"""
raddb/main.py
-------------
High-level interface for RadDB — a generic radar data archiving library.

``RadDB`` is a single, dual-role class:

* **Archive-bound** — ``db = RadDB(archive_dir, crs=2056)``.  Use it to
  :meth:`~RadDB.archive` DataTree volumes and to :meth:`~RadDB.open` archived
  data.  ``archive_dir`` / ``crs`` are the shared defaults for every task.
* **Data-carrying** — the object returned by :meth:`~RadDB.open` (and by
  ``filter`` / ``crop_* `` / ``extract_cross_section``).  It holds the loaded
  data as a **polars** DataFrame (``rdf.data``) and exposes the query,
  conversion, area-of-interest, cross-section and plotting methods.  These
  return a **new** ``RadDB`` (fluent ``open → filter → crop → plot``).

Note: network-specific constants such as a list of radar identifiers
(e.g. Swiss radars A, D, L, P, W) belong in the user script or in the
network-specific pipeline (e.g. the private ``raddb.mch`` subpackage),
not here.
"""
from __future__ import annotations

import datetime
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import shapely
import xarray as xr

from raddb.io_core import (
    archive_volume,
    archive_multiple_volumes,
    archive_volumes_multi_radar,
    open_any_datatree,
    scan_polar_parquet,
    dataframe_to_datatree,
)
from raddb.helper import ensure_utc, normalize_radar_name
from raddb.discovery import (
    find_datatree_files,
    _find_polar_files_in_range,
    _parse_datatree_file_time,
    _parse_pol_time,
)
from raddb.aoi import (
    _apply_gate_ids,
    _cross_section_gates,
    _load_aoi_polygon,
    _lut_centroids,
    _lut_cs_table,
    _radars_from_gate_ids,
    _reproject_to_2056,
    _resolve_aoi_centroids,
)
from raddb.lut import (
    RADAR_TO_IDX,
    generate_lut_from_datatree,
    load_radar_lut,
    load_radar_info,
    add_lut_projection,
)

logger = logging.getLogger(__name__)


# ================================================================
# Private helpers for end-to-end archiving
# ================================================================

def _iter_days(start: pd.Timestamp, end: pd.Timestamp):
    """Yield (day_start, day_end) pairs covering [start, end] inclusively."""
    day = start.normalize()
    last = end.normalize()
    while day <= last:
        day_start = max(day, start)
        day_end = min(day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), end)
        yield day_start, day_end
        day = day + pd.Timedelta(days=1)


def _load_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r") as f:
        return {line.strip() for line in f if line.strip()}


def _append_checkpoint(path: Path, entry: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(entry + "\n")


def _format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in a human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


# ================================================================
# Private helpers for the object model (filters, time, geometry)
# ================================================================

# Shapely-geometry columns are stored WKB-encoded (Binary) inside the polars
# backend and decoded back to shapely on ``to_pandas``.
_GEOM_COLS = ("cs_polygon",)


def _filter_expr(var: str, logic: str, threshold) -> "pl.Expr":
    """Build a polars boolean expression ``var <logic> threshold``."""
    col = pl.col(var)
    ops = {
        "==": col == threshold,
        "!=": col != threshold,
        ">": col > threshold,
        ">=": col >= threshold,
        "<": col < threshold,
        "<=": col <= threshold,
    }
    if logic not in ops:
        raise ValueError(
            f"Unknown filter logic {logic!r}; use one of {sorted(ops)}."
        )
    return ops[logic]


def _resolve_filters(filters) -> list[tuple[str, str, float]]:
    """Normalize a filter dict / list-of-dicts to ``[(var, logic, threshold), ...]``."""
    if filters is None:
        return []
    if isinstance(filters, dict):
        filters = [filters]
    specs = []
    for f in filters:
        specs.append((f["var"], f.get("logic", ">"), f.get("threshold", 0.0)))
    return specs


def _normalize_time_period(time_period):
    """Return ``(start, end)`` datetimes from str | datetime | (start, end)."""
    def _u(x):
        return ensure_utc(x) if x is not None else None

    if time_period is None:
        return None, None
    if isinstance(time_period, (tuple, list)):
        if len(time_period) == 1:
            return _u(time_period[0]), None
        return _u(time_period[0]), _u(time_period[1])
    return _u(time_period), None


def _encode_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """WKB-encode shapely-geometry columns so the frame can enter polars."""
    for c in _GEOM_COLS:
        if c in df.columns and len(df):
            sample = df[c].dropna()
            if len(sample) and hasattr(sample.iloc[0], "geom_type"):
                df = df.copy()
                df[c] = [None if g is None else shapely.to_wkb(g) for g in df[c]]
    return df


def _decode_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Decode WKB-encoded geometry columns back to shapely objects."""
    for c in _GEOM_COLS:
        if c in df.columns and len(df):
            sample = df[c].dropna()
            if len(sample) and isinstance(sample.iloc[0], (bytes, bytearray)):
                df[c] = [None if b is None else shapely.from_wkb(b) for b in df[c]]
    return df


def _to_polars(df: pd.DataFrame) -> "pl.DataFrame":
    """pandas -> polars, WKB-encoding any shapely-geometry columns first."""
    return pl.from_pandas(_encode_geometry(df))


def _radar_from_filename(path) -> str:
    """Infer the radar id from a saved DataTree filename (``<RADAR>_...``)."""
    return Path(path).stem.split("_")[0]


def _format_size(n_bytes: float) -> str:
    """Human-readable byte count."""
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024 or unit == "GB":
            return f"{n_bytes:,.1f} {unit}" if unit != "B" else f"{n_bytes:,.0f} B"
        n_bytes /= 1024
    return f"{n_bytes:,.1f} TB"


def _path_size(path: Path) -> int:
    """Size of a file, or the total size of a directory (e.g. a ``.zarr`` store)."""
    # ponytail: stats every file; fine for archives up to ~1e5 volumes, add a
    # `size=False` switch if scanning ever gets slow.
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _time_span(times: list) -> str:
    """``'first .. last'`` for a list of timestamps (``'unknown'`` when empty)."""
    known = sorted(t for t in times if t is not None)
    if not known:
        return "unknown (no timestamp in the filenames)"
    if len(known) == 1 or known[0] == known[-1]:
        return f"{known[0]:%Y-%m-%d %H:%M:%S}"
    return f"{known[0]:%Y-%m-%d %H:%M:%S} .. {known[-1]:%Y-%m-%d %H:%M:%S}"


def _print_daily_breakdown(times: list, indent: str = "      ") -> None:
    """One line per day: number of volumes and the first/last time of that day."""
    per_day = defaultdict(list)
    for t in times:
        if t is not None:
            per_day[t.date()].append(t)
    for day in sorted(per_day):
        ts = sorted(per_day[day])
        print(f"{indent}{day}  {len(ts):>5} volume(s)  {ts[0]:%H:%M:%S} .. {ts[-1]:%H:%M:%S}")
    n_undated = sum(1 for t in times if t is None)
    if n_undated:
        print(f"{indent}(no parseable timestamp)  {n_undated:>5} file(s)")


def _list_archive_radars(archive_dir: Path) -> list[str]:
    """Single-letter radar subdirectories that hold archived data."""
    if not archive_dir.exists():
        return []
    return sorted(
        p.name
        for p in archive_dir.iterdir()
        if p.is_dir() and len(p.name) == 1 and p.name.isalpha()
    )


class RadDB:
    """Generic radar-data archiving and analysis interface (see module docstring).

    Examples
    --------
    >>> db = RadDB(archive_dir="/data/raddb", crs=2056)
    >>> db.archive(datatree_dir="/data/MCH_datatree")        # or datatree=dt
    >>> rdf = db.open(time_period=("2024-08-26", "2024-08-27"))
    >>> rdf.filter({"var": "DBZH", "logic": ">", "threshold": 20})\
    ...    .crop_by_bbox(extent=rdf.extent())\
    ...    .plot_ppi(variable="DBZH", save="ppi.png")
    """

    # ================================================================
    # Construction & state
    # ================================================================

    def __init__(
        self,
        archive_dir: str | None = None,
        crs: int | str | None = None,
        network: str = "",
        *,
        _data: "pl.DataFrame | None" = None,
        _meta: dict | None = None,
    ):
        """Create an archive-bound RadDB.

        Parameters
        ----------
        archive_dir : str, optional
            Base directory of the RadDB archive (LUT + POL parquet under
            ``{archive_dir}/{radar}/``).  Shared default for every task.
        crs : int, str or CRS-coercible, optional
            Projection for the LUT (e.g. ``2056`` for CH1903+/LV95).  Used both
            when generating LUTs during :meth:`archive` and to resolve projected
            coordinates (:meth:`extent`, :meth:`to_geopandas`).
        network : str, optional
            Network label stored in generated LUT metadata.

        The private ``_data`` / ``_meta`` arguments carry the polars backend of
        a data-carrying instance and are not meant to be passed directly; use
        :meth:`open`.
        """
        self.archive_dir = Path(archive_dir) if archive_dir is not None else None
        self._crs = crs
        self.network = network
        self._data = _data
        self._meta = dict(_meta) if _meta else {}

    def _derive(self, data: "pl.DataFrame", *, archive_dir=None, crs=None, **meta) -> "RadDB":
        """Build a new data-carrying ``RadDB`` sharing this one's configuration."""
        return RadDB(
            archive_dir=str(archive_dir) if archive_dir is not None
            else (str(self.archive_dir) if self.archive_dir else None),
            crs=crs if crs is not None else self._crs,
            network=self.network,
            _data=data,
            _meta={**self._meta, **meta},
        )

    def _require_archive_dir(self) -> Path:
        if self.archive_dir is None:
            raise ValueError(
                "This RadDB has no archive_dir; pass it to RadDB(archive_dir=...) "
                "or to the method call."
            )
        return self.archive_dir

    def _require_data(self) -> "pl.DataFrame":
        if self._data is None:
            raise ValueError(
                "This RadDB carries no data (it is archive-bound). Load data "
                "first with db.open(...)."
            )
        return self._data

    @property
    def data(self) -> "pl.DataFrame":
        """The loaded data as a polars DataFrame."""
        return self._require_data()

    def __len__(self) -> int:
        return 0 if self._data is None else self._data.height

    def __repr__(self) -> str:
        if self._data is None:
            return (
                f"RadDB(archive_dir={self.archive_dir!s}, crs={self._crs!r}) "
                f"[archive-bound, no data loaded]"
            )
        lines = [f"RadDB [{len(self):,} gates]"]
        try:
            lines.append(f"  radars     : {self.radars()}")
        except Exception:
            pass
        try:
            t0, t1 = self.start_time(), self.end_time()
            lines.append(f"  time range : {t0} .. {t1}")
        except Exception:
            pass
        schema = self._data.schema
        cols = ", ".join(f"{n}:{str(t)}" for n, t in list(schema.items())[:12])
        more = "" if len(schema) <= 12 else f" (+{len(schema) - 12} more)"
        lines.append(f"  columns    : {cols}{more}")
        lines.append(f"  archive_dir: {self.archive_dir}")
        return "\n".join(lines)

    __str__ = __repr__

    # ================================================================
    # ARCHIVE
    # ================================================================

    def archive(
        self,
        datatree_dir: str | None = None,
        datatree=None,
        archive_dir: str | None = None,
        crs: int | str | None = None,
        radar: str | list[str] | None = None,
        filter: dict | None = None,
        time_period=None,
    ) -> dict:
        """Archive DataTree volumes to the RadDB Parquet store.

        Exactly one input source must be given:

        * ``datatree`` — an in-memory ``xr.DataTree``, a list of them, a
          ``{label: DataTree}`` dict (one radar), or a ``{radar: [DataTree, ...]}``
          dict (several radars).
        * ``datatree_dir`` — a directory of **saved** DataTree files
          (``.zarr`` / ``.nc``); volumes are grouped by radar from the filename.

        Passing both raises ``ValueError``.

        Parameters
        ----------
        datatree_dir : str, optional
            Directory of saved DataTree files to archive.
        datatree : xr.DataTree, list, or dict, optional
            In-memory volume(s) to archive.
        archive_dir, crs : optional
            Override the instance defaults for this call.
        radar : str or list of str, optional
            Restrict / assign the radar(s).  For ``datatree_dir`` with
            ``radar=None`` (default) the radar is inferred per file from the
            filename and **all** radars are archived.  For a single in-memory
            ``DataTree`` / list, ``radar`` is required.
        filter : dict, optional
            Gate filter ``{"var", "logic", "threshold"}`` defining which gates
            to keep.  Default: ``{"var": "DBZH", "logic": ">", "threshold": 0}``
            (drops no-echo gates).
        time_period : str, datetime or (start, end), optional
            For ``datatree_dir``: keep only files whose filename timestamp falls
            in the period.

        Returns
        -------
        dict
            ``{"n_archived": int, "n_failed": int, "radars": [...]}``.
        """
        archive_dir = Path(archive_dir) if archive_dir is not None else self.archive_dir
        if archive_dir is None:
            raise ValueError("archive_dir must be given (via RadDB(...) or this call).")
        crs = crs if crs is not None else self._crs

        if (datatree is None) == (datatree_dir is None):
            raise ValueError(
                "Pass exactly one of `datatree` (in-memory) or `datatree_dir` "
                "(saved files)."
            )

        if filter is None:
            feat, logic, thr = "DBZH", ">", 0.0
        else:
            feat = filter.get("var", "DBZH")
            logic = filter.get("logic", ">")
            thr = filter.get("threshold", 0.0)

        t0 = time.time()
        if datatree is not None:
            radars_done, n_ok, n_fail = self._archive_in_memory(
                datatree, radar, archive_dir, crs, feat, logic, thr
            )
        else:
            radars_done, n_ok, n_fail = self._archive_from_disk(
                datatree_dir, radar, archive_dir, crs, feat, logic, thr, time_period
            )

        print("=" * 70)
        print("RadDB archive")
        print(f"  archive_dir : {archive_dir}")
        print(f"  crs         : {crs}")
        print(f"  radars      : {radars_done}")
        print(f"  filter      : keep {feat} {logic} {thr}")
        print(f"  volumes     : {n_ok} archived, {n_fail} failed")
        print(f"  elapsed     : {_format_elapsed_time(time.time() - t0)}")
        print("=" * 70)
        return {"n_archived": n_ok, "n_failed": n_fail, "radars": radars_done}

    def _ensure_lut(self, radar: str, sample_dt, archive_dir: Path, crs) -> None:
        """Generate the per-radar LUT from a sample volume if it does not exist."""
        lut_path = archive_dir / radar / "LUT" / f"{radar}_LUT.parquet"
        if lut_path.exists():
            return
        try:
            generate_lut_from_datatree(
                dt=sample_dt,
                radar=radar,
                output_base_path=str(archive_dir),
                network=self.network,
                projection_epsg=crs if isinstance(crs, int) else None,
                projection_crs=None if isinstance(crs, int) else crs,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [{radar}] LUT generation failed: {e}")

    def _archive_in_memory(self, datatree, radar, archive_dir, crs, feat, logic, thr):
        archive_dir = Path(archive_dir)
        # {radar: [DataTree, ...]} -- multi-radar
        if isinstance(datatree, dict) and datatree and all(
            not isinstance(v, xr.DataTree) for v in datatree.values()
        ):
            for r, vols in datatree.items():
                rn = normalize_radar_name(r)
                first = next(iter(vols.values())) if isinstance(vols, dict) else vols[0]
                self._ensure_lut(rn, first, archive_dir, crs)
            results = archive_volumes_multi_radar(
                volumes_by_radar=datatree,
                base_output_path=str(archive_dir),
                filter_feature=feat, filter_threshold=thr, filter_logic=logic,
                verbose=False,
            )
            n = sum(len(v) for v in results.values())
            return list(results.keys()), n, 0

        if radar is None or not isinstance(radar, str):
            raise ValueError(
                "For in-memory archiving, pass a single radar letter, e.g. "
                "archive(datatree=dt, radar='A')."
            )
        r = normalize_radar_name(radar)
        if isinstance(datatree, xr.DataTree):
            self._ensure_lut(r, datatree, archive_dir, crs)
            archive_volume(
                dt=datatree, radar=r, base_output_path=str(archive_dir),
                filter_feature=feat, filter_threshold=thr, filter_logic=logic,
            )
            return [r], 1, 0
        # list or {label: DataTree}
        first = next(iter(datatree.values())) if isinstance(datatree, dict) else datatree[0]
        self._ensure_lut(r, first, archive_dir, crs)
        results = archive_multiple_volumes(
            volumes=datatree, radar=r, base_output_path=str(archive_dir),
            filter_feature=feat, filter_threshold=thr, filter_logic=logic,
            verbose=False,
        )
        return [r], len(results), 0

    def _archive_from_disk(self, datatree_dir, radar, archive_dir, crs, feat, logic, thr, time_period):
        archive_dir = Path(archive_dir)
        start, end = _normalize_time_period(time_period)
        files = find_datatree_files(
            Path(datatree_dir), recursive=True, start_time=start, end_time=end
        )
        if isinstance(radar, str):
            # A single radar name: archive every file as that radar (the
            # filename need not encode the radar, e.g. generic ``vol_*`` files).
            by_radar: dict[str, list] = {normalize_radar_name(radar): list(files)}
        else:
            # Infer the radar per file from its filename (``<RADAR>_...``).
            by_radar = defaultdict(list)
            for f in files:
                by_radar[_radar_from_filename(f)].append(f)
            if radar is not None:  # a list of radars -> keep only those
                wanted = {normalize_radar_name(x) for x in radar}
                by_radar = {r: fs for r, fs in by_radar.items() if r in wanted}

        radars_done, total_ok, total_fail = [], 0, 0
        for r, rfiles in sorted(by_radar.items()):
            n_ok, n_fail = self._archive_files_one_radar(
                r, sorted(rfiles), archive_dir, crs, feat, logic, thr
            )
            radars_done.append(r)
            total_ok += n_ok
            total_fail += n_fail
        return radars_done, total_ok, total_fail

    def _archive_files_one_radar(self, radar, files, archive_dir, crs, feat, logic, thr):
        """Archive every saved DataTree file for one radar (LUT autogen, resume)."""
        if radar not in RADAR_TO_IDX:
            print(
                f"  [skip] radar {radar!r} is not a single letter A-Z "
                f"(gate_id encoding limit); skipping {len(files)} file(s)."
            )
            return (0, 0)
        archive_dir.mkdir(parents=True, exist_ok=True)
        ckpt = archive_dir / f"_archive_checkpoint_datatrees_{radar}.txt"
        seen = _load_checkpoint(ckpt)

        preopened: dict = {}
        lut_path = archive_dir / radar / "LUT" / f"{radar}_LUT.parquet"
        if not lut_path.exists() and files:
            try:
                dt0 = open_any_datatree(files[0])
                preopened[files[0]] = dt0
                self._ensure_lut(radar, dt0, archive_dir, crs)
            except Exception as e:  # noqa: BLE001
                print(f"  [{radar}] LUT generation failed: {e}")

        n_ok = n_fail = 0
        for f in files:
            stem = Path(f).stem
            key = f"{radar}:{stem}"
            if key in seen:
                preopened.pop(f, None)
                continue
            try:
                dt = preopened.pop(f, None)
                if dt is None:
                    dt = open_any_datatree(f)
                archive_volume(
                    dt=dt, radar=radar, base_output_path=str(archive_dir),
                    filter_feature=feat, filter_threshold=thr, filter_logic=logic,
                    volume=stem,
                )
                _append_checkpoint(ckpt, key)
                seen.add(key)
                n_ok += 1
                del dt
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                print(f"  [{radar}] FAIL {stem}: {e}")
        return (n_ok, n_fail)

    # ---- LUT read accessors (archive-bound) ----

    def get_lut(self, radar: str) -> "pl.DataFrame":
        """Load the LUT (static gate geometry) for a radar, as polars."""
        return load_radar_lut(normalize_radar_name(radar), self._require_archive_dir())

    def get_radar_info(self, radar: str) -> dict:
        """Load radar metadata (location, sweep geometry)."""
        return load_radar_info(normalize_radar_name(radar), self._require_archive_dir())

    def add_lut_projection(self, radar: str, epsg: int | None = None, crs=None) -> "pl.DataFrame":
        """Return the radar LUT enriched with projected ``x_{epsg}`` / ``y_{epsg}`` columns."""
        lut_df = load_radar_lut(normalize_radar_name(radar), self._require_archive_dir())
        return add_lut_projection(lut_df, epsg=epsg, crs=crs)

    def list_radars(self) -> list[str]:
        """List radar identifiers that have data in the archive."""
        return _list_archive_radars(self._require_archive_dir())

    # ---- what is on disk? ----

    def inventory(self, datatree_dir: str | None = None, detailed: bool = False,
                  archive_dir: str | None = None) -> None:
        """Print what data is available on disk — which radars, which time periods.

        Answers "what can I analyse?" before :meth:`open` (archive side) or
        :meth:`archive` (input side).  Prints only; nothing is loaded into memory.

        Parameters
        ----------
        datatree_dir : str, optional
            Scan a directory of **saved DataTree files** (``.zarr`` / ``.nc``) —
            i.e. data not archived yet.  Radar and time come from the filename
            (``<RADAR>_<YYYYMMDD>_<HHMMSS>``).  When omitted, the **RadDB
            archive** is scanned instead.
        detailed : bool
            Add a per-radar, per-day breakdown; for the archive also the LUT size,
            the sweep count and the moment columns stored per volume.
        archive_dir : str, optional
            Override the archive location for this call.

        Examples
        --------
        >>> db.inventory()                                  # what is archived
        >>> db.inventory(detailed=True)                     # ... day by day
        >>> db.inventory(datatree_dir="/data/MCH_datatree") # what could be archived
        """
        if datatree_dir is not None:
            self._inventory_datatrees(Path(datatree_dir), detailed)
        else:
            base = Path(archive_dir) if archive_dir is not None else self._require_archive_dir()
            self._inventory_archive(base, detailed)

    def _inventory_datatrees(self, directory: Path, detailed: bool) -> None:
        """Inventory of saved DataTree files (the input side of the archive)."""
        files = find_datatree_files(directory, recursive=True)
        print("=" * 78)
        print("RadDB inventory — DataTree files on disk (not archived yet)")
        print(f"  directory : {directory}")
        if not files:
            print("  no .zarr / .nc / .nc4 / .cdf files found")
            print("=" * 78)
            return

        by_radar: dict[str, list] = defaultdict(list)
        for f in files:
            by_radar[_radar_from_filename(f)].append(f)

        times_all = [_parse_datatree_file_time(f) for f in files]
        print(f"  files     : {len(files)}")
        print(f"  radars    : {', '.join(sorted(by_radar))}  (from the filename prefix)")
        print(f"  time range: {_time_span(times_all)}")
        print("-" * 78)
        print(f"  {'radar':<7}{'files':>8}  {'time range':<45}{'size':>12}")
        for r in sorted(by_radar):
            rfiles = by_radar[r]
            times = [_parse_datatree_file_time(f) for f in rfiles]
            size = sum(_path_size(f) for f in rfiles)
            print(f"  {r:<7}{len(rfiles):>8}  {_time_span(times):<45}{_format_size(size):>12}")
            if detailed:
                _print_daily_breakdown(times)
                if r not in RADAR_TO_IDX:
                    print(f"      [!] {r!r} is not a single letter A-Z — archive() would skip it "
                          f"unless you pass radar='<letter>'")
        print("-" * 78)
        print(f"  archive with: db.archive(datatree_dir={str(directory)!r})")
        print("=" * 78)

    def _inventory_archive(self, base: Path, detailed: bool) -> None:
        """Inventory of an existing RadDB archive (the output side)."""
        print("=" * 78)
        print("RadDB inventory — archived data")
        print(f"  archive_dir : {base}")
        radars = _list_archive_radars(base)
        if not radars:
            print("  no radar directories found — nothing archived here yet")
            print("=" * 78)
            return

        per_radar = {}
        for r in radars:
            pol = _find_polar_files_in_range(base / r)
            per_radar[r] = (pol, [_parse_pol_time(f) for f in pol])

        n_vol = sum(len(v[0]) for v in per_radar.values())
        all_times = [t for v in per_radar.values() for t in v[1]]
        print(f"  radars      : {', '.join(radars)}")
        print(f"  volumes     : {n_vol}")
        print(f"  time range  : {_time_span(all_times)}")
        print("-" * 78)
        print(f"  {'radar':<7}{'volumes':>8}  {'time range':<45}{'size':>12}")
        for r in radars:
            pol, times = per_radar[r]
            size = sum(_path_size(f) for f in pol)
            print(f"  {r:<7}{len(pol):>8}  {_time_span(times):<45}{_format_size(size):>12}")
            if not detailed:
                continue
            lut_path = base / r / "LUT" / f"{r}_LUT.parquet"
            if lut_path.exists():
                try:
                    info = load_radar_info(r, base)
                    print(f"      LUT: {_format_size(_path_size(lut_path))}, "
                          f"{len(info.get('sweeps', {}))} sweeps, site "
                          f"({info.get('latitude'):.4f}, {info.get('longitude'):.4f}) "
                          f"at {info.get('altitude'):.0f} m")
                except Exception as e:  # noqa: BLE001
                    print(f"      LUT: present, metadata unreadable ({e})")
            else:
                print("      LUT: MISSING — open() will have no geometry for this radar")
            if pol:
                try:
                    cols = pl.read_parquet_schema(pol[0]).keys()
                    print(f"      columns: {', '.join(c for c in cols if c != 'gate_id')}")
                except Exception as e:  # noqa: BLE001
                    print(f"      columns: unreadable ({e})")
            _print_daily_breakdown(times)
        print("-" * 78)
        print("  load with   : db.open(radars=..., time_period=(start, end))")
        print("=" * 78)

    # ================================================================
    # OPEN & FILTER
    # ================================================================

    def open(
        self,
        time_period=None,
        radars: str | list[str] | None = None,
        columns: list[str] | None = None,
        filters=None,
        archive_dir: str | None = None,
    ) -> "RadDB":
        """Load archived data into a data-carrying ``RadDB``.

        Parameters
        ----------
        time_period : str, datetime or (start, end), optional
            Time range to load.  ``None`` loads everything available.
        radars : str or list of str, optional
            Radar(s) to load.  ``None`` (default) loads all radars in the archive.
        columns : list of str, optional
            Subset of moment columns to load (``gate_id`` is always included).
        filters : dict or list of dict, optional
            Gate filter(s) ``{"var", "logic", "threshold"}`` applied after load
            (a list is combined with AND).
        archive_dir : str, optional
            Override the archive location for this call.

        Returns
        -------
        RadDB
            A data-carrying instance (``rdf``).
        """
        archive_dir = Path(archive_dir) if archive_dir is not None else self._require_archive_dir()
        start, end = _normalize_time_period(time_period)

        if radars is None:
            radars = _list_archive_radars(archive_dir)
        elif isinstance(radars, str):
            radars = [radars]

        if columns is not None and "gate_id" not in columns:
            columns = ["gate_id", *columns]

        scans = []
        for r in radars:
            lf = scan_polar_parquet(
                radar=normalize_radar_name(r), base_path=archive_dir,
                start_time=start, end_time=end, columns=columns,
            )
            if lf is not None:
                scans.append(lf)

        # Filters are applied to the plan, so polars only materialises the rows
        # that survive them.
        if scans:
            plan = pl.concat(scans, how="vertical_relaxed")
            for var, logic, thr in _resolve_filters(filters):
                plan = plan.filter(_filter_expr(var, logic, thr))
            data = plan.collect()
        else:
            data = pl.DataFrame()
        return self._derive(data, archive_dir=archive_dir)

    def filter(self, filters) -> "RadDB":
        """Keep only gates satisfying ``filters`` (row removal); returns a new RadDB.

        ``filters`` is a dict ``{"var", "logic", "threshold"}`` or a list of such
        dicts (combined with AND).  Vertical subsetting works the same way on an
        ``altitude`` column, e.g.
        ``rdf.filter([{"var": "altitude", "logic": ">", "threshold": 2000},
        {"var": "altitude", "logic": "<", "threshold": 4000}])``.

        Filtering on a static column (``altitude``, ``sweep``, ``latitude`` …)
        joins it from the LUT just long enough to evaluate the predicate and
        drops it again, so the returned frame still carries dynamic values only.
        """
        data = self._require_data()
        specs = _resolve_filters(filters)

        borrowed = []
        missing = [v for v, _, _ in specs if v not in data.columns]
        if missing:
            geo = self._gate_geometry()
            borrowed = [c for c in dict.fromkeys(missing) if c in geo.columns]
            unknown = [c for c in missing if c not in geo.columns]
            if unknown:
                raise KeyError(
                    f"cannot filter on {unknown}: not a data column "
                    f"{sorted(data.columns)} nor a LUT column {sorted(geo.columns)}."
                )
            data = data.join(geo.select("gate_id", *borrowed), on="gate_id", how="left")

        for var, logic, thr in specs:
            data = data.filter(_filter_expr(var, logic, thr))
        if borrowed:
            data = data.drop(borrowed)
        return self._derive(data)

    def add_feature(self, name: str, compute_fn) -> "RadDB":
        """Add a computed column ``name`` and return a new RadDB.

        ``compute_fn`` receives the polars DataFrame and returns a polars
        ``Expr`` or a column-length array/Series, e.g.
        ``rdf.add_feature("ZDR_lin", lambda df: 10 ** (df["ZDR"] / 10))``.
        """
        data = self._require_data()
        res = compute_fn(data)
        if isinstance(res, pl.Expr):
            data = data.with_columns(res.alias(name))
        else:
            data = data.with_columns(pl.Series(name, res))
        return self._derive(data)

    # ---- converters ----

    def to_pandas(self, with_geometry: bool = False) -> pd.DataFrame:
        """Return the data as a pandas DataFrame.

        With ``with_geometry=True`` the per-gate LUT geometry
        (``latitude``/``longitude``/``altitude`` and, if ``crs`` is set,
        ``x_{epsg}``/``y_{epsg}``) is merged in on ``gate_id``.
        """
        data = self._require_data()
        if with_geometry:
            data = data.join(self._gate_geometry(), on="gate_id", how="left", suffix="_lut")
        return _decode_geometry(data.to_pandas())

    def to_geopandas(self):
        """Return a GeoDataFrame with a per-gate point geometry and CRS."""
        import geopandas as gpd

        df = self.to_pandas(with_geometry=True)
        epsg = int(self._crs) if isinstance(self._crs, int) else None
        xcol, ycol = (f"x_{epsg}", f"y_{epsg}") if epsg else (None, None)
        if xcol and xcol in df.columns:
            geom = gpd.points_from_xy(df[xcol], df[ycol])
            crs = self._crs
        else:
            geom = gpd.points_from_xy(df["longitude"], df["latitude"])
            crs = 4326
        return gpd.GeoDataFrame(df, geometry=geom, crs=crs)

    def to_geoarrow(
        self,
        geometry: str = "point",
        columns: list[str] | None = None,
        max_rows: int | None = 5_000_000,
    ):
        """Return the data as a GeoArrow-tagged ``pyarrow.Table``.

        This is the export path for GPU map renderers (deck.gl / lonboard) and
        for GeoParquet.  The dynamic columns come straight from the polars frame
        with no copy; the LUT geometry is joined **here and only here**.

        Parameters
        ----------
        geometry : {"point", "polygon"}
            ``"point"`` gives one gate centroid per row (~24 B/gate).
            ``"polygon"`` gives the exact wedge — the gate's four corners taken
            from the LUT's per-sweep edge arrays (~100 B/gate).
        columns : list of str, optional
            Dynamic columns to export.  ``None`` (default) exports all of them.
            ``gate_id`` is always included.
        max_rows : int, optional
            Guardrail: raise if the frame is larger, since renderers top out
            around a few million features.  Pass ``None`` to disable.

        Returns
        -------
        pyarrow.Table
            The dynamic columns plus a ``geometry`` column carrying the GeoArrow
            extension metadata.

        Examples
        --------
        >>> import lonboard
        >>> lonboard.viz(rdf.crop_by_bbox(bounds).to_geoarrow(geometry="polygon"))
        """
        import pyarrow as pa

        from raddb.lut import gate_polygons_geoarrow, geoarrow_field

        if geometry not in ("point", "polygon"):
            raise ValueError(f"geometry must be 'point' or 'polygon'; got {geometry!r}.")

        data = self._require_data()
        if max_rows is not None and len(data) > max_rows:
            raise ValueError(
                f"to_geoarrow() would build {len(data):,} features, over the "
                f"max_rows={max_rows:,} guardrail. Narrow the selection first "
                "(crop_by_bbox / crop_by_polygone / crop_around_point / filter), "
                "or pass max_rows=None to override."
            )
        if columns is not None:
            data = data.select(dict.fromkeys(["gate_id", *columns]))

        table = data.to_arrow()
        gate_ids = data["gate_id"].to_numpy()

        if geometry == "polygon":
            radars = self.radars()
            if len(radars) != 1:
                raise ValueError(
                    f"polygon geometry needs a single radar; data spans {radars}. "
                    "Select one with open(radars=...) or filter first."
                )
            geom = gate_polygons_geoarrow(
                normalize_radar_name(radars[0]), self._require_archive_dir(), gate_ids,
            )
            field = geoarrow_field("geometry", geom.type, "polygon", "EPSG:4326")
        else:
            geo = (
                pl.DataFrame({"gate_id": gate_ids, "_ord": np.arange(len(gate_ids))})
                .join(self._gate_geometry(), on="gate_id", how="left")
                .sort("_ord")
            )
            xy = np.column_stack([geo["longitude"].to_numpy(), geo["latitude"].to_numpy()])
            geom = pa.FixedSizeListArray.from_arrays(
                pa.array(xy.reshape(-1), type=pa.float64()), 2
            )
            field = geoarrow_field("geometry", geom.type, "point", "EPSG:4326")

        return table.append_column(field, geom)

    def to_datatree(self, radar: str | None = None, timestep=None, label_column: str = "DBZH") -> xr.DataTree:
        """Reconstruct a single-volume ``xr.DataTree`` from the loaded data.

        Recovers geometry from the LUT and NaN-fills gates absent from the data.
        Selects one radar and one volume:

        - ``radar`` — inferred when the data covers exactly one radar, else required.
        - ``timestep`` — nearest ``volume_time``; required only when several volumes.
        """
        df = self.to_pandas()
        if "gate_id" not in df.columns:
            raise KeyError("data has no 'gate_id' column; cannot reconstruct a DataTree.")

        present = _radars_from_gate_ids(df["gate_id"])
        if radar is None:
            if len(present) != 1:
                raise ValueError(f"data spans radars {present}; pass radar= to pick one.")
            radar = present[0]
        radar = normalize_radar_name(radar)

        if "radar" in df.columns:
            df_r = df[df["radar"] == radar]
        else:
            df_r = df[df["gate_id"].to_numpy() // 1_000_000_000_000 == RADAR_TO_IDX[radar]]
        if df_r.empty:
            raise ValueError(f"No rows for radar {radar!r} in data.")

        if "volume_time" in df_r.columns and df_r["volume_time"].notna().any():
            vols = pd.to_datetime(df_r["volume_time"]).dropna().unique()
            if timestep is None:
                if len(vols) != 1:
                    raise ValueError(f"data holds {len(vols)} volumes; pass timestep= to pick one.")
                chosen = pd.Timestamp(vols[0])
            else:
                ts = pd.to_datetime(timestep)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                chosen = min((pd.Timestamp(v) for v in vols), key=lambda v: abs(v - ts))
            df_vol = df_r[pd.to_datetime(df_r["volume_time"]) == chosen]
        else:
            df_vol = df_r

        return dataframe_to_datatree(
            df=df_vol, radar=radar, base_path=str(self._require_archive_dir()),
            label_column=label_column,
        )

    # ---- accessors (return values; repr prints the summary) ----

    def head(self, n: int = 5) -> "pl.DataFrame":
        """First ``n`` rows (polars)."""
        return self._require_data().head(n)

    def tail(self, n: int = 5) -> "pl.DataFrame":
        """Last ``n`` rows (polars)."""
        return self._require_data().tail(n)

    def columns(self) -> list[str]:
        """Column names of the loaded data."""
        return list(self._require_data().columns)

    def radars(self) -> list[str]:
        """Radar identifiers present in the loaded data."""
        data = self._require_data()
        if "radar" in data.columns:
            return sorted(data["radar"].unique().to_list())
        return list(_radars_from_gate_ids(data["gate_id"].to_pandas()))

    def _time_column(self) -> str:
        data = self._require_data()
        for c in ("volume_time", "time"):
            if c in data.columns:
                return c
        raise KeyError("data has no 'volume_time' or 'time' column.")

    def start_time(self) -> datetime.datetime:
        """Earliest volume/ray time in the loaded data."""
        return self._require_data()[self._time_column()].min()

    def end_time(self) -> datetime.datetime:
        """Latest volume/ray time in the loaded data."""
        return self._require_data()[self._time_column()].max()

    def _gate_geometry(self) -> "pl.DataFrame":
        """LUT geometry (lon/lat/alt [+ projected x/y]) for the gates present.

        Returned as its **own** table — the LUT is never carried alongside the
        dynamic values; the ``to_*`` converters join it on demand.
        """
        archive_dir = self._require_archive_dir()
        epsg = int(self._crs) if isinstance(self._crs, int) else None
        radars = self.radars()
        if not radars:  # empty selection (e.g. a crop that matched nothing)
            cols = ["gate_id", "latitude", "longitude", "altitude", "sweep"]
            if epsg:
                cols += [f"x_{epsg}", f"y_{epsg}"]
            return pl.DataFrame(schema={c: (pl.Int64 if c == "gate_id" else pl.Float64) for c in cols})
        parts = []
        for r in radars:
            lut = load_radar_lut(normalize_radar_name(r), archive_dir)
            keep = [c for c in ("gate_id", "latitude", "longitude", "altitude", "sweep") if c in lut.columns]
            if epsg and f"x_{epsg}" in lut.columns:
                keep += [f"x_{epsg}", f"y_{epsg}"]
            parts.append(lut.select(keep))
        geo = pl.concat(parts, how="vertical").unique(subset="gate_id", maintain_order=True)
        present = self._require_data().select("gate_id").unique()
        return geo.join(present, on="gate_id", how="semi")

    def extent(self) -> list[float]:
        """Projected bounding box ``[xmin, xmax, ymin, ymax]`` (in ``crs``)."""
        epsg = int(self._crs) if isinstance(self._crs, int) else None
        if epsg is None:
            raise ValueError("extent() needs a projected crs; set RadDB(crs=<epsg>).")
        geo = self._gate_geometry()
        xcol, ycol = f"x_{epsg}", f"y_{epsg}"
        if xcol not in geo.columns:
            raise ValueError(
                f"LUT has no {xcol} column; archive with crs={epsg} to store projected coords."
            )
        return [float(geo[xcol].min()), float(geo[xcol].max()),
                float(geo[ycol].min()), float(geo[ycol].max())]

    def geographic_extent(self) -> list[float]:
        """Geographic bounding box ``[lon_min, lon_max, lat_min, lat_max]``."""
        geo = self._gate_geometry()
        return [float(geo["longitude"].min()), float(geo["longitude"].max()),
                float(geo["latitude"].min()), float(geo["latitude"].max())]

    def crs(self):
        """Projected CRS (x, y) as a ``pyproj.CRS``."""
        import pyproj

        if self._crs is None:
            raise ValueError("No projected crs set; pass RadDB(crs=<epsg>).")
        return pyproj.CRS.from_user_input(self._crs)

    def geographic_crs(self):
        """Geographic CRS (lat, lon) as a ``pyproj.CRS`` (EPSG:4326)."""
        import pyproj

        return pyproj.CRS.from_epsg(4326)

    # ================================================================
    # EXTRACT AREA OF INTEREST
    # ================================================================

    def crop_by_bbox(self, bounds=None, extent=None, crs: int | str = 2056, quicklook: bool = False) -> "RadDB":
        """Crop to a rectangle; returns a new RadDB.

        Give **exactly one** of ``bounds=(xmin, ymin, xmax, ymax)`` or
        ``extent=[xmin, xmax, ymin, ymax]`` (the latter matches :meth:`extent`),
        in ``crs`` (default EPSG:2056).  A single horizontal footprint selects
        the whole vertical column above it.
        """
        if (bounds is None) == (extent is None):
            raise ValueError("Pass exactly one of `bounds` or `extent`.")
        if extent is not None:
            xmin, xmax, ymin, ymax = extent
        else:
            xmin, ymin, xmax, ymax = bounds
        if not (xmin < xmax and ymin < ymax):
            raise ValueError("expected xmin < xmax and ymin < ymax.")
        geom = _reproject_to_2056(shapely.box(xmin, ymin, xmax, ymax), crs)
        return self._derive(self._crop_to_aoi(self._require_data(), geom, quicklook))

    def crop_by_polygone(self, polygon, crs: int | str | None = None, quicklook: bool = False) -> "RadDB":
        """Crop to an arbitrary polygon (shapely, GeoDataFrame/GeoSeries, or a
        ``.shp``/``.geojson`` path); returns a new RadDB.  ``crs=None`` auto-detects.
        """
        geom = _load_aoi_polygon(polygon, crs)
        return self._derive(self._crop_to_aoi(self._require_data(), geom, quicklook))

    def crop_around_point(self, point, distance: float, crs: int | str | None = None, quicklook: bool = False) -> "RadDB":
        """Crop to a circle of radius ``distance`` (metres) around ``point``;
        returns a new RadDB.  ``point`` is ``(x, y)`` or a shapely Point in ``crs``.
        """
        if distance <= 0:
            raise ValueError(f"distance must be positive (metres); got {distance!r}.")
        if hasattr(point, "geom_type"):
            if point.geom_type != "Point":
                raise TypeError(f"point geometry must be a Point; got {point.geom_type}.")
            pt = point
        elif isinstance(point, (tuple, list)) and len(point) == 2:
            pt = shapely.Point(float(point[0]), float(point[1]))
        else:
            raise TypeError(f"point must be (x, y) or a shapely Point; got {type(point).__name__}.")
        geom = _reproject_to_2056(pt, crs).buffer(distance)
        return self._derive(self._crop_to_aoi(self._require_data(), geom, quicklook))

    def _crop_to_aoi(self, data: "pl.DataFrame", geom, quicklook: bool = False) -> "pl.DataFrame":
        """Intersect ``geom`` (EPSG:2056) with LUT centroids and keep the matching rows.

        This **selects** rows, it never widens them: the LUT geometry stays in its
        own table, reachable through the ``to_*`` converters.  The intersection
        itself runs against the static LUT centroids, so it costs the same no
        matter how many volumes ``data`` spans.
        """
        if "gate_id" not in data.columns:
            raise KeyError("data has no 'gate_id' column; cannot crop by AOI.")
        radars = _radars_from_gate_ids(data["gate_id"].to_numpy())
        centroids = _lut_centroids(self._require_archive_dir(), radars)
        aoi_cen = _resolve_aoi_centroids(centroids, geom)
        data_aoi = _apply_gate_ids(data, aoi_cen["gate_id"].to_numpy())

        if quicklook:
            from raddb.viz.plot import plot_aoi_quicklook
            # ponytail: the quicklook is the one consumer that needs coordinates,
            # so join them for the plot only — never into the returned frame.
            selected = data_aoi.join(
                aoi_cen.select("gate_id", "x_2056", "y_2056"), on="gate_id", how="left",
            )
            plot_aoi_quicklook(geom, selected=selected, radars=radars, base_path=self.archive_dir)
        return data_aoi

    def interactive_crop(self, **kwargs):
        """Draw an AOI on an interactive ipyleaflet map and crop (Jupyter only).

        Returns an ``AOISelector``; after drawing a shape and clicking *Apply
        crop*, the cropped result (a new ``RadDB``) is on ``selector.result``.
        """
        self._require_data()
        from raddb.viz.interactive import AOISelector
        return AOISelector(self, **kwargs).display()

    # ================================================================
    # EXTRACT CROSS-SECTION
    # ================================================================

    def extract_cross_section(self, p1, p2, crs: int | str | None = None, beamwidth_deg: float = 1.0, quicklook: bool = False) -> "RadDB":
        """Extract a vertical cross-section along the line ``p1 -> p2``; returns a new RadDB.

        The line need not pass through a radar.  Each selected gate gets a polygon
        in the (distance-along-line, altitude) plane (``cs_polygon``) plus
        ``d_near/d_far/z_near/z_far`` and ``d_center/z_center``; visualize with
        :meth:`plot_cross_section`.  ``p1``/``p2`` are ``(x, y)`` or shapely Points
        in ``crs``; distance is measured from ``p1``.
        """
        data = self._require_data()
        if "gate_id" not in data.columns:
            raise KeyError("data has no 'gate_id' column; cannot extract a cross-section.")

        pts = []
        for name, p in (("p1", p1), ("p2", p2)):
            if hasattr(p, "geom_type"):
                if p.geom_type != "Point":
                    raise TypeError(f"{name} must be a Point; got {p.geom_type}.")
                pt = p
            elif isinstance(p, (tuple, list)) and len(p) == 2:
                pt = shapely.Point(float(p[0]), float(p[1]))
            else:
                raise TypeError(f"{name} must be (x, y) or a shapely Point.")
            pt = _reproject_to_2056(pt, crs)
            pts.append((pt.x, pt.y))
        (x1, y1), (x2, y2) = pts

        radars = _radars_from_gate_ids(data["gate_id"].to_numpy())
        cs_t = _lut_cs_table(self._require_archive_dir(), radars, beamwidth_deg=beamwidth_deg)
        cs_geom = _cross_section_gates(cs_t, (x1, y1), (x2, y2), beamwidth_deg=beamwidth_deg)

        gate_ids = cs_geom["gate_id"].to_numpy(dtype=np.int64) if len(cs_geom) else np.empty(0, dtype=np.int64)
        data_cs = _apply_gate_ids(data, gate_ids)

        # Unlike an AOI crop these columns are *not* LUT data: they are the
        # per-gate geometry of this particular section line, computed here and
        # available nowhere else, so they travel with the rows.
        geom_cols = [
            c for c in (
                "radar", "sweep", "azimuth", "range", "elevation_angle",
                "x_2056", "y_2056", "altitude",
                "d_near", "d_far", "z_near", "z_far", "d_center", "z_center",
                "cs_polygon",
            )
            if c in cs_geom.columns and (c == "cs_polygon" or c not in data_cs.columns)
        ]
        if not data_cs.is_empty() and len(cs_geom):
            data_cs = data_cs.join(
                _to_polars(cs_geom[["gate_id", *geom_cols]]), on="gate_id", how="left",
            )

        if quicklook:
            from raddb.viz.plot import plot_aoi_quicklook
            plot_aoi_quicklook(
                shapely.LineString([(x1, y1), (x2, y2)]),
                selected=data_cs, radars=radars, base_path=self.archive_dir,
            )
        return self._derive(data_cs)

    # ================================================================
    # PLOT
    # ================================================================

    @staticmethod
    def _save_fig(ret, save, kwargs):
        if not save:
            return
        import matplotlib.pyplot as plt
        fig = getattr(ret, "figure", None) or getattr(getattr(ret, "axes", None), "figure", None) or plt.gcf()
        fig.savefig(save, bbox_inches="tight", dpi=kwargs.get("dpi", 150))

    def plot_ppi(self, sweep: int | str = 1, variable: str = "DBZH", radar: str | None = None,
                 timestep=None, save: str | None = None, **kwargs):
        """Plot a PPI of one sweep from the loaded data. Returns the matplotlib artist."""
        from raddb.viz.plot import plot_ppi as _plot_ppi
        dt = self.to_datatree(radar=radar, timestep=timestep, label_column=variable)
        ret = _plot_ppi(dt, sweep=sweep, variable=variable, **kwargs)
        self._save_fig(ret, save, kwargs)
        return ret

    def plot_rhi(self, azimuth: float, variable: str = "DBZH", radar: str | None = None,
                 timestep=None, save: str | None = None, **kwargs):
        """Plot a pseudo-RHI through a radar site at ``azimuth``. Returns the matplotlib artist."""
        from raddb.viz.plot import plot_rhi as _plot_rhi
        dt = self.to_datatree(radar=radar, timestep=timestep, label_column=variable)
        r = radar or (self.radars()[0] if self.radars() else "")
        ret = _plot_rhi(dt, azimuth=azimuth, variable=variable, radar=r, **kwargs)
        self._save_fig(ret, save, kwargs)
        return ret

    def plot_cross_section(self, variable: str = "DBZH", radar: str | None = None,
                           timestep=None, save: str | None = None, **kwargs):
        """Plot a vertical cross-section from an :meth:`extract_cross_section` result."""
        from raddb.viz.plot import plot_cross_section as _plot_cs
        data = self.to_pandas()
        if "cs_polygon" not in data.columns:
            raise ValueError(
                "no 'cs_polygon' column; call extract_cross_section() first."
            )
        if "radar" in data.columns:
            present = sorted(data["radar"].dropna().unique())
            if radar is None:
                if len(present) > 1:
                    raise ValueError(f"data spans radars {present}; pass radar= to pick one.")
            else:
                data = data[data["radar"] == normalize_radar_name(radar)]
                if data.empty:
                    raise ValueError(f"no rows for radar {radar!r}.")
        if "volume_time" in data.columns and data["volume_time"].notna().any():
            vols = pd.to_datetime(data["volume_time"]).dropna().unique()
            if timestep is None:
                if len(vols) > 1:
                    raise ValueError(f"data holds {len(vols)} volumes; pass timestep= to pick one.")
            else:
                ts = pd.to_datetime(timestep)
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                chosen = min((pd.Timestamp(v) for v in vols), key=lambda v: abs(v - ts))
                data = data[pd.to_datetime(data["volume_time"]) == chosen]
        ret = _plot_cs(data, variable=variable, **kwargs)
        self._save_fig(ret, save, kwargs)
        return ret
