"""
raddb/api.py
------------
High-level API for RadDB — a generic radar data archiving library.

The ``RadDB`` class provides a user-friendly interface for:
- Archiving xarray DataTree volumes to Parquet
- Generating and managing Look-Up Tables (LUT)
- Loading archived data as DataFrames or DataTrees
- Multi-radar support

Note: network-specific constants such as a list of radar identifiers
(e.g. Swiss radars A, D, L, P, W) belong in the user script or in the
network-specific pipeline (e.g. ``mch_pipeline.py``), not here.
"""
from __future__ import annotations

import datetime
import logging
import time
from pathlib import Path

import pandas as pd
import xarray as xr

from raddb.io_core import (
    archive_volume,
    archive_multiple_volumes,
    archive_volumes_multi_radar,
    parquet_to_dataframe,
    parquet_to_datatree,
    add_feature_to_df,
    add_feature_to_dt,
)
from raddb.helper import filter_df, filter_dt, normalize_radar_name
from raddb.discovery import (
    _group_files_by_volume,
    _parse_volume_time,
    print_available_data,
)
from raddb.lut import (
    generate_lut_from_datatree,
    load_radar_lut,
    load_radar_info,
    add_lut_projection,
)

logger = logging.getLogger(__name__)


# ================================================================
# Private helpers for end-to-end archiving (ported from main.py)
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


class RadDB:
    """High-level interface for RadDB operations.

    RadDB is a **generic** radar data archiving library.  It takes xarray
    DataTrees (from any source) and archives them as Parquet files with
    an efficient LUT-based layout.

    Examples
    --------
    >>> db = RadDB(base_path="/data/raddb")
    >>>
    >>> # Generate LUT from a sample DataTree
    >>> db.generate_lut(radar="A", sample_datatree=dt_sample)
    >>>
    >>> # Archive a volume from a DataTree
    >>> db.archive_volume(dt, radar="A")
    >>>
    >>> # Load data back
    >>> dt = db.load_datatree(radar="A", start_time="2024-01-01", end_time="2024-01-02")
    >>> df = db.load_dataframe(radar="A", start_time="2024-01-01", end_time="2024-01-02")
    """

    def __init__(
        self,
        base_path: str = "/ltenas8/users/giacobbi/raddb",
        raw_data_path: str | None = None,
        network: str = "MCH_LTE",
        static_vis_dir: str | None = None,
        qpegrid_to_rad_dir: str | None = None,
    ):
        """Initialize RadDB interface.

        Parameters
        ----------
        base_path : str
            **Output** base directory for the RadDB archive.  All LUT and
            POL parquet files will be stored under ``{base_path}/{radar}/``.
        raw_data_path : str, optional
            **Input** root of the raw METRANET data tree (e.g.
            ``/ltenas8/data/RADAR``).  Required for
            :meth:`show_available_data` and :meth:`archive_from_raw`.
            Not needed when you archive DataTrees you already hold in
            memory (:meth:`archive_volume` & co.).
        network : str
            Network identifier for raw ingestion (default ``"MCH_LTE"``).
        static_vis_dir : str, optional
            Directory with static visibility LUTs (MCH ingestion).
        qpegrid_to_rad_dir : str, optional
            Directory with QPE-grid-to-radar LUTs (MCH ingestion; needed
            for HZT / temperature).
        """
        self.base_path = Path(base_path)
        self.raw_data_path = Path(raw_data_path) if raw_data_path else None
        self.network = network
        self.static_vis_dir = static_vis_dir
        self.qpegrid_to_rad_dir = qpegrid_to_rad_dir

    # ================================================================
    # Raw data discovery & end-to-end archiving
    # ================================================================

    def show_available_data(
        self,
        radars: list[str] | None = None,
        detail: bool = False,
    ) -> pd.DataFrame:
        """Print a summary of the raw data available for archiving.

        Scans ``raw_data_path`` (pure filesystem walk — no pyart needed)
        and prints, per radar: the covered period, number of days and
        volumes, and which products are present (POL / HYM / HZT).  Use
        this to decide which radar, time period, and filter to archive
        with :meth:`archive_from_raw`.

        Parameters
        ----------
        radars : list of str, optional
            Restrict the scan to these radar letters (e.g. ``["A", "L"]``).
        detail : bool
            Also print a per-day breakdown (volumes and time span per day).

        Returns
        -------
        pd.DataFrame
            One row per (radar, day) with ``n_volumes``, ``first_volume``,
            ``last_volume``, ``has_hym``, ``has_hzt`` — for programmatic use.
        """
        if self.raw_data_path is None:
            raise ValueError(
                "RadDB was initialized without raw_data_path. "
                "Pass raw_data_path=<METRANET root> when creating RadDB "
                "to use data discovery."
            )
        return print_available_data(self.raw_data_path, radars=radars, detail=detail)

    def archive_from_raw(
        self,
        radar: str | list[str],
        start_time: str | datetime.datetime,
        end_time: str | datetime.datetime,
        filter_feature: str = "DBZH",
        filter_threshold: float = 0.0,
        filter_logic: str = ">",
        hzt_enabled: bool = True,
        hym_enabled: bool = True,
        compute_pyart_hc: bool = True,
        projection_epsg: int | None = 2056,
        resume: bool = True,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> dict[str, tuple[int, int]]:
        """Run the full raw-METRANET → Parquet archiving pipeline.

        This is the end-to-end entry point (formerly ``main.py``): it walks
        the requested time range day by day, processes every volume with the
        MCH ingestion chain (visibility, KDP, attenuation, HZT, hydrometeor
        classification), filters gates by
        ``filter_feature [filter_logic] filter_threshold``, and archives
        each volume to ``base_path``.  The radar LUT is generated
        automatically on the first archived volume.

        Memory stays bounded: each volume is processed, archived, and
        dropped before the next one is loaded.  A checkpoint file
        (``{base_path}/_archive_checkpoint_{year}.txt``) records archived
        volumes so an interrupted run resumes where it left off.

        Requires ``raw_data_path`` at initialization, plus the optional
        MCH ancillary dirs (``static_vis_dir``, ``qpegrid_to_rad_dir``)
        for the corresponding processing stages.  Needs **pyart** and
        **radar_api** installed (imported only here, not by the package).

        Parameters
        ----------
        radar : str or list of str
            Radar letter(s): ``"A"``, ``"A,D"``, ``["A", "D"]`` or
            ``"all"`` for all Swiss radars (A, D, L, P, W).
        start_time, end_time : str or datetime
            Time range to archive (inclusive).
        filter_feature, filter_threshold, filter_logic
            Gate filter defining *which data to keep* in the archive
            (default: ``DBZH > 0``).
        hzt_enabled, hym_enabled, compute_pyart_hc : bool
            Toggle the optional MCH processing stages.
        projection_epsg : int, optional
            EPSG code added to the generated LUT (default ``2056``,
            CH1903+/LV95).  ``None`` disables the extra projection.
        resume : bool
            Skip volumes already recorded in the checkpoint file
            (default ``True``).  ``False`` re-archives everything.
        verbose : bool
            Print per-volume progress messages.
        show_progress : bool
            Show tqdm progress bars (if tqdm is installed).

        Returns
        -------
        dict
            ``{radar: (n_archived, n_failed)}`` totals.
        """
        try:
            from raddb import mch_pipeline as mch
        except ImportError as exc:
            raise ImportError(
                "archive_from_raw requires the MCH ingestion dependencies "
                "(pyart, radar_api, scipy). Install them or archive "
                "pre-built DataTrees with archive_volume() instead."
            ) from exc

        if self.raw_data_path is None:
            raise ValueError(
                "RadDB was initialized without raw_data_path — required "
                "by archive_from_raw."
            )

        # --- resolve radars ---
        if isinstance(radar, str):
            if radar.strip().lower() == "all":
                radars = list(mch.RADAR_LETTERS)
            else:
                radars = [r.strip().upper() for r in radar.split(",") if r.strip()]
        else:
            radars = [normalize_radar_name(r) for r in radar]
        unknown = [r for r in radars if r not in mch.RADAR_LETTERS]
        if unknown:
            raise ValueError(
                f"Unknown radar(s): {unknown}. Valid: {mch.RADAR_LETTERS}"
            )

        start = pd.Timestamp(start_time)
        end = pd.Timestamp(end_time)
        if end < start:
            raise ValueError("end_time must be >= start_time")

        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(it, **kwargs):  # noqa: ANN001 - graceful fallback
                return it
            show_progress = False

        self.base_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.base_path / f"_archive_checkpoint_{start.year}.txt"
        checkpoint_seen = _load_checkpoint(checkpoint_path) if resume else set()

        print("=" * 70)
        print("RadDB archiving run")
        print(f"  range     : {start} -> {end}")
        print(f"  radars    : {radars}")
        print(f"  filter    : {filter_feature} {filter_logic} {filter_threshold}")
        print(f"  base_path : {self.base_path}")
        print(f"  resuming  : {len(checkpoint_seen)} volume(s) already archived")
        print("=" * 70)

        run_t0 = time.time()
        totals: dict[str, list[int]] = {r: [0, 0] for r in radars}

        days = list(_iter_days(start, end))
        day_iter = tqdm(days, desc="days", unit="day") if show_progress else days
        for day_start, day_end in day_iter:
            for r in radars:
                n_ok, n_fail = self._archive_raw_day(
                    mch=mch,
                    radar=r,
                    day_start=day_start,
                    day_end=day_end,
                    filter_feature=filter_feature,
                    filter_threshold=filter_threshold,
                    filter_logic=filter_logic,
                    hzt_enabled=hzt_enabled,
                    hym_enabled=hym_enabled,
                    compute_pyart_hc=compute_pyart_hc,
                    projection_epsg=projection_epsg,
                    checkpoint_path=checkpoint_path,
                    checkpoint_seen=checkpoint_seen,
                    verbose=verbose,
                    show_progress=show_progress,
                )
                totals[r][0] += n_ok
                totals[r][1] += n_fail

        print("\n" + "=" * 70)
        print("Run complete")
        for r, (n_ok, n_fail) in totals.items():
            print(f"  {r}: {n_ok} archived, {n_fail} failed")
        print(f"  checkpoint: {checkpoint_path}")
        print(f"  elapsed time: {_format_elapsed_time(time.time() - run_t0)}")
        print("=" * 70)
        return {r: tuple(v) for r, v in totals.items()}

    def _archive_raw_day(
        self,
        *,
        mch,
        radar: str,
        day_start: pd.Timestamp,
        day_end: pd.Timestamp,
        filter_feature: str,
        filter_threshold: float,
        filter_logic: str,
        hzt_enabled: bool,
        hym_enabled: bool,
        compute_pyart_hc: bool,
        projection_epsg: int | None,
        checkpoint_path: Path,
        checkpoint_seen: set[str],
        verbose: bool,
        show_progress: bool,
    ) -> tuple[int, int]:
        """Process every volume for one (radar, day). Returns (n_ok, n_fail)."""
        try:
            pol_files = mch.find_files_with_fallback(
                network=self.network,
                radar=radar,
                start_time=day_start.to_pydatetime(),
                end_time=day_end.to_pydatetime(),
                product="POL",
                raw_data_dir=str(self.raw_data_path),
                verbose=verbose,
            )
        except Exception as e:
            print(f"  [{radar} {day_start:%Y-%m-%d}] find_files failed: {e}")
            return (0, 0)

        if not pol_files:
            return (0, 0)

        volumes = _group_files_by_volume(pol_files)
        n_ok = n_fail = 0

        items = sorted(volumes.items())
        if show_progress:
            from tqdm import tqdm
            items = tqdm(
                items, desc=f"{radar} {day_start:%Y-%m-%d}",
                unit="vol", leave=False,
            )

        # Generate the LUT up front from the first volume's sweeps if missing.
        # generate_mch_lut reads the raw files directly, so this also covers
        # resumed runs where every volume is checkpoint-skipped.
        lut_path = self.base_path / radar / "LUT" / f"{radar}_LUT.parquet"
        if not lut_path.exists() and volumes:
            first_sweep_paths = sorted(volumes.items())[0][1]
            try:
                mch.generate_mch_lut(
                    radar=radar,
                    network=self.network,
                    sample_volume_filepaths=first_sweep_paths,
                    output_base_path=str(self.base_path),
                    qpegrid_to_rad_dir=self.qpegrid_to_rad_dir,
                    projection_epsg=projection_epsg,
                )
            except Exception as e:
                print(f"  [{radar}] LUT generation failed: {e}")

        for stem, sweep_paths in items:
            ckpt_key = f"{radar}:{stem}"
            if ckpt_key in checkpoint_seen:
                continue

            try:
                volume_time = _parse_volume_time(stem)
                vol_dt = mch.process_mch_volume(
                    sweep_filepaths=sweep_paths,
                    network=self.network,
                    radar=radar,
                    volume_time=volume_time,
                    raw_data_dir=str(self.raw_data_path),
                    static_vis_dir=self.static_vis_dir,
                    qpegrid_to_rad_dir=self.qpegrid_to_rad_dir,
                    hzt_enabled=hzt_enabled,
                    hym_enabled=hym_enabled,
                    compute_pyart_hc=compute_pyart_hc,
                    verbose=verbose,
                    volume_label=stem,
                )

                self.archive_volume(
                    dt=vol_dt,
                    radar=radar,
                    filter_feature=filter_feature,
                    filter_threshold=filter_threshold,
                    filter_logic=filter_logic,
                    volume_label=stem,
                )

                _append_checkpoint(checkpoint_path, ckpt_key)
                checkpoint_seen.add(ckpt_key)
                n_ok += 1

                del vol_dt

            except Exception as e:
                n_fail += 1
                print(f"  [{radar}] FAIL {stem}: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()

        return (n_ok, n_fail)

    # ================================================================
    # LUT Generation & Access
    # ================================================================

    def generate_lut(
        self,
        radar: str,
        sample_datatree: xr.DataTree,
        ke: float = 1.25,
        network: str = "",
        projection_epsg: int | None = None,
        projection_crs=None,
    ) -> str:
        """Generate a Look-Up Table for a radar from a sample DataTree.

        The LUT stores static spatial information (gate coordinates,
        elevation angles, etc.) and is generated **once per radar**.
        Subsequent calls skip regeneration if the LUT already exists.

        Parameters
        ----------
        radar : str
            Radar name (e.g. ``"A"``).
        sample_datatree : xr.DataTree
            A single complete volume scan used to derive the radar geometry.
        ke : float
            Scale factor for Earth's effective radius (default 1.25).
        network : str, optional
            Network identifier stored in the metadata YAML.
        projection_epsg : int, optional
            EPSG code for an additional projected coordinate system
            (e.g. ``2056`` for CH1903+ / LV95).  Adds ``x_{epsg}`` /
            ``y_{epsg}`` columns via :func:`add_lut_projection`.
        projection_crs : pyproj.CRS or CRS-coercible, optional
            Alternative to ``projection_epsg``.

        Returns
        -------
        str
            Path to the generated LUT parquet file.
        """
        radar = normalize_radar_name(radar)
        return generate_lut_from_datatree(
            dt=sample_datatree,
            radar=radar,
            output_base_path=str(self.base_path),
            ke=ke,
            network=network,
            projection_epsg=projection_epsg,
            projection_crs=projection_crs,
        )

    def get_lut(self, radar: str) -> pd.DataFrame:
        """Load the LUT for a radar."""
        radar = normalize_radar_name(radar)
        return load_radar_lut(radar, self.base_path)

    def get_radar_info(self, radar: str) -> dict:
        """Load radar metadata (location, sweep geometry)."""
        radar = normalize_radar_name(radar)
        return load_radar_info(radar, self.base_path)

    def add_lut_projection(
        self,
        radar: str,
        epsg: int | None = None,
        crs=None,
    ) -> pd.DataFrame:
        """Add projected coordinates to the LUT of a radar.

        Loads the LUT, converts lat/lon to the target CRS, and returns
        the enriched DataFrame.  The LUT file on disk is **not** modified;
        save the returned DataFrame yourself if persistence is needed.

        Parameters
        ----------
        radar : str
            Radar identifier.
        epsg : int, optional
            EPSG code for the target CRS.
            Example: ``2056`` for CH1903+ / LV95 (Swiss national grid).
        crs : pyproj.CRS or any CRS-coercible object, optional
            Alternative to ``epsg``.

        Returns
        -------
        pd.DataFrame
            LUT with appended ``x_{suffix}`` / ``y_{suffix}`` columns.

        Examples
        --------
        >>> lut_ch = db.add_lut_projection(radar="A", epsg=2056)
        >>> lut_ch[["x_2056", "y_2056"]].head()
        """
        radar = normalize_radar_name(radar)
        lut_df = load_radar_lut(radar, self.base_path)
        return add_lut_projection(lut_df, epsg=epsg, crs=crs)

    # ================================================================
    # Archiving
    # ================================================================

    def archive_volume(
        self,
        dt: xr.DataTree,
        radar: str,
        filter_feature: str = "DBZH",
        filter_threshold: float = 0.0,
        filter_logic: str = ">",
        timer=None,
        volume_label: str | None = None,
    ) -> str:
        """Archive a single DataTree volume to Parquet.

        Gates that do not satisfy
        ``filter_feature [filter_logic] filter_threshold`` are dropped
        (row removal — zeros in surviving gates are preserved as zeros).

        Parameters
        ----------
        dt : xr.DataTree
            Volume DataTree to archive.
        radar : str
            Radar identifier.
        filter_feature : str
            Column to filter on (default ``"DBZH"``).
        filter_threshold : float
            Threshold value (default ``0.0``).
        filter_logic : str
            Comparison operator (default ``">"``).
        timer : StageTimer, optional
            Profiling timer.
        volume_label : str, optional
            Label for timer records.

        Returns
        -------
        str
            Path to saved POL parquet file.
        """
        radar = normalize_radar_name(radar)
        return archive_volume(
            dt=dt,
            radar=radar,
            base_output_path=str(self.base_path),
            filter_feature=filter_feature,
            filter_threshold=filter_threshold,
            filter_logic=filter_logic,
            timer=timer,
            volume=volume_label,
        )

    def archive_multiple_volumes(
        self,
        volumes: list[xr.DataTree] | dict[str, xr.DataTree],
        radar: str,
        filter_feature: str = "DBZH",
        filter_threshold: float = 0.0,
        filter_logic: str = ">",
        verbose: bool = True,
        timer=None,
    ) -> list[dict]:
        """Archive multiple DataTree volumes sequentially.

        Parameters
        ----------
        volumes : list or dict of DataTree
            Volumes to archive.  If a dict, keys are used as labels.
        radar : str
            Radar identifier.
        filter_feature : str
            Column to filter on (default ``"DBZH"``).
        filter_threshold : float
            Threshold value (default ``0.0``).
        filter_logic : str
            Comparison operator (default ``">"``).
        verbose : bool
            Print progress messages.
        timer : StageTimer, optional
            Profiling timer.

        Returns
        -------
        list of dict
            Results for each volume.
        """
        radar = normalize_radar_name(radar)
        return archive_multiple_volumes(
            volumes=volumes,
            radar=radar,
            base_output_path=str(self.base_path),
            filter_feature=filter_feature,
            filter_threshold=filter_threshold,
            filter_logic=filter_logic,
            verbose=verbose,
            timer=timer,
        )

    def archive_multi_radar(
        self,
        volumes_by_radar: dict[str, list[xr.DataTree] | dict[str, xr.DataTree]],
        filter_feature: str = "DBZH",
        filter_threshold: float = 0.0,
        filter_logic: str = ">",
        verbose: bool = True,
        timer=None,
    ) -> dict[str, list[dict]]:
        """Archive volumes for multiple radars sequentially.

        Parameters
        ----------
        volumes_by_radar : dict
            Keys are radar names, values are lists/dicts of DataTrees.
            Example: ``{"A": [dt1, dt2], "D": [dt3]}``
        filter_feature : str
            Column to filter on (default ``"DBZH"``).
        filter_threshold : float
            Threshold value (default ``0.0``).
        filter_logic : str
            Comparison operator (default ``">"``).
        verbose : bool
            Print progress.
        timer : StageTimer, optional
            Profiling timer.

        Returns
        -------
        dict
            Keys are radar names, values are lists of result dicts.
        """
        return archive_volumes_multi_radar(
            volumes_by_radar=volumes_by_radar,
            base_output_path=str(self.base_path),
            filter_feature=filter_feature,
            filter_threshold=filter_threshold,
            filter_logic=filter_logic,
            verbose=verbose,
            timer=timer,
        )

    # ================================================================
    # Data Loading
    # ================================================================

    def load_dataframe(
        self,
        radar: str,
        start_time: str | datetime.datetime | None = None,
        end_time: str | datetime.datetime | None = None,
        columns: list[str] | None = None,
        merge_lut: bool = False,
    ) -> pd.DataFrame:
        """Load archived data as a pandas DataFrame.

        Parameters
        ----------
        radar : str
            Radar name (e.g. ``"A"`` or ``"MLA"``).
        start_time, end_time : optional
            Time range filter.
        columns : list of str, optional
            Columns to load.
        merge_lut : bool
            If True, merge with LUT to include spatial coordinates.

        Returns
        -------
        pd.DataFrame
        """
        radar = normalize_radar_name(radar)
        return parquet_to_dataframe(
            radar=radar,
            base_path=self.base_path,
            start_time=start_time,
            end_time=end_time,
            columns=columns,
            merge_lut=merge_lut,
        )

    def load_datatree(
        self,
        radar: str,
        start_time: str | datetime.datetime,
        end_time: str | datetime.datetime,
        label_column: str = "DBZH",
        max_workers: int = 1,
    ) -> xr.DataTree:
        """Load archived data and reconstruct an xarray DataTree.

        Loads all volumes in the time range, joins with the LUT to
        recover spatial coordinates, and reconstructs a DataTree.

        Parameters
        ----------
        radar : str
            Radar name.
        start_time, end_time : str or datetime
            Time range.
        label_column : str
            Column to use for reconstruction.
        max_workers : int
            Parallel workers for sweep reconstruction.

        Returns
        -------
        xr.DataTree
        """
        radar = normalize_radar_name(radar)
        return parquet_to_datatree(
            radar=radar,
            base_path=self.base_path,
            start_time=start_time,
            end_time=end_time,
            label_column=label_column,
            max_workers=max_workers,
        )

    def parquet_to_dt(
        self,
        parquet_file: str,
        radar: str | None = None,
        label_column: str = "DBZH",
        max_workers: int = 1,
    ) -> xr.DataTree:
        """Load a single POL parquet file and reconstruct a DataTree.

        Convenience method for loading from a direct file path rather than
        a time range.  The LUT is looked up automatically under ``base_path``.

        Parameters
        ----------
        parquet_file : str or Path
            Path to a ``*_POL.parquet`` file.
        radar : str, optional
            Radar identifier (e.g. ``"A"``).  Inferred from the filename if
            omitted.
        label_column : str
            Feature column used for reconstruction (default ``"DBZH"``).
        max_workers : int
            Parallel workers for sweep reconstruction.

        Returns
        -------
        xr.DataTree
        """
        parquet_file = Path(parquet_file)
        if radar is None:
            radar = parquet_file.name.split("_")[0]
        radar = normalize_radar_name(radar)

        # Parse the volume timestamp from the filename:
        # {radar}_{YYYYMMDD}_{HHMMSS}_POL.parquet
        stem = parquet_file.stem.replace("_POL", "")
        parts = stem.split("_")
        ts = pd.to_datetime(parts[-2] + "_" + parts[-1], format="%Y%m%d_%H%M%S")

        return parquet_to_datatree(
            radar=radar,
            base_path=self.base_path,
            start_time=ts,
            end_time=ts,
            label_column=label_column,
            max_workers=max_workers,
        )

    def load_multi_radar_dataframe(
        self,
        radars: list[str],
        start_time: str | datetime.datetime | None = None,
        end_time: str | datetime.datetime | None = None,
        columns: list[str] | None = None,
        merge_lut: bool = False,
    ) -> pd.DataFrame:
        """Load data from multiple radars into a single DataFrame.

        A ``radar`` column is added to identify which radar each row
        belongs to.

        Parameters
        ----------
        radars : list of str
            Radar names to load (e.g. ``["A", "D", "L"]``).
        start_time, end_time : optional
            Time range filter.
        columns : list of str, optional
            Columns to load.
        merge_lut : bool
            If True, merge with LUT.

        Returns
        -------
        pd.DataFrame
        """
        if isinstance(radars, str):
            radars = [radars]

        dfs = []
        for radar in radars:
            radar = normalize_radar_name(radar)
            df = self.load_dataframe(
                radar=radar,
                start_time=start_time,
                end_time=end_time,
                columns=columns,
                merge_lut=merge_lut,
            )
            if not df.empty:
                df["radar"] = radar
                dfs.append(df)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    # ================================================================
    # Utilities
    # ================================================================

    @staticmethod
    def filter_dt(
        dt: xr.DataTree,
        feature: str = "DBZH",
        threshold: float = 0.0,
        logic: str = ">",
    ) -> xr.DataTree:
        """Mask gates in a DataTree that do not satisfy the filter condition.

        Convenience static method wrapping :func:`raddb.helper.filter_dt`.

        Gates where ``feature [logic] threshold`` is False are set to NaN
        across all variables.  Gates where the condition is True keep their
        original values unchanged (including legitimate zero values).

        Parameters
        ----------
        dt : xr.DataTree
        feature : str
            Variable to filter on (default ``"DBZH"``).
        threshold : float
        logic : str
            ``'>'``, ``'>='``, ``'<'``, ``'<='``, ``'=='``, ``'!='``.
        """
        return filter_dt(dt, feature=feature, threshold=threshold, logic=logic)

    @staticmethod
    def filter_df(
        df: pd.DataFrame,
        feature: str = "DBZH",
        threshold: float = 0.0,
        logic: str = ">",
    ) -> pd.DataFrame:
        """Drop rows from a DataFrame that do not satisfy the filter condition.

        Convenience static method wrapping :func:`raddb.helper.filter_df`.

        Parameters
        ----------
        df : pd.DataFrame
        feature : str
            Column to filter on (default ``"DBZH"``).
        threshold : float
        logic : str
            ``'>'``, ``'>='``, ``'<'``, ``'<='``, ``'=='``, ``'!='``.
        """
        return filter_df(df, feature=feature, threshold=threshold, logic=logic)

    @staticmethod
    def add_feature_to_df(
        df: pd.DataFrame,
        feature_name: str,
        compute_fn: callable,
    ) -> pd.DataFrame:
        """Add a new column to a DataFrame computed from existing columns.

        Convenience static method wrapping :func:`raddb.io_core.add_feature_to_df`.
        """
        return add_feature_to_df(df, feature_name=feature_name, compute_fn=compute_fn)

    @staticmethod
    def add_feature_to_dt(
        dt: xr.DataTree,
        feature_name: str,
        compute_fn: callable,
    ) -> xr.DataTree:
        """Add a new variable to every sweep in a DataTree.

        Convenience static method wrapping :func:`raddb.io_core.add_feature_to_dt`.
        """
        return add_feature_to_dt(dt, feature_name=feature_name, compute_fn=compute_fn)

    def list_available_radars(self) -> list[str]:
        """List radar identifiers that have data in the archive."""
        if not self.base_path.exists():
            return []
        return sorted(
            p.name
            for p in self.base_path.iterdir()
            if p.is_dir() and len(p.name) == 1 and p.name.isalpha()
        )

    # ================================================================
    # Plotting
    # ================================================================

    def plot_ppi(
        self,
        radar: str,
        timestep: str | datetime.datetime | pd.Timestamp,
        sweep: int | str,
        variable: str,
        **kwargs,
    ):
        """Plot a PPI for one volume loaded from the archive.

        Parameters
        ----------
        radar : str
            Radar identifier (e.g. ``"D"``).
        timestep : str or datetime
            Volume timestamp. A single volume at this time is loaded.
        sweep : int or str
            Sweep index (``3``) or group name (``"sweep_3"``).
        variable : str
            Variable to plot (``"DBZH"``, ``"ZDR"``, ``"HC_MCH"``, ...).
        **kwargs
            Forwarded to :func:`raddb.plot.plot_ppi` (``ax``, ``coords``,
            ``use_cartopy``, ``vmin``, ``vmax``, ``cmap``, ...).
        """
        from raddb.viz.plot import plot_ppi as _plot_ppi
        ts = pd.to_datetime(timestep)
        dt = self.load_datatree(
            radar=radar, start_time=ts, end_time=ts, label_column=variable,
        )
        return _plot_ppi(dt, sweep=sweep, variable=variable, **kwargs)

    def plot_rhi(
        self,
        radar: str,
        timestep: str | datetime.datetime | pd.Timestamp,
        azimuth: float,
        variable: str,
        **kwargs,
    ):
        """Plot a pseudo-RHI cross-section through a volume PPI scan.

        Replicates ``pyart.util.cross_section_ppi`` + ``plot_rhi``: picks the
        nearest ray per sweep to ``azimuth``, regrids to a common range axis,
        and renders gate edges (4/3 Earth) with ``shading="flat"``.

        Parameters
        ----------
        radar : str
        timestep : str or datetime
            Volume timestamp.
        azimuth : float
            Cross-section azimuth (degrees, 0 = North, clockwise).
        variable : str
        **kwargs
            Forwarded to :func:`raddb.plot.plot_rhi`.
        """
        from raddb.viz.plot import plot_rhi as _plot_rhi
        ts = pd.to_datetime(timestep)
        dt = self.load_datatree(
            radar=radar, start_time=ts, end_time=ts, label_column=variable,
        )
        return _plot_rhi(dt, azimuth=azimuth, variable=variable, radar=radar, **kwargs)

    def plot_cross_section_ppi(
        self,
        radar: str,
        timestep: str | datetime.datetime | pd.Timestamp,
        azimuth: float,
        variable: str,
        **kwargs,
    ):
        """Backwards-compatibility alias for :meth:`plot_rhi`."""
        return self.plot_rhi(
            radar=radar, timestep=timestep, azimuth=azimuth,
            variable=variable, **kwargs,
        )

    # ================================================================
    # Backwards compatibility aliases
    # ================================================================

    def load_parquet_data(self, *args, **kwargs) -> pd.DataFrame:
        """Backwards-compatible alias for :meth:`load_dataframe`."""
        return self.load_dataframe(*args, **kwargs)
