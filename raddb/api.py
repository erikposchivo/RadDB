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
network-specific pipeline (e.g. the private ``raddb.mch`` subpackage),
not here.
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
    open_any_datatree,
    parquet_to_dataframe,
    parquet_to_datatree,
    add_feature_to_df,
    add_feature_to_dt,
)
from raddb.helper import ensure_utc, filter_df, filter_dt, normalize_radar_name
from raddb.discovery import _parse_datatree_file_time, find_datatree_files
from raddb.lut import (
    RADAR_TO_IDX,
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
        base_path: str,
        network: str = "",
    ):
        """Initialize RadDB interface.

        Parameters
        ----------
        base_path : str
            **Output** base directory for the RadDB archive.  All LUT and
            POL parquet files will be stored under ``{base_path}/{radar}/``.
        network : str, optional
            Network label stored in the generated LUT info YAML.
        """
        self.base_path = Path(base_path)
        self.network = network

    # ================================================================
    # End-to-end archiving from DataTree files on disk
    # ================================================================

    def archive_from_datatrees(
        self,
        source: str | Path | list[str | Path],
        radar: str,
        start_time: str | datetime.datetime | None = None,
        end_time: str | datetime.datetime | None = None,
        filter_feature: str = "DBZH",
        filter_threshold: float = 0.0,
        filter_logic: str = ">",
        recursive: bool = True,
        engine: str | None = None,
        lut_ke: float = 1.25,
        projection_epsg: int | None = None,
        projection_crs=None,
        resume: bool = True,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> tuple[int, int]:
        """Archive DataTree files (NetCDF / Zarr) end to end.

        The public counterpart of the raw-ingestion pipelines: discovers
        DataTree files on disk, loads each volume with
        :func:`raddb.io_core.open_any_datatree`, generates the radar LUT
        from the first volume if missing, filters gates by
        ``filter_feature [filter_logic] filter_threshold``, and archives
        every volume to ``base_path``.

        Memory stays bounded: each volume is loaded, archived, and dropped
        before the next one is opened.  A checkpoint file
        (``{base_path}/_archive_checkpoint_datatrees_{radar}.txt``) records
        archived volumes so an interrupted run resumes where it left off.

        In-memory DataTrees don't need this method — archive them directly
        with :meth:`archive_volume` / :meth:`archive_multiple_volumes`.

        Parameters
        ----------
        source : str, Path or list
            Directory to scan for DataTree files (``*.nc`` / ``*.zarr``),
            a single file/store, or an explicit list of paths.
        radar : str
            Radar identifier — must resolve to a single letter A–Z
            (``gate_id`` encoding limit).
        start_time, end_time : str or datetime, optional
            Keep only files whose filename timestamp falls in this range
            (files without a parseable timestamp are kept).
        filter_feature, filter_threshold, filter_logic
            Gate filter defining *which data to keep* in the archive
            (default: ``DBZH > 0``).
        recursive : bool
            Recurse into subdirectories when ``source`` is a directory.
        engine : str, optional
            xarray backend override (e.g. ``"h5netcdf"``); auto-detected
            by default (``zarr`` for ``*.zarr`` stores).
        lut_ke : float
            Effective Earth radius scale factor for LUT generation
            (default ``1.25``).
        projection_epsg : int, optional
            EPSG code added to the generated LUT.
        projection_crs : pyproj.CRS or CRS-coercible, optional
            Alternative to ``projection_epsg``.
        resume : bool
            Skip volumes already recorded in the checkpoint file
            (default ``True``).  ``False`` re-archives everything.
        verbose : bool
            Print per-volume progress messages.
        show_progress : bool
            Show tqdm progress bars (if tqdm is installed).

        Returns
        -------
        tuple
            ``(n_archived, n_failed)`` totals.
        """
        # Validate before normalizing: normalize_radar_name() silently
        # truncates arbitrary names to their last character.
        r = radar.upper().strip()
        if r.startswith("ML") and len(r) == 3:
            r = r[-1]
        if len(r) != 1 or r not in RADAR_TO_IDX:
            raise ValueError(
                f"radar must be a single letter A-Z or 'ML<letter>' "
                f"(gate_id encoding limit); got {radar!r}"
            )
        radar = r

        # --- resolve input files ---
        if isinstance(source, (str, Path)):
            src = Path(source)
            if src.is_dir() and src.suffix.lower() != ".zarr":
                files = find_datatree_files(
                    src,
                    recursive=recursive,
                    start_time=start_time,
                    end_time=end_time,
                )
            else:
                files = [src]
        else:
            files = [Path(p) for p in source]
            if start_time is not None or end_time is not None:
                start_dt = ensure_utc(start_time) if start_time else None
                end_dt = ensure_utc(end_time) if end_time else None
                kept = []
                for f in files:
                    ts = _parse_datatree_file_time(f)
                    if ts is not None:
                        if start_dt and ts < start_dt:
                            continue
                        if end_dt and ts > end_dt:
                            continue
                    kept.append(f)
                files = kept

        if not files:
            logger.warning(
                "archive_from_datatrees: no DataTree files to archive."
            )
            return (0, 0)

        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(it, **kwargs):  # noqa: ANN001 - graceful fallback
                return it
            show_progress = False

        self.base_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = (
            self.base_path / f"_archive_checkpoint_datatrees_{radar}.txt"
        )
        checkpoint_seen = _load_checkpoint(checkpoint_path) if resume else set()

        print("=" * 70)
        print("RadDB archiving run (DataTree files)")
        print(f"  files     : {len(files)}")
        print(f"  radar     : {radar}")
        print(f"  filter    : {filter_feature} {filter_logic} {filter_threshold}")
        print(f"  base_path : {self.base_path}")
        print(f"  resuming  : {len(checkpoint_seen)} volume(s) already archived")
        print("=" * 70)

        run_t0 = time.time()

        # Generate the LUT up front from the first volume if missing; keep
        # the opened DataTree so the first volume is not opened twice.
        preopened: dict[Path, xr.DataTree] = {}
        lut_path = self.base_path / radar / "LUT" / f"{radar}_LUT.parquet"
        if not lut_path.exists():
            try:
                dt0 = open_any_datatree(files[0], engine=engine)
                preopened[files[0]] = dt0
                generate_lut_from_datatree(
                    dt0,
                    radar,
                    str(self.base_path),
                    ke=lut_ke,
                    network=self.network,
                    projection_epsg=projection_epsg,
                    projection_crs=projection_crs,
                )
            except Exception as e:
                print(f"  [{radar}] LUT generation failed: {e}")

        n_ok = n_fail = 0
        file_iter = (
            tqdm(files, desc=f"{radar} volumes", unit="vol")
            if show_progress
            else files
        )
        for f in file_iter:
            stem = Path(f).stem
            ckpt_key = f"{radar}:{stem}"
            if ckpt_key in checkpoint_seen:
                preopened.pop(Path(f), None)
                continue

            try:
                vol_dt = preopened.pop(Path(f), None)
                if vol_dt is None:
                    vol_dt = open_any_datatree(f, engine=engine)

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

        print("\n" + "=" * 70)
        print("Run complete")
        print(f"  {radar}: {n_ok} archived, {n_fail} failed")
        print(f"  checkpoint: {checkpoint_path}")
        print(f"  elapsed time: {_format_elapsed_time(time.time() - run_t0)}")
        print("=" * 70)
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
