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
from pathlib import Path

import pandas as pd
import xarray as xr

from raddb.pipeline import (
    archive_volume,
    archive_multiple_volumes,
    archive_volumes_multi_radar,
    filter_dt,
    filter_df,
)
from raddb.io_core import (
    parquet_to_dataframe,
    parquet_to_datatree,
    reconstruct_datatree,
    add_feature_to_df,
    add_feature_to_dt,
)
from raddb.lut import (
    generate_lut_from_datatree,
    load_radar_lut,
    load_radar_info,
    add_lut_projection,
)
from raddb.helper import (
    normalize_radar_name,
    _find_polar_files_in_range,
    ensure_utc,
)

logger = logging.getLogger(__name__)


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
    ):
        """Initialize RadDB interface.

        Parameters
        ----------
        base_path : str
            Base directory for RadDB storage.  All LUT and POL parquet
            files will be stored under ``{base_path}/{radar}/``.
        """
        self.base_path = Path(base_path)

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

        Convenience static method wrapping :func:`raddb.pipeline.filter_dt`.

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

        Convenience static method wrapping :func:`raddb.pipeline.filter_df`.

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
        from raddb.plot import plot_ppi as _plot_ppi
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
        """Plot a RHI (vertical cross-section) for one volume.

        Parameters
        ----------
        radar : str
            Radar identifier (e.g. ``"D"``).
        timestep : str or datetime
            Volume timestamp.
        azimuth : float
            Cross-section azimuth in degrees (0..360, 0 = North, clockwise).
        variable : str
            Variable to plot.
        **kwargs
            Forwarded to :func:`raddb.plot.plot_rhi` (``az_tol``,
            ``max_range_km``, ``ax``, ``vmin``, ``vmax``, ``cmap``, ...).
        """
        from raddb.plot import plot_rhi as _plot_rhi
        ts = pd.to_datetime(timestep)
        dt = self.load_datatree(
            radar=radar, start_time=ts, end_time=ts, label_column=variable,
        )
        return _plot_rhi(dt, azimuth=azimuth, variable=variable, **kwargs)

    # ================================================================
    # Backwards compatibility aliases
    # ================================================================

    def load_parquet_data(self, *args, **kwargs) -> pd.DataFrame:
        """Backwards-compatible alias for :meth:`load_dataframe`."""
        return self.load_dataframe(*args, **kwargs)
