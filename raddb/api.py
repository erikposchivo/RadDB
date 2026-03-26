"""
raddb/api.py
------------
High-level API for RadDB — a generic radar data archiving library.

The ``RadDB`` class provides a user-friendly interface for:
- Archiving xarray DataTree volumes to Parquet
- Generating and managing Look-Up Tables (LUT)
- Loading archived data as DataFrames or DataTrees
- Multi-radar support
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from raddb.pipeline import (
    archive_volume,
    archive_volumes,
    archive_volumes_dask,
    archive_volumes_multi_radar,
    filter_clear_sky,
)
from raddb.io_core import (
    parquet_to_dataframe,
    parquet_to_datatree,
    reconstruct_datatree,
)
from raddb.lut import (
    generate_lut_from_datatree,
    load_radar_lut,
    load_radar_info,
)
from raddb.helper import (
    read_parquet_dataset,
    normalize_radar_name,
    _find_polar_files_in_range,
    ensure_utc,
)

logger = logging.getLogger(__name__)

SWISS_RADARS = ["A", "D", "L", "P", "W"]


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
    >>> # Archive a volume
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
            Base directory for RadDB storage.  All LUT and POLAR parquet
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
        )

    def get_lut(self, radar: str) -> pd.DataFrame:
        """Load the LUT for a radar."""
        radar = normalize_radar_name(radar)
        return load_radar_lut(radar, self.base_path)

    def get_radar_info(self, radar: str) -> dict:
        """Load radar metadata (location, sweep geometry)."""
        radar = normalize_radar_name(radar)
        return load_radar_info(radar, self.base_path)

    # ================================================================
    # Archiving
    # ================================================================

    def archive_volume(
        self,
        dt: xr.DataTree,
        radar: str,
        dbzh_threshold: float = 0.0,
        timer=None,
        volume_label: str | None = None,
    ) -> str:
        """Archive a single DataTree volume to Parquet.

        Clear-sky gates (``DBZH <= dbzh_threshold``) are removed
        automatically.

        Parameters
        ----------
        dt : xr.DataTree
            Volume DataTree to archive.
        radar : str
            Radar identifier.
        dbzh_threshold : float
            Minimum DBZH to keep (default 0.0 removes clear sky).
        timer : StageTimer, optional
            Profiling timer.
        volume_label : str, optional
            Label for timer records.

        Returns
        -------
        str
            Path to saved POLAR parquet file.
        """
        radar = normalize_radar_name(radar)
        return archive_volume(
            dt=dt,
            radar=radar,
            base_output_path=str(self.base_path),
            dbzh_threshold=dbzh_threshold,
            timer=timer,
            volume=volume_label,
        )

    def archive_volumes(
        self,
        volumes: list[xr.DataTree] | dict[str, xr.DataTree],
        radar: str,
        dbzh_threshold: float = 0.0,
        use_dask: bool = False,
        verbose: bool = True,
        timer=None,
    ) -> list[dict]:
        """Archive multiple DataTree volumes.

        Parameters
        ----------
        volumes : list or dict of DataTree
            Volumes to archive.  If a dict, keys are used as labels.
        radar : str
            Radar identifier.
        dbzh_threshold : float
            Clear-sky threshold.
        use_dask : bool
            If True, parallelize with Dask.  Requires an active Dask cluster.
        verbose : bool
            Print progress messages.
        timer : StageTimer, optional
            Profiling timer (sequential mode only).

        Returns
        -------
        list of dict
            Results for each volume.
        """
        radar = normalize_radar_name(radar)
        if use_dask:
            return archive_volumes_dask(
                volumes=volumes,
                radar=radar,
                base_output_path=str(self.base_path),
                dbzh_threshold=dbzh_threshold,
                verbose=verbose,
            )
        return archive_volumes(
            volumes=volumes,
            radar=radar,
            base_output_path=str(self.base_path),
            dbzh_threshold=dbzh_threshold,
            verbose=verbose,
            timer=timer,
        )

    def archive_multi_radar(
        self,
        volumes_by_radar: dict[str, list[xr.DataTree] | dict[str, xr.DataTree]],
        dbzh_threshold: float = 0.0,
        use_dask: bool = False,
        verbose: bool = True,
        timer=None,
    ) -> dict[str, list[dict]]:
        """Archive volumes for multiple radars at once.

        Parameters
        ----------
        volumes_by_radar : dict
            Keys are radar names, values are lists/dicts of DataTrees.
            Example: ``{"A": [dt1, dt2], "D": [dt3]}``
        dbzh_threshold : float
            Clear-sky threshold.
        use_dask : bool
            If True, parallelize with Dask.
        verbose : bool
            Print progress.
        timer : StageTimer, optional
            Profiling timer (sequential mode only).

        Returns
        -------
        dict
            Keys are radar names, values are lists of result dicts.
        """
        return archive_volumes_multi_radar(
            volumes_by_radar=volumes_by_radar,
            base_output_path=str(self.base_path),
            dbzh_threshold=dbzh_threshold,
            use_dask=use_dask,
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

    def load_multi_radar_dataframe(
        self,
        radars: list[str] | str,
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
        radars : list of str, or "all"
            Radar names.  ``"all"`` loads all Swiss radars (A, D, L, P, W).
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
            if radars.upper() == "ALL":
                radars = SWISS_RADARS
            else:
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
    def filter_clear_sky(
        dt: xr.DataTree,
        threshold: float = 0.0,
        variable: str = "DBZH",
    ) -> xr.DataTree:
        """Remove clear-sky gates from a DataTree.

        Convenience static method wrapping
        :func:`raddb.pipeline.filter_clear_sky`.
        """
        return filter_clear_sky(dt, threshold=threshold, variable=variable)

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
    # Backwards compatibility aliases
    # ================================================================

    def load_parquet_data(self, *args, **kwargs) -> pd.DataFrame:
        """Backwards-compatible alias for :meth:`load_dataframe`."""
        return self.load_dataframe(*args, **kwargs)
