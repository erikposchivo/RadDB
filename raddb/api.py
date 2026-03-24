"""
raddb/api.py
------------
High-level API for easy access to RadDB functionality.
"""
from __future__ import annotations
import datetime
from pathlib import Path
import pandas as pd
import xarray as xr
import logging

from raddb.pipeline import process_and_archive_metranet, process_metranet_volume
from raddb.io_core import reconstruct_datatree
from raddb.lut import generate_radar_lut, load_radar_lut, load_radar_info
from raddb.helper import read_parquet_dataset, normalize_radar_name, _find_polar_files_in_range

logger = logging.getLogger(__name__)


class RadDB:
    """
    High-level interface for RadDB operations.

    Examples
    --------
    >>> raddb = RadDB(base_path="/ltenas8/users/giacobbi/raddb")
    >>>
    >>> # Process and store radar data
    >>> raddb.process_and_store(
    ...     radar="MLA",
    ...     start_time="2021-08-28 06:00",
    ...     end_time="2021-08-28 18:00"
    ... )
    >>>
    >>> # Load data for analysis
    >>> dt = raddb.load_datatree(
    ...     radar="MLA",
    ...     start_time="2021-08-28 12:00",
    ...     end_time="2021-08-28 13:00"
    ... )
    """

    def __init__(
        self,
        base_path: str = "/ltenas8/users/giacobbi/raddb",
        raw_data_dir: str | None = None,
        network: str = "MCH_LTE",
        static_vis_dir: str = "/ltenas8/data/Rad4Alp_LUTs/static_vis",
        qpegrid_to_rad_dir: str = "/ltenas8/data/Rad4Alp_LUTs/qpegrid_to_rad",
    ):
        """
        Initialize RadDB interface.

        Parameters
        ----------
        base_path : str
            Base directory for RadDB storage (output directory for processed data)
        raw_data_dir : str, optional
            Base directory for raw METRANET data files.
            If None, will use radar_api configuration file.
        network : str
            Radar network identifier
        static_vis_dir : str
            Directory with static visibility LUTs
        qpegrid_to_rad_dir : str
            Directory with QPE grid LUTs
        """
        self.base_path = Path(base_path)
        self.raw_data_dir = raw_data_dir
        self.network = network
        self.static_vis_dir = static_vis_dir
        self.qpegrid_to_rad_dir = qpegrid_to_rad_dir

    def process_and_store(
        self,
        radar: str,
        start_time: str | datetime.datetime,
        end_time: str | datetime.datetime,
        hzt_enabled: bool = True,
        hym_enabled: bool = True,
        compute_pyart_hc: bool = True,
        product: str = "POL",
        max_workers: int = 1,
        verbose: bool = True,
    ) -> list[dict]:
        """
        Process METRANET radar data and store to parquet.

        Parameters
        ----------
        radar : str
            Radar name (e.g., "MLA", "A"). Will be normalized to single letter.
        start_time : str or datetime
            Start time
        end_time : str or datetime
            End time
        hzt_enabled : bool
            Process HZT (isotherm height)
        hym_enabled : bool
            Include operational hydrometeor classification
        compute_pyart_hc : bool
            Compute PyART hydrometeor classification
        product : str
            Product type
        max_workers : int
            Number of parallel workers
        verbose : bool
            Verbose output

        Returns
        -------
        list of dict
            Processing results for each volume
        """
        # Normalize radar name to single letter
        radar = normalize_radar_name(radar)

        return process_and_archive_metranet(
            network=self.network,
            radar=radar,
            start_time=start_time,
            end_time=end_time,
            base_output_path=str(self.base_path),
            raw_data_dir=self.raw_data_dir,
            static_vis_dir=self.static_vis_dir,
            qpegrid_to_rad_dir=self.qpegrid_to_rad_dir,
            hzt_enabled=hzt_enabled,
            hym_enabled=hym_enabled,
            compute_pyart_hc=compute_pyart_hc,
            product=product,
            max_workers=max_workers,
            verbose=verbose,
        )

    def generate_lut(
        self,
        radar: str,
        sample_volume_filepaths: list[str],
        ke: float = 1.25,
    ) -> str:
        """
        Generate Look-Up Table for a radar.

        Parameters
        ----------
        radar : str
            Radar name (e.g., "MLA", "A"). Will be normalized to single letter.
        sample_volume_filepaths : list of str
            List of file paths for one complete volume (all sweeps)
        ke : float
            Scale factor for Earth's effective radius

        Returns
        -------
        str
            Path to generated LUT parquet file
        """
        # Normalize radar name to single letter
        radar = normalize_radar_name(radar)

        return generate_radar_lut(
            radar=radar,
            network=self.network,
            sample_volume_filepaths=sample_volume_filepaths,
            output_base_path=str(self.base_path),
            qpegrid_to_rad_dir=self.qpegrid_to_rad_dir,
            ke=ke,
        )

    def load_parquet_data(
        self,
        radar: str,
        start_time: str | datetime.datetime | None = None,
        end_time: str | datetime.datetime | None = None,
        columns: list[str] | None = None,
        data_type: str = "POLAR",
    ) -> pd.DataFrame:
        """
        Load parquet data for a radar and time range.

        Parameters
        ----------
        radar : str
            Radar name (e.g., "MLA", "A"). Will be normalized to single letter.
        start_time : str or datetime, optional
            Start time for filtering
        end_time : str or datetime, optional
            End time for filtering
        columns : list of str, optional
            Columns to load
        data_type : str
            "POLAR" for dual-pol data or "LUT" for look-up table

        Returns
        -------
        pd.DataFrame
            Loaded data
        """
        # Normalize radar name to single letter
        radar = normalize_radar_name(radar)

        radar_path = self.base_path / radar
        pattern = f"**/*{data_type}.parquet"

        if not radar_path.exists():
            logger.warning(f"Radar directory not found: {radar_path}")
            return pd.DataFrame()

        df = read_parquet_dataset(
            base_path=radar_path,
            pattern=pattern,
            columns=columns,
            verbose=True,
        )

        if df.empty:
            logger.warning(f"No {data_type} data found for radar {radar}")
            return df

        # Filter by time if requested
        if start_time or end_time:
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
                if start_time:
                    df = df[df["time"] >= pd.to_datetime(start_time)]
                if end_time:
                    df = df[df["time"] <= pd.to_datetime(end_time)]

        return df

    def load_datatree(
        self,
        radar: str,
        start_time: str | datetime.datetime,
        end_time: str | datetime.datetime,
        label_column: str = "HC_PYART",
    ) -> xr.DataTree:
        """
        Load and reconstruct a DataTree from parquet files.

        Loads the **latest** volume in the given time range, joins it
        with the central LUT (to recover azimuth/range coordinates),
        and reconstructs an xarray DataTree.

        Parameters
        ----------
        radar : str
            Radar name (e.g., "MLA", "A"). Will be normalized to single letter.
        start_time : str or datetime
            Start time
        end_time : str or datetime
            End time
        label_column : str
            Column to use for reconstruction

        Returns
        -------
        xr.DataTree
            Reconstructed DataTree

        Raises
        ------
        ValueError
            If no data is found for the specified time range
        FileNotFoundError
            If LUT or radar info files are not found
        """
        # Normalize radar name to single letter
        radar = normalize_radar_name(radar)

        # Check LUT / info existence
        lut_path = self.base_path / radar / "LUT" / f"{radar}_LUT.parquet"
        info_path = self.base_path / radar / "LUT" / f"{radar}_info.yaml"

        if not lut_path.exists():
            raise FileNotFoundError(f"LUT not found at {lut_path}. Run generate_lut() first.")
        if not info_path.exists():
            raise FileNotFoundError(f"Radar info not found at {info_path}.")

        # Find POLAR parquet files in the time range
        radar_path = self.base_path / radar
        polar_files = _find_polar_files_in_range(radar_path, start_time, end_time)

        if not polar_files:
            raise ValueError(f"No POLAR data found for {radar} between {start_time} and {end_time}")

        # Load the latest volume file
        df_polar = pd.read_parquet(polar_files[-1], engine="pyarrow")

        if df_polar.empty:
            raise ValueError(f"Latest POLAR file is empty for {radar}")

        # Join with LUT to recover azimuth / range coordinates
        lut_df = pd.read_parquet(lut_path, engine="pyarrow")
        df_joined = df_polar.merge(
            lut_df[["gate_id", "azimuth", "range"]],
            on="gate_id",
            how="left",
        )
        df_joined = df_joined.dropna(subset=["azimuth", "range"])

        if df_joined.empty:
            raise ValueError("No matching gates found between POLAR data and LUT.")

        # Reconstruct datatree
        dt = reconstruct_datatree(
            df_joined=df_joined,
            lut_path=lut_path,
            radar_info_path=info_path,
            label_column=label_column,
        )

        return dt

    def get_radar_info(self, radar: str) -> dict:
        """
        Get radar metadata.

        Parameters
        ----------
        radar : str
            Radar name (e.g., "MLA", "A"). Will be normalized to single letter.

        Returns
        -------
        dict
            Radar metadata
        """
        # Normalize radar name to single letter
        radar = normalize_radar_name(radar)
        return load_radar_info(radar, self.base_path)

    def get_lut(self, radar: str) -> pd.DataFrame:
        """
        Get radar Look-Up Table.

        Parameters
        ----------
        radar : str
            Radar name (e.g., "MLA", "A"). Will be normalized to single letter.

        Returns
        -------
        pd.DataFrame
            Radar LUT
        """
        # Normalize radar name to single letter
        radar = normalize_radar_name(radar)
        return load_radar_lut(radar, self.base_path)
