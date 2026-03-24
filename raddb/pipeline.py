"""
raddb/pipeline.py
-----------------
High-level pipeline orchestrator for METRANET radar data processing using PyART.
"""
from __future__ import annotations
import concurrent.futures
import datetime
import logging
import glob
from collections import defaultdict
from pathlib import Path

import pandas as pd
import xarray as xr
import radar_api
import numpy as np

from raddb.radar_processing import (
    load_metranet_sweep,
    hzt_hourly_to_5min,
    read_static_visibility,
)
from raddb.io_core import pyart_volume_to_datatree, datatree_to_dataframe, _save_polar_parquet, POLAR_COLUMNS
from raddb.lut import generate_gate_id
from raddb.helper import list_sweep_names, normalize_radar_name

logger = logging.getLogger(__name__)


# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def _parse_volume_time(stem: str) -> datetime.datetime:
    """Parse volume timestamp from METRANET filename."""
    try:
        y, j, h, m = int(stem[3:5]), int(stem[5:8]), int(stem[8:10]), int(stem[10:12])
        return datetime.datetime(2000 + y, 1, 1) + datetime.timedelta(days=j - 1, hours=h, minutes=m)
    except Exception:
        return datetime.datetime(1970, 1, 1)


def _group_files_by_volume(paths: list[str]) -> dict:
    """Group sweep files by volume (based on filename stem)."""
    vols = defaultdict(list)
    for p in paths:
        stem = Path(p).stem
        vols[stem].append(p)
    return dict(vols)


def _raw_data_dir_candidates(raw_data_dir: str | None, network: str) -> list[str | None]:
    """Return candidate base directories for radar_api.find_files.

    Some users pass a directory already including the network folder
    (for example, ".../RADAR/MCH"). For MCH_LTE, radar_api local patterns
    already append "MCH", so we also try the parent directory fallback.
    """
    if raw_data_dir is None:
        return [None]

    raw_path = Path(raw_data_dir)
    candidates: list[str | None] = [str(raw_path)]

    network_prefix = network.split("_", 1)[0].upper() if isinstance(network, str) else ""
    if raw_path.name.upper() == network_prefix:
        candidates.append(str(raw_path.parent))

    # Preserve order while removing duplicates.
    seen = set()
    unique_candidates: list[str | None] = []
    for candidate in candidates:
        if candidate not in seen:
            unique_candidates.append(candidate)
            seen.add(candidate)
    return unique_candidates


def _find_files_with_fallback(
    *,
    network: str,
    radar: str,
    start_time,
    end_time,
    product: str,
    raw_data_dir: str | None,
    verbose: bool = False,
) -> list[str]:
    """Find files and retry with a parent directory fallback when needed."""
    last_error = None

    for candidate_base_dir in _raw_data_dir_candidates(raw_data_dir=raw_data_dir, network=network):
        try:
            fpaths = radar_api.find_files(
                network=network,
                radar=radar,
                start_time=start_time,
                end_time=end_time,
                product=product,
                protocol="local",
                base_dir=candidate_base_dir,
            )
        except Exception as e:
            last_error = e
            continue

        if fpaths:
            if verbose and raw_data_dir is not None and candidate_base_dir != raw_data_dir:
                logger.info(
                    "No files found with raw_data_dir='%s'. Retrying with '%s' succeeded.",
                    raw_data_dir,
                    candidate_base_dir,
                )
            return fpaths

    if last_error is not None:
        raise last_error
    return []


def find_hzt_files(radar: str, network: str, volume_time: datetime.datetime, raw_data_dir: str | None = None) -> dict:
    """Find HZT files around the volume time."""
    hzt_start = volume_time - datetime.timedelta(hours=1)
    hzt_end = volume_time + datetime.timedelta(hours=1)

    try:
        hzt_files = _find_files_with_fallback(
            network=network,
            radar=radar,
            start_time=hzt_start,
            end_time=hzt_end,
            product="HZT",
            raw_data_dir=raw_data_dir,
        )
    except Exception as e:
        logger.warning(f"Error finding HZT files: {e}")
        return {}

    # Group by timestamp
    hzt_dict = {}
    if hzt_files and isinstance(hzt_files, list):
        for hzt_file in hzt_files:
            try:
                # Parse HZT timestamp (assuming format similar to METRANET)
                stem = Path(hzt_file).stem
                hzt_time = _parse_volume_time(stem)
                hzt_dict[hzt_time] = hzt_file
            except Exception as e:
                logger.warning(f"Error parsing HZT file {hzt_file}: {e}")
                continue

    return hzt_dict


def find_hydroclass_files(sweep_filepaths: list[str]) -> dict[int, str]:
    """Find hydrometeor classification files corresponding to sweeps."""
    hym_paths = {}
    for sweep_path in sweep_filepaths:
        norm = Path(sweep_path)
        sweep_num = int(norm.suffix[1:])

        # Build HYM file pattern
        hym_ext = ".8" + norm.suffix[2:]
        hym_dir = norm.parent.parent / norm.parent.name.replace("ML", "YM")
        hym_pattern = str(hym_dir / (norm.stem.replace("ML", "YM")[:-2] + "*L" + hym_ext))

        matches = glob.glob(hym_pattern)
        if matches:
            hym_paths[sweep_num] = matches[-1]

    return hym_paths


# ================================================================================
# VOLUME PROCESSING
# ================================================================================

def process_metranet_volume(
    sweep_filepaths: list[str],
    network: str,
    radar: str,
    volume_time: datetime.datetime,
    raw_data_dir: str | None = None,
    static_vis_dir: str | None = None,
    qpegrid_to_rad_dir: str | None = None,
    hzt_enabled: bool = True,
    hym_enabled: bool = True,
    compute_pyart_hc: bool = True,
    verbose: bool = False,
) -> xr.DataTree:
    """
    Process a complete METRANET volume scan.

    Steps:
    1. List and group radar files by volume
    2. Load static visibility for each sweep
    3. Find and interpolate HZT files
    4. Find HYM files
    5. Process each sweep with PyART (visibility, KDP, attenuation, HC)
    6. Convert all radar objects to xradar datatree

    Parameters
    ----------
    sweep_filepaths : list
        List of file paths for all sweeps in the volume
    network : str
        Network identifier
    radar : str
        Radar name
    volume_time : datetime
        Volume scan timestamp
    raw_data_dir : str, optional
        Base directory for raw METRANET data (for finding HZT files)
    static_vis_dir : str, optional
        Directory containing static visibility LUTs
    qpegrid_to_rad_dir : str, optional
        Directory containing QPE grid to radar LUTs
    hzt_enabled : bool
        Whether to include HZT data
    hym_enabled : bool
        Whether to include operational hydrometeor classification
    compute_pyart_hc : bool
        Whether to compute PyART hydrometeor classification
    verbose : bool
        Verbose logging

    Returns
    -------
    xr.DataTree
        Processed volume as xradar DataTree
    """
    # Normalize radar name to single letter (handles both "MLA" and "A" formats)
    radar_letter = normalize_radar_name(radar)

    # Load static visibility
    vis_dict = {}
    if static_vis_dir:
        try:
            vis_dict = read_static_visibility(radar_letter, static_vis_dir, verbose=verbose)
        except Exception as e:
            logger.warning(f"Could not load visibility for radar {radar_letter}: {e}")

    # Find and interpolate HZT
    hzt_interpolated_dict = {}
    if hzt_enabled and qpegrid_to_rad_dir:
        try:
            hzt_files = find_hzt_files(radar, network, volume_time, raw_data_dir=raw_data_dir)
            if verbose:
                logger.info(f"Found {len(hzt_files) if isinstance(hzt_files, dict) else 0} HZT files")

            if isinstance(hzt_files, dict) and len(hzt_files) >= 2:
                hzt_interpolated_dict = hzt_hourly_to_5min(hzt_files, tsteps_min=5)
            else:
                if verbose:
                    logger.warning(f"Need at least 2 HZT files for interpolation, got {len(hzt_files) if isinstance(hzt_files, dict) else 'invalid type'}")
        except Exception as e:
            logger.warning(f"Could not process HZT: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    # Find HYM files
    hym_paths = {}
    if hym_enabled:
        hym_paths = find_hydroclass_files(sweep_filepaths)

    # Process each sweep
    rad_objs = {}
    sorted_paths = sorted(sweep_filepaths)

    for sweep_filepath in sorted_paths:
        sweep_num = int(Path(sweep_filepath).suffix[1:])

        # Get visibility for this sweep
        visibility = vis_dict.get(sweep_num) if vis_dict else None

        # Get HZT cartesian for this time - find closest timestamp
        hzt_cartesian = None
        if hzt_interpolated_dict:
            # Find closest timestamp in HZT dict
            hzt_times = list(hzt_interpolated_dict.keys())
            if hzt_times:
                closest_time = min(hzt_times, key=lambda t: abs((t - volume_time).total_seconds()))
                hzt_cartesian = hzt_interpolated_dict[closest_time]

        # Get HYM file
        hym_file = hym_paths.get(sweep_num)

        # Process sweep
        try:
            rad_obj, x, y, z = load_metranet_sweep(
                radar_fpath=sweep_filepath,
                hydroclassif_fpath=hym_file,
                hzt_cartesian=hzt_cartesian,
                visibility=visibility,
                qpegrid_to_rad_dir=qpegrid_to_rad_dir,
                compute_pyart_hc=compute_pyart_hc,
                verbose=verbose,
            )
            rad_objs[sweep_num] = rad_obj
        except Exception as e:
            logger.error(f"Error processing sweep {sweep_num}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    if not rad_objs:
        raise ValueError("No sweeps were successfully processed")

    # Convert to xradar DataTree
    dt = pyart_volume_to_datatree(rad_objs)
    return dt


# ================================================================================
# ARCHIVAL TO PARQUET
# ================================================================================

def archive_volume_to_parquet(
    dt: xr.DataTree,
    radar: str,
    base_output_path: str,
    dbzh_threshold: float = 0.0,
) -> str:
    """
    Archive a processed volume DataTree to parquet format.

    Only saves polarimetric (POLAR) data. The static LUT is generated
    once per radar via ``generate_radar_lut`` and stored separately.

    Parameters
    ----------
    dt : xr.DataTree
        Processed volume
    radar : str
        Radar name
    base_output_path : str
        Base output directory
    dbzh_threshold : float
        Minimum DBZH value to include in POLAR file

    Returns
    -------
    str
        Path to saved POLAR parquet file
    """
    df = datatree_to_dataframe(dt)

    # Generate gate_id to link back to the central LUT
    df["gate_id"] = df.apply(
        lambda row: generate_gate_id(radar, int(row["sweep"]), float(row["azimuth"]), float(row["range"])),
        axis=1
    )

    # Extract only POLAR columns and filter by DBZH threshold
    df_polar_full = df[df["DBZH"] > dbzh_threshold].copy()
    df_polar = df_polar_full[[c for c in POLAR_COLUMNS if c in df_polar_full.columns]].copy()

    return _save_polar_parquet(df_polar, radar, base_output_path)


# ================================================================================
# BATCH PROCESSING
# ================================================================================

def process_and_archive_metranet(
    network: str,
    radar: str,
    start_time: str | datetime.datetime,
    end_time: str | datetime.datetime,
    base_output_path: str,
    raw_data_dir: str | None = None,
    static_vis_dir: str | None = None,
    qpegrid_to_rad_dir: str | None = None,
    hzt_enabled: bool = True,
    hym_enabled: bool = True,
    compute_pyart_hc: bool = True,
    product: str = "POL",
    max_workers: int = 1,
    verbose: bool = True,
) -> list[dict]:
    """
    Process and archive METRANET radar data for a time range.

    Parameters
    ----------
    network : str
        Network identifier (e.g., "MCH_LTE")
    radar : str
        Radar name (e.g., "MLA")
    start_time : str or datetime
        Start time
    end_time : str or datetime
        End time
    base_output_path : str
        Base output directory for parquet files
    raw_data_dir : str, optional
        Base directory for raw METRANET data files.
        If None, will use radar_api configuration file.
    static_vis_dir : str, optional
        Directory with static visibility LUTs
    qpegrid_to_rad_dir : str, optional
        Directory with QPE grid LUTs
    hzt_enabled : bool
        Include HZT processing
    hym_enabled : bool
        Include operational HC
    compute_pyart_hc : bool
        Compute PyART HC
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

    # Find files
    try:
        fps = _find_files_with_fallback(
            network=network,
            radar=radar,
            start_time=start_time,
            end_time=end_time,
            product=product,
            raw_data_dir=raw_data_dir,
            verbose=verbose,
        )
    except Exception as e:
        logger.error(f"Error finding files for radar {radar}: {e}")
        return []

    if not fps:
        logger.warning(f"No files found for radar {radar} between {start_time} and {end_time}")
        return []

    # Group by volume
    volumes = _group_files_by_volume(fps)
    logger.info(f"Found {len(volumes)} volumes to process")

    # Process each volume
    results = []
    for i, (stem, sweep_paths) in enumerate(volumes.items(), 1):
        vol_time = _parse_volume_time(stem)
        if verbose:
            logger.info(f"Processing volume {i}/{len(volumes)}: {stem} -> {vol_time}, {len(sweep_paths)} sweeps")

        result = {
            "stem": stem,
            "time": vol_time,
            "success": False,
            "error": None,
            "n_gates": 0,
            "n_sweeps": len(sweep_paths),
        }

        try:
            if verbose:
                logger.info(f"Calling process_metranet_volume for {vol_time}")

            # Process volume
            dt = process_metranet_volume(
                sweep_filepaths=sweep_paths,
                network=network,
                radar=radar,
                volume_time=vol_time,
                raw_data_dir=raw_data_dir,
                static_vis_dir=static_vis_dir,
                qpegrid_to_rad_dir=qpegrid_to_rad_dir,
                hzt_enabled=hzt_enabled,
                hym_enabled=hym_enabled,
                compute_pyart_hc=compute_pyart_hc,
                verbose=verbose,
            )

            if verbose:
                logger.info(f"Volume processed successfully, archiving to parquet...")

            # Archive to parquet (only POLAR — LUT is generated once separately)
            polar_path = archive_volume_to_parquet(dt, radar, base_output_path)

            result["success"] = True
            result["polar_path"] = polar_path

            # Count gates
            df_polar = pd.read_parquet(polar_path)
            result["n_gates"] = len(df_polar)

            if verbose:
                logger.info(f"[{i}/{len(volumes)}] {vol_time} - OK ({result['n_gates']} gates)")

        except Exception as e:
            result["error"] = str(e)
            if verbose:
                logger.error(f"[{i}/{len(volumes)}] {vol_time} - FAIL: {e}")
                import traceback
                traceback.print_exc()
            else:
                logger.error(f"[{i}/{len(volumes)}] {vol_time} - FAIL: {e}")

        results.append(result)

    return results


# Backwards compatibility alias
archive_metranet_to_parquet = process_and_archive_metranet
