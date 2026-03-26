"""
raddb/pipeline.py
-----------------
Generic archiving pipeline for radar DataTrees.

This module handles the conversion from xarray DataTree to archived
Parquet files.  It is **network-agnostic** — any DataTree with the
standard xradar layout (sweep_N groups with azimuth/range coordinates
and polarimetric variables) can be archived.

Features:
- Clear-sky filtering (remove gates with DBZH <= threshold)
- Single-volume archiving
- Batch archiving with Dask parallelization
- Multi-radar support
"""
from __future__ import annotations

import logging
import time as _time
from contextlib import nullcontext as _nullctx
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from raddb.io_core import (
    POLAR_COLUMNS,
    _save_polar_parquet,
    datatree_to_dataframe,
)
from raddb.lut import generate_gate_id
from raddb.helper import (
    list_sweep_names,
    normalize_radar_name,
    StageTimer,
    _vprint,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CLEAR-SKY FILTER
# ============================================================================

def filter_clear_sky(
    dt: xr.DataTree,
    threshold: float = 0.0,
    variable: str = "DBZH",
) -> xr.DataTree:
    """Remove clear-sky gates from a DataTree.

    Sets gates where ``variable <= threshold`` to NaN across **all**
    data variables in each sweep.  This significantly reduces parquet
    file sizes when archiving.

    Parameters
    ----------
    dt : xr.DataTree
        Input DataTree with sweep groups.
    threshold : float
        Gates with ``variable <= threshold`` are considered clear sky.
        Default is 0.0 (removes all DBZH <= 0 dBZ).
    variable : str
        Variable to threshold on (default ``"DBZH"``).

    Returns
    -------
    xr.DataTree
        New DataTree with clear-sky gates set to NaN.
    """
    sweep_names = list_sweep_names(dt)
    dict_ds = {}

    for sweep_name in sweep_names:
        ds = dt[sweep_name].to_dataset()
        if variable in ds:
            mask = ds[variable] <= threshold
            ds = ds.where(~mask)
        dict_ds[sweep_name] = ds

    return xr.DataTree.from_dict(dict_ds)


# ============================================================================
# SINGLE-VOLUME ARCHIVING
# ============================================================================

def archive_volume(
    dt: xr.DataTree,
    radar: str,
    base_output_path: str,
    dbzh_threshold: float = 0.0,
    timer=None,
    volume: str | None = None,
) -> str:
    """Archive a single DataTree volume to Parquet format.

    Converts the DataTree to a DataFrame, generates ``gate_id`` for each
    gate (linking back to the LUT), filters out clear-sky gates
    (``DBZH <= dbzh_threshold``), and saves the result as a POLAR parquet
    file.

    Parameters
    ----------
    dt : xr.DataTree
        Processed volume (any radar network).
    radar : str
        Radar identifier (single letter, e.g. ``"A"``).
    base_output_path : str
        Base output directory for parquet files.
    dbzh_threshold : float
        Minimum DBZH value to keep.  Gates with ``DBZH <= threshold``
        are excluded (clear-sky removal).  Default 0.0.
    timer : StageTimer, optional
        Profiling timer.
    volume : str, optional
        Volume label for timer records.

    Returns
    -------
    str
        Path to the saved POLAR parquet file.
    """
    radar = normalize_radar_name(radar)

    with (
        timer.time_stage("datatree_to_df", volume=volume)
        if timer
        else _nullctx()
    ):
        df = datatree_to_dataframe(dt)

    # Generate gate_id to link back to the central LUT
    with (
        timer.time_stage("generate_gate_ids", volume=volume)
        if timer
        else _nullctx()
    ):
        df["gate_id"] = df.apply(
            lambda row: generate_gate_id(
                radar,
                int(row["sweep"]),
                float(row["azimuth"]),
                float(row["range"]),
            ),
            axis=1,
        )
        # Clear-sky removal
        df_polar_full = df[df["DBZH"] > dbzh_threshold].copy()
        df_polar = df_polar_full[
            [c for c in POLAR_COLUMNS if c in df_polar_full.columns]
        ].copy()

    with (
        timer.time_stage("save_parquet", volume=volume)
        if timer
        else _nullctx()
    ):
        return _save_polar_parquet(df_polar, radar, base_output_path)


# ============================================================================
# BATCH ARCHIVING (sequential)
# ============================================================================

def archive_volumes(
    volumes: list[xr.DataTree] | dict[str, xr.DataTree],
    radar: str,
    base_output_path: str,
    dbzh_threshold: float = 0.0,
    verbose: bool = True,
    timer: StageTimer | None = None,
) -> list[dict]:
    """Archive multiple DataTree volumes sequentially.

    Parameters
    ----------
    volumes : list or dict
        If a list, each element is a DataTree.
        If a dict, keys are volume labels and values are DataTrees.
    radar : str
        Radar identifier.
    base_output_path : str
        Base output directory.
    dbzh_threshold : float
        Clear-sky threshold.
    verbose : bool
        Print progress.
    timer : StageTimer, optional
        Profiling timer.

    Returns
    -------
    list of dict
        Results with keys: label, success, error, polar_path, n_gates.
    """
    radar = normalize_radar_name(radar)

    if isinstance(volumes, dict):
        items = list(volumes.items())
    else:
        items = [(f"volume_{i}", dt) for i, dt in enumerate(volumes)]

    results = []
    pipeline_t0 = _time.perf_counter()

    for i, (label, dt) in enumerate(items, 1):
        _vprint(
            f"\n>>  Volume {i}/{len(items)}: {label}",
            verbose,
        )

        result = {
            "label": label,
            "radar": radar,
            "success": False,
            "error": None,
            "n_gates": 0,
        }

        vol_t0 = _time.perf_counter()
        try:
            with (
                timer.time_stage("archive_volume", volume=label)
                if timer
                else _nullctx()
            ):
                polar_path = archive_volume(
                    dt,
                    radar=radar,
                    base_output_path=base_output_path,
                    dbzh_threshold=dbzh_threshold,
                    timer=timer,
                    volume=label,
                )

            result["success"] = True
            result["polar_path"] = polar_path

            df_polar = pd.read_parquet(polar_path)
            result["n_gates"] = len(df_polar)

            vol_elapsed = _time.perf_counter() - vol_t0
            _vprint(
                f"OK  Volume {i}/{len(items)} done in "
                f"{vol_elapsed:.1f}s -- {result['n_gates']:,} gates saved",
                verbose,
            )

        except Exception as e:
            vol_elapsed = _time.perf_counter() - vol_t0
            result["error"] = str(e)
            _vprint(
                f"FAIL  Volume {i}/{len(items)} FAILED in "
                f"{vol_elapsed:.1f}s: {e}",
                verbose,
            )
            logger.error(f"[{i}/{len(items)}] {label} - FAIL: {e}")

        results.append(result)

    total_elapsed = _time.perf_counter() - pipeline_t0
    n_ok = sum(1 for r in results if r["success"])
    _vprint(
        f"\nArchiving complete: {n_ok}/{len(results)} volumes "
        f"in {total_elapsed:.1f}s",
        verbose,
    )
    return results


# ============================================================================
# BATCH ARCHIVING WITH DASK
# ============================================================================

def archive_volumes_dask(
    volumes: list[xr.DataTree] | dict[str, xr.DataTree],
    radar: str,
    base_output_path: str,
    dbzh_threshold: float = 0.0,
    verbose: bool = True,
) -> list[dict]:
    """Archive multiple DataTree volumes in parallel using Dask.

    The user **must** initialize a Dask cluster before calling this
    function.  Example::

        from dask.distributed import Client
        client = Client(n_workers=4, threads_per_worker=1)
        # Monitor at http://localhost:8787/status

        results = raddb.archive_volumes_dask(volumes, radar="A", ...)

    Parameters
    ----------
    volumes : list or dict
        DataTrees to archive.
    radar : str
        Radar identifier.
    base_output_path : str
        Base output directory.
    dbzh_threshold : float
        Clear-sky threshold.
    verbose : bool
        Print progress.

    Returns
    -------
    list of dict
        Results for each volume.
    """
    import dask

    radar = normalize_radar_name(radar)

    if isinstance(volumes, dict):
        items = list(volumes.items())
    else:
        items = [(f"volume_{i}", dt) for i, dt in enumerate(volumes)]

    @dask.delayed
    def _try_archive(label, dt_vol):
        result = {
            "label": label,
            "radar": radar,
            "success": False,
            "error": None,
            "n_gates": 0,
        }
        try:
            with dask.config.set(scheduler="synchronous"):
                polar_path = archive_volume(
                    dt_vol,
                    radar=radar,
                    base_output_path=base_output_path,
                    dbzh_threshold=dbzh_threshold,
                )
            result["success"] = True
            result["polar_path"] = polar_path
            df_polar = pd.read_parquet(polar_path)
            result["n_gates"] = len(df_polar)
        except Exception as e:
            result["error"] = (
                f"Archive of volume {label} failed: {e}"
            )
        return result

    tasks = [_try_archive(label, dt) for label, dt in items]

    _vprint(
        f"Submitting {len(tasks)} volume(s) to Dask...", verbose
    )
    _vprint(
        "Monitor at the Dask dashboard "
        "(default: http://localhost:8787/status)",
        verbose,
    )

    t_i = _time.perf_counter()
    results = list(dask.compute(*tasks))
    elapsed = _time.perf_counter() - t_i

    n_ok = sum(1 for r in results if r["success"])
    n_fail = sum(1 for r in results if not r["success"])
    _vprint(
        f"Dask archiving complete: {n_ok} succeeded, {n_fail} failed "
        f"in {elapsed:.1f}s",
        verbose,
    )

    for r in results:
        if r["error"]:
            logger.error(r["error"])
            _vprint(f"  ERROR: {r['error']}", verbose)

    return results


# ============================================================================
# MULTI-RADAR ARCHIVING
# ============================================================================

def archive_volumes_multi_radar(
    volumes_by_radar: dict[str, list[xr.DataTree] | dict[str, xr.DataTree]],
    base_output_path: str,
    dbzh_threshold: float = 0.0,
    use_dask: bool = False,
    verbose: bool = True,
    timer: StageTimer | None = None,
) -> dict[str, list[dict]]:
    """Archive volumes for multiple radars.

    Parameters
    ----------
    volumes_by_radar : dict
        Keys are radar names, values are lists or dicts of DataTrees.
        Example: ``{"A": [dt1, dt2], "D": [dt3, dt4]}``
    base_output_path : str
        Base output directory.
    dbzh_threshold : float
        Clear-sky threshold.
    use_dask : bool
        If True, use Dask for parallel archiving.  Requires an active
        Dask cluster.
    verbose : bool
        Print progress.
    timer : StageTimer, optional
        Profiling timer (only used with sequential archiving).

    Returns
    -------
    dict
        Keys are radar names, values are lists of result dicts.
    """
    all_results = {}

    for radar, volumes in volumes_by_radar.items():
        radar = normalize_radar_name(radar)
        _vprint(
            f"\n{'='*60}\nArchiving radar {radar}\n{'='*60}",
            verbose,
        )

        if use_dask:
            results = archive_volumes_dask(
                volumes,
                radar=radar,
                base_output_path=base_output_path,
                dbzh_threshold=dbzh_threshold,
                verbose=verbose,
            )
        else:
            results = archive_volumes(
                volumes,
                radar=radar,
                base_output_path=base_output_path,
                dbzh_threshold=dbzh_threshold,
                verbose=verbose,
                timer=timer,
            )

        all_results[radar] = results

    return all_results
