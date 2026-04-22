"""
raddb/pipeline.py
-----------------
Generic archiving pipeline for radar DataTrees.

This module handles the conversion from xarray DataTree to archived
Parquet files.  It is **network-agnostic** — any DataTree with the
standard xradar layout (sweep_N groups with azimuth/range coordinates
and polarimetric variables) can be archived.

Features:
- Flexible filtering (any feature, threshold, and comparison logic)
- Single-volume archiving
- Sequential batch archiving
- Multi-radar support
- Utilities to add new features to DataTree or DataFrame
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
    _cast_hc_column,
    _POLAR_FLOAT32_COLS,
    _compute_gate_temperature,
)
from raddb.lut import RADAR_TO_IDX
from raddb.helper import (
    list_sweep_names,
    normalize_radar_name,
    StageTimer,
    _vprint,
)

logger = logging.getLogger(__name__)


# ============================================================================
# FILTER LOGIC REGISTRY
# ============================================================================

FILTER_LOGICS: dict[str, callable] = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">":  lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<":  lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
}


# ============================================================================
# DATAFRAME FILTER
# ============================================================================

def filter_df(
    df: pd.DataFrame,
    feature: str = "DBZH",
    threshold: float = 0.0,
    logic: str = ">",
) -> pd.DataFrame:
    """Filter a DataFrame, keeping rows where ``feature [logic] threshold``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature : str
        Column name to filter on (default ``"DBZH"``).
    threshold : float
        Comparison value.
    logic : str
        Comparison operator: ``'>'``, ``'>='``, ``'<'``, ``'<='``,
        ``'=='``, ``'!='``.  Default ``'>'``.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with non-matching rows dropped (index reset).

    Raises
    ------
    KeyError
        If ``feature`` is not a column of ``df``.
    ValueError
        If ``logic`` is not one of the supported operators.
    """
    fn = FILTER_LOGICS.get(logic)
    if fn is None:
        raise ValueError(
            f"Unknown logic '{logic}'. Choose from: {list(FILTER_LOGICS)}"
        )
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found in DataFrame columns.")
    mask = fn(df[feature].to_numpy(), threshold)
    return df[mask].reset_index(drop=True)


# ============================================================================
# DATATREE FILTER
# ============================================================================

def filter_dt(
    dt: xr.DataTree,
    feature: str = "DBZH",
    threshold: float = 0.0,
    logic: str = ">",
) -> xr.DataTree:
    """Filter a DataTree, masking gates where ``feature [logic] threshold`` is False.

    Gates that do **not** satisfy the condition are set to NaN across all
    data variables in each sweep.  Gates that *do* satisfy the condition
    keep their original values unchanged — including legitimate zero values.

    .. note::
        This operates on the multidimensional DataTree structure via
        ``xr.Dataset.where()``.  For tabular (row-level) filtering use
        :func:`filter_df` instead, which drops non-matching rows entirely.

    Parameters
    ----------
    dt : xr.DataTree
        Input DataTree with ``sweep_N`` groups.
    feature : str
        Variable name to use as the filter criterion (default ``"DBZH"``).
    threshold : float
        Comparison value.
    logic : str
        Comparison operator: ``'>'``, ``'>='``, ``'<'``, ``'<='``,
        ``'=='``, ``'!='``.  Default ``'>'``.

    Returns
    -------
    xr.DataTree
        New DataTree where non-matching gates have NaN for all variables.
        Matching gates are left entirely unchanged (zeros remain zeros).

    Raises
    ------
    ValueError
        If ``logic`` is not a supported operator.
    """
    fn = FILTER_LOGICS.get(logic)
    if fn is None:
        raise ValueError(
            f"Unknown logic '{logic}'. Choose from: {list(FILTER_LOGICS)}"
        )

    sweep_names = list_sweep_names(dt)
    dict_ds = {}

    for sweep_name in sweep_names:
        ds = dt[sweep_name].to_dataset()
        if feature in ds:
            keep_mask = fn(ds[feature], threshold)
            ds = ds.where(keep_mask)
        dict_ds[sweep_name] = ds

    return xr.DataTree.from_dict(dict_ds)


# ============================================================================
# SINGLE-VOLUME ARCHIVING
# ============================================================================

def archive_volume(
    dt: xr.DataTree,
    radar: str,
    base_output_path: str,
    filter_feature: str = "DBZH",
    filter_threshold: float = 0.0,
    filter_logic: str = ">",
    timer=None,
    volume: str | None = None,
) -> str:
    """Archive a single DataTree volume to Parquet format.

    Converts the DataTree to a DataFrame, generates a ``gate_id`` for each
    gate (linking back to the LUT), drops gates that do not satisfy
    ``filter_feature [filter_logic] filter_threshold``, and saves the
    result as a POL parquet file.

    Non-matching gates are **dropped** (row removal), so zero values in
    surviving gates are never converted to NaN.  NaN values in the
    reconstruction come only from gates absent in the parquet (i.e. gates
    that were filtered out or had no data in the original DataTree).

    HZT availability check: if ``"HZT"`` is not present in the DataTree,
    ``"HC_PYART"`` is also skipped because it requires HZT to be meaningful.

    Parameters
    ----------
    dt : xr.DataTree
        Processed volume (any radar network).
    radar : str
        Radar identifier (single letter, e.g. ``"A"``).
    base_output_path : str
        Base output directory for parquet files.
    filter_feature : str
        Column to use for gate filtering (default ``"DBZH"``).
    filter_threshold : float
        Threshold value for the filter (default ``0.0``).
    filter_logic : str
        Comparison operator (default ``">"``).
    timer : StageTimer, optional
        Profiling timer.
    volume : str, optional
        Volume label for timer records.

    Returns
    -------
    str
        Path to the saved POL parquet file.
    """
    radar = normalize_radar_name(radar)
    fn = FILTER_LOGICS.get(filter_logic)
    if fn is None:
        raise ValueError(
            f"Unknown filter_logic '{filter_logic}'. "
            f"Choose from: {list(FILTER_LOGICS)}"
        )

    with (
        timer.time_stage("datatree_to_df", volume=volume)
        if timer
        else _nullctx()
    ):
        df = datatree_to_dataframe(dt)

    with (
        timer.time_stage("generate_gate_ids", volume=volume)
        if timer
        else _nullctx()
    ):
        # Step 1: filter mask — row dropping, NOT NaN conversion.
        # This preserves zero values in surviving gates as zeros.
        if filter_feature in df.columns:
            _mask = fn(df[filter_feature].to_numpy(), filter_threshold)
        else:
            logger.warning(
                "filter_feature '%s' not found in DataFrame; keeping all gates.",
                filter_feature,
            )
            _mask = np.ones(len(df), dtype=bool)

        # Step 2: gate_ids on surviving rows only (pure numpy int64)
        _radar_idx = np.int64(RADAR_TO_IDX[radar.upper()])
        _sweep_v   = df["sweep"].to_numpy(dtype=np.int64)[_mask]
        _az_int    = np.round(
            df["azimuth"].to_numpy(dtype=np.float64)[_mask] * 10
        ).astype(np.int64)
        _rng_int   = df["range"].to_numpy(dtype=np.int64)[_mask]
        _gate_ids  = (
            _radar_idx * np.int64(1_000_000_000_000)
            + _sweep_v  * np.int64(   10_000_000_000)
            + _az_int   * np.int64(        1_000_000)
            + _rng_int
        )

        # Step 3: column selection.
        # Skip HC_PYART when HZT is not available (HC_PYART requires HZT).
        _hzt_available = "HZT" in df.columns
        if not _hzt_available:
            logger.debug(
                "HZT not available in DataTree — skipping HC_PYART column."
            )
        _polar_cols = [
            c for c in POLAR_COLUMNS
            if c in df.columns
            and c != "gate_id"
            and not (c == "HC_PYART" and not _hzt_available)
        ]

        # Step 4: build output DataFrame from numpy arrays (no intermediate copy)
        df_polar = pd.DataFrame(
            {
                "gate_id": _gate_ids,
                **{c: df[c].to_numpy()[_mask] for c in _polar_cols},
            }
        )

    with (
        timer.time_stage("save_parquet", volume=volume)
        if timer
        else _nullctx()
    ):
        for col in list(df_polar.columns):
            if col == "HC_MCH":
                df_polar[col] = _cast_hc_column(df_polar[col], shift=1)
            elif col == "HC_PYART":
                df_polar[col] = _cast_hc_column(df_polar[col], shift=0)
            elif col in _POLAR_FLOAT32_COLS:
                df_polar[col] = df_polar[col].astype(np.float32)
        _temp = _compute_gate_temperature(df, _mask)
        if _temp is not None:
            df_polar["TEMP"] = _temp
        return _save_polar_parquet(df_polar, radar, base_output_path)


# ============================================================================
# BATCH ARCHIVING (sequential)
# ============================================================================

def archive_multiple_volumes(
    volumes: list[xr.DataTree] | dict[str, xr.DataTree],
    radar: str,
    base_output_path: str,
    filter_feature: str = "DBZH",
    filter_threshold: float = 0.0,
    filter_logic: str = ">",
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
    filter_feature : str
        Column to use for gate filtering (default ``"DBZH"``).
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
        _vprint(f"\n>>  Volume {i}/{len(items)}: {label}", verbose)

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
                    filter_feature=filter_feature,
                    filter_threshold=filter_threshold,
                    filter_logic=filter_logic,
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
# MULTI-RADAR ARCHIVING
# ============================================================================

def archive_volumes_multi_radar(
    volumes_by_radar: dict[str, list[xr.DataTree] | dict[str, xr.DataTree]],
    base_output_path: str,
    filter_feature: str = "DBZH",
    filter_threshold: float = 0.0,
    filter_logic: str = ">",
    verbose: bool = True,
    timer: StageTimer | None = None,
) -> dict[str, list[dict]]:
    """Archive volumes for multiple radars sequentially.

    Parameters
    ----------
    volumes_by_radar : dict
        Keys are radar names, values are lists or dicts of DataTrees.
        Example: ``{"A": [dt1, dt2], "D": [dt3, dt4]}``
    base_output_path : str
        Base output directory.
    filter_feature : str
        Column to use for gate filtering (default ``"DBZH"``).
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
    all_results = {}

    for radar, volumes in volumes_by_radar.items():
        radar = normalize_radar_name(radar)
        _vprint(
            f"\n{'='*60}\nArchiving radar {radar}\n{'='*60}",
            verbose,
        )

        results = archive_multiple_volumes(
            volumes,
            radar=radar,
            base_output_path=base_output_path,
            filter_feature=filter_feature,
            filter_threshold=filter_threshold,
            filter_logic=filter_logic,
            verbose=verbose,
            timer=timer,
        )

        all_results[radar] = results

    return all_results
