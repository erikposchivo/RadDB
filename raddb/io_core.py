"""
raddb/io_core.py
----------------
Core I/O conversion functions for radar data.

This module provides generic conversions between xarray DataTree,
pandas DataFrame, and Parquet files.  It does **not** depend on pyart
or radar_api — all MCH-specific I/O lives in ``mch_pipeline.py``.
"""
from __future__ import annotations

import concurrent.futures
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from raddb.lut import RADAR_TO_IDX, get_full_sweep_index
from raddb.helper import list_sweep_names, _find_polar_files_in_range, ensure_utc

logger = logging.getLogger(__name__)

# --- Constants --- #
POL_FEATURES = ["DBZH", "ZDR", "RHOHV", "PHIDP"]
POLAR_COLUMNS = [
    "gate_id", "time",
    "DBZH", "ZDR", "RHOHV", "PHIDP",
    "HC_MCH", "HC_PYART", "HZT",
]
LUT_COLUMNS = [
    "gate_id", "sweep", "azimuth", "range", "elevation_angle",
    "latitude", "longitude", "altitude",
    "x", "y", "z",
]


# ============================================================================
# DataTree -> DataFrame / Parquet
# ============================================================================

def datatree_to_dataset(dt: xr.DataTree, sweep: str | int) -> xr.Dataset:
    """Extract a single sweep Dataset from a DataTree."""
    sweep_name = f"sweep_{sweep}" if isinstance(sweep, int) else sweep
    return dt[sweep_name].to_dataset()


def datatree_to_dataframe(
    dt: xr.DataTree, max_workers: int = 1
) -> pd.DataFrame:
    """Flatten a DataTree into a single pandas DataFrame.

    Each sweep is converted independently and concatenated, with a ``sweep``
    column indicating the source sweep number.
    """
    names = list_sweep_names(dt)

    def _flatten(name):
        df = dt[name].to_dataset().to_dataframe().reset_index()
        df["sweep"] = int(name.split("_")[-1])
        return df

    if max_workers <= 1:
        list_df = [_flatten(name) for name in names]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers) as ex:
            list_df = list(ex.map(_flatten, names))

    return pd.concat(list_df, ignore_index=True)


def _save_polar_parquet(
    df_polar: pd.DataFrame, radar: str, base_path: str
) -> str:
    """Save a POLAR DataFrame to the standard directory layout."""
    vol_time = pd.to_datetime(df_polar["time"].min())
    save_dir = (
        Path(base_path)
        / radar
        / str(vol_time.year)
        / f"{vol_time.month:02d}"
        / f"{vol_time.day:02d}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = vol_time.strftime("%Y%m%d_%H%M%S")
    pp = save_dir / f"{radar}_{ts}_POL.parquet"
    df_polar.to_parquet(pp, index=False, engine="pyarrow")
    return str(pp)


def datatree_to_parquet(
    dt: xr.DataTree,
    radar: str,
    base_output_path: str,
    filter_feature: str = "DBZH",
    filter_threshold: float = 0.0,
    filter_logic: str = ">",
    max_workers: int = 1,
) -> str:
    """Convert a DataTree to a filtered POL parquet file.

    Gates that do not satisfy ``filter_feature [filter_logic] filter_threshold``
    are dropped (row removal) before saving.  Zero values in surviving gates
    are preserved as-is.

    If ``"HZT"`` is absent from the DataTree, ``"HC_PYART"`` is also skipped
    because it requires HZT to be meaningful.
    """
    from raddb.pipeline import FILTER_LOGICS

    fn = FILTER_LOGICS.get(filter_logic)
    if fn is None:
        raise ValueError(
            f"Unknown filter_logic '{filter_logic}'. "
            f"Choose from: {list(FILTER_LOGICS)}"
        )

    df = datatree_to_dataframe(dt, max_workers)

    if filter_feature in df.columns:
        _mask = fn(df[filter_feature].to_numpy(), filter_threshold)
    else:
        _mask = np.ones(len(df), dtype=bool)

    _radar_idx = np.int64(RADAR_TO_IDX[radar.upper()])
    _sweep_v   = df["sweep"].to_numpy(dtype=np.int64)[_mask]
    _az_int    = np.round(df["azimuth"].to_numpy(dtype=np.float64)[_mask] * 10).astype(np.int64)
    _rng_int   = df["range"].to_numpy(dtype=np.int64)[_mask]
    _gate_ids  = (
        _radar_idx * np.int64(1_000_000_000_000)
        + _sweep_v  * np.int64(   10_000_000_000)
        + _az_int   * np.int64(        1_000_000)
        + _rng_int
    )

    _hzt_available = "HZT" in df.columns
    _polar_cols = [
        c for c in POLAR_COLUMNS
        if c in df.columns
        and c != "gate_id"
        and not (c == "HC_PYART" and not _hzt_available)
    ]
    df_polar = pd.DataFrame(
        {"gate_id": _gate_ids, **{c: df[c].to_numpy()[_mask] for c in _polar_cols}},
    )

    return _save_polar_parquet(df_polar, radar, base_output_path)


# ============================================================================
# Parquet -> DataFrame / DataTree  (reading archived data)
# ============================================================================

def parquet_to_dataframe(
    radar: str,
    base_path: str | Path,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
    columns: list[str] | None = None,
    merge_lut: bool = False,
) -> pd.DataFrame:
    """Load archived POLAR parquet files as a single DataFrame.

    Parameters
    ----------
    radar : str
        Single-letter radar identifier.
    base_path : str or Path
        RadDB base directory.
    start_time, end_time : optional
        Filter by volume timestamp.
    columns : list of str, optional
        Columns to load from parquet files.
    merge_lut : bool
        If True, merge with the LUT to add spatial coordinates
        (azimuth, range, latitude, longitude, altitude, x, y, z).

    Returns
    -------
    pd.DataFrame
    """
    radar_path = Path(base_path) / radar
    if not radar_path.exists():
        logger.warning(f"Radar directory not found: {radar_path}")
        return pd.DataFrame()

    polar_files = _find_polar_files_in_range(radar_path, start_time, end_time)

    if not polar_files:
        logger.warning(
            f"No POLAR data found for radar {radar} "
            f"between {start_time} and {end_time}"
        )
        return pd.DataFrame()

    dfs = []
    for f in polar_files:
        try:
            df = pd.read_parquet(f, columns=columns, engine="pyarrow")
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Error reading {f}: {e}")
            continue

    if not dfs:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)

    if merge_lut:
        lut_path = radar_path / "LUT" / f"{radar}_LUT.parquet"
        if lut_path.exists():
            lut_df = pd.read_parquet(lut_path, engine="pyarrow")
            lut_cols = [
                "gate_id",
                "sweep",
                "azimuth",
                "range",
                "elevation_angle",
                "latitude",
                "longitude",
                "altitude",
                "x",
                "y",
                "z",
            ]
            lut_cols = [c for c in lut_cols if c in lut_df.columns]
            df_all = df_all.merge(lut_df[lut_cols], on="gate_id", how="left")
        else:
            logger.warning(
                f"LUT not found at {lut_path}. "
                "Returning data without spatial coordinates."
            )

    return df_all


def parquet_to_datatree(
    radar: str,
    base_path: str | Path,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
    label_column: str = "DBZH",
    max_workers: int = 1,
) -> xr.DataTree:
    """Load archived POLAR parquet files and reconstruct a DataTree.

    Loads all volumes in the given time range, joins with the LUT to
    recover azimuth/range coordinates, and reconstructs an xarray DataTree.

    Parameters
    ----------
    radar : str
        Single-letter radar identifier.
    base_path : str or Path
        RadDB base directory.
    start_time, end_time : optional
        Filter by volume timestamp.
    label_column : str
        Column to use for reconstruction (default ``"DBZH"``).
    max_workers : int
        Parallel workers for sweep reconstruction.

    Returns
    -------
    xr.DataTree

    Raises
    ------
    FileNotFoundError
        If the LUT or radar info files are missing.
    ValueError
        If no data is found.
    """
    base = Path(base_path)
    radar_path = base / radar

    lut_path = radar_path / "LUT" / f"{radar}_LUT.parquet"
    info_path = radar_path / "LUT" / f"{radar}_info.yaml"

    if not lut_path.exists():
        raise FileNotFoundError(
            f"LUT not found at {lut_path}. Run generate_lut() first."
        )
    if not info_path.exists():
        raise FileNotFoundError(f"Radar info not found at {info_path}.")

    polar_files = _find_polar_files_in_range(radar_path, start_time, end_time)
    if not polar_files:
        raise ValueError(
            f"No POLAR data found for {radar} "
            f"between {start_time} and {end_time}"
        )

    # Load all POLAR files in range
    dfs = []
    for f in polar_files:
        try:
            dfs.append(pd.read_parquet(f, engine="pyarrow"))
        except Exception as e:
            logger.warning(f"Error reading {f}: {e}")
            continue

    if not dfs:
        raise ValueError(f"All POLAR files for {radar} failed to load.")

    df_polar = pd.concat(dfs, ignore_index=True)

    # Join with LUT — include "sweep" since it is no longer stored in the
    # POL parquet (removed from POLAR_COLUMNS to avoid redundancy with gate_id).
    lut_df = pd.read_parquet(lut_path, engine="pyarrow")
    df_joined = df_polar.merge(
        lut_df[["gate_id", "sweep", "azimuth", "range"]],
        on="gate_id",
        how="left",
    )
    df_joined = df_joined.dropna(subset=["azimuth", "range"])

    # When multiple volumes are loaded, keep only the latest observation
    # per gate to avoid duplicate (azimuth, range) entries in reconstruction.
    if "time" in df_joined.columns:
        df_joined = df_joined.sort_values("time")
    df_joined = df_joined.drop_duplicates(
        subset=["sweep", "azimuth", "range"], keep="last"
    )

    if df_joined.empty:
        raise ValueError(
            "No matching gates found between POLAR data and LUT."
        )

    # Reconstruct DataTree
    return reconstruct_datatree(
        df_joined=df_joined,
        lut_path=lut_path,
        radar_info_path=info_path,
        label_column=label_column,
        max_workers=max_workers,
    )


# ============================================================================
# Reconstruction (Parquet + LUT -> DataTree)
# ============================================================================

def labels_to_dataframe(
    labels: np.ndarray,
    gate_ids,
    extra_columns: dict | None = None,
) -> pd.DataFrame:
    """Create a DataFrame from prediction labels and gate IDs."""
    df = pd.DataFrame({"gate_id": gate_ids, "hydrometeor_class": labels})
    if extra_columns:
        for k, v in extra_columns.items():
            df[k] = v
    return df


def join_labels_with_lut(
    df_labels: pd.DataFrame, lut_path: str | Path
) -> pd.DataFrame:
    """Join label data with the LUT to recover spatial coordinates."""
    df_lut = pd.read_parquet(str(lut_path), engine="pyarrow")
    cols = [c for c in df_labels.columns if c != "gate_id"]
    return df_lut.merge(
        df_labels[["gate_id"] + cols], on="gate_id", how="left"
    )


def _get_sweep_coords(sweep, radar_info):
    coords = {
        "latitude": radar_info["latitude"],
        "longitude": radar_info["longitude"],
        "altitude": radar_info["altitude"],
        "sweep_number": sweep,
    }
    meta = radar_info.get("sweeps", {}).get(sweep, {})
    if "elevation" in meta:
        coords["elevation_angle"] = meta["elevation"]
    return coords


def reconstruct_sweep_dataset(
    df_joined: pd.DataFrame,
    sweep: int,
    lut_df: pd.DataFrame,
    radar_info: dict,
    label_column: str = "hydrometeor_class",
) -> xr.Dataset:
    """Reconstruct a single sweep Dataset from joined data."""
    df_sweep = df_joined[df_joined["sweep"] == sweep].copy()
    keep_cols = [
        c
        for c in df_sweep.columns
        if c not in ("gate_id", "sweep", "azimuth", "range")
    ]

    idx = get_full_sweep_index(lut_df, sweep)
    df_reidx = df_sweep.set_index(["azimuth", "range"])[keep_cols].reindex(idx)

    ds = df_reidx.to_xarray().assign_coords(
        _get_sweep_coords(sweep, radar_info)
    )
    return ds


def reconstruct_datatree(
    df_joined: pd.DataFrame,
    lut_path: str | Path,
    radar_info_path: str | Path,
    label_column: str = "hydrometeor_class",
    max_workers: int = 1,
) -> xr.DataTree:
    """Reconstruct a full DataTree from joined data + LUT + radar info."""
    lut_df = pd.read_parquet(str(lut_path), engine="pyarrow")
    with open(str(radar_info_path)) as f:
        radar_info = yaml.safe_load(f)

    s_serie = pd.to_numeric(df_joined["sweep"], errors="coerce").dropna().unique()
    sweeps = [int(s) for s in s_serie]
    if not sweeps:
        raise ValueError("No valid sweep numbers found in df_joined.")

    def _rec(sw):
        try:
            return sw, reconstruct_sweep_dataset(
                df_joined,
                sweep=sw,
                lut_df=lut_df,
                radar_info=radar_info,
                label_column=label_column,
            )
        except Exception as exc:
            logger.warning(f"Failed to reconstruct sweep {sw}: {exc}")
            return sw, None

    dict_ds = {}
    if max_workers <= 1:
        for sw in sweeps:
            k, ds = _rec(sw)
            if ds is not None:
                dict_ds[f"sweep_{k}"] = ds
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers) as ex:
            for fut in concurrent.futures.as_completed(
                [ex.submit(_rec, sw) for sw in sweeps]
            ):
                k, ds = fut.result()
                if ds is not None:
                    dict_ds[f"sweep_{k}"] = ds

    if not dict_ds:
        raise ValueError("No sweeps could be reconstructed.")
    return xr.DataTree.from_dict(dict_ds)


# ============================================================================
# Feature addition utilities
# ============================================================================

def add_feature_to_df(
    df: pd.DataFrame,
    feature_name: str,
    compute_fn: callable,
) -> pd.DataFrame:
    """Add a new column to a DataFrame computed from existing columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (e.g. from :func:`parquet_to_dataframe`).
    feature_name : str
        Name of the new column to add.
    compute_fn : callable
        Function that takes ``df`` and returns a Series or array of the
        same length.  Example::

            def my_feature(df):
                return df["ZDR"] + df["DBZH"] * 0.1

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with the new column appended.
    """
    df = df.copy()
    df[feature_name] = compute_fn(df)
    return df


def add_feature_to_dt(
    dt: xr.DataTree,
    feature_name: str,
    compute_fn: callable,
) -> xr.DataTree:
    """Add a new variable to every sweep in a DataTree.

    Parameters
    ----------
    dt : xr.DataTree
        Input DataTree with ``sweep_N`` groups.
    feature_name : str
        Name of the new variable to add to each sweep Dataset.
    compute_fn : callable
        Function that takes an ``xr.Dataset`` (one sweep) and returns an
        ``xr.DataArray`` with matching dimensions.  Example::

            def kdp_proxy(ds):
                return (ds["PHIDP"].diff("range") / 0.250).clip(0)

    Returns
    -------
    xr.DataTree
        New DataTree with the computed variable added to every sweep.
    """
    from raddb.helper import list_sweep_names

    sweep_names = list_sweep_names(dt)
    dict_ds = {}
    for sweep_name in sweep_names:
        ds = dt[sweep_name].to_dataset()
        new_var = compute_fn(ds)
        ds = ds.assign({feature_name: new_var})
        dict_ds[sweep_name] = ds
    return xr.DataTree.from_dict(dict_ds)
