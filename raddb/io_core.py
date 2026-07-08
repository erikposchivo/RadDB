"""
raddb/io_core.py
----------------
Core I/O conversion functions for radar data.

This module provides generic conversions between xarray DataTree,
pandas DataFrame, and Parquet files.  It does **not** depend on pyart
or radar_api — all MCH-specific I/O lives in the private ``raddb.mch``
subpackage.
"""
from __future__ import annotations

import concurrent.futures
import logging
import re
import time as _time
from contextlib import nullcontext as _nullctx
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from raddb.lut import _parse_corners_npz, encode_gate_ids, get_full_sweep_index
from raddb.helper import (
    StageTimer,
    _vprint,
    list_sweep_names,
    normalize_radar_name,
    resolve_filter_logic,
)
from raddb.discovery import _find_polar_files_in_range

logger = logging.getLogger(__name__)

# --- Constants --- #
POL_FEATURES = ["DBZH", "ZDR", "RHOHV", "PHIDP"]
POLAR_COLUMNS = [
    "gate_id", "time",
    "DBZH", "DBZH_raw", "ZDR", "ZDR_raw", "KDP", "RHOHV", "PHIDP",
    "HC_MCH", "HC_PYART", "HZT", "TEMP",
]
LUT_COLUMNS = [
    "gate_id", "sweep", "azimuth", "range", "elevation_angle",
    "latitude", "longitude", "altitude",
    "x", "y", "z",
]

# float32 gives 7 significant digits — sufficient for all radar variables.
_POLAR_FLOAT32_COLS: frozenset = frozenset({"DBZH", "DBZH_raw", "ZDR", "ZDR_raw", "KDP", "RHOHV", "PHIDP", "HZT", "HC_MCH", "HC_PYART", "TEMP"})
_LAPSE_RATE: float = -0.0065  # °C/m (standard environmental lapse rate, -6.5 °C/km)


def _projection_columns(df: pd.DataFrame) -> list[str]:
    """Columns added by :func:`raddb.lut.add_lut_projection` (e.g. x_2056 / y_2056)."""
    return [c for c in df.columns if re.match(r"^[xy]_\w+$", c)]


# ============================================================================
# DataTree file loading  (NetCDF / Zarr)
# ============================================================================

def open_any_datatree(
    path: str | Path,
    engine: str | None = None,
    **open_kwargs,
) -> xr.DataTree:
    """Open a DataTree from disk — NetCDF file or Zarr store.

    Engine resolution: an explicit ``engine`` wins; a ``*.zarr`` suffix or a
    directory containing ``.zgroup`` / ``zarr.json`` selects ``"zarr"``;
    otherwise xarray auto-detects the NetCDF backend (netCDF4 / h5netcdf).

    Parameters
    ----------
    path : str or Path
        Path to a NetCDF file or Zarr store.
    engine : str, optional
        xarray backend override (e.g. ``"h5netcdf"``, ``"zarr"``).
    **open_kwargs
        Forwarded to :func:`xarray.open_datatree`.

    Returns
    -------
    xr.DataTree
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DataTree file/store not found: {p}")

    if engine is None and (
        p.suffix.lower() == ".zarr"
        or (p.is_dir() and ((p / ".zgroup").exists() or (p / "zarr.json").exists()))
    ):
        engine = "zarr"

    try:
        return xr.open_datatree(p, engine=engine, **open_kwargs)
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            f"Opening {p.name} requires an xarray backend that is not "
            "installed (netCDF4/h5netcdf for NetCDF, zarr for Zarr stores). "
            "Install with: pip install raddb[io]"
        ) from exc


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


def _cast_hc_column(arr, shift: int = 0) -> np.ndarray:
    """Cast an HC column to float32, optionally shifting values first.

    HC_MCH raw = 0-8; shift=1 → parquet 1-9.
    HC_PYART after PYART_TO_OPE remapping = 1-8; shift=1 → parquet 2-9.
    Both land on the same 1-9 scale.  Values outside [1, 9] after shifting → NaN.
    """
    arr_f = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    if shift:
        arr_f = np.where(np.isnan(arr_f), np.nan, arr_f + shift)
    arr_f[(arr_f < 1) | (arr_f > 9)] = np.nan
    return arr_f.astype(np.float32)


def _compute_gate_temperature(df: pd.DataFrame, mask: np.ndarray) -> np.ndarray | None:
    """Compute temperature (°C) at surviving gates using standard lapse rate.

    TEMP = _LAPSE_RATE x (gate_altitude - HZT)
    gate_altitude = radar site altitude + z-height above radar (4/3 Earth radius model).
    Returns NaN array if HZT is absent; returns None only if geometry columns
    (range, elevation, altitude) are missing so no array can be sized.
    """
    geom_required = {"range", "elevation", "altitude"}
    if not geom_required.issubset(df.columns):
        return None
    n        = int(mask.sum())
    r        = df["range"].to_numpy()[mask]
    el_rad   = np.deg2rad(df["elevation"].to_numpy()[mask])
    site_alt = df["altitude"].to_numpy()[mask]
    ke, Re   = 4.0 / 3.0, 6_371_000.0
    z_gate   = np.sqrt(r**2 + (ke * Re)**2 + 2 * r * ke * Re * np.sin(el_rad)) - ke * Re
    gate_alt = site_alt + z_gate
    if "HZT" not in df.columns:
        return np.full(n, np.nan, dtype=np.float32)
    hzt = df["HZT"].to_numpy()[mask]
    return (_LAPSE_RATE * (gate_alt - hzt)).astype(np.float32)


def _build_polar_dataframe(
    df: pd.DataFrame,
    radar: str,
    filter_feature: str,
    filter_threshold: float,
    filter_logic: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Filter a flattened volume DataFrame and attach gate_ids.

    Rows that do not satisfy ``filter_feature [filter_logic] filter_threshold``
    are dropped (row removal — zeros in surviving gates stay zeros).  HC_PYART
    is skipped when HZT is absent (it requires HZT to be meaningful).

    This is the single shared core of :func:`datatree_to_parquet` and
    :func:`archive_volume`.

    Returns
    -------
    (df_polar, mask) : the polar DataFrame (gate_id + polar columns) and the
    boolean row mask, needed afterwards for TEMP computation.
    """
    fn = resolve_filter_logic(filter_logic)

    if filter_feature in df.columns:
        mask = fn(df[filter_feature].to_numpy(), filter_threshold)
    else:
        logger.warning(
            "filter_feature '%s' not found in DataFrame; keeping all gates.",
            filter_feature,
        )
        mask = np.ones(len(df), dtype=bool)

    gate_ids = encode_gate_ids(
        radar,
        df["sweep"].to_numpy(dtype=np.int64)[mask],
        df["azimuth"].to_numpy(dtype=np.float64)[mask],
        df["range"].to_numpy()[mask],
    )

    hzt_available = "HZT" in df.columns
    polar_cols = [
        c for c in POLAR_COLUMNS
        if c in df.columns
        and c != "gate_id"
        and not (c == "HC_PYART" and not hzt_available)
    ]
    df_polar = pd.DataFrame(
        {"gate_id": gate_ids, **{c: df[c].to_numpy()[mask] for c in polar_cols}},
    )
    return df_polar, mask


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
    resolve_filter_logic(filter_logic)  # fail fast before flattening

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
        df_polar, _mask = _build_polar_dataframe(
            df, radar, filter_feature, filter_threshold, filter_logic
        )

    with (
        timer.time_stage("save_parquet", volume=volume)
        if timer
        else _nullctx()
    ):
        df_polar = _finalize_polar_dtypes(df_polar, df, _mask)
        return _save_polar_parquet(df_polar, radar, base_output_path)


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


def _finalize_polar_dtypes(
    df_polar: pd.DataFrame, df: pd.DataFrame, mask: np.ndarray
) -> pd.DataFrame:
    """Apply dtype optimisations and add the TEMP column.

    HC columns are shifted +1 to the 1-based parquet scale; all polar
    variables are cast to float32; TEMP is computed from gate geometry + HZT.
    """
    for col in list(df_polar.columns):
        if col in ("HC_MCH", "HC_PYART"):
            df_polar[col] = _cast_hc_column(df_polar[col], shift=1)
        elif col in _POLAR_FLOAT32_COLS:
            df_polar[col] = df_polar[col].astype(np.float32)
    temp = _compute_gate_temperature(df, mask)
    if temp is not None:
        df_polar["TEMP"] = temp
    return df_polar


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
    resolve_filter_logic(filter_logic)  # fail fast before flattening

    df = datatree_to_dataframe(dt, max_workers)
    df_polar, mask = _build_polar_dataframe(
        df, radar, filter_feature, filter_threshold, filter_logic
    )
    df_polar = _finalize_polar_dtypes(df_polar, df, mask)
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
            # Include any projected coordinate columns added by add_lut_projection
            # (e.g. x_2056, y_2056 for Swiss LV95 / EPSG:2056)
            lut_cols += _projection_columns(lut_df)
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
    # POL parquet (removed from POLAR_COLUMNS to avoid redundancy with gate_id),
    # plus per-gate spatial coords (latitude, longitude, altitude, x, y, z and
    # any x_<epsg> / y_<epsg> projection columns) so the reconstructed DataTree
    # is plot-ready.
    lut_df = pd.read_parquet(lut_path, engine="pyarrow")
    _spatial_cols = ["latitude", "longitude", "altitude", "x", "y", "z"]
    _proj_cols = _projection_columns(lut_df)
    _join_cols = ["gate_id", "sweep", "azimuth", "range"] + _spatial_cols + _proj_cols
    _join_cols = [c for c in _join_cols if c in lut_df.columns]
    df_joined = df_polar.merge(
        lut_df[_join_cols],
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


_PER_GATE_COORDS = ("latitude", "longitude", "altitude", "x", "y", "z")


def _get_sweep_coords(sweep, radar_info):
    coords = {
        "site_latitude":  radar_info["latitude"],
        "site_longitude": radar_info["longitude"],
        "site_altitude":  radar_info["altitude"],
        "sweep_number":   sweep,
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
    sweep_corners: dict | None = None,
) -> xr.Dataset:
    """Reconstruct a single sweep Dataset from joined data.

    Data variables come from the filtered POL parquet (NaN where gates were
    dropped). Per-gate spatial coords (lat/lon/alt/x/y/z and any x_<epsg>/
    y_<epsg>) come from the LUT directly so every gate has a valid geometry
    regardless of filtering.

    If ``sweep_corners`` is provided, per-gate edge arrays
    (``x_edges``, ``y_edges``, ``z_edges``, ``lon_edges``, ``lat_edges``) of
    shape ``(n_az+1, n_range+1)`` are attached as data_vars for pcolormesh
    rendering with ``shading="flat"``.
    """
    df_sweep = df_joined[df_joined["sweep"] == sweep].copy()

    # Identify spatial columns that should come from the LUT (always populated)
    # and non-spatial columns that should come from the polar data (may be NaN).
    spatial_cols = [c for c in _PER_GATE_COORDS if c in lut_df.columns]
    spatial_cols += _projection_columns(lut_df)

    non_spatial = [
        c for c in df_sweep.columns
        if c not in ("gate_id", "sweep", "azimuth", "range") and c not in spatial_cols
    ]

    idx = get_full_sweep_index(lut_df, sweep)

    # Data variables (may have NaN for filtered-out gates)
    df_reidx = df_sweep.set_index(["azimuth", "range"])[non_spatial].reindex(idx)

    # Spatial coords from the LUT, aligned to the same (azimuth, range) index
    lut_sweep = lut_df[lut_df["sweep"] == sweep].set_index(["azimuth", "range"])
    lut_spatial = lut_sweep[spatial_cols].reindex(idx)
    df_full = pd.concat([df_reidx, lut_spatial], axis=1)

    ds = df_full.to_xarray().assign_coords(
        _get_sweep_coords(sweep, radar_info)
    )

    # Promote per-gate spatial vars to coords so the Dataset is plot-ready.
    promote = [c for c in spatial_cols if c in ds.data_vars]
    if promote:
        ds = ds.set_coords(promote)

    # Attach per-sweep gate edge arrays (N_az+1, N_range+1) for accurate
    # pcolormesh(shading="flat") rendering. These use their own dims
    # (azimuth_edge, range_edge) so they coexist with the primary grid.
    if sweep_corners:
        for key in ("x_edges", "y_edges", "z_edges", "lon_edges", "lat_edges"):
            if key in sweep_corners:
                arr = np.asarray(sweep_corners[key])
                ds[key] = (("azimuth_edge", "range_edge"), arr)
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

    # Optional: load per-sweep gate corners from <radar>_corners.npz.
    # These enable pcolormesh(shading="flat") rendering in plot_ppi. Missing
    # file → plots fall back to centroid-based rendering (less accurate).
    lut_dir = Path(lut_path).parent
    radar_name = Path(lut_path).name.split("_")[0]
    corners_file = lut_dir / f"{radar_name}_corners.npz"
    sweep_corners_all: dict[int, dict] = {}
    if corners_file.exists():
        try:
            sweep_corners_all = _parse_corners_npz(corners_file)
        except Exception as exc:
            logger.warning(f"Failed to load sweep corners from {corners_file}: {exc}")

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
                sweep_corners=sweep_corners_all.get(sw),
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
