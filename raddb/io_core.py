"""
raddb/io_core.py
----------------
Core atomic conversion functions for radar data and reconstruction.
"""
#%%
from __future__ import annotations
import concurrent.futures
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import yaml
import pyart

import radar_api
from raddb.lut import generate_gate_id, get_full_sweep_index
from raddb.helper import list_sweep_names

logger = logging.getLogger(__name__)

# --- Constants --- #
POL_FEATURES = ["DBZH", "ZDR", "RHOHV", "PHIDP"]
POLAR_COLUMNS = ["gate_id", "sweep", "time", "DBZH", "ZDR", "RHOHV", "PHIDP", "HC_MCH", "HC_PYART", "HZT"]
LUT_COLUMNS = ["gate_id", "sweep", "azimuth", "range", "latitude", "longitude", "altitude", "x", "y", "z", "elevation_angle"]

# Field name mapping from PyART to xradar
FIELD_MAPPING = {
    'reflectivity': 'DBZH',
    'reflectivity_visibilitycorr': 'DBZH',
    'reflectivity_vv': 'DBZV',
    'corrected_reflectivity': 'DBZH_corrected',
    'corrected_reflectivity_vv': 'DBZV_corrected',
    'differential_reflectivity': 'ZDR',
    'corrected_differential_reflectivity': 'ZDR_corrected',
    'uncorrected_cross_correlation_ratio': 'RHOHV',
    'cross_correlation_ratio': 'RHOHV',
    'uncorrected_differential_phase': 'PHIDP',
    'differential_phase': 'PHIDP',
    'propagation_differential_phase': 'PHIDP_smoothed',
    'specific_differential_phase': 'KDP',
    'hydrometeor_classification_mch': 'HC_MCH',
    'radar_echo_classification': 'HC_PYART',
    'iso0_height': 'HZT',
    'height_over_iso0': 'height_over_iso0',
    'temperature': 'temperature',
}

#%%
# --- PyART to xradar conversion --- #
def pyart_to_xradar_dataset(rad_obj, sweep_idx: int = 0, fields_to_extract: list[str] | None = None) -> xr.Dataset:
    """
    Convert a PyART radar object to an xradar-compatible xarray Dataset.

    Parameters
    ----------
    rad_obj : pyart.Radar
        PyART radar object
    sweep_idx : int
        Sweep index (0-based) to extract from radar object
    fields_to_extract : list of str, optional
        List of field names to extract. If None, extracts all available fields.

    Returns
    -------
    xr.Dataset
        xradar-compatible dataset
    """
    # Get sweep data
    sweep_slice = rad_obj.get_slice(sweep_idx)

    # Create coordinate arrays
    time_data = pyart.util.datetime_from_radar(rad_obj)
    azimuth = rad_obj.azimuth['data'][sweep_slice]
    elevation = rad_obj.elevation['data'][sweep_slice]
    range_data = rad_obj.range['data']

    # Create coordinates dict
    # Handle both scalar and array-like time_data
    if isinstance(time_data, (list, np.ndarray)):
        time_coord = time_data[sweep_slice] if np.size(time_data) > 1 else [time_data[0]] * len(azimuth)
    else:
        # time_data is a scalar datetime object
        time_coord = [time_data] * len(azimuth)

    coords = {
        'azimuth': (['azimuth'], azimuth),
        'range': (['range'], range_data),
        'time': (['azimuth'], time_coord),
        'elevation': (['azimuth'], elevation),
    }

    # Add radar location
    if rad_obj.latitude:
        coords['latitude'] = float(rad_obj.latitude['data'][0])
    if rad_obj.longitude:
        coords['longitude'] = float(rad_obj.longitude['data'][0])
    if rad_obj.altitude:
        coords['altitude'] = float(rad_obj.altitude['data'][0])

    # Add sweep metadata
    if rad_obj.fixed_angle:
        coords['elevation_angle'] = float(rad_obj.fixed_angle['data'][sweep_idx])

    # Extract fields
    data_vars = {}
    available_fields = list(rad_obj.fields.keys())
    if fields_to_extract is None:
        fields_to_extract = available_fields

    for field_name in fields_to_extract:
        if field_name not in available_fields:
            continue

        field_data = rad_obj.get_field(sweep_idx, field_name)
        mapped_name = FIELD_MAPPING.get(field_name, field_name)

        data_vars[mapped_name] = (['azimuth', 'range'], field_data, {
            'long_name': rad_obj.fields[field_name].get('long_name', mapped_name),
            'units': rad_obj.fields[field_name].get('units', ''),
        })

    # Create dataset
    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs['sweep_number'] = sweep_idx + 1

    return ds


def pyart_volume_to_datatree(rad_objs: dict[int, object], fields_to_extract: list[str] | None = None) -> xr.DataTree:
    """
    Convert multiple PyART radar objects (sweeps) to an xradar DataTree.

    Parameters
    ----------
    rad_objs : dict
        Dictionary mapping sweep numbers (1-based) to PyART radar objects
    fields_to_extract : list of str, optional
        List of field names to extract

    Returns
    -------
    xr.DataTree
        xradar DataTree with sweep groups
    """
    dict_ds = {}
    for sweep_num, rad_obj in rad_objs.items():
        ds = pyart_to_xradar_dataset(rad_obj, sweep_idx=0, fields_to_extract=fields_to_extract)
        dict_ds[f"sweep_{sweep_num}"] = ds

    return xr.DataTree.from_dict(dict_ds)

#%%
# --- Input Conversions --- #
def metranet_to_datatree(filepath: str | Path, network: str = "MCH_LTE", product: str = "POL") -> xr.DataTree:
    return radar_api.open_datatree(str(filepath), network=network, product=product)

#%%
def _load_sweep_into_datatree(idx_path: tuple[int, str], network, product):
    sweep_idx, sweep_filepath = idx_path
    try:
        dt_sweep = metranet_to_datatree(sweep_filepath, network=network, product=product)
        names = list_sweep_names(dt_sweep)
        if not names: return None, None
        ds = dt_sweep[names[0]].to_dataset().assign_attrs({"sweep_number": sweep_idx})
        return f"sweep_{sweep_idx}", ds
    except Exception as exc:
        logger.error(f"Error loading {sweep_filepath}: {exc}")
        return None, None

def volume_to_datatree(sweep_filepaths: list, network: str = "MCH_LTE", product: str = "POL", max_workers: int = 1) -> xr.DataTree:
    paths = sorted([str(p) for p in sweep_filepaths])
    args = [(i, p) for i, p in enumerate(paths, 1)]
    dict_ds = {}
    
    if max_workers <= 1:
        for arg in args:
            k, ds = _load_sweep_into_datatree(arg, network, product)
            if k and ds is not None: dict_ds[k] = ds
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            for future in concurrent.futures.as_completed([executor.submit(_load_sweep_into_datatree, arg, network, product) for arg in args]):
                k, ds = future.result()
                if k and ds is not None: dict_ds[k] = ds
                
    if not dict_ds: raise ValueError("No valid sweep datasets loaded.")
    return xr.DataTree.from_dict(dict_ds)

# --- Output Conversions --- #
def datatree_to_dataset(dt: xr.DataTree, sweep: str | int) -> xr.Dataset:
    sweep_name = f"sweep_{sweep}" if isinstance(sweep, int) else sweep
    return dt[sweep_name].to_dataset()

def datatree_to_dataframe(dt: xr.DataTree, max_workers: int = 1) -> pd.DataFrame:
    names = list_sweep_names(dt)
    def _flatten(name):
        df = dt[name].to_dataset().to_dataframe().reset_index()
        df["sweep"] = int(name.split("_")[-1])
        return df
    
    list_df = []
    if max_workers <= 1:
        list_df = [_flatten(name) for name in names]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers) as ex:
            list_df = list(ex.map(_flatten, names))
            
    return pd.concat(list_df, ignore_index=True)

def _save_polar_parquet(df_polar: pd.DataFrame, radar: str, base_path: str) -> str:
    vol_time = pd.to_datetime(df_polar["time"].min())
    save_dir = Path(base_path) / radar / str(vol_time.year) / f"{vol_time.month:02d}" / f"{vol_time.day:02d}"
    save_dir.mkdir(parents=True, exist_ok=True)
    ts = vol_time.strftime("%Y%m%d_%H%M%S")
    pp = save_dir / f"{radar}_{ts}_POLAR.parquet"
    df_polar.to_parquet(pp, index=False, engine="pyarrow")
    return str(pp)

def datatree_to_parquet(dt: xr.DataTree, radar: str, base_output_path: str, dbzh_threshold: float = 0.0, max_workers: int = 1) -> str:
    df = datatree_to_dataframe(dt, max_workers)
    df["gate_id"] = df.apply(lambda row: generate_gate_id(radar, int(row["sweep"]), float(row["azimuth"]), float(row["range"])), axis=1)

    df_polar_full = df[df["DBZH"] > dbzh_threshold].copy()
    df_polar = df_polar_full[[c for c in POLAR_COLUMNS if c in df_polar_full.columns]].copy()

    return _save_polar_parquet(df_polar, radar, base_output_path)

# --- Reconstruction --- #
def labels_to_dataframe(labels: np.ndarray, gate_ids, extra_columns: dict | None = None) -> pd.DataFrame:
    df = pd.DataFrame({"gate_id": gate_ids, "hydrometeor_class": labels})
    if extra_columns:
        for k, v in extra_columns.items(): df[k] = v
    return df

def join_labels_with_lut(df_labels: pd.DataFrame, lut_path: str | Path) -> pd.DataFrame:
    df_lut = pd.read_parquet(str(lut_path), engine="pyarrow")
    cols = [c for c in df_labels.columns if c != "gate_id"]
    return df_lut.merge(df_labels[["gate_id"] + cols], on="gate_id", how="left")

def _get_sweep_coords(sweep, radar_info):
    coords = {
        "latitude": radar_info["latitude"],
        "longitude": radar_info["longitude"],
        "altitude": radar_info["altitude"],
        "sweep_number": sweep,
    }
    meta = radar_info.get("sweeps", {}).get(sweep, {})
    if "elevation" in meta: coords["elevation_angle"] = meta["elevation"]
    return coords

def reconstruct_sweep_dataset(df_joined: pd.DataFrame, sweep: int, lut_df: pd.DataFrame, radar_info: dict, label_column: str = "hydrometeor_class") -> xr.Dataset:
    df_sweep = df_joined[df_joined["sweep"] == sweep].copy()
    keep_cols = [c for c in df_sweep.columns if c not in ("gate_id", "sweep", "azimuth", "range")]
    
    idx = get_full_sweep_index(lut_df, sweep)
    df_reidx = df_sweep.set_index(["azimuth", "range"])[keep_cols].reindex(idx)
    
    ds = df_reidx.to_xarray().assign_coords(_get_sweep_coords(sweep, radar_info))
    return ds

def reconstruct_datatree(df_joined: pd.DataFrame, lut_path: str | Path, radar_info_path: str | Path, label_column: str = "hydrometeor_class", max_workers: int = 1) -> xr.DataTree:
    lut_df = pd.read_parquet(str(lut_path), engine="pyarrow")
    with open(str(radar_info_path)) as f: radar_info = yaml.safe_load(f)
        
    s_serie = pd.to_numeric(df_joined["sweep"], errors='coerce').dropna().unique()
    sweeps = [int(s) for s in s_serie]
    if not sweeps: raise ValueError("No valid sweep numbers found in df_joined.")
    
    def _rec(sw): 
        try: return sw, reconstruct_sweep_dataset(df_joined, sweep=sw, lut_df=lut_df, radar_info=radar_info, label_column=label_column)
        except Exception as exc:
            logger.warning(f"Failed to reconstruct sweep {sw}: {exc}")
            return sw, None
        
    dict_ds = {}
    if max_workers <= 1:
        for sw in sweeps:
            k, ds = _rec(sw)
            if ds is not None: dict_ds[f"sweep_{k}"] = ds
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers) as ex:
            for fut in concurrent.futures.as_completed([ex.submit(_rec, sw) for sw in sweeps]):
                k, ds = fut.result()
                if ds is not None: dict_ds[f"sweep_{k}"] = ds
                
    if not dict_ds: raise ValueError("No sweeps could be reconstructed.")
    return xr.DataTree.from_dict(dict_ds)
