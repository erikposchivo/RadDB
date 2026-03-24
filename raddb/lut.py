"""
raddb/lut.py
------------
Look-Up Table (LUT) generation and loading utilities.
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import xarray as xr
import re
import radar_api
import pyart.core.transforms as transforms

logger = logging.getLogger(__name__)

# --- Coordinate and ID generation --- #
def compute_gate_xyz(rad_obj, ke: float = 1.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return transforms.antenna_vectors_to_cartesian(
        rad_obj.range["data"], rad_obj.azimuth["data"], rad_obj.elevation["data"], ke=ke
    )

def generate_gate_id(radar: str, sweep: int, azimuth: float, range_m: float) -> str:
    az_str = f"a{round(azimuth, 1):.1f}"
    return f"{radar.upper()}_s{sweep:02d}_{az_str}_r{int(range_m):06d}"

# --- Internal LUT generation --- #
def _process_lut_sweep(sweep_filepath, sweep_idx, radar, network, ke):
    rad_obj = radar_api.open_pyart(sweep_filepath, network=network, product="POL")
    dt_sweep = radar_api.open_datatree(sweep_filepath, network=network, product="POL")
    
    radar_lat = float(rad_obj.latitude["data"][0])
    radar_lon = float(rad_obj.longitude["data"][0])
    radar_alt = float(rad_obj.altitude["data"][0])
    elevation = float(np.mean(rad_obj.elevation["data"]))
    
    x_raw, y_raw, z_raw = compute_gate_xyz(rad_obj, ke=ke)
    azimuths, ranges = rad_obj.azimuth["data"], rad_obj.range["data"]
    n_az, n_rng = len(azimuths), len(ranges)
    
    groups = [g.lstrip("/") for g in dt_sweep.groups if re.match(r"sweep_\d+", g.lstrip("/"))]
    ds_sweep = dt_sweep[groups[0]].to_dataset()
    site_lat = float(ds_sweep.coords.get("latitude", radar_lat))
    site_lon = float(ds_sweep.coords.get("longitude", radar_lon))
    site_alt = float(ds_sweep.coords.get("altitude", radar_alt))
    
    records = []
    for az_idx in range(n_az):
        az = float(azimuths[az_idx])
        for rng_idx in range(n_rng):
            rng = float(ranges[rng_idx])
            gate_id = generate_gate_id(radar, sweep_idx, az, rng)
            records.append({
                "gate_id": gate_id, "sweep": sweep_idx, "azimuth": az, "range": rng,
                "latitude": site_lat, "longitude": site_lon, "altitude": site_alt,
                "x": float(x_raw[az_idx, rng_idx]), "y": float(y_raw[az_idx, rng_idx]),
                "z": float(z_raw[az_idx, rng_idx]), "elevation_angle": elevation,
            })
            
    meta = {"n_azimuths": n_az, "n_ranges": n_rng, "elevation": round(elevation, 2)}
    return records, meta, radar_lat, radar_lon, radar_alt

def _save_lut_outputs(lut_dir, radar, df_lut, radar_info):
    lut_dir.mkdir(parents=True, exist_ok=True)
    lut_path = lut_dir / f"{radar}_LUT.parquet"
    info_path = lut_dir / f"{radar}_info.yaml"

    if lut_path.exists() and info_path.exists():
        logger.info("LUT and radar info already exist at %s — skipping creation.", lut_dir)
        return str(lut_path)

    df_lut.to_parquet(lut_path, index=False, engine="pyarrow")
    logger.info("LUT saved → %s", lut_path)

    with open(info_path, "w") as f:
        yaml.dump(radar_info, f, default_flow_style=False, sort_keys=False)
    logger.info("Radar info saved → %s", info_path)
    return str(lut_path)

# --- Main LUT Generator --- #
def generate_radar_lut(
    radar: str, network: str, sample_volume_filepaths: list, output_base_path: str, qpegrid_to_rad_dir=None, ke=1.25,
) -> str:
    lut_dir = Path(output_base_path) / radar / "LUT"
    lut_path = lut_dir / f"{radar}_LUT.parquet"
    info_path = lut_dir / f"{radar}_info.yaml"
    if lut_path.exists() and info_path.exists():
        logger.info("LUT already exists at %s — skipping generation.", lut_path)
        return str(lut_path)

    sorted_paths = sorted([str(p) for p in sample_volume_filepaths])
    lut_records, sweep_meta = [], {}
    radar_lat, radar_lon, radar_alt = None, None, None
    
    for sweep_idx, sweep_filepath in enumerate(sorted_paths, start=1):
        recs, meta, lat, lon, alt = _process_lut_sweep(sweep_filepath, sweep_idx, radar, network, ke)
        lut_records.extend(recs)
        sweep_meta[sweep_idx] = meta
        radar_lat, radar_lon, radar_alt = lat, lon, alt
        
    df_lut = pd.DataFrame.from_records(lut_records)
    logger.info("LUT built: %d total gates, %d sweeps.", len(df_lut), len(sweep_meta))
    
    radar_info = {
        "radar": radar, "network": network,
        "latitude": radar_lat, "longitude": radar_lon, "altitude": radar_alt,
        "sweeps": sweep_meta,
    }
    return _save_lut_outputs(Path(output_base_path) / radar / "LUT", radar, df_lut, radar_info)

# --- Loaders --- #
def load_radar_lut(radar: str, lut_base_path: str | Path) -> pd.DataFrame:
    lut_path = Path(lut_base_path) / radar / "LUT" / f"{radar}_LUT.parquet"
    if not lut_path.exists(): raise FileNotFoundError(f"LUT not found at {lut_path}.")
    return pd.read_parquet(lut_path, engine="pyarrow")

def load_radar_info(radar: str, lut_base_path: str | Path) -> dict:
    info_path = Path(lut_base_path) / radar / "LUT" / f"{radar}_info.yaml"
    if not info_path.exists(): raise FileNotFoundError(f"Info not found at {info_path}.")
    with open(info_path) as f:
        return yaml.safe_load(f)

def get_full_sweep_index(lut_df: pd.DataFrame, sweep: int) -> pd.MultiIndex:
    sweep_lut = lut_df[lut_df["sweep"] == sweep]
    if sweep_lut.empty: raise ValueError(f"No LUT entries found for sweep={sweep}.")
    return sweep_lut.set_index(["azimuth", "range"]).index
