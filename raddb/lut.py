"""
raddb/lut.py
------------
Look-Up Table (LUT) generation and loading utilities.

The LUT stores one record per radar gate (azimuth x range x sweep) with
static spatial information (Cartesian coordinates, lat/lon, elevation).
It is generated once per radar and reused for every subsequent volume.

This module is **generic** — it works with any xarray DataTree that
has ``azimuth``, ``range``, and ``elevation`` coordinates per sweep.
No pyart or radar_api dependency.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

from raddb.helper import list_sweep_names

logger = logging.getLogger(__name__)


# ============================================================================
# Coordinate transforms  (pure numpy, no pyart dependency)
# ============================================================================

def antenna_vectors_to_cartesian(
    ranges: np.ndarray,
    azimuths: np.ndarray,
    elevations: np.ndarray,
    ke: float = 4.0 / 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert radar antenna coordinates to Cartesian (x, y, z).

    Replicates the standard radar beam propagation formula used by PyART
    (``pyart.core.transforms.antenna_vectors_to_cartesian``) using only
    numpy.  The 4/3 effective-Earth-radius model is used by default.

    Parameters
    ----------
    ranges : 1-D array, shape (n_gates,)
        Range to each gate in meters.
    azimuths : 1-D array, shape (n_rays,)
        Azimuth angle of each ray in degrees (0 = North, clockwise).
    elevations : 1-D array, shape (n_rays,)
        Elevation angle of each ray in degrees.
    ke : float
        Effective Earth radius scale factor (default 4/3).

    Returns
    -------
    x, y, z : arrays, shape (n_rays, n_gates)
        Cartesian coordinates in meters relative to the radar.
    """
    r = np.atleast_1d(np.asarray(ranges, dtype=np.float64))
    theta_e = np.deg2rad(np.atleast_1d(np.asarray(elevations, dtype=np.float64)))
    theta_a = np.deg2rad(np.atleast_1d(np.asarray(azimuths, dtype=np.float64)))

    R = 6371.0 * 1000.0 * ke  # effective earth radius [m]

    # Height of each gate above radar (n_rays, n_gates)
    z = (
        np.sqrt(
            r[np.newaxis, :] ** 2
            + R ** 2
            + 2.0 * r[np.newaxis, :] * R * np.sin(theta_e[:, np.newaxis])
        )
        - R
    )

    # Ground-range arc length
    s = R * np.arcsin(
        r[np.newaxis, :] * np.cos(theta_e[:, np.newaxis]) / (R + z)
    )

    # Cartesian
    x = s * np.sin(theta_a[:, np.newaxis])
    y = s * np.cos(theta_a[:, np.newaxis])

    return x, y, z


def cartesian_to_geographic(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    radar_lat: float,
    radar_lon: float,
    radar_alt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert gate Cartesian offsets to geographic coordinates.

    Uses an equirectangular approximation which is accurate for
    typical radar ranges (< 250 km).

    Parameters
    ----------
    x, y, z : arrays
        Cartesian offsets from the radar in meters (x = east, y = north,
        z = height above radar).
    radar_lat, radar_lon : float
        Radar site latitude and longitude in degrees.
    radar_alt : float
        Radar site altitude in meters above sea level.

    Returns
    -------
    lat, lon, alt : arrays (same shape as input)
        Geographic coordinates of each gate.
    """
    R_EARTH = 6_371_000.0  # mean Earth radius [m]
    lat = radar_lat + np.degrees(y / R_EARTH)
    lon = radar_lon + np.degrees(x / (R_EARTH * np.cos(np.radians(radar_lat))))
    alt = radar_alt + z
    return lat, lon, alt


def compute_gate_xyz(
    ranges: np.ndarray,
    azimuths: np.ndarray,
    elevations: np.ndarray,
    ke: float = 1.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute gate Cartesian coordinates from polar coordinates.

    Thin wrapper around :func:`antenna_vectors_to_cartesian` for
    backwards compatibility.
    """
    return antenna_vectors_to_cartesian(ranges, azimuths, elevations, ke=ke)


# ============================================================================
# Gate ID generation
# ============================================================================

def generate_gate_id(
    radar: str, sweep: int, azimuth: float, range_m: float
) -> str:
    """Create a unique gate identifier string.

    Format: ``{RADAR}_s{SWEEP:02d}_a{AZ:.1f}_r{RANGE:06d}``
    """
    az_str = f"a{round(azimuth, 1):.1f}"
    return f"{radar.upper()}_s{sweep:02d}_{az_str}_r{int(range_m):06d}"


# ============================================================================
# Generic LUT generation from DataTree
# ============================================================================

def generate_lut_from_datatree(
    dt: xr.DataTree,
    radar: str,
    output_base_path: str,
    ke: float = 1.25,
    network: str = "",
) -> str:
    """Generate a LUT from an xarray DataTree.

    This is the **generic** LUT generator — it works with any DataTree
    that has the standard xradar coordinate layout (azimuth, range,
    elevation per sweep).

    For MCH-specific LUT generation from raw METRANET files, use
    ``mch_pipeline.generate_mch_lut()`` instead.

    Parameters
    ----------
    dt : xr.DataTree
        DataTree with ``sweep_N`` groups, each containing ``azimuth``,
        ``range``, and ``elevation`` coordinates.
    radar : str
        Radar identifier (single letter, e.g. ``"A"``).
    output_base_path : str
        Base directory for LUT storage.
    ke : float
        Effective Earth radius scale factor (default 1.25 for Switzerland).
    network : str, optional
        Network identifier stored in radar info YAML.

    Returns
    -------
    str
        Path to the saved LUT parquet file.
    """
    lut_dir = Path(output_base_path) / radar / "LUT"
    lut_path = lut_dir / f"{radar}_LUT.parquet"
    info_path = lut_dir / f"{radar}_info.yaml"

    if lut_path.exists() and info_path.exists():
        logger.info(
            "LUT already exists at %s -- skipping generation.", lut_path
        )
        return str(lut_path)

    sweep_names = list_sweep_names(dt)
    if not sweep_names:
        raise ValueError("DataTree has no sweep groups (sweep_N).")

    lut_records = []
    sweep_meta = {}
    radar_lat, radar_lon, radar_alt = None, None, None

    for sweep_name in sweep_names:
        sweep_idx = int(sweep_name.split("_")[-1])
        ds = dt[sweep_name].to_dataset()

        azimuths = ds["azimuth"].values
        ranges = ds["range"].values
        elevations = ds["elevation"].values
        n_az, n_rng = len(azimuths), len(ranges)

        # Compute Cartesian coordinates
        x_raw, y_raw, z_raw = antenna_vectors_to_cartesian(
            ranges, azimuths, elevations, ke=ke
        )

        # Extract site coordinates
        site_lat = float(ds.coords.get("latitude", 0.0))
        site_lon = float(ds.coords.get("longitude", 0.0))
        site_alt = float(ds.coords.get("altitude", 0.0))
        elevation_angle = float(
            ds.coords.get("elevation_angle", np.mean(elevations))
        )

        if radar_lat is None:
            radar_lat = site_lat
            radar_lon = site_lon
            radar_alt = site_alt

        # Compute per-gate geographic coordinates from Cartesian offsets
        gate_lat, gate_lon, gate_alt = cartesian_to_geographic(
            x_raw, y_raw, z_raw, radar_lat, radar_lon, radar_alt
        )

        for az_idx in range(n_az):
            az = float(azimuths[az_idx])
            for rng_idx in range(n_rng):
                rng = float(ranges[rng_idx])
                gate_id = generate_gate_id(radar, sweep_idx, az, rng)
                lut_records.append({
                    "gate_id": gate_id,
                    "sweep": sweep_idx,
                    "azimuth": az,
                    "range": rng,
                    "elevation_angle": elevation_angle,
                    "latitude": float(gate_lat[az_idx, rng_idx]),
                    "longitude": float(gate_lon[az_idx, rng_idx]),
                    "altitude": float(gate_alt[az_idx, rng_idx]),
                    "x": float(x_raw[az_idx, rng_idx]),
                    "y": float(y_raw[az_idx, rng_idx]),
                    "z": float(z_raw[az_idx, rng_idx]),
                })

        sweep_meta[sweep_idx] = {
            "n_azimuths": n_az,
            "n_ranges": n_rng,
            "elevation": round(elevation_angle, 2),
        }

    df_lut = pd.DataFrame.from_records(lut_records)
    logger.info(
        "LUT built: %d total gates, %d sweeps.", len(df_lut), len(sweep_meta)
    )

    radar_info = {
        "radar": radar,
        "network": network,
        "latitude": radar_lat,
        "longitude": radar_lon,
        "altitude": radar_alt,
        "sweeps": sweep_meta,
    }

    return _save_lut_outputs(lut_dir, radar, df_lut, radar_info)


# ============================================================================
# LUT storage helpers
# ============================================================================

def _save_lut_outputs(lut_dir, radar, df_lut, radar_info):
    """Save LUT parquet and radar info YAML to disk."""
    lut_dir = Path(lut_dir)
    lut_dir.mkdir(parents=True, exist_ok=True)
    lut_path = lut_dir / f"{radar}_LUT.parquet"
    info_path = lut_dir / f"{radar}_info.yaml"

    if lut_path.exists() and info_path.exists():
        logger.info(
            "LUT and radar info already exist at %s -- skipping creation.",
            lut_dir,
        )
        return str(lut_path)

    df_lut.to_parquet(lut_path, index=False, engine="pyarrow")
    logger.info("LUT saved -> %s", lut_path)

    with open(info_path, "w") as f:
        yaml.dump(radar_info, f, default_flow_style=False, sort_keys=False)
    logger.info("Radar info saved -> %s", info_path)
    return str(lut_path)


# ============================================================================
# Loaders
# ============================================================================

def load_radar_lut(
    radar: str, lut_base_path: str | Path
) -> pd.DataFrame:
    """Load the LUT parquet for a radar."""
    lut_path = Path(lut_base_path) / radar / "LUT" / f"{radar}_LUT.parquet"
    if not lut_path.exists():
        raise FileNotFoundError(f"LUT not found at {lut_path}.")
    return pd.read_parquet(lut_path, engine="pyarrow")


def load_radar_info(
    radar: str, lut_base_path: str | Path
) -> dict:
    """Load the radar info YAML for a radar."""
    info_path = (
        Path(lut_base_path) / radar / "LUT" / f"{radar}_info.yaml"
    )
    if not info_path.exists():
        raise FileNotFoundError(f"Info not found at {info_path}.")
    with open(info_path) as f:
        return yaml.safe_load(f)


def get_full_sweep_index(
    lut_df: pd.DataFrame, sweep: int
) -> pd.MultiIndex:
    """Get the full (azimuth, range) MultiIndex for a sweep from the LUT."""
    sweep_lut = lut_df[lut_df["sweep"] == sweep]
    if sweep_lut.empty:
        raise ValueError(f"No LUT entries found for sweep={sweep}.")
    return sweep_lut.set_index(["azimuth", "range"]).index
