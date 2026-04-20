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

# Maps single-letter radar identifiers to integer indices for numeric gate_id.
# A=0, B=1, ..., Z=25  (26 radars supported)
RADAR_TO_IDX: dict[str, int] = {chr(ord("A") + i): i for i in range(26)}


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
) -> int:
    """Create a unique gate identifier as a 64-bit integer.

    Encoding: ``radar_idx * 10^12 + sweep * 10^10 + az_int * 10^6 + range_int``

    where ``az_int = round(azimuth * 10)`` (1 decimal place precision) and
    ``range_int = int(range_m)`` (integer metres).
    """
    radar_idx = RADAR_TO_IDX[radar.upper()]
    az_int = int(round(azimuth * 10))
    range_int = int(range_m)
    return (
        radar_idx * 1_000_000_000_000
        + sweep    *    10_000_000_000
        + az_int   *         1_000_000
        + range_int
    )


# ============================================================================
# Generic LUT generation from DataTree
# ============================================================================

def generate_lut_from_datatree(
    dt: xr.DataTree,
    radar: str,
    output_base_path: str,
    ke: float = 1.25,
    network: str = "",
    projection_epsg: int | None = None,
    projection_crs=None,
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
    projection_epsg : int, optional
        EPSG code for an additional projected coordinate system to include
        in the LUT (e.g. ``2056`` for CH1903+ / LV95).  Adds
        ``x_{epsg}`` / ``y_{epsg}`` columns via :func:`add_lut_projection`.
    projection_crs : pyproj.CRS or CRS-coercible, optional
        Alternative to ``projection_epsg``.

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

    lut_dfs = []
    sweep_meta = {}
    radar_lat, radar_lon, radar_alt = None, None, None
    _radar_idx = np.int64(RADAR_TO_IDX[radar.upper()])

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

        # Vectorised int64 gate_id generation (zero string allocations)
        _az_int  = np.round(azimuths * 10).astype(np.int64)   # (n_az,)
        _rng_int = ranges.astype(np.int64)                      # (n_rng,)
        _gate_ids = (
            _radar_idx         * np.int64(1_000_000_000_000)
            + sweep_idx        * np.int64(   10_000_000_000)
            + _az_int[:, None] * np.int64(        1_000_000)
            + _rng_int[None, :]
        ).ravel()

        lut_dfs.append(pd.DataFrame({
            "gate_id":         _gate_ids,
            "sweep":           np.full(n_az * n_rng, sweep_idx, dtype=np.int32),
            "azimuth":         np.repeat(azimuths, n_rng),
            "range":           np.tile(ranges, n_az),
            "elevation_angle": np.full(n_az * n_rng, elevation_angle),
            "latitude":        gate_lat.ravel(),
            "longitude":       gate_lon.ravel(),
            "altitude":        gate_alt.ravel(),
            "x":               x_raw.ravel(),
            "y":               y_raw.ravel(),
            "z":               z_raw.ravel(),
        }))

        sweep_meta[sweep_idx] = {
            "n_azimuths": n_az,
            "n_ranges": n_rng,
            "elevation": round(elevation_angle, 2),
        }

    df_lut = pd.concat(lut_dfs, ignore_index=True)
    logger.info(
        "LUT built: %d total gates, %d sweeps.", len(df_lut), len(sweep_meta)
    )

    # Add projected coordinates if requested
    if projection_epsg is not None or projection_crs is not None:
        df_lut = add_lut_projection(
            df_lut, epsg=projection_epsg, crs=projection_crs
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


# ============================================================================
# Projection utilities
# ============================================================================

def add_lut_projection(
    lut_df: pd.DataFrame,
    epsg: int | None = None,
    crs=None,
) -> pd.DataFrame:
    """Add projected coordinates to a LUT DataFrame.

    Converts the ``latitude`` / ``longitude`` columns to the target CRS and
    appends ``x_{suffix}`` / ``y_{suffix}`` columns, where ``suffix`` is the
    EPSG code (if available) or ``"custom"``.

    Requires **pyproj** (``pip install pyproj``).

    Parameters
    ----------
    lut_df : pd.DataFrame
        LUT DataFrame with ``latitude`` and ``longitude`` columns (degrees,
        WGS-84 / EPSG:4326).
    epsg : int, optional
        EPSG code for the target CRS.
        Example: ``2056`` for CH1903+ / LV95 (Swiss national grid).
    crs : pyproj.CRS or any CRS-coercible object, optional
        Alternative to ``epsg``.  Used when ``epsg`` is ``None``.
        Can be a pyproj ``CRS`` object, a WKT string, or any value accepted
        by ``pyproj.CRS()``.

    Returns
    -------
    pd.DataFrame
        Copy of ``lut_df`` with two new columns:
        ``x_{suffix}`` (easting / metres) and ``y_{suffix}`` (northing / metres).

    Raises
    ------
    ValueError
        If neither ``epsg`` nor ``crs`` is provided.
    ImportError
        If pyproj is not installed.

    Examples
    --------
    Add Swiss LV95 (CH1903+) coordinates to a LUT:

    >>> lut_ch = add_lut_projection(lut_df, epsg=2056)
    >>> lut_ch[["x_2056", "y_2056"]].head()

    Use a custom pyproj CRS:

    >>> import pyproj
    >>> my_crs = pyproj.CRS.from_epsg(32632)   # UTM zone 32N
    >>> lut_utm = add_lut_projection(lut_df, crs=my_crs)
    """
    try:
        import pyproj
    except ImportError as exc:
        raise ImportError(
            "pyproj is required for add_lut_projection. "
            "Install it with: pip install pyproj"
        ) from exc

    if epsg is not None:
        target_crs = pyproj.CRS.from_epsg(epsg)
        col_suffix = str(epsg)
    elif crs is not None:
        target_crs = crs if isinstance(crs, pyproj.CRS) else pyproj.CRS(crs)
        detected_epsg = target_crs.to_epsg()
        col_suffix = str(detected_epsg) if detected_epsg is not None else "custom"
    else:
        raise ValueError("Provide either 'epsg' or 'crs'.")

    # Use proj4 string for WGS-84 to avoid requiring the PROJ database
    wgs84 = pyproj.CRS.from_proj4(
        "+proj=longlat +datum=WGS84 +no_defs"
    )
    transformer = pyproj.Transformer.from_crs(
        wgs84, target_crs, always_xy=True
    )
    x_proj, y_proj = transformer.transform(
        lut_df["longitude"].to_numpy(),
        lut_df["latitude"].to_numpy(),
    )

    lut_out = lut_df.copy()
    lut_out[f"x_{col_suffix}"] = x_proj
    lut_out[f"y_{col_suffix}"] = y_proj
    return lut_out
