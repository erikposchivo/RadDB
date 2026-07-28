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
import polars as pl
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

def _interpolate_range_edges(ranges: np.ndarray) -> np.ndarray:
    """Interpolate the edges of range gates (PyART's formula).

    Returns an array of size ``ranges.size + 1``: the midpoints between
    consecutive gates, with first and last edges extrapolated by half a
    step and clipped to non-negative values.
    """
    r = np.asarray(ranges, dtype=np.float64)
    edges = np.empty(r.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    edges[0]    = r[0]  - 0.5 * (r[1]  - r[0])
    edges[-1]   = r[-1] + 0.5 * (r[-1] - r[-2])
    edges[edges < 0] = 0.0
    return edges


def _interpolate_elevation_edges(elevations: np.ndarray) -> np.ndarray:
    """Interpolate elevation-angle edges (linear, clipped to [-90, 90])."""
    el = np.asarray(elevations, dtype=np.float64)
    edges = np.empty(el.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (el[:-1] + el[1:])
    edges[0]    = el[0]  - 0.5 * (el[1]  - el[0])
    edges[-1]   = el[-1] + 0.5 * (el[-1] - el[-2])
    return np.clip(edges, -90.0, 90.0)


def _interpolate_azimuth_edges(azimuths: np.ndarray) -> np.ndarray:
    """Interpolate azimuth-angle edges using complex-plane midpoints.

    Complex-number interpolation handles the 360°→0° wrap-around correctly:
    the midpoint between 359° and 1° is 0°, not 180°.

    Returns
    -------
    edges : np.ndarray
        Shape ``(azimuths.size + 1,)``, values in [0, 360).
    """
    az = np.asarray(azimuths, dtype=np.float64)
    z = np.exp(1j * np.deg2rad(az))
    midpoints = 0.5 * (z[:-1] + z[1:])
    first = z[0]  - (midpoints[0]  - z[0])
    last  = z[-1] + (z[-1] - midpoints[-1])
    edges = np.concatenate(([first], midpoints, [last]))
    return np.rad2deg(np.angle(edges)) % 360.0


def antenna_vectors_to_cartesian(
    ranges: np.ndarray,
    azimuths: np.ndarray,
    elevations: np.ndarray,
    ke: float = 4.0 / 3.0,
    edges: bool = False,
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
    edges : bool
        If True, interpolate ranges/azimuths/elevations to gate edges
        (size N+1) before computing Cartesian coords. Used for
        ``pcolormesh(shading="flat")`` rendering.

    Returns
    -------
    x, y, z : arrays, shape (n_rays, n_gates) — or (n_rays+1, n_gates+1) if ``edges``.
        Cartesian coordinates in meters relative to the radar.
    """
    if edges:
        ranges     = _interpolate_range_edges(ranges)
        azimuths   = _interpolate_azimuth_edges(azimuths)
        elevations = _interpolate_elevation_edges(elevations)

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

def encode_gate_ids(
    radar: str,
    sweeps: int | np.ndarray,
    azimuths: np.ndarray,
    ranges: np.ndarray,
) -> np.ndarray:
    """Vectorised 64-bit gate identifier encoding.

    Encoding: ``radar_idx * 10^12 + sweep * 10^10 + az_int * 10^6 + range_int``

    where ``az_int = round(azimuth * 10)`` (1 decimal place precision) and
    ``range_int = int(range_m)`` (integer metres).  This is the single
    canonical implementation, used by LUT generation and volume archiving.

    Parameters
    ----------
    radar : str
        Radar identifier (single letter, e.g. ``"A"``).
    sweeps : int or array
        Sweep number(s) — a scalar (applied to all gates) or an array
        aligned with ``azimuths`` / ``ranges``.
    azimuths, ranges : arrays (aligned per gate)
        Azimuth [deg] and range [m] of each gate.

    Returns
    -------
    np.ndarray of int64
    """
    radar_idx = np.int64(RADAR_TO_IDX[radar.upper()])
    sweep_v = np.asarray(sweeps, dtype=np.int64)
    az_int  = np.round(np.asarray(azimuths, dtype=np.float64) * 10).astype(np.int64)
    rng_int = np.asarray(ranges).astype(np.int64)
    return (
        radar_idx * np.int64(1_000_000_000_000)
        + sweep_v  * np.int64(   10_000_000_000)
        + az_int   * np.int64(        1_000_000)
        + rng_int
    )


def generate_gate_id(
    radar: str, sweep: int, azimuth: float, range_m: float
) -> int:
    """Create a unique gate identifier as a 64-bit integer.

    Scalar convenience wrapper around :func:`encode_gate_ids` — see there
    for the encoding definition.
    """
    return int(encode_gate_ids(radar, sweep, azimuth, range_m))


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
    ``raddb.mch.generate_mch_lut()`` instead.

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

        gate_az  = np.repeat(azimuths, n_rng)
        gate_rng = np.tile(ranges, n_az)
        gate_ids = encode_gate_ids(radar, sweep_idx, gate_az, gate_rng)

        lut_dfs.append(pd.DataFrame({
            "gate_id":         gate_ids,
            "sweep":           np.full(n_az * n_rng, sweep_idx, dtype=np.int32),
            "azimuth":         gate_az,
            "range":           gate_rng,
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

def compute_sweep_corners(
    ranges: np.ndarray,
    azimuths: np.ndarray,
    elevations: np.ndarray,
    radar_lat: float,
    radar_lon: float,
    radar_alt: float,
    ke: float = 4.0 / 3.0,
) -> dict:
    """Compute per-sweep gate corner arrays for pcolormesh rendering.

    Uses PyART's edge-interpolation (complex-plane for azimuth wrap-around)
    plus the standard 4/3 Earth beam propagation.

    Returns
    -------
    dict with keys ``x_edges``, ``y_edges``, ``z_edges``, ``lon_edges``,
    ``lat_edges``, each shape ``(n_az+1, n_range+1)``.
    """
    x_e, y_e, z_e = antenna_vectors_to_cartesian(
        ranges, azimuths, elevations, ke=ke, edges=True,
    )
    lat_e, lon_e, _ = cartesian_to_geographic(
        x_e, y_e, z_e, radar_lat=radar_lat, radar_lon=radar_lon, radar_alt=radar_alt,
    )
    # float64 throughout: these edges are the gate polygon vertices, and gate
    # position precision is a hard requirement (float32 costs ~20 cm, and the
    # error does not shrink with range).
    return {
        "x_edges":   x_e.astype(np.float64),
        "y_edges":   y_e.astype(np.float64),
        "z_edges":   z_e.astype(np.float64),
        "lon_edges": lon_e.astype(np.float64),
        "lat_edges": lat_e.astype(np.float64),
    }


def save_sweep_corners(
    corners_by_sweep: dict[int, dict], corners_path: str | Path
) -> str:
    """Save per-sweep corner arrays to a single ``.npz`` file.

    Parameters
    ----------
    corners_by_sweep : dict {sweep_number: {"x_edges", "y_edges", ...}}
    corners_path : str or Path
        Target ``.npz`` file path.
    """
    flat = {}
    for sw, d in corners_by_sweep.items():
        for k, v in d.items():
            flat[f"sweep_{int(sw)}_{k}"] = v
    corners_path = Path(corners_path)
    corners_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(corners_path, **flat)
    logger.info("Sweep corners saved -> %s", corners_path)
    return str(corners_path)


def compute_corners_from_lut(
    radar: str,
    lut_base_path: str | Path,
    ke: float = 4.0 / 3.0,
) -> str:
    """Rebuild ``{radar}_corners.npz`` from an existing LUT parquet + info YAML.

    Useful when the corners file is missing (e.g. for LUTs generated before
    corners were introduced). Reads the per-sweep unique (azimuth, range,
    elevation) triples from the LUT, computes corner arrays via
    :func:`compute_sweep_corners`, and saves them.

    Returns
    -------
    str : path to the written ``.npz`` file.
    """
    lut_df = load_radar_lut(radar, lut_base_path)
    info = load_radar_info(radar, lut_base_path)
    corners_by_sweep: dict[int, dict] = {}
    for sweep_num in sorted(lut_df["sweep"].unique().to_list()):
        sub = lut_df.filter(pl.col("sweep") == sweep_num)
        azimuths = np.sort(sub["azimuth"].unique().to_numpy())
        ranges   = np.sort(sub["range"].unique().to_numpy())
        n_az, n_rng = len(azimuths), len(ranges)
        # Rebuild per-ray elevation from the LUT (stored as constant per sweep
        # in elevation_angle). If the per-ray elevation varies within a sweep,
        # we'd need the raw antenna data; for Swiss radars the fixed-angle
        # approximation is accurate to < 0.1°.
        el_mean = float(sub["elevation_angle"][0])
        elevations = np.full(n_az, el_mean, dtype=np.float64)
        corners_by_sweep[int(sweep_num)] = compute_sweep_corners(
            ranges=ranges, azimuths=azimuths, elevations=elevations,
            radar_lat=info["latitude"],
            radar_lon=info["longitude"],
            radar_alt=info["altitude"],
            ke=ke,
        )
    corners_path = Path(lut_base_path) / radar / "LUT" / f"{radar}_corners.npz"
    save_sweep_corners(corners_by_sweep, corners_path)
    return str(corners_path)


def _parse_corners_npz(corners_path: str | Path) -> dict[int, dict]:
    """Parse a ``*_corners.npz`` file into ``{sweep: {array_name: array}}``."""
    data = np.load(corners_path)
    out: dict[int, dict] = {}
    for key in data.files:
        # key format: "sweep_{N}_{array_name}"
        parts = key.split("_", 2)
        if len(parts) != 3 or parts[0] != "sweep":
            continue
        try:
            sw = int(parts[1])
        except ValueError:
            continue
        out.setdefault(sw, {})[parts[2]] = data[key]
    return out


def load_sweep_corners(
    radar: str, lut_base_path: str | Path
) -> dict[int, dict]:
    """Load per-sweep corner arrays from ``{radar}_corners.npz``.

    Returns
    -------
    dict {sweep_number: {"x_edges", "y_edges", ...}}  — empty dict if the
    corners file does not exist (backwards-compatible).
    """
    corners_path = Path(lut_base_path) / radar / "LUT" / f"{radar}_corners.npz"
    if not corners_path.exists():
        return {}
    return _parse_corners_npz(corners_path)


# ============================================================================
# GeoArrow export  (gate wedge polygons)
# ============================================================================

# Per-radar cache of gate_id -> (sweep, azimuth index, range index) into the
# corner arrays: {(base_path, radar): pl.DataFrame}.  ~27 MB per radar, built
# once per session.
_GRID_CACHE: dict[tuple[str, str], "pl.DataFrame"] = {}


def decode_gate_ids(gate_ids) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`encode_gate_ids` (radar index excluded).

    Returns
    -------
    (sweeps, azimuths, ranges) : np.ndarray
        Sweep number (int64), azimuth in degrees (float64, 1 decimal) and range
        in metres (float64), one entry per gate_id.
    """
    ids = np.asarray(gate_ids, dtype=np.int64)
    sweeps = (ids // np.int64(10_000_000_000)) % np.int64(100)
    az_int = (ids // np.int64(1_000_000)) % np.int64(10_000)
    rng_m = ids % np.int64(1_000_000)
    return sweeps, az_int.astype(np.float64) / 10.0, rng_m.astype(np.float64)


def _gate_grid_index(radar: str, lut_base_path: str | Path) -> "pl.DataFrame":
    """Map every ``gate_id`` to its position in the per-sweep corner arrays.

    Returns a frame ``[gate_id, sweep, az_idx, rng_idx]``.  The indices are
    computed against ``np.sort(unique(...))`` of the LUT's own azimuth/range
    values — the exact ordering :func:`compute_corners_from_lut` uses — so they
    address :func:`compute_sweep_corners` output directly.

    The lookup is keyed on ``gate_id`` rather than on azimuth/range values
    because ``gate_id`` stores azimuth rounded to 0.1° while the LUT keeps the
    raw antenna azimuth (~0.03° jitter); matching the floats would fail.
    """
    key = (str(lut_base_path), radar)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached

    lut_path = Path(lut_base_path) / radar / "LUT" / f"{radar}_LUT.parquet"
    if not lut_path.exists():
        raise FileNotFoundError(f"LUT not found at {lut_path}.")
    lut = pl.read_parquet(lut_path, columns=["gate_id", "sweep", "azimuth", "range"])

    parts = []
    for (sweep_num,), sub in lut.group_by(["sweep"]):
        az_grid = np.sort(sub["azimuth"].unique().to_numpy())
        rng_grid = np.sort(sub["range"].unique().to_numpy())
        parts.append(sub.select(
            "gate_id",
            pl.lit(int(sweep_num), dtype=pl.Int32).alias("sweep"),
            pl.Series("az_idx", np.searchsorted(az_grid, sub["azimuth"].to_numpy()), dtype=pl.Int32),
            pl.Series("rng_idx", np.searchsorted(rng_grid, sub["range"].to_numpy()), dtype=pl.Int32),
        ))
    table = pl.concat(parts, how="vertical")
    _GRID_CACHE[key] = table
    return table


def gate_polygons_geoarrow(
    radar: str,
    lut_base_path: str | Path,
    gate_ids,
    frame: str = "geographic",
):
    """Build the gate wedge polygons for ``gate_ids`` as a GeoArrow array.

    Each gate becomes a 4-corner ring (closed, 5 vertices) taken from the
    per-sweep edge arrays produced by :func:`compute_sweep_corners` — the exact
    wedge, not a rectangle. The corners file is rebuilt via
    :func:`compute_corners_from_lut` if it does not exist yet.

    Parameters
    ----------
    radar : str
        Single-letter radar identifier.
    lut_base_path : str or Path
        RadDB archive base directory.
    gate_ids : array-like of int64
        Gates to build polygons for, in the order they should appear.
    frame : {"geographic", "cartesian"}
        ``"geographic"`` uses ``lon_edges``/``lat_edges`` (EPSG:4326, the frame
        web maps expect); ``"cartesian"`` uses ``x_edges``/``y_edges`` (metres
        from the radar).

    Returns
    -------
    pyarrow.Array
        A ``geoarrow.polygon`` extension array, one polygon per gate_id.
        Gates whose (azimuth, range) is absent from the LUT grid yield null.
    """
    import pyarrow as pa

    if frame not in ("geographic", "cartesian"):
        raise ValueError(f"frame must be 'geographic' or 'cartesian'; got {frame!r}.")
    xkey, ykey = (("lon_edges", "lat_edges") if frame == "geographic"
                  else ("x_edges", "y_edges"))

    corners = load_sweep_corners(radar, lut_base_path)
    if not corners:
        compute_corners_from_lut(radar, lut_base_path)
        corners = load_sweep_corners(radar, lut_base_path)

    ids = np.asarray(gate_ids, dtype=np.int64)
    n = len(ids)
    # Left-join keeps the caller's row order and leaves unknown gates null.
    located = (
        pl.DataFrame({"gate_id": ids, "_ord": np.arange(n, dtype=np.int64)})
        .join(_gate_grid_index(radar, lut_base_path), on="gate_id", how="left")
        .sort("_ord")
    )
    sweeps = located["sweep"].to_numpy()
    az_idx = located["az_idx"].fill_null(0).to_numpy()
    rng_idx = located["rng_idx"].fill_null(0).to_numpy()
    known = located["sweep"].is_not_null().to_numpy()

    # 5 vertices x 2 coordinates per gate; NaN marks a gate we could not place.
    ring_xy = np.full((n, 5, 2), np.nan, dtype=np.float64)

    for sweep_num in np.unique(sweeps[known]):
        sw = int(sweep_num)
        if sw not in corners:
            logger.warning("sweep %d missing from the corners file; gates skipped.", sw)
            continue
        rows = np.flatnonzero(known & (sweeps == sweep_num))
        ai, ri = az_idx[rows], rng_idx[rows]
        xe, ye = corners[sw][xkey], corners[sw][ykey]

        # Ring: (az_i, r_j) -> (az_i, r_j+1) -> (az_i+1, r_j+1) -> (az_i+1, r_j) -> close.
        for k, (ii, jj) in enumerate(((0, 0), (0, 1), (1, 1), (1, 0), (0, 0))):
            ring_xy[rows, k, 0] = xe[ai + ii, ri + jj]
            ring_xy[rows, k, 1] = ye[ai + ii, ri + jj]

    valid = ~np.isnan(ring_xy[:, 0, 0])
    if not valid.all():
        logger.warning(
            "%d of %d gate_ids could not be placed on the LUT grid (null geometry).",
            int((~valid).sum()), n,
        )

    # geoarrow.polygon = List<List<FixedSizeList<double>[2]>>: polygon -> rings -> xy.
    # A null in the outer offsets makes that polygon null (unplaceable gate).
    coords = pa.FixedSizeListArray.from_arrays(
        pa.array(ring_xy.reshape(-1), type=pa.float64()), 2
    )
    rings = pa.ListArray.from_arrays(np.arange(n + 1, dtype=np.int32) * 5, coords)
    offsets = pa.array(
        np.arange(n + 1, dtype=np.int32),
        mask=np.concatenate([~valid, [False]]),
    )
    return pa.ListArray.from_arrays(offsets, rings)


def geoarrow_field(name: str, dtype, kind: str, crs: str | None = None):
    """Build a ``pyarrow.Field`` tagged as a GeoArrow extension type.

    GeoArrow identifies geometry through *field* metadata, so the tag can only be
    attached where the array is placed into a table/schema.

    Parameters
    ----------
    name : str
        Column name.
    dtype : pyarrow.DataType
        Type of the geometry array (e.g. ``polygons.type``).
    kind : str
        GeoArrow geometry kind, e.g. ``"point"`` or ``"polygon"``.
    crs : str, optional
        CRS identifier such as ``"EPSG:4326"``.
    """
    import pyarrow as pa

    meta = {b"ARROW:extension:name": f"geoarrow.{kind}".encode()}
    if crs:
        meta[b"ARROW:extension:metadata"] = (
            f'{{"crs":"{crs}","crs_type":"authority_code"}}'
        ).encode()
    return pa.field(name, dtype, metadata=meta)


def load_radar_lut(
    radar: str, lut_base_path: str | Path
) -> pl.DataFrame:
    """Load the LUT parquet for a radar as a **polars** DataFrame.

    Call :meth:`polars.DataFrame.to_pandas` on the result if you need pandas.
    """
    lut_path = Path(lut_base_path) / radar / "LUT" / f"{radar}_LUT.parquet"
    if not lut_path.exists():
        raise FileNotFoundError(f"LUT not found at {lut_path}.")
    return pl.read_parquet(lut_path)


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
    lut_df: "pl.DataFrame | pd.DataFrame",
    epsg: int | None = None,
    crs=None,
) -> "pl.DataFrame | pd.DataFrame":
    """Add projected coordinates to a LUT DataFrame.

    Converts the ``latitude`` / ``longitude`` columns to the target CRS and
    appends ``x_{suffix}`` / ``y_{suffix}`` columns, where ``suffix`` is the
    EPSG code (if available) or ``"custom"``.

    Accepts either a polars or a pandas frame and returns the same kind — LUT
    *loading* is polars, but LUT *generation* still builds pandas frames.

    Requires **pyproj** (``pip install pyproj``).

    Parameters
    ----------
    lut_df : pl.DataFrame or pd.DataFrame
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
    pl.DataFrame or pd.DataFrame
        Copy of ``lut_df`` (same kind) with two new columns:
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

    if isinstance(lut_df, pl.DataFrame):
        return lut_df.with_columns(
            pl.Series(f"x_{col_suffix}", x_proj),
            pl.Series(f"y_{col_suffix}", y_proj),
        )

    lut_out = lut_df.copy()
    lut_out[f"x_{col_suffix}"] = x_proj
    lut_out[f"y_{col_suffix}"] = y_proj
    return lut_out
