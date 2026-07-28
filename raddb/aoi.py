"""
raddb/aoi.py
------------
Area-of-Interest (AOI) selection internals.

These are the **private** building blocks behind the public ``RadDB.crop_*``
methods.  The design is LUT-first: an AOI geometry is intersected once with the
static per-radar LUT **centroids** (in Swiss LV95 / EPSG:2056) to resolve a set
of ``gate_id`` values, and that set then filters any number of dynamic volume
DataFrames — cheaply and identically across timesteps.

Because the intersection is done on gate centroids over **all** sweeps, a single
horizontal footprint selects the whole vertical column above it (every elevation).

Coordinate convention: AOI geometries are handled in EPSG:2056 throughout; input
in another CRS is reprojected via :func:`_reproject_to_2056` before intersection.

No pyart / geocube dependency — only numpy, polars, pandas, shapely, pyproj.

The centroid tables are **polars** frames (the package-wide default); the spatial
predicate itself runs on plain numpy arrays via shapely, so no geometry objects
are ever materialised for the ~1.7M gates of a full LUT.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import shapely
import shapely.ops

from raddb.lut import RADAR_TO_IDX, add_lut_projection

logger = logging.getLogger(__name__)

# EPSG code of the canonical AOI working frame (Swiss LV95 / CH1903+).
AOI_EPSG: int = 2056

# LUT columns kept for centroid intersection (small footprint, cached in RAM).
_CENTROID_COLS = ["gate_id", "sweep", "x_2056", "y_2056", "z", "altitude"]

# gate_id encoding base: radar_idx * 10**12 + ...  (see raddb.lut.encode_gate_ids)
_GATE_ID_RADAR_BASE = np.int64(1_000_000_000_000)

# Inverse of RADAR_TO_IDX for decoding radar letters from gate_ids.
_IDX_TO_RADAR = {v: k for k, v in RADAR_TO_IDX.items()}

# Per-radar centroid cache: {(base_path, radar): DataFrame}. LUTs are static, so
# a radar's centroid table is loaded from disk at most once per session.
_CENTROID_CACHE: dict[tuple[str, str], pl.DataFrame] = {}


def _radars_from_gate_ids(gate_ids) -> list[str]:
    """Decode the distinct radar letters present in a set of ``gate_id`` values.

    The radar index is the leading factor of the gate_id encoding
    (``radar_idx = gate_id // 10**12``), so the radars a DataFrame spans can be
    inferred without a separate ``radar`` column.

    Parameters
    ----------
    gate_ids : array-like of int64

    Returns
    -------
    list of str
        Sorted radar letters (e.g. ``["L", "P"]``).  Indices with no known
        letter are skipped with a warning.
    """
    ids = np.asarray(gate_ids, dtype=np.int64)
    if ids.size == 0:
        return []
    idxs = np.unique(ids // _GATE_ID_RADAR_BASE)
    radars = []
    for i in idxs:
        letter = _IDX_TO_RADAR.get(int(i))
        if letter is None:
            logger.warning("gate_id radar index %d has no known radar letter.", int(i))
            continue
        radars.append(letter)
    return sorted(radars)


def _lut_centroids(base_path: str | Path, radars: list[str]) -> pl.DataFrame:
    """Load and concatenate LUT centroid tables for the given radars.

    Returns one row per gate with ``[gate_id, radar, sweep, x_2056, y_2056, z,
    altitude]``.  Results are cached per (base_path, radar); ``x_2056``/``y_2056``
    are computed on the fly via :func:`add_lut_projection` if a LUT predates the
    projected-coordinate columns.

    Parameters
    ----------
    base_path : str or Path
        RadDB archive base directory.
    radars : list of str
        Radar letters whose LUTs to load (e.g. ``["L", "P"]``).

    Returns
    -------
    pl.DataFrame
    """
    base = Path(base_path)
    frames = []
    for radar in radars:
        key = (str(base), radar)
        cached = _CENTROID_CACHE.get(key)
        if cached is None:
            cached = _load_one_centroid_table(base, radar)
            _CENTROID_CACHE[key] = cached
        frames.append(cached)
    if not frames:
        return pl.DataFrame(
            schema={"gate_id": pl.Int64, "radar": pl.String, "sweep": pl.Int32,
                    "x_2056": pl.Float64, "y_2056": pl.Float64,
                    "z": pl.Float64, "altitude": pl.Float64}
        )
    return pl.concat(frames, how="vertical")


def _load_one_centroid_table(base: Path, radar: str) -> pl.DataFrame:
    """Load a single radar's LUT centroid columns (helper for :func:`_lut_centroids`)."""
    lut_path = base / radar / "LUT" / f"{radar}_LUT.parquet"
    if not lut_path.exists():
        raise FileNotFoundError(
            f"LUT not found at {lut_path}. Cannot resolve AOI for radar {radar!r}."
        )

    # Column names come from the parquet footer — reading the whole LUT just to
    # inspect `.columns` would load ~100 MB and throw it away.
    available = set(pq.read_schema(lut_path).names)
    has_proj = {"x_2056", "y_2056"}.issubset(available)

    if has_proj:
        cols = [c for c in _CENTROID_COLS if c in available]
        lut = pl.read_parquet(lut_path, columns=cols)
    else:
        # Older LUT without projected coords: load lat/lon + geometry and project.
        base_cols = ["gate_id", "sweep", "z", "altitude", "latitude", "longitude"]
        cols = [c for c in base_cols if c in available]
        lut = add_lut_projection(pl.read_parquet(lut_path, columns=cols), epsg=AOI_EPSG)
        lut = lut.select([c for c in _CENTROID_COLS if c in lut.columns])

    return lut.with_columns(pl.lit(radar).alias("radar"))


# proj4 definitions for the AOI's canonical frames.  Using proj4 strings (rather
# than EPSG lookups) keeps reprojection working even where the PROJ database is
# unavailable — the same tactic raddb.lut.add_lut_projection uses for WGS-84.
# The LV95 string is the standard 7-parameter CH1903+ definition; it agrees with
# the grid-based EPSG:2056 transform to well under a metre, negligible at the
# kilometre scale of AOI gate selection.
_PROJ4 = {
    4326: "+proj=longlat +datum=WGS84 +no_defs",
    2056: (
        "+proj=somerc +lat_0=46.9524055555556 +lon_0=7.43958333333333 "
        "+k_0=1 +x_0=2600000 +y_0=1200000 +ellps=bessel "
        "+towgs84=674.374,15.056,405.346,0,0,0,0 +units=m +no_defs"
    ),
}


def _to_pyproj_crs(spec):
    """Build a ``pyproj.CRS``, preferring DB-free proj4 strings for known frames.

    Falls back to :func:`pyproj.CRS` for other specs (which needs a working PROJ
    database); a proj4 string is routed to ``from_proj4`` so custom definitions
    work database-free too.
    """
    import pyproj

    if isinstance(spec, pyproj.CRS):
        return spec
    if isinstance(spec, int) and spec in _PROJ4:
        return pyproj.CRS.from_proj4(_PROJ4[spec])
    if isinstance(spec, str) and "+proj" in spec:
        return pyproj.CRS.from_proj4(spec)
    return pyproj.CRS(spec)


def _reproject_to_2056(geom, crs: int | str | None):
    """Reproject a shapely geometry into EPSG:2056 if it is in another CRS.

    Parameters
    ----------
    geom : shapely geometry
        AOI geometry expressed in ``crs``.
    crs : int, str, or None
        CRS of ``geom``.  ``None`` or ``2056`` (int or ``"EPSG:2056"``) is a
        no-op — the primary workflow, needing no pyproj at all.  Anything else is
        transformed via pyproj (``always_xy=True``); WGS-84 (4326) and LV95 (2056)
        work without a PROJ database, other frames need one installed.

    Returns
    -------
    shapely geometry in EPSG:2056.
    """
    if crs is None:
        return geom
    # Normalise common spellings of "already 2056" → no-op, no pyproj needed.
    if crs in (AOI_EPSG, str(AOI_EPSG), f"EPSG:{AOI_EPSG}", f"epsg:{AOI_EPSG}"):
        return geom

    src = _to_pyproj_crs(crs)
    tgt = _to_pyproj_crs(AOI_EPSG)
    if src.equals(tgt):
        return geom

    import pyproj

    transformer = pyproj.Transformer.from_crs(src, tgt, always_xy=True)
    return shapely.ops.transform(transformer.transform, geom)


def _resolve_aoi_centroids(centroids: pl.DataFrame, aoi_geom_2056) -> pl.DataFrame:
    """Return the centroid rows whose ``(x_2056, y_2056)`` lie inside an AOI.

    A cheap bounding-box mask pre-filters the (potentially millions of) centroids
    before the vectorized ``shapely.contains`` point-in-geometry test, which keeps
    the intersection at a few milliseconds even for arbitrary polygons.

    Because ``centroids`` spans every sweep, the result covers the full vertical
    column above the AOI footprint.  Returning the rows (not just the ids) lets
    callers clip by ``altitude`` and merge geometry without a second lookup.

    Parameters
    ----------
    centroids : pl.DataFrame
        Output of :func:`_lut_centroids` (needs ``x_2056``, ``y_2056``, ``gate_id``).
    aoi_geom_2056 : shapely geometry
        AOI footprint in EPSG:2056.

    Returns
    -------
    pl.DataFrame
        Subset of ``centroids`` inside the AOI (may be empty).
    """
    if centroids.is_empty():
        return centroids.clear()

    x = centroids["x_2056"].to_numpy()
    y = centroids["y_2056"].to_numpy()

    minx, miny, maxx, maxy = aoi_geom_2056.bounds
    bbox_mask = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
    if not bbox_mask.any():
        return centroids.clear()

    # Exact point-in-geometry only on the bbox survivors.
    pts = shapely.points(x[bbox_mask], y[bbox_mask])
    inside = shapely.contains(aoi_geom_2056, pts)

    keep = np.zeros(len(x), dtype=bool)
    keep[np.flatnonzero(bbox_mask)[inside]] = True
    return centroids.filter(keep)


def _resolve_gate_ids(centroids: pl.DataFrame, aoi_geom_2056) -> np.ndarray:
    """Resolve the ``gate_id`` set whose centroid lies inside an AOI geometry.

    Thin wrapper over :func:`_resolve_aoi_centroids` returning just the ids.

    Returns
    -------
    np.ndarray of int64
        gate_ids inside the AOI (may be empty).
    """
    sub = _resolve_aoi_centroids(centroids, aoi_geom_2056)
    if sub.is_empty():
        return np.empty(0, dtype=np.int64)
    return sub["gate_id"].to_numpy().astype(np.int64, copy=False)


def _apply_gate_ids(df: pl.DataFrame, gate_ids: np.ndarray) -> pl.DataFrame:
    """Filter a dynamic DataFrame to the rows whose ``gate_id`` is in ``gate_ids``.

    A semi-join: rows are **selected**, never widened — the static LUT geometry
    stays in its own table and is joined only by the ``to_*`` converters.

    Parameters
    ----------
    df : pl.DataFrame
        Dynamic volume data (must have a ``gate_id`` column).
    gate_ids : np.ndarray
        gate_id set from :func:`_resolve_gate_ids`.

    Returns
    -------
    pl.DataFrame
        Subset of ``df`` holding exactly the AOI rows, with ``df``'s columns.
    """
    if "gate_id" not in df.columns:
        raise KeyError("dynamic DataFrame has no 'gate_id' column; cannot apply AOI selection.")
    if len(gate_ids) == 0:
        return df.clear()
    keep = pl.DataFrame({"gate_id": np.asarray(gate_ids, dtype=np.int64)})
    return df.join(keep, on="gate_id", how="semi")


# ============================================================================
# Context geometry (map background for quicklooks)
# ============================================================================

# Loaded-once cache for the reprojected Swiss border ("unset" = not yet tried).
_SWISS_BORDER: object = "unset"


def _swiss_border_2056():
    """Return Switzerland's border as a shapely geometry in EPSG:2056, or None.

    Loaded once per process from cartopy's cached Natural Earth 10 m admin-0
    dataset and reprojected with the DB-free transform.  Returns ``None`` (with a
    warning) if cartopy or the dataset is unavailable, so a quicklook still works
    — just without the country outline.
    """
    global _SWISS_BORDER
    if _SWISS_BORDER != "unset":
        return _SWISS_BORDER
    try:
        from cartopy.io import shapereader as shpreader

        path = shpreader.natural_earth(
            resolution="10m", category="cultural", name="admin_0_countries"
        )
        geom = None
        for rec in shpreader.Reader(path).records():
            attrs = rec.attributes
            if attrs.get("NAME") == "Switzerland" or attrs.get("ADMIN") == "Switzerland":
                geom = rec.geometry
                break
        # simplify (~300 m) to keep the outline light, then project to LV95.
        _SWISS_BORDER = None if geom is None else _reproject_to_2056(geom.simplify(0.003), 4326)
    except Exception as exc:  # noqa: BLE001 - context is optional; never fatal
        logger.warning("Swiss border context unavailable (%s); drawn without it.", exc)
        _SWISS_BORDER = None
    return _SWISS_BORDER


def _resolve_context(context):
    """Resolve a map-background context geometry to EPSG:2056, or ``None``.

    Parameters
    ----------
    context : None, str, shapely geometry, or GeoDataFrame/GeoSeries
        - ``None`` → no context.
        - ``"switzerland"`` (or ``"ch"``) → the Swiss border (Natural Earth).
        - a shapely geometry → assumed already in EPSG:2056.
        - a GeoDataFrame/GeoSeries → dissolved and reprojected from its ``.crs``.

    Returns
    -------
    shapely geometry in EPSG:2056, or None.
    """
    if context is None:
        return None

    import shapely.geometry.base as _base

    if isinstance(context, str):
        if context.lower() in ("switzerland", "ch", "suisse", "schweiz", "svizzera"):
            return _swiss_border_2056()
        raise ValueError(
            f"unknown context {context!r}; use 'switzerland', None, or a geometry."
        )
    if isinstance(context, _base.BaseGeometry):
        return context  # assume already EPSG:2056
    if hasattr(context, "crs"):  # GeoDataFrame / GeoSeries
        try:
            geom = context.union_all() if hasattr(context, "union_all") else context.unary_union
            crs = getattr(context, "crs", None)
            epsg = crs.to_epsg() if crs is not None else None
            return _reproject_to_2056(geom, epsg) if epsg not in (None, AOI_EPSG) else geom
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not resolve context geometry (%s); drawn without it.", exc)
            return None
    return None


# ============================================================================
# Polygon AOI loading  (shapely / GeoDataFrame / shapefile / GeoJSON)
# ============================================================================

def _load_aoi_polygon(polygon, crs=None):
    """Resolve a polygon AOI to a shapely geometry in EPSG:2056.

    Accepts a shapely ``Polygon``/``MultiPolygon``, a GeoDataFrame/GeoSeries, or a
    path to a ``.shp`` or ``.geojson`` / ``.json`` file.  The source CRS is taken
    from ``crs`` when given, else auto-detected (GeoDataFrame ``.crs``; a GeoJSON
    ``crs`` member or WGS-84; a shapefile ``.prj``), else assumed EPSG:2056.

    File reading is dependency-light: GeoJSON via :mod:`json` + shapely, shapefiles
    via :mod:`shapefile` (pyshp) — no geopandas/pyogrio/fiona needed.
    """
    import shapely.geometry.base as _base

    src_crs = None
    if isinstance(polygon, (str, Path)):
        geom, src_crs = _read_polygon_file(Path(polygon))
    elif isinstance(polygon, _base.BaseGeometry):
        geom = polygon
    elif hasattr(polygon, "crs"):  # GeoDataFrame / GeoSeries
        geom = polygon.union_all() if hasattr(polygon, "union_all") else polygon.unary_union
        src_crs = _crs_to_spec(getattr(polygon, "crs", None))
    else:
        raise TypeError(
            f"crop_polygon: unsupported polygon input {type(polygon).__name__}; pass a "
            "shapely (Multi)Polygon, a GeoDataFrame, or a .shp / .geojson path."
        )

    if geom is None or geom.is_empty:
        raise ValueError("crop_polygon: the polygon AOI is empty.")
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(
            f"crop_polygon expects a Polygon/MultiPolygon; got {geom.geom_type}."
        )

    effective = crs if crs is not None else (src_crs if src_crs is not None else AOI_EPSG)
    return _reproject_to_2056(geom, effective)


def _crs_to_spec(crs):
    """Reduce a pyproj CRS (or None) to an EPSG int when possible, else pass through."""
    if crs is None:
        return None
    try:
        epsg = crs.to_epsg()
        return epsg if epsg is not None else crs
    except Exception:  # noqa: BLE001
        return crs


def _read_polygon_file(path: Path):
    """Read a polygon file → ``(geom, src_crs)``.  ``.shp`` via pyshp, ``.geojson`` via json."""
    if not path.exists():
        raise FileNotFoundError(f"Polygon file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".geojson", ".json"):
        return _read_geojson(path)
    if suffix == ".shp":
        return _read_shapefile(path)
    raise ValueError(f"Unsupported polygon file type {suffix!r}; use .shp or .geojson.")


def _read_geojson(path: Path):
    import json
    from shapely.geometry import shape

    data = json.loads(path.read_text())
    kind = data.get("type")
    if kind == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in data.get("features", []) if f.get("geometry")]
    elif kind == "Feature":
        geoms = [shape(data["geometry"])]
    elif kind in ("Polygon", "MultiPolygon", "GeometryCollection"):
        geoms = [shape(data)]
    else:
        raise ValueError(f"Unrecognised GeoJSON object type {kind!r}.")
    if not geoms:
        raise ValueError(f"No geometries found in {path}.")
    return shapely.union_all(geoms), _geojson_crs(data)


def _geojson_crs(data):
    """CRS of a GeoJSON: legacy ``crs`` member if present, else WGS-84 (RFC 7946)."""
    member = data.get("crs")
    if isinstance(member, dict):
        import re

        name = str(member.get("properties", {}).get("name", ""))
        m = re.search(r"EPSG:{1,2}(\d+)", name)
        if m:
            return int(m.group(1))
    return 4326


def _read_shapefile(path: Path):
    try:
        import shapefile  # pyshp
    except ImportError as exc:
        raise ImportError(
            "Reading .shp needs pyshp (or pass a GeoDataFrame / .geojson). "
            "Install with: pip install pyshp"
        ) from exc
    from shapely.geometry import shape

    reader = shapefile.Reader(str(path))
    geoms = [shape(s.__geo_interface__) for s in reader.shapes() if s.shapeType]
    if not geoms:
        raise ValueError(f"No geometries found in {path}.")
    return shapely.union_all(geoms), _prj_crs(path.with_suffix(".prj"))


def _prj_crs(prj_path: Path):
    """Best-effort CRS from a shapefile ``.prj`` (WKT): EPSG int, WKT string, or None."""
    if not prj_path.exists():
        return None
    wkt = prj_path.read_text().strip()
    if not wkt:
        return None
    try:
        import pyproj

        return pyproj.CRS.from_wkt(wkt).to_epsg() or wkt
    except Exception:  # noqa: BLE001 - broken PROJ db: sniff for Swiss LV95, else pass WKT
        low = wkt.lower()
        if "2056" in wkt or "ch1903+" in low or "lv95" in low:
            return AOI_EPSG
        return wkt


# ============================================================================
# Vertical cross-section geometry
# ----------------------------------------------------------------------------
# Ported from the RADAR_POLYGONS_CROSS_SECTIONS prototypes:
#   - gate half-dimensions + corner math  <- radDB_csts.get_rad_db /
#     get_gate_corner_xyzd / get_gate_3D_coordinates ("Eo" face = horizontal
#     footprint quad)
#   - line/footprint intersection + beam-following endpoint altitudes +
#     perpendicular +-dE offsets -> per-gate polygon in the
#     (distance-along-line, altitude) plane
#     <- radDB_spatial_plot.get_rad_gdb_vert_cross_section
# Adapted to the current data model (x_2056/y_2056/altitude from the LUT) and
# fully vectorised (numpy point-to-segment prefilter replaces the KDTree).
# ============================================================================

_CS_LUT_COLS = [
    "gate_id", "sweep", "azimuth", "range", "elevation_angle",
    "x_2056", "y_2056", "altitude",
]
# Per-radar cross-section geometry cache: {(base_path, radar, beamwidth): df}.
_CS_CACHE: dict = {}


def _lut_cs_table(base_path: str | Path, radars: list[str], beamwidth_deg: float = 1.0) -> pd.DataFrame:
    """Static per-gate geometry for cross-sections: centers + half-dimensions.

    Half-dimensions (prototype convention):
      - ``dR``: half the radial gate spacing, derived per sweep from the LUT's
        range grid (500 m -> 250 m for MCH radars);
      - ``dA`` (= dE): half the across-beam extent, ``range * tan(beamwidth/2)``
        — grows with range.
    """
    frames = []
    for radar in radars:
        key = (str(base_path), radar, float(beamwidth_deg))
        t = _CS_CACHE.get(key)
        if t is None:
            lut_path = Path(base_path) / radar / "LUT" / f"{radar}_LUT.parquet"
            if not lut_path.exists():
                raise FileNotFoundError(
                    f"LUT not found at {lut_path}. Cannot build cross-section for radar {radar!r}."
                )
            t = pd.read_parquet(lut_path, columns=_CS_LUT_COLS, engine="pyarrow")
            # Radial spacing per sweep from the unique range grid -> dR = spacing/2.
            ur = t[["sweep", "range"]].drop_duplicates().sort_values(["sweep", "range"])
            spacing = ur.groupby("sweep")["range"].apply(
                lambda r: float(np.median(np.diff(r.to_numpy(dtype=np.float64))))
            )
            t = t.copy()
            t["dR"] = (t["sweep"].map(spacing) / 2.0).astype(np.float64)
            t["dA"] = t["range"].to_numpy(dtype=np.float64) * np.tan(np.deg2rad(beamwidth_deg / 2.0))
            t["radar"] = radar
            _CS_CACHE[key] = t
        frames.append(t)
    if not frames:
        raise ValueError("no radars given for cross-section geometry.")
    return pd.concat(frames, ignore_index=True)


def _gate_footprints(sub: pd.DataFrame, half_bw_tan: float) -> np.ndarray:
    """Horizontal footprint quad per gate (the prototype's ``Eo_xyz`` face).

    Corner = center displaced along the beam (±dR, with the elevation-angle
    horizontal foreshortening cos(el)) and across it (±dA evaluated at
    range±dR).  Vectorised over all gates -> array of shapely Polygons.
    """
    az = np.deg2rad(sub["azimuth"].to_numpy(dtype=np.float64))
    el = np.deg2rad(sub["elevation_angle"].to_numpy(dtype=np.float64))
    xc = sub["x_2056"].to_numpy(dtype=np.float64)
    yc = sub["y_2056"].to_numpy(dtype=np.float64)
    rng = sub["range"].to_numpy(dtype=np.float64)
    dR = sub["dR"].to_numpy(dtype=np.float64)

    sin_az, cos_az, cos_el = np.sin(az), np.cos(az), np.cos(el)
    rings = []
    for s_r, s_a in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
        dr = s_r * dR
        da = s_a * (rng + dr) * half_bw_tan
        rings.append(np.stack([
            xc + dr * cos_el * sin_az + da * cos_az,
            yc + dr * cos_el * cos_az - da * sin_az,
        ], axis=1))
    return shapely.polygons(np.stack(rings, axis=1))


def _endpoint_d_z(pt_xy: np.ndarray, sub: pd.DataFrame, origin: tuple[float, float]):
    """(distance-along-line, altitude) of chord endpoints on the beam surface.

    The endpoint's altitude follows the gate's tilted beam: project the
    center->endpoint vector onto the beam direction (angle ``delta`` between
    them) and climb ``u * tan(el)`` from the gate-center altitude.
    """
    az = np.deg2rad(sub["azimuth"].to_numpy(dtype=np.float64))
    el = np.deg2rad(sub["elevation_angle"].to_numpy(dtype=np.float64))
    dx = pt_xy[:, 0] - sub["x_2056"].to_numpy(dtype=np.float64)
    dy = pt_xy[:, 1] - sub["y_2056"].to_numpy(dtype=np.float64)
    delta = (2.0 * np.pi - az - np.arctan2(dy, dx) + np.pi / 2.0) % (2.0 * np.pi)
    u = np.hypot(dx, dy) * np.cos(delta)
    z = sub["altitude"].to_numpy(dtype=np.float64) + u * np.tan(el)
    d = np.hypot(pt_xy[:, 0] - origin[0], pt_xy[:, 1] - origin[1])
    return d, z


def _cross_section_gates(
    cs_t: pd.DataFrame,
    p1: tuple[float, float],
    p2: tuple[float, float],
    beamwidth_deg: float = 1.0,
    min_chord_m: float = 0.5,
) -> pd.DataFrame:
    """Gates whose horizontal footprint crosses the line ``p1 -> p2``.

    Returns one row per crossed gate with its cross-section geometry:
    chord endpoints ``(d_near, z_near) / (d_far, z_far)``, center
    ``(d_center, z_center)``, and ``cs_polygon`` — the 4-corner shapely polygon
    in the (distance-along-line [m], altitude [m ASL]) plane, built by
    offsetting the chord perpendicularly by ±dA (the vertical half-beamwidth
    extent).  ``d`` is measured from ``p1``.
    """
    ox, oy = float(p1[0]), float(p1[1])
    ex, ey = float(p2[0]), float(p2[1])
    length = float(np.hypot(ex - ox, ey - oy))
    if length < 1.0:
        raise ValueError("cross-section line is degenerate (< 1 m long).")
    half_bw_tan = float(np.tan(np.deg2rad(beamwidth_deg / 2.0)))

    # --- vectorised point-to-segment prefilter (replaces the KDTree) ---
    px = cs_t["x_2056"].to_numpy(dtype=np.float64)
    py = cs_t["y_2056"].to_numpy(dtype=np.float64)
    diag = np.hypot(cs_t["dR"].to_numpy(dtype=np.float64),
                    cs_t["dA"].to_numpy(dtype=np.float64)) * 1.05
    t_par = ((px - ox) * (ex - ox) + (py - oy) * (ey - oy)) / (length * length)
    t_par = np.clip(t_par, 0.0, 1.0)
    dist = np.hypot(px - (ox + t_par * (ex - ox)), py - (oy + t_par * (ey - oy)))
    sub = cs_t[dist <= diag].reset_index(drop=True)
    if sub.empty:
        return sub.assign(cs_polygon=pd.Series(dtype=object))

    # --- exact footprint / line intersection -> per-gate chord ---
    footprints = _gate_footprints(sub, half_bw_tan)
    line = shapely.LineString([(ox, oy), (ex, ey)])
    chords = shapely.intersection(footprints, line)
    keep = ~shapely.is_empty(chords)
    sub, chords = sub[keep].reset_index(drop=True), chords[keep]
    if sub.empty:
        return sub.assign(cs_polygon=pd.Series(dtype=object))

    # First/last coordinate of each chord (Points collapse to zero length).
    coords, gidx = shapely.get_coordinates(chords, return_index=True)
    starts = np.searchsorted(gidx, np.arange(len(chords)))
    ends = np.append(starts[1:], len(gidx)) - 1
    p0, p1_ = coords[starts], coords[ends]
    good = np.hypot(*(p1_ - p0).T) > min_chord_m
    sub, p0, p1_ = sub[good].reset_index(drop=True), p0[good], p1_[good]
    if sub.empty:
        return sub.assign(cs_polygon=pd.Series(dtype=object))

    # --- endpoint (d, z) on the beam, ordered near/far along the line ---
    d0, z0 = _endpoint_d_z(p0, sub, (ox, oy))
    d1, z1 = _endpoint_d_z(p1_, sub, (ox, oy))
    swap = d0 > d1
    d_near = np.where(swap, d1, d0)
    d_far = np.where(swap, d0, d1)
    z_near = np.where(swap, z1, z0)
    z_far = np.where(swap, z0, z1)

    # --- perpendicular ±dE offsets -> (d, z) polygon per gate ---
    incl = np.arctan2(z_far - z_near, d_far - d_near)   # chord inclination
    s_i, c_i = np.sin(incl), np.cos(incl)
    dE = sub["dA"].to_numpy(dtype=np.float64)           # dE == dA (same beamwidth)
    ring = np.stack([
        np.stack([d_near - s_i * dE, z_near + c_i * dE], axis=1),   # near, top
        np.stack([d_far - s_i * dE, z_far + c_i * dE], axis=1),     # far, top
        np.stack([d_far + s_i * dE, z_far - c_i * dE], axis=1),     # far, bottom
        np.stack([d_near + s_i * dE, z_near - c_i * dE], axis=1),   # near, bottom
    ], axis=1)

    out = sub.copy()
    out["d_near"] = d_near
    out["d_far"] = d_far
    out["z_near"] = z_near
    out["z_far"] = z_far
    out["d_center"] = 0.5 * (d_near + d_far)
    out["z_center"] = 0.5 * (z_near + z_far)
    out["cs_polygon"] = shapely.polygons(ring)
    return out


# ============================================================================
# Regular-grid resampling  (AOI gates -> per-radar 3-D grid in LV95 + altitude)
# ============================================================================

# Measurement columns eligible for gridding (auto-detected from the df).
_GRID_VARIABLES = [
    "DBZH", "DBZH_raw", "ZDR", "ZDR_raw", "KDP", "RHOHV", "PHIDP",
    "HC_MCH", "HC_PYART", "HZT", "TEMP",
]
# Refuse to build absurdly large grids by accident.
_GRID_MAX_CELLS = 50_000_000


def _grid_axis(vmin: float, vmax: float, step: float) -> np.ndarray:
    """Bin edges aligned to multiples of ``step`` covering [vmin, vmax]."""
    lo = np.floor(vmin / step) * step
    hi = np.ceil(vmax / step) * step
    if hi <= lo:
        hi = lo + step
    return np.arange(lo, hi + 0.5 * step, step)


def _aoi_to_grid(
    base_path,
    df: pd.DataFrame,
    variables=None,
    resolution: float = 1000.0,
    z_resolution: float = 500.0,
    bounds=None,
    z_range=None,
    radars=None,
    beamwidth_deg: float = 1.0,
    time_round: str | None = "5min",
):
    """Resample AOI gates onto a regular per-radar 3-D grid.

    Output dims: ``(time?, radar, z, y, x)`` —

    - ``x`` / ``y``: pixel centres in Swiss LV95 / EPSG:2056 metres,
    - ``z``: **absolute altitude** bin centres (m ASL) — sweep numbers are not
      comparable across radars (different site altitudes/elevation programs),
      so voxels are keyed by real height,
    - ``radar``: one layer per radar, **never merged** — the common grid is what
      makes per-voxel cross-radar comparison meaningful,
    - ``time``: present when the df carries ``volume_time``.

    Pixel filling uses **gate footprint coverage** (the prototype's geocube
    semantics, dependency-free): a pixel takes a gate's value when the gate's
    horizontal footprint polygon covers the pixel centre.  A voxel hit by
    several gates (overlapping beams) keeps the single gate whose altitude is
    closest to the bin centre — a real measurement, never an average.  The
    winning sweep number is recorded in the ``source_sweep`` variable.
    """
    import xarray as xr

    if "gate_id" not in df.columns:
        raise KeyError("df has no 'gate_id' column; cannot grid an AOI.")
    if df.empty:
        raise ValueError("aoi_to_grid: input DataFrame is empty.")

    if radars is None:
        radars = _radars_from_gate_ids(df["gate_id"])
    if variables is None:
        variables = [v for v in _GRID_VARIABLES if v in df.columns]
    missing = [v for v in variables if v not in df.columns]
    if missing:
        raise KeyError(f"variables not in df: {missing}")
    if not variables:
        raise ValueError("no measurement variables found in df to grid.")

    # Authoritative static geometry (incl. dR/dA for footprints) from the LUT.
    cs_t = _lut_cs_table(base_path, radars, beamwidth_deg=beamwidth_deg)
    keep = ["gate_id", *variables] + (["volume_time"] if "volume_time" in df.columns else [])
    data = df[keep].merge(cs_t, on="gate_id", how="inner")
    if data.empty:
        raise ValueError("no AOI gates matched the LUT geometry.")

    # --- axes ---
    if bounds is not None:
        xmin, ymin, xmax, ymax = bounds
    else:
        xmin, xmax = data["x_2056"].min(), data["x_2056"].max()
        ymin, ymax = data["y_2056"].min(), data["y_2056"].max()
    x_edges = _grid_axis(xmin, xmax, resolution)
    y_edges = _grid_axis(ymin, ymax, resolution)
    if z_range is not None:
        z_edges = _grid_axis(z_range[0], z_range[1], z_resolution)
    else:
        z_edges = _grid_axis(data["altitude"].min(), data["altitude"].max(), z_resolution)
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    nx, ny, nz = len(xc), len(yc), len(zc)

    times = None
    if "volume_time" in data.columns:
        # naive-UTC (tz-aware datetimes don't fit numpy/xarray time coords).
        vt = pd.to_datetime(data["volume_time"], utc=True).dt.tz_localize(None)
        # Snap to the scan cycle so near-simultaneous volumes of different
        # radars (e.g. L 22:35:02 vs P 22:35:03) share one time bin — required
        # for same-time cross-radar comparison, the per-radar grid's purpose.
        if time_round is not None:
            vt = vt.dt.floor(time_round)
        data = data.assign(_vt=vt)
        times = np.sort(data["_vt"].unique())
    n_t = len(times) if times is not None else 1
    n_cells = n_t * len(radars) * nz * ny * nx
    if n_cells > _GRID_MAX_CELLS:
        raise ValueError(
            f"grid would have {n_cells:,} cells; coarsen resolution/z_resolution "
            "or pass bounds/z_range."
        )

    # Pixel-centre points, flattened row-major as (iy, ix).
    gx, gy = np.meshgrid(xc, yc)                    # (ny, nx)
    pix_pts = shapely.points(gx.ravel(), gy.ravel())
    tree = shapely.STRtree(pix_pts)
    half_bw_tan = float(np.tan(np.deg2rad(beamwidth_deg / 2.0)))

    shape = (n_t, len(radars), nz, ny, nx)
    arrays = {v: np.full(shape, np.nan, dtype=np.float32) for v in variables}
    src_sweep = np.full(shape, -1, dtype=np.int16)

    group_cols = (["_vt"] if times is not None else []) + ["radar"]
    for gkey, sub in data.groupby(group_cols, sort=False):
        gkey = gkey if isinstance(gkey, tuple) else (gkey,)
        it = int(np.searchsorted(times, np.datetime64(gkey[0]))) if times is not None else 0
        ir = radars.index(gkey[-1])

        # footprint polygons of this radar's gates (all sweeps at once — hits are
        # later resolved per-voxel by altitude, so sweeps can be batched)
        foot = _gate_footprints(sub, half_bw_tan)
        poly_i, pt_i = tree.query(foot, predicate="covers")
        if len(poly_i) == 0:
            continue

        alt = sub["altitude"].to_numpy(dtype=np.float64)[poly_i]
        iz = np.floor((alt - z_edges[0]) / z_resolution).astype(np.int64)
        ok = (iz >= 0) & (iz < nz)
        poly_i, pt_i, alt, iz = poly_i[ok], pt_i[ok], alt[ok], iz[ok]
        if len(poly_i) == 0:
            continue

        # nearest-to-bin-centre gate wins: sort hits by |alt - zc| descending and
        # write in order, so the closest gate is written last for each voxel.
        dist = np.abs(alt - zc[iz])
        order = np.argsort(-dist, kind="stable")
        p, q, z_i = poly_i[order], pt_i[order], iz[order]
        iy, ix = np.divmod(q, nx)

        sub_sweep = sub["sweep"].to_numpy(dtype=np.int16)
        src_sweep[it, ir, z_i, iy, ix] = sub_sweep[p]
        for v in variables:
            vals = sub[v].to_numpy(dtype=np.float32)
            arrays[v][it, ir, z_i, iy, ix] = vals[p]

    # --- assemble the Dataset ---
    dims = ("time", "radar", "z", "y", "x") if times is not None else ("radar", "z", "y", "x")
    if times is None:
        arrays = {v: a[0] for v, a in arrays.items()}
        src_sweep = src_sweep[0]
    coords = {"radar": list(radars), "z": zc, "y": yc, "x": xc}
    if times is not None:
        coords["time"] = times
    ds = xr.Dataset(
        {v: (dims, arrays[v]) for v in variables} | {"source_sweep": (dims, src_sweep)},
        coords=coords,
        attrs={
            "crs": "EPSG:2056",
            "resolution_m": float(resolution),
            "z_resolution_m": float(z_resolution),
            "z_units": "m ASL (absolute altitude)",
            "beamwidth_deg": float(beamwidth_deg),
            "fill_method": "gate footprint covers pixel centre; nearest-altitude gate per voxel",
        },
    )
    ds["z"].attrs["units"] = "m ASL"
    ds["x"].attrs["units"] = "m (LV95 easting)"
    ds["y"].attrs["units"] = "m (LV95 northing)"
    return ds
