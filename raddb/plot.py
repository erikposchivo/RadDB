"""
raddb/plot.py
-------------
PPI and RHI plotting for DataTrees reconstructed from RadDB parquet archives.

A radar gate is not a rectangle: it is a curved trapezoid in geographic space
whose footprint depends on range, azimuth, elevation, and Earth curvature.
The right way to render it is to feed matplotlib's ``pcolormesh`` the 2-D
centroid coordinates of every gate — matplotlib then builds the correct
curved quadrilaterals automatically.

The reconstructed DataTree already carries the per-gate ``(lon, lat, alt, x, y, z)``
coords on every sweep Dataset (attached during ``parquet_to_datatree``), so
these plot functions do not need any geometry computation of their own.

If cartopy is available the PPI is drawn on an Azimuthal Equidistant map
centred on the radar site (coastlines, borders, gridlines). Otherwise it
falls back to a plain matplotlib axis using either lon/lat or radar-centric
cartesian coords.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import xarray as xr


# ============================================================================
# Per-variable plotting defaults
# ============================================================================

# Hydrometeor class labels for HC_MCH and HC_PYART (already shifted to 1..9).
_HC_CLASSES = [
    "NC",   # 1 no classification / no echo
    "AG",   # 2 aggregates
    "CR",   # 3 ice crystals
    "LR",   # 4 light rain
    "RP",   # 5 rimed particles
    "RN",   # 6 rain
    "VI",   # 7 vertically-aligned ice
    "WS",   # 8 wet snow
    "MH",   # 9 melting hail / heavy precipitation
]

_PLOT_DEFAULTS: dict[str, dict] = {
    "DBZH":     dict(cmap="turbo",    vmin=-10,  vmax=60,   label="Reflectivity [dBZ]"),
    "ZDR":      dict(cmap="RdBu_r",   vmin=-2,   vmax=5,    label="Differential reflectivity [dB]"),
    "RHOHV":    dict(cmap="viridis",  vmin=0.5,  vmax=1.0,  label="Co-polar correlation"),
    "PHIDP":    dict(cmap="twilight", vmin=-180, vmax=180,  label="Differential phase [deg]"),
    "HZT":      dict(cmap="viridis",  vmin=0,    vmax=5000, label="Freezing level height [m]"),
    "TEMP":     dict(cmap="RdBu_r",   vmin=-30,  vmax=15,   label="Temperature [°C]"),
    "HC_MCH":   dict(discrete=True,   classes=_HC_CLASSES,  label="MCH hydrometeor class"),
    "HC_PYART": dict(discrete=True,   classes=_HC_CLASSES,  label="PyART hydrometeor class"),
}


# ============================================================================
# Internal helpers
# ============================================================================

def _resolve_plot_kwargs(variable: str, user_kwargs: dict):
    """Merge per-variable defaults with user overrides.

    Returns (plot_kwargs, is_discrete, class_labels, cbar_label).
    """
    defaults = _PLOT_DEFAULTS.get(variable, {})
    is_discrete = bool(defaults.get("discrete", False))
    class_labels = defaults.get("classes")
    cbar_label = defaults.get("label", variable)

    plot_kwargs = dict(user_kwargs)
    if is_discrete:
        n = len(class_labels)
        base_cmap = plt.get_cmap(plot_kwargs.pop("cmap", "tab10"), n)
        cmap = ListedColormap([base_cmap(i) for i in range(n)])
        bounds = np.arange(0.5, n + 1.5)
        norm = BoundaryNorm(bounds, cmap.N)
        plot_kwargs.setdefault("cmap", cmap)
        plot_kwargs.setdefault("norm", norm)
    else:
        for k in ("cmap", "vmin", "vmax"):
            if k in defaults:
                plot_kwargs.setdefault(k, defaults[k])
    return plot_kwargs, is_discrete, class_labels, cbar_label


def _maybe_cartopy():
    """Try to import cartopy. Return (ccrs, cfeature) or (None, None)."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except ImportError:
        return None, None


def _get_sweep_dataset(dt, sweep) -> xr.Dataset:
    """Return a sweep Dataset from a DataTree, or pass-through a Dataset."""
    if isinstance(dt, xr.Dataset):
        return dt
    name = sweep if isinstance(sweep, str) else f"sweep_{int(sweep)}"
    groups = [g.lstrip("/") for g in dt.groups]
    if name not in groups:
        available = sorted(g for g in groups if g.startswith("sweep_"))
        raise KeyError(f"Sweep '{name}' not found. Available: {available}")
    return dt[name].to_dataset()


def _add_colorbar(p, ax, is_discrete: bool, class_labels, label: str):
    """Attach either a continuous or a categorical colorbar."""
    if is_discrete and class_labels is not None:
        n = len(class_labels)
        cbar = plt.colorbar(
            p, ax=ax, ticks=np.arange(1, n + 1),
            boundaries=np.arange(0.5, n + 1.5),
            spacing="uniform", fraction=0.046, pad=0.04,
        )
        cbar.ax.set_yticklabels(class_labels)
        cbar.set_label(label)
    else:
        cbar = plt.colorbar(p, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label)
    return cbar


def _draw_range_rings_xy(ax, distances_km=(50, 100, 150), **kwargs):
    """Draw range rings as dashed circles on a cartesian (km) axis."""
    kwargs.setdefault("color", "k")
    kwargs.setdefault("linewidth", 0.5)
    kwargs.setdefault("linestyle", "--")
    theta = np.linspace(0, 2 * np.pi, 361)
    for d in distances_km:
        ax.plot(d * np.cos(theta), d * np.sin(theta), **kwargs)


def _volume_time_str(ds) -> str:
    """Extract a readable timestamp from a sweep Dataset for plot titles."""
    if "time" not in ds.coords and "time" not in ds.data_vars:
        return ""
    vals = np.asarray(ds["time"].values).ravel()
    mask = ~pd.isna(vals)
    if not mask.any():
        return ""
    return str(vals[mask][0])[:19]


# ============================================================================
# PPI
# ============================================================================

def plot_ppi(
    dt,
    sweep,
    variable: str,
    ax=None,
    use_cartopy: bool | None = None,
    coords: str = "geo",
    add_range_rings: bool = True,
    add_colorbar: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 8),
    **plot_kwargs,
):
    """Plan Position Indicator from a single sweep of a reconstructed DataTree.

    Parameters
    ----------
    dt : xr.DataTree or xr.Dataset
        Reconstructed DataTree (from ``parquet_to_datatree``) or single sweep Dataset.
    sweep : int or str
        Sweep index (``3``) or group name (``"sweep_3"``). Ignored if
        ``dt`` is already a Dataset.
    variable : str
        Variable to plot: ``"DBZH"``, ``"ZDR"``, ``"RHOHV"``, ``"PHIDP"``,
        ``"HZT"``, ``"TEMP"``, ``"HC_MCH"``, ``"HC_PYART"``.
    ax : matplotlib.axes.Axes, optional
    use_cartopy : bool, optional
        If None (default), auto-detect: use cartopy when installed and
        ``coords == "geo"``. If True and cartopy is missing, raise ImportError.
    coords : {"geo", "cartesian"}
        ``"geo"`` plots on ``(longitude, latitude)``. ``"cartesian"`` plots on
        ``(x, y)`` in km from the radar site (no cartopy needed).
    add_range_rings : bool
        For cartesian mode, draw 50/100/150 km range rings.
    add_colorbar : bool
    title : str, optional
    figsize : tuple
    **plot_kwargs
        Forwarded to ``pcolormesh`` (overrides defaults for ``cmap``, ``vmin``,
        ``vmax``, ``norm``, etc.).

    Returns
    -------
    matplotlib.collections.QuadMesh
    """
    ds = _get_sweep_dataset(dt, sweep)
    if variable not in ds.variables:
        raise KeyError(f"Variable '{variable}' not in sweep. Available: {list(ds.data_vars)}")
    da = ds[variable]

    plot_kwargs, is_discrete, class_labels, cbar_label = _resolve_plot_kwargs(
        variable, plot_kwargs
    )

    ccrs, cfeature = _maybe_cartopy()
    if use_cartopy is None:
        use_cartopy = (coords == "geo") and (ccrs is not None)
    elif use_cartopy and ccrs is None:
        raise ImportError(
            "use_cartopy=True but cartopy is not installed. "
            "Run: pip install 'raddb[viz]' or: pip install cartopy"
        )

    # -- Case 1: geographic with cartopy --------------------------------
    if use_cartopy and coords == "geo":
        site_lon = float(ds["site_longitude"])
        site_lat = float(ds["site_latitude"])
        proj = ccrs.AzimuthalEquidistant(
            central_longitude=site_lon, central_latitude=site_lat,
        )
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": proj}, figsize=figsize)
        p = ax.pcolormesh(
            ds["longitude"].values, ds["latitude"].values, da.values,
            transform=ccrs.PlateCarree(), shading="auto", **plot_kwargs,
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
        ax.plot(site_lon, site_lat, "k^", markersize=7,
                transform=ccrs.PlateCarree(), zorder=5)

    # -- Case 2: geographic without cartopy ----------------------------
    elif coords == "geo":
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        p = ax.pcolormesh(
            ds["longitude"].values, ds["latitude"].values, da.values,
            shading="auto", **plot_kwargs,
        )
        ax.set_xlabel("Longitude [°]")
        ax.set_ylabel("Latitude [°]")
        ax.set_aspect("equal")
        ax.plot(float(ds["site_longitude"]), float(ds["site_latitude"]),
                "k^", markersize=7, zorder=5)

    # -- Case 3: cartesian (x, y) in km -------------------------------
    elif coords == "cartesian":
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        p = ax.pcolormesh(
            ds["x"].values / 1000.0, ds["y"].values / 1000.0, da.values,
            shading="auto", **plot_kwargs,
        )
        ax.set_xlabel("East from radar [km]")
        ax.set_ylabel("North from radar [km]")
        ax.set_aspect("equal")
        if add_range_rings:
            _draw_range_rings_xy(ax)
        ax.plot(0, 0, "k^", markersize=7, zorder=5)
    else:
        raise ValueError(f"coords must be 'geo' or 'cartesian', got {coords!r}")

    if add_colorbar:
        _add_colorbar(p, ax, is_discrete, class_labels, cbar_label)

    sweep_num = int(sweep) if isinstance(sweep, (int, np.integer)) else sweep
    tstr = _volume_time_str(ds)
    ax.set_title(
        title or f"{variable} — sweep {sweep_num} — {tstr}".rstrip(" —")
    )
    return p


# ============================================================================
# RHI
# ============================================================================

def plot_rhi(
    dt,
    azimuth: float,
    variable: str,
    az_tol: float = 1.0,
    max_range_km: float | None = None,
    ax=None,
    add_colorbar: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 4),
    **plot_kwargs,
):
    """Range-Height Indicator — vertical cross-section at a given azimuth.

    For every sweep in the DataTree, selects the ray whose azimuth is closest
    to ``azimuth`` (within ``az_tol`` degrees, wrap-around safe) and stacks
    them into a 2-D ``(ground_range, height_ASL)`` pcolormesh. Earth curvature
    is already baked into the per-gate ``z`` coord (computed with the 4/3 Earth
    radius model at LUT generation), so no geometry code is needed here.

    Parameters
    ----------
    dt : xr.DataTree
        Reconstructed DataTree (from ``parquet_to_datatree``).
    azimuth : float
        Target azimuth in degrees (0..360; 0 = North, clockwise).
    variable : str
    az_tol : float
        Maximum allowed angular distance between the requested azimuth and the
        nearest available ray, in degrees.
    max_range_km : float, optional
        Clip the ground-range axis.
    ax, add_colorbar, title, figsize, **plot_kwargs
        As in ``plot_ppi``.

    Returns
    -------
    matplotlib.collections.QuadMesh
    """
    sweep_names = sorted(
        [g.lstrip("/") for g in dt.groups
         if g.lstrip("/").startswith("sweep_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    if not sweep_names:
        raise ValueError("No sweep_* groups found in DataTree.")

    rays = []  # list of (ground_range_1d, z_1d, values_1d, el_deg)
    for name in sweep_names:
        ds = dt[name].to_dataset()
        if variable not in ds.variables:
            continue
        az = ds["azimuth"].values
        diff = np.abs(((az - azimuth + 180.0) % 360.0) - 180.0)
        j = int(np.argmin(diff))
        if diff[j] > az_tol:
            continue
        ray = ds.isel(azimuth=j)
        gr = np.sqrt(ray["x"].values ** 2 + ray["y"].values ** 2)
        try:
            el = float(ray["elevation_angle"])
        except Exception:
            el = np.nan
        rays.append((gr, ray["z"].values, ray[variable].values, el))

    if not rays:
        raise ValueError(
            f"No sweep has a ray within ±{az_tol}° of azimuth {azimuth}° "
            f"for variable '{variable}'."
        )

    # Sort by elevation (ascending) so low sweeps are drawn at the bottom.
    rays.sort(key=lambda r: r[3] if np.isfinite(r[3]) else 0.0)

    # Sweeps have different gate counts — trim to the shortest common axis.
    n_rng = min(len(r[0]) for r in rays)
    gr2d = np.stack([r[0][:n_rng] for r in rays])
    z2d  = np.stack([r[1][:n_rng] for r in rays])
    v2d  = np.stack([r[2][:n_rng] for r in rays])

    plot_kwargs, is_discrete, class_labels, cbar_label = _resolve_plot_kwargs(
        variable, plot_kwargs
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    p = ax.pcolormesh(
        gr2d / 1000.0, z2d / 1000.0, v2d, shading="auto", **plot_kwargs,
    )
    ax.set_xlabel("Ground range [km]")
    ax.set_ylabel("Height ASL [km]")
    if max_range_km is not None:
        ax.set_xlim(0, max_range_km)
    ax.set_ylim(bottom=0)

    if add_colorbar:
        _add_colorbar(p, ax, is_discrete, class_labels, cbar_label)

    tstr = ""
    for name in sweep_names:
        ds = dt[name].to_dataset()
        if variable in ds.variables:
            tstr = _volume_time_str(ds)
            break
    ax.set_title(
        title or f"{variable} — RHI @ azimuth {azimuth:.1f}° — {tstr}".rstrip(" —")
    )
    return p
