"""
raddb/plot.py
-------------
PPI, RHI, and latent-space scatter plots for RadDB.

A radar gate is not a rectangle: it is a curved trapezoid in geographic space
whose footprint depends on range, azimuth, elevation, and Earth curvature.
The right way to render it is to feed matplotlib's ``pcolormesh`` the 2-D
centroid coordinates of every gate — matplotlib then builds the correct
curved quadrilaterals automatically.

The reconstructed DataTree already carries the per-gate ``(lon, lat, alt, x, y, z)``
coords on every sweep Dataset (attached during ``parquet_to_datatree``), so
these plot functions do not need any geometry computation of their own.

If cartopy is available the PPI is drawn on an Azimuthal Equidistant map
centred on the radar site (coastlines, borders, gridlines). This applies to
both ``coords="geo"`` and ``coords="cartesian"`` — in the latter case the
axes tick labels are formatted in km from the radar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
import xarray as xr

from raddb.hc_mapping import HC_CLASSES as _HC_CLASSES, HC_COLORS as _HC_COLORS


# ============================================================================
# Per-variable plotting defaults
# ============================================================================

_PLOT_DEFAULTS: dict[str, dict] = {
    "DBZH":     dict(cmap="turbo",    vmin=-10,  vmax=60,   label="Reflectivity [dBz]"),
    "DBZH_raw": dict(cmap="turbo",    vmin=-10,  vmax=60,   label="Raw reflectivity [dBz]"),
    "ZDR":      dict(cmap="RdBu_r",   vmin=-2,   vmax=5,    label="Differential reflectivity [dB]"),
    "ZDR_raw":  dict(cmap="RdBu_r",   vmin=-2,   vmax=5,    label="Raw differential reflectivity [dB]"),
    "KDP":      dict(cmap="RdBu_r",   vmin=-1,   vmax=3,    label="Specific differential phase [°/km]"),
    "RHOHV":    dict(cmap="viridis",  vmin=0.5,  vmax=1.0,  label="Co-polar correlation [-]"),
    "PHIDP":    dict(cmap="twilight", vmin=-180, vmax=180,  label="Differential phase [deg]"),
    "HZT":      dict(cmap="viridis",  vmin=0,    vmax=5000, label="Freezing level height [m]"),
    "TEMP":     dict(cmap="RdBu_r",   vmin=-30,  vmax=15,   label="Temperature [°C]"),
    "HC_MCH":   dict(discrete=True,   classes=_HC_CLASSES,  colors=_HC_COLORS, label="MCH hydrometeor class"),
    "HC_PYART": dict(discrete=True,   classes=_HC_CLASSES,  colors=_HC_COLORS, label="PyART hydrometeor class"),
}


# ============================================================================
# Internal helpers
# ============================================================================

def _resolve_plot_kwargs(variable: str, user_kwargs: dict):
    """Merge per-variable defaults with user overrides.

    Returns (plot_kwargs, is_discrete, class_labels, cbar_label). Always sets
    ``cmap.set_bad("none")`` so NaN (filtered) gates render transparent
    instead of picking up a colormap endpoint.
    """
    defaults = _PLOT_DEFAULTS.get(variable, {})
    is_discrete = bool(defaults.get("discrete", False))
    class_labels = defaults.get("classes")
    cbar_label = defaults.get("label", variable)

    plot_kwargs = dict(user_kwargs)
    if is_discrete:
        n = len(class_labels)
        class_colors = defaults.get("colors")
        if class_colors is not None:
            cmap = ListedColormap(class_colors[:n])
        else:
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

    # Make NaN gates transparent (not bottom-of-colormap colored).
    cmap_val = plot_kwargs.get("cmap")
    if cmap_val is not None:
        cmap_obj = plt.get_cmap(cmap_val).copy() if isinstance(cmap_val, str) else cmap_val.copy()
        cmap_obj.set_bad("none")
        plot_kwargs["cmap"] = cmap_obj

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
    figsize: tuple[float, float] = (6, 6),
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
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
        If None (default), auto-detect: use cartopy when installed. Pass
        ``False`` to force plain matplotlib. Pass ``True`` to require cartopy
        (raises ImportError if missing).
    coords : {"geo", "cartesian"}
        ``"geo"`` plots on ``(longitude, latitude)``. ``"cartesian"`` plots on
        ``(x, y)`` km from the radar site. When cartopy is available, both
        modes add a context map (coastlines, borders). In ``"cartesian"`` mode
        the axes tick labels are formatted in km.
    add_range_rings : bool
        For cartesian mode, draw 50/100/150 km range rings.
    add_colorbar : bool
    title : str, optional
    figsize : tuple
        Figure size in inches. Default ``(6, 6)``.
    xlim : tuple, optional
        x-axis limits in the natural units of the chosen ``coords``
        (degrees for ``"geo"``, km for ``"cartesian"``).
    ylim : tuple, optional
        y-axis limits (same units as ``xlim``).
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
        use_cartopy = ccrs is not None  # auto-use cartopy for both geo and cartesian
    elif use_cartopy and ccrs is None:
        raise ImportError(
            "use_cartopy=True but cartopy is not installed. "
            "Run: pip install 'raddb[viz]' or: pip install cartopy"
        )

    # If the user passed a non-GeoAxes axis, silently disable cartopy.
    if use_cartopy and ax is not None:
        try:
            from cartopy.mpl.geoaxes import GeoAxes
            if not isinstance(ax, GeoAxes):
                use_cartopy = False
        except ImportError:
            use_cartopy = False

    has_edges = all(k in ds.variables for k in ("x_edges", "y_edges"))

    # ------------------------------------------------------------------ geo
    if coords == "geo":
        if use_cartopy:
            site_lon = float(ds["site_longitude"])
            site_lat = float(ds["site_latitude"])
            proj = ccrs.AzimuthalEquidistant(
                central_longitude=site_lon, central_latitude=site_lat,
            )
            if ax is None:
                fig, ax = plt.subplots(subplot_kw={"projection": proj}, figsize=figsize)
            if has_edges:
                p = ax.pcolormesh(
                    ds["x_edges"].values, ds["y_edges"].values, da.values,
                    transform=proj, shading="flat", **plot_kwargs,
                )
            else:
                p = ax.pcolormesh(
                    ds["longitude"].values, ds["latitude"].values, da.values,
                    transform=ccrs.PlateCarree(), shading="auto", **plot_kwargs,
                )
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
            ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.3)
            ax.plot(site_lon, site_lat, "kx", markersize=7, markeredgewidth=2,
                    transform=ccrs.PlateCarree(), zorder=5)
            if xlim is not None or ylim is not None:
                lon_min, lon_max = xlim if xlim is not None else (None, None)
                lat_min, lat_max = ylim if ylim is not None else (None, None)
                cur = ax.get_extent(crs=ccrs.PlateCarree())
                ax.set_extent([
                    lon_min if lon_min is not None else cur[0],
                    lon_max if lon_max is not None else cur[1],
                    lat_min if lat_min is not None else cur[2],
                    lat_max if lat_max is not None else cur[3],
                ], crs=ccrs.PlateCarree())
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            if has_edges:
                p = ax.pcolormesh(
                    ds["lon_edges"].values, ds["lat_edges"].values, da.values,
                    shading="flat", **plot_kwargs,
                )
            else:
                p = ax.pcolormesh(
                    ds["longitude"].values, ds["latitude"].values, da.values,
                    shading="auto", **plot_kwargs,
                )
            ax.set_xlabel("Longitude [°]")
            ax.set_ylabel("Latitude [°]")
            ax.set_aspect("equal")
            ax.plot(float(ds["site_longitude"]), float(ds["site_latitude"]),
                    "kx", markersize=7, markeredgewidth=2, zorder=5)
            ax.grid(True, alpha=0.3)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

    # ------------------------------------------------------------ cartesian
    elif coords == "cartesian":
        if use_cartopy and ("site_longitude" in ds.variables or "site_longitude" in ds.coords):
            site_lon = float(ds["site_longitude"])
            site_lat = float(ds["site_latitude"])
            proj = ccrs.AzimuthalEquidistant(
                central_longitude=site_lon, central_latitude=site_lat,
            )
            if ax is None:
                fig, ax = plt.subplots(subplot_kw={"projection": proj}, figsize=figsize)
            # x/y from dataset are in metres; plot directly in AEQD native units.
            if has_edges:
                p = ax.pcolormesh(
                    ds["x_edges"].values, ds["y_edges"].values, da.values,
                    transform=proj, shading="flat", **plot_kwargs,
                )
            else:
                p = ax.pcolormesh(
                    ds["x"].values, ds["y"].values, da.values,
                    transform=proj, shading="auto", **plot_kwargs,
                )
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
            ax.gridlines(linewidth=0.3, alpha=0.3, draw_labels=False)
            ax.plot(0, 0, "kx", markersize=7, markeredgewidth=2, transform=proj, zorder=5)
            if add_range_rings:
                theta = np.linspace(0, 2 * np.pi, 361)
                for d_km in (50, 100, 150):
                    ax.plot(
                        d_km * 1e3 * np.cos(theta), d_km * 1e3 * np.sin(theta),
                        "k--", linewidth=0.5, transform=proj,
                    )
            # Format ticks in km (AEQD native units are metres).
            ax.xaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}")
            )
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v / 1e3:.0f}")
            )
            ax.set_xlabel("East from radar [km]")
            ax.set_ylabel("North from radar [km]")
            if xlim is not None:
                ax.set_xlim(xlim[0] * 1e3, xlim[1] * 1e3)
            if ylim is not None:
                ax.set_ylim(ylim[0] * 1e3, ylim[1] * 1e3)
        else:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            if has_edges:
                p = ax.pcolormesh(
                    ds["x_edges"].values / 1000.0, ds["y_edges"].values / 1000.0,
                    da.values, shading="flat", **plot_kwargs,
                )
            else:
                p = ax.pcolormesh(
                    ds["x"].values / 1000.0, ds["y"].values / 1000.0, da.values,
                    shading="auto", **plot_kwargs,
                )
            ax.set_xlabel("East from radar [km]")
            ax.set_ylabel("North from radar [km]")
            ax.set_aspect("equal")
            if add_range_rings:
                _draw_range_rings_xy(ax)
            ax.plot(0, 0, "kx", markersize=7, markeredgewidth=2, zorder=5)
            ax.grid(True, alpha=0.3)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
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
# RHI (pseudo-RHI from PPI volume)
# ============================================================================

def plot_rhi(
    dt,
    azimuth: float,
    variable: str,
    radar: str = "",
    az_tol: float = 1.0,
    max_range_km: float | None = None,
    max_height_km: float | None = 20.0,
    ke: float = 4.0 / 3.0,
    ax=None,
    add_colorbar: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 4),
    **plot_kwargs,
):
    """Pseudo-RHI from a volume PPI scan — PyART's ``cross_section_ppi``.

    For every sweep in the DataTree, selects the ray whose azimuth is closest
    to ``azimuth`` (within ``az_tol``, wrap-around safe), regrids all rays to
    a common range axis, and renders a 2-D
    ``(ground_range, height_ASL)`` pcolormesh using **gate edges** computed
    with the 4/3 Earth-radius model — so curved-trapezoid gates and Earth
    curvature are both physically correct.

    Parameters
    ----------
    dt : xr.DataTree
        Reconstructed DataTree (from ``parquet_to_datatree``).
    azimuth : float
        Target azimuth in degrees (0..360; 0 = North, clockwise).
    variable : str
    radar : str, optional
        Radar identifier shown in the plot title (e.g. ``"A"``).
    az_tol : float
        Maximum allowed angular distance between the requested azimuth and
        the nearest available ray.
    max_range_km : float, optional
        Clip the ground-range axis.
    max_height_km : float, optional
        Clip the height axis.
    ke : float
        Effective Earth radius factor (default 4/3, pyart standard).
    ax, add_colorbar, title, figsize, **plot_kwargs : see ``plot_ppi``.

    Returns
    -------
    matplotlib.collections.QuadMesh
    """
    from raddb.lut import (
        antenna_vectors_to_cartesian,
        _interpolate_range_edges,
        _interpolate_elevation_edges,
    )

    sweep_names = sorted(
        [g.lstrip("/") for g in dt.groups
         if g.lstrip("/").startswith("sweep_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    if not sweep_names:
        raise ValueError("No sweep_* groups found in DataTree.")

    # 1. Pick the nearest ray per sweep.
    rays = []
    site_alt = 0.0
    for name in sweep_names:
        ds = dt[name].to_dataset()
        if variable not in ds.variables:
            continue
        az_arr = ds["azimuth"].values
        diff = np.abs(((az_arr - azimuth + 180.0) % 360.0) - 180.0)
        j = int(np.argmin(diff))
        if diff[j] > az_tol:
            continue
        site_alt = float(ds["site_altitude"])
        try:
            el = float(ds["elevation_angle"])
        except Exception:
            el = np.nan
        rays.append({
            "range":     np.asarray(ds["range"].values,    dtype=np.float64),
            "values":    np.asarray(ds[variable].isel(azimuth=j).values, dtype=np.float64),
            "elevation": el,
            "actual_az": float(az_arr[j]),
        })

    if not rays:
        raise ValueError(
            f"No sweep has a ray within ±{az_tol}° of azimuth {azimuth}° "
            f"for variable '{variable}'."
        )

    # 2. Sort by elevation (low sweeps at the bottom of the RHI).
    rays.sort(key=lambda r: r["elevation"] if np.isfinite(r["elevation"]) else 0.0)

    # 3. Build a common range axis (widest sweep).
    widest = max(rays, key=lambda r: r["range"].max() if r["range"].size else 0.0)
    common_range = widest["range"]

    v2d = np.stack([
        np.interp(common_range, r["range"], r["values"],
                  left=np.nan, right=np.nan)
        for r in rays
    ])  # (n_sweeps, n_range)

    # 4. Gate-edge arrays via 4/3-Earth model.
    range_edges = _interpolate_range_edges(common_range)
    elevations  = np.array([r["elevation"] for r in rays], dtype=np.float64)
    el_edges    = _interpolate_elevation_edges(elevations)
    mean_az     = float(np.mean([r["actual_az"] for r in rays]))
    az_edges    = np.full(el_edges.size, mean_az)

    x_e, y_e, z_e = antenna_vectors_to_cartesian(
        ranges=range_edges, azimuths=az_edges, elevations=el_edges, ke=ke,
    )
    ground_range_edges = np.sqrt(x_e ** 2 + y_e ** 2)
    height_edges_asl   = z_e + site_alt

    # 5. Render.
    plot_kwargs, is_discrete, class_labels, cbar_label = _resolve_plot_kwargs(
        variable, plot_kwargs
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    p = ax.pcolormesh(
        ground_range_edges / 1000.0,
        height_edges_asl   / 1000.0,
        v2d,
        shading="flat", **plot_kwargs,
    )
    ax.set_xlabel("Ground range [km]")
    ax.set_ylabel("Height ASL [km]")
    if max_range_km is not None:
        ax.set_xlim(0, max_range_km)
    if max_height_km is not None:
        ax.set_ylim(site_alt / 1000.0, max_height_km)
    else:
        ax.set_ylim(bottom=site_alt / 1000.0)
    ax.grid(True, alpha=0.3)

    if add_colorbar:
        _add_colorbar(p, ax, is_discrete, class_labels, cbar_label)

    tstr = ""
    for name in sweep_names:
        ds = dt[name].to_dataset()
        if variable in ds.variables:
            tstr = _volume_time_str(ds)
            break

    if title is None:
        parts = []
        if radar:
            parts.append(f"radar: {radar}")
        parts.append(f"feature: {variable}")
        parts.append(f"azimuth: {azimuth:.1f}°")
        if tstr:
            parts.append(f"date: {tstr}")
        title = " | ".join(parts)
    ax.set_title(title)
    return p


# ============================================================================
# LATENT SPACE SCATTER (AMT publication figure)
# ============================================================================

def plot_latent_scatter(
    df: "pd.DataFrame",
    config: list[dict],
    figsize: tuple[float, float] | None = None,
    fig_height: float = 4.6,
    **scatter_kwargs,
):
    """Publication-ready 2×3 AMT latent-space scatter figure.

    Creates a 2-row × 3-column figure with width=6.9 inches (AMT full-column
    width). Each subplot shows a scatter of ``df["L1"]`` vs ``df["L2"]``
    coloured by one radar variable. A compact inset colorbar with a white
    semi-transparent background is placed inside each subplot.

    x-tick labels and x-axis labels are shown only on the bottom row.
    y-tick labels and y-axis labels are shown only on the first column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ``"L1"``, ``"L2"``, and the variable column
        named in each panel's ``"var"`` key.
    config : list of dict
        Exactly 6 panel descriptors, one per subplot (row-major: panels
        0-2 fill row 0, panels 3-5 fill row 1). Each dict must have:

        - ``"var"`` : str — column in ``df`` to use for colouring.
        - ``"label"`` : str — colorbar label text.
        - ``"cmap"`` : str or Colormap — colormap passed to ``scatter``.
        - ``"norm"`` : matplotlib Normalize, optional — normalisation.
        - ``"cbar_kwargs"`` : dict — extra kwargs forwarded to
          ``fig.colorbar()`` (e.g. ``{"ticks": [...]}``, ``{"extend": "both"}``).
        - ``"scatter_kwargs"`` : dict, optional — extra kwargs for
          ``ax.scatter()`` (e.g. ``{"s": 1, "alpha": 0.5}``).
    figsize : tuple, optional
        Override figure size. Default ``(6.9, fig_height)``.
    fig_height : float
        Figure height in inches. Default ``4.6``.
    **scatter_kwargs
        Global fallback kwargs for ``ax.scatter()`` (overridden per-panel
        by ``config[i]["scatter_kwargs"]``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of shape (2, 3)

    Example config
    --------------
    config = [
        {"var": "DBZH",    "label": "DBZH [dBZ]",   "cmap": "turbo",
         "norm": Normalize(-10, 60), "cbar_kwargs": {}, "scatter_kwargs": {"s": 0.5}},
        {"var": "ZDR",     "label": "ZDR [dB]",      "cmap": "RdBu_r",
         "norm": Normalize(-2, 5),  "cbar_kwargs": {}, "scatter_kwargs": {"s": 0.5}},
        ...
    ]
    """
    if len(config) != 6:
        raise ValueError(f"config must have exactly 6 entries, got {len(config)}.")

    n_rows, n_cols = 2, 3
    fw = figsize[0] if figsize is not None else 6.9
    fh = figsize[1] if figsize is not None else fig_height
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fw, fh),
        gridspec_kw={"hspace": 0, "wspace": 0},
    )

    # Colorbar inset geometry (in axes-fraction coordinates).
    cbar_inset_axes = [0.04, 0.87, 0.50, 0.05]
    cbar_fontsize = 7
    x_pad = 0.05
    y_pad = 0.10

    for idx, panel in enumerate(config):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        var = panel["var"]
        cmap = panel.get("cmap", "viridis")
        norm = panel.get("norm", None)
        panel_scatter_kw = {**scatter_kwargs, **panel.get("scatter_kwargs", {})}

        m = ax.scatter(
            df["L1"], df["L2"],
            c=df[var],
            cmap=cmap,
            norm=norm,
            **panel_scatter_kw,
        )

        # ---- inset colorbar with white fancy-box background ----
        cax = ax.inset_axes(cbar_inset_axes)
        fancybox_zorder = cax.get_zorder() + 1
        cax.set_zorder(cax.get_zorder() + 2)

        cb = fig.colorbar(
            m,
            cax=cax,
            orientation="horizontal",
            **panel.get("cbar_kwargs", {}),
        )
        cb.set_label(panel["label"], fontsize=cbar_fontsize, labelpad=3.1)
        cb.ax.xaxis.set_label_position("top")
        cb.ax.tick_params(labelsize=6, pad=1, length=2)
        cb.outline.set_linewidth(0.5)

        fancy_box_coords = (cbar_inset_axes[0] - x_pad, cbar_inset_axes[1] - y_pad)
        fancy_box_width  = cbar_inset_axes[2] + 2 * x_pad
        fancy_box_height = cbar_inset_axes[3] + 2 * y_pad
        fancy_patch = mpatches.FancyBboxPatch(
            fancy_box_coords,
            width=fancy_box_width,
            height=fancy_box_height,
            boxstyle="square,pad=0",
            fc="white",
            ec="none",
            lw=0.5,
            alpha=0.6,
            transform=ax.transAxes,
            zorder=fancybox_zorder,
            clip_on=False,
        )
        ax.add_artist(fancy_patch)
        for spine in ax.spines.values():
            spine.set_zorder(fancybox_zorder + 2)

        # ---- tick / label visibility ----
        if row < n_rows - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("$L_1$")

        if col > 0:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("$L_2$")

    return fig, axes
