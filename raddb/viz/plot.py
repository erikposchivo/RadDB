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
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm
import xarray as xr

from raddb.hc_mapping import HC_CLASSES as _HC_CLASSES, HC_COLORS as _HC_COLORS


# ============================================================================
# Per-variable plotting defaults
# ============================================================================

def _first_available_cmap(*names: str) -> str:
    """First registered colormap among ``names`` (last is the guaranteed fallback).

    Lets ``DBZH`` prefer Py-ART's ``HomeyerRainbow`` when pyart has been imported
    (registering it), falling back to ``turbo`` otherwise.
    """
    available = set(plt.colormaps())
    for name in names:
        if name in available:
            return name
    return names[-1]


# Colormaps chosen to match raddb/viz/report_raddb_figures.py (raddb_ppi_ex.png),
# with two deliberate departures: KDP uses a non-cyclic map (the report's twilight
# wraps around and is misleading for a signed quantity), and TEMP is a 0-centred
# diverging (coolwarm's midpoint is grey) via TwoSlopeNorm so 0 °C reads grey.

_PLOT_DEFAULTS: dict[str, dict] = {
    "DBZH":     dict(cmap="HomeyerRainbow",   vmin=0,    vmax=60,   label="Reflectivity [dBz]"),
    "DBZH_raw": dict(cmap="HomeyerRainbow",   vmin=0,    vmax=60,   label="Raw reflectivity [dBz]"),
    "ZDR":      dict(cmap="viridis",  vmin=-2,   vmax=7,    label="Differential reflectivity [dB]"),
    "ZDR_raw":  dict(cmap="viridis",  vmin=-2,   vmax=7,    label="Raw differential reflectivity [dB]"),
    "KDP":      dict(cmap="plasma",   vmin=-2,   vmax=5,    label="Specific differential phase [°/km]"),
    "RHOHV":    dict(cmap="cividis",  vmin=0.5,  vmax=1.0,  label="Co-polar correlation [-]"),
    "PHIDP":    dict(cmap="twilight", vmin=-180, vmax=180,  label="Differential phase [deg]"),
    "HZT":      dict(cmap="viridis",  vmin=0,    vmax=5000, label="Freezing level height [m]"),
    "TEMP":     dict(cmap="coolwarm",
                     norm=lambda: TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30),
                     label="Temperature [°C]"),
    "HC_MCH":   dict(discrete=True,   classes=_HC_CLASSES,  colors=_HC_COLORS, label="MCH hydrometeor class"),
    "HC_PYART": dict(discrete=True,   classes=_HC_CLASSES,  colors=_HC_COLORS, label="PyART hydrometeor class"),
}


# ============================================================================
# Internal helpers
# ============================================================================

_PYART_CMAPS_TRIED = False


def _ensure_cmap_registered(name):
    """Return a usable colormap name; register Py-ART's colormaps on demand.

    Non-string values (Colormap instances) pass through.  A string already known
    to matplotlib is returned as-is.  Otherwise Py-ART is imported **once** — that
    registers its colormaps (e.g. ``HomeyerRainbow``) with matplotlib — and the
    name is re-checked.  If it's still missing (pyart absent or unknown name),
    fall back to ``turbo`` with a warning so plotting never hard-crashes.
    """
    global _PYART_CMAPS_TRIED
    if not isinstance(name, str) or name in plt.colormaps():
        return name
    if not _PYART_CMAPS_TRIED:
        _PYART_CMAPS_TRIED = True
        try:
            import pyart  # noqa: F401  # registers Py-ART colormaps with matplotlib
        except Exception:  # noqa: BLE001 - pyart optional
            pass
    if name in plt.colormaps():
        return name
    import warnings
    warnings.warn(
        f"colormap {name!r} is unavailable (Py-ART colormaps need pyart installed); "
        "falling back to 'turbo'.",
        stacklevel=2,
    )
    return "turbo"


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
        if "cmap" in defaults:
            plot_kwargs.setdefault("cmap", defaults["cmap"])
        if "norm" in defaults:
            # a default norm (e.g. 0-centred diverging); build a fresh instance so
            # it isn't shared/mutated across figures. Skipped if the caller passed
            # any explicit scale (norm / vmin / vmax) to avoid a matplotlib clash.
            if not any(k in plot_kwargs for k in ("norm", "vmin", "vmax")):
                nrm = defaults["norm"]
                plot_kwargs["norm"] = nrm() if callable(nrm) else nrm
        else:
            for k in ("vmin", "vmax"):
                if k in defaults:
                    plot_kwargs.setdefault(k, defaults[k])

    # Make NaN gates transparent (not bottom-of-colormap colored).
    cmap_val = plot_kwargs.get("cmap")
    if cmap_val is not None:
        cmap_val = _ensure_cmap_registered(cmap_val)  # register pyart cmaps if needed
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


_LV95_BORDERS = "unset"  # cache: list of country-border lines in EPSG:2056


def _lv95_border_lines():
    """Country-border lines near Switzerland, reprojected to EPSG:2056 (cached).

    Sourced from cartopy's Natural Earth 10 m admin-0 boundary lines, clipped to
    a lon/lat box around Switzerland before reprojection (LV95 is only valid near
    CH).  Returns a list of shapely (Multi)LineStrings in LV95, or ``[]`` if
    cartopy / the data is unavailable — so the swiss PPI still draws, just without
    borders.  Drawn on a plain matplotlib axis, so no GeoAxes / extent quirks.
    """
    global _LV95_BORDERS
    if _LV95_BORDERS != "unset":
        return _LV95_BORDERS
    lines = []
    try:
        import shapely
        from cartopy.io import shapereader as shpreader
        from raddb.aoi import _reproject_to_2056

        path = shpreader.natural_earth(
            resolution="10m", category="cultural", name="admin_0_boundary_lines_land"
        )
        clip = shapely.box(3.0, 43.0, 13.5, 49.5)  # Switzerland + neighbours
        for geom in shpreader.Reader(path).geometries():
            piece = geom.intersection(clip)
            if not piece.is_empty:
                lines.append(_reproject_to_2056(piece, 4326))
    except Exception as exc:  # noqa: BLE001 - cartopy missing / data not cached
        import warnings
        warnings.warn(
            f"cartopy country borders unavailable ({exc}); swiss PPI drawn without them.",
            stacklevel=2,
        )
    _LV95_BORDERS = lines
    return _LV95_BORDERS


def _draw_lv95_borders(ax):
    """Plot the cached LV95 country borders as light lines (clipped to the axes)."""
    def _iter_lines(g):
        if g.geom_type == "LineString":
            yield g
        elif g.geom_type in ("MultiLineString", "GeometryCollection"):
            for sub in g.geoms:
                yield from _iter_lines(sub)

    for geom in _lv95_border_lines():
        for line in _iter_lines(geom):
            ax.plot(*line.xy, color="0.4", linewidth=0.6, zorder=1)


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

def plot_aoi_quicklook(
    aoi_geom,
    selected=None,
    radars=None,
    base_path=None,
    context="switzerland",
    ax=None,
    figsize=(9, 9),
    title=None,
    range_rings_km=(100,),
    show_gates=False,
    gate_sample=50_000,
    xlim=None,
    ylim=(1_040_000, 1_310_000),
    save_path=None,
):
    """Map an AOI on a country-scale background for a quick sanity-check.

    Answers "**is my AOI where I think it is?**": the AOI footprint (red) drawn on
    the **Switzerland outline** with the involved radar sites, in a **square**
    national-extent view.  Rendered in Swiss LV95 (EPSG:2056), axis units km.  The
    Swiss outline comes from cartopy's cached Natural Earth data; everything else
    is dependency-free, so the map still draws (without the outline) if that data
    is missing.

    Parameters
    ----------
    aoi_geom : shapely geometry
        AOI footprint in EPSG:2056 (Polygon for bbox/polygon/point AOIs;
        LineString for a cross-section line).
    selected : pd.DataFrame, optional
        Selected gates carrying ``x_2056`` / ``y_2056``.  Only scattered when
        ``show_gates=True`` — off by default so the map stays readable.
    radars : list of str, optional
        Radar letters to mark (needs ``base_path`` to load their site coords).
    base_path : str or Path, optional
        RadDB archive base directory, for loading radar site coordinates.
    context : str, shapely geometry, GeoDataFrame, or None
        Map background. ``"switzerland"`` (default) draws the national outline;
        ``None`` draws none; a geometry/GeoDataFrame draws a custom context.
    ax : matplotlib Axes, optional
        Draw into an existing axis instead of creating a figure.
    figsize : tuple
        Default ``(9, 9)`` — square, sized to show the whole country.
    title : str, optional
    range_rings_km : number or iterable of number, optional
        Range-ring radii (km) drawn dashed around each radar site.  Accepts a
        single value (``100``) or several (``(50, 100)``); ``None`` omits them.
    show_gates : bool
        Scatter the selected gate centroids (default False).
    gate_sample : int
        Cap the number of scattered centroids (random subsample); ``None`` = all.
    xlim, ylim : (min, max) in EPSG:2056 metres, optional
        Axis limits.  ``xlim`` defaults to auto (fills the context/AOI extent);
        ``ylim`` defaults to the Swiss north band (1.04–1.31 Mm ≈ North 40–310 km).
        Pass ``None`` to either for auto-framing of that axis.
    save_path : str or Path, optional
        If given, save the figure (dpi=150, tight).

    Returns
    -------
    (fig, ax)
    """
    import shapely
    from raddb.aoi import _resolve_context

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # --- context background (Switzerland outline) ---
    ctx_geom = _resolve_context(context)
    if ctx_geom is not None:
        _draw_context(ax, ctx_geom)

    # --- radar site positions (also used to frame the view) ---
    sites: dict[str, tuple[float, float]] = {}
    if radars and base_path is not None:
        from raddb.lut import load_radar_info
        from raddb.aoi import _reproject_to_2056

        for r in radars:
            try:
                info = load_radar_info(r, base_path)
            except Exception:  # noqa: BLE001 - missing info shouldn't kill the quicklook
                continue
            pt = _reproject_to_2056(shapely.Point(info["longitude"], info["latitude"]), 4326)
            sites[r] = (pt.x, pt.y)

    # --- optional selected gate centroids ---
    if (
        selected is not None
        and show_gates
        and len(selected)
        and {"x_2056", "y_2056"}.issubset(selected.columns)
    ):
        xs = selected["x_2056"].to_numpy()
        ys = selected["y_2056"].to_numpy()
        if gate_sample and len(xs) > gate_sample:
            idx = np.random.default_rng(0).choice(len(xs), gate_sample, replace=False)
            xs, ys = xs[idx], ys[idx]
        ax.scatter(
            xs, ys, s=2, c="tab:blue", alpha=0.25, linewidths=0,
            label=f"selected gates (n={len(selected):,})", zorder=2,
        )

    # --- radar sites (+ dashed range rings) ---
    theta = np.linspace(0, 2 * np.pi, 361)
    # accept None, a single number (e.g. 100), or an iterable of radii
    if range_rings_km is None:
        rings = ()
    elif isinstance(range_rings_km, (int, float)):
        rings = (range_rings_km,)
    else:
        rings = tuple(range_rings_km)
    ring_label = (
        f"range rings ({', '.join(str(int(d)) for d in rings)} km)" if rings else None
    )
    first_site = True
    ring_labeled = False
    for r, (sx, sy) in sites.items():
        for d_km in rings:
            ax.plot(
                sx + d_km * 1e3 * np.cos(theta), sy + d_km * 1e3 * np.sin(theta),
                color="0.5", lw=0.7, ls="--", zorder=1,
                label=None if ring_labeled else ring_label,
            )
            ring_labeled = True
        ax.plot(sx, sy, "k^", ms=9, zorder=5, label="radar" if first_site else None)
        ax.annotate(
            r, (sx, sy), textcoords="offset points", xytext=(5, 5),
            fontweight="bold", zorder=6,
        )
        first_site = False

    # --- AOI footprint ---
    _draw_aoi_outline(ax, aoi_geom)

    # --- frame: x fills the context/AOI extent; y defaults to the Swiss band ---
    boxes = [aoi_geom.bounds]
    if ctx_geom is not None:
        boxes.append(ctx_geom.bounds)
    boxes += [(x, y, x, y) for x, y in sites.values()]
    xmin = min(b[0] for b in boxes)
    ymin = min(b[1] for b in boxes)
    xmax = max(b[2] for b in boxes)
    ymax = max(b[3] for b in boxes)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        mx = 0.06 * max(xmax - xmin, 1.0)
        ax.set_xlim(xmin - mx, xmax + mx)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        my = 0.06 * max(ymax - ymin, 1.0)
        ax.set_ylim(ymin - my, ymax + my)

    # --- cosmetics: equal aspect, LV95 km ticks ---
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{(v - 2e6) / 1e3:.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{(v - 1e6) / 1e3:.0f}"))
    ax.set_xlabel("East [km]")
    ax.set_ylabel("North [km]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title(title or "Quicklook - Area Of Interest")

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, ax


def _draw_context(ax, geom, label="Switzerland"):
    """Draw a light-gray filled context outline (country/region) behind the AOI."""
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    first = True
    for g in polys:
        if g.geom_type != "Polygon":
            continue
        ax.fill(*g.exterior.xy, fc="0.93", ec="0.55", lw=0.8, zorder=0,
                label=label if first else None)
        first = False


def _draw_aoi_outline(ax, geom, color="red"):
    """Draw an AOI geometry outline (Polygon fill+edge, or LineString path)."""
    gt = geom.geom_type
    if gt == "Polygon":
        ax.fill(*geom.exterior.xy, color=color, alpha=0.25, zorder=3)
        ax.plot(*geom.exterior.xy, color=color, lw=2, zorder=4, label="AOI")
    elif gt in ("LineString", "LinearRing"):
        ax.plot(*geom.xy, color=color, lw=2, zorder=4, label="AOI")
    elif gt in ("MultiPolygon", "GeometryCollection"):
        first = True
        for g in geom.geoms:
            lbl = "AOI" if first else None
            if g.geom_type == "Polygon":
                ax.fill(*g.exterior.xy, color=color, alpha=0.25, zorder=3)
                ax.plot(*g.exterior.xy, color=color, lw=2, zorder=4, label=lbl)
            first = False
    else:  # Point or other
        ax.plot(geom.x, geom.y, marker="*", color=color, ms=12, zorder=4, label="AOI")


# ============================================================================
# Regular-grid slice maps (from RadDB.aoi_to_grid)
# ============================================================================

def plot_grid(
    ds,
    variable: str = "DBZH",
    radar=None,
    z: float | None = None,
    iz: int | None = None,
    time=None,
    ax=None,
    figsize=(8, 7),
    title: str | None = None,
    add_colorbar: bool = True,
    use_cartopy: bool | None = None,
    xlim=None,
    ylim=None,
    **plot_kwargs,
):
    """Map one horizontal slice of an :meth:`RadDB.aoi_to_grid` Dataset.

    Selects one radar (or a **pair for a difference map**), one altitude layer,
    and (if present) one volume time, then draws the (y, x) slice in the Swiss
    LV95 frame (East/North km ticks, optional cartopy country borders) — the
    same frame as the AOI quicklook and ``coords="swiss"`` PPIs.

    Parameters
    ----------
    ds : xr.Dataset
        Output of :meth:`RadDB.aoi_to_grid` (dims ``(time?, radar, z, y, x)``).
    variable : str
        Data variable to draw (default ``"DBZH"``).
    radar : str or (str, str), optional
        A radar letter selects that radar's layer (inferred when the grid holds
        exactly one).  A **pair** ``("P", "L")`` draws the per-voxel difference
        ``P − L`` with a symmetric diverging colormap — the cross-radar
        comparison the per-radar grid exists for.
    z : float, optional
        Altitude of the layer to draw (m ASL); the nearest ``z`` bin is used.
    iz : int, optional
        Alternative to ``z``: direct index into the ``z`` dimension.
    time : str or datetime, optional
        Volume time (nearest match); required only when the grid has several.
    ax : matplotlib Axes, optional
    figsize, title, add_colorbar
        Usual matplotlib options.
    use_cartopy : bool, optional
        Draw LV95-reprojected country borders (auto when cartopy is available).
    xlim, ylim : (min, max) in LV95 metres, optional
    **plot_kwargs
        ``cmap`` / ``vmin`` / ``vmax`` / ``norm`` overrides.

    Returns
    -------
    (fig, ax, mesh)
    """
    if variable not in ds.data_vars:
        raise KeyError(f"variable {variable!r} not in grid; available: {list(ds.data_vars)}")

    sel = ds[variable]

    # --- time selection (grid times are naive UTC) ---
    if "time" in sel.dims:
        if time is not None:
            ts = pd.to_datetime(time)
            ts = ts.tz_convert("UTC").tz_localize(None) if ts.tzinfo is not None else ts
            sel = sel.sel(time=ts.to_datetime64(), method="nearest")
        elif sel.sizes["time"] == 1:
            sel = sel.isel(time=0)
        else:
            raise ValueError(f"grid holds {sel.sizes['time']} volumes; pass time= to pick one.")

    # --- altitude layer ---
    if iz is not None:
        sel = sel.isel(z=int(iz))
    elif z is not None:
        sel = sel.sel(z=float(z), method="nearest")
    elif sel.sizes.get("z", 1) == 1:
        sel = sel.isel(z=0)
    else:
        raise ValueError("pass z= (altitude, m ASL) or iz= to pick the layer to draw.")
    z_val = float(sel["z"])

    # --- radar selection: single layer or difference of a pair ---
    radars_in = [str(r) for r in np.atleast_1d(ds["radar"].values)]
    is_diff = isinstance(radar, (tuple, list)) and len(radar) == 2
    if is_diff:
        r0, r1 = radar
        data2d = sel.sel(radar=r0) - sel.sel(radar=r1)
        default_title = f"{variable} difference {r0}−{r1} — z≈{z_val:.0f} m ASL"
        plot_kwargs.setdefault("cmap", "RdBu_r")
        if not any(k in plot_kwargs for k in ("vmin", "vmax", "norm")):
            vmax = float(np.nanmax(np.abs(data2d.values))) if np.isfinite(data2d.values).any() else 1.0
            plot_kwargs["vmin"], plot_kwargs["vmax"] = -vmax, vmax
        resolved, is_discrete, class_labels, cbar_label = _resolve_plot_kwargs("__diff__", plot_kwargs)
        cbar_label = f"Δ{variable} ({r0}−{r1})"
    else:
        if radar is None:
            if len(radars_in) != 1:
                raise ValueError(f"grid holds radars {radars_in}; pass radar= (or a pair for a difference).")
            radar = radars_in[0]
        data2d = sel.sel(radar=radar)
        default_title = f"{variable} — radar {radar} — z≈{z_val:.0f} m ASL"
        resolved, is_discrete, class_labels, cbar_label = _resolve_plot_kwargs(variable, plot_kwargs)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # pixel-edge mesh from centres (regular spacing)
    res = float(ds.attrs.get("resolution_m", np.diff(ds["x"].values).mean()))
    xe = np.append(ds["x"].values - res / 2, ds["x"].values[-1] + res / 2)
    ye = np.append(ds["y"].values - res / 2, ds["y"].values[-1] + res / 2)
    p = ax.pcolormesh(xe, ye, data2d.values, shading="flat", **resolved)

    ccrs, _ = _maybe_cartopy()
    if use_cartopy is None:
        use_cartopy = ccrs is not None
    if use_cartopy:
        _draw_lv95_borders(ax)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(*(xlim if xlim is not None else (xe[0], xe[-1])))
    ax.set_ylim(*(ylim if ylim is not None else (ye[0], ye[-1])))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{(v - 2e6) / 1e3:.0f}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{(v - 1e6) / 1e3:.0f}"))
    ax.set_xlabel("East [km]")
    ax.set_ylabel("North [km]")
    ax.set_title(title or default_title)

    if add_colorbar:
        _add_colorbar(p, ax, is_discrete, class_labels, cbar_label)
    return fig, ax, p


# ============================================================================
# Vertical cross-section (arbitrary line, from crop_cross_section)
# ============================================================================

def plot_cross_section(
    df_cs,
    variable: str = "DBZH",
    ax=None,
    figsize=(12, 5),
    title: str | None = None,
    add_colorbar: bool = True,
    edgecolor="none",
    xlim=None,
    ylim=None,
    **plot_kwargs,
):
    """Render a vertical cross-section from a :meth:`RadDB.crop_cross_section` result.

    Each row's ``cs_polygon`` — the gate's 4-corner polygon in the
    (distance-along-line, altitude) plane — is drawn as a filled patch coloured
    by ``variable`` (per-variable colormap defaults apply, incl. discrete HC
    classes).  Axes: distance along the section line [km] (from ``p1``) vs
    altitude [km ASL].

    Note: pass a **single volume** (filter by ``volume_time``) and ideally a
    single radar — overlapping radars/volumes draw on top of each other.

    Parameters
    ----------
    df_cs : pd.DataFrame
        Output of :meth:`RadDB.crop_cross_section` (needs ``cs_polygon`` +
        ``variable`` columns).
    variable : str
        Column to colour by (default ``"DBZH"``).
    ax : matplotlib Axes, optional
    figsize : tuple
    title : str, optional
    add_colorbar : bool
    edgecolor : matplotlib color
        Patch edge colour (default ``"none"``; e.g. ``"k"`` to outline gates).
    xlim : (dmin_km, dmax_km), optional
        Along-section distance limits in km.
    ylim : (zmin_km, zmax_km), optional
        Altitude limits in km.
    **plot_kwargs
        ``cmap`` / ``vmin`` / ``vmax`` / ``norm`` overrides.

    Returns
    -------
    (fig, ax, collection)
    """
    from matplotlib.collections import PolyCollection

    if "cs_polygon" not in df_cs.columns:
        raise KeyError("df_cs has no 'cs_polygon' column; use RadDB.crop_cross_section first.")
    if variable not in df_cs.columns:
        raise KeyError(f"variable {variable!r} not in df_cs columns.")

    data = df_cs[df_cs[variable].notna() & df_cs["cs_polygon"].notna()]
    if data.empty:
        raise ValueError(f"no non-NaN {variable!r} values on this cross-section.")

    plot_kwargs, is_discrete, class_labels, cbar_label = _resolve_plot_kwargs(
        variable, plot_kwargs
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # polygons in km on both axes
    verts = [np.asarray(p.exterior.coords)[:, :2] / 1000.0 for p in data["cs_polygon"]]
    pc = PolyCollection(verts, array=data[variable].to_numpy(), edgecolor=edgecolor, linewidth=0.1)
    if "cmap" in plot_kwargs:
        pc.set_cmap(plot_kwargs["cmap"])
    if "norm" in plot_kwargs:
        pc.set_norm(plot_kwargs["norm"])
    else:
        pc.set_clim(plot_kwargs.get("vmin"), plot_kwargs.get("vmax"))
    ax.add_collection(pc)

    d_all = np.concatenate([v[:, 0] for v in verts])
    z_all = np.concatenate([v[:, 1] for v in verts])
    ax.set_xlim(*(xlim if xlim is not None else (d_all.min(), d_all.max())))
    ax.set_ylim(*(ylim if ylim is not None else (max(0.0, z_all.min() - 0.2), z_all.max() + 0.2)))

    ax.set_xlabel("Distance along section [km]")
    ax.set_ylabel("Altitude [km ASL]")
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    if add_colorbar:
        _add_colorbar(pc, ax, is_discrete, class_labels, cbar_label)
    return fig, ax, pc


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
    coords : {"geo", "cartesian", "swiss"}
        Coordinate frame for the axes:

        - ``"geo"`` — ``(longitude, latitude)`` degrees.
        - ``"cartesian"`` — ``(x, y)`` km **relative to the radar** (origin at the
          site).  With cartopy this is an AEQD map centred on the radar.
        - ``"swiss"`` (aka ``"lv95"`` / ``"2056"``) — absolute **Swiss LV95**
          ``(x_2056, y_2056)``, axis ticks in km (E from the 2 000 km
          false-easting, N from 1 000 km).  This is the **same frame as the AOI
          quicklook**, so a cropped-df PPI lines up with `crop_*` overlays.  With
          cartopy (``use_cartopy``) it overlays country borders reprojected to
          LV95 as a basemap; otherwise the same plot without borders.
    add_range_rings : bool
        For ``cartesian`` / ``swiss`` modes, draw 50/100/150 km range rings.
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

    # If the user passed a non-GeoAxes axis, silently disable cartopy — except
    # for "swiss", which draws on a plain axis (borders are reprojected lines).
    if use_cartopy and ax is not None and coords not in ("swiss", "lv95", "2056"):
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
    # ---------------------------------------------------------- swiss (LV95)
    elif coords in ("swiss", "lv95", "2056"):
        # Absolute Swiss LV95 (EPSG:2056) — same frame as the AOI quicklook.
        # With cartopy, an LV95 GeoAxes adds a country-border basemap; otherwise
        # plain matplotlib.  Either way the axes are LV95 km (radar x, range rings).
        if "x_2056" not in ds.coords:
            raise KeyError(
                "coords='swiss' needs x_2056/y_2056 in the sweep; reconstruct the "
                "DataTree from a LUT that has the EPSG:2056 projection columns."
            )

        # Gate mesh in LV95: reproject the corner mesh for flat shading, else centroids.
        import shapely
        from raddb.aoi import _reproject_to_2056
        if all(k in ds.variables for k in ("lon_edges", "lat_edges")):
            import pyproj
            from raddb.aoi import _to_pyproj_crs
            _tf = pyproj.Transformer.from_crs(
                _to_pyproj_crs(4326), _to_pyproj_crs(2056), always_xy=True
            )
            lon_e = ds["lon_edges"].values
            lat_e = ds["lat_edges"].values
            xe, ye = _tf.transform(lon_e.ravel(), lat_e.ravel())
            mx = np.asarray(xe).reshape(lon_e.shape)
            my = np.asarray(ye).reshape(lat_e.shape)
            mshade = "flat"
        else:
            mx, my = ds["x_2056"].values, ds["y_2056"].values
            mshade = "auto"
        site = _reproject_to_2056(
            shapely.Point(float(ds["site_longitude"]), float(ds["site_latitude"])), 4326
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        # Data range now, so borders drawn afterwards don't expand the view.
        data_xlim = xlim if xlim is not None else (float(np.nanmin(mx)), float(np.nanmax(mx)))
        data_ylim = ylim if ylim is not None else (float(np.nanmin(my)), float(np.nanmax(my)))

        p = ax.pcolormesh(mx, my, da.values, shading=mshade, **plot_kwargs)
        # cartopy country-border basemap (reprojected to LV95, plain-axis clipped)
        if use_cartopy:
            _draw_lv95_borders(ax)
        ax.set_aspect("equal")
        ax.plot(site.x, site.y, "kx", markersize=7, markeredgewidth=2, zorder=5)
        if add_range_rings:
            theta = np.linspace(0, 2 * np.pi, 361)
            for d_km in (50, 100, 150):
                ax.plot(site.x + d_km * 1e3 * np.cos(theta),
                        site.y + d_km * 1e3 * np.sin(theta),
                        "k--", linewidth=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(data_xlim)
        ax.set_ylim(data_ylim)

        # LV95 km tick labels (same convention as the AOI quicklook).
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{(v - 2e6) / 1e3:.0f}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{(v - 1e6) / 1e3:.0f}"))
        ax.set_xlabel("East [km]")
        ax.set_ylabel("North [km]")
    else:
        raise ValueError(f"coords must be 'geo', 'cartesian', or 'swiss', got {coords!r}")

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
