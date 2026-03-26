"""
raddb/plot.py
-------------
Plotting functions for raw and classified radar DataTrees.
"""
from __future__ import annotations
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xarray as xr
from raddb.helper import list_sweep_names

logger = logging.getLogger(__name__)

# --- Configurations --- #
_CMAPS = {"DBZH": "tab20", "ZDR": "RdBu", "RHOHV": "viridis", "PHIDP": "hsv", "HC": "Set1", "hydrometeor_class": "Set1", "HZT": "viridis"}
_VLIMS = {"DBZH": (-10, 60), "ZDR": (-2, 10), "RHOHV": (0.6, 1.05), "PHIDP": (0, 180), "HC": (0, 9), "HZT": (0, 5000)}
HC_CLASS_NAMES = {0: "NC", 1: "AG", 2: "CR", 3: "LR", 4: "RP", 5: "RN", 6: "VI", 7: "WS", 8: "MH", 9: "IH/HDG"}

def _get_cmap_vlim(var: str, vmin, vmax, cmap):
    c = cmap or _CMAPS.get(var, "viridis")
    vn = vmin if vmin is not None else _VLIMS.get(var, (None, None))[0]
    vx = vmax if vmax is not None else _VLIMS.get(var, (None, None))[1]
    return c, vn, vx

def _save_or_show(fig: plt.Figure, save_path: str | None) -> None:
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

# --- Internal Data Extractor --- #
def _get_nearest_azimuth_data(dt, var, azimuth, az_tol):
    all_data, elevations, ranges = [], [], None
    for sw in list_sweep_names(dt):
        ds = dt[sw].to_dataset()
        if var not in ds: continue
        diffs = np.abs(ds["azimuth"].values - azimuth)
        diffs = np.minimum(diffs, 360 - diffs)
        idx = int(np.argmin(diffs))
        if diffs[idx] > az_tol: continue
        
        all_data.append(ds[var].values[idx, :])
        elevations.append(float(ds.coords.get("elevation_angle", ds["elevation"].values[idx])))
        if ranges is None: ranges = ds["range"].values
    
    if not all_data: raise ValueError(f"No data near az={azimuth}°.")
    idx_sort = np.argsort(elevations)
    return np.vstack(all_data)[idx_sort, :], np.array(elevations)[idx_sort], ranges

# --- General Plotting --- #
def plot_ppi(dt, sweep, variable, vmin=None, vmax=None, cmap=None, title=None, save_path=None) -> plt.Figure:
    sweep_name = f"sweep_{sweep}" if isinstance(sweep, int) else sweep
    ds = dt[sweep_name].to_dataset()
    if variable not in ds: raise KeyError(f"Variable '{variable}' not found in {sweep_name}.")
    
    c, vn, vx = _get_cmap_vlim(variable, vmin, vmax, cmap)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.pcolormesh(ds[variable].values, cmap=c, vmin=vn, vmax=vx)
    plt.colorbar(im, ax=ax, label=variable)
    ax.set_title(title or f"PPI — {variable} — {sweep_name}")
    _save_or_show(fig, save_path)
    return fig

def plot_rhi(dt, azimuth, variable, vmin=None, vmax=None, cmap=None, title=None, save_path=None, az_tol=1.0) -> plt.Figure:
    data, elevs, ranges = _get_nearest_azimuth_data(dt, variable, azimuth, az_tol)
    c, vn, vx = _get_cmap_vlim(variable, vmin, vmax, cmap)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.pcolormesh(ranges / 1000.0, elevs, data, cmap=c, vmin=vn, vmax=vx, shading="nearest")
    plt.colorbar(im, ax=ax, label=variable)
    ax.set_title(title or f"RHI — {variable} — az={azimuth:.1f}°")
    _save_or_show(fig, save_path)
    return fig

def plot_cappi(dt, altitude_m, variable, vmin=None, vmax=None, cmap=None, title=None, save_path=None) -> plt.Figure:
    names = list_sweep_names(dt)
    c, vn, vx = _get_cmap_vlim(variable, vmin, vmax, cmap)
    vals, g_alt, ref_ds = [], [], None
    
    for sw in names:
        ds = dt[sw].to_dataset()
        if variable not in ds: continue
        z = ds["z"].values if "z" in ds else float(ds.coords.get("altitude", 0)) + ds["range"].values[np.newaxis, :] * np.sin(np.deg2rad(float(np.mean(ds["elevation"]))))
        vals.append(ds[variable].values)
        g_alt.append(z)
        if ref_ds is None: ref_ds = ds
        
    n_az, n_rng = vals[0].shape
    cappi = np.full((n_az, n_rng), np.nan)
    for az in range(n_az):
        for rng in range(n_rng):
            best = int(np.argmin(np.abs(np.array([g[az, rng] for g in g_alt]) - altitude_m)))
            cappi[az, rng] = vals[best][az, rng]
            
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.pcolormesh(cappi, cmap=c, vmin=vn, vmax=vx)
    plt.colorbar(im, ax=ax, label=variable)
    ax.set_title(title or f"CAPPI — {variable} — {altitude_m:.0f} m ASL")
    _save_or_show(fig, save_path)
    return fig

def plot_volume_panel(dt, variable, sweeps=None, ncols=4, vmin=None, vmax=None, cmap=None, save_path=None) -> plt.Figure:
    all_sweeps = list_sweep_names(dt)
    sel = [f"sweep_{s}" if isinstance(s, int) else s for s in sweeps] if sweeps else all_sweeps
    sel = [s for s in sel if s in all_sweeps]
    
    c, vn, vx = _get_cmap_vlim(variable, vmin, vmax, cmap)
    nrows = int(np.ceil(len(sel) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes_flat = np.array(axes).ravel()
    
    for i, sw in enumerate(sel):
        ax = axes_flat[i]
        ds = dt[sw].to_dataset()
        if variable in ds:
            ax.pcolormesh(ds[variable].values, cmap=c, vmin=vn, vmax=vx, rasterized=True)
            ax.set_title(f"{sw} | el={float(np.mean(ds['elevation'])):.1f}°", fontsize=8)
        else: ax.set_title(f"{sw} — N/A", fontsize=8)
        ax.axis("off")
        
    for j in range(len(sel), len(axes_flat)): axes_flat[j].set_visible(False)
    fig.suptitle(f"Volume Panel — {variable}", fontsize=12, y=1.01)
    _save_or_show(fig, save_path)
    return fig

# --- Classified Plots (from reconstruct.py) --- #
def _create_discrete_cbar(fig, ax, im, cmap_str, n_classes, label="Class"):
    cmap = plt.get_cmap(cmap_str, n_classes)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n_classes + 0.5, 1), n_classes)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(n_classes))
    cbar.set_ticklabels([HC_CLASS_NAMES.get(i, str(i)) for i in range(n_classes)])
    cbar.set_label(label)
    return cmap, norm

def plot_classified_ppi(dt, sweep, variable="hydrometeor_class", cmap="Set1", title=None, save_path=None, show_colorbar=True) -> plt.Figure:
    sweep_name = f"sweep_{sweep}" if isinstance(sweep, int) else sweep
    ds = dt[sweep_name].to_dataset()
    if variable not in ds: raise KeyError(f"'{variable}' not in {sweep_name}.")
    
    n_classes = 10
    cmap_disc = plt.get_cmap(cmap, n_classes)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n_classes + 0.5, 1), n_classes)
    
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.pcolormesh(ds[variable].values.astype(float), cmap=cmap_disc, norm=norm, rasterized=True)
    if show_colorbar: _create_discrete_cbar(fig, ax, im, cmap, n_classes, "Hydrometeor class")
        
    ax.set_title(title or f"Classified PPI — {sweep_name} | el={ds.coords.get('elevation_angle', '?')}°")
    ax.set_aspect("equal")
    _save_or_show(fig, save_path)
    return fig

def plot_classified_rhi(dt, azimuth, variable="hydrometeor_class", cmap="Set1", title=None, save_path=None, az_tol=1.0) -> plt.Figure:
    data, elevs, ranges = _get_nearest_azimuth_data(dt, variable, azimuth, az_tol)
    n_classes = 10
    cmap_disc = plt.get_cmap(cmap, n_classes)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, n_classes + 0.5, 1), n_classes)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.pcolormesh(ranges / 1000.0, elevs, data, cmap=cmap_disc, norm=norm, shading="nearest")
    _create_discrete_cbar(fig, ax, im, cmap, n_classes, "Hydrometeor class")
    
    ax.set_title(title or f"Classified RHI — az={azimuth:.1f}°")
    _save_or_show(fig, save_path)
    return fig
