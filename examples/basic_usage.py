"""
Basic Usage Example for RadDB
==============================

This example demonstrates the essential workflow for archiving radar data:
1. Initialize RadDB
2. Generate the LUT from a sample DataTree (once per radar)
3. Archive DataTree volumes to Parquet
4. Load and visualize the results

RadDB is a **generic** library — it works with any xarray DataTree.
MCH/METRANET-specific ingestion is handled separately by mch_pipeline.py.

Author: RadDB Package
Date: 2026-03-25
"""
#%%
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
print(f"Added to PYTHONPATH: {sys.path[0]}")
import raddb

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base directory for RadDB storage (LUT + POLAR parquet files)
BASE_PATH = "/home/erik_poschivo/Desktop/LTE_project/ltenas8/users/giacobbi/raddb"

# Radar identifier (single letter)
RADAR_NAME = "A"

# =============================================================================
# STEP 1: Initialize RadDB
# =============================================================================

print("Initializing RadDB...")
db = raddb.RadDB(base_path=BASE_PATH)


#%%
# =============================================================================
# STEP 2: Get a DataTree (from MCH pipeline or any other source)
# =============================================================================
# RadDB is generic — it archives any xarray DataTree.  Here we show
# how to use the MCH-specific pipeline to produce DataTrees from
# METRANET files.  You can skip this and provide DataTrees from any source.

print("\nLoading DataTree via MCH pipeline...")

# Import the MCH pipeline (not part of the RadDB package)
from mch_pipeline import process_mch_volume, find_files_with_fallback, _group_files_by_volume, _parse_volume_time

RAW_DATA_DIR = "/home/erik_poschivo/Desktop/LTE_project/ltenas8/data/RADAR"
NETWORK = "MCH_LTE"
START_TIME = "2024-08-01 12:00"
END_TIME = "2024-08-15 12:00"

# Find METRANET files
fps = find_files_with_fallback(
    network=NETWORK,
    radar=RADAR_NAME,
    start_time=START_TIME,
    end_time=END_TIME,
    product="POL",
    raw_data_dir=RAW_DATA_DIR,
    verbose=True,
)

# Group into volumes
volumes = _group_files_by_volume(fps)
print(f"Found {len(fps)} sweep files -> {len(volumes)} volume(s)")

#%%
# =============================================================================
# STEP 3: Generate the LUT (run ONCE per radar)
# =============================================================================
# The LUT stores static spatial information and is reused for every volume.
# It requires one sample DataTree to derive the geometry.

print(f"\nGenerating LUT for radar {RADAR_NAME} (skipped if already exists)...")

# Process the first volume to get a sample DataTree
first_stem, first_paths = list(volumes.items())[0]
vol_time = _parse_volume_time(first_stem)

dt_sample = process_mch_volume(
    sweep_filepaths=first_paths,
    network=NETWORK,
    radar=RADAR_NAME,
    volume_time=vol_time,
    raw_data_dir=RAW_DATA_DIR,
    hzt_enabled=False,
    hym_enabled=False,
    compute_pyart_hc=False,
    verbose=True,
)

lut_path = db.generate_lut(
    radar=RADAR_NAME,
    sample_datatree=dt_sample,
    ke=1.25,
    network=NETWORK,
)
print(f"  LUT ready at: {lut_path}")

#%%
# =============================================================================
# STEP 4: Archive DataTree Volumes to Parquet
# =============================================================================
# Each volume is archived as a POLAR parquet file.  Clear-sky gates
# (DBZH <= 0) are automatically removed to reduce file sizes.

print(f"\nArchiving {len(volumes)} volume(s)...")

timer = raddb.StageTimer()

# Process and archive each volume
for stem, sweep_paths in list(volumes.items())[:3]:  # first 3 for demo
    vol_time = _parse_volume_time(stem)
    print(f"\n  Processing volume {stem}...")

    # Step A: MCH pipeline -> DataTree
    dt = process_mch_volume(
        sweep_filepaths=sweep_paths,
        network=NETWORK,
        radar=RADAR_NAME,
        volume_time=vol_time,
        raw_data_dir=RAW_DATA_DIR,
        hzt_enabled=False,
        hym_enabled=False,
        compute_pyart_hc=False,
        verbose=False,
    )

    # Step B: Archive with RadDB (generic)
    polar_path = db.archive_volume(
        dt,
        radar=RADAR_NAME,
        dbzh_threshold=0.0,  # remove clear sky
        timer=timer,
        volume_label=stem,
    )
    print(f"  Saved: {polar_path}")

timer.print_summary()

#%%
# =============================================================================
# STEP 5: Load Processed Data
# =============================================================================

print(f"\nLoading processed data for radar {RADAR_NAME}...")

START_TIME = "2024-07-01 00:00"
END_TIME = "2024-07-15 23:59"

# Load as DataFrame (for ML / analysis)
df = db.load_dataframe(
    radar=RADAR_NAME,
    start_time=START_TIME,
    end_time=END_TIME,
    merge_lut=True
)
print(f"  DataFrame: {len(df):,} rows, columns: {list(df.columns)}")

#%%
# =============================================================================
# STEP 5b: Corner arrays for correct gate rendering
# =============================================================================
# Plotting uses a PyART-style approach: 2-D gate CORNERS (N_az+1, N_range+1)
# computed with the 4/3 Earth-radius model and complex-plane azimuth
# interpolation (handles the 360°/0° seam). Corners live in a separate file
# `{radar}_corners.npz` alongside the LUT parquet.
#
# For LUTs generated before this feature existed, rebuild the corners from
# the existing LUT parquet + info yaml:
from raddb.lut import compute_corners_from_lut

corners_path = Path(BASE_PATH) / RADAR_NAME / "LUT" / f"{RADAR_NAME}_corners.npz"
if not corners_path.exists():
    print(f"Generating corners for radar {RADAR_NAME} ...")
    compute_corners_from_lut(RADAR_NAME, BASE_PATH, ke=1.25)
else:
    print(f"Corners already present at {corners_path}")

#%%
# =============================================================================
# STEP 5c: Test the corrected plotting functions (PPI + cross-section)
# =============================================================================
# Two ways to call the plotting:
#   1. High-level on the RadDB instance:
#        db.plot_ppi(radar, timestep, sweep, variable)
#        db.plot_cross_section_ppi(radar, timestep, azimuth, variable)
#   2. Low-level on a DataTree:
#        raddb.plot_ppi(dt, sweep, variable)
#        raddb.plot_cross_section_ppi(dt, azimuth, variable)
import matplotlib.pyplot as plt

PLOT_TIMESTEP = "2024-07-15 23:00:11"   # single volume timestamp
PLOT_SWEEP    = 3
PLOT_AZIMUTH  = 90.0

# Load the DataTree once for the low-level sanity check
dt_plot = db.load_datatree(
    radar=RADAR_NAME,
    start_time=PLOT_TIMESTEP,
    end_time=PLOT_TIMESTEP,
)
# Verify corner arrays are attached (size should be N_az+1, N_range+1).
ds_sw = dt_plot[f"sweep_{PLOT_SWEEP}"].to_dataset()
assert "x_edges" in ds_sw.data_vars, "corner arrays missing — rebuild with compute_corners_from_lut"
print(f"x_edges shape: {ds_sw['x_edges'].shape}  (N_az+1, N_range+1)")
print(f"lon_edges shape: {ds_sw['lon_edges'].shape}")

# --- PPI: DBZH, cartesian view (uses x/y edges, no cartopy required) ---
db.plot_ppi(
    radar=RADAR_NAME, timestep=PLOT_TIMESTEP,
    sweep=PLOT_SWEEP, variable="DBZH", coords="cartesian",
)
plt.show()

# --- PPI: DBZH, geographic view (uses metre edges + AEQD transform, PyART-style) ---
db.plot_ppi(
    radar=RADAR_NAME, timestep=PLOT_TIMESTEP,
    sweep=PLOT_SWEEP, variable="DBZH", coords="geo",
)
plt.show()

# --- PPI: classified variable (HC_PYART) → discrete colorbar with class labels ---
db.plot_ppi(
    radar=RADAR_NAME, timestep=PLOT_TIMESTEP,
    sweep=PLOT_SWEEP, variable="HC_PYART", coords="cartesian",
)
plt.show()

# --- PPI: temperature (TEMP) — derived feature ---
db.plot_ppi(
    radar=RADAR_NAME, timestep=PLOT_TIMESTEP,
    sweep=PLOT_SWEEP, variable="TEMP", coords="cartesian",
)
plt.show()

# --- Cross-section PPI: vertical slice at a given azimuth ---
# PyART's cross_section_ppi + plot_rhi pipeline: nearest ray per sweep,
# regrid to common range, render 4/3-Earth corners with shading="flat".
db.plot_cross_section_ppi(
    radar=RADAR_NAME, timestep=PLOT_TIMESTEP,
    azimuth=PLOT_AZIMUTH, variable="DBZH",
    max_range_km=150, max_height_km=15,
)
plt.show()

# --- Cross-section PPI: classified variable ---
db.plot_cross_section_ppi(
    radar=RADAR_NAME, timestep=PLOT_TIMESTEP,
    azimuth=PLOT_AZIMUTH, variable="HC_PYART",
    max_range_km=150, max_height_km=15,
)
plt.show()

#%%
# --- plot_rhi is a backwards-compat alias for plot_cross_section_ppi ---
db.plot_rhi(
    radar=RADAR_NAME, timestep=PLOT_TIMESTEP,
    azimuth=PLOT_AZIMUTH, variable="TEMP",
    max_range_km=150, max_height_km=15,
)
plt.show()

# --- Low-level call: pass the DataTree directly ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
raddb.plot_ppi(dt_plot, sweep=PLOT_SWEEP, variable="DBZH",
               coords="cartesian", ax=axes[0])
raddb.plot_cross_section_ppi(dt_plot, azimuth=PLOT_AZIMUTH, variable="DBZH",
                             max_range_km=150, max_height_km=15, ax=axes[1])
plt.tight_layout()
plt.show()

#%%
#check the A_LUT.paruqet file
import pandas as pd

radar_path = Path(BASE_PATH) / RADAR_NAME
lut_path = radar_path / "LUT" / f"{RADAR_NAME}_LUT.parquet"
lut_df = pd.read_parquet(lut_path, engine="pyarrow")

#%%
# Load as DataTree (for visualization)
dt = db.load_datatree(
    radar=RADAR_NAME,
    start_time=START_TIME,
    end_time=END_TIME,
)
sweeps = raddb.list_sweep_names(dt)
print(f"  DataTree: {len(sweeps)} sweeps: {sweeps}")



