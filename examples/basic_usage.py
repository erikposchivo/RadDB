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
BASE_PATH = "/ltenas8/users/giacobbi/raddb"

# Radar identifier (single letter)
RADAR_NAME = "L"

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

START_TIME = "2022-08-01 12:00"
END_TIME = "2022-08-10 12:00"

# Load as DataFrame (for ML / analysis)
df = db.load_dataframe(
    radar=RADAR_NAME,
    start_time=START_TIME,
    end_time=END_TIME,
    merge_lut=True
)
print(f"  DataFrame: {len(df):,} rows, columns: {list(df.columns)}")
print(df.head())

#%%
# Load as DataTree (for visualization)
dt = db.load_datatree(
    radar=RADAR_NAME,
    start_time=START_TIME,
    end_time=END_TIME,
)
sweeps = raddb.list_sweep_names(dt)
print(f"  DataTree: {len(sweeps)} sweeps: {sweeps}")

#%%
# =============================================================================
# STEP 6: Visualize
# =============================================================================

print("\nGenerating PPI plot...")
raddb.plot_ppi(
    dt,
    sweep=1,
    variable="DBZH",
    title=f"Radar {RADAR_NAME} - Reflectivity (Sweep 1)",
)

# Get radar metadata
radar_info = db.get_radar_info(RADAR_NAME)
print(f"\nRadar location: {radar_info['latitude']}, {radar_info['longitude']}")

print("\nBasic workflow complete!")

#%%
# =============================================================================
# ADDITIONAL: Load with spatial coordinates (merge with LUT)
# =============================================================================

df_with_coords = db.load_dataframe(
    radar=RADAR_NAME,
    start_time=START_TIME,
    end_time=END_TIME,
    merge_lut=True,  # adds azimuth, range, x, y, z, lat, lon, alt
)
print(f"\nDataFrame with LUT: {len(df_with_coords):,} rows")
print(f"Columns: {list(df_with_coords.columns)}")
