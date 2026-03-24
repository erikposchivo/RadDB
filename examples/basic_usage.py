"""
Basic Usage Example for RadDB
==============================

This example demonstrates the essential workflow for processing radar data:
1. Initialize RadDB
2. Generate the LUT (once per radar — stores static spatial info)
3. Process and store polarimetric data
4. Load and visualize the results

The key design idea: the LUT (gate coordinates, elevation angles, etc.) is
**static** and generated only once.  Each volume scan then only stores the
**polarimetric** data (DBZH, ZDR, RHOHV, …) together with a ``gate_id`` that
links every gate back to the LUT.  This keeps downstream ML datasets small
and avoids redundant spatial information.

Requirements:
- Radar data files in METRANET format

Author: RadDB Package
Date: 2026-03-24
"""
#%%
from pathlib import Path
import raddb
import radar_api

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory paths — adjust these to match your system
BASE_PATH = "/home/erik_poschivo/Desktop/LTE_project/ltenas8/users/giacobbi/raddb"
RAW_DATA_DIR = "/home/erik_poschivo/Desktop/LTE_project/ltenas8/data/RADAR"

# Radar and network configuration
# Note: Radar name can be "MLA" or just "A" — both formats are supported
RADAR_NAME = "A"  # Or "MLA", "D", "MLD", etc.
NETWORK = "MCH_LTE"

# Time range to process
START_TIME = "2021-08-28 06:00"
END_TIME = "2021-08-28 19:55"

#%%
# =============================================================================
# STEP 1: Initialize RadDB
# =============================================================================

print("Initializing RadDB...")
db = raddb.RadDB(
    base_path=BASE_PATH,
    raw_data_dir=RAW_DATA_DIR,
    network=NETWORK,
    # Optional: disable features if not needed or not available
    static_vis_dir=None,  # Set to path if you have visibility LUTs
    qpegrid_to_rad_dir=None,  # Set to path if you have QPE grid LUTs
)

#%%
# =============================================================================
# STEP 2: Generate the LUT (run ONCE per radar)
# =============================================================================
# The LUT stores static spatial information (gate coordinates, elevation
# angles, etc.) and is reused for every subsequent volume.  If the LUT
# already exists on disk it will NOT be regenerated.

print(f"\nGenerating LUT for radar {RADAR_NAME} (skipped if already exists)...")

# We need one sample volume (all sweeps) to derive the radar geometry.
sample_files = radar_api.find_files(
    network=NETWORK,
    radar=RADAR_NAME,
    start_time=START_TIME,
    end_time=END_TIME,  # just one volume is enough
    product="POL",
    protocol="local",
    base_dir=RAW_DATA_DIR,
)

if sample_files:
    lut_path = db.generate_lut(radar=RADAR_NAME, sample_volume_filepaths=sample_files)
    print(f"  LUT ready at: {lut_path}")
else:
    print("  WARNING: no sample files found — cannot generate LUT.")

#%%
# =============================================================================
# STEP 3: Process and Store Polarimetric Data
# =============================================================================
# Only the polarimetric measurements (+ gate_id) are stored per volume.
# Missing HZT / HYM files are tolerated — the corresponding columns are
# filled with NaN so that downstream ML training can still proceed.

print(f"\nProcessing radar {RADAR_NAME} from {START_TIME} to {END_TIME}...")

results = db.process_and_store(
    radar=RADAR_NAME,
    start_time=START_TIME,
    end_time=END_TIME,
    hzt_enabled=False,  # Disable HZT (isotherm height) if not available
    hym_enabled=False,  # Disable operational hydrometeor classification
    compute_pyart_hc=False,  # Disable PyART hydrometeor classification
    max_workers=8,
    verbose=True,
)

successful = sum(1 for r in results if r["success"])
failed = sum(1 for r in results if not r["success"])

print(f"\nProcessing complete:")
print(f"  Successful: {successful}/{len(results)}")
print(f"  Failed:     {failed}/{len(results)}")

if failed > 0:
    print("\nFailed volumes:")
    for r in results:
        if not r["success"]:
            print(f"  - {r['time']}: {r['error']}")

#%%
# =============================================================================
# STEP 4: Load Processed Data (as xarray DataTree)
# =============================================================================
# load_datatree() loads the latest volume in the time range, joins it with
# the LUT (via gate_id) to restore spatial coordinates, and reconstructs
# a full xarray DataTree suitable for plotting.

if successful > 0:
    print(f"\nLoading processed data for radar {RADAR_NAME}...")

    dt = db.load_datatree(
        radar=RADAR_NAME,
        start_time=START_TIME,
        end_time=END_TIME,
    )

    sweeps = raddb.list_sweep_names(dt)
    print(f"  Loaded {len(sweeps)} sweeps: {sweeps}")

    # ==================================================================
    # STEP 5: Visualize Data
    # ==================================================================

    print("\nGenerating visualization...")

    raddb.plot_ppi(
        dt,
        sweep=1,
        variable="DBZH",
        title=f"Radar {RADAR_NAME} - Reflectivity (Sweep 1)",
    )

    print("  Plot created successfully!")
    print("\nBasic workflow complete!")

else:
    print("\nNo data was successfully processed. Check the error messages above.")

#%%
# =============================================================================
# ADDITIONAL OPTIONS
# =============================================================================

# Load polarimetric data as a pandas DataFrame (for ML training):
# df = db.load_parquet_data(
#     radar=RADAR_NAME,
#     start_time=START_TIME,
#     end_time=END_TIME,
#     data_type="POLAR",
# )
# print(df.head())
# The DataFrame contains a "gate_id" column that links every row back to
# the LUT so you can recover spatial coordinates after classification.

# Get radar metadata:
# radar_info = db.get_radar_info(RADAR_NAME)
# print(f"Radar location: {radar_info['latitude']}, {radar_info['longitude']}")

# Get the Look-Up Table directly:
# lut = db.get_lut(RADAR_NAME)
# print(lut.head())
