"""
Advanced Usage Example for RadDB
=================================

This example demonstrates advanced features:
1. Multi-radar archiving
2. Dask parallelization for batch processing
3. Clear-sky filtering at the DataTree level
4. Loading data from multiple radars
5. Parquet -> DataTree reconstruction
6. Using the MCH batch pipeline with dask

Author: RadDB Package
Date: 2026-03-25
"""
#%%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import raddb
from raddb.helper import StageTimer, plot_profiling_dashboard

# MCH pipeline (not part of the RadDB package)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mch_pipeline import (
    process_mch_volume,
    process_and_archive_mch,
    batch_archive_mch_volumes,
    find_files_with_fallback,
    _group_files_by_volume,
    _parse_volume_time,
    generate_mch_lut,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_PATH = "/home/erik_poschivo/Desktop/LTE_project/ltenas8/users/giacobbi/raddb"
RAW_DATA_DIR = "/home/erik_poschivo/Desktop/LTE_project/ltenas8/data/RADAR"
STATIC_VIS_DIR = "/home/erik_poschivo/Desktop/LTE_project/ltenas8/data/Rad4Alp_LUTs/static_vis"
QPEGRID_TO_RAD_DIR = "/home/erik_poschivo/Desktop/LTE_project/ltenas8/data/Rad4Alp_LUTs/qpegrid_to_rad"

NETWORK = "MCH_LTE"
START_TIME = "2024-08-01 12:00"
END_TIME = "2024-08-15 12:00"

#%%
# =============================================================================
# PART 1: MCH BATCH PROCESSING (Sequential, Single Radar)
# =============================================================================

print("=" * 70)
print("PART 1: MCH Batch Processing (Sequential)")
print("=" * 70)

timer = StageTimer()

# process_and_archive_mch does everything: find files, process, archive
results = process_and_archive_mch(
    network=NETWORK,
    radar="A",                        # single radar
    start_time=START_TIME,
    end_time=END_TIME,
    base_output_path=BASE_PATH,
    raw_data_dir=RAW_DATA_DIR,
    static_vis_dir=STATIC_VIS_DIR,
    qpegrid_to_rad_dir=QPEGRID_TO_RAD_DIR,
    hzt_enabled=True,
    hym_enabled=True,
    compute_pyart_hc=True,
    dbzh_threshold=0.0,              # remove clear sky
    verbose=True,
    timer=timer,
)

# Analyze results
successful = [r for r in results if r["success"]]
failed = [r for r in results if not r["success"]]
print(f"\nResults: {len(successful)} OK, {len(failed)} failed")
if failed:
    print("Failed volumes:")
    for r in failed:
        print(f"  - {r['stem']}: {r['error']}")

timer.print_summary()
plot_profiling_dashboard(timer, title_prefix="Radar A - Full Features")

# Generate LUT from first successful volume
successful_with_files = [r for r in results if r["success"] and r.get("sweep_filepaths")]
if successful_with_files:
    lut_path = generate_mch_lut(
        radar="A",
        network=NETWORK,
        sample_volume_filepaths=successful_with_files[0]["sweep_filepaths"],
        output_base_path=BASE_PATH,
        ke=1.25,
    )
    print(f"\nLUT saved to: {lut_path}")
else:
    print("\nNo successful volumes — cannot generate LUT for radar A")

#%%
# =============================================================================
# PART 2: MULTI-RADAR PROCESSING
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: Multi-Radar Processing")
print("=" * 70)

# Process multiple radars sequentially
# Pass "all" for all five Swiss radars (A, D, L, P, W)
results_multi = process_and_archive_mch(
    network=NETWORK,
    radar=["A", "D"],                 # or "all" for all radars
    start_time=START_TIME,
    end_time=END_TIME,
    base_output_path=BASE_PATH,
    raw_data_dir=RAW_DATA_DIR,
    static_vis_dir=STATIC_VIS_DIR,
    qpegrid_to_rad_dir=QPEGRID_TO_RAD_DIR,
    hzt_enabled=False,
    hym_enabled=False,
    compute_pyart_hc=False,
    verbose=True,
)

for r in results_multi:
    status = "OK" if r["success"] else f"FAIL: {r['error']}"
    print(f"  Radar {r['radar']} | {r['stem']} | {status}")

# Generate LUT for each radar (uses first successful volume per radar)
for radar_letter in ["A", "D"]:
    first_ok = next(
        (r for r in results_multi if r["success"] and r["radar"] == radar_letter and r.get("sweep_filepaths")),
        None,
    )
    if first_ok:
        lut_path = generate_mch_lut(
            radar=radar_letter,
            network=NETWORK,
            sample_volume_filepaths=first_ok["sweep_filepaths"],
            output_base_path=BASE_PATH,
            ke=1.25,
        )
        print(f"  LUT radar {radar_letter}: {lut_path}")
    else:
        print(f"  No successful volume for radar {radar_letter} — skipping LUT")

#%%
# =============================================================================
# PART 3: DASK PARALLEL PROCESSING
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: Dask Parallel Processing")
print("=" * 70)

# Initialize Dask cluster BEFORE calling batch functions
from dask.distributed import Client

# Local cluster — adjust resources as needed:
#   n_workers:          number of parallel workers
#   threads_per_worker: threads per worker (1 for CPU-bound tasks)
#   memory_limit:       RAM per worker
client = Client(n_workers=10, threads_per_worker=1, memory_limit="16GB")
print(f"Dask dashboard: {client.dashboard_link}")

# For remote cluster (e.g., on LTESRV1):
#   client = Client("tcp://scheduler-address:8786")
#   SSH tunnel: ssh -L 8787:localhost:8787 user@ltesrv1.epfl.ch

# Batch archive with Dask parallelization
results_dask = batch_archive_mch_volumes(
    network=NETWORK,
    radar="A",
    start_time=START_TIME,
    end_time=END_TIME,
    base_output_path=BASE_PATH,
    raw_data_dir=RAW_DATA_DIR,
    static_vis_dir=STATIC_VIS_DIR,
    qpegrid_to_rad_dir=QPEGRID_TO_RAD_DIR,
    hzt_enabled=True,
    hym_enabled=True,
    compute_pyart_hc=True,
    verbose=True,
)

n_ok = sum(1 for r in results_dask if r["success"])
print(f"\nDask results: {n_ok}/{len(results_dask)} volumes succeeded")

# Generate LUT from first successful Dask volume
first_ok_dask = next(
    (r for r in results_dask if r["success"] and r.get("sweep_filepaths")),
    None,
)
if first_ok_dask:
    lut_path = generate_mch_lut(
        radar="A",
        network=NETWORK,
        sample_volume_filepaths=first_ok_dask["sweep_filepaths"],
        output_base_path=BASE_PATH,
        ke=1.25,
    )
    print(f"LUT saved to: {lut_path}")
else:
    print("No successful Dask volumes — cannot generate LUT")

client.close()

#%%
# =============================================================================
# PART 4: LOADING ARCHIVED DATA
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: Loading Archived Data")
print("=" * 70)

db = raddb.RadDB(base_path=BASE_PATH)

# 4a. Load as DataFrame
print("\n4a. Loading as DataFrame...")
df = db.load_dataframe(
    radar="A",
    start_time=START_TIME,
    end_time=END_TIME,
)
print(f"  {len(df):,} rows, columns: {list(df.columns)}")

# 4b. Load with spatial coordinates
print("\n4b. Loading with LUT merge...")
df_full = db.load_dataframe(
    radar="A",
    start_time=START_TIME,
    end_time=END_TIME,
    merge_lut=True,
)
print(f"  {len(df_full):,} rows, columns: {list(df_full.columns)}")

#%%
# 4c. Load as DataTree (for visualization)
print("\n4c. Reconstructing DataTree...")
dt = db.load_datatree(
    radar="A",
    start_time=START_TIME,
    end_time=END_TIME,
)
sweeps = raddb.list_sweep_names(dt)
print(f"  {len(sweeps)} sweeps: {sweeps}")

#%%
# 4d. Multi-radar DataFrame
print("\n4d. Multi-radar loading...")
df_multi = db.load_multi_radar_dataframe(
    radars=["A", "D"],  # or "all"
    start_time=START_TIME,
    end_time=END_TIME,
    merge_lut=True
)
if not df_multi.empty:
    print(f"  {len(df_multi):,} rows from radars: {df_multi['radar'].unique().tolist()}")

# 4e. List available radars
available = db.list_available_radars()
print(f"\n4e. Available radars in archive: {available}")

#%%
# =============================================================================
# PART 5: CLEAR-SKY FILTERING
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: Clear-Sky Filtering at DataTree Level")
print("=" * 70)

# You can filter clear sky at the DataTree level before archiving
print("Before filter:")
ds_before = dt["sweep_1"].to_dataset()
n_valid_before = int((~ds_before["DBZH"].isnull()).sum())
print(f"  sweep_1 valid DBZH gates: {n_valid_before:,}")

dt_filtered = raddb.filter_clear_sky(dt, threshold=0.0)

print("After filter (DBZH <= 0 removed):")
ds_after = dt_filtered["sweep_1"].to_dataset()
n_valid_after = int((~ds_after["DBZH"].isnull()).sum())
print(f"  sweep_1 valid DBZH gates: {n_valid_after:,}")
print(f"  Reduction: {(1 - n_valid_after / max(n_valid_before, 1)) * 100:.1f}%")

#%%
# =============================================================================
# PART 6: VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 70)
print("PART 6: Visualizations")
print("=" * 70)

# PPI
raddb.plot_ppi(dt, sweep=1, variable="DBZH", title="Radar A - DBZH (Sweep 1)")

# RHI
raddb.plot_rhi(dt, azimuth=90, variable="DBZH", title="Radar A - RHI at 90 deg")

# Volume panel
raddb.plot_volume_panel(dt, variable="DBZH", ncols=4)

# Classified (if HC data available)
ds_1 = dt["sweep_1"].to_dataset()
if "HC_PYART" in ds_1:
    raddb.plot_classified_ppi(dt, sweep=1, title="Radar A - HC (Sweep 1)")

print("\nAdvanced workflow complete!")
print("=" * 70)
print("Key features demonstrated:")
print("  - MCH batch processing (sequential & Dask parallel)")
print("  - Multi-radar support")
print("  - Parquet -> DataFrame / DataTree reconstruction")
print("  - Clear-sky filtering")
print("  - Visualizations (PPI, RHI, Volume Panel, Classified)")
