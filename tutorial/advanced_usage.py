"""
Advanced Usage Example for RadDB
=================================

This example demonstrates the two-step workflow:

  **Step 1 (MCH-specific):**  Raw METRANET files  ->  xarray DataTrees
      via ``mch_pipeline`` (Swiss-specific processing: visibility,
      KDP, attenuation, HZT, hydrometeor classification).

  **Step 2 (generic RadDB):**  DataTrees  ->  archived Parquet files
      via ``raddb.RadDB`` (works with any DataTree from any source).

Features demonstrated:
1. Sequential MCH processing + RadDB archiving (single radar)
2. Multi-radar sequential processing
3. Loading data from the archive (DataFrame, DataTree, multi-radar)
4. Clear-sky filtering at the DataTree level
5. Visualizations (PPI, RHI, Volume Panel, Classified)

For end-to-end processing of a long time range on a server, use
``main.py`` at the package root (resumable, volume-by-volume archive).
"""
#%%
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import raddb
from raddb import StageTimer, plot_profiling_dashboard

# MCH pipeline (not part of the RadDB package)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from raddb.mch_pipeline import (
    process_mch_volume,
    process_mch_volumes,
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
START_TIME = "2024-08-18 00:00"
END_TIME = "2024-08-18 10:00"

# Swiss MeteoSwiss radar identifiers — network-specific constant kept here,
# outside the generic raddb library.
SWISS_RADARS = ["A", "D", "L", "P", "W"]

# RadDB instance — used for all archiving and loading operations
db = raddb.RadDB(base_path=BASE_PATH)


#%%
# =============================================================================
# PART 1: MCH BATCH PROCESSING (Sequential, Single Radar)
# =============================================================================

print("=" * 70)
print("PART 1: MCH Processing (Sequential) + RadDB Archiving")
print("=" * 70)

timer = StageTimer()

# ---------------------------------------------------------------------------
# Step 1 (MCH-specific): process raw METRANET files -> DataTrees
# ---------------------------------------------------------------------------
results = process_mch_volumes(
    network=NETWORK,
    radar="A",                        # single radar
    start_time=START_TIME,
    end_time=END_TIME,
    raw_data_dir=RAW_DATA_DIR,
    static_vis_dir=STATIC_VIS_DIR,
    qpegrid_to_rad_dir=QPEGRID_TO_RAD_DIR,
    hzt_enabled=True,
    hym_enabled=True,
    compute_pyart_hc=True,
    verbose=True,
    timer=timer,
)

successful = [r for r in results if r["success"]]
failed = [r for r in results if not r["success"]]
print(f"\nProcessing: {len(successful)} OK, {len(failed)} failed")
if failed:
    print("Failed volumes:")
    for r in failed:
        print(f"  - {r['stem']}: {r['error']}")

# ---------------------------------------------------------------------------
# Step 2 (generic RadDB): archive DataTrees -> Parquet
# ---------------------------------------------------------------------------
if successful:
    # Collect DataTrees as a dict for batch archiving
    volumes_to_archive = {r["stem"]: r["datatree"] for r in successful}
    archive_results = db.archive_multiple_volumes(
        volumes=volumes_to_archive,
        radar="A",
        filter_feature="DBZH",
        filter_threshold=0.0,           # remove clear sky
        filter_logic=">",
        verbose=True,
        timer=timer,
    )
    n_archived = sum(1 for r in archive_results if r["success"])
    print(f"Archived: {n_archived}/{len(archive_results)} volumes")

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

RAD = ["A", "D", "P"]

# ---------------------------------------------------------------------------
# Step 1 (MCH-specific): process multiple radars sequentially
# ---------------------------------------------------------------------------
results_multi = process_mch_volumes(
    network=NETWORK,
    radar=RAD,                 # or "all" for all radars
    start_time=START_TIME,
    end_time=END_TIME,
    raw_data_dir=RAW_DATA_DIR,
    static_vis_dir=STATIC_VIS_DIR,
    qpegrid_to_rad_dir=QPEGRID_TO_RAD_DIR,
    hzt_enabled=True,
    hym_enabled=True,
    compute_pyart_hc=True,
    verbose=True,
)

for r in results_multi:
    status = "OK" if r["success"] else f"FAIL: {r['error']}"
    print(f"  Radar {r['radar']} | {r['stem']} | {status}")

# ---------------------------------------------------------------------------
# Step 2 (generic RadDB): archive per radar
# ---------------------------------------------------------------------------
for radar_letter in RAD:
    radar_ok = [r for r in results_multi if r["success"] and r["radar"] == radar_letter]
    if radar_ok:
        volumes_dict = {r["stem"]: r["datatree"] for r in radar_ok}
        db.archive_multiple_volumes(
            volumes=volumes_dict,
            radar=radar_letter,
            verbose=True,
        )
        print(f"  Radar {radar_letter}: archived {len(radar_ok)} volume(s)")

# Generate LUT for each radar (uses first successful volume per radar)
for radar_letter in RAD:
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
# PART 3: LOADING ARCHIVED DATA
# =============================================================================

print("\n" + "=" * 70)
print("PART 3: Loading Archived Data")
print("=" * 70)

# 3a. Load as DataFrame
print("\n3a. Loading as DataFrame...")
df = db.load_dataframe(
    radar="A",
    start_time=START_TIME,
    end_time=END_TIME,
)
print(f"  {len(df):,} rows, columns: {list(df.columns)}")

# 3b. Load with spatial coordinates
print("\n3b. Loading with LUT merge...")
df_full = db.load_dataframe(
    radar="A",
    start_time=START_TIME,
    end_time=END_TIME,
    merge_lut=True,
)
print(f"  {len(df_full):,} rows, columns: {list(df_full.columns)}")

#%%
# 3c. Load as DataTree (for visualization)
print("\n3c. Reconstructing DataTree...")
dt = db.load_datatree(
    radar="A",
    start_time=START_TIME,
    end_time=END_TIME,
)
sweeps = raddb.list_sweep_names(dt)
print(f"  {len(sweeps)} sweeps: {sweeps}")

#%%
# 3d. Multi-radar DataFrame
print("\n3d. Multi-radar loading...")
df_multi = db.load_multi_radar_dataframe(
    radars=["A", "D"],  # or "all"
    start_time=START_TIME,
    end_time=END_TIME,
    merge_lut=True
)
if not df_multi.empty:
    print(f"  {len(df_multi):,} rows from radars: {df_multi['radar'].unique().tolist()}")

# 3e. List available radars
available = db.list_available_radars()
print(f"\n3e. Available radars in archive: {available}")

#%%
# =============================================================================
# PART 4: CLEAR-SKY FILTERING
# =============================================================================

print("\n" + "=" * 70)
print("PART 4: Clear-Sky Filtering at DataTree Level")
print("=" * 70)

# You can filter clear sky at the DataTree level before archiving
print("Before filter:")
ds_before = dt["sweep_1"].to_dataset()
n_valid_before = int((~ds_before["DBZH"].isnull()).sum())
print(f"  sweep_1 valid DBZH gates: {n_valid_before:,}")

dt_filtered = raddb.filter_dt(dt, feature="DBZH", threshold=0.0, logic=">")

print("After filter (DBZH <= 0 removed):")
ds_after = dt_filtered["sweep_1"].to_dataset()
n_valid_after = int((~ds_after["DBZH"].isnull()).sum())
print(f"  sweep_1 valid DBZH gates: {n_valid_after:,}")
print(f"  Reduction: {(1 - n_valid_after / max(n_valid_before, 1)) * 100:.1f}%")

#%%
# =============================================================================
# PART 5: VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 70)
print("PART 5: Visualizations")
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
print("  - MCH processing (sequential) -> DataTrees")
print("  - RadDB archiving (DataTrees -> Parquet)")
print("  - Multi-radar support")
print("  - Parquet -> DataFrame / DataTree reconstruction")
print("  - Clear-sky filtering")
print("  - Visualizations (PPI, RHI, Volume Panel, Classified)")
