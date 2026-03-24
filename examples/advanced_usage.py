"""
Advanced Usage Example for RadDB
=================================

This example demonstrates advanced features:
1. Generating Look-Up Tables (LUT)
2. Processing with all optional features enabled
3. Multiple visualization types
4. Low-level API usage for custom workflows

Requirements:
- Radar data files in METRANET format
- Static visibility LUTs (optional)
- QPE grid LUTs (optional)
- HZT (isotherm height) files (optional)

Author: RadDB Package
Date: 2026-03-24
"""

import raddb
import radar_api
from pathlib import Path
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory paths
BASE_PATH = "/ltenas8/users/giacobbi/raddb"
RAW_DATA_DIR = "/ltenas8/data/RADAR"
STATIC_VIS_DIR = "/ltenas8/data/Rad4Alp_LUTs/static_vis"
QPEGRID_TO_RAD_DIR = "/ltenas8/data/Rad4Alp_LUTs/qpegrid_to_rad"

# Radar configuration
RADAR_NAME = "A"  # Supports both "A" and "MLA" formats
NETWORK = "MCH_LTE"

# Time range
START_TIME = "2021-08-28 06:00"
END_TIME = "2021-08-28 12:00"

# =============================================================================
# PART 1: HIGH-LEVEL API WITH ALL FEATURES
# =============================================================================

print("=" * 70)
print("PART 1: High-Level API")
print("=" * 70)

# Initialize with all features enabled
db = raddb.RadDB(
    base_path=BASE_PATH,
    raw_data_dir=RAW_DATA_DIR,
    network=NETWORK,
    static_vis_dir=STATIC_VIS_DIR,
    qpegrid_to_rad_dir=QPEGRID_TO_RAD_DIR,
)

# Step 1: Generate LUT (only needed once per radar)
# Uncomment this section if you need to generate a new LUT
"""
print("\n1. Generating Look-Up Table...")

# Find one complete volume for LUT generation
sample_files = radar_api.find_files(
    network=NETWORK,
    radar=RADAR_NAME,
    start_time="2021-08-28 12:00",
    end_time="2021-08-28 12:05",
    product="POL",
    protocol="local",
    base_dir=RAW_DATA_DIR,
)

if sample_files:
    lut_path = db.generate_lut(
        radar=RADAR_NAME,
        sample_volume_filepaths=sample_files,
        ke=1.25,  # Earth's effective radius scale factor
    )
    print(f"  LUT generated: {lut_path}")
else:
    print("  ⚠ No sample files found for LUT generation")
"""

# Step 2: Process with all features enabled
print("\n2. Processing radar data with all features...")

results = db.process_and_store(
    radar=RADAR_NAME,
    start_time=START_TIME,
    end_time=END_TIME,
    hzt_enabled=True,  # Include isotherm height interpolation
    hym_enabled=True,  # Include operational hydrometeor classification
    compute_pyart_hc=True,  # Compute PyART hydrometeor classification
    max_workers=1,  # Use parallel processing (increase for speed)
    verbose=True,
)

# Analyze results
successful = [r for r in results if r["success"]]
failed = [r for r in results if not r["success"]]

print(f"\nProcessing summary:")
print(f"  Total volumes: {len(results)}")
print(f"  Successful: {len(successful)}")
print(f"  Failed: {len(failed)}")

if successful:
    total_gates = sum(r.get("n_gates", 0) for r in successful)
    avg_gates = total_gates / len(successful)
    print(f"  Total gates processed: {total_gates:,}")
    print(f"  Average gates per volume: {avg_gates:.0f}")

# Step 3: Load and explore data
if successful:
    print("\n3. Loading processed data...")

    # Load one volume as DataTree
    first_time = successful[0]["time"]
    dt = db.load_datatree(
        radar=RADAR_NAME,
        start_time=first_time,
        end_time=first_time,
    )

    sweeps = raddb.list_sweep_names(dt)
    print(f"  Loaded {len(sweeps)} sweeps")

    # Step 4: Advanced visualizations
    print("\n4. Creating visualizations...")

    # PPI plot - Plan Position Indicator
    raddb.plot_ppi(
        dt,
        sweep=1,
        variable="DBZH",
        title=f"Radar {RADAR_NAME} - Reflectivity (Sweep 1)",
    )

    # RHI plot - Range Height Indicator
    raddb.plot_rhi(
        dt,
        azimuth=90,
        variable="DBZH",
        title=f"Radar {RADAR_NAME} - RHI at 90° Azimuth",
    )

    # Volume panel - All sweeps at once
    raddb.plot_volume_panel(
        dt,
        variable="DBZH",
        ncols=4,
    )

    # Hydrometeor classification plots (if available)
    first_sweep_ds = dt["sweep_1"].to_dataset()
    if "HC_PYART" in first_sweep_ds:
        print("  Creating hydrometeor classification plots...")
        raddb.plot_classified_ppi(
            dt,
            sweep=1,
            title=f"Radar {RADAR_NAME} - Hydrometeor Classification",
        )
        raddb.plot_classified_rhi(
            dt,
            azimuth=90,
            title=f"Radar {RADAR_NAME} - HC RHI at 90°",
        )

    print("  Visualizations complete!")

    # Step 5: Load data as DataFrame for analysis
    print("\n5. Loading as DataFrame for analysis...")

    df = db.load_parquet_data(
        radar=RADAR_NAME,
        start_time=START_TIME,
        end_time=END_TIME,
        data_type="POLAR",
    )

    print(f"  Loaded {len(df):,} data points")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Sample data:")
    print(df.head())

    # Get radar metadata
    radar_info = db.get_radar_info(RADAR_NAME)
    print(f"\n6. Radar metadata:")
    print(f"  Location: {radar_info['latitude']:.4f}°N, {radar_info['longitude']:.4f}°E")
    print(f"  Altitude: {radar_info['altitude']:.1f} m")
    print(f"  Network: {radar_info['network']}")
    print(f"  Number of sweeps: {len(radar_info.get('sweeps', {}))}")

# =============================================================================
# PART 2: LOW-LEVEL API FOR CUSTOM WORKFLOWS
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: Low-Level API")
print("=" * 70)

print("\nDemonstrating low-level API for custom processing...")

# Find files manually
fps = radar_api.find_files(
    network=NETWORK,
    radar=RADAR_NAME,
    start_time=START_TIME,
    end_time=END_TIME,
    product="POL",
    protocol="local",
    base_dir=RAW_DATA_DIR,
)

if fps:
    print(f"  Found {len(fps)} sweep files")

    # Group files by volume
    volumes = defaultdict(list)
    for fp in fps:
        stem = Path(fp).stem
        volumes[stem].append(fp)

    print(f"  Grouped into {len(volumes)} volumes")

    # Process one volume using low-level functions
    if volumes:
        stem, sweep_paths = list(volumes.items())[0]
        print(f"\n  Processing volume: {stem}")

        # Load volume as DataTree
        dt_volume = raddb.volume_to_datatree(
            sweep_filepaths=sweep_paths,
            network=NETWORK,
            product="POL",
            max_workers=1,
        )

        print(f"  Loaded with {len(raddb.list_sweep_names(dt_volume))} sweeps")

        # Convert to DataFrame
        df_volume = raddb.datatree_to_dataframe(dt_volume)
        print(f"  Converted to DataFrame: {len(df_volume):,} rows")

        # Load LUT (if available)
        try:
            lut_df = db.get_lut(RADAR_NAME)
            print(f"  Loaded LUT: {len(lut_df):,} gates")

            # Join data with LUT for spatial information
            df_joined = lut_df.merge(df_volume, on="gate_id", how="right")
            print(f"  Joined with LUT: {len(df_joined):,} rows")

            # Now you can perform custom analysis with spatial coordinates
            # Example: Filter by distance from radar
            import numpy as np

            df_joined["distance"] = np.sqrt(
                df_joined["x"] ** 2 + df_joined["y"] ** 2
            ) / 1000  # Convert to km

            print(f"\n  Distance statistics:")
            print(f"    Min: {df_joined['distance'].min():.1f} km")
            print(f"    Max: {df_joined['distance'].max():.1f} km")
            print(f"    Mean: {df_joined['distance'].mean():.1f} km")

        except FileNotFoundError:
            print("  ⚠ LUT not found - generate it first using db.generate_lut()")

else:
    print("  ⚠ No files found")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("Advanced workflow complete!")
print("=" * 70)
print("\nKey features demonstrated:")
print("  ✓ LUT generation")
print("  ✓ Full-featured radar data processing")
print("  ✓ Multiple visualization types")
print("  ✓ DataFrame operations and analysis")
print("  ✓ Low-level API for custom workflows")
print("  ✓ Spatial data joining")
