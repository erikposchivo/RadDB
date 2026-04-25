"""
storage_analysis.py
-------------------
Quantify how much disk space RadDB saves compared to storing radar volumes
as unfiltered, coordinate-inclusive parquet files.

Analysis steps:
  1. Single-volume comparison: raw parquet (all gates + all coords) vs. the
     RadDB POL parquet (clear-sky filtered, coordinates separated into LUT).
  2. 1-year extrapolation: count METRANET files for one year, derive number
     of volumes, multiply by per-volume savings.
  3. Multi-year file counts: collect n_files / n_volumes for each year in YEARS.
  4. Multi-year statistics: median, IQR, std, var, Q25, Q75, min, max for
     n_files, n_volumes, and total saved space per year.

Run from the RadDB package root:
    python storage_analysis.py
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from mch_pipeline import process_mch_volume, _parse_volume_time, _group_files_by_volume
from raddb.io_core import datatree_to_dataframe

# ============================================================================
# CONFIGURATION — adjust paths to match your environment
# ============================================================================

BASE_PATH      = "/home/erik_poschivo/Desktop/LTE_project/ltenas8/users/giacobbi/raddb"
RAW_DATA_DIR   = "/home/erik_poschivo/Desktop/LTE_project/ltenas8/data/RADAR"
STATIC_VIS_DIR = "/home/erik_poschivo/Desktop/LTE_project/ltenas8/data/Rad4Alp_LUTs/static_vis"
QPEGRID_DIR    = "/home/erik_poschivo/Desktop/LTE_project/ltenas8/data/Rad4Alp_LUTs/qpegrid_to_rad"

RADAR             = "A"
NETWORK           = "MCH_LTE"
SWEEPS_PER_VOLUME = 20
YEARS             = [2024]  # add more years as data becomes available

# Reference volume: a volume that exists in BOTH raw METRANET files and the
# processed RadDB archive so we can compare sizes.
# REF_METRANET_DIR may contain multiple volumes (one per 5-min slot); the
# script uses _group_files_by_volume to isolate the exact 20-sweep group
# that corresponds to REF_RADDB_PARQUET.
REF_METRANET_DIR  = f"{RAW_DATA_DIR}/MCH/2024/07/12/23/ML{RADAR}"
REF_VOLUME_STEM   = f"ML{RADAR}2419423300U"  # 2024-07-12 23:30 UTC
REF_RADDB_PARQUET = f"{BASE_PATH}/{RADAR}/2024/07/12/{RADAR}_20240712_233005_POL.parquet"

_SEP = "=" * 62


# ============================================================================
# 1. RAW PARQUET CREATION
# ============================================================================

def create_raw_parquet(datatree, output_path: str | Path) -> Path:
    """Flatten DataTree to parquet with ALL columns and ALL gates (no filtering).

    Uses datatree_to_dataframe which calls xr.Dataset.to_dataframe() per sweep,
    preserving every variable and every coordinate (azimuth, range, elevation,
    latitude, longitude, altitude, x, y, z, etc.) with no clear-sky removal.
    """
    df = datatree_to_dataframe(datatree)
    out = Path(output_path)
    df.to_parquet(out, engine="pyarrow", index=False)
    return out


# ============================================================================
# 2. SINGLE-VOLUME COMPARISON
# ============================================================================

def analyze_single_volume() -> dict:
    """Process the reference volume and compare raw vs. RadDB parquet sizes."""
    all_files = sorted(Path(REF_METRANET_DIR).glob(f"ML{RADAR}*"))
    if not all_files:
        raise FileNotFoundError(f"No METRANET files found in {REF_METRANET_DIR}")

    # The directory may hold multiple volumes (one per 5-min slot).
    # Isolate the exact 20-sweep group for the reference volume.
    volumes = _group_files_by_volume([str(p) for p in all_files])
    if REF_VOLUME_STEM not in volumes:
        raise KeyError(
            f"Reference stem '{REF_VOLUME_STEM}' not found in {REF_METRANET_DIR}. "
            f"Available stems: {sorted(volumes)}"
        )
    sweep_files_str = sorted(volumes[REF_VOLUME_STEM])
    volume_time = _parse_volume_time(REF_VOLUME_STEM)

    print(f"  Processing volume: {volume_time}  ({len(sweep_files_str)} sweeps)...")
    datatree = process_mch_volume(
        sweep_filepaths=sweep_files_str,
        network=NETWORK,
        radar=RADAR,
        volume_time=volume_time,
        raw_data_dir=RAW_DATA_DIR,
        static_vis_dir=STATIC_VIS_DIR,
        qpegrid_to_rad_dir=QPEGRID_DIR,
        verbose=False,
    )

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        create_raw_parquet(datatree, tmp_path)
        raw_bytes  = os.path.getsize(tmp_path)
    finally:
        os.unlink(tmp_path)

    raddb_path = Path(REF_RADDB_PARQUET)
    if not raddb_path.exists():
        raise FileNotFoundError(f"RadDB parquet not found: {raddb_path}")
    raddb_bytes = os.path.getsize(raddb_path)

    saved_bytes  = raw_bytes - raddb_bytes
    saving_pct   = saved_bytes / raw_bytes * 100

    return {
        "volume_time":   volume_time,
        "raw_bytes":     raw_bytes,
        "raddb_bytes":   raddb_bytes,
        "saved_bytes":   saved_bytes,
        "saving_pct":    saving_pct,
    }


# ============================================================================
# 3. FILE / VOLUME COUNTING — 1 YEAR
# ============================================================================

def count_metranet_files_year(raw_data_dir: str, radar: str, year: int) -> dict:
    """Count METRANET POL files and derive volume count for one year."""
    files = list(Path(raw_data_dir).glob(
        f"MCH/{year}/*/*/*/ML{radar}/ML{radar}*"
    ))
    n_files   = len(files)
    n_volumes = n_files // SWEEPS_PER_VOLUME
    return {"year": year, "n_files": n_files, "n_volumes": n_volumes}


# ============================================================================
# 4. FILE / VOLUME COUNTING — MULTIPLE YEARS
# ============================================================================

def count_metranet_files_multi_year(
    raw_data_dir: str, radar: str, years: list[int]
) -> pd.DataFrame:
    """Count METRANET files and volumes for each year in *years*."""
    rows = [count_metranet_files_year(raw_data_dir, radar, y) for y in years]
    return pd.DataFrame(rows)


# ============================================================================
# 5. STATISTICS OVER MULTIPLE YEARS
# ============================================================================

def compute_savings_statistics(
    multi_year_df: pd.DataFrame,
    saved_bytes_per_volume: float,
) -> pd.DataFrame:
    """Compute descriptive statistics for n_files, n_volumes, and saved_space_gb.

    Returns a DataFrame indexed by statistic name with one column per metric.
    """
    df = multi_year_df.copy()
    df["saved_space_gb"] = df["n_volumes"] * saved_bytes_per_volume / 1e9

    metrics = ["n_files", "n_volumes", "saved_space_gb"]
    stats: dict[str, dict] = {}

    for col in metrics:
        arr = df[col].to_numpy(dtype=float)
        q25, q75 = np.percentile(arr, [25, 75])
        stats[col] = {
            "median": float(np.median(arr)),
            "q25":    float(q25),
            "q75":    float(q75),
            "iqr":    float(q75 - q25),
            "std":    float(np.std(arr, ddof=1) if len(arr) > 1 else np.nan),
            "var":    float(np.var(arr, ddof=1) if len(arr) > 1 else np.nan),
            "min":    float(np.min(arr)),
            "max":    float(np.max(arr)),
        }

    return pd.DataFrame(stats)


# ============================================================================
# PRINTING HELPERS
# ============================================================================

def _print_single_volume(result: dict) -> None:
    print(f"\n{_SEP}")
    print(f"SINGLE VOLUME COMPARISON  ({RADAR}, {result['volume_time']} UTC)")
    print(_SEP)
    raw_mb   = result["raw_bytes"]   / 1e6
    raddb_mb = result["raddb_bytes"] / 1e6
    saved_mb = result["saved_bytes"] / 1e6
    print(f"  Raw parquet (all gates + all coords): {raw_mb:>8.2f} MB")
    print(f"  RadDB parquet (filtered, no LUT):     {raddb_mb:>8.2f} MB")
    print(f"  Saved per volume:                     {saved_mb:>8.2f} MB  ({result['saving_pct']:.1f}%)")
    print("-" * 62)


def _print_one_year(counts: dict, saved_bytes_per_volume: float) -> None:
    year       = counts["year"]
    n_files    = counts["n_files"]
    n_volumes  = counts["n_volumes"]
    saved_gb   = n_volumes * saved_bytes_per_volume / 1e9
    print(f"\n{_SEP}")
    print(f"1-YEAR EXTRAPOLATION  (Radar: {RADAR}, Year: {year})")
    print(_SEP)
    print(f"  METRANET files:     {n_files:>8,}")
    print(f"  Volumes:            {n_volumes:>8,}")
    print(f"  Estimated savings:  {saved_gb:>8.3f} GB")
    print("-" * 62)


def _print_multi_year(df: pd.DataFrame, saved_bytes_per_volume: float) -> None:
    display = df.copy()
    display["saved_space_gb"] = display["n_volumes"] * saved_bytes_per_volume / 1e9
    print(f"\n{_SEP}")
    print(f"MULTI-YEAR FILE COUNTS  (Radar: {RADAR})")
    print(_SEP)
    print(f"  {'Year':>6}  {'Files':>10}  {'Volumes':>10}  {'Saved (GB)':>12}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}")
    for _, row in display.iterrows():
        print(
            f"  {int(row.year):>6}  {int(row.n_files):>10,}  "
            f"{int(row.n_volumes):>10,}  {row.saved_space_gb:>12.3f}"
        )
    print("-" * 62)


def _print_statistics(stats_df: pd.DataFrame) -> None:
    print(f"\n{_SEP}")
    print(f"MULTI-YEAR STATISTICS  (Radar: {RADAR}, Years: {YEARS})")
    print(_SEP)
    col_width = 16
    header = f"  {'Statistic':>10}" + "".join(f"  {c:>{col_width}}" for c in stats_df.columns)
    print(header)
    print(f"  {'-'*10}" + f"  {'-'*col_width}" * len(stats_df.columns))
    for stat, row in stats_df.iterrows():
        vals = "".join(f"  {v:>{col_width}.3f}" for v in row)
        print(f"  {stat:>10}{vals}")
    print("-" * 62)


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print("\nRadDB Storage Savings Analysis")
    print(f"Radar: {RADAR}  |  Network: {NETWORK}")

    # --- single volume ---
    print(f"\n[1/4] Analyzing single volume comparison...")
    result = analyze_single_volume()
    _print_single_volume(result)
    saved_bytes_per_volume = result["saved_bytes"]

    # --- 1-year extrapolation (first year in YEARS) ---
    first_year = YEARS[0]
    print(f"\n[2/4] Counting METRANET files for {first_year}...")
    one_year_counts = count_metranet_files_year(RAW_DATA_DIR, RADAR, first_year)
    _print_one_year(one_year_counts, saved_bytes_per_volume)

    # --- multi-year counts ---
    #print(f"\n[3/4] Counting METRANET files for all years: {YEARS}...")
    #multi_year_df = count_metranet_files_multi_year(RAW_DATA_DIR, RADAR, YEARS)
    #_print_multi_year(multi_year_df, saved_bytes_per_volume)

    # --- statistics ---
    #print(f"\n[4/4] Computing statistics across years...")
    #stats_df = compute_savings_statistics(multi_year_df, saved_bytes_per_volume)
    #_print_statistics(stats_df)


if __name__ == "__main__":
    main()
