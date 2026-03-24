"""
raddb/helper.py
---------------
Shared utilities and configuration for RadDB.
"""
#from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
#from donfig import Config
#from raddb.configs import read_configs

# --- Config Initialization ---
'''def _read_donfig():
    try:
        return {k: v for k, v in read_configs().items() if v is not None}
    except Exception:
        return {}

def init_config():
    defaults = {"base_dir": None}
    defaults.update(_read_donfig())
    return Config("radar", defaults=[defaults], paths=[])

config = init_config()'''

# --- DataTree Helpers ---
def list_sweep_names(dt: xr.DataTree) -> list[str]:
    """Return sorted sweep group names from a DataTree."""
    pat = re.compile(r"^sweep_\d+$")
    return sorted(s.lstrip("/") for s in dt.groups if pat.match(s.lstrip("/")))

def nan_field_like(ds: xr.Dataset, reference_var: str = "DBZH") -> xr.DataArray:
    """Return a DataArray of NaNs with the same shape/dims as reference_var."""
    return xr.full_like(ds[reference_var], fill_value=np.nan, dtype=float)

# --- Parquet Helpers ---
def read_parquet_dataset(
    base_path: str | Path,
    pattern: str = "**/*POLAR.parquet",
    columns: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    files = sorted(Path(base_path).rglob(pattern))
    if not files:
        if verbose: print(f"[RadDB] No files found matching '{pattern}' in {base_path}")
        return pd.DataFrame()
    if verbose: print(f"[RadDB] Found {len(files)} file(s) — loading...")
    return pd.concat([pd.read_parquet(f, columns=columns, engine="pyarrow") for f in files], ignore_index=True)

def check_dataframe(df: pd.DataFrame) -> None:
    print("-" * 50)
    print(f"Shape:   {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("-" * 50)
    print(f"Missing values:\n{df.isnull().sum()}")
    print("-" * 50)
    print(df.head())
    print("-" * 50)

# --- File Discovery ---
def _find_polar_files_in_range(
    radar_path: Path,
    start_time: str | pd.Timestamp | None = None,
    end_time: str | pd.Timestamp | None = None,
) -> list[Path]:
    """Return POLAR parquet files within the given time range, sorted by timestamp."""
    polar_files = sorted(radar_path.rglob("*_POLAR.parquet"))
    if not polar_files:
        return []

    start_dt = pd.to_datetime(start_time) if start_time else None
    end_dt = pd.to_datetime(end_time) if end_time else None

    valid = []
    for f in polar_files:
        # Filename: {radar}_{YYYYMMDD}_{HHMMSS}_POLAR.parquet
        stem = f.stem.replace("_POLAR", "")
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        try:
            ts = pd.to_datetime(parts[-2] + "_" + parts[-1], format="%Y%m%d_%H%M%S")
        except Exception:
            continue
        if start_dt and ts < start_dt:
            continue
        if end_dt and ts > end_dt:
            continue
        valid.append((ts, f))

    valid.sort(key=lambda x: x[0])
    return [f for _, f in valid]


# --- Radar Name Normalization ---
def normalize_radar_name(radar: str) -> str:
    """
    Normalize radar name to use only the single letter identifier.

    Handles both formats:
    - "MLA" -> "A"
    - "A" -> "A"
    - "MLW" -> "W"

    Parameters
    ----------
    radar : str
        Radar name (can be "ML*" format or just the letter)

    Returns
    -------
    str
        Normalized single-letter radar name (uppercase)

    Examples
    --------
    >>> normalize_radar_name("MLA")
    'A'
    >>> normalize_radar_name("A")
    'A'
    >>> normalize_radar_name("MLW")
    'W'
    """
    radar_upper = radar.upper().strip()

    # If it starts with "ML", extract the last character
    if radar_upper.startswith("ML") and len(radar_upper) > 2:
        return radar_upper[-1]

    # Otherwise, return the last character (or the string if it's already a single char)
    return radar_upper[-1] if len(radar_upper) > 0 else radar_upper
