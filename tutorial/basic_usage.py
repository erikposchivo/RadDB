"""
Basic Usage Example for RadDB
==============================

This example demonstrates the essential public workflow for archiving
radar data starting from **DataTree files on disk** (NetCDF / Zarr):

1. Build (or obtain) DataTree volumes and save them to disk
2. Discover the DataTree files with ``find_datatree_files``
3. Archive them with ``RadDB.archive(datatree_dir=...)``
   (the radar LUT is generated automatically from the first volume)
4. ``RadDB.open(...)`` returns a data-carrying ``RadDB`` (``rdf``) with a
   polars backend; filter / convert / plot from it

RadDB is a **generic** library — it works with any xarray DataTree with
the standard xradar coordinate layout.  No pyart / radar_api needed.
If you already hold DataTrees in memory, skip the disk round-trip and
call ``db.archive(datatree=dt, radar=...)`` directly.
"""
#%%
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import raddb

# =============================================================================
# CONFIGURATION
# =============================================================================

WORK_DIR = Path("./raddb_tutorial")          # scratch dir for this demo
INPUT_DIR = WORK_DIR / "datatrees"           # where the DataTree files live
BASE_PATH = WORK_DIR / "archive"             # RadDB output (LUT + parquet)
RADAR_NAME = "A"                             # single letter A-Z

INPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STEP 1: Create sample DataTree volumes and save them to disk
# =============================================================================
# In a real workflow these files come from your own converter (any radar
# format -> xradar-style DataTree, saved as NetCDF or Zarr).


def make_volume(vol_time: pd.Timestamp, n_az: int = 90, n_rng: int = 120,
                n_sweeps: int = 3) -> xr.DataTree:
    """Synthetic multi-sweep volume with the standard xradar layout."""
    az = np.linspace(0, 360 - 360 / n_az, n_az)
    rng_vals = np.linspace(500, 60_000, n_rng)
    time_vals = np.array([vol_time] * n_az, dtype="datetime64[ns]")

    dict_ds = {}
    for sweep_idx in range(1, n_sweeps + 1):
        dbzh = np.random.uniform(-5, 45, (n_az, n_rng)).astype(np.float32)
        ds = xr.Dataset(
            {
                "DBZH":  (["azimuth", "range"], dbzh),
                "ZDR":   (["azimuth", "range"], np.random.uniform(-1, 3, (n_az, n_rng)).astype(np.float32)),
                "RHOHV": (["azimuth", "range"], np.full((n_az, n_rng), 0.97, np.float32)),
                "PHIDP": (["azimuth", "range"], np.zeros((n_az, n_rng), np.float32)),
                "time":  (["azimuth"], time_vals),
            },
            coords={
                "azimuth":         az,
                "range":           rng_vals,
                "elevation":       (["azimuth"], np.full(n_az, 0.5 * sweep_idx)),
                "elevation_angle": 0.5 * sweep_idx,
                "latitude":        46.04,
                "longitude":       8.83,
                "altitude":        1626.0,
            },
        )
        ds.attrs["sweep_number"] = sweep_idx
        dict_ds[f"sweep_{sweep_idx}"] = ds
    return xr.DataTree.from_dict(dict_ds)


print("Writing sample DataTree volumes to disk...")
for minute in (0, 5, 10):
    vol_time = pd.Timestamp(f"2024-07-15 12:{minute:02d}:00")
    dt = make_volume(vol_time)
    fname = INPUT_DIR / f"{RADAR_NAME}_{vol_time:%Y%m%d_%H%M%S}.nc"
    dt.to_netcdf(fname)
    print(f"  {fname}")
    # Zarr works exactly the same way:
    # dt.to_zarr(INPUT_DIR / f"{RADAR_NAME}_{vol_time:%Y%m%d_%H%M%S}.zarr")

#%%
# =============================================================================
# STEP 2: Discover DataTree files on disk
# =============================================================================

files = raddb.find_datatree_files(
    INPUT_DIR,
    recursive=True,
    start_time="2024-07-15 12:00",
    end_time="2024-07-15 12:59",
)
print(f"\nDiscovered {len(files)} DataTree file(s):")
for f in files:
    print(f"  {f}")

# Load a single file manually if you want to inspect it first:
dt_preview = raddb.open_any_datatree(files[0])
print(f"\nPreview of {files[0].name}: {raddb.list_sweep_names(dt_preview)}")

#%%
# =============================================================================
# STEP 3: Archive end to end (LUT generated automatically)
# =============================================================================
# Gates failing the filter (default DBZH > 0) are dropped to keep the
# archive small.  A checkpoint file makes interrupted runs resumable.

db = raddb.RadDB(archive_dir=str(BASE_PATH), crs=2056)

result = db.archive(
    datatree_dir=INPUT_DIR,     # directory of saved .nc / .zarr volumes
    radar=RADAR_NAME,
    filter={"var": "DBZH", "logic": ">", "threshold": 0.0},
)
print(f"\nArchived: {result['n_archived']} volume(s), failed: {result['n_failed']}")

# Already have DataTrees in memory?  Archive them directly instead:
#   db.archive(datatree=dt, radar=RADAR_NAME)
#   db.archive(datatree=[dt1, dt2], radar=RADAR_NAME)

#%%
# =============================================================================
# STEP 4: Load the archive back
# =============================================================================

# open() returns a data-carrying RadDB (``rdf``) backed by polars
rdf = db.open(
    radars=RADAR_NAME,
    time_period=("2024-07-15 00:00", "2024-07-15 23:59"),
)
print(rdf)                                       # rich multi-line summary
print(f"gates: {len(rdf):,} | radars: {rdf.radars()} | columns: {rdf.columns()}")
print(f"extent (LV95): {rdf.extent()}")

# subset / convert as needed (each returns a new RadDB or a frame)
strong = rdf.filter({"var": "DBZH", "logic": ">", "threshold": 20})
pdf = rdf.to_pandas(with_geometry=True)          # pandas + gate coordinates
# gdf = rdf.to_geopandas()                        # GeoDataFrame with a CRS
print(f"strong-echo gates: {len(strong):,}")

#%%
# =============================================================================
# STEP 5: Plot
# =============================================================================

import matplotlib.pyplot as plt

rdf.plot_ppi(sweep=1, variable="DBZH", coords="cartesian")
plt.show()
