# Troubleshooting Guide for RadDB

## Issues Fixed

### 1. **Radar Name Consistency**
**Problem**: You processed with `radar="A"` but tried to load with `radar="MLA"`.

**Fix**: Use consistent radar naming throughout:
```python
# WRONG - Mixed naming
results = db.process_and_store(radar="A", ...)
dt = db.load_datatree(radar="MLA", ...)  # Will fail!

# CORRECT - Consistent naming
results = db.process_and_store(radar="MLA", ...)
dt = db.load_datatree(radar="MLA", ...)
```

**Recommendation**: Use the full radar name (e.g., "MLA", "MLD", "MLL", "MLP", "MLW") rather than just the letter.

### 2. **Visibility File Not Found**
**Problem**: `[Errno 2] No such file or directory: '/ltenas8/data/Rad4Alp_LUTs/static_vis/lut_visibility_radA.p'`

**Solutions**:
- **Option A**: Disable visibility correction if files are missing:
  ```python
  db = raddb.RadDB(
      base_path="...",
      raw_data_dir="...",
      static_vis_dir=None,  # Disable visibility
  )
  ```

- **Option B**: Provide correct path to visibility LUTs if you have them:
  ```python
  db = raddb.RadDB(
      base_path="...",
      raw_data_dir="...",
      static_vis_dir="/correct/path/to/static_vis",
  )
  ```

### 3. **HZT Interpolation Error**
**Problem**: `object of type 'datetime.datetime' has no len()`

**Fixes Applied**:
- Added better error handling in `hzt_hourly_to_5min()`
- Fixed HZT timestamp matching to find closest available timestamp
- Added check for minimum 2 HZT files required for interpolation

**Workaround** (if HZT files are missing or causing issues):
```python
results = db.process_and_store(
    radar="MLA",
    start_time="...",
    end_time="...",
    hzt_enabled=False,  # Disable HZT processing
)
```

### 4. **Radar Letter Extraction**
**Problem**: Code assumed radar names like "MLA" but failed with just "A".

**Fix**: Updated to handle both formats:
```python
# Now works with both:
radar_letter = radar[-1].upper() if len(radar) > 1 else radar.upper()
```

## Recommended Testing Workflow

Start with minimal features enabled and gradually add more:

### Step 1: Basic Processing (No Optional Features)
```python
import raddb

db = raddb.RadDB(
    base_path="/home/erik_poschivo/Desktop/LTE_project/ltenas8/users/giacobbi/raddb",
    raw_data_dir="/ltenas8/data/RADAR",
    network="MCH_LTE",
    static_vis_dir=None,  # Disable for now
    qpegrid_to_rad_dir=None,  # Disable for now
)

# Process single volume
results = db.process_and_store(
    radar="MLA",
    start_time="2021-08-28 10:00",
    end_time="2021-08-28 10:05",  # Just one volume
    hzt_enabled=False,
    hym_enabled=False,
    compute_pyart_hc=False,
    verbose=True,
)

print(results)
```

### Step 2: Add Visibility
```python
db = raddb.RadDB(
    base_path="...",
    raw_data_dir="...",
    static_vis_dir="/ltenas8/data/Rad4Alp_LUTs/static_vis",  # Enable
    qpegrid_to_rad_dir=None,
)

results = db.process_and_store(
    radar="MLA",
    start_time="2021-08-28 10:00",
    end_time="2021-08-28 10:05",
    hzt_enabled=False,
    hym_enabled=False,
    compute_pyart_hc=False,
)
```

### Step 3: Add HZT (if files available)
```python
db = raddb.RadDB(
    base_path="...",
    raw_data_dir="...",
    static_vis_dir="/ltenas8/data/Rad4Alp_LUTs/static_vis",
    qpegrid_to_rad_dir="/ltenas8/data/Rad4Alp_LUTs/qpegrid_to_rad",  # Enable
)

results = db.process_and_store(
    radar="MLA",
    start_time="2021-08-28 10:00",
    end_time="2021-08-28 10:05",
    hzt_enabled=True,  # Enable
    hym_enabled=False,
    compute_pyart_hc=False,
)
```

### Step 4: Full Processing
```python
results = db.process_and_store(
    radar="MLA",
    start_time="2021-08-28 06:00",
    end_time="2021-08-28 18:00",
    hzt_enabled=True,
    hym_enabled=True,
    compute_pyart_hc=True,
    verbose=True,
)
```

## Checking Results

Always check the results:

```python
# Check processing results
print(f"Total volumes: {len(results)}")
print(f"Successful: {sum(1 for r in results if r['success'])}")
print(f"Failed: {sum(1 for r in results if not r['success'])}")

# Print errors for failed volumes
for r in results:
    if not r['success']:
        print(f"Failed at {r['time']}: {r['error']}")

# Only try to load if some succeeded
if any(r['success'] for r in results):
    # Get time range of successful processing
    successful_times = [r['time'] for r in results if r['success']]
    first_time = min(successful_times)

    dt = db.load_datatree(
        radar="MLA",
        start_time=first_time,
        end_time=first_time + pd.Timedelta(minutes=5)
    )

    raddb.plot_ppi(dt, sweep=1, variable="DBZH")
```

## Common Error Messages

### "No files found for {radar}"
- Check `raw_data_dir` path is correct
- Verify radar files exist: `ls /ltenas8/data/RADAR/MCH_LTE/MLA/2021/08/28/`
- Check radar name is correct ("MLA", not "A")

### "No data found for {radar} between {start} and {end}"
- Processing failed or no data was saved
- Check the `results` from `process_and_store()` for errors
- Verify output directory has parquet files

### "LUT not found"
- Run `db.generate_lut()` first before processing
- Or LUT generation happened but in wrong directory

### "Could not load visibility/HZT"
- Files missing - either provide correct paths or disable these features
- Use `hzt_enabled=False` or `static_vis_dir=None`

## Directory Structure Check

Verify your data is organized correctly:

```bash
# Raw data structure
/ltenas8/data/RADAR/
└── MCH_LTE/
    └── MLA/
        └── 2021/
            └── 08/
                └── 28/
                    ├── MLA2108628100000U.001
                    ├── MLA2108628100000U.002
                    └── ...

# Output structure (after processing)
/ltenas8/users/giacobbi/raddb/
└── MLA/
    ├── LUT/
    │   ├── MLA_LUT.parquet
    │   └── MLA_info.yaml
    └── 2021/
        └── 08/
            └── 28/
                ├── MLA_20210828_100000_LUT.parquet
                ├── MLA_20210828_100000_POLAR.parquet
                └── ...
```

## Quick Diagnostic Script

```python
import raddb
import os

# Configuration
base_path = "/home/erik_poschivo/Desktop/LTE_project/ltenas8/users/giacobbi/raddb"
raw_data_dir = "/ltenas8/data/RADAR"
radar = "MLA"
test_date = "2021-08-28"

# Check raw data exists
raw_path = f"{raw_data_dir}/MCH_LTE/{radar}/{test_date[:4]}/{test_date[5:7]}/{test_date[8:10]}"
print(f"Raw data path: {raw_path}")
print(f"Exists: {os.path.exists(raw_path)}")
if os.path.exists(raw_path):
    files = [f for f in os.listdir(raw_path) if f.startswith(radar[:2])]
    print(f"Found {len(files)} files")
    print(f"First file: {files[0] if files else 'None'}")

# Check processed data
processed_path = f"{base_path}/{radar}/{test_date[:4]}/{test_date[5:7]}/{test_date[8:10]}"
print(f"\nProcessed data path: {processed_path}")
print(f"Exists: {os.path.exists(processed_path)}")
if os.path.exists(processed_path):
    parquet_files = [f for f in os.listdir(processed_path) if f.endswith('.parquet')]
    print(f"Found {len(parquet_files)} parquet files")
    for pf in parquet_files[:5]:
        print(f"  - {pf}")

# Check LUT
lut_path = f"{base_path}/{radar}/LUT/{radar}_LUT.parquet"
print(f"\nLUT path: {lut_path}")
print(f"Exists: {os.path.exists(lut_path)}")
```

## Next Steps

1. Run the diagnostic script to verify your directory structure
2. Start with minimal processing (Step 1 above)
3. Check results and gradually enable more features
4. If errors persist, share the full error traceback

Remember: Start simple, add complexity gradually!
