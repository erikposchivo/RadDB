# RadDB - Radar Database Package

A comprehensive Python package for processing, storing, and analyzing Swiss weather radar data (METRANET format) using PyART and xradar.

## Overview

RadDB provides a complete pipeline for:
1. **Reading** METRANET radar files (dual-polarimetric data)
2. **Processing** with PyART: visibility correction, KDP computation, attenuation correction
3. **Enriching** with hydrometeor classification (operational & PyART-based) and isotherm height (HZT)
4. **Converting** PyART radar objects to xradar DataTrees
5. **Storing** data efficiently in parquet format with gate_id indexing
6. **Loading** and visualizing processed data (PPI, RHI, CAPPI plots)

## Key Features

### 🚀 Complete Processing Pipeline
- Visibility correction using static LUTs
- KDP (specific differential phase) computation
- Attenuation correction (ZPHI algorithm)
- Hydrometeor classification (operational MCH & PyART semi-supervised)
- Isotherm height (HZT) integration with 5-minute interpolation

### 📊 Efficient Storage
- **LUT (Look-Up Table)**: Static geographical information (lat, lon, alt, x, y, z) stored separately
- **POLAR Data**: Dynamic dual-polarization values linked via `gate_id`
- Parquet format for fast I/O and compression
- Directory structure: `base_path/radar/year/month/day/`

### 🎨 Visualization
- PPI (Plan Position Indicator)
- RHI (Range Height Indicator)
- CAPPI (Constant Altitude PPI)
- Volume panels
- Classified hydrometeor displays

### 🔧 Flexible API
- **High-level API**: Simple `RadDB` class for common workflows
- **Low-level API**: Fine-grained control over processing steps
- **PyART integration**: Direct access to radar objects

## Installation

```bash
# Install in development mode
cd /path/to/RadDB
pip install -e .
```

### Dependencies
- pyart
- pyart-mch
- xradar
- xarray
- pandas
- numpy
- pyyaml
- radar-api

## Quick Start

### 1. High-Level API (Recommended)

```python
import raddb

# Initialize
db = raddb.RadDB(
    base_path="/ltenas8/users/giacobbi/raddb",  # Output directory for processed data
    raw_data_dir="/ltenas8/data/RADAR",          # Input directory for raw METRANET files
    network="MCH_LTE",
)

# Generate LUT (once per radar)
# db.generate_lut(radar="MLA", sample_volume_filepaths=sweep_files)

# Process and store data
results = db.process_and_store(
    radar="MLA",
    start_time="2021-08-28 06:00",
    end_time="2021-08-28 18:00",
    hzt_enabled=True,
    hym_enabled=True,
    compute_pyart_hc=True,
)

# Load as DataFrame
df = db.load_parquet_data(
    radar="MLA",
    start_time="2021-08-28 12:00",
    end_time="2021-08-28 13:00",
)

# Load as DataTree for plotting
dt = db.load_datatree(
    radar="MLA",
    start_time="2021-08-28 12:00",
    end_time="2021-08-28 12:05",
)

# Visualize
raddb.plot_ppi(dt, sweep=1, variable="DBZH")
raddb.plot_rhi(dt, azimuth=90, variable="DBZH")
raddb.plot_classified_ppi(dt, sweep=1)
```

## Architecture

### Processing Pipeline

```
METRANET Files (sweeps 1-20)
    ↓
PyART Radar Object Processing
    ├─ Visibility correction
    ├─ KDP computation
    ├─ Attenuation correction
    ├─ HZT integration (5-min interpolation)
    ├─ Operational HC (from files)
    └─ PyART HC (semi-supervised)
    ↓
xradar DataTree
    ↓
DataFrame with gate_id
    ├─ LUT: Static geo info
    └─ POLAR: Dynamic values
    ↓
Parquet Storage
```

### Directory Structure

```
/ltenas8/users/giacobbi/raddb/
├── MLA/
│   ├── LUT/
│   │   ├── MLA_LUT.parquet        # Static geographical data
│   │   └── MLA_info.yaml          # Radar metadata
│   ├── 2021/
│   │   └── 08/
│   │       └── 28/
│   │           ├── MLA_20210828_120000_LUT.parquet
│   │           ├── MLA_20210828_120000_POLAR.parquet
│   │           ├── MLA_20210828_120500_LUT.parquet
│   │           └── MLA_20210828_120500_POLAR.parquet
├── MLD/
│   └── ...
└── ...
```

### Data Features

#### LUT (Look-Up Table)
```
gate_id, sweep, azimuth, range, latitude, longitude, altitude, x, y, z, elevation_angle
```

#### POLAR Data
```
gate_id, time, DBZH, ZDR, RHOHV, PHIDP, HC_MCH, HC_PYART, HZT
```

Where:
- `gate_id`: Unique identifier (e.g., "MLA_s01_a0.5_r001000")
- `DBZH`: Reflectivity (dBZ)
- `ZDR`: Differential reflectivity (dB)
- `RHOHV`: Cross-correlation coefficient
- `PHIDP`: Differential phase (degrees)
- `HC_MCH`: Operational hydrometeor classification (MeteoSwiss)
- `HC_PYART`: PyART semi-supervised hydrometeor classification
- `HZT`: Height of 0°C isotherm (m)

## Module Structure

```
raddb/
├── __init__.py           # Public API exports
├── api.py                # High-level RadDB class
├── pipeline.py           # Processing orchestration
├── radar_processing.py   # PyART processing functions
├── io_core.py            # Data conversion functions
├── lut.py                # LUT generation and management
├── plot.py               # Visualization functions
└── helper.py             # Utility functions
```

## Advanced Usage

### Working with PyART Objects

```python
import raddb

# Load and process a single sweep
rad_obj, x, y, z = raddb.load_metranet_sweep(
    radar_fpath="/path/to/MLA2108628120000U.001",
    hydroclassif_fpath="/path/to/YM_file",  # optional
    hzt_cartesian=hzt_array,                # optional
    visibility=vis_array,                   # optional
    compute_pyart_hc=True,
)

# Access PyART fields
print(rad_obj.fields.keys())
dbz = rad_obj.get_field(0, 'reflectivity')

# Convert to xradar dataset
ds = raddb.pyart_to_xradar_dataset(rad_obj)
```

### Custom Processing

```python
import raddb

# Process volume with custom parameters
dt = raddb.process_metranet_volume(
    sweep_filepaths=sweep_files,
    network="MCH_LTE",
    radar="MLA",
    volume_time=datetime(2021, 8, 28, 12, 0),
    static_vis_dir="/path/to/static_vis",
    qpegrid_to_rad_dir="/path/to/qpegrid",
    hzt_enabled=True,
    hym_enabled=True,
    compute_pyart_hc=True,
)

# Archive with custom threshold
lut_path, polar_path = raddb.archive_volume_to_parquet(
    dt=dt,
    radar="MLA",
    base_output_path="/output/path",
    dbzh_threshold=10.0,  # Only save gates with DBZH > 10
)
```

## Configuration

Default paths can be configured when initializing `RadDB`:

```python
db = raddb.RadDB(
    base_path="/custom/output/path",           # Where to save processed parquet files
    raw_data_dir="/path/to/raw/METRANET/data", # Where raw radar files are located
    network="MCH_LTE",
    static_vis_dir="/path/to/visibility/luts",
    qpegrid_to_rad_dir="/path/to/qpe/luts",
)
```

**Important**: The `raw_data_dir` parameter is required if you don't have a `radar_api` configuration file set up. This directory should contain your raw METRANET radar files organized by the radar_api convention (network/radar/year/month/day/).

## Examples

See `examples/example_usage.py` for comprehensive examples covering:
- High-level API workflow
- Low-level processing control
- Direct PyART object manipulation
- Custom visualization

## Hydrometeor Classification

### Classes
- **NC**: Not classified
- **AG**: Aggregates
- **CR**: Ice crystals
- **LR**: Light rain
- **RP**: Rimed particles
- **RN**: Rain
- **VI**: Vertically oriented ice
- **WS**: Wet snow
- **MH**: Melting hail
- **IH/HDG**: Dry hail / high density graupel

### Two Types
1. **HC_MCH**: Operational classification from MeteoSwiss (from HYM files)
2. **HC_PYART**: Semi-supervised classification using PyART's algorithm

## Performance Tips

1. **Parallel Processing**: Use `max_workers > 1` for batch processing
2. **Memory**: Process volumes sequentially for large time ranges
3. **Storage**: Use `dbzh_threshold` to filter low-reflectivity gates
4. **Plotting**: Use `rasterized=True` in plots for large datasets

## References

- PyART: [https://arm-doe.github.io/pyart/](https://arm-doe.github.io/pyart/)
- xradar: [https://docs.openradarscience.org/projects/xradar/](https://docs.openradarscience.org/projects/xradar/)
- RainForest: [https://github.com/MeteoSwiss/rainforest](https://github.com/MeteoSwiss/rainforest)

## License

See LICENSE file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
