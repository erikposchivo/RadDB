# RadDB — generic radar data archiving & analysis

RadDB archives xarray **DataTree** radar volumes as compact Parquet files and
gives you a small, fluent object API to load, filter, crop, cross-section and
plot them.  It is **network-agnostic**: any DataTree with the standard
[xradar](https://docs.openradarscience.org/projects/xradar/) coordinate layout
(MeteoSwiss, NEXRAD, …) can be archived and analysed — no pyart required.

MeteoSwiss/METRANET-specific ingestion (pyart + radar_api) lives in the private
`raddb.mch` subpackage and is kept out of the generic core.

## Storage model

A radar is stored as a **static LUT** (per-gate geometry, generated once) plus one
**POL parquet per volume** (the dynamic moments), linked by an integer `gate_id`:

```
{archive_dir}/{radar}/LUT/{radar}_LUT.parquet          # gate_id, lat/lon/alt, x_<epsg>/y_<epsg>, sweep, …
{archive_dir}/{radar}/{YYYY}/{MM}/{DD}/{radar}_{YYYYMMDD}_{HHMMSS}_POL.parquet
```

No-echo gates are dropped at archive time (default `DBZH > 0`), so the archive
stays small.

## Installation

```bash
pip install -e .            # core
pip install -e ".[viz]"     # + cartopy / pyproj / shapely for maps
```

Core runtime deps: `numpy, pandas, polars, geopandas, xarray, pyarrow, dask,
fsspec, s3fs, matplotlib`.

## Quick start

`RadDB` is a single, dual-role class:

* **archive-bound** — `db = RadDB(archive_dir, crs=…)`; use it to `archive()` and `open()`.
* **data-carrying** — the object returned by `open()` (call it `rdf`).  It holds the
  data as a **polars** DataFrame (`rdf.data`) and exposes the query / convert /
  crop / cross-section / plot methods.  Each of those returns a **new** `RadDB`,
  so calls chain fluently.

```python
import raddb

db = raddb.RadDB(archive_dir="/data/raddb", crs=2056)   # 2056 = CH1903+/LV95

# --- archive -----------------------------------------------------------------
# From saved DataTree files on disk (.zarr / .nc); the LUT is auto-generated:
db.archive(datatree_dir="/data/MCH_datatree")           # radar inferred per file
# ...or archive in-memory DataTrees directly:
# db.archive(datatree=dt, radar="A")
# db.archive(datatree=[dt1, dt2], radar="A")
# db.archive(datatree={"A": [dt1], "D": [dt2]})         # multi-radar

# --- open --------------------------------------------------------------------
rdf = db.open(time_period=("2024-08-26", "2024-08-27"), radars="L")
print(rdf)                     # rich summary: gates, radars, time range, columns
len(rdf), rdf.columns(), rdf.radars()
rdf.start_time(), rdf.end_time()
rdf.extent()                   # [xmin, xmax, ymin, ymax] in `crs`
rdf.geographic_extent()        # [lon_min, lon_max, lat_min, lat_max]
rdf.crs(), rdf.geographic_crs()

# --- filter / convert --------------------------------------------------------
strong = rdf.filter({"var": "DBZH", "logic": ">", "threshold": 20})
strong = rdf.filter([{"var": "DBZH", "logic": ">", "threshold": 20},
                     {"var": "RHOHV", "logic": ">", "threshold": 0.9}])   # AND
pdf = rdf.to_pandas(with_geometry=True)     # pandas + gate coordinates
gdf = rdf.to_geopandas()                    # GeoDataFrame (with CRS)
dt  = rdf.to_datatree()                     # xarray DataTree (for plotting)

# --- area of interest --------------------------------------------------------
box   = rdf.crop_by_bbox(extent=[2.60e6, 2.62e6, 1.11e6, 1.13e6])   # or bounds=(xmin,ymin,xmax,ymax)
poly  = rdf.crop_by_polygone("catchment.geojson")
disc  = rdf.crop_around_point((2.61e6, 1.12e6), distance=20_000)    # metres
# rdf.interactive_crop()      # draw an AOI on a Jupyter map (needs ipyleaflet)

# --- cross-section -----------------------------------------------------------
cs = rdf.extract_cross_section(p1=(2.60e6, 1.12e6), p2=(2.63e6, 1.12e6))

# --- plot --------------------------------------------------------------------
rdf.plot_ppi(sweep=1, variable="DBZH", save="ppi.png")
cs.plot_cross_section(variable="DBZH", save="xsec.png")
rdf.plot_rhi(azimuth=270, variable="DBZH")

# fluent: open -> filter -> crop -> plot
rdf.filter({"var": "DBZH", "logic": ">", "threshold": 20}) \
   .crop_by_bbox(extent=rdf.extent()) \
   .plot_ppi(variable="DBZH", save="strong.png")
```

`filter` / `crs` argument shapes: filters are `{"var", "logic", "threshold"}`
dicts (`logic` ∈ `==,!=,>,>=,<,<=`); `crs` is an EPSG int (e.g. `2056`), a
CRS-coercible object, or `None`.

### What is on disk? (archive-bound)

```python
db.inventory()                      # radars, volume counts, time ranges, size
db.inventory(detailed=True)         # + LUT info, stored moments, day-by-day counts
db.inventory(datatree_dir="/data/MCH_datatree")   # DataTree files not archived yet
```

### LUT accessors (archive-bound)

```python
db.list_radars()                    # radars present in the archive
db.get_lut("L")                     # the static LUT (pandas)
db.get_radar_info("L")              # site location / sweep geometry
db.add_lut_projection("L", epsg=2056)
```

## Module structure

```
raddb/
├── __init__.py     # public API surface
├── main.py         # the RadDB class (this API)
├── io_core.py      # DataTree <-> DataFrame <-> Parquet + archive backends
├── lut.py          # LUT generation / geo projection
├── aoi.py          # AOI / crop / cross-section geometry
├── discovery.py    # find_datatree_files + filename-time parsing
├── helper.py       # filters, radar-name normalisation, timers
├── viz/            # plot.py (PPI/RHI/cross-section), interactive.py
└── mch/            # private MeteoSwiss/METRANET ingestion (pyart) — not in the public wheel
```

Sample-data scripts live under `scripts/` (`make_sample_mch_datatrees.py`,
`download_nexrad_datatree.py`).

## Notes

* **Projected coordinates / `crs`.** Generating a LUT with a projection (e.g.
  `crs=2056`) and the projected accessors (`extent`, `to_geopandas`) use `pyproj`,
  which needs the PROJ database.  A `PROJ_DATA` / `PROJ_LIB` inherited from another
  environment (a conda base env, a system PROJ) points at a proj.db of the wrong
  PROJ version and makes every projection fail with *"no database context
  specified"* — `import raddb` detects that and repoints PROJ at the running
  interpreter's own `share/proj` (see `raddb/_proj.py`; `raddb.PROJ_DATA` reports
  what it changed, `None` when nothing had to be).
* **Zarr / NetCDF.** Saved volumes are read back with `raddb.open_any_datatree`
  (engine auto-detected); either format works.

## License

See LICENSE (MIT).
