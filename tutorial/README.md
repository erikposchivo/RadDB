# RadDB tutorials

Five notebooks that walk through the whole workflow, in order. Each one is
self-contained — if you jump straight to number 3, it builds the archive it needs.

| # | notebook | covers |
|---|---|---|
| 1 | [Archiving](01_archiving.ipynb) | the storage model, the CRS contract, `archive()`, what lands on disk |
| 2 | [Opening and filtering](02_opening_and_filtering.ipynb) | `open()`, `filter()`, `sel()`, computed columns, converters |
| 3 | [Areas of interest](03_area_of_interest.ipynb) | bbox / point / polygon crops, cross-sections, the interactive map |
| 4 | [Plots](04_plots.ipynb) | PPI, RHI, CAPPI, vertical cross-section |
| 5 | [Demo pipeline](05_demo_pipeline.ipynb) | the whole pipeline on data it downloads itself — NEXRAD, FMI and IDEAM volumes, archived, plotted, cut |

The notebooks are stored **with their output**, so you can read them on GitHub
without running anything.

## Running them yourself

RadDB is network-agnostic — any xarray `DataTree` with the standard
[xradar](https://docs.openradarscience.org/projects/xradar/) layout works. The
notebooks use MeteoSwiss and NEXRAD volumes stored as Zarr. Point them at your own
data with environment variables, or edit the configuration cell at the top of each
notebook:

```bash
export RADDB_DATATREE_DIR=/path/to/MCH_datatree      # radars L, W
export RADDB_NEXRAD_DIR=/path/to/NEXRAD_datatree     # radar KTLX
export RADDB_TUTORIAL_ARCHIVE=/tmp/raddb_tutorial_archive   # where to write

jupyter lab
```

Then run notebook 1 first — it creates the archive the others read.

Notebook 5 needs no local data at all: it downloads three public volumes (US,
Finland, Colombia) over HTTP and adds them to the same archive.

Needed beyond the core install: `jupyter`, and `ipyleaflet` + `ipywidgets` for the
interactive map in notebook 3 (`pip install raddb[viz]`).
