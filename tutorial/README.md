# RadDB tutorials

Five notebooks that walk through the whole workflow, in order. Each one is
self-contained.

| #   | notebook                                                | covers                                                                                              |
| --- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 1   | [Archiving](01_archiving.ipynb)                         | the storage model, the CRS contract, `archive()`, what lands on disk                                |
| 2   | [Opening and filtering](02_opening_and_filtering.ipynb) | `open()`, `filter()`, `sel()`, computed columns, converters                                         |
| 3   | [Areas of interest](03_area_of_interest.ipynb)          | bbox / point / polygon crops, cross-sections, the interactive map                                   |
| 4   | [Plots](04_plots.ipynb)                                 | PPI, RHI, CAPPI, vertical cross-section                                                             |
| 5   | [Demo pipeline](05_demo_pipeline.ipynb)                 | the whole pipeline on data it downloads itself — NEXRAD and FMI volumes, archived, plotted, cropped |

## Running them yourself

RadDB is network-agnostic — any xarray `DataTree` with the standard
[xradar](https://docs.openradarscience.org/projects/xradar/) layout works. The
notebooks use Finnish (FMI) and US (NEXRAD) volumes stored as Zarr — both are
open data. Point them at your own data by editing the configuration cell at the
top of each notebook:

```python
FMI_DIR = Path("/path/to/FMI_datatree_zarr")  # radars FANJ, FKOR, FKUO
NEXRAD_DIR = Path("/path/to/NEXRAD_datatree_zarr")  # radars KTLX, KMLB, KLOT
ARCHIVE_DIR = Path("/tmp/raddb_tutorial_archive")  # where to write
```

Then run notebook 1 first — it creates the archive the others read.

Notebook 5 needs no local data at all: it downloads two public volumes (US,
Finland) over HTTP and adds them to the same archive.
