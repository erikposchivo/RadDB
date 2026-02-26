<div align="center">

# Welcome to RadDB

![Radars currently accessible through RadDB](/docs/source/static/raddb_coverage.png)

|                   |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Deployment        | [![PyPI](https://badge.fury.io/py/raddb.svg?style=flat)](https://pypi.org/project/raddb/) [![Conda](https://img.shields.io/conda/vn/conda-forge/radar-api.svg?logo=conda-forge&logoColor=white&style=flat)](https://anaconda.org/conda-forge/radar-api)                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Activity          | [![PyPI Downloads](https://img.shields.io/pypi/dm/raddb.svg?label=PyPI%20downloads&style=flat)](https://pypi.org/project/raddb/) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/radar-api.svg?label=Conda%20downloads&style=flat)](https://anaconda.org/conda-forge/radar-api)                                                                                                                                                                                                                                                                                                                                                                                       |
| Python Versions   | [![Python Versions](https://img.shields.io/badge/Python-3.10%20%203.11%20%203.12%20%203.13-blue?style=flat)](https://www.python.org/downloads/)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Supported Systems | [![Linux](https://img.shields.io/github/actions/workflow/status/erikposchivo/raddb/.github/workflows/tests.yaml?label=Linux&style=flat)](https://github.com/erikposchivo/raddb/actions/workflows/tests.yaml) [![macOS](https://img.shields.io/github/actions/workflow/status/erikposchivo/raddb/.github/workflows/tests.yaml?label=macOS&style=flat)](https://github.com/erikposchivo/raddb/actions/workflows/tests.yaml) [![Windows](https://img.shields.io/github/actions/workflow/status/erikposchivo/raddb/.github/workflows/tests_windows.yaml?label=Windows&style=flat)](https://github.com/erikposchivo/raddb/actions/workflows/tests_windows.yaml)                                                |
| Project Status    | [![Project Status](https://www.repostatus.org/badges/latest/active.svg?style=flat)](https://www.repostatus.org/#active)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Build Status      | [![Tests](https://github.com/erikposchivo/raddb/actions/workflows/tests.yaml/badge.svg?style=flat)](https://github.com/erikposchivo/raddb/actions/workflows/tests.yaml) [![Lint](https://github.com/erikposchivo/raddb/actions/workflows/lint.yaml/badge.svg?style=flat)](https://github.com/erikposchivo/raddb/actions/workflows/lint.yaml) [![Docs](https://readthedocs.org/projects/raddb/badge/?version=latest&style=flat)](https://radar-api.readthedocs.io/en/latest/)                                                                                                                                                                                                                      |
| Linting           | [![Black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat)](https://github.com/psf/black) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json&style=flat)](https://github.com/astral-sh/ruff) [![Codespell](https://img.shields.io/badge/Codespell-enabled-brightgreen?style=flat)](https://github.com/codespell-project/codespell)                                                                                                                                                                                                                                                                 |
| Code Coverage     | [![Coveralls](https://coveralls.io/repos/github/erikposchivo/raddb/badge.svg?branch=main&style=flat)](https://coveralls.io/github/erikposchivo/raddb?branch=main) [![Codecov](https://codecov.io/gh/erikposchivo/raddb/branch/main/graph/badge.svg?token=G7IESZ02CW?style=flat)](https://codecov.io/gh/erikposchivo/raddb)                                                                                                                                                                                                                                                                                                                                                                            |
| Code Quality      | [![Codefactor](https://www.codefactor.io/repository/github/erikposchivo/raddb/badge?style=flat)](https://www.codefactor.io/repository/github/erikposchivo/raddb) [![Codebeat](https://codebeat.co/badges/57498d71-f042-473f-bb8e-9b45e50572d8?style=flat)](https://codebeat.co/projects/github-com-erikposchivo-raddb-main) [![Codacy](https://app.codacy.com/project/badge/Grade/bee842cb10004ad8bb9288256f2fc8af?style=flat)](https://app.codacy.com/gh/erikposchivo/raddb/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade) [![Codescene](https://codescene.io/projects/63299/status-badges/average-code-health?style=flat)](https://codescene.io/projects/63299) |
| License           | [![License](https://img.shields.io/github/license/erikposchivo/raddb?style=flat)](https://github.com/erikposchivo/raddb/blob/main/LICENSE)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| Community         | [![Discourse](https://img.shields.io/badge/Slack-raddb-green.svg?logo=slack&style=flat)](https://openradar.discourse.group/) [![GitHub Discussions](https://img.shields.io/badge/GitHub-Discussions-green?logo=github&style=flat)](https://github.com/erikposchivo/raddb/discussions)                                                                                                                                                                                                                                                                                                                                                                                                       |
| Citation          | [![DOI](https://zenodo.org/badge/922589509.svg?style=flat)](https://doi.org/10.5281/zenodo.14743651)                                                                                                                                                                                                                                                                                                                                                                                    </div>                                                                                                                                                                                                |

[**Documentation: https://radar-api.readthedocs.io**](https://radar-api.readthedocs.io/)

<div align="left">

## 🚀 Quick start

RadDB provides an easy-to-use python interface to find, download and
read weather radar data from several meteorological services.

RadDB currently provides data access to the following
radar networks: `NEXRAD`, `IDEAM` and `FMI`.

The list of available radars can be retrieved using:

```python
import raddb

raddb.available_networks()
raddb.available_radars()
raddb.available_radars(network="NEXRAD")
```

Before starting using RadDB, we highly suggest to save into a configuration file
the directory on your local disk where to save the radar data of interest.

To facilitate the creation of the RadDB configuration file, you can adapt and execute the following script:

```python
import raddb

base_dir = (
    "<path/to/directory/RADAR"  # path to the directory where to download the data
)
raddb.define_configs(base_dir=base_dir)

# You can check that the config file has been correctly created with:
configs = raddb.read_configs()
print(configs)
```

______________________________________________________________________

### 📥 Download radar data

You can start to download radar data editing the following code example:

```python
import raddb

start_time = "2021-09-07 17:00:00"
end_time = "2021-09-07 17:30:00"

radar = "KMKX"
network = "NEXRAD"

filepaths = raddb.download_files(
    network=network,
    radar=radar,
    start_time=start_time,
    end_time=end_time,
)
```

______________________________________________________________________

### 💫 Open radar files into xarray or pyart

RadDB allows to read directly radar data from the cloud without the
need to previously download and save the files on your disk.

RadDB make use of pyart and xradar readers to open the files into either
an xarray object or pyart radar object.

```python
import raddb
import pyart

# Search for files on cloud bucket
filepaths = raddb.find_files(
    network=network,
    radar=radar,
    start_time=start_time,
    end_time=end_time,
    protocol="s3",
)
print(filepaths)

# Define the file to open
filepath = filepaths[0]

# Open all sweeps of a radar volume into a xradar datatree
dt = raddb.open_datatree(filepath, network=network)

# Extract the radar sweep of interest
ds = dt["sweep_0"].to_dataset()

# Open directly a single radar sweep into a xradar dataset
ds = raddb.open_dataset(filepath, network=network, sweep="sweep_0")

# Open all sweeps of a radar volume into a pyart radar object
radar_obj = raddb.open_pyart(filepath, network=network)

# Display the data with pyart
display = pyart.graph.RadarDisplay(radar_obj)
display.plot("reflectivity", cmap="pyart_ChaseSpectral", vmin=-20, vmax=70)
display.set_limits((-150, 150), (-150, 150))
```

______________________________________________________________________

## 📖 Documentation

To discover RadDB utilities and functionalities,
please read the software documentation available at [https://radar-api.readthedocs.io/en/latest/](https://radar-api.readthedocs.io/en/latest/).

All RadDB tutorials are available as Jupyter Notebooks in the [`tutorial`](https://github.com/erikposchivo/raddb/tree/main/tutorials) directory.

______________________________________________________________________

## 🛠️ Installation

### conda

RadDB can be installed via [conda][conda_link] on Linux, Mac, and Windows.
Install the package by typing the following command in the terminal:

```bash
conda install radar-api
```

In case conda-forge is not set up for your system yet, see the easy to follow instructions on [conda-forge][conda_forge_link].

### pip

RadDB can be installed also via [pip][pip_link] on Linux, Mac, and Windows.
On Windows you can install [WinPython][winpy_link] to get Python and pip running.
Then, install the RadDB package by typing the following command in the terminal:

```bash
pip install radar-api
```

To install the latest development version via pip, see the [documentation][dev_install_link].

## 💭 Feedback and Contributing Guidelines

If you aim to contribute your data or discuss the future development of RadDB,
we suggest to join the [**Open Radar Science Discourse Group**](https://openradar.discourse.group/).

Feel free to also open a [GitHub Issue](https://github.com/erikposchivo/raddb/issues) or a [GitHub Discussion](https://github.com/erikposchivo/raddb/discussions) specific to your questions or ideas.

## Citation

If you are using RadDB in your publication please cite our Zenodo repository:

> Ghiggi Gionata. erikposchivo/raddb. Zenodo. [![<https://doi.org/10.5281/zenodo.14743651>](https://zenodo.org/badge/922589509.svg?style=flat)](https://doi.org/10.5281/zenodo.14743651)

If you want to cite a specific software version, have a look at the [Zenodo site](https://doi.org/10.5281/zenodo.14743651).

## License

The content of this repository is released under the terms of the [MIT license](LICENSE).

</div>

[conda_forge_link]: https://github.com/conda-forge/radar-api-feedstock#installing-radar-api
[conda_link]: https://docs.conda.io/en/latest/miniconda.html
[dev_install_link]: https://radar-api.readthedocs.io/en/latest/02_installation.html#installation-for-contributors
[pip_link]: https://pypi.org/project/radar-api
[winpy_link]: https://winpython.github.io/
