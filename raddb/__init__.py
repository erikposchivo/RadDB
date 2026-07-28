"""
RadDB Package — Generic radar data archiving and reconstruction.

RadDB archives xarray DataTree volumes as Parquet files with an efficient
LUT-based layout.  It is network-agnostic: any DataTree with the standard
xradar coordinate layout can be archived and reconstructed.

MCH/METRANET-specific ingestion code lives in the private ``raddb.mch``
subpackage (gitignored in the public repository; never imported here).
"""
from __future__ import annotations

import contextlib
import os
from importlib.metadata import PackageNotFoundError, version

# PROJ data directory — MUST stay the first raddb import: it repairs a
# PROJ_DATA/PROJ_LIB inherited from another environment before pyproj is
# imported (by geopandas, cartopy, raddb.lut, ...).
from raddb._proj import PROJ_DATA

# High-level interface
from raddb.main import RadDB

# Helper functions
from raddb.helper import (
    read_parquet_files,
    check_dataframe,
    list_sweep_names,
    normalize_radar_name,
    StageTimer,
)

# I/O functions
from raddb.io_core import (
    datatree_to_dataset,
    datatree_to_dataframe,
    datatree_to_parquet,
    open_any_datatree,
    parquet_to_dataframe,
    parquet_to_datatree,
    scan_polar_parquet,
    dataframe_to_datatree,
    labels_to_dataframe,
    join_labels_with_lut,
    reconstruct_sweep_dataset,
    reconstruct_datatree,
    add_feature_to_df,
    add_feature_to_dt,
)

# Discovery functions
from raddb.discovery import find_datatree_files

# LUT functions
from raddb.lut import (
    RADAR_TO_IDX,
    antenna_vectors_to_cartesian,
    cartesian_to_geographic,
    compute_gate_xyz,
    generate_gate_id,
    generate_lut_from_datatree,
    load_radar_lut,
    load_radar_info,
    get_full_sweep_index,
    add_lut_projection,
)

# Filtering & archiving functions
from raddb.helper import FILTER_LOGICS, filter_df, filter_dt
from raddb.io_core import (
    archive_volume,
    archive_multiple_volumes,
    archive_volumes_multi_radar,
)

# Plotting functions
from raddb.viz.plot import (
    plot_ppi,
    plot_rhi,
    plot_latent_scatter,
)

# Profiling helpers
from raddb.viz.profiling import (
    plot_stage_totals,
    plot_volume_timing,
    plot_sweep_timing,
    plot_profiling_dashboard,
)

__all__ = [
    # High-level API
    "RadDB",
    # Environment
    "PROJ_DATA",
    # Helper functions
    "read_parquet_files",
    "check_dataframe",
    "list_sweep_names",
    "normalize_radar_name",
    "StageTimer",
    # I/O functions
    "datatree_to_dataset",
    "datatree_to_dataframe",
    "datatree_to_parquet",
    "open_any_datatree",
    "parquet_to_dataframe",
    "parquet_to_datatree",
    "scan_polar_parquet",
    "dataframe_to_datatree",
    "labels_to_dataframe",
    "join_labels_with_lut",
    "reconstruct_sweep_dataset",
    "reconstruct_datatree",
    "add_feature_to_df",
    "add_feature_to_dt",
    # Discovery functions
    "find_datatree_files",
    # LUT functions
    "RADAR_TO_IDX",
    "antenna_vectors_to_cartesian",
    "cartesian_to_geographic",
    "compute_gate_xyz",
    "generate_gate_id",
    "generate_lut_from_datatree",
    "load_radar_lut",
    "load_radar_info",
    "get_full_sweep_index",
    "add_lut_projection",
    # Pipeline functions
    "FILTER_LOGICS",
    "filter_df",
    "filter_dt",
    "archive_volume",
    "archive_multiple_volumes",
    "archive_volumes_multi_radar",
    # Plotting functions
    "plot_ppi",
    "plot_rhi",
    "plot_latent_scatter",
    # Profiling helpers
    "plot_stage_totals",
    "plot_volume_timing",
    "plot_sweep_timing",
    "plot_profiling_dashboard",
]

_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("raddb")
