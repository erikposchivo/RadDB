"""
RadDB Package — Generic radar data archiving and reconstruction.

RadDB archives xarray DataTree volumes as Parquet files with an efficient
LUT-based layout.  It is network-agnostic: any DataTree with the standard
xradar coordinate layout can be archived and reconstructed.

MCH/METRANET-specific ingestion code lives in ``mch_pipeline.py``
(outside the package, excluded via .gitignore).
"""
from __future__ import annotations

import contextlib
import os
from importlib.metadata import PackageNotFoundError, version

# High-level API
from raddb.api import RadDB

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
    parquet_to_dataframe,
    parquet_to_datatree,
    labels_to_dataframe,
    join_labels_with_lut,
    reconstruct_sweep_dataset,
    reconstruct_datatree,
    add_feature_to_df,
    add_feature_to_dt,
)

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

# Pipeline functions
from raddb.pipeline import (
    FILTER_LOGICS,
    filter_df,
    filter_dt,
    archive_volume,
    archive_multiple_volumes,
    archive_volumes_multi_radar,
)

# Plotting functions
from raddb.plot import (
    plot_ppi,
    plot_rhi,
    plot_cappi,
    plot_volume_panel,
    plot_classified_ppi,
    plot_classified_rhi,
)

# Profiling helpers
from raddb.helper import (
    plot_stage_totals,
    plot_volume_timing,
    plot_sweep_timing,
    plot_profiling_dashboard,
)

__all__ = [
    # High-level API
    "RadDB",
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
    "parquet_to_dataframe",
    "parquet_to_datatree",
    "labels_to_dataframe",
    "join_labels_with_lut",
    "reconstruct_sweep_dataset",
    "reconstruct_datatree",
    "add_feature_to_df",
    "add_feature_to_dt",
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
    "plot_cappi",
    "plot_volume_panel",
    "plot_classified_ppi",
    "plot_classified_rhi",
    # Profiling helpers
    "plot_stage_totals",
    "plot_volume_timing",
    "plot_sweep_timing",
    "plot_profiling_dashboard",
]

_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("raddb")
