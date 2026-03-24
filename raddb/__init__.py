"""
RadDB Package — PyART-based radar data processing and storage.
"""
from __future__ import annotations
import contextlib
import os
from importlib.metadata import PackageNotFoundError, version

# High-level API
from raddb.api import RadDB

# Helper functions
from raddb.helper import (
    read_parquet_dataset,
    check_dataframe,
    list_sweep_names,
    normalize_radar_name,
)

# I/O functions
from raddb.io_core import (
    metranet_to_datatree,
    volume_to_datatree,
    pyart_to_xradar_dataset,
    pyart_volume_to_datatree,
    datatree_to_dataset,
    datatree_to_dataframe,
    datatree_to_parquet,
    labels_to_dataframe,
    join_labels_with_lut,
    reconstruct_sweep_dataset,
    reconstruct_datatree,
)

# LUT functions
from raddb.lut import (
    compute_gate_xyz,
    generate_gate_id,
    generate_radar_lut,
    load_radar_lut,
    load_radar_info,
    get_full_sweep_index,
)

# Pipeline functions
from raddb.pipeline import (
    process_metranet_volume,
    archive_volume_to_parquet,
    process_and_archive_metranet,
    archive_metranet_to_parquet,  # backwards compatibility
)

# Radar processing functions
from raddb.radar_processing import (
    load_metranet_sweep,
    hzt_hourly_to_5min,
    read_static_visibility,
    read_qpegrid_to_rad,
    correct_gate_cartesian_coordinates,
    add_visibility,
    correct_reflectivity_for_visibility,
    compute_kdp,
    correct_attenuation,
    add_hydroclass_from_file,
    compute_hydroclass_semisupervised,
    add_hzt_data,
    add_height_over_iso0,
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

__all__ = [
    # High-level API
    "RadDB",
    # Helper functions
    "read_parquet_dataset",
    "check_dataframe",
    "list_sweep_names",
    "normalize_radar_name",
    # I/O functions
    "metranet_to_datatree",
    "volume_to_datatree",
    "pyart_to_xradar_dataset",
    "pyart_volume_to_datatree",
    "datatree_to_dataset",
    "datatree_to_dataframe",
    "datatree_to_parquet",
    "labels_to_dataframe",
    "join_labels_with_lut",
    "reconstruct_sweep_dataset",
    "reconstruct_datatree",
    # LUT functions
    "compute_gate_xyz",
    "generate_gate_id",
    "generate_radar_lut",
    "load_radar_lut",
    "load_radar_info",
    "get_full_sweep_index",
    # Pipeline functions
    "process_metranet_volume",
    "archive_volume_to_parquet",
    "process_and_archive_metranet",
    "archive_metranet_to_parquet",
    # Radar processing functions
    "load_metranet_sweep",
    "hzt_hourly_to_5min",
    "read_static_visibility",
    "read_qpegrid_to_rad",
    "correct_gate_cartesian_coordinates",
    "add_visibility",
    "correct_reflectivity_for_visibility",
    "compute_kdp",
    "correct_attenuation",
    "add_hydroclass_from_file",
    "compute_hydroclass_semisupervised",
    "add_hzt_data",
    "add_height_over_iso0",
    # Plotting functions
    "plot_ppi",
    "plot_rhi",
    "plot_cappi",
    "plot_volume_panel",
    "plot_classified_ppi",
    "plot_classified_rhi",
]

_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("raddb")
