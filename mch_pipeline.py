"""
mch_pipeline.py
===============
MCH (MeteoSwiss) radar data ingestion pipeline.

This standalone module handles everything specific to the Swiss METRANET radar
network: reading raw files via ``radar_api`` / ``pyart``, applying visibility
corrections, KDP, attenuation, hydrometeor classification, and converting the
processed PyART radar objects into xarray DataTrees that RadDB can archive.

**This file is intentionally excluded from the RadDB package** (via .gitignore)
so that RadDB remains a generic, dependency-light library that works with any
xarray DataTree.  Users who need the MCH ingestion chain should keep this file
alongside their project and import from it directly.

Usage
-----
The recommended workflow is:

1. Use ``process_mch_volume()`` (single volume) or ``process_mch_volumes()``
   (sequential batch) to produce xarray DataTrees from raw METRANET files.
2. Pass the DataTrees to RadDB for archiving:
   ``db.archive_volume()`` or ``db.archive_multiple_volumes()``.

For end-to-end processing of a long time range, see ``main.py`` at the
package root — it fuses processing and archiving volume-by-volume with
a resumable checkpoint, avoiding the RAM blow-up of holding all
DataTrees in memory at once.

Dependencies (not required by RadDB itself)
-------------------------------------------
- pyart  (``pyart_mch`` or ``arm_pyart``)
- radar_api
- scipy
- xmltodict
"""
from __future__ import annotations

import concurrent.futures
import datetime
import functools
import glob
import logging
import os
import pickle
import time as _time
from collections import defaultdict
from contextlib import nullcontext as _nullctx
from pathlib import Path

import numpy as np
import pandas as pd
import pyart
import radar_api
import xarray as xr
import xmltodict
from pyart.aux_io import read_cartesian_metranet
from pyart.correct import calculate_attenuation_zphi, smooth_phidp_single_window
from pyart.retrieve import kdp_leastsquare_single_window
from scipy.spatial import KDTree

# RadDB imports (generic package)
import raddb
from raddb.helper import StageTimer, _vprint, normalize_radar_name
from raddb.lut import RADAR_TO_IDX

logger = logging.getLogger(__name__)

RADAR_LETTERS = ["A", "D", "L", "P", "W"]

# Field name mapping from PyART to xradar naming convention
FIELD_MAPPING = {
    "reflectivity": "DBZH",
    "reflectivity_visibilitycorr": "DBZH",
    "reflectivity_vv": "DBZV",
    "corrected_reflectivity": "DBZH_corrected",
    "corrected_reflectivity_vv": "DBZV_corrected",
    "differential_reflectivity": "ZDR",
    "corrected_differential_reflectivity": "ZDR_corrected",
    "uncorrected_cross_correlation_ratio": "RHOHV",
    "cross_correlation_ratio": "RHOHV",
    "uncorrected_differential_phase": "PHIDP",
    "differential_phase": "PHIDP",
    "propagation_differential_phase": "PHIDP_smoothed",
    "specific_differential_phase": "KDP",
    "hydrometeor_classification_mch": "HC_MCH",
    "radar_echo_classification": "HC_PYART",
    "iso0_height": "HZT",
    "height_over_iso0": "height_over_iso0",
    "temperature": "temperature",
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def check_radar_letter(radar_letter: str) -> bool:
    """Check if a given radar-identification letter is valid."""
    return radar_letter.upper() in RADAR_LETTERS


def read_status(status_file: str, verbose: bool = False) -> dict:
    """Read a radar XML status file."""
    try:
        status = xmltodict.parse(
            open(status_file, "r").read().replace("-P/", "-P_")
        )
        return status
    except Exception as e:
        if verbose:
            logger.warning(f"ERROR reading status file {status_file}: {e}")
        return {}


@functools.lru_cache(maxsize=None)
def read_qpegrid_to_rad(
    radar_letter: str, qpegrid_to_rad_dir: str, verbose: bool = False
) -> dict:
    """Read lookup table containing the Swiss grid (used for QPE).

    Result is cached in-process so repeated calls (e.g. one per sweep)
    only pay the pickle-load cost once per radar letter per worker.
    """
    if check_radar_letter(radar_letter):
        vis_fname = f"lut_qpegrid_to_rad{radar_letter.upper()}.p"
        vis_content = pickle.load(
            open(os.path.join(qpegrid_to_rad_dir, vis_fname), "rb")
        )
        return vis_content
    if verbose:
        logger.warning(
            f"ERROR: Impossible to read radar visibility, "
            f"radar letter {radar_letter} is not valid."
        )
    return {}


def read_static_visibility(
    radar_letter: str, static_vis_dir: str, verbose: bool = False
) -> dict:
    """Return the static visibility dictionary for a given radar."""
    if check_radar_letter(radar_letter):
        vis_fname = f"lut_visibility_rad{radar_letter.upper()}.p"
        vis_content = pickle.load(
            open(os.path.join(static_vis_dir, vis_fname), "rb")
        )
        return vis_content
    if verbose:
        logger.warning(
            f"ERROR: Impossible to read radar visibility, "
            f"radar letter {radar_letter} is not valid."
        )
    return {}


# ============================================================================
# HZT INTERPOLATION
# ============================================================================


def hzt_hourly_to_5min(filedict: dict, tsteps_min: int = 5) -> dict:
    """Interpolate hourly isothermal fields to 5-minute resolution."""
    if not filedict or len(filedict) < 2:
        logger.warning(
            f"Need at least 2 HZT files for interpolation, got {len(filedict)}"
        )
        return {}

    timestep_list = sorted(filedict.keys())
    tstamp_hzt0 = timestep_list[0]
    tstamp_hzt1 = timestep_list[1]

    hzt = {}
    try:
        hzt[tstamp_hzt0] = read_cartesian_metranet(
            filedict[tstamp_hzt0], reader="python"
        ).fields["iso0_height"]["data"][0]
        hzt[tstamp_hzt1] = read_cartesian_metranet(
            filedict[tstamp_hzt1], reader="python"
        ).fields["iso0_height"]["data"][0]
    except Exception as e:
        logger.error(f"Error reading HZT files: {e}")
        return {}

    dt = datetime.timedelta(minutes=tsteps_min)
    ndt = np.arange(1, int(60 / tsteps_min))
    deltaHZT = (hzt[tstamp_hzt1] - hzt[tstamp_hzt0]) / (len(ndt) + 1)

    for idx in ndt:
        if idx == ndt[0]:
            deltaHZT_temp = deltaHZT.copy()
        else:
            deltaHZT_temp += deltaHZT
        hzt[tstamp_hzt0 + dt * idx] = hzt[tstamp_hzt0] + deltaHZT_temp
    return hzt


# ============================================================================
# NOISE COMPUTATION
# ============================================================================


def compute_noise(
    rad_obj, status_fpath: str, sweep_number: int, verbose: bool = False
) -> None:
    """Compute noise estimate for a radar object."""
    NOISE_100 = 5

    if not os.path.isfile(status_fpath):
        if verbose:
            logger.warning("Cannot find status file. Using default noise value.")
        noisedBADU_h = NOISE_100
        noisedBADU_v = NOISE_100
    else:
        status = read_status(status_fpath)
        if not len(status.keys()):
            if verbose:
                logger.warning(
                    "Cannot read status file. Using default noise value."
                )
            noisedBADU_h = NOISE_100
            noisedBADU_v = NOISE_100
        else:
            try:
                noise_h = float(
                    status["status"]["sweep"][sweep_number]["RADAR"]["STAT"][
                        "CALIB"
                    ]["noisepower_frontend_h_inuse"]["@value"]
                )
                rconst_h = float(
                    status["status"]["sweep"][sweep_number]["RADAR"]["STAT"][
                        "CALIB"
                    ]["rconst_h"]["@value"]
                )
                noisedBADU_h = 10.0 * np.log10(noise_h) + rconst_h

                noise_v = float(
                    status["status"]["sweep"][sweep_number]["RADAR"]["STAT"][
                        "CALIB"
                    ]["noisepower_frontend_v_inuse"]["@value"]
                )
                rconst_v = float(
                    status["status"]["sweep"][sweep_number]["RADAR"]["STAT"][
                        "CALIB"
                    ]["rconst_v"]["@value"]
                )
                noisedBADU_v = 10.0 * np.log10(noise_v) + rconst_v
            except Exception:
                noisedBADU_h = NOISE_100
                noisedBADU_v = NOISE_100

    noisedBZ_h = pyart.retrieve.compute_noisedBZ(
        rad_obj.nrays,
        noisedBADU_h,
        rad_obj.range["data"],
        100.0,
        noise_field="noisedBZ_hh",
    )
    noisedBZ_v = pyart.retrieve.compute_noisedBZ(
        rad_obj.nrays,
        noisedBADU_v,
        rad_obj.range["data"],
        100.0,
        noise_field="noisedBZ_vv",
    )

    noisedBZ_h["data"] = np.ma.array(
        noisedBZ_h["data"], mask=np.isnan(noisedBZ_h["data"])
    )
    noisedBZ_v["data"] = np.ma.array(
        noisedBZ_v["data"], mask=np.isnan(noisedBZ_v["data"])
    )

    rad_obj.add_field("noise_h", noisedBZ_h)
    rad_obj.add_field("noise_v", noisedBZ_v)


# ============================================================================
# CARTESIAN COORDINATES CORRECTION
# ============================================================================


def correct_gate_cartesian_coordinates(rad_obj, ke: float = 1.25):
    """Convert radar antenna vectors to Cartesian coordinates using Swiss standard."""
    x_radar_raw, y_radar_raw, z_radar_raw = (
        pyart.core.transforms.antenna_vectors_to_cartesian(
            rad_obj.range["data"],
            rad_obj.azimuth["data"],
            rad_obj.elevation["data"],
            ke=ke,
        )
    )
    return x_radar_raw, y_radar_raw, z_radar_raw


# ============================================================================
# VISIBILITY
# ============================================================================


def add_visibility(rad_obj, visibility: np.ndarray) -> None:
    """Add visibility field to radar object."""
    z_h = rad_obj.get_field(0, "reflectivity")
    visib_sweep = np.ma.array(
        visibility.astype(np.float32), mask=np.isnan(visibility)
    )
    if visib_sweep.shape[1] < z_h.shape[1]:
        raise ValueError("Reflectivity shape does not match the expected one")
    visib_sweep = visib_sweep[0 : z_h.shape[0], 0 : z_h.shape[1]]
    rad_obj.add_field("visibility", {"data": visib_sweep})


def correct_reflectivity_for_visibility(
    rad_obj,
    visibility: np.ndarray,
    minimum_visibility: int = 37,
    maximum_correction_dbz: float = 2,
) -> None:
    """Mask radar data at low visibility and correct reflectivity."""
    z_h = rad_obj.get_field(0, "reflectivity")
    z_v = rad_obj.get_field(0, "reflectivity_vv")

    corr = np.full(visibility.shape, np.nan, dtype=np.float32)
    positive_vis = visibility > 0
    corr[positive_vis] = 1.0 / (visibility[positive_vis] / 100.0)
    corr[corr >= maximum_correction_dbz] = maximum_correction_dbz

    visib_mask = visibility < minimum_visibility

    z_h_linear = 10 ** (0.1 * z_h)
    z_h_corrected = 10.0 * np.log10(z_h_linear * corr)

    z_v_linear = 10 ** (0.1 * z_v)
    z_v_corrected = 10.0 * np.log10(z_v_linear * corr)

    z_h_vis_corrected = np.ma.array(
        z_h_corrected.astype(np.float32), mask=visib_mask
    )
    z_v_vis_corrected = np.ma.array(
        z_v_corrected.astype(np.float32), mask=visib_mask
    )
    visib_mask_for_rdr = np.ma.array(visib_mask, mask=np.isnan(visib_mask))

    rad_obj.add_field(
        "reflectivity_visibilitycorr", {"data": z_h_vis_corrected}
    )
    rad_obj.add_field(
        "reflectivity_vv_visibilitycorr", {"data": z_v_vis_corrected}
    )
    rad_obj.add_field("visibility_mask", {"data": visib_mask_for_rdr})


# ============================================================================
# KDP COMPUTATION
# ============================================================================


def compute_kdp(
    rad_obj,
    kdp_rmin: float,
    kdp_rmax: float,
    kdp_rcell: float,
    kdp_rwind: float,
    kdp_zmin: float,
    kdp_zmax: float,
) -> None:
    """Compute KDP using moving least-square algorithm."""
    r = rad_obj.range["data"]
    ind_rmin = np.where(r > kdp_rmin)[0][0]
    ind_rmax = np.where(r < kdp_rmax)[0][-1]
    r_res = r[1] - r[0]
    min_rcons = int(kdp_rcell / r_res)
    wind_len = int(kdp_rwind / r_res)
    min_valid = int(wind_len / 2 + 1)

    psidp_field = "uncorrected_differential_phase"
    if "reflectivity_visibilitycorr" in rad_obj.fields:
        refl_field = "reflectivity_visibilitycorr"
    else:
        refl_field = "reflectivity"
    phidp_field = "propagation_differential_phase"
    kdp_field = "specific_differential_phase"

    phidp = smooth_phidp_single_window(
        rad_obj,
        ind_rmin=ind_rmin,
        ind_rmax=ind_rmax,
        min_rcons=min_rcons,
        zmin=kdp_zmin,
        zmax=kdp_zmax,
        wind_len=wind_len,
        min_valid=min_valid,
        psidp_field=psidp_field,
        refl_field=refl_field,
        phidp_field=phidp_field,
    )
    rad_obj.add_field(phidp_field, phidp)

    kdp = kdp_leastsquare_single_window(
        rad_obj,
        wind_len=wind_len,
        min_valid=min_valid,
        phidp_field=phidp_field,
        kdp_field=kdp_field,
        vectorize=True,
    )
    rad_obj.add_field(kdp_field, kdp)


# ============================================================================
# ATTENUATION CORRECTION
# ============================================================================


def correct_attenuation(rad_obj) -> None:
    """Correct for attenuation using ZPHI algorithm."""
    ah, pia, zh_corr, _, pida, zdr_corr = calculate_attenuation_zphi(
        rad_obj,
        refl_field="reflectivity_visibilitycorr",
        zdr_field="differential_reflectivity",
        phidp_field="propagation_differential_phase",
        iso0_field="height_over_iso0",
        temp_ref="height_over_iso0",
        doc=15,
    )
    rad_obj.add_field("attenuation_h", ah)
    rad_obj.add_field("corrected_reflectivity", zh_corr)
    rad_obj.add_field("corrected_differential_reflectivity", zdr_corr)

    zv_corr = (
        pia["data"]
        - pida["data"]
        + rad_obj.get_field(0, "reflectivity_vv")
    )
    rad_obj.add_field("corrected_reflectivity_vv", {"data": zv_corr})


# ============================================================================
# HYDROMETEOR CLASSIFICATION
# ============================================================================


def add_hydroclass_from_file(rad_obj, hydroclassif_fpath: str) -> None:
    """Add hydrometeor classification from file."""
    if hydroclassif_fpath != "no_data" and hydroclassif_fpath is not None:
        try:
            hydroclassif = pyart.aux_io.read_file_py(
                hydroclassif_fpath, moment="ZDRP"
            )
            hydroclassif_scaled = np.array(hydroclassif.data / 25)
            hydroclassif_masked = np.ma.array(
                hydroclassif_scaled.astype(np.uint8),
                mask=np.logical_or(
                    hydroclassif_scaled < 0, hydroclassif_scaled > 9
                ),
            )
        except Exception as e:
            logger.warning(
                f"Error reading hydroclass file {hydroclassif_fpath}: {e}"
            )
            nan_data = np.full(
                (rad_obj.nrays, rad_obj.ngates), np.nan, dtype=np.float32
            )
            hydroclassif_masked = np.ma.array(
                nan_data, mask=np.isnan(nan_data)
            )
    else:
        nan_data = np.full(
            (rad_obj.nrays, rad_obj.ngates), np.nan, dtype=np.float32
        )
        hydroclassif_masked = np.ma.array(nan_data, mask=np.isnan(nan_data))
    rad_obj.add_field(
        "hydrometeor_classification_mch", {"data": hydroclassif_masked}
    )


def compute_hydroclass_semisupervised(rad_obj) -> None:
    """Compute hydrometeor classification using PyART semi-supervised method."""
    if "height_over_iso0" not in rad_obj.fields:
        logger.warning(
            "Cannot compute PyART hydroclass: height_over_iso0 field missing"
        )
        return

    try:
        hydro = pyart.retrieve.hydroclass_semisupervised(
            rad_obj,
            hydro_names=(
                "AG", "CR", "LR", "RP", "RN", "VI", "WS", "MH", "IH",
            ),
            var_names=("dBZ", "ZDR", "KDP", "RhoHV", "H_ISO0"),
            refl_field=(
                "reflectivity_visibilitycorr"
                if "reflectivity_visibilitycorr" in rad_obj.fields
                else "reflectivity"
            ),
            zdr_field="differential_reflectivity",
            kdp_field="specific_differential_phase",
            rhv_field="uncorrected_cross_correlation_ratio",
            iso0_field="height_over_iso0",
        )["hydro"]
        rad_obj.add_field("radar_echo_classification", hydro)
    except Exception as e:
        logger.warning(f"Error computing PyART hydroclass: {e}")


# ============================================================================
# HZT DATA ADDITION
# ============================================================================


def add_hzt_data(
    rad_obj,
    hzt_cartesian: np.ndarray,
    radar_letter: str,
    sweep_number: int,
    qpegrid_to_rad_dir: str,
    verbose: bool = False,
) -> None:
    """Add Height of Freezing Level (HZT) field to radar object."""
    NBINS_Y = 710
    NBINS_X = 640
    Y_SWISS_GRID = np.linspace(255, 965, NBINS_Y + 1)
    X_SWISS_GRID = np.linspace(480, -160, NBINS_X + 1)

    qpegrid_to_rad = read_qpegrid_to_rad(
        radar_letter, qpegrid_to_rad_dir, verbose
    )
    lut_sweep = qpegrid_to_rad[qpegrid_to_rad[:, 0] == (sweep_number - 1)]
    nrange = lut_sweep[:, 2].max() + 1
    naz = lut_sweep[:, 1].max() + 1

    idxx = (lut_sweep[:, -1] - X_SWISS_GRID[-1]).astype(int) - 1
    idxy = (lut_sweep[:, -2] - Y_SWISS_GRID[0]).astype(int) - 1
    idxaz = lut_sweep[:, 1]
    idxrange = lut_sweep[:, 2]

    hzt_pol = np.zeros((naz, nrange))
    npts = np.zeros((naz, nrange))

    toadd = hzt_cartesian[idxx.ravel(), idxy.ravel()]
    hzt_pol[idxaz.ravel(), idxrange.ravel()] += toadd
    npts[idxaz.ravel(), idxrange.ravel()] += np.ones(toadd.shape)

    npts[npts == 0] = np.nan
    hzt_pol /= npts

    x, y = np.mgrid[0:naz, 0:nrange]
    xygood = np.array((x[~np.isnan(hzt_pol)], y[~np.isnan(hzt_pol)])).T
    xybad = np.array((x[np.isnan(hzt_pol)], y[np.isnan(hzt_pol)])).T

    hzt_pol[np.isnan(hzt_pol)] = hzt_pol[~np.isnan(hzt_pol)][
        KDTree(xygood).query(xybad)[1]
    ]

    hzt_pol_field = np.zeros((rad_obj.nrays, rad_obj.ngates)) + np.nan
    hzt_pol_field[:, 0 : hzt_pol.shape[1]] = hzt_pol

    hzt_dict = {
        "data": hzt_pol_field,
        "units": "m",
        "long_name": "Height of freezing level",
        "standard_name": "HZT",
    }
    rad_obj.add_field("iso0_height", hzt_dict)


def add_height_over_iso0(rad_obj, z_gates: np.ndarray) -> None:
    """Add height above freezing level to radar object."""
    height_over_iso0 = z_gates - rad_obj.fields["iso0_height"]["data"]
    height_over_iso0_masked = np.ma.array(
        height_over_iso0.astype(np.float32), mask=np.isnan(height_over_iso0)
    )

    iso0_dict = {
        "data": height_over_iso0_masked,
        "units": "m",
        "long_name": "height of freezing level with respect to radar gate altitude",
        "standard_name": "height_over_iso0",
    }
    rad_obj.add_field("height_over_iso0", iso0_dict)

    lapse_rate = -6.5  # C/km
    temperature = 0 + lapse_rate * height_over_iso0 / 1000
    temperature = np.ma.array(
        temperature.astype(np.float32), mask=np.isnan(temperature)
    )
    temp_dict = {
        "data": temperature,
        "units": "C",
        "long_name": "Temperature",
        "standard_name": "air_temperature",
    }
    rad_obj.add_field("temperature", temp_dict)


# ============================================================================
# COMPLETE SWEEP PROCESSING
# ============================================================================


def load_metranet_sweep(
    radar_fpath: str,
    hydroclassif_fpath: str | None = None,
    hzt_cartesian: np.ndarray | None = None,
    visibility: np.ndarray | None = None,
    qpegrid_to_rad_dir: str | None = None,
    compute_pyart_hc: bool = True,
    verbose: bool = False,
    timer=None,
    volume: str | None = None,
):
    """Load and process a single METRANET sweep file through all stages."""
    radar_filename = os.path.basename(radar_fpath)
    radar_letter = radar_filename[2]
    sweep_number = int(radar_filename.split(".")[-1])

    # --- Read raw file ---
    _vprint(f"    -> Reading METRANET: {radar_filename}", verbose)
    with (
        timer.time_stage("read_metranet", volume=volume, sweep=sweep_number)
        if timer
        else _nullctx()
    ):
        rad_obj = pyart.aux_io.read_metranet(
            radar_fpath, reader="C", nbytes=4
        )

    # --- Cartesian coordinates ---
    _vprint(f"    -> Computing Cartesian coordinates...", verbose)
    with (
        timer.time_stage("coordinates", volume=volume, sweep=sweep_number)
        if timer
        else _nullctx()
    ):
        x_radar_raw, y_radar_raw, z_radar_raw = (
            correct_gate_cartesian_coordinates(rad_obj)
        )

    # --- Visibility correction ---
    if visibility is not None:
        _vprint(f"    -> Applying visibility correction...", verbose)
        with (
            timer.time_stage("visibility", volume=volume, sweep=sweep_number)
            if timer
            else _nullctx()
        ):
            add_visibility(rad_obj, visibility)
            correct_reflectivity_for_visibility(
                rad_obj,
                visibility,
                minimum_visibility=37,
                maximum_correction_dbz=2,
            )

    # --- KDP computation ---
    _vprint(f"    -> Computing KDP...", verbose)
    KDP_RMIN, KDP_RMAX, KDP_RCELL = 1000.0, 50000.0, 1000.0
    KDP_ZMIN, KDP_ZMAX, KDP_RWIND = 20.0, 40.0, 6000.0
    with (
        timer.time_stage("kdp", volume=volume, sweep=sweep_number)
        if timer
        else _nullctx()
    ):
        compute_kdp(
            rad_obj,
            kdp_rmin=KDP_RMIN,
            kdp_rmax=KDP_RMAX,
            kdp_rcell=KDP_RCELL,
            kdp_rwind=KDP_RWIND,
            kdp_zmin=KDP_ZMIN,
            kdp_zmax=KDP_ZMAX,
        )

    # --- HZT and attenuation ---
    if hzt_cartesian is not None and qpegrid_to_rad_dir is not None:
        _vprint(f"    -> Adding HZT / height-over-iso0...", verbose)
        with (
            timer.time_stage("hzt", volume=volume, sweep=sweep_number)
            if timer
            else _nullctx()
        ):
            add_hzt_data(
                rad_obj,
                hzt_cartesian,
                radar_letter,
                sweep_number,
                qpegrid_to_rad_dir=qpegrid_to_rad_dir,
                verbose=verbose,
            )
            z_gates = rad_obj.altitude["data"][0] + z_radar_raw
            add_height_over_iso0(rad_obj, z_gates)

        _vprint(f"    -> Correcting attenuation (ZPHI)...", verbose)
        with (
            timer.time_stage("attenuation", volume=volume, sweep=sweep_number)
            if timer
            else _nullctx()
        ):
            correct_attenuation(rad_obj)

    # --- Hydrometeor classification (MCH operational) ---
    _vprint(f"    -> Loading MCH hydrometeor classification...", verbose)
    with (
        timer.time_stage("hydroclass_mch", volume=volume, sweep=sweep_number)
        if timer
        else _nullctx()
    ):
        add_hydroclass_from_file(rad_obj, hydroclassif_fpath)

    # --- Hydrometeor classification (PyART semi-supervised) ---
    if compute_pyart_hc and "height_over_iso0" in rad_obj.fields:
        _vprint(f"    -> Computing PyART hydroclass...", verbose)
        with (
            timer.time_stage(
                "hydroclass_pyart", volume=volume, sweep=sweep_number
            )
            if timer
            else _nullctx()
        ):
            compute_hydroclass_semisupervised(rad_obj)

    # Ensure all expected fields exist
    _expected_fields = [
        "iso0_height",
        "hydrometeor_classification_mch",
        "radar_echo_classification",
    ]
    for field in _expected_fields:
        if field not in rad_obj.fields:
            nan_data = np.full(
                (rad_obj.nrays, rad_obj.ngates), np.nan, dtype=np.float32
            )
            rad_obj.add_field(
                field, {"data": np.ma.array(nan_data, mask=np.isnan(nan_data))}
            )

    return rad_obj, x_radar_raw, y_radar_raw, z_radar_raw


# ============================================================================
# PyART -> xarray CONVERSION
# ============================================================================


def pyart_to_xradar_dataset(
    rad_obj, sweep_idx: int = 0, fields_to_extract: list[str] | None = None
) -> xr.Dataset:
    """Convert a PyART radar object to an xradar-compatible xarray Dataset."""
    sweep_slice = rad_obj.get_slice(sweep_idx)

    time_data = pyart.util.datetime_from_radar(rad_obj)
    azimuth = rad_obj.azimuth["data"][sweep_slice]
    elevation = rad_obj.elevation["data"][sweep_slice]
    range_data = rad_obj.range["data"]

    if isinstance(time_data, (list, np.ndarray)):
        time_coord = (
            time_data[sweep_slice]
            if np.size(time_data) > 1
            else [time_data[0]] * len(azimuth)
        )
    else:
        time_coord = [time_data] * len(azimuth)

    coords = {
        "azimuth": (["azimuth"], azimuth),
        "range": (["range"], range_data),
        "time": (["azimuth"], time_coord),
        "elevation": (["azimuth"], elevation),
    }

    if rad_obj.latitude:
        coords["latitude"] = float(rad_obj.latitude["data"][0])
    if rad_obj.longitude:
        coords["longitude"] = float(rad_obj.longitude["data"][0])
    if rad_obj.altitude:
        coords["altitude"] = float(rad_obj.altitude["data"][0])
    if rad_obj.fixed_angle:
        coords["elevation_angle"] = float(
            rad_obj.fixed_angle["data"][sweep_idx]
        )

    data_vars = {}
    available_fields = list(rad_obj.fields.keys())
    if fields_to_extract is None:
        fields_to_extract = available_fields

    for field_name in fields_to_extract:
        if field_name not in available_fields:
            continue

        field_data = rad_obj.get_field(sweep_idx, field_name)
        mapped_name = FIELD_MAPPING.get(field_name, field_name)

        data_vars[mapped_name] = (
            ["azimuth", "range"],
            field_data,
            {
                "long_name": rad_obj.fields[field_name].get(
                    "long_name", mapped_name
                ),
                "units": rad_obj.fields[field_name].get("units", ""),
            },
        )

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs["sweep_number"] = sweep_idx + 1
    return ds


def pyart_volume_to_datatree(
    rad_objs: dict[int, object],
    fields_to_extract: list[str] | None = None,
) -> xr.DataTree:
    """Convert multiple PyART radar objects (sweeps) to an xradar DataTree."""
    dict_ds = {}
    for sweep_num, rad_obj in rad_objs.items():
        ds = pyart_to_xradar_dataset(
            rad_obj, sweep_idx=0, fields_to_extract=fields_to_extract
        )
        dict_ds[f"sweep_{sweep_num}"] = ds
    return xr.DataTree.from_dict(dict_ds)


# ============================================================================
# METRANET -> xarray (via radar_api)
# ============================================================================


def metranet_to_datatree(
    filepath: str | Path,
    network: str = "MCH_LTE",
    product: str = "POL",
) -> xr.DataTree:
    """Read a METRANET file directly into an xarray DataTree via radar_api."""
    return radar_api.open_datatree(str(filepath), network=network, product=product)


def volume_to_datatree(
    sweep_filepaths: list,
    network: str = "MCH_LTE",
    product: str = "POL",
    max_workers: int = 1,
) -> xr.DataTree:
    """Load all sweeps of a volume from METRANET files into a DataTree."""
    from raddb.helper import list_sweep_names

    def _load_sweep(idx_path):
        sweep_idx, sweep_filepath = idx_path
        try:
            dt_sweep = metranet_to_datatree(
                sweep_filepath, network=network, product=product
            )
            names = list_sweep_names(dt_sweep)
            if not names:
                return None, None
            ds = dt_sweep[names[0]].to_dataset().assign_attrs(
                {"sweep_number": sweep_idx}
            )
            return f"sweep_{sweep_idx}", ds
        except Exception as exc:
            logger.error(f"Error loading {sweep_filepath}: {exc}")
            return None, None

    paths = sorted([str(p) for p in sweep_filepaths])
    args = [(i, p) for i, p in enumerate(paths, 1)]
    dict_ds = {}

    if max_workers <= 1:
        for arg in args:
            k, ds = _load_sweep(arg)
            if k and ds is not None:
                dict_ds[k] = ds
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
            futures = [executor.submit(_load_sweep, arg) for arg in args]
            for future in concurrent.futures.as_completed(futures):
                k, ds = future.result()
                if k and ds is not None:
                    dict_ds[k] = ds

    if not dict_ds:
        raise ValueError("No valid sweep datasets loaded.")
    return xr.DataTree.from_dict(dict_ds)


# ============================================================================
# FILE DISCOVERY HELPERS
# ============================================================================


def _parse_volume_time(stem: str) -> datetime.datetime:
    """Parse volume timestamp from METRANET filename."""
    try:
        y, j, h, m = (
            int(stem[3:5]),
            int(stem[5:8]),
            int(stem[8:10]),
            int(stem[10:12]),
        )
        return datetime.datetime(2000 + y, 1, 1) + datetime.timedelta(
            days=j - 1, hours=h, minutes=m
        )
    except Exception:
        return datetime.datetime(1970, 1, 1)


def _group_files_by_volume(paths: list[str]) -> dict:
    """Group sweep files by volume (based on filename stem)."""
    vols = defaultdict(list)
    for p in paths:
        stem = Path(p).stem
        vols[stem].append(p)
    return dict(vols)


def _raw_data_dir_candidates(
    raw_data_dir: str | None, network: str
) -> list[str | None]:
    """Return candidate base directories for radar_api.find_files."""
    if raw_data_dir is None:
        return [None]

    raw_path = Path(raw_data_dir)
    candidates: list[str | None] = [str(raw_path)]

    network_prefix = (
        network.split("_", 1)[0].upper() if isinstance(network, str) else ""
    )
    if raw_path.name.upper() == network_prefix:
        candidates.append(str(raw_path.parent))

    seen = set()
    unique: list[str | None] = []
    for c in candidates:
        if c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


def find_files_with_fallback(
    *,
    network: str,
    radar: str,
    start_time,
    end_time,
    product: str,
    raw_data_dir: str | None,
    verbose: bool = False,
) -> list[str]:
    """Find files and retry with a parent directory fallback when needed."""
    last_error = None
    for candidate_base_dir in _raw_data_dir_candidates(
        raw_data_dir=raw_data_dir, network=network
    ):
        try:
            fpaths = radar_api.find_files(
                network=network,
                radar=radar,
                start_time=start_time,
                end_time=end_time,
                product=product,
                protocol="local",
                base_dir=candidate_base_dir,
            )
        except Exception as e:
            last_error = e
            continue

        if fpaths:
            if (
                verbose
                and raw_data_dir is not None
                and candidate_base_dir != raw_data_dir
            ):
                logger.info(
                    "No files found with raw_data_dir='%s'. "
                    "Retrying with '%s' succeeded.",
                    raw_data_dir,
                    candidate_base_dir,
                )
            return fpaths

    if last_error is not None:
        raise last_error
    return []


def find_hzt_files(
    radar: str,
    network: str,
    volume_time: datetime.datetime,
    raw_data_dir: str | None = None,
) -> dict:
    """Find HZT files around the volume time."""
    hzt_start = volume_time - datetime.timedelta(hours=1)
    hzt_end = volume_time + datetime.timedelta(hours=1)

    try:
        hzt_files = find_files_with_fallback(
            network=network,
            radar=radar,
            start_time=hzt_start,
            end_time=hzt_end,
            product="HZT",
            raw_data_dir=raw_data_dir,
        )
    except Exception as e:
        logger.warning(f"Error finding HZT files: {e}")
        return {}

    hzt_dict = {}
    if hzt_files and isinstance(hzt_files, list):
        for hzt_file in hzt_files:
            try:
                stem = Path(hzt_file).stem
                hzt_time = _parse_volume_time(stem)
                hzt_dict[hzt_time] = hzt_file
            except Exception as e:
                logger.warning(f"Error parsing HZT file {hzt_file}: {e}")
                continue
    return hzt_dict


def find_hydroclass_files(sweep_filepaths: list[str]) -> dict[int, str]:
    """Find hydrometeor classification files corresponding to sweeps."""
    hym_paths = {}
    for sweep_path in sweep_filepaths:
        norm = Path(sweep_path)
        sweep_num = int(norm.suffix[1:])

        hym_ext = ".8" + norm.suffix[2:]
        hym_dir = norm.parent.parent / norm.parent.name.replace("ML", "YM")
        hym_pattern = str(
            hym_dir
            / (norm.stem.replace("ML", "YM")[:-2] + "*L" + hym_ext)
        )

        matches = glob.glob(hym_pattern)
        if matches:
            hym_paths[sweep_num] = matches[-1]
    return hym_paths


# ============================================================================
# MCH VOLUME PROCESSING  (METRANET -> DataTree)
# ============================================================================


def process_mch_volume(
    sweep_filepaths: list[str],
    network: str,
    radar: str,
    volume_time: datetime.datetime,
    raw_data_dir: str | None = None,
    static_vis_dir: str | None = None,
    qpegrid_to_rad_dir: str | None = None,
    hzt_enabled: bool = True,
    hym_enabled: bool = True,
    compute_pyart_hc: bool = True,
    verbose: bool = False,
    timer=None,
    volume_label: str | None = None,
) -> xr.DataTree:
    """
    Process a complete METRANET volume scan.

    Reads raw sweep files, applies all corrections (visibility, KDP,
    attenuation, HZT, hydrometeor classification), and returns an xarray
    DataTree ready for archiving with RadDB.

    Parameters
    ----------
    sweep_filepaths : list
        File paths for all sweeps in the volume.
    network : str
        Network identifier (e.g. ``"MCH_LTE"``).
    radar : str
        Radar name (e.g. ``"MLA"`` or ``"A"``).
    volume_time : datetime
        Volume scan timestamp.
    raw_data_dir, static_vis_dir, qpegrid_to_rad_dir : str, optional
        Directory paths for ancillary data.
    hzt_enabled, hym_enabled, compute_pyart_hc : bool
        Toggle optional processing stages.
    verbose : bool
        Print progress messages.
    timer : StageTimer, optional
        Profiling timer.
    volume_label : str, optional
        Label for timer records.

    Returns
    -------
    xr.DataTree
        Processed volume as xradar DataTree.
    """
    radar_letter = normalize_radar_name(radar)
    vol_id = volume_label or str(volume_time)

    # --- Load static visibility ---
    vis_dict = {}
    if static_vis_dir:
        _vprint(
            f"  Loading visibility LUT for radar {radar_letter}...", verbose
        )
        with (
            timer.time_stage("load_visibility_lut", volume=vol_id)
            if timer
            else _nullctx()
        ):
            try:
                vis_dict = read_static_visibility(
                    radar_letter, static_vis_dir, verbose=verbose
                )
            except Exception as e:
                logger.warning(
                    f"Could not load visibility for radar {radar_letter}: {e}"
                )

    # --- Find and interpolate HZT ---
    hzt_interpolated_dict = {}
    if hzt_enabled and qpegrid_to_rad_dir:
        _vprint(f"  Finding and interpolating HZT files...", verbose)
        with (
            timer.time_stage("load_hzt", volume=vol_id)
            if timer
            else _nullctx()
        ):
            try:
                hzt_files = find_hzt_files(
                    radar, network, volume_time, raw_data_dir=raw_data_dir
                )
                _vprint(
                    f"  Found {len(hzt_files) if isinstance(hzt_files, dict) else 0} HZT files",
                    verbose,
                )
                if isinstance(hzt_files, dict) and len(hzt_files) >= 2:
                    hzt_interpolated_dict = hzt_hourly_to_5min(
                        hzt_files, tsteps_min=5
                    )
                else:
                    _vprint(
                        f"  Need >=2 HZT files for interpolation, got "
                        f"{len(hzt_files) if isinstance(hzt_files, dict) else 'invalid'}",
                        verbose,
                    )
            except Exception as e:
                logger.warning(f"Could not process HZT: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()

    # --- Find HYM files ---
    hym_paths = {}
    if hym_enabled:
        _vprint(f"  Finding HYM hydrometeor-class files...", verbose)
        with (
            timer.time_stage("find_hym_files", volume=vol_id)
            if timer
            else _nullctx()
        ):
            hym_paths = find_hydroclass_files(sweep_filepaths)
        _vprint(f"  Found {len(hym_paths)} HYM file(s)", verbose)

    # --- Process each sweep ---
    rad_objs = {}
    sorted_paths = sorted(sweep_filepaths)
    n_sweeps = len(sorted_paths)

    for i, sweep_filepath in enumerate(sorted_paths, 1):
        sweep_num = int(Path(sweep_filepath).suffix[1:])
        _vprint(
            f"  Processing sweep {i}/{n_sweeps}  (sweep #{sweep_num})...",
            verbose,
        )

        visibility = vis_dict.get(sweep_num) if vis_dict else None

        hzt_cartesian = None
        if hzt_interpolated_dict:
            hzt_times = list(hzt_interpolated_dict.keys())
            if hzt_times:
                closest_time = min(
                    hzt_times,
                    key=lambda t: abs((t - volume_time).total_seconds()),
                )
                hzt_cartesian = hzt_interpolated_dict[closest_time]

        hym_file = hym_paths.get(sweep_num)

        try:
            with (
                timer.time_stage(
                    "load_metranet_sweep", volume=vol_id, sweep=sweep_num
                )
                if timer
                else _nullctx()
            ):
                rad_obj, x, y, z = load_metranet_sweep(
                    radar_fpath=sweep_filepath,
                    hydroclassif_fpath=hym_file,
                    hzt_cartesian=hzt_cartesian,
                    visibility=visibility,
                    qpegrid_to_rad_dir=qpegrid_to_rad_dir,
                    compute_pyart_hc=compute_pyart_hc,
                    verbose=verbose,
                    timer=timer,
                    volume=vol_id,
                )
            rad_objs[sweep_num] = rad_obj
        except Exception as e:
            logger.error(f"Error processing sweep {sweep_num}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    if not rad_objs:
        raise ValueError("No sweeps were successfully processed")

    # --- Convert to xradar DataTree ---
    _vprint(
        f"  Converting {len(rad_objs)} sweep(s) to xarray DataTree...", verbose
    )
    with (
        timer.time_stage("pyart_to_datatree", volume=vol_id)
        if timer
        else _nullctx()
    ):
        dt = pyart_volume_to_datatree(rad_objs)
    return dt


def process_mch_volumes(
    network: str,
    radar: str | list[str],
    start_time: str | datetime.datetime,
    end_time: str | datetime.datetime,
    raw_data_dir: str | None = None,
    static_vis_dir: str | None = None,
    qpegrid_to_rad_dir: str | None = None,
    hzt_enabled: bool = True,
    hym_enabled: bool = True,
    compute_pyart_hc: bool = True,
    product: str = "POL",
    verbose: bool = True,
    timer: StageTimer | None = None,
) -> list[dict]:
    """
    Process MCH radar data for a time range into DataTrees (sequential).

    Finds raw METRANET files, applies all Swiss-specific corrections
    (visibility, KDP, attenuation, HZT, hydrometeor classification), and
    returns the processed DataTrees.  Archiving to Parquet is done
    separately via RadDB (``db.archive_volume()`` /
    ``db.archive_multiple_volumes()``).

    Supports multiple radars: pass a list of radar names or ``"all"`` to
    process all five Swiss radars (A, D, L, P, W).

    Parameters
    ----------
    network : str
        Network identifier (e.g. ``"MCH_LTE"``).
    radar : str or list of str
        Radar name(s).  ``"all"`` expands to all five Swiss radars.
    start_time, end_time : str or datetime
        Processing time window.
    raw_data_dir : str, optional
        Override for raw data location.
    static_vis_dir : str, optional
        Path to static visibility LUTs.
    qpegrid_to_rad_dir : str, optional
        Path to QPE grid-to-radar mapping LUTs.
    hzt_enabled : bool
        Compute isothermal height (HZT). Default ``True``.
    hym_enabled : bool
        Apply MCH operational hydrometeor classification. Default ``True``.
    compute_pyart_hc : bool
        Apply PyART semi-supervised HC. Default ``True``.
    product : str
        METRANET product type. Default ``"POL"``.
    verbose : bool
        Print progress messages. Default ``True``.
    timer : StageTimer, optional
        Profiling timer.

    Returns
    -------
    list of dict
        One dict per volume with keys: ``radar``, ``stem``, ``time``,
        ``success``, ``error``, ``datatree``, ``n_sweeps``,
        ``sweep_filepaths``.  Use ``result["datatree"]`` for archiving
        with RadDB.
    """
    # Handle multi-radar
    if isinstance(radar, str):
        if radar.upper() == "ALL":
            radars = RADAR_LETTERS
        else:
            radars = [normalize_radar_name(radar)]
    else:
        radars = [normalize_radar_name(r) for r in radar]

    all_results = []

    for radar_letter in radars:
        _vprint(
            f"\n{'='*60}\nProcessing radar {radar_letter}\n{'='*60}",
            verbose,
        )

        # --- Find files ---
        _vprint(
            f"Finding {product} files for radar {radar_letter} "
            f"({start_time} -> {end_time})...",
            verbose,
        )
        with (
            timer.time_stage("find_files", volume=f"radar_{radar_letter}")
            if timer
            else _nullctx()
        ):
            try:
                fps = find_files_with_fallback(
                    network=network,
                    radar=radar_letter,
                    start_time=start_time,
                    end_time=end_time,
                    product=product,
                    raw_data_dir=raw_data_dir,
                    verbose=verbose,
                )
            except Exception as e:
                logger.error(
                    f"Error finding files for radar {radar_letter}: {e}"
                )
                continue

        if not fps:
            _vprint(
                f"No files found for radar {radar_letter} "
                f"between {start_time} and {end_time}",
                verbose,
            )
            continue

        # Group by volume
        volumes = _group_files_by_volume(fps)
        _vprint(
            f"Found {len(fps)} sweep file(s) -> {len(volumes)} volume(s)",
            verbose,
        )

        # Process each volume
        pipeline_t0 = _time.perf_counter()

        for i, (stem, sweep_paths) in enumerate(volumes.items(), 1):
            vol_time = _parse_volume_time(stem)
            _vprint(
                f"\n>>  Volume {i}/{len(volumes)}: {stem}  "
                f"({vol_time.strftime('%Y-%m-%d %H:%M')}, "
                f"{len(sweep_paths)} sweeps)",
                verbose,
            )

            result = {
                "radar": radar_letter,
                "stem": stem,
                "time": vol_time,
                "success": False,
                "error": None,
                "datatree": None,
                "n_sweeps": len(sweep_paths),
                "sweep_filepaths": sweep_paths,
            }

            vol_t0 = _time.perf_counter()
            try:
                with (
                    timer.time_stage("process_volume", volume=stem)
                    if timer
                    else _nullctx()
                ):
                    dt = process_mch_volume(
                        sweep_filepaths=sweep_paths,
                        network=network,
                        radar=radar_letter,
                        volume_time=vol_time,
                        raw_data_dir=raw_data_dir,
                        static_vis_dir=static_vis_dir,
                        qpegrid_to_rad_dir=qpegrid_to_rad_dir,
                        hzt_enabled=hzt_enabled,
                        hym_enabled=hym_enabled,
                        compute_pyart_hc=compute_pyart_hc,
                        verbose=verbose,
                        timer=timer,
                        volume_label=stem,
                    )

                result["success"] = True
                result["datatree"] = dt

                vol_elapsed = _time.perf_counter() - vol_t0
                _vprint(
                    f"OK  Volume {i}/{len(volumes)} done in "
                    f"{vol_elapsed:.1f}s",
                    verbose,
                )

            except Exception as e:
                vol_elapsed = _time.perf_counter() - vol_t0
                result["error"] = str(e)
                _vprint(
                    f"FAIL  Volume {i}/{len(volumes)} FAILED in "
                    f"{vol_elapsed:.1f}s: {e}",
                    verbose,
                )
                logger.error(
                    f"[{i}/{len(volumes)}] {vol_time} - FAIL: {e}"
                )
                if verbose:
                    import traceback
                    traceback.print_exc()

            all_results.append(result)

        total_elapsed = _time.perf_counter() - pipeline_t0
        n_ok = sum(
            1 for r in all_results if r["success"] and r["radar"] == radar_letter
        )
        n_total = sum(
            1 for r in all_results if r["radar"] == radar_letter
        )
        _vprint(
            f"\nRadar {radar_letter} complete: {n_ok}/{n_total} volumes "
            f"processed in {total_elapsed:.1f}s",
            verbose,
        )

    return all_results


# ============================================================================
# LUT GENERATION (MCH-specific, uses radar_api + pyart)
# ============================================================================


def generate_mch_lut(
    radar: str,
    network: str,
    sample_volume_filepaths: list[str],
    output_base_path: str,
    qpegrid_to_rad_dir: str | None = None,
    ke: float = 1.25,
) -> str:
    """
    Generate a RadDB LUT from METRANET sample files.

    This is the MCH-specific LUT generator. It reads raw METRANET files
    via radar_api/pyart to extract the radar geometry, then delegates
    to the generic RadDB LUT storage.

    For a generic (non-MCH) workflow, use ``raddb.generate_lut_from_datatree()``.
    """
    import re
    import yaml

    radar = normalize_radar_name(radar)
    lut_dir = Path(output_base_path) / radar / "LUT"
    lut_path = lut_dir / f"{radar}_LUT.parquet"
    info_path = lut_dir / f"{radar}_info.yaml"

    if lut_path.exists() and info_path.exists():
        logger.info(
            "LUT already exists at %s -- skipping generation.", lut_path
        )
        return str(lut_path)

    sorted_paths = sorted([str(p) for p in sample_volume_filepaths])
    lut_dfs, sweep_meta = [], {}
    radar_lat, radar_lon, radar_alt = None, None, None

    for sweep_idx, sweep_filepath in enumerate(sorted_paths, start=1):
        rad_obj = radar_api.open_pyart(
            sweep_filepath, network=network, product="POL"
        )
        dt_sweep = radar_api.open_datatree(
            sweep_filepath, network=network, product="POL"
        )

        radar_lat = float(rad_obj.latitude["data"][0])
        radar_lon = float(rad_obj.longitude["data"][0])
        radar_alt = float(rad_obj.altitude["data"][0])
        elevation = float(np.mean(rad_obj.elevation["data"]))

        x_raw, y_raw, z_raw = (
            pyart.core.transforms.antenna_vectors_to_cartesian(
                rad_obj.range["data"],
                rad_obj.azimuth["data"],
                rad_obj.elevation["data"],
                ke=ke,
            )
        )
        azimuths = rad_obj.azimuth["data"]
        ranges = rad_obj.range["data"]
        n_az, n_rng = len(azimuths), len(ranges)

        groups = [
            g.lstrip("/")
            for g in dt_sweep.groups
            if re.match(r"sweep_\d+", g.lstrip("/"))
        ]
        ds_sweep = dt_sweep[groups[0]].to_dataset()

        # Compute per-gate geographic coordinates from Cartesian offsets.
        # Use Py-ART's AEQD transform to keep consistency with radar geometry.
        gate_lon, gate_lat = pyart.core.transforms.cartesian_to_geographic_aeqd(
            x_raw,
            y_raw,
            lon_0=radar_lon,
            lat_0=radar_lat,
        )
        gate_alt = radar_alt + z_raw

        # Vectorised int64 gate_id generation (zero string allocations)
        _radar_idx = np.int64(RADAR_TO_IDX[radar.upper()])
        _az_int    = np.round(azimuths * 10).astype(np.int64)   # (n_az,)
        _rng_int   = ranges.astype(np.int64)                     # (n_rng,)
        _gate_ids  = (
            _radar_idx          * np.int64(1_000_000_000_000)
            + sweep_idx         * np.int64(   10_000_000_000)
            + _az_int[:, None]  * np.int64(        1_000_000)
            + _rng_int[None, :]
        ).ravel()

        lut_dfs.append(pd.DataFrame({
            "gate_id":         _gate_ids,
            "sweep":           np.full(n_az * n_rng, sweep_idx, dtype=np.int32),
            "azimuth":         np.repeat(azimuths, n_rng),
            "range":           np.tile(ranges, n_az),
            "elevation_angle": np.full(n_az * n_rng, elevation),
            "latitude":        gate_lat.ravel(),
            "longitude":       gate_lon.ravel(),
            "altitude":        gate_alt.ravel(),
            "x":               x_raw.ravel(),
            "y":               y_raw.ravel(),
            "z":               z_raw.ravel(),
        }))

        sweep_meta[sweep_idx] = {
            "n_azimuths": n_az,
            "n_ranges": n_rng,
            "elevation": round(elevation, 2),
        }

    df_lut = pd.concat(lut_dfs, ignore_index=True)
    logger.info(
        "LUT built: %d total gates, %d sweeps.", len(df_lut), len(sweep_meta)
    )

    radar_info = {
        "radar": radar,
        "network": network,
        "latitude": radar_lat,
        "longitude": radar_lon,
        "altitude": radar_alt,
        "sweeps": sweep_meta,
    }

    lut_dir.mkdir(parents=True, exist_ok=True)
    df_lut.to_parquet(lut_path, index=False, engine="pyarrow")
    logger.info("LUT saved -> %s", lut_path)

    with open(info_path, "w") as f:
        yaml.dump(radar_info, f, default_flow_style=False, sort_keys=False)
    logger.info("Radar info saved -> %s", info_path)

    return str(lut_path)
