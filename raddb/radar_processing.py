"""
raddb/radar_processing.py
--------------------------
PyART radar object processing functions for METRANET data.
Adapted from RadarProcessing_example_code/functions.py
"""
from __future__ import annotations
import datetime
import os
import pickle
import logging
import numpy as np
import pyart
from pyart.aux_io import read_cartesian_metranet
from pyart.correct import calculate_attenuation_zphi, smooth_phidp_single_window
from pyart.retrieve import kdp_leastsquare_single_window
from scipy.spatial import KDTree
import xmltodict

logger = logging.getLogger(__name__)

RADAR_LETTERS = ['A', 'D', 'L', 'P', 'W']

# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

def check_radar_letter(radar_letter: str) -> bool:
    """Check if a given radar-identification letter is valid."""
    return radar_letter.upper() in RADAR_LETTERS


def read_status(status_file: str, verbose: bool = False) -> dict:
    """Read a radar XML status file."""
    try:
        status = xmltodict.parse(open(status_file, 'r').read().replace('-P/', '-P_'))
        return status
    except Exception as e:
        if verbose:
            logger.warning(f'ERROR reading status file {status_file}: {e}')
        return {}


def read_qpegrid_to_rad(radar_letter: str, qpegrid_to_rad_dir: str, verbose: bool = False) -> dict:
    """Read lookup table containing the Swiss grid (used for QPE)."""
    if check_radar_letter(radar_letter):
        vis_fname = f'lut_qpegrid_to_rad{radar_letter.upper()}.p'
        vis_content = pickle.load(open(os.path.join(qpegrid_to_rad_dir, vis_fname), 'rb'))
        return vis_content
    if verbose:
        logger.warning(f'ERROR: Impossible to read radar visibility, radar letter {radar_letter} is not valid.')
    return {}


def read_static_visibility(radar_letter: str, static_vis_dir: str, verbose: bool = False) -> dict:
    """Return the static visibility dictionary for a given radar."""
    if check_radar_letter(radar_letter):
        vis_fname = f'lut_visibility_rad{radar_letter.upper()}.p'
        vis_content = pickle.load(open(os.path.join(static_vis_dir, vis_fname), 'rb'))
        return vis_content
    if verbose:
        logger.warning(f'ERROR: Impossible to read radar visibility, radar letter {radar_letter} is not valid.')
    return {}


# ================================================================================
# HZT INTERPOLATION
# ================================================================================

def hzt_hourly_to_5min(filedict: dict, tsteps_min: int = 5) -> dict:
    """Interpolate hourly isothermal fields to 5-minute resolution."""
    if not filedict or len(filedict) < 2:
        logger.warning(f"Need at least 2 HZT files for interpolation, got {len(filedict)}")
        return {}

    timestep_list = sorted(filedict.keys())
    tstamp_hzt0 = timestep_list[0]
    tstamp_hzt1 = timestep_list[1]

    hzt = {}
    try:
        hzt[tstamp_hzt0] = read_cartesian_metranet(
            filedict[tstamp_hzt0], reader="python"
        ).fields['iso0_height']['data'][0]
        hzt[tstamp_hzt1] = read_cartesian_metranet(
            filedict[tstamp_hzt1], reader="python"
        ).fields['iso0_height']['data'][0]
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


# ================================================================================
# NOISE COMPUTATION
# ================================================================================

def compute_noise(rad_obj, status_fpath: str, sweep_number: int, verbose: bool = False) -> None:
    """Compute noise estimate for a radar object."""
    NOISE_100 = 5

    if not os.path.isfile(status_fpath):
        if verbose:
            logger.warning('Cannot find status file. Using default noise value.')
        noisedBADU_h = NOISE_100
        noisedBADU_v = NOISE_100
    else:
        status = read_status(status_fpath)
        if not len(status.keys()):
            if verbose:
                logger.warning('Cannot read status file. Using default noise value.')
            noisedBADU_h = NOISE_100
            noisedBADU_v = NOISE_100
        else:
            try:
                noise_h = float(status['status']['sweep'][sweep_number]['RADAR']['STAT']
                                ['CALIB']['noisepower_frontend_h_inuse']['@value'])
                rconst_h = float(status['status']['sweep'][sweep_number]['RADAR']['STAT']
                                 ['CALIB']['rconst_h']['@value'])
                noisedBADU_h = 10. * np.log10(noise_h) + rconst_h

                noise_v = float(status['status']['sweep'][sweep_number]['RADAR']['STAT']
                                ['CALIB']['noisepower_frontend_v_inuse']['@value'])
                rconst_v = float(status['status']['sweep'][sweep_number]['RADAR']['STAT']
                                 ['CALIB']['rconst_v']['@value'])
                noisedBADU_v = 10. * np.log10(noise_v) + rconst_v
            except Exception:
                noisedBADU_h = NOISE_100
                noisedBADU_v = NOISE_100

    noisedBZ_h = pyart.retrieve.compute_noisedBZ(
        rad_obj.nrays, noisedBADU_h, rad_obj.range['data'], 100., noise_field='noisedBZ_hh'
    )
    noisedBZ_v = pyart.retrieve.compute_noisedBZ(
        rad_obj.nrays, noisedBADU_v, rad_obj.range['data'], 100., noise_field='noisedBZ_vv'
    )

    noisedBZ_h['data'] = np.ma.array(noisedBZ_h['data'], mask=np.isnan(noisedBZ_h['data']))
    noisedBZ_v['data'] = np.ma.array(noisedBZ_v['data'], mask=np.isnan(noisedBZ_v['data']))

    rad_obj.add_field('noise_h', noisedBZ_h)
    rad_obj.add_field('noise_v', noisedBZ_v)


# ================================================================================
# CARTESIAN COORDINATES CORRECTION
# ================================================================================

def correct_gate_cartesian_coordinates(rad_obj, ke: float = 1.25):
    """Convert radar antenna vectors to Cartesian coordinates using Swiss standard."""
    x_radar_raw, y_radar_raw, z_radar_raw = pyart.core.transforms.antenna_vectors_to_cartesian(
        rad_obj.range['data'],
        rad_obj.azimuth['data'],
        rad_obj.elevation['data'],
        ke=ke
    )
    return x_radar_raw, y_radar_raw, z_radar_raw


# ================================================================================
# VISIBILITY
# ================================================================================

def add_visibility(rad_obj, visibility: np.ndarray) -> None:
    """Add visibility field to radar object."""
    z_h = rad_obj.get_field(0, 'reflectivity')
    visib_sweep = np.ma.array(visibility.astype(np.float32), mask=np.isnan(visibility))
    if visib_sweep.shape[1] < z_h.shape[1]:
        raise ValueError("Reflectivity shape does not match the expected one")
    visib_sweep = visib_sweep[0:z_h.shape[0], 0:z_h.shape[1]]
    rad_obj.add_field('visibility', {'data': visib_sweep})


def correct_reflectivity_for_visibility(
    rad_obj, visibility: np.ndarray, minimum_visibility: int = 37, maximum_correction_dbz: float = 2
) -> None:
    """Mask radar data at low visibility and correct reflectivity."""
    z_h = rad_obj.get_field(0, 'reflectivity')
    z_v = rad_obj.get_field(0, 'reflectivity_vv')

    corr = np.full(visibility.shape, np.nan, dtype=np.float32)
    positive_vis = visibility > 0
    corr[positive_vis] = 1. / (visibility[positive_vis] / 100.)
    corr[corr >= maximum_correction_dbz] = maximum_correction_dbz

    visib_mask = visibility < minimum_visibility

    z_h_linear = 10 ** (0.1 * z_h)
    z_h_corrected = 10. * np.log10(z_h_linear * corr)

    z_v_linear = 10 ** (0.1 * z_v)
    z_v_corrected = 10. * np.log10(z_v_linear * corr)

    z_h_vis_corrected = np.ma.array(z_h_corrected.astype(np.float32), mask=visib_mask)
    z_v_vis_corrected = np.ma.array(z_v_corrected.astype(np.float32), mask=visib_mask)
    visib_mask_for_rdr = np.ma.array(visib_mask, mask=np.isnan(visib_mask))

    rad_obj.add_field('reflectivity_visibilitycorr', {'data': z_h_vis_corrected})
    rad_obj.add_field('reflectivity_vv_visibilitycorr', {'data': z_v_vis_corrected})
    rad_obj.add_field('visibility_mask', {'data': visib_mask_for_rdr})


# ================================================================================
# KDP COMPUTATION
# ================================================================================

def compute_kdp(
    rad_obj, kdp_rmin: float, kdp_rmax: float, kdp_rcell: float,
    kdp_rwind: float, kdp_zmin: float, kdp_zmax: float
) -> None:
    """Compute KDP using moving least-square algorithm."""
    r = rad_obj.range['data']
    ind_rmin = np.where(r > kdp_rmin)[0][0]
    ind_rmax = np.where(r < kdp_rmax)[0][-1]
    r_res = r[1] - r[0]
    min_rcons = int(kdp_rcell / r_res)
    wind_len = int(kdp_rwind / r_res)
    min_valid = int(wind_len / 2 + 1)

    psidp_field = 'uncorrected_differential_phase'
    if "reflectivity_visibilitycorr" in rad_obj.fields:
        refl_field = 'reflectivity_visibilitycorr'
    else:
        refl_field = "reflectivity"
    phidp_field = 'propagation_differential_phase'
    kdp_field = 'specific_differential_phase'

    phidp = smooth_phidp_single_window(
        rad_obj, ind_rmin=ind_rmin, ind_rmax=ind_rmax, min_rcons=min_rcons,
        zmin=kdp_zmin, zmax=kdp_zmax, wind_len=wind_len, min_valid=min_valid,
        psidp_field=psidp_field, refl_field=refl_field, phidp_field=phidp_field
    )
    rad_obj.add_field(phidp_field, phidp)

    kdp = kdp_leastsquare_single_window(
        rad_obj, wind_len=wind_len, min_valid=min_valid,
        phidp_field=phidp_field, kdp_field=kdp_field, vectorize=True
    )
    rad_obj.add_field(kdp_field, kdp)


# ================================================================================
# ATTENUATION CORRECTION
# ================================================================================

def correct_attenuation(rad_obj) -> None:
    """Correct for attenuation using ZPHI algorithm."""
    ah, pia, zh_corr, _, pida, zdr_corr = calculate_attenuation_zphi(
        rad_obj,
        refl_field='reflectivity_visibilitycorr',
        zdr_field='differential_reflectivity',
        phidp_field='propagation_differential_phase',
        iso0_field='height_over_iso0',
        temp_ref='height_over_iso0',
        doc=15
    )
    rad_obj.add_field('attenuation_h', ah)
    rad_obj.add_field('corrected_reflectivity', zh_corr)
    rad_obj.add_field('corrected_differential_reflectivity', zdr_corr)

    zv_corr = pia['data'] - pida['data'] + rad_obj.get_field(0, 'reflectivity_vv')
    rad_obj.add_field('corrected_reflectivity_vv', {'data': zv_corr})


# ================================================================================
# HYDROMETEOR CLASSIFICATION
# ================================================================================

def add_hydroclass_from_file(rad_obj, hydroclassif_fpath: str) -> None:
    """Add hydrometeor classification from file."""
    if hydroclassif_fpath != 'no_data' and hydroclassif_fpath is not None:
        try:
            hydroclassif = pyart.aux_io.read_file_py(hydroclassif_fpath, moment='ZDRP')
            hydroclassif_scaled = np.array(hydroclassif.data / 25)
            hydroclassif_masked = np.ma.array(
                hydroclassif_scaled.astype(np.uint8),
                mask=np.logical_or(hydroclassif_scaled < 0, hydroclassif_scaled > 9)
            )
        except Exception as e:
            logger.warning(f"Error reading hydroclass file {hydroclassif_fpath}: {e}")
            nan_data = np.full((rad_obj.nrays, rad_obj.ngates), np.nan, dtype=np.float32)
            hydroclassif_masked = np.ma.array(nan_data, mask=np.isnan(nan_data))
    else:
        nan_data = np.full((rad_obj.nrays, rad_obj.ngates), np.nan, dtype=np.float32)
        hydroclassif_masked = np.ma.array(nan_data, mask=np.isnan(nan_data))
    rad_obj.add_field('hydrometeor_classification_mch', {'data': hydroclassif_masked})


def compute_hydroclass_semisupervised(rad_obj) -> None:
    """Compute hydrometeor classification using PyART semi-supervised method."""
    if "height_over_iso0" not in rad_obj.fields:
        logger.warning("Cannot compute PyART hydroclass: height_over_iso0 field missing")
        return

    try:
        hydro = pyart.retrieve.hydroclass_semisupervised(
            rad_obj,
            hydro_names=("AG", "CR", "LR", "RP", "RN", "VI", "WS", "MH", "IH"),
            var_names=("dBZ", "ZDR", "KDP", "RhoHV", "H_ISO0"),
            refl_field="reflectivity_visibilitycorr" if "reflectivity_visibilitycorr" in rad_obj.fields else "reflectivity",
            zdr_field="differential_reflectivity",
            kdp_field="specific_differential_phase",
            rhv_field="uncorrected_cross_correlation_ratio",
            iso0_field="height_over_iso0",
        )["hydro"]
        rad_obj.add_field("radar_echo_classification", hydro)
    except Exception as e:
        logger.warning(f"Error computing PyART hydroclass: {e}")


# ================================================================================
# HZT DATA ADDITION
# ================================================================================

def add_hzt_data(
    rad_obj, hzt_cartesian: np.ndarray, radar_letter: str,
    sweep_number: int, qpegrid_to_rad_dir: str, verbose: bool = False
) -> None:
    """Add Height of Freezing Level (HZT) field to radar object."""
    NBINS_Y = 710
    NBINS_X = 640
    Y_SWISS_GRID = np.linspace(255, 965, NBINS_Y + 1)
    X_SWISS_GRID = np.linspace(480, -160, NBINS_X + 1)

    qpegrid_to_rad = read_qpegrid_to_rad(radar_letter, qpegrid_to_rad_dir, verbose)
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
    hzt_pol_field[:, 0:hzt_pol.shape[1]] = hzt_pol

    hzt_dict = {
        'data': hzt_pol_field, 'units': 'm',
        'long_name': 'Height of freezing level',
        'standard_name': 'HZT'
    }
    rad_obj.add_field('iso0_height', hzt_dict)


def add_height_over_iso0(rad_obj, z_gates: np.ndarray) -> None:
    """Add height above freezing level to radar object."""
    height_over_iso0 = z_gates - rad_obj.fields['iso0_height']['data']
    height_over_iso0_masked = np.ma.array(
        height_over_iso0.astype(np.float32),
        mask=np.isnan(height_over_iso0)
    )

    iso0_dict = {
        'data': height_over_iso0_masked, 'units': 'm',
        'long_name': 'height of freezing level with respect to radar gate altitude',
        'standard_name': 'height_over_iso0'
    }
    rad_obj.add_field('height_over_iso0', iso0_dict)

    lapse_rate = -6.5  # C/km
    temperature = 0 + lapse_rate * height_over_iso0 / 1000
    temperature = np.ma.array(temperature.astype(np.float32), mask=np.isnan(temperature))
    temp_dict = {
        'data': temperature, 'units': 'C',
        'long_name': 'Temperature',
        'standard_name': 'air_temperature'
    }
    rad_obj.add_field("temperature", temp_dict)


# ================================================================================
# COMPLETE SWEEP PROCESSING
# ================================================================================

def load_metranet_sweep(
    radar_fpath: str,
    hydroclassif_fpath: str | None = None,
    hzt_cartesian: np.ndarray | None = None,
    visibility: np.ndarray | None = None,
    qpegrid_to_rad_dir: str | None = None,
    compute_pyart_hc: bool = True,
    verbose: bool = False
):
    """Load and process METRANET sweep data."""
    rad_obj = pyart.aux_io.read_metranet(radar_fpath, reader="C", nbytes=4)

    radar_filename = os.path.basename(radar_fpath)
    radar_letter = radar_filename[2]
    sweep_number = int(radar_filename.split('.')[-1])

    x_radar_raw, y_radar_raw, z_radar_raw = correct_gate_cartesian_coordinates(rad_obj)

    # Visibility correction
    if visibility is not None:
        add_visibility(rad_obj, visibility)
        correct_reflectivity_for_visibility(rad_obj, visibility, minimum_visibility=37, maximum_correction_dbz=2)

    # KDP computation
    KDP_RMIN, KDP_RMAX, KDP_RCELL = 1000., 50000., 1000.
    KDP_ZMIN, KDP_ZMAX, KDP_RWIND = 20., 40., 6000.
    compute_kdp(rad_obj, kdp_rmin=KDP_RMIN, kdp_rmax=KDP_RMAX, kdp_rcell=KDP_RCELL,
                kdp_rwind=KDP_RWIND, kdp_zmin=KDP_ZMIN, kdp_zmax=KDP_ZMAX)

    # HZT and attenuation
    if hzt_cartesian is not None and qpegrid_to_rad_dir is not None:
        add_hzt_data(rad_obj, hzt_cartesian, radar_letter, sweep_number,
                     qpegrid_to_rad_dir=qpegrid_to_rad_dir, verbose=verbose)
        z_gates = rad_obj.altitude['data'][0] + z_radar_raw
        add_height_over_iso0(rad_obj, z_gates)
        correct_attenuation(rad_obj)

    # Hydrometeor classification
    add_hydroclass_from_file(rad_obj, hydroclassif_fpath)

    if compute_pyart_hc and "height_over_iso0" in rad_obj.fields:
        compute_hydroclass_semisupervised(rad_obj)

    # Ensure all expected fields exist — fill with NaN if not computed
    _expected_fields = ['iso0_height', 'hydrometeor_classification_mch', 'radar_echo_classification']
    for field in _expected_fields:
        if field not in rad_obj.fields:
            nan_data = np.full((rad_obj.nrays, rad_obj.ngates), np.nan, dtype=np.float32)
            rad_obj.add_field(field, {'data': np.ma.array(nan_data, mask=np.isnan(nan_data))})

    return rad_obj, x_radar_raw, y_radar_raw, z_radar_raw
