import datetime
import os
import glob
import pickle 
import pyart
import numpy as np
from pyart.aux_io import read_cartesian_metranet
from pyart.correct import calculate_attenuation_zphi
from pyart.correct import smooth_phidp_single_window
from pyart.retrieve import kdp_leastsquare_single_window
from scipy.spatial import KDTree
import xmltodict


RADAR_LETTERS = ['A', 'D', 'L', 'P', 'W']


def check_radar_letter(radar_letter):
    """
    Check if a given radar-identification letter (single character) is valid.

    Parameters
    ----------
    radar_letter : str
        A string representing the radar letter.W

    Returns
    -------
    bool
        True if the radar letter is valid (i.e., 'A', 'D', 'L', 'P', or 'W'),
        False otherwise.
    """
    if radar_letter.upper() in RADAR_LETTERS:
        return True
    return False



def read_status(status_file, verbose=False):
    """
    Reads a radar xml status file.
    
    Parameters
    ----------
    fname : str
        Full path of the status xml file to be read
    add_wet_Radome : boolean (optional)
        For older files, there is not information about the wet radome. 
        If this is true, the script will estimate the wet radome precipitation
        as a 3 x 3 mean of the RZC product at the given time (as is done for
        more recent files)
        
    Returns
    -------
    dict
        The status as a Python dictionary.

    Notes
    -----
    This function is based on the `read_status` function of the rainforest library:
    https://github.com/MeteoSwiss/rainforest/blob/5faa02d8ab1d5ba0c494a05381f5eaf96104f0ff/rainforest/common/io_data.py
    The wet radome computation has been removed.
    If errors appear while reading the xml file, an empty dictionary is returned
    """
    # Reads a xml status file 
    try:
        status = xmltodict.parse(open(status_file,'r').read().replace('-P/','-P_'))
    except:
        if verbose:
            print('ERROR reading status file %s.' % status_file)
        return {}

    # Differently from RainForest, we don't look at wet radome
    return status


def read_qpegrid_to_rad(radar_letter, qpegrid_to_rad_dir, verbose=False):
    """
    Reads a lookup table containing the Swiss grid (used in RainForest for QPE.)
    
    Parameters
    ----------
    fname : str
        Full path of the status xml file to be read
    add_wet_Radome : boolean (optional)
        For older files, there is not information about the wet radome. 
        If this is true, the script will estimate the wet radome precipitation
        as a 3 x 3 mean of the RZC product at the given time (as is done for
        more recent files)
        
    Returns
    -------
    dict
        The status as a Python dictionary.

    Notes
    -----
    This function is based on the `get_lookup` function of the rainforest library:
    https://github.com/MeteoSwiss/rainforest/blob/5faa02d8ab1d5ba0c494a05381f5eaf96104f0ff/rainforest/common/lookup.py
    The wet radome computation has been removed.
    """
    if check_radar_letter(radar_letter):
        vis_fname = 'lut_qpegrid_to_rad%s.p' % radar_letter.upper()
        vis_content = pickle.load(open(os.path.join(qpegrid_to_rad_dir, vis_fname),'rb'))
        return vis_content
    # If the letter does not identify a radar, we return an empty dictionary
    if verbose:
        print('ERROR: Impossible to read radar visibility, radar letter %s is not valid.' % radar_letter)
    return {}


def hzt_hourly_to_5min(filedict, tsteps_min=5):
    """
    Interpolate hourly isothermal fields to 5-minute resolution.

    Parameters
    ----------
    filedict : dict
        A dictionary containing file paths (str), with their date and time 
        (datetime.datetime) as keys.
    tsteps_min : int, optional
        The time interval between consecutive interpolated values, in minutes.
        Default is 5.

    Returns
    -------
    hzt : dict
        A dictionary containing date and time (datetime.datetime) as keys and 
        interpolated isothermal heights as values, at 5-minute intervals.

    Notes
    -----
    This function reads the isothermal fields from the given file paths, and
    interpolates them to create a time series at a resolution of 5 minutes. The
    interpolation is performed by calculating the incremental difference between
    the isothermal heights at the first two timestamps, and then adding this
    increment to the isothermal height at each subsequent timestamp. The resulting
    time series is returned as a dictionary.
    """
    # Datetime.datetime of the two HZT files
    timestep_list = list(filedict.keys())
    tstamp_hzt0 = timestep_list[0]
    tstamp_hzt1 = timestep_list[1]

    # Reading the HZT files
    # test = read_cartesian_metranet('/t5500/ltenas8/mch/Radar/2021/21193/HZT21193/HZT2119321000L.801',
    #                                            reader=READER).fields['iso0_height']['data'][0]
    hzt = {}
    
    hzt[tstamp_hzt0] = read_cartesian_metranet(filedict[tstamp_hzt0],
                                            reader="python").fields['iso0_height']['data'][0]
    hzt[tstamp_hzt1] = read_cartesian_metranet(filedict[tstamp_hzt1],
                                            reader="python").fields['iso0_height']['data'][0]
    
    # data_ex = read_cartesian_metranet(filedict[tstamp_hzt1], reader="python")
    # print(data_ex.metadata)
    
    # Get the incremental difference for e.g. 5min steps (divided by 12):
    dt = datetime.timedelta(minutes=tsteps_min)
    ndt = np.arange(1,int(60/tsteps_min))
    deltaHZT = (hzt[tstamp_hzt1]-hzt[tstamp_hzt0])/ (len(ndt)+1)

    # Loop through all min increments and add the calculated increment of deltaHZT
    for idx in ndt:
        if idx == ndt[0]:
            deltaHZT_temp = deltaHZT.copy()
        else:
            deltaHZT_temp += deltaHZT
        hzt[tstamp_hzt0+dt*idx] = hzt[tstamp_hzt0]+deltaHZT_temp
    return hzt

def compute_noise(rad_obj, status_fpath, sweep_number, verbose=False):
    """
    Computes a noise estimate for a given radar object based on the calibration data 
    contained in the status dictionary for a given sweep. 

    Parameters
    -----------
    rad_obj : object
        A pyart radar object containing range and other data for a single sweep.
    status_fpath : str
        The full path to the status file corresponding to the current radar file.
    sweep_number : int
        An integer indicating which sweep in the radar object to use for calibration data.
    verbose : bool, optional
        If True, print warning messages to the console. Default is False.

    Returns
    --------
    None

    Notes
    -----
    This function is based on the `compute_noise` function of the rainforest library:
    https://github.com/MeteoSwiss/rainforest/blob/5faa02d8ab1d5ba0c494a05381f5eaf96104f0ff/rainforest/common/radarprocessing.py

    The function modifies the radar object in place by adding the following fields:
        noise_h : array
            The estimated noise on the horizontal polarization channel.
        noise_v : array
            The estimated noise on the vertical polarization channel.

    """
    NOISE_100 = 5 # Noise level at 100 km
    
    if not os.path.isfile(status_fpath):
        # If the file does not exist, we use the default noise
        if verbose:
            print('WARNING: Cannot find status file for current scan. Using default value.')
        noisedBADU_h = NOISE_100
        noisedBADU_v = NOISE_100
    else:
        # If the status file exists we proceed
        status = read_status(status_fpath)

        if not len(status.keys()):
            # If problem occurr during reading, we use the default noise
            if verbose:
                print('WARNING: Cannot find status file for current scan. Using default value.')
            noisedBADU_h = NOISE_100
            noisedBADU_v = NOISE_100
        else:
            # Otherwise we try to read the noise from the status dictionary
            try:
                noise_h = float(status['status']['sweep'][sweep_number]['RADAR']['STAT']
                                        ['CALIB']['noisepower_frontend_h_inuse']['@value'])
                rconst_h = float(status['status']['sweep'][sweep_number]['RADAR']['STAT']
                                            ['CALIB']['rconst_h']['@value'])
                noisedBADU_h = 10.*np.log10(noise_h) + rconst_h

                noise_v = float(status['status']['sweep'][sweep_number]['RADAR']['STAT']
                                        ['CALIB']['noisepower_frontend_v_inuse']['@value'])
                rconst_v = float(status['status']['sweep'][sweep_number]['RADAR']['STAT']
                                        ['CALIB']['rconst_v']['@value'])
                noisedBADU_v = 10.*np.log10(noise_v) + rconst_v
            except:
                # If the file exist and we can't load all needed quantities, we use the default noise
                noisedBADU_h = NOISE_100
                noisedBADU_v = NOISE_100
 
    noisedBZ_h = pyart.retrieve.compute_noisedBZ(rad_obj.nrays, noisedBADU_h,
            rad_obj.range['data'], 100.,
            noise_field='noisedBZ_hh')
        
    noisedBZ_v = pyart.retrieve.compute_noisedBZ(rad_obj.nrays, noisedBADU_v,
            rad_obj.range['data'], 100.,
            noise_field='noisedBZ_vv')
    
    # Convert to masked array for consistency
    noisedBZ_h['data'] = np.ma.array(noisedBZ_h['data'], 
                                mask = np.isnan(noisedBZ_h['data'])) 
    noisedBZ_v['data'] = np.ma.array(noisedBZ_v['data'], 
                                mask = np.isnan(noisedBZ_v['data']))    
    
    # Adding it to the radar object
    rad_obj.add_field('noise_h', noisedBZ_h)
    rad_obj.add_field('noise_v', noisedBZ_v)


def correct_gate_cartesian_coordinates(rad_obj, ke = 1.25):
    """
    Convert radar antenna vectors to Cartesian coordinates using Swiss standard for gate altitude correction.

    Parameters
    ----------
    rad_obj : Py-ART Radar object
        Py-ART Radar object containing (at least) range, azimuth and elevation data.

    Returns
    -------
    x_radar_raw : ndarray
        Cartesian x-coordinate of radar gate locations corrected for altitude.
    y_radar_raw : ndarray
        Cartesian y-coordinate of radar gate locations corrected for altitude.
    z_radar_raw : ndarray
        Cartesian z-coordinate of radar gate locations corrected for altitude.
    ke: float 
        Scale factor for the effective radius of Earth


    Notes
    -----
    According to the Swiss standard, the default gate_altitude given by `rad_obj.get_gate_x_y_z(0)` is not used. 
    Instead of the constant radar scale factor of (ke) of 4/3, a value of 1.25 is used for the gate altitude correction.

    This function is based on the `correct_gate_altitude` function of the rainforest library:
    https://github.com/MeteoSwiss/rainforest/blob/5faa02d8ab1d5ba0c494a05381f5eaf96104f0ff/rainforest/common/radarprocessing.py
    """
    x_radar_raw, \
        y_radar_raw, \
            z_radar_raw = pyart.core.transforms.antenna_vectors_to_cartesian(rad_obj.range['data'],
                                                                             rad_obj.azimuth['data'],
                                                                             rad_obj.elevation['data'],
                                                                             ke = 1.25)
    return x_radar_raw, y_radar_raw, z_radar_raw


def read_static_visibility(radar_letter, static_vis_dir, verbose=False):
    """
    Return the static visibility dictionary for a given radar.

    Parameters
    ----------
    radar_letter : str
        A string representing the radar letter ('A', 'D', 'L', 'P', or 'W').
    verbose : bool, optional
        If True, print an error message when the radar letter is invalid.
        Default is False.

    Returns
    -------
    dict
        A dictionary containing the static visibility values for the given
        radar letter.

    Notes
    -----
    This function is based on the `get_lookup` function of the rainforest library:
    https://github.com/MeteoSwiss/rainforest/blob/5faa02d8ab1d5ba0c494a05381f5eaf96104f0ff/rainforest/common/lookup.py
    The wet radome computation has been removed.
    """
    if check_radar_letter(radar_letter):
        vis_fname = 'lut_visibility_rad%s.p' % radar_letter.upper()
        vis_content = pickle.load(open(os.path.join(static_vis_dir, vis_fname),'rb'))
        return vis_content
    # If the letter does not identify a radar, we return an empty dictionary
    if verbose:
        print('ERROR: Impossible to read radar visibility, radar letter %s is not valid.' % radar_letter)
    return {}

   
def add_visibility(rad_obj, visibility):
    z_h = rad_obj.get_field(0, 'reflectivity')
    visib_sweep = np.ma.array(visibility.astype(np.float32), 
                              mask = np.isnan(visibility))
    if visib_sweep.shape[1] < z_h.shape[1]:
        # Sometimes there are weird sizes, so we stop the processing
        raise ValueError("Reflectivity shape does not match the expected one for {radar_fpath}")
   
    # If the size is compatible with the static visibility, we continue
    visib_sweep = visib_sweep[0:z_h.shape[0], 0:z_h.shape[1]]
    rad_obj.add_field('visibility', {'data': visib_sweep})
    

def correct_reflectivity_for_visibility(rad_obj, visibility, minimum_visibility=37, maximum_correction_dbz=2): 
    """Mask the radar data at low visibility and correct the reflectivity (horizontal and vertical).

    Parameters
    ----------
    rad_obj : pyart.radar
        The reflectivity data in dBZ for horizontal polarization.
    visibility : numpy.ndarray
        The visibility data in percent.
    minimum_visibility : int
        The minimal visibility below which the data is masked.
    maximum_correction_dbz : float
        The maximum correction factor in dBZ. The visibility correction
        is 100 / VISIB (with VISIB in %) and can be thresholded with this
        parameter. This is usually set to 2 at MeteoSwiss.

    Returns
    -------
    visib_mask : numpy.ndarray
        A boolean mask where True values represent data below the minimum visibility.
    z_h_corrected : numpy.ndarray
        The reflectivity data in dBZ for horizontal polarization, corrected
        for visibility.
    z_v_corrected : numpy.ndarray
        The reflectivity data in dBZ for vertical polarization, corrected
        for visibility.

    Notes
    -----
    This function is based on the `compute_kdp` function of the rainforest library:
    https://github.com/MeteoSwiss/rainforest/blob/5faa02d8ab1d5ba0c494a05381f5eaf96104f0ff/rainforest/common/radarprocessing.py

    The correction is computed as follows:
        corr = 1 / (visib / 100.)
    The reflectivity values are then corrected as follows:
        z_h_corrected = 10. * np.log10(z_h_linear * corr)
        z_v_corrected = 10. * np.log10(z_v_linear * corr)
    """
    # Maximum correction for Z_H and Z_V from visibility
    # Minimum accepted static visibility in percent
    
    z_h = rad_obj.get_field(0, 'reflectivity')
    z_v = rad_obj.get_field(0, 'reflectivity_vv')
    
    # Computing the correction from the visibilty
    corr = np.full(visibility.shape, np.nan, dtype=np.float32)
    positive_vis = visibility > 0 # To avoid warning
    corr[positive_vis] = 1./(visibility[positive_vis]/100.)
    
    # Imposing a maximum threshold to the correction
    corr[corr >= maximum_correction_dbz] = maximum_correction_dbz

    # Defining a mask based on minimum visibility
    visib_mask = visibility < minimum_visibility
    
    # Correcting and masking Z_H
    z_h_linear = 10 ** (0.1 * z_h)
    z_h_corrected = 10. * np.log10(z_h_linear * corr)

    # Correcting and masking Z_V
    z_v_linear = 10 ** (0.1 * z_v)
    z_v_corrected = 10. * np.log10(z_v_linear * corr)
    
    z_h_vis_corrected = np.ma.array(z_h_corrected.astype(np.float32), 
                                   mask = visib_mask)
    z_v_vis_corrected = np.ma.array(z_v_corrected.astype(np.float32), 
                                   mask = visib_mask)
    visib_mask_for_rdr = np.ma.array(visib_mask, 
                                     mask = np.isnan(visib_mask))
    rad_obj.add_field('reflectivity_visibilitycorr', {'data': z_h_vis_corrected})
    rad_obj.add_field('reflectivity_vv_visibilitycorr', {'data': z_v_vis_corrected})
    rad_obj.add_field('visibility_mask', {'data': visib_mask_for_rdr})


def compute_kdp(rad_obj, kdp_rmin, kdp_rmax, kdp_rcell, kdp_rwind, kdp_zmin, kdp_zmax):
    """
    Computes KDP using the simple moving least-square algorithm.

    Parameters
    ----------
    rad_obj : object
        Py-ART radar object.
    kdp_rmin : float
        Minimum range (in m) index where to look for continuous precipitation.
    kdp_rmax : float
        Maximum range (in m) index where to look for continuous precipitation.
    kdp_rcell : float
        Minimum length (in m) of consecutive gates to consider it a rain cell.
    kdp_rwind : float
        Width of the smoothing window used to compute KDP (in m).
    kdp_zmin : float
        Minimum reflectivity factor (in dBZ) to consider it a rain cell.
    kdp_zmax : float
        Maximum reflectivity factor (in dBZ) to consider it a rain cell.

    Returns
    -------
    None

    Notes
    -----
    This function is based on the `correct_attenuation` function of the rainforest library:
    https://github.com/MeteoSwiss/rainforest/blob/5faa02d8ab1d5ba0c494a05381f5eaf96104f0ff/rainforest/common/radarprocessing.py

    The function modifies the radar object in place by adding the following fields:
        propagation_differential_phase : array
            The estimated propagation differential phase.
        specific_differential_phase : array
            The estimated specific differential phase on propagation.
    """
    # Loading parameters
    r = rad_obj.range['data']
    ind_rmin = np.where(r > kdp_rmin)[0][0]
    ind_rmax = np.where(r < kdp_rmax)[0][-1]
    r_res = r[1] - r[0]
    min_rcons = int(kdp_rcell/r_res)
    wind_len = int(kdp_rwind/r_res)
    min_valid = int(wind_len/2+1)

    # Defining filed names
    psidp_field = 'uncorrected_differential_phase'
    if "reflectivity_visibilitycorr" in rad_obj.fields:
        refl_field = 'reflectivity_visibilitycorr'
    else: 
        refl_field = "reflectivity"
    phidp_field = 'propagation_differential_phase'
    kdp_field = 'specific_differential_phase'

    # Smooth Phidp
    phidp = smooth_phidp_single_window(rad_obj, ind_rmin=ind_rmin,
                                       ind_rmax=ind_rmax, min_rcons=min_rcons,
                                       zmin=kdp_zmin, zmax=kdp_zmax, wind_len=wind_len,
                                       min_valid=min_valid, psidp_field=psidp_field,
                                       refl_field=refl_field,phidp_field=phidp_field)
    
    rad_obj.add_field(phidp_field, phidp)
    
    # Compute KDP
    kdp = kdp_leastsquare_single_window(rad_obj, wind_len=wind_len, min_valid=min_valid, 
                                        phidp_field=phidp_field, kdp_field=kdp_field, 
                                        vectorize = True)
    
    rad_obj.add_field(kdp_field, kdp)


def correct_attenuation(rad_obj):
    """
    Corrects for attenuation using the ZPHI algorithm (Testud et al.)
    using the 0° isothermal altitude.

    Parameters
    ----------
    rad_obj : Radar
        The radar object containing the data to be corrected.

    Returns
    -------
    None

    Notes
    -----
    This function is based on the `correct_attenuation` function of the rainforest library:
    https://github.com/MeteoSwiss/rainforest/blob/5faa02d8ab1d5ba0c494a05381f5eaf96104f0ff/rainforest/common/radarprocessing.py

    The function modifies the radar object in place by adding the following fields:
        attenuation_h : array
            The estimated attenuation in dB/km for horizontal polarization.
        reflectivity_corr : array
            The corrected reflectivity for horizontal polarization.
        differential_reflectivity_corr : array
            The corrected differential reflectivity for horizontal polarization.
        reflectivity_vv_corr : array
            The corrected reflectivity for vertical polarization.
    """
    # Computing the correction
    ah, pia, zh_corr, _,\
        pida, zdr_corr = calculate_attenuation_zphi(rad_obj,
                                                    refl_field='reflectivity_visibilitycorr',
                                                    zdr_field = 'differential_reflectivity', # visibility_corr?
                                                    phidp_field = 'propagation_differential_phase',
                                                    iso0_field = 'height_over_iso0',
                                                    temp_ref = 'height_over_iso0',
                                                    doc = 15)
    # Adding the fields to the radar object
    rad_obj.add_field('attenuation_h', ah)
    rad_obj.add_field('corrected_reflectivity', zh_corr)
    rad_obj.add_field('corrected_differential_reflectivity', zdr_corr)
    
    # Computing correction for V channel
    zv_corr = pia['data'] - pida['data'] + rad_obj.get_field(0, 'reflectivity_vv')
    rad_obj.add_field('corrected_reflectivity_vv', {'data': zv_corr})


def add_hydroclass_from_file(rad_obj, hydroclassif_fpath):
    if hydroclassif_fpath != 'no_data':
        hydroclassif = pyart.aux_io.read_file_py(hydroclassif_fpath, moment='ZDRP')
        hydroclassif_scaled = np.array(hydroclassif.data/25)
        hydroclassif_masked = np.ma.array(hydroclassif_scaled.astype(np.uint8), 
                                        mask = np.logical_or(hydroclassif_scaled < 0, hydroclassif_scaled > 9))
    else:
        hydroclassif_masked = np.zeros_like(rad_obj.get_field(0, 'reflectivity'))
    rad_obj.add_field('hydrometeor_classification', {'data': hydroclassif_masked})
   
    
def add_hzt_data(rad_obj, hzt_cartesian, radar_letter, sweep_number, 
                 qpegrid_to_rad_dir, verbose=False):
    """
    Add the Height of Freezing Level (HZT) field data to a Py-ART Radar object.

    Parameters
    ----------
    rad_obj : Py-ART Radar object
        Py-ART Radar object containing radar data.
    hzt_cartesian : ndarray
        Array of Cartesian coordinates of HZT data.
    radar_letter : str
        Letter indicating which radar the data comes from ('A', 'D', 'L', 'P', or 'W').
    sweep_number : int
        Sweep number of the current file (1-20)
    verbose : bool, optional
        If True, print additional information to the console. Default is False.

    Returns
    -------
    None

    Notes
    -----
    The Height of Freezing Level (HZT) field within the radar field of view is computed from the
    Cartesian coordinates of HZT data and added to the `rad_obj` under the name 'iso0_height'.
    The HZT data is interpolated using nearest neighbor interpolation to fill any missing data points.

    The resulting field has the following attributes:
        - data type: ndarray
        - units: meters
        - long_name: Height of freezing level
        - standard_name: HZT
    
    This function is based on the `add_hzt_data` function of the rainforest library:
    https://github.com/MeteoSwiss/rainforest/blob/5faa02d8ab1d5ba0c494a05381f5eaf96104f0ff/rainforest/common/radarprocessing.py
    """
    # Swiss grid info
    NBINS_Y = 710
    NBINS_X = 640
    Y_SWISS_GRID = np.linspace(255, 965, NBINS_Y + 1)
    X_SWISS_GRID = np.linspace(480, -160,  NBINS_X + 1)

    # Get lookup table for the Swiss grid at the chosen radar
    qpegrid_to_rad = read_qpegrid_to_rad(radar_letter, qpegrid_to_rad_dir, verbose)

    # Lookup tables the sweeps are labelled 0-19, while in our indexing the sweeps are 1-20 # TODO CHECK WHAT WE DO
    lut_sweep = qpegrid_to_rad[qpegrid_to_rad[:,0] == (sweep_number-1)]
    nrange = lut_sweep[:,2].max() + 1 
    naz = lut_sweep[:,1].max() + 1

    # Get Cartesian and polar indexes
    idxx = (lut_sweep[:,-1] - X_SWISS_GRID[-1]).astype(int) - 1 
    idxy = (lut_sweep[:,-2] - Y_SWISS_GRID[0]).astype(int) - 1

    idxaz = lut_sweep[:,1]
    idxrange = lut_sweep[:,2]

    # Initialize polar arrays
    hzt_pol = np.zeros((naz, nrange))
    npts = np.zeros((naz, nrange))

    # Get part of Cart HZT that covers radar
    toadd = hzt_cartesian[idxx.ravel(), idxy.ravel()]
    
    # Update grid
    hzt_pol[idxaz.ravel(), idxrange.ravel()] += toadd
    npts[idxaz.ravel(), idxrange.ravel()] += np.ones(toadd.shape)
    
    # To avoid a division trhough 0, which causes a python runtime warning:
    npts[npts == 0] = np.nan
    hzt_pol /= npts

    # Fill holes with nearest neighbour interpolation
    x,y=np.mgrid[0:naz, 0:nrange]
    xygood = np.array((x[~np.isnan(hzt_pol)],
                    y[~np.isnan(hzt_pol)])).T
    xybad = np.array((x[np.isnan(hzt_pol)],
                    y[np.isnan(hzt_pol)])).T
    
    hzt_pol[np.isnan(hzt_pol)] = hzt_pol[~np.isnan(hzt_pol)][
        KDTree(xygood).query(xybad)[1]]

    # Assure same sized fields
    hzt_pol_field = np.zeros((rad_obj.nrays, rad_obj.ngates)) + np.nan
    hzt_pol_field[:,0:hzt_pol.shape[1]] = hzt_pol

    hzt_dict = {'data':hzt_pol_field, 'units':'m',
                'long_name':'Height of freezing level',
                'standard_name' :'HZT'}

    rad_obj.add_field('iso0_height', hzt_dict)



def add_height_over_iso0(rad_obj, z_gates):
    """
    Aadd the height above the freezing level of each gate to a Py-ART Radar object.

    Parameters
    ----------
    rad_obj : Py-ART Radar object
        Py-ART Radar object containing iso0_height field data.
    z_gates : ndarray
        Array of gate altitudes.

    Returns
    -------
    None

    Notes
    -----
    The height of the radar gates altitude with respect to the freezing level is calculated as the difference 
    between the gate altitudes in `z_gates` and the iso0_height field data in `rad_obj`. The resulting field 
    is added to the `rad_obj` under the name `height_over_iso0`.

    The resulting field has the following attributes:
        - data type: ndarray
        - units: meters
        - long_name: height of freezing level with respect to radar gate altitude
        - standard_name: height_over_iso0

    This function is based on the `add_height_over_iso0` function of the rainforest library:
    https://github.com/MeteoSwiss/rainforest/blob/5faa02d8ab1d5ba0c494a05381f5eaf96104f0ff/rainforest/common/radarprocessing.py
    """
    # Add heigh over isotherm 0 degree
    height_over_iso0 = z_gates - rad_obj.fields['iso0_height']['data']
    height_over_iso0_masked = np.ma.array(height_over_iso0.astype(np.float32), 
                                          mask = np.isnan(height_over_iso0))
    
    iso0_dict = {'data':height_over_iso0_masked, 'units':'m', 
                 'long_name':'height of freezing level with respect to radar gate altitude',
                 'standard_name' :'height_over_iso0'}
    
    rad_obj.add_field('height_over_iso0', iso0_dict)
    
    # Add temperature 
    lapse_rate = -6.5 # C/km
    temperature = 0 + lapse_rate * height_over_iso0/1000
    temperature = np.ma.array(temperature.astype(np.float32), 
                                          mask = np.isnan(temperature))
    iso0_dict = {'data': temperature, 'units':'C', 
                 'long_name':'Temperature',
                 'standard_name' :'air_temperature'}
    rad_obj.add_field("temperature", temperature)


def load_metranet_sweep(radar_fpath, 
                        hydroclassif_fpath=None, 
                        hzt_cartesian=None,
                        visibility=None,
                        verbose=False):
    """
    Load and corrects METRANET sweep data.

    Parameters
    ----------
    radar_fpath : str
        Filepath of the radar data file.
    hzt_cartesian : Tuple[float, float]
        Cartesian coordinates of the isotherm 0 (in meters) in (x, y) format.
    visibility: np.array 
        Visibility array for the sweep
    verbose : bool, optional
        Whether to print information about the process (default is False).
    
    Returns
    -------
    rad_obj : pyart.core.Radar
        Radar object with corrected data fields.
    x_radar_raw : numpy.ndarray
        x coordinate of the gates, computed by the function "correct_gate_cartesian_coordinates".
    y_radar_raw : numpy.ndarray
        y coordinate of the gates, computed by the function "correct_gate_cartesian_coordinates".
    z_radar_raw : numpy.ndarray
        z coordinate of the gates, computed by the function "correct_gate_cartesian_coordinates".

    Notes
    -----
    This function performs the following operations on the radar data:
        1. Reads the radar file.
        2. Loads the status file.
        3. Computes the noise on H and V.
        4. Identifies radar and sweep number.
        5. Loads and corrects x, y, z coordinates.
        6. Loads the static visibility.
        7. Masks and corrects reflectivity using the visibility.
        8. Computes KDP.
        9. Loads the hydrometeor classification.
        10. Adds the height of the isotherm 0 to the radar object.
        11. Adds the height_above_iso0 as field in radar object.
        12. Corrects attenuation on H, V and differential attenuation.

    This function is inspired by several functions in the "radarprocessing" code of the rainforest library:
    https://github.com/MeteoSwiss/rainforest/blob/5faa02d8ab1d5ba0c494a05381f5eaf96104f0ff/rainforest/common/radarprocessing.py
    """
    # Reading the radar file
    rad_obj = pyart.aux_io.read_metranet(radar_fpath, reader="python", nbytes=4)
    
    # Retrieve timestep
    # time_str = radar_fpath[-10:-6]

    # ----------------------------------------------------------------------------------
    # TO BE IMPLEMENTED:
    # mask: masked = clutter (nan) or no precip (0 converted to value)
    # Set no precip to nan either, to avoid that they become "precip" after visibility and attenuation correction
    # ----------------------------------------------------------------------------------

    # Checking which scan number are we at
    sweep_number = int(radar_fpath.split('.')[-1])

    # Finding the status file
     # status_fpath = find_status_fpath(radar_fpath)

    # Computing noise on H and V
    # compute_noise(rad_obj, status_fpath, sweep_number, verbose=verbose)

    # Identifying radar and sweep number
    radar_filename = os.path.basename(radar_fpath)
    radar_letter = radar_filename[2]
    sweep_number = int(radar_filename.split('.')[-1])
    
    # Loading and correcting x,y,z coordinates
    x_radar_raw, y_radar_raw, z_radar_raw = correct_gate_cartesian_coordinates(rad_obj)
    
    #-------------------------------------------------
    # Add static visibility and correct reflectivity for visibility
    if visibility is not None:
        add_visibility(rad_obj)
        
        # Add reflectivity_visibilitycorr, reflectivity_vv_visibilitycorr, visibility_mask
        correct_reflectivity_for_visibility(rad_obj, visibility, minimum_visibility=37, max_correction_db=2)
     
    #-------------------------------------------------
    # Computing KDP
    KDP_RMIN = 1000.
    KDP_RMAX = 50000.
    KDP_RCELL = 1000.
    KDP_ZMIN = 20.
    KDP_ZMAX = 40.
    KDP_RWIND = 6000.
    compute_kdp(rad_obj, kdp_rmin=KDP_RMIN, kdp_rmax=KDP_RMAX, kdp_rcell=KDP_RCELL,
                kdp_rwind=KDP_RWIND, kdp_zmin=KDP_ZMIN, kdp_zmax=KDP_ZMAX)    
    
    #-------------------------------------------------
    # If isotherm available, add iso0_height, height_above_iso0, temperature and correct for attenuation
    if hzt_cartesian is not None:
        # Adding the height of the isotherm 0 to the radar object
        # --> Add iso0_height variable to pyart radar object
        qpegrid_to_rad_dir = "/ltenas8/data/Rad4Alp_LUTs/qpegrid_to_rad"
        add_hzt_data(rad_obj, hzt_cartesian, radar_letter, sweep_number, 
                     qpegrid_to_rad_dir=qpegrid_to_rad_dir, verbose=False)
        
        # Adding the height_above_iso0 as field in radar object
        # -- > Negative value below 0 isotherm
        z_gates = rad_obj.altitude['data'][0] + z_radar_raw
        add_height_over_iso0(rad_obj, z_gates)
                
        # Correcting attenuation on H, V and differential attenuation
        # ATTENTION: iso0_height required !
        correct_attenuation(rad_obj)

    #-------------------------------------------------
    # Add hydroclass from file 
    if hydroclassif_fpath is not None:
        add_hydroclass_from_file(rad_obj, hydroclassif_fpath=hydroclassif_fpath)
    elif "height_above_iso0" in rad_obj.fields:
        # Compute hydrometeor classification
        hydro = pyart.retrieve.hydroclass_semisupervised(
            rad_obj,
            hydro_names=("AG", "CR", "LR", "RP", "RN", "VI", "WS", "MH", "IH"),
            var_names=("dBZ", "ZDR", "KDP", "RhoHV", "H_ISO0"),
            refl_field="reflectivity", # reflectivity_corr
            zdr_field="differential_reflectivity", # "corrected_differential_reflectivity",
            kdp_field="specific_differential_phase",
            rhv_field="uncorrected_cross_correlation_ratio",
            iso0_field="height_above_iso0",
            # temp_field="temperature",
        )["hydro"]
        rad_obj.add_field("radar_echo_classification", hydro)

    #-------------------------------------------------
    # Return radar object and cartesian coordinates
    return rad_obj, x_radar_raw, y_radar_raw, z_radar_raw 

