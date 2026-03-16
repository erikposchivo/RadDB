import datetime
import os
import glob
from collections import OrderedDict
import math


def convert_radar_date_to_datetime(year, day_of_the_year, hour, minutes):
    """
    Converts a radar date (year, day of the year, hour, minutes) to a datetime object.

    Parameters
    ----------
    year : int
        The year.
    day_of_the_year : int
        The day of the year (1 to 366).
    hour : int
        The hour (0 to 23).
    minutes : int
        The minute (0 to 59).

    Returns
    -------
    datetime.datetime
        A datetime object representing the input radar date.

    Examples
    --------
    >>> convert_radar_date_to_datetime(2023, 102, 12, 30)
    datetime.datetime(2023, 4, 12, 12, 30)
    """
    date = datetime.date.fromordinal(datetime.date(year, 1, 1).toordinal() + day_of_the_year - 1)
    time = datetime.time(hour=hour, minute=minutes)
    return datetime.datetime.combine(date=date, time=time)


def extract_date_from_fname(file_name):
    """
    Extracts the date and time from a given file name and returns a corresponding datetime object.

    Parameters
    ----------
    file_name : str
        The base name (not full path) of the file containing the date and time information.

    Returns
    -------
    datetime.datetime
        A datetime object representing the date and time extracted from the file name.

    Examples
    --------
    >>> extract_date_from_fname('MLX1021300000U.nc')
    datetime.datetime(2010, 8, 1, 0, 0)
    """
    year = int('20' + file_name[3:5])
    day_num = int(file_name[5:8])
    hour = int(file_name[8:10])
    minutes = int(file_name[10:12])
    return convert_radar_date_to_datetime(year, day_num, hour,minutes)


def round_time_to_previous_hour(dt):
    """
    Rounds down the given datetime object to the nearest previous hour.

    Parameters
    ----------
    dt : datetime.datetime
        The datetime object to be rounded down.

    Returns
    -------
    datetime.datetime
        A new datetime object representing the nearest previoushour to the input datetime.

    Notes
    -----
    This function is called when we are looking for the first of the two HZT files whose
    content will be interpolated at 5 minutes resolution. Therefore, in the particular
    case in which the input is in the form HH:00, we will round up the hour to HH and not
    to HH-1, since we want the first HZT file in the interpolation to be the one at HH:00.

    Examples
    --------
    >>> round_time_to_previous_hour(datetime.datetime(2023, 4, 15, 12, 38))
    datetime.datetime(2023, 4, 15, 0, 0)
    """
    return datetime.datetime(dt.year, dt.month, dt.day, dt.hour)


def round_time_to_next_hour(dt):
    """
    Rounds up the given datetime object to the nearest next hour.

    Parameters
    ----------
    dt : datetime.datetime
        The datetime object to be rounded up.

    Returns
    -------
    datetime.datetime
        A new datetime object representing the nearest next hour to the input datetime.

    Examples
    --------
    >>> round_time_to_next_hour(datetime.datetime(2023, 4, 15, 12, 37))
    datetime.datetime(2023, 4, 16, 0, 0)
    """
    return datetime.datetime(dt.year, dt.month, dt.day, dt.hour) + datetime.timedelta(hours=1)


def round_time_to_previous_5_minutes(dt):
    """
    Rounds down the given datetime object to the nearest previous multiple of 5 minutes.

    Parameters
    ----------
    dt : datetime.datetime
        The datetime object to be rounded down.

    Returns
    -------
    datetime.datetime
        A new datetime object representing the nearest previous multiple of 5 minutes to the input datetime.

    Examples
    --------
    >>> round_time_to_previous_5_minutes(datetime.datetime(2023, 4, 15, 12, 38))
    datetime.datetime(2023, 4, 15, 12, 35)
    """
    minutes = dt.minute
    new_minutes = minutes // 5 * 5 
    return dt - datetime.timedelta(minutes=minutes-new_minutes)


def round_time_to_next_5_minutes(dt):
    """
    Rounds up the given datetime object to the nearest next multiple of 5 minutes.

    Parameters
    ----------
    dt : datetime.datetime
        The datetime object to be rounded up.

    Returns
    -------
    datetime.datetime
        A new datetime object representing the nearest next multiple of 5 minutes to the input datetime.

    Examples
    --------
    >>> round_time_to_next_5_minutes(datetime.datetime(2023, 4, 15, 12, 37))
    datetime.datetime(2023, 4, 15, 12, 40)
    """
    minutes = dt.minute
    new_minutes = math.ceil(minutes / 5) * 5 
    return dt + datetime.timedelta(minutes=new_minutes-minutes)


def find_status_fpath(radar_fpath):
    """
    Given the path to a radar file, returns the path to its corresponding status file.

    Parameters
    ----------
    radar_fpath : str
        The full path to the radar file.

    Returns
    -------
    str
        The full path to the status ('.xml') file.

    Examples
    --------
    >>> find_status_fpath('/path/to/MLX20100/MLX2010010300U.001')
    '/path/to/STX20100/ALX2010010300U.xml'
    """
    norm_rad_fname = os.path.normpath(radar_fpath)
    splitted_fname = norm_rad_fname.split(os.sep)
    # Replace the radar folder with the status one
    status_folder = splitted_fname[-2].replace('ML', 'ST')
    # Replace the radar filename with the status filename
    status_fname = splitted_fname[-1].replace('M', 'A').split('.')[0] + '.xml'
    # List of all full path components
    status_path_list = splitted_fname
    status_path_list[-2] = status_folder
    status_path_list[-1] = status_fname
    # Mergin path into string and returning it
    return os.sep.join(status_path_list)



def find_hydroclassif_fpath(radar_fpath):
    """
    Given the path to a radar file, returns the path to its corresponding hydrometeor
    classification file.

    Parameters
    ----------
    radar_fpath : str
        The full path to the radar file.

    Returns
    -------
    str
        The full path to file containing the hydrometeor classification.

    Examples
    --------
    >>> find_status_fpath('/path/to/MLX20100/MLX2010010300U.001')
    '/path/to/YMX20100/YMX2010010307L.801'

    References
    ----------
    The hydrometeor classification has been performed using the algorithm described by:

    Besic, N., Figueras i Ventura, J., Grazioli, J., Gabella, M., Germann, U.,
    and Berne, A.: Hydrometeor classification through statistical clustering of
    polarimetric radar measurements: a semi-supervised approach, Atmos. Meas. Tech.,
    9, 4425-4445, https://doi.org/10.5194/amt-9-4425-2016, 2016. 
    """
    norm_rad_fname = os.path.normpath(radar_fpath)
    splitted_fname = norm_rad_fname.split(os.sep)
    # Replace the radar folder with the hydroclassif one
    hydroclassif_folder = splitted_fname[-2].replace('ML', 'YM')
    # Replace the radar filename with the hydroclassif filename
    hydroclassif_name_list = splitted_fname[-1].replace('ML', 'YM').split('.')
    hydroclassif_extension = '.8' + hydroclassif_name_list[-1][1:]
    # The last number before the letter "L" may vary (example on day 21-205')
    hydroclassif_fname = hydroclassif_name_list[0][:-2] + '*L' + hydroclassif_extension
    # List of all full path components
    hydroclassif_path_list = splitted_fname
    hydroclassif_path_list[-2] = hydroclassif_folder
    hydroclassif_path_list[-1] = hydroclassif_fname
    # Mergin path into string (os specific separator)
    hydroclassif_path_merged = os.sep.join(hydroclassif_path_list)
    # Now we look for a file that matches the path above and we return it
    if len(glob.glob(hydroclassif_path_merged)) < 1:
        print(f'WARNING: no {hydroclassif_path_merged} file')
        return 'no_data'
    else:
        return glob.glob(hydroclassif_path_merged)[-1]


def find_hzt_files_at_time(all_hzt_files, date_f, verbose=False):
    """
    Given a dictionary of all HZT files and a datetime object, finds the HZT file at the hour before and after the 
    radar file. Returns a dictionary with the two files found. If either file is missing, it tries to use the available 
    file twice. If both files are missing, an empty dictionary is returned.

    Parameters
    ----------
    all_hzt_files : dict
        A dictionary containing all HZT files. The keys are datetime objects representing the timestamp when the HZT 
        file was created, and the values are the HZT file names.
    date_f : datetime.datetime
        A datetime object representing the timestamp for which the corresponding HZT files need to be found.
    verbose : bool, optional
        If True, warnings and errors will be printed to the console. Defaults to False.

    Returns
    -------
    dict
        A dictionary containing the HZT files found. The keys are datetime objects representing the timestamp when the 
        HZT file was created, and the values are the HZT file names. If both HZT files are missing, an empty dictionary is 
        returned.
    """
    # Finding the HZT file at the hour before and after the radar file
    hour_before = round_time_to_previous_hour(date_f)
    hour_after = round_time_to_next_hour(date_f)
    
    if (hour_before in all_hzt_files.keys()) and (hour_after in all_hzt_files.keys()):
        htz_files = {hour_before: all_hzt_files[hour_before],
                        hour_after: all_hzt_files[hour_after]}
    elif (hour_before in all_hzt_files.keys()) and not (hour_after in all_hzt_files.keys()):
        if verbose:
            print('WARNINGS: HZT file missing at ', hour_after)
        htz_files = {hour_before: all_hzt_files[hour_before],
                        hour_after: all_hzt_files[hour_before]}
    elif not (hour_before in all_hzt_files.keys()) and (hour_after in all_hzt_files.keys()):
        if verbose:
            print('WARNINGS: HZT file missing at ', hour_before)
        htz_files = {hour_before: all_hzt_files[hour_after],
                        hour_after: all_hzt_files[hour_after]}
    else:
        # If we don't find any, then we cannot proceed
        if verbose:
            print('ERROR: No HTZ file available, skipping time' % date_f)
        htz_files = {}
    return htz_files



def find_all_hzt_files(RADAR_BASE_DATA_DIR, year, day_start, hour_start, minute_start, day_end, hour_end, minute_end):
    """
    Find all HZT (0 degree isotherm) files for a time period, specified in year, day, hour and minute.

    Parameters:
    -----------
    year: int
        The year of the time period to search in.
    day_start: int
        The starting day of the time period to search in, as an integer in the range 1-366.
    hour_start: int
        The starting hour of the time period to search in, as an integer in the range 0-23.
    minute_start: int
        The starting minute of the time period to search in, as an integer in the range 0-59.
    day_end: int
        The ending day of the time period to search in, as an integer in the range 1-366.
    hour_end: int
        The ending hour of the time period to search in, as an integer in the range 0-23.
    minute_end: int
        The ending minute of the time period to search in, as an integer in the range 0-59.

    Returns:
    --------
    selected_hzt_files: dict
        A dictionary containing:
        - the date and time of the HTZ file (datetime.datetime object) as keys,
        - the string containing the path to the HTZ file at that date and time.

    Notes
    -----
    The output dictionary has a different structure from the one of "find_all_radar_files":
    for the current function, the datetime.dateime objects are the keys, the paths are the values.
    """
    if day_start == day_end:
        hzt_dir = os.path.join(RADAR_BASE_DATA_DIR, str(year), str(year)[2:] + str(day_start),
                                'HZT' + str(year)[2:] + str(day_start))
        all_hzt_files = sorted(glob.glob(os.path.join(hzt_dir, 'HZT*.800')))
    else:
        hzt_dir_1 = os.path.join(RADAR_BASE_DATA_DIR, str(year), str(year)[2:] + str(day_start),
                                'HZT' + str(year)[2:] + str(day_start))
        hzt_dir_2 = os.path.join(RADAR_BASE_DATA_DIR, str(year), str(year)[2:] + str(day_end),
                                'HZT' + str(year)[2:] + str(day_end))
        all_hzt_files = sorted(glob.glob(os.path.join(hzt_dir_1, 'HZT*.800'))) + sorted(glob.glob(os.path.join(hzt_dir_2, 'HZT*.800')))
    
    # Looping over all files and selecting only relevant HTZ files
    # Note: differently from radar case, in this case we have hourly files.
    date_start = round_time_to_previous_hour(convert_radar_date_to_datetime(year, day_start, hour_start, minute_start))
    date_end = round_time_to_next_hour(convert_radar_date_to_datetime(year, day_end, hour_end, minute_end))

    selected_hzt_files = OrderedDict()
    for file_path in all_hzt_files:
        date = extract_date_from_fname(os.path.basename(file_path))
        if date >= date_start and date <= date_end:
            selected_hzt_files[date] = file_path
    return selected_hzt_files


def find_all_radar_files(RADAR_BASE_DATA_DIR, radar_name, year, day_start, hour_start, minute_start, day_end, hour_end, minute_end):
    """
    Find all radar files (polar 500m) for a given radar name and time period, specified in year, day, hour and minute.

    Parameters:
    -----------
    radar_name: str
        The short name of the radar to search for (MLL, MLA, etc...).
    year: int
        The year of the time period to search in.
    day_start: int
        The starting day of the time period to search in, as an integer in the range 1-366.
    hour_start: int
        The starting hour of the time period to search in, as an integer in the range 0-23.
    minute_start: int
        The starting minute of the time period to search in, as an integer in the range 0-59.
    day_end: int
        The ending day of the time period to search in, as an integer in the range 1-366.
    hour_end: int
        The ending hour of the time period to search in, as an integer in the range 0-23.
    minute_end: int
        The ending minute of the time period to search in, as an integer in the range 0-59.

    Returns:
    --------
    selected_radar_files: dict
        A dictionary containing:
        - the string containing the path to the radar polar (500m) fileas keys,
        - the date and time of the radar file (datetime.datetime object) as values.

    Notes
    -----
    The output dictionary has a different structure from the one of "find_all_hzt_files":
    for the current function, the paths are the keys, the datetime.dateime objects are the values.
    This choice is due to the existence of multiple files at each date-time, since the time
    used in our function indicate the beginning of a scan cycle.
    """
    if day_start == day_end:
        radar_dir = os.path.join(RADAR_BASE_DATA_DIR, str(year), str(year)[2:] + str(day_start),
                                radar_name + str(year)[2:] + str(day_start))
        all_radar_files = sorted(glob.glob(os.path.join(radar_dir, radar_name + '*.*')))
    else:
        radar_dir_1 = os.path.join(RADAR_BASE_DATA_DIR, str(year), str(year)[2:] + str(day_start),
                                radar_name + str(year)[2:] + str(day_start))
        radar_dir_2 = os.path.join(RADAR_BASE_DATA_DIR, str(year), str(year)[2:] + str(day_end),
                                radar_name + str(year)[2:] + str(day_end))
        all_radar_files = sorted(glob.glob(os.path.join(radar_dir_1, radar_name + '*.*'))) + sorted(glob.glob(os.path.join(radar_dir_2, radar_name + '*.*')))

    # Looping over all files and selecting only relevant scans
    date_start = round_time_to_previous_5_minutes(convert_radar_date_to_datetime(year, day_start, hour_start, minute_start))
    date_end = round_time_to_next_5_minutes(convert_radar_date_to_datetime(year, day_end, hour_end, minute_end))

    selected_radar_files = OrderedDict()
    for file_path in all_radar_files:
        date = extract_date_from_fname(os.path.basename(file_path))
        if date >= date_start and date <= date_end:
            selected_radar_files[file_path] = date 

    return selected_radar_files