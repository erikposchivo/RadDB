import matplotlib.pyplot as plt
import radar_api
from radar_api.checks import check_time
import datetime

####---------------------------------------------------------------------------.
#### Read single metranet file 
radar_fpath = "/home/ghiggi/data/MLL2220617000U.001"
verbose = True 
radar_letter = "L"
sweep_number = 1

rad_obj, x_radar_raw, y_radar_raw, z_radar_raw  = load_metranet_sweep(radar_fpath,
                                                                      hydroclassif_fpath=None, 
                                                                      hzt_cartesian=None,
                                                                      visibility=None,
                                                                      verbose=verbose)

# TODO: check x,y,z coordinates are accurately calculated 
# --> x,y,z,lon,lat could varies as function of refractive_index (bending of the beam) 
# --> LUT for various bending could be prepared ...
####--------------------------------------------------------------------------.
#### Load static visibility fields
static_vis_dir = "/ltenas8/data/Rad4Alp_LUTs/static_vis"
vis_dict = read_static_visibility(radar_letter=radar_letter,
                                  static_vis_dir=static_vis_dir, 
                                  verbose=verbose)
visibility = vis_dict[sweep_number]

####--------------------------------------------------------------------------.
#### TODO: Rewrite visibility LUT to netCDF
# --> Create xr.DataTree and save to netCDF (with lon,lat, x,y, z, r,a coordinates)
# --> dt_visibility = xr.open_datatree(statitic_vis_filepath)
# --> dt_visibility["sweep1"].xradar_dev.plot_map() 
# --> dt_visibility["sweep1"].xradar_dev.plot_image() 
plt.imshow(vis_dict[sweep_number])
 
####--------------------------------------------------------------------------.
#### Input Data Archive 
# /ltenas8/data/RADAR/MCH/2017/03/06/16/MLA/MLA1706516000U.001
# /ltenas8/data/RADAR/MCH/2017/03/06/16/MLA/HZT (hourly 0 isotherm) 
# /ltenas8/data/RADAR/MCH/2017/03/06/16/MLA/HYM (hydrometeor class) 


####--------------------------------------------------------------------------.
#### RADB PRODUCTION SCRIPT
qpegrid_to_rad_dir = "/ltenas8/data/Rad4Alp_LUTs/qpegrid_to_rad"

# Define radar 
network = "MCH_LTE"
radar = "A" #  MLA, MLD, MLL, MLP, MLW 
start_time = "2021-02-01 12:00:00"
end_time = "2021-02-31 13:00:00"
 
# List radar files 
filepaths = radar_api.find_files(
    network=network,
    radar=radar,
    start_time=start_time,
    end_time=end_time,
    product="POL", 
    protocol="local",
)

# Group by radar volumes
# --> TODO: In radar-api add function to identify groups based on start_time, 
volumes_filepaths = radar_api.group_filepaths(filepaths, network=network, groups="volume")
 
# Process each volume 
for volume_start_time, sweep_filepaths in volumes_filepaths.items():
    
    # Find relevant hourly HZT files around radar scan volume
    volume_start_time = check_time(volume_start_time)   # ensure datetime.datetime
    hzt_start_time = volume_start_time - datetime.timedelta(hours=1)
    hzt_end_time = volume_start_time + datetime.timedelta(hours=1)
    hzt_filepaths = radar_api.find_files(
        network=network,
        radar=radar,
        start_time=hzt_start_time,
        end_time=hzt_end_time,
        product="HZT", 
        protocol="local",
    )
    # CHeck if not missing HZT files 
    # --> Maybe skip processing otherwise
    # --> See checks in find_hzt_files_at_time in .io.py  
    
    # Load zero degree isotherm height map (interpolated at volume time)
    # TODO: ADAPT CODE 
    hzt_interpolated_dict = hzt_hourly_to_5min(hzt_filepaths, tsteps_min=5)
    hzt_cartesian = hzt_interpolated_dict[date_f]    
    
    # Process METRANET sweeps data
    dict_sweeps = {}
    for sweep_filepath in sweep_filepaths:
        
        # Retrieve sweep number
        sweep_number = int(sweep_filepath[-3:])
        
        # Find operational hydrometeor classification
        # TODO: adapt logic in find_hydroclassif_fpath io.py module
        # - Given sweep filepath, define hydroclass file name and define filepath in HZT/directory
        hydroclassif_fpath = find_hydroclassif_fpath(sweep_filepath) 
        
        # Alternatively we could: 
        # hydroclassif_fpath = radar_api.find_files(
        #     network=network,
        #     radar=radar,
        #     start_time=sweep_start_time,
        #     end_time=sweep_end_time,
        #     product="HYM", 
        #     protocol="local",
        # )
                
        # Retrieve visibility 
        visibility = vis_dict[sweep_number] # TODO: check if index start at 0 or at 1 for vis_dict
        # visibility = dt_visibility[sweep_number].data
        
        # Process data with pyart 
        rad_obj, x_radar_raw, y_radar_raw, z_radar_raw  = load_metranet_sweep(sweep_filepath,
                                                                              hydroclassif_fpath=hydroclassif_fpath, 
                                                                              hzt_cartesian=hzt_cartesian,
                                                                              visibility=visibility,
                                                                              verbose=verbose)
        # Converto to xradar dataset 
        ds = convert_pyart_to_xradar_dataset(rad_obj) # TODO implement
        # Ensure we have all info required by xradar data format 
        
        # Add to dictionary
        dict_sweeps[f"sweep_{sweep_number}"] = ds
        
    ####----------------------------------------------.
    #### Create xradar DataTree
    dt = xr.DataTree.from_dict(dict_sweeps)
    
    ####----------------------------------------------.
    #### Archive to RadDB 
    # --> to be wrapped into rad.archive_xradar(dt, radar, radb_dir)  

    # [MAYBE NOT NECESSARY TO CREATE DATATREE FOR RADB ARCHIVE PRODUCTION ]
    # for sweep_numer, ds in dict_sweeps.item():
    #      df = ds.to_dataframe().reset_index()
    #      # ...
    
    # List sweeps group 
    # --> Need to remove sweep_fixed_angle, volume_number and other groups
    list_sweeps = [s for s in dt if re.fullmatch(r"sweep_\d+", s)]
    
    # Convert to dataset
    list_df = []
    for sweep in list_sweeps:  # dict_sweeps.
        ds = dt[sweep].to_dataset() 
        # Convert to pandas
        df = ds.to_dataframe().reset_index()
        df["sweep"] = int(sweep.split("_"))
        list_df.append(df) 
    
    df = pd.concat(list_df)
    
    # Define gate_id, ... 
    
    # Remove x,y,z,lon,lat, r,a,z
    # --> We will need to save it just once when we have full volume to have it as LUT
    # --> Need to save the LUT before filtering !
    
    # Filter row for e.g. reflectivity < 0 
    # ... 
    
    # Define filename 
    # ...
    
    # Define filepath 
    filepath = "/tmp/.parquet"
    
    # Save to parquet
    df.to_parquet("/tmp/.parquet")


####----------------------------------------------------------------------
#### Hydroclassification
# NC = not classified
# AG = aggregates
# CR = ice crystals
# LR = light rain
# RP = rimed particles
# RN = rain
# VI = vertically oriented ice
# WS = wet snow
# MH = melting hail
# IH/HDG = dry hail / high density graupel

####--------------------------------------------------------------------
#### Define dummy DataTree
import time
import numpy as np
import pandas as pd
import xarray as xr
 
# Create coordinates
time = pd.date_range("2025-01-01", periods=5, freq="h")
range_vals = np.arange(0, 1000, 200)

# Create dummy datasets
ds1 = xr.Dataset(
    {
        "reflectivity": (("time", "range"), np.random.rand(len(time), len(range_vals))),
        "velocity": (("time", "range"), np.random.randn(len(time), len(range_vals))),
    },
    coords={
        "time": time,
        "range": range_vals,
    },
    attrs={"sweep_id": 1}
)

ds2 = xr.Dataset(
    {
        "reflectivity": (("time", "range"), np.random.rand(len(time), len(range_vals))),
        "velocity": (("time", "range"), np.random.randn(len(time), len(range_vals))),
    },
    coords={
        "time": time,
        "range": range_vals,
    },
    attrs={"sweep_id": 2}
)

# Create DataTree with groups
dt = xr.DataTree.from_dict({"sweep_1": ds1, "sweep_2": ds2})
 
####--------------------------------------------------------------------.



