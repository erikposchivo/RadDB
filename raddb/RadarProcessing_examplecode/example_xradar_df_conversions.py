import re 
import pandas as pd
import numpy as np
import xarray as xr
import radar_api
#import gpm.gv.xradar # to load xradar_dev accessor

radar_api.available_networks(only_online=False)
radar_api.available_networks(only_online=True)

radar_api.available_radars(only_online=False)
radar_api.available_radars(network="IDEAM")
radar_api.available_products(network="MCH_LTE")

#-----------------------------------------------------------------------------.
radar = "fiika"
network = "FMI"
start_time = "2021-09-07 16:20:00"
end_time = "2021-09-07 17:22:00"  
filepaths = radar_api.find_files(
    network=network,
    radar=radar,
    start_time=start_time,
    end_time=end_time,
    verbose=True,
    protocol="local",
)

network = "MCH_LTE"
radar = "L"
product = "POL" # products: ["HYM", "HZT", "POL"] 

#selected time range (UTC)
start_time = "2021-08-28 18:00:00"
end_time = "2021-08-28 18:15:00" 

filepaths = radar_api.find_files(
    network=network,
    radar=radar,
    start_time=start_time,
    end_time=end_time,
    product=product,
    verbose=True,
    protocol="local",
)
  
#-----------------------------------------------------------------------------.
# Currently radar_api open single volume 
# FUTURE NICETOHAVE key in etc/network/<product>.yaml
# - combine_sweeps --> Combine Dataset sweeps into DataTree volume
# - combine_volumes --> Combine DataTree volumes along time
# --> Keys poiting to function doing the heavy lifting to combine stuffs (specific to each radar)

#-----------------------------------------------------------------------------.
#### Conversion of single volume to xradar xr.DataTree

# Open FMI Volume with xradar DataTree
dt = radar_api.open_datatree(filepaths[0], network=network, product='POL')

pza = radar_api.open_pyart(filepaths[0], network=network, product='POL')


# List sweeps group 
# --> Need to remove sweep_fixed_angle, volume_number and other groups
list_sweeps = [s for s in dt if re.fullmatch(r"sweep_\d+", s)]

# Convert to dataframe
list_df = []
for sweep_number in list_sweeps:
    ds = dt[sweep_number].to_dataset() 
    # Convert to pandas
    df = ds.to_dataframe().reset_index()
    df["sweep"] = sweep_number 
    list_df.append(df) 

df = pd.concat(list_df)

# Conversion back to xr.DataTree
dict_ds = {}
for sweep in np.unique(df["sweep"]):
    ds_sweep = df[df["sweep"] == sweep].set_index(["azimuth", "range"]).to_xarray()
    dict_ds[sweep] = ds_sweep
dt_rec = xr.DataTree.from_dict(dict_ds)

# Compare structure (try to make more similar as possible)
dt_rec["sweep_0"]
dt["sweep_0"]

# Ensure plot_map works
dt["sweep_0"].to_dataset()["DBZH"].xradar_dev.plot_map()  # OK
dt["sweep_0"].to_dataset().coords

dt_rec["sweep_0"].to_dataset().coords # latitude, longitude, altitude missing !
# --> These is the location of the radar ... when converting from radDB back to xradar, 
#     we need to attach this info (from e.g. a yaml file containing this info)
#     --> In radDB we will save parquet with radar_variable, LUT of gate geolocation, radar_info.yaml file
dt_rec["sweep_0"].to_dataset()["DBZH"].xradar_dev.plot_map()   # fail without radar geolocation

ds = dt_rec["sweep_0"].to_dataset()
ds = ds.assign_coords(dt["sweep_0"].to_dataset().coords)
ds["DBZH"].xradar_dev.plot_map()   # now it works 

ds["HCLASS"].xradar_dev.plot_map()  # plot hydroclass (we could compare also against finnish hydroclass)


#-----------------------------------------------------------------------------.
#### Conversion after filtering df 
df_subset = df[df["DBZH"] > 0] # to simulate actual content of RadDB
print((1 - len(df_subset) / len(df)) * 100) # reduction percentage 

# Now just for example purpose high threshold to show possible problem arising
df_subset = df[df["DBZH"] > 20]
 
# reconstruct one sweep for example
sweep = "sweep_0"
ds_sweep = df_subset[df_subset["sweep"] == sweep].set_index(["azimuth", "range"]).to_xarray() 

# Compare reconstructed vs original 
ds_sweep.sizes
dict_ds[sweep].sizes # different dimensions shape ... because any gate at range 992 in filtered dataset is present 

ds_sweep["range"]
dict_ds[sweep]["range"]

ds_sweep["azimuth"]
dict_ds[sweep]["azimuth"]

# In extreme case, no data at given sweep at certain timestep
# In many case, missing range or azimuth in reconstructed version
# --> Because in the filtered df any gate at given azimuth is present 
# --> We must ensure to reconstruct always the original full shape sweep
# --> Otherwise 
#     - to_xarray use only (range, azimuth) coordinate left in df
#     - plotting results in bad stuffs (neighbour azimuth e.g. are not in reality neighbouring)
#     - shape can varies across timesteps and can lead to problem during concatenation 
# - We need to set_index to have all az-range indices for each given sweep (shape vary across sweep number !)

# Retrieve full index
full_index = (
    df[df["sweep"] == sweep]
    .set_index(["azimuth", "range"])
    .index
)

# Use reindex with full index
ds_sweep_rec = df_subset[df_subset["sweep"] == sweep].set_index(["azimuth", "range"]).reindex(full_index).to_xarray()

ds_sweep_rec.sizes
dict_ds[sweep].sizes 

# We will need to define full_index from LUT or radar_info.yaml info we store 

#-----------------------------------------------------------------------------.
#### Conversion of multiple volumes from radDB to xr.DataTree (with time dimension)
list_time_df = []
for filepath in filepaths[0:2]:
    # Open FMI Volume with xradar DataTree
    dt = radar_api.open_datatree(filepath, network=network)
    
    # List sweeps group 
    # --> Need to remove sweep_fixed_angle, volume_number and other groups
    list_sweeps = [s for s in dt if re.fullmatch(r"sweep_\d+", s)]

    # Convert to dataframe
    list_df = []
    for sweep_number in list_sweeps:
        ds = dt[sweep_number].to_dataset() 
        # Convert to pandas
        df = ds.to_dataframe().reset_index()
        df["sweep"] = sweep_number 
        list_df.append(df) 
    
    df = pd.concat(list_df)
    df["sweep_time"] = df["time"] # or azimuth_start_time # sweep_start_time
    df["time"] = df["sweep_time"].min() # or one that make sense (e.g. filename)
    
    list_time_df.append(df)

df = pd.concat(list_time_df)

# Conversion back to xr.DataTree
dict_ds = {}
for sweep in np.unique(df["sweep"]):
    ds_sweep = df[df["sweep"] == sweep].set_index(["azimuth", "range", "time"]).to_xarray()
    dict_ds[sweep] = ds_sweep
dt_temporal = xr.DataTree.from_dict(dict_ds)

dt_temporal["sweep_0"].sizes # time dimension !

#-----------------------------------------------------------------------------.



