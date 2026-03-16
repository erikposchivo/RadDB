#%%
# autoreload, remove after testing
#%load_ext autoreload
#%autoreload 2

# standard library
import os
import re
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import concurrent.futures # for parallel processing
from collections import defaultdict

# radar specific libraries
import xarray as xr
import xradar as xd

# MCH pyart fork: pyart_mch
import pyart

# feature-local-archive branch (Gionata Ghiggi)
import radar_api
#from radar_api.utils.xradar import get_mch_datatree_from_pyart


#%%
network = "MCH_LTE"
radar = "L"
product = "POL" # products: ["HYM", "HZT", "POL"] 
start_time = "2021-08-28 06:00:00"
end_time = "2021-08-28 18:20:00" 

#%%
# used to check, the files, this function is inside archive_metranet_to_parquet()
'''filepaths = radar_api.find_files(
    network=network,
    radar=radar,
    start_time=start_time,
    end_time=end_time,
    product=product,
    verbose=True,
    protocol="local",
)

print(f"Number of files found: {len(filepaths)}")'''


#%%
def create_volume_dataframe(metranet_filepaths, network="MCH_LTE", product="POL"):
    """
    1. Converts a list of METRANET sweep filepaths (1 volume scan) into a single DataFrame.
       Reindexes sweeps from 1 to 20 based on their sorted order.
    """
    list_df = []
    
    # Sort to ensure we process .001 through .020 in order
    sorted_filepaths = sorted(metranet_filepaths)
    
    for sweep_idx, sweep_filepath in enumerate(sorted_filepaths, start=1):        
        # Open datatree directly using radar_api
        dt = radar_api.open_datatree(sweep_filepath, network=network, product=product)
        # Extract the sweep group (e.g., 'sweep_0')
        list_sweeps = [s for s in dt if re.fullmatch(r"sweep_\d+", s)]
        sweep_name = list_sweeps[0] if list_sweeps else None

        if sweep_name:
            ds = dt[sweep_name].to_dataset()
            df_sweep = ds.to_dataframe().reset_index()
        else:
            df_sweep = dt.to_dataset().to_dataframe().reset_index()
        
        # Reindex sweep from 1 to 20
        df_sweep["sweep"] = sweep_idx
        list_df.append(df_sweep)
        
    # Concatenate all 20 sweeps into one volume DataFrame
    volume_df = pd.concat(list_df, ignore_index=True)
    return volume_df

# test on first 20 metranet files: 1 volume scan
#volume_df = create_volume_dataframe(filepaths[:20], network=network, product=product)

#%%
def filter_and_split_volume(df, radar_name, z_col="DBZH"):
    """
    2. Filters out empty sky (Z <= 0), generates gate_ids, and splits the data
       into a spatial LUT DataFrame and a Polarimetric DataFrame.
    """    
    # Filter: Remove Z <= 0 (Keep strictly > 0)
    df_filtered = df[df[z_col] > 0].copy()
    
    # Use min time infofor gate_id col
    date_str = pd.to_datetime(df_filtered["time"].min()).strftime("%Y-%m-%d_%H:%M:%S")
    
    # Create unique gate_id for the remaining points
    df_filtered["gate_id"] = (
        f"{radar_name}_{date_str}_" + 
        "s" + df_filtered["sweep"].astype(str) +
        "_a" + df_filtered["azimuth"].round(1).astype(str) + 
        "_r" + df_filtered["range"].astype(str)
    )
    #print(f"gate_id example: {df_filtered['gate_id'].iloc[0]}")
    
    # Split the DataFrame
    lut_columns = ["gate_id", "latitude", "longitude", "altitude", "elevation", "range", "azimuth", "sweep"]
    pol_columns = ["gate_id", "time", "DBZH", "ZDR", "RHOHV", "PHIDP"]

    # Safely get LUT columns that exist
    lut_cols_present = [c for c in lut_columns if c in df_filtered.columns]
    df_lut = df_filtered[lut_cols_present]
    
    polar_cols_present = [c for c in pol_columns if c in df_filtered.columns]
    df_polar = df_filtered[polar_cols_present]

    # Return volume_time as well, so your saving function knows where to route the files
    return df_lut, df_polar

# test using volume_df from previous step
#df_lut, df_polar = filter_and_split_volume(volume_df, radar_name=radar, z_col="DBZH")

#%%

def save_volume_parquet(df_lut, df_polar, radar_name, base_output_path):
    """
    3. Saves the split DataFrames to Parquet files in the structured archive directory.
    Extracts the volume time directly from the polarimetric dataframe.
    """
    # Extract the volume start time directly from the polarimetric dataframe
    # since df_polar retained the exact 'time' column.
    volume_time = pd.to_datetime(df_polar["time"].min())
    
    # Build directory: base_output_path / radar / YYYY / MM / DD
    save_dir = Path(base_output_path) / radar_name / str(volume_time.year) / f"{volume_time.month:02d}" / f"{volume_time.day:02d}"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a clean string for the filename (avoiding colons which can break some filesystems)
    filename_date_str = volume_time.strftime("%Y%m%d_%H%M%S")
    
    lut_filepath = save_dir / f"{radar_name}_{filename_date_str}_LUT.parquet"
    polar_filepath = save_dir / f"{radar_name}_{filename_date_str}_POLAR.parquet"
    
    # Save using pyarrow engine for maximum speed
    df_lut.to_parquet(lut_filepath, index=False, engine="pyarrow")
    df_polar.to_parquet(polar_filepath, index=False, engine="pyarrow")
    
    return str(lut_filepath), str(polar_filepath)


#base_output_path = "/ltenas8/users/giacobbi/raddb/test_output"
#lut_path, polar_path = save_volume_parquet(df_lut, df_polar, radar_name=radar, base_output_path=base_output_path)


#%%

def group_filepaths_by_volume(filepaths):
    """Groups METRANET filepaths into volumes based on their base filename."""
    volumes_dict = defaultdict(list)
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        volume_id = filename.split('.')[0] # e.g., 'MLL2124017550U'
        volumes_dict[volume_id].append(filepath)
    return volumes_dict

def _worker_process_volume(args):
    """
    Helper function to run on a single core. Unpacks arguments and runs the 3 steps.
    """
    sweep_filepaths, volume_time, network, radar, product, base_output_path = args
    
    try:
        # Step 1
        df_volume = create_volume_dataframe(sweep_filepaths, network=network, product=product)
        # Step 2
        df_lut, df_polar = filter_and_split_volume(df_volume, radar_name=radar, z_col="DBZH")
        # Step 3
        save_volume_parquet(df_lut, df_polar, radar, base_output_path)
        
        return f"Success: {volume_time} ({len(df_polar)} gates kept)"
    except Exception as e:
        return f"Error on {volume_time}: {str(e)}"

def archive_metranet_to_parquet(network, radar, product, start_time, end_time, base_output_path, max_workers=2):
    """
    Finds files, groups them into volumes, and processes each volume scan in parallel.
    """
    # Find all files in the time range
    filepaths = radar_api.find_files(
        network=network,
        radar=radar,
        start_time=start_time,
        end_time=end_time,
        product=product,
        verbose=True,
        protocol="local",
    )
    
    if not filepaths:
        print("No files found.")
        return

    # Group filepaths into discrete volumes using our local fix
    volumes_filepaths = group_filepaths_by_volume(filepaths)
    print(f"Grouped into {len(volumes_filepaths)} volumes based on filename patterns.")

    # Prepare the arguments for each parallel worker
    tasks = []
    for volume_start_time, sweep_paths in volumes_filepaths.items():
        tasks.append((sweep_paths, volume_start_time, network, radar, product, base_output_path))

    # Execute in parallel
    # Note: Adjust max_workers based on your machine's CPU cores and RAM limits
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_worker_process_volume, tasks))
        
    for res in results:
        print(res)
        
    print("Archive complete.")


#%% TEST ENTIRE PROCESS

# test archive function on small list of files (1 volume scan)
base_output_path = "/ltenas8/users/giacobbi/raddb/test_output"
archive_metranet_to_parquet(
    network=network,
    radar=radar,
    product=product,
    start_time=start_time,
    end_time=end_time,
    base_output_path=base_output_path,
    max_workers=4, 
)


