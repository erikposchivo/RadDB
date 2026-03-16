import pandas as pd
from pathlib import Path
import glob

# for the moement it takes all *POLAR.parquet files inside the base_path directory and concatenate into one single dataframe
# next step: find a way ro train the UMAP, sine we cannot concatenate all the data into one single dataframe (OOM)
def read_parquet_dataset(base_path, pattern="**/*POLAR.parquet", columns=None, verbose=True):
    """
    Reads Parquet files recursively from a base directory into a single DataFrame.
    
    Parameters
    ----------
    base_path : str
        Root directory containing the parquet files.
    pattern : str
        Glob pattern to match files (default matches POLAR files from your archive).
    columns : list, optional
        List of columns to read.
        
    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame.
    """
    path = Path(base_path)
    files = list(path.rglob(pattern))
    
    if not files:
        print(f"No files found matching {pattern} in {base_path}")
        return pd.DataFrame()
    
    if verbose:
        print(f"Found {len(files)} files. Loading...")
        
    # Read files
    df_list = [pd.read_parquet(f, columns=columns) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    
    return df

def check_dataframe(df):
    """Prints basic summary of the DataFrame."""
    print("-" * 30)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("-" * 30)
    print("Missing values:\n", df.isnull().sum())
    print("-" * 30)
    print(f"Head:\n{df.head()}")
    print("-" * 30)
    print(f"Tail:\n{df.tail()}")
    print("-" * 30)
    print(f"info:\n{df.info()}")
    print("-" * 30)
    print(f"describe:\n{df.describe()}")
    print("-" * 30)

