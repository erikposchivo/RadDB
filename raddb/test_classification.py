#%%
import pandas as pd
from utils import read_parquet_dataset, check_dataframe
from ml import normalize_data, apply_umap, PolyClassifier, HDBSCANClassifier, GMMClassifier
from vis import plot_umap

#%%
# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = "/ltenas8/users/giacobbi/raddb/test_output"
# Features to use for UMAP (Polarimetric variables)
FEATURES = ["DBZH", "ZDR", "RHOHV", "PHIDP"] 

# Testing configuration for date, NEXT STEP: AUTOMATIZE THE ENTIRE PROCESS
DATA_DIR += "/L/2021/08/28/"

#%%
# -----------------------------------------------------------------------------
# 1. Read Data
# -----------------------------------------------------------------------------
print("Step 1: Reading Parquet files...")
# We specifically target POLAR files to get the variables
df = read_parquet_dataset(DATA_DIR, pattern="**/*POLAR.parquet")
#check_dataframe(df)

# Drop NaNs to ensure UMAP works
#print("Missing values before dropping NaNs:\n", df.isnull().sum())
df = df.dropna(subset=FEATURES)
#print("Missing values after dropping NaNs:\n", df.isnull().sum())

if df.empty:
    raise ValueError("DataFrame is empty. Check your data path.")

#%%
# -----------------------------------------------------------------------------
# 2. Data Transformation (Normalization)
# -----------------------------------------------------------------------------
print("Step 2: Normalizing data...")
df_scaled, scaler = normalize_data(df, FEATURES, method="minmax")

#check the scaled data
#print(f"Scaler parameters: {scaler.get_params()}")
print("Scaled data summary:")
print(f"Scaled data summary: {df_scaled[FEATURES].describe()}")

#%%
# -----------------------------------------------------------------------------
# 3. Apply UMAP
# -----------------------------------------------------------------------------
print("Step 3: Applying UMAP reduction...")
# Using a small subset for testing speed if needed, remove .sample() for full run
#df_sample = df_scaled.sample(n=min(10000, len(df_scaled)), random_state=42)
embedding, reducer = apply_umap(df_scaled, FEATURES, n_neighbors=15, min_dist=0.1)

#%%
# -----------------------------------------------------------------------------
# 4. Plot Reprojected Data (Unclassified)
# -----------------------------------------------------------------------------
print("Step 4: Plotting unclassified data...")
plot_umap(embedding, title="UMAP Projection (Unclassified)")

#%%
# -----------------------------------------------------------------------------
# 5. Classify using PolyClassifier
# -----------------------------------------------------------------------------
print("Step 5: Classifying data...")

# Example Polygons, this is just dummy coordinates for demonstration
example_polygons = {
    "Light Rain": [(-5, -5), (-5, 0), (0, 0), (0, -5)],
    "Heavy Rain": [(0, 0), (0, 5), (5, 5), (5, 0)]
}

classifier = PolyClassifier(classes_polygons=example_polygons)
labels = classifier.predict(embedding)

# Plot Classified Data
plot_umap(embedding, labels=labels, title="UMAP Projection (Classified)")

#%%
# -----------------------------------------------------------------------------
# 6. Classify using HDBSCANClassifier and GMMClassifier
# -----------------------------------------------------------------------------
print("Step 6a: Clustering data with HDBSCAN...")
# Tune min_cluster_size based on how many gates constitute a 'storm' in your sample
hdb_classifier = HDBSCANClassifier(min_cluster_size=200) 
hdb_labels = hdb_classifier.fit_predict(embedding)

print("Step 6b: Clustering data with GMM...")
# Tell GMM roughly how many hydrometeor classes you expect (e.g., 6 or 7)
gmm_classifier = GMMClassifier(n_components=7)
gmm_labels = gmm_classifier.fit_predict(embedding)

#%%
# -----------------------------------------------------------------------------
# 7. Plot the Comparisons
# -----------------------------------------------------------------------------
# Convert HDBSCAN -1 labels to 'Noise' for a cleaner plot legend
hdb_str_labels = [f"Class {lbl}" if lbl != -1 else "Noise" for lbl in hdb_labels]
gmm_str_labels = [f"Class {lbl}" for lbl in gmm_labels]

plot_umap(embedding, labels=hdb_str_labels, title="UMAP Projection (HDBSCAN)")
plot_umap(embedding, labels=gmm_str_labels, title="UMAP Projection (GMM)")

