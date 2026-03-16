import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib.path import Path as MplPath
import umap
import hdbscan
from sklearn.mixture import GaussianMixture


def normalize_data(df, features, method="standard"):
    """
    Normalizes specified features in the DataFrame.
    Returns the scaled DataFrame and the scaler object.
    """
    data = df[features].values
    
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Method must be 'minmax' or 'standard'")
    
    scaled_data = scaler.fit_transform(data)
    
    # Return a new DataFrame with scaled values
    df_scaled = df.copy()
    df_scaled[features] = scaled_data
    
    return df_scaled, scaler

def apply_umap(df, features, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Applies UMAP dimensionality reduction.
    """
    if umap is None:
        raise ImportError("Please install 'umap-learn' to use this function.")
        
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    
    embedding = reducer.fit_transform(df[features])
    return embedding, reducer

class PolyClassifier:
    """
    A simple classifier that assigns labels based on 2D polygons.
    """
    def __init__(self, classes_polygons=None):
        """
        Parameters
        ----------
        classes_polygons : dict
            Dictionary where key is the class name (str) and value is a list 
            of (x, y) tuples defining the polygon vertices.
        """
        self.classes_polygons = classes_polygons if classes_polygons else {}

    def predict(self, embedding):
        """
        Predicts class labels for the given 2D embedding.
        """
        # Default label
        labels = np.array(["Unclassified"] * len(embedding), dtype=object)
        
        # Check points against each polygon
        for class_name, poly_coords in self.classes_polygons.items():
            path = MplPath(poly_coords)
            mask = path.contains_points(embedding)
            labels[mask] = class_name
            
        return labels

class HDBSCANClassifier:
    """
    Density-based clustering optimized for UMAP embeddings.
    """
    def __init__(self, min_cluster_size=100, min_samples=15):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.clusterer = None

    def fit_predict(self, embedding):
        if hdbscan is None:
            raise ImportError("Please install 'hdbscan' (pip install hdbscan).")
            
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            gen_min_span_tree=True
        )
        
        # Returns an array of integer labels. -1 means 'Noise'.
        labels = self.clusterer.fit_predict(embedding)
        return labels

class GMMClassifier:
    """
    Gaussian Mixture Model for probabilistic clustering on UMAP embeddings.
    Great for continuous physical transitions (like melting hydrometeors).
    """
    def __init__(self, n_components=5, random_state=42):
        """
        n_components: The number of distinct hydrometeor classes you expect to find.
        """
        self.n_components = n_components
        self.clusterer = GaussianMixture(
            n_components=self.n_components, 
            covariance_type='full', # Allows elliptical clusters
            random_state=random_state
        )

    def fit_predict(self, embedding):
        """Returns the hard integer labels for the most likely class."""
        return self.clusterer.fit_predict(embedding)
        
    def predict_proba(self, embedding):
        """Returns the probability matrix for each class (useful for uncertainty)."""
        self.clusterer.fit(embedding)
        return self.clusterer.predict_proba(embedding)

