import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score

def silhouette_eval(X, labels) -> float:
    """
    Compute the Silhouette Score for clustering.  If X is a square matrix
    (n_samples×n_samples), treat it as a precomputed distance matrix.
    """
    # get raw array
    arr = X.values if isinstance(X, pd.DataFrame) else X

    # if square, assume it's a distance matrix
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        return silhouette_score(arr, labels, metric='precomputed')
    else:
        return silhouette_score(arr, labels)
