import pandas as pd
from sklearn.metrics import calinski_harabasz_score

def calinski_harabasz_eval(X: pd.DataFrame, labels: pd.Series) -> float:
    """
    Compute the Calinski-Harabasz Index for clustering.

    Parameters
    ----------
    X : DataFrame or array-like of shape (n_samples, n_features)
        Feature matrix used for clustering.
    labels : Series or array-like of shape (n_samples,)
        Cluster labels for each sample.

    Returns
    -------
    float
        Calinski-Harabasz score (higher is better).
    """
    return calinski_harabasz_score(X, labels)