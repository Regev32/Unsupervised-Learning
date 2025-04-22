import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

def mutual_info_eval(y_true: pd.Series, labels: pd.Series) -> float:
    """
    Compute the Adjusted Mutual Information between true labels and cluster labels.

    Parameters
    ----------
    y_true : Series or array-like of shape (n_samples,)
        Ground-truth labels.
    labels : Series or array-like of shape (n_samples,)
        Cluster labels for each sample.

    Returns
    -------
    float
        Adjusted Mutual Information score (range: 0 to 1; higher is better).
    """
    return adjusted_mutual_info_score(y_true, labels)