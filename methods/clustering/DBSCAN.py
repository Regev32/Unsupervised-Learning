import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

def dbscan_cluster(df: pd.DataFrame,
                   eps: float = 0.5,
                   min_samples: int = 5,
                   metric: str = 'euclidean') -> pd.Series:
    """
    DBSCAN clustering labels of the numeric columns of `df`.
    """
    X = df.select_dtypes(include="number")
    if X.shape[1] == 0:
        raise ValueError("No numeric columns for DBSCAN.")
    X_std = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    labels = db.fit_predict(X_std)
    return pd.Series(labels, index=df.index, name="dbscan_label")
