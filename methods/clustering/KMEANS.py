import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def kmeans_cluster(df: pd.DataFrame,
                   n_clusters: int = 3,
                   random_state: int = 42) -> pd.Series:
    """
    K‑Means clustering labels of the numeric columns of `df`.
    """
    X = df.select_dtypes(include="number")
    if X.shape[1] == 0:
        raise ValueError("No numeric columns for K‑Means.")
    X_std = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(X_std)
    return pd.Series(labels, index=df.index, name="kmeans_label")
