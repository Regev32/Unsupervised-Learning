import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering

def spectral_cluster(df: pd.DataFrame,
                     n_clusters: int = 3,
                     affinity: str = 'rbf',
                     gamma: float = None,
                     random_state: int = 42) -> pd.Series:
    """
    Spectral Clustering labels of the numeric columns of `df`.
    """
    X = df.select_dtypes(include="number")
    if X.shape[1] == 0:
        raise ValueError("No numeric columns for Spectral Clustering.")
    X_std = StandardScaler().fit_transform(X)
    sc = SpectralClustering(n_clusters=n_clusters,
                            affinity=affinity,
                            gamma=gamma,
                            random_state=random_state,
                            assign_labels="kmeans")
    labels = sc.fit_predict(X_std)
    return pd.Series(labels, index=df.index, name="spectral_label")
