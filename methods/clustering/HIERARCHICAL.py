import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

def hierarchical_cluster(df: pd.DataFrame,
                         n_clusters: int = 3,
                         linkage: str = 'ward') -> pd.Series:
    """
    Agglomerative (hierarchical) clustering labels of the numeric columns of `df`.
    """
    X = df.select_dtypes(include="number")
    if X.shape[1] == 0:
        raise ValueError("No numeric columns for hierarchical clustering.")
    X_std = StandardScaler().fit_transform(X)
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = hc.fit_predict(X_std)
    return pd.Series(labels, index=df.index, name="hierarchical_label")
