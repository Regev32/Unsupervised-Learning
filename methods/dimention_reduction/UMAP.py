import pandas as pd
from sklearn.preprocessing import StandardScaler
from umap.umap_ import UMAP    # <- note the submodule import

def umap_embed(df: pd.DataFrame,
               n_dim: int = 2,
               n_neighbors: int = 15,
               min_dist: float = 0.1,
               metric: str = 'euclidean',
               random_state: int = 42) -> pd.DataFrame:
    """
    UMAP embedding of the numeric columns of `df`.
    """
    # 1) extract numeric data
    X = df.select_dtypes(include="number")
    if X.shape[1] == 0:
        raise ValueError("No numeric columns for UMAP.")

    # 2) scale
    X_std = StandardScaler().fit_transform(X)

    # 3) embed
    umap = UMAP(n_components=n_dim,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state)
    emb = umap.fit_transform(X_std)

    # 4) wrap as DataFrame
    cols = [f"UMAP{i+1}" for i in range(n_dim)]
    return pd.DataFrame(emb, index=df.index, columns=cols)
