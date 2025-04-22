import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA

def kpca_embed(df: pd.DataFrame,
               n_dim: int,
               kernel: str = 'rbf',
               gamma: float = None,
               degree: int = 3,
               random_state: int = 42) -> pd.DataFrame:
    """
    Kernel PCA embedding of the numeric columns of `df`.
    """
    X = df.select_dtypes(include="number")
    if X.shape[1] == 0:
        raise ValueError("No numeric columns for Kernel PCA.")
    X_std = StandardScaler().fit_transform(X)
    kpca = KernelPCA(n_components=n_dim,
                     kernel=kernel,
                     gamma=gamma,
                     degree=degree,
                     random_state=random_state)
    emb = kpca.fit_transform(X_std)
    cols = [f"KPC{i+1}" for i in range(n_dim)]
    return pd.DataFrame(emb, index=df.index, columns=cols)
