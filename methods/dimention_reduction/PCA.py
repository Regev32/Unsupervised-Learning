import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca(df: pd.DataFrame, n_dim: int) -> pd.DataFrame:
    """
    Fit PCA on the **numeric** columns of `df` and return an
    n_dim‑column DataFrame of the low‑dimensional embeddings.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data frame (mix of dtypes is fine – only numeric
        columns are used).
    n_dim : int
        How many principal components to keep.

    Returns
    -------
    pandas.DataFrame
        Shape = (len(df), n_dim); columns are PC1, PC2, …
    """
    # 1 · grab numeric columns only
    X_num = df.select_dtypes(include="number")
    if X_num.empty:
        raise ValueError("No numeric columns to run PCA on.")

    # 2 · standardise — crucial for PCA
    X_std = StandardScaler().fit_transform(X_num)

    # 3 · fit PCA and project rows (“scores”)
    pca    = PCA(n_components=n_dim, random_state=42)
    scores = pca.fit_transform(X_std)

    # 4 · wrap back into a DataFrame
    emb_df = pd.DataFrame(scores,
                          index=df.index,
                          columns=[f"PC{i+1}" for i in range(n_dim)])
    return emb_df
