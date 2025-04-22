import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA

def ica_embed(df: pd.DataFrame,
              n_dim: int,
              random_state: int = 42,
              tol: float = 0.0001,
              max_iter: int = 200
             ) -> pd.DataFrame:
    """
    Fit ICA on the **numeric** columns of `df` and return an
    n_dim‑column DataFrame of the independent component embeddings.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data frame (mixed dtypes OK – only numeric cols are used).
    n_dim : int
        Number of independent components to extract.
    random_state : int, default=42
        Seed for reproducibility.
    tol : float, default=1e-4
        Tolerance on update for stopping criteria.
    max_iter : int, default=200
        Maximum number of iterations during fit.

    Returns
    -------
    pandas.DataFrame
        Shape = (len(df), n_dim); columns are IC1, IC2, …
    """
    # 1. select numeric columns
    X_num = df.select_dtypes(include="number")
    if X_num.shape[1] == 0:
        raise ValueError("No numeric columns to run ICA on.")

    # 2. standardize
    X_std = StandardScaler().fit_transform(X_num)

    # 3. fit and transform with FastICA
    ica = FastICA(
        n_components=n_dim,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state
    )
    components = ica.fit_transform(X_std)

    # 4. wrap into DataFrame
    cols = [f"IC{i+1}" for i in range(n_dim)]
    emb_df = pd.DataFrame(components, index=df.index, columns=cols)
    return emb_df
