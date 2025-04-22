import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

def gmm_cluster(df: pd.DataFrame,
                n_components: int = 3,
                covariance_type: str = 'full',
                random_state: int = 42) -> pd.Series:
    """
    Gaussian Mixture Model clustering labels of the numeric columns of `df`.
    """
    X = df.select_dtypes(include="number")
    if X.shape[1] == 0:
        raise ValueError("No numeric columns for GMM.")
    X_std = StandardScaler().fit_transform(X)
    gm = GaussianMixture(n_components=n_components,
                         covariance_type=covariance_type,
                         random_state=random_state)
    labels = gm.fit_predict(X_std)
    return pd.Series(labels, index=df.index, name="gmm_label")
