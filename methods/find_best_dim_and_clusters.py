import warnings
import numpy as np
import pandas as pd
import umap
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, mutual_info_score

# ---------- suppress warnings ----------
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore', over='ignore')


def bootstrap_cluster_evaluation(
    file_path="../data/stroke.csv",
    n_bootstrap=50,
    silhouette_output="silhouette_scores.csv",
    mi_output="mi_scores.csv",
    random_seed=42
):
    """
    Perform full-size bootstrap (with replacement) and evaluate each
    (reduction + clustering) method by silhouette score and mutual information.

    Saves two CSVs of shape (n_bootstrap x n_methods):
      - silhouette scores
      - mutual information scores
    """
    # load full dataset
    df_full = pd.read_csv(file_path)
    N = len(df_full)

    # prepare dimension‐reduction methods
    reductions = {
        'PCA': lambda n: PCA(n_components=n, random_state=random_seed),
        'KPCA': lambda n: KernelPCA(n_components=n, kernel='rbf', random_state=random_seed),
        'UMAP': lambda n: umap.UMAP(n_components=n, random_state=random_seed)
    }

    # clustering configurations
    def get_clustering_configs():
        configs = []
        # DBSCAN
        for eps in np.arange(0.1, 1.1, 0.1):
            configs.append(('DBSCAN', {'eps': eps, 'min_samples': 5}))
        # GMM, Agglomerative, KMeans, Spectral
        for k in range(2, 11):
            configs += [
                ('GMM', {'n_components': k, 'covariance_type': 'full', 'random_state': random_seed}),
                ('Agglomerative', {'n_clusters': k, 'linkage': 'ward'}),
                ('KMeans', {'n_clusters': k, 'n_init': 10, 'random_state': random_seed}),
                ('Spectral', {'n_clusters': k, 'affinity': 'nearest_neighbors', 'random_state': random_seed})
            ]
        return configs

    silhouette_rows = []
    mi_rows = []
    rng = np.random.RandomState(random_seed)

    for b in range(n_bootstrap):
        # full-size bootstrap sample
        df = df_full.sample(n=N, replace=True, random_state=rng.randint(0, 2**32 - 1))
        X = df.drop(columns=['id', 'stroke'], errors='ignore')
        y = df['stroke'].values

        # scale features
        X_scaled = StandardScaler().fit_transform(X)

        sil_row = {}
        mi_row = {}

        # iterate over each DR + clustering combination
        for method_name, reducer_func in reductions.items():
            for n_comp in range(2, 11):
                X_red = reducer_func(n_comp).fit_transform(X_scaled)
                for cluster_name, params in get_clustering_configs():
                    # instantiate clustering model
                    if cluster_name == 'DBSCAN':
                        model = DBSCAN(**params)
                    elif cluster_name == 'GMM':
                        model = GaussianMixture(**params)
                    elif cluster_name == 'Agglomerative':
                        model = AgglomerativeClustering(**params)
                    elif cluster_name == 'KMeans':
                        model = KMeans(**params)
                    elif cluster_name == 'Spectral':
                        model = SpectralClustering(**params)
                    else:
                        continue

                    # fit & predict clusters
                    if cluster_name == 'GMM':
                        model.fit(X_red)
                        labels = model.predict(X_red)
                    else:
                        labels = model.fit_predict(X_red)

                    # compute silhouette score (or NaN if invalid)
                    try:
                        sil = silhouette_score(X_red, labels)
                    except:
                        sil = np.nan
                    # compute mutual information
                    mi = mutual_info_score(y, labels)

                    # build unique key
                    param_str = "_".join(f"{k}{v}" for k, v in params.items())
                    key = f"{method_name}_{n_comp}_{cluster_name}_{param_str}"

                    sil_row[key] = sil
                    mi_row[key]  = mi

        silhouette_rows.append(sil_row)
        mi_rows.append(mi_row)

    # build DataFrames and save
    sil_df = pd.DataFrame(silhouette_rows)
    mi_df  = pd.DataFrame(mi_rows)

    os.makedirs(os.path.dirname(silhouette_output) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(mi_output) or '.', exist_ok=True)
    sil_df.to_csv(silhouette_output, index=False)
    mi_df.to_csv(mi_output, index=False)
    print(f"Saved silhouette scores to {silhouette_output}")
    print(f"Saved MI scores to {mi_output}")

    return sil_df, mi_df


if __name__ == "__main__":
    bootstrap_cluster_evaluation()
