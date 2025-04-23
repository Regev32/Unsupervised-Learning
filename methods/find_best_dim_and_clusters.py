import warnings
import numpy as np
import pandas as pd
import umap
import glob, os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
np.seterr(divide='ignore', invalid='ignore', over='ignore')


def evaluate_stroke_clusters(cluster_labels: pd.Series,
                             stroke_series: pd.Series):
    """
    Evaluate clusters by the percentage of stroke cases (1's) in each cluster.

    Returns the highest percentage and the cluster ID achieving it,
    plus a DataFrame of metrics per cluster.
    """
    # align and drop missing
    df = pd.DataFrame({
        'cluster': cluster_labels,
        'stroke': stroke_series
    }).dropna()

    rows = []
    for cid, grp in df.groupby('cluster'):
        size = len(grp)
        positives = int(grp['stroke'].sum())
        pct = positives / size if size > 0 else 0.0

        rows.append({
            'cluster': cid,
            'size': size,
            'positives': positives,
            'pct': pct
        })

    metrics = pd.DataFrame(rows).set_index('cluster')
    best_cluster = metrics['pct'].idxmax()
    best_pct = metrics.loc[best_cluster, 'pct']
    return best_pct, best_cluster, metrics


def analyze_stroke_datasets(input_dir='../data/filled', output_file='../results/clustering_results.csv'):
    """Iterate over all CSVs, reduce, cluster, and evaluate by max-%-of-1s."""
    results = []
    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

    for file_path in csv_files:
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_csv(file_path)

        X = df.drop(columns=['id', 'stroke'], errors='ignore')
        y = pd.Series(df['stroke'])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reductions = {
            'PCA': lambda n: PCA(n_components=n, random_state=42),
            'KPCA': lambda n: KernelPCA(n_components=n, kernel='rbf', random_state=42),
            'UMAP': lambda n: umap.UMAP(n_components=n, random_state=42)
        }

        for method, reducer_func in reductions.items():
            for n_components in range(2, 11):
                reducer = reducer_func(n_components)
                X_red = reducer.fit_transform(X_scaled)

                clustering_configs = []
                # DBSCAN
                for eps in np.arange(0.1, 1.1, 0.1):
                    clustering_configs.append(('DBSCAN', {'eps': eps, 'min_samples': 5}))
                # GMM, Agglomerative, KMeans, Spectral
                for k in range(2, 11):
                    clustering_configs += [
                        ('GMM', {'n_components': k, 'covariance_type': 'full', 'random_state': 42}),
                        ('Agglomerative', {'n_clusters': k, 'linkage': 'ward'}),
                        ('KMeans', {'n_clusters': k, 'n_init': 10, 'random_state': 42}),
                        ('Spectral', {'n_clusters': k, 'affinity': 'nearest_neighbors', 'random_state': 42})
                    ]

                for method_name, params in clustering_configs:
                    # instantiate
                    if method_name == 'DBSCAN':
                        model = DBSCAN(**params)
                    elif method_name == 'GMM':
                        model = GaussianMixture(**params)
                    elif method_name == 'Agglomerative':
                        model = AgglomerativeClustering(**params)
                    elif method_name == 'KMeans':
                        model = KMeans(**params)
                    elif method_name == 'Spectral':
                        model = SpectralClustering(**params)
                    else:
                        continue

                    # fit & predict
                    if method_name == 'GMM':
                        model.fit(X_red)
                        cluster_labels = model.predict(X_red)
                    else:
                        cluster_labels = model.fit_predict(X_red)

                    # evaluate by % of 1's
                    best_pct, best_cluster, metrics_df = evaluate_stroke_clusters(
                        pd.Series(cluster_labels), y
                    )

                    # record result
                    result = {
                        'dataset': dataset_name,
                        'reduction': method,
                        'n_components': n_components,
                        'cluster_method': method_name,
                        'params': ", ".join(f"{k}={v}" for k, v in params.items()),
                        'best_pct': best_pct,
                        'best_cluster': best_cluster
                    }
                    # also log the size and positives of that best cluster
                    best_stats = metrics_df.loc[best_cluster]
                    result['size'] = best_stats['size']
                    result['positives'] = best_stats['positives']

                    results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    return results_df


if __name__ == "__main__":
    analyze_stroke_datasets()
