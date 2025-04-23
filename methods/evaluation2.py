import warnings
import numpy as np
import pandas as pd
import umap
import glob, os
import hdbscan
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import mutual_info_score

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
np.seterr(divide='ignore', invalid='ignore', over='ignore')


def evaluate_stroke_clusters_mi(cluster_labels: pd.Series,
                                stroke_series: pd.Series):
    """
    Compute mutual information between stroke and membership in each cluster;
    return best_mi, best_cluster, full metrics DataFrame.
    """
    df = pd.DataFrame({'cluster': cluster_labels, 'stroke': stroke_series}).dropna()
    rows = []
    for cid in df['cluster'].unique():
        membership = (df['cluster'] == cid).astype(int)
        mi = mutual_info_score(df['stroke'], membership)
        rows.append({'cluster': cid, 'mi': mi})
    metrics = pd.DataFrame(rows).set_index('cluster')
    best_cluster = metrics['mi'].idxmax()
    return metrics.loc[best_cluster, 'mi'], best_cluster, metrics


def _get_clustering_configs():
    """Return list of (method_name, params) including DBSCAN, GMM, Agglomerative,
    KMeans, Spectral, and HDBSCAN."""
    configs = []
    # DBSCAN grid
    for eps in np.arange(0.1, 1.1, 0.1):
        configs.append(('DBSCAN', {'eps': eps, 'min_samples': 5}))
    # k-based methods
    for k in range(2, 11):
        configs += [
            ('GMM',           {'n_components': k, 'covariance_type': 'full', 'random_state': 42}),
            ('Agglomerative', {'n_clusters':    k, 'linkage': 'ward'}),
            ('KMeans',        {'n_clusters':    k, 'n_init': 10, 'random_state': 42}),
            ('Spectral',      {'n_clusters':    k, 'affinity': 'nearest_neighbors', 'random_state': 42}),
        ]
    # HDBSCAN
    for mcs in [5, 10, 20, 50]:
        configs.append(('HDBSCAN', {'min_cluster_size': mcs}))
    return configs


if __name__ == '__main__':
    os.makedirs('heatmaps', exist_ok=True)
    input_dir = '../data/filled'

    # -----------------------
    # reduce -> cluster pipeline
    # -----------------------
    rtc_results = []
    reductions = {
        'PCA':  lambda n: PCA(n_components=n, random_state=42),
        'KPCA': lambda n: KernelPCA(n_components=n, kernel='rbf', random_state=42),
        'UMAP': lambda n: umap.UMAP(n_components=n, random_state=42),
    }
    clustering_configs = _get_clustering_configs()

    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
    for fp in csv_files:
        ds_name = os.path.splitext(os.path.basename(fp))[0]
        df      = pd.read_csv(fp)
        X       = df.drop(columns=['id', 'stroke'], errors='ignore')
        y       = df['stroke']
        X_scaled = StandardScaler().fit_transform(X)

        for red_name, red_func in reductions.items():
            for n_comp in range(2, 11):
                X_red = red_func(n_comp).fit_transform(X_scaled)
                for method_name, params in clustering_configs:
                    # fit & predict
                    if method_name == 'DBSCAN':
                        labels = DBSCAN(**params).fit_predict(X_red)
                        cluster_k = None
                    elif method_name == 'GMM':
                        gm = GaussianMixture(**params)
                        gm.fit(X_red)
                        labels = gm.predict(X_red)
                        cluster_k = params['n_components']
                    elif method_name == 'Agglomerative':
                        labels = AgglomerativeClustering(**params).fit_predict(X_red)
                        cluster_k = params['n_clusters']
                    elif method_name == 'KMeans':
                        labels = KMeans(**params).fit_predict(X_red)
                        cluster_k = params['n_clusters']
                    elif method_name == 'Spectral':
                        labels = SpectralClustering(**params).fit_predict(X_red)
                        cluster_k = params['n_clusters']
                    elif method_name == 'HDBSCAN':
                        labels = hdbscan.HDBSCAN(**params).fit_predict(X_red)
                        cluster_k = None
                    else:
                        continue

                    best_mi, best_cluster, _ = evaluate_stroke_clusters_mi(pd.Series(labels), y)
                    rtc_results.append({
                        'dataset':        ds_name,
                        'pipeline':       'reduce_then_cluster',
                        'reduction':      red_name,
                        'n_components':   n_comp,
                        'cluster_method': method_name,
                        'cluster_k':      cluster_k,
                        'best_mi':        best_mi,
                    })

    rtc_df = pd.DataFrame(rtc_results)
    rtc_df.to_csv('../results/reduce_then_cluster_results.csv', index=False)
    print('Saved reduce_then_cluster_results.csv')

    # -----------------------
    # cluster -> reduce pipeline
    # -----------------------
    cf_results = []
    for fp in csv_files:
        ds_name = os.path.splitext(os.path.basename(fp))[0]
        df      = pd.read_csv(fp)
        X_scaled = StandardScaler().fit_transform(
            df.drop(columns=['id', 'stroke'], errors='ignore')
        )
        y = df['stroke']

        for method_name, params in clustering_configs:
            # fit & predict on scaled data
            if method_name == 'DBSCAN':
                labels = DBSCAN(**params).fit_predict(X_scaled)
                cluster_k = None
            elif method_name == 'GMM':
                gm = GaussianMixture(**params)
                gm.fit(X_scaled)
                labels = gm.predict(X_scaled)
                cluster_k = params['n_components']
            elif method_name == 'Agglomerative':
                labels = AgglomerativeClustering(**params).fit_predict(X_scaled)
                cluster_k = params['n_clusters']
            elif method_name == 'KMeans':
                labels = KMeans(**params).fit_predict(X_scaled)
                cluster_k = params['n_clusters']
            elif method_name == 'Spectral':
                labels = SpectralClustering(**params).fit_predict(X_scaled)
                cluster_k = params['n_clusters']
            elif method_name == 'HDBSCAN':
                labels = hdbscan.HDBSCAN(**params).fit_predict(X_scaled)
                cluster_k = None
            else:
                continue

            best_mi, best_cluster, _ = evaluate_stroke_clusters_mi(pd.Series(labels), y)
            cf_results.append({
                'dataset':        ds_name,
                'pipeline':       'cluster_then_reduce',
                'reduction':      'Scaled',  # no reduction step
                'n_components':   0,
                'cluster_method': method_name,
                'cluster_k':      cluster_k,
                'best_mi':        best_mi,
            })

    cf_df = pd.DataFrame(cf_results)
    cf_df.to_csv('../results/cluster_first_results.csv', index=False)
    print('Saved cluster_first_results.csv')

    # -----------------------
    # Heatmap generation for both pipelines
    # -----------------------
    combined = pd.concat([rtc_df, cf_df], ignore_index=True)
    grouped = combined.dropna(subset=['cluster_k']).groupby(
        ['dataset', 'pipeline', 'reduction', 'cluster_method']
    )

    for (ds, pipe, red, method), grp in grouped:
        pivot = grp.pivot(index='n_components',
                          columns='cluster_k',
                          values='best_mi').sort_index().sort_index(axis=1)
        if pivot.empty:
            continue

        fig, ax = plt.subplots()
        cax = ax.imshow(pivot.values, aspect='auto')
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel('cluster_k')
        ax.set_ylabel('n_components')
        ax.set_title(f'{ds} | {pipe} | {red} + {method}')
        fig.colorbar(cax, ax=ax, label='Mutual Information')

        fname = f"../results/heatmaps/{ds}__{pipe}__{red}__{method}.png"
        fig.tight_layout()
        fig.savefig(fname)
        plt.close(fig)
        print(f'Saved heatmap: {fname}')
