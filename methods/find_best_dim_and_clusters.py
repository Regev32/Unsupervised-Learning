import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# your existing DR wrappers
from methods.dimention_reduction.PCA import pca
from methods.dimention_reduction.ICA import ica_embed
from methods.dimention_reduction.KPCA import kpca_embed
from methods.dimention_reduction.UMAP import umap_embed

# your existing clustering wrappers
from methods.clustering.KMEANS import kmeans_cluster
from methods.clustering.SPECTRAL import spectral_cluster
from methods.clustering.HIERARCHICAL import hierarchical_cluster
from methods.clustering.GMM import gmm_cluster
from methods.clustering.DBSCAN import dbscan_cluster

# your existing evaluation wrappers
from methods.cluster_evaluation.SILHOUETTE import silhouette_eval
from methods.cluster_evaluation.CALINSKI_HARABASZ import calinski_harabasz_eval
from methods.cluster_evaluation.ADJUSTED_MUTUAL_INFORMATION import adjusted_mutual_info_eval
from methods.cluster_evaluation.MUTUAL_INFORMATION import mutual_info_eval


def run_single_dataset(data_path: str, results_dir: str, label: str):
    os.makedirs(results_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    y  = df['stroke']
    X  = df.drop(['stroke', 'id'], axis=1)
    X_std = StandardScaler().fit_transform(X)

    red_methods = {
        'PCA': pca,
        'ICA': ica_embed,
        'KPCA': kpca_embed,
        'UMAP': umap_embed
    }
    cl_methods = {
        'KMeans':       kmeans_cluster,
        'Spectral':     spectral_cluster,
        'Hierarchical': hierarchical_cluster,
        'GMM':          gmm_cluster,
        'DBSCAN':       dbscan_cluster
    }
    metrics = {
        'Silhouette':       silhouette_eval,
        'CalinskiHarabasz': calinski_harabasz_eval,
        'AMI':              adjusted_mutual_info_eval,
        'MI':               mutual_info_eval
    }

    records = []

    for rname, rfn in red_methods.items():
        for cname, cfn in cl_methods.items():
            # one heatmap per metric
            heatmaps = {
                m: pd.DataFrame(index=range(1,11), columns=range(2,11), dtype=float)
                for m in metrics
            }

            for d in range(1, 11):
                # do DR once per (rname, d)
                E = rfn(pd.DataFrame(X_std), n_dim=d)
                # remove any NaN/inf
                E = np.nan_to_num(E, nan=0.0, posinf=0.0, neginf=0.0)

                # precompute pairwise-distances once
                D = pairwise_distances(E, metric='euclidean')

                for k in range(2, 11):
                    labels = cfn(pd.DataFrame(E), n_clusters=k)

                    for mname, mfn in metrics.items():
                        if mname == 'Silhouette':
                            score = mfn(D, labels)
                        elif mname == 'CalinskiHarabasz':
                            score = mfn(E, labels)
                        else:
                            score = mfn(y, labels)

                        heatmaps[mname].at[d, k] = score
                        records.append({
                            'dataset':    label,
                            'reduction':  rname,
                            'clustering': cname,
                            'n_dim':      d,
                            'n_clusters': k,
                            'metric':     mname,
                            'score':      score
                        })

            # save out each heatmap
            for mname, hm in heatmaps.items():
                plt.figure(figsize=(6,5))
                plt.title(f"{label} — {rname} + {cname} — {mname}")
                im = plt.imshow(hm.values, aspect='auto')
                plt.colorbar(im)
                plt.xticks(range(len(hm.columns)), hm.columns)
                plt.yticks(range(len(hm.index)),   hm.index)
                for i, di in enumerate(hm.index):
                    for j, kj in enumerate(hm.columns):
                        plt.text(j, i, f"{hm.at[di,kj]:.2f}",
                                 ha='center', va='center', fontsize=8)
                plt.xlabel("n_clusters")
                plt.ylabel("n_dim")
                fname = f"{label}_{rname}_{cname}_{mname}.png"
                plt.savefig(os.path.join(results_dir, fname),
                            bbox_inches='tight')
                plt.close()

    # write master CSV
    result_csv = os.path.join(results_dir,
                              f"{label}_cluster_eval_results.csv")
    pd.DataFrame(records).to_csv(result_csv, index=False)


if __name__ == '__main__':
    datasets = {
        'stroke_mean':       '../data/filled/stroke_mean.csv',
        'stroke_median':     '../data/filled/stroke_median.csv',
        'stroke_imputation': '../data/filled/stroke_imputation.csv'
    }
    for label, path in datasets.items():
        out_dir = os.path.join('..', 'results', label)
        run_single_dataset(path, out_dir, label)
