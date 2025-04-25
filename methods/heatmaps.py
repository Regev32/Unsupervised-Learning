import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_heatmaps():
    # Load both result CSVs
    rtc_df = pd.read_csv('../results/reduce_then_cluster_results.csv')
    cf_df = pd.read_csv('../results/cluster_first_results.csv')

    # Combine them
    combined = pd.concat([rtc_df, cf_df], ignore_index=True)

    # Filter out methods that don't have cluster_k (like DBSCAN/HDBSCAN)
    grouped = combined.dropna(subset=['cluster_k']).groupby(
        ['dataset', 'pipeline', 'reduction', 'cluster_method']
    )

    # Create base heatmap directory
    base_dir = '../results/heatmaps'
    os.makedirs(base_dir, exist_ok=True)

    for (ds, pipe, red, method), grp in grouped:
        pivot = grp.pivot(index='n_components',
                          columns='cluster_k',
                          values='best_mi').sort_index().sort_index(axis=1)
        if pivot.empty:
            continue

        fig, ax = plt.subplots()
        cax = ax.imshow(pivot.values, aspect='auto', cmap='viridis')
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel('cluster_k')
        ax.set_ylabel('n_components')
        ax.set_title(f'{ds} | {pipe} | {red} + {method}')
        fig.colorbar(cax, ax=ax, label='Mutual Information')

        # Create subfolder based on pipeline type
        subfolder = os.path.join(base_dir, pipe)
        os.makedirs(subfolder, exist_ok=True)

        fname = os.path.join(subfolder, f'{ds}__{red}__{method}.png')
        fig.tight_layout()
        fig.savefig(fname)
        plt.close(fig)
        print(f'Saved heatmap: {fname}')

if __name__ == '__main__':
    generate_heatmaps()
