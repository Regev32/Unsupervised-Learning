import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def fill_missing_bmi(data_path: str = "../data/stroke.csv") -> None:
    """
    Reads the stroke CSV and fills missing 'bmi' in three ways, then writes
    three new CSVs under data/filled/:
      - stroke_mean.csv       : missing BMI filled with the overall arithmetic mean
      - stroke_median.csv     : missing BMI filled with the median (50th percentile)
      - stroke_imputation.csv : missing BMI filled by kNN-5 imputation

    Note:
    - Mean is the average of all values and can be skewed by outliers.
    - Median is the middle value, robust to extreme values.
    """
    df = pd.read_csv(data_path)

    # 1) Fill with mean
    df_mean = df.copy()
    mean_bmi = df_mean['bmi'].mean(skipna=True)
    df_mean['bmi'] = df_mean['bmi'].fillna(mean_bmi)

    # 2) Fill with median
    df_median = df.copy()
    median_bmi = df_median['bmi'].median(skipna=True)
    df_median['bmi'] = df_median['bmi'].fillna(median_bmi)

    # 3) kNN-based imputation
    df_imp = df.copy()
    num_feats = df_imp.select_dtypes(include=[np.number]).drop(columns=['id','bmi','stroke'])

    known_mask = df_imp['bmi'].notna()
    unknown_mask = df_imp['bmi'].isna()

    X_known = num_feats.loc[known_mask].values
    bmi_known = df_imp.loc[known_mask, 'bmi'].values
    X_unknown = num_feats.loc[unknown_mask].values

    nbrs = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nbrs.fit(X_known)

    distances, neighbors = nbrs.kneighbors(X_unknown)
    for i, idx in enumerate(df_imp.loc[unknown_mask].index):
        neigh_idxs = neighbors[i]
        df_imp.at[idx, 'bmi'] = bmi_known[neigh_idxs].mean()

    out_dir = os.path.join(os.path.dirname(data_path), 'filled')
    os.makedirs(out_dir, exist_ok=True)

    df_mean.to_csv(os.path.join(out_dir, 'stroke_mean.csv'), index=False)
    print(f"Filled missing BMI with mean: {mean_bmi}")
    df_median.to_csv(os.path.join(out_dir, 'stroke_median.csv'), index=False)
    print(f"Filled missing BMI with median: {median_bmi}")
    df_imp.to_csv(os.path.join(out_dir, 'stroke_imputation.csv'), index=False)
    print("Filled missing BMI with 5NN imputation.")


if __name__ == "__main__":
    fill_missing_bmi()
