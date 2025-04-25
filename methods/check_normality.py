# normality_test.py
# ------------------ Normality Testing for Silhouette & MI Scores ------------------

import pandas as pd
from scipy.stats import shapiro

counter = 0
i = 0
def normality_report(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Runs a Shapiro–Wilk test on each column of the DataFrame.
    Returns a DataFrame with W-statistic, p-value, and normality boolean per method.
    """
    results = []
    for col in df.columns:
        vals = df[col].dropna().values
        w_stat, p_val = shapiro(vals)
        normal = p_val >= alpha
        results.append({'method': col, 'W': w_stat, 'p_value': p_val, 'normal': normal})
        global counter, i
        counter = counter + 1 if normal else counter
        i += 1
    return pd.DataFrame(results)


def main(sil_path: str, mi_path: str, alpha: float = 0.05):
    # Load data
    sil_df = pd.read_csv(sil_path)
    mi_df  = pd.read_csv(mi_path)

    # Perform tests
    print(f"Normality test (alpha={alpha}) for silhouette scores from '{sil_path}':")
    sil_report = normality_report(sil_df, alpha)
    print(sil_report.to_string(index=False))

    print(f"\nNormality test (alpha={alpha}) for MI scores from '{mi_path}':")
    mi_report = normality_report(mi_df, alpha)
    print(mi_report.to_string(index=False))

    # Optionally save reports
    sil_report.to_csv('../results/silhouette_normality_report.csv', index=False)
    mi_report.to_csv('../results/mi_normality_report.csv', index=False)
    print("\nReports saved to 'silhouette_normality_report.csv' and 'mi_normality_report.csv'.")


if __name__ == "__main__":
    silhouette_path = '../results/silhouette_scores.csv'
    mi_path = '../results/mi_scores.csv'
    alpha = 0.05
    main(silhouette_path, mi_path, alpha)
    print(counter)
    print(i)
