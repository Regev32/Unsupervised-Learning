# statistical_tests.py
# ------------------ Omnibus, Post-hoc, Top-100 Extraction, and Wilcoxon Fallback ------------------

import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

try:
    import scikit_posthocs as sp
    HAS_SP = True
except ImportError:
    HAS_SP = False


def run_friedman(df: pd.DataFrame, name: str, alpha: float = 0.05):
    """
    Runs Friedman omnibus test on method score DataFrame.
    Drops methods with missing values so all comparisons are complete.
    Prints chi-square, p-value, and decision.
    """
    df_clean = df.dropna(axis=1, how='any')
    n_rep, n_methods = df_clean.shape
    if n_methods < 2:
        print(f"Not enough complete methods ({n_methods}) for Friedman on {name}.")
        return None, None

    arrays = [df_clean[col].values for col in df_clean.columns]
    stat, p_val = friedmanchisquare(*arrays)

    print(f"Friedman test for {name}: chi2={stat:.3f}, p={p_val:.3e} (replicates={n_rep}, methods={n_methods})")
    if p_val < alpha:
        print(f"→ reject H0 at alpha={alpha}: methods differ significantly")
    else:
        print(f"→ fail to reject H0 at alpha={alpha}: no significant differences")
    return stat, p_val


def run_nemenyi(df: pd.DataFrame, name: str, alpha: float = 0.05):
    """
    Runs Nemenyi post-hoc test (requires scikit-posthocs).
    Drops methods with missing values to match Friedman input.
    Prints average ranks, pairwise p-values, and identifies best method(s).
    """
    if not HAS_SP:
        print("scikit-posthocs not installed; skipping Nemenyi post-hoc.")
        return None

    df_clean = df.dropna(axis=1, how='any')
    n_rep, n_methods = df_clean.shape
    if n_methods < 2:
        print(f"Not enough complete methods ({n_methods}) for Nemenyi on {name}.")
        return None

    print(f"\nRunning Nemenyi post-hoc for {name} (methods={n_methods})...")
    matrix = sp.posthoc_nemenyi_friedman(df_clean.values)
    methods = df_clean.columns.tolist()
    nemenyi = pd.DataFrame(matrix, index=methods, columns=methods)

    avg_ranks = df_clean.rank(axis=1, method='average').mean().sort_values()
    print("Average ranks (lower is better):")
    print(avg_ranks.to_string())

    print(f"\nNemenyi pairwise p-values (alpha={alpha}):")
    print(nemenyi.to_string())

    best = avg_ranks.index[0]
    pvals = nemenyi.loc[best, :]
    tied = [best] + [m for m, p in pvals.items() if m != best and p >= alpha]
    if len(tied) == 1:
        print(f"\nBest method overall for {name}: {best}")
    else:
        print(f"\nTop tied methods for {name}: {tied}")

    return nemenyi


def save_top100(df: pd.DataFrame, name: str, alpha: float = 0.05):
    """
    Extracts the 100 best methods by average rank and their p-value vs the best.
    Saves to '{name}_top100.csv'.
    """
    if not HAS_SP:
        print(f"Cannot save top-100 for {name}: scikit-posthocs missing.")
        return

    df_clean = df.dropna(axis=1, how='any')
    avg_ranks = df_clean.rank(axis=1, method='average').mean().sort_values()
    if avg_ranks.empty:
        print(f"No methods to save for {name}.")
        return

    matrix = sp.posthoc_nemenyi_friedman(df_clean.values)
    methods = df_clean.columns.tolist()
    nemenyi = pd.DataFrame(matrix, index=methods, columns=methods)

    best = avg_ranks.index[0]
    top_methods = avg_ranks.index[:100]

    records = []
    for m in top_methods:
        p_val = 1.0 if m == best else float(nemenyi.loc[best, m])
        records.append({
            'method': m,
            'avg_rank': avg_ranks[m],
            'p_value_vs_best': p_val
        })

    top_df = pd.DataFrame(records)
    filename = f"{name}_top100.csv"
    top_df.to_csv(filename, index=False)
    print(f"Saved top 100 {name} methods to {filename}")


def print_best_with_pvalue(df: pd.DataFrame, name: str, alpha: float = 0.05):
    """
    Identifies the best and second-best methods by average rank and prints their p-value.
    Falls back to Wilcoxon signed-rank if Nemenyi yields NaN or tied data.
    Only prints: name, best method, second best, and p-value.
    """
    import warnings
    from scipy.stats import wilcoxon

    # Keep only methods with complete scores
    df_clean = df.dropna(axis=1, how='any')
    avg_ranks = df_clean.rank(axis=1, method='average').mean().sort_values()
    if len(avg_ranks) < 2:
        print(f"Not enough methods to compare for {name}.")
        return

    # Initialize p_val
    p_val = np.nan
    best = avg_ranks.index[0]
    second = avg_ranks.index[1]

    # Attempt Nemenyi test
    if HAS_SP:
        matrix = sp.posthoc_nemenyi_friedman(df_clean.values)
        methods = df_clean.columns.tolist()
        nemenyi = pd.DataFrame(matrix, index=methods, columns=methods)
        p_val = nemenyi.loc[best, second]

    # If p_val is nan or methods tied perfectly, fallback
    if not HAS_SP or np.isnan(p_val):
        x = df_clean[best].values
        y = df_clean[second].values
        # if identical vectors, p=1.0 (no difference)
        if np.array_equal(x, y):
            p_val = 1.0
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                try:
                    _, p_val = wilcoxon(x, y)
                except Exception:
                    p_val = 1.0

    print(f"{name}: best = {best}, p-value = {p_val:.3e}")

    # compute Nemenyi matrix if available
    if HAS_SP:
        matrix = sp.posthoc_nemenyi_friedman(df_clean.values)
        methods = df_clean.columns.tolist()
        nemenyi = pd.DataFrame(matrix, index=methods, columns=methods)
        best = avg_ranks.index[0]
        second = avg_ranks.index[1]
        p_val = nemenyi.loc[best, second]
    else:
        best = avg_ranks.index[0]
        second = avg_ranks.index[1]
        p_val = np.nan

    # fallback to Wilcoxon if p_val is nan
    if np.isnan(p_val):
        x = df_clean[best].values
        y = df_clean[second].values
        try:
            stat, p_val = wilcoxon(x, y)
        except Exception:
            p_val = np.nan

    print(f"{name}: best = {best}, p-value = {p_val:.3e}")


def main(sil_csv: str = '../results/silhouette_scores.csv', mi_csv: str = '../results/mi_scores.csv'):
    sil_df = pd.read_csv(sil_csv)
    mi_df  = pd.read_csv(mi_csv)
    print_best_with_pvalue(sil_df, 'Silhouette')
    print_best_with_pvalue(mi_df, 'Mutual Information')


if __name__ == '__main__':
    main()
