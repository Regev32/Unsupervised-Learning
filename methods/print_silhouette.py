#!/usr/bin/env python3
import pandas as pd
import sys

COLUMN = "PCA_4_DBSCAN_eps0.1_min_samples5"
CSV_PATH = "../results/silhouette_scores.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    if COLUMN not in df.columns:
        print(f"Error: column '{COLUMN}' not found in {CSV_PATH}", file=sys.stderr)
        sys.exit(1)
    values = df[COLUMN].tolist()
    print(f"Values for {COLUMN}:")
    for i, v in enumerate(values, 1):
        print(f"  Replicate {i}: {v}")

if __name__ == "__main__":
    main()
