import pandas as pd
import matplotlib.pyplot as plt
import os

# ───────────── global defaults ─────────────
plt.rcParams['axes.grid'] = False      # turn grids off
# (leave the rotation line out)

# ───────────── load data ─────────────
df = pd.read_csv("../data/stroke.csv").drop('id', axis=1)

hist_dir = "../results/histograms"
os.makedirs(hist_dir, exist_ok=True)

# ───────────── plotting loop ─────────────
for col in df.columns:
    plt.figure(figsize=(6, 4))

    if df[col].nunique() <= 2:                # binary → bar plot
        ax = (
            df[col]
            .value_counts(dropna=False)
            .sort_index()
            .plot(kind="bar", width=0.8, edgecolor="black")
        )
        ax.set_ylabel("count")

        # keep 0/1 labels horizontal
        ax.tick_params(axis="x", rotation=0)

        # annotate bars
        for p in ax.patches:
            height = int(p.get_height())
            ax.annotate(
                text=str(height),
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold"
            )

    else:                                     # numeric / multi‑class
        df[col].hist(bins=20, edgecolor="black")
        plt.ylabel("frequency")

    plt.title(col)
    plt.xlabel(col)
    plt.tight_layout()
    plt.savefig(os.path.join(hist_dir, f"{col}.png"))
    plt.close()

print(f"Histograms saved to {hist_dir}")
