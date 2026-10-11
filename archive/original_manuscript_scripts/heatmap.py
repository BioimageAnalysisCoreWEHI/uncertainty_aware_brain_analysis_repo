# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:35:27 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# df = your dataframe
df = pd.read_csv(P(r"<IMAGING>\Ishrat\JoM\final_count_results\full_final_results_with_ssim.csv"))
df = df[df['Sample ID'] == "pp451"]
# Make heatmap table
heatmap_df = df.pivot_table(
    index="grid",
    columns="bending",
    values="preprocessed_snpc_CellCount",
    aggfunc="mean"   # use mean in case there are duplicate bending/grid combos
)

# Optional: sort axes nicely
heatmap_df = heatmap_df.sort_index()
heatmap_df = heatmap_df.reindex(sorted(heatmap_df.columns), axis=1)

# Plot
plt.figure(figsize=(8, 6))

ax = sns.heatmap(
    heatmap_df,
    cmap="viridis",
    annot=True,
    fmt=".0f",
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"label": "TH+ cell count"}
)

ax.set_xlabel("Bending Energy Weight", fontsize=14, labelpad=10)
ax.set_ylabel("Grid Spacing", fontsize=14, labelpad=10)
ax.set_title("TH+ Cell Count Across Registration Parameters", fontsize=15, pad=15)

ax.tick_params(axis="both", labelsize=12)
ax.figure.axes[-1].yaxis.label.set_size(13)
ax.figure.axes[-1].tick_params(labelsize=11)

plt.tight_layout()

# Save publication-quality figure
plt.savefig(P(r"<IMAGING>\Ishrat\JoM\final_figures\Figure1\pp451_preprocessed_snpc_cellcount_heatmap.png"), dpi=600, bbox_inches="tight")
plt.savefig(P(r"<IMAGING>\Ishrat\JoM\final_figures\Figure1\pp451_preprocessed_snpc_cellcount_heatmap.svg"), bbox_inches="tight")

plt.show()