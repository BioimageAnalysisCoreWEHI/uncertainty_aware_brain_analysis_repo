# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:45:33 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ============================================================
# CONFIG
# ============================================================

BASE = Path(P(r"<IMAGING>\Ishrat\JoM"))

INPUT_CSV = BASE / "final_count_results" / "full_final_results_with_ssim.csv"

OUTDIR = BASE / "final_count_results" / "snpc_count_distribution_plots"
OUTDIR_CLEAN = BASE / "final_count_results" / "snpc_count_distribution_plots_clean"

OUTDIR.mkdir(parents=True, exist_ok=True)
OUTDIR_CLEAN.mkdir(parents=True, exist_ok=True)

PROCESSING_STATUSES = ["raw", "preprocessed"]

COMBINATIONS = [
    {"qc_filter": "passed_qc", "ssim_filter": "top_25pct", "top_fraction": 0.25},
    {"qc_filter": "passed_qc", "ssim_filter": "top_50pct", "top_fraction": 0.50},
    {"qc_filter": "passed_qc", "ssim_filter": "all", "top_fraction": None},

    {"qc_filter": "all_runs", "ssim_filter": "top_25pct", "top_fraction": 0.25},
    {"qc_filter": "all_runs", "ssim_filter": "top_50pct", "top_fraction": 0.50},
    {"qc_filter": "all_runs", "ssim_filter": "all", "top_fraction": None},
]

FIGSIZE = (13, 8)
DPI = 300

BOX_FACE = "#9AA4B2"
BOX_EDGE = "#4B5563"
MEDIAN_COLOR = "#111827"
POINT_EDGE = "#374151"

CMAP = "magma"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")].copy()
df["Sample ID"] = df["Sample ID"].astype(str)

samples = sorted(df["Sample ID"].unique())
# Create clean labels
sample_labels = {sample: f"Sample {i+1}" for i, sample in enumerate(samples)}

# Reverse mapping (for saving)
label_to_sample = {v: k for k, v in sample_labels.items()}

# Save mapping
mapping_df = pd.DataFrame({
    "sample_id": samples,
    "plot_label": [sample_labels[s] for s in samples]
})

mapping_df.to_csv(
    OUTDIR_CLEAN / "sample_id_mapping.csv",
    index=False
)

# ============================================================
# HELPERS
# ============================================================

def get_subset(df, sample_id, processing_status, qc_filter, ssim_filter, top_fraction=None):
    pass_col = f"{processing_status}_pass_final"
    ssim_col = f"{processing_status}_ssim_value"

    sub = df[df["Sample ID"] == str(sample_id)].copy()

    if qc_filter == "passed_qc":
        sub = sub[sub[pass_col] == True]
    elif qc_filter == "all_runs":
        pass
    else:
        raise ValueError(f"Unknown qc_filter: {qc_filter}")

    if ssim_filter in ["top_25pct", "top_50pct"]:
        sub = sub.dropna(subset=[ssim_col])
        sub = sub.sort_values(ssim_col, ascending=False)
        n_keep = max(1, int(np.ceil(len(sub) * top_fraction)))
        sub = sub.iloc[:n_keep]

    elif ssim_filter == "all":
        pass
    else:
        raise ValueError(f"Unknown ssim_filter: {ssim_filter}")

    return sub.reset_index(drop=True)


def make_plotting_df(df):
    rows = []

    for processing_status in PROCESSING_STATUSES:
        count_col = f"{processing_status}_snpc_CellCount"
        ssim_col = f"{processing_status}_ssim_value"

        for combo in COMBINATIONS:
            qc_filter = combo["qc_filter"]
            ssim_filter = combo["ssim_filter"]
            top_fraction = combo["top_fraction"]
            combination_name = f"{qc_filter}_{ssim_filter}"

            for sample_id in samples:
                sub = get_subset(
                    df=df,
                    sample_id=sample_id,
                    processing_status=processing_status,
                    qc_filter=qc_filter,
                    ssim_filter=ssim_filter,
                    top_fraction=top_fraction,
                )

                sub = sub.dropna(subset=[count_col, ssim_col])

                for _, row in sub.iterrows():
                    rows.append({
                        "sample_id": sample_id,
                        "processing_status": processing_status,
                        "qc_filter": qc_filter,
                        "ssim_filter": ssim_filter,
                        "top_fraction": top_fraction,
                        "combination_name": combination_name,
                        "snpc_cell_count": row[count_col],
                        "ssim": row[ssim_col],
                    })

    return pd.DataFrame(rows)


plot_df = make_plotting_df(df)

plot_df.to_csv(
    OUTDIR / "plotting_data_all_snpc_count_distributions.csv",
    index=False,
)


# ============================================================
# GLOBAL NORMALISATION AND AXIS LIMITS
# ============================================================

ssim_min = plot_df["ssim"].min()
ssim_max = plot_df["ssim"].max()

count_min = plot_df["snpc_cell_count"].min()
count_max = plot_df["snpc_cell_count"].max()

count_pad = (count_max - count_min) * 0.08
ylim = (count_min - count_pad, count_max + count_pad)

norm = Normalize(vmin=ssim_min, vmax=ssim_max)
cmap = plt.get_cmap(CMAP)


# ============================================================
# PLOT FUNCTION
# ============================================================

def plot_one_distribution(plot_df, processing_status, combination_name):
    sub = plot_df[
        (plot_df["processing_status"] == processing_status) &
        (plot_df["combination_name"] == combination_name)
    ].copy()

    if sub.empty:
        print(f"No data for {processing_status} | {combination_name}")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    data_by_sample = [
        sub.loc[sub["sample_id"] == sample, "snpc_cell_count"].values
        for sample in samples
    ]

    positions = np.arange(1, len(samples) + 1)

    bp = ax.boxplot(
        data_by_sample,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=MEDIAN_COLOR, linewidth=2.5),
        boxprops=dict(facecolor=BOX_FACE, edgecolor=BOX_EDGE, linewidth=2),
        whiskerprops=dict(color=BOX_EDGE, linewidth=2),
        capprops=dict(color=BOX_EDGE, linewidth=2),
    )

    rng = np.random.default_rng(42)

    for i, sample in enumerate(samples, start=1):
        sample_sub = sub[sub["sample_id"] == sample]

        x = rng.normal(loc=i, scale=0.055, size=len(sample_sub))
        y = sample_sub["snpc_cell_count"].values
        colours = cmap(norm(sample_sub["ssim"].values))

        ax.scatter(
            x,
            y,
            c=colours,
            s=55,
            edgecolor=POINT_EDGE,
            linewidth=0.45,
            alpha=0.9,
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [sample_labels[s] for s in samples],
        fontsize=15
    )
    ax.set_ylim(ylim)

    ax.set_ylabel("SNpc Cell Count", fontsize=18)
    ax.set_xlabel("Biological Sample", fontsize=18)

    title = f"{processing_status.capitalize()} | {combination_name.replace('_', ' ')}"
    ax.set_title(title, fontsize=20, pad=18)

    ax.tick_params(axis="y", labelsize=15)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Local SSIM Score", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    fig.tight_layout()

    outname = f"{processing_status}_{combination_name}_snpc_count_distribution"
    fig.savefig(OUTDIR_CLEAN / f"{outname}.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(OUTDIR_CLEAN / f"{outname}.svg", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {outname}")


# ============================================================
# MAKE ALL INDIVIDUAL PLOTS
# ============================================================

for processing_status in PROCESSING_STATUSES:
    for combo in COMBINATIONS:
        combination_name = f"{combo['qc_filter']}_{combo['ssim_filter']}"
        plot_one_distribution(
            plot_df=plot_df,
            processing_status=processing_status,
            combination_name=combination_name,
        )


# ============================================================
# OPTIONAL: FACET SUMMARY FIGURE
# ============================================================

def plot_facet_summary(plot_df, processing_status):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(22, 13),
        sharey=True,
    )

    axes = axes.ravel()

    for ax, combo in zip(axes, COMBINATIONS):
        combination_name = f"{combo['qc_filter']}_{combo['ssim_filter']}"

        sub = plot_df[
            (plot_df["processing_status"] == processing_status) &
            (plot_df["combination_name"] == combination_name)
        ].copy()

        data_by_sample = [
            sub.loc[sub["sample_id"] == sample, "snpc_cell_count"].values
            for sample in samples
        ]

        positions = np.arange(1, len(samples) + 1)

        ax.boxplot(
            data_by_sample,
            positions=positions,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color=MEDIAN_COLOR, linewidth=2),
            boxprops=dict(facecolor=BOX_FACE, edgecolor=BOX_EDGE, linewidth=1.8),
            whiskerprops=dict(color=BOX_EDGE, linewidth=1.8),
            capprops=dict(color=BOX_EDGE, linewidth=1.8),
        )

        rng = np.random.default_rng(42)

        for i, sample in enumerate(samples, start=1):
            sample_sub = sub[sub["sample_id"] == sample]
            x = rng.normal(loc=i, scale=0.055, size=len(sample_sub))
            y = sample_sub["snpc_cell_count"].values
            colours = cmap(norm(sample_sub["ssim"].values))

            ax.scatter(
                x,
                y,
                c=colours,
                s=40,
                edgecolor=POINT_EDGE,
                linewidth=0.35,
                alpha=0.9,
                zorder=3,
            )

        ax.set_title(combination_name.replace("_", " "), fontsize=17)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [sample_labels[s] for s in samples],
            fontsize=13
        )
        ax.set_ylim(ylim)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="y", labelsize=13)

    fig.supxlabel("Biological Sample", fontsize=20)
    fig.supylabel("SNpc Cell Count", fontsize=20)
    fig.suptitle(
        f"{processing_status.capitalize()} SNpc Count Distributions",
        fontsize=24,
        y=0.995,
    )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=axes, pad=0.01, shrink=0.85)
    cbar.set_label("Local SSIM Score", fontsize=17)
    cbar.ax.tick_params(labelsize=14)

    fig.savefig(
        OUTDIR_CLEAN / f"{processing_status}_ALL_combinations_snpc_count_distribution_facet.png",
        dpi=DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTDIR_CLEAN / f"{processing_status}_ALL_combinations_snpc_count_distribution_facet.svg",
        bbox_inches="tight",
    )
    plt.close(fig)


for processing_status in PROCESSING_STATUSES:
    plot_facet_summary(plot_df, processing_status)

print("\nDone!")
print(f"Plots saved to: {OUTDIR}")


summary_stats = (
    plot_df
    .groupby([
        "processing_status",
        "combination_name",
        "sample_id"
    ])["snpc_cell_count"]
    .agg(
        mean_count="mean",
        std_count="std",
        n_runs="count"
    )
    .reset_index()
)

summary_stats.to_csv(
    OUTDIR_CLEAN / "summary_stats_mean_std_per_sample.csv",
    index=False
)

print("Saved summary stats")

baseline = summary_stats[
    summary_stats["combination_name"] == "all_runs_all"
].copy()

baseline = baseline.rename(columns={
    "std_count": "baseline_std"
})

sd_compare = summary_stats.merge(
    baseline[["processing_status", "sample_id", "baseline_std"]],
    on=["processing_status", "sample_id"],
    how="left"
)

sd_compare["std_reduction_pct"] = (
    (sd_compare["baseline_std"] - sd_compare["std_count"])
    / sd_compare["baseline_std"]
) * 100

sd_compare.to_csv(
    OUTDIR_CLEAN / "sd_reduction_vs_baseline.csv",
    index=False
)

print("Saved SD reduction table")

sd_summary = (
    sd_compare
    .groupby(["processing_status", "combination_name"])
    .agg(
        mean_std_reduction=("std_reduction_pct", "mean"),
        mean_std=("std_count", "mean"),
        mean_runs=("n_runs", "mean"),
        mean_count=("mean_count", "mean"),   # 👈 ADD THIS
    )
    .reset_index()
)

sd_summary.to_csv(
    OUTDIR_CLEAN / "sd_reduction_summary_across_samples.csv",
    index=False
)

print(sd_summary)

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 7))

for i, status in enumerate(["raw", "preprocessed"]):
    sub = sd_summary[sd_summary["processing_status"] == status]

    x = np.arange(len(sub)) + (i * 0.35)

    ax.bar(
        x,
        sub["mean_std_reduction"],
        width=0.35,
        label=status.capitalize(),
    )

ax.set_xticks(np.arange(len(sub)))
ax.set_xticklabels(
    sd_summary["combination_name"].unique(),
    rotation=45,
    ha="right",
    fontsize=12
)

ax.set_ylabel("Mean SD Reduction (%)", fontsize=16)
ax.set_title("Reduction in SNpc Cell Count Variability vs Baseline", fontsize=18)

ax.legend(fontsize=13)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTDIR_CLEAN / "sd_reduction_barplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    OUTDIR_CLEAN / "sd_reduction_barplot.svg",
    bbox_inches="tight"
)

plt.close()

print("Saved SD reduction plot")

fig, ax = plt.subplots(figsize=(10, 7))

for status in ["raw", "preprocessed"]:
    sub = summary_stats[summary_stats["processing_status"] == status]

    grouped = sub.groupby("combination_name").agg(
        mean=("mean_count", "mean"),
        std=("std_count", "mean")
    ).reset_index()

    ax.scatter(
        grouped["mean"],
        grouped["std"],
        label=status,
        s=120
    )

    for _, row in grouped.iterrows():
        ax.text(
            row["mean"],
            row["std"],
            row["combination_name"],
            fontsize=9
        )

ax.set_xlabel("Mean SNpc Cell Count", fontsize=14)
ax.set_ylabel("SD (Variability)", fontsize=14)
ax.set_title("Mean vs Variability Across Selection Strategies", fontsize=16)

ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTDIR / "mean_vs_sd.png", dpi=300)
plt.close()

print("Saved mean vs SD plot")


runs_summary = (
    summary_stats
    .groupby(["processing_status", "combination_name"])
    .agg(mean_runs=("n_runs", "mean"))
    .reset_index()
)


order = [
    "all_runs_all",
    "passed_qc_all",
    "passed_qc_top_50pct",
    "passed_qc_top_25pct",
]

runs_summary["combination_name"] = pd.Categorical(
    runs_summary["combination_name"],
    categories=order,
    ordered=True
)

runs_summary = runs_summary.sort_values("combination_name")
runs_summary = runs_summary.dropna()
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35

for i, status in enumerate(["raw", "preprocessed"]):
    sub = runs_summary[runs_summary["processing_status"] == status]

    x = np.arange(len(sub)) + (i * bar_width)

    ax.bar(
        x,
        sub["mean_runs"],
        width=bar_width,
        label=status.capitalize(),
    )

# xticks centered between bars
ax.set_xticks(np.arange(len(order)) + bar_width / 2)
ax.set_xticklabels(
    [name.replace("_", " ") for name in order],
    rotation=45,
    ha="right",
    fontsize=12
)

ax.set_ylabel("Mean number of runs retained", fontsize=16)
ax.set_title("Tradeoff between filtering and data retention", fontsize=18)

ax.legend(fontsize=13)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(OUTDIR /"panelC_runs_retained.png", dpi=300, bbox_inches="tight")
plt.savefig(OUTDIR /"panelC_runs_retained.svg", bbox_inches="tight")

plt.show()