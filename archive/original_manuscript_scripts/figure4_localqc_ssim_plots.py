# -*- coding: utf-8 -*-
"""
Created on Wed May  6 09:47:25 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Paths
# -------------------------

OUTDIR = P(r"<IMAGING>\Ishrat\JoM\final_figures\Figure4 - Local QC + SSIM")
CSV_PATH = P(r"<IMAGING>\Ishrat\JoM\final_count_results\full_final_results_with_ssim.csv")

os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

SAMPLE_COL = "Sample ID"


# -------------------------
# Make long dataframe
# -------------------------

def make_long_count_df(df, top_frac=0.5):
    rows = []

    for status in ["raw", "preprocessed"]:
        temp = pd.DataFrame({
            "Sample ID": df[SAMPLE_COL],
            "Folder Name": df["Folder Name"],
            "bending": df["bending"],
            "grid": df["grid"],
            "smoothing": df["smoothing"],
            "status": status,
            "status_label": "Raw" if status == "raw" else "Preprocessed",
            "snpc_count": df[f"{status}_snpc_CellCount"],
            "ssim_value": df[f"{status}_ssim_value"],
            "pass_final": df[f"{status}_pass_final"],
        })

        rows.append(temp)

    long_df = pd.concat(rows, ignore_index=True)

    # Clean sample labels
    sample_order = sorted(long_df["Sample ID"].unique())
    sample_map = {
        sample_id: f"Sample {i + 1}"
        for i, sample_id in enumerate(sample_order)
    }

    long_df["Sample label"] = long_df["Sample ID"].map(sample_map)

    mapping_df = pd.DataFrame({
        "Sample label": [sample_map[s] for s in sample_order],
        "Original Sample ID": sample_order
    })

    mapping_df.to_csv(
        os.path.join(OUTDIR, "sample_id_mapping.csv"),
        index=False
    )

    # Stage 1: all registrations
    all_runs = long_df.copy()
    all_runs["qc_stage"] = "All registrations"

    # Stage 2: after global QC
    passed = long_df[long_df["pass_final"]].copy()
    passed["qc_stage"] = "After global QC"

    # Stage 3: top 50% SSIM after global QC, per sample and status
    top_ssim = (
        passed
        .groupby(["Sample ID", "status_label"], group_keys=False)
        .apply(lambda g: g[g["ssim_value"] >= g["ssim_value"].quantile(1 - top_frac)])
        .copy()
    )

    top_ssim["qc_stage"] = f"Top {int(top_frac * 100)}% SSIM"

    plot_df = pd.concat([all_runs, passed, top_ssim], ignore_index=True)

    plot_df["qc_stage"] = pd.Categorical(
        plot_df["qc_stage"],
        categories=[
            "All registrations",
            "After global QC",
            f"Top {int(top_frac * 100)}% SSIM"
        ],
        ordered=True
    )

    return plot_df, mapping_df


# -------------------------
# Plot function
# -------------------------

def plot_qc_stages_by_ssim(
    plot_df,
    status_label,
    y_limits,
    ssim_limits,
    outpath=None
):
    df_plot = plot_df[plot_df["status_label"] == status_label].copy()

    sns.set_context("paper", font_scale=2.0)
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6))

    stage_order = [
        "All registrations",
        "After global QC",
        "Top 50% SSIM"
    ]

    box_palette = {
        "All registrations":  "#F5F2F7",
        "After global QC": "#D9D9D9",
        "Top 50% SSIM": "#505054",
    }

    sns.boxplot(
        data=df_plot,
        x="Sample label",
        y="snpc_count",
        hue="qc_stage",
        order=sorted(df_plot["Sample label"].unique()),
        hue_order=stage_order,
        palette=box_palette,
        showfliers=False,
        linewidth=1.2,
        width=0.7,
        ax=ax
    )

    if ax.legend_:
        ax.legend_.remove()

    sample_order = sorted(df_plot["Sample label"].unique())

    stage_offsets = {
        "All registrations": -0.25,
        "After global QC": 0.0,
        "Top 50% SSIM": 0.25,
    }

    rng = np.random.default_rng(42)
    sc = None

    for stage in stage_order:
        sub = df_plot[df_plot["qc_stage"] == stage].copy()

        x = np.array([
            sample_order.index(s) + stage_offsets[stage]
            for s in sub["Sample label"]
        ])

        x = x + rng.normal(0, 0.03, size=len(x))

        # sc = ax.scatter(
        #     x,
        #     sub["snpc_count"],
        #     c=sub["ssim_value"],
        #     cmap="magma",
        #     vmin=ssim_limits[0],
        #     vmax=ssim_limits[1],
        #     s=42,
        #     edgecolor="black",
        #     linewidth=0.35,
        #     alpha=0.9,
        #     zorder=3
        # )

    ax.set_ylim(y_limits)
    ax.set_xticks(range(len(sample_order)))
    ax.set_xticklabels(sample_order)

    # cbar = fig.colorbar(sc, ax=ax)
    # cbar.set_label("Local SSIM value", fontsize=13)
    # cbar.ax.tick_params(labelsize=11)

    from matplotlib.patches import Patch

    handles = [
        Patch(
            facecolor=box_palette[stage],
            edgecolor="black",
            label=stage
        )
        for stage in stage_order
    ]
    
    ax.legend(handles=handles, title="QC stage", frameon=True)
    ax.set_title(
        f"SNpc TH+ cell counts across QC stages — {status_label}",
        fontsize=20,
        pad=14
    )
    ax.set_xlabel("Biological sample", fontsize=18)
    ax.set_ylabel("SNpc TH+ cell count", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)

    sns.despine()
    plt.tight_layout()

    if outpath:
        plt.savefig(
            os.path.join(OUTDIR, outpath),
            dpi=600,
            bbox_inches="tight"
        )

    plt.show()


# -------------------------
# Run
# -------------------------

plot_df, mapping_df = make_long_count_df(df, top_frac=0.5)

# y-axis from ALL registrations, so this matches before/after QC plots
ymin = plot_df["snpc_count"].min()
ymax = plot_df["snpc_count"].max()
ypad = (ymax - ymin) * 0.05
y_limits = (ymin - ypad, ymax + ypad)

# shared SSIM colour scale across raw + preprocessed + all stages
ssim_limits = (
    plot_df["ssim_value"].min(),
    plot_df["ssim_value"].max()
)

plot_qc_stages_by_ssim(
    plot_df,
    status_label="Raw",
    y_limits=y_limits,
    ssim_limits=ssim_limits,
    outpath="raw_all_global_qc_top50_ssim_big_text.png"
)

plot_qc_stages_by_ssim(
    plot_df,
    status_label="Preprocessed",
    y_limits=y_limits,
    ssim_limits=ssim_limits,
    outpath="preprocessed_all_global_qc_top50_ssim_big_text.png"
)