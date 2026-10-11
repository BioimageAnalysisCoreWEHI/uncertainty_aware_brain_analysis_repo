# -*- coding: utf-8 -*-
"""
Created on Tue May  5 13:07:04 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os


# -----------------------------
# Settings
# -----------------------------
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")

OUTDIR = P(r"<IMAGING>\Ishrat\JoM\final_figures\Figure3 - Global QC")
os.makedirs(OUTDIR, exist_ok=True)

IOU_THRESH = 0.90
GRAD_P95_THRESH = 0.045

SAMPLE_COL = "Sample ID"

RAW_COLOR = "#7A7A7A"
PREP_COLOR = "#4C72B0"
PASS_COLOR = "#2E8B57"
FAIL_COLOR = "#B22222"

df = pd.read_csv(P(r"<IMAGING>\Ishrat\JoM\final_count_results\full_final_results_with_ssim.csv"))

def make_global_qc_long_df(df):
    records = []

    for status in ["raw", "preprocessed"]:
        temp = pd.DataFrame({
            "Sample ID": df[SAMPLE_COL],
            "Folder Name": df["Folder Name"],
            "status": status,
            "bending": df["bending"],
            "grid": df["grid"],
            "iou": df[f"{status}_iou"],
            "grad_p95": df[f"{status}_grad_p95"],
            "snpc_count": df[f"{status}_snpc_CellCount"],
            "pass_iou": df[f"{status}_pass_iou"],
            "pass_grad_p95": df[f"{status}_pass_grad_p95"],
            "pass_final": df[f"{status}_pass_final"],
            "iou_thresh": df[f"{status}_iou_thresh"],
            "grad_p95_thresh": df[f"{status}_grad_p95_thresh"],
            "image_path": df[f"{status}_image_path"],
            "boundaries_path": df[f"{status}_boundaries_path"],
            "mask_path": df[f"{status}_snpc_mask_path"],
        })

        records.append(temp)

    long_df = pd.concat(records, ignore_index=True)

    long_df["status_label"] = long_df["status"].map({
        "raw": "Raw",
        "preprocessed": "Preprocessed"
    })

    long_df["qc_label"] = np.where(long_df["pass_final"], "Pass", "Fail")

    return long_df


qc_df = make_global_qc_long_df(df)
qc_df.head()

# Create consistent sample mapping
sample_order = sorted(qc_df["Sample ID"].unique())

sample_map = {
    sample_id: f"Sample {i+1}"
    for i, sample_id in enumerate(sample_order)
}

sample_map

qc_df["Sample label"] = qc_df["Sample ID"].map(sample_map)


mapping_df = pd.DataFrame({
    "Sample label": [sample_map[s] for s in sample_order],
    "Original Sample ID": sample_order
})

mapping_df.to_csv("sample_id_mapping.csv", index=False)

mapping_df

summary_counts = (
    qc_df
    .groupby(["status_label", "Sample ID"])
    .agg(
        total_runs=("snpc_count", "count"),
        pass_iou=("pass_iou", "sum"),
        pass_grad=("pass_grad_p95", "sum"),
        pass_final=("pass_final", "sum"),
    )
    .reset_index()
)

summary_counts["pct_pass_final"] = (
    summary_counts["pass_final"] / summary_counts["total_runs"] * 100
)

summary_counts

schematic_numbers = (
    qc_df
    .groupby("status_label")
    .agg(
        total_runs=("snpc_count", "count"),
        pass_iou=("pass_iou", "sum"),
        pass_grad=("pass_grad_p95", "sum"),
        pass_final=("pass_final", "sum"),
    )
    .reset_index()
)

schematic_numbers["pct_pass_final"] = (
    schematic_numbers["pass_final"] / schematic_numbers["total_runs"] * 100
)

schematic_numbers

def plot_iou_vs_grad(qc_df, outpath=None):
    g = sns.FacetGrid(
        qc_df,
        col="status_label",
        height=5,
        aspect=1.1,
        sharex=True,
        sharey=True
    )

    g.map_dataframe(
        sns.scatterplot,
        x="iou",
        y="grad_p95",
        hue="qc_label",
        palette={"Pass": PASS_COLOR, "Fail": FAIL_COLOR},
        s=55,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.85
    )

    for ax, status in zip(g.axes.flat, qc_df["status_label"].unique()):
        sub = qc_df[qc_df["status_label"] == status]

        # Use actual thresholds from df if present
        iou_thresh = sub["iou_thresh"].dropna().median()
        grad_thresh = sub["grad_p95_thresh"].dropna().median()

        ax.axvline(iou_thresh, linestyle="--", color="black", linewidth=1.2)
        ax.axhline(grad_thresh, linestyle="--", color="black", linewidth=1.2)

        ax.set_xlabel("Tissue-atlas IoU")
        ax.set_ylabel("Jacobian gradient p95")
        ax.set_title(status)

    g.add_legend(title="Global QC")
    g.fig.suptitle("Global QC identifies feasible registrations", y=1.05)

    if outpath:
        outdir = os.path.join(OUTDIR, outpath)
        plt.savefig(outdir, dpi=300, bbox_inches="tight")

    plt.show()


plot_iou_vs_grad(qc_df,  "B_iou_vs_grad_p95_global_qc.png")



def plot_iou_vs_grad_separate(qc_df, outname_base="iou_vs_grad"):
    # -------------------------
    # Shared axis limits
    # -------------------------
    x_min, x_max = qc_df["iou"].min(), qc_df["iou"].max()
    y_min, y_max = qc_df["grad_p95"].min(), qc_df["grad_p95"].max()

    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    x_limits = (x_min - x_pad, x_max + x_pad)
    y_limits = (y_min - y_pad, y_max + y_pad)

    # -------------------------
    # Shared thresholds
    # -------------------------
    iou_thresh = qc_df["iou_thresh"].dropna().median()
    grad_thresh = qc_df["grad_p95_thresh"].dropna().median()

    # -------------------------
    # Loop over Raw / Preprocessed
    # -------------------------
    for status in ["Raw", "Preprocessed"]:
        df_plot = qc_df[qc_df["status_label"] == status].copy()

        fig, ax = plt.subplots(figsize=(6.5, 5.8))

        sns.scatterplot(
            data=df_plot,
            x="iou",
            y="grad_p95",
            hue="qc_label",
            palette={"Pass": PASS_COLOR, "Fail": FAIL_COLOR},
            s=65,
            edgecolor="black",
            linewidth=0.35,
            alpha=0.85,
            ax=ax
        )

        # thresholds
        ax.axvline(iou_thresh, linestyle="--", color="black", linewidth=1.2)
        ax.axhline(grad_thresh, linestyle="--", color="black", linewidth=1.2)

        # enforce SAME axes
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)

        ax.set_xlabel("Tissue-atlas IoU", fontsize=13)
        ax.set_ylabel("Jacobian gradient p95", fontsize=13)
        ax.set_title("Registration quality assessed using IoU and Jacobian gradient (p95)", fontsize=14, pad=12)

        ax.tick_params(axis="both", labelsize=11)

        ax.legend(
            title="Global QC",
            frameon=True,
            fontsize=10
        )

        sns.despine()
        plt.tight_layout()

        # save
        outpath = os.path.join(
            OUTDIR,
            f"{outname_base}_{status.lower()}.png"
        )

        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close()
        
plot_iou_vs_grad_separate(
    qc_df,
    outname_base="B_iou_vs_grad_p95_global_qc"
)    
        

def make_before_after_qc_df(qc_df):
    all_runs = qc_df.copy()
    all_runs["filter_stage"] = "All registrations"

    passed = qc_df[qc_df["pass_final"]].copy()
    passed["filter_stage"] = "After global QC"

    plot_df = pd.concat([all_runs, passed], ignore_index=True)

    plot_df["filter_stage"] = pd.Categorical(
        plot_df["filter_stage"],
        categories=["All registrations", "After global QC"],
        ordered=True
    )

    return plot_df


count_qc_df = make_before_after_qc_df(qc_df)
count_qc_df["Sample label"] = count_qc_df["Sample ID"].map(sample_map)

def compute_variability_summary(qc_df):
    rows = []

    for (status, sample), sub in qc_df.groupby(["status_label", "Sample label"]):
        all_counts = sub["snpc_count"].dropna()

        passed_counts = sub.loc[sub["pass_final"], "snpc_count"].dropna()

        if len(all_counts) < 2 or len(passed_counts) < 2:
            continue

        row = {
            "status_label": status,
            "Sample label": sample,

            "n_all": len(all_counts),
            "n_pass": len(passed_counts),

            "sd_all": all_counts.std(),
            "sd_pass": passed_counts.std(),

            "cv_all": all_counts.std() / all_counts.mean() * 100,
            "cv_pass": passed_counts.std() / passed_counts.mean() * 100,

            "range_all": all_counts.max() - all_counts.min(),
            "range_pass": passed_counts.max() - passed_counts.min(),

            "iqr_all": all_counts.quantile(0.75) - all_counts.quantile(0.25),
            "iqr_pass": passed_counts.quantile(0.75) - passed_counts.quantile(0.25),
        }

        row["sd_reduction_pct"] = (
            (row["sd_all"] - row["sd_pass"]) / row["sd_all"] * 100
        )

        row["cv_reduction_pct"] = (
            (row["cv_all"] - row["cv_pass"]) / row["cv_all"] * 100
        )

        row["range_reduction_pct"] = (
            (row["range_all"] - row["range_pass"]) / row["range_all"] * 100
        )

        row["iqr_reduction_pct"] = (
            (row["iqr_all"] - row["iqr_pass"]) / row["iqr_all"] * 100
        )

        rows.append(row)

    return pd.DataFrame(rows)


var_summary = compute_variability_summary(qc_df)
var_summary



def plot_sd_before_after(var_summary, outpath=None):
    plot_df = var_summary.melt(
        id_vars=["Sample ID", "status_label"],
        value_vars=["sd_all", "sd_pass"],
        var_name="stage",
        value_name="sd"
    )

    plot_df["stage"] = plot_df["stage"].map({
        "sd_all": "All registrations",
        "sd_pass": "After global QC"
    })

    plot_df["stage"] = pd.Categorical(
        plot_df["stage"],
        categories=["All registrations", "After global QC"],
        ordered=True
    )

    plt.figure(figsize=(7, 5))

    ax = sns.pointplot(
        data=plot_df,
        x="stage",
        y="sd",
        hue="status_label",
        dodge=0.25,
        errorbar=None,
        markers="o",
        linestyles="-",
        palette={"Raw": RAW_COLOR, "Preprocessed": PREP_COLOR}
    )

    sns.stripplot(
        data=plot_df,
        x="stage",
        y="sd",
        hue="status_label",
        dodge=True,
        alpha=0.8,
        size=6,
        palette={"Raw": RAW_COLOR, "Preprocessed": PREP_COLOR},
        ax=ax
    )

    # Clean duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Input")

    ax.set_xlabel("")
    ax.set_ylabel("Standard deviation of SNpc count")
    ax.set_title("Global QC reduces count variability")

    plt.tight_layout()

    if outpath:
        outdir = os.path.join(OUTDIR, outpath)
        plt.savefig(outdir, dpi=300, bbox_inches="tight")


    plt.show()


plot_sd_before_after(var_summary, "F1_sd_before_after_global_qc.png")



def plot_variability_reduction(var_summary, metric="sd_reduction_pct", outpath=None):
    ylabel_map = {
        "sd_reduction_pct": "SD reduction after global QC (%)",
        "cv_reduction_pct": "CV reduction after global QC (%)",
        "range_reduction_pct": "Range reduction after global QC (%)",
        "iqr_reduction_pct": "IQR reduction after global QC (%)",
    }

    # Ensure consistent sample ordering (Sample 1, Sample 2, ...)
    sample_order = sorted(
        var_summary["Sample label"].unique(),
        key=lambda x: int(x.split()[-1])
    )

    plt.figure(figsize=(7, 5))

    ax = sns.barplot(
        data=var_summary,
        x="Sample label",
        y=metric,
        hue="status_label",
        order=sample_order,
        hue_order=["Raw", "Preprocessed"],  # 🔥 THIS fixes left/right order
        palette={"Raw": RAW_COLOR, "Preprocessed": PASS_COLOR},
        edgecolor="black",
        linewidth=0.8
    )

    ax.axhline(0, color="black", linewidth=1)

    ax.set_xlabel("Sample")
    ax.set_ylabel(ylabel_map.get(metric, metric))
    ax.set_title("Global QC reduces registration-induced variability")

    ax.legend(title="Input")
    sns.despine()
    plt.tight_layout()

    if outpath:
        outdir = os.path.join(OUTDIR, outpath)
        plt.savefig(outdir, dpi=300, bbox_inches="tight")

    plt.show()

plot_variability_reduction(
    var_summary,
    metric="sd_reduction_pct",
    outpath="F2_sd_reduction_percent_global_qc.png"
)

plot_variability_reduction(
    var_summary,
    metric="cv_reduction_pct",
    outpath="F3_cv_reduction_percent_global_qc.png"
)


poor_iou_examples = (
    qc_df
    .query("pass_iou == False")
    .sort_values("iou", ascending=True)
    [[
        "Sample ID", "status_label", "Folder Name",
        "bending", "grid", "iou", "grad_p95",
        "image_path", "boundaries_path"
    ]]
)

poor_iou_examples.head(10)



high_deformation_examples = (
    qc_df
    .query("pass_iou == True and pass_grad_p95 == False")
    .sort_values("grad_p95", ascending=False)
    [[
        "Sample ID", "status_label", "Folder Name",
        "bending", "grid", "iou", "grad_p95",
        "image_path", "boundaries_path"
    ]]
)

high_deformation_examples.head(10)



def plot_qc_failure_modes(qc_df):
    temp = qc_df.copy()

    temp["failure_mode"] = np.select(
        [
            temp["pass_iou"] == False,
            (temp["pass_iou"] == True) & (temp["pass_grad_p95"] == False),
            temp["pass_final"] == True
        ],
        [
            "Fail IoU",
            "Fail deformation",
            "Pass"
        ],
        default="Unclassified"
    )

    summary = (
        temp
        .groupby(["status_label", "failure_mode"])
        .size()
        .reset_index(name="count")
    )

    plt.figure(figsize=(6, 5))

    sns.barplot(
        data=summary,
        x="status_label",
        y="count",
        hue="failure_mode",
        palette={
            "Fail IoU": "#D62728",
            "Fail deformation": "#FF7F0E",
            "Pass": "#2E8B57",
            "Unclassified": "#BBBBBB"
        },
        edgecolor="black",
        linewidth=0.8
    )

    plt.ylabel("Number of registrations")
    plt.xlabel("")
    plt.title("Global QC failure modes")

    plt.tight_layout()
    plt.show()


plot_qc_failure_modes(qc_df)






import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ymin = count_qc_df["snpc_count"].min()
ymax = count_qc_df["snpc_count"].max()

# add padding so points aren’t squashed at edges
padding = (ymax - ymin) * 0.05
ymin -= padding
ymax += padding

sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")

BOX_PALETTE = {
    "All registrations": "#D0D0D0",   # light grey
    "After global QC": "#A5D6A7"      # light green
}

POINT_PALETTE = {
    "All registrations": "#4A4A4A",   # dark grey
    "After global QC": "#1B5E20"      # dark green
}
def make_before_after_qc_df(qc_df):
    all_runs = qc_df.copy()
    all_runs["qc_stage"] = "All registrations"

    passed = qc_df[qc_df["pass_final"]].copy()
    passed["qc_stage"] = "After global QC"

    plot_df = pd.concat([all_runs, passed], ignore_index=True)

    plot_df["qc_stage"] = pd.Categorical(
        plot_df["qc_stage"],
        categories=["All registrations", "After global QC"],
        ordered=True
    )

    return plot_df


count_qc_df = make_before_after_qc_df(qc_df)


def plot_before_after_qc_by_status(
    count_qc_df,
    status_label="Raw",
    y_limits=None,
    outpath=None
):
    plot_df = count_qc_df[count_qc_df["status_label"] == status_label].copy()

    plt.figure(figsize=(10, 5.5))

    ax = sns.boxplot(
        data=plot_df,
        x="Sample label",
        y="snpc_count",
        hue="qc_stage",
        palette=BOX_PALETTE,
        showfliers=False,
        linewidth=1.2,
        width=0.55
    )

    sns.stripplot(
        data=plot_df,
        x="Sample label",
        y="snpc_count",
        hue="qc_stage",
        palette=POINT_PALETTE,
        dodge=True,
        jitter=0.18,
        size=5,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.3,
        ax=ax
    )

    # 🔥 enforce SAME y-axis
    if y_limits is not None:
        ax.set_ylim(y_limits)

    # clean legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="QC stage")

    ax.set_title(f"SNpc TH+ cell counts before and after global QC — {status_label}")
    ax.set_xlabel("Biological sample")
    ax.set_ylabel("SNpc TH+ cell count")

    sns.despine()
    plt.tight_layout()

    if outpath:
        outdir = os.path.join(OUTDIR, outpath)
        plt.savefig(outdir, dpi=300, bbox_inches="tight")

    plt.show()

y_limits = (ymin, ymax)

plot_before_after_qc_by_status(
    count_qc_df,
    status_label="Raw",
    y_limits=y_limits,
    outpath="raw_counts_qc.png"
)

plot_before_after_qc_by_status(
    count_qc_df,
    status_label="Preprocessed",
    y_limits=y_limits,
    outpath="preprocessed_counts_qc.png"
)



def plot_global_qc_pass_proportion_by_sample(qc_df, outpath=None):
    plot_df = qc_df.copy()

    pass_summary = (
        plot_df
        .groupby(["Sample label", "status_label"])
        .agg(
            total_registrations=("pass_final", "count"),
            passed_registrations=("pass_final", "sum")
        )
        .reset_index()
    )

    pass_summary["percent_pass"] = (
        pass_summary["passed_registrations"] /
        pass_summary["total_registrations"] * 100
    )

    sample_order = sorted(
        pass_summary["Sample label"].unique(),
        key=lambda x: int(x.split()[-1])
    )

    status_order = ["Raw", "Preprocessed"]

    plt.figure(figsize=(8, 5))

    ax = sns.barplot(
        data=pass_summary,
        x="Sample label",
        y="percent_pass",
        hue="status_label",
        order=sample_order,
        hue_order=status_order,
        palette={"Raw": RAW_COLOR, "Preprocessed": PASS_COLOR},
        edgecolor="black",
        linewidth=1.0
    )

    # Annotate bars robustly
    sorted_summary = pass_summary.sort_values(["Sample label", "status_label"])

    for patch, (_, row) in zip(ax.patches, sorted_summary.iterrows()):
        height = patch.get_height()
        x = patch.get_x() + patch.get_width() / 2

        ax.text(
            x,
            height + 1.5,
            f"{row['percent_pass']:.0f}%\n"
            f"{int(row['passed_registrations'])}/{int(row['total_registrations'])}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold"
        )

    ax.set_ylim(0, 95)
    ax.set_xlabel("Biological sample")
    ax.set_ylabel("Registrations passing global QC (%)")
    ax.set_title("Preprocessing increases globally valid registrations per sample")

    ax.legend(title="Input")
    sns.despine()
    plt.tight_layout()

    if outpath:
        outdir = os.path.join(OUTDIR, outpath)
        plt.savefig(outdir, dpi=300, bbox_inches="tight")

    plt.show()

    return pass_summary


pass_summary_by_sample = plot_global_qc_pass_proportion_by_sample(
    qc_df,
    "global_qc_pass_proportion_by_sample_anonymised.png"
)




