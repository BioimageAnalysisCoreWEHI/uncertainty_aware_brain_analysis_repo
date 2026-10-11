# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:43:44 2026

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
from matplotlib.patches import Patch

# Optional only needed if loading TIFF label images
import tifffile as tiff


# ============================================================
# Paths
# ============================================================

PROB_CSV = P(r"<IMAGING>\Ishrat\JoM\final_count_results\cell_region_probability_outputs\ALL_cell_region_grouped_probabilities_long.csv")
OUTDIR = P(r"<IMAGING>\Ishrat\JoM\final_figures\Figure5 - Consensus Regions")

os.makedirs(OUTDIR, exist_ok=True)

prob_df = pd.read_csv(PROB_CSV)


# ============================================================
# Settings
# ============================================================

REGION_COLOURS = {
    "SNc": "#E41A1C",
    "VTA": "#377EB8",
    "SNr": "#4DAF4A",
    "MB": "#984EA3",
    "other": "#BDBDBD",
}

BOUNDARY_CMAP = "magma"

# confidence below this gets called a boundary/uncertain cell
BOUNDARY_THRESH = 0.80


# ============================================================
# Convert long probability df to one row per cell
# ============================================================

def get_dominant_region_df(prob_df, boundary_thresh=0.8):
    """
    Converts long df:
        one row per cell-region probability

    Into:
        one row per cell with dominant region and confidence
    """

    group_cols = ["sample_id", "processing_status", "cell_id"]

    idx = (
        prob_df
        .groupby(group_cols)["probability"]
        .idxmax()
    )

    dom_df = prob_df.loc[idx].copy()

    dom_df = dom_df.rename(columns={
        "region_group": "dominant_region",
        "probability": "confidence"
    })

    dom_df["is_boundary"] = dom_df["confidence"] < boundary_thresh

    return dom_df.reset_index(drop=True)


dom_df = get_dominant_region_df(
    prob_df,
    boundary_thresh=BOUNDARY_THRESH
)


# ============================================================
# Centroid plot
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

COLOURS = {
    "SNc": "#E41A1C",              # red
    "VTA": "#377EB8",              # blue
    "MB": "#984EA3",               # purple
    "Other": "#BDBDBD",            # grey
    "Uncertain": "#FFD92F"         # yellow
}

def plot_centroid_slab_with_uncertainty(
    dom_df,
    sample_id,
    processing_status="raw",
    z_min=105,
    z_max=110,
    boundary_thresh=0.8,
    outpath=None,
    point_size=18
):
    # -------------------------
    # Filter slab
    # -------------------------
    df_plot = dom_df[
        (dom_df["sample_id"].astype(str) == str(sample_id)) &
        (dom_df["processing_status"] == processing_status) &
        (dom_df["z"] >= z_min) &
        (dom_df["z"] <= z_max)
    ].copy()

    if df_plot.empty:
        raise ValueError("No cells found in this slab.")

    # -------------------------
    # Assign plotting category
    # -------------------------
    df_plot["plot_region"] = "Other"

    # low confidence → uncertain
    df_plot.loc[
        df_plot["confidence"] < boundary_thresh,
        "plot_region"
    ] = "Uncertain"

    # confident cells → assign real region
    confident = df_plot["confidence"] >= boundary_thresh

    df_plot.loc[
        confident & (df_plot["dominant_region"] == "SNc"),
        "plot_region"
    ] = "SNc"

    df_plot.loc[
        confident & (df_plot["dominant_region"] == "VTA"),
        "plot_region"
    ] = "VTA"

    df_plot.loc[
        confident & (df_plot["dominant_region"] == "MB"),
        "plot_region"
    ] = "MB"

    # everything else stays "Other"

    # -------------------------
    # Plot
    # -------------------------
    fig, ax = plt.subplots(figsize=(7, 7))

    plot_order = ["Other", "MB", "VTA", "SNc", "Uncertain"]

    for region in plot_order:
        sub = df_plot[df_plot["plot_region"] == region]

        if len(sub) == 0:
            continue

        ax.scatter(
            sub["x"],
            sub["y"],
            s=point_size,
            c=COLOURS[region],
            edgecolor="black" if region != "Other" else "none",
            linewidth=0.25,
            alpha=0.4 if region == "Other" else 0.9,
            label=region,
            zorder=2 if region == "Other" else 3
        )

    ax.set_aspect("equal")
    ax.invert_yaxis()

    ax.set_title(
        f"Cell assignment (z {z_min}–{z_max})\n"
        f"Sample {sample_id} — {processing_status}",
        fontsize=14,
        pad=12
    )

    ax.set_xlabel("x centroid coordinate", fontsize=12)
    ax.set_ylabel("y centroid coordinate", fontsize=12)
    ax.tick_params(labelsize=10)

    # -------------------------
    # Legend (rectangles)
    # -------------------------
    handles = [
        Patch(facecolor=COLOURS[r], edgecolor="black", label=r)
        for r in ["SNc", "VTA", "MB", "Other", "Uncertain"]
    ]

    ax.legend(
        handles=handles,
        title="Cell assignment",
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=5,
        fontsize=11,
        title_fontsize=12
    )
    
    plt.subplots_adjust(bottom=0.18)
    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()
    
    
    
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# -------------------------
# Colours
# -------------------------

COLOURS = {
    "SNc": "#D7191C",        # strong red
    "VTA": "#2C7BB6",        # strong blue
    "MB": "#7B3294",         # purple
    "Uncertain": "#FFD92F",  # yellow
    "Other": "#D0D0D0"       # light grey
}

PLOT_ORDER = ["Other", "MB", "VTA", "SNc", "Uncertain"]
LEGEND_ORDER = ["SNc", "VTA", "MB", "Uncertain", "Other"]


# -------------------------
# Slab centroid plot
# -------------------------

def plot_centroid_slab_polished(
    dom_df,
    sample_id,
    processing_status="preprocessed",
    z_min=96,
    z_max=99,
    boundary_thresh=0.8,
    outpath=None,
    point_size=22,
    axis_limits=None,
    jitter=False,
    jitter_sd=1.5,
):
    df_plot = dom_df[
        (dom_df["sample_id"].astype(str) == str(sample_id)) &
        (dom_df["processing_status"] == processing_status) &
        (dom_df["z"] >= z_min) &
        (dom_df["z"] <= z_max)
    ].copy()

    if df_plot.empty:
        raise ValueError(
            f"No cells found for sample {sample_id}, {processing_status}, z {z_min}–{z_max}"
        )

    # -------------------------
    # Assign display category
    # -------------------------

    df_plot["plot_region"] = "Other"

    confident = df_plot["confidence"] >= boundary_thresh

    df_plot.loc[confident & (df_plot["dominant_region"] == "SNc"), "plot_region"] = "SNc"
    df_plot.loc[confident & (df_plot["dominant_region"] == "VTA"), "plot_region"] = "VTA"
    df_plot.loc[confident & (df_plot["dominant_region"] == "MB"), "plot_region"] = "MB"

    # low-confidence cells override region assignment
    df_plot.loc[~confident, "plot_region"] = "Uncertain"

    # Optional tiny jitter only for visual overlap reduction
    if jitter:
        rng = np.random.default_rng(42)
        df_plot["plot_x"] = df_plot["x"] + rng.normal(0, jitter_sd, len(df_plot))
        df_plot["plot_y"] = df_plot["y"] + rng.normal(0, jitter_sd, len(df_plot))
    else:
        df_plot["plot_x"] = df_plot["x"]
        df_plot["plot_y"] = df_plot["y"]

    # -------------------------
    # Plot
    # -------------------------

    sns.set_context("paper", font_scale=1.35)
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(7.4, 5.4))

    style = {
        "Other": {
            "s": point_size * 0.65,
            "alpha": 0.8,
            "edgecolor": "none",
            "linewidth": 0,
            "zorder": 1,
        },
        "MB": {
            "s": point_size,
            "alpha": 0.85,
            "edgecolor": "black",
            "linewidth": 0.25,
            "zorder": 2,
        },
        "VTA": {
            "s": point_size,
            "alpha": 0.9,
            "edgecolor": "black",
            "linewidth": 0.25,
            "zorder": 3,
        },
        "SNc": {
            "s": point_size,
            "alpha": 0.9,
            "edgecolor": "black",
            "linewidth": 0.25,
            "zorder": 4,
        },
        "Uncertain": {
            "s": point_size * 1.35,
            "alpha": 0.9,
            "edgecolor": "black",
            "linewidth": 0.35,
            "zorder": 5,
        },
    }

    for region in PLOT_ORDER:
        sub = df_plot[df_plot["plot_region"] == region]

        if sub.empty:
            continue

        ax.scatter(
            sub["plot_x"],
            sub["plot_y"],
            c=COLOURS[region],
            **style[region],
        )

    ax.set_aspect("equal")
    ax.invert_yaxis()

    if axis_limits is not None:
        ax.set_xlim(axis_limits["x"])
        ax.set_ylim(axis_limits["y"])

    ax.set_title(
        f"Cell assignment — Sample {sample_id} ({processing_status})\n"
        f"z = {z_min}–{z_max}",
        fontsize=15,
        pad=12,
    )

    ax.set_xlabel("x centroid coordinate", fontsize=13)
    ax.set_ylabel("y centroid coordinate", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    # lighter grid
    ax.grid(True, linewidth=0.6, alpha=0.35)

    # legend as rectangles
    handles = [
        Patch(
            facecolor=COLOURS[region],
            edgecolor="black",
            label=region,
        )
        for region in LEGEND_ORDER
    ]

    ax.legend(
        handles=handles,
        title="Cell assignment",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
        handlelength=1.8,
        handleheight=1.1,
        loc="lower right",
    )

    sns.despine(ax=ax)

    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=600, bbox_inches="tight")
        plt.savefig(outpath.replace(".png", ".svg"), bbox_inches="tight")

    plt.show()
    
def plot_cell_centroids_by_region(
    dom_df,
    sample_id,
    processing_status="raw",
    outpath=None,
    boundary_thresh=0.8,
    point_size=10
):
    df_plot = dom_df[
        (dom_df["sample_id"].astype(str) == str(sample_id)) &
        (dom_df["processing_status"] == processing_status)
    ].copy()

    if df_plot.empty:
        raise ValueError(
            f"No cells found for sample_id={sample_id}, "
            f"processing_status={processing_status}"
        )

    confident = df_plot[~df_plot["is_boundary"]]
    boundary = df_plot[df_plot["is_boundary"]]

    fig, ax = plt.subplots(figsize=(7, 7))

    # confident cells coloured by dominant region
    for region, sub in confident.groupby("dominant_region"):
        ax.scatter(
            sub["x"],
            sub["y"],
            s=point_size,
            c=REGION_COLOURS.get(region, "#000000"),
            alpha=0.9,
            linewidth=0,
            label=region
        )

    # boundary cells coloured by confidence
    sc = None
    if len(boundary) > 0:
        sc = ax.scatter(
            boundary["x"],
            boundary["y"],
            s=point_size * 1.3,
            c=boundary["confidence"],
            cmap=BOUNDARY_CMAP,
            vmin=0,
            vmax=boundary_thresh,
            edgecolor="black",
            linewidth=0.25,
            alpha=0.95,
            label="Boundary / low confidence"
        )

    ax.set_aspect("equal")
    ax.invert_yaxis()

    ax.set_title(
        f"Cell dominant region assignment\n"
        f"Sample {sample_id} — {processing_status}",
        fontsize=14,
        pad=12
    )

    ax.set_xlabel("x centroid coordinate", fontsize=12)
    ax.set_ylabel("y centroid coordinate", fontsize=12)

    # region legend
    region_handles = [
        Patch(
            facecolor=colour,
            edgecolor="black",
            label=region
        )
        for region, colour in REGION_COLOURS.items()
        if region in df_plot["dominant_region"].unique()
    ]

    ax.legend(
        handles=region_handles,
        title="Dominant region",
        frameon=True,
        bbox_to_anchor=(1.04, 1),
        loc="upper left"
    )

    # confidence colourbar
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(
            f"Boundary cell confidence\n(< {boundary_thresh})",
            fontsize=11
        )
        cbar.ax.tick_params(labelsize=10)

    ax.tick_params(labelsize=10)

    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()


# ============================================================
# Optional: plot raw + preprocessed side-by-side
# ============================================================

def plot_raw_vs_preprocessed_centroids(
    dom_df,
    sample_id,
    outpath=None,
    boundary_thresh=0.8,
    point_size=10
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

    statuses = ["raw", "preprocessed"]

    sc = None

    for ax, status in zip(axes, statuses):
        df_plot = dom_df[
            (dom_df["sample_id"].astype(str) == str(sample_id)) &
            (dom_df["processing_status"] == status)
        ].copy()

        confident = df_plot[~df_plot["is_boundary"]]
        boundary = df_plot[df_plot["is_boundary"]]

        for region, sub in confident.groupby("dominant_region"):
            ax.scatter(
                sub["x"],
                sub["y"],
                s=point_size,
                c=REGION_COLOURS.get(region, "#000000"),
                alpha=0.9,
                linewidth=0
            )

        if len(boundary) > 0:
            sc = ax.scatter(
                boundary["x"],
                boundary["y"],
                s=point_size * 1.3,
                c=boundary["confidence"],
                cmap=BOUNDARY_CMAP,
                vmin=0,
                vmax=boundary_thresh,
                edgecolor="black",
                linewidth=0.25,
                alpha=0.95
            )

        ax.set_title(status.capitalize(), fontsize=14)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel("x centroid coordinate", fontsize=12)
        ax.tick_params(labelsize=10)

    axes[0].set_ylabel("y centroid coordinate", fontsize=12)

    fig.suptitle(
        f"Cell dominant region assignment — Sample {sample_id}",
        fontsize=15,
        y=1.02
    )

    region_handles = [
        Patch(
            facecolor=colour,
            edgecolor="black",
            label=region
        )
        for region, colour in REGION_COLOURS.items()
    ]

    axes[1].legend(
        handles=region_handles,
        title="Dominant region",
        frameon=True,
        bbox_to_anchor=(1.04, 1),
        loc="upper left"
    )

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, fraction=0.035, pad=0.08)
        cbar.set_label(
            f"Boundary cell confidence\n(< {boundary_thresh})",
            fontsize=11
        )
        cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()


# ============================================================
# Optional: colour the full segmentation label image
# ============================================================
def plot_region_label_slices(
    label_img,
    dom_df,
    sample_id,
    processing_status="raw",
    z_slices=[80, 90, 100],
    outpath=None
):
    from matplotlib.patches import Patch

    n = len(z_slices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]

    for ax, z in zip(axes, z_slices):
        # 2D label slice
        label_slice = label_img[z]

        # only cells with centroid in this slice
        slice_df = dom_df[
            (dom_df["sample_id"].astype(str) == str(sample_id)) &
            (dom_df["processing_status"] == processing_status) &
            (dom_df["z"].round().astype(int) == z)
        ].copy()

        rgb = make_region_coloured_label_image(
            label_img=label_slice,
            dom_df=slice_df,
            sample_id=sample_id,
            processing_status=processing_status,
            boundary_thresh=BOUNDARY_THRESH
        )

        ax.imshow(rgb)
        ax.set_title(f"z = {z}", fontsize=13)
        ax.axis("off")

    region_handles = [
        Patch(
            facecolor=colour,
            edgecolor="black",
            label=region
        )
        for region, colour in REGION_COLOURS.items()
    ]

    fig.legend(
        handles=region_handles,
        title="Dominant region",
        frameon=True,
        loc="center right",
        bbox_to_anchor=(1.04, 0.5)
    )

    fig.suptitle(
        f"Region-coloured cell labels — Sample {sample_id} — {processing_status}",
        fontsize=15,
        y=1.02
    )

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()
    
    
def make_region_coloured_label_image(
    label_img,
    dom_df,
    sample_id,
    processing_status="raw",
    boundary_thresh=0.8
):
    """
    Creates an RGB image where each labelled cell is coloured by:
    - dominant region if confident
    - boundary confidence colour if low-confidence
    """

    from matplotlib.colors import to_rgb
    from matplotlib import cm

    df_plot = dom_df[
        (dom_df["sample_id"].astype(str) == str(sample_id)) &
        (dom_df["processing_status"] == processing_status)
    ].copy()

    rgb = np.zeros(label_img.shape + (3,), dtype=np.float32)

    cmap = cm.get_cmap(BOUNDARY_CMAP)

    for _, row in df_plot.iterrows():
        cell_id = int(row["cell_id"])

        if row["is_boundary"]:
            # normalise confidence from 0 to boundary_thresh
            norm_conf = np.clip(row["confidence"] / boundary_thresh, 0, 1)
            colour = cmap(norm_conf)[:3]
        else:
            colour = to_rgb(
                REGION_COLOURS.get(row["dominant_region"], "#000000")
            )

        rgb[label_img == cell_id] = colour

    return rgb


def plot_region_label_image(
    label_img,
    dom_df,
    sample_id,
    processing_status="raw",
    z_slice=None,
    outpath=None
):
    """
    If label_img is 3D, provide z_slice.
    If z_slice is None, uses max projection.
    """

    rgb = make_region_coloured_label_image(
        label_img=label_img,
        dom_df=dom_df,
        sample_id=sample_id,
        processing_status=processing_status,
        boundary_thresh=BOUNDARY_THRESH
    )

    if rgb.ndim == 4:
        if z_slice is None:
            show_img = rgb.max(axis=0)
            title_extra = "max projection"
        else:
            show_img = rgb[z_slice]
            title_extra = f"z={z_slice}"
    else:
        show_img = rgb
        title_extra = "2D"

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(show_img)
    ax.set_title(
        f"Region-coloured segmentation\n"
        f"Sample {sample_id} — {processing_status} — {title_extra}",
        fontsize=14
    )
    ax.axis("off")

    region_handles = [
        Patch(
            facecolor=colour,
            edgecolor="black",
            label=region
        )
        for region, colour in REGION_COLOURS.items()
    ]

    ax.legend(
        handles=region_handles,
        title="Dominant region",
        frameon=True,
        bbox_to_anchor=(1.04, 1),
        loc="upper left"
    )

    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.show()




# ============================================================
# Example usage
# ============================================================

sample_id = "pp451"

plot_cell_centroids_by_region(
    dom_df,
    sample_id=sample_id,
    processing_status="raw",
    outpath=os.path.join(OUTDIR, f"sample_{sample_id}_raw_centroids_region_confidence.png"),
    boundary_thresh=BOUNDARY_THRESH
)


plot_centroid_slab_with_uncertainty(
    dom_df,
    sample_id="162",
    processing_status="preprocessed",
    z_min=96,
    z_max=99,
    boundary_thresh=0.8,
    outpath=os.path.join(OUTDIR, "sample_162_raw_z105_110_slab.png")
)

plot_centroid_slab_polished(
    dom_df,
    sample_id="162",
    processing_status="preprocessed",
    z_min=96,
    z_max=99,
    boundary_thresh=0.8,
    outpath=os.path.join(OUTDIR, "sample_162_preprocessed_z96_99_polished.png"),
    point_size=22,
    jitter=False
)


plot_cell_centroids_by_region(
    dom_df,
    sample_id=sample_id,
    processing_status="preprocessed",
    outpath=os.path.join(OUTDIR, f"sample_{sample_id}_preprocessed_centroids_region_confidence.png"),
    boundary_thresh=BOUNDARY_THRESH
)

plot_raw_vs_preprocessed_centroids(
    dom_df,
    sample_id=sample_id,
    outpath=os.path.join(OUTDIR, f"sample_{sample_id}_raw_vs_preprocessed_centroids.png"),
    boundary_thresh=BOUNDARY_THRESH
)


# ============================================================
# Example usage with label image
# ============================================================

# label_img_path = r"<IMAGING>\path\to\cell_segmentation_labels.tif"
# label_img = tiff.imread(label_img_path)
#
# plot_region_label_image(
#     label_img,
#     dom_df,
#     sample_id=sample_id,
#     processing_status="raw",
#     z_slice=None,   # use max projection
#     outpath=os.path.join(OUTDIR, f"sample_{sample_id}_raw_label_image_region_coloured.png")
# )