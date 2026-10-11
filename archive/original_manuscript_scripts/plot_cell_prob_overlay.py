# -*- coding: utf-8 -*-
"""
Created on Wed May  6 12:25:36 2026

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
from matplotlib.colors import to_rgb
import tifffile as tiff


# -------------------------
# Colours
# -------------------------

COLOURS = {
    "SNc": "#D7191C",        # red
    "VTA": "#2C7BB6",        # blue
    "MB": "#7B3294",         # purple
    "Uncertain": "#FFD92F",  # yellow
    "Other": "#D0D0D0"       # grey
}

LEGEND_ORDER = ["SNc", "VTA", "MB", "Uncertain", "Other"]
OUTDIR = P(r"<IMAGING>\Ishrat\JoM\final_figures\Figure5 - Consensus Regions")

# -------------------------
# Helper: make dominant cell df
# -------------------------

def get_dominant_region_df(prob_df, boundary_thresh=0.8):
    idx = (
        prob_df
        .groupby(["sample_id", "processing_status", "cell_id"])["probability"]
        .idxmax()
    )

    dom_df = prob_df.loc[idx].copy()

    dom_df = dom_df.rename(columns={
        "region_group": "dominant_region",
        "probability": "confidence"
    })

    dom_df["is_boundary"] = dom_df["confidence"] < boundary_thresh

    return dom_df.reset_index(drop=True)


# -------------------------
# Helper: normalise image for display
# -------------------------

def normalise_image(img, p_low=1, p_high=99.8):
    img = img.astype(np.float32)

    vmin, vmax = np.percentile(img, [p_low, p_high])

    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-8)

    return img


# -------------------------
# Main overlay function
# -------------------------
import tifffile as tiff

def save_overlay_image_exact(final_rgb, outpath):
    # convert to 8-bit
    img_uint8 = (np.clip(final_rgb, 0, 1) * 255).astype(np.uint8)

    tiff.imwrite(outpath, img_uint8)
    
    
def plot_region_label_overlay_slab(
    image,
    label_img,
    dom_df,
    sample_id,
    processing_status="preprocessed",
    z_min=96,
    z_max=99,
    boundary_thresh=0.8,
    overlay_alpha=0.65,
    outpath=None
):
    """
    image:
        3D microscopy image, shape (z, y, x)

    label_img:
        3D segmentation label image, shape (z, y, x)
        Each cell has a unique integer label matching cell_id.

    dom_df:
        one-row-per-cell dataframe with:
        sample_id, processing_status, cell_id, dominant_region, confidence

    """

    # -------------------------
    # Select z slab
    # -------------------------

    image_slab = image[z_min:z_max + 1]
    label_slab = label_img[z_min:z_max + 1]

    # max projection for display
    image_proj = image_slab.max(axis=0)
    label_proj = label_slab.max(axis=0)

    image_norm = normalise_image(image_proj)

    # grayscale base as RGB
    base_rgb = np.dstack([image_norm, image_norm, image_norm])

    # -------------------------
    # Filter cell metadata
    # -------------------------

    df_plot = dom_df[
        (dom_df["sample_id"].astype(str) == str(sample_id)) &
        (dom_df["processing_status"] == processing_status)
    ].copy()

    if df_plot.empty:
        raise ValueError(
            f"No cells found for sample {sample_id}, {processing_status}"
        )

    # -------------------------
    # Assign plot category
    # -------------------------

    df_plot["plot_region"] = "Other"

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

    df_plot.loc[~confident, "plot_region"] = "Uncertain"

    # -------------------------
    # Build RGB overlay from labels
    # -------------------------

    overlay_rgb = np.zeros_like(base_rgb)
    overlay_mask = np.zeros(label_proj.shape, dtype=bool)

    cell_to_region = dict(zip(df_plot["cell_id"], df_plot["plot_region"]))

    for cell_id, region in cell_to_region.items():
        cell_id = int(cell_id)

        mask = label_proj == cell_id

        if not np.any(mask):
            continue

        overlay_rgb[mask] = to_rgb(COLOURS[region])
        overlay_mask[mask] = True

    # alpha blend only where labels exist
    final_rgb = base_rgb.copy()
    final_rgb[overlay_mask] = (
        (1 - overlay_alpha) * base_rgb[overlay_mask]
        + overlay_alpha * overlay_rgb[overlay_mask]
    )
    # clean (for paper figure panel)
    save_overlay_image_exact(final_rgb, os.path.join(OUTDIR,"overlay_clean.tif"))
    # -------------------------
    # Plot
    # -------------------------

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.imshow(final_rgb)
    ax.set_title(
        f"Region-coloured cell overlay — Sample {sample_id} ({processing_status})\n"
        f"z = {z_min}–{z_max} max projection",
        fontsize=14,
        pad=12
    )

    ax.axis("off")

    handles = [
        Patch(
            facecolor=COLOURS[region],
            edgecolor="black",
            label=region
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
        loc="lower right"
    )

    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=600, bbox_inches="tight")
        plt.savefig(outpath.replace(".png", ".svg"), bbox_inches="tight")
        #save_overlay_image_exact(final_rgb, outpath.replace(".png", ".tif"))
    plt.show()
    


# Load probability df
prob_df = pd.read_csv(P(r"<IMAGING>\Ishrat\JoM\final_count_results\cell_region_probability_outputs\ALL_cell_region_grouped_probabilities_long.csv"))

dom_df = get_dominant_region_df(
    prob_df,
    boundary_thresh=0.8
)

# Load images
image = tiff.imread(P(r"<IMAGING>\Ishrat\JoM\results_preprocessed\extracted_regions_preprocessed_no_padding_union\162_AF_level1_preprocessed_freeform_bending_0.9_grid_-30_smoothing_-1_atlas_perens_lsfm_mouse_20um\162_snpc_image_region_scale.tif"))[:, :, :, 0]
label_img = tiff.imread(P(r"<IMAGING>\Ishrat\JoM\results_preprocessed\cellpose_masked_cells\for_cellpose\162_AF_level1_preprocessed_freeform_bending_0.0_grid_-2_smoothing_-1_atlas_perens_lsfm_mouse_20um\results_python\snpc\whole_segmentation_python.tif"))

plot_region_label_overlay_slab(
    image=image,
    label_img=label_img,
    dom_df=dom_df,
    sample_id="162",
    processing_status="preprocessed",
    z_min=96,
    z_max=99,
    boundary_thresh=0.8,
    overlay_alpha=0.65,
    outpath=P(r"<IMAGING>\Ishrat\JoM\final_figures\Figure5 - Consensus Regions\sample_162_preprocessed_overlay_z96_99.png")
)

