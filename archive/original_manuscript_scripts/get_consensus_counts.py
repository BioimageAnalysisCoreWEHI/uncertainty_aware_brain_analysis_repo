# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:14:15 2026

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
from tifffile import imread, imwrite
from skimage.measure import regionprops

import time

start_time = time.perf_counter()
# ============================================================
# CONFIG
# ============================================================

BASE = Path(P(r"<IMAGING>\Ishrat\JoM"))

INPUT_CSV = BASE / "final_count_results" / "full_final_results_with_ssim.csv"

OUTDIR = BASE / "final_count_results" / "consensus_threshold_cell_counts_overlap_logic"
OUTDIR.mkdir(parents=True, exist_ok=True)

PROCESSING_STATUSES = ["raw", "preprocessed"]

THRESHOLDS = np.round(np.arange(0.05, 1.01, 0.05), 2)

CONSENSUS_COMBINATIONS = [
    {"qc_filter": "passed_qc", "ssim_filter": "top_25pct", "top_fraction": 0.25},
    {"qc_filter": "passed_qc", "ssim_filter": "top_50pct", "top_fraction": 0.50},
    {"qc_filter": "passed_qc", "ssim_filter": "all", "top_fraction": None},

    {"qc_filter": "all_runs", "ssim_filter": "top_25pct", "top_fraction": 0.25},
    {"qc_filter": "all_runs", "ssim_filter": "top_50pct", "top_fraction": 0.50},
    {"qc_filter": "all_runs", "ssim_filter": "all", "top_fraction": None},
]

CELL_LABEL_BASES = {
    "raw": BASE / "results_raw" / "cellpose_masked_cells" / "for_cellpose",
    "preprocessed": BASE / "results_preprocessed" / "cellpose_masked_cells" / "for_cellpose",
}


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")].copy()
df["Sample ID"] = df["Sample ID"].astype(str)


# ============================================================
# HELPERS
# ============================================================

def get_subset(df, sample_id, processing_status, qc_filter, ssim_filter, top_fraction=None):
    pass_col = f"{processing_status}_pass_final"
    ssim_col = f"{processing_status}_ssim_value"

    sub = df[df["Sample ID"] == str(sample_id)].copy()

    # Step 1: QC filter
    if qc_filter == "passed_qc":
        sub = sub[sub[pass_col] == True]

    elif qc_filter == "all_runs":
        pass

    else:
        raise ValueError(f"Unknown qc_filter: {qc_filter}")

    # Step 2: SSIM filter
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

def infer_cell_label_path(sample_id, processing_status, sub):
    col = f"{processing_status}_cell_labels_path"

    if col in sub.columns:
        vals = sub[col].dropna().astype(str).unique()
        if len(vals) > 0:
            return Path(vals[0])

    folder = str(sub.iloc[0]["Folder Name"])

    return (
        CELL_LABEL_BASES[processing_status]
        / f"{sample_id}_AF_level1_preprocessed_freeform_bending_0.0_grid_-2_smoothing_-1_atlas_perens_lsfm_mouse_20um"
        / "results_python"
        / "snpc"
        / "whole_segmentation_python.tif"
    )


def get_cell_centroids(cell_label_path):
    labels = imread(cell_label_path)
    props = regionprops(labels)

    cell_ids = np.array([p.label for p in props], dtype=int)
    coords = np.round(np.array([p.centroid for p in props])).astype(int)

    return labels, cell_ids, coords


def build_consensus_probability(sub, processing_status):
    mask_col = f"{processing_status}_mask_axial_path"

    masks = []

    for _, row in sub.iterrows():
        mask_path = row[mask_col]

        if not isinstance(mask_path, str) or not os.path.exists(mask_path):
            print(f"Missing mask: {mask_path}")
            continue

        mask = imread(mask_path)
        masks.append((mask > 0).astype(np.uint8))

    if len(masks) == 0:
        raise ValueError("No masks found.")

    shapes = {m.shape for m in masks}
    if len(shapes) > 1:
        raise ValueError(f"Masks have different shapes: {shapes}")

    stack = np.stack(masks, axis=0)
    consensus_prob = stack.mean(axis=0).astype(np.float32)

    return consensus_prob, len(masks)


def count_cells_in_mask(cell_labels, consensus_mask):
    """
    Count cells if ANY voxel of the labelled cell overlaps the consensus mask.
    Background label 0 is excluded.
    """

    if cell_labels.shape != consensus_mask.shape:
        raise ValueError(
            f"Shape mismatch: cell_labels {cell_labels.shape} vs "
            f"consensus_mask {consensus_mask.shape}"
        )

    mask_bool = consensus_mask > 0

    included_cell_ids = np.unique(cell_labels[mask_bool])
    included_cell_ids = included_cell_ids[included_cell_ids != 0]

    return len(included_cell_ids), included_cell_ids


# ============================================================
# BATCH RUN
# ============================================================

summary_rows = []
cell_rows = []

samples = sorted(df["Sample ID"].unique())

for sample_id in samples:
    for processing_status in PROCESSING_STATUSES:
        for combo in CONSENSUS_COMBINATIONS:

            qc_filter = combo["qc_filter"]
            ssim_filter = combo["ssim_filter"]
            top_fraction = combo["top_fraction"]

            combination_name = f"{qc_filter}_{ssim_filter}"

            print("\n===================================")
            print(f"Sample: {sample_id}")
            print(f"Processing: {processing_status}")
            print(f"Combination: {combination_name}")
            print("===================================")

            try:
                sub = get_subset(
                    df=df,
                    sample_id=sample_id,
                    processing_status=processing_status,
                    qc_filter=qc_filter,
                    ssim_filter=ssim_filter,
                    top_fraction=top_fraction,
                )

                if sub.empty:
                    print("No registrations found.")
                    continue

                consensus_prob, n_masks = build_consensus_probability(
                    sub=sub,
                    processing_status=processing_status,
                )

                cell_label_path = infer_cell_label_path(
                    sample_id=sample_id,
                    processing_status=processing_status,
                    sub=sub,
                )

                if not cell_label_path.exists():
                    raise FileNotFoundError(f"Missing cell label image: {cell_label_path}")

                cell_labels, cell_ids, coords = get_cell_centroids(cell_label_path)

                save_dir = (
                    OUTDIR
                    / str(sample_id)
                    / processing_status
                    / combination_name
                )
                save_dir.mkdir(parents=True, exist_ok=True)

                imwrite(
                    save_dir / f"{sample_id}_{processing_status}_{combination_name}_consensus_probability.tif",
                    consensus_prob,
                    compression="zlib",
                )

                for threshold in THRESHOLDS:
                    consensus_mask = (consensus_prob >= threshold).astype(np.uint8)

                    count, included_cell_ids = count_cells_in_mask(
                        cell_labels=cell_labels,
                        consensus_mask=consensus_mask,
                    )

                    summary_rows.append({
                        "sample_id": sample_id,
                        "processing_status": processing_status,
                        "qc_filter": qc_filter,
                        "ssim_filter": ssim_filter,
                        "top_fraction": top_fraction,
                        "combination_name": combination_name,
                        "threshold": threshold,
                        "n_masks_used": n_masks,
                        "n_total_segmented_cells": len(cell_ids),
                        "consensus_cell_count": count,
                        "consensus_volume_voxels": int(consensus_mask.sum()),
                        "cell_label_path": str(cell_label_path),
                    })

                    for cid in included_cell_ids:
                        cell_rows.append({
                            "sample_id": sample_id,
                            "processing_status": processing_status,
                            "qc_filter": qc_filter,
                            "ssim_filter": ssim_filter,
                            "top_fraction": top_fraction,
                            "combination_name": combination_name,
                            "threshold": threshold,
                            "cell_id": int(cid),
                            "included_in_consensus": True,
                        })

                print(f"Done. N cells total: {len(cell_ids)}")

            except Exception as e:
                print(f"FAILED: {sample_id} | {processing_status} | {combination_name}")
                print(e)

                summary_rows.append({
                    "sample_id": sample_id,
                    "processing_status": processing_status,
                    "qc_filter": qc_filter,
                    "ssim_filter": ssim_filter,
                    "top_fraction": top_fraction,
                    "combination_name": combination_name,
                    "threshold": np.nan,
                    "status": "failed",
                    "error": str(e),
                })

# ============================================================
# SAVE CENTRAL OUTPUTS
# ============================================================

summary_df = pd.DataFrame(summary_rows)
cell_df = pd.DataFrame(cell_rows)

summary_df.to_csv(
    OUTDIR / "ALL_consensus_threshold_cell_counts_summary.csv",
    index=False,
)

cell_df.to_csv(
    OUTDIR / "ALL_consensus_threshold_included_cells_long.csv",
    index=False,
)

print("\nFinished!")
print(f"Saved to: {OUTDIR}")

end_time = time.perf_counter()
elapsed = end_time - start_time

print(f"\nTotal runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")