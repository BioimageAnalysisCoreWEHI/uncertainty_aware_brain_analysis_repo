# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 20:01:54 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import os
from pathlib import Path
from collections import Counter
import traceback

import numpy as np
import pandas as pd
from tifffile import imread
from skimage.measure import regionprops

import time

start_time = time.perf_counter()

# ============================================================
# CONFIG
# ============================================================

BASE = Path(P(r"<IMAGING>\Ishrat\JoM"))

INPUT_CSV = BASE / "final_count_results" / "full_final_results_with_ssim.csv"
ONTOLOGY_CSV = BASE / "ARA2_annotation_info_avail_regions.csv"

OUTDIR = BASE / "final_count_results" / "cell_region_probability_outputs_all_runs"
OUTDIR.mkdir(parents=True, exist_ok=True)

PROCESSING_STATUSES = ["raw", "preprocessed"]

REGIONS_OF_INTEREST = ["SNc", "SNr", "VTA", "MRN", "RR", "MB"]

REGION_MASK_BASES = {
    "raw": BASE / "results_raw" / "extracted_regions_raw_no_padding_union",
    "preprocessed": BASE / "results_preprocessed" / "extracted_regions_preprocessed_no_padding_union",
}

CELL_LABEL_BASES = {
    "raw": BASE / "results_raw" / "cellpose_masked_cells" / "for_cellpose",
    "preprocessed": BASE / "results_preprocessed" / "cellpose_masked_cells" / "for_cellpose",
}


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")].copy()

ontology = pd.read_csv(ONTOLOGY_CSV)
id_to_name = dict(zip(ontology["id"].astype(int), ontology["name"]))
id_to_acronym = dict(zip(ontology["id"].astype(int), ontology["acronym"]))


# ============================================================
# PATH HELPERS
# ============================================================

def add_region_label_paths(df):
    df = df.copy()

    for status in PROCESSING_STATUSES:
        base = REGION_MASK_BASES[status]
        col = f"{status}_region_labels_path"

        df[col] = df.apply(
            lambda row: str(
                base
                / str(row["Folder Name"])
                / f"{row['Sample ID']}_snpc_atlas_region_full.tif"
            ),
            axis=1,
        )

    return df


def infer_cell_label_path(sample_id, processing_status, df_sub):
    """
    Prefer an existing *_cell_labels_path column if present.
    Otherwise infer from Folder Name.
    """

    col = f"{processing_status}_cell_labels_path"
    if col in df_sub.columns:
        vals = df_sub[col].dropna().astype(str).unique()
        if len(vals) > 0:
            return vals[0]

    # fallback: infer from first folder name
    folder = str(df_sub.iloc[0]["Folder Name"])

    return str(
        CELL_LABEL_BASES[processing_status]
        / f"{sample_id}_AF_level1_preprocessed_freeform_bending_0.0_grid_-2_smoothing_-1_atlas_perens_lsfm_mouse_20um"
        / "results_python"
        / "snpc"
        / "whole_segmentation_python.tif"
    )


# ============================================================
# CORE FUNCTIONS
# ============================================================

def get_cell_centroids(cell_label_path):
    cell_labels = imread(cell_label_path)
    props = regionprops(cell_labels)

    cell_ids = np.array([p.label for p in props], dtype=int)
    coords = np.round(np.array([p.centroid for p in props])).astype(int)

    return cell_ids, coords


def compute_cell_region_probabilities(
    df,
    sample_id,
    processing_status,
    use_only_passed=True,
):
    atlas_label_col = f"{processing_status}_region_labels_path"
    pass_col = f"{processing_status}_pass_final"

    sub = df[df["Sample ID"].astype(str) == str(sample_id)].copy()

    if use_only_passed and pass_col in sub.columns:
        sub = sub[sub[pass_col] == True]

    sub = sub.dropna(subset=[atlas_label_col]).reset_index(drop=True)

    if sub.empty:
        raise ValueError(f"No selected registrations for {sample_id}, {processing_status}")

    cell_label_path = infer_cell_label_path(sample_id, processing_status, sub)

    if not os.path.exists(cell_label_path):
        raise FileNotFoundError(f"Missing cell label file: {cell_label_path}")

    existing = sub[atlas_label_col].apply(os.path.exists)
    sub = sub[existing].reset_index(drop=True)

    if sub.empty:
        raise FileNotFoundError(f"No atlas region masks exist for {sample_id}, {processing_status}")

    cell_ids, coords = get_cell_centroids(cell_label_path)


    cell_centroids = pd.DataFrame({
        "sample_id": sample_id,
        "processing_status": processing_status,
        "cell_id": cell_ids,
        "z": coords[:, 0],
        "y": coords[:, 1],
        "x": coords[:, 2],
    })
    
    n_cells = len(cell_ids)
    n_regs = len(sub)

    assignments = {int(cid): [] for cid in cell_ids}

    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]

    for _, row in sub.iterrows():
        atlas_path = row[atlas_label_col]
        atlas = imread(atlas_path)

        valid = (
            (z >= 0) & (z < atlas.shape[0]) &
            (y >= 0) & (y < atlas.shape[1]) &
            (x >= 0) & (x < atlas.shape[2])
        )

        region_ids = np.full(n_cells, -1, dtype=int)
        region_ids[valid] = atlas[z[valid], y[valid], x[valid]].astype(int)

        for cid, rid in zip(cell_ids, region_ids):
            rid = int(rid)
            acronym = id_to_acronym.get(rid, f"unknown_{rid}")
            name = id_to_name.get(rid, f"unknown_{rid}")
            assignments[int(cid)].append((rid, acronym, name))

    long_rows = []

    for cid, regs in assignments.items():
        counts = Counter([r[1] for r in regs])

        for acronym, count in counts.items():
            rid, acronym, name = [r for r in regs if r[1] == acronym][0]

            long_rows.append({
                "sample_id": sample_id,
                "processing_status": processing_status,
                "cell_id": cid,
                "region_id": rid,
                "region_acronym": acronym,
                "region_name": name,
                "n_assignments": count,
                "n_registrations": n_regs,
                "probability": count / n_regs,
                "cell_label_path": cell_label_path,
            })

    long = pd.DataFrame(long_rows)

    wide = (
        long
        .pivot_table(
            index=["sample_id", "processing_status", "cell_id"],
            columns="region_acronym",
            values="probability",
            fill_value=0,
        )
        .reset_index()
    )

    dominant = (
        long.sort_values("probability", ascending=False)
        .groupby(["sample_id", "processing_status", "cell_id"])
        .first()
        .reset_index()
    )

    dominant = dominant[[
        "sample_id",
        "processing_status",
        "cell_id",
        "region_acronym",
        "region_name",
        "probability",
    ]].rename(columns={
        "region_acronym": "dominant_region",
        "region_name": "dominant_region_name",
        "probability": "dominant_probability",
    })

    wide = wide.merge(
        dominant,
        on=["sample_id", "processing_status", "cell_id"],
        how="left",
    )

    grouped_long = long.copy()
    grouped_long["region_group"] = np.where(
        grouped_long["region_acronym"].isin(REGIONS_OF_INTEREST),
        grouped_long["region_acronym"],
        "other",
    )

    grouped_long = (
        grouped_long
        .groupby(
            ["sample_id", "processing_status", "cell_id", "region_group"],
            as_index=False,
        )["probability"]
        .sum()
    )

    grouped_wide = (
        grouped_long
        .pivot_table(
            index=["sample_id", "processing_status", "cell_id"],
            columns="region_group",
            values="probability",
            fill_value=0,
        )
        .reset_index()
    )


    long = long.merge(
        cell_centroids,
        on=["sample_id", "processing_status", "cell_id"],
        how="left",
    )
    
    wide = wide.merge(
        cell_centroids,
        on=["sample_id", "processing_status", "cell_id"],
        how="left",
    )
    
    grouped_long = grouped_long.merge(
        cell_centroids,
        on=["sample_id", "processing_status", "cell_id"],
        how="left",
    )
    
    grouped_wide = grouped_wide.merge(
        cell_centroids,
        on=["sample_id", "processing_status", "cell_id"],
        how="left",
    )
    
    
    summary = {
        "sample_id": sample_id,
        "processing_status": processing_status,
        "n_cells": n_cells,
        "n_registrations": n_regs,
        "cell_label_path": cell_label_path,
    }

    return long, wide, grouped_long, grouped_wide, summary


# ============================================================
# BATCH RUNNER
# ============================================================

df = add_region_label_paths(df)

# Save updated master table with paths
df.to_csv(OUTDIR / "master_results_with_region_paths.csv", index=False)

samples = sorted(df["Sample ID"].astype(str).unique())

all_long = []
all_wide = []
all_grouped_long = []
all_grouped_wide = []
manifest = []

for sample_id in samples:
    for status in PROCESSING_STATUSES:
        print(f"\n=== Running {sample_id} | {status} ===")

        try:
            long, wide, grouped_long, grouped_wide, summary = compute_cell_region_probabilities(
                df=df,
                sample_id=sample_id,
                processing_status=status,
                use_only_passed=False,
            )

            sample_outdir = OUTDIR / str(sample_id) / status
            sample_outdir.mkdir(parents=True, exist_ok=True)

            long.to_csv(sample_outdir / "cell_region_probabilities_long.csv", index=False)
            wide.to_csv(sample_outdir / "cell_region_probabilities_wide.csv", index=False)
            grouped_long.to_csv(sample_outdir / "cell_region_grouped_probabilities_long.csv", index=False)
            grouped_wide.to_csv(sample_outdir / "cell_region_grouped_probabilities_wide.csv", index=False)

            all_long.append(long)
            all_wide.append(wide)
            all_grouped_long.append(grouped_long)
            all_grouped_wide.append(grouped_wide)

            summary["status"] = "success"
            summary["error"] = ""
            manifest.append(summary)

        except Exception as e:
            print(f"FAILED: {sample_id} | {status}")
            print(e)

            manifest.append({
                "sample_id": sample_id,
                "processing_status": status,
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })


# ============================================================
# SAVE COMBINED OUTPUTS
# ============================================================

manifest_df = pd.DataFrame(manifest)
manifest_df.to_csv(OUTDIR / "run_manifest.csv", index=False)

if all_long:
    pd.concat(all_long, ignore_index=True).to_csv(
        OUTDIR / "ALL_cell_region_probabilities_long.csv",
        index=False,
    )

if all_wide:
    pd.concat(all_wide, ignore_index=True).to_csv(
        OUTDIR / "ALL_cell_region_probabilities_wide.csv",
        index=False,
    )

if all_grouped_long:
    pd.concat(all_grouped_long, ignore_index=True).to_csv(
        OUTDIR / "ALL_cell_region_grouped_probabilities_long.csv",
        index=False,
    )

if all_grouped_wide:
    pd.concat(all_grouped_wide, ignore_index=True).to_csv(
        OUTDIR / "ALL_cell_region_grouped_probabilities_wide.csv",
        index=False,
    )

print("\nDone!")
print(f"Results saved to: {OUTDIR}")

end_time = time.perf_counter()
elapsed = end_time - start_time

print(f"\nTotal runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")




# ===========================================================
# VISUALSING IN NAPARI 
# ===========================================================

import napari
import pandas as pd
from tifffile import imread

sample_id = "pp451"
status = "preprocessed"

img = imread(P(r"<IMAGING>\Ishrat\JoM\results_preprocessed\param_sweep_masks_preprocessed_no_padding\snpc\162_masks\162_AF_level1_preprocessed_freeform_bending_0.0_grid_-10_smoothing_-1_atlas_perens_lsfm_mouse_20um_snpc_coronal_mask_brainreg.tif"))
all_wide_df = pd.concat(all_wide, ignore_index=True)
df_vis = all_wide_df[
    (all_wide_df["sample_id"].astype(str) == sample_id) &
    (all_wide_df["processing_status"] == status)
].copy()

points = df_vis[["z", "y", "x"]].values

viewer = napari.Viewer()
viewer.add_image(img, name="background", contrast_limits=[0, img.max()])

viewer.add_points(
    points,
    name="cells_coloured_by_region",
    properties={
        "cell_id": df_vis["cell_id"].astype(str).values,
        "dominant_region": df_vis["dominant_region"].astype(str).values,
        "dominant_probability": df_vis["dominant_probability"].values,
    },
    face_color="dominant_region",
    size=5,
)

df_vis["uncertainty"] = 1 - df_vis["dominant_probability"]

viewer.add_points(
    points,
    name="cells_coloured_by_uncertainty",
    properties={
        "uncertainty": df_vis["uncertainty"].values,
        "dominant_region": df_vis["dominant_region"].astype(str).values,
    },
    face_color="uncertainty",
    face_colormap="viridis",
    size=5,
)

uncertain = df_vis[
    (df_vis["SNc"] > 0) &
    (df_vis["SNc"] < 0.8)
]

boundary = df_vis[
    (df_vis["SNc"] > 0) &
    (df_vis["VTA"] > 0)
]

viewer.add_points(
    boundary[["z", "y", "x"]].values,
    name="SNc_VTA_boundary_uncertain_cells",
    properties={
        "cell_id": boundary["cell_id"].astype(str).values,
        "SNc": boundary["SNc"].values,
        "VTA": boundary["VTA"].values,
        "dominant_region": boundary["dominant_region"].astype(str).values,
    },
    face_color="yellow",
    size=8,
)