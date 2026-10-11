# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:14:44 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff


def extract_single(text, pattern):
    match = re.search(pattern, text)
    return match.group(1) if match else None


def parse_folder_params(folder_name):
    return {
        "Sample ID": folder_name.split("_")[0],
        "Folder Name": folder_name,
        "bending": extract_single(folder_name, r"_bending_([^_]+)"),
        "grid": extract_single(folder_name, r"_grid_([^_]+)"),
        "smoothing": extract_single(folder_name, r"_smoothing_([^_]+)"),
        "atlas": extract_single(folder_name, r"_atlas_([^_]+)")
    }


def load_binary_mask(path):
    arr = tiff.imread(path)
    return arr != 0


def iou_score(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return np.nan

    return intersection / union


def atlas_inside_tissue_fraction(tissue_mask, atlas_mask):
    atlas_voxels = atlas_mask.sum()
    if atlas_voxels == 0:
        return np.nan

    inside = np.logical_and(tissue_mask, atlas_mask).sum()
    return inside / atlas_voxels


def atlas_outside_tissue_fraction(tissue_mask, atlas_mask):
    atlas_voxels = atlas_mask.sum()
    if atlas_voxels == 0:
        return np.nan

    outside = np.logical_and(~tissue_mask, atlas_mask).sum()
    return outside / atlas_voxels


def tissue_inside_atlas_fraction(tissue_mask, atlas_mask):
    tissue_voxels = tissue_mask.sum()
    if tissue_voxels == 0:
        return np.nan

    inside = np.logical_and(tissue_mask, atlas_mask).sum()
    return inside / tissue_voxels


def tissue_outside_atlas_fraction(tissue_mask, atlas_mask):
    tissue_voxels = tissue_mask.sum()
    if tissue_voxels == 0:
        return np.nan

    outside = np.logical_and(tissue_mask, ~atlas_mask).sum()
    return outside / tissue_voxels


def get_registered_atlas_path(folder_path):
    possible_names = [
        "registered_atlas.tif",
        "registered_atlas.tiff"
    ]

    for name in possible_names:
        candidate = folder_path / name
        if candidate.exists():
            return candidate

    return None


def get_tissue_mask_path(sample_id):
    return Path(
        P(rf"<IMAGING>\Ishrat\JoM\raw_tiff_stacks\downsampled\{sample_id}_AF_level1\{sample_id}_AF_level1_mask_coronal_flipped.tif")
    )


def collect_overlap_scores(result_directory, status):
    rows = []
    result_directory = Path(result_directory)

    print("\n" + "=" * 80)
    print(f"STARTING DIRECTORY: {result_directory}")
    print(f"STATUS: {status}")
    print("=" * 80)

    if not result_directory.exists():
        print(f"[ERROR] Directory does not exist: {result_directory}")
        return pd.DataFrame()

    folders = sorted([f for f in result_directory.iterdir() if f.is_dir()])
    print(f"[INFO] Found {len(folders)} folders to process.\n")

    success_count = 0
    skip_count = 0

    for i, folder in enumerate(folders, start=1):
        folder_name = folder.name
        parsed = parse_folder_params(folder_name)
        sample_id = parsed["Sample ID"]

        print("-" * 80)
        print(f"[{i}/{len(folders)}] Processing folder: {folder_name}")
        print(f"    Sample ID: {sample_id}")
        print(f"    bending={parsed['bending']}, grid={parsed['grid']}, smoothing={parsed['smoothing']}, atlas={parsed['atlas']}")

        tissue_mask_path = get_tissue_mask_path(sample_id)
        atlas_path = get_registered_atlas_path(folder)

        print(f"    Tissue mask path: {tissue_mask_path}")
        print(f"    Registered atlas path: {atlas_path if atlas_path is not None else 'NOT FOUND'}")

        if not tissue_mask_path.exists():
            print(f"    [SKIP] Tissue mask not found.")
            skip_count += 1
            continue

        if atlas_path is None:
            print(f"    [SKIP] registered_atlas.tif/.tiff not found.")
            skip_count += 1
            continue

        try:
            print("    [LOAD] Loading tissue mask...")
            tissue_mask = load_binary_mask(tissue_mask_path)
            print(f"    [LOAD] Tissue mask loaded. Shape: {tissue_mask.shape}, Non-zero voxels: {tissue_mask.sum()}")

            print("    [LOAD] Loading atlas annotation mask...")
            atlas_mask = load_binary_mask(atlas_path)
            print(f"    [LOAD] Atlas mask loaded. Shape: {atlas_mask.shape}, Non-zero voxels: {atlas_mask.sum()}")

        except Exception as e:
            print(f"    [SKIP] Error loading files: {e}")
            skip_count += 1
            continue

        if tissue_mask.shape != atlas_mask.shape:
            print(f"    [SKIP] Shape mismatch.")
            print(f"           Tissue mask shape: {tissue_mask.shape}")
            print(f"           Atlas mask shape:  {atlas_mask.shape}")
            skip_count += 1
            continue

        try:
            print("    [CALC] Computing overlap metrics...")
            iou = iou_score(tissue_mask, atlas_mask)

            atlas_inside_frac = atlas_inside_tissue_fraction(tissue_mask, atlas_mask)
            atlas_outside_frac = atlas_outside_tissue_fraction(tissue_mask, atlas_mask)

            tissue_inside_frac = tissue_inside_atlas_fraction(tissue_mask, atlas_mask)
            tissue_outside_frac = tissue_outside_atlas_fraction(tissue_mask, atlas_mask)

            print(f"    [DONE] IoU = {iou:.6f}" if pd.notna(iou) else "    [DONE] IoU = NaN")
            print(f"    [DONE] Atlas inside tissue fraction = {atlas_inside_frac:.6f}" if pd.notna(atlas_inside_frac) else "    [DONE] Atlas inside tissue fraction = NaN")
            print(f"    [DONE] Atlas outside tissue fraction = {atlas_outside_frac:.6f}" if pd.notna(atlas_outside_frac) else "    [DONE] Atlas outside tissue fraction = NaN")
            print(f"    [DONE] Tissue inside atlas fraction = {tissue_inside_frac:.6f}" if pd.notna(tissue_inside_frac) else "    [DONE] Tissue inside atlas fraction = NaN")
            print(f"    [DONE] Tissue outside atlas fraction = {tissue_outside_frac:.6f}" if pd.notna(tissue_outside_frac) else "    [DONE] Tissue outside atlas fraction = NaN")

            parsed[f"{status}_iou"] = iou
            parsed[f"{status}_atlas_inside_tissue"] = atlas_inside_frac
            parsed[f"{status}_atlas_outside_tissue"] = atlas_outside_frac
            parsed[f"{status}_tissue_inside_atlas"] = tissue_inside_frac
            parsed[f"{status}_tissue_outside_atlas"] = tissue_outside_frac

            rows.append(parsed)
            success_count += 1

        except Exception as e:
            print(f"    [SKIP] Error computing metrics: {e}")
            skip_count += 1
            continue

    print("\n" + "=" * 80)
    print(f"FINISHED DIRECTORY: {result_directory}")
    print(f"[SUMMARY] Successful: {success_count}")
    print(f"[SUMMARY] Skipped:    {skip_count}")
    print("=" * 80 + "\n")

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

raw_directory = Path(P(r"<SCRATCH>\jom_v2_raw"))
preprocessed_directory = Path(P(r"<SCRATCH>\jom_v2_preprocessed"))
output_csv = Path(P(r"<IMAGING>\Ishrat\JoM\qc_results\jom_overlap_scores.csv"))

print("\n" + "#" * 80)
print("STARTING OVERLAP ANALYSIS")
print("#" * 80)
print(f"Raw directory:          {raw_directory}")
print(f"Preprocessed directory: {preprocessed_directory}")
print(f"Output CSV:             {output_csv}")
print("#" * 80)

# Collect metrics
raw_df = collect_overlap_scores(raw_directory, status="raw")
pre_df = collect_overlap_scores(preprocessed_directory, status="preprocessed")

print("[INFO] Raw dataframe shape:", raw_df.shape)
print("[INFO] Preprocessed dataframe shape:", pre_df.shape)

merge_cols = ["Sample ID", "Folder Name", "bending", "grid", "smoothing", "atlas"]

print("\n[INFO] Merging raw and preprocessed results...")
final_df = pd.merge(
    raw_df,
    pre_df[
        merge_cols + [
            "preprocessed_iou",
            "preprocessed_atlas_inside_tissue",
            "preprocessed_atlas_outside_tissue",
            "preprocessed_tissue_inside_atlas",
            "preprocessed_tissue_outside_atlas"
        ]
    ],
    on=merge_cols,
    how="outer"
)

print(f"[INFO] Final dataframe shape after merge: {final_df.shape}")

print("[INFO] Sorting final dataframe...")
final_df = final_df.sort_values(
    by=["Sample ID", "bending", "grid", "smoothing"],
    na_position="last"
).reset_index(drop=True)

print("[INFO] Saving CSV...")
final_df.to_csv(output_csv, index=False)

print("\n" + "#" * 80)
print("ANALYSIS COMPLETE")
print("#" * 80)
print(f"[SUCCESS] Saved results to: {output_csv}")
print(f"[SUCCESS] Final number of rows: {len(final_df)}")
print("\nPreview of final dataframe:")
print(final_df.head(10))
print("#" * 80)