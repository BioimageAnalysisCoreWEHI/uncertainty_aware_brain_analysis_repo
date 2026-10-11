# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:56:40 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

# -*- coding: utf-8 -*-
"""
Transform already-downsampled masks into the same final coronal
orientation as the light-sheet intensity images.

Transform:
    1. transpose (2, 0, 1)
    2. flip axes (1, 2)

Output:
    sample_id_mask_coronal.tif
"""

import os
import pandas as pd
import numpy as np
import tifffile


# ============================================================
# CONFIG
# ============================================================

manifest_csv = (
    P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_datasets"
    r"\lightsheet_data\metadata\analysis_manifest.csv")
)

base_dir = (
    P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_datasets"
    r"\lightsheet_data\results\downsampled_25um_images")
)

output_dir = (
    P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_datasets"
    r"\lightsheet_data\results\downsampled_25um_masks")
)

os.makedirs(output_dir, exist_ok=True)


# ============================================================
# LOAD MANIFEST
# ============================================================

df = pd.read_csv(manifest_csv)

if "include_in_combined_run" in df.columns:
    df = df[
        df["include_in_combined_run"]
        .astype(str)
        .str.lower()
        .isin(["yes", "true", "1"])
    ]

df = df.reset_index(drop=True)

print(f"Found {len(df)} samples to process.\n")


# ============================================================
# PROCESS EACH SAMPLE
# ============================================================

for _, row in df.iterrows():

    sample_id = str(row["sample_id"])

    # Each sample has its own folder
    sample_dir = os.path.join(
        base_dir,
        sample_id
    )

    print("=" * 70)
    print(f"Processing: {sample_id}")
    print(f"Folder: {sample_dir}")

    if not os.path.isdir(sample_dir):
        print("⚠️ Sample folder not found. Skipping.\n")
        continue


    # ========================================================
    # FIND FILE WHOSE NAME ENDS IN "mask"
    #
    # e.g.
    # something_mask.tif
    # something_mask.tiff
    # ========================================================

    mask_files = sorted([
        f for f in os.listdir(sample_dir)
        if os.path.splitext(f)[0].lower().endswith("mask")
        and f.lower().endswith((".tif", ".tiff"))
    ])

    if not mask_files:
        print("⚠️ No file ending in 'mask' found. Skipping.\n")
        continue

    if len(mask_files) > 1:
        print("⚠️ Multiple masks found:")
        for f in mask_files:
            print(f"   {f}")
        print("Using the first one.")

    mask_filename = mask_files[0]

    mask_path = os.path.join(
        sample_dir,
        mask_filename
    )

    print(f"Found mask: {mask_filename}")


    # ========================================================
    # LOAD MASK
    # ========================================================

    mask = tifffile.imread(mask_path)

    print(f"Original shape: {mask.shape}")
    print(f"Original dtype: {mask.dtype}")


    # ========================================================
    # CORONAL TRANSFORM
    #
    # Same as intensity image:
    #
    # (0, 1, 2) -> (2, 0, 1)
    # ========================================================

    mask_coronal = np.transpose(
        mask,
        (2, 0, 1)
    )


    # ========================================================
    # 180° ROTATION
    #
    # Same as intensity image:
    # ========================================================

    mask_coronal = np.flip(
        mask_coronal,
        axis=(1, 2)
    )

    mask_coronal = np.ascontiguousarray(
        mask_coronal
    )


    # ========================================================
    # SAVE
    # ========================================================

    output_path = os.path.join(
        output_dir,
        f"{sample_id}_mask_coronal.tif"
    )

    tifffile.imwrite(
        output_path,
        mask_coronal
    )

    print(f"Final shape: {mask_coronal.shape}")
    print(f"✅ Saved: {output_path}\n")


print("=" * 70)
print("✅ Finished processing all samples.")