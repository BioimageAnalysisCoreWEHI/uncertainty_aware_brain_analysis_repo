# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:40:47 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)
import os
import pandas as pd

# Windows directories
image_dir = P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_datasets\amap\amap_validation_brains\downscaledBrains_tiff")
mask_dir = P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\results\tissue_masks_coronal")

# Corresponding Linux/HPC directories
image_dir_linux = P("<IMAGING>/Ishrat/JoM/post_review_analysis/validation_datasets/amap/amap_validation_brains/downscaledBrains_tiff")
mask_dir_linux = P("<IMAGING>/Ishrat/JoM/post_review_analysis/results/tissue_masks_coronal")

rows = []

# Find all NIfTI brains
image_files = sorted(
    f for f in os.listdir(image_dir)
    if f.lower().endswith((".tif", ".tiff"))
)

for filename in image_files:

    # Use filename without .nii/.nii.gz as sample ID
    if filename.lower().endswith(".tif"):
        sample_id = filename[:-4]
    else:
        sample_id = os.path.splitext(filename)[0]

    # Find the corresponding tissue mask
    mask_files = [
        f for f in os.listdir(mask_dir)
        if "_".join(sample_id.split("_")[:2]) in f
    ]

    if len(mask_files) == 0:
        print(f"WARNING: No tissue mask found for {sample_id}")
        continue

    if len(mask_files) > 1:
        print(f"WARNING: Multiple tissue masks found for {sample_id}: {mask_files}")
        continue

    mask_filename = mask_files[0]

    rows.append({
        "sample_id": sample_id,
        "input_type": "raw",

        "af_image":
            f"{image_dir_linux}/{filename}",

        "unprocessed_image":
            f"{image_dir_linux}/{filename}",

        "tissue_mask":
            f"{mask_dir_linux}/{mask_filename}",

        "output_group": "raw"
    })


config = pd.DataFrame(rows)

output_path = (
    P(r"<IMAGING>\Ishrat\JoM\post_review_analysis"
    r"\config\samples_amap_validation_raw.csv")
)

config.to_csv(output_path, index=False)

print(config)
print(f"\nGenerated config for {len(config)} brains.")