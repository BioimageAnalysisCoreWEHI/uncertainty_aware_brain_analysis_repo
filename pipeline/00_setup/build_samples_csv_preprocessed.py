# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 09:28:03 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import os
import pandas as pd

# Windows directories
image_dir = P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\results\images_preprocessed")
unprocessed_dir = P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_datasets\amap\amap_validation_brains\downscaledBrains_tiff")
mask_dir = P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\results\tissue_masks_coronal")

# Corresponding Linux/HPC directories
image_dir_linux = P("<IMAGING>/Ishrat/JoM/post_review_analysis/results/images_preprocessed")
unprocessed_dir_linux = P("<IMAGING>/Ishrat/JoM/post_review_analysis/validation_datasets/amap/amap_validation_brains/downscaledBrains_tiff")
mask_dir_linux =  P("<IMAGING>/Ishrat/JoM/post_review_analysis/results/tissue_masks_coronal")

rows = []


def files_for(sample_id, folder, key=None):
    """Files in `folder` belonging to exactly this sample: the name must equal the sample id or start with
    '<sample id>_' / '<sample id>.'. A plain substring test is not enough, because one sample id can be contained
    in another (e.g. 'TH' in 'Rep2-TH'), which previously paired brains with the wrong unprocessed image."""
    key = key or sample_id
    return sorted(f for f in os.listdir(folder)
                  if os.path.splitext(f)[0] == key or f.startswith(key + "_") or f.startswith(key + "."))


# Find all NIfTI brains
image_files = sorted(
    f for f in os.listdir(image_dir)
    if f.lower().endswith((".tif", ".tiff"))
)

for filename in image_files:

    # Use filename without .nii/.nii.gz as sample ID
    if filename.lower().endswith(".tif"):
        sample_id = filename[:-17]
    else:
        sample_id = os.path.splitext(filename)[0]

    # Find the corresponding tissue mask
    mask_files = files_for(sample_id, mask_dir, key="_".join(sample_id.split("_")[:2]))
    unprocessed_files = files_for(sample_id, unprocessed_dir)

    if len(mask_files) != 1 or len(unprocessed_files) != 1:
        print(f"WARNING: expected exactly one tissue mask and one unprocessed image for {sample_id}, found "
              f"{mask_files} and {unprocessed_files}; sample skipped")
        continue

    mask_filename = mask_files[0]
    unprocessed_filename = unprocessed_files[0]

    rows.append({
        "sample_id": sample_id,
        "input_type": "preprocessed",

        "af_image":
            f"{image_dir_linux}/{filename}",

        "unprocessed_image":
            f"{unprocessed_dir_linux}/{unprocessed_filename}",

        "tissue_mask":
            f"{mask_dir_linux}/{mask_filename}",

        "output_group": "preprocessed"
    })


config = pd.DataFrame(rows)

output_path = (
    P(r"<IMAGING>\Ishrat\JoM\post_review_analysis"
    r"\config\samples_amap_validation_preprocessed_proper.csv")
)

config.to_csv(output_path, index=False)

print(config)
print(f"\nGenerated config for {len(config)} brains.")