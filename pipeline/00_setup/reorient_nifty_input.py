# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:41:30 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import os
import numpy as np
import nibabel as nib
import tifffile


# ============================================================
# PATHS
# ============================================================

input_dir = P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_datasets\amap\amap_validation_brains\downscaledBrains")

output_dir = P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_datasets\amap\amap_validation_brains\downscaledBrains_tiff")

suffix = "_reoriented"


# ============================================================
# SETUP
# ============================================================

os.makedirs(output_dir, exist_ok=True)

nii_files = sorted(
    f for f in os.listdir(input_dir)
    if f.lower().endswith((".nii", ".nii.gz"))
)

print(f"Found {len(nii_files)} NIfTI files.\n")


# ============================================================
# CONVERT
# ============================================================

for filename in nii_files:

    input_path = os.path.join(input_dir, filename)

    # Remove .nii or .nii.gz cleanly
    if filename.lower().endswith(".nii.gz"):
        base_name = filename[:-7]
    else:
        base_name = filename[:-4]

    output_filename = f"{base_name}{suffix}.tif"
    output_path = os.path.join(output_dir, output_filename)

    print(f"Processing: {filename}")

    # --------------------------------------------------------
    # Load NIfTI without converting unnecessarily to float64
    # --------------------------------------------------------

    img = nib.load(input_path)
    data = np.asanyarray(img.dataobj)

    print(f"  Original shape: {data.shape}")
    print(f"  Voxel size:     {img.header.get_zooms()[:3]}")
    print(f"  Axis codes:     {nib.aff2axcodes(img.affine)}")

    if data.ndim != 3:
        print(f"  WARNING: Expected 3D image, got {data.ndim}D. Skipping.\n")
        continue

    # --------------------------------------------------------
    # Reorder NIfTI (X, Y, Z) -> TIFF stack (Z, Y, X)
    # --------------------------------------------------------

    data_tiff = np.transpose(data, (2, 1, 0))

    print(f"  TIFF shape:     {data_tiff.shape}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    tifffile.imwrite(
        output_path,
        data_tiff,
        bigtiff=True,
    )

    print(f"  Saved: {output_filename}\n")


print("=" * 60)
print(f"Finished converting {len(nii_files)} files.")
print(f"Output directory:\n{output_dir}")