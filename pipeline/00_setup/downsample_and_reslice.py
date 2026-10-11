# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:22:04 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)
import os
import sys
import pandas as pd
import SimpleITK as sitk


# ============================================================
# CONFIG
# ============================================================

manifest_csv = (
    P("<IMAGING>/Ishrat/JoM/post_review_analysis/config/"
    "samples_amap_validation_raw.csv")
)

output_base = (
    P("<IMAGING>/Ishrat/JoM/post_review_analysis/results/"
    "downsampled_25um_images")
)

input_spacing = (12.5, 12.5, 12.5)  # X, Y, Z in µm
target_spacing = 25.0                # isotropic output spacing in µm

os.makedirs(output_base, exist_ok=True)


# ============================================================
# FUNCTIONS
# ============================================================

def resample_to_isotropic(img, target_spacing, is_mask=False):
    """
    Resample a 3D SimpleITK image to isotropic voxel spacing.

    AF image:
        Linear interpolation.

    Tissue mask:
        Nearest-neighbour interpolation to preserve discrete labels.
    """

    orig_spacing = img.GetSpacing()  # X, Y, Z
    orig_size = img.GetSize()        # X, Y, Z

    new_size = [
        int(round(size * spacing / target_spacing))
        for size, spacing in zip(orig_size, orig_spacing)
    ]

    resampler = sitk.ResampleImageFilter()

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    resampler.SetOutputSpacing([target_spacing] * 3)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())

    return resampler.Execute(img)


def shorten_sample_name(sample_id):
    """
    Keep only the first two underscore-separated elements.

    Example:
        MV_Ntsr1_165_whatever
        -> MV_Ntsr1
    """

    parts = sample_id.split("_")

    if len(parts) >= 2:
        return "_".join(parts[:2])

    return sample_id


def reorient_image(img):
    """
    Apply the required reslicing/orientation corrections.

    Operations:
        1. Reslice: X,Y,Z -> X,Z,Y
        2. Flip up/down
        3. Reverse slice order
        4. Flip left/right

    This function is used identically for the AF image and tissue mask
    so that they remain spatially aligned.
    """

    # Reslice
    img = sitk.PermuteAxes(
        img,
        [0, 2, 1]
    )

    # Flip up/down
    img = sitk.Flip(
        img,
        [False, True, False]
    )

    # Reverse slice order
    img = sitk.Flip(
        img,
        [False, False, True]
    )

    # Flip left/right
    img = sitk.Flip(
        img,
        [True, False, False]
    )

    return img


# ============================================================
# SLURM ARRAY HANDLING
# ============================================================

try:
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
except KeyError:
    print("ERROR: SLURM_ARRAY_TASK_ID is not set.")
    print("Run this script using a SLURM array.")
    sys.exit(1)


# ============================================================
# LOAD MANIFEST
# ============================================================

df = pd.read_csv(manifest_csv)
df = df.reset_index(drop=True)

if task_id >= len(df):
    print(
        f"ERROR: Task ID {task_id} exceeds manifest sample count "
        f"({len(df)} samples)."
    )
    sys.exit(1)


row = df.iloc[task_id]

full_sample_id = str(row["sample_id"])
sample_id = shorten_sample_name(full_sample_id)

af_image_path = str(row["af_image"])
tissue_mask_path = str(row["tissue_mask"])


print("=" * 70)
print(f"Processing sample {task_id + 1}/{len(df)}")
print(f"Original sample ID: {full_sample_id}")
print(f"Output sample ID:   {sample_id}")
print()
print(f"AF image:    {af_image_path}")
print(f"Tissue mask: {tissue_mask_path}")
print()
print(f"Input spacing:  {input_spacing} µm")
print(f"Target spacing: {target_spacing} µm isotropic")
print("=" * 70)


# ============================================================
# CHECK INPUTS
# ============================================================

if not os.path.isfile(af_image_path):
    print("ERROR: AF image does not exist:")
    print(af_image_path)
    sys.exit(1)

if not os.path.isfile(tissue_mask_path):
    print("ERROR: tissue mask does not exist:")
    print(tissue_mask_path)
    sys.exit(1)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

sample_output_dir = os.path.join(
    output_base,
    sample_id
)

os.makedirs(sample_output_dir, exist_ok=True)


# ============================================================
# LOAD AF IMAGE
# ============================================================

print("\n--- AF IMAGE ---")
print("Reading AF image...")

af_image = sitk.ReadImage(af_image_path)

print(f"Original size:    {af_image.GetSize()}")
print(f"Original spacing: {af_image.GetSpacing()}")
print(f"Pixel type:       {af_image.GetPixelIDTypeAsString()}")

# Explicitly set known voxel spacing
af_image.SetSpacing(input_spacing)


# ============================================================
# LOAD TISSUE MASK
# ============================================================

print("\n--- TISSUE MASK ---")
print("Reading tissue mask...")

tissue_mask = sitk.ReadImage(tissue_mask_path)

print(f"Original size:    {tissue_mask.GetSize()}")
print(f"Original spacing: {tissue_mask.GetSpacing()}")
print(f"Pixel type:       {tissue_mask.GetPixelIDTypeAsString()}")

# Explicitly set the same known voxel spacing
tissue_mask.SetSpacing(input_spacing)


# ============================================================
# CHECK AF / MASK DIMENSIONS
# ============================================================

if af_image.GetSize() != tissue_mask.GetSize():
    print("\nWARNING:")
    print("AF image and tissue mask do not have the same dimensions!")
    print(f"AF size:   {af_image.GetSize()}")
    print(f"Mask size: {tissue_mask.GetSize()}")
else:
    print("\nAF image and tissue mask dimensions match.")


# ============================================================
# RESAMPLE AF IMAGE
# ============================================================

print("\n--- RESAMPLING AF IMAGE ---")

af_resampled = resample_to_isotropic(
    af_image,
    target_spacing=target_spacing,
    is_mask=False
)

print(f"Resampled AF size:    {af_resampled.GetSize()}")
print(f"Resampled AF spacing: {af_resampled.GetSpacing()}")


# ============================================================
# RESAMPLE TISSUE MASK
# ============================================================

print("\n--- RESAMPLING TISSUE MASK ---")

mask_resampled = resample_to_isotropic(
    tissue_mask,
    target_spacing=target_spacing,
    is_mask=True
)

print(f"Resampled mask size:    {mask_resampled.GetSize()}")
print(f"Resampled mask spacing: {mask_resampled.GetSpacing()}")


# ============================================================
# SAVE DOWNSAMPLED, UN-RESLICED VERSIONS
# ============================================================

af_downsampled_path = os.path.join(
    sample_output_dir,
    f"{sample_id}_downsampled_25um.tif"
)

mask_downsampled_path = os.path.join(
    sample_output_dir,
    f"{sample_id}_tissue_mask_downsampled_25um.tif"
)

sitk.WriteImage(
    af_resampled,
    af_downsampled_path
)

sitk.WriteImage(
    mask_resampled,
    mask_downsampled_path
)


# ============================================================
# RESLICE + ORIENTATION CORRECTIONS
# ============================================================

print("\n--- REORIENTING AF IMAGE ---")

af_final = reorient_image(af_resampled)

print("\n--- REORIENTING TISSUE MASK ---")

mask_final = reorient_image(mask_resampled)


# ============================================================
# CHECK FINAL DIMENSIONS
# ============================================================

print("\nFinal dimensions:")
print(f"AF:   {af_final.GetSize()}")
print(f"Mask: {mask_final.GetSize()}")

if af_final.GetSize() != mask_final.GetSize():
    print("WARNING: final AF and mask dimensions do not match!")
else:
    print("Final AF and mask dimensions match.")


# ============================================================
# SAVE FINAL AF IMAGE
# ============================================================

af_final_path = os.path.join(
    sample_output_dir,
    f"{sample_id}_downsampled_25um_resliced.tif"
)

sitk.WriteImage(
    af_final,
    af_final_path
)


# ============================================================
# SAVE FINAL TISSUE MASK
# ============================================================

mask_final_path = os.path.join(
    sample_output_dir,
    f"{sample_id}_tissue_mask_downsampled_25um_resliced.tif"
)

sitk.WriteImage(
    mask_final,
    mask_final_path
)


# ============================================================
# SAVE NIFTI VERSIONS
# ============================================================

af_nii_path = os.path.join(
    sample_output_dir,
    f"{sample_id}_downsampled_25um_resliced.nii.gz"
)

mask_nii_path = os.path.join(
    sample_output_dir,
    f"{sample_id}_tissue_mask_downsampled_25um_resliced.nii.gz"
)

sitk.WriteImage(
    af_final,
    af_nii_path
)

sitk.WriteImage(
    mask_final,
    mask_nii_path
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 70)
print(f"✅ Done: {sample_id}")
print()
print("Final AF image:")
print(af_final_path)
print()
print("Final tissue mask:")
print(mask_final_path)
print()
print(f"All outputs saved to:")
print(sample_output_dir)
print("=" * 70)