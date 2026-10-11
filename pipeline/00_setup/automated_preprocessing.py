# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 14:04:26 2025

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

# %% ================== 🧠 Automated Preprocessing Pipeline ==================
"""
This script:
1. Clips intensities (1–99%)
2️. Matches histogram to a reference atlas
3️. Masks background using a binary mask
4. Saves both NIfTI (.nii.gz) and TIFF (.tif) outputs

"""

# %% 🔧 Imports and Setup
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import traceback
import tifffile

# %% 📦 Helper Functions
def read_image(path, dtype=sitk.sitkFloat32):
    """Read an image with SimpleITK."""
    print(f"📥 Reading: {os.path.basename(path)}")
    return sitk.ReadImage(path, dtype)

def clip_percentile(image, lower=1, upper=99):
    """Clip image intensities to given percentiles."""
    arr = sitk.GetArrayFromImage(image)
    low, high = np.percentile(arr, [lower, upper])
    arr = np.clip(arr, low, high)
    clipped = sitk.GetImageFromArray(arr)
    clipped.CopyInformation(image)
    return clipped

def match_histogram(input_img, reference_img, num_levels=256, num_match_points=64):
    """Match histogram of input image to reference."""
    
    # Ensure both images are the same pixel type
    input_img = sitk.Cast(input_img, sitk.sitkFloat32)
    reference_img = sitk.Cast(reference_img, sitk.sitkFloat32)
    
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(num_levels)
    matcher.SetNumberOfMatchPoints(num_match_points)
    matcher.ThresholdAtMeanIntensityOn()
    
    return matcher.Execute(input_img, reference_img)

def mask_background(image, mask):
    """Apply binary mask to remove background."""
    img_arr = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(mask)
    masked = sitk.GetImageFromArray(img_arr * mask_arr)
    masked.CopyInformation(image)
    return masked

def save_image(image, path, is_tiff=False):
    """Save image as NIfTI or normalized TIFF."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if is_tiff:
        arr = sitk.GetArrayFromImage(image)
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr_uint8 = (arr_norm * 255).astype(np.uint8)
        img_tif = sitk.GetImageFromArray(arr_uint8)
        img_tif.CopyInformation(image)
        sitk.WriteImage(img_tif, path)
    else:
        sitk.WriteImage(image, path)
    print(f"💾 Saved: {os.path.basename(path)}")

def plot_histogram(image, title):
    """Plot histogram of image intensities."""
    arr = sitk.GetArrayFromImage(image).flatten()
    plt.figure(figsize=(6, 4))
    plt.hist(arr, bins=100, alpha=0.75, color='steelblue')
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def save_tiff_sequence(image, out_dir):
    """Save a 3D image as a TIFF sequence (one file per slice)."""
    os.makedirs(out_dir, exist_ok=True)
    arr = sitk.GetArrayFromImage(image)  # [z, y, x]
    for i, slice_ in enumerate(arr):
        out_path = os.path.join(out_dir, f"slice_{i:04d}.tif")
        tifffile.imwrite(out_path, slice_.astype(np.float32))
    print(f"📸 Saved {len(arr)} TIFF slices to {out_dir}")
    
# %% 🚀 Main Processing Function
import glob
def process_image(parent_dir, folder, atlas_path, visualize=False):
    """Process one image: clip → match → mask → save."""
    try:
        sample_id = "_".join(folder.split("_")[:2])
        print(sample_id)
        input_img_paths = glob.glob(
            os.path.join(parent_dir, f"{sample_id}_*.tif")
        )
        if not input_img_paths:
            raise FileNotFoundError(
                f"No TIFF starting with '{sample_id}_' found in {parent_dir}"
            )

        input_img_path = input_img_paths[0]

        mask_path_dir = P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\results\tissue_masks")
        mask_paths = glob.glob(
            os.path.join(mask_path_dir, f"MASK_{sample_id}_*.tif")
        )

        if not mask_paths:
            raise FileNotFoundError(
                f"No mask starting with 'MASK_{sample_id}_' found"
            )

        mask_path = mask_paths[0]

        out_base = os.path.join(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\results\images_preprocessed"), f"{sample_id}_preprocessed")

        print(f"\n🧩 Processing: {folder}")
        input_img = read_image(input_img_path)
        mask_img = read_image(mask_path, dtype=sitk.sitkUInt8)
        atlas_img = read_image(atlas_path)

        # --- Step 1: Clip intensities ---
        #clipped = clip_percentile(input_img, 1, 99)
        #if visualize:
        #    plot_histogram(clipped, f"{folder}: After clipping")

        # --- Step 2: Histogram match ---
        matched = match_histogram(input_img, atlas_img)
        if visualize:
            plot_histogram(matched, f"{folder}: After histogram matching")

        # --- Step 3: Apply mask ---
        masked = mask_background(matched, mask_img)

        # --- Step 4: Save ---
        save_image(masked, f"{out_base}.nii.gz")
        save_image(masked, f"{out_base}.tif", is_tiff=True)
        
        sequence_path = P(rf"<IMAGING>\Ishrat\JoM\post_review_analysis\results\preprocessed_stacks\{folder}_preprocessed")
        save_tiff_sequence(masked, sequence_path)

        print(f"✅ Done: {folder}\n")

    except Exception as e:
        print(f"❌ Error processing {folder}: {e}")
        traceback.print_exc()

# %% 📁 Paths and Folder Selection
parent_dir = P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_datasets\amap\amap_validation_brains\downscaledBrains_tiff")
atlas_path = P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\config\osten_mouse_25um_reference.tif")

# Automatically find matching folders
folders = [
    f for f in os.listdir(parent_dir)
    if "_" in f # adjust filters if needed
]
print(f"🧠 Found {len(folders)} folders to process: {folders}\n")

# %% ▶️ Run preprocessing
visualize = False  # Set to True to see intensity histograms

parent_dir = P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_datasets\amap\amap_validation_brains\downscaledBrains_tiff")
atlas_path = P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\config\osten_mouse_25um_reference.tif")

for folder in folders:
    process_image(parent_dir, folder, atlas_path, visualize=visualize)
    

print("🎉 All preprocessing complete!")
