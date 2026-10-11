# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:44:53 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)



import tifffile
import numpy as np
import os


def threshold_probability_map(image_path, threshold=0.5):
    """Threshold a probability map (> threshold), save a binary mask,
    a scaled binary mask, and a coronal-resliced binary mask."""

    print(f"⚙️ Thresholding: {os.path.basename(image_path)} at > {threshold}")

    # Load probability map
    img = tifffile.imread(image_path).astype(np.float32)

    # Threshold
    mask = (img > threshold).astype(np.uint8)

    # Versions to save
    mask_uint8 = mask
    mask_uint8_scaled = mask * 255

    # Output paths
    output_path = image_path.replace(
        "Probabilities.tif", "mask_thresholded.tif"
    )
    output_path_scaled = image_path.replace(
        "Probabilities.tif", "mask_thresholded_scaled.tif"
    )
    output_path_coronal = image_path.replace(
        "Probabilities.tif", "mask_thresholded_scaled_coronal.tif"
    )

    # Save original masks
    tifffile.imwrite(output_path, mask_uint8)
    tifffile.imwrite(output_path_scaled, mask_uint8_scaled)

    # ------------------------------------------------------------------
    # Coronal reslice (Z, Y, X) -> (Y, Z, X)
    # ------------------------------------------------------------------
    mask_coronal = np.transpose(mask_uint8_scaled, (1, 0, 2))
    tifffile.imwrite(output_path_coronal, mask_coronal)
    
    # Left-right flipped coronal
    output_path_coronal_flipped = image_path.replace(
        "Probabilities.tif", "mask_thresholded_scaled_coronal_flipped.tif"
    )
    
    mask_coronal_flipped = np.flip(mask_coronal, axis=2)
    tifffile.imwrite(output_path_coronal_flipped, mask_coronal_flipped)

    print(f"💾 Saved binary mask: {output_path}")
    print(f"💾 Saved scaled mask: {output_path_scaled}")
    print(f"💾 Saved coronal mask: {output_path_coronal}\n")
    print(f"💾 Saved left-right flipped coronal mask: {output_path_coronal_flipped}\n")
    
    return output_path


parent_dir = P(r"<IMAGING>\Ishrat\analysis_of_whole_brain_projects\results\downsampled_20um_images")
folders = [f for f in os.listdir(parent_dir) if "AF" in f]

for folder in folders:
    curr_dir = os.path.join(parent_dir, folder)
    print(f"\n🧩 Thresholding: {curr_dir}")

    prob_brain = [f for f in os.listdir(curr_dir) if "Probabilities" in f]

    if not prob_brain:
        print("⚠️ No probability map found.")
        continue

    prob_brain_path = os.path.join(curr_dir, prob_brain[0])

    threshold_probability_map(prob_brain_path, threshold=0.4)
    
    
    
    
    
### DELETING FILES

import os

# Parent directory containing all sample folders
parent_dir = P(r"<IMAGING>\Ishrat\analysis_of_whole_brain_projects\results\downsampled_20um_images")

deleted = 0

for root, dirs, files in os.walk(parent_dir):
    for file in files:
        if file.endswith("mask_thresholded_coronal.tiff"):
            file_path = os.path.join(root, file)

            try:
                os.remove(file_path)
                print(f"🗑️ Deleted: {file_path}")
                deleted += 1
            except Exception as e:
                print(f"❌ Could not delete {file_path}: {e}")

print(f"\n✅ Finished. Deleted {deleted} file(s).")