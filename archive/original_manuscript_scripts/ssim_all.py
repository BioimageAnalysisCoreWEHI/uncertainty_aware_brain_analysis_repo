# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 10:36:58 2025

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import os
import json
import numpy as np
import pandas as pd
import skimage.io
from skimage.metrics import structural_similarity as ssim

# ----------------------------
# Parameters
# ----------------------------
# Use the directory where your brainreg outputs (downsampled_standard) live
parent_dir = P(r"<SCRATCH>\am_round3_jom_raw")
atlas_file = P(r"<IMAGING>\Ishrat\AnneMiller_round3\perens_lsfm_mouse_20um_coronal.tif")
bbox_file  = P(r"<IMAGING>\Ishrat\AnneMiller_round3\midbrain_qc_bbox.json")

# ----------------------------
# Load Reference Data
# ----------------------------
print("Loading reference atlas...")
atlas_af = skimage.io.imread(atlas_file).astype(np.float32)

with open(bbox_file) as f:
    b = json.load(f)

# Crop Atlas to the Midbrain region defined by your bounding box
atlas_crop = atlas_af[b["z0"]:b["z1"], b["y0"]:b["y1"], b["x0"]:b["x1"]]

# Standardize Atlas range to [0, 1] for SSIM consistency
atlas_min, atlas_max = atlas_crop.min(), atlas_crop.max()
atlas_prep = (atlas_crop - atlas_min) / (atlas_max - atlas_min + 1e-6)

# ----------------------------
# Analysis Loop
# ----------------------------
all_results = []
# Get all folders that start with your sample IDs (pp451, etc)
sample_folders = sorted([f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))])

for folder in sample_folders:
    path = os.path.join(parent_dir, folder)
    sample_id = folder.split("_")[0]
    
    # Locate the "standard space" image produced by brainreg
    # It usually contains "downsampled_standard" in the filename
    target_files = [f for f in os.listdir(path) if "downsampled_standard" in f]
    
    if not target_files:
        continue
        
    print(f"Calculating SSIM for: {folder}")
    
    # Load and crop the sample image
    sample_img = skimage.io.imread(os.path.join(path, target_files[0])).astype(np.float32)
    sample_crop = sample_img[b["z0"]:b["z1"], b["y0"]:b["y1"], b["x0"]:b["x1"]]
    
    # Standardize sample range to [0, 1]
    # Since you've already histogram-matched, this ensures the math is identical to the atlas
    s_min, s_max = sample_crop.min(), sample_crop.max()
    sample_prep = (sample_crop - s_min) / (s_max - s_min + 1e-6)

    # Calculate SSIM (Global Structural Similarity)
    # This evaluates luminance, contrast, and structure (texture/edges)
    # win_size=7 is standard for 3D volumes at this resolution
    score_ssim = ssim(atlas_prep, sample_prep, data_range=1.0)

    all_results.append({
        "sample_id": sample_id,
        "registration_folder": folder,
        "raw_ssim": score_ssim
    })
    print(f"{folder} RAW range: min={sample_crop.min():.5f}, max={sample_crop.max():.5f}")
    print(f"{folder} NORMALISED range: min={sample_prep.min():.5f}, max={sample_prep.max():.5f}")

# ----------------------------
# Post-Processing & Export
# ----------------------------
df = pd.DataFrame(all_results)

# Create the Relative Score per sample for your dot plot
# This tells you which registration was "best" for that specific brain
df['relative_qc_score'] = df.groupby('sample_id')['raw_ssim'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
)

output_path = P(r"<IMAGING>\Ishrat\AnneMiller_round3\JoM\ssim_midbrain_qc_results_raw.csv")
df.to_csv(output_path, index=False)
print(f"Success! Results saved to: {output_path}")

