# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 10:11:04 2025

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import os
import sys
import csv
from imaris_ims_file_reader.ims import ims

# === CONFIG ===
parent_path = P("<IMAGING>/Ishrat/JoM/post_review_analysis/validation_datasets/lightsheet_data/results/raw_ims_data")
output_base = P("<IMAGING>/Ishrat/JoM/post_review_analysis/validation_datasets/lightsheet_data/results/raw_tiff_stacks")
os.makedirs(output_base, exist_ok=True)
metadata_csv = os.path.join(output_base, "ims_metadata_summary.csv")
resolution_level = 1

# === Get all .ims files ===
ims_files = sorted([f for f in os.listdir(parent_path) if f.endswith(".ims")])
if not ims_files:
    print("No .ims files found in directory:", parent_path)
    sys.exit(1)

# === Get SLURM task ID ===
try:
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
except KeyError:
    print("Error: SLURM_ARRAY_TASK_ID not set. Run this within a SLURM array job.")
    sys.exit(1)

# === Get the .ims file for this job ===
if task_id < 0 or task_id >= len(ims_files):
    print(f"Invalid task ID {task_id}, only {len(ims_files)} files available.")
    sys.exit(1)

img_name = ims_files[task_id]
img_path = os.path.join(parent_path, img_name)
print(f"Processing file {task_id + 1}/{len(ims_files)}: {img_name}")

# === Extract prefix (before first underscore) ===
fname_prefix = img_name.split("_")[0]

# === Read image ===
imaris_img = ims(img_path)
meta = imaris_img.metaData

# === Extract pixel size + shape info ===
res = meta[(resolution_level, 0, 0, "resolution")]  # x,y,z voxel size (µm)
shape = imaris_img.shape  # typically (t, c, z, y, x)

pixel_sizes = [round(r, 4) for r in res]
num_pixels = shape[-3:]  # z, y, x
print(f"Pixel sizes (µm): {pixel_sizes}, Num pixels (z,y,x): {num_pixels}")

# === Save metadata ===
header = ["filename", "pixel_size_z", "pixel_size_y", "pixel_size_x", "z", "y", "x"]
metadata_row = [img_name, *pixel_sizes, num_pixels[-3], num_pixels[-2], num_pixels[-1]]

# Save a temporary per-job CSV
metadata_job_csv = os.path.join(output_base, f"{fname_prefix}_metadata.csv")
with open(metadata_job_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerow(metadata_row)
    
# === Save TIFF stacks for each channel ===


# Channel 0 -> the registration channel: autofluorescence for the original LSFM brains; for the single-colour
# independent LSFM (ScienceDB) brains it is the labelled marker channel. Output folders keep the historical
# '_AF_' name, which downstream scripts expect.
af_path = os.path.join(output_base, f"{fname_prefix}_AF_level{resolution_level}")
imaris_img.save_Tiff_Series(
    af_path, overwrite=True, resolutionLevel=resolution_level, channels=(0, 0)
)

# Channel 1 → TH
# th_path = os.path.join(output_base, f"{fname_prefix}_atlas_level{resolution_level}")
# imaris_img.save_Tiff_Series(
#     th_path, overwrite=True, resolutionLevel=resolution_level, channels=(1, 1)
# )

print(f"Finished {img_name}")
