# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 14:54:21 2025

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)


import os
import numpy as np
import tifffile
import pandas as pd
from cellpose import models, io
from cellpose.io import imread
from scipy.ndimage import binary_closing, binary_fill_holes, generate_binary_structure
from skimage.measure import label, regionprops_table
from skimage.morphology import remove_small_objects
import argparse

# === CONFIGURATION ===

#parent_dir = "<IMAGING>\Ishrat\brainreg\AnneMiller\region_extract_python_test_v2"
model_path = P("<IMAGING>/Pradeep/Verena/Grant_Dewson/level1_cropped/cellpose/final_model/CP_epochs500_lr0.1_SGD_True_wd_1e-05")
region = "snpc"

parser = argparse.ArgumentParser()
parser.add_argument("--parent_dir", required=True)
parser.add_argument("--folder", required=True)
args = parser.parse_args()

folder_path = os.path.join(args.parent_dir, args.folder)

# === FUNCTIONS ===

def count_cells_by_region_overlap(cell_labels, region_labels, region_ids):
    results = []
    for region_id in region_ids:
        region_mask = region_labels == region_id
        overlapping_labels = np.unique(cell_labels[region_mask])
        overlapping_labels = overlapping_labels[overlapping_labels != 0]  # exclude background
        cell_count = len(overlapping_labels)
        volume = np.sum(region_mask)
        results.append((region_id, cell_count, volume))
    return results

# === INITIALIZE MODEL ONCE ===

io.logger_setup()
model = models.CellposeModel(gpu=True, pretrained_model=model_path, diam_mean=9, nchan=2)

# === PROCESS EACH SUBFOLDER ===

#folder_path = os.path.join(parent_dir, folder_name)


print(f"\nProcessing folder: {folder_path}")
sample = args.folder.split("_")[0]
image_name = f"{sample}_{region}_image_region.tif"
mask_name = f"{sample}_{region}_mask_region_resized.tif"
image_path = os.path.join(folder_path, image_name)
mask_path = os.path.join(folder_path, mask_name)

if not (os.path.exists(image_path) and os.path.exists(mask_path)):
    print(f"Skipping {sample} – missing image or mask")
    

output_dir = os.path.join(folder_path, "results_python", region)
os.makedirs(output_dir, exist_ok=True)

# Load image and mask
image = imread(image_path)
mask = imread(mask_path)

# Run Cellpose segmentation
masks, flows, styles = model.eval(
    image,
    channels=[0, 0],
    diameter=9,
    anisotropy=1.0,
    cellprob_threshold=-3.0,
    flow_threshold=0.8,
    stitch_threshold=0.3,
    min_size=10
)

segmented_path = os.path.join(output_dir, "whole_segmentation_python.tif")
tifffile.imwrite(segmented_path, masks.astype(np.uint16))

#masks = imread(r"<IMAGING>\Ishrat\brainreg\AnneMiller\region_extract_python_test\137_freeform_bending_0.6_grid_-10_smoothing_-1_atlas_perens_lsfm_mouse_20um\results_\137_snpc_whole_segmentation.tif")

# Mask cells using region mask (no hole fill here, can add if needed)
masked_cells = masks * mask
masked_path = os.path.join(output_dir, "masked_cells_python.tif")
tifffile.imwrite(masked_path, masked_cells.astype(np.uint16))

# Label regions and get IDs
region_labels = label(mask)
props = regionprops_table(region_labels, properties=("label", "bbox", "area"))
props_df = pd.DataFrame(props)
region_ids = props_df["label"].tolist()

# Remove small objects in cell labels (cells smaller than 1 px)
labeled_cells = remove_small_objects(masked_cells, min_size=1)

# Count cells in each region
counts = count_cells_by_region_overlap(labeled_cells, region_labels, region_ids)

# Save results CSV
df = pd.DataFrame(counts, columns=["RegionID", "CellCount", "RegionVolume_px3"])
# df["Sample"] = folder_name
# csv_path = os.path.join(output_dir, "cell_counts_by_region_python.csv")
# df.to_csv(csv_path, index=False)

# Build a dictionary with desired column structure
row = {"Sample": sample}

# Total cells across all regions
cellpose_labels = np.unique(masks)
row["TotalCellCount"] = len(cellpose_labels)

# Add region volumes and cell counts as RegionVolume_1, CellCount_1, etc.
for i, (region_id, cell_count, volume) in enumerate(counts, start=1):
    row[f"RegionVolume_{i}"] = volume
    row[f"CellCount_{i}"] = cell_count

# Convert to one-row DataFrame
summary_df = pd.DataFrame([row])

# Save
csv_path = os.path.join(output_dir, "cell_counts_by_region_python.csv")
summary_df.to_csv(csv_path, index=False)

print(f"✅ Saved cell counts for {sample} to {csv_path}")



# import os

# # === CONFIGURATION ===
# parent_dir = r"<IMAGING>\Ishrat\brainreg\AnneMiller\region_extraction"
# file_of_interest = "cell_counts_by_region_python.csv"  # or "masked_cells_python.tif" if that's what you're using

# # === COUNTING ===
# count = 0
# matching_paths = []

# for root, dirs, files in os.walk(parent_dir):
#     if file_of_interest in files:
#         count += 1
#         matching_paths.append(os.path.join(root, file_of_interest))

# # === OUTPUT ===
# print(f"Found {count} file(s) named '{file_of_interest}'")
# # Optional: print out where they were found
# for path in matching_paths:
#     print(" -", path)
    
    
# import os

# # === CONFIGURATION ===
# parent_dir = r"<IMAGING>\Ishrat\brainreg\AnneMiller\region_extraction"
# file_of_interest = "cell_counts_by_region_python.csv"

# # === GET FIRST-LEVEL FOLDERS ===
# top_level_folders = [
#     f for f in os.listdir(parent_dir)
#     if os.path.isdir(os.path.join(parent_dir, f))
# ]

# # === CHECK EACH FOLDER ===
# folders_with_file = set()

# for folder in top_level_folders:
#     folder_path = os.path.join(parent_dir, folder)
#     for root, dirs, files in os.walk(folder_path):
#         if file_of_interest in files:
#             folders_with_file.add(folder)
#             break  # No need to search deeper in this folder

# # === FOLDERS MISSING THE FILE ===
# missing_folders = [f for f in top_level_folders if f not in folders_with_file]

# # === OUTPUT ===
# print(f"\nFolders missing '{file_of_interest}':")
# for folder in missing_folders:
#     print(" -", folder)

# print(f"\n✅ {len(missing_folders)} folder(s) missing the file.")

# import os
# import shutil

# # === CONFIGURATION ===
# parent_dir = r"<IMAGING>\Ishrat\brainreg\AnneMiller\region_extraction"
# file_of_interest = "cell_counts_by_region_python.csv"
# output_dir = r"<IMAGING>\Ishrat\brainreg\AnneMiller\missing_folders3"

# # Create output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# # Get all top-level folders
# top_level_folders = [
#     f for f in os.listdir(parent_dir)
#     if os.path.isdir(os.path.join(parent_dir, f))
# ]

# # Identify folders missing the file
# missing_folders = []

# for folder in top_level_folders:
#     folder_path = os.path.join(parent_dir, folder)
#     found = False
#     for root, dirs, files in os.walk(folder_path):
#         if file_of_interest in files:
#             found = True
#             break
#     if not found:
#         missing_folders.append(folder)

# # Copy missing folders to the output directory
# for folder in missing_folders:
#     src = os.path.join(parent_dir, folder)
#     dst = os.path.join(output_dir, folder)
#     print(f"📁 Copying {folder}...")
#     shutil.copytree(src, dst)

# print(f"\n✅ Done! {len(missing_folders)} folder(s) copied to:\n{output_dir}")


# import os
# import shutil

# # === CONFIGURATION ===
# parent_dir = r"<IMAGING>\Ishrat\brainreg\AnneMiller\region_extraction"
# file_of_interest = "cell_counts_by_region_python.csv"
# destination_dir = r"<IMAGING>\Ishrat\brainreg\AnneMiller\collected_cell_counts"

# # Create destination folder if it doesn't exist
# os.makedirs(destination_dir, exist_ok=True)

# # Loop through all subdirectories in parent_dir
# for folder_name in os.listdir(parent_dir):
#     folder_path = os.path.join(parent_dir, folder_name)
#     if not os.path.isdir(folder_path):
#         continue

#     # Walk through each subdirectory inside the folder
#     for root, dirs, files in os.walk(folder_path):
#         if file_of_interest in files:
#             source_path = os.path.join(root, file_of_interest)
#             new_filename = f"{folder_name}_cell_counts_python.csv"
#             destination_path = os.path.join(destination_dir, new_filename)
#             shutil.copy2(source_path, destination_path)
#             print(f"✅ Copied: {source_path} → {destination_path}")
#             break  # Stop after finding the first matching file

# print("\n📦 All files collected and renamed.")

# import os
# import shutil

# # === CONFIGURATION ===
# source_dir = r"<SCRATCH>\AnneMiller_testing"
# dest_dir = r"<IMAGING>\Ishrat\brainreg\AnneMiller_testing"

# # Make sure destination directory exists
# os.makedirs(dest_dir, exist_ok=True)

# # Loop through folders in source
# for folder_name in os.listdir(source_dir):
#     folder_name = folder_name.strip()  # Removes \n, \r, spaces

#     src_path = os.path.join(source_dir, folder_name)
#     dst_path = os.path.join(dest_dir, folder_name)

#     # Check if it's a folder and if it doesn't already exist in dest
#     if os.path.isdir(src_path) and not os.path.exists(dst_path):
#         try:
#             shutil.copytree(src_path, dst_path)
#             print(f"✅ Copied: {folder_name}")
#         except Exception as e:
#             print(f"❌ Failed to copy {folder_name}: {e}")
#     else:
#         print(f"⏩ Skipped: {folder_name} (already exists or not a folder)")


