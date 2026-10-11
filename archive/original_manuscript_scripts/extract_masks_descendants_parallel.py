# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 12:31:54 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import os
import sys
import numpy as np
import pandas as pd
import skimage.io
import skimage.transform
import dask_image.imread
import warnings

warnings.filterwarnings("ignore")

# --- Paths (Updated for Linux/HPC style) ---
parent_dir = P('<SCRATCH>/jom_v2_preprocessed')
dims_csv_path = P("<IMAGING>/Ishrat/JoM/raw_tiff_stacks/001_image_dimensions_level1.csv")
ontology_path = P("<IMAGING>/Ishrat/brainreg/padded/final_automated_am/ARA2_annotation_info_avail_regions.csv")
results_base_dir = P("<IMAGING>/Ishrat/JoM/param_sweep_masks_preprocessed_no_padding")
os.makedirs(results_base_dir, exist_ok=True)
# Regions to extract
MASK_REGIONS = [374, 749]
MASK_NAMES = {374: "snpc", 749: "vta"}

def get_all_descendants(region_df, parent_id):
    descendants = set()
    stack = [parent_id]
    while stack:
        current = stack.pop()
        children = region_df[region_df["parent_id"] == current]["id"].tolist()
        for child in children:
            if child not in descendants:
                descendants.add(child)
                stack.append(child)
    return list(descendants)

def process_single_folder(dataset_name, region_df, df_dims):
    data_id_prefix = dataset_name.split("_")[0]
    
    # Load atlas
    image_path = os.path.join(parent_dir, dataset_name, 'registered_atlas.tiff')
    if not os.path.exists(image_path):
        return f"Skip: {dataset_name} (no atlas)"

    image = skimage.io.imread(image_path)
    
    # Load raw image for shape reference
    raw_img_path = os.path.join(parent_dir, dataset_name, f"downsampled_{data_id_prefix}_TH_level1_preprocessed.tiff")
    if not os.path.exists(raw_img_path):
        return f"Skip: {dataset_name} (no raw tiff)"
    
    orig_shape = dask_image.imread.imread(raw_img_path).shape

    for mask_id in MASK_REGIONS:
        mask_name = MASK_NAMES[mask_id]
        
        # Setup results directory: .../snpc/136_masks/
        specific_results_dir = os.path.join(results_base_dir, mask_name, f"{data_id_prefix}_masks")
        os.makedirs(specific_results_dir, exist_ok=True)
        
        output_path = os.path.join(specific_results_dir, f"{dataset_name}_{mask_name}_coronal_mask_brainreg.tif")
        
        if os.path.exists(output_path):
            continue

        # Get mask IDs
        if mask_id in [374, 749]:
            ids_to_extract = [mask_id]
        else:
            ids_to_extract = get_all_descendants(region_df, mask_id)

        # Extract and resize
        mask_region = np.isin(image, ids_to_extract)
        mask_resized = skimage.transform.resize(mask_region, orig_shape, order=0, preserve_range=True).astype(np.uint8)
        
        skimage.io.imsave(output_path, mask_resized, check_contrast=False)
        
    
    return f"Success: {dataset_name}"

if __name__ == "__main__":
    # Get all subfolders in parent_dir
    all_folders = sorted([f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))])
    
    # Get index from SLURM Array
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    
    if task_id < len(all_folders):
        folder_to_process = all_folders[task_id]
        
        # Load metadata once per task
        region_df = pd.read_csv(ontology_path)
        df_dims = pd.read_csv(dims_csv_path)
        
        print(process_single_folder(folder_to_process, region_df, df_dims))