# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 13:15:41 2026

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
import skimage.measure
import skimage.transform
import dask_image.imread
from tifffile import imwrite
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# Path Configuration
# ----------------------------
# The MASTER axial union mask (defines the crop for ALL 270 jobs)
union_mask_path = P('<IMAGING>/Ishrat/AnneMiller_round3/param_sweep_masks_raw_no_padding_consensus/pp451_snpc_consensus_masks/pp451_consensus_0.01_axial.tif')

# The folder containing the 270 individual iteration masks (Coronal)
iteration_masks_dir = P('<IMAGING>/Ishrat/AnneMiller_round3/param_sweep_masks_raw_no_padding/snpc/pp451_masks')

# Raw Data and Metadata
raw_img_dir = P('<IMAGING>/Ishrat/AnneMiller_round3/raw_tiff_stacks')
dims_csv = P('<IMAGING>/Ishrat/AnneMiller_round3/001_image_dimensions_level1.csv')
results_base_dir = P('<IMAGING>/Ishrat/AnneMiller_round3/extracted_regions_raw_no_padding_union')

sample_id = "pp451"
region_name = "snpc"

def process_iteration(task_id):
    df = pd.read_csv(dims_csv)
    
    # 1. Identify the specific mask for this task
    # We sort to ensure consistency across the array
    all_masks = sorted([f for f in os.listdir(iteration_masks_dir) if region_name in f])
    if task_id >= len(all_masks):
        return f"Task {task_id} out of bounds."
    
    target_mask_name = all_masks[task_id]
    iteration_id = target_mask_name.replace(".tif", "")
    
    # 2. Define the MASTER BBox from the Axial Union Mask
    union_mask = skimage.io.imread(union_mask_path)
    props = skimage.measure.regionprops((union_mask == 1).astype(np.uint8))
    bbox = list(props[0].bbox) # [z_min, y_min, x_min, z_max, y_max, x_max]
    
    # Apply standard padding
    bbox[0] = max(0, bbox[0] - 10)
    bbox[3] = bbox[3] + 10 

    # 3. Calculate Scaling and Extract Raw Crop
    meta = df[df['Folder'].str.contains(sample_id)].iloc[0]
    # Scaling from 20um Atlas to Raw Pixels
    z_fac, y_fac, x_fac = meta['z']/union_mask.shape[0], meta['y']/union_mask.shape[1], meta['x']/union_mask.shape[2]
    
    raw_bbox = [int(bbox[0]*z_fac), int(bbox[1]*y_fac), int(bbox[2]*x_fac),
                int(bbox[3]*z_fac), int(bbox[4]*y_fac), int(bbox[5]*x_fac)]

    # Load Raw TH Channel
    raw_folder = [f for f in os.listdir(raw_img_dir) if sample_id in f and 'TH' in f][0]
    raw_path = os.path.join(raw_img_dir, raw_folder, '*.tif')
    orig_image = dask_image.imread.imread(raw_path)
    
    # Extract the Master Raw Crop
    crop = orig_image[raw_bbox[0]:raw_bbox[3], 
                      raw_bbox[1]:raw_bbox[4], 
                      raw_bbox[2]:raw_bbox[5]].compute()

    # 4. Process the Iteration-Specific Mask
    it_mask_coronal = skimage.io.imread(os.path.join(iteration_masks_dir, target_mask_name))
    
    # Transform Coronal mask to Axial to match Union BBox
    it_mask_axial = np.flip(np.transpose(it_mask_coronal, (1, 0, 2)), axis=2)
    
    # Crop using Union coordinates
    it_cropped = it_mask_axial[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
    
    # Resize to match the high-res raw crop
    it_resized = skimage.transform.resize(it_cropped, crop.shape, order=0, preserve_range=True).astype(np.uint8)

    # 5. Save Outputs
    out_dir = os.path.join(results_base_dir, iteration_id)
    os.makedirs(out_dir, exist_ok=True)
    
    # Primary Outputs
    skimage.io.imsave(os.path.join(out_dir, f'{sample_id}_{region_name}_image_region.tif'), crop)
    skimage.io.imsave(os.path.join(out_dir, f'{sample_id}_{region_name}_mask_region_resized.tif'), it_resized)
    skimage.io.imsave(os.path.join(out_dir, f'{sample_id}_{region_name}_mask_region_downsampled.tif'), it_cropped)

    # MIPs for fast QC
    skimage.io.imsave(os.path.join(out_dir, f'{sample_id}_{region_name}_image_region_mip.tif'), np.max(crop, axis=0))
    skimage.io.imsave(os.path.join(out_dir, f'{sample_id}_{region_name}_mask_region_resized_mip.tif'), np.max(it_resized, axis=0))
    
    return f"Success: {iteration_id}"

if __name__ == "__main__":
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    print(process_iteration(task_id))