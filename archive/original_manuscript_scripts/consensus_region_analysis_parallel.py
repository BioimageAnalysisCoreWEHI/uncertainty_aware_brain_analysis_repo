# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 13:01:40 2026

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
from tifffile import imread, imwrite
import warnings

warnings.filterwarnings("ignore")

# --------------------------
# Configuration
# --------------------------
# Path where region folders (snpc, vta) live
parent_dir = P("<IMAGING>/Ishrat/JoM/param_sweep_masks_raw_no_padding")
# Output destination
parent_output_dir = P("<IMAGING>/Ishrat/JoM/param_sweep_masks_raw_no_padding_consensus")

thresholds = [0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mask_regions = [374, 749]
mask_names = {374: "snpc", 749: "vta"}

def save_consensus_masks(prob_map, sample, thresholds, out_dir):
    for t in thresholds:
        consensus_mask = (prob_map >= t).astype(np.uint8)
        # Coronal
        imwrite(os.path.join(out_dir, f"{sample}_consensus_{t:.2f}_coronal.tif"), consensus_mask)
        # Axial
        axial = np.flip(np.transpose(consensus_mask, (1, 0, 2)), axis=2)
        imwrite(os.path.join(out_dir, f"{sample}_consensus_{t:.2f}_axial.tif"), axial)

def save_prob_maps(prob_map, sample, out_dir):
    # Coronal
    imwrite(os.path.join(out_dir, f"{sample}_prob_map_coronal.tif"), prob_map.astype('float32'))
    # Axial
    axial = np.flip(np.transpose(prob_map, (1, 0, 2)), axis=2)
    imwrite(os.path.join(out_dir, f"{sample}_prob_map_axial.tif"), axial.astype('float32'))

def save_union_mask(mask_stack, sample, out_dir):
    union_mask = mask_stack.any(axis=0).astype(np.uint8)
    # Coronal
    imwrite(os.path.join(out_dir, f"{sample}_union_coronal.tif"), union_mask)
    # Axial
    axial = np.flip(np.transpose(union_mask, (1, 0, 2)), axis=2)
    imwrite(os.path.join(out_dir, f"{sample}_union_axial.tif"), axial)

def process_sample(sample_id):
    print(f"\n--- Starting Consensus for Sample: {sample_id} ---")
    
    for region_id in mask_regions:
        region_name = mask_names[region_id]
        
        # Structure: parent_dir/region_name/sample_id_masks
        mask_dir = os.path.join(parent_dir, region_name, f"{sample_id}_masks")
        
        if not os.path.exists(mask_dir):
            print(f"Directory not found: {mask_dir}")
            continue

        # Find all iteration masks (filtering for 'perens' as per your original logic)
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".tif") and "perens" in f]
        
        if not mask_files:
            print(f"No masks found in {mask_dir}")
            continue

        output_dir = os.path.join(parent_output_dir, f"{sample_id}_{region_name}_consensus_masks")
        os.makedirs(output_dir, exist_ok=True)

        print(f"[{region_name}] Loading {len(mask_files)} masks...")
        # Load and stack (Bool to save memory)
        mask_stack = np.stack([imread(os.path.join(mask_dir, f)).astype(bool) for f in mask_files])
        
        print(f"[{region_name}] Computing Probability Map...")
        prob_map = mask_stack.mean(axis=0)

        print(f"[{region_name}] Saving all variations...")
        save_prob_maps(prob_map, sample_id, output_dir)
        save_consensus_masks(prob_map, sample_id, thresholds, output_dir)
        save_union_mask(mask_stack, sample_id, output_dir)
        
    return f"Finished processing Sample {sample_id}"

if __name__ == "__main__":
    # The 3 specific sample IDs you are working with
    unique_samples = ["162", "527", "pp451", "xl353", "xl385"] 
    
    task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    if task_id < len(unique_samples):
        print(process_sample(unique_samples[task_id]))