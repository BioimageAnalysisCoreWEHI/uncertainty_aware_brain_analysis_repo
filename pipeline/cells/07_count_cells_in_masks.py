# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:37:37 2026

@author: zaman.i
"""

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.morphology import remove_small_objects


def extract_param_from_name(name, key):
    match = re.search(rf"{key}_([^_\.]+)", name)
    return match.group(1) if match else None


def parse_threshold_from_name(name):
    match = re.search(r"thresh_([^_]+)_fullres", name)
    if match:
        return match.group(1).replace("p", ".").replace("m", "-")
    return None


def count_cells_by_masked_labels(mask_path, segmentation_path, min_size=1):
    region_mask = tiff.imread(mask_path) > 0
    labels = tiff.imread(segmentation_path)

    if region_mask.shape != labels.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"mask: {region_mask.shape}\n"
            f"labels: {labels.shape}"
        )

    # Old logic: clip Cellpose labels by region mask first
    masked_cells = labels * region_mask.astype(labels.dtype)

    if min_size is not None and min_size > 1:
        masked_cells = remove_small_objects(masked_cells, min_size=min_size)

    cell_ids = np.unique(masked_cells)
    cell_ids = cell_ids[cell_ids != 0]

    return int(len(cell_ids)), cell_ids.astype(int), masked_cells


def count_parameter_masks(region_root, out_csv, min_size=1, save_masked=True):
    rows = []
    region_root = Path(region_root)

    for sample_dir in sorted(region_root.iterdir()):
        if not sample_dir.is_dir():
            continue

        sample_id = sample_dir.name

        for input_dir in sorted(sample_dir.iterdir()):
            if not input_dir.is_dir():
                continue

            input_type = input_dir.name

            segmentation_path = input_dir / "cellpose_outputs" / "cellpose_labels.tif"
            mask_dir = input_dir / "parameter_masks_fullres_crop"
            masked_dir = input_dir / "parameter_masked_cell_labels"

            if not segmentation_path.exists() or not mask_dir.exists():
                continue

            if save_masked:
                masked_dir.mkdir(parents=True, exist_ok=True)

            for mask_path in sorted(mask_dir.glob("*.tif")):
                try:
                    count, cell_ids, masked_cells = count_cells_by_masked_labels(
                        mask_path=mask_path,
                        segmentation_path=segmentation_path,
                        min_size=min_size,
                    )

                    masked_path = ""
                    if save_masked:
                        masked_path = masked_dir / f"{mask_path.stem}__masked_cells.tif"
                        tiff.imwrite(masked_path, masked_cells.astype(np.uint16))

                    rows.append({
                        "sample_id": sample_id,
                        "input_type": input_type,
                        "mask_type": "parameter",
                        "mask_path": str(mask_path),
                        "segmentation_path": str(segmentation_path),
                        "masked_cell_labels_path": str(masked_path),
                        "bending_energy_weight": extract_param_from_name(mask_path.stem, "bend"),
                        "grid_spacing": extract_param_from_name(mask_path.stem, "grid"),
                        "smoothing_sigma_floating": extract_param_from_name(mask_path.stem, "smooth"),
                        "cell_count": count,
                        #"cell_ids": ";".join(map(str, cell_ids)),
                        "error": "",
                    })

                except Exception as e:
                    rows.append({
                        "sample_id": sample_id,
                        "input_type": input_type,
                        "mask_type": "parameter",
                        "mask_path": str(mask_path),
                        "segmentation_path": str(segmentation_path),
                        "masked_cell_labels_path": "",
                        "bending_energy_weight": extract_param_from_name(mask_path.stem, "bend"),
                        "grid_spacing": extract_param_from_name(mask_path.stem, "grid"),
                        "smoothing_sigma_floating": extract_param_from_name(mask_path.stem, "smooth"),
                        "cell_count": np.nan,
                        #"cell_ids": "",
                        "error": str(e),
                    })

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved parameter-mask counts to: {out_csv}")


def count_consensus_masks(region_root, out_csv, min_size=1, save_masked=True):
    rows = []
    region_root = Path(region_root)

    for sample_dir in sorted(region_root.iterdir()):
        if not sample_dir.is_dir():
            continue

        sample_id = sample_dir.name

        for input_dir in sorted(sample_dir.iterdir()):
            if not input_dir.is_dir():
                continue

            input_type = input_dir.name

            segmentation_path = input_dir / "cellpose_outputs" / "cellpose_labels.tif"
            mask_dir = input_dir / "consensus_masks_fullres_crop"
            masked_dir = input_dir / "consensus_masked_cell_labels"

            if not segmentation_path.exists() or not mask_dir.exists():
                continue

            if save_masked:
                masked_dir.mkdir(parents=True, exist_ok=True)

            for mask_path in sorted(mask_dir.glob("*consensus_thresh*.tif")):
                threshold = parse_threshold_from_name(mask_path.stem)

                try:
                    count, cell_ids, masked_cells = count_cells_by_masked_labels(
                        mask_path=mask_path,
                        segmentation_path=segmentation_path,
                        min_size=min_size,
                    )

                    masked_path = ""
                    if save_masked:
                        masked_path = masked_dir / f"{mask_path.stem}__masked_cells.tif"
                        tiff.imwrite(masked_path, masked_cells.astype(np.uint16))

                    rows.append({
                        "sample_id": sample_id,
                        "input_type": input_type,
                        "mask_type": "consensus",
                        "threshold": threshold,
                        "mask_path": str(mask_path),
                        "segmentation_path": str(segmentation_path),
                        "masked_cell_labels_path": str(masked_path),
                        "cell_count": count,
                        #"cell_ids": ";".join(map(str, cell_ids)),
                        "error": "",
                    })

                except Exception as e:
                    rows.append({
                        "sample_id": sample_id,
                        "input_type": input_type,
                        "mask_type": "consensus",
                        "threshold": threshold,
                        "mask_path": str(mask_path),
                        "segmentation_path": str(segmentation_path),
                        "masked_cell_labels_path": "",
                        "cell_count": np.nan,
                        #"cell_ids": "",
                        "error": str(e),
                    })

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved consensus-mask counts to: {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Count Cellpose-detected cells within parameter and consensus SNpc masks."
    )
    parser.add_argument("--region-root", default="results/region_assets")
    parser.add_argument("--out-dir", default="results/cell_counts")
    parser.add_argument("--min-size", type=int, default=1)
    parser.add_argument(
        "--no-save-masked",
        action="store_true",
        help="Do not save masked cell-label images.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_masked = not args.no_save_masked

    count_parameter_masks(
        region_root=args.region_root,
        out_csv=out_dir / "cell_counts_by_parameter_mask.csv",
        min_size=args.min_size,
        save_masked=save_masked,
    )

    count_consensus_masks(
        region_root=args.region_root,
        out_csv=out_dir / "cell_counts_by_consensus_mask.csv",
        min_size=args.min_size,
        save_masked=save_masked,
    )


if __name__ == "__main__":
    main()