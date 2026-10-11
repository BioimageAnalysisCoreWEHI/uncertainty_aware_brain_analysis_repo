# -*- coding: utf-8 -*-
"""
Created on Tue May 26 13:43:18 2026

@author: zaman.i
"""

from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P  # noqa: E402
import argparse
import json

import numpy as np
import pandas as pd
import tifffile as tiff
import yaml
from cellpose import models
import logging
from cellpose import io

def main():
    parser = argparse.ArgumentParser(
        description="Run Cellpose locally on all TH crops in region_assets."
    )

    parser.add_argument(
        "--config",
        default="config/config.yaml",
    )

    parser.add_argument(
        "--region-assets",
        default="results/region_assets",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Cellpose outputs.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cp_cfg = cfg["cellpose"]


    # ------------------------------------------------------------
    # Enable Cellpose logging
    # ------------------------------------------------------------
    io.logger_setup()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    
    print("Cellpose logging enabled", flush=True)
        # ------------------------------------------------------------
    # Initialise Cellpose model ONCE
    # ------------------------------------------------------------
    custom_weights = cp_cfg.get("custom_weights", None)
    if custom_weights not in [None, "null", "None", ""]:
        custom_weights = P(custom_weights)  # e.g. <DATA>/cellpose_model/...

    if custom_weights in [None, "null", "None", ""]:
        print("Using built-in Cellpose model")

        model = models.CellposeModel(
            gpu=bool(cp_cfg.get("use_gpu", False)),
            model_type=cp_cfg.get("model_type", "cyto2"),
        )

    else:
        print(f"Using custom Cellpose weights: {custom_weights}")

        model = models.CellposeModel(
            gpu=bool(cp_cfg.get("use_gpu", False)),
            pretrained_model=custom_weights,
        )

    # ------------------------------------------------------------
    # Find all TH crops
    # ------------------------------------------------------------
    region_root = Path(args.region_assets)

    th_crops = sorted(
        region_root.glob("*/*/fullres_crops/TH_crop_fullres_axial.tif")
    )

    print(f"Found {len(th_crops)} TH crops")

    # ------------------------------------------------------------
    # Process each sample
    # ------------------------------------------------------------
    summary_rows = []

    for i, th_crop_path in enumerate(th_crops, start=1):

        sample_id = th_crop_path.parents[2].name
        input_type = th_crop_path.parents[1].name

        out_dir = th_crop_path.parents[1] / "cellpose_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_mask = out_dir / "cellpose_labels.tif"

        print("=" * 80)
        print(f"[{i}/{len(th_crops)}]")
        print(f"Sample: {sample_id}")
        print(f"Input type: {input_type}")
        print(f"Image: {th_crop_path}")
        print("=" * 80)

        if out_mask.exists() and not args.overwrite:
            print("Output already exists, skipping.")

            summary_rows.append({
                "sample_id": sample_id,
                "input_type": input_type,
                "image": str(th_crop_path),
                "output": str(out_mask),
                "status": "skipped_existing",
            })

            continue

        # ------------------------------------------------------------
        # Load image
        # ------------------------------------------------------------
        img = tiff.imread(th_crop_path)

        print(f"Image shape: {img.shape}")

        # ------------------------------------------------------------
        # Run Cellpose
        # ------------------------------------------------------------
        print("Starting Cellpose segmentation...", flush=True)
        
        masks, flows, styles = model.eval(
            img,
            diameter=cp_cfg.get("diameter", None),
            channels=cp_cfg.get("channels", [0, 0]),
            flow_threshold=cp_cfg.get("flow_threshold", 0.4),
            cellprob_threshold=cp_cfg.get("cellprob_threshold", 0.0),
            stitch_threshold=cp_cfg.get("stitch_threshold", 0.0),
            do_3D=bool(cp_cfg.get("do_3D", False)),
        )
        
        print("Cellpose finished successfully", flush=True)

        # ------------------------------------------------------------
        # Save outputs
        # ------------------------------------------------------------
        tiff.imwrite(
            out_mask,
            masks.astype(np.uint16),
        )

        metadata = {
            "sample_id": sample_id,
            "input_type": input_type,
            "image": str(th_crop_path),
            "output": str(out_mask),
            "cellpose_config": cp_cfg,
            "n_labels": int(masks.max()),
            "image_shape": list(img.shape),
        }

        with open(out_dir / "cellpose_run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Saved labels to: {out_mask}")
        print(f"Detected labels: {int(masks.max())}")

        summary_rows.append({
            "sample_id": sample_id,
            "input_type": input_type,
            "image": str(th_crop_path),
            "output": str(out_mask),
            "n_labels": int(masks.max()),
            "status": "success",
        })

    # ------------------------------------------------------------
    # Save summary table
    # ------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)

    summary_path = region_root / "cellpose_run_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("=" * 80)
    print("DONE")
    print(f"Saved summary to: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()