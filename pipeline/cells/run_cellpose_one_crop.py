# -*- coding: utf-8 -*-
"""
Created on Tue May 26 12:58:09 2026

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
import tifffile as tiff
import yaml
from cellpose import models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    cp_cfg = cfg["cellpose"]

    image_path = Path(args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = tiff.imread(image_path)

    custom_weights = cp_cfg.get("custom_weights", None)
    if custom_weights not in [None, "null", "None", ""]:
        custom_weights = P(custom_weights)  # e.g. <DATA>/cellpose_model/...

    if custom_weights in [None, "null", "None", ""]:
        model = models.CellposeModel(
            gpu=bool(cp_cfg.get("use_gpu", False)),
            model_type=cp_cfg.get("model_type", "cyto2"),
        )
    else:
        model = models.CellposeModel(
            gpu=bool(cp_cfg.get("use_gpu", False)),
            pretrained_model=custom_weights,
        )

    masks, flows, styles = model.eval(
        img,
        diameter=cp_cfg.get("diameter", None),
        channels=cp_cfg.get("channels", [0, 0]),
        flow_threshold=cp_cfg.get("flow_threshold", 0.8),
        cellprob_threshold=cp_cfg.get("cellprob_threshold", 0.0),
        stitch_threshold=cp_cfg.get("stitch_threshold", 0.3),
        do_3D=False,
    )

    tiff.imwrite(out_dir / "cellpose_labels.tif", masks.astype(np.uint16))

    metadata = {
        "image": str(image_path),
        "output": str(out_dir / "cellpose_labels.tif"),
        "cellpose_config": cp_cfg,
        "n_labels": int(masks.max()),
        "image_shape": list(img.shape),
    }

    with open(out_dir / "cellpose_run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved labels to: {out_dir / 'cellpose_labels.tif'}")
    print(f"Detected labels: {int(masks.max())}")


if __name__ == "__main__":
    main()