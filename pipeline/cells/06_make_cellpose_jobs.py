# -*- coding: utf-8 -*-
"""
Created on Tue May 26 12:39:43 2026

@author: zaman.i
"""

from pathlib import Path
import argparse
import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--region-assets", default="results/region_assets")
    parser.add_argument("--out", default="results/job_tables/cellpose_jobs.csv")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    region_root = Path(args.region_assets)
    rows = []

    for th_crop in region_root.glob("*/*/fullres_crops/TH_crop_fullres_axial.tif"):
        sample_id = th_crop.parents[2].name
        input_type = th_crop.parents[1].name

        out_dir = th_crop.parents[1] / "cellpose_outputs"

        rows.append({
            "sample_id": sample_id,
            "input_type": input_type,
            "th_crop": th_crop.as_posix(),
            "out_dir": out_dir.as_posix(),
        })

    jobs = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(out_path, index=False)

    print(f"Saved {len(jobs)} Cellpose jobs to {out_path}")


if __name__ == "__main__":
    main()