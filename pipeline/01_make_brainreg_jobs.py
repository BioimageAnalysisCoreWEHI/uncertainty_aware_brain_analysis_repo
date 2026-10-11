# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:40:13 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

from pathlib import Path
import argparse
import itertools
import json
import os
import pandas as pd
import yaml


def safe_value(value):
    """Make parameter values safe for folder names."""
    return str(value).replace("-", "m").replace(".", "p")


def make_save_dir(sample_id, output_group, bending, grid, smoothing):

    folder = (
        f"bend_{safe_value(bending)}"
        f"__grid_{safe_value(grid)}"
        f"__smooth_{safe_value(smoothing)}"
    )

    save_dir = (
        Path(P("<SCRATCH>/jom_review_sweep"))
        / str(sample_id)
        / str(output_group)
        / folder
    )

    return save_dir.as_posix()


def main():
    parser = argparse.ArgumentParser(
        description="Generate brainreg SLURM job table from samples.csv and config.yaml"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--samples", default="config/samples.csv")
    parser.add_argument("--out", default="results/job_tables/brainreg_jobs.csv")
    args = parser.parse_args()

    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    
    samples = pd.read_csv(args.samples)

    output_root = cfg["brainreg"]["output_root"]
    atlas = cfg["atlas"]["name"]

    voxel_size = cfg["brainreg"].get("voxel_size", [20, 20, 20])
    orientation = cfg["brainreg"].get("orientation", "sal")

    bending_values = cfg["brainreg"]["bending_energy_weight"]
    grid_values = cfg["brainreg"]["grid_spacing"]
    smoothing_values = cfg["brainreg"].get("smoothing_sigma_floating", [-1.0])

    affine_n_steps = cfg["brainreg"].get("affine_n_steps", 3)
    affine_use_n_steps = cfg["brainreg"].get("affine_use_n_steps", 2)
    freeform_n_steps = cfg["brainreg"].get("freeform_n_steps", 4)
    freeform_use_n_steps = cfg["brainreg"].get("freeform_use_n_steps", 4)

    rows = []

    for _, sample in samples.iterrows():
        for bending, grid, smoothing in itertools.product(
            bending_values, grid_values, smoothing_values
        ):
            save_dir = make_save_dir(
                sample_id=sample["sample_id"],
                output_group=sample["output_group"],
                bending=bending,
                grid=grid,
                smoothing=smoothing,
            )
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "input_type": sample["input_type"],
                    "output_group": sample["output_group"],
                    "af_image": sample["af_image"],
                    #"th_image": sample["th_image"],
                    "unprocessed_image": sample["unprocessed_image"],
                    "save_dir": str(save_dir),
                    "atlas": atlas,
                    "voxel_size_x": voxel_size[0],
                    "voxel_size_y": voxel_size[1],
                    "voxel_size_z": voxel_size[2],
                    "orientation": orientation,
                    "bending_energy_weight": bending,
                    "grid_spacing": grid,
                    "smoothing_sigma_floating": smoothing,
                    "affine_n_steps": affine_n_steps,
                    "affine_use_n_steps": affine_use_n_steps,
                    "freeform_n_steps": freeform_n_steps,
                    "freeform_use_n_steps": freeform_use_n_steps,
                }
            )

    jobs = pd.DataFrame(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(out_path, index=False)

    # Also save a quick metadata JSON
    meta_path = out_path.with_suffix(".metadata.json")
    with open(meta_path, "w") as f:
        json.dump(
            {
                "n_samples_rows": len(samples),
                "n_jobs": len(jobs),
                "config": args.config,
                "samples": args.samples,
            },
            f,
            indent=4,
        )

    print(f"Saved {len(jobs)} jobs to: {out_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()