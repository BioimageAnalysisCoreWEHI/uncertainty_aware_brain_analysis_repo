# -*- coding: utf-8 -*-
"""
07d_calibration.py

Calibration of the consensus probability maps against GT: among voxels
where p = k/n of the registrations called "region", what fraction are
actually region in GT? A well-calibrated map has fraction ~= p.

Evaluation domain: a 10-voxel dilation of (GT region | p > 0 in ANY
subset), shared by the three subsets so their scores are comparable, and
so the huge far-away background (p = 0, GT = 0) doesn't swamp the p = 0
bin or the Brier score. p is binned by its natural 1/n steps (n = number of
registrations in the subset).

Outputs (results/validation_gt/):
    calibration_bins.csv     one row per (sample, input, region, subset, k)
    calibration_summary.csv  one row per probability map: Brier, ECE
"""

import argparse

import numpy as np
import pandas as pd
from scipy import ndimage

from gt_utils import (
    OUT_DIR, SUBSETS,
    region_table, load_label_mapping, load_gt_atlas, gt_region_mask,
    region_dir, n_registrations, bbox_from_mask, run_groups, read_tiff,
)

DILATION_VOXELS = 10


def score_group(sample_id, input_type) -> pd.DataFrame:
    _, agg = load_label_mapping()
    gt = load_gt_atlas(sample_id)
    bins, summary = [], []

    for r in region_table().itertuples():
        gt_mask = gt_region_mask(gt, r.acronym, agg)

        probs = {}
        for subset in SUBSETS:
            prob = read_tiff(region_dir(sample_id, input_type, r.acronym, subset) / "probability.tif")
            if prob.shape != gt.shape:
                raise ValueError(f"{sample_id}/{input_type}/{r.acronym}/{subset}: probability shape mismatch")
            probs[subset] = prob

        # One domain shared by all subsets, so Brier/ECE are comparable
        # between them (a per-subset domain would give all_registrations,
        # with its wide p > 0 spread, many extra easy background voxels).
        support = gt_mask.copy()
        for prob in probs.values():
            support |= prob > 0
        bbox = bbox_from_mask(support, padding=DILATION_VOXELS + 1, min_size=1)
        sl = np.s_[bbox["z0"]:bbox["z1"], bbox["y0"]:bbox["y1"], bbox["x0"]:bbox["x1"]]
        domain = ndimage.binary_dilation(support[sl], iterations=DILATION_VOXELS)

        for subset, prob in probs.items():
            n = n_registrations(sample_id, input_type, r.acronym, subset)
            p = prob[sl][domain].astype(np.float64)
            g = gt_mask[sl][domain]
            k = np.rint(p * n).astype(int)

            n_vox = np.bincount(k, minlength=n + 1)
            n_pos = np.bincount(k, weights=g, minlength=n + 1)
            sum_p = np.bincount(k, weights=p, minlength=n + 1)

            key = {"sample_id": sample_id, "input_type": input_type,
                   "region": r.acronym, "subset": subset, "n_registrations": n}
            for kk in np.flatnonzero(n_vox):
                bins.append({**key, "k": int(kk), "p": sum_p[kk] / n_vox[kk],
                             "n_voxels": int(n_vox[kk]), "n_gt_positive": int(n_pos[kk]),
                             "frac_gt_positive": n_pos[kk] / n_vox[kk]})

            nz = n_vox > 0
            ece = np.sum(n_vox[nz] / n_vox.sum() * np.abs(n_pos[nz] / n_vox[nz] - sum_p[nz] / n_vox[nz]))
            summary.append({**key, "n_voxels_domain": int(domain.sum()),
                            "brier": float(np.mean((p - g) ** 2)), "ece": float(ece)})

    print(f"done: {sample_id} / {input_type}", flush=True)
    # Both tables travel back through run_groups as one frame, split in main().
    return pd.concat(
        [pd.DataFrame(bins).assign(_table="bins"), pd.DataFrame(summary).assign(_table="summary")],
        ignore_index=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore per-group results cached by an earlier run")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = run_groups(score_group, args.workers, cache_name="calibration", fresh=args.fresh)

    for table, name in [("bins", "calibration_bins.csv"), ("summary", "calibration_summary.csv")]:
        out = df[df["_table"] == table].drop(columns="_table").dropna(axis=1, how="all")
        out.to_csv(OUT_DIR / name, index=False)
        print(f"Wrote {len(out)} rows -> {OUT_DIR / name}")


if __name__ == "__main__":
    main()
