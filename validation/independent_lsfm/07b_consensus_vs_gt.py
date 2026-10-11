# -*- coding: utf-8 -*-
"""
07b_consensus_vs_gt.py

Compare every consensus mask against the GT region mask.

Loops sample x input_type x region x subset x threshold, loading
results/region_consensus/<sample>/<input>/<ACR>_both/<subset>/consensus_thresh_<t>.tif
and scoring it against the GT atlas region (gt_utils.gt_region_mask).
0.5 is the headline threshold; all config thresholds are scored.

Output: results/validation_gt/consensus_vs_gt.csv
"""

import argparse

import pandas as pd

from gt_utils import (
    OUT_DIR, SUBSETS, THRESHOLDS,
    region_table, load_label_mapping, load_gt_atlas, gt_region_mask,
    consensus_path, load_mask, n_registrations, seg_metrics, run_groups,
)


def score_group(sample_id, input_type) -> pd.DataFrame:
    _, agg = load_label_mapping()
    gt = load_gt_atlas(sample_id)
    rows = []

    for r in region_table().itertuples():
        gt_mask = gt_region_mask(gt, r.acronym, agg)

        for subset in SUBSETS:
            n_reg = n_registrations(sample_id, input_type, r.acronym, subset)

            for threshold in THRESHOLDS:
                pred = load_mask(
                    consensus_path(sample_id, input_type, r.acronym, subset, threshold),
                    gt.shape,
                )
                rows.append({
                    "sample_id": sample_id,
                    "input_type": input_type,
                    "region": r.acronym,
                    "subset": subset,
                    "threshold": threshold,
                    "n_registrations": n_reg,
                    **seg_metrics(pred, gt_mask),
                })

    print(f"done: {sample_id} / {input_type}", flush=True)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--fresh", action="store_true", help="ignore per-group results cached by an earlier run")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = run_groups(score_group, args.workers, cache_name="consensus_vs_gt", fresh=args.fresh)
    out = OUT_DIR / "consensus_vs_gt.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
