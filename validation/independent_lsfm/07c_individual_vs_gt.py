# -*- coding: utf-8 -*-
"""
07c_individual_vs_gt.py

Score every single brainreg registration's region mask against GT, and
attach that registration's QC metrics, so we can ask:
  - does the consensus beat a typical single registration?
  - do the pipeline's QC gates (global IoU/grad_p95, local SSIM top
    fraction) actually select registrations that agree better with GT?

Individual masks are read from all_registrations/individual_masks/
(every registration; the per-subset folders hold duplicates). Each mask's
filename is rebuilt from its registrations_used.csv row with
pipeline_common.make_output_name, so masks and QC rows are joined exactly.

Output: results/validation_gt/individual_vs_gt.csv
"""

import argparse

import pandas as pd

from gt_utils import (
    OUT_DIR, FINAL_QC_SUBSET,
    region_table, load_label_mapping, load_gt_atlas, gt_region_mask,
    region_dir, load_mask, seg_metrics, run_groups,
)
from pipeline_common import make_output_name

PARAM_COLS = ["bending_energy_weight", "grid_spacing", "smoothing_sigma_floating"]


def score_group(sample_id, input_type) -> pd.DataFrame:
    _, agg = load_label_mapping()
    gt = load_gt_atlas(sample_id)
    rows = []

    for r in region_table().itertuples():
        gt_mask = gt_region_mask(gt, r.acronym, agg)
        all_dir = region_dir(sample_id, input_type, r.acronym, "all_registrations")
        ssim_col = f"local_ssim__{r.acronym}_both"

        used = pd.read_csv(all_dir / "registrations_used.csv")
        final = pd.read_csv(
            region_dir(sample_id, input_type, r.acronym, FINAL_QC_SUBSET) / "registrations_used.csv"
        )
        final_names = {make_output_name(row, prefix="mask") for row in final.to_dict("records")}

        for row in used.to_dict("records"):
            name = make_output_name(row, prefix="mask")
            pred = load_mask(all_dir / "individual_masks" / name, gt.shape)
            rows.append({
                "sample_id": sample_id,
                "input_type": input_type,
                "region": r.acronym,
                "mask_name": name,
                **{c: row[c] for c in PARAM_COLS},
                "qc_iou": row["iou"],
                "qc_grad_p95": row["grad_p95"],
                "qc_local_ssim": row[ssim_col],
                "passes_global_qc": bool(row["passes_global_qc"]),
                "in_final_qc": name in final_names,
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
    df = run_groups(score_group, args.workers, cache_name="individual_vs_gt", fresh=args.fresh)
    out = OUT_DIR / "individual_vs_gt.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
