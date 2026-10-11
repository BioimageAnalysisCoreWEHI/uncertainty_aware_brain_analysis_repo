# -*- coding: utf-8 -*-
"""
11_snc_registration_diagnostics.py

Targeted version of 07c for the SNc question: score every single registration's SNc and CP mask (CP = large-region
control) against the published registration, with its QC metrics and QC outcome. Read-only on all inputs.

Output: results/validation_gt/snc_cp_individual_vs_gt.csv
"""

import pandas as pd

from gt_utils import (OUT_DIR, FINAL_QC_SUBSET, load_label_mapping, load_gt_atlas, gt_region_mask, region_dir,
                      load_mask, seg_metrics, run_groups)
from pipeline_common import make_output_name

REGIONS = ["SNc", "CP"]


def score_group(sample_id, input_type):
    _, agg = load_label_mapping()
    gt = load_gt_atlas(sample_id)
    rows = []
    for reg in REGIONS:
        ref = gt_region_mask(gt, reg, agg)
        all_dir = region_dir(sample_id, input_type, reg, "all_registrations")
        used = pd.read_csv(all_dir / "registrations_used.csv")
        final = pd.read_csv(region_dir(sample_id, input_type, reg, FINAL_QC_SUBSET) / "registrations_used.csv")
        final_names = {make_output_name(r, prefix="mask") for r in final.to_dict("records")}
        for r in used.to_dict("records"):
            name = make_output_name(r, prefix="mask")
            pred = load_mask(all_dir / "individual_masks" / name, gt.shape)
            rows.append(dict(sample_id=sample_id, input_type=input_type, region=reg, mask_name=name,
                             bending_energy_weight=r["bending_energy_weight"], grid_spacing=r["grid_spacing"],
                             qc_iou=r["iou"], qc_grad_p95=r["grad_p95"], qc_local_ssim=r[f"local_ssim__{reg}_both"],
                             passes_global_qc=bool(r["passes_global_qc"]), in_final_qc=name in final_names,
                             **seg_metrics(pred, ref)))
    print(f"done: {sample_id} / {input_type}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_groups(score_group, None, cache_name="snc_cp_individual_vs_gt")
    out = OUT_DIR / "snc_cp_individual_vs_gt.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")
