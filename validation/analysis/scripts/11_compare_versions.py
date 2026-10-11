"""Old vs new results after the pipeline regeneration of 2026-09-29 (grad_p95 threshold 0.045 -> 0.058, preprocessed
registrations rerun). Old = archive/2026-09-28_grad0p045, new = current results. Writes
results/tables/version_comparison_*.csv and prints a summary.
"""
from __future__ import annotations

import pandas as pd

from common import STRUCTS_MAIN, TABLES, VA, log

OLD = VA / "archive" / "2026-09-28_grad0p045"
KEY = ["brain", "structure", "variant"]


def load(name, old=False, focused=False):
    base = (OLD if old else VA) / ("focused/tables" if focused else "results/tables")
    return pd.read_csv(base / f"{name}.csv")


if __name__ == "__main__":
    out = {}
    # QC pass counts per brain x variant
    r_new, r_old = load("amap_runs"), load("amap_runs", old=True)
    cnt = []
    for tag, r in (("old", r_old), ("new", r_new)):
        g = r.groupby(["brain", "variant", "run"]).agg(glob=("passes_global_qc", "first")).reset_index()
        f = r[r.structure.isin(STRUCTS_MAIN)].groupby(["brain", "variant"]).passes_final_qc.mean() * 64
        c = g.groupby(["brain", "variant"]).glob.sum().rename(f"n_global_{tag}").to_frame()
        c[f"mean_n_final_{tag}"] = f
        cnt.append(c)
    cnt = pd.concat(cnt, axis=1).reset_index()
    out["qc_counts"] = cnt
    # per-case consensus Dice (primary)
    a_new, a_old = load("amap_agreement_cases"), load("amap_agreement_cases", old=True)
    cols = ["dice_paper", "dice_fixed", "n_runs"]
    case = a_old[KEY + cols].merge(a_new[KEY + cols], on=KEY, how="outer", suffixes=("_old", "_new"))
    case["delta_dice_paper"] = case.dice_paper_new - case.dice_paper_old
    case["delta_dice_fixed"] = case.dice_fixed_new - case.dice_fixed_old
    out["cases"] = case
    # summary tables
    s_new, s_old = load("amap_agreement_summary"), load("amap_agreement_summary", old=True)
    sc = ["variant", "brains", "structure", "n_cases", "pipeline_dice_median", "pipeline_ci_lo", "pipeline_ci_hi",
          "pipeline_dice_fixed_median", "human_loo_dice_median"]
    out["agreement"] = s_old[sc].merge(s_new[sc], on=["variant", "brains", "structure"], how="outer",
                                      suffixes=("_old", "_new"))
    b_new, b_old = load("amap_benefit_summary"), load("amap_benefit_summary", old=True)
    bc = ["variant", "comparator", "median_delta", "ci_lo", "ci_hi", "wins", "ties", "losses"]
    out["benefit"] = b_old[bc].merge(b_new[bc], on=["variant", "comparator"], suffixes=("_old", "_new"))
    f_new, f_old = load("focused_structure_summary", focused=True), load("focused_structure_summary", old=True,
                                                                          focused=True)
    fc = ["structure", "n_cases", "consensus_dice_amap", "consensus_iou_amap", "consensus_dice_staple_plane",
          "rater_dice_median", "cases_within_rater_range", "median_percentile_among_raters"]
    out["focused"] = f_old[fc].merge(f_new[fc], on="structure", suffixes=("_old", "_new"))
    for k, df in out.items():
        df.to_csv(TABLES / f"version_comparison_{k}.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for k, df in out.items():
        log(f"\n== {k} ==")
        log(df.round(3).to_string())
    c = case[case.structure.isin(STRUCTS_MAIN)]
    log("\nper-case change in primary Dice (paper protocol), by variant:")
    log(c.groupby("variant").delta_dice_paper.describe().round(3).to_string())
    log("\nlargest changes:")
    log(c.reindex(c.delta_dice_paper.abs().sort_values(ascending=False).index).head(12).round(3).to_string())
