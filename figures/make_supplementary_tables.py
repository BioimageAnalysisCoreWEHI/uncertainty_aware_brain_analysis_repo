"""Supplementary Tables S1-S5 (manuscript numbering of 2026-10-10) as CSV, computed from the analysis tables.

  S1  consensus SNpc count vs SSIM retention fraction (top 25 % / 50 % / all global-QC passers), original LSFM
  S2  SNpc TH+ counts per brain: retained registrations, consensus at 0.5 / 0.25 / 0.75, single-registration ranges
  S3  consensus vs single-registration strategies (aMAP, raw input)
  S4  independent LSFM, agreement between the consensus and the published registration by region
  S5  global QC rules of the sensitivity analysis: thresholds and % of raw registrations retained, three datasets

Usage:
  python make_supplementary_tables.py --tables <validation results/tables> --indep <consensus_vs_published.csv>
                                      --out <folder>
The independent-LSFM table may be given under its pipeline name (consensus_vs_gt.csv) or its released name
(consensus_vs_published.csv); "gt" columns are read as "published".
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

LAB = {"162": "Sample 1", "527": "Sample 2", "pp451": "Sample 3", "xl353": "Sample 4", "xl385": "Sample 5"}
# the eight global-QC rules (setting ids of the sensitivity grid), as in the Fig. 6 scripts
REGIMES = {
    "No global QC": {"amap": "iounone_gradnone", "paper": "iounone_gradnone", "lsfm2": "iounone_gradnone"},
    "Very lenient": {"amap": "iou0p8_grad0p08", "paper": "iou0p84_grad0p08", "lsfm2": "iou0p5_grad0p13"},
    "Lenient": {"amap": "iou0p875_grad0p065", "paper": "iou0p86_grad0p07", "lsfm2": "iou0p6_grad0p1"},
    "Untuned start (median ± 3·MAD)": {"amap": "iou0p8537_grad0p0528", "paper": "iou0p8107_grad0p0795",
                                      "lsfm2": "iou0p7455_grad0p0847"},
    "Current": {"amap": "iou0p9_grad0p058", "paper": "iou0p9_grad0p045", "lsfm2": "iou0p8_grad0p08"},
    "Strict": {"amap": "iou0p9078_grad0p055", "paper": "iou0p91_grad0p042", "lsfm2": "iou0p82_grad0p07"},
    "Very strict": {"amap": "iou0p9078_grad0p047", "paper": "iou0p92_grad0p038", "lsfm2": "iou0p84_grad0p067"},
    "Median (marginal medians)": {"amap": "iou0p9078_grad0p0454", "paper": "iou0p9016_grad0p042",
                                  "lsfm2": "iou0p8632_grad0p067"},
}
N_RUNS = {"amap": 64, "paper": 48, "lsfm2": 64}
PER_CASE = {"amap": "qcsens_amap_per_case.csv", "paper": "qcsens_paper_per_case.csv", "lsfm2": "qcsens_lsfm2_per_case.csv"}
COMPARATORS = [("default", "Brainreg default (0.95, −10)"), ("top_ssim", "Highest-SSIM QC-passing registration"),
               ("random_qc_run (expected)", "Random QC-passing registration"),
               ("lobo_top1", "Leave-one-brain-out parameter"),
               ("qc_run_p10 (tail)", "10th-percentile QC-passing registration*")]


def thr(sid):
    m = re.fullmatch(r"iou(none|[0-9p]+)_grad(none|[0-9p]+)", sid)
    f = lambda s: "–" if s == "none" else s.replace("p", ".")
    return f(m.group(1)), f(m.group(2))


def table_s1_s2(T):
    c = pd.read_csv(T / "lsfm_consensus.csv")
    c = c[c.qc_filter == "passed_qc"].copy()
    c["sample"] = c["sample"].astype(str)
    r = pd.read_csv(T / "lsfm_per_run.csv")
    r["sample"] = r["sample"].astype(str)
    s1, s2 = [], []
    for st in ("raw", "preprocessed"):
        for s, lab in LAB.items():
            g = c[(c["sample"] == s) & (c.status == st)]
            at = lambda f, t: g[(g.ssim_filter == f) & np.isclose(g.threshold, t)].iloc[0]
            b = at("top_50pct", 0.5)
            row = dict(sample=lab, input=st)
            for f, name in (("top_25pct", "top25"), ("top_50pct", "top50_used"), ("all", "all_global_qc_passers")):
                x = at(f, 0.5)
                row[f"n_{name}"], row[f"count_{name}"] = int(x.n_runs), int(x["count"])
                if f != "top_50pct":
                    row[f"change_{name}_vs_top50_pct"] = round((x["count"] - b["count"]) / b["count"] * 100, 1)
            s1.append(row)
            rr = r[(r["sample"] == s) & (r.status == st)]
            ret = rr[rr.pass_qc & rr.in_ssim_top50].count_overlap
            assert len(ret) == int(b.n_runs) and len(rr) == 48
            s2.append(dict(sample=lab, input=st, retained_registrations=int(b.n_runs),
                           consensus_count_0p5=int(b["count"]), consensus_count_0p25=int(at("top_50pct", 0.25)["count"]),
                           consensus_count_0p75=int(at("top_50pct", 0.75)["count"]),
                           retained_single_min=int(ret.min()), retained_single_max=int(ret.max()),
                           all48_single_min=int(rr.count_overlap.min()), all48_single_max=int(rr.count_overlap.max())))
    return pd.DataFrame(s1), pd.DataFrame(s2)


def table_s3(T):
    b = pd.read_csv(T / "amap_benefit_summary.csv")
    b = b[b.variant == "raw"].set_index("comparator")
    return pd.DataFrame([dict(comparator=lab, median_delta_dice=round(b.loc[k, "median_delta"], 3),
                              wins=int(b.loc[k, "wins"]), ties=int(b.loc[k, "ties"]), losses=int(b.loc[k, "losses"]),
                              n_cases=int(b.loc[k, "n"])) for k, lab in COMPARATORS])


def table_s4(indep_csv):
    g = pd.read_csv(indep_csv)
    g.columns = [re.sub(r"(?<![a-z])gt(?![a-z])", "published", c) for c in g.columns]
    g = g[(g.input_type == "raw") & (g.subset == "passes_final_qc_top_0p5") & np.isclose(g.threshold, 0.5)]
    rows = []
    for reg, x in g.groupby("region"):
        q = lambda v: (round(v.median(), 3), round(v.quantile(.25), 3), round(v.quantile(.75), 3))
        d, b = q(x.dice), q(x.assd_um)
        rows.append(dict(region=reg, published_volume_mm3=round(x.vol_published_mm3.median(), 3), n_brains=len(x),
                         dice_median=d[0], dice_q1=d[1], dice_q3=d[2],
                         mean_boundary_distance_um_median=b[0], mean_boundary_distance_um_q1=b[1],
                         mean_boundary_distance_um_q3=b[2]))
    return pd.DataFrame(rows).sort_values("published_volume_mm3", ascending=False)


def table_s5(T):
    ret = {}
    for ds, f in PER_CASE.items():
        p = pd.read_csv(T / f, usecols=["sample", "variant", "setting_id", "n_global"])
        p = p[p.variant == "raw"]
        n_samples = p["sample"].nunique()
        for rule, ids in REGIMES.items():
            g = p[p.setting_id == ids[ds]].groupby("sample").n_global.first()
            ret[(rule, ds)] = 100 * g.sum() / (N_RUNS[ds] * n_samples)
    rows = []
    for rule, ids in REGIMES.items():
        row = dict(rule=rule)
        for ds, name in (("amap", "aMAP"), ("paper", "original_LSFM"), ("lsfm2", "independent_LSFM")):
            i, gr = thr(ids[ds])
            row[f"{name}_IoU"], row[f"{name}_grad_p95"] = i, gr
            row[f"{name}_retained_pct"] = round(ret[(rule, ds)], 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True, type=Path)
    ap.add_argument("--indep", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s1, s2 = table_s1_s2(a.tables)
    for name, df in (("TableS1_SSIM_retention_fraction", s1), ("TableS2_SNpc_counts_per_brain", s2),
                     ("TableS3_consensus_vs_single_registrations_aMAP", table_s3(a.tables)),
                     ("TableS4_independent_LSFM_agreement_by_region", table_s4(a.indep)),
                     ("TableS5_global_QC_rules", table_s5(a.tables))):
        df.to_csv(a.out / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} rows")


if __name__ == "__main__":
    main()
