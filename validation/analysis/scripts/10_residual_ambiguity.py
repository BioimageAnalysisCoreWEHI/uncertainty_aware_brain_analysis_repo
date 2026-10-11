"""A16: residual ambiguity among LSFM registrations that pass ALL QC (global QC + SSIM top 50 %), per brain.

This is the choice the consensus is meant to resolve. Per brain x input:
  n retained; min / median / max / IQR / SD / CV of their SNpc counts (overlap rule); range and IQR as % of the
  consensus count; consensus minus the median retained run; the same for the global-QC-only set, for context.
Cohort level:
  within-brain SD (all-QC set) vs between-animal SD of the consensus;
  how often a random choice of one retained run per brain preserves the consensus ranking of the five animals
  (Kendall tau, 5,000 draws).
Raw vs preprocessed:
  overlap (Jaccard) of the retained parameter sets, to explain why the consensus is not more reproducible across
  inputs than a fixed parameter.
Outputs: results/tables/residual_*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import CFG, log, record, save_table, load_table

if __name__ == "__main__":
    pr = load_table("lsfm_per_run")
    pr["sample"] = pr["sample"].astype(str)
    cons = load_table("lsfm_consensus")
    cons["sample"] = cons["sample"].astype(str)
    prim = cons[(cons.qc_filter == "passed_qc") & (cons.ssim_filter == "top_50pct") & (cons.threshold == 0.5)] \
        .set_index(["sample", "status"])["count"]
    rows = []
    for (s, st), g in pr.groupby(["sample", "status"]):
        P = prim[(s, st)]
        for name, sel in (("all_qc (global QC + SSIM top 50 %)", g.in_ssim_top50.astype(bool)),
                          ("global_qc_only", g.pass_qc)):
            x = g[sel].count_overlap.to_numpy(float)
            q1, q3 = np.percentile(x, [25, 75])
            rows.append(dict(sample=s, status=st, set=name, n_retained=len(x), consensus=P, min=x.min(),
                             median=np.median(x), max=x.max(), sd=x.std(ddof=1), cv=x.std(ddof=1) / x.mean(),
                             range_pct_of_consensus=100 * (x.max() - x.min()) / P,
                             iqr_pct_of_consensus=100 * (q3 - q1) / P,
                             max_abs_dev_pct=100 * np.abs(x / P - 1).max(),
                             consensus_minus_median_pct=100 * (P / np.median(x) - 1),
                             retained_params=" ".join(g[sel].run_key) if name.startswith("all") else ""))
    res = pd.DataFrame(rows)
    save_table(res, "residual_ambiguity_per_brain")

    rng = np.random.default_rng(int(CFG["seed"]))
    coh = []
    for st in ("raw", "preprocessed"):
        samples = sorted(pr["sample"].unique())
        ref = np.array([prim[(s, st)] for s in samples])
        sets = [pr[(pr["sample"] == s) & (pr.status == st) & pr.in_ssim_top50.astype(bool)].count_overlap.to_numpy(float)
                for s in samples]
        taus = []
        for _ in range(5000):
            x = np.array([rng.choice(v) for v in sets])
            taus.append(stats.kendalltau(ref, x).statistic)
        taus = np.array(taus)
        within = np.sqrt(np.mean([v.var(ddof=1) for v in sets]))
        coh.append(dict(status=st, between_animal_sd=ref.std(ddof=1), between_animal_cv=ref.std(ddof=1) / ref.mean(),
                        within_brain_sd_all_qc=within, within_to_between_sd=within / ref.std(ddof=1),
                        p_ranking_identical=float((taus > 0.999).mean()), median_tau=float(np.median(taus)),
                        p_tau_ge_0p6=float((taus >= 0.6 - 1e-9).mean()), min_tau=float(taus.min())))
    coh = pd.DataFrame(coh)
    save_table(coh, "residual_ambiguity_cohort")

    jac = []
    for s in sorted(pr["sample"].unique()):
        a = set(pr[(pr["sample"] == s) & (pr.status == "raw") & pr.in_ssim_top50.astype(bool)].run_key)
        b = set(pr[(pr["sample"] == s) & (pr.status == "preprocessed") & pr.in_ssim_top50.astype(bool)].run_key)
        ga = set(pr[(pr["sample"] == s) & (pr.status == "raw") & pr.pass_qc].run_key)
        gb = set(pr[(pr["sample"] == s) & (pr.status == "preprocessed") & pr.pass_qc].run_key)
        jac.append(dict(sample=s, n_raw=len(a), n_pre=len(b), n_shared=len(a & b), jaccard_all_qc=len(a & b) / len(a | b),
                        jaccard_global_qc=len(ga & gb) / len(ga | gb)))
    jac = pd.DataFrame(jac)
    rp = load_table("lsfm_raw_vs_preprocessed")
    rp["sample"] = rp["sample"].astype(str)
    for m, col in (("primary_consensus", "consensus_raw_vs_pre_pct"), ("default_0.95_-10", "default_raw_vs_pre_pct"),
                   ("same_param_single_run_(median over params passing QC in both)", "same_param_median_raw_vs_pre_pct")):
        jac = jac.merge(rp[rp.method == m][["sample", "abs_rel_diff"]].rename(columns={"abs_rel_diff": col}), on="sample")
        jac[col] *= 100
    save_table(jac, "residual_raw_vs_pre_membership")
    record("residual", dict(per_brain=res[res.set.str.startswith("all")].round(4).to_dict("records"),
                            cohort=coh.round(4).to_dict("records"), membership=jac.round(4).to_dict("records")))
    pd.set_option("display.width", 250)
    log(res.drop(columns="retained_params").round(3).to_string())
    log(coh.round(3).to_string())
    log(jac.round(3).to_string())
