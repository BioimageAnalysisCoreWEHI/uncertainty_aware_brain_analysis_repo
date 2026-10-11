"""A3-A13 statistics on the aMAP tables written by 02/03. Unit = brain x structure (case); brains are the
independent unit for CIs (bootstrap over brains). Primary: raw, registered hemispheres, final QC set, threshold 0.5,
paper protocol, DG-sg excluded, 5 QC-passing brains (ER reported separately).

Outputs: results/tables/amap_*.csv (+ numbers.json entries).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from common import (AMAP_OUT, BRAINS, CFG, QC_BRAINS, REJECTED_BRAINS, STRUCTS_MAIN, VOX, boot_brain, boot_spearman,
                    centred_spearman, holm, load_table, log, record, save_table, wilcoxon_p)

A = CFG["amap"]
TIE = float(CFG["tie_band"])
FINAL = "passes_final_qc_top_0p5"


def auc(score, positive):
    """P(score of a positive case > score of a negative case) (Mann-Whitney AUC)."""
    s, y = np.asarray(score, float), np.asarray(positive, bool)
    ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    return float(stats.mannwhitneyu(s[y], s[~y]).statistic / (y.sum() * (~y).sum()))


def primary_sets(sets):
    """Primary consensus per case: final QC set at 0.5; brains with no QC-passing run use all registrations."""
    p = sets[(sets.set_kind == "production") & (sets.threshold == 0.5)]
    fin = p[p.set_name == FINAL]
    rej = p[p.brain.isin(REJECTED_BRAINS) & (p.set_name == "all_registrations")]
    return pd.concat([fin, rej], ignore_index=True)


if __name__ == "__main__":
    runs = load_table("amap_runs")
    sets = load_table("amap_sets")
    qcg = load_table("amap_qcgrid")
    pred = load_table("amap_predictors")
    calib = load_table("amap_calib")
    relb = load_table("amap_reliability")
    hum = load_table("human_loo")
    key = ["brain", "structure", "variant"]
    prim = primary_sets(sets)

    # ======================================================================================================
    # A2/A3 agreement: pipeline vs experts vs humans
    # ======================================================================================================
    hcase = hum.groupby(["brain", "structure"]).agg(
        human_loo_median=("dice_loo", "median"), human_loo_q25=("dice_loo", lambda x: x.quantile(.25)),
        human_loo_q75=("dice_loo", lambda x: x.quantile(.75)), human_self_median=("dice_self", "median"),
        human_hd95_loo_median=("hd95_loo_um", "median"), human_msd_loo_median=("msd_loo_um", "median"),
        loo_vs_full_staple_dice=("dice_loo_vs_full_staple", "median"), n_ratings=("dice_loo", "size")).reset_index()
    ag = prim[key + ["n_runs", "dice_paper", "dice_fixed", "hd95_paper_um", "msd_paper_um", "hd_paper_um",
                     "area_ratio_paper", "A_best", "at_window_edge"]].merge(hcase, on=["brain", "structure"])
    pct = []
    for r in ag.itertuples():
        h = hum[(hum.brain == r.brain) & (hum.structure == r.structure)].dice_loo
        pct.append(float((h < r.dice_paper).mean() + 0.5 * (h == r.dice_paper).mean()))
    ag["pipeline_percentile_in_humans"] = pct
    ag["pipeline_minus_human_loo"] = ag.dice_paper - ag.human_loo_median
    save_table(ag, "amap_agreement_cases")
    rows = []
    main = ag[ag.structure.isin(STRUCTS_MAIN)]
    for variant in ("raw", "preprocessed"):
        groups = [(QC_BRAINS, "QC brains")] + ([(REJECTED_BRAINS, "QC-rejected (all runs)")] if REJECTED_BRAINS else [])
        for brains, label in groups:
            m = main[(main.variant == variant) & main.brain.isin(brains)]
            h = hum[hum.brain.isin(brains) & hum.structure.isin(STRUCTS_MAIN)]
            for st in STRUCTS_MAIN + ["ALL"]:
                mm = m if st == "ALL" else m[m.structure == st]
                hh = h if st == "ALL" else h[h.structure == st]
                est, lo, hi = boot_brain(mm, "dice_paper") if st == "ALL" and len(brains) > 1 else (
                    mm.dice_paper.median(), np.nan, np.nan)
                hest, hlo, hhi = boot_brain(hh, "dice_loo") if st == "ALL" and len(brains) > 1 else (
                    hh.dice_loo.median(), np.nan, np.nan)
                rows.append(dict(variant=variant, brains=label, structure=st, n_cases=len(mm),
                                 pipeline_dice_median=est, pipeline_ci_lo=lo, pipeline_ci_hi=hi,
                                 pipeline_dice_fixed_median=mm.dice_fixed.median(),
                                 pipeline_hd95_median_um=mm.hd95_paper_um.median(),
                                 pipeline_msd_median_um=mm.msd_paper_um.median(),
                                 human_loo_dice_median=hest, human_ci_lo=hlo, human_ci_hi=hhi,
                                 human_self_dice_median=hh.dice_self.median(),
                                 human_hd95_median_um=hh.hd95_loo_um.median(), human_msd_median_um=hh.msd_loo_um.median(),
                                 n_ratings=len(hh), median_pipeline_percentile=mm.pipeline_percentile_in_humans.median(),
                                 frac_cases_within_human_iqr=float((mm.dice_paper >= mm.human_loo_q25).mean())
                                 if len(mm) else np.nan,
                                 mannwhitney_p=stats.mannwhitneyu(mm.dice_paper, hh.dice_loo).pvalue
                                 if len(mm) and len(hh) else np.nan))
    agsum = pd.DataFrame(rows)
    save_table(agsum, "amap_agreement_summary")
    q = main[main.brain.isin(QC_BRAINS)]
    pr = q.pivot_table(index=["brain", "structure"], columns="variant", values="dice_paper").dropna()
    record("agreement", dict(
        raw=agsum[(agsum.variant == "raw") & (agsum.brains == "QC brains") & (agsum.structure == "ALL")].iloc[0].to_dict(),
        preprocessed=agsum[(agsum.variant == "preprocessed") & (agsum.brains == "QC brains") &
                           (agsum.structure == "ALL")].iloc[0].to_dict(),
        pre_minus_raw_median=float((pr.preprocessed - pr.raw).median()),
        pre_better_cases=int((pr.preprocessed > pr.raw + TIE).sum()), raw_better_cases=int((pr.raw > pr.preprocessed + TIE).sum()),
        n_paired=len(pr), wilcoxon_pre_vs_raw_p=wilcoxon_p(pr.preprocessed - pr.raw),
        human_loo_vs_full_staple=float(hum.dice_loo_vs_full_staple.median()),
        human_loo_le_self_frac=float((hum.dice_loo <= hum.dice_self + 1e-12).mean())))
    bb = q.groupby(["variant", "brain"]).dice_paper.median().unstack(0)
    save_table(bb.reset_index(), "amap_agreement_by_brain")

    # ======================================================================================================
    # A4 / A13 consensus vs realistic single choices
    # ======================================================================================================
    choice = sets[(sets.set_kind == "choice") & (sets.threshold == 0.5)]
    wide = prim[key + ["dice_paper", "hd95_paper_um", "area_ratio_paper", "dice_fixed"]].rename(
        columns={"dice_paper": "cons_dice", "hd95_paper_um": "cons_hd95", "area_ratio_paper": "cons_area",
                 "dice_fixed": "cons_dice_fixed"})
    for name, g in choice.groupby("set_name"):
        wide = wide.merge(g[key + ["dice_paper", "hd95_paper_um", "area_ratio_paper", "dice_fixed"]].rename(
            columns={"dice_paper": f"{name}_dice", "hd95_paper_um": f"{name}_hd95",
                     "area_ratio_paper": f"{name}_area", "dice_fixed": f"{name}_dice_fixed"}), on=key, how="left")
    for name in ("all_registrations", "passes_global_qc"):
        g = sets[(sets.set_kind == "production") & (sets.set_name == name) & (sets.threshold == 0.5)]
        wide = wide.merge(g[key + ["dice_paper", "hd95_paper_um", "area_ratio_paper"]].rename(
            columns={"dice_paper": f"{name}_cons_dice", "hd95_paper_um": f"{name}_cons_hd95",
                     "area_ratio_paper": f"{name}_cons_area"}), on=key, how="left")
    rs = []
    for (b, s, v), g in runs.groupby(key):
        qcr = g[g.passes_global_qc] if g.passes_global_qc.any() else g
        rs.append(dict(brain=b, structure=s, variant=v, n_qc=int(g.passes_global_qc.sum()),
                       qc_mean_dice=qcr.dice_paper.mean(), qc_median_dice=qcr.dice_paper.median(),
                       qc_p10_dice=qcr.dice_paper.quantile(0.10), qc_p05_dice=qcr.dice_paper.quantile(0.05),
                       qc_mean_hd95=qcr.hd95_paper_um.mean(), qc_mean_abs_area_err=(qcr.area_ratio_paper - 1).abs().mean(),
                       all_median_dice=g.dice_paper.median(), all_p10_dice=g.dice_paper.quantile(.1),
                       oracle_dice=g.dice_paper.max(), oracle_qc_dice=qcr.dice_paper.max(),
                       sd_dice_all=g.dice_paper.std(), iqr_dice_all=g.dice_paper.quantile(.75) - g.dice_paper.quantile(.25),
                       frac_all_below_0p5=float((g.dice_paper < 0.5).mean()), sd_dice_qc=qcr.dice_paper.std(),
                       best_run=g.sort_values("dice_paper", ascending=False).run.iloc[0],
                       qc_volume_cv=qcr.slab_volume_px.std() / qcr.slab_volume_px.mean()))
    rs = pd.DataFrame(rs)
    wide = wide.merge(rs, on=key, how="left")
    for r in wide.itertuples():
        g = runs[(runs.brain == r.brain) & (runs.structure == r.structure) & (runs.variant == r.variant)]
        qcr = g[g.passes_global_qc] if g.passes_global_qc.any() else g
        wide.loc[r.Index, "cons_pct_all"] = float((g.dice_paper < r.cons_dice).mean())
        wide.loc[r.Index, "cons_pct_qc"] = float((qcr.dice_paper < r.cons_dice).mean())
    # P8 transfer mismatch: gap between the case's best parameter and the LOBO-selected one, and its rank
    sel = load_table("amap_selections")
    for r in wide.itertuples():
        g = runs[(runs.brain == r.brain) & (runs.structure == r.structure) & (runs.variant == r.variant)]
        lob = sel[(sel.brain == r.brain) & (sel.variant == r.variant)].lobo_top1.iloc[0]
        d = g.set_index("run").dice_paper
        wide.loc[r.Index, "P8_transfer_gap"] = d.max() - d.get(lob, np.nan)
        wide.loc[r.Index, "P8_lobo_rank"] = int((d > d.get(lob, np.nan)).sum() + 1)
        wide.loc[r.Index, "lobo_run_passes_qc"] = bool(g.set_index("run").passes_global_qc.get(lob, False))
    comps = {"default": "default_dice", "top_ssim": "top_ssim_dice", "lobo_top1": "lobo_top1_dice",
             f"lobo_top{A['lobo_ensemble_k']}_ensemble": f"lobo_top{A['lobo_ensemble_k']}_ensemble_dice",
             "insample_best_param (optimistic)": "insample_best_param_dice",
             "insample_stiff_family (optimistic)": "insample_stiff_family_dice",
             "random_qc_run (expected)": "qc_mean_dice", "qc_run_p10 (tail)": "qc_p10_dice",
             "median_qc_run": "qc_median_dice", "all_registrations_consensus": "all_registrations_cons_dice",
             "global_qc_consensus": "passes_global_qc_cons_dice", "oracle_any_run (unattainable)": "oracle_dice",
             "oracle_qc_run (unattainable)": "oracle_qc_dice"}
    for k, col in comps.items():
        wide[f"delta__{k}"] = wide.cons_dice - wide[col]
    wide = wide.merge(pred, on=key, how="left").merge(
        hcase[["brain", "structure", "human_loo_median", "human_loo_q25", "human_loo_q75"]], on=["brain", "structure"],
        how="left").merge(calib[key + ["pipeline_band_width_um", "human_band_width_um",
                                       "spearman_pipeline_vs_human", "truth_boundary_in_pipeline_band"]], on=key, how="left")
    wide["human_loo_iqr"] = wide.human_loo_q75 - wide.human_loo_q25
    save_table(wide, "amap_benefit_cases")

    bsum = []
    for variant in ("raw", "preprocessed"):
        w = wide[(wide.variant == variant) & wide.brain.isin(QC_BRAINS) & wide.structure.isin(STRUCTS_MAIN)]
        for k in comps:
            d = w[f"delta__{k}"]
            est, lo, hi = boot_brain(w.assign(d=d), "d")
            bsum.append(dict(variant=variant, comparator=k, n=int(d.notna().sum()), median_delta=est, ci_lo=lo, ci_hi=hi,
                             mean_delta=d.mean(), wins=int((d > TIE).sum()), ties=int((d.abs() <= TIE).sum()),
                             losses=int((d < -TIE).sum()), wilcoxon_p=wilcoxon_p(d),
                             brains_consensus_better=int((w.assign(d=d).groupby("brain").d.mean() > 0).sum())))
    bsum = pd.DataFrame(bsum)
    save_table(bsum, "amap_benefit_summary")
    record("benefit_summary_raw", bsum[bsum.variant == "raw"].set_index("comparator")[
        ["median_delta", "ci_lo", "ci_hi", "wins", "ties", "losses", "brains_consensus_better"]].to_dict("index"))
    record("benefit_summary_pre", bsum[bsum.variant == "preprocessed"].set_index("comparator")[
        ["median_delta", "ci_lo", "ci_hi", "wins", "ties", "losses", "brains_consensus_better"]].to_dict("index"))

    # robustness: worst case per brain (minimax) and the risk of a bad result
    mm = []
    methods = {"consensus": "cons_dice", "default": "default_dice", "top_ssim": "top_ssim_dice",
               "lobo_top1": "lobo_top1_dice", f"lobo_top{A['lobo_ensemble_k']}_ensemble":
               f"lobo_top{A['lobo_ensemble_k']}_ensemble_dice", "random_qc_run (expected)": "qc_mean_dice",
               "qc_run_p10": "qc_p10_dice", "all_registrations_consensus": "all_registrations_cons_dice"}
    for variant in ("raw", "preprocessed"):
        w = wide[(wide.variant == variant) & wide.brain.isin(QC_BRAINS) & wide.structure.isin(STRUCTS_MAIN)]
        for m, col in methods.items():
            mm.append(dict(variant=variant, method=m, median=w[col].median(), p10=w[col].quantile(.1),
                           min=w[col].min(), n_below_0p6=int((w[col] < 0.6).sum()), n_below_0p7=int((w[col] < 0.7).sum()),
                           worst_per_brain_median=w.groupby("brain")[col].min().median(),
                           mean_abs_area_error=np.nan))
    save_table(pd.DataFrame(mm), "amap_robustness")
    # risk curves (source data): P(Dice < x) for runs by stage, consensus and deterministic choices
    xs = np.round(np.arange(0.0, 1.0001, 0.01), 2)
    rc = []
    for variant in ("raw", "preprocessed"):
        r = runs[(runs.variant == variant) & runs.brain.isin(QC_BRAINS) & runs.structure.isin(STRUCTS_MAIN)]
        w = wide[(wide.variant == variant) & wide.brain.isin(QC_BRAINS) & wide.structure.isin(STRUCTS_MAIN)]
        groups = {"all runs": r.dice_paper, "global-QC runs": r[r.passes_global_qc].dice_paper,
                  "final-QC runs": r[r.passes_final_qc].dice_paper, "consensus": w.cons_dice,
                  "default": w.default_dice, "LOBO parameter": w.lobo_top1_dice}
        for gname, vals in groups.items():
            vals = vals.dropna().to_numpy()
            for x in xs:
                rc.append(dict(variant=variant, group=gname, x=x, p_below=float((vals < x).mean()), n=len(vals)))
    save_table(pd.DataFrame(rc), "amap_risk_curves")

    # A13 predictors of benefit (pre-specified; all cases; Holm across predictors x comparators)
    P = {"P1_sd_dice_all [GT]": "sd_dice_all", "P1_volume_cv_all": "gf_volume_cv_all",
         "P2_pairwise_dice_final": "gf_pairwise_dice_final", "P2_band_fraction": "gf_band_fraction",
         "P3_frac_runs_dice_lt_0p5 [GT]": "frac_all_below_0p5", "P3_frac_runs_disagree": "gf_frac_runs_disagree",
         "P4_sd_dice_qc [GT]": "sd_dice_qc", "P4_sd_dice_to_cons_global": "gf_sd_dice_to_cons_global",
         "P5_truth_area_um2": "truth_area_um2", "P6_human_loo_median": "human_loo_median",
         "P6_human_loo_iqr": "human_loo_iqr", "P7_entropy": "gf_entropy", "P7_band_width_um": "gf_band_width_um",
         "P8_transfer_gap [GT]": "P8_transfer_gap", "P8_lobo_rank [GT]": "P8_lobo_rank"}
    targets = {k: f"delta__{k}" for k in ("default", "top_ssim", "lobo_top1", "insample_best_param (optimistic)",
                                          "random_qc_run (expected)", "qc_run_p10 (tail)")}
    pr_rows = []
    for variant in ("raw", "preprocessed"):
        w = wide[(wide.variant == variant) & wide.brain.isin(QC_BRAINS) & wide.structure.isin(STRUCTS_MAIN)]
        for (pn, pc), (tn, tc) in itertools.product(P.items(), targets.items()):
            rho, p, lo, hi = boot_spearman(w[pc], w[tc], w.brain)
            pr_rows.append(dict(variant=variant, predictor=pn, comparator=tn, n=int(w[[pc, tc]].dropna().shape[0]),
                                spearman=rho, p=p, ci_lo=lo, ci_hi=hi,
                                centred_spearman=centred_spearman(w, pc, tc)))
    prd = pd.DataFrame(pr_rows)
    for variant in ("raw", "preprocessed"):
        m = prd.variant == variant
        prd.loc[m, "p_holm"] = holm(prd.loc[m, "p"].to_numpy())
    save_table(prd, "amap_benefit_predictors")
    # benefit / tie / harm classes and predictor medians per class
    cl = []
    for variant in ("raw", "preprocessed"):
        w = wide[(wide.variant == variant) & wide.brain.isin(QC_BRAINS) & wide.structure.isin(STRUCTS_MAIN)]
        for tn, tc in targets.items():
            c = np.where(w[tc] > TIE, "benefit", np.where(w[tc] < -TIE, "harm", "tie"))
            for cls in ("benefit", "tie", "harm"):
                sub = w[c == cls]
                cl.append(dict(variant=variant, comparator=tn, cls=cls, n=len(sub),
                               cases=" ".join(f"{b.split('_')[0]}{'_' + b.split('_')[-1] if b.startswith('MV') else ''}:{s}"
                                              for b, s in zip(sub.brain, sub.structure)),
                               **{pn: sub[pc].median() for pn, pc in P.items()}))
    save_table(pd.DataFrame(cl), "amap_benefit_classes")

    # uncertainty information: do GT-free signals predict the consensus' own error?
    ui = []
    for variant in ("raw", "preprocessed"):
        w = wide[(wide.variant == variant) & wide.structure.isin(STRUCTS_MAIN)]
        wq = w[w.brain.isin(QC_BRAINS)]
        for pn, pc in P.items():
            if "[GT]" in pn:
                continue
            for tgt, col in (("consensus Dice", "cons_dice"), ("consensus HD95", "cons_hd95")):
                rho, p, lo, hi = boot_spearman(wq[pc], wq[col], wq.brain)
                ui.append(dict(variant=variant, signal=pn, target=tgt, spearman=rho, p=p, ci_lo=lo, ci_hi=hi,
                               centred_spearman=centred_spearman(wq, pc, col),
                               auc_flag_poor=auc(wq[pc] * (1 if "pairwise" not in pn and "human_loo_median" not in pn
                                                           else -1), wq.cons_dice < A["poor_dice"])
                               if tgt == "consensus Dice" else np.nan,
                               n_poor=int((wq.cons_dice < A["poor_dice"]).sum())))
    ui = pd.DataFrame(ui)
    save_table(ui, "amap_uncertainty_information")

    # ======================================================================================================
    # A5 parameter landscape / A6 generalisation
    # ======================================================================================================
    rm = runs[runs.structure.isin(STRUCTS_MAIN)]
    land = rm.groupby(["variant", "brain", "bending_energy", "grid"]).dice_paper.mean().reset_index()
    pooled = rm[rm.brain.isin(QC_BRAINS)].groupby(["variant", "bending_energy", "grid"]).agg(
        mean_dice=("dice_paper", "mean"), median_dice=("dice_paper", "median"), min_dice=("dice_paper", "min"),
        frac_pass_global_qc=("passes_global_qc", "mean")).reset_index()
    save_table(land, "amap_landscape_by_brain")
    save_table(pooled, "amap_landscape_pooled")
    win = rm.loc[rm.groupby(key).dice_paper.idxmax()]
    wf = win.groupby(["variant", "bending_energy", "grid"]).size().rename("n_cases_best").reset_index()
    save_table(wf, "amap_winner_frequency")
    conc = []
    for variant in ("raw", "preprocessed"):
        piv = land[land.variant == variant].pivot_table(index=["bending_energy", "grid"], columns="brain", values="dice_paper")
        for b1, b2 in itertools.combinations([b for b in BRAINS if b in piv], 2):
            conc.append(dict(variant=variant, brain1=b1, brain2=b2, spearman=stats.spearmanr(piv[b1], piv[b2]).statistic))
    conc = pd.DataFrame(conc)
    save_table(conc, "amap_param_rank_concordance")
    record("landscape", dict(
        n_distinct_winners={v: int(win[win.variant == v].run.nunique()) for v in ("raw", "preprocessed")},
        n_cases={v: int((win.variant == v).sum()) for v in ("raw", "preprocessed")},
        rank_concordance_median=conc.groupby("variant").spearman.median().to_dict(),
        top5_pooled={v: pooled[pooled.variant == v].nlargest(5, "mean_dice")[["bending_energy", "grid", "mean_dice",
                                                                               "frac_pass_global_qc"]].to_dict("records")
                     for v in ("raw", "preprocessed")},
        fine_grid_mean={v: float(pooled[(pooled.variant == v) & pooled.grid.isin([-2, -5])].mean_dice.mean())
                        for v in ("raw", "preprocessed")}))
    # LOBO for the consensus threshold and the QC thresholds (selected on 4 brains by median Dice)
    lob = []
    pset = sets[(sets.set_kind == "production") & (sets.set_name == FINAL) & sets.structure.isin(STRUCTS_MAIN)]
    for variant in ("raw", "preprocessed"):
        for b in QC_BRAINS:
            tr = pset[(pset.variant == variant) & (pset.brain != b) & pset.brain.isin(QC_BRAINS)]
            t_best = tr.groupby("threshold").dice_paper.median().idxmax()
            te = pset[(pset.variant == variant) & (pset.brain == b)]
            g = qcg[(qcg.variant == variant) & qcg.structure.isin(STRUCTS_MAIN)]
            gtr = g[(g.brain != b) & g.brain.isin(QC_BRAINS)]
            ok = gtr.groupby(["iou_thr", "grad_thr", "ssim_frac"]).dice_paper.agg(["median", "count"])
            ok = ok[ok["count"] == ok["count"].max()]
            best = ok["median"].idxmax()
            gte = g[(g.brain == b) & (g.iou_thr == best[0]) & (g.grad_thr == best[1]) & (g.ssim_frac == best[2])]
            lob.append(dict(variant=variant, held_out=b, lobo_threshold=t_best,
                            dice_at_lobo_threshold=te[te.threshold == t_best].dice_paper.median(),
                            dice_at_0p5=te[te.threshold == 0.5].dice_paper.median(),
                            lobo_qc=str(best), dice_at_lobo_qc=gte.dice_paper.median(),
                            dice_at_production_qc=te[te.threshold == 0.5].dice_paper.median()))
    save_table(pd.DataFrame(lob), "amap_lobo_thresholds")

    # ======================================================================================================
    # A7 QC metrics vs accuracy
    # ======================================================================================================
    qa = []
    for (b, s, v), g in rm.groupby(key):
        for m in ("iou", "grad_p95", "ssim", "bending_energy", "grid"):
            x = g[[m, "dice_paper"]].dropna()
            qa.append(dict(brain=b, structure=s, variant=v, metric=m,
                           spearman=stats.spearmanr(x[m], x.dice_paper).statistic if x[m].nunique() > 1 else np.nan))
        if g.passes_global_qc.any() and (~g.passes_global_qc).any():
            qa.append(dict(brain=b, structure=s, variant=v, metric="pass_minus_fail_median_dice",
                           spearman=g[g.passes_global_qc].dice_paper.median() - g[~g.passes_global_qc].dice_paper.median()))
    qa = pd.DataFrame(qa)
    save_table(qa, "amap_qc_vs_dice_cases")
    qs = []
    for (v, m), g in qa.groupby(["variant", "metric"]):
        gq = g[g.brain.isin(QC_BRAINS)]
        est, lo, hi = boot_brain(gq, "spearman")
        qs.append(dict(variant=v, metric=m, median_within_case=est, ci_lo=lo, ci_hi=hi, n_cases=int(gq.spearman.notna().sum()),
                       er_median=g[g.brain == "ER_Glt25d2Cre_9"].spearman.median()))
    qs = pd.DataFrame(qs)
    save_table(qs, "amap_qc_vs_dice_summary")
    record("qc_vs_dice", qs.set_index(["variant", "metric"])[["median_within_case", "ci_lo", "ci_hi"]]
           .reset_index().to_dict("records"))
    # per-run QC scatter (one point per registration; Dice = median over the 9 structures)
    sc = rm.groupby(["brain", "variant", "run", "bending_energy", "grid"]).agg(
        iou=("iou", "first"), grad_p95=("grad_p95", "first"), passes_global_qc=("passes_global_qc", "first"),
        median_dice=("dice_paper", "median")).reset_index()
    save_table(sc, "amap_qc_scatter")
    # QC-threshold grid summary (threshold 0.5), including which settings would admit ER
    gs = qcg[qcg.structure.isin(STRUCTS_MAIN)].groupby(["variant", "iou_thr", "grad_thr", "ssim_frac"]).apply(
        lambda g: pd.Series(dict(
            median_dice_qc_brains=g[g.brain.isin(QC_BRAINS)].dice_paper.median(),
            n_qc_brains_with_runs=int(g[g.brain.isin(QC_BRAINS)].groupby("brain").n_runs.max().gt(0).sum()),
            er_admitted=bool(g[g.brain == "ER_Glt25d2Cre_9"].n_runs.max() > 0),
            er_median_dice=g[g.brain == "ER_Glt25d2Cre_9"].dice_paper.median(),
            median_n_runs=g.n_runs.median())), include_groups=False).reset_index()
    save_table(gs, "amap_qcgrid_summary")
    er = runs[runs.brain == "ER_Glt25d2Cre_9"]
    record("er", dict(
        n_runs=int(er.run.nunique()),
        n_pass_iou={v: int(er[er.variant == v].groupby("run").iou.first().gt(A["iou_threshold"]).sum()) for v in ("raw", "preprocessed")},
        n_pass_grad={v: int(er[er.variant == v].groupby("run").grad_p95.first().lt(A["grad_p95_threshold"]).sum()) for v in ("raw", "preprocessed")},
        median_run_dice={v: float(er[(er.variant == v) & er.structure.isin(STRUCTS_MAIN)].dice_paper.median()) for v in ("raw", "preprocessed")},
        primary_consensus_dice={v: float(prim[(prim.brain == "ER_Glt25d2Cre_9") & (prim.variant == v) &
                                               prim.structure.isin(STRUCTS_MAIN)].dice_paper.median()) for v in ("raw", "preprocessed")},
        qc_brain_consensus_dice={v: float(prim[prim.brain.isin(QC_BRAINS) & (prim.variant == v) &
                                               prim.structure.isin(STRUCTS_MAIN)].dice_paper.median()) for v in ("raw", "preprocessed")}))

    # ======================================================================================================
    # A8 calibration / A9 threshold / A10 sensitivities / A11 extras
    # ======================================================================================================
    rl = relb[relb.structure.isin(STRUCTS_MAIN) & relb.brain.isin(QC_BRAINS)].copy()
    rl["n_exp"] = rl.frac_expert * rl.n_pixels
    pooled_rel = rl.groupby(["variant", "p_bin_lo", "p_bin_hi"]).agg(n_pixels=("n_pixels", "sum"), n_exp=("n_exp", "sum"),
                                                                    mean_p=("mean_p", "mean")).reset_index()
    pooled_rel["frac_expert"] = pooled_rel.n_exp / pooled_rel.n_pixels
    # brain-bootstrap band
    rng = np.random.default_rng(int(CFG["seed"]))
    bands = []
    for v in ("raw", "preprocessed"):
        r = rl[rl.variant == v]
        for _ in range(int(CFG["bootstrap_n"])):
            pick = rng.choice(QC_BRAINS, len(QC_BRAINS), replace=True)
            s = pd.concat([r[r.brain == b] for b in pick]).groupby("p_bin_lo")[["n_pixels", "n_exp"]].sum()
            bands.append(pd.DataFrame(dict(variant=v, p_bin_lo=s.index.to_numpy(), f=(s.n_exp / s.n_pixels).to_numpy())))
    bands = pd.concat(bands).groupby(["variant", "p_bin_lo"]).f.quantile([.025, .975]).unstack().reset_index()
    pooled_rel = pooled_rel.merge(bands.rename(columns={0.025: "ci_lo", 0.975: "ci_hi"}), on=["variant", "p_bin_lo"])
    save_table(pooled_rel, "amap_reliability_pooled")
    cm = calib[calib.structure.isin(STRUCTS_MAIN) & calib.brain.isin(QC_BRAINS)]
    record("calibration", cm.groupby("variant")[["spearman_pipeline_vs_human", "truth_boundary_in_pipeline_band",
                                                 "pipeline_band_width_um", "truth_boundary_in_human_band",
                                                 "human_band_width_um"]].median().to_dict("index"))
    # threshold curves and volume bias
    th = sets[(sets.set_kind == "production") & sets.structure.isin(STRUCTS_MAIN)]
    tc = th.groupby(["variant", "set_name", "structure", "threshold"]).agg(
        dice=("dice_paper", "median"), area_ratio=("area_ratio_paper", "median")).reset_index()
    save_table(tc, "amap_threshold_curves")
    tf = th[(th.set_name == FINAL)]
    opt = tf.loc[tf.groupby(key).dice_paper.idxmax()][key + ["threshold", "dice_paper", "area_ratio_paper"]]
    at05 = tf[tf.threshold == 0.5][key + ["dice_paper", "area_ratio_paper"]].rename(
        columns={"dice_paper": "dice_at_0p5", "area_ratio_paper": "area_ratio_at_0p5"})
    opt = opt.merge(at05, on=key)
    opt["gain_from_case_optimum"] = opt.dice_paper - opt.dice_at_0p5
    save_table(opt, "amap_threshold_optimum_cases")
    record("threshold", dict(
        optimum_counts={v: opt[opt.variant == v].threshold.value_counts().sort_index().to_dict() for v in ("raw", "preprocessed")},
        median_gain_from_case_optimum=opt.groupby("variant").gain_from_case_optimum.median().to_dict(),
        spearman_area_ratio_vs_optimum={v: float(stats.spearmanr(opt[opt.variant == v].area_ratio_at_0p5,
                                                                 opt[opt.variant == v].threshold).statistic)
                                        for v in ("raw", "preprocessed")},
        pooled_curve_raw=tc[(tc.variant == "raw") & (tc.set_name == FINAL)].groupby("threshold").dice.median().to_dict()))
    # knob sensitivity of the regional measurement (area in the slab) -- the aMAP analogue of the LSFM counts
    kn = []
    for r in wide[wide.structure.isin(STRUCTS_MAIN)].itertuples():
        g = runs[(runs.brain == r.brain) & (runs.structure == r.structure) & (runs.variant == r.variant)]
        qcr = g[g.passes_global_qc] if g.passes_global_qc.any() else g
        s = sets[(sets.brain == r.brain) & (sets.structure == r.structure) & (sets.variant == r.variant) &
                 (sets.set_kind == "production")]
        sp = s[s.set_name == (FINAL if r.brain not in REJECTED_BRAINS else "all_registrations")].set_index("threshold")
        V = sp.loc[0.5, "slab_volume_px"]
        kn.append(dict(brain=r.brain, structure=r.structure, variant=r.variant,
                       single_qc_run_iqr=(qcr.slab_volume_px.quantile(.75) - qcr.slab_volume_px.quantile(.25)) / V,
                       single_qc_run_range=(qcr.slab_volume_px.max() - qcr.slab_volume_px.min()) / V,
                       threshold_0p25_0p75=(sp.loc[0.25, "slab_volume_px"] - sp.loc[0.75, "slab_volume_px"]) / V,
                       threshold_0p4_0p6=(sp.loc[0.4, "slab_volume_px"] - sp.loc[0.6, "slab_volume_px"]) / V,
                       qc_set_choice=(s[s.threshold == 0.5].slab_volume_px.max() - s[s.threshold == 0.5].slab_volume_px.min()) / V,
                       default_vs_consensus=abs(g.set_index("run").slab_volume_px.get(A["default_run"], np.nan) / V - 1),
                       area_err_consensus=abs(r.cons_area - 1), area_err_default=abs(r.default_area - 1),
                       area_err_lobo=abs(r.lobo_top1_area - 1), area_err_qc_mean=r.qc_mean_abs_area_err))
    kn = pd.DataFrame(kn)
    save_table(kn, "amap_measurement_stability")
    record("measurement_stability", kn[kn.brain.isin(QC_BRAINS)].groupby("variant").median(numeric_only=True).to_dict("index"))
    # sensitivities: hemisphere mode, window, preprocessing
    old = pd.read_csv(AMAP_OUT / "pipeline_comparison" / "tables" / "main.csv")
    from common import PAPER_NAME
    old["structure"] = old.region.map(PAPER_NAME)
    old = old[(old.qc_set == FINAL) & (old.threshold == 0.5)][key + ["dice_paper"]].rename(columns={"dice_paper": "dice_amap_midline"})
    sens = prim[key + ["dice_paper", "dice_fixed", "dice_w5", "dice_w10", "A_best", "at_window_edge"]].merge(old, on=key, how="left")
    ap = pd.read_csv(AMAP_OUT / "pipeline_comparison_registered_hemispheres" / "tables" / "ap.csv")
    ap["structure"] = ap.region.map(PAPER_NAME)
    sens = sens.merge(ap[["brain", "structure", "A0", "rater_A_min", "rater_A_max", "rater_A_q25", "rater_A_q75"]],
                      on=["brain", "structure"], how="left")
    sens["dA_um"] = (sens.A_best - sens.A0) * VOX
    sens["A_best_within_rater_range"] = (sens.A_best >= sens.rater_A_min) & (sens.A_best <= sens.rater_A_max)
    save_table(sens, "amap_sensitivity_cases")
    ss = sens[sens.structure.isin(STRUCTS_MAIN) & sens.brain.isin(QC_BRAINS)].groupby("variant").agg(
        paper=("dice_paper", "median"), fixed=("dice_fixed", "median"), w5=("dice_w5", "median"),
        w10=("dice_w10", "median"), amap_midline=("dice_amap_midline", "median"),
        frac_edge=("at_window_edge", "mean"), dA_median_um=("dA_um", "median"),
        frac_within_rater_range=("A_best_within_rater_range", "mean")).reset_index()
    save_table(ss, "amap_sensitivity_summary")
    record("sensitivity", ss.set_index("variant").to_dict("index"))
    # A11: does human disagreement / size / brain predict pipeline error?
    ex = []
    w = wide[(wide.variant == "raw") & wide.brain.isin(QC_BRAINS) & wide.structure.isin(STRUCTS_MAIN)]
    for pn, pc in (("human LOO median Dice", "human_loo_median"), ("human LOO IQR", "human_loo_iqr"),
                   ("structure area", "truth_area_um2")):
        rho, p, lo, hi = boot_spearman(w[pc], w.cons_dice, w.brain)
        ex.append(dict(variant="raw", predictor=pn, target="consensus Dice", spearman=rho, p=p, ci_lo=lo, ci_hi=hi))
    save_table(pd.DataFrame(ex), "amap_extras")
    log("A3-A13 tables written")
