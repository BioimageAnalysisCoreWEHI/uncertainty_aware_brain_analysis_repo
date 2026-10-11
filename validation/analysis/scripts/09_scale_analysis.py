"""A15: robustness / reliability at scale (cohort use without per-sample tuning).

A. Selection freedom (LSFM). Among registrations an analyst would accept (pass global QC; or the SSIM top-50 % set),
   how far can the choice of one registration per brain move the SNpc count, and how does that compare with the
   between-animal spread of the consensus? Bias bound for group comparisons: for every split of the 5 brains into
   groups of 2 and 3, the apparent difference in group means under (i) the consensus, (ii) a random accepted run per
   brain, and (iii) the most extreme selection (highest-count runs in one group, lowest in the other).
B. Sweep size (LSFM counts, aMAP Dice): repeatedly subsample k of the parameter settings, apply the unchanged pipeline
   rules to the subset, and measure the deviation from the full-sweep consensus. Also three pre-specified structured
   3 x 3 subsets. Cost uses the manuscript's 92.2 h / 240 registrations = 23 min per registration.
C. Tail risk across a cohort (aMAP): expected shortfall (mean Dice of the worst 10 % of cases), fraction of cases
   below 0.6 / 0.7, and the expected worst case in a cohort of N cases (resampling the 45 cases).

Outputs: results/tables/scale_*.csv
"""
from __future__ import annotations

import importlib.util
import itertools
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import CACHE, CFG, QC_BRAINS, STRUCTS_MAIN, VA, load_table, log, record, save_table

SEED = int(CFG["seed"])
MIN_PER_REG = 92.2 * 60 / 240          # manuscript: cumulative runtime / registrations


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, str(VA / "scripts" / fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------------------
# A. selection freedom (LSFM)
# ------------------------------------------------------------------------------------------------------------
def selection_freedom():
    pr = load_table("lsfm_per_run")
    pr["sample"] = pr["sample"].astype(str)
    cons = load_table("lsfm_consensus")
    cons["sample"] = cons["sample"].astype(str)
    prim = cons[(cons.qc_filter == "passed_qc") & (cons.ssim_filter == "top_50pct") & (cons.threshold == 0.5)] \
        .set_index(["sample", "status"])["count"]
    rows, split_rows = [], []
    rng = np.random.default_rng(SEED)
    for st in ("raw", "preprocessed"):
        samples = sorted(pr["sample"].unique())
        acc = {}
        for s in samples:
            g = pr[(pr["sample"] == s) & (pr.status == st)]
            for setname, sel in (("global_qc", g.pass_qc), ("ssim_top50", g.in_ssim_top50.astype(bool))):
                x = g[sel].count_overlap.to_numpy(float)
                P = prim[(s, st)]
                rows.append(dict(status=st, sample=s, accepted_set=setname, n_accepted=len(x), consensus=P,
                                 min=x.min(), max=x.max(), sd=x.std(ddof=1), up=(x.max() / P - 1), down=(x.min() / P - 1),
                                 range_rel=(x.max() - x.min()) / P))
                acc[(s, setname)] = x
        between_sd = prim.xs(st, level="status").std(ddof=1)
        between_mean = prim.xs(st, level="status").mean()
        for setname in ("global_qc", "ssim_top50"):
            within = np.sqrt(np.mean([acc[(s, setname)].var(ddof=1) for s in samples]))
            rows.append(dict(status=st, sample="ALL", accepted_set=setname, between_animal_sd_consensus=between_sd,
                             between_animal_cv_consensus=between_sd / between_mean, within_selection_sd=within,
                             selection_to_between_variance=within ** 2 / between_sd ** 2))
            for ga in itertools.combinations(samples, 2):
                gb = [s for s in samples if s not in ga]
                c = lambda grp: np.mean([prim[(s, st)] for s in grp])  # noqa: E731
                true = c(ga) / c(gb) - 1
                hi = np.mean([acc[(s, setname)].max() for s in ga]) / np.mean([acc[(s, setname)].min() for s in gb]) - 1
                lo = np.mean([acc[(s, setname)].min() for s in ga]) / np.mean([acc[(s, setname)].max() for s in gb]) - 1
                rnd = [np.mean([rng.choice(acc[(s, setname)]) for s in ga]) /
                       np.mean([rng.choice(acc[(s, setname)]) for s in gb]) - 1 for _ in range(2000)]
                split_rows.append(dict(status=st, accepted_set=setname, group_a="+".join(ga), group_b="+".join(gb),
                                       consensus_diff=true, extreme_selection_hi=hi, extreme_selection_lo=lo,
                                       random_selection_p05=np.percentile(rnd, 5),
                                       random_selection_p95=np.percentile(rnd, 95),
                                       random_selection_sd=np.std(rnd)))
    a, b = pd.DataFrame(rows), pd.DataFrame(split_rows)
    save_table(a, "scale_selection_freedom")
    save_table(b, "scale_selection_group_bias")
    return a, b


# ------------------------------------------------------------------------------------------------------------
# B. sweep size -- LSFM
# ------------------------------------------------------------------------------------------------------------
def structured_lsfm():
    return {"3x3 moderate": [f"{b:g}_{g:g}" for b in (0.1, 0.5, 0.9) for g in (-10, -30, -50)],
            "3x3 default-centred": [f"{b:g}_{g:g}" for b in (0.7, 0.9, 0.95) for g in (-5, -10, -30)],
            "3x3 stiff (aMAP-optimal)": [f"{b:g}_{g:g}" for b in (0.9, 0.95, 1.0) for g in (-10, -30, -50)]}


def sweep_lsfm(m7, reps=200, ks=(4, 8, 12, 16, 24, 36, 48)):
    runs = pd.read_csv(CACHE / "lsfm" / "runs.csv")
    runs["sample"] = runs["sample"].astype(str)
    rows = []
    rng = np.random.default_rng(SEED)
    for s in sorted(runs["sample"].unique()):
        for st in ("raw", "preprocessed"):
            c = m7.Counter(s, st)
            sub = runs[(runs["sample"] == s) & (runs.status == st)].reset_index(drop=True)
            full = c.count(m7.select(sub), 0.5)
            keys = sub.run_key.to_numpy()
            designs = [(f"random {k}", k, [list(rng.choice(keys, k, replace=False)) for _ in range(reps if k < 48 else 1)])
                       for k in ks]
            designs += [(n, 9, [r]) for n, r in structured_lsfm().items()]
            for name, k, draws in designs:
                for d in draws:
                    ss = sub[sub.run_key.isin(d)]
                    sel = m7.select(ss)
                    n_pass = int(((ss.iou > 0.9) & (ss.grad_p95 < 0.045)).sum())
                    val = c.count(sel, 0.5) if n_pass else np.nan
                    rows.append(dict(sample=s, status=st, design=name, k=k, n_pass=n_pass, count=val, full=full,
                                     abs_rel_dev=abs(val / full - 1) if n_pass else np.nan))
            log(f"  sweep LSFM {s} {st} done")
    df = pd.DataFrame(rows)
    save_table(df, "scale_sweep_lsfm_draws")
    summ = df.groupby(["status", "design", "k"]).agg(
        median_abs_dev=("abs_rel_dev", "median"), p90_abs_dev=("abs_rel_dev", lambda x: x.quantile(.9)),
        frac_no_run_passes=("n_pass", lambda x: float((x == 0).mean())), median_n_pass=("n_pass", "median")).reset_index()
    summ["cpu_hours_per_brain"] = summ.k * MIN_PER_REG / 60
    save_table(summ, "scale_sweep_lsfm")
    return summ


# ------------------------------------------------------------------------------------------------------------
# B. sweep size -- aMAP (Dice), parallel over cases
# ------------------------------------------------------------------------------------------------------------
def _amap_case(job):
    brain, structure, variant, qrows, ks, reps, seed = job
    m3 = _load("m3", "03_amap_metrics.py")
    cs = m3.Case(brain, structure, variant)
    rng = np.random.default_rng(seed)
    A = CFG["amap"]
    full_runs = m3.qc_subset(qrows, structure, A["iou_threshold"], A["grad_p95_threshold"], A["ssim_top_fraction"])
    full = cs.score(cs.consensus(full_runs, 0.5), full=False)["dice_paper"]
    runs = np.array(cs.runs)
    out = []
    structured = {"3x3 moderate": [f"bend_{b}__grid_m{g}__smooth_m1p0" for b in ("0p3", "0p7", "0p95") for g in (10, 20, 30)],
                  "3x3 default-centred": [f"bend_{b}__grid_m{g}__smooth_m1p0" for b in ("0p9", "0p95", "0p99") for g in (7, 10, 15)],
                  "3x3 stiff": [f"bend_{b}__grid_m{g}__smooth_m1p0" for b in ("0p95", "0p99", "1p0") for g in (20, 25, 30)]}
    designs = [(f"random {k}", k, [list(rng.choice(runs, k, replace=False)) for _ in range(reps if k < len(runs) else 1)])
               for k in ks] + [(n, 9, [r]) for n, r in structured.items()]
    for name, k, draws in designs:
        for d in draws:
            q = qrows[qrows.run.isin(d)]
            sel = m3.qc_subset(q, structure, A["iou_threshold"], A["grad_p95_threshold"], A["ssim_top_fraction"])
            dd = cs.score(cs.consensus(sel, 0.5), full=False)["dice_paper"] if sel else np.nan
            out.append(dict(brain=brain, structure=structure, variant=variant, design=name, k=k, n_sel=len(sel),
                            dice=dd, full_dice=full, delta=dd - full if sel else np.nan))
    return out


def sweep_amap(reps=100, ks=(4, 8, 16, 24, 32, 48, 64)):
    m3 = _load("m3", "03_amap_metrics.py")
    qc = m3.load_qc()
    jobs = [(b, s, v, qc[(qc.brain == b) & (qc.variant == v)], ks, reps, SEED + i)
            for i, (b, s, v) in enumerate((b, s, v) for b in QC_BRAINS for s in STRUCTS_MAIN for v in ("raw", "preprocessed"))]
    with ProcessPoolExecutor(max_workers=8) as ex:
        df = pd.DataFrame([r for rs in ex.map(_amap_case, jobs) for r in rs])
    save_table(df, "scale_sweep_amap_draws")
    # summarise per draw index across cases: median Dice of the 45 cases for each (design, draw)
    summ = df.groupby(["variant", "design", "k"]).agg(
        median_delta=("delta", "median"), p10_delta=("delta", lambda x: x.quantile(.1)),
        frac_below_minus_0p05=("delta", lambda x: float((x < -0.05).mean())),
        frac_no_selection=("n_sel", lambda x: float((x == 0).mean())), median_dice=("dice", "median")).reset_index()
    summ["cpu_hours_per_brain"] = summ.k * MIN_PER_REG / 60
    save_table(summ, "scale_sweep_amap")
    return summ


# ------------------------------------------------------------------------------------------------------------
# C. tail risk across a cohort (aMAP)
# ------------------------------------------------------------------------------------------------------------
def tail_risk():
    w = load_table("amap_benefit_cases")
    runs = load_table("amap_runs")
    w = w[w.brain.isin(QC_BRAINS) & w.structure.isin(STRUCTS_MAIN)]
    rng = np.random.default_rng(SEED)
    rows = []
    for v in ("raw", "preprocessed"):
        ww = w[w.variant == v]
        r = runs[(runs.variant == v) & runs.brain.isin(QC_BRAINS) & runs.structure.isin(STRUCTS_MAIN) & runs.passes_global_qc]
        methods = {"consensus": ww.cons_dice.to_numpy(), "default": ww.default_dice.to_numpy(),
                   "top-SSIM run": ww.top_ssim_dice.to_numpy(), "LOBO parameter": ww.lobo_top1_dice.to_numpy()}
        # a random QC-passing run per case (resampled)
        grp = [g.dice_paper.to_numpy() for _, g in r.groupby(["brain", "structure"])]
        for name, x in methods.items():
            x = x[~np.isnan(x)]
            es = np.sort(x)[: max(1, int(np.ceil(0.1 * len(x))))].mean()
            cohort = {N: np.mean([rng.choice(x, N).min() for _ in range(4000)]) for N in (10, 50, 100)}
            rows.append(dict(variant=v, method=name, median=np.median(x), expected_shortfall_10pct=es,
                             frac_below_0p6=(x < 0.6).mean(), frac_below_0p7=(x < 0.7).mean(),
                             **{f"expected_worst_of_{N}": val for N, val in cohort.items()}))
        sims = np.array([[rng.choice(g) for g in grp] for _ in range(2000)])
        es = np.mean([np.sort(s)[:5].mean() for s in sims])
        cohort = {N: np.mean([rng.choice(sims[i % 2000], N).min() for i in range(4000)]) for N in (10, 50, 100)}
        rows.append(dict(variant=v, method="random QC-passing run", median=np.median(sims), expected_shortfall_10pct=es,
                         frac_below_0p6=(sims < 0.6).mean(), frac_below_0p7=(sims < 0.7).mean(),
                         **{f"expected_worst_of_{N}": val for N, val in cohort.items()}))
    df = pd.DataFrame(rows)
    save_table(df, "scale_tail_risk")
    return df


if __name__ == "__main__":
    t0 = time.time()
    a, b = selection_freedom()
    log(a[a["sample"] == "ALL"].round(3).to_string())
    log(b.groupby(["status", "accepted_set"])[["consensus_diff", "extreme_selection_hi", "extreme_selection_lo",
                                                  "random_selection_sd"]].describe().round(3).T.to_string())
    tr = tail_risk()
    log(tr.round(3).to_string())
    m7 = _load("m7", "07_lsfm_analysis.py")
    sl = sweep_lsfm(m7)
    log(sl.round(3).to_string())
    sa = sweep_amap()
    log(sa.round(3).to_string())
    record("scale_selection", a[a["sample"] == "ALL"].to_dict("records"))
    record("scale_tail", tr.to_dict("records"))
    log(f"done ({time.time() - t0:.0f}s)")
