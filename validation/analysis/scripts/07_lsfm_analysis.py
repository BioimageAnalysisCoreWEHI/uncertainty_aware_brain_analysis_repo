"""A14: SNpc TH+ counts in the manuscript's LSFM data -- stability, reproducibility and knob sensitivity of
consensus vs deterministic single-registration choices. No ground truth exists, so nothing here measures accuracy.

Counting rule for every method compared here (unless stated): the manuscript's overlap rule
(get_consensus_counts.py): a cell counts if ANY of its voxels is inside the region mask; consensus mask =
float32(mean of the retained masks) >= threshold. A single registration is the consensus of one mask.
Primary consensus = global QC (IoU > 0.9 & grad_p95 < 0.045) + SSIM top 50 % (ceil rule of get_subset), threshold 0.5.

Outputs: results/tables/lsfm_*.csv
"""
from __future__ import annotations

import importlib.util
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from common import CACHE, CFG, LSFM, log, record, save_table

spec = importlib.util.spec_from_file_location("lc", str(CACHE.parent / "scripts" / "06_lsfm_cache.py"))
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)

L = CFG["lsfm"]
THR = [float(t) for t in L["thresholds"]]
DEFAULT = f"{L['default_params'][0]:g}_{L['default_params'][1]:g}"
STATUSES = ["raw", "preprocessed"]


class Counter:
    """Exact overlap-rule counts for any subset of runs and any threshold, from the voxel-vote cache."""

    def __init__(self, sample, status):
        self.cell, self.votes, self.runs, self.n_cells = lc.load_votes(sample, status)
        self.col = {r: i for i, r in enumerate(self.runs)}
        # compress identical vote patterns (many voxels share one pattern) per cell
        key = np.packbits(self.votes, axis=1)
        comb = np.concatenate([self.cell[:, None].view(np.uint8).reshape(len(self.cell), -1), key], axis=1)
        u, inv = np.unique(comb, axis=0, return_index=True)
        self.cell_u = self.cell[inv]
        self.votes_u = self.votes[inv]

    def count(self, runs, thr):
        ids = [self.col[r] for r in runs]
        if not ids:
            return np.nan
        n = len(ids)
        # float32 probability compared with a float64 threshold, exactly as get_consensus_counts.py (a float32
        # comparison would, e.g., include 21/30 at threshold 0.7)
        frac = (self.votes_u[:, ids].sum(1) / n).astype(np.float32).astype(np.float64)
        inc = frac >= np.float64(thr)
        return int(np.unique(self.cell_u[inc]).size)


def ssim_top(sub: pd.DataFrame, frac):
    s = sub.dropna(subset=["ssim_value"]).sort_values("ssim_value", ascending=False)
    return s.iloc[: max(1, int(np.ceil(len(s) * frac)))] if len(s) else s


def select(sub, qc="passed_qc", ssim="top_50pct", iou_t=0.9, grad_t=0.045):
    g = sub
    if qc == "passed_qc":
        g = sub[(sub.iou > iou_t) & (sub.grad_p95 < grad_t)]
    if ssim != "all":
        g = ssim_top(g, {"top_25pct": 0.25, "top_50pct": 0.5}[ssim])
    return list(g.run_key)


def rel(a, b):
    return (a - b) / b if b else np.nan


if __name__ == "__main__":
    runs = pd.read_csv(CACHE / "lsfm" / "runs.csv")
    runs["sample"] = runs["sample"].astype(str)
    # production flag check: pass_final == IoU > 0.9 & grad_p95 < 0.045
    rep = (runs.iou > 0.9) & (runs.grad_p95 < 0.045)
    log(f"pass_final reproduced by thresholds: {(rep == runs.pass_final).mean():.1%}")
    samples = sorted(runs["sample"].unique())
    counters = {(s, st): Counter(s, st) for s in samples for st in STATUSES}
    log("counters ready")

    # ---- per-run counts (overlap rule) vs the table's per-run counts -----------------------------------------
    rows = []
    for (s, st), c in counters.items():
        sub = runs[(runs["sample"] == s) & (runs.status == st)]
        for r in sub.itertuples():
            rows.append(dict(sample=s, status=st, run_key=r.run_key, bending=r.bending, grid=r.grid, iou=r.iou,
                             grad_p95=r.grad_p95, pass_qc=bool(r.pass_final), ssim=r.ssim_value,
                             count_overlap=c.count([r.run_key], 0.5), count_table=r.snpc_CellCount,
                             volume_mm3=r.Volume_RegionSum_snpc_mm3))
    per_run = pd.DataFrame(rows)
    for (s, st), g in per_run.groupby(["sample", "status"]):
        top = set(select(runs[(runs["sample"] == s) & (runs.status == st)]))
        per_run.loc[g.index, "in_ssim_top50"] = g.run_key.isin(top)
    save_table(per_run, "lsfm_per_run")

    # ---- consensus counts: production combos x thresholds (verification vs the manuscript's table) -----------
    combos = [(q, sf) for q in ("passed_qc", "all_runs") for sf in ("top_25pct", "top_50pct", "all")]
    cons = []
    for (s, st), c in counters.items():
        sub = runs[(runs["sample"] == s) & (runs.status == st)]
        for q, sf in combos:
            sel = select(sub, q, sf)
            for t in THR:
                cons.append(dict(sample=s, status=st, qc_filter=q, ssim_filter=sf, threshold=t, n_runs=len(sel),
                                 count=c.count(sel, t)))
    cons = pd.DataFrame(cons)
    save_table(cons, "lsfm_consensus")
    old = pd.read_csv(LSFM / "final_count_results" / "consensus_threshold_cell_counts_overlap_logic" /
                      "ALL_consensus_threshold_cell_counts_summary.csv")
    old["sample"] = old.sample_id.astype(str)
    v = old.merge(cons, left_on=["sample", "processing_status", "qc_filter", "ssim_filter", "threshold"],
                  right_on=["sample", "status", "qc_filter", "ssim_filter", "threshold"])
    ver = dict(n_compared=len(v), n_exact=int((v.consensus_cell_count == v["count"]).sum()),
               max_abs_diff=float((v.consensus_cell_count - v["count"]).abs().max()),
               n_runs_match=int((v.n_masks_used == v.n_runs).sum()))
    save_table(pd.DataFrame([ver]), "lsfm_verification")
    log(f"verification vs manuscript consensus counts: {ver}")

    prim = cons[(cons.qc_filter == "passed_qc") & (cons.ssim_filter == "top_50pct") & (cons.threshold == 0.5)]
    prim = prim.set_index(["sample", "status"])["count"]

    # ---- variability by stage (replicates the manuscript's SD reduction) --------------------------------------
    var = []
    for (s, st), g in per_run.groupby(["sample", "status"]):
        for rule in ("count_table", "count_overlap"):
            stages = {"all": g, "global_qc": g[g.pass_qc], "ssim_top50": g[g.in_ssim_top50.astype(bool)]}
            sd = {k: x[rule].std(ddof=1) for k, x in stages.items()}
            for k, x in stages.items():
                var.append(dict(sample=s, status=st, rule=rule, stage=k, n=len(x), mean=x[rule].mean(), sd=sd[k],
                                cv=sd[k] / x[rule].mean(), iqr_rel=(x[rule].quantile(.75) - x[rule].quantile(.25))
                                / x[rule].median(), range_rel=(x[rule].max() - x[rule].min()) / x[rule].median(),
                                sd_reduction_vs_all=1 - sd[k] / sd["all"],
                                sd_reduction_vs_global=1 - sd[k] / sd["global_qc"]))
    var = pd.DataFrame(var)
    save_table(var, "lsfm_variability_by_stage")

    # ---- deterministic choices vs the primary consensus ---------------------------------------------------------
    all_pass = per_run.groupby("run_key").pass_qc.all()
    all_pass = sorted(all_pass[all_pass].index)
    comp = []
    for (s, st), c in counters.items():
        g = per_run[(per_run["sample"] == s) & (per_run.status == st)].set_index("run_key")
        P = prim[(s, st)]
        qc_runs = list(g.index[g.pass_qc])
        others = [(o, st) for o in samples if o != s]
        # GT-free LOBO rule: the parameter whose single count is closest (mean |rel. dev|) to the primary consensus
        # on the other brains, among parameters passing QC in all of them
        cand = [r for r in g.index if all(per_run.set_index(["sample", "status", "run_key"]).loc[(o, st2, r), "pass_qc"]
                                          for o, st2 in others)]
        dev = {r: np.mean([abs(rel(per_run[(per_run["sample"] == o) & (per_run.status == st2) &
                                           (per_run.run_key == r)].count_overlap.iloc[0], prim[(o, st2)]))
                           for o, st2 in others]) for r in cand}
        lobo = sorted(dev, key=dev.get)
        stiff = [r for r in g.index if g.loc[r, "bending"] >= 0.95 and g.loc[r, "grid"] in (-30, -50)]
        choices = {"default_0.95_-10": [DEFAULT], "lobo_central_param": lobo[:1],
                   "lobo_central_top4_ensemble": lobo[:4], "median_qc_run": None, "random_qc_run": None,
                   "stiff_family_BE>=0.95_grid-30/-50": stiff, "affine_equivalent_BE1.0": ["1_-10"],
                   "all_runs_consensus": list(g.index), "global_qc_consensus": qc_runs}
        for name, rs in choices.items():
            if name == "median_qc_run":
                val = float(np.median(g.loc[qc_runs, "count_overlap"]))
            elif name == "random_qc_run":
                val = float(np.mean(np.abs(g.loc[qc_runs, "count_overlap"] / P - 1)))  # expected |rel dev|
                comp.append(dict(sample=s, status=st, method=name, count=np.nan, rel_dev=np.nan, abs_rel_dev=val,
                                 n_runs=len(qc_runs), runs_pass_qc=True, primary=P))
                continue
            else:
                val = float(c.count(rs, 0.5))
            passes = bool(np.all(g.loc[rs, "pass_qc"])) if rs else True
            comp.append(dict(sample=s, status=st, method=name, count=val, rel_dev=rel(val, P),
                             abs_rel_dev=abs(rel(val, P)), n_runs=len(rs) if rs else len(qc_runs),
                             runs_pass_qc=passes, runs=" ".join(rs) if rs else "", primary=P))
        for r in all_pass:
            val = float(g.loc[r, "count_overlap"])
            comp.append(dict(sample=s, status=st, method=f"param_{r}", count=val, rel_dev=rel(val, P),
                             abs_rel_dev=abs(rel(val, P)), n_runs=1, runs_pass_qc=True, runs=r, primary=P))
    comp = pd.DataFrame(comp)
    save_table(comp, "lsfm_choices_vs_consensus")

    # ---- raw vs preprocessed reproducibility per method -----------------------------------------------------------
    rp = []
    methods = [m for m in comp.method.unique() if m != "random_qc_run"]
    for s in samples:
        for m in ["primary_consensus"] + methods:
            if m == "primary_consensus":
                a, b = prim[(s, "raw")], prim[(s, "preprocessed")]
            else:
                x = comp[(comp["sample"] == s) & (comp.method == m)].set_index("status")["count"]
                a, b = x.get("raw", np.nan), x.get("preprocessed", np.nan)
            rp.append(dict(sample=s, method=m, raw=a, preprocessed=b, abs_rel_diff=abs(a - b) / ((a + b) / 2)))
        # typical single QC run: all raw QC x preprocessed QC pairs of the same parameter
        g = per_run[per_run["sample"] == s].pivot_table(index="run_key", columns="status", values="count_overlap")
        q = per_run[per_run["sample"] == s].pivot_table(index="run_key", columns="status", values="pass_qc")
        both = g[(q.raw == 1) & (q.preprocessed == 1)]
        d = (both.raw - both.preprocessed).abs() / ((both.raw + both.preprocessed) / 2)
        rp.append(dict(sample=s, method="same_param_single_run_(median over params passing QC in both)",
                       raw=np.nan, preprocessed=np.nan, abs_rel_diff=float(d.median()), n_params=len(both)))
    rp = pd.DataFrame(rp)
    save_table(rp, "lsfm_raw_vs_preprocessed")

    # ---- knob sensitivity (all relative to the primary consensus count) ------------------------------------------
    kn = []
    for (s, st), c in counters.items():
        P = prim[(s, st)]
        g = per_run[(per_run["sample"] == s) & (per_run.status == st)]
        qcr = g[g.pass_qc].count_overlap
        cs = cons[(cons["sample"] == s) & (cons.status == st)]
        pc = cs[(cs.qc_filter == "passed_qc") & (cs.ssim_filter == "top_50pct")].set_index("threshold")["count"]
        at05 = cs[cs.threshold == 0.5]["count"]
        sub = runs[(runs["sample"] == s) & (runs.status == st)]
        grid_counts = []
        for it, gt in itertools.product(L["iou_grid"], L["grad_grid"]):
            sel = select(sub, "passed_qc", "top_50pct", it, gt)
            grid_counts.append(c.count(sel, 0.5) if sel else np.nan)
        grid_counts = np.array(grid_counts, float)
        kn += [dict(sample=s, status=st, knob="single QC-run choice (IQR)",
                    rel_spread=(qcr.quantile(.75) - qcr.quantile(.25)) / P),
               dict(sample=s, status=st, knob="single QC-run choice (range)", rel_spread=(qcr.max() - qcr.min()) / P),
               dict(sample=s, status=st, knob="consensus threshold 0.25-0.75",
                    rel_spread=(pc.loc[0.25] - pc.loc[0.75]) / P),
               dict(sample=s, status=st, knob="consensus threshold 0.4-0.6", rel_spread=(pc.loc[0.4] - pc.loc[0.6]) / P),
               dict(sample=s, status=st, knob="consensus QC/SSIM filter set (6 combos, t=0.5)",
                    rel_spread=(at05.max() - at05.min()) / P),
               dict(sample=s, status=st, knob="QC thresholds IoU 0.85-0.95 x grad 0.040-0.055 (t=0.5)",
                    rel_spread=(np.nanmax(grid_counts) - np.nanmin(grid_counts)) / P,
                    n_grid_empty=int(np.isnan(grid_counts).sum()))]
    kn = pd.DataFrame(kn)
    save_table(kn, "lsfm_knob_sensitivity")

    # ---- membership rule: centroid probabilities (manuscript Fig. 3I-J) vs overlap rule ---------------------------
    mem = []
    for folder, label in (("cell_region_probability_outputs", "qc_ssim_filtered"),
                          ("cell_region_probability_outputs_all_runs", "all_runs")):
        w = pd.read_csv(LSFM / "final_count_results" / folder / "ALL_cell_region_grouped_probabilities_wide.csv",
                        usecols=["sample_id", "processing_status", "cell_id", "SNc"], dtype={"sample_id": str})
        raw_ids = pd.read_csv(LSFM / "final_count_results" / folder / "ALL_cell_region_grouped_probabilities_wide.csv",
                              usecols=["sample_id"]).sample_id
        types = raw_ids.map(lambda x: type(x).__name__).value_counts().to_dict()
        for (s, st), g in w.groupby(["sample_id", "processing_status"]):
            dup = int(g.cell_id.duplicated().sum())
            mem.append(dict(source=label, sample=s, status=st, n_cells=len(g), n_duplicate_cell_ids=dup,
                            expected_count_sum_p=float(g.SNc.sum()), count_p_ge_0p5=int((g.SNc >= 0.5).sum()),
                            count_p_gt_0=int((g.SNc > 0).sum()), sample_id_types_in_file=str(types)))
    mem = pd.DataFrame(mem)
    mem["overlap_consensus_t0p5"] = [prim.get((s, st), np.nan) if src == "qc_ssim_filtered" else
                                     cons[(cons["sample"] == s) & (cons.status == st) & (cons.qc_filter == "all_runs")
                                          & (cons.ssim_filter == "all") & (cons.threshold == 0.5)]["count"].squeeze()
                                     for src, s, st in zip(mem.source, mem["sample"], mem.status)]
    save_table(mem, "lsfm_membership_rule")

    # ---- degenerate parameters and ranking stability --------------------------------------------------------------
    be1 = per_run[per_run.bending == 1.0].groupby(["sample", "status"]).agg(
        n=("count_overlap", "size"), n_unique_counts=("count_overlap", "nunique"),
        n_unique_table_counts=("count_table", "nunique"), frac_pass_qc=("pass_qc", "mean"),
        count=("count_overlap", "median")).reset_index()
    med = per_run.groupby(["sample", "status"]).count_overlap.median().rename("median_all_runs").reset_index()
    be1 = be1.merge(med, on=["sample", "status"])
    be1["ratio_to_median"] = be1["count"] / be1.median_all_runs
    save_table(be1, "lsfm_be1_affine")
    rk = []
    for st in STATUSES:
        ref = prim.xs(st, level="status").reindex(samples)
        for m in comp.method.unique():
            x = comp[(comp.status == st) & (comp.method == m)].set_index("sample")["count"].reindex(samples)
            if x.isna().any():
                continue
            tau = stats.kendalltau(ref, x).statistic
            rk.append(dict(status=st, method=m, kendall_tau_vs_primary=tau,
                           identical_order=bool(list(ref.sort_values().index) == list(x.sort_values().index))))
    rk = pd.DataFrame(rk)
    save_table(rk, "lsfm_ranking")

    # ---- key numbers ------------------------------------------------------------------------------------------------
    k = kn.groupby("knob").rel_spread.agg(["median", "min", "max"])
    record("lsfm_knobs", k.to_dict("index"))
    c2 = comp[comp.method.isin(["default_0.95_-10", "lobo_central_param", "lobo_central_top4_ensemble",
                                "random_qc_run", "median_qc_run", "stiff_family_BE>=0.95_grid-30/-50",
                                "affine_equivalent_BE1.0"])]
    record("lsfm_choice_abs_rel_dev", c2.groupby("method").abs_rel_dev.agg(["median", "min", "max"]).to_dict("index"))
    record("lsfm_raw_vs_pre", rp.groupby("method").abs_rel_diff.median().to_dict())
    record("lsfm_variability", var[var.rule == "count_table"].groupby("stage")[["cv", "sd_reduction_vs_all",
                                                                              "sd_reduction_vs_global"]].median()
           .to_dict("index"))
    log(k.to_string())
    log(c2.groupby("method").abs_rel_dev.describe().to_string())
    log(rp.groupby("method").abs_rel_diff.median().to_string())
