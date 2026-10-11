"""A3-A10 engine: per-run and per-consensus scores for every aMAP case, recomputed from the cache (A1).

Pass 1 (runs):  every sweep run restricted to its own registered hemisphere (the annotated label), scored with the
                paper protocol (best coronal slice in the raters' slab), the fixed STAPLE slice A0 and wider windows.
Pass 2 (sets):  consensus = mean of a run subset >= threshold, restricted to the subset's hemisphere vote, for
                * the pipeline's production sets (all / global QC / final = global QC + SSIM top 50 %) at many
                  thresholds;
                * a QC-threshold grid (IoU x grad_p95 x SSIM fraction) at threshold 0.5;
                * deterministic choices and small ensembles: brainreg default, top-SSIM run, LOBO-selected parameter
                  and LOBO top-k ensemble (chosen on the other brains), in-sample best parameter and in-sample
                  "stiff family" ensemble (both labelled optimistic).
Also: GT-free and GT-based per-case predictors (A13), calibration / reliability of the final-set probability (A8).
Verification: recomputed run and production-consensus Dice must equal the earlier compare tables.

Outputs (results/tables/): amap_runs, amap_sets, amap_qcgrid, amap_predictors, amap_calib, amap_reliability,
amap_verification, amap_selections.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import (AMAP_OUT, BRAINS, CFG, DEFAULT_RUN, PIPE, QC_BRAINS, STRUCTS, STRUCTS_MAIN, C, binary_entropy,
                    bwperim8, case_path, load_case, log, run_params, save_table, seg_metrics)

A = CFG["amap"]
THR_FINE = [float(t) for t in A["thresholds_fine"]]
PIPE_THR = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
WINDOWS = {"fixed": None, "paper": 0, "w5": 5, "w10": 10}


# -------------------------------------------------------------------------------------------------------------
# QC table (per run: IoU, grad_p95, per-region SSIM)
# -------------------------------------------------------------------------------------------------------------
def load_qc() -> pd.DataFrame:
    frames = []
    for f in ("qc_results.csv", "qc_results_preprocessed.csv"):
        q = pd.read_csv(PIPE / "qc_tables" / f)
        # the preprocessed table names SSIM columns local_ssim__<ACR>_both, the raw one local_ssim__<ACR>
        q = q.rename(columns={c: c[:-5] for c in q.columns if c.startswith("local_ssim__") and c.endswith("_both")})
        frames.append(q)
    q = pd.concat(frames, ignore_index=True)
    q["run"] = q.registration_dir.str.rstrip("/").str.split("/").str[-1]
    inv = {v: k for k, v in C.SAMPLE_PREFIX.items()}
    q["brain"] = q.sample_id.map(lambda s: next(inv[p] for p in inv if s.startswith(p)))
    q["variant"] = q.input_type
    dup = q.duplicated(["brain", "variant", "run"]).sum()
    q = q.drop_duplicates(["brain", "variant", "run"], keep="first")
    q.attrs["n_duplicates_dropped"] = int(dup)
    return q


def qc_subset(qrows: pd.DataFrame, structure: str, iou_t, grad_t, ssim_frac):
    """Runs passing IoU > iou_t and grad_p95 < grad_t, then SSIM >= quantile(1 - f) among those (pipeline rule)."""
    g = qrows[(qrows.iou > iou_t) & (qrows.grad_p95 < grad_t)]
    if ssim_frac >= 1.0 or len(g) == 0:
        return list(g.run)
    s = g[f"local_ssim__{structure}"]
    return list(g.run[s >= s.quantile(1 - ssim_frac)])


# -------------------------------------------------------------------------------------------------------------
# scoring helpers on one cached case
# -------------------------------------------------------------------------------------------------------------
class Case:
    def __init__(self, brain, structure, variant):
        d = load_case(brain, structure, variant)
        self.__dict__.update(d)
        self.brain, self.structure, self.variant = brain, structure, variant
        self.idx = {r: i for i, r in enumerate(self.runs)}
        self.hl = self.h1 if self.label == 1 else self.h2          # runs' own annotated-side maps
        self.ho = self.h2 if self.label == 1 else self.h1
        self.i0 = self.A_list.index(self.A0)
        s0, s1 = int(self.slab[0]), int(self.slab[1])
        lo, hi = min(s0, self.A0), max(s1, self.A0)
        self.win = {"fixed": [self.i0]}
        for name, extra in WINDOWS.items():
            if extra is not None:
                self.win[name] = [i for i, a in enumerate(self.A_list) if lo - extra <= a <= hi + extra]
        self.slab_idx = self.win["paper"]
        self.tarea = self.truth.sum()

    def slice_dice(self, stack):
        """stack (nA, H, W) -> Dice per slice vs truth."""
        inter = (stack & self.truth).sum((1, 2))
        den = stack.sum((1, 2)) + self.tarea
        return np.where(den > 0, 2 * inter / np.maximum(den, 1), np.nan)

    def score(self, stack, full=True):
        d = self.slice_dice(stack)
        out = {}
        for name, ids in self.win.items():
            dd = d[ids]
            if np.all(np.isnan(dd)):
                out[f"dice_{name}"] = np.nan
                continue
            k = ids[int(np.nanargmax(dd))]
            out[f"dice_{name}"] = float(d[k])
            if name == "paper":
                out["A_best"] = self.A_list[k]
                out["at_window_edge"] = bool(k in (self.slab_idx[0], self.slab_idx[-1]))
                if full:
                    m = seg_metrics(stack[k], self.truth)
                    out.update(hd_paper_um=m["hd_um"], hd95_paper_um=m["hd95_um"], msd_paper_um=m["msd_um"],
                               area_ratio_paper=m["area_ratio"])
            if name == "fixed" and full:
                m = seg_metrics(stack[k], self.truth)
                out.update(hd_fixed_um=m["hd_um"], hd95_fixed_um=m["hd95_um"], msd_fixed_um=m["msd_um"],
                           area_ratio_fixed=m["area_ratio"])
        out["slab_volume_px"] = int(stack[self.slab_idx].sum())
        return out

    def run_mask(self, run):
        k = self.idx[run]
        return self.masks[k] & self.hl[k]

    def probability(self, runs):
        ids = [self.idx[r] for r in runs if r in self.idx]
        cnt = self.masks[ids].sum(0)
        c1 = self.hl[ids].sum(0)
        c2 = self.ho[ids].sum(0)
        keep = c1 > c2                                             # vote == annotated label
        return cnt, len(ids), keep

    def consensus(self, runs, thr):
        cnt, n, keep = self.probability(runs)
        if n == 0:
            return None
        return (cnt >= thr * n - 1e-6) & keep


def dice3d(a, b):
    s = a.sum() + b.sum()
    return 2 * (a & b).sum() / s if s else np.nan


# -------------------------------------------------------------------------------------------------------------
# pass 1
# -------------------------------------------------------------------------------------------------------------
def pass1(job):
    brain, structure, variant, qrows = job
    cs = Case(brain, structure, variant)
    rows = []
    fin = qc_subset(qrows, structure, A["iou_threshold"], A["grad_p95_threshold"], A["ssim_top_fraction"])
    glob = qc_subset(qrows, structure, A["iou_threshold"], A["grad_p95_threshold"], 1.0)
    ref_runs = fin if fin else cs.runs
    ref = cs.consensus(ref_runs, 0.5)[cs.slab_idx]
    q = qrows.set_index("run")
    for run in cs.runs:
        m = cs.run_mask(run)
        be, g = run_params(run)
        r = dict(brain=brain, structure=structure, variant=variant, run=run, bending_energy=be, grid=g,
                 iou=q.loc[run, "iou"] if run in q.index else np.nan,
                 grad_p95=q.loc[run, "grad_p95"] if run in q.index else np.nan,
                 ssim=q.loc[run, f"local_ssim__{structure}"] if run in q.index else np.nan,
                 passes_global_qc=run in glob, passes_final_qc=run in fin,
                 dice_to_ref_consensus_3d=dice3d(m[cs.slab_idx], ref))
        r.update(cs.score(m))
        rows.append(r)
    return rows


# -------------------------------------------------------------------------------------------------------------
# pass 2
# -------------------------------------------------------------------------------------------------------------
def pass2(job):
    brain, structure, variant, qrows, named = job
    cs = Case(brain, structure, variant)
    base = dict(brain=brain, structure=structure, variant=variant)
    sets, grid_rows, pred, calib, rel = [], [], [], [], []

    def add(kind, name, runs, thresholds, full=True, **extra):
        runs = [r for r in runs if r in cs.idx]
        for t in thresholds:
            m = cs.consensus(runs, t)
            row = dict(base, set_kind=kind, set_name=name, threshold=t, n_runs=len(runs), **extra)
            if m is not None:
                row.update(cs.score(m, full=full))
            sets.append(row)

    prod = {"all_registrations": cs.runs,
            "passes_global_qc": qc_subset(qrows, structure, A["iou_threshold"], A["grad_p95_threshold"], 1.0),
            "passes_final_qc_top_0p5": qc_subset(qrows, structure, A["iou_threshold"], A["grad_p95_threshold"],
                                                 A["ssim_top_fraction"])}
    for name, runs in prod.items():
        if runs:
            add("production", name, runs, sorted(set(THR_FINE) | set(PIPE_THR)))
    for name, runs in named.items():
        if runs:
            add("choice", name, runs, [0.5])
    # QC-threshold grid (threshold 0.5), Dice only
    for it in A["iou_grid"]:
        for gt in A["grad_grid"]:
            for sf in A["ssim_grid"]:
                runs = qc_subset(qrows, structure, it, gt, sf)
                row = dict(base, iou_thr=it, grad_thr=gt, ssim_frac=sf, n_runs=len(runs))
                m = cs.consensus(runs, 0.5) if runs else None
                if m is not None:
                    s = cs.score(m, full=False)
                    row.update(dice_paper=s["dice_paper"], dice_fixed=s["dice_fixed"])
                grid_rows.append(row)

    # predictors (A13) on the primary set; ER (no QC passers) uses all registrations
    fin = prod["passes_final_qc_top_0p5"] or cs.runs
    glob = prod["passes_global_qc"] or cs.runs
    cnt, n, keep = cs.probability(fin)
    p = np.where(keep, cnt / n, 0.0)[cs.slab_idx]
    cons = p >= 0.5 - 1e-9
    band = (p > 0) & (p < 1)
    per = sum(bwperim8(c).sum() for c in cons)
    run_m = {r: cs.run_mask(r)[cs.slab_idx] for r in cs.runs}
    vols = np.array([run_m[r].sum() for r in cs.runs], float)
    fin_masks = [run_m[r] for r in fin]
    pair = [dice3d(fin_masks[i], fin_masks[j]) for i in range(len(fin_masks)) for j in range(i + 1, len(fin_masks))]
    d_all = np.array([dice3d(run_m[r], cons) for r in cs.runs])
    d_glob = np.array([dice3d(run_m[r], cons) for r in glob])
    pred.append(dict(base, n_final=len(prod["passes_final_qc_top_0p5"]), n_global=len(prod["passes_global_qc"]),
                     gf_volume_cv_all=float(vols.std() / vols.mean()) if vols.mean() else np.nan,
                     gf_pairwise_dice_final=float(np.nanmean(pair)) if pair else np.nan,
                     gf_band_fraction=float(band.sum() / max(cons.sum(), 1)),
                     gf_band_width_um=float(band.sum() / max(per, 1)) * 25.0,
                     gf_entropy=float(binary_entropy(p[p > 0]).mean()) if (p > 0).any() else np.nan,
                     gf_frac_runs_disagree=float((d_all < A["gt_free_agree"]).mean()),
                     gf_sd_dice_to_cons_global=float(np.nanstd(d_glob)),
                     truth_area_um2=float(cs.tarea) * 625.0,
                     human_p_band_frac=float(((cs.human_p > 0) & (cs.human_p < 1)).sum() / max(cs.tarea, 1))))

    # calibration of the primary-set probability on its paper-protocol best slice
    m = cs.consensus(fin, 0.5)
    k = cs.slab_idx[int(np.nanargmax(cs.slice_dice(m)[cs.slab_idx]))]
    pk = np.where(keep[k], cnt[k] / n, 0.0)
    calib.append(dict(base, A=cs.A_list[k], set_name="passes_final_qc_top_0p5" if prod["passes_final_qc_top_0p5"]
                      else "all_registrations", **C._calibration(pk, cs.human_p, cs.truth)))
    rel += [dict(base, **r) for r in C._reliability(pk, cs.truth)]
    return sets, grid_rows, pred, calib, rel


# -------------------------------------------------------------------------------------------------------------
# selections that depend on other brains (LOBO) or on all cases (in-sample)
# -------------------------------------------------------------------------------------------------------------
def selections(runs: pd.DataFrame, k_ens: int) -> dict:
    """{(brain, variant): {name: [runs]}} with LOBO / in-sample / default / top-SSIM choices."""
    out, info = {}, []
    main = runs[runs.structure.isin(STRUCTS_MAIN)]
    for variant in C.VARIANTS:
        v = main[main.variant == variant]
        pooled = v[v.brain.isin(QC_BRAINS)].groupby("run").dice_paper.mean().sort_values(ascending=False)
        stiff = [r for r in pooled.index if run_params(r)[0] >= 0.99 and run_params(r)[1] in (-25, -30)]
        for brain in BRAINS:
            train = v[v.brain.isin([b for b in QC_BRAINS if b != brain])]
            rank = train.groupby("run").dice_paper.mean().sort_values(ascending=False)
            sel = {"default": [DEFAULT_RUN],
                   "lobo_top1": [rank.index[0]],
                   f"lobo_top{k_ens}_ensemble": list(rank.index[:k_ens]),
                   "insample_best_param": [pooled.index[0]],
                   "insample_stiff_family": stiff}
            out[(brain, variant)] = sel
            info.append(dict(brain=brain, variant=variant, lobo_top1=rank.index[0],
                             lobo_topk=" ".join(rank.index[:k_ens]), insample_best=pooled.index[0],
                             stiff_family=" ".join(stiff)))
    return out, pd.DataFrame(info)


def verify(runs, sets):
    """Recomputed Dice vs the earlier compare tables (registered hemispheres)."""
    t = AMAP_OUT / "pipeline_comparison_registered_hemispheres" / "tables"
    old_runs = pd.read_csv(t / "runs.csv")
    old_runs = old_runs[old_runs.qc_set == "all_registrations"].copy()
    old_runs["structure"] = old_runs.region.map(C.PAPER_NAME)
    a = old_runs.merge(runs, on=["brain", "structure", "variant", "run"], suffixes=("_old", ""))
    old_main = pd.read_csv(t / "main.csv")
    old_main["structure"] = old_main.region.map(C.PAPER_NAME)
    s = sets[sets.set_kind == "production"].rename(columns={"set_name": "qc_set"})
    b = old_main.merge(s, on=["brain", "structure", "variant", "qc_set", "threshold"], suffixes=("_old", ""))
    rows = []
    for name, df in (("runs", a), ("production consensus", b)):
        for col_old, col_new in (("dice_paper_old", "dice_paper"), ("dice_fixed_old", "dice_fixed")):
            diff = (df[col_old] - df[col_new]).abs()
            rows.append(dict(table=name, metric=col_new, n_compared=len(df), max_abs_diff=float(diff.max()),
                             n_diff_gt_1e9=int((diff > 1e-9).sum()), n_diff_gt_0p01=int((diff > 0.01).sum())))
    return pd.DataFrame(rows), b


if __name__ == "__main__":
    t0 = time.time()
    qc = load_qc()
    log(f"QC rows {len(qc)}; duplicate (brain, variant, run) rows dropped: {qc.attrs['n_duplicates_dropped']}")
    # check the pipeline's passes_global_qc flag is reproduced by IoU > 0.9 and grad_p95 < 0.045
    rep = (qc.iou > A["iou_threshold"]) & (qc.grad_p95 < A["grad_p95_threshold"])
    log(f"passes_global_qc reproduced for {(rep == qc.passes_global_qc.astype(bool)).mean():.1%} of rows")
    cases = [(b, s, v) for b in BRAINS for s in STRUCTS for v in C.VARIANTS if case_path(b, s, v).exists()]
    log(f"{len(cases)} cached cases")
    qrows = {(b, v): qc[(qc.brain == b) & (qc.variant == v)] for b in BRAINS for v in C.VARIANTS}
    with ProcessPoolExecutor(max_workers=8) as ex:
        runs = pd.DataFrame([r for rs in ex.map(pass1, [(b, s, v, qrows[(b, v)]) for b, s, v in cases]) for r in rs])
    save_table(runs, "amap_runs")
    log(f"pass 1: {len(runs)} run rows ({time.time() - t0:.0f}s)")
    sel, sel_info = selections(runs, int(A["lobo_ensemble_k"]))
    save_table(sel_info, "amap_selections")
    # top-SSIM single run among global-QC passers (per case)
    jobs = []
    for b, s, v in cases:
        named = dict(sel[(b, v)])
        r = runs[(runs.brain == b) & (runs.structure == s) & (runs.variant == v) & runs.passes_global_qc]
        if len(r):
            named["top_ssim"] = [r.sort_values("ssim", ascending=False).run.iloc[0]]
        jobs.append((b, s, v, qrows[(b, v)], named))
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(pass2, jobs))
    names = ["amap_sets", "amap_qcgrid", "amap_predictors", "amap_calib", "amap_reliability"]
    out = {n: pd.DataFrame([r for x in res for r in x[i]]) for i, n in enumerate(names)}
    for n, df in out.items():
        save_table(df, n)
    ver, _ = verify(runs, out["amap_sets"])
    save_table(ver, "amap_verification")
    log(ver.to_string())
    log(f"done ({time.time() - t0:.0f}s)")
