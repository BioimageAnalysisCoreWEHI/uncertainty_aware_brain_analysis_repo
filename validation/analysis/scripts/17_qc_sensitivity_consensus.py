"""Sensitivity of the final consensus regions to the global-QC thresholds (tissue IoU, grad_p95).

For every threshold pair on a grid spanning each dataset's full range ("everything passes" to "nothing passes"), plus
the named settings (chosen, untuned starting point, Tukey, all registrations):
  global QC recomputed (IoU > t and grad_p95 < t) -> top 50 % by SSIM among the passers -> probability map and
  consensus regions from the pipeline's per-registration masks -> comparison of all 0.5 consensus regions with each
  other and, for aMAP, with the expert STAPLE.

Datasets
  amap  - 6 brains x raw/preprocessed x 11 regions; masks in results/region_consensus/.../all_registrations/
          individual_masks; SSIM rule = pipeline region_utils.apply_ssim_ranking (>= quantile, per region).
  paper - 5 LSFM brains x raw/preprocessed, SNpc; masks in results_{raw,preprocessed}/param_sweep_masks_*_no_padding/
          snpc; full QC table dataset_qc_comparison/original_lsfm_full_final_results_with_ssim.csv; SSIM rule =
          07_lsfm_analysis.select (top ceil(n/2) by ssim_value), which reproduces the published counts.

Stages
  prepare   (local, seconds): settings.csv, qc_tables/<setting>/*.csv (recomputed pass columns), distinct run sets,
            index.csv, one task JSON per case and jobs.csv for the HPC array; prints distinct sets and a disk estimate.
  (HPC)     scripts/hpc/submit_qcsens.sh runs qcsens_worker.py per case: writes the maps and 3-D metrics.
  aggregate (local): merges worker metrics; aMAP STAPLE scoring on the validation cache (Case); paper SNpc counts
            (Counter); tables, figures and verification checks.

  lsfm2 - independent LSFM (6 brains x raw/preprocessed x 16 regions), same layout and SSIM rule as aMAP; every set's
          0.5 consensus is also scored against the published registration (worker: dice_vs_ref, volume_ratio_vs_ref,
          hd95_vs_ref_um).

Usage: python 17_qc_sensitivity_consensus.py prepare|aggregate|relabel --dataset amap|paper|lsfm2
Outputs: results/qc_sensitivity/<dataset>/..., results/tables/qcsens_<dataset>_*.csv,
         figures/supplementary/FigS_qc_sensitivity_<dataset>.{pdf,png}, source_data/FigS_qc_sensitivity_<dataset>_*.csv
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import (C, CFG, FIG_SUPP, PIPE, SRC, STRUCTS, STRUCTS_MAIN, VA, amap_brain, load_case, log,  # noqa: E402
                    save_table)
from qc_start import start_thresholds  # noqa: E402

sys.path.insert(0, str(Path(CFG["paths"]["pipeline_config"]).parents[1] / "scripts"))
from region_utils import apply_ssim_ranking  # noqa: E402  (pipeline's own SSIM rule, read-only import)


def _load(name, fname):
    s = importlib.util.spec_from_file_location(name, str(VA / "scripts" / fname))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


OUTROOT = VA / "results" / "qc_sensitivity"
THRESHOLDS = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
SSIM_FRAC = 0.5
DS = {
    "amap": dict(iou=[None, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.94, 0.95, 0.96],
                 grad=[0.040, 0.043, 0.045, 0.047, 0.050, 0.053, 0.055, 0.058, 0.060, 0.065, 0.070, 0.080, 0.090,
                       0.110, None],
                 chosen=(0.9, 0.058), plausible=((0.80, 0.95), (0.045, 0.070)), voxel_um=25.0, style="amap"),
    "paper": dict(iou=[None, 0.80, 0.82, 0.84, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.97],
                  grad=[0.030, 0.033, 0.035, 0.038, 0.040, 0.042, 0.045, 0.048, 0.050, 0.055, 0.060, 0.070, 0.080,
                        0.095, None],
                  chosen=(0.9, 0.045), plausible=((0.85, 0.95), (0.035, 0.060)), voxel_um=20.0, style="paper"),
    # independent LSFM (ScienceDB VTA projectome): 6 brains x raw/preprocessed x 16 regions; same pipeline code and
    # layout as aMAP; observed raw range IoU ~0.38-0.92, grad_p95 ~0.043-0.13
    "lsfm2": dict(iou=[None, 0.40, 0.50, 0.60, 0.70, 0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.87, 0.88, 0.90, 0.92],
                  grad=[0.045, 0.050, 0.055, 0.060, 0.063, 0.066, 0.070, 0.075, 0.080, 0.085, 0.090, 0.100, 0.110,
                        0.130, None],
                  chosen=(0.8, 0.08), plausible=((0.75, 0.88), (0.060, 0.100)), voxel_um=25.0, style="amap"),
}
LS = VA.parent / "validation_datasets" / "lightsheet_data"
LS_PIPE = LS / "results"
PIPES = {"amap": PIPE, "lsfm2": LS_PIPE}
PAPER_TABLE = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\dataset_qc_comparison\original_lsfm_full_final_results_with_ssim.csv"))
VARIANTS = ["raw", "preprocessed"]


def posix(p) -> str:
    """Path as the cluster sees it (job tables are read by the HPC workers)."""
    return to_cluster(p)


def fmt(v):
    return "none" if v is None else f"{v:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def sid_of(runs):
    return hashlib.sha1(";".join(sorted(runs)).encode()).hexdigest()[:12] if runs else "empty"


def axes(ds, raw_iou, raw_grad):
    """Grid axes: the fixed values plus, from the raw sweep, the median and the median -/+ 3 MAD starting point."""
    cfg = DS[ds]
    st = start_thresholds(raw_iou, raw_grad)
    med = (round(float(np.median(raw_iou)), 4), round(float(np.median(raw_grad)), 4))
    iou = [None] + sorted({v for v in cfg["iou"] if v is not None} | {st[0], med[0]})
    grad = sorted({v for v in cfg["grad"] if v is not None} | {st[1], med[1]}) + [None]
    return iou, grad, st, med


def settings_table(ds, raw_iou, raw_grad):
    cfg = DS[ds]
    ious, grads, st, med = axes(ds, raw_iou, raw_grad)
    rows = [dict(setting_id=f"iou{fmt(i)}_grad{fmt(g)}", iou_thr=i, grad_thr=g, ssim="top50", in_grid=True)
            for i in ious for g in grads]
    tk = _load("qc12", "12_qc_threshold_procedure.py").tukey(pd.DataFrame(dict(iou=raw_iou, grad_p95=raw_grad)))
    rows.append(dict(setting_id="tukey", iou_thr=tk[0], grad_thr=tk[1], ssim="top50", in_grid=False))
    rows.append(dict(setting_id="all_registrations", iou_thr=None, grad_thr=None, ssim="all", in_grid=False))
    s = pd.DataFrame(rows)
    s["label"] = ""
    at = lambda i, g: s.in_grid & np.isclose(s.iou_thr, i) & np.isclose(s.grad_thr, g)
    s.loc[at(*cfg["chosen"]), "label"] = "chosen"
    s.loc[at(*st), "label"] = "starting point (median ± 3·MAD)"
    s.loc[at(*med), "label"] = "median"
    s.loc[s.iou_thr.isna() & s.grad_thr.isna() & s.in_grid, "label"] = "no global QC"
    s.loc[s.setting_id == "tukey", "label"] = "Tukey 1.5·IQR"
    s.loc[s.setting_id == "all_registrations", "label"] = "all registrations (no QC, no SSIM)"
    s["axis_iou_start"], s["axis_grad_start"], s["axis_iou_median"], s["axis_grad_median"] = st[0], st[1], med[0], med[1]
    (pi0, pi1), (pg0, pg1) = cfg["plausible"]
    s["plausible"] = s.in_grid & s.iou_thr.between(pi0, pi1) & s.grad_thr.between(pg0, pg1)
    return s


def thr(v, none):
    return none if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


# ------------------------------------------------------------------------------------------------------------
# selection per dataset: -> long table (setting_id, case key..., runs) and per-setting QC tables
# ------------------------------------------------------------------------------------------------------------
def amap_selection(settings, out, pipe=PIPE):
    """aMAP-style pipeline output (also used for the independent LSFM dataset, pipe=LS_PIPE)."""
    tabs = {"raw": pd.read_csv(pipe / "qc_tables" / "qc_results.csv"),
            "preprocessed": pd.read_csv(pipe / "qc_tables" / "qc_results_preprocessed.csv")}
    for t in tabs.values():
        t["run"] = t.registration_dir.astype(str).str.rstrip("/").str.split("/").str[-1]
    suffixes = [c[len("local_ssim__"):] for c in tabs["raw"].columns if c.startswith("local_ssim__")]
    rows = []
    for s in settings.itertuples():
        ti, tg = thr(s.iou_thr, -np.inf), thr(s.grad_thr, np.inf)
        d = out / "qc_tables" / s.setting_id
        d.mkdir(parents=True, exist_ok=True)
        for v, t in tabs.items():
            q = t.copy()
            q["passes_global_qc"] = (q.iou > ti) & (q.grad_p95 < tg)
            if s.ssim == "all":
                q["passes_global_qc"] = True
                for sf in suffixes:
                    q[f"passes_local_ssim_qc__{sf}"] = True
            elif not q.passes_global_qc.any():  # nothing passes: apply_ssim_ranking cannot assign an empty result
                for sf in suffixes:
                    q[f"ssim_rank__{sf}"], q[f"ssim_percentile__{sf}"] = np.nan, np.nan
                    q[f"passes_local_ssim_qc__{sf}"] = False
            else:
                q = apply_ssim_ranking(q, SSIM_FRAC, "passes_global_qc", group_cols=("sample_id", "input_type"))
            for sf in suffixes:
                q[f"passes_final_qc__{sf}"] = q.passes_global_qc & q[f"passes_local_ssim_qc__{sf}"].astype(bool)
            q.drop(columns="run").to_csv(d / ("qc_results.csv" if v == "raw" else "qc_results_preprocessed.csv"),
                                         index=False)
            for (smp, it), g in q.groupby(["sample_id", "input_type"]):
                n_glob = int(g.passes_global_qc.sum())
                for sf in suffixes:
                    runs = sorted(g.run[g[f"passes_final_qc__{sf}"]])
                    rows.append(dict(setting_id=s.setting_id, sample=smp, variant=it, region=sf, n_global=n_glob,
                                     n_final=len(runs), runs=runs))
    sel = pd.DataFrame(rows)
    masks = {}
    for (smp, v, sf), _ in sel.groupby(["sample", "variant", "region"]):
        d = pipe / "region_consensus" / smp / v / sf / "all_registrations" / "individual_masks"
        masks[(smp, v, sf)] = {r: posix(d / f"mask__{r}.tif") for r in sorted(tabs[v][tabs[v].sample_id == smp].run)}
    return sel, masks, ["sample", "variant", "region"]


def paper_selection(settings, out):
    m7 = _load("m7", "07_lsfm_analysis.py")
    df = pd.read_csv(PAPER_TABLE)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")].copy()
    df["Sample ID"] = df["Sample ID"].astype(str)
    df["run_key"] = df.bending.map(lambda b: f"{b:g}") + "_" + df.grid.map(lambda g: f"{g:g}")   # as 06_lsfm_cache
    long = {}
    for st in VARIANTS:
        cols = {c: c[len(st) + 1:] for c in df.columns if c.startswith(st + "_")}
        long[st] = df[["Sample ID", "bending", "grid", "run_key"] + list(cols)].rename(columns=cols).rename(
            columns={"Sample ID": "sample"})
    rows = []
    for s in settings.itertuples():
        ti, tg = thr(s.iou_thr, -np.inf), thr(s.grad_thr, np.inf)
        d = out / "qc_tables" / s.setting_id
        d.mkdir(parents=True, exist_ok=True)
        for st, t in long.items():
            q = t[["sample", "bending", "grid", "run_key", "iou", "grad_p95", "ssim_value", "snpc_mask_path"]].copy()
            q["passes_global_qc"] = True if s.ssim == "all" else (q.iou > ti) & (q.grad_p95 < tg)
            q["passes_final_qc"] = False
            for smp, g in q.groupby("sample"):
                runs = list(g.run_key) if s.ssim == "all" else m7.select(g, "passed_qc", "top_50pct", ti, tg)
                q.loc[g.index[g.run_key.isin(runs)], "passes_final_qc"] = True
                rows.append(dict(setting_id=s.setting_id, sample=smp, variant=st, n_global=int(g.passes_global_qc.sum()),
                                 n_final=len(runs), runs=sorted(runs)))
            q.to_csv(d / ("qc_results.csv" if st == "raw" else "qc_results_preprocessed.csv"), index=False)
    sel = pd.DataFrame(rows)
    masks = {(smp, st): {r.run_key: posix(r.snpc_mask_path) for r in long[st][long[st]["sample"] == smp].itertuples()}
             for st in VARIANTS for smp in long[st]["sample"].unique()}
    return sel, masks, ["sample", "variant"]


def write_reference_masks(out, sel):
    """lsfm2: the published ("independent") registration's mask for every sample x region, on the consensus grid,
    built with the validation code itself (gt_utils, as for Fig. 8). Written once; returns {(sample, region): posix}."""
    sys.path.insert(0, str(LS / "scripts" / "validation"))
    sys.path.insert(1, str(LS / "scripts"))
    gtu = importlib.util.module_from_spec(s := importlib.util.spec_from_file_location(
        "gt_utils", str(LS / "scripts" / "validation" / "gt_utils.py")))
    s.loader.exec_module(gtu)
    import tifffile
    _, agg = gtu.load_label_mapping()
    paths = {}
    for smp in sorted(sel["sample"].unique()):
        gt = None
        for region in sorted(sel.region.unique()):
            f = out / "reference_masks" / smp / f"{region}.tif"
            if not f.exists():
                gt = gtu.load_gt_atlas(smp) if gt is None else gt
                f.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(f, gtu.gt_region_mask(gt, region.replace("_both", ""), agg).astype(np.uint8),
                                 compression="zlib")
            paths[(smp, region)] = posix(f)
    log(f"lsfm2: {len(paths)} reference masks in {out / 'reference_masks'}")
    return paths


# ------------------------------------------------------------------------------------------------------------
def prepare(ds):
    out = OUTROOT / ds
    out.mkdir(parents=True, exist_ok=True)
    if ds in PIPES:
        q = pd.read_csv(PIPES[ds] / "qc_tables" / "qc_results.csv")
        settings = settings_table(ds, q.iou, q.grad_p95)
    else:
        t = pd.read_csv(PAPER_TABLE)
        settings = settings_table(ds, t.raw_iou, t.raw_grad_p95)
    settings.to_csv(out / "settings.csv", index=False)
    if ds in PIPES:
        sel, masks, keys = amap_selection(settings, out, PIPES[ds])
    else:
        sel, masks, keys = paper_selection(settings, out)
    ref_masks = write_reference_masks(out, sel) if ds == "lsfm2" else {}
    sel["set_id"] = sel.runs.map(sid_of)
    chosen_id = settings.setting_id[settings.label == "chosen"].iloc[0]
    noqc_id = settings.setting_id[settings.label == "no global QC"].iloc[0]
    grid_ids = set(settings.setting_id[settings.in_grid])
    plaus_ids = set(settings.setting_id[settings.plausible])
    jobs, index = [], []
    jdir = out / "tasks"
    jdir.mkdir(exist_ok=True)
    for task, (key, s_all) in enumerate(sel.groupby(keys, sort=True)):
        key = key if isinstance(key, tuple) else (key,)
        g = s_all[s_all.set_id != "empty"]
        sets = {sid: runs for sid, runs in zip(g.set_id, g.runs)}
        set_settings = g.groupby("set_id").setting_id.apply(list).to_dict()
        label = "/".join(map(str, key))
        if ds in PIPES:
            smp, v, region = key
            out_dir = out / "region_consensus" / smp / v / region / "sets"
            ref = PIPES[ds] / "region_consensus" / smp / v / region / "passes_final_qc_top_0p5"
            extra = dict(reference_dir=posix(ref), prefix="")
            if ds == "lsfm2":  # the published registration's region mask, scored per set by the worker
                extra["ref_mask"] = ref_masks[(smp, region)]
        else:
            smp, v = key
            out_dir = out / "consensus" / smp / v / "sets"
            extra = dict(reference_dir=None, prefix=smp)
        pick = lambda ids: sorted({r.set_id for r in s_all.itertuples() if r.setting_id in ids and r.set_id != "empty"})
        chosen = s_all.set_id[s_all.setting_id == chosen_id].iloc[0]
        noqc = s_all.set_id[s_all.setting_id == noqc_id].iloc[0]
        job = dict(task=task, root=posix(out), case_label=label, style=DS[ds]["style"], thresholds=THRESHOLDS,
                   voxel_um=DS[ds]["voxel_um"], masks=masks[key], sets=sets, set_settings=set_settings,
                   chosen=None if chosen == "empty" else chosen, noqc=None if noqc == "empty" else noqc,
                   grid_sets=pick(grid_ids), plausible_sets=pick(plaus_ids), out_dir=posix(out_dir), **extra)
        jp = jdir / f"{task}.json"
        jp.write_text(json.dumps(job))
        jobs.append(dict(task=task, case_label=label, job_json=posix(jp), n_sets=len(sets)))
        for r in s_all.itertuples():
            index.append(dict(task=task, **{k: getattr(r, k) for k in keys}, setting_id=r.setting_id,
                              set_id=r.set_id, n_global=r.n_global, n_final=r.n_final))
    jobs = pd.DataFrame(jobs)
    jobs.to_csv(out / "jobs.csv", index=False)
    pd.DataFrame(index).to_csv(out / "index.csv", index=False)
    sel.assign(runs=sel.runs.map(";".join)).to_csv(out / "selection.csv", index=False)
    # ---- dry-run report ----
    n_sets = int(jobs.n_sets.sum())
    if ds in PIPES:
        ref = next((PIPES[ds] / "region_consensus").glob("*/raw/ACA_both/passes_final_qc_top_0p5"))
        per_set = sum(f.stat().st_size for f in ref.glob("*.tif"))
    else:
        per_set = 3.0e6  # rough: zlib full-size float32 probability + 6 uint8 maps of a small region
    need = n_sets * per_set
    free = shutil.disk_usage(OUTROOT.anchor or ".").free
    n_missing = sum(not Path(to_local(p)).exists() for m in masks.values() for p in m.values())
    log(f"{ds}: {len(settings)} settings, {len(jobs)} cases, {n_sets} distinct non-empty sets "
        f"(median {jobs.n_sets.median():.0f} per case, max {jobs.n_sets.max()}); "
        f"estimated {need / 1e9:.1f} GB for maps; V: free {free / 1e9:.0f} GB; missing mask files: {n_missing}")
    if ds in PIPES:  # chosen-setting run sets vs the pipeline's own passes_final_qc_top_0p5 sets
        bad = 0
        for j in jobs.itertuples():
            job = json.loads(Path(to_local(j.job_json)).read_text())
            u = pd.read_csv(Path(to_local(job["reference_dir"])) / "registrations_used.csv")
            ref_runs = sorted(u.registration_dir.astype(str).str.rstrip("/").str.split("/").str[-1])
            bad += ref_runs != sorted(job["sets"].get(job["chosen"], []))
        log(f"{ds}: chosen-setting run sets differing from the pipeline's passes_final_qc_top_0p5: {bad}/{len(jobs)}")
    else:
        t = pd.read_csv(PAPER_TABLE)
        for st in VARIANTS:
            rep = ((t[f"{st}_iou"] > 0.9) & (t[f"{st}_grad_p95"] < 0.045)) == t[f"{st}_pass_final"].astype(bool)
            log(f"paper {st}: table pass_final reproduced by 0.9/0.045: {rep.mean():.0%}")
    return settings, jobs


# ------------------------------------------------------------------------------------------------------------
def aggregate(ds):
    out = OUTROOT / ds
    settings = pd.read_csv(out / "settings.csv")
    jobs = pd.read_csv(out / "jobs.csv")
    index = pd.read_csv(out / "index.csv")
    missing = [t for t in jobs.task if not (out / "done" / f"{t}.done").exists()]
    if missing and "--partial" in sys.argv:  # testing only: aggregate the finished tasks
        log(f"PARTIAL: {len(missing)} of {len(jobs)} tasks not finished - results are incomplete")
        jobs = jobs[~jobs.task.isin(missing)]
        index = index[index.task.isin(jobs.task)]
    elif missing:
        sys.exit(f"{len(missing)} tasks not finished, e.g. {missing[:10]} - resubmit with --array="
                 + ",".join(map(str, missing)))
    ms = pd.concat([pd.read_csv(out / "metrics" / f"{t}_sets.csv") for t in jobs.task], ignore_index=True)
    mc = pd.concat([pd.read_csv(out / "metrics" / f"{t}_case.csv") for t in jobs.task], ignore_index=True)
    per = index.merge(ms.drop(columns="n"), on=["task", "set_id"], how="left").merge(
        settings[["setting_id", "iou_thr", "grad_thr", "in_grid", "plausible", "label"]], on="setting_id")
    sel = pd.read_csv(out / "selection.csv")
    runs_of = {r.set_id: (r.runs.split(";") if isinstance(r.runs, str) and r.runs else [])
               for r in sel.drop_duplicates("set_id").itertuples()}
    if ds == "lsfm2":  # scored against the published registration by the worker (dice_vs_ref, ...); no figure here
        per["structure"] = per.region.str.replace("_both", "", regex=False)
        save_table(per, f"qcsens_{ds}_per_case")
        save_table(mc, f"qcsens_{ds}_stability")
        g = per[per.n_final > 0].groupby(["setting_id", "variant"])
        summ = g.agg(median_dice_vs_ref=("dice_vs_ref", "median"), median_dice3d_vs_chosen=("dice_vs_chosen", "median"),
                     n_cases=("dice_vs_ref", "size")).reset_index()
        snc = per[(per.structure == "SNc") & (per.n_final > 0)].groupby(["setting_id", "variant"]).dice_vs_ref.median()
        summ = summ.merge(snc.rename("median_snc_dice_vs_ref").reset_index(), on=["setting_id", "variant"], how="left")
        empty = per[per.n_final == 0].groupby(["setting_id", "variant"]).size().rename("n_cases_empty").reset_index()
        summ = summ.merge(empty, on=["setting_id", "variant"], how="left").fillna({"n_cases_empty": 0}).merge(
            settings[["setting_id", "iou_thr", "grad_thr", "label"]], on="setting_id")
        save_table(summ, f"qcsens_{ds}_grid_summary")
        log(summ[summ.label.fillna("") != ""].round(3).to_string(index=False))
        log(mc.filter(regex="ref_|unstable|pairwise").describe().round(4).to_string())
        return
    if ds == "amap":
        per, checks = amap_staple(per, runs_of, out)
    else:
        per, checks = paper_counts(per, runs_of)
    save_table(per, f"qcsens_{ds}_per_case")
    save_table(mc, f"qcsens_{ds}_stability")
    summ = summarise(per, ds)
    save_table(summ, f"qcsens_{ds}_grid_summary")
    log(checks)
    log(mc.filter(regex="ref_|unstable|pairwise").describe().round(4).to_string())
    figure(per, summ, mc, settings, ds)


def amap_staple(per, runs_of, out):
    m3 = _load("m3", "03_amap_metrics.py")
    per = per.copy()
    per["brain"] = per["sample"].map(amap_brain)
    per["structure"] = per.region.str.replace("_both", "", regex=False)
    scores = {}
    check = []
    for (b, s, v), g in per[per.structure.isin(STRUCTS)].groupby(["brain", "structure", "variant"]):
        cs = m3.Case(b, s, v)
        for sid in g.set_id.unique():
            runs = runs_of.get(sid, [])
            cons = cs.consensus(runs, 0.5) if runs else None
            scores[(b, s, v, sid)] = cs.score(cons, full=False) if cons is not None else {}
        # cache vs new volume at A0 for the chosen set (no hemisphere restriction on either side)
        ch = g.set_id[g.label == "chosen"].iloc[0]
        if runs_of.get(ch):
            import tifffile
            smp = g["sample"].iloc[0]
            f = out / "region_consensus" / smp / v / f"{s}_both" / "sets" / ch / "consensus_thresh_0p5.tif"
            if (out / "set_folders.csv").exists():  # after relabel: readable folder names
                fm = pd.read_csv(out / "set_folders.csv")
                f = out / fm.set_path[(fm.task == g.task.iloc[0]) & (fm.set_id == ch)].iloc[0] / "consensus_thresh_0p5.tif"
            with tifffile.TiffFile(f) as t:
                page = t.pages[cs.A0].asarray() > 0
            z = load_case(b, s, v)
            s0, s1, r0, r1 = z["bbox"]
            ours = C.to_our_grid(page, tuple(z["full_shape"]))[s0:s1, r0:r1]
            cnt, n, _ = cs.probability(runs_of[ch])
            cache = cnt[cs.i0] >= 0.5 * n - 1e-6
            check.append(float((ours == cache).mean()))
    for k in ("dice_paper", "dice_fixed"):
        per[k] = [scores.get((r.brain, r.structure, r.variant, r.set_id), {}).get(k, np.nan) for r in per.itertuples()]
    msg = (f"cache vs new volume at A0 (chosen set): {len(check)} cases, pixel agreement min {min(check):.4f}, "
           f"identical in {sum(c == 1 for c in check)}")
    ch = per[(per.label == "chosen") & per.structure.isin(STRUCTS_MAIN)]
    msg += (f"\nchosen setting: median dice_paper raw {ch[ch.variant == 'raw'].dice_paper.median():.3f} / pre "
            f"{ch[ch.variant == 'preprocessed'].dice_paper.median():.3f}; dice_fixed raw "
            f"{ch[ch.variant == 'raw'].dice_fixed.median():.3f}")
    return per, msg


def paper_counts(per, runs_of):
    m7 = _load("m7", "07_lsfm_analysis.py")
    per = per.copy()
    cnt = {}
    for (smp, v), g in per.groupby(["sample", "variant"]):
        c = m7.Counter(str(smp), v)
        for sid in g.set_id.unique():
            cnt[(smp, v, sid)] = c.count(runs_of[sid], 0.5) if runs_of.get(sid) else np.nan
    per["count"] = [cnt[(r.sample, r.variant, r.set_id)] for r in per.itertuples()]
    ref = per[per.label == "chosen"].set_index(["sample", "variant"])["count"]
    per["count_change_pct"] = [100 * (r.count / ref[(r.sample, r.variant)] - 1) for r in per.itertuples()]
    msg = "chosen setting SNpc counts: " + ", ".join(f"{k[0]} {k[1]} {v:.0f}" for k, v in ref.items())
    return per, msg


def summarise(per, ds):
    rows = []
    main = per[per.structure.isin(STRUCTS_MAIN)] if ds == "amap" else per
    for (sid, v), g in main.groupby(["setting_id", "variant"]):
        r = dict(setting_id=sid, variant=v, iou_thr=g.iou_thr.iloc[0], grad_thr=g.grad_thr.iloc[0],
                 label=g.label.iloc[0], in_grid=g.in_grid.iloc[0], n_cases=len(g), n_empty=int((g.n_final == 0).sum()),
                 median_n_final=g.n_final.median(), median_dice3d_vs_chosen=g.dice_vs_chosen.median(),
                 min_dice3d_vs_chosen=g.dice_vs_chosen.min(), median_hd95_vs_chosen_um=g.hd95_vs_chosen_um.median(),
                 median_volume_ratio_vs_chosen=g.volume_ratio_vs_chosen.median())
        if ds == "amap":
            r.update(median_dice_paper=g.dice_paper.median(), q1_dice_paper=g.dice_paper.quantile(0.25),
                     q3_dice_paper=g.dice_paper.quantile(0.75), min_dice_paper=g.dice_paper.min(),
                     median_dice_fixed=g.dice_fixed.median())
        else:
            ch = per[(per.label == "chosen") & (per.variant == v)].set_index("sample")["count"]
            cur = g.set_index("sample")["count"].reindex(ch.index)
            ok = cur.notna()
            r.update(median_abs_count_change_pct=g.count_change_pct.abs().median(),
                     max_abs_count_change_pct=g.count_change_pct.abs().max(),
                     kendall_tau_vs_chosen=stats.kendalltau(ch[ok], cur[ok]).statistic if ok.sum() >= 3 else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------
def figure(per, summ, mc, settings, ds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    grid_s = settings[settings.in_grid.astype(bool)]
    ious = [None] + sorted(grid_s.iou_thr.dropna().unique())
    grads = sorted(grid_s.grad_thr.dropna().unique()) + [None]
    lab = lambda v: "none" if v is None else f"{v:g}"
    key = lambda v: -1.0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
    maps = ([("median_dice_paper", "median Dice vs expert STAPLE", "viridis", (0.5, 0.9))] if ds == "amap" else
            [("median_abs_count_change_pct", "median |Δ SNpc count| vs chosen (%)", "magma_r", (0, 40))])
    maps.append(("median_dice3d_vs_chosen", "median 3-D Dice to chosen consensus", "viridis", (0.5, 1.0)))
    fig = plt.figure(figsize=(7.2, 8.6))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], hspace=0.55, wspace=0.35)
    src = []
    g_s = summ[summ.in_grid.astype(bool)]
    for i, (col, title, cmap, lim) in enumerate(maps):
        for j, v in enumerate(VARIANTS):
            ax = fig.add_subplot(gs[i, j])
            s = g_s[g_s.variant == v]
            M = np.full((len(ious), len(grads)), np.nan)
            E = np.zeros_like(M)
            for r in s.itertuples():
                a = [key(x) for x in ious].index(key(r.iou_thr))
                b = [key(x) for x in grads].index(key(r.grad_thr))
                M[a, b], E[a, b] = getattr(r, col), r.n_empty
            im = ax.imshow(M, cmap=cmap, vmin=lim[0], vmax=lim[1], origin="lower", aspect="auto")
            for a in range(M.shape[0]):
                for b in range(M.shape[1]):
                    if E[a, b] > 0:
                        ax.text(b, a, f"{int(E[a, b])}", ha="center", va="center", fontsize=4.5, color="w")
                        ax.add_patch(plt.Rectangle((b - .5, a - .5), 1, 1, fill=False, hatch="////", lw=0, ec="0.8"))
            # chosen = red box; starting point = white circle; median = white triangle (grid cells); Tukey = white
            # square at its interpolated position (not on the grid)
            for labl, mk in (("chosen", None), ("starting point (median ± 3·MAD)", "o"), ("median", "^"),
                             ("Tukey 1.5·IQR", "s")):
                r = settings[settings.label == labl].iloc[0]
                num = lambda arr: np.array([x for x in arr if x is not None], float)
                ii = np.interp(r.iou_thr, num(ious), np.arange(len(ious))[[x is not None for x in ious]])
                gg = np.interp(r.grad_thr, num(grads), np.arange(len(grads))[[x is not None for x in grads]])
                if mk is None:
                    ax.add_patch(plt.Rectangle((gg - .5, ii - .5), 1, 1, fill=False, ec="r", lw=1.3))
                else:
                    ax.plot(gg, ii, mk, mfc="none", mec="w", ms=4.5, mew=1.1)
            ax.set_xticks(range(len(grads)), [lab(x) for x in grads], rotation=90, fontsize=5)
            ax.set_yticks(range(len(ious)), [lab(x) for x in ious], fontsize=5)
            ax.set_xlabel("grad_p95 <")
            ax.set_ylabel("IoU >")
            ax.set_title(f"{title}\n({v})", fontsize=6.5)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            if j == 0:
                ax.text(-0.3, 1.12, "ab"[i], transform=ax.transAxes, fontsize=9, fontweight="bold")
            src.append(s.assign(panel="ab"[i], metric=col))
    # (c) per structure / brain spread across grid settings; (d) unstable fraction
    ax = fig.add_subplot(gs[2, 0])
    grid = per[per.in_grid.astype(bool)]
    if ds == "amap":
        g = grid[grid.structure.isin(STRUCTS_MAIN)]
        ycol, ylab, grp = "dice_paper", "Dice vs STAPLE (all grid settings)", "structure"
    else:
        g = grid
        ycol, ylab, grp = "count", "SNpc consensus count (all grid settings)", "sample"
    cats = sorted(g[grp].astype(str).unique())
    for k, c in enumerate(cats):
        for dv, v in ((-0.15, "raw"), (0.15, "preprocessed")):
            y = g[(g[grp].astype(str) == c) & (g.variant == v)][ycol].dropna()
            ax.plot(np.full(len(y), k + dv) + np.random.default_rng(0).uniform(-.06, .06, len(y)), y, ".",
                    ms=1.5, color="#0072B2" if v == "raw" else "#56B4E9", alpha=0.5)
            chv = per[(per.label == "chosen") & (per[grp].astype(str) == c) & (per.variant == v)][ycol]
            if ds == "amap":
                chv = per[(per.label == "chosen") & (per.structure == c) & (per.variant == v)][ycol].median()
            else:
                chv = chv.iloc[0] if len(chv) else np.nan
            ax.plot(k + dv, chv, "_", color="r", ms=8, mew=1.5)
    ax.set_xticks(range(len(cats)), cats, rotation=45, fontsize=6)
    ax.set_ylabel(ylab)
    ax.set_title("red = chosen" + (" (median over brains)" if ds == "amap" else ""), fontsize=6.5)
    ax.text(-0.3, 1.12, "c", transform=ax.transAxes, fontsize=9, fontweight="bold")
    ax = fig.add_subplot(gs[2, 1])
    data = [mc.unstable_fraction_plausible.dropna(), mc.unstable_fraction_grid.dropna()]
    ax.boxplot(data, widths=0.5, showfliers=True, flierprops=dict(ms=2))
    ax.set_xticks([1, 2], ["plausible range", "full grid"])
    ax.set_ylabel("unstable fraction of the 0.5 consensus\n(union − intersection) / union")
    ax.set_ylim(0, 1)
    ax.text(-0.3, 1.12, "d", transform=ax.transAxes, fontsize=9, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(FIG_SUPP / f"FigS_qc_sensitivity_{ds}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    pd.concat(src).to_csv(SRC / f"FigS_qc_sensitivity_{ds}_ab_heatmaps.csv", index=False)
    g.to_csv(SRC / f"FigS_qc_sensitivity_{ds}_c_spread.csv", index=False)
    mc.to_csv(SRC / f"FigS_qc_sensitivity_{ds}_d_stability.csv", index=False)
    log(f"figure FigS_qc_sensitivity_{ds}")


# ------------------------------------------------------------------------------------------------------------
TAGS = {"chosen": "chosen", "starting point (median ± 3·MAD)": "start", "median": "median",
        "no global QC": "no-global-QC", "all registrations (no QC, no SSIM)": "all-registrations",
        "Tukey 1.5·IQR": "tukey"}


def relabel(ds):
    """Rename the hash-named set folders to human-readable names and write setting_lookup.csv.

    Name = n<registrations>__<named settings producing this set, if any>__iou<t>_grad<t>, where the thresholds are
    those of the first named grid setting or else the least strict grid setting producing the set (lowest IoU, then
    highest grad). Every folder keeps settings.txt with all settings that give exactly this region. Idempotent:
    the mapping is stored in set_folders.csv and reused."""
    out = OUTROOT / ds
    settings = pd.read_csv(out / "settings.csv")
    index = pd.read_csv(out / "index.csv")
    jobs = pd.read_csv(out / "jobs.csv")
    mp_file = out / "set_folders.csv"
    old = pd.read_csv(mp_file) if mp_file.exists() else pd.DataFrame(columns=["task", "set_id", "set_folder"])
    st = settings.set_index("setting_id")
    tf = lambda v, p: "none" if pd.isna(v) else f"{v:.4f}".rstrip("0").rstrip(".")
    maps = []
    if ds == "amap":  # long pipeline sample folders -> brain names (readable, and keeps paths under MAX_PATH)
        rc = out / "region_consensus"
        for d in list(rc.iterdir()):
            if d.is_dir() and d.name not in C.SAMPLE_PREFIX:
                target = rc / amap_brain(d.name)
                if not target.exists():
                    d.rename(target)
    for j in jobs.itertuples():
        job = json.loads(Path(to_local(j.job_json)).read_text())
        base = Path(to_local(job["out_dir"]))
        if ds == "amap":
            smp = base.parents[2].name
            base = out / "region_consensus" / amap_brain(smp) / base.parents[1].name / base.parent.name / "sets"
        for sid, runs in job["sets"].items():
            ids = job["set_settings"][sid]
            lab = [TAGS[st.label[i]] for i in ids if isinstance(st.label[i], str) and st.label[i] in TAGS]
            order = ["chosen", "start", "median", "tukey", "no-global-QC", "all-registrations"]
            lab = sorted(set(lab), key=order.index)
            grid = [i for i in ids if st.in_grid[i]]
            named_grid = [i for i in grid if isinstance(st.label[i], str) and st.label[i] in TAGS]
            if named_grid:
                rep = named_grid[0]
            elif "tukey" in lab:
                rep = "tukey"
            elif grid:
                rep = sorted(grid, key=lambda i: (-np.inf if pd.isna(st.iou_thr[i]) else st.iou_thr[i],
                                                  -(np.inf if pd.isna(st.grad_thr[i]) else st.grad_thr[i])))[0]
            else:
                rep = ids[0]
            thr_txt = f"iou{tf(st.iou_thr[rep], 'iou')}_grad{tf(st.grad_thr[rep], 'grad')}"
            name = f"n{len(runs):02d}__" + ("+".join(lab) + "__" if lab else "") + thr_txt
            if lab and lab[0] in ("no-global-QC", "all-registrations") and len(lab) == 1:
                name = f"n{len(runs):02d}__{lab[0]}"
            prev = old[(old.task == j.task) & (old.set_id == sid)]
            cur = base / (prev.set_folder.iloc[0] if len(prev) else sid)
            dest = base / name
            if cur.exists() and cur != dest:
                cur.rename(dest)
            maps.append(dict(task=j.task, set_id=sid, set_folder=name,
                             set_path=str(dest.relative_to(out)).replace("\\", "/")))
    mp = pd.DataFrame(maps)
    mp.to_csv(mp_file, index=False)
    lk = index.merge(mp, on=["task", "set_id"], how="left").merge(
        settings[["setting_id", "iou_thr", "grad_thr", "label"]], on="setting_id")
    if ds == "amap":
        lk.insert(1, "brain", lk["sample"].map(amap_brain))
    lk["note"] = np.where(lk.n_final == 0, "no registration passes this setting", "")
    lk.to_csv(out / "setting_lookup.csv", index=False)
    for t in ("per_case",):
        p = VA / "results" / "tables" / f"qcsens_{ds}_{t}.csv"
        if p.exists():
            d = pd.read_csv(p).drop(columns=["set_folder", "set_path"], errors="ignore")
            d.merge(mp, on=["task", "set_id"], how="left").to_csv(p, index=False)
    log(f"{ds}: {len(mp)} set folders named; lookup {len(lk)} rows -> {out / 'setting_lookup.csv'}")


if __name__ == "__main__":
    stages = {"prepare": prepare, "aggregate": aggregate, "relabel": relabel}
    if len(sys.argv) < 4 or sys.argv[1] not in stages or "--dataset" not in sys.argv:
        sys.exit(__doc__)
    ds = sys.argv[sys.argv.index("--dataset") + 1]
    stages[sys.argv[1]](ds)
