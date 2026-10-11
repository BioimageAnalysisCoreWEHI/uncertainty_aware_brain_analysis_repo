"""Worker for the global-QC sensitivity analysis (17_qc_sensitivity_consensus.py): one task = one case.

A case is one brain x input x region (aMAP) or one brain x input (paper LSFM, SNpc). The task JSON (written by
`17_qc_sensitivity_consensus.py prepare`) lists the case's per-registration masks and every distinct set of
registrations selected by some global-QC setting. The worker:
  1. reads each mask once (all masks of a case share one full-size grid) and keeps it cropped to the union bounding box;
  2. for each distinct set: probability = mean of the selected masks (float32); consensus = probability >= t for each
     threshold; writes full-size zlib TIFFs in the same grid as the source masks (skipped if already written);
  3. computes 3-D stability metrics of the 0.5 consensus: Dice / volume ratio / HD95 vs the chosen setting's set,
     Dice vs the no-global-QC set, mean |probability difference| vs chosen, and per case the unstable fraction
     ((union - intersection) / union over sets) and pairwise Dice among sets;
  4. for aMAP, compares the chosen set's maps with the pipeline's own passes_final_qc_top_0p5 outputs.
Writes <root>/metrics/<task>_sets.csv, <task>_case.csv and <root>/done/<task>.done.

Standalone (numpy, pandas, tifffile, scipy) so it runs in the HPC env. Paths in the JSON are POSIX <IMAGING>/...;
on Windows they are mapped to <IMAGING>\\ automatically.

Usage: python qcsens_worker.py --job-table <jobs.csv> --task N [--threads 4]
       python qcsens_worker.py --job-table <jobs.csv> --all [--threads 4]      (local fallback, sequential tasks)
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

WIN = os.name == "nt"


def local(p: str) -> Path:
    """Stored (cluster or placeholder) path -> path on this machine."""
    return Path(to_local(p))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------------------
def read_masks(paths: dict, threads: int):
    """-> runs, stack (n, z, y, x) bool cropped to the union bbox, offset (z0, y0, x0), full shape."""
    runs = list(paths)

    def one(r):
        m = tifffile.imread(local(paths[r])) > 0
        nz = [np.flatnonzero(m.any(axis=tuple(j for j in range(3) if j != i))) for i in range(3)]
        if any(len(a) == 0 for a in nz):
            return r, m.shape, None, None
        lo = [int(a[0]) for a in nz]
        hi = [int(a[-1]) + 1 for a in nz]
        return r, m.shape, lo, m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy()

    with ThreadPoolExecutor(max_workers=threads) as ex:
        res = {r: (shape, lo, crop) for r, shape, lo, crop in ex.map(one, runs)}
    shapes = {v[0] for v in res.values()}
    if len(shapes) != 1:
        raise ValueError(f"masks of one case have different shapes: {shapes}")
    full = shapes.pop()
    los = [v[1] for v in res.values() if v[1] is not None]
    if not los:  # every mask empty
        return runs, np.zeros((len(runs), 1, 1, 1), bool), (0, 0, 0), full
    lo = np.min(los, 0)
    hi = np.max([np.add(v[1], v[2].shape) for v in res.values() if v[1] is not None], 0)
    stack = np.zeros((len(runs),) + tuple(hi - lo), bool)
    for k, r in enumerate(runs):
        shape, l0, crop = res[r]
        if crop is not None:
            o = np.subtract(l0, lo)
            stack[k, o[0]:o[0] + crop.shape[0], o[1]:o[1] + crop.shape[1], o[2]:o[2] + crop.shape[2]] = crop
    return runs, stack, tuple(int(v) for v in lo), full


def surface(a):
    return a & ~ndi.binary_erosion(a, border_value=0)


def hd95(a, b, vox_um):
    """Symmetric 95th-percentile surface distance (um) between two 3-D masks."""
    if not a.any() or not b.any():
        return np.nan
    pad = [(1, 1)] * 3
    a, b = np.pad(a, pad), np.pad(b, pad)
    sa, sb = surface(a), surface(b)
    da = ndi.distance_transform_edt(~sb)[sa]
    db = ndi.distance_transform_edt(~sa)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95) * vox_um)


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return 2 * int((a & b).sum()) / s if s else np.nan


def thr_name(t, style):
    if style == "paper":
        return f"{t:.2f}"
    s = f"{float(t):g}"
    return (s if "." in s else s + ".0").replace(".", "p")


# ------------------------------------------------------------------------------------------------------------
def run_task(job: dict, threads: int):
    t0 = time.time()
    root = local(job["root"])
    task = int(job["task"])
    done = root / "done" / f"{task}.done"
    if done.exists():
        log(f"task {task}: already done")
        return
    style, thresholds, vox = job["style"], [float(t) for t in job["thresholds"]], float(job["voxel_um"])
    runs, stack, off, full = read_masks(job["masks"], threads)
    col = {r: i for i, r in enumerate(runs)}
    log(f"task {task} {job['case_label']}: {len(runs)} masks read, crop {stack.shape[1:]} of {full} "
        f"({time.time() - t0:.0f}s)")
    sets = job["sets"]                                   # {set_id: [runs]}
    out_base = local(job["out_dir"])
    # reused full-size buffers (one per output file, so the 7 files of a set are compressed + written in parallel
    # threads; zlib releases the GIL); only the crop is rewritten, the rest stays zero
    fullf = np.zeros(full, np.float32)
    fullu = [np.zeros(full, np.uint8) for _ in thresholds]
    sl = tuple(slice(o, o + s) for o, s in zip(off, stack.shape[1:]))
    pool = ThreadPoolExecutor(max_workers=min(threads, len(thresholds) + 1))
    prob, cons05 = {}, {}
    n_written = 0
    for sid, rs in sets.items():
        ids = [col[r] for r in rs]
        if not ids:
            continue
        p = (stack[ids].sum(0, dtype=np.float64) / len(ids)).astype(np.float32)
        prob[sid] = p
        # aMAP: exactly the pipeline's expression (float32 array >= python float); paper: float64 comparison, as
        # get_consensus_counts.py / the Counter used for the published counts
        ge = (lambda t: p >= t) if style == "amap" else (lambda t: p.astype(np.float64) >= np.float64(t))
        cons05[sid] = ge(0.5)
        d = out_base / sid
        names = ([f"{job['prefix']}_prob_map_coronal.tif"] + [f"{job['prefix']}_consensus_{thr_name(t, style)}_coronal.tif"
                  for t in thresholds]) if style == "paper" else (
                 ["probability.tif"] + [f"consensus_thresh_{thr_name(t, style)}.tif" for t in thresholds])
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"run": rs}).to_csv(d / "registrations_used.csv", index=False)   # always refreshed (cheap)
        (d / "settings.txt").write_text("\n".join(job["set_settings"][sid]) + "\n")
        if all((d / n).exists() for n in names):
            continue
        fullf[sl] = p
        for buf, t in zip(fullu, thresholds):
            buf[sl] = ge(t)
        futs = [pool.submit(tifffile.imwrite, d / names[0], fullf, compression="zlib")]
        futs += [pool.submit(tifffile.imwrite, d / n, buf, compression="zlib") for n, buf in zip(names[1:], fullu)]
        for f in futs:
            f.result()
        n_written += 1
    pool.shutdown()
    del fullf, fullu
    log(f"task {task}: {len(prob)} sets with registrations, {n_written} written ({time.time() - t0:.0f}s)")

    # ---- 3-D stability metrics (cropped space) ----
    ch, nq = job.get("chosen"), job.get("noqc")
    rows = []
    for sid in sets:
        r = dict(set_id=sid, n=len(sets[sid]), volume_vox=int(cons05[sid].sum()) if sid in cons05 else 0)
        if sid in cons05 and ch in cons05:
            a, b = cons05[sid], cons05[ch]
            r.update(dice_vs_chosen=dice(a, b), volume_ratio_vs_chosen=a.sum() / b.sum() if b.sum() else np.nan,
                     hd95_vs_chosen_um=0.0 if sid == ch else hd95(a, b, vox))
            supp = (prob[sid] > 0) | (prob[ch] > 0)
            r["prob_mad_vs_chosen"] = float(np.abs(prob[sid] - prob[ch])[supp].mean()) if supp.any() else np.nan
        if sid in cons05 and nq in cons05:
            r["dice_vs_noqc"] = dice(cons05[sid], cons05[nq])
        rows.append(r)
    case = dict(task=task, case_label=job["case_label"], n_distinct_sets=len(sets), n_nonempty_sets=len(cons05),
                crop_shape="x".join(map(str, stack.shape[1:])), full_shape="x".join(map(str, full)))
    for name in ("grid", "plausible"):
        ids = [s for s in job[f"{name}_sets"] if s in cons05]
        if ids:
            u = np.logical_or.reduce([cons05[s] for s in ids])
            i = np.logical_and.reduce([cons05[s] for s in ids])
            case[f"unstable_fraction_{name}"] = float((u.sum() - i.sum()) / u.sum()) if u.sum() else np.nan
            pw = [dice(cons05[a], cons05[b]) for a, b in combinations(ids, 2)]
            case[f"pairwise_dice_median_{name}"] = float(np.nanmedian(pw)) if pw else np.nan
            case[f"pairwise_dice_min_{name}"] = float(np.nanmin(pw)) if pw else np.nan
            case[f"n_sets_{name}"] = len(ids)
    # ---- optional: every set's 0.5 consensus vs an external reference mask (independent LSFM dataset) ----
    if job.get("ref_mask"):
        ref_full = tifffile.imread(local(job["ref_mask"])) > 0
        if ref_full.shape != tuple(full):
            raise ValueError(f"reference mask shape {ref_full.shape} != mask grid {full}")
        lo_c = np.array(off)
        hi_c = lo_c + np.array(stack.shape[1:])
        if ref_full.any():
            nz = np.argwhere(ref_full)
            lo_b, hi_b = np.minimum(lo_c, nz.min(0)), np.maximum(hi_c, nz.max(0) + 1)
        else:
            lo_b, hi_b = lo_c, hi_c
        box = tuple(slice(a, b) for a, b in zip(lo_b, hi_b))
        R = ref_full[box]
        inner = tuple(slice(a - b0, a - b0 + s) for a, b0, s in zip(lo_c, lo_b, stack.shape[1:]))
        case["ref_mask_voxels"] = int(R.sum())
        for r in rows:
            if r["set_id"] in cons05:
                A = np.zeros(R.shape, bool)
                A[inner] = cons05[r["set_id"]]
                r.update(dice_vs_ref=dice(A, R), volume_ratio_vs_ref=A.sum() / R.sum() if R.sum() else np.nan,
                         hd95_vs_ref_um=hd95(A, R, vox))
    # ---- check against the pipeline's own output for the chosen setting ----
    ref = job.get("reference_dir")
    if ref and ch in prob:
        refd = local(ref)
        try:
            rp = tifffile.imread(refd / "probability.tif")
            case["ref_prob_max_abs_diff"] = float(np.abs(rp[sl] - prob[ch]).max())
            outside = rp.copy()
            outside[sl] = 0
            case["ref_prob_outside_crop_max"] = float(outside.max())
            rc = tifffile.imread(refd / "consensus_thresh_0p5.tif") > 0
            case["ref_cons05_identical"] = bool((rc[sl] == cons05[ch]).all() and not rc.sum() - rc[sl].sum())
            used = pd.read_csv(refd / "registrations_used.csv")
            ref_runs = sorted(used.registration_dir.astype(str).str.rstrip("/").str.split("/").str[-1])
            case["ref_runs_identical"] = ref_runs == sorted(sets[ch])
        except Exception as e:  # report, don't fail the task
            case["ref_check_error"] = str(e)
    mdir = root / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).assign(task=task).to_csv(mdir / f"{task}_sets.csv", index=False)
    pd.DataFrame([case]).to_csv(mdir / f"{task}_case.csv", index=False)
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {time.time() - t0:.0f}s\n")
    log(f"task {task}: done ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-table", required=True)
    ap.add_argument("--task", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    a = ap.parse_args()
    jobs = pd.read_csv(local(a.job_table))
    tasks = list(jobs.task) if a.all else [a.task]
    for t in tasks:
        job = json.loads(local(jobs.loc[jobs.task == t, "job_json"].iloc[0]).read_text())
        run_task(job, a.threads)


if __name__ == "__main__":
    sys.exit(main())
