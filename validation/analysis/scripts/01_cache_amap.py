"""A1: cache every sweep run's structure mask and hemisphere labels over the raters' slab (+/- extra slices).

One compressed .npz per brain x structure x variant in cache/amap/<brain>/, cropped to the bounding box of
everything that is ever compared (all runs' masks in the window, STAPLE, rater probability, rater masks).
Run masks are the pipeline's own ``all_registrations/individual_masks`` (both hemispheres); hemisphere labels are
each run's ``registered_hemispheres.tiff`` pages. Any subset consensus / threshold / window can then be recomputed
exactly without touching the network share again.

Usage: python 01_cache_amap.py [brain ...]
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import AMAP_OUT, BRAINS, CACHE, CFG, PIPE, C, case_path, log

EXTRA = int(CFG["amap"]["window_extra_slices"])
MARGIN = 4


def build_brain(brain: str) -> list[dict]:
    from amap_validation import nifti
    from amap_validation.config import load_config
    from amap_validation.reslice import AsrGrid, AsrLayout
    import json

    cfg = load_config()
    lay = AsrLayout(AMAP_OUT / "userStapleResults" / "asr25um")
    slices = pd.read_csv(lay.slices_log)
    raters = pd.read_csv(lay.raters_log)
    ap = pd.read_csv(AMAP_OUT / "pipeline_comparison_registered_hemispheres" / "tables" / "ap.csv")
    h = nifti.read_header(cfg.brain_file(brain))
    grid = AsrGrid(h["shape"], h["sform"])
    nA = grid.shape25[0]
    out = []
    (CACHE / "amap" / brain).mkdir(parents=True, exist_ok=True)
    for srow in slices[slices.brain == brain].itertuples():
        t0 = time.time()
        region = srow.region
        structure = C.PAPER_NAME[region]
        truth = nifti.load(lay.dest / srow.staple_file)[0] > 0
        human_p = nifti.load(lay.dest / srow.probability_file)[0].astype(np.float32)
        target = truth.shape
        A0 = int(srow.asr25_slice_A)
        a = ap[(ap.brain == brain) & (ap.region == region)].iloc[0]
        slab0, slab1 = int(a.slab_A0), int(a.slab_A1)
        lo, hi = max(0, min(slab0, A0) - EXTRA), min(nA - 1, max(slab1, A0) + EXTRA)
        A_list = list(range(lo, hi + 1))
        rr = raters[(raters.brainRegionComb == srow.brainRegionComb) & (raters.variant == "medianPlane")]
        rmasks = np.stack([nifti.load(lay.dest / f)[0] > 0 for f in rr.file])
        for variant in C.VARIANTS:
            dest = case_path(brain, structure, variant)
            if dest.exists():
                out.append(dict(brain=brain, structure=structure, variant=variant, status="cached"))
                continue
            d_all = C.set_dir(PIPE, brain, variant, region, "all_registrations")
            run_dirs = C._run_dirs(d_all)
            files = sorted((d_all / "individual_masks").glob("mask__*.tif"))
            runs = [f.stem.replace("mask__", "") for f in files]
            masks = np.zeros((len(runs), len(A_list)) + target, bool)
            h1 = np.zeros_like(masks)
            h2 = np.zeros_like(masks)
            for k, (run, f) in enumerate(zip(runs, files)):
                pages = C.read_pages(f, A_list)
                n = len(pages)  # pages beyond the stack end are dropped by read_pages (A_list starts >= 0)
                masks[k, :n] = np.stack([C.to_our_grid(p > 0, target) for p in pages])
                hp = C.read_pages(C.hemisphere_file(PIPE, brain, variant, run, run_dirs[run]), A_list)
                hh = np.stack([C.to_our_grid(p, target) for p in hp])
                h1[k, :len(hh)] = hh == 1
                h2[k, :len(hh)] = hh == 2
            # annotated hemisphere label: majority of all runs' vote over the STAPLE pixels at A0 (as compare.py)
            i0 = A_list.index(A0)
            c1, c2 = h1[:, i0].sum(0), h2[:, i0].sum(0)
            vote = np.where(c1 > c2, 1, np.where(c2 > c1, 2, 0))[truth]
            label = 1 if (vote == 1).sum() >= (vote == 2).sum() else 2
            union = masks.any((0, 1)) | truth | (human_p > 0) | rmasks.any(0)
            rows_, cols_ = np.flatnonzero(union.any(1)), np.flatnonzero(union.any(0))
            s0, s1 = max(0, rows_[0] - MARGIN), min(target[0], rows_[-1] + MARGIN + 1)
            r0, r1 = max(0, cols_[0] - MARGIN), min(target[1], cols_[-1] + MARGIN + 1)
            crop = (slice(None), slice(None), slice(s0, s1), slice(r0, r1))
            m, a1, a2 = masks[crop], h1[crop], h2[crop]
            np.savez_compressed(
                dest, runs=np.array(runs), A_list=np.array(A_list), A0=A0, slab=np.array([slab0, slab1]),
                bbox=np.array([s0, s1, r0, r1]), shape=np.array(m.shape), label=label,
                masks=np.packbits(m), h1=np.packbits(a1), h2=np.packbits(a2),
                truth=truth[s0:s1, r0:r1], human_p=human_p[s0:s1, r0:r1],
                raters=rmasks[:, s0:s1, r0:r1], rater_user=rr.user.to_numpy(), rater_idx=rr.idx.to_numpy(),
                rater_attempt=rr.attempt.to_numpy(), full_shape=np.array(target))
            out.append(dict(brain=brain, structure=structure, variant=variant, status="written", n_runs=len(runs),
                            n_slices=len(A_list), crop=f"{s1 - s0}x{r1 - r0}", label=label,
                            seconds=round(time.time() - t0)))
            print(f"{brain} {structure} {variant}: {len(runs)} runs, {len(A_list)} slices, crop {s1 - s0}x{r1 - r0}, "
                  f"label {label} ({time.time() - t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    brains = sys.argv[1:] or BRAINS
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(6, len(brains))) as ex:
        res = [r for rs in ex.map(build_brain, brains) for r in rs]
    df = pd.DataFrame(res)
    (CACHE / "amap").mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE / "amap" / f"cache_log_{int(time.time())}.csv", index=False)
    log(df.groupby(["status"]).size().to_string(), f"\ntotal {time.time() - t0:.0f}s")
