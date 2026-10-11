"""A14 cache: per-voxel registration votes on the cellpose cell voxels of each LSFM brain (read-only inputs).

For each sample x processing status, the cellpose label image (``whole_segmentation_python.tif``, the one used by
``get_consensus_counts.py``) and the 48 per-run SNpc masks (``<status>_mask_axial_path``) are read once. For every
voxel that belongs to a cell we keep the 48-bit vector of which registrations include it. Any consensus count
(any run subset, any threshold) under the manuscript's overlap rule -- a cell counts if ANY of its voxels lies in the
consensus mask, consensus = float32(mean of masks) >= threshold -- is then exact and cheap:
    count(S, t) = #unique cells with some voxel where float32(votes[S].sum / |S|) >= t.
Single-run counts under the same rule are the special case |S| = 1.

Output: cache/lsfm/<sample>_<status>.npz and cache/lsfm/runs.csv.
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from common import CACHE, LSFM, log

OUT = CACHE / "lsfm"
TABLE = LSFM / "final_count_results" / "full_final_results_with_ssim.csv"
BASES = {"raw": LSFM / "results_raw" / "cellpose_masked_cells" / "for_cellpose",
         "preprocessed": LSFM / "results_preprocessed" / "cellpose_masked_cells" / "for_cellpose"}


def label_path(sample, status):
    return (BASES[status] / f"{sample}_AF_level1_preprocessed_freeform_bending_0.0_grid_-2_smoothing_-1_atlas_perens_"
            "lsfm_mouse_20um" / "results_python" / "snpc" / "whole_segmentation_python.tif")


def build(job):
    sample, status, sub = job
    t0 = time.time()
    dest = OUT / f"{sample}_{status}.npz"
    if dest.exists():
        return f"{sample} {status}: cached"
    labels = tifffile.imread(label_path(sample, status))
    idx = np.flatnonzero(labels)
    cell = labels.reshape(-1)[idx].astype(np.int32)
    n_cells = int(np.unique(cell).size)
    del labels
    votes = np.zeros((len(idx), len(sub)), bool)
    for k, p in enumerate(sub[f"{status}_mask_axial_path"]):
        m = tifffile.imread(p)
        votes[:, k] = m.reshape(-1)[idx] > 0
    np.savez_compressed(dest, cell=cell, votes=np.packbits(votes, axis=1), n_runs=len(sub), n_cells=n_cells,
                        runs=sub.run_key.to_numpy().astype(str))
    return f"{sample} {status}: {n_cells} cells, {len(idx)} cell voxels, {len(sub)} runs ({time.time() - t0:.0f}s)"


def load_votes(sample, status):
    z = np.load(OUT / f"{sample}_{status}.npz")
    n = int(z["n_runs"])
    votes = np.unpackbits(z["votes"], axis=1)[:, :n].astype(bool)
    return z["cell"], votes, [str(r) for r in z["runs"]], int(z["n_cells"])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TABLE)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")].copy()
    df["Sample ID"] = df["Sample ID"].astype(str)
    df["run_key"] = df.bending.map(lambda b: f"{b:g}") + "_" + df.grid.map(lambda g: f"{g:g}")
    keep = ["Sample ID", "bending", "grid", "run_key"]
    long = []
    for st in ("raw", "preprocessed"):
        cols = {c: c[len(st) + 1:] for c in df.columns if c.startswith(st + "_")}
        d = df[keep + list(cols)].rename(columns=cols).assign(status=st)
        long.append(d)
    runs = pd.concat(long, ignore_index=True).rename(columns={"Sample ID": "sample"})
    runs.to_csv(OUT / "runs.csv", index=False)
    jobs = []
    for (s, st), sub in runs.groupby(["sample", "status"]):
        sub = sub.sort_values(["bending", "grid"]).rename(columns={"mask_axial_path": f"{st}_mask_axial_path"})
        missing = [p for p in sub[f"{st}_mask_axial_path"] if not Path(p).exists()]
        if missing:
            log(f"{s} {st}: {len(missing)} missing masks, e.g. {missing[0]}")
        jobs.append((s, st, sub))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=5) as ex:
        for msg in ex.map(build, jobs):
            log(msg)
    log(f"total {time.time() - t0:.0f}s")
