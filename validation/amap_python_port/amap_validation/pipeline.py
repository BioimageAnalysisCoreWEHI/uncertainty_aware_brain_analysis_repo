"""Agreement segmentations (createAgreementSegs.m / userRater) and manual scoring (segAnalyser.scoreManualSegs)."""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import masks as M
from . import nifti
from .config import Config, matlab_num2str
from .scoring import dice, hausdorff, pad_to_larger
from .segtable import build_segtable
from .staple import sba, staple, valid_slices


@dataclass
class RunOptions:
    out_dir: Path
    mode: str = ""              # '', 'range' or 'quantiles'  (userRater.create4dNii)
    arg: object = None          # 7 for 'range', [0.25, 0.75] for 'quantiles'
    repeats: str = "both"       # 'both' = MATLAB code; 'first' = paper ("only the first segmentation")
    unc: bool = True            # seg_LabFusion -unc
    workers: int = 6
    only: tuple[str, ...] = ()  # restrict to these brainRegionComb values

    @property
    def suffix(self) -> str:
        s = ""
        if self.mode == "range":
            s += f"_zRange{self.arg}"
        elif self.mode == "quantiles":
            s += "_zRangeQuartiles"
        if self.repeats == "first":
            s += "_firstAttemptOnly"
        if not self.unc:
            s += "_noUnc"
        return s

    def staple_dir(self) -> Path:
        return self.out_dir / ("userStapleResults" + self.suffix)

    def cache_dir(self, scale: float) -> Path:
        return self.out_dir / f"maskCache_{matlab_num2str(scale)}"


def _mask_job(args):
    path, canvas, scale, cache = args
    M.cached_masks(path, canvas, scale, Path(cache))
    return path


def prepare_masks(table: pd.DataFrame, canvas, scale: float, cache: Path, workers: int, log=print):
    todo = [(p, canvas, scale, str(cache)) for p in table["path"]]
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k, _ in enumerate(ex.map(_mask_job, todo, chunksize=4), 1):
                if k % 50 == 0 or k == len(todo):
                    log(f"  masks {k}/{len(todo)}  ({time.time() - t0:.0f}s)")
    else:
        for k, job in enumerate(todo, 1):
            _mask_job(job)
            if k % 50 == 0 or k == len(todo):
                log(f"  masks {k}/{len(todo)}  ({time.time() - t0:.0f}s)")


def _save_like_matlab(img_xy: np.ndarray, path: Path):
    """make_nii + save_nii: pixdim 1, sform_code 1, srow = diag(1) with +1 offset, qform_code 0."""
    aff = np.eye(4)
    aff[:3, 3] = 1.0
    nifti.save(path, img_xy.astype(np.uint8), aff, sform_code=1, qform_code=0)


def select_rows(table: pd.DataFrame, brc: str, opts: RunOptions) -> pd.DataFrame:
    cur = table[table["brainRegionComb"] == brc]
    if opts.repeats == "first":
        cur = cur[cur["attempt"] == 1]
    keep = valid_slices(cur["sliceNum"], opts.mode, opts.arg)
    return cur[cur["sliceNum"].isin(keep)]


def run_agreement(cfg: Config, opts: RunOptions, log=print) -> pd.DataFrame:
    table = build_segtable(cfg.seg_dir)
    canvas = M.canvas_size(table["path"])
    log(f"{len(table)} segmentations, {table['brainRegionComb'].nunique()} brain x structure groups, "
        f"canvas (maxX, maxY) = {canvas}")
    cache = opts.cache_dir(cfg.scale)
    needed = table[table["brainRegionComb"].isin(opts.only)] if opts.only else table
    prepare_masks(needed, canvas, cfg.scale, cache, opts.workers, log)

    out = opts.staple_dir()
    out.mkdir(parents=True, exist_ok=True)
    (out / "canvas.json").write_text(json.dumps({"maxX_maxY": list(canvas), "scale": cfg.scale}))
    sc = matlab_num2str(cfg.scale)
    table = table.assign(p=np.nan, q=np.nan, inStaple=False)
    combos = sorted(table["brainRegionComb"].unique())
    if opts.only:
        combos = [c for c in combos if c in opts.only]
    summary = []
    for brc in combos:
        cur = select_rows(table, brc, opts)
        stack = np.stack([M.cached_masks(p, canvas, cfg.scale, cache)[0] for p in cur["path"]])
        res = staple(stack, unc=opts.unc)
        sba_img = sba(stack)
        _save_like_matlab(res.labels, out / f"{brc}_{sc}_NiftySegSTAPLE.nii")
        _save_like_matlab(sba_img, out / f"{brc}_{sc}_NiftySegSBA.nii")
        raters = [f"{u}:{int(i)}" for u, i in zip(cur["user"], cur["idx"])]
        info = dict(brainRegionComb=brc, n_segmentations=len(cur), raters=raters,
                    slices=[int(s) for s in cur["sliceNum"]],
                    sensitivity_p=res.sensitivity.tolist(), specificity_q=res.specificity.tolist(),
                    prior=res.prior, iterations=res.iterations, converged=res.converged,
                    unc=opts.unc, repeats=opts.repeats, mode=opts.mode, arg=opts.arg,
                    staple_area_px=int(res.labels.sum()), sba_area_px=int(sba_img.sum()),
                    dice_staple_vs_sba=dice(res.labels, sba_img),
                    array_orientation="MATLAB [x=TIFF column, y=TIFF row]", pixel_um=0.001 / cfg.scale * 1000)
        (out / f"{brc}_{sc}_NiftySegSTAPLE.json").write_text(json.dumps(info, indent=1))
        table.loc[cur.index, "p"] = res.sensitivity
        table.loc[cur.index, "q"] = res.specificity
        table.loc[cur.index, "inStaple"] = True
        summary.append(dict(brainRegionComb=brc, n=len(cur), iterations=res.iterations, converged=res.converged,
                            staple_area_px=info["staple_area_px"], sba_area_px=info["sba_area_px"],
                            dice_staple_vs_sba=info["dice_staple_vs_sba"]))
        log(f"  {brc:22s} n={len(cur):2d} it={res.iterations:4d} area={info['staple_area_px']:7d} "
            f"dice(STAPLE,SBA)={info['dice_staple_vs_sba']:.3f}")
    table.to_csv(out / "segTable_with_pq.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "staple_summary.csv", index=False)
    return table


def run_manual_scoring(cfg: Config, opts: RunOptions, log=print) -> pd.DataFrame:
    """segAnalyser.scoreManualSegs for every brain: each included segmentation vs its STAPLE/SBA consensus."""
    out = opts.staple_dir()
    table = pd.read_csv(out / "segTable_with_pq.csv")
    canvas = tuple(json.loads((out / "canvas.json").read_text())["maxX_maxY"])
    cache = opts.cache_dir(cfg.scale)
    sc = matlab_num2str(cfg.scale)
    rows = []
    for brc, cur in table[table["inStaple"]].groupby("brainRegionComb"):
        st_path = out / f"{brc}_{sc}_NiftySegSTAPLE.nii"
        sb_path = out / f"{brc}_{sc}_NiftySegSBA.nii"
        if not st_path.exists():
            continue
        st = nifti.load(st_path)
        sb = nifti.load(sb_path)
        for _, r in cur.iterrows():
            bic = M.cached_masks(r["path"], canvas, cfg.scale, cache)[1]
            a, s1 = pad_to_larger(bic, st)
            a2, s2 = pad_to_larger(bic, sb)
            rows.append(dict(r, diceSTAPLE=dice(a, s1), diceSBA=dice(a2, s2),
                             hdSTAPLE=hausdorff(a, s1), hdSBA=hausdorff(a2, s2)))
        log(f"  scored {brc}")
    res = pd.DataFrame(rows)
    res["hdSTAPLE_um"] = res["hdSTAPLE"] / cfg.scale  # native TIFF pixel = 1 um
    res["hdSBA_um"] = res["hdSBA"] / cfg.scale
    res.to_csv(out / "manual_scores.csv", index=False)
    return res
