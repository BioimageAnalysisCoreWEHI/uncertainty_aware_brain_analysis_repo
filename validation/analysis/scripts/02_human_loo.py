"""A2: human baselines at 25 um. For every rater (user), STAPLE of the *other* raters' segmentations
(leave-one-rater-out; both attempts of a user are left out together for the repeated brains), computed exactly as
the full consensus (same cached 4 um canvas masks, ``staple(unc=True)``), placed on the STAPLE median plane and
resliced to the asr 25 um grid with the same functions as the reference products.

Each rater's medianPlane mask is then scored against (a) its LOO consensus (primary human baseline) and (b) the
full STAPLE (self-inclusive; sensitivity). Outputs: cache/human_loo/<brain>_<structure>.npz (LOO masks, full frame)
and results/tables/human_loo.csv.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import AMAP_OUT, CACHE, PAPER_NAME, log, save_table, seg_metrics

OUT = CACHE / "human_loo"


def do_case(brc: str) -> list[dict]:
    from amap_validation import masks as M
    from amap_validation import nifti
    from amap_validation.config import load_config, matlab_num2str
    from amap_validation.geometry import StackMapping, canvas_to_volume
    from amap_validation.reslice import AsrGrid, AsrLayout, sheet_to_asr25
    from amap_validation.staple import staple
    from amap_validation.tobrain import stack_name

    cfg = load_config()
    sd = AMAP_OUT / "userStapleResults"
    lay = AsrLayout(sd / "asr25um")
    seg = pd.read_csv(sd / "segTable_with_pq.csv")
    cur = seg[seg.inStaple & (seg.brainRegionComb == brc)]
    brain, region = cur.brain.iloc[0], cur.region.iloc[0]
    structure = PAPER_NAME[region]
    canvas = tuple(json.loads((sd / "canvas.json").read_text())["maxX_maxY"])
    cache = AMAP_OUT / f"maskCache_{matlab_num2str(cfg.scale)}"
    info = json.loads((sd / "inBrainSpace" / f"{brc}_{matlab_num2str(cfg.scale)}_STAPLE_inBrain.json").read_text())
    j0 = float(info["J0_median_slice"])
    h = nifti.read_header(cfg.brain_file(brain))
    grid = AsrGrid(h["shape"], h["sform"])
    mp = StackMapping(cfg.stack(brain, stack_name(cur.path.iloc[0])), cfg.scale)
    slices = pd.read_csv(lay.slices_log).set_index("brainRegionComb")
    truth = nifti.load(lay.dest / slices.loc[brc, "staple_file"])[0] > 0
    raters = pd.read_csv(lay.raters_log)
    rr = raters[(raters.brainRegionComb == brc) & (raters.variant == "medianPlane")]
    stack = np.stack([M.cached_masks(p, canvas, cfg.scale, cache)[0] for p in cur.path])
    users = sorted(cur.user.unique())
    loo, rows = {}, []
    t0 = time.time()
    for u in users:
        keep = (cur.user != u).to_numpy()
        res = staple(stack[keep], unc=True)
        vox, _ = canvas_to_volume(res.labels, mp, j0, h["shape"][:3])
        img = sheet_to_asr25(vox, np.ones(len(vox)), grid, binary=True)[0] > 0
        loo[u] = img
        for r in rr[rr.user == u].itertuples():
            m = nifti.load(lay.dest / r.file)[0] > 0
            a = seg_metrics(m, img)
            b = seg_metrics(m, truth)
            rows.append(dict(brain=brain, region=region, structure=structure, user=u, idx=int(r.idx),
                             attempt=int(r.attempt), n_loo=int(keep.sum()), loo_area_px=int(img.sum()),
                             dice_loo=a["dice"], hd_loo_um=a["hd_um"], hd95_loo_um=a["hd95_um"],
                             msd_loo_um=a["msd_um"], area_ratio_loo=a["area_ratio"],
                             dice_self=b["dice"], hd_self_um=b["hd_um"], hd95_self_um=b["hd95_um"],
                             msd_self_um=b["msd_um"], area_ratio_self=b["area_ratio"],
                             dice_loo_vs_full_staple=seg_metrics(img, truth)["dice"]))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{brain}_{structure}.npz", users=np.array(users),
                        loo=np.packbits(np.stack([loo[u] for u in users])), shape=np.array(truth.shape))
    print(f"{brc}: {len(users)} LOO consensuses, {len(rows)} rater scores ({time.time() - t0:.0f}s)", flush=True)
    return rows


def load_loo(brain, structure):
    z = np.load(OUT / f"{brain}_{structure}.npz")
    shp = tuple(int(s) for s in z["shape"])
    users = [str(u) for u in z["users"]]
    m = np.unpackbits(z["loo"])[: len(users) * shp[0] * shp[1]].reshape((len(users),) + shp).astype(bool)
    return users, m


if __name__ == "__main__":
    seg = pd.read_csv(AMAP_OUT / "userStapleResults" / "segTable_with_pq.csv")
    combos = sorted(seg[seg.inStaple].brainRegionComb.unique())
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = [r for rs in ex.map(do_case, combos) for r in rs]
    df = pd.DataFrame(rows)
    save_table(df, "human_loo")
    log(f"{len(df)} rater scores; median Dice LOO {df.dice_loo.median():.3f} vs self-inclusive "
        f"{df.dice_self.median():.3f}; LOO <= self in {(df.dice_loo <= df.dice_self + 1e-12).mean():.1%} "
        f"({time.time() - t0:.0f}s)")
