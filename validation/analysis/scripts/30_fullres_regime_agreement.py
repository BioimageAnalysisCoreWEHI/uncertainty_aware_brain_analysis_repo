"""How much do the QC rules change the original-LSFM SNpc consensus, measured on the full-resolution axial masks and
the cellpose cells they contain (QC_REGIME_FULLRES_AXIAL_MASKS, written by script 24)?

For each sample x input: streamed slice by slice over all rules' masks and masked-label files,
  - voxel Dice and volume for every pair of rules,
  - the set of cell labels counted under each rule -> cell Jaccard and the fraction of the current rule's cells kept.
Read-only. Output: QC_REGIME_FULLRES_AXIAL_MASKS/agreement_pairwise.csv, agreement_vs_current.csv
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

OUT = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\QC_REGIME_FULLRES_AXIAL_MASKS"))
RULES = ["no_global_QC", "very_lenient", "untuned_start", "lenient", "current", "median", "strict", "very_strict"]


def case(args):
    s, v = args
    d = OUT / s / v
    rules = [r for r in RULES if (d / r).exists()]
    mf = {r: tifffile.TiffFile(d / r / f"{s}_{v}_{r}_consensus0p5_mask_fullres_axial.tif") for r in rules}
    cf = {r: tifffile.TiffFile(d / r / f"{s}_{v}_{r}_masked_cells.tif") for r in rules}
    nz = len(mf[rules[0]].pages)
    vol = {r: 0 for r in rules}
    inter = {p: 0 for p in combinations(rules, 2)}
    cells = {r: set() for r in rules}
    for z in range(nz):
        m = {r: mf[r].pages[z].asarray() > 0 for r in rules}
        for r in rules:
            vol[r] += int(m[r].sum())
            cells[r].update(np.unique(cf[r].pages[z].asarray()).tolist())
        for a, b in inter:
            inter[(a, b)] += int((m[a] & m[b]).sum())
    for f in list(mf.values()) + list(cf.values()):
        f.close()
    for r in rules:
        cells[r].discard(0)
    rows = []
    for (a, b), i in inter.items():
        ca, cb = cells[a], cells[b]
        rows.append(dict(sample=s, input=v, rule_a=a, rule_b=b, voxel_dice=2 * i / (vol[a] + vol[b]) if vol[a] + vol[b] else np.nan,
                         vol_a=vol[a], vol_b=vol[b], cells_a=len(ca), cells_b=len(cb),
                         cell_jaccard=len(ca & cb) / len(ca | cb) if ca | cb else np.nan,
                         cells_shared=len(ca & cb)))
    print(f"{s} {v}: done ({nz} slices)", flush=True)
    return rows


if __name__ == "__main__":
    jobs = [(s.name, v.name) for s in sorted(OUT.iterdir()) if s.is_dir() for v in sorted(s.iterdir()) if v.is_dir()]
    if len(sys.argv) > 1:
        jobs = [tuple(sys.argv[1:3])]
    with ProcessPoolExecutor(max_workers=5) as ex:
        rows = [r for rs in ex.map(case, jobs) for r in rs]
    pw = pd.DataFrame(rows)
    pw.to_csv(OUT / "agreement_pairwise.csv", index=False)
    cur = []
    for r in pw.itertuples():
        if "current" in (r.rule_a, r.rule_b):
            other = r.rule_b if r.rule_a == "current" else r.rule_a
            vc, vo = (r.vol_a, r.vol_b) if r.rule_a == "current" else (r.vol_b, r.vol_a)
            cc, co = (r.cells_a, r.cells_b) if r.rule_a == "current" else (r.cells_b, r.cells_a)
            cur.append(dict(sample=r.sample, input=r.input, rule=other, voxel_dice=r.voxel_dice,
                            volume_change_pct=100 * (vo / vc - 1), count_change_pct=100 * (co / cc - 1),
                            cell_jaccard=r.cell_jaccard, frac_current_cells_kept=r.cells_shared / cc))
    pd.DataFrame(cur).to_csv(OUT / "agreement_vs_current.csv", index=False)
    print("written")
