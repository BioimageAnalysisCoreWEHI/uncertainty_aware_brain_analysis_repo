"""SUB sensitivity analysis (aMAP): score the subiculum without the ventral part that the experts did not annotate.

The atlas SUB label (osten_mouse_25um, id 502) covers dorsal and ventral subiculum and the atlas has no dorsal/ventral
subdivision. The aMAP raters outlined only the dorsal subiculum. At the scored sections, the atlas-derived SUB mask
consists of a dorsal component (overlapping the annotations) and a separate ventral component.

Rule (fixed before scoring; uses only the annotations, never the STAPLE or Dice): on every coronal section, keep the
2-D connected components (8-connectivity) of a mask that intersect the union of all expert annotations of that case;
discard the others. Applied identically to the consensus and to every single registration; the aMAP protocol (best
section within the raters' stack) is otherwise unchanged. Expert scores are unaffected (their outlines are dorsal).
Raw input, final QC set, consensus threshold 0.5. Read-only.

Output: REVISION_OUTPUTS_FOCUSED/source_data/SUB_dorsal_only_per_case.csv, SUB_dorsal_only_singles.csv,
        SUB_dorsal_only_summary.csv
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from common import BRAINS, SHORT, VA, TABLES

s = importlib.util.spec_from_file_location("f01", str(VA / "focused" / "scripts" / "f01_focused_validation.py"))
f01 = importlib.util.module_from_spec(s)
s.loader.exec_module(f01)
m3 = f01.m3
SRC = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\REVISION_OUTPUTS_FOCUSED\source_data"))
EIGHT = np.ones((3, 3), bool)


def dice(a, b):
    t = a.sum() + b.sum()
    return 2 * (a & b).sum() / t if t else np.nan


def keep_annotated(m2d, ann_union):
    lab, n = ndimage.label(m2d, structure=EIGHT)
    if n == 0:
        return m2d
    keep = np.unique(lab[ann_union & (lab > 0)])
    return np.isin(lab, keep[keep > 0])


def protocol(stack, cs, ann_union, restrict):
    best, bestA, bestk = -1.0, None, None
    for k in cs.slab_idx:
        m = keep_annotated(stack[k], ann_union) if restrict else stack[k]
        d = dice(m, cs.truth)
        if d > best:
            best, bestA, bestk = d, cs.A_list[k], k
    return best, bestA


def main():
    qc = m3.load_qc()
    rows, singles = [], []
    for b in BRAINS:
        cs, final = f01.load(b, "SUB", qc)
        used = final if final else cs.runs
        ann_union = cs.raters.any(0)
        cons = cs.consensus(used, 0.5)
        d0, a0 = protocol(cons, cs, ann_union, False)
        d1, a1 = protocol(cons, cs, ann_union, True)
        k1 = cs.A_list.index(a1)
        removed = cons[k1] & ~keep_annotated(cons[k1], ann_union)
        rows.append(dict(brain=b, sample=SHORT.get(b, b), n_retained=len(used), dice_consensus_full=d0,
                         best_section_full=a0, dice_consensus_dorsal_only=d1, best_section_dorsal_only=a1,
                         consensus_px_at_section=int(cons[k1].sum()), removed_ventral_px=int(removed.sum()),
                         staple_px=int(cs.truth.sum())))
        for r in used:
            st = cs.run_mask(r)
            s0, _ = protocol(st, cs, ann_union, False)
            s1, _ = protocol(st, cs, ann_union, True)
            singles.append(dict(brain=b, run=r, dice_full=s0, dice_dorsal_only=s1))
        print(f"{b}: consensus {d0:.3f} -> {d1:.3f}", flush=True)
    pc = pd.DataFrame(rows)
    sg = pd.DataFrame(singles)
    # check: unrestricted values reproduce the published SUB scores
    ref = pd.read_csv(VA / "focused" / "tables" / "focused_case_scores.csv")
    ref = ref[ref.structure == "SUB"].set_index("brain").dice_amap_protocol
    pc["matches_published_full"] = np.isclose(pc.dice_consensus_full, pc.brain.map(ref))
    runs = pd.read_csv(TABLES / "amap_runs.csv")
    runs = runs[(runs.variant == "raw") & (runs.structure == "SUB") & runs.passes_final_qc.astype(bool)]
    raters = pd.read_csv(VA / "focused" / "tables" / "focused_rater_scores.csv")
    rat = raters[raters.structure == "SUB"].dice
    summ = pd.DataFrame([
        dict(group="annotators (unchanged)", n=len(rat), median_full=rat.median(), median_dorsal_only=rat.median()),
        dict(group="single retained registrations", n=len(sg), median_full=sg.dice_full.median(),
             median_dorsal_only=sg.dice_dorsal_only.median()),
        dict(group="pipeline consensus", n=len(pc), median_full=pc.dice_consensus_full.median(),
             median_dorsal_only=pc.dice_consensus_dorsal_only.median())])
    summ["published_single_median_check"] = [np.nan, runs.dice_paper.median(), np.nan]
    pc.to_csv(SRC / "SUB_dorsal_only_per_case.csv", index=False)
    sg.to_csv(SRC / "SUB_dorsal_only_singles.csv", index=False)
    summ.to_csv(SRC / "SUB_dorsal_only_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(pc.round(3).to_string(index=False))
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
