"""Expert STAPLE references of the aMAP validation as 2-D sections in the space of the released consensus regions.

The expert STAPLE consensus is one coronal section per brain x structure. The aMAP Python port places it on an asr
25 um grid (userStapleResults/asr25um/<brain>/<structure>/<brain><code>_STAPLE_asr25um.tif, plane A recorded in
asr25um_checks_<brain>.json). Scoring places each registration's (S, R) slice on that grid by direct indexing
(compare.to_our_grid, offset 0), so brainreg's 25 um grid (the grid of the released regions) and the asr grid share
indices; brainreg's grid can be one row shorter where it rounded an odd dimension down. Here each STAPLE section is
cropped to the region grid (the cropped rows are checked to be empty) and saved as a 2-D TIFF, with a table of the
coronal plane and hemisphere of every section.

functions:
  write_sections(R, asr_dir, out_rel, region_dir_rel) -> DataFrame of sections (adds files to Release R)
  check_dice(release_amap_dir, sections, amap_sets_csv) -> DataFrame comparing recomputed and published Dice
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

CODE_TO_STRUCT = {"CING": "ACA", "S1BF": "SSp", "SG": "DG-sg", "V1": "VISp", "V2M": "VISam", "AHN": "AHN",
                  "MV": "MV", "RSP": "RSP", "SUB": "SUB", "VPM": "VPM"}
STRUCT_TO_CODE = {v: k for k, v in CODE_TO_STRUCT.items()}
PORT = Path(__file__).resolve().parents[1] / "validation" / "amap_python_port"


def _port_config():
    sys.path.insert(0, str(PORT))
    from amap_validation.config import FLIPPED_HEMISPHERE, MIDLINES  # noqa: E402
    return MIDLINES, FLIPPED_HEMISPHERE


def hemisphere_keep(brain, code, transform, nR):
    """(nR,) bool: columns on the annotated side of aMAP's midline (compare.hemisphere_keep)."""
    MIDLINES, FLIPPED = _port_config()
    r_centre = 2 * np.arange(nR) + 0.5
    nx = transform["original_shape_xyz"][0]
    x_centre = (nx - 1 - r_centre) if transform["mirror_lr"] else r_centre
    keep = x_centre >= MIDLINES[brain] - 0.5
    return ~keep if (brain, code) in FLIPPED else keep


def write_sections(R, asr_dir, out_rel, region_dir_rel):
    rows = []
    for bd in sorted(p for p in Path(asr_dir).iterdir() if p.is_dir()):
        brain = bd.name
        checks = json.loads((bd / f"asr25um_checks_{brain}.json").read_text())
        transform = json.loads((bd / f"{brain}_transform.json").read_text())
        with tifffile.TiffFile(R.out / region_dir_rel / brain / f"{brain}_brain_25um.tif") as t:
            nA, nS, nR = t.series[0].shape
        for s in checks["structures"]:
            code, st = s["region"], CODE_TO_STRUCT[s["region"]]
            src = bd / st / f"{brain}{code}_STAPLE_asr25um.tif"
            sl = tifffile.imread(src)
            assert sl.ndim == 2 and sl.shape[1] == nR and sl.shape[0] >= nS and s["A"] < nA, (brain, st, sl.shape)
            assert not sl[nS:].any(), (brain, st, "STAPLE in rows outside the region grid")
            crop = np.ascontiguousarray((sl[:nS] > 0).astype(np.uint8))
            keep = hemisphere_keep(brain, code, transform, nR)
            rel = f"{out_rel}/{brain}_{st}_expert_STAPLE.tif"
            desc = dict(brain=brain, structure=st, content="expert STAPLE consensus (2-D coronal section)",
                        coronal_plane_A=int(s["A"]), axes="SR (rows: superior->inferior, columns: R as in the region "
                        "grid)", grid=f"plane A of the 25 um region grid ({nA}, {nS}, {nR})")
            tifffile.imwrite(R._dst(rel), crop, compression="zlib", metadata=None, description=json.dumps(desc))
            R.rows.append(dict(release_path=rel, from_path=str(src), operation="cropped to the region grid (empty rows)",
                               note="expert STAPLE section"))
            rows.append(dict(brain=brain, structure=st, aMAP_region_code=code, coronal_plane_A=int(s["A"]),
                             n_pixels=int(crop.sum()), file=Path(rel).name))
    sec = pd.DataFrame(rows)
    R.write_text(f"{out_rel}/expert_STAPLE_sections.csv", sec.to_csv(index=False), asr_dir,
                 "generated table", "coronal plane and annotated hemisphere of every STAPLE section")
    return sec


def _run_name(be, g):
    f = lambda v: f"{v:.10g}".replace(".", "p") if "." in f"{v:.10g}" else f"{v:.10g}p0"
    return f"bend_{f(be)}__grid_m{abs(int(g))}__smooth_m1p0"


def write_hemispheres(R, cache_dir, sections, out_rel, region_dir_rel, variant="raw"):
    """Annotated-hemisphere mask used in the paper for every case: on each plane of the raters' slab, the voxels
    where the retained registrations' registered_hemispheres vote is the annotated label (strict majority), as in
    validation/analysis/scripts/03_amap_metrics.py (Case.probability). Saved as a 3-D volume with the shape of the
    brain (planes outside the slab are 0), so the paper's best-section search can be re-run; returns the sections
    table with the slab range and annotated label added."""
    rows = []
    for r in sections.itertuples():
        z = np.load(Path(cache_dir) / r.brain / f"{r.structure}_{variant}.npz")
        n, nA, hs, hw = (int(v) for v in z["shape"])
        unpack = lambda k: np.unpackbits(z[k])[: n * nA * hs * hw].reshape(n, nA, hs, hw).astype(bool)
        h1, h2 = unpack("h1"), unpack("h2")
        hl, ho = (h1, h2) if int(z["label"]) == 1 else (h2, h1)
        qc = pd.read_csv(R.out / region_dir_rel / r.brain / f"{r.brain}_registrations_qc.csv")
        ret = qc[qc[f"retained_{r.structure}"].astype(bool)]
        names = [_run_name(b, g) for b, g in zip(ret.bending_energy_weight, ret.grid_spacing)]
        idx = {str(x): i for i, x in enumerate(z["runs"])}
        missing = [x for x in names if x not in idx]
        assert not missing, (r.brain, r.structure, missing[:3], list(z["runs"])[:3])
        ids = [idx[x] for x in names]
        keep = hl[ids].sum(0) > ho[ids].sum(0)                       # (nA, hs, hw), cropped to bbox
        A_list = [int(a) for a in z["A_list"]]
        slab0, slab1 = (int(v) for v in z["slab"])
        sel = [i for i, a in enumerate(A_list) if slab0 <= a <= slab1]
        s0, s1, r0, r1 = (int(v) for v in z["bbox"])
        with tifffile.TiffFile(R.out / region_dir_rel / r.brain / f"{r.brain}_brain_25um.tif") as t:
            nA_brain, nS, nR = t.series[0].shape
        planes = [A_list[i] for i in sel]
        assert all(0 <= a < nA_brain for a in planes), (r.brain, r.structure, planes)
        sec_sr = np.zeros((len(sel), int(z["full_shape"][0]), int(z["full_shape"][1])), np.uint8)
        sec_sr[:, s0:s1, r0:r1] = keep[sel]
        assert not sec_sr[:, nS:].any() and sec_sr.shape[2] == nR, (r.brain, r.structure)
        vol = np.zeros((nA_brain, nS, nR), np.uint8)                # same shape as <brain>_brain_25um.tif
        vol[planes] = sec_sr[:, :nS]
        rel = f"{out_rel}/{r.brain}_{r.structure}_annotated_hemisphere_slab.tif"
        tifffile.imwrite(R._dst(rel), vol, compression="zlib", metadata=None, description=json.dumps(dict(
            brain=r.brain, structure=r.structure, axes="ASR, same grid as the brain and the regions",
            coronal_planes_A=planes, n_retained_registrations=len(ids),
            note="mask defined on the planes of the raters' 40-section stack only (the planes scored in the paper); "
                 "all other planes are 0",
            rule="registered_hemispheres majority vote of the retained registrations = annotated label")))
        R.rows.append(dict(release_path=rel, from_path=str(Path(cache_dir) / r.brain / f"{r.structure}_{variant}.npz"),
                           operation="majority vote of the retained registrations' registered hemispheres",
                           note="annotated hemisphere on the raters' slab (paper protocol)"))
        rows.append(dict(brain=r.brain, structure=r.structure, slab_A_first=slab0, slab_A_last=slab1,
                         annotated_hemisphere_label=int(z["label"]), n_retained_registrations=len(ids),
                         hemisphere_file=Path(rel).name))
    return sections.merge(pd.DataFrame(rows), on=["brain", "structure"], how="left")


def check_dice_paper(amap_dir, sections, amap_sets_csv):
    """Exact paper protocol from released files: per slab section, consensus & annotated hemisphere vs the 2-D
    STAPLE; the best section's Dice must equal the published dice_paper."""
    sets = pd.read_csv(amap_sets_csv)
    sets = sets[(sets.variant == "raw") & (sets.set_name == "passes_final_qc_top_0p5") & np.isclose(sets.threshold, 0.5)]
    out = []
    for r in sections.itertuples():
        row = sets[(sets.brain == r.brain) & (sets.structure == r.structure)]
        if row.empty:
            continue
        cons = tifffile.imread(Path(amap_dir) / r.brain / f"{r.brain}_{r.structure}_consensus0p5_25um.tif") > 0
        truth = tifffile.imread(Path(amap_dir) / "expert_STAPLE" / r.file) > 0
        with tifffile.TiffFile(Path(amap_dir) / "expert_STAPLE" / r.hemisphere_file) as t:
            hem = t.series[0].asarray() > 0
            planes = json.loads(t.pages[0].description)["coronal_planes_A"]
        assert hem.shape == cons.shape, (r.brain, r.structure, hem.shape, cons.shape)
        best = (-1.0, None)
        for A in planes:
            m = cons[A] & hem[A]
            s = m.sum() + truth.sum()
            d = 2 * (m & truth).sum() / s if s else np.nan
            if d > best[0]:
                best = (d, A)
        out.append(dict(brain=r.brain, structure=r.structure, dice_recomputed=best[0], A_best_recomputed=best[1],
                        dice_published=float(row.dice_paper.iloc[0]), A_best_published=int(row.A_best.iloc[0])))
    return pd.DataFrame(out)


def check_dice(amap_dir, sections, amap_sets_csv):
    """Recompute the published consensus Dice (aMAP protocol: best section, here at the published A_best) from the
    released 3-D consensus and the 2-D STAPLE, with the aMAP-midline hemisphere rule."""
    sets = pd.read_csv(amap_sets_csv)
    sets = sets[(sets.variant == "raw") & (sets.set_name == "passes_final_qc_top_0p5") & np.isclose(sets.threshold, 0.5)]
    out = []
    for r in sections.itertuples():
        row = sets[(sets.brain == r.brain) & (sets.structure == r.structure)]
        if row.empty:
            continue
        A = int(row.A_best.iloc[0])
        cons = tifffile.imread(Path(amap_dir) / r.brain / f"{r.brain}_{r.structure}_consensus0p5_25um.tif")[A] > 0
        truth = tifffile.imread(Path(amap_dir) / "expert_STAPLE" / r.file) > 0
        keep = np.zeros(cons.shape[1], bool)
        keep[r.annotated_hemisphere_R_from:r.annotated_hemisphere_R_to + 1] = True
        m = cons & keep[None, :]
        d = 2 * (m & truth).sum() / (m.sum() + truth.sum())
        out.append(dict(brain=r.brain, structure=r.structure, dice_recomputed=d,
                        dice_published=float(row.dice_paper.iloc[0]), A_best=A))
    return pd.DataFrame(out)
