"""STAPLE consensus, individual rater annotations and rater agreement placed in downscaled-brain space.

For each brain x structure, on the grid and header of its downscaledBrains/*.nii volume:
* STAPLE mask on the median rater slice (``J0``),
* rater-agreement probability (fraction of raters) on the same plane,
* every rater annotation on its own slice and on the J0 plane,
plus one label map per brain. See PLAN_inBrainSpace.md and the README for the conventions.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import masks as M
from . import nifti
from .config import Config, matlab_num2str
from .geometry import StackMapping, canvas_to_volume, matlab_round, stack_plane_from_volume
from .qc import PAPER_NAME
from .roi import read_roi
from .scoring import bwperim8, dice
from .staple import matlab_quantile

VOXEL_UM = 12.5


def stack_name(tif_path: str) -> str:
    """'…/Seg37AB_Gad67_223_rspV1V2M.tif001-RSP_Slice23..tif' -> 'rspV1V2M'."""
    name = str(tif_path).replace("\\", "/").split("/")[-1]
    return name[: name.index(".tif")].rsplit("_", 1)[1]


def roi_path(tif_path: str) -> str:
    p = str(tif_path)
    return (p[:-5] if p.endswith("..tif") else p[:-4]) + ".roi"


def label_ids(cfg: Config) -> dict[str, int]:
    """Label IDs 1..10 in globalData.yml regionsUsed order."""
    return {r.abbrev: k for k, r in enumerate(cfg.regions, 1)}


def _plane_boundary_tree(voxels):
    """KD-tree of the 8-connected boundary of a plane mask, in (x, z) voxel coordinates."""
    xz = voxels[:, [0, 2]]
    lo = xz.min(0) - 1
    img = np.zeros(xz.max(0) - lo + 2, bool)
    img[tuple((xz - lo).T)] = True
    return cKDTree(np.argwhere(bwperim8(img)) + lo)


def _vertex_distances_um(tree, verts_nii):
    return tree.query(verts_nii[:, [0, 2]])[0] * VOXEL_UM


def _rater_vertices(tif_path):
    paths, _ = read_roi(roi_path(tif_path))
    return np.vstack(paths) if paths else None


def _write(path: Path, grid, dtype, voxels, values):
    shape, affine = grid
    path.parent.mkdir(parents=True, exist_ok=True)
    nifti.save_sparse(path, shape, dtype, voxels, values, affine)


def _header_matches(path: Path, grid) -> bool:
    h = nifti.read_header(path)
    return h["shape"][:3] == tuple(grid[0]) and np.allclose(h["sform"], grid[1]) and h["sform_code"] == 1


def place_combo(cfg: Config, brc: str, cur: pd.DataFrame, canvas, cache: Path, staple_dir: Path, dest: Path,
                grid, with_raters: bool = True) -> dict:
    """Write the brain-space outputs for one brain x structure and return its sidecar/check record."""
    sc = matlab_num2str(cfg.scale)
    shape = tuple(grid[0])
    brain, region = cur["brain"].iloc[0], cur["region"].iloc[0]
    stack = stack_name(cur["path"].iloc[0])
    mp = StackMapping(cfg.stack(brain, stack), cfg.scale)
    slices = cur["sliceNum"].to_numpy(float)
    j0 = float(np.median(slices))

    # STAPLE on the median plane
    st = nifti.load(staple_dir / f"{brc}_{sc}_NiftySegSTAPLE.nii")
    vox, _ = canvas_to_volume(st, mp, j0, shape)
    f_st = dest / f"{brc}_{sc}_STAPLE_inBrain.nii.gz"
    _write(f_st, grid, np.uint8, vox, np.ones(len(vox)))

    # rater-agreement probability on the same plane (the pooled input STAPLE sees)
    cmasks = [M.cached_masks(p, canvas, cfg.scale, cache)[0] for p in cur["path"]]
    n = len(cmasks)
    prob = np.mean(np.stack(cmasks), axis=0).astype(np.float32)
    pvox, pval = canvas_to_volume(prob, mp, j0, shape)
    f_pr = dest / f"{brc}_{sc}_raterProbability_inBrain.nii.gz"
    _write(f_pr, grid, np.float32, pvox, pval)

    # checks 1-4 (STAPLE)
    rb_vox, _ = nifti.load_sparse(f_st)
    same_file = len(rb_vox) == len(vox) and set(map(tuple, rb_vox)) == set(map(tuple, vox))
    cols = len({(x, z) for x, _, z in vox})
    area_canvas_um2 = float(st.sum()) * (1.0 / cfg.scale) ** 2  # native TIFF pixel = 1 um
    area_vol_um2 = len(vox) * VOXEL_UM ** 2
    back = stack_plane_from_volume(vox, np.ones(len(vox), np.uint8), mp, j0, st.shape)
    ci, cj = np.argwhere(st).mean(0) if st.any() else (np.nan, np.nan)
    cen = mp.staple_to_nii(ci, cj, j0)
    stat_names = ["min", "q25", "median", "q75", "max"]
    stat_vals = [slices.min(), *matlab_quantile(slices, [0.25]), j0, *matlab_quantile(slices, [0.75]), slices.max()]
    fg = np.argwhere(st)
    fg_nii = np.floor(mp.staple_to_nii(fg[:, 0], fg[:, 1], j0) + 0.5) if len(fg) else np.zeros((0, 3))
    off_grid = int(np.sum(np.any((fg_nii < 0) | (fg_nii >= np.array(shape)), axis=1)))

    # raters (both variants), checks 5, 7, 8, 9
    tree_st = _plane_boundary_tree(vox) if len(vox) else None
    rater_rows, d_st, d_own, dy_own, rt_dice = [], [], [], [], []
    count = {}
    for row, cm in zip(cur.itertuples(), cmasks):
        verts = _rater_vertices(row.path)
        vnii = mp.tiff_to_nii(verts[:, 0], verts[:, 1], row.sliceNum) if verts is not None else None
        if vnii is not None and tree_st is not None:
            d_st.append(np.median(_vertex_distances_um(tree_st, vnii)))
        rec = dict(user=row.user, idx=int(row.idx), attempt=int(row.attempt), slice=int(row.sliceNum), inStaple=True)
        for variant, jv in (("ownSlice", float(row.sliceNum)), ("medianPlane", j0)):
            rv, _ = canvas_to_volume(cm, mp, jv, shape)
            name = f"{row.user}_Seg{int(row.idx)}_slice{int(row.sliceNum)}_{variant}_inBrain.nii.gz"
            if with_raters:
                _write(dest / "raters" / brc / name, grid, np.uint8, rv, np.ones(len(rv)))
            rec[f"{variant}_file"] = name if with_raters else None
            rec[f"{variant}_J"] = jv
            rec[f"{variant}_voxels"] = int(len(rv))
            rec[f"{variant}_y_range"] = [int(rv[:, 1].min()), int(rv[:, 1].max())] if len(rv) else None
            if variant == "ownSlice" and len(rv) and vnii is not None:
                d_own.append(np.median(_vertex_distances_um(_plane_boundary_tree(rv), vnii)))
                # distance of the rater's own vertices from the voxel plane actually written (rounded y)
                y_written = matlab_round(mp.plane_y(np.round(vnii[:, 0]), np.round(vnii[:, 2]), jv))
                dy_own.append(float(np.median(np.abs(vnii[:, 1] - y_written))) * VOXEL_UM)
            if variant == "medianPlane":
                back_r = stack_plane_from_volume(rv, np.ones(len(rv), np.uint8), mp, j0, cm.shape)
                rt_dice.append(dice(back_r, cm))
                for v in map(tuple, rv):
                    count[v] = count.get(v, 0) + 1
        rater_rows.append(rec)
    if with_raters:
        (dest / "raters" / brc).mkdir(parents=True, exist_ok=True)
        (dest / "raters" / brc / "raters.json").write_text(json.dumps(
            dict(brainRegionComb=brc, brain_file=cfg.brain_file(brain).name, stack=stack, J0=j0, raters=rater_rows),
            indent=1))
    pmap = dict(zip(map(tuple, pvox), pval))
    prob_exact = set(pmap) == set(count) and all(abs(pmap[v] - np.float32(count[v] / n)) < 1e-6 for v in count)
    maj = {v for v, p in pmap.items() if p > 0.5}
    stset = set(map(tuple, vox))
    dice_maj = 2 * len(maj & stset) / (len(maj) + len(stset)) if (maj or stset) else float("nan")
    rater_headers_ok = all(
        _header_matches(dest / "raters" / brc / r[f"{v}_file"], grid)
        for r in rater_rows for v in ("ownSlice", "medianPlane")) if with_raters else None

    info = dict(
        brainRegionComb=brc, brain=brain, region=region, stack=stack, brain_file=cfg.brain_file(brain).name,
        grid_shape=list(shape), n_raters=n, raters=[f"{r['user']}:{r['idx']}" for r in rater_rows],
        J0_median_slice=j0,
        slice_stats={k: float(v) for k, v in zip(stat_names, stat_vals)},
        slice_stats_nii_y={k: float(mp.staple_to_nii(ci, cj, v)[1]) for k, v in zip(stat_names, stat_vals)},
        staple_voxels=int(len(vox)), staple_plane_y_range=[int(vox[:, 1].min()), int(vox[:, 1].max())] if len(vox) else None,
        staple_centroid_voxel=[float(c) for c in cen],
        staple_centroid_world_mm=[float(c) for c in (grid[1] @ np.r_[cen, 1.0])[:3]],
        area_canvas_um2=area_canvas_um2, area_volume_um2=area_vol_um2,
        area_ratio=area_vol_um2 / area_canvas_um2 if area_canvas_um2 else float("nan"),
        canvas_pixels_off_grid=off_grid,
        # checks
        check1_header_ok=_header_matches(f_st, grid) and _header_matches(f_pr, grid) and same_file,
        check2_one_voxel_per_column=cols == len(vox),
        check4_roundtrip_dice=dice(back, st),
        check5_rater_vertex_to_staple_boundary_um_median=float(np.median(d_st)) if d_st else float("nan"),
        check7_own_vertex_to_own_boundary_um_median=float(np.median(d_own)) if d_own else float("nan"),
        check7_own_vertex_off_plane_um_median=float(np.median(dy_own)) if dy_own else float("nan"),
        check7_rater_headers_ok=rater_headers_ok,
        check8_median_plane_roundtrip_dice_min=float(np.min(rt_dice)),
        check8_median_plane_roundtrip_dice_median=float(np.median(rt_dice)),
        check9_probability_equals_rater_mean=bool(prob_exact),
        check9_probability_max=float(pval.max()) if len(pval) else 0.0,
        check9_dice_majority_vs_staple=dice_maj,
    )
    (dest / f"{brc}_{sc}_STAPLE_inBrain.json").write_text(json.dumps(info, indent=1))
    info["_voxels"] = vox
    return info


def write_label_map(cfg: Config, brain: str, items, grid, dest: Path):
    """items: list of (label_id, abbrev, voxels). First ID (regionsUsed order) wins where structures overlap."""
    nx, ny, _ = grid[0]
    items = sorted(items, key=lambda t: t[0])
    lin = np.concatenate([v[:, 0] + nx * (v[:, 1] + ny * v[:, 2]) for _, _, v in items]) if items else np.zeros(0, int)
    lab = np.concatenate([np.full(len(v), k, np.uint8) for k, _, v in items]) if items else np.zeros(0, np.uint8)
    uniq, first = np.unique(lin, return_index=True)  # first occurrence = lowest ID (sorted above)
    voxels = np.stack([uniq % nx, (uniq // nx) % ny, uniq // (nx * ny)], axis=1)
    path = dest / f"{brain}_STAPLE_labels.nii.gz"
    _write(path, grid, np.uint8, voxels, lab[first])
    names = {r.abbrev: r for r in cfg.regions}
    meta = dict(brain=brain, brain_file=cfg.brain_file(brain).name, overlap_rule="lowest label ID wins",
                overlap_voxels=int(len(lin) - len(uniq)),
                labels=[dict(id=k, abbrev=a, paper_name=PAPER_NAME[a], allen_id=names[a].allen_id,
                             allen_name=names[a].allen_name, voxels=int(len(v))) for k, a, v in items],
                included=[a for _, a, _ in items], complete=len(items) == len(cfg.regions))
    (dest / f"{brain}_STAPLE_labels.json").write_text(json.dumps(meta, indent=1))
    return meta


def run_to_brain(cfg: Config, opts, with_raters: bool = True, log=print) -> pd.DataFrame:
    staple_dir = opts.staple_dir()
    dest = staple_dir / "inBrainSpace"
    dest.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(staple_dir / "segTable_with_pq.csv")
    table = table[table["inStaple"]]
    canvas = tuple(json.loads((staple_dir / "canvas.json").read_text())["maxX_maxY"])
    cache = opts.cache_dir(cfg.scale)
    combos = sorted(table["brainRegionComb"].unique())
    if opts.only:
        combos = [c for c in combos if c in opts.only]
    ids = label_ids(cfg)
    grids, labels, rows = {}, {}, []
    logf = open(dest / "tobrain.log", "a", encoding="utf-8")

    def say(msg):
        log(msg)
        logf.write(msg + "\n")
        logf.flush()

    say(f"== tobrain {time.strftime('%Y-%m-%d %H:%M:%S')}: {len(combos)} brain x structure, raters={with_raters}")
    t_all = time.time()
    try:
        for k, brc in enumerate(combos, 1):
            t0 = time.time()
            cur = table[table["brainRegionComb"] == brc]
            brain = cur["brain"].iloc[0]
            if brain not in grids:
                h = nifti.read_header(cfg.brain_file(brain))
                grids[brain] = (h["shape"][:3], h["sform"])
            info = place_combo(cfg, brc, cur, canvas, cache, staple_dir, dest, grids[brain], with_raters)
            labels.setdefault(brain, []).append((ids[info["region"]], info["region"], info.pop("_voxels")))
            rows.append(info)
            say(f"  [{k}/{len(combos)}] {brc:22s} n={info['n_raters']:2d} J0={info['J0_median_slice']:5.1f} "
                f"vox={info['staple_voxels']:5d} area={info['area_ratio']:.3f} rt={info['check4_roundtrip_dice']:.3f} "
                f"prob_ok={info['check9_probability_equals_rater_mean']} ({time.time() - t0:.0f}s)")
        for brain, items in labels.items():
            meta = write_label_map(cfg, brain, items, grids[brain], dest)
            say(f"  label map {brain}: {len(items)} structures, overlap voxels {meta['overlap_voxels']}")
    finally:
        say(f"== done in {time.time() - t_all:.0f}s")
        logf.close()
    summary = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in rows])
    path = dest / "inBrain_summary.csv"
    if opts.only and path.exists():  # keep results of earlier runs for other structures
        old = pd.read_csv(path)
        summary = pd.concat([old[~old["brainRegionComb"].isin(summary["brainRegionComb"])], summary])
    summary.sort_values("brainRegionComb").to_csv(path, index=False)
    return summary
