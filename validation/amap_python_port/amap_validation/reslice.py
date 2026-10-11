"""Reslice the downscaled brains and the brain-space consensus products to coronal BrainGlobe ``asr`` at 25 um.

``asr`` = axis 0 anterior -> posterior (coronal slices), axis 1 superior -> inferior, axis 2 right -> left
(the allen_mouse_25um convention). The downscaled brains are [x, y (+ anterior), z (+ dorsal)] at 12.5 um and
were rigidly aligned to the Allen average, so the reslice is a pure permutation + flips and 12.5 -> 25 um is an
exact 2 x 2 x 2 block reduction. All products go through the same ``AsrGrid``:

    a = ny - 1 - y,  s = nz - 1 - z,  r = nx - 1 - x    (asr at 12.5 um, mirrored left-right by default:
    A = a // 2,      S = s // 2,      R = r // 2         r increases AWAY from the segmented hemisphere)

The left-right mirror (``mirror_lr=True``) is a display/matching choice requested by the user; the anatomical
L/R labelling of the original x axis is only inferred (see the README). ``mirror_lr=False`` gives r = x.
Structures are written as 2-D coronal images (one asr slice each) plus a slice-index log. See PLAN_asr25um.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from . import nifti
from .config import FLIPPED_HEMISPHERE, MIDLINES, Config, matlab_num2str

ORIENTATION = "asr"
VOXEL_UM = 25.0


class AsrGrid:
    """Index maps and affines between a brain's original 12.5 um grid and its asr grids (12.5 and 25 um)."""

    def __init__(self, shape12, sform12, mirror_lr: bool = True):
        self.nx, self.ny, self.nz = (int(n) for n in shape12[:3])
        self.mirror_lr = bool(mirror_lr)
        self.sform12 = np.asarray(sform12, float)
        self.shape_asr12 = (self.ny, self.nz, self.nx)
        self.shape25 = tuple(-(-n // 2) for n in self.shape_asr12)
        # block centre (A, S, R) -> original 0-based voxel (x, y, z)
        x_row = [0, 0, -2, self.nx - 1.5] if self.mirror_lr else [0, 0, 2, 0.5]
        self.P25 = np.array([x_row,
                             [-2, 0, 0, self.ny - 1.5],
                             [0, -2, 0, self.nz - 1.5],
                             [0, 0, 0, 1]], float)
        self.affine25 = self.sform12 @ self.P25

    def _r(self, x):
        return self.nx - 1 - x if self.mirror_lr else x

    # 0-based index maps (the x <-> r map is its own inverse)
    def to_asr12(self, v):
        v = np.asarray(v)
        return np.stack([self.ny - 1 - v[..., 1], self.nz - 1 - v[..., 2], self._r(v[..., 0])], axis=-1)

    def from_asr12(self, u):
        u = np.asarray(u)
        return np.stack([self._r(u[..., 2]), self.ny - 1 - u[..., 0], self.nz - 1 - u[..., 1]], axis=-1)

    def to_asr25(self, v):
        return self.to_asr12(v) // 2

    def slice_affine(self, A: int) -> np.ndarray:
        """Affine of a (1, nS, nR) image that is asr slice A of the 25 um grid."""
        aff = self.affine25.copy()
        aff[:, 3] = self.affine25 @ np.array([A, 0, 0, 1.0])
        return aff

    def to_json(self, brain: str, brain_file: str) -> dict:
        return dict(
            brain=brain, brain_file=brain_file, orientation=ORIENTATION,
            axes={"0": "anterior -> posterior (coronal slice index A)", "1": "superior -> inferior (S)",
                  "2": ("R: mirrored left-right relative to the original x; increases AWAY from the segmented "
                        "hemisphere" if self.mirror_lr else "R = original x; increases toward the segmented hemisphere")},
            mirror_lr=self.mirror_lr,
            mirror_note="left-right mirror is a display/matching choice; anatomical L/R of the original x is inferred",
            original_shape_xyz=[self.nx, self.ny, self.nz], original_voxel_um=12.5,
            asr_shape_12um5=list(self.shape_asr12), asr_shape_25um=list(self.shape25), voxel_um=VOXEL_UM,
            index_map_to_asr_12um5="a = ny-1-y, s = nz-1-z, " + ("r = nx-1-x" if self.mirror_lr else "r = x"),
            index_map_to_asr_25um="A = a//2, S = s//2, R = r//2 (odd dims: last block has one voxel)",
            P_block_centre_to_original_voxel=self.P25.tolist(),
            original_sform=self.sform12.tolist(), asr_25um_affine=self.affine25.tolist(),
        )


def _block_sums_2d(arr):
    """Sum over 2 x 2 blocks of a 2-D array (odd edges -> partial blocks) and the voxel count per block."""
    n0, n1 = arr.shape
    p = np.zeros((n0 + n0 % 2, n1 + n1 % 2), dtype=np.float64)
    p[:n0, :n1] = arr
    c = np.zeros_like(p)
    c[:n0, :n1] = 1
    s = p.reshape(p.shape[0] // 2, 2, p.shape[1] // 2, 2).sum(axis=(1, 3))
    k = c.reshape(c.shape[0] // 2, 2, c.shape[1] // 2, 2).sum(axis=(1, 3))
    return s, k


def _to_dtype(x, dtype):
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.floor(x + 0.5), info.min, info.max).astype(dtype)
    return x.astype(dtype)


def downsample_to_asr25(vol, grid: AsrGrid, dtype=None, log=None) -> np.ndarray:
    """2 x 2 x 2 block mean of an (nx, ny, nz) volume (ndarray or memmap) in the asr 25 um grid (A, S, R).

    Streams over pairs of original z planes (= one S row of the output), so only two planes are in memory.
    Means are over the valid voxels of each block, so odd dimensions do not bias the edge blocks.
    """
    dtype = np.dtype(dtype or vol.dtype)
    nA, nS, nR = grid.shape25
    out = np.zeros((nA, nS, nR), dtype)
    t0 = time.time()
    for S in range(nS):
        z_hi = grid.nz - 1 - 2 * S
        z_lo = max(z_hi - 1, 0)
        chunk = np.asarray(vol[:, :, z_lo:z_hi + 1], dtype=np.float64)  # (nx, ny, 1..2)
        zsum, zcnt = chunk.sum(axis=2), chunk.shape[2]
        plane_ar = zsum[:, ::-1].T  # [a, r]: a = ny-1-y, r = x ...
        if grid.mirror_lr:
            plane_ar = plane_ar[:, ::-1]  # ... or r = nx-1-x
        s, k = _block_sums_2d(plane_ar)
        out[:, S, :] = _to_dtype(s / (k * zcnt), dtype)
        if log and (S % 50 == 0 or S == nS - 1):
            log(f"    S {S + 1}/{nS} ({time.time() - t0:.0f}s)")
    return out


def sheet_to_asr25(voxels, values, grid: AsrGrid, binary: bool):
    """A one-voxel-thick coronal sheet (sparse 12.5 um voxels) -> 2-D (nS, nR) image on the 25 um grid.

    The sheet is projected along a (one voxel per (s, r) column, see geometry.canvas_to_volume), then reduced
    over 2 x 2 in-plane blocks: block mean for probabilities, mean >= 0.5 for binary masks.
    Returns (image, A_median, sorted list of A values touched, a-range at 12.5 um); A_median is None if the
    sheet is empty.
    """
    _, nS, nR = grid.shape25
    voxels = np.asarray(voxels, int).reshape(-1, 3)
    if len(voxels) == 0:
        return np.zeros((nS, nR), np.uint8 if binary else np.float32), None, [], None
    u = grid.to_asr12(voxels)
    img12 = np.zeros((grid.nz, grid.nx), np.float64)
    img12[u[:, 1], u[:, 2]] = values
    s, k = _block_sums_2d(img12)
    mean = s / k
    img = (mean >= 0.5).astype(np.uint8) if binary else mean.astype(np.float32)
    a = u[:, 0]
    A_all = sorted(set((a // 2).tolist()))
    A_med = int(np.median(a)) // 2
    return img[:nS, :nR], A_med, A_all, (int(a.min()), int(a.max()))


def save_tiff(path: Path, data: np.ndarray):
    """ImageJ TIFF (pages = asr slices for 3-D) with 25 um spacing in the metadata."""
    data = np.asarray(data)
    meta = {"spacing": VOXEL_UM, "unit": "um", "axes": "ZYX" if data.ndim == 3 else "YX",
            "orientation": ORIENTATION}
    tifffile.imwrite(path, data, imagej=True, resolution=(1 / VOXEL_UM, 1 / VOXEL_UM), metadata=meta)


class AsrLayout:
    """All output paths of the asr 25 um stage (one brain folder, one sub-folder per structure)."""

    def __init__(self, dest: Path, gzip: bool = False):
        self.dest = Path(dest)
        self.ext = ".nii.gz" if gzip else ".nii"

    def rel(self, path: Path) -> str:
        return Path(path).relative_to(self.dest).as_posix()

    def _f(self, folder: Path, stem: str, tif: bool) -> Path:
        return folder / (stem + (".tif" if tif else self.ext))

    def brain_dir(self, brain):
        return self.dest / brain

    def brain_img(self, brain, tif=False):
        return self._f(self.brain_dir(brain), f"{brain}_asr_25um", tif)

    def labels(self, brain, tif=False):
        return self._f(self.brain_dir(brain), f"{brain}_STAPLE_labels_asr_25um", tif)

    def transform(self, brain):
        return self.brain_dir(brain) / f"{brain}_transform.json"

    def checks(self, brain):
        return self.brain_dir(brain) / f"asr25um_checks_{brain}.json"

    def struct_dir(self, brain, region):
        from .qc import PAPER_NAME
        return self.brain_dir(brain) / PAPER_NAME[region]

    def staple(self, brain, region, tif=False):
        return self._f(self.struct_dir(brain, region), f"{brain}{region}_STAPLE_asr25um", tif)

    def prob(self, brain, region, tif=False):
        return self._f(self.struct_dir(brain, region), f"{brain}{region}_raterProbability_asr25um", tif)

    def rater(self, brain, region, variant, user, idx, attempt, slice_num, tif=False):
        folder = self.struct_dir(brain, region) / "raters" / variant
        return self._f(folder, f"{user}_Seg{int(idx)}_att{int(attempt)}_slice{int(slice_num)}_{variant}_asr25um", tif)

    @property
    def slices_log(self):
        return self.dest / "asr25um_slices.csv"

    @property
    def raters_log(self):
        return self.dest / "asr25um_raters.csv"

    @property
    def master_log(self):
        return self.dest / "asr25um_all_masks.csv"


def _save_2d(nii_path: Path, tif_path: Path, arr: np.ndarray, affine: np.ndarray):
    nii_path.parent.mkdir(parents=True, exist_ok=True)
    nifti.save(nii_path, arr[None], affine)  # (1, S, R) placed at its slice by the affine
    save_tiff(tif_path, arr)


def _write_table(df: pd.DataFrame, csv_path: Path):
    df.to_csv(csv_path, index=False)
    csv_path.with_suffix(".json").write_text(df.to_json(orient="records", indent=1))


def _merge_rows(path: Path, new: pd.DataFrame, brains) -> pd.DataFrame:
    """Replace the rows of the given brains in an existing table (runs can be done brain by brain)."""
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["brain"].isin(brains)], new], ignore_index=True)
    return new


def run_asr25(cfg: Config, opts, brains=None, gzip: bool = False, stamp: bool = False, log=print) -> pd.DataFrame:
    """Brains, STAPLE and probability images at asr 25 um, plus the structure-level slice log."""
    from .qc import PAPER_NAME
    from .tobrain import label_ids
    staple_dir = opts.staple_dir()
    src = staple_dir / "inBrainSpace"
    lay = AsrLayout(staple_dir / "asr25um", gzip)
    sc = matlab_num2str(cfg.scale)
    ids = label_ids(cfg)
    brains = list(brains or cfg.brains)
    rows = []
    for brain in brains:
        t0 = time.time()
        h = nifti.read_header(cfg.brain_file(brain))
        grid = AsrGrid(h["shape"], h["sform"])
        lay.brain_dir(brain).mkdir(parents=True, exist_ok=True)
        lay.transform(brain).write_text(json.dumps(grid.to_json(brain, cfg.brain_file(brain).name), indent=1))
        log(f"{brain}: {h['shape']} -> asr 25 um {grid.shape25}")
        vol = nifti.load(cfg.brain_file(brain), mmap=True)
        img = downsample_to_asr25(vol, grid, h["dtype"].newbyteorder("="), log=log)
        nifti.save(lay.brain_img(brain), img, grid.affine25)
        save_tiff(lay.brain_img(brain, tif=True), img)
        log(f"  brain written ({time.time() - t0:.0f}s)")
        for region, lab in ids.items():
            brc = brain + region
            f_st = src / f"{brc}_{sc}_STAPLE_inBrain.nii.gz"
            if not f_st.exists():
                continue
            info = json.loads((src / f"{brc}_{sc}_STAPLE_inBrain.json").read_text())
            st_vox, _ = nifti.load_sparse(f_st)
            pr_vox, pr_val = nifti.load_sparse(src / f"{brc}_{sc}_raterProbability_inBrain.nii.gz")
            m25, A, A_all, a_rng = sheet_to_asr25(st_vox, np.ones(len(st_vox)), grid, binary=True)
            p25, _, _, _ = sheet_to_asr25(pr_vox, pr_val, grid, binary=False)
            aff = grid.slice_affine(A)
            _save_2d(lay.staple(brain, region), lay.staple(brain, region, tif=True), m25, aff)
            _save_2d(lay.prob(brain, region), lay.prob(brain, region, tif=True), p25, aff)
            rows.append(dict(
                brain=brain, brainRegionComb=brc, region=region, label_id=lab, paper_name=PAPER_NAME[region],
                n_raters=info["n_raters"], J0_median_slice=info["J0_median_slice"],
                slice_min=info["slice_stats"]["min"], slice_q25=info["slice_stats"]["q25"],
                slice_q75=info["slice_stats"]["q75"], slice_max=info["slice_stats"]["max"],
                y12_range=f"{info['staple_plane_y_range'][0]}-{info['staple_plane_y_range'][1]}",
                a12_range=f"{a_rng[0]}-{a_rng[1]}", asr25_slice_A=A, asr25_slices_touched=" ".join(map(str, A_all)),
                world_y_mm_of_slice=float((grid.affine25 @ np.array([A, 0, 0, 1.0]))[1]),
                area25_um2=float(m25.sum()) * VOXEL_UM ** 2, area_canvas_um2=info["area_canvas_um2"],
                area_ratio=float(m25.sum()) * VOXEL_UM ** 2 / info["area_canvas_um2"],
                staple_file=lay.rel(lay.staple(brain, region)),
                probability_file=lay.rel(lay.prob(brain, region))))
        log(f"  {brain} done ({time.time() - t0:.0f}s)")
    table = _merge_rows(lay.slices_log, pd.DataFrame(rows), brains).sort_values(["brain", "label_id"])
    _write_table(table, lay.slices_log)
    if stamp:  # needs the slice log written above
        for brain in brains:
            stamp_labels(cfg, brain, lay.dest, gzip=gzip, write=True)
            log(f"  {brain}: 3-D label volume stamped")
    write_master_log(cfg, lay)
    return table


def run_raters_asr25(cfg: Config, opts, brains=None, gzip: bool = False, log=print) -> pd.DataFrame:
    """Every rater mask at asr 25 um, on its own slice and on the STAPLE (median) slice, with a log."""
    from .geometry import StackMapping, canvas_to_volume
    from . import masks as M
    from .qc import PAPER_NAME
    from .tobrain import label_ids, stack_name
    staple_dir = opts.staple_dir()
    src = staple_dir / "inBrainSpace"
    lay = AsrLayout(staple_dir / "asr25um", gzip)
    sc = matlab_num2str(cfg.scale)
    ids = label_ids(cfg)
    seg = pd.read_csv(staple_dir / "segTable_with_pq.csv")
    seg = seg[seg["inStaple"]]
    canvas = tuple(json.loads((staple_dir / "canvas.json").read_text())["maxX_maxY"])
    cache = opts.cache_dir(cfg.scale)
    slices = pd.read_csv(lay.slices_log).set_index("brainRegionComb")
    brains = list(brains or cfg.brains)
    rows = []
    for brain in brains:
        t0 = time.time()
        h = nifti.read_header(cfg.brain_file(brain))
        grid = AsrGrid(h["shape"], h["sform"])
        shape12 = h["shape"][:3]
        for brc, cur in seg[seg["brain"] == brain].groupby("brainRegionComb"):
            region = cur["region"].iloc[0]
            info = json.loads((src / f"{brc}_{sc}_STAPLE_inBrain.json").read_text())
            j0, a_staple = float(info["J0_median_slice"]), int(slices.loc[brc, "asr25_slice_A"])
            mp = StackMapping(cfg.stack(brain, stack_name(cur["path"].iloc[0])), cfg.scale)
            for r in cur.itertuples():
                cm = M.cached_masks(r.path, canvas, cfg.scale, cache)[0]
                area_canvas = float(cm.sum()) * (1.0 / cfg.scale) ** 2
                for variant, jv in (("ownSlice", float(r.sliceNum)), ("medianPlane", j0)):
                    vox, _ = canvas_to_volume(cm, mp, jv, shape12)
                    img, A, A_all, _ = sheet_to_asr25(vox, np.ones(len(vox)), grid, binary=True)
                    if A is None:  # nothing landed on the grid; keep an empty image on the STAPLE slice
                        A = a_staple
                    f = lay.rater(brain, region, variant, r.user, r.idx, r.attempt, r.sliceNum)
                    _save_2d(f, lay.rater(brain, region, variant, r.user, r.idx, r.attempt, r.sliceNum, tif=True),
                             img, grid.slice_affine(A))
                    rows.append(dict(
                        brain=brain, brainRegionComb=brc, region=region, label_id=ids[region],
                        paper_name=PAPER_NAME[region], variant=variant, user=r.user, idx=int(r.idx),
                        attempt=int(r.attempt), rater_stack_slice=int(r.sliceNum), J_used=jv,
                        asr25_slice_A=int(A), asr25_slices_touched=" ".join(map(str, A_all)),
                        staple_slice_A=a_staple, A_minus_staple_A=int(A) - a_staple,
                        area25_um2=float(img.sum()) * VOXEL_UM ** 2, area_canvas_um2=area_canvas,
                        area_ratio=float(img.sum()) * VOXEL_UM ** 2 / area_canvas if area_canvas else np.nan,
                        file=lay.rel(f)))
        log(f"  {brain}: {sum(1 for x in rows if x['brain'] == brain)} rater masks ({time.time() - t0:.0f}s)")
    table = _merge_rows(lay.raters_log, pd.DataFrame(rows), brains)
    table = table.sort_values(["brain", "label_id", "variant", "user", "idx"])
    _write_table(table, lay.raters_log)
    write_master_log(cfg, lay)
    return table


def write_master_log(cfg: Config, lay: AsrLayout) -> pd.DataFrame:
    """One row per written image: brain stacks, label volumes, STAPLE, probability and rater masks."""
    rows = []
    for brain in cfg.brains:
        for kind, path in (("brain", lay.brain_img(brain)), ("labels_3d", lay.labels(brain))):
            if path.exists():
                rows.append(dict(kind=kind, brain=brain, file=lay.rel(path), tif=lay.rel(path.with_name(
                    path.name.replace(lay.ext, ".tif")))))
    if lay.slices_log.exists():
        for r in pd.read_csv(lay.slices_log).itertuples():
            for kind, rel in (("STAPLE", r.staple_file), ("probability", r.probability_file)):
                rows.append(dict(kind=kind, brain=r.brain, region=r.region, label_id=r.label_id,
                                 paper_name=r.paper_name, n_raters=r.n_raters, asr25_slice_A=r.asr25_slice_A,
                                 asr25_slices_touched=r.asr25_slices_touched, area25_um2=r.area25_um2,
                                 area_ratio=r.area_ratio, file=rel, tif=rel.replace(lay.ext, ".tif")))
    if lay.raters_log.exists():
        for r in pd.read_csv(lay.raters_log).itertuples():
            rows.append(dict(kind=f"rater_{r.variant}", brain=r.brain, region=r.region, label_id=r.label_id,
                             paper_name=r.paper_name, user=r.user, idx=r.idx, attempt=r.attempt,
                             rater_stack_slice=r.rater_stack_slice, asr25_slice_A=r.asr25_slice_A,
                             asr25_slices_touched=r.asr25_slices_touched, area25_um2=r.area25_um2,
                             area_ratio=r.area_ratio, file=r.file, tif=r.file.replace(lay.ext, ".tif")))
    df = pd.DataFrame(rows)
    _write_table(df, lay.master_log)
    return df


def migrate_legacy_layout(dest: Path, log=print) -> int:
    """Move outputs of the first asr25 layout (brains/, structures/) into the per-brain/per-structure layout."""
    from .qc import PAPER_NAME
    lay = AsrLayout(dest)
    moved = 0
    for old_dir in (dest / "brains", dest / "structures"):
        if not old_dir.exists():
            continue
        for f in sorted(old_dir.iterdir()):
            name = f.name
            brain = next((b for b in MIDLINES if name.startswith(b)), None)
            if brain is None:
                continue
            if old_dir.name == "brains":
                target = lay.brain_dir(brain) / name
            else:
                rest = name[len(brain):]
                region = next(r for r in PAPER_NAME if rest.startswith(r + "_"))
                target = lay.struct_dir(brain, region) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            f.replace(target)
            moved += 1
        if not any(old_dir.iterdir()):
            old_dir.rmdir()
    for f in dest.glob("asr25um_checks_*.json"):
        brain = f.stem.replace("asr25um_checks_", "")
        f.replace(lay.checks(brain))
        moved += 1
    if lay.slices_log.exists():  # rewrite file columns to the new relative paths
        t = pd.read_csv(lay.slices_log)
        t["staple_file"] = [lay.rel(lay.staple(b, r)) for b, r in zip(t["brain"], t["region"])]
        t["probability_file"] = [lay.rel(lay.prob(b, r)) for b, r in zip(t["brain"], t["region"])]
        _write_table(t, lay.slices_log)
    log(f"  migrated {moved} files to the per-brain layout")
    return moved


def stamp_labels(cfg: Config, brain: str, dest: Path, gzip: bool = False, write: bool = False):
    """Rebuild a 3-D asr 25 um label volume from the 2-D structure images and the slice log (lowest ID wins)."""
    lay = AsrLayout(dest, gzip)
    h = nifti.read_header(cfg.brain_file(brain))
    grid = AsrGrid(h["shape"], h["sform"])
    log_df = pd.read_csv(lay.slices_log)
    vol = np.zeros(grid.shape25, np.uint8)
    for row in log_df[log_df["brain"] == brain].sort_values("label_id", ascending=False).itertuples():
        m = nifti.load(lay.dest / row.staple_file)[0]
        vol[row.asr25_slice_A][m > 0] = row.label_id
    if write:
        nifti.save(lay.labels(brain), vol, grid.affine25)
        save_tiff(lay.labels(brain, tif=True), vol)
    return vol


def verify_asr25(cfg: Config, opts, brain: str, n_blocks: int = 4000, seed: int = 0, gzip: bool = False) -> dict:
    """Checks for one brain's asr 25 um products (read back from disk), written to <brain>/asr25um_checks_<brain>.json."""
    staple_dir = opts.staple_dir()
    src = staple_dir / "inBrainSpace"
    lay = AsrLayout(staple_dir / "asr25um", gzip)
    sc = matlab_num2str(cfg.scale)
    h = nifti.read_header(cfg.brain_file(brain))
    grid = AsrGrid(h["shape"], h["sform"])
    orig = nifti.load(cfg.brain_file(brain), mmap=True)
    hb = nifti.read_header(lay.brain_img(brain))
    img = nifti.load(lay.brain_img(brain), mmap=not gzip)
    with tifffile.TiffFile(lay.brain_img(brain, tif=True)) as tif:
        tif_ok = (tif.series[0].shape == tuple(grid.shape25)
                  and abs(tif.imagej_metadata.get("spacing", 0) - VOXEL_UM) < 1e-9)

    # 1. affine: 25 um voxel centre = mean world position of its (full) 2x2x2 children
    rng = np.random.default_rng(seed)
    full = [n // 2 for n in grid.shape_asr12]
    blocks = np.stack([rng.integers(0, f, n_blocks) for f in full], axis=1)

    def children(A, S, R):
        return np.array([[a, s, r] for a in (2 * A, 2 * A + 1) for s in (2 * S, 2 * S + 1) for r in (2 * R, 2 * R + 1)])

    aff_err = 0.0
    for A, S, R in blocks[:200]:
        w_kids = (grid.sform12 @ np.c_[grid.from_asr12(children(A, S, R)), np.ones(8)].T)[:3].mean(1)
        aff_err = max(aff_err, float(np.abs(w_kids - (hb["sform"] @ np.array([A, S, R, 1.0]))[:3]).max()))
    # 2. values: exact block mean from the original volume at random blocks
    val_err = 0
    for A, S, R in blocks:
        v = grid.from_asr12(children(A, S, R))
        ref = _to_dtype(np.array([np.mean([float(orig[x, y, z]) for x, y, z in v])]), img.dtype)[0]
        val_err = max(val_err, abs(float(img[A, S, R]) - float(ref)))

    # 3-4. structures
    rows = pd.read_csv(lay.slices_log)
    rows = rows[rows["brain"] == brain]
    per = []
    for row in rows.itertuples():
        m25 = nifti.load(lay.dest / row.staple_file)[0]
        hs = nifti.read_header(lay.dest / row.staple_file)
        st_vox, _ = nifti.load_sparse(src / f"{row.brainRegionComb}_{sc}_STAPLE_inBrain.nii.gz")
        u = grid.to_asr12(st_vox)
        sheet = np.zeros((grid.nz, grid.nx), bool)
        sheet[u[:, 1], u[:, 2]] = True
        up = np.repeat(np.repeat(m25, 2, 0), 2, 1)[: grid.nz, : grid.nx] > 0
        d_up = 2 * (up & sheet).sum() / (up.sum() + sheet.sum())
        A_expect = int(grid.ny - 1 - float(np.median(st_vox[:, 1]))) // 2
        S_c, R_c = (np.argwhere(m25).mean(0) if m25.any() else (np.nan, np.nan))
        slice_aff_ok = np.allclose(hs["sform"], grid.slice_affine(int(row.asr25_slice_A)), atol=1e-7)
        # hemisphere: segmented side is original x >= midline (1-based midline = last zeroed x, segAnalyser.m:68-86)
        R_mid = int(grid.to_asr12(np.array([MIDLINES[brain] - 0.5, 0, 0]))[2]) / 2
        segmented_side = (R_c < R_mid) if grid.mirror_lr else (R_c > R_mid)
        expect = (brain, row.region) not in FLIPPED_HEMISPHERE
        per.append(dict(brainRegionComb=row.brainRegionComb, region=row.region, A=int(row.asr25_slice_A),
                        A_expected_from_y=A_expect, A_consistent=abs(int(row.asr25_slice_A) - A_expect) <= 1,
                        S_centroid=float(S_c), R_centroid=float(R_c), R_midline=R_mid,
                        hemisphere_ok=bool(segmented_side == expect),
                        area_ratio=float(row.area_ratio), dice_upsampled_vs_12um5_sheet=float(d_up),
                        slice_affine_ok=bool(slice_aff_ok)))
    per_df = pd.DataFrame(per).set_index("region") if per else pd.DataFrame()
    directions = {}
    if not per_df.empty and {"CING", "MV", "AHN", "RSP"} <= set(per_df.index):
        directions = dict(
            ACA_anterior_of_MV=bool(per_df.loc["CING", "A"] < per_df.loc["MV", "A"]),
            RSP_superior_of_AHN=bool(per_df.loc["RSP", "S_centroid"] < per_df.loc["AHN", "S_centroid"]))
    res = dict(brain=brain, mirror_lr=grid.mirror_lr,
               all_hemispheres_ok=bool(all(p["hemisphere_ok"] for p in per)),
               asr_shape_25um=list(grid.shape25), nii_shape=list(hb["shape"]), nii_dtype=str(hb["dtype"]),
               shape_ok=tuple(hb["shape"]) == tuple(grid.shape25), tif_ok=bool(tif_ok),
               affine_block_centre_max_err_mm=aff_err, block_mean_max_abs_err=val_err, blocks_checked=int(n_blocks),
               directions=directions, structures=per)
    lay.checks(brain).write_text(json.dumps(res, indent=1, default=float))
    return res


def decompress(paths, out_dir: Path, log=print):
    """Plain .nii copies of chosen .nii.gz files (e.g. the 12.5 um brain-space volumes)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        p = Path(p)
        target = out_dir / p.name[:-3]
        hdr = nifti.read_header(p)
        vox, vals = nifti.load_sparse(p)
        # stream the plain copy the same way, keeping the original sform
        nifti.save_sparse(target, hdr["shape"][:3], hdr["dtype"].newbyteorder("="), vox, vals, hdr["sform"])
        log(f"  {target}")
