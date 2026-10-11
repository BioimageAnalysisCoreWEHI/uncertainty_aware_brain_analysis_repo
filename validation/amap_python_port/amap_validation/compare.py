"""Validate the uncertainty-aware (brainreg sweep + QC + consensus) pipeline against the aMAP expert consensus.

Both live on the same grid: each brain's own space at 25 um in (mirrored) BrainGlobe ``asr``, i.e. brainreg's
``registered_atlas.tiff`` grid (``reslice.AsrGrid``). Ground truth = the 2-D STAPLE consensus on its slice A0; the
pipeline masks are 3-D, restricted to the annotated hemisphere with aMAP's own midline
(``segAnalyser.removeUnsegmentedHemisphere``). Scoring follows the aMAP paper (best Dice over the coronal slices of
the raters' stack) plus a fixed-slice variant and uncertainty-aware analyses. See PLAN section 12 / PIPELINE.md.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import stats

from . import nifti
from .config import FLIPPED_HEMISPHERE, MIDLINES, Config
from .geometry import StackMapping
from .qc import PAPER_NAME
from .reslice import AsrGrid, AsrLayout
from .scoring import dice, hausdorff

VOXEL_UM = 25.0
QC_SETS = ["all_registrations", "passes_global_qc", "passes_final_qc_top_0p5"]
FINAL_SET = "passes_final_qc_top_0p5"
THRESHOLDS = ["0p01", "0p1", "0p25", "0p5", "0p75", "1p0"]
VARIANTS = ["raw", "preprocessed"]
SCORING_BRAINS = ["ER_Glt25d2Cre_9", "AB_Gad67_223", "MV131017_7", "MV_Ntsr1_169"]  # aMAP's own test brains

# pipeline sample_id (folder name) prefix -> aMAP brain name
SAMPLE_PREFIX = {
    "MV_Ntsr1_165": "MV_Ntsr1_165_StitchedImagesPaths",
    "MV_Ntsr1_169": "Ntsr1_169_StitchedImagesPaths",
    "CR_Syt6CreRfp_35": "CR_Syt6_cre_rfp_35_StitchedImagesPaths",
    "ER_Glt25d2Cre_9": "ER_Glt25d2Cre_9_5umstack",
    "AB_Gad67_223": "GAD67GFP_223_StitchedImagesPaths",
    "MV131017_7": "MV_131017_MV_7_StitchedImagesPaths",
}
try:  # repo-level path resolver (uaa_paths.py); without it, stored paths are used unchanged
    import sys as _sys
    _sys.path.insert(0, str(next(_p for _p in Path(__file__).resolve().parents if (_p / "uaa_paths.py").exists())))
    from uaa_paths import to_local
except (StopIteration, ImportError):
    def to_local(s):
        return s


# ---------------------------------------------------------------------------------------------------------------
# locating and reading the pipeline outputs
# ---------------------------------------------------------------------------------------------------------------
def long_path(p) -> Path:
    """Windows extended-length form (\\\\?\\...), so paths over 260 characters (e.g. CR's per-run masks) open."""
    import os
    s = os.path.abspath(str(p))
    if os.name != "nt" or s.startswith("\\\\?\\"):
        return Path(s)
    return Path("\\\\?\\UNC\\" + s[2:] if s.startswith("\\\\") else "\\\\?\\" + s)


def sample_dir(root: Path, brain: str) -> Path:
    base = long_path(Path(root) / "region_consensus")
    hits = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(SAMPLE_PREFIX[brain])]
    if len(hits) != 1:
        raise FileNotFoundError(f"{brain}: expected one sample folder starting {SAMPLE_PREFIX[brain]!r}, got {hits}")
    return hits[0]


def set_dir(root: Path, brain: str, variant: str, region: str, qc_set: str) -> Path:
    return sample_dir(root, brain) / variant / f"{PAPER_NAME[region]}_both" / qc_set


def read_pages(path: Path, pages) -> np.ndarray:
    """Coronal slices (pages = asr axis 0) of a pipeline TIFF, without decoding the whole stack."""
    with tifffile.TiffFile(path) as tf:
        n = len(tf.pages)
        pages = [p for p in pages if 0 <= p < n]
        return np.stack([tf.pages[p].asarray() for p in pages]) if pages else np.zeros((0, 1, 1))


def tiff_shape(path: Path):
    with tifffile.TiffFile(path) as tf:
        return (len(tf.pages),) + tuple(tf.pages[0].shape)


def to_our_grid(sl: np.ndarray, target_sr, offset_sr=(0, 0)) -> np.ndarray:
    """Place a pipeline (S, R) slice on our (S, R) grid: our index = pipeline index + offset, zero-filled."""
    out = np.zeros(target_sr, sl.dtype)
    s0, r0 = offset_sr
    src = sl[max(0, -s0): target_sr[0] - s0, max(0, -r0): target_sr[1] - r0]
    out[max(0, s0): max(0, s0) + src.shape[0], max(0, r0): max(0, r0) + src.shape[1]] = src
    return out


# ---------------------------------------------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------------------------------------------
def hemisphere_keep(grid: AsrGrid, brain: str, region: str) -> np.ndarray:
    """(nR,) bool: 25 um columns on the annotated side of aMAP's midline (x >= midline, 0-based; mirrored in asr)."""
    nR = grid.shape25[2]
    r_centre = 2 * np.arange(nR) + 0.5
    x_centre = (grid.nx - 1 - r_centre) if grid.mirror_lr else r_centre
    keep = x_centre >= MIDLINES[brain] - 0.5
    return ~keep if (brain, region) in FLIPPED_HEMISPHERE else keep


def slab_A_range(cfg: Config, grid: AsrGrid, brain: str, stack: str, centroid_voxel, n_planes: int = 40):
    """asr 25 um slices covered by the raters' 40-plane stack, at the consensus' in-plane centroid."""
    mp = StackMapping(cfg.stack(brain, stack), cfg.scale)
    x, _, z = centroid_voxel
    ys = mp.plane_y(np.full(n_planes, x), np.full(n_planes, z), np.arange(1, n_planes + 1))
    A = ((grid.ny - 1 - np.floor(ys + 0.5)) // 2).astype(int)
    return int(A.min()), int(A.max())


def perimeter_band(prob: np.ndarray, truth: np.ndarray):
    """Fraction of the truth boundary lying in the uncertain band 0 < p < 1, and the band's mean width (um)."""
    from .scoring import bwperim8
    per = bwperim8(truth > 0)
    band = (prob > 0) & (prob < 1)
    frac = float(band[per].mean()) if per.any() else np.nan
    width = float(band.sum() / max(per.sum(), 1)) * VOXEL_UM
    return frac, width


# ---------------------------------------------------------------------------------------------------------------
# alignment check: brainreg's own downsampled image vs our asr 25 um brain
# ---------------------------------------------------------------------------------------------------------------
def alignment_check(root: Path, brain: str, ours: np.ndarray, step: int = 2) -> dict:
    """NCC at integer shifts {-1,0,1}^3 and a parabolic sub-voxel estimate per axis."""
    csv = next((set_dir(root, brain, "raw", r, "all_registrations") / "registrations_used.csv"
                for r in PAPER_NAME if (set_dir(root, brain, "raw", r, "all_registrations") /
                                        "registrations_used.csv").exists()), None)
    run_dir = Path(to_local(pd.read_csv(csv)["registration_dir"].iloc[0]))
    theirs = tifffile.imread(run_dir / "downsampled.tiff")
    a = ours.astype(np.float32)
    b = theirs.astype(np.float32)
    n = [min(x, y) for x, y in zip(a.shape, b.shape)]
    core = tuple(slice(2, k - 2, step) for k in n)
    bb = b[core]
    bb = (bb - bb.mean()) / bb.std()
    ncc = {}
    for da in (-1, 0, 1):
        for ds in (-1, 0, 1):
            for dr in (-1, 0, 1):
                sl = tuple(slice(2 + d, k - 2 + d, step) for d, k in zip((da, ds, dr), n))
                aa = a[sl]
                aa = (aa - aa.mean()) / aa.std()
                ncc[(da, ds, dr)] = float((aa * bb).mean())
    best = max(ncc, key=ncc.get)
    sub = []
    for ax in range(3):
        e = [0, 0, 0]
        vals = []
        for d in (-1, 0, 1):
            e[ax] = d
            vals.append(ncc[tuple(e)])
        den = vals[0] - 2 * vals[1] + vals[2]
        sub.append(float(0.5 * (vals[0] - vals[2]) / den) if den != 0 else 0.0)
    return dict(brain=brain, run_dir=str(run_dir), their_shape=list(theirs.shape), our_shape=list(ours.shape),
                best_integer_shift_ASR=list(best), ncc_at_0=ncc[(0, 0, 0)], ncc_best=ncc[best],
                subvoxel_shift_ASR_vox=sub, subvoxel_shift_ASR_um=[s * VOXEL_UM for s in sub],
                ok=best == (0, 0, 0))


# ---------------------------------------------------------------------------------------------------------------
# scoring one brain x structure
# ---------------------------------------------------------------------------------------------------------------
def _keep_at(keep, i):
    """keep is either (nR,) (aMAP midline, same for every slice) or (n_slices, S, R) (registered hemispheres)."""
    return keep[None, :] if keep.ndim == 1 else keep[i]


def _score_stack(pages_sr: np.ndarray, A_list, truth, A0, keep_r, target_sr):
    """Paper protocol (best slice in slab) and fixed slice (A0) for a stack of pipeline slices."""
    best = (-1.0, None, None)
    fixed = (np.nan, np.nan)
    for i, (sl, A) in enumerate(zip(pages_sr, A_list)):
        m = to_our_grid(sl > 0, target_sr) & _keep_at(keep_r, i)
        d = dice(m, truth)
        if A == A0:
            fixed = (d, hausdorff(m, truth) * VOXEL_UM)
        if d > best[0]:
            best = (d, A, m)
    hd = hausdorff(best[2], truth) * VOXEL_UM if best[2] is not None else np.nan
    return dict(dice_paper=best[0], A_best=best[1], hd_paper_um=hd, dice_fixed=fixed[0], hd_fixed_um=fixed[1])


def _run_dirs(d: Path) -> dict:
    """run name (e.g. bend_0p1__grid_m25__smooth_m1p0) -> Windows registration folder, from registrations_used.csv."""
    f = d / "registrations_used.csv"
    if not f.exists():
        return {}
    dirs = pd.read_csv(f, usecols=["registration_dir"]).registration_dir
    return {p.rstrip("/").split("/")[-1]: p.replace(SCRATCH[0], SCRATCH[1]).replace("/", "\\") for p in dirs}


def hemisphere_file(root: Path, brain: str, variant: str, run: str, reg_dir: str) -> Path:
    """A run's registered_hemispheres.tiff: the collected copy if present, else the original run folder."""
    collected = (long_path(Path(root) / "registration_images_collected") / sample_dir(root, brain).name / variant
                 / f"{run}__registered_hemispheres.tiff")
    return collected if collected.exists() else long_path(Path(reg_dir)) / "registered_hemispheres.tiff"


class RegisteredHemispheres:
    """Per-run hemisphere labels (1/2) on the slab slices, stored bit-packed, plus the majority vote over runs."""

    def __init__(self, root, brain, variant, runs: dict, A_list, target_sr):
        self.A_list, self.shape = list(A_list), (len(A_list),) + tuple(target_sr)
        self.packed = {}
        c1 = np.zeros(self.shape, np.int16)
        c2 = np.zeros(self.shape, np.int16)
        for run, reg_dir in runs.items():
            pages = read_pages(hemisphere_file(root, brain, variant, run, reg_dir), self.A_list)
            h = np.stack([to_our_grid(p, target_sr) for p in pages])
            self.packed[run] = (np.packbits(h == 1), np.packbits(h == 2))
            c1 += h == 1
            c2 += h == 2
        self.vote = np.where(c1 > c2, 1, np.where(c2 > c1, 2, 0)).astype(np.uint8)  # ties / outside -> 0

    def _unpack(self, bits):
        return np.unpackbits(bits)[: int(np.prod(self.shape))].reshape(self.shape).astype(bool)

    def run_label(self, run, label):
        return self._unpack(self.packed[run][label - 1])

    def annotated_label(self, truth, A0) -> int:
        """The hemisphere label (1 or 2) that the expert consensus lies in (majority of its pixels at A0)."""
        v = self.vote[self.A_list.index(A0)][truth]
        n1, n2 = int((v == 1).sum()), int((v == 2).sum())
        return 1 if n1 >= n2 else 2


def score_structure(cfg: Config, root: Path, lay: AsrLayout, grid: AsrGrid, srow, raters: pd.DataFrame,
                    variants=VARIANTS, individual: bool = True, hemisphere: str = "amap"):
    brain, region = srow.brain, srow.region
    truth = nifti.load(lay.dest / srow.staple_file)[0] > 0
    human_p = nifti.load(lay.dest / srow.probability_file)[0]
    A0 = int(srow.asr25_slice_A)
    target_sr = truth.shape
    keep_amap = hemisphere_keep(grid, brain, region)
    info = json.loads((lay.dest.parent / "inBrainSpace" / f"{srow.brainRegionComb}_0.25_STAPLE_inBrain.json").read_text())
    a0, a1 = slab_A_range(cfg, grid, brain, info["stack"], info["staple_centroid_voxel"])
    A_list = list(range(min(a0, A0), max(a1, A0) + 1))
    base = dict(brain=brain, region=region, structure=PAPER_NAME[region], A0=A0, slab_A0=a0, slab_A1=a1,
                hemisphere=hemisphere)
    main, runs, calib, rel = [], [], [], []
    for variant in variants:
        for qc_set in QC_SETS:
            d = set_dir(root, brain, variant, region, qc_set)
            if not d.exists():
                continue
            hem, label = None, None
            if hemisphere == "registered":
                hem = RegisteredHemispheres(root, brain, variant, _run_dirs(d), A_list, target_sr)
                label = hem.annotated_label(truth, A0)
                keep = hem.vote == label
            else:
                keep = keep_amap
            extra = dict(hemisphere_label=label, n_hemisphere_runs=len(hem.packed) if hem else None)
            for thr in THRESHOLDS:
                f = d / f"consensus_thresh_{thr}.tif"
                if f.exists():
                    s = _score_stack(read_pages(f, A_list), A_list, truth, A0, keep, target_sr)
                    main.append(dict(base, variant=variant, qc_set=qc_set, threshold=float(thr.replace("p", ".")),
                                     n_runs=_n_runs(d), **extra, **s))
            if qc_set == FINAL_SET or (qc_set == "all_registrations" and not (d.parent / FINAL_SET).exists()):
                pf = d / "probability.tif"
                if pf.exists():
                    best = next((m for m in main[::-1] if m["variant"] == variant and m["qc_set"] == qc_set
                                 and m["threshold"] == 0.5), None)
                    Ab = best["A_best"] if best and best["A_best"] is not None else A0
                    p = to_our_grid(read_pages(pf, [Ab])[0], target_sr) * _keep_at(keep, A_list.index(Ab))
                    calib.append(dict(base, variant=variant, qc_set=qc_set, A=Ab, **_calibration(p, human_p, truth)))
                    rel += [dict(base, variant=variant, qc_set=qc_set, **r) for r in _reliability(p, truth)]
            if individual and (d / "individual_masks").exists():
                for f in sorted((d / "individual_masks").glob("mask__*.tif")):
                    run = f.stem.replace("mask__", "")
                    k = hem.run_label(run, label) if hem is not None and run in hem.packed else keep
                    s = _score_stack(read_pages(f, A_list), A_list, truth, A0, k, target_sr)
                    runs.append(dict(base, variant=variant, qc_set=qc_set, run=run, **s))
    human = []
    for r in raters[(raters["brainRegionComb"] == srow.brainRegionComb) & (raters["variant"] == "medianPlane")].itertuples():
        m = nifti.load(lay.dest / r.file)[0] > 0
        human.append(dict(base, user=r.user, idx=r.idx, attempt=r.attempt, dice=dice(m, truth),
                          hd_um=hausdorff(m, truth) * VOXEL_UM))
    own = raters[(raters["brainRegionComb"] == srow.brainRegionComb) & (raters["variant"] == "ownSlice")]
    ap = dict(base, rater_A_q25=float(own["asr25_slice_A"].quantile(0.25)),
              rater_A_q75=float(own["asr25_slice_A"].quantile(0.75)),
              rater_A_min=int(own["asr25_slice_A"].min()), rater_A_max=int(own["asr25_slice_A"].max()))
    return main, runs, calib, rel, human, ap


def _n_runs(d: Path) -> int:
    f = d / "registrations_used.csv"
    return int(len(pd.read_csv(f, usecols=[0]))) if f.exists() else -1


def _calibration(p: np.ndarray, human_p: np.ndarray, truth: np.ndarray) -> dict:
    union = (p > 0) | (human_p > 0)
    rho = stats.spearmanr(p[union], human_p[union]).statistic if union.sum() > 2 else np.nan
    frac_p, width_p = perimeter_band(p, truth)
    frac_h, width_h = perimeter_band(human_p, truth)
    return dict(spearman_pipeline_vs_human=float(rho), truth_boundary_in_pipeline_band=frac_p,
                pipeline_band_width_um=width_p, truth_boundary_in_human_band=frac_h, human_band_width_um=width_h)


def _reliability(p: np.ndarray, truth: np.ndarray, edges=(0, 1e-9, 0.2, 0.4, 0.6, 0.8, 1 - 1e-9, 1.0 + 1e-9)):
    """P(expert = 1 | pipeline probability bin), on the union of pipeline > 0 and expert (bins: 0, (0,.2], ..., 1)."""
    region = (p > 0) | truth
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = region & (p >= lo) & (p < hi)
        out.append(dict(p_bin_lo=lo, p_bin_hi=min(hi, 1.0), n_pixels=int(sel.sum()),
                        mean_p=float(p[sel].mean()) if sel.any() else np.nan,
                        frac_expert=float(truth[sel].mean()) if sel.any() else np.nan))
    return out


# ---------------------------------------------------------------------------------------------------------------
# statistics and report
# ---------------------------------------------------------------------------------------------------------------
def pipeline_vs_human_stats(main: pd.DataFrame, human: pd.DataFrame, variant: str, brains) -> pd.DataFrame:
    sel = main[(main.variant == variant) & (main.qc_set == FINAL_SET) & (main.threshold == 0.5) & main.brain.isin(brains)]
    hum = human[human.brain.isin(brains)]
    rows = []
    for st in sorted(set(sel.structure) | set(hum.structure)):
        p, h = sel[sel.structure == st].dice_paper.dropna(), hum[hum.structure == st].dice.dropna()
        mw = stats.mannwhitneyu(p, h) if len(p) > 0 and len(h) > 0 else None
        rows.append(dict(structure=st, n_pipeline=len(p), n_human=len(h), median_pipeline=p.median(),
                         median_human=h.median(), mannwhitney_p=mw.pvalue if mw else np.nan))
    p_all = sel[sel.structure != "DG-sg"].dice_paper.dropna()
    h_all = hum[hum.structure != "DG-sg"].dice.dropna()
    lev = stats.levene(p_all, h_all) if len(p_all) > 1 and len(h_all) > 1 else None
    mw = stats.mannwhitneyu(p_all, h_all) if len(p_all) and len(h_all) else None
    rows.append(dict(structure="ALL (excl. DG-sg)", n_pipeline=len(p_all), n_human=len(h_all),
                     median_pipeline=p_all.median(), median_human=h_all.median(),
                     mannwhitney_p=mw.pvalue if mw else np.nan, sd_pipeline=p_all.std(), sd_human=h_all.std(),
                     levene_p=lev.pvalue if lev else np.nan))
    return pd.DataFrame(rows)


def run_compare(cfg: Config, opts, pipeline_root: Path, brains=None, individual: bool = True,
                hemisphere: str = "amap", log=print):
    """hemisphere: 'amap' = aMAP's constant per-brain midline (paper protocol); 'registered' = the pipeline's own
    registered_hemispheres.tiff (majority vote over the runs of each consensus; each run's own map for single runs)."""
    staple_dir = opts.staple_dir()
    lay = AsrLayout(staple_dir / "asr25um")
    out = opts.out_dir / ("pipeline_comparison" if hemisphere == "amap" else "pipeline_comparison_registered_hemispheres")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    slices = pd.read_csv(lay.slices_log)
    raters = pd.read_csv(lay.raters_log)
    brains = list(brains or cfg.brains)
    names = ("main", "runs", "calib", "rel", "human", "ap", "align")
    for brain in brains:
        t0 = time.time()
        acc = {k: [] for k in names}
        h = nifti.read_header(cfg.brain_file(brain))
        grid = AsrGrid(h["shape"], h["sform"])
        ours = nifti.load(lay.brain_img(brain), mmap=True)
        al = alignment_check(pipeline_root, brain, np.asarray(ours))
        acc["align"].append(al)
        log(f"{brain}: alignment best shift {al['best_integer_shift_ASR']} NCC {al['ncc_at_0']:.4f} "
            f"sub-voxel {np.round(al['subvoxel_shift_ASR_um'], 1)} um  ({time.time() - t0:.0f}s)")
        if al["ok"]:
            for srow in slices[slices.brain == brain].itertuples():
                res = score_structure(cfg, pipeline_root, lay, grid, srow, raters, individual=individual,
                                      hemisphere=hemisphere)
                for k, v in zip(names[:5], res[:5]):
                    acc[k] += v
                acc["ap"].append(res[5])
                log(f"  {srow.brainRegionComb:22s} done ({time.time() - t0:.0f}s)")
        else:
            log(f"  !! {brain}: best alignment is not (0,0,0); scoring skipped for this brain")
        for k in names:  # save after every brain, replacing this brain's earlier rows
            path = out / "tables" / f"{k}.csv"
            df = pd.DataFrame(acc[k])
            if path.exists():
                old = pd.read_csv(path)
                old = old[old["brain"] != brain] if "brain" in old else old
                df = pd.concat([old, df], ignore_index=True)
            df.to_csv(path, index=False)
        log(f"  {brain}: tables saved ({time.time() - t0:.0f}s)")
    tables = {k: pd.read_csv(out / "tables" / f"{k}.csv") if (out / "tables" / f"{k}.csv").stat().st_size > 2
              else pd.DataFrame() for k in names}
    write_summary(tables, out, log)
    from .qc import comparison_figures
    comparison_figures(cfg, opts, tables, out / "figures", lay, pipeline_root, log=log)
    return tables


def md_table(df: pd.DataFrame, index: bool = True, digits: int = 3) -> str:
    """Markdown table without the optional tabulate dependency."""
    d = df.reset_index() if index else df
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(fmt(x) for x in v) + "]"
        return str(v)
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "---|" * len(d.columns)
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def _mode(main: pd.DataFrame) -> str:
    if "hemisphere" in main and (main.hemisphere == "registered").any():
        return "the pipeline's registered_hemispheres (majority vote over the consensus' runs)"
    return "aMAP's constant per-brain midline"


def write_summary(t: dict, out: Path, log=print):
    main, human, runs, calib, ap, align = (t.get(k, pd.DataFrame()) for k in ("main", "human", "runs", "calib", "ap", "align"))
    lines = ["# Uncertainty-aware pipeline vs aMAP expert consensus\n",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `python -m amap_validation compare`. Tables in `tables/`, "
             "figures in `figures/`. Primary: raw inputs, final QC set (`passes_final_qc_top_0p5`), threshold 0.5, "
             "aMAP paper protocol (best coronal slice within the raters' stack), annotated hemisphere only "
             f"(hemisphere mode: **{_mode(main)}**), DG-sg excluded from pooled numbers.\n"]

    def section(title, table, **kw):
        lines.extend([f"## {title}\n", md_table(table, **kw), ""])

    if len(align):
        section("Grid alignment (brainreg downsampled vs our asr 25 µm)",
                align[["brain", "best_integer_shift_ASR", "ncc_at_0", "subvoxel_shift_ASR_um", "ok"]], index=False)
    if len(main) and len(human):
        brains = sorted(main.brain.unique())
        for variant in VARIANTS:
            if variant not in set(main.variant):
                continue
            for label, subset in (("all brains", brains),
                                  ("aMAP scoring brains", [b for b in brains if b in SCORING_BRAINS])):
                if subset:
                    section(f"Dice: pipeline ({variant}) vs humans, {label}",
                            pipeline_vs_human_stats(main, human, variant, subset), index=False)
        noDG = main[main.structure != "DG-sg"]
        section("Median Dice vs consensus threshold (final QC set, excl. DG-sg)",
                noDG[noDG.qc_set == FINAL_SET].groupby(["variant", "threshold"]).dice_paper.median().unstack(0))
        section("Median Dice by QC stage (threshold 0.5, excl. DG-sg)",
                noDG[noDG.threshold == 0.5].groupby(["variant", "qc_set"]).dice_paper.median().unstack(0))
        section("Paper protocol vs fixed STAPLE slice (medians, final QC, threshold 0.5)",
                noDG[(noDG.qc_set == FINAL_SET) & (noDG.threshold == 0.5)].groupby("variant")[
                    ["dice_paper", "dice_fixed", "hd_paper_um", "hd_fixed_um"]].median())
    if len(runs):
        section("Individual registrations (paper-protocol Dice, excl. DG-sg)",
                runs[runs.structure != "DG-sg"].groupby(["variant", "qc_set"]).dice_paper.describe()[
                    ["count", "25%", "50%", "75%"]])
    if len(calib):
        section("Uncertainty vs human disagreement (medians, excl. DG-sg)",
                calib[calib.structure != "DG-sg"].groupby("variant")[
                    ["spearman_pipeline_vs_human", "truth_boundary_in_pipeline_band", "pipeline_band_width_um",
                     "truth_boundary_in_human_band", "human_band_width_um"]].median())
    if len(main) and len(ap):
        m = main[(main.qc_set == FINAL_SET) & (main.threshold == 0.5)].copy()
        m["dA_um"] = (m.A_best - m.A0) * VOXEL_UM
        section("AP position: pipeline best slice minus STAPLE slice (µm)",
                m.groupby("variant").dA_um.describe(), digits=1)
    er = main[main.brain == "ER_Glt25d2Cre_9"] if len(main) else pd.DataFrame()
    if len(er):
        section("ER_Glt25d2Cre_9 (rejected by global QC; sensitivity on all_registrations, threshold 0.5)",
                er[er.threshold == 0.5][["variant", "structure", "dice_paper", "dice_fixed"]], index=False)
    (out / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"  wrote {out / 'results_summary.md'}")
