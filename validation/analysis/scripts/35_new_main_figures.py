"""New versions of manuscript Figures 1-4 (original LSFM dataset), restyled to match the validation figures (Figs 5-8)
and merged where Reviewer 2 found the panels redundant (old 2F, 3C, 3D and 4F -> new Fig 3C).

Every panel is re-rendered from stored data; nothing upstream is modified.
  counts       per-registration SNpc TH+ counts: final_count_results table (snpc_CellCount);
               consensus counts: overlap rule from the voxel-vote cache (07_lsfm_analysis.Counter), i.e. a cell counts if
               any of its voxels lies in the consensus (mean of retained masks >= threshold)
  selection    global QC IoU > 0.9 and grad_p95 < 0.045, then the top ceil(n/2) by midbrain SSIM (07_lsfm_analysis.select)
  SNpc masks   original pipeline masks (coronal 20 um, and full-resolution crops in the TH image space)
  whole-brain  downsampled.tiff / boundaries.tiff / downsampled_standard.tiff from the rerun of the same sweep
  images       (jom_review_sweep_original_data; same inputs and parameters, SNpc Dice 0.93-1.00 vs the original in spot
               checks) -- only used for anatomy display, never for numbers
  atlas        brainglobe perens_lsfm_mouse_20um v1.2; SSIM box = midbrain_qc_bbox.json (the box the SSIM was computed in)

Usage: python 35_new_main_figures.py [fig1|fig2|fig3|fig4|all]
Output: REVISION_OUTPUTS_FOCUSED/new_main_figures/Fig{1-4}_*.{pdf,svg,png} (600 dpi) + source_data/*.csv
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from skimage.segmentation import find_boundaries  # noqa: E402

VA = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_analysis"))
sys.path.insert(0, str(VA / "scripts"))
os.chdir(VA / "scripts")
_spec = importlib.util.spec_from_file_location("la", str(VA / "scripts" / "07_lsfm_analysis.py"))
la = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(la)

OUT = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\REVISION_OUTPUTS_FOCUSED\new_main_figures"))
SRC = OUT / "source_data"
for _d in (OUT, SRC):
    _d.mkdir(parents=True, exist_ok=True)
REG = Path(P(r"<SCRATCH>\jom_review_sweep_original_data"))
ATLAS = Path.home() / ".brainglobe" / "perens_lsfm_mouse_20um_v1.2"
SSIM_BOX = json.load(open(P(r"<IMAGING>\Ishrat\AnneMiller_round3\midbrain_qc_bbox.json")))
HPC_LOGS = Path(P(r"<IMAGING>\Ishrat\JoM\hpc_processing_logs"))
SNPC_ID = 374

SAMPLES = ["162", "527", "pp451", "xl353", "xl385"]                      # paper order: Sample 1-5
LAB = {s: f"Sample {i + 1}" for i, s in enumerate(SAMPLES)}
SCOL = dict(zip(SAMPLES, ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]))
C_RUN, C_CONS, C_REF = "#56B4E9", "#7B3294", "white"
C_PASS, C_FAIL = "#1a9850", "#d73027"
C_RAW, C_PRE = "#E08214", "#2A9D8F"
C_B, C_C = "#D55E00", "#0072B2"                                          # the two Fig 1 registrations
IOU_T, GRAD_T = 0.9, 0.045
PX_DS = 20.0                                                             # um per downsampled / atlas voxel
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5})
BOX = dict(facecolor="black", alpha=0.6, pad=1.2, lw=0)


# ------------------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------------------
RUNS = pd.read_csv(VA / "cache" / "lsfm" / "runs.csv")
RUNS["sample"] = RUNS["sample"].astype(str)
RUNS["passes_global"] = (RUNS.iou > IOU_T) & (RUNS.grad_p95 < GRAD_T)


def runs(s, v):
    return RUNS[(RUNS["sample"] == s) & (RUNS.status == v)].sort_values(["bending", "grid"])


def retained(s, v):
    return la.select(runs(s, v))


_COUNTERS = {}


def counter(s, v):
    if (s, v) not in _COUNTERS:
        _COUNTERS[(s, v)] = la.Counter(s, v)
    return _COUNTERS[(s, v)]


def row(s, v, key):
    return runs(s, v).set_index("run_key").loc[key]


def rerun_dir(s, v, bending, grid):
    t = f"{float(bending):g}"
    t = (t if "." in t else t + ".0").replace(".", "p")
    return REG / s / v / f"bend_{t}__grid_m{abs(int(grid))}__smooth_m1p0"


def page(path, z):
    with tifffile.TiffFile(path) as t:
        return t.pages[int(z)].asarray()


def n_pages(path):
    with tifffile.TiffFile(path) as t:
        return len(t.pages)


def fullres_um(s, v):
    """In-plane voxel size of the full-resolution TH crop: the pipeline resized the 20 um mask crop to it."""
    p = Path(row(s, v, runs(s, v).run_key.iloc[0]).mask_axial_path).parent
    dn = tifffile.TiffFile(next(p.glob("*_snpc_mask_region_downsampled.tif"))).series[0].shape
    up = tifffile.TiffFile(next(p.glob("*_snpc_mask_region_resized.tif"))).series[0].shape
    return PX_DS * dn[1] / up[1], PX_DS * dn[2] / up[2]


def norm(img, lo=1, hi=99.5, mask=None):
    img = np.asarray(img, np.float32)
    ref = img[mask] if mask is not None and mask.any() else img
    a, b = np.percentile(ref, [lo, hi])
    return np.clip((img - a) / (b - a + 1e-9), 0, 1)


# ------------------------------------------------------------------------------------------------------------
# drawing helpers
# ------------------------------------------------------------------------------------------------------------
def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def letter(ax, s, x=-0.02, y=1.02, fig=None):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def clean(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def contour(ax, m, color, lw, ls="-", alpha=1.0, z=3):
    m = np.asarray(m, float)
    if m.any():
        ax.contour(m, [0.5], colors=[color], linewidths=lw, linestyles=[ls], alpha=alpha, zorder=z)


def scalebar(ax, shape, um, px_um, txt=None, frac_x=0.04, frac_y=0.07, color="white", fs=5.5):
    h, w = shape[:2]
    L = um / px_um
    x0, y0 = frac_x * w, h * (1 - frac_y)
    ax.plot([x0, x0 + L], [y0, y0], color=color, lw=1.6, solid_capstyle="butt")
    ax.text(x0, y0 - 0.02 * h, txt or (f"{um / 1000:g} mm" if um >= 1000 else f"{um:g} µm"), color=color,
            ha="left", va="bottom", fontsize=fs)


def tag(ax, txt, fs=5.8, loc="tl", color="white"):
    x, ha = (0.03, "left") if loc[1] == "l" else (0.97, "right")
    y, va = (0.97, "top") if loc[0] == "t" else (0.03, "bottom")
    ax.text(x, y, txt, transform=ax.transAxes, ha=ha, va=va, color=color, fontsize=fs, bbox=BOX, linespacing=1.15)


def bbox_of(mask, margin, shape=None):
    rr, cc = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    shape = shape or mask.shape
    return (slice(max(rr[0] - margin, 0), min(rr[-1] + margin + 1, shape[0])),
            slice(max(cc[0] - margin, 0), min(cc[-1] + margin + 1, shape[1])))


def show(ax, img, cmap="gray"):
    ax.imshow(img, cmap=cmap, interpolation="nearest", vmin=0, vmax=1)
    clean(ax)


def boundary_rgba(lbl, color, alpha):
    b = find_boundaries(lbl, mode="inner") & (lbl > 0)
    rgba = np.zeros(lbl.shape + (4,), np.float32)
    rgba[b] = list(matplotlib.colors.to_rgb(color)) + [alpha]
    return rgba


def strip_box(ax, x, vals, color, width=0.6, pts=True, pcolor=None, s=4, box_fc="none", seed=0, cmap=None,
              norm_=None, lw=0.7):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if not len(vals):
        return
    bp = ax.boxplot(vals, positions=[x], widths=width, showfliers=False, patch_artist=True, whis=(0, 100),
                    medianprops=dict(color="black", lw=0.9), boxprops=dict(facecolor=box_fc, edgecolor=color, lw=lw),
                    whiskerprops=dict(color=color, lw=lw), capprops=dict(color=color, lw=lw))
    if pts:
        rng = np.random.default_rng(seed)
        jx = x + rng.uniform(-width * 0.32, width * 0.32, len(vals))
        if cmap is not None:
            ax.scatter(jx, vals, s=s, c=pcolor, cmap=cmap, norm=norm_, lw=0.25, edgecolor="white", zorder=3)
        else:
            ax.scatter(jx, vals, s=s, color=pcolor or color, lw=0.25, edgecolor="white", zorder=3)
    return bp


# ------------------------------------------------------------------------------------------------------------
# atlas
# ------------------------------------------------------------------------------------------------------------
_ATL = {}


def atlas():
    if not _ATL:
        _ATL["ref"] = tifffile.imread(ATLAS / "reference.tiff")
        _ATL["ann"] = tifffile.imread(ATLAS / "annotation.tiff")
    return _ATL["ref"], _ATL["ann"]


# ============================================================================================================
# Figure 1
# ============================================================================================================
def hpc_times():
    cache = SRC / "Fig1E_hpc_runtimes.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["start_time", "end_time"])
    rec = []
    for f in HPC_LOGS.rglob("*inner*"):
        if not f.is_file():
            continue
        t = f.read_text(errors="ignore")
        a = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+PM.*Starting logging", t, re.DOTALL)
        b = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+PM.*Finished\.\s+Total time taken", t, re.DOTALL)
        r = re.search(r"Finished\. Total time taken:\s*([0-9]+:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?)", t)
        if a and b and r:
            rec.append(dict(log_file=f.name, start_time=pd.to_datetime(a.group(1)), end_time=pd.to_datetime(b.group(1)),
                            runtime_h=pd.to_timedelta(r.group(1)).total_seconds() / 3600))
    d = pd.DataFrame(rec).sort_values("start_time")
    d.to_csv(cache, index=False)
    return d


def fig1():
    s, v = "pp451", "raw"
    kB, kC = "0.95_-30", "0.5_-10"
    rB, rC = row(s, v, kB), row(s, v, kC)
    cntB, cntC = int(rB.snpc_CellCount), int(rC.snpc_CellCount)
    src = []

    fig = plt.figure(figsize=(7.2, 8.9))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.0, 0.9, 0.9, 0.8, 1.32], hspace=0.42)

    # ---- A: atlas ------------------------------------------------------------------------------------------
    ref, ann = atlas()
    sn = ann == SNPC_ID
    ap = int(np.argmax(sn.sum((1, 2))))
    dv = int(np.argmax(sn.sum((0, 2))))
    gA = gs[0].subgridspec(1, 3, width_ratios=[461 / 323, 461 / 621 * 1.0, 1.35], wspace=0.06)
    axs = [fig.add_subplot(gA[i]) for i in range(3)]
    cor = norm(ref[ap].astype(float), 0.5, 99.8)
    axl = norm(ref[:, dv, :].astype(float), 0.5, 99.8)
    for ax, img, lbl in ((axs[0], cor, ann[ap]), (axs[1], axl, ann[:, dv, :])):
        show(ax, img)
        ax.imshow(boundary_rgba(lbl, "black", 0.28), interpolation="nearest")
        m = lbl == SNPC_ID
        ax.imshow(np.dstack([np.ones_like(m, float)] * 3 + [m * 0.55]), interpolation="nearest")
        contour(ax, m, C_REF, 0.6)
    scalebar(axs[0], cor.shape, 1000, PX_DS)
    m = ann[:, dv, :] == SNPC_ID
    sl = bbox_of(m, 22)
    axs[1].add_patch(Rectangle((sl[1].start, sl[0].start), sl[1].stop - sl[1].start, sl[0].stop - sl[0].start,
                               fill=False, ec="white", lw=0.8))
    scalebar(axs[1], axl.shape, 1000, PX_DS)
    show(axs[2], axl[sl])
    axs[2].imshow(boundary_rgba(ann[:, dv, :][sl], "black", 0.35), interpolation="nearest")
    axs[2].imshow(np.dstack([np.ones_like(m[sl], float)] * 3 + [m[sl] * 0.55]), interpolation="nearest")
    contour(axs[2], m[sl], C_REF, 0.9)
    scalebar(axs[2], axl[sl].shape, 500, PX_DS)
    for ax, t in zip(axs, ("coronal", "horizontal", "horizontal, boxed region")):
        ax.set_title(t, fontsize=6.5, pad=2)
    letter(axs[0], "A")
    axs[0].text(0.03, 0.97, "Perens LSFM atlas\nSNpc (white)", transform=axs[0].transAxes, ha="left", va="top",
                color="white", fontsize=5.8, bbox=BOX)

    # ---- B, C: two registrations of the same brain -----------------------------------------------------------
    mC = tifffile.imread(rC.snpc_mask_path) > 0
    z = int(np.argmax(mC.sum((1, 2))))                                    # coronal slice with the largest SNpc (C)
    fz = None
    for i, (key, r, cnt, col, L) in enumerate(((kB, rB, cntB, C_B, "B"), (kC, rC, cntC, C_C, "C"))):
        g = gs[1 + i].subgridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.04)
        a1, a2 = fig.add_subplot(g[0]), fig.add_subplot(g[1])
        d = rerun_dir(s, v, r.bending, r.grid)
        img = page(d / "downsampled.tiff", z).astype(float)
        bnd = page(d / "boundaries.tiff", z) > 0
        tissue = img > np.percentile(img, 60)
        sl = bbox_of(ndi.binary_opening(tissue, iterations=2), 6)
        show(a1, norm(img[sl], 1, 99.7))
        rgba = np.zeros(bnd[sl].shape + (4,), np.float32)
        rgba[bnd[sl]] = list(matplotlib.colors.to_rgb(col)) + [0.9]
        a1.imshow(rgba, interpolation="nearest")
        msk = page(r.snpc_mask_path, z) > 0
        contour(a1, msk[sl], "white", 0.7)
        scalebar(a1, img[sl].shape, 1000, PX_DS)
        tag(a1, f"bending energy {r.bending:g}\ngrid spacing {int(r.grid)}")
        # full-resolution TH crop, same registration's SNpc
        if fz is None:
            mres = tifffile.imread(rC.mask_axial_path)
            fz = int(np.argmax((mres > 0).sum((1, 2))))
            del mres
            both = (page(rB.mask_axial_path, fz) > 0) | (page(rC.mask_axial_path, fz) > 0)
            tsl = bbox_of(both, 170)
            th = page(r.snpc_image_region_path, fz).astype(float)[tsl]
            thn = norm(th, 1, 99.7)
        mm = (page(r.mask_axial_path, fz) > 0)[tsl]
        show(a2, thn)
        contour(a2, mm, col, 0.9)
        py, px = fullres_um(s, v)
        scalebar(a2, thn.shape, 500, px)
        a2.set_title(f"SNpc TH+ cells: {cnt:,}", fontsize=7.5, color=col, fontweight="bold", pad=2)
        letter(a1, L)
        src.append(dict(panel=L, sample=LAB[s], input=v, run=key, bending=r.bending, grid=r.grid,
                        snpc_count=cnt, coronal_slice_20um=z, fullres_page=fz, iou=r.iou, grad_p95=r.grad_p95))
    fig.text(0.5, 0.0, "", fontsize=1)

    # ---- D: pipeline schematic ------------------------------------------------------------------------------
    gDE = gs[3].subgridspec(1, 2, width_ratios=[1.75, 1.0], wspace=0.42)
    axD = fig.add_subplot(gDE[0])
    axD.set_xlim(0, 12.3), axD.set_ylim(0, 4.2)
    clean(axD)
    bend = sorted(RUNS.bending.unique())
    grids = sorted(RUNS.grid.unique())
    # 8 x 6 parameter grid
    x0, y0, cw, ch = 0.55, 0.45, 0.32, 0.4
    for i in range(len(bend)):
        for j in range(len(grids)):
            axD.add_patch(Rectangle((x0 + i * cw, y0 + j * ch), cw * 0.86, ch * 0.82, fc=C_RUN, ec="none", alpha=0.85))
    axD.text(x0 + len(bend) * cw / 2, y0 + len(grids) * ch + 0.1, "48 registrations per brain\n(one HPC job each)",
             ha="center", va="bottom", fontsize=5.6, linespacing=1.15)
    axD.text(x0 + len(bend) * cw / 2, y0 - 0.1, "8 bending energies", ha="center", va="top", fontsize=5.2,
             color="0.35")
    axD.text(x0 - 0.1, y0 + len(grids) * ch / 2, "6 grid spacings", ha="right", va="center", rotation=90,
             fontsize=5.2, color="0.35")
    steps = [("global QC", "tissue–atlas IoU\nand deformation\n(grad_p95)", "0.35"),
             ("local QC", "midbrain SSIM\nkeep top 50%", "0.35"),
             ("consensus", "inclusion\nfrequency of\neach voxel / cell", C_CONS)]
    bx, bw, step = 3.75, 2.35, 2.9
    for k, (t1, t2, c) in enumerate(steps):
        xx = bx + k * step
        axD.annotate("", xy=(xx - 0.06, 1.75), xytext=(xx - 0.5, 1.75),
                     arrowprops=dict(arrowstyle="-|>", color="0.3", lw=0.8, mutation_scale=7))
        axD.add_patch(FancyBboxPatch((xx, 0.7), bw, 2.1, boxstyle="round,pad=0.04,rounding_size=0.15",
                                     fc="white", ec=c, lw=1.0))
        axD.text(xx + bw / 2, 2.42, t1, ha="center", va="center", fontsize=6.3, fontweight="bold",
                 color=c if c == C_CONS else "0.15")
        axD.text(xx + bw / 2, 1.42, t2, ha="center", va="center", fontsize=5.2, linespacing=1.15)
    letter(axD, "D", x=0.0, y=0.92)

    # ---- E: HPC timing ----------------------------------------------------------------------------------------
    hp = hpc_times()
    seq = hp.runtime_h.sum()
    wall = (hp.end_time.max() - hp.start_time.min()).total_seconds() / 3600
    axE = fig.add_subplot(gDE[1])
    axE.barh([1, 0], [seq, wall], color=["0.6", C_CONS], height=0.5)
    axE.set_yticks([1, 0], ["sequential", "HPC"])
    axE.set_xlabel("time to run the sweep (h)")
    axE.text(seq * 0.97, 1, f"{seq:.1f} h (sum of {len(hp)} run times)", va="center", ha="right", fontsize=5.8,
             color="white")
    axE.text(wall + seq * 0.02, 0, f"{wall:.1f} h observed wall time ({seq / wall:.0f}× faster)", va="center",
             fontsize=5.8)
    axE.set_xlim(0, seq * 1.05)
    axE.set_ylim(-0.6, 1.6)
    axE.tick_params(axis="y", length=0)
    axE.text(seq * 0.02, 1.38, "", fontsize=1)
    letter(axE, "E", x=-0.2, y=0.92)
    pd.DataFrame([dict(n_registrations=len(hp), sequential_h=seq, wall_h=wall, speedup=seq / wall)]).to_csv(
        SRC / "Fig1E_hpc_summary.csv", index=False)

    # ---- F: heatmap, G: per-brain spread -------------------------------------------------------------------
    gFG = gs[4].subgridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.38)
    axF = fig.add_subplot(gFG[0])
    sub = runs(s, v)
    H = sub.pivot(index="grid", columns="bending", values="snpc_CellCount").sort_index()
    im = axF.imshow(H.values, cmap="Blues", aspect="auto", vmin=H.values.min() * 0.75)
    for i, gg in enumerate(H.index):
        for j, bb in enumerate(H.columns):
            val = H.iloc[i, j]
            axF.text(j, i, f"{val:,.0f}", ha="center", va="center", fontsize=5.1,
                     color="white" if val > H.values.min() + 0.55 * (H.values.max() - H.values.min()) else "black")
    axF.set_xticks(range(len(H.columns)), [f"{b:g}" for b in H.columns])
    axF.set_yticks(range(len(H.index)), [f"{int(gg)}" for gg in H.index])
    axF.set_xlabel("bending energy weight")
    axF.set_ylabel("grid spacing")
    for key, col in ((kB, C_B), (kC, C_C)):
        r = row(s, v, key)
        i, j = list(H.index).index(r.grid), list(H.columns).index(r.bending)
        axF.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=col, lw=1.6, zorder=4))
    for sp in axF.spines.values():
        sp.set_visible(False)
    axF.tick_params(length=0)
    cb = fig.colorbar(im, ax=axF, fraction=0.045, pad=0.02)
    cb.set_label("SNpc TH+ cells", fontsize=6.3)
    cb.ax.tick_params(labelsize=5.8)
    cb.outline.set_linewidth(0.4)
    axF.set_title(f"{LAB[s]}, raw input: all 48 registrations", fontsize=6.5, pad=3)
    letter(axF, "F", x=-0.12)
    H.to_csv(SRC / "Fig1F_heatmap_counts.csv")

    axG = fig.add_subplot(gFG[1])
    for i, ss in enumerate(SAMPLES):
        vals = runs(ss, v).snpc_CellCount.values
        strip_box(axG, i, vals, "0.25", width=0.55, pcolor=SCOL[ss], s=6, seed=i)
        src.append(dict(panel="G", sample=LAB[ss], input=v, n=len(vals), min=vals.min(), median=np.median(vals),
                        max=vals.max(), range_pct_of_median=100 * (vals.max() - vals.min()) / np.median(vals)))
        axG.text(i, vals.max() + 90, f"{100 * (vals.max() - vals.min()) / np.median(vals):.0f}%", ha="center",
                 va="bottom", fontsize=5.6, color="0.3")
    axG.set_xticks(range(len(SAMPLES)), [LAB[x] for x in SAMPLES])
    axG.set_ylabel("SNpc TH+ cells")
    axG.set_title("all 48 registrations per brain, raw input (range as % of the median)", fontsize=6.3, pad=3)
    axG.set_ylim(1000, 5500)
    axG.grid(axis="y", color="0.92", lw=0.4)
    axG.set_axisbelow(True)
    letter(axG, "G", x=-0.1)
    save(fig, "Fig1_parameter_sensitivity_and_pipeline")
    pd.DataFrame(src).to_csv(SRC / "Fig1_source.csv", index=False)


# ============================================================================================================
# Figure 2
# ============================================================================================================
def quadrant(r):
    return ("hi" if r.iou > IOU_T else "lo") + "_" + ("lo" if r.grad_p95 < GRAD_T else "hi")


def fig2():
    v = "raw"
    d = RUNS[RUNS.status == v].copy()
    d["quad"] = d.apply(quadrant, axis=1)
    # examples: Sample 3 (as in Fig 1), the registration closest to the median of its quadrant (normalised metrics)
    s = "pp451"
    e = d[d["sample"] == s]
    spec = [("lo_hi", "B", "low IoU, high deformation"), ("hi_hi", "C", "high IoU, high deformation"),
            ("lo_lo", "D", "low IoU, low deformation"), ("hi_lo", "E", "high IoU, low deformation")]
    ex = {}
    for q, L, _ in spec:
        g = e[e.quad == q]
        if g.empty:
            g = d[d.quad == q]
        zi = (g.iou - g.iou.median()) / d.iou.std()
        zg = (g.grad_p95 - g.grad_p95.median()) / d.grad_p95.std()
        ex[L] = g.loc[(zi ** 2 + zg ** 2).idxmin()]

    fig = plt.figure(figsize=(7.2, 3.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.12)
    ax = fig.add_subplot(gs[0])
    ok = d.passes_global
    ax.scatter(d.iou[~ok], d.grad_p95[~ok], s=7, color=C_FAIL, alpha=0.75, lw=0.2, edgecolor="white", label="fail")
    ax.scatter(d.iou[ok], d.grad_p95[ok], s=7, color=C_PASS, alpha=0.85, lw=0.2, edgecolor="white", label="pass")
    ax.axvline(IOU_T, color="0.25", ls="--", lw=0.7)
    ax.axhline(GRAD_T, color="0.25", ls="--", lw=0.7)
    ax.axvspan(IOU_T, 1.0, ymax=(GRAD_T - 0.028) / (0.095 - 0.028), color=C_PASS, alpha=0.07, lw=0)
    for L, r in ex.items():
        ax.scatter(r.iou, r.grad_p95, s=34, facecolor="none", edgecolor="black", lw=0.8, zorder=4)
        ax.text(r.iou + 0.0025, r.grad_p95 + 0.0018, L, fontsize=7, fontweight="bold", zorder=5)
    ax.set_xlim(0.83, 0.975)
    ax.set_ylim(0.028, 0.095)
    ax.set_xlabel("tissue–atlas IoU")
    ax.set_ylabel("deformation (grad_p95)")
    ax.text(0.97, 0.04, f"pass: {ok.sum()}/{len(d)}", transform=ax.transAxes, ha="right", fontsize=6.3,
            color=C_PASS)
    ax.legend(loc="upper right", handletextpad=0.1, markerscale=1.3, title="global QC", title_fontsize=6.3)
    ax.set_title("all 240 registrations (5 brains × 48), raw input", fontsize=6.5, pad=3)
    letter(ax, "A", x=-0.16)

    g = gs[1].subgridspec(2, 2, wspace=0.04, hspace=0.32)
    src = []
    for k, (q, L, desc) in enumerate(spec):
        r = ex[L]
        a = fig.add_subplot(g[k // 2, k % 2])
        rr = row(r["sample"], v, r.run_key)
        m3 = tifffile.imread(rr.snpc_mask_path) > 0
        z = int(np.argmax(m3.sum((1, 2)))) if m3.any() else n_pages(rr.snpc_mask_path) // 2
        dd = rerun_dir(r["sample"], v, r.bending, r.grid)
        img = page(dd / "downsampled.tiff", z).astype(float)
        bnd = page(dd / "boundaries.tiff", z) > 0
        tissue = img > np.percentile(img, 60)
        sl = bbox_of(ndi.binary_opening(tissue, iterations=2), 6)
        show(a, norm(img[sl], 1, 99.7))
        col = C_PASS if r.passes_global else C_FAIL
        rgba = np.zeros(bnd[sl].shape + (4,), np.float32)
        rgba[bnd[sl]] = list(matplotlib.colors.to_rgb(col)) + [0.5]
        a.imshow(rgba, interpolation="nearest")
        outer = ndi.binary_fill_holes(page(dd / "registered_atlas.tiff", z) > 0)
        contour(a, outer[sl], col, 1.1)                                  # registered atlas outline (IoU)
        scalebar(a, img[sl].shape, 1000, PX_DS)
        a.set_title(f"{desc}\nIoU {r.iou:.3f} · grad_p95 {r.grad_p95:.3f}", fontsize=6, pad=2, linespacing=1.15)
        tag(a, "pass" if r.passes_global else "fail", fs=6, loc="tr", color=col)
        letter(a, L, x=-0.01)
        src.append(dict(panel=L, sample=LAB[r["sample"]], run=r.run_key, bending=r.bending, grid=r.grid, iou=r.iou,
                        grad_p95=r.grad_p95, passes_global_qc=bool(r.passes_global), coronal_slice=z))
    fig.text(0.72, 0.005, "thick line: outer outline of the registered atlas (compared with the tissue for IoU)\n"
             "thin lines: registered atlas region borders", ha="center", va="top", fontsize=5.8, color="0.3")
    save(fig, "Fig2_global_QC")
    pd.DataFrame(src).to_csv(SRC / "Fig2_examples.csv", index=False)
    d[["sample", "run_key", "bending", "grid", "iou", "grad_p95", "passes_global"]].assign(
        sample=lambda x: x["sample"].map(LAB)).to_csv(SRC / "Fig2A_registrations.csv", index=False)


# ============================================================================================================
# Figure 3
# ============================================================================================================
def ssim_examples():
    """Sample with the widest SSIM range among raw global-QC passers; its lowest- and highest-SSIM passers."""
    best = None
    for s in SAMPLES:
        p = runs(s, "raw")
        p = p[p.passes_global]
        rng = p.ssim_value.max() - p.ssim_value.min()
        if best is None or rng > best[0]:
            best = (rng, s, p.loc[p.ssim_value.idxmin()], p.loc[p.ssim_value.idxmax()])
    return best[1], best[2], best[3]


def cell_inclusion(s, v, run_keys):
    """Per cell: max over its voxels of the fraction of the given registrations that include the voxel (the cell
    is counted at threshold t iff this >= t -- the overlap rule)."""
    c = counter(s, v)
    ids = [c.col[k] for k in run_keys]
    frac = (c.votes[:, ids].sum(1) / len(ids)).astype(np.float32)
    best = np.zeros(int(c.cell.max()) + 1, np.float32)
    np.maximum.at(best, c.cell, frac)
    return best


def fig3():
    v = "raw"
    s, lo, hi = ssim_examples()
    ret = retained(s, v)
    allk = list(runs(s, v).run_key)
    fig = plt.figure(figsize=(7.2, 10.4))
    gs = fig.add_gridspec(5, 1, height_ratios=[0.62, 0.62, 1.25, 0.95, 1.1], hspace=0.5)
    src = []

    # ---- A, B: low vs high SSIM ------------------------------------------------------------------------------
    ref, ann = atlas()
    b = SSIM_BOX
    snb = ann[b["z0"]:b["z1"], b["y0"]:b["y1"], b["x0"]:b["x1"]] == SNPC_ID
    za = b["z0"] + int(np.argmax(snb.sum((1, 2))))
    aref = norm(ref[za, b["y0"]:b["y1"], b["x0"]:b["x1"]].astype(float), 0.5, 99.8)
    alab = ann[za, b["y0"]:b["y1"], b["x0"]:b["x1"]]
    py, px = fullres_um(s, v)
    for i, (r, L, word) in enumerate(((lo, "A", "lowest"), (hi, "B", "highest"))):
        g = gs[i].subgridspec(1, 3, width_ratios=[1, 1, 1.0], wspace=0.04)
        a0, a1, a2 = (fig.add_subplot(g[k]) for k in range(3))
        dd = rerun_dir(s, v, r.bending, r.grid)
        reg = norm(page(dd / "downsampled_standard.tiff", za)[b["y0"]:b["y1"], b["x0"]:b["x1"]].astype(float),
                   0.5, 99.8)
        for a, img in ((a0, aref), (a1, reg)):
            show(a, img)
            a.set_aspect("auto")
            a.imshow(boundary_rgba(alab, "white", 0.35), interpolation="nearest", aspect="auto")
            contour(a, alab == SNPC_ID, "#FFD92F", 0.8)
        rr = row(s, v, r.run_key)
        mres = tifffile.imread(rr.mask_axial_path)
        fz = int(np.argmax((mres > 0).sum((1, 2))))
        mm = mres[fz] > 0
        del mres
        tsl = bbox_of(mm, 220)
        th = norm(page(rr.snpc_image_region_path, fz).astype(float)[tsl], 1, 99.7)
        show(a2, th)
        a2.set_aspect("auto")
        contour(a2, mm[tsl], C_RUN, 0.9)
        scalebar(a2, th.shape, 500, px)
        scalebar(a0, aref.shape, 500, PX_DS)
        if i == 0:
            a0.set_title("atlas (midbrain SSIM box)", fontsize=6.5, pad=2)
            a1.set_title("registered brain in atlas space", fontsize=6.5, pad=2)
            a2.set_title("TH, with this registration's SNpc", fontsize=6.5, pad=2)
        tag(a1, f"SSIM {r.ssim_value:.2f} ({word} of the\n{int(runs(s, v).passes_global.sum())} global-QC passers)",
            fs=5.4)
        tag(a2, f"{int(rr.snpc_CellCount):,} SNpc TH+ cells", fs=5.6)
        a0.text(-0.03, 0.5, f"{LAB[s]}\nbending {r.bending:g}, grid {int(r.grid)}", transform=a0.transAxes, rotation=90,
                ha="right", va="center", fontsize=5.8)
        letter(a0, L, x=-0.1)
        src.append(dict(panel=L, sample=LAB[s], run=r.run_key, ssim=r.ssim_value, iou=r.iou, grad_p95=r.grad_p95,
                        snpc_count=int(rr.snpc_CellCount), atlas_coronal_slice=za, fullres_page=fz))
    leg = [Line2D([], [], color="#FFD92F", lw=1, label="atlas SNpc"),
           Line2D([], [], color="0.6", lw=0.6, label="atlas region borders"),
           Line2D([], [], color=C_RUN, lw=1, label="SNpc of this registration")]
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, fontsize=6)

    # ---- C: merged count panel --------------------------------------------------------------------------------
    axC = fig.add_subplot(gs[2])
    cm = plt.get_cmap("viridis")
    nrm = Normalize(RUNS.ssim_value.min(), RUNS.ssim_value.max())
    stage_fc = {"all": "white", "qc": "0.88", "ssim": "0.62"}
    xt, xl = [], []
    stats = []
    x = 0.0
    for si, ss in enumerate(SAMPLES):
        for vi, vv in enumerate(("raw", "preprocessed")):
            sub = runs(ss, vv)
            stages = {"all": sub, "qc": sub[sub.passes_global],
                      "ssim": sub[sub.run_key.isin(retained(ss, vv))]}
            for k, (st, g) in enumerate(stages.items()):
                strip_box(axC, x + k, g.snpc_CellCount, "0.3", width=0.72, pcolor=g.ssim_value.values, cmap=cm,
                          norm_=nrm, s=4.5, box_fc=stage_fc[st], seed=si * 10 + vi * 3 + k)
                stats.append(dict(sample=LAB[ss], input=vv, stage=st, n=len(g), median=g.snpc_CellCount.median(),
                                  sd=g.snpc_CellCount.std(), min=g.snpc_CellCount.min(),
                                  max=g.snpc_CellCount.max()))
            cons = counter(ss, vv).count(retained(ss, vv), 0.5)
            axC.scatter(x + 2.62, cons, marker="D", s=16, color=C_CONS, edgecolor="white", lw=0.4, zorder=5)
            stats.append(dict(sample=LAB[ss], input=vv, stage="consensus_0.5", n=len(retained(ss, vv)), median=cons))
            axC.text(x + 1, 1180, "raw" if vv == "raw" else "pre-\nprocessed", ha="center", va="top", fontsize=5.6,
                     color=C_RAW if vv == "raw" else C_PRE, linespacing=0.9)
            x += 3.6
        xt.append(x - 3.6 - 1.0)
        x += 1.2
        if si < len(SAMPLES) - 1:
            axC.axvline(x - 1.0, color="0.88", lw=0.6)
    axC.set_xticks(xt, [LAB[ss] for ss in SAMPLES])
    axC.tick_params(axis="x", length=0, pad=17)
    axC.set_xlim(-0.8, x - 1.6)
    axC.set_ylim(1200, 5200)
    axC.set_ylabel("SNpc TH+ cells")
    axC.grid(axis="y", color="0.93", lw=0.4)
    axC.set_axisbelow(True)
    h = [Patch(fc=stage_fc[k], ec="0.3", lw=0.6, label=t) for k, t in
         (("all", "all 48 registrations"), ("qc", "after global QC"), ("ssim", "after SSIM (top 50%)"))]
    h.append(Line2D([], [], ls="", marker="D", ms=3.8, color=C_CONS, label="consensus (inclusion ≥ 0.5)"))
    axC.legend(handles=h, loc="upper left", ncol=4, bbox_to_anchor=(0.0, 1.13), fontsize=6, handlelength=1.4,
               columnspacing=1.2)
    cax = axC.inset_axes([0.86, 1.06, 0.13, 0.035])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=nrm, cmap=cm), cax=cax, orientation="horizontal")
    cbar.set_label("midbrain SSIM (points)", fontsize=5.6, labelpad=1)
    cbar.ax.tick_params(labelsize=5.2, length=1.5, pad=1)
    cbar.outline.set_linewidth(0.4)
    letter(axC, "C", x=-0.045, y=1.04)
    pd.DataFrame(stats).to_csv(SRC / "Fig3C_counts_by_stage.csv", index=False)

    # ---- D: consensus construction, E: thresholds on TH -----------------------------------------------------
    gDE = gs[3].subgridspec(1, 3, width_ratios=[0.95, 1.0, 1.45], wspace=0.06)
    masks = {k: tifffile.imread(row(s, v, k).snpc_mask_path) > 0 for k in ret}
    prob = np.mean([m.astype(np.float32) for m in masks.values()], axis=0)
    zc = int(np.argmax((prob >= 0.5).sum((1, 2))))
    p2 = prob[zc]
    sl = bbox_of(p2 > 0, 10)
    dd = rerun_dir(s, v, *(row(s, v, ret[0])[["bending", "grid"]]))
    bg = norm(page(dd / "downsampled.tiff", zc).astype(float)[sl], 1, 99.7)
    gD = gDE[0].subgridspec(2, 2, wspace=0.03, hspace=0.05)
    for k in range(4):
        a = fig.add_subplot(gD[k // 2, k % 2])
        show(a, bg)
        a.set_aspect("auto")
        contour(a, masks[ret[k]][zc][sl], C_RUN, 0.7)
        a.text(0.04, 0.95, f"{ret[k].replace('_', ' / ')}", transform=a.transAxes, color="white", fontsize=4.6,
               va="top")
        if k == 0:
            letter(a, "D", x=-0.05)
            a.set_title(f"retained registrations ({len(ret)})", fontsize=6.2, pad=2, loc="left")
    a = fig.add_subplot(gDE[1])
    show(a, bg)
    a.set_aspect("auto")
    pc = plt.get_cmap("Purples")
    im = a.imshow(np.ma.masked_equal(p2[sl], 0), cmap=pc, vmin=0, vmax=1, alpha=0.9, interpolation="nearest",
                  aspect="auto")
    contour(a, p2[sl] >= 0.5, C_CONS, 1.0)
    a.set_title("inclusion frequency", fontsize=6.2, pad=2)
    cax = a.inset_axes([0.06, 0.08, 0.5, 0.05])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=5, colors="white", length=1.5, pad=1)
    cb.outline.set_edgecolor("white")
    cb.outline.set_linewidth(0.4)
    scalebar(a, bg.shape, 500, PX_DS, frac_x=0.62)
    a.annotate("", xy=(-0.02, 0.5), xytext=(-0.13, 0.5), xycoords="axes fraction",
               arrowprops=dict(arrowstyle="-|>", color="0.3", lw=0.8, mutation_scale=7))
    # E: full-resolution consensus at three thresholds on TH
    aE = fig.add_subplot(gDE[2])
    r0 = row(s, v, ret[0])
    fr = [tifffile.imread(row(s, v, k).mask_axial_path) for k in ret[:1]]
    fz = int(np.argmax((fr[0] > 0).sum((1, 2))))
    del fr
    pf = np.mean([page(row(s, v, k).mask_axial_path, fz) > 0 for k in ret], axis=0)
    esl = bbox_of(pf > 0, 160)
    pf = pf[esl]
    th = norm(page(r0.snpc_image_region_path, fz).astype(float)[esl], 1, 99.7)
    show(aE, th)
    aE.set_aspect("auto")
    styles = [(0.1, ":", "#e7d4e8", 0.9), (0.5, "-", C_CONS, 1.0), (1.0, "--", "#FFD92F", 0.8)]
    for t, ls, col, lw in styles:
        contour(aE, pf >= t - 1e-6, col, lw, ls)
    scalebar(aE, th.shape, 500, px)
    aE.set_title("consensus SNpc on TH at three inclusion thresholds", fontsize=6.2, pad=2)
    aE.legend(handles=[Line2D([], [], color=c, ls=ls, lw=1, label=f"≥ {t:g}") for t, ls, c, _ in styles],
              loc="lower right", fontsize=5.6, labelcolor="white", handlelength=1.8, ncol=3, columnspacing=0.8)
    letter(aE, "E", x=-0.02)

    # ---- F, G: counts vs threshold ------------------------------------------------------------------------------
    gFG = gs[4].subgridspec(1, 4, width_ratios=[1, 1, 1.1, 1.1], wspace=0.3)
    thr = la.THR
    curves = []
    axF = fig.add_subplot(gFG[0])
    axG = fig.add_subplot(gFG[1], sharey=axF)
    for ss in SAMPLES:
        for ax_, key, sel in ((axF, "retained", retained(ss, v)), (axG, "all", list(runs(ss, v).run_key))):
            y = [counter(ss, v).count(sel, t) for t in thr]
            ax_.plot(thr, y, color=SCOL[ss], lw=1.0, marker="o", ms=1.6, label=LAB[ss])
            curves += [dict(sample=LAB[ss], input=v, run_set=key, n_runs=len(sel), threshold=t, count=c)
                       for t, c in zip(thr, y)]
    for ax_, t in ((axF, f"after global QC + SSIM"), (axG, "all 48 registrations")):
        ax_.axvline(0.5, color="0.5", ls=":", lw=0.7)
        ax_.set_xlabel("inclusion threshold")
        ax_.set_title(t, fontsize=6.3, pad=2)
        ax_.set_xlim(0, 1.02)
        ax_.grid(axis="y", color="0.93", lw=0.4)
    axF.set_ylabel("consensus SNpc TH+ cells")
    axF.set_ylim(0, 7000)
    plt.setp(axG.get_yticklabels(), visible=False)
    axF.legend(loc="lower left", fontsize=5.4, handlelength=1.2, ncol=1, borderaxespad=0.2)
    letter(axF, "F", x=-0.12)
    letter(axG, "G", x=-0.02)
    pd.DataFrame(curves).to_csv(SRC / "Fig3FG_counts_vs_threshold.csv", index=False)

    # ---- H, I: per-cell inclusion frequency -------------------------------------------------------------------
    lab_path = la.lc.label_path(s, v)
    pf_all = np.mean([page(row(s, v, kk).mask_axial_path, fz) > 0 for kk in allk], axis=0)
    hsl = bbox_of(pf_all > 0, 90)
    labels = page(lab_path, fz)[hsl]
    thc = page(r0.snpc_image_region_path, fz).astype(float)[hsl]
    cmap_c = LinearSegmentedColormap.from_list("inc", ["#fde0ef", "#c51b7d", "#40004b"])
    cmap_c = plt.get_cmap("plasma")
    cell_rows = []
    for k, (L, keys, t) in enumerate((("H", ret, f"inclusion frequency per cell:\nretained ({len(ret)})"),
                                      ("I", allk, "inclusion frequency per cell:\nall 48 registrations"))):
        a = fig.add_subplot(gFG[2 + k])
        inc = cell_inclusion(s, v, keys)
        f2 = inc[labels]
        f2[labels == 0] = np.nan
        base = norm(thc, 1, 99.7) * 0.55
        rgb = np.dstack([base] * 3)
        on = ~np.isnan(f2)
        zero = on & (f2 == 0)
        rgb[zero] = [0.55, 0.55, 0.55]
        pos = on & (f2 > 0)
        rgb[pos] = cmap_c(f2[pos])[:, :3]
        a.imshow(rgb, interpolation="nearest", aspect="auto")
        clean(a)
        pf_k = np.mean([page(row(s, v, kk).mask_axial_path, fz) > 0 for kk in keys], axis=0)[hsl]
        contour(a, pf_k >= 0.5, "white", 0.7)
        a.set_title(t, fontsize=6.2, pad=2)
        ids = np.unique(labels[labels > 0])
        fr_ = inc[ids]
        n_mid = int(((fr_ > 0) & (fr_ < 1)).sum())
        tag(a, f"partially included (0 < f < 1):\n{n_mid} of {int((fr_ > 0).sum())} cells", fs=5.2, loc="tl")
        letter(a, L, x=-0.02)
        cell_rows += [dict(panel=L, sample=LAB[s], input=v, run_set=("retained" if k == 0 else "all"),
                           cell_id=int(c_), inclusion_frequency=float(inc[c_])) for c_ in ids]
        if k == 1:
            scalebar(a, labels.shape, 500, px)
            cax = a.inset_axes([1.03, 0.1, 0.05, 0.8])
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap_c), cax=cax, ticks=[0, 0.5, 1])
            cb.ax.tick_params(labelsize=5.2, length=1.5)
            cb.outline.set_linewidth(0.4)
            cb.set_label("inclusion frequency", fontsize=5.6)
    pd.DataFrame(cell_rows).to_csv(SRC / "Fig3HI_cell_inclusion.csv", index=False)
    pd.DataFrame(src + [dict(panel="D-I", sample=LAB[s], input=v, retained=";".join(ret), coronal_slice_20um=zc,
                             fullres_page=fz)]).to_csv(SRC / "Fig3_source.csv", index=False)
    save(fig, "Fig3_SSIM_and_consensus")


# ============================================================================================================
# Figure 4
# ============================================================================================================
def any_run_dir(s, v):
    return next(p for p in sorted((REG / s / v).iterdir()) if (p / "downsampled.tiff").exists())


def fig4():
    sB = "xl353"
    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.3)
    gTop = gs[0].subgridspec(1, 2, width_ratios=[0.95, 1.6], wspace=0.15)
    src = []

    # ---- A: histograms ----------------------------------------------------------------------------------------
    ref, ann = atlas()
    vols = {}
    for vv in ("raw", "preprocessed"):
        vols[vv] = tifffile.imread(any_run_dir(sB, vv) / "downsampled.tiff")
    gA = gTop[0].subgridspec(3, 1, hspace=0.45)
    hist_rows = []
    bins = np.linspace(0, 450, 91)
    for k, (nm, arr, col) in enumerate((("raw autofluorescence", vols["raw"], C_RAW),
                                        ("atlas reference", ref, "0.35"),
                                        ("preprocessed", vols["preprocessed"], C_PRE))):
        a = fig.add_subplot(gA[k])
        x = arr[arr > 0].astype(np.float32)
        hi_ = float(np.percentile(x, 99.9))
        bins = np.linspace(0, hi_, 81)
        hh, _ = np.histogram(np.clip(x, 0, hi_ - 1e-3), bins=bins)
        frac = 100 * hh / hh.sum()
        a.bar(bins[:-1], frac, width=np.diff(bins), align="edge", color=col, lw=0)
        a.set_xlim(0, hi_)
        a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        a.text(0.98, 0.85, nm, transform=a.transAxes, ha="right", va="top", fontsize=6.2, color=col)
        a.set_ylabel("% voxels", fontsize=6)
        if k == 2:
            a.set_xlabel("intensity (a.u.; each axis to its 99.9th percentile)")
        hist_rows += [dict(image=nm, bin_lo=b0, bin_hi=b1, pct_voxels=f) for b0, b1, f in zip(bins[:-1], bins[1:], frac)]
        if k == 0:
            letter(a, "A", x=-0.2)
            a.set_title(f"{LAB[sB]} (non-zero voxels)", fontsize=6.3, pad=2)
    pd.DataFrame(hist_rows).to_csv(SRC / "Fig4A_histograms.csv", index=False)

    # ---- B: horizontal sections -------------------------------------------------------------------------------
    gB = gTop[1].subgridspec(1, 3, wspace=0.04)
    yB, yA = 108, 141                                                    # sections used in the original figure
    imgs = [(norm(vols["raw"][:, yB, :].astype(float), 1, 99.7), "raw", C_RAW),
            (norm(vols["preprocessed"][:, yB, :].astype(float), 1, 99.7), "preprocessed", C_PRE),
            (norm(ref[:, yA, :].astype(float), 0.5, 99.8), "atlas", "0.2")]
    for k, (im, t, col) in enumerate(imgs):
        a = fig.add_subplot(gB[k])
        from skimage.filters import threshold_otsu
        tissue = ndi.binary_fill_holes(im > threshold_otsu(im))
        lab_, _ = ndi.label(ndi.binary_opening(tissue, iterations=2))
        sz = np.bincount(lab_.ravel())
        big = np.isin(lab_, np.flatnonzero(sz > 0.02 * sz[1:].sum())) & (lab_ > 0)
        sl = bbox_of(big, 4)
        show(a, im[sl])
        a.set_title(t, fontsize=6.5, color=col, pad=2)
        scalebar(a, im[sl].shape, 1000, PX_DS)
        if k == 0:
            letter(a, "B", x=-0.03)
    del vols

    # ---- C: same parameters, raw vs preprocessed -------------------------------------------------------------
    gBot = gs[1].subgridspec(1, 3, width_ratios=[0.85, 1.0, 1.05], wspace=0.32)
    gC = gBot[0].subgridspec(2, 1, hspace=0.06)
    sC, bC, grC, zC = "pp451", 0.5, -3, 483
    for k, (vv, col) in enumerate((("raw", C_RAW), ("preprocessed", C_PRE))):
        a = fig.add_subplot(gC[k])
        dd = rerun_dir(sC, vv, bC, grC)
        img = page(dd / "downsampled.tiff", zC).astype(float)
        bnd = page(dd / "boundaries.tiff", zC) > 0
        sl = bbox_of(ndi.binary_opening(img > np.percentile(img, 60), iterations=2), 6)
        show(a, norm(img[sl], 1, 99.7))
        rgba = np.zeros(bnd[sl].shape + (4,), np.float32)
        rgba[bnd[sl]] = list(matplotlib.colors.to_rgb(col)) + [0.9]
        a.imshow(rgba, interpolation="nearest")
        r = row(sC, vv, f"{bC:g}_{grC}")
        tag(a, f"{vv}\nIoU {r.iou:.3f} · grad_p95 {r.grad_p95:.3f}", fs=5.2)
        tag(a, "pass" if r.passes_global else "fail", fs=5.8, loc="tr", color=C_PASS if r.passes_global else C_FAIL)
        scalebar(a, img[sl].shape, 1000, PX_DS)
        src.append(dict(panel="C", sample=LAB[sC], input=vv, bending=bC, grid=grC, iou=r.iou, grad_p95=r.grad_p95,
                        passes_global_qc=bool(r.passes_global), coronal_slice=zC))
        if k == 0:
            letter(a, "C", x=-0.03)
            a.set_title(f"{LAB[sC]}, bending {bC:g}, grid {grC}", fontsize=6.3, pad=2)

    # ---- D: pass rate per brain ---------------------------------------------------------------------------------
    aD = fig.add_subplot(gBot[1])
    w = 0.38
    for i, ss in enumerate(SAMPLES):
        for j, (vv, col) in enumerate((("raw", C_RAW), ("preprocessed", C_PRE))):
            n = int(runs(ss, vv).passes_global.sum())
            xx = i + (j - 0.5) * w
            aD.bar(xx, 100 * n / 48, width=w * 0.92, color=col, lw=0)
            aD.text(xx, 100 * n / 48 + 1.2, f"{n}", ha="center", va="bottom", fontsize=5.4)
            src.append(dict(panel="D", sample=LAB[ss], input=vv, n_pass=n, n=48, pct=100 * n / 48))
    aD.set_xticks(range(len(SAMPLES)), [LAB[x].replace("Sample ", "S") for x in SAMPLES])
    aD.set_ylabel("registrations passing global QC (%)")
    aD.set_ylim(0, 85)
    aD.set_xlabel("brain (numbers above bars: n of 48)")
    aD.legend(handles=[Patch(color=C_RAW, label="raw"), Patch(color=C_PRE, label="preprocessed")],
              loc="upper left", fontsize=6)
    aD.grid(axis="y", color="0.93", lw=0.4)
    aD.set_axisbelow(True)
    letter(aD, "D", x=-0.14)

    # ---- E: shift of the QC landscape ---------------------------------------------------------------------------
    aE = fig.add_subplot(gBot[2])
    for vv, col, a_, z_ in (("raw", C_RAW, 0.55, 2), ("preprocessed", C_PRE, 0.8, 3)):
        d = RUNS[RUNS.status == vv]
        aE.scatter(d.iou, d.grad_p95, s=6, color=col, alpha=a_, lw=0.2, edgecolor="white", zorder=z_,
                   label=f"{vv}: {int(d.passes_global.sum())}/240 pass")
    aE.axvline(IOU_T, color="0.25", ls="--", lw=0.7)
    aE.axhline(GRAD_T, color="0.25", ls="--", lw=0.7)
    aE.set_xlim(0.83, 0.975)
    aE.set_ylim(0.028, 0.095)
    aE.set_xlabel("tissue–atlas IoU")
    aE.set_ylabel("deformation (grad_p95)")
    aE.legend(loc="upper right", fontsize=6, handletextpad=0.1, markerscale=1.4)
    letter(aE, "E", x=-0.16)
    save(fig, "Fig4_preprocessing")
    pd.DataFrame(src).to_csv(SRC / "Fig4_source.csv", index=False)


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    for name, fn in (("fig1", fig1), ("fig2", fig2), ("fig3", fig3), ("fig4", fig4)):
        if "all" in which or name in which:
            fn()
