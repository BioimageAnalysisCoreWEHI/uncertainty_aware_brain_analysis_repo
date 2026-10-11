"""Figure 2 (v6): preprocessing, global QC (raw vs preprocessed), local SSIM, and SNpc counts at each QC stage.
Implements mockups/Fig2_v6_mockup.png with real data only.

  A  axial sections of Sample 4 (raw | preprocessed | atlas; the old Fig 4B images) + intensity distributions of the
     actual registration inputs of Sample 4 and the atlas reference (non-zero voxels)
  B  tissue-atlas IoU vs grad_p95 for all 240 raw and 240 preprocessed registrations; arrows join three matched
     registrations of Sample 3 (same parameters) from raw to preprocessed input
  C/D the three matched registrations, raw (C) and preprocessed (D): coronal section through the SNpc with the
     registered atlas (thick: outer outline used for IoU; thin: region borders), green = passes global QC, red = fails
  E/F the old Fig 3A/B examples (brain pp455, illustrative): atlas midbrain crop, registered brain, TH with SNpc
  G  SNpc TH+ counts per brain at each stage, raw and preprocessed (numbers: passed global QC / retained, of 48)

Numbers: final_count_results/full_final_results_with_ssim.csv (via validation_analysis/cache/lsfm/runs.csv);
selection = global QC IoU > 0.9 and grad_p95 < 0.045, then top ceil(n/2) by midbrain SSIM.
Images: original pipeline masks/crops; whole-brain sections and borders from the rerun of the same sweep (display only).
Output: manuscript_figures/Fig2_v6.{png,pdf,svg} (600 dpi), Fig2_v6_source_data.csv, Fig2_v6_caption.md
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P  # noqa: E402  (resolves <IMAGING>/<SCRATCH> placeholders)

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

VA = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_analysis"))
_REPO = next(_p for _p in Path(__file__).resolve().parents if (_p / "uaa_paths.py").exists())
_CODE = _REPO / "validation" / "analysis" / "scripts"          # analysis scripts in this repository
_s = importlib.util.spec_from_file_location("m35", str(_CODE / "35_new_main_figures.py"))
m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m)

OUT = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\manuscript_figures"))
OLD4 = OUT / "figures" / "Figure4 - Preprocessing"
J = Path(P(r"<IMAGING>\Ishrat\JoM"))
C_RAW, C_PRE, C_ATL = "#E08214", "#2C7BB6", "0.25"
RAW_SH = ["#fde0c2", "#f4a259", "#d9690b"]           # all / passed global QC / retained (light -> dark)
PRE_SH = ["#d4e6f5", "#7fb2db", "#1f6aa8"]
C_PASS, C_FAIL = "#2ca25f", "#c0392b"
# Sample 3 (pp451): the four raw examples of the old Fig 2 (figures/Figure2 - Global QC, params in the file names),
# one per failure mode, each with the preprocessed registration of the same parameters
PAIRS = ["0.5_-3", "0.3_-5", "1_-10", "0.7_-10"]
Z_SECTION = 483                            # one coronal level for all eight sections (as the old Fig 4C pair)
S_PAIRS, S_SSIM, S_HIST = "pp451", "527", "xl353"


def L(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


# ------------------------------------------------------------------------------------------------------------
def panel_a(fig, gs, src):
    g = gs.subgridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.22)
    gi = g[0].subgridspec(1, 3, wspace=0.03)
    files = [("raw", OLD4 / "axial_xl353_slice108_raw_1000um_scale.tif", C_RAW),
             ("preprocessed", OLD4 / "axial_xl353_slice108_preprocessed_1000um_scale.tif", C_PRE),
             ("atlas", OLD4 / "axial_reference_atlas_slice141_1000um_scale.tif", C_ATL)]
    for k, (t, f, c) in enumerate(files):
        a = fig.add_subplot(gi[k])
        img = tifffile.imread(f)
        a.imshow(img, interpolation="nearest")
        m.clean(a)
        a.set_title(t, fontsize=6.8, color=c, pad=2)
        if k == 0:
            L(a, "A")
    ah = fig.add_subplot(g[1])
    vols = {"raw": J / "final_preprocessed_stacks_raw" / f"{S_HIST}_AF_level1_preprocessed",
            "preprocessed": J / "final_preprocessed_stacks" / f"{S_HIST}_AF_level1_preprocessed"}
    data = {}
    for k, folder in vols.items():
        fs = sorted(folder.glob("*.tif*"))
        v = np.stack([tifffile.imread(f) for f in fs]).ravel()
        data[k] = v[v > 0]
    ref, _ = m.atlas()
    data["atlas"] = ref[ref > 0].ravel().astype(np.float32)
    xmax = float(np.percentile(data["raw"], 99.5))
    bins = np.linspace(0, xmax, 241)
    ctr = (bins[1:] + bins[:-1]) / 2
    for k, c, z in (("raw", C_RAW, 3), ("atlas", C_ATL, 1), ("preprocessed", C_PRE, 2)):
        h, _ = np.histogram(np.clip(data[k], 0, xmax), bins=bins)
        h = h / h.max()
        ah.fill_between(ctr, h, color=c, alpha=0.18 if k != "atlas" else 0.22, lw=0, zorder=z)
        ah.plot(ctr, h, color=c, lw=1.0, label=k, zorder=z)
        src += [dict(panel="A", image=k, sample=m.LAB[S_HIST] if k != "atlas" else "atlas", bin_centre=b_, fraction_of_peak=y_)
                for b_, y_ in zip(ctr, h)]
    ah.set_xlim(0, xmax)
    ah.set_ylim(0, 1.05)
    ah.set_yticks([])
    ah.set_xlabel("intensity (a.u.)")
    ah.set_ylabel("voxels (normalised to peak)")
    ah.legend(loc="upper right", fontsize=6, handlelength=1.4)
    ah.set_title(f"intensity distributions ({m.LAB[S_HIST]}, non-zero voxels)", fontsize=6.5, pad=3)


def panel_b(fig, ax, src):
    d = m.RUNS
    for vv, c, mk, z in (("raw", C_RAW, "o", 2), ("preprocessed", C_PRE, "^", 3)):
        x = d[d.status == vv]
        n = int(x.passes_global.sum())
        ax.scatter(x.iou, x.grad_p95, s=7 if mk == "o" else 8, marker=mk, color=c, alpha=0.75, lw=0.2,
                   edgecolor="white", zorder=z, label=f"{vv}: {n}/240 pass")
    ax.axvline(m.IOU_T, color="0.25", ls="--", lw=0.7)
    ax.axhline(m.GRAD_T, color="0.25", ls="--", lw=0.7)
    ax.add_patch(Rectangle((m.IOU_T, 0.0), 0.2, m.GRAD_T, color=C_PASS, alpha=0.08, lw=0, zorder=0))
    ax.text(0.968, 0.0305, "pass", color=C_PASS, fontsize=6, ha="right", va="bottom")
    for i, key in enumerate(PAIRS, 1):
        r0, r1 = m.row(S_PAIRS, "raw", key), m.row(S_PAIRS, "preprocessed", key)
        ax.annotate("", xy=(r1.iou, r1.grad_p95), xytext=(r0.iou, r0.grad_p95),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8, mutation_scale=7,
                                    shrinkA=4, shrinkB=4), zorder=5)
        for r, c in ((r0, C_RAW), (r1, C_PRE)):
            ax.scatter(r.iou, r.grad_p95, s=46, color=c, edgecolor="black", lw=0.8, zorder=6)
            ax.text(r.iou, r.grad_p95, str(i), color="white", fontsize=5.5, fontweight="bold", ha="center",
                    va="center", zorder=7)
        src.append(dict(panel="B-D", pair=i, sample=m.LAB[S_PAIRS], run=key, bending=r0.bending, grid=r0.grid,
                        iou_raw=r0.iou, grad_p95_raw=r0.grad_p95, pass_raw=bool(r0.passes_global),
                        iou_preprocessed=r1.iou, grad_p95_preprocessed=r1.grad_p95,
                        pass_preprocessed=bool(r1.passes_global)))
    ax.set_xlim(0.825, 0.97)
    ax.set_ylim(0.03, 0.093)
    ax.set_xlabel("tissue–atlas IoU")
    ax.set_ylabel("deformation (grad_p95)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=5.8, handletextpad=0.2, markerscale=1.5,
              borderaxespad=0.1, frameon=True, framealpha=0.9, edgecolor="none")
    ax.set_title("Global QC, raw vs preprocessed input", fontsize=6.8, pad=3)
    L(ax, "B", x=-0.17)


def section(fig, ax, s, v, key, z):
    r = m.row(s, v, key)
    dd = m.rerun_dir(s, v, r.bending, r.grid)
    img = m.page(dd / "downsampled.tiff", z).astype(float)
    bnd = m.page(dd / "boundaries.tiff", z) > 0
    sl = m.bbox_of(ndi.binary_opening(img > np.percentile(img, 60), iterations=2), 6)
    m.show(ax, m.norm(img[sl], 1, 99.7))
    col = C_PASS if r.passes_global else C_FAIL
    rgba = np.zeros(bnd[sl].shape + (4,), np.float32)
    rgba[bnd[sl]] = list(matplotlib.colors.to_rgb(col)) + [1.0]
    ax.imshow(rgba, interpolation="nearest")
    m.scalebar(ax, img[sl].shape, 1000, m.PX_DS, fs=5)
    m.tag(ax, "pass" if r.passes_global else "fail", fs=5.8, loc="tr", color=col)
    return r


def panel_cd(fig, gs, src):
    g = gs.subgridspec(2, 4, wspace=0.04, hspace=0.42)
    for i, key in enumerate(PAIRS):
        z = Z_SECTION
        for j, v in enumerate(("raw", "preprocessed")):
            a = fig.add_subplot(g[j, i])
            r = section(fig, a, S_PAIRS, v, key, z)
            why = [t for t, bad in (("IoU", r.iou <= m.IOU_T), ("grad_p95", r.grad_p95 >= m.GRAD_T)) if bad]
            verdict = "pass" if r.passes_global else "fails " + " and ".join(why)
            head = f"{i + 1}  {verdict}" if v == "raw" else f"{i + 1}  same params: {verdict}"
            a.set_title(f"{head}\nIoU {r.iou:.3f} · grad {r.grad_p95:.3f}", fontsize=5.3, pad=2, loc="left",
                        linespacing=1.15)
            if i == 0:
                a.text(-0.08, 0.5, "raw" if v == "raw" else "preprocessed", transform=a.transAxes, rotation=90,
                       ha="right", va="center", fontsize=6.8, color=C_RAW if v == "raw" else C_PRE)
                L(a, "C" if v == "raw" else "D", x=-0.14, y=1.18)
            src.append(dict(panel="C" if v == "raw" else "D", pair=i + 1, sample=m.LAB[S_PAIRS], run=key,
                            input=v, iou=r.iou, grad_p95=r.grad_p95, passes_global_qc=bool(r.passes_global),
                            coronal_slice=z))


def panel_ef(fig, gs_e, gs_f, src):
    """Old Fig 3A/B images (brain pp455, an additional brain outside the five quantified brains), as exported:
    atlas midbrain crop | registered brain (coronal, atlas space) | TH with the registration's SNpc (red)."""
    F3 = OUT / "figures" / "Figure3 - Local QC + SSIM"
    atlas = trim(plt.imread(F3 / "Atlas AF (reference)_1000um_scale_napari.png"))
    rows = (("E", gs_e, "Low local SSIM (midbrain box): misaligned landmarks", "pp455_0.8_-40_-2_coronal_low_ssim.png",
             "pp455_0.8_-40_-2_slice84_500um_scale_napari.png", 1695, (0.8, -40, -2)),
            ("F", gs_f, "High local SSIM: SNpc boundary over the TH+ cluster", "pp455_0.4_-10_-1_coronal_high_ssim.png",
             "pp455_0.4_-10_-1_slice95_500um_scale_napari.png", 4089, (0.4, -10, -1)))
    for letter, gsx, title, cor, th, count, (be, gs_, sm) in rows:
        reg = trim(plt.imread(F3 / cor))
        thi = trim(plt.imread(F3 / th))
        ims = (atlas, reg, thi)
        g = gsx.subgridspec(1, 3, wspace=0.03, width_ratios=[im.shape[1] / im.shape[0] for im in ims])
        axs = [fig.add_subplot(g[k]) for k in range(3)]
        for a, im in zip(axs, ims):
            a.imshow(im, interpolation="nearest", aspect="auto")
            m.clean(a)
        if letter == "E":
            axs[0].set_title("atlas (midbrain box)", fontsize=6, pad=2)
            axs[1].set_title("registered brain in atlas space", fontsize=6, pad=2)
            axs[2].set_title("TH with this registration's SNpc", fontsize=6, pad=2)
        m.tag(axs[2], f"{count:,} SNpc TH+ cells", fs=5.8)
        y = 1.17 if letter == "E" else 1.04
        axs[0].text(0.0, y, f"{title} (bending {be:g}, grid {gs_}, smoothing {sm})", transform=axs[0].transAxes,
                    fontsize=6.5, va="bottom", ha="left")
        L(axs[0], letter, x=-0.02, y=y)
        src.append(dict(panel=letter, brain="pp455 (illustrative, not one of Samples 1-5)", bending=be, grid=gs_,
                        smoothing=sm, snpc_count_as_in_old_figure=count, images=f"{cor}; {th}"))


def trim(img, thr=0.04):
    """Crop the black margin of an exported image (keeps burned-in scale bars inside the image)."""
    g = img[..., :3].mean(-1) if img.ndim == 3 else img
    rr, cc = np.flatnonzero((g > thr).any(1)), np.flatnonzero((g > thr).any(0))
    return img[rr[0]:rr[-1] + 1, cc[0]:cc[-1] + 1]


def panel_g(fig, ax, src):
    x, xt = 0.0, []
    top = 5450
    for si, s in enumerate(m.SAMPLES):
        for vv, sh, c in (("raw", RAW_SH, C_RAW), ("preprocessed", PRE_SH, C_PRE)):
            sub = m.runs(s, vv)
            ret = m.retained(s, vv)
            stages = [("all", sub), ("global", sub[sub.passes_global]), ("retained", sub[sub.run_key.isin(ret)])]
            for k, (st, g) in enumerate(stages):
                m.strip_box(ax, x + k * 0.8, g.snpc_CellCount, "0.2", width=0.62, pcolor="0.35", s=2.2,
                            box_fc=sh[k], seed=si * 10 + k + (5 if vv != "raw" else 0), lw=0.6)
                vals = g.snpc_CellCount
                src.append(dict(panel="G", sample=m.LAB[s], input=vv, stage=st, n=len(g), median=vals.median(),
                                sd=vals.std(), min=vals.min(), max=vals.max()))
            ax.text(x + 0.8, top, f"{int(sub.passes_global.sum())}\n{len(ret)}", ha="center", va="bottom",
                    fontsize=5.6, color=c, linespacing=1.0)
            ax.text(x + 0.8, 1060, "raw" if vv == "raw" else "pre", ha="center", va="bottom", fontsize=5.6, color=c)
            x += 2.9
        xt.append(x - 2.9 - 1.25)
        x += 0.9
    ax.set_xticks(xt, [m.LAB[s] for s in m.SAMPLES])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, x - 1.3)
    ax.set_ylim(1000, 5950)
    ax.set_yticks(range(1000, 5001, 1000))
    ax.set_ylabel("SNpc TH+ cell count")
    ax.grid(axis="y", color="0.93", lw=0.4)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Count at each QC stage, raw and preprocessed (numbers on top: passed global QC / retained, of 48)",
                 fontsize=6.8, pad=3, loc="left")
    h = [Patch(fc=RAW_SH[k], ec="0.2", lw=0.6, label=f"raw: {t}") for k, t in
         enumerate(("all 48", "passed global QC", "+ top 50% SSIM (retained)"))]
    h += [Patch(fc=PRE_SH[k], ec="0.2", lw=0.6, label=f"pre: {t}") for k, t in
          enumerate(("all 48", "passed global QC", "+ top 50% SSIM (retained)"))]
    ax.legend(handles=[h[0], h[3], h[1], h[4], h[2], h[5]], loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3,
              fontsize=6, handlelength=1.5, columnspacing=1.5)
    L(ax, "G", x=-0.05, y=1.01)


def main():
    src = []
    fig = plt.figure(figsize=(7.2, 10.3))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.05, 1.45, 1.05, 1.35], hspace=0.42)
    panel_a(fig, gs[0], src)
    gb = gs[1].subgridspec(1, 2, width_ratios=[0.85, 2.0], wspace=0.2)
    panel_b(fig, fig.add_subplot(gb[0]), src)
    panel_cd(fig, gb[1], src)
    gef = gs[2].subgridspec(2, 1, hspace=0.42)
    panel_ef(fig, gef[0], gef[1], src)
    panel_g(fig, fig.add_subplot(gs[3]), src)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"Fig2_v6.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.DataFrame(src).to_csv(OUT / "Fig2_v6_source_data.csv", index=False)
    print("saved", OUT / "Fig2_v6.png")


if __name__ == "__main__":
    main()
