# -*- coding: utf-8 -*-
"""
18_fig4v2_independent_lsfm.py

Revision Fig. 4 v2 (manuscript Fig. 7): the independent LSFM results in one figure = original Fig. 4 (examples and
per-region agreement) + the revised Supplementary Fig. S6 (midbrain). Raw input, final QC set, threshold 0.5.
  A-C  example regions (same fixed rule as 22_focused_revision_figures.fig4): highest median Dice; median Dice nearest
       the median of regions; best region whose consensus is at least as large as the reference. Median-Dice brain.
  D    3-D Dice with the published registration for 16 regions (largest -> smallest), one dot per brain; the median
       mean boundary distance of each region is printed under its name.
  E    midbrain close-up of all six brains: published SNc (white dashed) and VTA (orange dotted), retained single
       registrations' SNc (thin blue), consensus SNc (purple); SNc Dice and agreement between retained registrations.
  F    agreement between retained registrations (SNc, median pairwise Dice) vs consensus SNc Dice, per brain.
  G    marker brightness (x median tissue intensity) in the published VTA, published SNc, a 100 um shell around the
       published SNc, and the consensus SNc.
  H    midbrain marker brightness (published VTA) vs agreement between retained registrations; Spearman rho.
Inputs: results/validation_gt/consensus_vs_gt.csv, results/fig8_region_masks/, results/midbrain_inspection/
midbrain_inspection.csv. Read-only. The original Fig. 4 and Fig. S6 files are kept.
Output: REVISION_OUTPUTS_FOCUSED/figures/Fig4_independent_LSFM_agreement_v2.{pdf,svg,png} + source_data/Fig4v2_*.csv
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import stats
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from gt_utils import (ROOT, FINAL_QC_SUBSET, SAMPLES, load_label_mapping, load_gt_atlas, gt_region_mask, image_path,
                      region_dir, load_mask, read_tiff, consensus_path)  # noqa: E402
from pipeline_common import make_output_name  # noqa: E402

PRA = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis"))
FIG, SRC = PRA / "REVISION_OUTPUTS_FOCUSED" / "figures", PRA / "REVISION_OUTPUTS_FOCUSED" / "source_data"
MASKS = ROOT / "results" / "fig8_region_masks"
INSP = ROOT / "results" / "midbrain_inspection" / "midbrain_inspection.csv"
ORDER = ["GAD2", "Rep2-GAD2", "TH", "Rep2-TH", "vGlut2", "Rep2-vGlut2"]
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
COL = dict(zip(sorted(SAMPLES), OKABE))
C_REF, C_VTA, C_RUN, C_CONS = "white", "#E69F00", "#56B4E9", "#7B3294"
BOX = dict(facecolor="black", alpha=0.6, pad=1.2, lw=0)
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 6.5,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "svg.fonttype": "none"})


def outline(ax, m, color, lw, ls="-", alpha=1.0):
    if m.any():
        ax.contour(m.astype(float), [0.5], colors=[color], linewidths=lw, linestyles=[ls], alpha=alpha)


def clean(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def letter(ax, s, x=-0.04, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def scalebar(ax, h, px=20, txt="500 µm"):
    ax.plot([5, 5 + px], [h - 4, h - 4], color="white", lw=1.6, solid_capstyle="butt")
    ax.text(5, h - 6, txt, color="white", ha="left", va="bottom", fontsize=5.2)


def main(view="coronal"):
    """view: "coronal" (v2) or "axial" (v3: panel E shows the horizontal section through the SNc, cropped to the same
    medio-lateral extent as the coronal close-up; all other panels identical)."""
    tag = "v2" if view == "coronal" else "v3"
    _, agg = load_label_mapping()
    c = pd.read_csv(ROOT / "results" / "validation_gt" / "consensus_vs_gt.csv")
    c = c[(c.subset == FINAL_QC_SUBSET) & (c.threshold == 0.5) & (c.input_type == "raw")]
    st = pd.read_csv(INSP).set_index("sample").loc[ORDER]
    fig = plt.figure(figsize=(7.2, 9.0))
    outer = fig.add_gridspec(4, 1, height_ratios=[0.75, 0.95, 1.3, 1.2], hspace=0.36, top=0.955, bottom=0.04)

    # ---------------- A-C examples (rule as in the original Fig. 4)
    stat = c.groupby("region").agg(dice=("dice", "median"), vol_ratio=("vol_ratio", "median"))
    a = stat.dice.idxmax()
    b_ = (stat.dice - stat.dice.median()).abs().drop(a).idxmin()
    cand = stat.drop([a, b_])
    cc_ = cand[cand.vol_ratio > 1].dice.idxmax()
    g0 = outer[0].subgridspec(1, 3, wspace=0.05)
    srcABC = []
    for j, reg in enumerate([a, b_, cc_]):
        g = c[c.region == reg].sort_values("dice").reset_index(drop=True)
        r = g.iloc[(len(g) - 1) // 2]
        gt = load_gt_atlas(r.sample_id)
        ref = gt_region_mask(gt, reg, agg)
        cons = load_mask(consensus_path(r.sample_id, "raw", reg, FINAL_QC_SUBSET, 0.5), gt.shape)
        img = read_tiff(image_path(r.sample_id)).astype(float)
        z = int(np.argmax(ref.sum((1, 2)) + cons.sum((1, 2))))
        u = ref[z] | cons[z]
        rr, cc = np.flatnonzero(u.any(1)), np.flatnonzero(u.any(0))
        sl = (slice(max(rr[0] - 10, 0), rr[-1] + 11), slice(max(cc[0] - 10, 0), cc[-1] + 11))
        im = img[z][sl]
        lo, hi = np.percentile(im, [1, 99.5])
        ax = fig.add_subplot(g0[0, j])
        ax.imshow(np.clip((im - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray", interpolation="nearest")
        outline(ax, ref[z][sl], C_REF, 1.1, "--")
        outline(ax, cons[z][sl], C_CONS, 1.3)
        clean(ax)
        scalebar(ax, im.shape[0])
        ax.set_title(f"{reg}, {r.sample_id} (median brain)", fontsize=6.8, pad=2)
        ax.text(0.03, 0.97, f"Dice {r.dice:.2f}", transform=ax.transAxes, ha="left", va="top", color="white",
                fontsize=6.2, bbox=BOX)
        letter(ax, "ABC"[j])
        srcABC.append(dict(panel="ABC"[j], region=reg, sample_id=r.sample_id, slice=z, dice=r.dice,
                           vol_ratio=r.vol_ratio))

    # ---------------- D per-region Dice with boundary distance
    ax = fig.add_subplot(outer[1])
    vol = c.groupby("region").vol_gt_mm3.median().sort_values(ascending=False)
    bd = c.groupby("region").assd_um.median()
    for i, reg in enumerate(vol.index):
        y = c[c.region == reg].set_index("sample_id").dice
        ax.scatter(np.full(len(y), i) + np.linspace(-0.13, 0.13, len(y)), y.values, s=11,
                   c=[COL[s_] for s_ in y.index], lw=0.3, edgecolor="white", zorder=3)
        ax.plot([i - 0.28, i + 0.28], [y.median()] * 2, color="0.25", lw=1.1, zorder=2)
    ax.set_xticks(range(len(vol)), [f"{r_}\n{bd[r_]:.0f} µm" for r_ in vol.index], fontsize=5.6)
    ax.set_xlim(-0.6, len(vol) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("3-D Dice with the\npublished registration")
    ax.set_xlabel("Regions, largest → smallest (below each: median mean boundary distance)", fontsize=6.2)
    ax.grid(axis="y", color="0.93", lw=0.4, zorder=0)
    spines(ax)
    letter(ax, "D", x=-0.07, y=1.06)
    ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=3.2, color=COL[b], label=b) for b in ORDER] +
              [Line2D([], [], color="0.25", lw=1.1, label="median")],
              loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=7, fontsize=5.4, frameon=False, handletextpad=0.2,
              columnspacing=0.8)

    # ---------------- E midbrain close-ups
    ge = outer[2].subgridspec(2, 3, wspace=0.04, hspace=0.3)
    srcE = []
    for k, b in enumerate(ORDER):
        gt = load_gt_atlas(b)
        snc, vta = gt_region_mask(gt, "SNc", agg), gt_region_mask(gt, "VTA", agg)
        cons = tifffile.imread(MASKS / b / "SNc__consensus.tif") != 0
        final = pd.read_csv(region_dir(b, "raw", "SNc", FINAL_QC_SUBSET) / "registrations_used.csv")
        all_dir = region_dir(b, "raw", "SNc", "all_registrations") / "individual_masks"
        runs = [load_mask(all_dir / make_output_name(r, prefix="mask"), gt.shape) for r in final.to_dict("records")]
        img = read_tiff(image_path(b)).astype(np.float32)
        z = int(np.argmax(snc.sum((1, 2))))
        u = snc[z] | vta[z] | cons[z]
        rr, cc = np.flatnonzero(u.any(1)), np.flatnonzero(u.any(0))
        sl = (slice(max(rr[0] - 14, 0), rr[-1] + 15), slice(max(cc[0] - 10, 0), cc[-1] + 11))
        if view == "coronal":
            plane = lambda v: v[z]                                              # noqa: E731
            level = z
        else:                                       # axial: horizontal section with the largest published SNc area
            level = int(np.argmax(snc.sum((0, 2))))
            u2 = (snc | vta | cons)[:, level, :]
            zz = np.flatnonzero(u2.any(1))
            sl = (slice(max(zz[0] - 10, 0), zz[-1] + 11), sl[1])              # same medio-lateral crop
            plane = lambda v: v[:, level, :]                                    # noqa: E731
        im = plane(img)[sl]
        lo, hi = np.percentile(im, [1, 99.5])
        ax = fig.add_subplot(ge[k % 2, k // 2])            # one column per marker: replicate 1 above replicate 2
        ax.imshow(np.clip((im - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray", interpolation="nearest")
        for m in runs:
            outline(ax, plane(m)[sl], C_RUN, 0.35, alpha=0.8)
        outline(ax, plane(vta)[sl], C_VTA, 0.9, ":")
        outline(ax, plane(snc)[sl], C_REF, 1.0, "--")
        outline(ax, plane(cons)[sl], C_CONS, 1.4)
        clean(ax)
        scalebar(ax, im.shape[0])
        r = st.loc[b]
        ax.set_title(b, fontsize=6.8, color=COL[b], fontweight="bold", pad=9)
        ax.text(0.5, 1.01, f"SNc Dice {r.dice_SNc:.2f}  ·  registrations agree {r.singles_pairwise_dice_median:.2f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=5.6, color="0.2")
        if k == 0:
            letter(ax, "E", x=-0.06, y=1.12)
        srcE.append(dict(sample_id=b, view=view, section_index=level, n_retained_registrations=len(runs), snc_dice=r.dice_SNc,
                         retained_registrations_pairwise_dice_median=r.singles_pairwise_dice_median,
                         consensus_volume_over_published=r.consensus_snc_vox / r.published_snc_vox))

    # ---------------- F, G, H
    gb = outer[3].subgridspec(1, 3, width_ratios=[1, 1.45, 1], wspace=0.48)
    axF = fig.add_subplot(gb[0, 0])
    for b in ORDER:
        r = st.loc[b]
        axF.scatter(r.singles_pairwise_dice_median, r.dice_SNc, s=22, color=COL[b], edgecolor="black", lw=0.4,
                    zorder=3)
        off = {"Rep2-TH": (-5, 2), "Rep2-vGlut2": (4, -4), "TH": (-3, 5), "vGlut2": (4, 2)}.get(b, (4, 2))
        axF.annotate(b, (r.singles_pairwise_dice_median, r.dice_SNc), xytext=off, textcoords="offset points",
                     fontsize=5.3, color="0.2", ha="right" if off[0] < 0 else "left")
    axF.set_xlim(0, 0.7), axF.set_ylim(-0.04, 0.7)
    axF.set_xlabel("Agreement between retained\nregistrations (SNc, pairwise Dice)")
    axF.set_ylabel("Consensus vs published SNc (Dice)")
    spines(axF)
    letter(axF, "F", x=-0.42)

    axG = fig.add_subplot(gb[0, 1])
    cats = [("rel_signal_published_vta", "VTA (published)", C_VTA, "o"),
            ("rel_signal_published_snc", "SNc (published)", "0.15", "s"),
            ("rel_signal_snc_shell", "SNc surround (100 µm)", "0.65", "D"),
            ("rel_signal_consensus_snc", "SNc (consensus)", C_CONS, "^")]
    for i, b in enumerate(ORDER):
        for j, (col, lab, colr, mk) in enumerate(cats):
            axG.scatter(i + (j - 1.5) * 0.17, st.loc[b, col], s=12, marker=mk, color=colr, edgecolor="black",
                        lw=0.3, zorder=3, label=lab if i == 0 else None)
    axG.axhline(1, color="0.6", lw=0.5, ls="--", zorder=1)
    axG.set_xticks(range(len(ORDER)), ORDER, rotation=45, ha="right", rotation_mode="anchor")
    for t, b in zip(axG.get_xticklabels(), ORDER):
        t.set_color(COL[b])
    axG.set_ylabel("Marker brightness\n(× median tissue intensity)")
    axG.set_yscale("log")
    axG.set_yticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    axG.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axG.set_ylim(0.85, 20)
    axG.legend(fontsize=5.2, frameon=False, loc="upper left", ncol=2, handletextpad=0.2, borderaxespad=0.1,
               columnspacing=0.6)
    spines(axG)
    letter(axG, "G", x=-0.3)

    axH = fig.add_subplot(gb[0, 2])
    for b in ORDER:
        r = st.loc[b]
        axH.scatter(r.rel_signal_published_vta, r.singles_pairwise_dice_median, s=22, color=COL[b],
                    edgecolor="black", lw=0.4, zorder=3)
        off = {"Rep2-GAD2": (4, -6), "Rep2-vGlut2": (-2, 7), "TH": (-4, -8), "Rep2-TH": (2, -9)}.get(b, (4, 2))
        axH.annotate(b, (r.rel_signal_published_vta, r.singles_pairwise_dice_median), xytext=off,
                     textcoords="offset points", fontsize=5.3, color="0.2", ha="right" if off[0] < 0 else "left")
    rho = stats.spearmanr(st.rel_signal_published_vta, st.singles_pairwise_dice_median).statistic
    axH.text(0.97, 0.95, f"ρ = {rho:.2f} (n = 6)", transform=axH.transAxes, ha="right", va="top", fontsize=5.6,
             color="0.2")
    axH.set_xscale("log")
    axH.set_xticks([2, 3, 4, 6, 8], ["2", "3", "4", "6", "8"])
    axH.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axH.set_xlim(2, 9), axH.set_ylim(0.1, 0.65)
    axH.set_xlabel("Midbrain marker brightness\n(published VTA, × median tissue)")
    axH.set_ylabel("Agreement between retained\nregistrations (SNc, pairwise Dice)")
    spines(axH)
    letter(axH, "H", x=-0.4)

    fig.legend(handles=[Line2D([], [], color="0.35", lw=1.0, ls="--", label="published registration (white)"),
                        Line2D([], [], color=C_VTA, lw=1.0, ls=":", label="published VTA"),
                        Line2D([], [], color=C_RUN, lw=0.8, label="retained single registrations (SNc)"),
                        Line2D([], [], color=C_CONS, lw=1.4, label="pipeline consensus")],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6, frameon=False)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"Fig4_independent_LSFM_agreement_{tag}.{ext}", dpi=600, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    pd.DataFrame(srcABC).to_csv(SRC / f"Fig4{tag}_ABC_examples.csv", index=False)
    c[["sample_id", "region", "dice", "vol_gt_mm3", "vol_pred_mm3", "vol_ratio", "hd95_um", "assd_um"]].to_csv(
        SRC / f"Fig4{tag}_D_regions.csv", index=False)
    pd.DataFrame(srcE).merge(st.reset_index().rename(columns={"sample": "sample_id"}), on="sample_id").to_csv(
        SRC / f"Fig4{tag}_E-H_per_brain.csv", index=False)
    print("examples:", [a, b_, cc_], "rho", round(rho, 3))
    print(bd.round(0).to_dict())


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "coronal")
