# -*- coding: utf-8 -*-
"""
15_figS6_midbrain.py

Revised Supplementary Fig. S6: the midbrain in the independent LSFM dataset (raw input, final QC set, threshold 0.5).
  A  Coronal close-up of the midbrain in all six brains (section with the largest published SNc area): published SNc
     (white dashed) and VTA (orange dotted), every retained single registration's SNc (thin blue), our consensus SNc
     (purple). Per brain: consensus SNc Dice with the published SNc, and agreement between our retained registrations
     (median pairwise Dice of their SNc masks).
  B  Agreement between our retained registrations vs agreement of the consensus with the published SNc, per brain.
  C  Marker brightness (mean intensity / median tissue intensity) in the published VTA, the published SNc, a 100 um
     shell around the published SNc, and our consensus SNc.
  D  Per brain: midbrain marker brightness (published VTA, x median tissue) against agreement between retained
     registrations (SNc, median pairwise Dice); Spearman rho.
Inputs: results/midbrain_inspection/midbrain_inspection.csv (14_midbrain_inspection.py), results/fig8_region_masks/
(13_export_fig8_masks.py), results/validation_gt/*.csv. Read-only on all inputs. The earlier Fig. S6 files are kept.
Output: REVISION_OUTPUTS_FOCUSED/figures/FigS6_independent_LSFM_midbrain.{pdf,svg,png} + source_data/FigS6_midbrain_*.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib
import matplotlib.ticker
import matplotlib.colors
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from gt_utils import (ROOT, FINAL_QC_SUBSET, SAMPLES, load_label_mapping, load_gt_atlas, gt_region_mask, image_path,
                      region_dir, load_mask, read_tiff)  # noqa: E402

_s = importlib.util.spec_from_file_location("s12", str(Path(__file__).with_name("12_snc_ssim_figure.py")))
s12 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(s12)
crop_bbox, minmax_norm, make_output_name = s12.crop_bbox, s12.minmax_norm, s12.make_output_name

FIG, SRC = s12.FIG, s12.SRC
INSP = ROOT / "results" / "midbrain_inspection" / "midbrain_inspection.csv"
MASKS = ROOT / "results" / "fig8_region_masks"
ORDER = ["GAD2", "Rep2-GAD2", "TH", "Rep2-TH", "vGlut2", "Rep2-vGlut2"]          # by marker
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
COL = dict(zip(sorted(SAMPLES), OKABE))                                          # same colours as Fig. 6A-C / S5c
C_REF, C_VTA, C_RUN, C_CONS = "white", "#E69F00", "#56B4E9", "#7B3294"
SMALL = ["SNc", "STN", "VTA", "DG-sg"]
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


def letter(ax, s, x=-0.18, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def main():
    _, agg = load_label_mapping()
    st = pd.read_csv(INSP).set_index("sample").loc[ORDER]
    fig = plt.figure(figsize=(7.2, 4.9))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.3], hspace=0.28, top=0.915, bottom=0.07)

    # ---------------- A: midbrain close-ups
    ga = outer[0].subgridspec(2, 3, wspace=0.04, hspace=0.22)
    srcA = []
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
        im = img[z][sl]
        lo, hi = np.percentile(im, [1, 99.5])
        ax = fig.add_subplot(ga[k // 3, k % 3])
        ax.imshow(np.clip((im - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray", interpolation="nearest")
        for r in runs:
            outline(ax, r[z][sl], C_RUN, 0.35, alpha=0.8)
        outline(ax, vta[z][sl], C_VTA, 0.9, ":")
        outline(ax, snc[z][sl], C_REF, 1.0, "--")
        outline(ax, cons[z][sl], C_CONS, 1.4)
        clean(ax)
        ax.plot([4, 24], [im.shape[0] - 5] * 2, color="white", lw=1.4)                # 20 px x 25 um = 500 um
        r = st.loc[b]
        ax.set_title(b, fontsize=6.8, color=COL[b], fontweight="bold", pad=9)
        ax.text(0.5, 1.01, f"SNc Dice {r.dice_SNc:.2f}  ·  registrations agree {r.singles_pairwise_dice_median:.2f}",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=5.6, color="0.2")
        if k == 0:
            letter(ax, "A", x=-0.06)
        srcA.append(dict(panel="A", sample_id=b, coronal_slice_z=z, n_retained_registrations=len(runs),
                         snc_dice=r.dice_SNc, retained_registrations_pairwise_dice_median=r.singles_pairwise_dice_median,
                         consensus_volume_over_published=r.consensus_snc_vox / r.published_snc_vox))
    fig.legend(handles=[Line2D([], [], color="0.35", lw=1.0, ls="--", label="published SNc (white)"),
                        Line2D([], [], color=C_VTA, lw=1.0, ls=":", label="published VTA"),
                        Line2D([], [], color=C_RUN, lw=0.8, label="retained single registrations (SNc)"),
                        Line2D([], [], color=C_CONS, lw=1.4, label="pipeline consensus SNc")],
               loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6, frameon=False)

    # ---------------- B, C, D
    gb = outer[1].subgridspec(1, 3, width_ratios=[1, 1.45, 1], wspace=0.48)
    axB = fig.add_subplot(gb[0, 0])
    for b in ORDER:
        r = st.loc[b]
        axB.scatter(r.singles_pairwise_dice_median, r.dice_SNc, s=22, color=COL[b], edgecolor="black", lw=0.4, zorder=3)
        off = {"Rep2-TH": (-5, 2), "Rep2-vGlut2": (4, -4), "TH": (-3, 5), "vGlut2": (4, 2)}.get(b, (4, 2))
        axB.annotate(b, (r.singles_pairwise_dice_median, r.dice_SNc), xytext=off, textcoords="offset points",
                     fontsize=5.3, color="0.2", ha="right" if off[0] < 0 else "left")
    axB.set_xlim(0, 0.7), axB.set_ylim(-0.04, 0.7)
    axB.set_xlabel("Agreement between retained\nregistrations (SNc, pairwise Dice)")
    axB.set_ylabel("Consensus vs published SNc (Dice)")
    spines(axB)
    letter(axB, "B", x=-0.42)

    axC = fig.add_subplot(gb[0, 1])
    cats = [("rel_signal_published_vta", "VTA (published)", C_VTA, "o"),
            ("rel_signal_published_snc", "SNc (published)", "0.15", "s"),
            ("rel_signal_snc_shell", "SNc surround (100 µm)", "0.65", "D"),
            ("rel_signal_consensus_snc", "SNc (consensus)", C_CONS, "^")]
    for i, b in enumerate(ORDER):
        for j, (c, lab, colr, mk) in enumerate(cats):
            axC.scatter(i + (j - 1.5) * 0.17, st.loc[b, c], s=12, marker=mk, color=colr, edgecolor="black", lw=0.3,
                        zorder=3, label=lab if i == 0 else None)
    axC.axhline(1, color="0.6", lw=0.5, ls="--", zorder=1)
    axC.set_xticks(range(len(ORDER)), ORDER, rotation=45, ha="right", rotation_mode="anchor")
    for t, b in zip(axC.get_xticklabels(), ORDER):
        t.set_color(COL[b])
    axC.set_ylabel("Marker brightness\n(× median tissue intensity)")
    axC.set_yscale("log")
    axC.set_yticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    axC.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axC.set_ylim(0.85, 20)
    axC.legend(fontsize=5.2, frameon=False, loc="upper left", ncol=2, handletextpad=0.2, borderaxespad=0.1,
               columnspacing=0.6)
    spines(axC)
    letter(axC, "C", x=-0.3)

    # ---------------- D: midbrain brightness vs agreement between retained registrations
    axF = fig.add_subplot(gb[0, 2])
    for b in ORDER:
        r = st.loc[b]
        axF.scatter(r.rel_signal_published_vta, r.singles_pairwise_dice_median, s=22, color=COL[b], edgecolor="black",
                    lw=0.4, zorder=3)
        off = {"Rep2-GAD2": (4, -6), "Rep2-vGlut2": (-2, 7), "TH": (-4, -8), "Rep2-TH": (2, -9)}.get(b, (4, 2))
        axF.annotate(b, (r.rel_signal_published_vta, r.singles_pairwise_dice_median), xytext=off,
                     textcoords="offset points", fontsize=5.3, color="0.2", ha="right" if off[0] < 0 else "left")
    rho_f = stats.spearmanr(st.rel_signal_published_vta, st.singles_pairwise_dice_median).statistic
    axF.text(0.97, 0.95, f"ρ = {rho_f:.2f} (n = 6)", transform=axF.transAxes, ha="right", va="top", fontsize=5.6,
             color="0.2")
    axF.set_xscale("log")
    axF.set_xticks([2, 3, 4, 6, 8], ["2", "3", "4", "6", "8"])
    axF.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axF.set_xlim(2, 9)
    axF.set_ylim(0.1, 0.65)
    axF.set_xlabel("Midbrain marker brightness\n(published VTA, × median tissue)")
    axF.set_ylabel("Agreement between retained\nregistrations (SNc, pairwise Dice)")
    spines(axF)
    letter(axF, "D", x=-0.4)

    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"FigS6_independent_LSFM_midbrain.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.DataFrame(srcA).merge(st.reset_index().rename(columns={"sample": "sample_id"}), on="sample_id").to_csv(
        SRC / "FigS6_midbrain_A-C_per_brain.csv", index=False)
    st.reset_index()[["sample", "rel_signal_published_vta", "singles_pairwise_dice_median"]].to_csv(
        SRC / "FigS6_midbrain_D_per_brain.csv", index=False)


if __name__ == "__main__":
    main()
