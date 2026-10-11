# -*- coding: utf-8 -*-
"""
19_figS_single_registration_range.py

Supplementary figure (independent LSFM, raw input): how widely the retained single registrations agree with the
published registration in each region, and where the consensus falls within that range (same layout as the aMAP
consensus-vs-single figure).
  A  per region (largest -> smallest) and brain: range (thin line) and IQR (thick line) of the 3-D Dice of every
     single registration retained by global QC + SSIM, and the consensus (purple dot, threshold 0.5).
  B  the same for the mean boundary distance (um), which is not inflated for small regions the way Dice is; axis
     capped at 250 um, longer ranges marked with an arrowhead and their maximum.
Inputs: results/validation_gt/individual_vs_gt.csv, consensus_vs_gt.csv. Read-only.
Output: REVISION_OUTPUTS_FOCUSED/figures/FigS_independent_LSFM_single_registration_range.{pdf,svg,png}
        + source_data/FigS_single_registration_range_per_case.csv, _per_region.csv
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from gt_utils import ROOT, FINAL_QC_SUBSET, SAMPLES  # noqa: E402

PRA = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis"))
FIG, SRC = PRA / "REVISION_OUTPUTS_FOCUSED" / "figures", PRA / "REVISION_OUTPUTS_FOCUSED" / "source_data"
ORDER = ["GAD2", "Rep2-GAD2", "TH", "Rep2-TH", "vGlut2", "Rep2-vGlut2"]
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
COL = dict(zip(sorted(SAMPLES), OKABE))
C_RUN, C_CONS = "#56B4E9", "#7B3294"
CAP = 250                                   # um; ranges reaching beyond are clipped and marked (value printed)
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "axes.linewidth": 0.6, "pdf.fonttype": 42, "svg.fonttype": "none"})


def main():
    V = ROOT / "results" / "validation_gt"
    ind = pd.read_csv(V / "individual_vs_gt.csv")
    ind = ind[(ind.input_type == "raw") & ind.in_final_qc]
    con = pd.read_csv(V / "consensus_vs_gt.csv")
    con = con[(con.input_type == "raw") & (con.subset == FINAL_QC_SUBSET) & (con.threshold == 0.5)]
    con = con.set_index(["sample_id", "region"])
    vol = ind.groupby("region").vol_gt_mm3.median().sort_values(ascending=False)
    rows = []
    for (s, r), g in ind.groupby(["sample_id", "region"]):
        cd, ca = con.loc[(s, r), "dice"], con.loc[(s, r), "assd_um"]
        rows.append(dict(sample_id=s, region=r, volume_mm3=g.vol_gt_mm3.iloc[0], n_retained=len(g),
                         dice_min=g.dice.min(), dice_q25=g.dice.quantile(.25), dice_median=g.dice.median(),
                         dice_q75=g.dice.quantile(.75), dice_max=g.dice.max(), consensus_dice=cd,
                         consensus_percentile_among_singles=100 * (g.dice < cd).mean(),
                         consensus_at_or_above_p10=bool(cd >= g.dice.quantile(.1) - 0.02),
                         bd_min=g.assd_um.min(), bd_q25=g.assd_um.quantile(.25), bd_q75=g.assd_um.quantile(.75),
                         bd_max=g.assd_um.max(), consensus_bd=ca))
    d = pd.DataFrame(rows)
    d["dice_range"] = d.dice_max - d.dice_min
    d["bd_range"] = d.bd_max - d.bd_min

    fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.9), sharex=True, gridspec_kw=dict(hspace=0.12))
    offs = np.linspace(-0.3, 0.3, len(ORDER))
    for ax, (lo, q1, q3, hi, cv, ylab) in zip(
            axs, (("dice_min", "dice_q25", "dice_q75", "dice_max", "consensus_dice",
                   "3-D Dice with the published registration"),
                  ("bd_min", "bd_q25", "bd_q75", "bd_max", "consensus_bd",
                   "Mean boundary distance\nfrom the published registration (µm)"))):
        for i, reg in enumerate(vol.index):
            if i % 2:
                ax.axvspan(i - 0.5, i + 0.5, color="0.96", zorder=0, lw=0)
            for o, b in zip(offs, ORDER):
                r = d[(d.region == reg) & (d.sample_id == b)].iloc[0]
                x = i + o
                top = r[hi] if ax is axs[0] else min(r[hi], CAP)
                ax.plot([x, x], [r[lo], top], color=C_RUN, lw=0.6, zorder=2)
                if ax is axs[1] and r[hi] > CAP:
                    ax.plot(x, CAP, marker="^", ms=2.6, color=C_RUN, zorder=3, clip_on=False)
                    ax.text(x + 0.05, CAP, f"{r[hi]:.0f}", fontsize=4.6, color="0.35", va="bottom", ha="left")
                ax.plot([x, x], [r[q1], r[q3]], color=C_RUN, lw=2.2, solid_capstyle="butt", zorder=2)
                ax.scatter(x, r[cv], s=9, color=C_CONS, edgecolor="white", lw=0.3, zorder=3)
        ax.set_ylabel(ylab)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="0.92", lw=0.4, zorder=0)
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, CAP)
    axs[0].text(-0.07, 1.02, "A", transform=axs[0].transAxes, fontsize=9, fontweight="bold")
    axs[1].text(-0.07, 1.02, "B", transform=axs[1].transAxes, fontsize=9, fontweight="bold")
    per_reg = d.groupby("region").agg(volume_mm3=("volume_mm3", "median"), dice_range=("dice_range", "median"),
                                      bd_range=("bd_range", "median"),
                                      consensus_percentile=("consensus_percentile_among_singles", "median")
                                      ).reindex(vol.index)
    for i, reg in enumerate(vol.index):
        axs[0].text(i, 1.01, f"{per_reg.loc[reg, 'dice_range']:.2f}", ha="center", va="bottom", fontsize=5.2,
                    color="0.35", transform=axs[0].get_xaxis_transform())
    axs[0].text(len(vol) - 0.4, 1.07, "median range of single registrations (Dice)", ha="right", va="bottom",
                fontsize=5.4, color="0.35", transform=axs[0].get_xaxis_transform())
    axs[1].set_xticks(range(len(vol)), [f"{r_}\n{vol[r_]:.2f} mm³" if vol[r_] < 1 else f"{r_}\n{vol[r_]:.1f} mm³"
                                        for r_ in vol.index], fontsize=5.6)
    axs[1].set_xlim(-0.6, len(vol) - 0.4)
    axs[1].set_xlabel("Regions, largest → smallest (published region volume); within each region, brains in the order of the legend")
    fig.legend(handles=[Line2D([], [], color=C_RUN, lw=0.6, label="range of retained single registrations"),
                        Line2D([], [], color=C_RUN, lw=2.2, label="interquartile range"),
                        Line2D([], [], ls="", marker="o", ms=3.5, color=C_CONS, label="pipeline consensus")],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=6, frameon=False)
    fig.text(0.5, 0.955, "Brains within each region (left → right): " + ", ".join(ORDER), ha="center", fontsize=5.6,
             color="0.35")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"FigS_independent_LSFM_single_registration_range.{ext}", dpi=600, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    d.to_csv(SRC / "FigS_single_registration_range_per_case.csv", index=False)
    per_reg.to_csv(SRC / "FigS_single_registration_range_per_region.csv")
    print(per_reg.round(3).to_string())
    print("rho Dice range vs log volume", round(stats.spearmanr(d.dice_range, np.log(d.volume_mm3)).statistic, 2),
          "| consensus >= p10 (within 0.02):", int(d.consensus_at_or_above_p10.sum()), "/", len(d),
          "| above median single:", int((d.consensus_percentile_among_singles > 50).sum()))


if __name__ == "__main__":
    main()
