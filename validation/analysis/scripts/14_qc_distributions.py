"""Distributions of the global-QC metrics (tissue IoU, grad_p95) per dataset and input.

Histograms with a Gaussian KDE, the chosen thresholds (solid black) and the raw-anchored Tukey default
(dashed grey), plus a shape table (skewness, excess kurtosis, bimodality coefficient, Shapiro-Wilk, KDE peaks).
The bimodality coefficient is inflated by strong skew, so shape is read from the histograms, not from it alone.

Inputs (read-only): post_review_analysis/dataset_qc_comparison/*.csv (via 12_qc_threshold_procedure.load_all).
Outputs: figures/supplementary/FigS_qc_distributions.{pdf,png}, source_data/FigS_qc_distributions.csv,
results/tables/qcproc_distribution_shape.csv.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIG_SUPP, SRC, VA, log, save_table  # noqa: E402

_s = importlib.util.spec_from_file_location("qcproc", str(VA / "scripts" / "12_qc_threshold_procedure.py"))
qc = importlib.util.module_from_spec(_s)
_s.loader.exec_module(qc)

METRICS = [("iou", "Tissue IoU"), ("grad_p95", "grad_p95")]
PEAK_PROMINENCE = 0.05  # KDE peaks reported if prominence >= 5 % of the maximum density


def kde_peaks(x):
    k = stats.gaussian_kde(x)
    xs = np.linspace(x.min(), x.max(), 400)
    y = k(xs)
    pk, _ = find_peaks(y, prominence=PEAK_PROMINENCE * y.max())
    return xs, y, xs[pk]


def shape_table(allqc):
    rows = []
    for ds in qc.DATASETS:
        for v in qc.VARIANTS:
            g = allqc[(allqc.dataset == ds) & (allqc.variant == v)]
            for col, _ in METRICS:
                x = g[col].dropna().to_numpy()
                n = len(x)
                sk, ku = stats.skew(x), stats.kurtosis(x)
                bc = (sk ** 2 + 1) / (ku + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
                rows.append(dict(dataset=ds, variant=v, metric=col, n=n, median=np.median(x),
                                 q1=np.quantile(x, 0.25), q3=np.quantile(x, 0.75), skew=sk, excess_kurtosis=ku,
                                 bimodality_coef=bc, shapiro_p=stats.shapiro(x).pvalue,
                                 kde_peaks=";".join(f"{p:.4f}" for p in kde_peaks(x)[2])))
    return pd.DataFrame(rows)


def figure(allqc):
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    raw = allqc[allqc.variant == "raw"]
    default = {ds: qc.tukey(raw[raw.dataset == ds]) for ds in qc.DATASETS}
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 8.2))
    src = []
    for j, ds in enumerate(qc.DATASETS):
        for m, (col, lab) in enumerate(METRICS):
            for vi, v in enumerate(qc.VARIANTS):
                ax = axes[2 * m + vi, j]
                x = allqc[(allqc.dataset == ds) & (allqc.variant == v)][col].dropna().to_numpy()
                ax.hist(x, bins=40, density=True, color="0.75", edgecolor="white", linewidth=0.3)
                xs, y, _ = kde_peaks(x)
                ax.plot(xs, y, color="0.15", lw=0.9)
                ax.axvline(qc.CHOSEN[ds][m], color="k", lw=1.0)
                ax.axvline(default[ds][m], color="0.45", lw=1.0, ls="--")
                ax.set_title(f"{ds} ({v})", fontsize=7)
                ax.set_xlabel(lab)
                if j == 0:
                    ax.set_ylabel("Density")
                src += [dict(dataset=ds, variant=v, metric=col, value=val) for val in x]
    for k, ax in enumerate(axes[:, 0]):
        ax.text(-0.32, 1.12, "ab"[k // 2] if k % 2 == 0 else "", transform=ax.transAxes,
                fontsize=9, fontweight="bold")
    fig.legend(handles=[plt.Line2D([], [], color="k", lw=1.0, label="chosen threshold"),
                        plt.Line2D([], [], color="0.45", lw=1.0, ls="--",
                                   label="Tukey default (set on raw, applied to both)"),
                        plt.Line2D([], [], color="0.15", lw=0.9, label="kernel density")],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=1.2, w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_SUPP / f"FigS_qc_distributions.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(src).to_csv(SRC / "FigS_qc_distributions.csv", index=False)


def main():
    allqc = qc.load_all()
    shp = shape_table(allqc)
    save_table(shp, "qcproc_distribution_shape")
    log(shp.round(4).to_string(index=False))
    figure(allqc)
    log("wrote figures/supplementary/FigS_qc_distributions.{pdf,png}")


if __name__ == "__main__":
    main()
