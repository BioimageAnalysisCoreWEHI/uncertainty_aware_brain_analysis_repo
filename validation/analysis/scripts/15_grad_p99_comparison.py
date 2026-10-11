"""grad_p99 vs grad_p95 as the global-QC deformation metric (IoU x grad), all three datasets.

Pre-specified comparison (written before looking at the p99 results):
  1. Agreement: Spearman rho between grad_p95 and grad_p99, among registrations passing the chosen IoU threshold.
  2. Same procedure for both metrics: raw-anchored Tukey default (IoU fence, then grad fence on IoU-passing runs).
  3. Ground-truth discrimination, among registrations passing the chosen IoU threshold (grad is applied after IoU):
     - aMAP: registration quality = median Dice vs expert STAPLE (aMAP protocol, dice_paper) over the nine structures
       (DG-sg excluded); poor = median Dice < config amap.poor_dice (0.7).
     - paper LSFM: registration quality = |SNpc count / brain reference - 1|; the reference is the median count of that
       brain's registrations passing the chosen IoU (independent of either grad metric); poor = deviation > 20 %.
     AUC of each metric for flagging poor registrations (higher grad = worse), with a brain-resampled 95 % CI of the
     difference AUC(p99) - AUC(p95). "Better" = a higher AUC whose CI excludes 0.
  4. Matched retention: the grad_p99 threshold that keeps as many registrations (with the chosen IoU) as the chosen
     grad_p95 threshold; Jaccard of the two pass sets, and poor registrations admitted / quality of retained ones.
  Validation LSFM has no per-registration ground truth, so only 1, 2 and 4 (Jaccard) are reported for it.

Inputs (read-only): dataset_qc_comparison/*.csv, results/tables/amap_runs.csv, cache/lsfm/runs.csv.
Outputs: results/tables/qcp99_*.csv, figures/supplementary/FigS_qc_grad_p99.{pdf,png}, source_data/FigS_qc_grad_p99_*.csv.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import CFG, FIG_SUPP, SRC, STRUCTS_MAIN, TABLES, VA, amap_brain, log, save_table  # noqa: E402

_s = importlib.util.spec_from_file_location("qcproc", str(VA / "scripts" / "12_qc_threshold_procedure.py"))
qc = importlib.util.module_from_spec(_s)
_s.loader.exec_module(qc)

GRADS = ["grad_p95", "grad_p99"]
POOR_DICE = float(CFG["amap"]["poor_dice"])
POOR_COUNT_DEV = 0.20
NBOOT = int(CFG["bootstrap_n"])
RNG = np.random.default_rng(int(CFG["seed"]))


def tukey(d, g):
    q1, q3 = np.percentile(d.iou, [25, 75])
    t_iou = q1 - 1.5 * (q3 - q1)
    x = d[d.iou >= t_iou][g]
    g1, g3 = np.percentile(x, [25, 75])
    return round(float(t_iou), 4), round(float(g3 + 1.5 * (g3 - g1)), 4)


def auc(score, poor):
    """P(score of a poor registration > score of a good one); ties count 1/2."""
    score, poor = np.asarray(score, float), np.asarray(poor, bool)
    n1, n0 = poor.sum(), (~poor).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(score)
    return (r[poor].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def matched_threshold(x_ref_pass: int, x: np.ndarray) -> float:
    """grad threshold keeping exactly as many registrations as the reference (strict '<' as in the pipeline)."""
    s = np.sort(x)
    if x_ref_pass <= 0:
        return float(s[0])
    if x_ref_pass >= len(s):
        return float(s[-1]) + 1e-9
    return float((s[x_ref_pass - 1] + s[x_ref_pass]) / 2)


# ------------------------------------------------------------------------------------------------------------
def quality_tables(allqc):
    """Per-registration ground-truth quality for aMAP and paper LSFM, with both grad metrics."""
    r = pd.read_csv(TABLES / "amap_runs.csv")
    r = r[r.structure.isin(STRUCTS_MAIN)]
    qa = (r.groupby(["brain", "variant", "bending_energy", "grid"]).dice_paper.median()
          .rename("quality").reset_index())
    a = allqc[allqc.dataset == "aMAP"].rename(columns={"sample": "brain"})
    qa = a.merge(qa, on=["brain", "variant", "bending_energy", "grid"], how="left", validate="1:1")
    qa["poor"] = qa.quality < POOR_DICE
    qa["unit"] = qa.brain

    l = pd.read_csv(VA / "cache" / "lsfm" / "runs.csv")
    l["sample"] = l["sample"].astype(str)
    l = l.rename(columns={"status": "variant", "bending": "bending_energy"})
    ci = qc.CHOSEN["paper LSFM"][0]
    ref = l[l.iou > ci].groupby(["sample", "variant"]).snpc_CellCount.median().rename("ref")
    l = l.join(ref, on=["sample", "variant"])
    l["quality"] = (l.snpc_CellCount / l.ref - 1).abs()
    l["poor"] = l.quality > POOR_COUNT_DEV
    l["unit"] = l["sample"]
    l["dataset"] = "paper LSFM"
    return {"aMAP": qa, "paper LSFM": l[["dataset", "variant", "sample", "unit", "bending_energy", "grid", "iou",
                                         "grad_p95", "grad_p99", "snpc_CellCount", "ref", "quality", "poor"]]}


def discrimination(qt):
    rows, boots = [], []
    for ds, d0 in qt.items():
        ci = qc.CHOSEN[ds][0]
        for v in qc.VARIANTS:
            d = d0[(d0.variant == v) & (d0.iou > ci)].dropna(subset=["quality"])
            a95, a99 = auc(d.grad_p95, d.poor), auc(d.grad_p99, d.poor)
            units = d.unit.unique()
            diffs = []
            for _ in range(NBOOT):
                b = pd.concat([d[d.unit == u] for u in RNG.choice(units, len(units), replace=True)])
                diffs.append(auc(b.grad_p99, b.poor) - auc(b.grad_p95, b.poor))
            diffs = np.asarray(diffs, float)
            lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
            rows.append(dict(dataset=ds, variant=v, iou_thr=ci, n=len(d), n_poor=int(d.poor.sum()),
                             auc_p95=a95, auc_p99=a99, auc_diff=a99 - a95, diff_ci_lo=lo, diff_ci_hi=hi,
                             rho_p95_quality=stats.spearmanr(d.grad_p95, d.quality).statistic,
                             rho_p99_quality=stats.spearmanr(d.grad_p99, d.quality).statistic))
    return pd.DataFrame(rows)


def agreement_and_matching(allqc, qt):
    rows = []
    for ds in qc.DATASETS:
        ci, cg = qc.CHOSEN[ds]
        raw = allqc[(allqc.dataset == ds) & (allqc.variant == "raw")]
        d95, d99 = tukey(raw, "grad_p95"), tukey(raw, "grad_p99")
        for v in qc.VARIANTS:
            d = allqc[(allqc.dataset == ds) & (allqc.variant == v)]
            di = d[d.iou > ci]
            p95 = (d.iou > ci) & (d.grad_p95 < cg)
            t99 = matched_threshold(int(p95.sum()), di.grad_p99.to_numpy())
            p99 = (d.iou > ci) & (d.grad_p99 < t99)
            row = dict(dataset=ds, variant=v, n=len(d), n_iou_pass=len(di),
                       rho_p95_p99=stats.spearmanr(di.grad_p95, di.grad_p99).statistic,
                       tukey_iou=d95[0], tukey_grad_p95=d95[1], tukey_grad_p99=d99[1],
                       pass_tukey_p95=int(((d.iou > d95[0]) & (d.grad_p95 < d95[1])).sum()),
                       pass_tukey_p99=int(((d.iou > d99[0]) & (d.grad_p99 < d99[1])).sum()),
                       chosen_grad_p95=cg, matched_grad_p99=round(t99, 4),
                       matched_p99_over_p95=t99 / cg, pass_chosen_p95=int(p95.sum()), pass_matched_p99=int(p99.sum()),
                       jaccard_pass_sets=float((p95 & p99).sum() / max((p95 | p99).sum(), 1)),
                       only_p95=int((p95 & ~p99).sum()), only_p99=int((p99 & ~p95).sum()))
            if ds in qt:
                g = qt[ds][qt[ds].variant == v].rename(columns={"brain": "sample"})
                key = ["sample", "bending_energy", "grid"]
                g = g.astype({"sample": str, "bending_energy": float, "grid": float}).set_index(key)
                dd = d.astype({"sample": str, "bending_energy": float, "grid": float}).set_index(key)
                m95 = dd[(dd.iou > ci) & (dd.grad_p95 < cg)].index
                m99 = dd[(dd.iou > ci) & (dd.grad_p99 < t99)].index
                row.update(poor_admitted_p95=int(g.loc[g.index.intersection(m95)].poor.sum()),
                           poor_admitted_p99=int(g.loc[g.index.intersection(m99)].poor.sum()),
                           median_quality_p95=float(g.loc[g.index.intersection(m95)].quality.median()),
                           median_quality_p99=float(g.loc[g.index.intersection(m99)].quality.median()),
                           quality_measure="median Dice (higher better)" if ds == "aMAP"
                           else "|count deviation| (lower better)")
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------
def figure(allqc, agr, disc):
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 9.4))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 0.9, 0.8], hspace=0.75, wspace=0.45)
    src_a, src_b = [], []
    for j, ds in enumerate(qc.DATASETS):
        ci, cg = qc.CHOSEN[ds]
        for i, v in enumerate(qc.VARIANTS):
            ax = fig.add_subplot(gs[i, j])
            d = allqc[(allqc.dataset == ds) & (allqc.variant == v)]
            a = agr[(agr.dataset == ds) & (agr.variant == v)].iloc[0]
            t99 = a.matched_grad_p99
            for k, (smp, g) in enumerate(d.groupby("sample")):
                ok = (g.iou > ci) & (g.grad_p99 < t99)
                c = qc.OI[k % 6]
                ax.scatter(g.iou[ok], g.grad_p99[ok], s=5, color=c, lw=0, label=smp)
                ax.scatter(g.iou[~ok], g.grad_p99[~ok], s=5, facecolor="none", edgecolor=c, lw=0.5)
            ax.axvline(ci, color="k", lw=0.9)
            ax.axhline(t99, color="k", lw=0.9)
            ax.axvline(a.tukey_iou, color="0.5", lw=0.9, ls="--")
            ax.axhline(a.tukey_grad_p99, color="0.5", lw=0.9, ls="--")
            ax.set_title(f"{ds} ({v})\nmatched p99 < {t99:.3f}; default {a.tukey_iou:.3f} / {a.tukey_grad_p99:.3f}",
                         fontsize=6.5)
            ax.set_xlabel("Tissue IoU")
            if j == 0:
                ax.set_ylabel("grad_p99")
            if i == 0:
                ax.legend(fontsize=5, frameon=False, markerscale=1.5, handletextpad=0.2, loc="upper left")
            src_a.append(d.assign(matched_grad_p99=t99, tukey_grad_p99=a.tukey_grad_p99)[
                ["dataset", "variant", "sample", "bending_energy", "grid", "iou", "grad_p95", "grad_p99",
                 "matched_grad_p99", "tukey_grad_p99"]])
    for j, ds in enumerate(qc.DATASETS):
        ax = fig.add_subplot(gs[2, j])
        ci, cg = qc.CHOSEN[ds]
        for v, c in zip(qc.VARIANTS, ("#0072B2", "#56B4E9")):
            d = allqc[(allqc.dataset == ds) & (allqc.variant == v) & (allqc.iou > ci)]
            ax.scatter(d.grad_p95, d.grad_p99, s=4, color=c, lw=0, label=v)
            src_b.append(d[["dataset", "variant", "sample", "grad_p95", "grad_p99"]])
        ax.axvline(cg, color="k", lw=0.8)
        rho = agr[agr.dataset == ds].set_index("variant").rho_p95_p99
        ax.set_title(f"{ds}, IoU > {ci}\nSpearman ρ raw {rho['raw']:.2f}, pre {rho['preprocessed']:.2f}", fontsize=6.5)
        ax.set_xlabel("grad_p95")
        if j == 0:
            ax.set_ylabel("grad_p99")
            ax.legend(fontsize=5, frameon=False, markerscale=1.5)
    ax = fig.add_subplot(gs[3, :2])
    lab = [f"{r.dataset}\n{r.variant} (poor {r.n_poor}/{r.n})" for r in disc.itertuples()]
    x = np.arange(len(disc))
    ax.bar(x - 0.18, disc.auc_p95, 0.36, color="#0072B2", label="grad_p95")
    ax.bar(x + 0.18, disc.auc_p99, 0.36, color="#D55E00", label="grad_p99")
    ax.axhline(0.5, color="0.6", lw=0.7, ls=":")
    for xi, r in zip(x, disc.itertuples()):
        ax.text(xi, max(r.auc_p95, r.auc_p99) + 0.03, f"Δ {r.auc_diff:+.2f}\n[{r.diff_ci_lo:+.2f}, {r.diff_ci_hi:+.2f}]",
                ha="center", fontsize=5.5)
    ax.set_xticks(x, lab, fontsize=5.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("AUC, flagging poor registrations\n(among IoU-passing)")
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax2 = fig.add_subplot(gs[3, 2])
    lab2 = [f"{r.dataset}\n{r.variant}" for r in agr.itertuples()]
    ax2.barh(np.arange(len(agr)), agr.jaccard_pass_sets, color="0.55")
    ax2.set_yticks(np.arange(len(agr)), lab2, fontsize=5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Jaccard of pass sets\n(p99 at matched retention)")
    for k, t in zip("abcd", (fig.axes[0], fig.axes[6], ax, ax2)):
        t.text(-0.3 if k != "c" else -0.12, 1.25, k, transform=t.transAxes, fontsize=9, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(FIG_SUPP / f"FigS_qc_grad_p99.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    pd.concat(src_a).to_csv(SRC / "FigS_qc_grad_p99_a_scatter.csv", index=False)
    pd.concat(src_b).to_csv(SRC / "FigS_qc_grad_p99_b_p95_vs_p99.csv", index=False)
    disc.to_csv(SRC / "FigS_qc_grad_p99_c_auc.csv", index=False)
    agr.to_csv(SRC / "FigS_qc_grad_p99_d_jaccard.csv", index=False)


def main():
    allqc = qc.load_all()
    qt = quality_tables(allqc)
    for ds, t in qt.items():
        miss = int(t.quality.isna().sum())
        log(f"{ds}: {len(t)} registrations with quality, {miss} missing")
    agr = agreement_and_matching(allqc, qt)
    disc = discrimination(qt)
    save_table(agr, "qcp99_agreement_matching")
    save_table(disc, "qcp99_discrimination")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    log(agr.round(4).to_string(index=False))
    log(disc.round(3).to_string(index=False))
    figure(allqc, agr, disc)
    log("wrote figures/supplementary/FigS_qc_grad_p99.{pdf,png}")


if __name__ == "__main__":
    main()
