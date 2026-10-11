# -*- coding: utf-8 -*-
"""
08_plot_validation.py

Figures + summary tables for the GT validation. Run after 07a-07d.

Reads results/validation_gt/{consensus_vs_gt, individual_vs_gt,
calibration_bins}.csv and writes to results/validation_gt/ (tables) and
results/validation_gt/figures/ (PNG + PDF):

  heatmap_{dice,hd95}_final_qc_t0p5    region x brain at the headline setting
  threshold_curves_{dice,hd95}_<input> metric vs consensus threshold, per subset
  subset_comparison_t0p5               paired all / global QC / final QC
  consensus_vs_individual_<input>      consensus vs the 64 single registrations
  qc_vs_gt                             do QC metrics/gates track GT agreement?
  reliability[_by_region]              probability-map calibration
  dice_vs_gt_volume                    Dice penalty for small structures
  overlays_<input>                     GT vs consensus contours, worst/median/best

Tables: summary_table.csv, subset_comparison_stats.csv,
consensus_vs_individual.csv, qc_spearman.csv, qc_gate_effect.csv
"""

import importlib.util

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import ndimage, stats

from gt_utils import (
    ROOT, OUT_DIR, SAMPLES, INPUT_TYPES, SUBSETS, FINAL_QC_SUBSET, HEADLINE_THRESHOLD,
    region_table, load_label_mapping, load_gt_atlas, gt_region_mask,
    consensus_path, image_path, load_mask, safe_value,
)
from pipeline_common import make_sample_style

_spec = importlib.util.spec_from_file_location("qc_calibration", ROOT / "scripts" / "04b_plot_qc_calibration.py")
_qc_calibration = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qc_calibration)
show_slice_with_boundaries = _qc_calibration.show_slice_with_boundaries
read_slice = _qc_calibration.read_slice

FIG_DIR = OUT_DIR / "figures"
DPI = 150

SUBSET_LABELS = {
    "all_registrations": "All registrations",
    "passes_global_qc": "Global QC",
    FINAL_QC_SUBSET: "Final QC",
}
# Subsets are ordinal (progressively stricter), so one hue light -> dark.
SUBSET_COLOURS = {
    "all_registrations": "#9dc3ef",
    "passes_global_qc": "#4f8fdc",
    FINAL_QC_SUBSET: "#1a4f95",
}
GT_EDGE = "#00b8d4"          # matches BOUNDARY_RGBA hue in 04b
CONSENSUS_EDGE = "#eb6834"
GRID_KW = dict(color="#dddddd", linewidth=0.6)

T_LABEL = safe_value(HEADLINE_THRESHOLD)


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure: {name}")


def tidy(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, **GRID_KW)
    ax.set_axisbelow(True)


def region_order(cons):
    """Regions ordered small -> large by median GT volume."""
    vol = cons.groupby("region")["vol_gt_mm3"].median().sort_values()
    return list(vol.index)


# ---------------------------------------------------------------------
# TABLES
# ---------------------------------------------------------------------

def iqr_str(s, fmt="{:.3f}"):
    s = s.dropna()
    if s.empty:
        return ""
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    return f"{fmt.format(med)} [{fmt.format(q1)}-{fmt.format(q3)}]"


def summary_table(cons, regions):
    head = cons[(cons.threshold == HEADLINE_THRESHOLD)]
    rows = []
    for (input_type, subset), g in head.groupby(["input_type", "subset"], sort=False):
        for region in regions:
            r = g[g.region == region]
            rows.append({
                "input_type": input_type, "subset": subset, "region": region,
                "n_brains": len(r),
                "gt_volume_mm3": iqr_str(r.vol_gt_mm3, "{:.3f}"),
                "dice": iqr_str(r.dice), "iou": iqr_str(r.iou),
                "sensitivity": iqr_str(r.sensitivity), "precision": iqr_str(r.precision),
                "vol_ratio": iqr_str(r.vol_ratio, "{:.2f}"),
                "hd95_um": iqr_str(r.hd95_um, "{:.0f}"), "assd_um": iqr_str(r.assd_um, "{:.0f}"),
                "centroid_dist_um": iqr_str(r.centroid_dist_um, "{:.0f}"),
                "dice_median": r.dice.median(), "hd95_median_um": r.hd95_um.median(),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "summary_table.csv", index=False)
    return out


# ---------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------

def plot_heatmaps(cons, regions):
    head = cons[(cons.threshold == HEADLINE_THRESHOLD) & (cons.subset == FINAL_QC_SUBSET)]
    specs = [
        ("dice", "Dice vs GT", "Blues", (0, 1), "{:.2f}"),
        ("hd95_um", "HD95 vs GT (µm)", "Blues_r", None, "{:.0f}"),
    ]
    for col, title, cmap, lims, fmt in specs:
        fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharey=True, constrained_layout=True)
        vmin, vmax = lims if lims else (0, np.nanpercentile(head[col], 98))
        for ax, input_type in zip(axes, INPUT_TYPES):
            mat = (head[head.input_type == input_type]
                   .pivot(index="region", columns="sample_id", values=col)
                   .reindex(index=regions, columns=SAMPLES))
            im = ax.imshow(mat.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat.values[i, j]
                    if np.isnan(v):
                        continue
                    frac = (v - vmin) / (vmax - vmin) if vmax > vmin else 0
                    dark_cell = frac > 0.6 if cmap == "Blues" else frac < 0.4
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                            color="white" if dark_cell else "#222222")
            ax.set_xticks(range(len(SAMPLES)), SAMPLES, rotation=40, ha="right")
            ax.set_yticks(range(len(regions)), regions)
            ax.set_title(f"{input_type}")
        axes[0].set_ylabel("Region (small → large GT volume)")
        fig.colorbar(im, ax=axes, shrink=0.6, label=title)
        fig.suptitle(f"{title}: final-QC consensus at threshold {HEADLINE_THRESHOLD}")
        save(fig, f"heatmap_{col.split('_')[0]}_final_qc_t{T_LABEL}")


def plot_threshold_curves(cons, regions):
    for col, ylabel in [("dice", "Dice vs GT"), ("hd95_um", "HD95 (µm)")]:
        for input_type in INPUT_TYPES:
            d = cons[cons.input_type == input_type]
            fig, axes = plt.subplots(4, 4, figsize=(13, 11), sharex=True, constrained_layout=True)
            for ax, region in zip(axes.flat, regions):
                r = d[d.region == region]
                for subset in SUBSETS:
                    s = r[r.subset == subset].groupby("threshold")[col]
                    med, lo, hi = s.median(), s.min(), s.max()
                    ax.fill_between(med.index, lo, hi, color=SUBSET_COLOURS[subset], alpha=0.15, linewidth=0)
                    ax.plot(med.index, med.values, color=SUBSET_COLOURS[subset], linewidth=2,
                            marker="o", markersize=4)
                ax.axvline(HEADLINE_THRESHOLD, color="#888888", linestyle=":", linewidth=1)
                ax.set_title(region, fontsize=10)
                if col == "dice":
                    ax.set_ylim(0, 1)
                tidy(ax)
            for ax in axes[-1]:
                ax.set_xlabel("Consensus threshold")
            for ax in axes[:, 0]:
                ax.set_ylabel(ylabel)
            handles = [Line2D([], [], color=SUBSET_COLOURS[s], linewidth=2, marker="o", label=SUBSET_LABELS[s])
                       for s in SUBSETS]
            fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), frameon=False)
            fig.suptitle(f"{ylabel} vs consensus threshold ({input_type}); line = median, band = range over 6 brains")
            save(fig, f"threshold_curves_{col.split('_')[0]}_{input_type}")


def subset_comparison(cons):
    head = cons[cons.threshold == HEADLINE_THRESHOLD]
    stats_rows = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    for j, input_type in enumerate(INPUT_TYPES):
        wide = {
            col: head[head.input_type == input_type]
            .pivot_table(index=["sample_id", "region"], columns="subset", values=col)[SUBSETS]
            for col in ("dice", "hd95_um")
        }
        for i, (col, label) in enumerate([("dice", "Dice vs GT"), ("hd95_um", "HD95 (µm)")]):
            ax = axes[i, j]
            w = wide[col]
            x = np.arange(len(SUBSETS))
            for _, row in w.iterrows():
                ax.plot(x, row.values, color="#bbbbbb", linewidth=0.6, alpha=0.6, zorder=1)
            ax.boxplot([w[s].dropna() for s in SUBSETS], positions=x, widths=0.35, showfliers=False,
                       medianprops=dict(color="#222222", linewidth=1.5), zorder=2)
            for k, s in enumerate(SUBSETS):
                ax.scatter(np.full(len(w), k), w[s], s=10, color=SUBSET_COLOURS[s], zorder=3)
            ax.set_xticks(x, [SUBSET_LABELS[s] for s in SUBSETS])
            ax.set_ylabel(label)
            if col == "hd95_um":
                ax.set_yscale("log")  # a few >2 mm failures would flatten the rest
            ax.set_title(f"{input_type}: paired brain × region, threshold {HEADLINE_THRESHOLD}", fontsize=10)
            tidy(ax)

            for a, b in [(SUBSETS[1], SUBSETS[0]), (SUBSETS[2], SUBSETS[0]), (SUBSETS[2], SUBSETS[1])]:
                pair = w[[a, b]].dropna()
                diff = pair[a] - pair[b]
                test = stats.wilcoxon(pair[a], pair[b]) if (diff != 0).any() else None
                stats_rows.append({
                    "input_type": input_type, "metric": col,
                    "comparison": f"{a} - {b}", "n_pairs": len(pair),
                    "median_difference": diff.median(), "mean_difference": diff.mean(),
                    "n_improved": int((diff > 0).sum() if col == "dice" else (diff < 0).sum()),
                    "wilcoxon_statistic": test.statistic if test else np.nan,
                    "wilcoxon_p": test.pvalue if test else np.nan,
                })

    save(fig, f"subset_comparison_t{T_LABEL}")
    out = pd.DataFrame(stats_rows)
    out.to_csv(OUT_DIR / "subset_comparison_stats.csv", index=False)
    return out


def consensus_vs_individual(cons, indiv, regions):
    head = cons[cons.threshold == HEADLINE_THRESHOLD]
    rows = []
    for (sample_id, input_type, region), g in indiv.groupby(["sample_id", "input_type", "region"]):
        single = g["dice"].dropna().values
        for subset in SUBSETS:
            c = head[(head.sample_id == sample_id) & (head.input_type == input_type)
                     & (head.region == region) & (head.subset == subset)]
            if c.empty:
                continue
            cd = float(c["dice"].item())
            rows.append({
                "sample_id": sample_id, "input_type": input_type, "region": region, "subset": subset,
                "consensus_dice": cd,
                "single_dice_median": np.median(single), "single_dice_max": single.max(),
                "single_dice_min": single.min(),
                "consensus_percentile_rank": stats.percentileofscore(single, cd, kind="weak"),
                "gain_over_median_single": cd - np.median(single),
                "gain_over_best_single": cd - single.max(),
            })
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "consensus_vs_individual.csv", index=False)

    style = make_sample_style(SAMPLES)
    rng = np.random.default_rng(0)
    for input_type in INPUT_TYPES:
        fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
        d = indiv[indiv.input_type == input_type]
        t = table[(table.input_type == input_type) & (table.subset == FINAL_QC_SUBSET)]
        for x, region in enumerate(regions):
            r = d[d.region == region]
            ax.scatter(x - 0.18 + rng.uniform(-0.12, 0.12, len(r)), r["dice"], s=6,
                       color="#b5b5b5", alpha=0.6, linewidths=0, zorder=1)
            for k, sample_id in enumerate(SAMPLES):
                row = t[(t.region == region) & (t.sample_id == sample_id)]
                if row.empty:
                    continue
                colour, marker = style[sample_id]
                ax.scatter(x + 0.12 + (k - 2.5) * 0.04, row["consensus_dice"], s=40, color=colour,
                           marker=marker, edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_xticks(range(len(regions)), regions, rotation=40, ha="right")
        ax.set_ylabel("Dice vs GT")
        ax.set_ylim(0, 1)
        ax.set_title(f"{input_type}: single registrations (grey, all 64 × 6 brains) vs "
                     f"final-QC consensus at {HEADLINE_THRESHOLD} (coloured, one per brain)", fontsize=10)
        tidy(ax)
        handles = [Line2D([], [], linestyle="", marker="o", color="#b5b5b5", label="Single registration")]
        handles += [Line2D([], [], linestyle="", marker=m, color=c, markeredgecolor="white", markersize=7, label=s)
                    for s, (c, m) in style.items()]
        ax.legend(handles=handles, ncol=7, loc="lower center", bbox_to_anchor=(0.5, 1.07), frameon=False, fontsize=8)
        save(fig, f"consensus_vs_individual_{input_type}")
    return table


def qc_vs_gt(indiv, regions):
    metrics = [("qc_iou", "Global IoU"), ("qc_grad_p95", "grad_p95"), ("qc_local_ssim", "Local SSIM")]
    rho_rows, gate_rows = [], []
    for (sample_id, input_type, region), g in indiv.groupby(["sample_id", "input_type", "region"]):
        for col, _ in metrics:
            ok = g[[col, "dice"]].dropna()
            rho, p = stats.spearmanr(ok[col], ok["dice"]) if len(ok) > 3 else (np.nan, np.nan)
            rho_rows.append({"sample_id": sample_id, "input_type": input_type, "region": region,
                             "qc_metric": col, "spearman_rho": rho, "p": p, "n": len(ok)})
        for gate, passed in [("passes_global_qc", g.passes_global_qc), ("in_final_qc", g.in_final_qc)]:
            a, b = g.loc[passed, "dice"].dropna(), g.loc[~passed, "dice"].dropna()
            test = stats.mannwhitneyu(a, b) if len(a) and len(b) else None
            gate_rows.append({"sample_id": sample_id, "input_type": input_type, "region": region,
                              "gate": gate, "n_pass": len(a), "n_fail": len(b),
                              "dice_median_pass": a.median(), "dice_median_fail": b.median(),
                              "dice_median_difference": a.median() - b.median(),
                              "mannwhitney_p": test.pvalue if test else np.nan})
    rho = pd.DataFrame(rho_rows)
    gates = pd.DataFrame(gate_rows)
    rho.to_csv(OUT_DIR / "qc_spearman.csv", index=False)
    gates.to_csv(OUT_DIR / "qc_gate_effect.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1, 1.6]))
    for ax, input_type in zip(axes[:2], INPUT_TYPES):
        mat = (rho[rho.input_type == input_type]
               .groupby(["region", "qc_metric"])["spearman_rho"].median().unstack()
               .reindex(index=regions, columns=[m for m, _ in metrics]))
        im = ax.imshow(mat.values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(v) > 0.6 else "#222222")
        ax.set_xticks(range(len(metrics)), [lab for _, lab in metrics])
        ax.set_yticks(range(len(regions)), regions)
        ax.set_title(f"{input_type}: median Spearman ρ\n(QC metric vs GT Dice, within brain)", fontsize=10)
    fig.colorbar(im, ax=axes[:2], shrink=0.6, label="Spearman ρ")

    ax = axes[2]
    gate_colours = {"passes_global_qc": SUBSET_COLOURS["passes_global_qc"],
                    "in_final_qc": SUBSET_COLOURS[FINAL_QC_SUBSET]}
    for k, (gate, label) in enumerate([("passes_global_qc", "Global QC"), ("in_final_qc", "Final QC")]):
        for m, input_type in enumerate(INPUT_TYPES):
            sub = gates[(gates.gate == gate) & (gates.input_type == input_type)]
            for x, region in enumerate(regions):
                vals = sub.loc[sub.region == region, "dice_median_difference"]
                offset = -0.3 + 0.2 * (2 * k + m)
                ax.scatter(np.full(len(vals), x + offset), vals, s=12, color=gate_colours[gate],
                           marker="o" if input_type == "raw" else "^", alpha=0.85, linewidths=0)
    ax.axhline(0, color="#666666", linewidth=0.8)
    ax.set_xticks(range(len(regions)), regions, rotation=40, ha="right")
    ax.set_ylabel("Median GT Dice: passing − failing registrations")
    ax.set_title("Does passing the QC gate predict better GT agreement? (one point per brain)", fontsize=10)
    tidy(ax)
    handles = [Line2D([], [], linestyle="", marker="o", color=gate_colours["passes_global_qc"], label="Global QC gate"),
               Line2D([], [], linestyle="", marker="o", color=gate_colours["in_final_qc"], label="Final QC gate"),
               Line2D([], [], linestyle="", marker="o", color="#666666", label="raw"),
               Line2D([], [], linestyle="", marker="^", color="#666666", label="preprocessed")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left")
    save(fig, "qc_vs_gt")
    return rho, gates


def _reliability_curve(bins, width=0.1, min_voxels=500):
    b = bins.copy()
    edges = np.arange(0, 1 + width, width)
    # p = 0 and p = 1 kept as their own points; interior binned.
    b["bin"] = np.where(b.p <= 0, -1, np.where(b.p >= 1, len(edges),
                                               np.digitize(b.p, edges[1:-1], right=True)))
    g = b.groupby("bin").apply(lambda x: pd.Series({
        "p": np.average(x.p, weights=x.n_voxels),
        "frac": x.n_gt_positive.sum() / x.n_voxels.sum(),
        "n_voxels": x.n_voxels.sum(),
    }))
    # Drop near-empty bins (e.g. p = 1 across all 64 registrations can be a
    # few dozen voxels) -- their fraction is noise, not calibration.
    return g[g.n_voxels >= min_voxels].sort_values("p")


def plot_reliability(bins, regions):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True, constrained_layout=True)
    for ax, input_type in zip(axes, INPUT_TYPES):
        ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
        for subset in SUBSETS:
            c = _reliability_curve(bins[(bins.input_type == input_type) & (bins.subset == subset)])
            ax.plot(c.p, c.frac, color=SUBSET_COLOURS[subset], linewidth=2, marker="o", markersize=4,
                    label=SUBSET_LABELS[subset])
        ax.set_xlabel("Consensus probability p (fraction of registrations)")
        ax.set_title(f"{input_type}: pooled over regions and brains")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        tidy(ax, "both")
    axes[0].set_ylabel("Fraction of voxels in GT region")
    axes[0].legend(frameon=False)
    save(fig, "reliability")

    for input_type in INPUT_TYPES:
        fig, axes = plt.subplots(4, 4, figsize=(12, 11), sharex=True, sharey=True, constrained_layout=True)
        for ax, region in zip(axes.flat, regions):
            ax.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
            for subset in SUBSETS:
                c = _reliability_curve(bins[(bins.input_type == input_type) & (bins.subset == subset)
                                            & (bins.region == region)])
                ax.plot(c.p, c.frac, color=SUBSET_COLOURS[subset], linewidth=1.8, marker="o", markersize=3)
            ax.set_title(region, fontsize=10)
            tidy(ax, "both")
        for ax in axes[-1]:
            ax.set_xlabel("p")
        for ax in axes[:, 0]:
            ax.set_ylabel("Frac. in GT")
        handles = [Line2D([], [], color=SUBSET_COLOURS[s], linewidth=2, label=SUBSET_LABELS[s]) for s in SUBSETS]
        fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), frameon=False)
        fig.suptitle(f"Probability-map reliability by region ({input_type}, pooled over brains)")
        save(fig, f"reliability_by_region_{input_type}")


def plot_dice_vs_volume(cons):
    head = cons[(cons.threshold == HEADLINE_THRESHOLD) & (cons.subset == FINAL_QC_SUBSET)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    for ax, input_type in zip(axes, INPUT_TYPES):
        d = head[head.input_type == input_type]
        ax.scatter(d.vol_gt_mm3, d.dice, s=14, color="#9dc3ef", linewidths=0, zorder=1)
        med = d.groupby("region")[["vol_gt_mm3", "dice"]].median()
        ax.scatter(med.vol_gt_mm3, med.dice, s=36, color="#1a4f95", edgecolors="white", linewidths=0.8, zorder=2)
        for region, row in med.iterrows():
            ax.annotate(region, (row.vol_gt_mm3, row.dice), xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("GT region volume (mm³, log)")
        ax.set_ylim(0, 1)
        ax.set_title(f"{input_type}: final-QC consensus at {HEADLINE_THRESHOLD}")
        tidy(ax, "both")
    axes[0].set_ylabel("Dice vs GT (light = brain, dark = region median)")
    save(fig, "dice_vs_gt_volume")


def plot_overlays(cons, regions):
    head = cons[(cons.threshold == HEADLINE_THRESHOLD) & (cons.subset == FINAL_QC_SUBSET)]
    _, agg = load_label_mapping()
    gt_cache = {}

    def gt_for(sample_id):
        if sample_id not in gt_cache:
            gt_cache[sample_id] = load_gt_atlas(sample_id)
        return gt_cache[sample_id]

    for input_type in INPUT_TYPES:
        d = head[head.input_type == input_type]
        fig, axes = plt.subplots(len(regions), 3, figsize=(9, 2.6 * len(regions)), constrained_layout=True)
        for i, region in enumerate(regions):
            r = d[d.region == region].dropna(subset=["dice"]).sort_values("dice")
            picks = {"worst": r.iloc[0], "median": r.iloc[len(r) // 2], "best": r.iloc[-1]}
            for j, (label, row) in enumerate(picks.items()):
                ax = axes[i, j]
                gt = gt_for(row.sample_id)
                gt_mask = gt_region_mask(gt, region, agg)
                pred = load_mask(consensus_path(row.sample_id, input_type, region, FINAL_QC_SUBSET,
                                                HEADLINE_THRESHOLD), gt.shape)
                z = int(np.argmax(gt_mask.sum(axis=(1, 2))))
                union = gt_mask[z] | pred[z]
                ys, xs = np.nonzero(union)
                pad = 12
                y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, union.shape[0])
                x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, union.shape[1])
                g_sl = gt_mask[z, y0:y1, x0:x1]
                boundary = g_sl & ~ndimage.binary_erosion(g_sl)
                image_slice = read_slice(image_path(row.sample_id), z)[y0:y1, x0:x1]
                show_slice_with_boundaries(ax, image_slice, boundary)
                p_sl = pred[z, y0:y1, x0:x1]
                if p_sl.any():
                    ax.contour(p_sl.astype(float), levels=[0.5], colors=[CONSENSUS_EDGE], linewidths=1.0)
                ax.set_title(f"{region} {label}: {row.sample_id}\nDice {row.dice:.2f}, HD95 "
                             f"{row.hd95_um:.0f} µm, z={z}", fontsize=8)
        handles = [Line2D([], [], color=GT_EDGE, linewidth=2, label="GT region boundary"),
                   Line2D([], [], color=CONSENSUS_EDGE, linewidth=2, label="Final-QC consensus (0.5)")]
        fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01), frameon=False)
        save(fig, f"overlays_{input_type}")


def main():
    # 07c/07d outputs are optional so the consensus figures can be made
    # while those (slower) steps are still running; rerun once they finish.
    cons = pd.read_csv(OUT_DIR / "consensus_vs_gt.csv")
    indiv_path, bins_path = OUT_DIR / "individual_vs_gt.csv", OUT_DIR / "calibration_bins.csv"
    indiv = pd.read_csv(indiv_path) if indiv_path.exists() else None
    bins = pd.read_csv(bins_path) if bins_path.exists() else None
    regions = region_order(cons)

    print("Tables + figures")
    summary = summary_table(cons, regions)
    sub_stats = subset_comparison(cons)
    plot_heatmaps(cons, regions)
    plot_threshold_curves(cons, regions)
    plot_dice_vs_volume(cons)
    plot_overlays(cons, regions)

    cvi = None
    if indiv is not None:
        cvi = consensus_vs_individual(cons, indiv, regions)
        qc_vs_gt(indiv, regions)
    else:
        print(f"  skipping consensus-vs-individual and QC-vs-GT ({indiv_path.name} not found)")
    if bins is not None:
        plot_reliability(bins, regions)
    else:
        print(f"  skipping reliability ({bins_path.name} not found)")

    head = summary[summary.subset == FINAL_QC_SUBSET]
    print("\nHeadline (final QC, threshold 0.5): median [IQR] over brains")
    print(head[["input_type", "region", "dice", "hd95_um", "vol_ratio"]].to_string(index=False))
    print("\nSubset comparison (Wilcoxon):")
    print(sub_stats.round(4).to_string(index=False))
    if cvi is None:
        return
    f = cvi[cvi.subset == FINAL_QC_SUBSET]
    print(f"\nFinal-QC consensus vs single registrations: median percentile rank "
          f"{f.consensus_percentile_rank.median():.0f}, median gain over median single "
          f"{f.gain_over_median_single.median():+.3f}, beats best single in "
          f"{(f.gain_over_best_single > 0).mean():.0%} of brain×region×input")


if __name__ == "__main__":
    main()
