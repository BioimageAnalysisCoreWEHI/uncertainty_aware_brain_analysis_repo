# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 2026

@author: zaman.i

Plot the global-QC threshold sensitivity sweep from
05_qc_threshold_sensitivity.py. Writes two figures (PNG + PDF each):

  *_profiles    Per-brain 1-D profiles, one row per region. Each metric is
                plotted against the IoU threshold (grad_p95 threshold held
                fixed) and against the grad_p95 threshold (IoU threshold
                held fixed). The fixed value is the reference gate if one
                was given to 05, otherwise "no gating" on that metric.
                  - consensus volume, as % of the ungated consensus volume
                  - Dice vs the reference gate (or vs ungated if no reference)
                Top row: registrations passing per brain.
                A flat line = the consensus region does not depend on the
                exact threshold in that range.

  *_heatmaps    Joint IoU x grad_p95 grid, one panel per region: the
                WORST brain's Dice at each gate (min across brains). Grey =
                at least one brain has fewer than min_registrations passing.
                First panel: fewest registrations passing in any brain.

Pass rule is the one used in 04_combine_qc_csvs.py:
    iou >= iou_threshold AND grad_p95 <= grad_threshold
so moving right on the IoU axis / down on the grad axis = stricter.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pipeline_common import make_sample_style


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

def load_sweep(sensitivity_dir: Path, consensus_threshold: float) -> pd.DataFrame:
    csv_files = sorted((sensitivity_dir / "per_sample").glob("*.csv"))

    if not csv_files:
        raise RuntimeError(f"No per-sample sweep CSVs found in {sensitivity_dir / 'per_sample'}")

    sweep = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    available = np.sort(sweep["consensus_threshold"].unique())
    match = available[np.isclose(available, consensus_threshold)]

    if len(match) == 0:
        raise ValueError(f"--consensus-threshold {consensus_threshold} not in sweep; available: {available.tolist()}")

    sweep = sweep[np.isclose(sweep["consensus_threshold"], match[0])].copy()

    if sweep["input_type"].nunique() > 1:
        sweep["sample_label"] = sweep["sample_id"].astype(str) + " (" + sweep["input_type"].astype(str) + ")"
    else:
        sweep["sample_label"] = sweep["sample_id"].astype(str)

    sweep["volume_pct_of_ungated"] = 100 * sweep["volume_mm3"] / sweep["volume_all_registrations_mm3"]

    print(f"Loaded {len(csv_files)} sample file(s), {len(sweep)} rows at consensus threshold {match[0]}.")

    return sweep


def nearest(values, target):
    values = np.asarray(values)
    return values[np.argmin(np.abs(values - target))]


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def plot_profile(ax, data, x_col, y_col, sample_style, reference_x=None):
    for label, (colour, marker) in sample_style.items():
        subset = data[data["sample_label"] == label].sort_values(x_col)
        ax.plot(
            subset[x_col], subset[y_col],
            color=colour, marker=marker, markersize=3.5, linewidth=1.5,
        )

    if reference_x is not None:
        ax.axvline(reference_x, color="0.3", linestyle="--", linewidth=1)

    ax.grid(color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)


def profile_figure(sweep, grid_info, sample_style, dice_col, dice_label, fixed_iou, fixed_grad, out_stem):
    iou_col = grid_info["iou_col"]
    grad_col = grid_info["grad_col"]
    reference_iou = grid_info["reference_iou"]
    reference_grad = grid_info["reference_grad"]

    acronyms = list(dict.fromkeys(sweep["acronym"]))

    along_iou = sweep[np.isclose(sweep["grad_threshold"], fixed_grad)]
    along_grad = sweep[np.isclose(sweep["iou_threshold"], fixed_iou)]

    n_rows = 1 + len(acronyms)
    fig, axes = plt.subplots(
        n_rows, 4, figsize=(16, 2.3 * n_rows), layout="constrained", squeeze=False,
    )

    iou_xlabel = f"{iou_col} threshold (keep >=)  [{grad_col} <= {fixed_grad}]"
    grad_xlabel = f"{grad_col} threshold (keep <=)  [{iou_col} >= {fixed_iou}]"

    # Row 0: registrations passing (independent of region -> first region only)
    first = acronyms[0]
    plot_profile(axes[0, 0], along_iou[along_iou["acronym"] == first], "iou_threshold", "n_pass",
                 sample_style, reference_iou)
    plot_profile(axes[0, 1], along_grad[along_grad["acronym"] == first], "grad_threshold", "n_pass",
                 sample_style, reference_grad)
    axes[0, 0].set_ylabel("registrations\npassing")
    axes[0, 0].set_title(f"vs {iou_col} threshold", loc="left", fontsize=10)
    axes[0, 1].set_title(f"vs {grad_col} threshold", loc="left", fontsize=10)
    axes[0, 1].invert_xaxis()

    axes[0, 2].axis("off")
    axes[0, 3].axis("off")
    handles = [
        Line2D([], [], color=c, marker=m, markersize=5, linewidth=1.5, label=l)
        for l, (c, m) in sample_style.items()
    ]
    if reference_iou is not None:
        handles.append(Line2D([], [], color="0.3", linestyle="--", label="reference gate"))
    axes[0, 2].legend(handles=handles, loc="center left", frameon=False, fontsize=9, ncol=2)

    for r, acronym in enumerate(acronyms, start=1):
        region_iou = along_iou[along_iou["acronym"] == acronym]
        region_grad = along_grad[along_grad["acronym"] == acronym]

        plot_profile(axes[r, 0], region_iou, "iou_threshold", "volume_pct_of_ungated", sample_style, reference_iou)
        plot_profile(axes[r, 1], region_grad, "grad_threshold", "volume_pct_of_ungated", sample_style, reference_grad)
        plot_profile(axes[r, 2], region_iou, "iou_threshold", dice_col, sample_style, reference_iou)
        plot_profile(axes[r, 3], region_grad, "grad_threshold", dice_col, sample_style, reference_grad)

        axes[r, 0].set_ylabel(f"{acronym}\nvolume (% ungated)")
        axes[r, 2].set_ylabel(dice_label)

        for c in (1, 3):
            axes[r, c].invert_xaxis()   # stricter to the right on every panel

        if r == 1:
            axes[r, 0].set_title("consensus volume", loc="left", fontsize=10)
            axes[r, 2].set_title("consensus overlap", loc="left", fontsize=10)

    for c, label in enumerate([iou_xlabel, grad_xlabel, iou_xlabel, grad_xlabel]):
        axes[-1, c].set_xlabel(label, fontsize=8)
    axes[0, 0].set_xlabel(iou_xlabel, fontsize=8)
    axes[0, 1].set_xlabel(grad_xlabel, fontsize=8)

    fig.suptitle(
        "Global-QC threshold sensitivity: per-brain consensus region vs gate threshold "
        f"(consensus probability >= {sweep['consensus_threshold'].iloc[0]})\n"
        "Stricter gate to the right on every panel. Flat lines = region insensitive to the exact threshold.",
        fontsize=12,
    )

    save(fig, out_stem)


def heatmap_figure(sweep, grid_info, dice_col, dice_label, min_registrations, out_stem):
    iou_col = grid_info["iou_col"]
    grad_col = grid_info["grad_col"]
    reference_iou = grid_info["reference_iou"]
    reference_grad = grid_info["reference_grad"]

    acronyms = list(dict.fromkeys(sweep["acronym"]))
    n_panels = 1 + len(acronyms)
    n_cols = 4
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.2 * n_cols, 3.6 * n_rows), layout="constrained", squeeze=False,
    )
    axes = axes.ravel()

    def draw(ax, pivot, title, cmap, vmin, vmax, cbar_label):
        mesh = ax.pcolormesh(
            pivot.columns, pivot.index, pivot.to_numpy(),
            shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax,
        )
        ax.set_facecolor("0.85")   # NaN cells (too few registrations) show as grey
        if reference_iou is not None:
            ax.plot(reference_iou, reference_grad, marker="*", color="#e34948",
                    markersize=12, markeredgecolor="white")
        ax.invert_yaxis()           # stricter grad gate at the bottom
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel(f"{iou_col} threshold (keep >=)", fontsize=8)
        ax.set_ylabel(f"{grad_col} threshold (keep <=)", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(mesh, ax=ax, shrink=0.85, label=cbar_label, extend="min" if vmin > 0 else "neither")

    # Panel 0: fewest registrations passing in any brain.
    first = sweep[sweep["acronym"] == acronyms[0]]
    n_pass = first.pivot_table(index="grad_threshold", columns="iou_threshold", values="n_pass", aggfunc="min")
    draw(axes[0], n_pass, "min registrations passing (any brain)", "Blues", 0, n_pass.to_numpy().max(), "n")
    axes[0].contour(
        n_pass.columns, n_pass.index, n_pass.to_numpy(), levels=[min_registrations - 0.5],
        colors="black", linewidths=1,
    )

    for ax, acronym in zip(axes[1:], acronyms):
        region = sweep[sweep["acronym"] == acronym]
        # min() over brains, but NaN if ANY brain is NaN at that gate.
        worst = region.pivot_table(
            index="grad_threshold", columns="iou_threshold", values=dice_col,
            aggfunc=lambda v: v.min() if v.notna().all() else np.nan, dropna=False,
        )
        draw(ax, worst, f"{acronym}: worst-brain {dice_label}", "Blues", 0.5, 1.0, "Dice")

    for ax in axes[n_panels:]:
        ax.axis("off")

    reference_text = (
        f"red star = reference gate ({iou_col} >= {reference_iou}, {grad_col} <= {reference_grad}); "
        if reference_iou is not None else ""
    )
    fig.suptitle(
        f"Global-QC threshold sensitivity: worst-brain {dice_label} over the joint threshold grid\n"
        f"{reference_text}grey = fewer than {min_registrations} registrations pass in at least one brain "
        f"(black contour in first panel). Stricter towards bottom-right.",
        fontsize=12,
    )

    save(fig, out_stem)


def save(fig, out_stem):
    for suffix in [".png", ".pdf"]:
        out_path = out_stem.with_suffix(suffix)
        fig.savefig(out_path, dpi=130)
        print(f"Saved: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sensitivity-dir",
        default="results/qc_tables/threshold_sensitivity/qc_results",
        help="Output folder of 05_qc_threshold_sensitivity.py.",
    )

    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.5,
        help="Which consensus probability threshold to plot.",
    )

    parser.add_argument(
        "--regions",
        nargs="+",
        default=None,
        help="Region acronyms to plot (default: all in the sweep).",
    )

    args = parser.parse_args()

    sensitivity_dir = Path(args.sensitivity_dir)

    with open(sensitivity_dir / "threshold_grid.json", "r") as file:
        grid_info = json.load(file)

    sweep = load_sweep(sensitivity_dir, args.consensus_threshold)

    if args.regions is not None:
        sweep = sweep[sweep["acronym"].isin(args.regions)]
        if sweep.empty:
            raise ValueError(f"None of {args.regions} found in the sweep.")

    has_reference = grid_info["reference_iou"] is not None

    if has_reference:
        dice_col, dice_label = "dice_vs_reference", "Dice vs reference gate"
        fixed_iou = nearest(grid_info["iou_thresholds"], grid_info["reference_iou"])
        fixed_grad = nearest(grid_info["grad_thresholds"], grid_info["reference_grad"])
    else:
        dice_col, dice_label = "dice_vs_all_registrations", "Dice vs ungated"
        fixed_iou = min(grid_info["iou_thresholds"])     # no IoU gating
        fixed_grad = max(grid_info["grad_thresholds"])   # no grad gating
        print("No reference gate in this sweep: profiles hold the other metric ungated, "
              "Dice is vs the ungated consensus. Re-run 05 with --reference-iou/--reference-grad "
              "once you have provisional thresholds.")

    sample_style = make_sample_style(sweep["sample_label"].unique())

    stem = f"threshold_sensitivity_consensus_{str(args.consensus_threshold).replace('.', 'p')}"

    profile_figure(
        sweep, grid_info, sample_style, dice_col, dice_label, fixed_iou, fixed_grad,
        sensitivity_dir / f"{stem}_profiles",
    )
    heatmap_figure(
        sweep, grid_info, dice_col, dice_label, grid_info["min_registrations"],
        sensitivity_dir / f"{stem}_heatmaps",
    )


if __name__ == "__main__":
    main()
