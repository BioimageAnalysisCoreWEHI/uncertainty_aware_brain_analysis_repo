# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 2026

@author: zaman.i

Dataset-level QC calibration figure.

Run AFTER 04_combine_qc_csvs.py (needs the combined qc_results*.csv) and
BEFORE any step that relies on passes_global_qc. This script is purely a
diagnostic: it reads the QC table and existing brainreg outputs, and writes
one figure (PNG + PDF) plus a small CSV of the example registrations shown.
It does NOT filter anything, does NOT choose thresholds, and does NOT modify
any registration output.

The figure has three parts:
  1. Scatter of ALL registrations (unfiltered): IoU (x) vs grad_p95 (y),
     coloured/shaped by sample. Example registrations are ringed and
     labelled (e.g. "B3") so they can be found in the image grid.
  2. Per-sample IoU and grad_p95 distributions (box + points), to spot a
     sample whose distribution is systematically shifted.
  3. A grid of example registrations spanning the observed QC space.
     Columns go low -> high IoU (left -> right), rows go high -> low
     grad_p95 (top -> bottom), i.e. the grid is laid out the same way as
     the scatter plot. Each panel is read straight from that
     registration's brainreg output folder (save_dir): the middle slice
     of downsampled.tiff (the sample in its own space) with that
     registration's boundaries.tiff (warped atlas outlines) drawn on top.
     All registrations of one sample share the same downsampled.tiff
     grid, so the slice is identical between them; different samples
     have their own volumes, so the middle slice is only approximately
     the same anatomy across samples.

How examples are selected
-------------------------
Both metrics are converted to percentile ranks (0-1) across the whole
dataset, so the grid adapts to wherever the data actually lie (this is
what makes it reusable when absolute grad_p95 values shift between
datasets). An N x N grid of target points is placed at evenly spaced
percentiles (0, 1/(N-1), ..., 1) of each metric -- so the extremes are
included. For each target, the closest not-yet-used registration (in
percentile-rank space) is chosen. If the closest registration is further
than ~half a grid cell away, that part of the QC space is empty (e.g. no
registration has both low IoU and low grad_p95) and the panel is left
blank rather than showing a misleading example.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import tifffile as tiff
import yaml
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pipeline_common import normalise_path, make_sample_style

BOUNDARY_RGB = (0.0, 0.9, 1.0)


# ---------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------

def shorten_sample_ids(sample_ids, min_tokens: int = 2) -> dict:
    """
    Map each full sample_id to the shortest leading run of "_"-separated
    tokens that is still unique across the dataset, e.g.
    "CR_Syt6_cre_rfp_35_StitchedImagesPaths_..." -> "CR_Syt6".
    Uses at least min_tokens tokens where the id has them, so labels stay
    readable. Short ids like "GAD2" are left unchanged.
    """

    ids = sorted({str(s) for s in sample_ids})
    tokens = {s: s.split("_") for s in ids}
    max_tokens = max(len(t) for t in tokens.values())

    for n in range(1, max_tokens + 1):
        short = {s: "_".join(t[:max(n, min(min_tokens, len(t)))]) for s, t in tokens.items()}
        if len(set(short.values())) == len(ids):
            return short

    return {s: s for s in ids}


def make_sample_labels(qc: pd.DataFrame, shorten: bool = True) -> tuple[pd.Series, dict]:
    """One label per sample (shortened unless shorten=False). Include
    input_type only if the QC table mixes several input types (e.g. raw +
    preprocessed). Returns (labels, {label: full sample_id})."""

    sample_ids = qc["sample_id"].astype(str)
    short = shorten_sample_ids(sample_ids) if shorten else {s: s for s in sample_ids.unique()}
    labels = sample_ids.map(short)

    if "input_type" in qc.columns and qc["input_type"].nunique() > 1:
        labels = labels + " (" + qc["input_type"].astype(str) + ")"

    key = dict(zip(labels, sample_ids))

    return labels, key


def add_percentile_ranks(qc: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """Add rank_x / rank_y: each metric's percentile rank across the whole
    dataset, rescaled so the lowest value is exactly 0 and the highest 1."""

    for rank_col, col in (("rank_x", x_col), ("rank_y", y_col)):
        rank = qc[col].rank(pct=True, method="average")
        qc[rank_col] = (rank - rank.min()) / max(rank.max() - rank.min(), 1e-12)

    return qc


def select_examples(
    qc: pd.DataFrame,
    x_col: str,
    y_col: str,
    grid_size: int,
) -> pd.DataFrame:
    """
    Pick up to grid_size x grid_size registrations spanning the joint
    x_col x y_col space (see module docstring).

    Returns one row per grid cell that has an example, with extra columns
    grid_row (0 = highest y), grid_col (0 = lowest x) and panel_label.
    Expects the percentile-rank columns added by add_percentile_ranks().
    """

    rank_x = qc["rank_x"].to_numpy()
    rank_y = qc["rank_y"].to_numpy()

    targets = np.linspace(0, 1, grid_size)
    step = 1 / (grid_size - 1)
    max_distance = 0.75 * step

    used = np.zeros(len(qc), dtype=bool)
    picks = []

    for grid_row in range(grid_size):
        target_y = targets[grid_size - 1 - grid_row]   # top row = highest y

        for grid_col in range(grid_size):
            target_x = targets[grid_col]

            distance = np.hypot(rank_x - target_x, rank_y - target_y)
            distance[used] = np.inf

            best = int(np.argmin(distance))

            if distance[best] > max_distance:
                continue

            used[best] = True

            pick = qc.iloc[best].copy()
            pick["grid_row"] = grid_row
            pick["grid_col"] = grid_col
            pick["panel_label"] = f"{chr(ord('A') + grid_row)}{grid_col + 1}"
            pick["target_pct_" + x_col] = target_x
            pick["target_pct_" + y_col] = target_y
            picks.append(pick)

    return pd.DataFrame(picks)


def find_existing_file(folder: Path, possible_names: list[str]) -> Path | None:
    for name in possible_names:
        path = folder / name
        if path.exists():
            return path
    return None


def volume_shape(path: Path) -> tuple:
    with tiff.TiffFile(path) as tif:
        return tuple(tif.series[0].shape)


def read_slice(path: Path, slice_index: int) -> np.ndarray:
    """Read one axis-0 slice without loading the whole volume where the
    TIFF stores one page per slice (brainreg's default)."""

    with tiff.TiffFile(path) as tif:
        shape = tif.series[0].shape

        if len(tif.pages) == shape[0]:
            return tif.pages[slice_index].asarray()

        return tif.series[0].asarray()[slice_index]


# ---------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------

def tissue_crop(image_slice, boundary_mask, margin=6):
    """Row/col slices covering the tissue AND the warped atlas outline (so
    an atlas hanging off the tissue stays visible), plus a small margin."""

    low, high = np.percentile(image_slice, [1, 99.5])
    keep = boundary_mask | (image_slice > low + 0.1 * (high - low))

    rows = np.flatnonzero(keep.any(axis=1))
    cols = np.flatnonzero(keep.any(axis=0))

    if len(rows) == 0 or len(cols) == 0:
        return slice(None), slice(None)

    return (
        slice(max(rows[0] - margin, 0), rows[-1] + margin + 1),
        slice(max(cols[0] - margin, 0), cols[-1] + margin + 1),
    )


def show_slice_with_boundaries(ax, image_slice, boundary_mask, boundary_alpha):
    crop = tissue_crop(image_slice, boundary_mask)
    image_slice = image_slice[crop]
    boundary_mask = boundary_mask[crop]

    low, high = np.percentile(image_slice, [1, 99.5])

    ax.imshow(image_slice, cmap="gray", vmin=low, vmax=max(high, low + 1))

    overlay = np.zeros(boundary_mask.shape + (4,), dtype=np.float32)
    overlay[boundary_mask] = (*BOUNDARY_RGB, boundary_alpha)
    ax.imshow(overlay, interpolation="nearest")

    ax.set_xticks([])
    ax.set_yticks([])


def plot_scatter(ax, qc, examples, x_col, y_col, sample_style, title, point_size=28):
    for label, (colour, marker) in sample_style.items():
        subset = qc[qc["sample_label"] == label]
        ax.scatter(
            subset[x_col], subset[y_col],
            s=point_size, c=colour, marker=marker, alpha=0.75,
            edgecolors="white", linewidths=0.5, label=label,
        )

    ax.scatter(
        examples[x_col], examples[y_col],
        s=point_size * 5, facecolors="none", edgecolors="black", linewidths=1.2,
    )

    for _, row in examples.iterrows():
        ax.annotate(
            row["panel_label"], (row[x_col], row[y_col]),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, fontweight="bold", clip_on=True,
        )

    ax.set_xlabel(f"{x_col} (tissue-atlas overlap; higher = better)")
    ax.set_ylabel(f"{y_col} (deformation roughness;\nlower = smoother)")
    ax.set_title(title, loc="left", fontsize=10)
    ax.grid(color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)


def core_limits(qc, x_col, y_col, lower_pct=10, upper_pct=90):
    """Axis limits for a zoom on the bulk of the data: IoU above its
    lower_pct percentile, grad_p95 below its upper_pct percentile."""

    x = qc[x_col]
    y = qc[y_col]

    x0, x1 = np.percentile(x, lower_pct), x.max()
    y0, y1 = y.min(), np.percentile(y, upper_pct)

    x_pad = 0.05 * max(x1 - x0, 1e-6)
    y_pad = 0.05 * max(y1 - y0, 1e-6)

    return (x0 - x_pad, x1 + x_pad), (y0 - y_pad, y1 + y_pad)


def plot_legend_and_key(ax, sample_style, label_key):
    """Sample legend plus a key from short label to full sample_id."""

    ax.axis("off")

    handles = [
        Line2D([], [], color=c, marker=m, linestyle="", markersize=7, label=l)
        for l, (c, m) in sample_style.items()
    ]
    handles.append(
        Line2D([], [], color="black", marker="o", markerfacecolor="none",
               linestyle="", markersize=10, label="example shown below")
    )
    # One row of legend entries across the top; key underneath it.
    ax.legend(
        handles=handles, loc="upper left", ncol=len(handles),
        frameon=False, fontsize=9, bbox_to_anchor=(0, 1), borderaxespad=0,
    )

    shortened = {label: full for label, full in label_key.items() if label != full}
    if shortened:
        key_text = "\n".join(f"{label} = {full}" for label, full in sorted(shortened.items()))
        ax.text(0, 0, "Sample key:\n" + key_text, fontsize=7.5, color="0.3",
                va="bottom", ha="left", linespacing=1.3, transform=ax.transAxes)


def plot_distribution(ax, qc, col, sample_style):
    labels = list(sample_style.keys())
    data = [qc.loc[qc["sample_label"] == label, col].to_numpy() for label in labels]

    # "orientation" replaced "vert" in matplotlib 3.10.
    if tuple(int(v) for v in matplotlib.__version__.split(".")[:2]) >= (3, 10):
        horizontal = {"orientation": "horizontal"}
    else:
        horizontal = {"vert": False}

    ax.boxplot(
        data, **horizontal, widths=0.55, showfliers=False,
        medianprops={"color": "black"},
        boxprops={"color": "0.4"}, whiskerprops={"color": "0.4"}, capprops={"color": "0.4"},
    )

    rng = np.random.default_rng(0)
    for i, (label, values) in enumerate(zip(labels, data), start=1):
        colour, marker = sample_style[label]
        jitter = rng.uniform(-0.18, 0.18, size=len(values))
        ax.scatter(values, i + jitter, s=10, c=colour, marker=marker, alpha=0.6, linewidths=0)

    ax.set_yticks(range(1, len(labels) + 1), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(col)
    ax.set_title(f"{col} per sample", loc="left", fontsize=10)
    ax.grid(axis="x", color="0.9", linewidth=0.6)
    ax.set_axisbelow(True)


def plot_position_inset(ax, qc, row):
    """QC-space map in the panel corner, in percentile-rank space (the
    space examples are picked in, so points are spread evenly rather than
    piled into one corner): all registrations in grey, this one in black.
    Right = higher IoU, up = higher grad_p95."""

    inset = ax.inset_axes([0.76, 0.02, 0.22, 0.22])
    inset.scatter(qc["rank_x"], qc["rank_y"], s=1.5, c="0.7", linewidths=0)
    inset.scatter([row["rank_x"]], [row["rank_y"]], s=28, c="black", edgecolors="white", linewidths=0.6)
    inset.set_xlim(-0.05, 1.05)
    inset.set_ylim(-0.05, 1.05)
    inset.set_xticks([])
    inset.set_yticks([])


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config/config.yaml",
    )

    parser.add_argument(
        "--qc",
        default="results/qc_tables/qc_results.csv",
        help="Combined QC table from 04_combine_qc_csvs.py.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/qc_tables/qc_calibration",
    )

    parser.add_argument(
        "--iou-col",
        default="iou",
    )

    parser.add_argument(
        "--grad-col",
        default="grad_p95",
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=4,
        help="Examples are picked on a grid-size x grid-size grid over QC space.",
    )

    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help=(
            "Axis-0 slice of each registration's downsampled.tiff to show "
            "(default: middle slice of that volume)."
        ),
    )

    parser.add_argument(
        "--boundary-alpha",
        type=float,
        default=0.4,
        help="Opacity of the atlas boundaries drawn over each example (0-1).",
    )

    parser.add_argument(
        "--full-labels",
        action="store_true",
        help="Use full sample_ids everywhere instead of shortened labels + key.",
    )

    args = parser.parse_args()

    if args.grid_size < 2:
        raise ValueError("--grid-size must be at least 2")

    # -----------------------------------------------------------------
    # Load configuration and QC table
    # -----------------------------------------------------------------

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)

    qc_path = Path(args.qc)
    qc = pd.read_csv(qc_path)

    n_total = len(qc)
    qc = qc.dropna(subset=[args.iou_col, args.grad_col]).reset_index(drop=True)

    if len(qc) < n_total:
        print(f"Skipping {n_total - len(qc)} registrations with missing {args.iou_col}/{args.grad_col}.")

    qc["sample_label"], label_key = make_sample_labels(qc, shorten=not args.full_labels)
    qc = add_percentile_ranks(qc, args.iou_col, args.grad_col)

    sample_labels = sorted(qc["sample_label"].unique())
    sample_style = make_sample_style(sample_labels)

    # -----------------------------------------------------------------
    # Select examples and locate their brainreg outputs
    # -----------------------------------------------------------------

    examples = select_examples(qc, args.iou_col, args.grad_col, args.grid_size)
    print(f"Selected {len(examples)} example registrations on a {args.grid_size}x{args.grid_size} grid.")

    downsampled_names = cfg["brainreg_outputs"].get(
        "downsampled_names", ["downsampled.tiff", "downsampled.tif"]
    )
    boundaries_names = cfg["brainreg_outputs"].get(
        "boundaries_names", ["boundaries.tiff", "boundaries.tif"]
    )

    image_paths = {}

    for _, row in examples.iterrows():
        save_dir = normalise_path(row["save_dir"])
        image_paths[row["panel_label"]] = (
            find_existing_file(save_dir, downsampled_names),
            find_existing_file(save_dir, boundaries_names),
        )

    # Panel aspect ratio from the first available downsampled.tiff.
    first_image = next((d for d, _ in image_paths.values() if d is not None), None)
    image_shape = volume_shape(first_image) if first_image else (1, 1, 1.5)

    # -----------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------

    n = args.grid_size
    panel_width = max(3.6, 16 / n)
    panel_height = panel_width * image_shape[1] / image_shape[2] + 0.75
    key_height = 0.7 + 0.17 * sum(label != full for label, full in label_key.items())
    top_height = 5.5 + key_height

    fig = plt.figure(
        figsize=(n * panel_width, top_height + n * panel_height),
        layout="constrained",
    )

    outer = fig.add_gridspec(2, 1, height_ratios=[top_height, n * panel_height])

    top = outer[0].subgridspec(3, 3, width_ratios=[1, 1, 0.9], height_ratios=[1, 1, key_height / 2.75])
    ax_scatter = fig.add_subplot(top[:2, 0])
    ax_zoom = fig.add_subplot(top[:2, 1])
    ax_iou = fig.add_subplot(top[0, 2])
    ax_grad = fig.add_subplot(top[1, 2])
    ax_key = fig.add_subplot(top[2, :])

    plot_scatter(
        ax_scatter, qc, examples, args.iou_col, args.grad_col, sample_style,
        title=f"All registrations, unfiltered (n = {len(qc)})",
    )

    # Zoom on the bulk of the data, where a threshold decision is made.
    x_lim, y_lim = core_limits(qc, args.iou_col, args.grad_col)
    plot_scatter(
        ax_zoom, qc, examples, args.iou_col, args.grad_col, sample_style,
        title="Zoom: dashed box on the left", point_size=36,
    )
    ax_zoom.set_xlim(*x_lim)
    ax_zoom.set_ylim(*y_lim)
    ax_scatter.add_patch(Rectangle(
        (x_lim[0], y_lim[0]), x_lim[1] - x_lim[0], y_lim[1] - y_lim[0],
        fill=False, linestyle="--", linewidth=1, edgecolor="0.3",
    ))

    plot_distribution(ax_iou, qc, args.iou_col, sample_style)
    plot_distribution(ax_grad, qc, args.grad_col, sample_style)
    plot_legend_and_key(ax_key, sample_style, label_key)

    fig.suptitle(
        f"QC calibration: {qc_path.name}  |  {len(sample_labels)} samples, {len(qc)} registrations\n"
        f"Example grid: columns = low -> high {args.iou_col}, rows = high -> low {args.grad_col} "
        f"(percentile-spaced).\n"
        f"Each panel: that registration's downsampled.tiff (sample space, "
        f"{'slice ' + str(args.slice_index) if args.slice_index is not None else 'middle slice'}) "
        f"with its boundaries.tiff in cyan, cropped to tissue.\n"
        f"Corner inset: position in percentile-rank space (right = higher {args.iou_col}, "
        f"up = higher {args.grad_col}).",
        fontsize=12,
    )

    # -----------------------------------------------------------------
    # Example grid
    # -----------------------------------------------------------------

    grid = outer[1].subgridspec(n, n)
    for grid_row in range(n):
        for grid_col in range(n):
            ax = fig.add_subplot(grid[grid_row, grid_col])
            ax.set_xticks([])
            ax.set_yticks([])

            match = examples[
                (examples["grid_row"] == grid_row) & (examples["grid_col"] == grid_col)
            ] if len(examples) else examples

            if len(match) == 0:
                ax.text(
                    0.5, 0.5, "no registrations\nin this part of QC space",
                    ha="center", va="center", fontsize=8, color="0.5",
                    transform=ax.transAxes,
                )
                for spine in ax.spines.values():
                    spine.set_color("0.85")
                continue

            row = match.iloc[0]

            downsampled_path, boundaries_path = image_paths[row["panel_label"]]
            slice_index = None

            if downsampled_path is None or boundaries_path is None:
                ax.text(0.5, 0.5, "downsampled / boundaries\nnot found in save_dir",
                        ha="center", va="center", fontsize=8, transform=ax.transAxes)
            else:
                slice_index = (
                    args.slice_index if args.slice_index is not None
                    else volume_shape(downsampled_path)[0] // 2
                )
                image_slice = read_slice(downsampled_path, slice_index).astype(np.float32)
                boundary_mask = read_slice(boundaries_path, slice_index) != 0
                show_slice_with_boundaries(ax, image_slice, boundary_mask, args.boundary_alpha)

            plot_position_inset(ax, qc, row)

            colour, marker = sample_style[row["sample_label"]]
            ax.plot(0.03, 0.95, marker=marker, color=colour, markersize=8,
                    markeredgecolor="white", transform=ax.transAxes)

            ax.set_title(
                f"{row['panel_label']}  {row['sample_label']}\n"
                f"{args.iou_col} = {row[args.iou_col]:.3f}   {args.grad_col} = {row[args.grad_col]:.4f}\n"
                f"bend {row.get('bending_energy_weight', 'NA')}, grid {row.get('grid_spacing', 'NA')}, "
                f"slice {slice_index}",
                fontsize=8, loc="left",
            )

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{qc_path.stem}_calibration"

    for suffix in [".png", ".pdf"]:
        out_path = output_dir / f"{stem}{suffix}"
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")

    plt.close(fig)

    example_cols = [
        "panel_label", "grid_row", "grid_col", "sample_label", "sample_id", "input_type",
        "bending_energy_weight", "grid_spacing", "smoothing_sigma_floating",
        args.iou_col, args.grad_col,
        f"target_pct_{args.iou_col}", f"target_pct_{args.grad_col}",
        "save_dir",
    ]
    examples_path = output_dir / f"{stem}_examples.csv"
    examples[[c for c in example_cols if c in examples.columns]].to_csv(examples_path, index=False)
    print(f"Saved: {examples_path}")


if __name__ == "__main__":
    main()
