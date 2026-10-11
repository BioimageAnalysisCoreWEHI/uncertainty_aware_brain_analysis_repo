# -*- coding: utf-8 -*-
"""
09_plot_region_gallery.py

Visual-inspection gallery: for every region x brain, one representative
coronal slice with the GT region boundary and the consensus boundary drawn
on the brain image, annotated with that consensus's 3D scores from
07b (consensus_vs_gt.csv).

One figure per region (rows = raw / preprocessed, columns = the 6 brains),
plus one multi-page PDF with all regions. Each input type is drawn on its
own image (raw -> *_downsampled_25um_coronal_flipped.tif; preprocessed ->
*_downsampled_25um_preprocessed.tif resliced with the same transpose(2,0,1)
+ flip(1,2) as setting_up/downsample_and_reslice.py), so the contour sits
on the image that registration actually saw.

Representative slice = the coronal slice (axis 0) with the largest GT
region area in that brain; the same slice and crop are used for both
input types so the two rows are directly comparable.

Run after 07b. Plot with spatialdata_env (matplotlib in analysis_env
crashes on savefig on this machine).

Outputs: results/validation_gt/figures/region_gallery/
    gallery_<ACR>_<subset>_t<thr>.png
    region_gallery_<subset>_t<thr>.pdf        (all regions, one page each)
"""

import argparse

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

from gt_utils import (
    IMAGE_DIR, OUT_DIR, SAMPLES, INPUT_TYPES, FINAL_QC_SUBSET, HEADLINE_THRESHOLD, SUBSETS,
    region_table, load_label_mapping, load_gt_atlas, gt_region_mask,
    consensus_path, image_path, load_mask, dice, safe_value,
)

GT_EDGE = "#00b8d4"
CONSENSUS_EDGE = "#eb6834"
CROP_PAD = 15
INPUT_LABELS = {"raw": "raw", "preprocessed": "preprocessed"}


def load_image(sample_id, input_type, expected_shape):
    if input_type == "raw":
        img = tiff.imread(image_path(sample_id))
    else:
        stem = f"{sample_id}_AF_level1"
        native = tiff.imread(IMAGE_DIR / stem / f"{stem}_downsampled_25um_preprocessed.tif")
        img = np.ascontiguousarray(np.flip(np.transpose(native, (2, 0, 1)), axis=(1, 2)))
    if img.shape != tuple(expected_shape):
        raise ValueError(f"{sample_id}/{input_type} image shape {img.shape} != GT {tuple(expected_shape)}")
    return img


def show(ax, image_slice, gt_slice, pred_slice):
    low, high = np.percentile(image_slice, [1, 99.5])
    ax.imshow(image_slice, cmap="gray", vmin=low, vmax=max(high, low + 1), interpolation="nearest")
    if gt_slice.any():
        ax.contour(gt_slice.astype(float), levels=[0.5], colors=[GT_EDGE], linewidths=1.3)
    if pred_slice.any():
        ax.contour(pred_slice.astype(float), levels=[0.5], colors=[CONSENSUS_EDGE], linewidths=1.3)
    else:
        ax.text(0.5, 0.06, "no consensus voxels on this slice", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7, color="white",
                bbox=dict(facecolor="black", alpha=0.6, linewidth=0, pad=2))
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default=FINAL_QC_SUBSET, choices=SUBSETS)
    parser.add_argument("--threshold", type=float, default=HEADLINE_THRESHOLD)
    args = parser.parse_args()

    tag = f"{args.subset}_t{safe_value(args.threshold)}"
    out_dir = OUT_DIR / "figures" / "region_gallery"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(OUT_DIR / "consensus_vs_gt.csv")
    scores = scores[(scores.subset == args.subset) & np.isclose(scores.threshold, args.threshold)]
    scores = scores.set_index(["sample_id", "input_type", "region"])

    _, agg = load_label_mapping()
    regions = region_table()

    # Load each brain once: GT atlas + both images.
    data = {}
    for sample_id in SAMPLES:
        gt = load_gt_atlas(sample_id)
        data[sample_id] = {
            "gt": gt,
            **{t: load_image(sample_id, t, gt.shape) for t in INPUT_TYPES},
        }
        print(f"loaded {sample_id}", flush=True)

    pdf_path = out_dir / f"region_gallery_{tag}.pdf"
    with PdfPages(pdf_path) as pdf:
        for r in regions.itertuples():
            fig, axes = plt.subplots(len(INPUT_TYPES), len(SAMPLES), figsize=(3.3 * len(SAMPLES), 3.1 * len(INPUT_TYPES)),
                                     constrained_layout=True, squeeze=False)
            for j, sample_id in enumerate(SAMPLES):
                d = data[sample_id]
                gt_mask = gt_region_mask(d["gt"], r.acronym, agg)
                preds = {t: load_mask(consensus_path(sample_id, t, r.acronym, args.subset, args.threshold),
                                      d["gt"].shape) for t in INPUT_TYPES}

                z = int(np.argmax(gt_mask.sum(axis=(1, 2))))
                union = gt_mask[z].copy()
                for p in preds.values():
                    union |= p[z]
                ys, xs = np.nonzero(union)
                y0, y1 = max(ys.min() - CROP_PAD, 0), min(ys.max() + CROP_PAD + 1, union.shape[0])
                x0, x1 = max(xs.min() - CROP_PAD, 0), min(xs.max() + CROP_PAD + 1, union.shape[1])
                crop = np.s_[y0:y1, x0:x1]

                for i, input_type in enumerate(INPUT_TYPES):
                    ax = axes[i, j]
                    pred = preds[input_type]
                    show(ax, d[input_type][z][crop], gt_mask[z][crop], pred[z][crop])
                    s = scores.loc[(sample_id, input_type, r.acronym)]
                    ax.set_title(
                        f"{sample_id} · {INPUT_LABELS[input_type]} · z={z}\n"
                        f"Dice {s.dice:.2f} · HD95 {s.hd95_um:.0f} µm · vol {s.vol_ratio:.2f}×\n"
                        f"slice Dice {dice(pred[z], gt_mask[z]):.2f}",
                        fontsize=8,
                    )

            handles = [Line2D([], [], color=GT_EDGE, linewidth=2, label="GT region (independent atlas)"),
                       Line2D([], [], color=CONSENSUS_EDGE, linewidth=2,
                              label=f"Consensus ({args.subset}, threshold {args.threshold})")]
            fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06), frameon=False)
            fig.suptitle(f"{r.acronym} — {r.name}   (scores are 3D, whole region; slice = max GT area)",
                         fontsize=11)
            fig.savefig(out_dir / f"gallery_{r.acronym}_{tag}.png", dpi=150, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"  {r.acronym}", flush=True)

    print(f"Wrote {len(regions)} PNGs + {pdf_path}")


if __name__ == "__main__":
    main()
