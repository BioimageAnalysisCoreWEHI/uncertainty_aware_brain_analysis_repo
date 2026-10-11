# -*- coding: utf-8 -*-
"""
12_snc_ssim_figure.py

Independent LSFM dataset: why local SSIM cannot select good SNc registrations here.
  A  SNc on the marker image (sample space, coronal section with the largest published SNc area): published
     registration (white dashed), the four QC-passing registrations ranked highest by SNc SSIM (thin blue) and the
     0.5 consensus (purple).
  B  What SSIM compares for the SNc: the atlas reference crop and the registered image crop (downsampled_standard,
     atlas space) inside the SNc bounding box, mid-section, each min-max normalised exactly as in
     region_utils.compute_local_ssim. SNc atlas outline in white.
  C  The same for the region whose SSIM box contains the least bright marker signal (chosen from the data: lowest
     median fraction of box voxels above the brain-wide 95th percentile).
Raw input, final-QC set. Read-only on all inputs.
Output: REVISION_OUTPUTS_FOCUSED/figures/FigS6_independent_LSFM_SNc_SSIM.{pdf,svg,png} + source_data/FigS6_*.csv
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from gt_utils import (ROOT, CFG, FINAL_QC_SUBSET, load_label_mapping, load_gt_atlas, gt_region_mask, image_path,
                      region_dir, load_mask, read_tiff)  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from region_utils import load_region_tree, build_region_lookup, crop_bbox, minmax_norm  # noqa: E402
from pipeline_common import make_output_name  # noqa: E402

PRA = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis"))
FIG, SRC = PRA / "REVISION_OUTPUTS_FOCUSED" / "figures", PRA / "REVISION_OUTPUTS_FOCUSED" / "source_data"
BRAINS = ["TH", "Rep2-TH", "Rep2-vGlut2", "GAD2"]
C_REF, C_RUN, C_CONS = "white", "#56B4E9", "#7B3294"
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.titlesize": 6.5, "pdf.fonttype": 42,
                     "svg.fonttype": "none"})
DIAG = ROOT / "results" / "validation_gt" / "snc_cp_individual_vs_gt.csv"


def unc(p):
    return Path(to_local(p))


def outline(ax, m, color, lw, ls="-", alpha=1.0):
    if m.any():
        ax.contour(m.astype(float), [0.5], colors=[color], linewidths=lw, linestyles=[ls], alpha=alpha)


def clean(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    _, agg = load_label_mapping()
    q = pd.read_csv(ROOT / "results" / "qc_tables" / "qc_results.csv")
    diag = pd.read_csv(DIAG)
    tree = load_region_tree(ROOT / CFG["roi"]["region_tree_csv"])
    ann = tifffile.imread(ROOT / CFG["roi"]["annotation_image"])
    lookup = build_region_lookup(CFG["roi"]["region_ids"], tree, ann, padding=CFG["roi"].get("bbox_padding_voxels", 5))
    atlas = tifffile.imread(ROOT / CFG["atlas"]["reference_image"]).astype(np.float32)
    acr = {v["acronym"]: k for k, v in lookup.items()}
    bbox = {a: lookup[k]["hemispheres"]["both"]["bbox"] for a, k in acr.items()}

    # ---- per brain: registrations to show (final QC, highest SNc SSIM) and the registered image of the top one
    picks, std = {}, {}
    for b in BRAINS:
        d = diag[(diag.sample_id == b) & (diag.input_type == "raw") & (diag.region == "SNc") & diag.in_final_qc]
        d = d.sort_values("qc_local_ssim", ascending=False).head(4)
        picks[b] = d
        top = d.iloc[0]
        row = q[(q.sample_id == b) & (q.input_type == "raw") &
                (q.registration_dir.str.rstrip("/").str.split("/").str[-1] ==
                 top.mask_name.replace("mask__", "").replace(".tif", ""))].iloc[0]
        std[b] = (tifffile.imread(unc(row.downsampled_standard_path)).astype(np.float32), row)
    # ---- low-signal region: fraction of box voxels above the brain-wide 95th percentile (median over brains)
    frac = []
    for b in BRAINS:
        img = std[b][0]
        p95 = np.percentile(img[img > 0], 95)
        for a, bb in bbox.items():
            frac.append(dict(sample_id=b, region=a, frac_bright=float((crop_bbox(img, bb) > p95).mean())))
    frac = pd.DataFrame(frac)
    rank = frac.groupby("region").frac_bright.median().sort_values()
    low = next(r for r in rank.index if r not in ("SNc", "VTA", "STN", "DG-sg"))
    frac.to_csv(SRC / "FigS6_bright_signal_fraction_by_region.csv", index=False)
    print("bright-signal fraction (median over brains):", rank.round(3).to_dict(), "-> low-signal region", low)

    fig = plt.figure(figsize=(7.2, 3.0))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.3, 1, 1], hspace=0.28, wspace=0.06)
    src = []
    for j, b in enumerate(BRAINS):
        gt = load_gt_atlas(b)
        ref = gt_region_mask(gt, "SNc", agg)
        img = read_tiff(image_path(b)).astype(float)
        cons = load_mask(region_dir(b, "raw", "SNc", FINAL_QC_SUBSET) / "consensus_thresh_0p5.tif", gt.shape)
        all_dir = region_dir(b, "raw", "SNc", "all_registrations") / "individual_masks"
        runs = [load_mask(all_dir / n, gt.shape) for n in picks[b].mask_name]
        z = int(np.argmax(ref.sum((1, 2))))
        u = ref[z] | cons[z] | np.logical_or.reduce([r[z] for r in runs])
        rr, cc = np.flatnonzero(u.any(1)), np.flatnonzero(u.any(0))
        sl = (slice(max(rr[0] - 12, 0), rr[-1] + 13), slice(max(cc[0] - 12, 0), cc[-1] + 13))
        im = img[z][sl]
        lo, hi = np.percentile(im, [1, 99.5])
        ax = fig.add_subplot(gs[0, j + 1])
        ax.imshow(np.clip((im - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray", interpolation="nearest")
        for r in runs:
            outline(ax, r[z][sl], C_RUN, 0.6, alpha=0.9)
        outline(ax, ref[z][sl], C_REF, 1.0, "--")
        outline(ax, cons[z][sl], C_CONS, 1.3)
        clean(ax)
        ax.plot([4, 24], [im.shape[0] - 4] * 2, color="white", lw=1.5)        # 20 px x 25 um = 500 um
        dd = picks[b].dice
        ax.set_title(f"{b}\nsingle SNc Dice {dd.min():.2f}–{dd.max():.2f}", fontsize=6.2)
        for k, r in enumerate(picks[b].itertuples()):
            src.append(dict(panel="A", sample_id=b, slice_z=z, registration=r.mask_name, snc_ssim=r.qc_local_ssim,
                            snc_dice=r.dice))
        # ---- B / C: what SSIM sees (atlas space, mid-section of the box)
        simg, row = std[b]
        for i, a in ((1, "SNc"), (2, low)):
            bb = bbox[a]
            rmask = crop_bbox(np.isin(ann, list(_desc(tree, acr[a]))), bb)
            k = int(np.argmax(rmask.sum((1, 2))))                         # section with the most of the region
            crop = minmax_norm(crop_bbox(simg, bb))[k]                    # normalised over the 3-D box, as SSIM
            axb = fig.add_subplot(gs[i, j + 1])
            axb.imshow(crop, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            outline(axb, rmask[k], C_REF, 0.8, "--")
            clean(axb)
            s = row[f"local_ssim__{a}_both"]
            axb.text(0.03, 0.97, f"SSIM {s:.2f}", transform=axb.transAxes, va="top", ha="left", color="white",
                     fontsize=5.8, bbox=dict(facecolor="black", alpha=0.6, pad=1, lw=0))
            src.append(dict(panel="B" if i == 1 else "C", sample_id=b, region=a, registration=row.registration_dir,
                            ssim=s, bbox=str(bb)))
    for i, a in ((1, "SNc"), (2, low)):
        bb = bbox[a]
        rmask = crop_bbox(np.isin(ann, list(_desc(tree, acr[a]))), bb)
        k = int(np.argmax(rmask.sum((1, 2))))
        axa = fig.add_subplot(gs[i, 0])
        axa.imshow(minmax_norm(crop_bbox(atlas, bb))[k], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        outline(axa, rmask[k], C_REF, 0.8, "--")
        clean(axa)
        axa.set_title(f"Atlas template\n{a} SSIM box", fontsize=6.2)
        axa.text(-0.12, 0.5, ["", "B", "C"][i], transform=axa.transAxes, fontsize=9, fontweight="bold", va="center")
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.text(-0.12, 0.92, "A", transform=ax0.transAxes, fontsize=9, fontweight="bold")
    ax0.legend(handles=[Line2D([], [], color="0.5", lw=1.0, ls="--", label="published SNc (white)"),
                        Line2D([], [], color=C_RUN, lw=0.8, label="QC-passing registrations\n(top 4 by SNc SSIM)"),
                        Line2D([], [], color=C_CONS, lw=1.3, label="pipeline consensus")],
               loc="center", fontsize=6, frameon=False)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"FigS6_independent_LSFM_SNc_SSIM.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.DataFrame(src).to_csv(SRC / "FigS6_panels.csv", index=False)
    print("written")


def _desc(tree, rid):
    from region_utils import get_descendant_ids
    return get_descendant_ids(rid, tree, include_self=True)


if __name__ == "__main__":
    main()
