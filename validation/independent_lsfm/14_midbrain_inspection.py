# -*- coding: utf-8 -*-
"""
14_midbrain_inspection.py

Why do the small midbrain nuclei (SNc, VTA) agree poorly with the published registration, and why better in some
brains (GAD2) than others (TH)? Visual + quantitative inspection of the downsampled 25 um coronal images that the
registrations were run on (raw input). Read-only on all inputs.

Figures (results/midbrain_inspection/):
  whole_images.png   per brain + Allen template: coronal at the SNc level (contrast 0.5-99.5 % of tissue, and the
                     full 0-max range to show saturation), sagittal and horizontal sections through the SNc
  midbrain_zoom.png  per brain: coronal and sagittal zoom on the midbrain; published SNc (white dashed) and VTA (cyan
                     dashed), our consensus SNc (purple), the final-QC single registrations' SNc (thin blue)
Table: midbrain_inspection.csv (image statistics, signal inside each outline, offsets, agreement)
"""
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gt_utils import (ROOT, FINAL_QC_SUBSET, VOXEL_UM, SAMPLES, load_label_mapping, load_gt_atlas, gt_region_mask,
                      image_path, region_dir, load_mask, read_tiff)  # noqa: E402
from pipeline_common import make_output_name  # noqa: E402

OUT = ROOT / "results" / "midbrain_inspection"
MASKS = ROOT / "results" / "fig8_region_masks"
C_REF, C_VTA, C_CONS, C_RUN = "white", "#56B4E9", "#C51B7D", "#4dd0ff"
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.titlesize": 6.5})


def norm(img, lo, hi):
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0, 1)


def contour(ax, m, color, lw=0.7, ls="-"):
    if m.any():
        ax.contour(m.astype(float), levels=[0.5], colors=[color], linewidths=lw, linestyles=ls)


def load_case(s, agg):
    img = read_tiff(image_path(s))
    gt = load_gt_atlas(s)
    snc_ref, vta_ref = gt_region_mask(gt, "SNc", agg), gt_region_mask(gt, "VTA", agg)
    snr_ref = gt_region_mask(gt, "SNr", agg) if "SNr" in str(agg) else None
    snc_cons = tifffile.imread(MASKS / s / "SNc__consensus.tif") != 0
    vta_cons = tifffile.imread(MASKS / s / "VTA__consensus.tif") != 0
    final = pd.read_csv(region_dir(s, "raw", "SNc", FINAL_QC_SUBSET) / "registrations_used.csv")
    all_dir = region_dir(s, "raw", "SNc", "all_registrations") / "individual_masks"
    singles = [load_mask(all_dir / make_output_name(r, prefix="mask"), gt.shape) for r in final.to_dict("records")]
    return img, gt, snc_ref, vta_ref, snc_cons, vta_cons, singles


def stats(s, img, snc_ref, vta_ref, snc_cons, vta_cons, singles):
    tissue = img > np.percentile(img, 50)                   # crude tissue mask (background is the dimmer half)
    t = img[tissue]
    p50, p95, p99, p999 = np.percentile(t, [50, 95, 99, 99.9])
    vmax = img.max()
    mb = ndimage.binary_dilation(snc_ref | vta_ref, iterations=8)   # midbrain neighbourhood (~200 um)
    shell = ndimage.binary_dilation(snc_ref, iterations=4) & ~snc_ref & ~vta_ref
    def m(x):
        return float(img[x].mean() / p50) if x.any() else np.nan
    cen = lambda x: np.array(ndimage.center_of_mass(x)) * VOXEL_UM if x.any() else np.full(3, np.nan)  # noqa: E731
    bright = img >= p99
    # spread of the single registrations' SNc: centroid scatter and pairwise agreement
    cs = np.array([cen(x) for x in singles if x.any()])
    scatter = float(np.sqrt(((cs - cs.mean(0)) ** 2).sum(1)).mean()) if len(cs) > 1 else np.nan
    pair = []
    for i in range(len(singles)):
        for j in range(i + 1, len(singles)):
            a, b = singles[i], singles[j]
            pair.append(2 * (a & b).sum() / max(1, a.sum() + b.sum()))
    off = cen(snc_cons) - cen(snc_ref)
    return dict(sample=s, dtype=str(img.dtype), image_max=int(vmax), tissue_p50=p50, tissue_p99=p99, tissue_p999=p999,
                frac_tissue_at_max=float((t >= vmax).mean()), frac_tissue_ge_0p95max=float((t >= 0.95 * vmax).mean()),
                frac_midbrain_ge_0p95max=float((img[mb] >= 0.95 * vmax).mean()),
                frac_midbrain_ge_tissue_p99=float(bright[mb].mean()),
                rel_signal_published_snc=m(snc_ref), rel_signal_published_vta=m(vta_ref), rel_signal_snc_shell=m(shell),
                rel_signal_consensus_snc=m(snc_cons),
                frac_published_snc_bright=float(bright[snc_ref].mean()), frac_consensus_snc_bright=float(bright[snc_cons].mean()) if snc_cons.any() else np.nan,
                published_snc_vox=int(snc_ref.sum()), consensus_snc_vox=int(snc_cons.sum()),
                consensus_minus_published_centroid_um_zyx=";".join(f"{v:.0f}" for v in off),
                consensus_to_published_centroid_um=float(np.linalg.norm(off)),
                n_final=len(singles), singles_centroid_scatter_um=scatter,
                singles_pairwise_dice_median=float(np.median(pair)) if pair else np.nan)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, agg = load_label_mapping()
    cases = {s: load_case(s, agg) for s in SAMPLES}
    rows = [stats(s, *[c[0], c[2], c[3], c[4], c[5], c[6]]) for s, c in cases.items()]
    st = pd.DataFrame(rows)
    gt_scores = pd.read_csv(ROOT / "results" / "validation_gt" / "consensus_vs_gt.csv")
    g = gt_scores[(gt_scores.input_type == "raw") & (gt_scores.subset == FINAL_QC_SUBSET) & (gt_scores.threshold == 0.5)]
    for reg in ("SNc", "VTA", "CP"):
        st[f"dice_{reg}"] = st["sample"].map(g[g.region == reg].set_index("sample_id").dice)
    ind = pd.read_csv(ROOT / "results" / "validation_gt" / "snc_cp_individual_vs_gt.csv")
    ind = ind[(ind.input_type == "raw") & (ind.region == "SNc") & ind.in_final_qc]
    st["snc_best_single_dice"] = st["sample"].map(ind.groupby("sample_id").dice.max())
    st["snc_median_single_dice"] = st["sample"].map(ind.groupby("sample_id").dice.median())
    st.to_csv(OUT / "midbrain_inspection.csv", index=False)
    pd.set_option("display.width", 250)
    print(st.T.to_string())

    # ---- whole images
    atlas = tifffile.imread(ROOT / "config" / "allen_mouse_25um_reference_coronal.tif")
    ann = tifffile.imread(ROOT / "config" / "allen_mouse_25um_reference_coronal_annotations.tif")
    rows_ = list(SAMPLES) + ["Allen template"]
    fig, axs = plt.subplots(len(rows_), 4, figsize=(7.2, 1.45 * len(rows_)),
                            gridspec_kw=dict(width_ratios=[1.2, 1.2, 1.6, 1.0], wspace=0.04, hspace=0.12))
    for i, s in enumerate(rows_):
        if s == "Allen template":
            img = atlas
            snc = ann == 374
            vta = ann == 749
        else:
            img, _, snc, vta = cases[s][0], None, cases[s][2], cases[s][3]
        tissue = img[img > np.percentile(img, 50)]
        lo, hi = np.percentile(tissue, [0.5, 99.5])
        z = int(np.argmax(snc.sum((1, 2))))
        zz, yy, xx = [int(v) for v in ndimage.center_of_mass(snc)]
        x_side = int(np.median(np.nonzero(snc[z])[1][np.nonzero(snc[z])[1] < img.shape[2] // 2])) \
            if (np.nonzero(snc[z])[1] < img.shape[2] // 2).any() else xx
        views = [(norm(img[z], lo, hi), snc[z], vta[z], f"coronal z={z} (0.5–99.5 %)"),
                 (norm(img[z], 0, img.max()), snc[z], vta[z], "same, full 0–max range"),
                 (norm(img[:, :, x_side].T, lo, hi), snc[:, :, x_side].T, vta[:, :, x_side].T, f"sagittal x={x_side}"),
                 (norm(img[:, yy, :], lo, hi), snc[:, yy, :], vta[:, yy, :], f"horizontal y={yy}")]
        for j, (im, a, b, t) in enumerate(views):
            ax = axs[i, j]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            contour(ax, a, C_REF, 0.5, "--")
            contour(ax, b, C_VTA, 0.5, "--")
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(t.split(" (")[0] if j != 1 else t, fontsize=6, color="white")
            if j == 0:
                ax.set_ylabel(s, fontsize=6.5, color="white")
            if j == 1:
                fr = (img >= 0.95 * img.max()).mean() * 100
                ax.text(0.02, 0.03, f"max {int(img.max())}", color="yellow", fontsize=5, transform=ax.transAxes)
    fig.suptitle("Downsampled 25 µm images used for registration (raw), with the published SNc (white) and VTA (cyan)",
                 fontsize=7, y=0.995, color="white")
    fig.savefig(OUT / "whole_images.png", dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    # ---- midbrain zoom
    fig, axs = plt.subplots(len(SAMPLES), 3, figsize=(7.2, 1.9 * len(SAMPLES)),
                            gridspec_kw=dict(wspace=0.04, hspace=0.18))
    for i, s in enumerate(SAMPLES):
        img, gt, snc, vta, sncc, vtac, singles = cases[s]
        tissue = img[img > np.percentile(img, 50)]
        lo, hi = np.percentile(tissue, [0.5, 99.5])
        z = int(np.argmax(snc.sum((1, 2))))
        ys, xs = np.nonzero(snc[z] | vta[z] | sncc[z])
        y0, y1 = max(0, ys.min() - 25), min(img.shape[1], ys.max() + 25)
        x0, x1 = max(0, xs.min() - 25), min(img.shape[2], xs.max() + 25)
        left = xs < img.shape[2] // 2
        xl = int(np.median(np.nonzero(snc[z])[1][np.nonzero(snc[z])[1] < img.shape[2] // 2]))
        zs = np.nonzero(snc.any((1, 2)) | sncc.any((1, 2)))[0]
        za, zb = max(0, zs.min() - 25), min(img.shape[0], zs.max() + 25)
        r = st.set_index("sample").loc[s]
        panels = [
            ("coronal, display 0.5–99.5 %", norm(img[z, y0:y1, x0:x1], lo, hi),
             [m[z, y0:y1, x0:x1] for m in singles], snc[z, y0:y1, x0:x1], vta[z, y0:y1, x0:x1], sncc[z, y0:y1, x0:x1]),
            ("coronal, full 0–max range", norm(img[z, y0:y1, x0:x1], 0, img.max()),
             [], snc[z, y0:y1, x0:x1], vta[z, y0:y1, x0:x1], sncc[z, y0:y1, x0:x1]),
            ("sagittal through left SNc", norm(img[za:zb, y0:y1, xl].T, lo, hi),
             [m[za:zb, y0:y1, xl].T for m in singles], snc[za:zb, y0:y1, xl].T, vta[za:zb, y0:y1, xl].T,
             sncc[za:zb, y0:y1, xl].T)]
        for j, (t, im, sg, a, b, c) in enumerate(panels):
            ax = axs[i, j]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            for m in sg:
                contour(ax, m, C_RUN, 0.3)
            contour(ax, b, C_VTA, 0.8, "--")
            contour(ax, a, C_REF, 0.8, "--")
            contour(ax, c, C_CONS, 1.0)
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(t, fontsize=6, color="white")
            if j == 0:
                ax.set_ylabel(s, fontsize=6.5, color="white")
                ax.text(0.02, 0.03, f"SNc Dice {r.dice_SNc:.2f}  VTA {r.dice_VTA:.2f}", color="yellow", fontsize=5,
                        transform=ax.transAxes)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=C_REF, ls="--", label="published SNc"), Line2D([], [], color=C_VTA, ls="--",
                                                                             label="published VTA"),
         Line2D([], [], color=C_CONS, label="our consensus SNc"), Line2D([], [], color=C_RUN, lw=0.5,
                                                                        label="final-QC single registrations (SNc)")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=6, frameon=False, labelcolor="white",
               bbox_to_anchor=(0.5, -0.005))
    fig.savefig(OUT / "midbrain_zoom.png", dpi=300, bbox_inches="tight", facecolor="black")
    plt.close(fig)


if __name__ == "__main__":
    main()
