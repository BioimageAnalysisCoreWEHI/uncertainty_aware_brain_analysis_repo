# -*- coding: utf-8 -*-
"""
10_extract_snc_reference.py

Extract the SNc label from each published ("ground truth") registered atlas of the independent LSFM dataset, exactly
as used for the Fig. 8 / consensus_vs_gt.csv scores (gt_utils.gt_region_mask: every registration value at or below
Allen SNc), and save it as a binary mask on the same 25 um coronal-flipped grid as the atlases, images and consensus
masks. Also writes per-sample volumes, voxel counts and per-hemisphere centroids.

Output: results/snc_published_registration/<sample>_SNc_published_registration_25um_coronal_flipped.tif (uint8 0/1,
uncompressed), snc_reference_summary.csv, README.md. Inputs are only read.
"""

import numpy as np
import pandas as pd
import tifffile

from gt_utils import ROOT, SAMPLES, VOXEL_MM3, gt_path, gt_region_mask, gt_region_values, load_gt_atlas, load_label_mapping

OUT = ROOT / "results" / "snc_published_registration"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _, agg = load_label_mapping()
    values = gt_region_values("SNc", agg)
    rows = []
    for s in SAMPLES:
        gt = load_gt_atlas(s)
        m = gt_region_mask(gt, "SNc", agg)
        f = OUT / f"{s}_SNc_published_registration_25um_coronal_flipped.tif"
        tifffile.imwrite(f, m.astype(np.uint8))
        z, y, x = np.nonzero(m)
        mid = gt.shape[2] / 2
        row = dict(sample_id=s, file=f.name, shape=str(gt.shape), registration_values=";".join(map(str, values)),
                   n_voxels=int(m.sum()), volume_mm3=float(m.sum() * VOXEL_MM3), source_atlas=gt_path(s).name)
        for side, sel in (("low_x", x < mid), ("high_x", x >= mid)):
            row[f"n_voxels_{side}"] = int(sel.sum())
            row[f"centroid_zyx_{side}"] = (" ".join(f"{v:.1f}" for v in (z[sel].mean(), y[sel].mean(), x[sel].mean()))
                                           if sel.any() else "")
        rows.append(row)
        print(f"{s}: {row['n_voxels']} voxels ({row['volume_mm3']:.3f} mm3) -> {f.name}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "snc_reference_summary.csv", index=False)
    (OUT / "README.md").write_text(
        "# SNc label from the published registrations (independent LSFM dataset)\n\n"
        "One binary mask per brain (uint8 0/1, uncompressed TIFF). Each mask is the SNc of the dataset authors' "
        "registered atlas.\n"
        "- **Grid:** the same 25 um coronal-flipped grid as `results/downsampled_25um_registered_atlas_ground_truth/`, "
        "`results/downsampled_25um_images/` and our consensus masks (`results/region_consensus/`), so they overlay "
        "directly.\n"
        "- **Labels:** every registration value at or below Allen SNc (`registration_label_mapping/`). This is exactly "
        "the reference mask behind the SNc scores in `results/validation_gt/consensus_vs_gt.csv` (Fig. 8).\n"
        "- **Hemispheres:** both are included. `snc_reference_summary.csv` gives voxels and centroids for each half of "
        "the x axis.\n\n"
        "Made by `scripts/validation/10_extract_snc_reference.py`; the source atlases were only read.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
