# -*- coding: utf-8 -*-
"""
16_export_snc_ssim_boxes.py

For every independent-LSFM brain (raw input), save the SNc SSIM bounding-box crops that local SSIM compares:
  * the registered image (downsampled_standard, i.e. the brain's image warped into Allen 25 um atlas space) of the
    retained (final-QC) registration with the highest SNc SSIM -- raw intensities and min-max normalised over the box
  * the Allen 25 um reference template cropped to the same box -- raw and min-max normalised
  * the atlas SNc mask (incl. descendants) within the box
Crop and normalisation are exactly region_utils.crop_bbox / minmax_norm as used by region_utils.compute_local_ssim.
SSIM is recomputed from the saved normalised crops and checked against the stored QC value.
Uncompressed 3-D TIFFs (z, y, x). Read-only on all inputs.

Output: results/snc_ssim_boxes/<sample>/..., manifest.csv, README.md
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)
import sys

import numpy as np
import pandas as pd
import tifffile
from skimage.metrics import structural_similarity as ssim

from gt_utils import ROOT, CFG, SAMPLES

sys.path.insert(0, str(ROOT / "scripts"))
from region_utils import load_region_tree, build_region_lookup, crop_bbox, minmax_norm, get_descendant_ids  # noqa: E402

OUT = ROOT / "results" / "snc_ssim_boxes"
DIAG = ROOT / "results" / "validation_gt" / "snc_cp_individual_vs_gt.csv"
REGION = "SNc"


def unc(p):
    return to_local(p)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tree = load_region_tree(ROOT / CFG["roi"]["region_tree_csv"])
    ann = tifffile.imread(ROOT / CFG["roi"]["annotation_image"])
    lookup = build_region_lookup(CFG["roi"]["region_ids"], tree, ann, padding=CFG["roi"].get("bbox_padding_voxels", 5))
    rid = next(k for k, v in lookup.items() if v["acronym"] == REGION)
    bb = lookup[rid]["hemispheres"]["both"]["bbox"]
    atlas = tifffile.imread(ROOT / CFG["atlas"]["reference_image"]).astype(np.float32)
    a_raw = crop_bbox(atlas, bb)
    a_norm = minmax_norm(a_raw)
    mask = crop_bbox(np.isin(ann, list(get_descendant_ids(rid, tree, include_self=True))), bb).astype(np.uint8)
    for name, arr in (("atlas_template__SNc_box.tif", a_raw), ("atlas_template__SNc_box__minmax_norm.tif", a_norm),
                      ("atlas_SNc_mask__SNc_box.tif", mask)):
        tifffile.imwrite(OUT / name, arr, compression=None, metadata={"axes": "ZYX"})

    q = pd.read_csv(ROOT / "results" / "qc_tables" / "qc_results.csv")
    q["run"] = q.registration_dir.astype(str).str.rstrip("/").str.split("/").str[-1]
    d = pd.read_csv(DIAG)
    d = d[(d.input_type == "raw") & (d.region == REGION) & d.in_final_qc]
    rows = []
    for s in SAMPLES:
        top = d[d.sample_id == s].sort_values("qc_local_ssim", ascending=False).iloc[0]
        run = top.mask_name.replace("mask__", "").replace(".tif", "")
        row = q[(q.sample_id == s) & (q.input_type == "raw") & (q.run == run)].iloc[0]
        img = tifffile.imread(unc(row.downsampled_standard_path)).astype(np.float32)
        if img.shape != atlas.shape:
            raise ValueError(f"{s}: downsampled_standard shape {img.shape} != atlas {atlas.shape}")
        s_raw = crop_bbox(img, bb)
        s_norm = minmax_norm(s_raw)
        dd = OUT / s
        dd.mkdir(exist_ok=True)
        tifffile.imwrite(dd / f"{s}__{run}__SNc_box.tif", s_raw, compression=None, metadata={"axes": "ZYX"})
        tifffile.imwrite(dd / f"{s}__{run}__SNc_box__minmax_norm.tif", s_norm, compression=None,
                         metadata={"axes": "ZYX"})
        recomputed = float(ssim(a_norm, s_norm, data_range=1.0))
        stored = float(row[f"local_ssim__{REGION}_both"])
        p95 = np.percentile(img[img > 0], 95)
        rows.append(dict(sample_id=s, input_type="raw", registration=run,
                         bending_energy=top.bending_energy_weight, grid_spacing=top.grid_spacing,
                         rank_note="highest SNc SSIM among final-QC (retained) registrations",
                         n_retained=int((d.sample_id == s).sum()), snc_ssim_stored=stored,
                         snc_ssim_recomputed_from_saved_crops=recomputed,
                         ssim_matches=bool(np.isclose(stored, recomputed, atol=1e-6)),
                         snc_dice_vs_published=float(top.dice),
                         box_raw_min=float(s_raw.min()), box_raw_max=float(s_raw.max()),
                         box_frac_voxels_above_brain_p95=float((s_raw > p95).mean()),
                         box_frac_voxels_below_0p1_after_norm=float((s_norm < 0.1).mean()),
                         atlas_frac_voxels_below_0p1_after_norm=float((a_norm < 0.1).mean()),
                         source_downsampled_standard=str(row.downsampled_standard_path),
                         files=f"{s}/{s}__{run}__SNc_box.tif; {s}/{s}__{run}__SNc_box__minmax_norm.tif"))
        print(f"{s}: {run}  SSIM stored {stored:.4f} recomputed {recomputed:.4f}", flush=True)
    man = pd.DataFrame(rows)
    man.to_csv(OUT / "manifest.csv", index=False)
    (OUT / "README.md").write_text(README.format(bbox=bb, shape=a_raw.shape, ok=int(man.ssim_matches.sum()),
                                                 n=len(man)), encoding="utf-8")
    print(man[["sample_id", "registration", "snc_ssim_stored", "snc_ssim_recomputed_from_saved_crops",
               "box_frac_voxels_below_0p1_after_norm", "atlas_frac_voxels_below_0p1_after_norm"]].round(3)
          .to_string(index=False))


README = """# SNc SSIM boxes (independent LSFM, raw input)

The 3-D crops that local SSIM compares for the SNc, exactly as the pipeline computes it
(`region_utils.compute_local_ssim`: crop to the region's bounding box, min-max normalise each crop over the whole box,
`skimage.metrics.structural_similarity(..., data_range=1.0)`).

Bounding box (atlas space, both hemispheres, with the configured padding): {bbox}; crop shape (z, y, x) = {shape};
Allen 25 um atlas space, coronal.

| File | Content |
|---|---|
| `atlas_template__SNc_box.tif` | Allen 25 um reference template, cropped to the SNc box (raw intensities) |
| `atlas_template__SNc_box__minmax_norm.tif` | the same, min-max normalised over the box (what SSIM sees) |
| `atlas_SNc_mask__SNc_box.tif` | atlas SNc mask (uint8 0/1) within the box |
| `<sample>/<sample>__<registration>__SNc_box.tif` | the brain's registered image (`downsampled_standard`, marker channel warped into atlas space) of the retained registration with the highest SNc SSIM, raw intensities |
| `<sample>/<sample>__<registration>__SNc_box__minmax_norm.tif` | the same, min-max normalised over the box (what SSIM sees) |

`manifest.csv`: per brain the registration, its stored SNc SSIM and the SSIM recomputed from the saved normalised crops
({ok}/{n} match), its SNc Dice with the published registration, intensity range, the fraction of box voxels above the
brain-wide 95th percentile, and the fraction of voxels below 0.1 after normalisation (brain vs template).
Written by `scripts/validation/16_export_snc_ssim_boxes.py`; uncompressed float32 TIFF. Read-only on all inputs.
"""

if __name__ == "__main__":
    main()
