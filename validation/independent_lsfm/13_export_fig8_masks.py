# -*- coding: utf-8 -*-
"""
13_export_fig8_masks.py

Export, for every brain and every region of manuscript Fig. 8 (revision Fig. 4), the two 3-D masks that were compared:
  * our pipeline consensus: raw input, final QC set (global QC + SSIM top 50 %), threshold 0.5
    (results/region_consensus/<s>/raw/<ACR>_both/passes_final_qc_top_0p5/consensus_thresh_0p5.tif)
  * the published (independent) registration: the region extracted from the dataset authors' registered atlas,
    rank labels converted to Allen IDs, all descendant labels included (gt_utils.gt_region_mask), both hemispheres.

Both masks are full volumes in the brain's own downsampled space (25 um isotropic, coronal), voxel-aligned with each
other and with <s>_AF_level1_downsampled_25um_coronal_flipped.tif. Written as uncompressed uint8 TIFF (0/1).
The Dice of each pair is recomputed and checked against results/validation_gt/consensus_vs_gt.csv.
Read-only on all inputs.

Output: results/fig8_region_masks/<sample>/<ACR>__consensus.tif, <ACR>__published_registration.tif,
        results/fig8_region_masks/manifest.csv, README.md
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tifffile

from gt_utils import (ROOT, FINAL_QC_SUBSET, VOXEL_UM, region_table, load_label_mapping, load_gt_atlas, gt_region_mask,
                      consensus_path, load_mask, gt_path, image_path, n_registrations, seg_metrics)

OUT = ROOT / "results" / "fig8_region_masks"
THR = 0.5
INPUT = "raw"


def export(sample):
    _, agg = load_label_mapping()
    gt = load_gt_atlas(sample)
    d = OUT / sample
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in region_table().itertuples():
        src = consensus_path(sample, INPUT, r.acronym, FINAL_QC_SUBSET, THR)
        cons = load_mask(src, gt.shape)
        ref = gt_region_mask(gt, r.acronym, agg)
        fc, fr = d / f"{r.acronym}__consensus.tif", d / f"{r.acronym}__published_registration.tif"
        for f, m in ((fc, cons), (fr, ref)):
            tifffile.imwrite(f, m.astype(np.uint8), compression=None, metadata={"axes": "ZYX"})
        back_c, back_r = tifffile.imread(fc) != 0, tifffile.imread(fr) != 0
        assert np.array_equal(back_c, cons) and np.array_equal(back_r, ref), f"write check failed: {sample} {r.acronym}"
        m = seg_metrics(cons, ref)
        rows.append(dict(sample_id=sample, region=r.acronym, input_type=INPUT, subset=FINAL_QC_SUBSET, threshold=THR,
                         n_registrations=n_registrations(sample, INPUT, r.acronym, FINAL_QC_SUBSET),
                         shape_zyx="x".join(map(str, gt.shape)), voxel_um=VOXEL_UM,
                         consensus_voxels=int(cons.sum()), published_voxels=int(ref.sum()),
                         dice_3d=m["dice"], vol_ratio=m["vol_ratio"], hd95_um=m["hd95_um"], assd_um=m["assd_um"],
                         consensus_file=str(fc.relative_to(OUT)), published_file=str(fr.relative_to(OUT)),
                         consensus_source=str(src), published_source=str(gt_path(sample)),
                         image_same_grid=str(image_path(sample))))
    print(f"done: {sample}", flush=True)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    samples = sorted(pd.read_csv(ROOT / "config" / "samples_raw.csv").sample_id.astype(str).unique())
    with ProcessPoolExecutor(max_workers=6) as ex:
        man = pd.DataFrame([row for rows in ex.map(export, samples) for row in rows])
    # check against the table behind Fig. 8
    ref = pd.read_csv(ROOT / "results" / "validation_gt" / "consensus_vs_gt.csv")
    ref = ref[(ref.input_type == INPUT) & (ref.subset == FINAL_QC_SUBSET) & (ref.threshold == THR)]
    chk = man.merge(ref[["sample_id", "region", "dice"]], on=["sample_id", "region"], how="left")
    man["dice_in_consensus_vs_gt_csv"] = chk["dice"].to_numpy()
    man["dice_matches"] = np.isclose(man.dice_3d, man.dice_in_consensus_vs_gt_csv, atol=1e-9)
    man.to_csv(OUT / "manifest.csv", index=False)
    med = man.groupby("region").dice_3d.median().sort_values(ascending=False)
    print(f"{len(man)} pairs; Dice matches consensus_vs_gt.csv: {int(man.dice_matches.sum())}/{len(man)}")
    print(med.round(3).to_string())
    (OUT / "README.md").write_text(README.format(n=len(man), ok=int(man.dice_matches.sum()), shape=man.shape_zyx.iloc[0],
                                                 regions=", ".join(region_table().acronym), samples=", ".join(samples)),
                                   encoding="utf-8")


README = """# Region masks behind manuscript Fig. 8 (revision Fig. 4)

For each brain ({samples}) and each of the 16 regions ({regions}), the two 3-D masks that were compared:

| File | Content |
|---|---|
| `<sample>/<REGION>__consensus.tif` | Our pipeline consensus: raw input, final QC set (global QC IoU > 0.8 and grad_p95 < 0.08, then SSIM top 50 %), consensus threshold 0.5, both hemispheres |
| `<sample>/<REGION>__published_registration.tif` | The same region from the dataset authors' own registered atlas (the independent registration; not a manual annotation). Rank labels converted to Allen structure IDs; all descendant labels included; both hemispheres |

* Full volumes, shape (z, y, x) = {shape}, 25 um isotropic, coronal, in each brain's own downsampled image space.
  The two masks of a pair are voxel-aligned with each other and with `<sample>_AF_level1_downsampled_25um_coronal_flipped.tif`.
* uint8, 0 = outside, 1 = inside; uncompressed TIFF.
* `manifest.csv` lists every pair with voxel counts, the 3-D Dice, volume ratio, HD95 and mean surface distance,
  the source files, and a check that the Dice equals the value in `results/validation_gt/consensus_vs_gt.csv`
  ({ok}/{n} match).
* Written by `scripts/validation/13_export_fig8_masks.py` (read-only on all inputs). The SNc published masks per
  hemisphere are also in `results/snc_published_registration/`.
"""

if __name__ == "__main__":
    main()
