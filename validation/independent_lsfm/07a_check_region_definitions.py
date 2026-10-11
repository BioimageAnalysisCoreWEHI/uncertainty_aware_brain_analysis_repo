# -*- coding: utf-8 -*-
"""
07a_check_region_definitions.py

Sanity gate for the GT validation -- run BEFORE 07b/07c/07d.

For each of the 16 regions of interest this checks that "region X" means
the same set of voxels in the pipeline and in the GT atlas:

1. ID sets. The pipeline builds a region mask as np.isin(atlas,
   get_descendant_ids(region_id)) against the ARA2 tree
   (build_region_consensus_wholebrain.py). The GT mask is built from the
   aggregate registration-value table. Here both are expressed as Allen
   IDs and compared, restricted to IDs that can actually carry voxels:
     - pipeline_only_painted: in the pipeline set AND painted in the
       Allen 25 um annotation brainreg used, but not in the GT region set.
       Voxels with these IDs would count as "region" for us, not for GT.
     - gt_only: in the GT region set but not the pipeline set.
   Both should be empty.

2. Volumes. Region volume in the Allen 25 um annotation vs in each GT
   brain, raw and normalised by whole-brain labelled volume. A consistent
   ratio far from 1 across all brains points to an atlas-version /
   definition difference (not a registration error) and must be kept in
   mind when reading Dice for that region.

Outputs (results/validation_gt/):
    region_definition_check.csv
    region_volume_check.csv
"""

import numpy as np
import pandas as pd
import tifffile as tiff

from gt_utils import (
    ROOT, CFG, OUT_DIR, SAMPLES, VOXEL_MM3,
    region_table, load_label_mapping, gt_region_values, load_gt_atlas,
    load_region_tree, get_descendant_ids,
)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    regions = region_table()
    tree = load_region_tree(ROOT / CFG["roi"]["region_tree_csv"])
    lut, agg = load_label_mapping()
    value_to_allen = dict(zip(lut["registration_value"], lut["allen_id"]))
    gt_label_ids = set(lut["allen_id"].astype(int))

    annotation = tiff.imread(ROOT / CFG["roi"]["annotation_image"])
    ann_ids, ann_counts = np.unique(annotation, return_counts=True)
    ann_count = dict(zip(ann_ids.astype(int), ann_counts.astype(int)))
    ann_brain_voxels = int(ann_counts[ann_ids != 0].sum())

    def_rows = []
    for r in regions.itertuples():
        pipeline_ids = get_descendant_ids(r.region_id, tree, include_self=True)
        gt_ids = {int(value_to_allen[v]) for v in gt_region_values(r.acronym, agg)}
        painted = pipeline_ids & set(ann_count)

        pipeline_only_painted = sorted(painted - gt_ids)
        gt_only = sorted(gt_ids - pipeline_ids)
        def_rows.append({
            "region_id": r.region_id,
            "acronym": r.acronym,
            "n_pipeline_ids": len(pipeline_ids),
            "n_pipeline_ids_painted_allen25": len(painted),
            "n_gt_ids": len(gt_ids),
            "pipeline_only_painted": ";".join(map(str, pipeline_only_painted)),
            "pipeline_only_painted_voxels": sum(ann_count[i] for i in pipeline_only_painted),
            "gt_only": ";".join(map(str, gt_only)),
            "gt_only_in_gt_label_set": ";".join(str(i) for i in gt_only if i in gt_label_ids),
            "_gt_only_values": [v for v in gt_region_values(r.acronym, agg)
                                if int(value_to_allen[v]) in gt_only],
            "gt_only_voxels_all_gt_brains": 0,
            "definitions_match": not pipeline_only_painted and not gt_only,
        })

    vol_rows = []
    for r in regions.itertuples():
        ids = get_descendant_ids(r.region_id, tree, include_self=True)
        n_ann = sum(ann_count.get(i, 0) for i in ids)
        vol_rows.append({
            "sample_id": "allen25_annotation", "acronym": r.acronym,
            "vol_mm3": n_ann * VOXEL_MM3, "brain_fraction": n_ann / ann_brain_voxels,
        })

    for sample_id in SAMPLES:
        gt = load_gt_atlas(sample_id)
        counts = np.bincount(gt.ravel(), minlength=673)
        brain_voxels = int(counts[1:].sum())
        for r in regions.itertuples():
            n = int(counts[gt_region_values(r.acronym, agg)].sum())
            vol_rows.append({
                "sample_id": sample_id, "acronym": r.acronym,
                "vol_mm3": n * VOXEL_MM3, "brain_fraction": n / brain_voxels,
            })
        for row in def_rows:
            row["gt_only_voxels_all_gt_brains"] += int(counts[row["_gt_only_values"]].sum())
        del gt

    # IDs only in the GT set are harmless if they carry no GT voxels
    # (e.g. RSPd4 / Allen 545: a GT label absent from the ARA2 tree and
    # empty in the GT atlases -- see registration_label_mapping/README.md).
    def_df = pd.DataFrame(def_rows).drop(columns="_gt_only_values")
    def_df["definitions_match_effective"] = (
        (def_df["pipeline_only_painted"] == "") & (def_df["gt_only_voxels_all_gt_brains"] == 0)
    )
    def_df.to_csv(OUT_DIR / "region_definition_check.csv", index=False)
    print(def_df[["acronym", "n_pipeline_ids_painted_allen25", "n_gt_ids", "pipeline_only_painted",
                  "gt_only", "gt_only_voxels_all_gt_brains", "definitions_match_effective"]].to_string(index=False))

    vol = pd.DataFrame(vol_rows)
    ref = vol[vol.sample_id == "allen25_annotation"].set_index("acronym")
    vol["vol_ratio_vs_allen25"] = vol["vol_mm3"] / vol["acronym"].map(ref["vol_mm3"])
    vol["fraction_ratio_vs_allen25"] = vol["brain_fraction"] / vol["acronym"].map(ref["brain_fraction"])
    vol.to_csv(OUT_DIR / "region_volume_check.csv", index=False)

    summary = (
        vol[vol.sample_id != "allen25_annotation"]
        .groupby("acronym", sort=False)["fraction_ratio_vs_allen25"]
        .agg(["median", "min", "max"])
        .round(3)
    )
    print("\nGT brain-fraction / Allen25 brain-fraction, across the 6 brains:")
    print(summary.to_string())

    if not def_df["definitions_match_effective"].all():
        print("\nWARNING: region definitions differ (with voxels) for: "
              + ", ".join(def_df.loc[~def_df.definitions_match_effective, "acronym"]))
    else:
        print("\nRegion definitions agree for all regions (up to empty GT-only labels).")


if __name__ == "__main__":
    main()
