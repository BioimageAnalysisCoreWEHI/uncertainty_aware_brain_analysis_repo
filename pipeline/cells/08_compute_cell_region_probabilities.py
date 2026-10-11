# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:57:09 2026

@author: zaman.i
"""

from pathlib import Path
import argparse
import traceback
from collections import Counter

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.measure import regionprops


REGIONS_OF_INTEREST = ["SNc", "SNr", "VTA", "MRN", "RR", "MB"]


def load_ontology(ontology_csv):
    ontology = pd.read_csv(ontology_csv)
    id_to_name = dict(zip(ontology["id"].astype(int), ontology["name"]))
    id_to_acronym = dict(zip(ontology["id"].astype(int), ontology["acronym"]))
    return id_to_name, id_to_acronym


def get_cell_centroids(cell_label_path):
    labels = tiff.imread(cell_label_path)
    props = regionprops(labels)

    cell_ids = np.array([p.label for p in props], dtype=int)
    coords = np.round(np.array([p.centroid for p in props])).astype(int)

    return cell_ids, coords, labels.shape


def compute_region_probabilities_for_sample(
    sample_id,
    input_type,
    cell_label_path,
    atlas_paths,
    id_to_name,
    id_to_acronym,
):
    cell_ids, coords, label_shape = get_cell_centroids(cell_label_path)

    n_cells = len(cell_ids)
    n_regs = len(atlas_paths)

    if n_regs == 0:
        raise ValueError("No registered atlas crops found.")

    assignments = {int(cid): [] for cid in cell_ids}

    z, y, x = coords[:, 0], coords[:, 1], coords[:, 2]

    for atlas_path in atlas_paths:
        atlas = tiff.imread(atlas_path)

        if atlas.shape != label_shape:
            raise ValueError(
                f"Shape mismatch:\n"
                f"atlas={atlas.shape}\n"
                f"cell labels={label_shape}\n"
                f"path={atlas_path}"
            )

        valid = (
            (z >= 0) & (z < atlas.shape[0]) &
            (y >= 0) & (y < atlas.shape[1]) &
            (x >= 0) & (x < atlas.shape[2])
        )

        region_ids = np.full(n_cells, -1, dtype=int)
        region_ids[valid] = atlas[z[valid], y[valid], x[valid]].astype(int)

        for cid, rid in zip(cell_ids, region_ids):
            rid = int(rid)
            acronym = id_to_acronym.get(rid, f"unknown_{rid}")
            name = id_to_name.get(rid, f"unknown_{rid}")
            assignments[int(cid)].append((rid, acronym, name))

    long_rows = []

    for cid, regs in assignments.items():
        counts = Counter([r[1] for r in regs])

        for acronym, count in counts.items():
            rid, _, name = [r for r in regs if r[1] == acronym][0]

            long_rows.append({
                "sample_id": sample_id,
                "input_type": input_type,
                "cell_id": cid,
                "region_id": rid,
                "region_acronym": acronym,
                "region_name": name,
                "n_assignments": count,
                "n_registrations": n_regs,
                "probability": count / n_regs,
                "cell_label_path": str(cell_label_path),
            })

    long = pd.DataFrame(long_rows)

    cell_centroids = pd.DataFrame({
        "sample_id": sample_id,
        "input_type": input_type,
        "cell_id": cell_ids,
        "z": coords[:, 0],
        "y": coords[:, 1],
        "x": coords[:, 2],
    })

    wide = (
        long
        .pivot_table(
            index=["sample_id", "input_type", "cell_id"],
            columns="region_acronym",
            values="probability",
            fill_value=0,
        )
        .reset_index()
    )

    dominant = (
        long
        .sort_values("probability", ascending=False)
        .groupby(["sample_id", "input_type", "cell_id"])
        .first()
        .reset_index()
    )

    dominant = dominant[[
        "sample_id",
        "input_type",
        "cell_id",
        "region_acronym",
        "region_name",
        "probability",
    ]].rename(columns={
        "region_acronym": "dominant_region",
        "region_name": "dominant_region_name",
        "probability": "dominant_probability",
    })

    wide = wide.merge(
        dominant,
        on=["sample_id", "input_type", "cell_id"],
        how="left",
    )

    grouped_long = long.copy()
    grouped_long["region_group"] = np.where(
        grouped_long["region_acronym"].isin(REGIONS_OF_INTEREST),
        grouped_long["region_acronym"],
        "other",
    )

    grouped_long = (
        grouped_long
        .groupby(
            ["sample_id", "input_type", "cell_id", "region_group"],
            as_index=False,
        )["probability"]
        .sum()
    )

    grouped_wide = (
        grouped_long
        .pivot_table(
            index=["sample_id", "input_type", "cell_id"],
            columns="region_group",
            values="probability",
            fill_value=0,
        )
        .reset_index()
    )

    long = long.merge(cell_centroids, on=["sample_id", "input_type", "cell_id"], how="left")
    wide = wide.merge(cell_centroids, on=["sample_id", "input_type", "cell_id"], how="left")
    grouped_long = grouped_long.merge(cell_centroids, on=["sample_id", "input_type", "cell_id"], how="left")
    grouped_wide = grouped_wide.merge(cell_centroids, on=["sample_id", "input_type", "cell_id"], how="left")

    summary = {
        "sample_id": sample_id,
        "input_type": input_type,
        "n_cells": n_cells,
        "n_registrations": n_regs,
        "cell_label_path": str(cell_label_path),
        "status": "success",
        "error": "",
    }

    return long, wide, grouped_long, grouped_wide, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-root", default="results/region_assets")
    parser.add_argument("--ontology-csv", required=True)
    parser.add_argument("--out-dir", default="results/cell_region_probabilities")
    args = parser.parse_args()

    region_root = Path(args.region_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_to_name, id_to_acronym = load_ontology(args.ontology_csv)

    all_long = []
    all_wide = []
    all_grouped_long = []
    all_grouped_wide = []
    manifest = []

    sample_dirs = sorted([p for p in region_root.iterdir() if p.is_dir()])

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name

        for input_dir in sorted([p for p in sample_dir.iterdir() if p.is_dir()]):
            input_type = input_dir.name

            print("=" * 80)
            print(f"Running region probabilities: {sample_id} | {input_type}")
            print("=" * 80)

            cell_label_path = input_dir / "cellpose_outputs" / "cellpose_labels.tif"
            atlas_dir = input_dir / "registered_atlas_fullres_crop"

            atlas_paths = sorted(atlas_dir.glob("*.tif"))

            sample_out = out_dir / str(sample_id) / str(input_type)
            sample_out.mkdir(parents=True, exist_ok=True)

            try:
                if not cell_label_path.exists():
                    raise FileNotFoundError(f"Missing Cellpose labels: {cell_label_path}")

                if not atlas_dir.exists():
                    raise FileNotFoundError(f"Missing atlas crop directory: {atlas_dir}")

                long, wide, grouped_long, grouped_wide, summary = compute_region_probabilities_for_sample(
                    sample_id=sample_id,
                    input_type=input_type,
                    cell_label_path=cell_label_path,
                    atlas_paths=atlas_paths,
                    id_to_name=id_to_name,
                    id_to_acronym=id_to_acronym,
                )

                long.to_csv(sample_out / "cell_region_probabilities_long.csv", index=False)
                wide.to_csv(sample_out / "cell_region_probabilities_wide.csv", index=False)
                grouped_long.to_csv(sample_out / "cell_region_grouped_probabilities_long.csv", index=False)
                grouped_wide.to_csv(sample_out / "cell_region_grouped_probabilities_wide.csv", index=False)

                all_long.append(long)
                all_wide.append(wide)
                all_grouped_long.append(grouped_long)
                all_grouped_wide.append(grouped_wide)
                manifest.append(summary)

            except Exception as e:
                print(f"FAILED: {sample_id} | {input_type}")
                print(e)

                manifest.append({
                    "sample_id": sample_id,
                    "input_type": input_type,
                    "status": "failed",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

    pd.DataFrame(manifest).to_csv(out_dir / "run_manifest.csv", index=False)

    if all_long:
        pd.concat(all_long, ignore_index=True).to_csv(
            out_dir / "ALL_cell_region_probabilities_long.csv",
            index=False,
        )

    if all_wide:
        pd.concat(all_wide, ignore_index=True).to_csv(
            out_dir / "ALL_cell_region_probabilities_wide.csv",
            index=False,
        )

    if all_grouped_long:
        pd.concat(all_grouped_long, ignore_index=True).to_csv(
            out_dir / "ALL_cell_region_grouped_probabilities_long.csv",
            index=False,
        )

    if all_grouped_wide:
        pd.concat(all_grouped_wide, ignore_index=True).to_csv(
            out_dir / "ALL_cell_region_grouped_probabilities_wide.csv",
            index=False,
        )

    print("Done.")
    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()