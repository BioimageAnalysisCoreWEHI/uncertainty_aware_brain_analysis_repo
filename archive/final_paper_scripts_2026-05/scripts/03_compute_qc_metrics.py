# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:24:00 2026

@author: zaman.i
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import tifffile as tiff
import yaml
from skimage.metrics import structural_similarity as ssim


def find_existing_file(folder: Path, possible_names: list[str]) -> Path | None:
    for name in possible_names:
        path = folder / name
        if path.exists():
            return path
    return None


def load_binary(path: Path) -> np.ndarray:
    return tiff.imread(path) != 0


def minmax_norm(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)


def compute_overlap_metrics(tissue_mask: np.ndarray, atlas_mask: np.ndarray) -> dict:
    intersection = np.logical_and(tissue_mask, atlas_mask).sum()
    union = np.logical_or(tissue_mask, atlas_mask).sum()

    atlas_voxels = atlas_mask.sum()
    tissue_voxels = tissue_mask.sum()

    return {
        "iou": intersection / union if union > 0 else np.nan,
        "atlas_inside_tissue": intersection / atlas_voxels if atlas_voxels > 0 else np.nan,
        "atlas_outside_tissue": np.logical_and(~tissue_mask, atlas_mask).sum() / atlas_voxels if atlas_voxels > 0 else np.nan,
        "tissue_inside_atlas": intersection / tissue_voxels if tissue_voxels > 0 else np.nan,
        "tissue_outside_atlas": np.logical_and(tissue_mask, ~atlas_mask).sum() / tissue_voxels if tissue_voxels > 0 else np.nan,
    }


def load_deformation_fields(folder: Path, names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    paths = [folder / name for name in names]
    if not all(p.exists() for p in paths):
        return None

    fields = [tiff.imread(p).astype(np.float32) for p in paths]

    if not (fields[0].shape == fields[1].shape == fields[2].shape):
        return None

    return fields[0], fields[1], fields[2]


def compute_deformation_metrics(d0: np.ndarray, d1: np.ndarray, d2: np.ndarray) -> dict:
    g0 = np.gradient(d0)
    g1 = np.gradient(d1)
    g2 = np.gradient(d2)

    grad_mag = np.sqrt(
        g0[0] ** 2 + g0[1] ** 2 + g0[2] ** 2 +
        g1[0] ** 2 + g1[1] ** 2 + g1[2] ** 2 +
        g2[0] ** 2 + g2[1] ** 2 + g2[2] ** 2
    )

    dux_dx, dux_dy, dux_dz = np.gradient(d0)
    duy_dx, duy_dy, duy_dz = np.gradient(d1)
    duz_dx, duz_dy, duz_dz = np.gradient(d2)

    J11 = 1 + dux_dx
    J12 = dux_dy
    J13 = dux_dz

    J21 = duy_dx
    J22 = 1 + duy_dy
    J23 = duy_dz

    J31 = duz_dx
    J32 = duz_dy
    J33 = 1 + duz_dz

    jac = (
        J11 * (J22 * J33 - J23 * J32)
        - J12 * (J21 * J33 - J23 * J31)
        + J13 * (J21 * J32 - J22 * J31)
    )

    return {
        "grad_mean": float(np.mean(grad_mag)),
        "grad_std": float(np.std(grad_mag)),
        "grad_median": float(np.median(grad_mag)),
        "grad_p95": float(np.percentile(grad_mag, 95)),
        "grad_p99": float(np.percentile(grad_mag, 99)),
        "grad_max": float(np.max(grad_mag)),
        "jac_mean": float(np.mean(jac)),
        "jac_std": float(np.std(jac)),
        "jac_median": float(np.median(jac)),
        "jac_p01": float(np.percentile(jac, 1)),
        "jac_p05": float(np.percentile(jac, 5)),
        "jac_p95": float(np.percentile(jac, 95)),
        "jac_p99": float(np.percentile(jac, 99)),
        "jac_min": float(np.min(jac)),
        "jac_max": float(np.max(jac)),
        "jac_frac_lt_0": float(np.mean(jac < 0)),
        "jac_frac_lt_0_5": float(np.mean(jac < 0.5)),
        "jac_frac_gt_2": float(np.mean(jac > 2.0)),
    }


def crop_bbox(arr: np.ndarray, bbox: dict) -> np.ndarray:
    return arr[
        bbox["z0"]:bbox["z1"],
        bbox["y0"]:bbox["y1"],
        bbox["x0"]:bbox["x1"],
    ]


def compute_local_ssim(sample_img: np.ndarray, atlas_img: np.ndarray, bbox: dict) -> float:
    atlas_crop = minmax_norm(crop_bbox(atlas_img, bbox))
    sample_crop = minmax_norm(crop_bbox(sample_img, bbox))

    if atlas_crop.shape != sample_crop.shape:
        return np.nan

    return float(ssim(atlas_crop, sample_crop, data_range=1.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--samples", default="config/samples.csv")
    parser.add_argument("--jobs", default="results/job_tables/brainreg_jobs.csv")
    parser.add_argument("--out", default="results/qc_tables/qc_results.csv")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    with open(cfg["roi"]["bbox_json"], "r") as f:
        bbox = json.load(f)

    atlas_img = tiff.imread(cfg["atlas"]["reference_image"]).astype(np.float32)

    samples = pd.read_csv(args.samples)
    jobs = pd.read_csv(args.jobs)

    jobs = jobs.merge(
        samples[["sample_id", "input_type", "tissue_mask"]],
        on=["sample_id", "input_type"],
        how="left",
    )

    registered_atlas_names = cfg["brainreg_outputs"]["registered_atlas_names"]
    downsampled_standard_names = cfg["brainreg_outputs"]["downsampled_standard_names"]
    deformation_names = cfg["brainreg_outputs"]["deformation_field_names"]

    iou_threshold = cfg["qc"]["iou_threshold"]
    grad_p95_threshold = cfg["qc"]["grad_p95_threshold"]
    ssim_top_fraction = cfg["qc"]["ssim_top_fraction"]

    rows = []

    for i, row in jobs.iterrows():
        save_dir = Path(row["save_dir"])

        print(f"[{i + 1}/{len(jobs)}] QC: {row['sample_id']} | {row['input_type']} | {save_dir.name}")

        result = row.to_dict()
        result["registration_dir"] = str(save_dir)

        registered_atlas_path = find_existing_file(save_dir, registered_atlas_names)
        standard_path = find_existing_file(save_dir, downsampled_standard_names)
        tissue_mask_path = Path(row["tissue_mask"]) if pd.notna(row["tissue_mask"]) else None

        result["registered_atlas_path"] = str(registered_atlas_path) if registered_atlas_path else None
        result["downsampled_standard_path"] = str(standard_path) if standard_path else None

        # -------------------------
        # Global overlap QC
        # -------------------------
        if registered_atlas_path and tissue_mask_path and tissue_mask_path.exists():
            tissue_mask = load_binary(tissue_mask_path)
            atlas_mask = load_binary(registered_atlas_path)

            if tissue_mask.shape == atlas_mask.shape:
                result.update(compute_overlap_metrics(tissue_mask, atlas_mask))
            else:
                result["qc_error_overlap"] = f"shape mismatch tissue={tissue_mask.shape}, atlas={atlas_mask.shape}"
        else:
            result["qc_error_overlap"] = "missing registered atlas or tissue mask"

        # -------------------------
        # Deformation/Jacobian QC
        # -------------------------
        deformation_fields = load_deformation_fields(save_dir, deformation_names)

        if deformation_fields is not None:
            result.update(compute_deformation_metrics(*deformation_fields))
        else:
            result["qc_error_deformation"] = "missing or mismatched deformation fields"

        # -------------------------
        # Local SSIM QC
        # -------------------------
        if standard_path and standard_path.exists():
            sample_standard = tiff.imread(standard_path).astype(np.float32)
            result["local_ssim"] = compute_local_ssim(sample_standard, atlas_img, bbox)
        else:
            result["qc_error_ssim"] = "missing downsampled_standard image"

        rows.append(result)

    qc = pd.DataFrame(rows)

    # -------------------------
    # Apply QC status columns
    # -------------------------
    qc["passes_global_qc"] = (
        (qc["iou"] > iou_threshold) &
        (qc["grad_p95"] < grad_p95_threshold)
    )

    qc["ssim_rank"] = qc.groupby(["sample_id", "input_type"])["local_ssim"].rank(
        ascending=False,
        method="first",
    )

    qc["ssim_percentile"] = qc.groupby(["sample_id", "input_type"])["local_ssim"].rank(
        ascending=True,
        pct=True,
    )

    qc["passes_local_ssim_qc"] = qc.groupby(["sample_id", "input_type"])["local_ssim"].transform(
        lambda x: x >= x.quantile(1 - ssim_top_fraction)
    )

    qc["passes_final_qc"] = qc["passes_global_qc"] & qc["passes_local_ssim_qc"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    qc.to_csv(out_path, index=False)

    print(f"\nSaved QC table to: {out_path}")
    print(f"Rows: {len(qc)}")


if __name__ == "__main__":
    main()