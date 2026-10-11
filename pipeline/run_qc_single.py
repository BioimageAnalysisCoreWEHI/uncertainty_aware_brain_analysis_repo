# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 11:50:50 2026
Updated for multi-region (leaf + composite) local SSIM QC.

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import to_local  # noqa: E402
from pathlib import Path
import argparse
import platform

import numpy as np
import pandas as pd
import tifffile as tiff
import yaml

from region_utils import (
    load_region_tree,
    build_region_lookup,
    compute_local_ssim,
    save_atlas_crops,
    save_sample_crop,
)


# ---------------------------------------------------------------------
# PATH HANDLING
# ---------------------------------------------------------------------

def normalise_path(path_str):
    """Map a stored path (placeholder form or a cluster path) to the local file system (see uaa_paths.py)."""
    if path_str is None:
        return None
    return Path(to_local(path_str))


def find_existing_file(
    folder: Path,
    possible_names: list[str],
) -> Path | None:
    """
    Find the first existing file from a list of possible filenames.
    """

    for name in possible_names:
        path = folder / name

        if path.exists():
            return path

    return None


# ---------------------------------------------------------------------
# IMAGE UTILITIES
# ---------------------------------------------------------------------

def load_binary(path: Path) -> np.ndarray:
    """
    Load a TIFF and convert all non-zero voxels to True.
    """

    return tiff.imread(path) != 0


# ---------------------------------------------------------------------
# GLOBAL OVERLAP QC
# ---------------------------------------------------------------------

def compute_overlap_metrics(
    tissue_mask: np.ndarray,
    atlas_mask: np.ndarray,
) -> dict:
    """
    Calculate overlap between the tissue mask and registered atlas.
    """

    intersection = np.count_nonzero(
        tissue_mask & atlas_mask
    )

    atlas_voxels = np.count_nonzero(
        atlas_mask
    )

    tissue_voxels = np.count_nonzero(
        tissue_mask
    )

    union = (
        atlas_voxels
        + tissue_voxels
        - intersection
    )

    atlas_outside = (
        atlas_voxels - intersection
    )

    tissue_outside = (
        tissue_voxels - intersection
    )

    return {
        "iou": (
            intersection / union
            if union > 0
            else np.nan
        ),

        "atlas_inside_tissue": (
            intersection / atlas_voxels
            if atlas_voxels > 0
            else np.nan
        ),

        "atlas_outside_tissue": (
            atlas_outside / atlas_voxels
            if atlas_voxels > 0
            else np.nan
        ),

        "tissue_inside_atlas": (
            intersection / tissue_voxels
            if tissue_voxels > 0
            else np.nan
        ),

        "tissue_outside_atlas": (
            tissue_outside / tissue_voxels
            if tissue_voxels > 0
            else np.nan
        ),
    }


# ---------------------------------------------------------------------
# DEFORMATION-FIELD QC
# ---------------------------------------------------------------------

def load_deformation_fields(
    folder: Path,
    names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Load the three deformation-field components.
    """

    paths = [
        folder / name
        for name in names
    ]

    if not all(path.exists() for path in paths):
        return None

    fields = [
        tiff.imread(path).astype(
            np.float32,
            copy=False,
        )
        for path in paths
    ]

    if not (
        fields[0].shape
        == fields[1].shape
        == fields[2].shape
    ):
        return None

    return (
        fields[0],
        fields[1],
        fields[2],
    )


def compute_deformation_metrics(
    d0: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
) -> dict:
    """
    Calculate deformation-gradient and Jacobian metrics.

    Each deformation-field gradient is calculated only once.
    """

    dux_dx, dux_dy, dux_dz = np.gradient(d0)
    duy_dx, duy_dy, duy_dz = np.gradient(d1)
    duz_dx, duz_dy, duz_dz = np.gradient(d2)

    grad_mag = np.sqrt(
        dux_dx ** 2
        + dux_dy ** 2
        + dux_dz ** 2
        + duy_dx ** 2
        + duy_dy ** 2
        + duy_dz ** 2
        + duz_dx ** 2
        + duz_dy ** 2
        + duz_dz ** 2
    )

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

    grad_percentiles = np.percentile(
        grad_mag,
        [50, 95, 99],
    )

    jac_percentiles = np.percentile(
        jac,
        [1, 5, 50, 95, 99],
    )

    return {
        "grad_mean": float(np.mean(grad_mag)),
        "grad_std": float(np.std(grad_mag)),
        "grad_median": float(grad_percentiles[0]),
        "grad_p95": float(grad_percentiles[1]),
        "grad_p99": float(grad_percentiles[2]),
        "grad_max": float(np.max(grad_mag)),

        "jac_mean": float(np.mean(jac)),
        "jac_std": float(np.std(jac)),
        "jac_median": float(jac_percentiles[2]),
        "jac_p01": float(jac_percentiles[0]),
        "jac_p05": float(jac_percentiles[1]),
        "jac_p95": float(jac_percentiles[3]),
        "jac_p99": float(jac_percentiles[4]),
        "jac_min": float(np.min(jac)),
        "jac_max": float(np.max(jac)),

        "jac_frac_lt_0": float(
            np.mean(jac < 0)
        ),

        "jac_frac_lt_0_5": float(
            np.mean(jac < 0.5)
        ),

        "jac_frac_gt_2": float(
            np.mean(jac > 2.0)
        ),
    }


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
        "--samples",
        default="config/samples.csv",
    )

    parser.add_argument(
        "--jobs",
        default="results/job_tables/brainreg_jobs.csv",
    )

    parser.add_argument(
        "--job-index",
        type=int,
        required=True,
        help=(
            "Zero-based row index in the merged brainreg jobs table."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/qc_tables/"
            "individual_registration_qc"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing result file.",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Load configuration
    # -----------------------------------------------------------------

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)

    atlas_path = normalise_path(
        cfg["atlas"]["reference_image"]
    )

    # -----------------------------------------------------------------
    # Build per-region masks/bboxes.
    #
    # NOTE: this recomputes the region lookup once per array-task
    # invocation (not per-registration -- there's only one registration
    # per invocation of this script). It's cheap: tree recursion plus a
    # handful of boolean masks against a single fixed atlas volume.
    # -----------------------------------------------------------------

    tree_df = load_region_tree(
        normalise_path(cfg["roi"]["region_tree_csv"])
    )

    annotation_volume = tiff.imread(
        normalise_path(cfg["roi"]["annotation_image"])
    )

    region_lookup = build_region_lookup(
        region_ids=cfg["roi"]["region_ids"],
        tree_df=tree_df,
        annotation_volume=annotation_volume,
        padding=cfg["roi"].get("bbox_padding_voxels", 5),
        split_hemispheres=cfg["roi"].get("split_hemispheres", False),
        lr_axis=cfg["roi"].get("lr_axis", 2),
        midline_index=cfg["roi"].get("midline_index"),
        low_side_label=cfg["roi"].get("low_side_label", "right"),
        high_side_label=cfg["roi"].get("high_side_label", "left"),
    )

    del annotation_volume

    save_crop_examples = cfg["roi"].get("save_crop_examples", False)
    crop_output_dir = cfg["roi"].get(
        "crop_output_dir", "results/qc_tables/region_crop_examples"
    )

    # Reference-atlas crops don't depend on the sample -- save one set,
    # unconditionally, regardless of save_crop_examples. Cheap and
    # idempotent even if every array task re-writes the same files.
    atlas_img = tiff.imread(atlas_path).astype(np.float32, copy=False)
    save_atlas_crops(atlas_img, region_lookup, crop_output_dir)

    # -----------------------------------------------------------------
    # Load and merge job tables
    # -----------------------------------------------------------------

    samples = pd.read_csv(
        args.samples
    )

    jobs = pd.read_csv(
        args.jobs
    )

    jobs = jobs.merge(
        samples[
            [
                "sample_id",
                "input_type",
                "tissue_mask",
            ]
        ],
        on=[
            "sample_id",
            "input_type",
        ],
        how="left",
        validate="many_to_one",
    )

    jobs = jobs.reset_index(drop=True)

    if (
        args.job_index < 0
        or args.job_index >= len(jobs)
    ):
        raise IndexError(
            f"Job index {args.job_index} is outside "
            f"the valid range 0-{len(jobs) - 1}"
        )

    # Select exactly one registration
    row = jobs.iloc[
        args.job_index
    ]

    save_dir = normalise_path(
        row["save_dir"]
    )

    print(
        f"QC registration {args.job_index + 1}/{len(jobs)}"
    )

    print(
        f"Sample: {row['sample_id']}"
    )

    print(
        f"Input type: {row['input_type']}"
    )

    print(
        f"Registration directory: {save_dir}"
    )

    # -----------------------------------------------------------------
    # Set output path
    # -----------------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / (
        f"qc_registration_"
        f"{args.job_index:05d}.csv"
    )

    if (
        output_path.exists()
        and not args.overwrite
    ):
        print(
            f"Output already exists; skipping: "
            f"{output_path}"
        )
        return

    # -----------------------------------------------------------------
    # Set up result row
    # -----------------------------------------------------------------

    result = row.to_dict()

    result["job_index"] = (
        args.job_index
    )

    result["registration_dir"] = str(
        save_dir
    )

    registered_atlas_names = cfg[
        "brainreg_outputs"
    ]["registered_atlas_names"]

    downsampled_standard_names = cfg[
        "brainreg_outputs"
    ]["downsampled_standard_names"]

    deformation_names = cfg[
        "brainreg_outputs"
    ]["deformation_field_names"]

    registered_atlas_path = find_existing_file(
        save_dir,
        registered_atlas_names,
    )

    standard_path = find_existing_file(
        save_dir,
        downsampled_standard_names,
    )

    tissue_mask_path = (
        normalise_path(row["tissue_mask"])
        if pd.notna(row["tissue_mask"])
        else None
    )

    result["registered_atlas_path"] = (
        str(registered_atlas_path)
        if registered_atlas_path
        else None
    )

    result["downsampled_standard_path"] = (
        str(standard_path)
        if standard_path
        else None
    )

    result["tissue_mask_path"] = (
        str(tissue_mask_path)
        if tissue_mask_path
        else None
    )

    # -----------------------------------------------------------------
    # Global overlap QC
    # -----------------------------------------------------------------

    if (
        registered_atlas_path
        and tissue_mask_path
        and tissue_mask_path.exists()
    ):
        tissue_mask = load_binary(
            tissue_mask_path
        )

        atlas_mask = load_binary(
            registered_atlas_path
        )

        if tissue_mask.shape == atlas_mask.shape:
            result.update(
                compute_overlap_metrics(
                    tissue_mask,
                    atlas_mask,
                )
            )

            print("Global overlap QC completed.")

        else:
            result["qc_error_overlap"] = (
                f"shape mismatch "
                f"tissue={tissue_mask.shape}, "
                f"atlas={atlas_mask.shape}"
            )

    else:
        result["qc_error_overlap"] = (
            "missing registered atlas or tissue mask"
        )

    # -----------------------------------------------------------------
    # Deformation/Jacobian QC
    # -----------------------------------------------------------------

    deformation_fields = load_deformation_fields(
        save_dir,
        deformation_names,
    )

    if deformation_fields is not None:
        result.update(
            compute_deformation_metrics(
                *deformation_fields
            )
        )

        print("Deformation QC completed.")

    else:
        result["qc_error_deformation"] = (
            "missing or mismatched deformation fields"
        )

    # -----------------------------------------------------------------
    # Local SSIM QC -- one score per requested region
    # -----------------------------------------------------------------

    if (
        standard_path
        and standard_path.exists()
    ):
        sample_standard = tiff.imread(
            standard_path
        ).astype(
            np.float32,
            copy=False,
        )

        for region_id, info in region_lookup.items():
            for side_label, side_info in info["hemispheres"].items():
                col = f"local_ssim__{info['acronym']}_{side_label}"
                result[col] = compute_local_ssim(
                    sample_standard,
                    atlas_img,
                    side_info["bbox"],
                )

                if save_crop_examples:
                    prefix = f"{row['sample_id']}_{row['input_type']}__{info['acronym']}_{side_label}"
                    save_sample_crop(
                        sample_standard, side_info["bbox"], crop_output_dir, prefix
                    )

        print("Local SSIM QC completed for all regions.")

    else:
        for region_id, info in region_lookup.items():
            for side_label in info["hemispheres"]:
                col = f"local_ssim__{info['acronym']}_{side_label}"
                result[col] = np.nan
                result[f"qc_error_ssim__{info['acronym']}_{side_label}"] = (
                    "missing downsampled standard image"
                )

    # -----------------------------------------------------------------
    # Save one-row result table
    # -----------------------------------------------------------------

    qc = pd.DataFrame(
        [result]
    )

    qc.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved QC result to: {output_path}"
    )


if __name__ == "__main__":
    main()