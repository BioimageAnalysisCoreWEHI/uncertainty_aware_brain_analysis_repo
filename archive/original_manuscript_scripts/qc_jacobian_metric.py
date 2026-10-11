# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:28:06 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff


def extract_single(text, pattern):
    match = re.search(pattern, text)
    return match.group(1) if match else None


def parse_folder_params(folder_name):
    return {
        "Sample ID": folder_name.split("_")[0],
        "Folder Name": folder_name,
        "bending": extract_single(folder_name, r"_bending_([^_]+)"),
        "grid": extract_single(folder_name, r"_grid_([^_]+)"),
        "smoothing": extract_single(folder_name, r"_smoothing_([^_]+)"),
        "atlas": extract_single(folder_name, r"_atlas_([^_]+)")
    }


def load_deformation_components(folder_path):
    f0 = folder_path / "deformation_field_0.tiff"
    f1 = folder_path / "deformation_field_1.tiff"
    f2 = folder_path / "deformation_field_2.tiff"

    if not (f0.exists() and f1.exists() and f2.exists()):
        return None, None, None

    d0 = tiff.imread(f0).astype(np.float32)
    d1 = tiff.imread(f1).astype(np.float32)
    d2 = tiff.imread(f2).astype(np.float32)

    return d0, d1, d2


def compute_gradient_and_jacobian_metrics(d0, d1, d2):
    # ---------------------------------------------------------
    # Gradient magnitude of deformation
    # ---------------------------------------------------------
    g0 = np.gradient(d0)
    g1 = np.gradient(d1)
    g2 = np.gradient(d2)

    grad_mag = np.sqrt(
        g0[0]**2 + g0[1]**2 + g0[2]**2 +
        g1[0]**2 + g1[1]**2 + g1[2]**2 +
        g2[0]**2 + g2[1]**2 + g2[2]**2
    )

    # ---------------------------------------------------------
    # Jacobian determinant
    # Assumes deformation fields are displacement fields
    # J = I + grad(u)
    # ---------------------------------------------------------
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

    metrics = {
        # gradient magnitude summaries
        "grad_mean": float(np.mean(grad_mag)),
        "grad_std": float(np.std(grad_mag)),
        "grad_median": float(np.median(grad_mag)),
        "grad_p95": float(np.percentile(grad_mag, 95)),
        "grad_p99": float(np.percentile(grad_mag, 99)),
        "grad_max": float(np.max(grad_mag)),

        # jacobian summaries
        "jac_mean": float(np.mean(jac)),
        "jac_std": float(np.std(jac)),
        "jac_median": float(np.median(jac)),
        "jac_p01": float(np.percentile(jac, 1)),
        "jac_p05": float(np.percentile(jac, 5)),
        "jac_p95": float(np.percentile(jac, 95)),
        "jac_p99": float(np.percentile(jac, 99)),
        "jac_min": float(np.min(jac)),
        "jac_max": float(np.max(jac)),

        # useful QC fractions
        "jac_frac_lt_0": float(np.mean(jac < 0)),
        "jac_frac_lt_0_5": float(np.mean(jac < 0.5)),
        "jac_frac_gt_2": float(np.mean(jac > 2.0)),
    }

    return metrics


def collect_deformation_metrics(result_directory, status):
    rows = []
    result_directory = Path(result_directory)

    print("\n" + "=" * 90)
    print(f"STARTING DEFORMATION ANALYSIS: {status}")
    print(f"DIRECTORY: {result_directory}")
    print("=" * 90)

    if not result_directory.exists():
        print(f"[ERROR] Directory does not exist: {result_directory}")
        return pd.DataFrame()

    folders = sorted([f for f in result_directory.iterdir() if f.is_dir()])
    print(f"[INFO] Found {len(folders)} folders.\n")

    success_count = 0
    skip_count = 0

    for i, folder in enumerate(folders, start=1):
        folder_name = folder.name
        parsed = parse_folder_params(folder_name)

        print("-" * 90)
        print(f"[{i}/{len(folders)}] Processing: {folder_name}")
        print(
            f"    Sample={parsed['Sample ID']} | "
            f"bending={parsed['bending']} | "
            f"grid={parsed['grid']} | "
            f"smoothing={parsed['smoothing']} | "
            f"atlas={parsed['atlas']}"
        )

        d0, d1, d2 = load_deformation_components(folder)

        if d0 is None:
            print("    [SKIP] deformation_field_0/1/2.tiff not found")
            skip_count += 1
            continue

        if not (d0.shape == d1.shape == d2.shape):
            print(f"    [SKIP] Shape mismatch: {d0.shape}, {d1.shape}, {d2.shape}")
            skip_count += 1
            continue

        print(f"    [LOAD] Shapes: {d0.shape}")
        print(
            f"    [LOAD] Ranges: "
            f"d0=({d0.min():.3f}, {d0.max():.3f}), "
            f"d1=({d1.min():.3f}, {d1.max():.3f}), "
            f"d2=({d2.min():.3f}, {d2.max():.3f})"
        )

        try:
            metrics = compute_gradient_and_jacobian_metrics(d0, d1, d2)

            for k, v in metrics.items():
                parsed[f"{status}_{k}"] = v

            rows.append(parsed)
            success_count += 1

            print(
                f"    [DONE] grad_mean={metrics['grad_mean']:.4f}, "
                f"grad_p95={metrics['grad_p95']:.4f}, "
                f"jac_min={metrics['jac_min']:.4f}, "
                f"jac_p95={metrics['jac_p95']:.4f}, "
                f"jac_frac_lt_0={metrics['jac_frac_lt_0']:.6f}"
            )

        except Exception as e:
            print(f"    [SKIP] Error computing metrics: {e}")
            skip_count += 1
            continue

    print("\n" + "=" * 90)
    print(f"FINISHED DEFORMATION ANALYSIS: {status}")
    print(f"[SUMMARY] Successful: {success_count}")
    print(f"[SUMMARY] Skipped:    {skip_count}")
    print("=" * 90 + "\n")

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
raw_directory = Path(P(r"<SCRATCH>\jom_v2_raw"))
preprocessed_directory = Path(P(r"<SCRATCH>\jom_v2_preprocessed"))
output_csv = Path(P(r"<IMAGING>\Ishrat\JoM\qc_results\jom_jacobian_deformation_metrics.csv"))

print("\n" + "#" * 90)
print("STARTING RAW + PREPROCESSED DEFORMATION METRIC ANALYSIS")
print("#" * 90)
print(f"Raw directory:          {raw_directory}")
print(f"Preprocessed directory: {preprocessed_directory}")
print(f"Output CSV:             {output_csv}")
print("#" * 90)

raw_df = collect_deformation_metrics(raw_directory, status="raw")
pre_df = collect_deformation_metrics(preprocessed_directory, status="preprocessed")

print(f"[INFO] raw_df shape: {raw_df.shape}")
print(f"[INFO] pre_df shape: {pre_df.shape}")

merge_cols = ["Sample ID", "Folder Name", "bending", "grid", "smoothing", "atlas"]

pre_metric_cols = [c for c in pre_df.columns if c not in merge_cols]

print("[INFO] Merging raw + preprocessed dataframes...")
final_df = pd.merge(
    raw_df,
    pre_df[merge_cols + pre_metric_cols],
    on=merge_cols,
    how="outer"
)

print(f"[INFO] Final merged shape: {final_df.shape}")

print("[INFO] Converting bending/grid/smoothing to numeric where possible...")
for col in ["bending", "grid", "smoothing"]:
    final_df[col] = pd.to_numeric(final_df[col], errors="coerce")

print("[INFO] Sorting final dataframe...")
final_df = final_df.sort_values(
    by=["Sample ID", "bending", "grid", "smoothing"],
    na_position="last"
).reset_index(drop=True)

print("[INFO] Saving CSV...")
final_df.to_csv(output_csv, index=False)

print("\n" + "#" * 90)
print("ANALYSIS COMPLETE")
print("#" * 90)
print(f"[SUCCESS] Saved final dataframe to: {output_csv}")
print(f"[SUCCESS] Final rows: {len(final_df)}")
print("\nPreview:")
print(final_df.head(10))
print("#" * 90)