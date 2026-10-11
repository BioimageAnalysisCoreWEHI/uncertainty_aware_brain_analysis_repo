# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 15:06:36 2026

@author: zaman.i
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 15:13:38 2026

@author: zaman.i

Changes from prior version
---------------------------
- SSIM ranking/percentile/final-QC selection has moved to
  05_prepare_crop_spaces.py. This script now only computes
  passes_global_qc; local_ssim rides through unchanged as a raw column.
  This means:
    (a) 05 can rank against whichever global-QC column it's pointed
        at (including manual overrides), without 04 ever needing to be
        rerun, and
    (b) there is exactly one place ranking happens, ever - no risk of
        04 and 05 disagreeing about ranking definitions.
- qc.ssim_top_fraction is no longer read here (05 owns that setting now).
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import yaml


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config/config.yaml",
    )

    parser.add_argument(
        "--input_dir",
        default="results/qc_tables/individual_registration_qc_hpc",
    )

    parser.add_argument(
        "--output",
        default="results/qc_tables/qc_results.csv",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load configuration
    # ------------------------------------------------------------

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    iou_threshold = cfg["qc"]["iou_threshold"]
    grad_p95_threshold = cfg["qc"]["grad_p95_threshold"]

    # ------------------------------------------------------------
    # Find all QC CSVs
    # ------------------------------------------------------------

    input_dir = Path(args.input_dir)

    csv_files = sorted(
        input_dir.glob("qc_registration_*.csv")
    )

    if len(csv_files) == 0:
        raise RuntimeError(
            f"No QC csv files found in:\n{input_dir}"
        )

    print(f"Found {len(csv_files)} QC result files.")

    # ------------------------------------------------------------
    # Read all tables
    # ------------------------------------------------------------

    dfs = []

    for csv in csv_files:

        try:
            dfs.append(
                pd.read_csv(csv)
            )

        except Exception as e:
            print(f"Failed to read {csv.name}: {e}")

    qc = pd.concat(
        dfs,
        ignore_index=True,
    )

    # ------------------------------------------------------------
    # Sort back into original order
    # ------------------------------------------------------------

    if "job_index" in qc.columns:

        qc = qc.sort_values(
            "job_index"
        ).reset_index(
            drop=True
        )

    print(f"Combined rows: {len(qc)}")

    # ------------------------------------------------------------
    # Apply QC thresholds
    # ------------------------------------------------------------

    qc["passes_global_qc"] = (
        (qc["iou"] >= iou_threshold)
        &
        (qc["grad_p95"] <= grad_p95_threshold)
    )

    # NOTE: local_ssim is passed through untouched. SSIM ranking,
    # percentile, and top-fraction selection (passes_final_qc, etc.) are
    # now computed downstream in 05_prepare_crop_spaces.py, scoped to
    # whichever global-QC column that script is run with.

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    qc.to_csv(
        output_path,
        index=False,
    )

    print()
    print(f"Saved: {output_path}")
    print(f"Rows: {len(qc)}")

    print()
    print(
        f"Passed global QC: {qc['passes_global_qc'].sum()}"
    )


if __name__ == "__main__":
    main()