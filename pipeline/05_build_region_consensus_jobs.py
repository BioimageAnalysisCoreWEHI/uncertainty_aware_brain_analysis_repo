# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:57:45 2026

@author: zaman.i
"""

# -*- coding: utf-8 -*-
"""
Build a job table listing one row per unique sample_id, for
parallelising build_region_consensus_wholebrain.py as a SLURM array
(one task per sample -- covers every input_type present for that sample
in a single task, same convention as build_crop_space_jobs.py).
"""

from pathlib import Path
import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a job table for parallelising build_region_consensus_wholebrain.py "
            "across SLURM array tasks, one task per sample_id."
        )
    )
    parser.add_argument("--qc-table", default="results/qc_tables/qc_results.csv")
    parser.add_argument("--out", default="results/job_tables/region_consensus_jobs.csv")
    args = parser.parse_args()

    qc = pd.read_csv(args.qc_table)

    if "sample_id" not in qc.columns:
        raise ValueError(f"--qc-table {args.qc_table} has no 'sample_id' column.")

    sample_ids = sorted(qc["sample_id"].astype(str).unique())

    if not sample_ids:
        raise RuntimeError(f"No sample_id values found in {args.qc_table}")

    jobs = pd.DataFrame({"sample_id": sample_ids})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(out_path, index=False, lineterminator="\n")

    print(f"Saved {len(jobs)} region-consensus jobs (one per sample_id) to: {out_path}")


if __name__ == "__main__":
    main()