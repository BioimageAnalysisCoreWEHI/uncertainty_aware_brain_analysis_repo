"""A0: provenance -- SHA-256 of every input table, the aMAP reproduction git commit, software versions.
Large image inputs are not hashed (they are read through the caches); their folders and file counts are recorded.
Output: results/tables/provenance_inputs.csv, results/tables/provenance_env.json
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy
import tifffile

from common import AMAP_OUT, AMAP_PY, CFG, LSFM, PIPE, TABLES, log

TABLE_INPUTS = [
    PIPE / "qc_tables" / "qc_results.csv", PIPE / "qc_tables" / "qc_results_preprocessed.csv",
    PIPE.parent / "config" / "config.yaml", PIPE.parent / "config" / "ARA2_annotation_info_avail_regions.csv",
    AMAP_OUT / "userStapleResults" / "segTable_with_pq.csv", AMAP_OUT / "userStapleResults" / "canvas.json",
    AMAP_OUT / "userStapleResults" / "asr25um" / "asr25um_slices.csv",
    AMAP_OUT / "userStapleResults" / "asr25um" / "asr25um_raters.csv",
    AMAP_OUT / "pipeline_comparison_registered_hemispheres" / "tables" / "main.csv",
    AMAP_OUT / "pipeline_comparison_registered_hemispheres" / "tables" / "runs.csv",
    AMAP_OUT / "pipeline_comparison_registered_hemispheres" / "tables" / "ap.csv",
    AMAP_OUT / "pipeline_comparison_registered_hemispheres" / "tables" / "align.csv",
    AMAP_OUT / "pipeline_comparison" / "tables" / "main.csv",
    LSFM / "final_count_results" / "full_final_results_with_ssim.csv",
    LSFM / "final_count_results" / "full_final_results_with_ssim_long.csv",
    LSFM / "final_count_results" / "consensus_threshold_cell_counts_overlap_logic" / "ALL_consensus_threshold_cell_counts_summary.csv",
    LSFM / "final_count_results" / "cell_region_probability_outputs" / "ALL_cell_region_grouped_probabilities_wide.csv",
    LSFM / "final_count_results" / "cell_region_probability_outputs_all_runs" / "ALL_cell_region_grouped_probabilities_wide.csv",
    LSFM / "analysis_scripts" / "get_consensus_counts.py",
    PIPE.parent / "scripts" / "build_region_consensus_wholebrain.py", PIPE.parent / "scripts" / "region_utils.py",
]
IMAGE_FOLDERS = [PIPE / "region_consensus", PIPE / "registration_images_collected",
                 AMAP_OUT / "userStapleResults" / "asr25um", LSFM / "results_raw" / "extracted_regions_raw_no_padding_union",
                 LSFM / "results_preprocessed", LSFM / "results_raw" / "cellpose_masked_cells"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    rows = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) if p.exists() else dict(path=str(p), bytes=None,
                                                                                                sha256="MISSING")
            for p in TABLE_INPUTS]
    for d in IMAGE_FOLDERS:
        rows.append(dict(path=str(d), bytes=None, sha256="(image folder; not hashed)", exists=d.exists()))
    pd.DataFrame(rows).to_csv(TABLES / "provenance_inputs.csv", index=False)
    try:
        commit = subprocess.run(["git", "-C", str(AMAP_PY.parent), "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(AMAP_PY.parent), "status", "--porcelain", "python"], capture_output=True,
                               text=True).stdout.strip()
    except Exception as e:  # git not on PATH
        commit, dirty = f"unavailable ({e})", ""
    import matplotlib
    env = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, pandas=pd.__version__,
               scipy=scipy.__version__, tifffile=tifffile.__version__, matplotlib=matplotlib.__version__,
               amap_repo_commit=commit, amap_python_uncommitted_changes=bool(dirty), config=CFG)
    (TABLES / "provenance_env.json").write_text(json.dumps(env, indent=1, default=str))
    log(f"{len(rows)} inputs recorded; aMAP commit {commit}")
