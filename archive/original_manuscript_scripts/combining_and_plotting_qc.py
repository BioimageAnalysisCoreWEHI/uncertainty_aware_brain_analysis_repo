# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:59:18 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import pandas as pd
import os
import re
import seaborn as sns

counts_raw = pd.read_csv(P(r"<IMAGING>\Ishrat\JoM\final_count_results\final_processed_data_snpc_raw.csv"))
counts_preprocessed = pd.read_csv(P(r"<IMAGING>\Ishrat\JoM\final_count_results\final_processed_data_snpc_preprocessed.csv"))

qc = pd.read_csv(P(r"<IMAGING>\Ishrat\JoM\qc_results\qc_results_full.csv"))
qc = qc[qc['grid'] != -1]

counts_raw = counts_raw[counts_raw['grid'] != -1]
counts_raw = counts_raw.rename(columns={"TotalCellCount_snpc_Image": "raw_TotalCellCount_snpc_Image",
                                        "snpc_CellCount": "raw_snpc_CellCount",
                                        "Volume_RegionSum_snpc_px3": "raw_Volume_RegionSum_snpc_px3",
                                        "Volume_RegionSum_snpc_mm3": "raw_Volume_RegionSum_snpc_mm3",
                                        'full_brain_mm3': "raw_full_brain_mm3",
                                        "Sample": "Folder Name", 
                                        "ID": "Sample ID"})

counts_preprocessed = counts_preprocessed[counts_preprocessed['grid'] != -1]
counts_preprocessed = counts_preprocessed.rename(columns={"TotalCellCount_snpc_Image": "preprocessed_TotalCellCount_snpc_Image",
                                        "snpc_CellCount": "preprocessed_snpc_CellCount",
                                        "Volume_RegionSum_snpc_px3": "preprocessed_Volume_RegionSum_snpc_px3",
                                        "Volume_RegionSum_snpc_mm3": "preprocessed_Volume_RegionSum_snpc_mm3",
                                        'full_brain_mm3': "preprocessed_full_brain_mm3",
                                        "Sample": "Folder Name", 
                                        "ID": "Sample ID"})


ssim_raw = pd.read_csv(P(r"<IMAGING>\Ishrat\JoM\qc_results\ssim_midbrain_qc_results_raw.csv"))
ssim_raw = ssim_raw.rename(columns={"registration_folder": "Folder Name",
                                        "raw_ssim": "raw_ssim_value",
                                        "relative_qc_score": "raw_relative_ssim_score",
                                        "sample_id": "Sample ID"})

ssim_preprocessed = pd.read_csv(P(r"<IMAGING>\Ishrat\JoM\qc_results\ssim_midbrain_qc_results_preprocessed.csv"))

ssim_preprocessed = ssim_preprocessed.rename(columns={"registration_folder": "Folder Name",
                                        "raw_ssim": "preprocessed_ssim_value",
                                        "relative_qc_score": "preprocessed_relative_ssim_score",
                                        "sample_id": "Sample ID"})


key = ["Folder Name"]
cols_to_add = [col for col in counts_raw.columns if col not in qc.columns or col in key]
counts_raw_subset = counts_raw[cols_to_add]
merged_qc_raw = qc.merge(counts_raw_subset, on=key, how="left")



cols_to_add = [col for col in counts_preprocessed.columns if col not in merged_qc_raw.columns or col in key]
counts_preprocessed_subset = counts_preprocessed[cols_to_add]
merged_qc_counts = merged_qc_raw.merge(counts_preprocessed_subset, on=key, how="left")



cols_to_add = [col for col in ssim_raw.columns if col not in merged_qc_counts.columns or col in key]
ssim_raw_subset = ssim_raw[cols_to_add]
merged_qc_counts_ssim_raw = merged_qc_counts.merge(ssim_raw_subset, on=key, how="left")


cols_to_add = [col for col in ssim_preprocessed.columns if col not in merged_qc_counts_ssim_raw.columns or col in key]
ssim_preprocessed_subset = ssim_preprocessed[cols_to_add]
merged_qc_full = merged_qc_counts_ssim_raw.merge(ssim_preprocessed_subset, on=key, how="left")


merged_qc_full.to_csv(P(r"<IMAGING>\Ishrat\JoM\final_count_results\full_final_results_with_ssim.csv"))
merged_df = merged_qc_full

shared_cols = [
    "Sample ID",
    "Folder Name",
    "bending",
    "grid",
    "smoothing",
    "atlas",
    "voxel_vol_um3",
    # add anything else that applies to both
]

raw_cols = [col for col in merged_df.columns if "raw" in col]
pre_cols = [col for col in merged_df.columns if "preprocessed" in col]


df_raw = merged_df[shared_cols + raw_cols].copy()
df_pre = merged_df[shared_cols + pre_cols].copy()


df_raw.columns = [col.replace("_raw", "") for col in df_raw.columns]
df_raw.columns = [col.replace("raw_", "") for col in df_raw.columns]

df_pre.columns = [col.replace("_preprocessed", "") for col in df_pre.columns]
df_pre.columns = [col.replace("preprocessed_", "") for col in df_pre.columns]

df_raw_filtered = df_raw[df_raw['pass_final'] == True]
df_pre_filtered = df_pre[df_pre['pass_final'] == True]


df_raw["processing"] = "raw"
df_pre["processing"] = "preprocessed"

df_long = pd.concat([df_raw, df_pre], ignore_index=True)
df_long_filtered = df_long[df_long["pass_final"] == True]

df_long_filtered_top_ssim = (
    df_long_filtered.groupby(['Sample ID', 'processing'], group_keys=False)
      .apply(lambda g: g[g['ssim_value'] >= g['ssim_value'].quantile(0.75)])
)

df_long_filtered_top10_ssim = (
    df_long_filtered
    .sort_values('ssim_value', ascending=False)
    .groupby(['Sample ID', 'processing'], group_keys=False)
    .head(10)
)

df_long.to_csv(P(r"<IMAGING>\Ishrat\JoM\final_count_results\full_final_results_with_ssim_long.csv"))

sns.boxplot(
    data=df_long_filtered_top10_ssim,
    x="Sample ID",
    y="snpc_CellCount",
    hue="processing"
)
