# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 15:48:16 2025

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

def plot_metric(df, metric, title, ylabel, filename, ylims=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x='ID',
        y=metric,
        showcaps=True,
        boxprops=dict(edgecolor='white'),
        whiskerprops=dict(color='white'),
        capprops=dict(color='white'),
        medianprops=dict(color='white'),
        flierprops=dict(markeredgecolor='white'),
        ax=ax
    )

    sns.stripplot(
        data=df,
        x='ID',
        y=metric,
        jitter=True,
        size=6,
        palette=sns.color_palette("magma", df['ID'].nunique()),
        ax=ax,
        dodge=False,
        edgecolor='white',
        linewidth=0.5
    )

    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.grid(True, color='gray', alpha=0.3)
    ax.set(xlabel="Sample ID", ylabel=ylabel, title=title)
    ax.set_xlim(auto=True)
    if ylims:
        ax.set_ylim(ylims)

    for spine in ax.spines.values():
        spine.set_color('white')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    plt.close()


# --- Load and preprocess general volume and voxel info ---
processing_status = "preprocessed"
df_volumes = pd.read_csv(P(rf"<IMAGING>\Ishrat\JoM\region_volumes_final_merged_{processing_status}.csv"))
df_volumes = df_volumes.rename(columns={"dataset": "Sample"})

px_info = pd.read_csv(P(r"<IMAGING>\Ishrat\JoM/001_image_dimensions_level1.csv"))
px_info["ID"] = px_info["Folder"].str.split("_").str[0]
px_info["voxel_vol_um3"] = px_info["pixel_size_z"] * px_info["pixel_size_x"] * px_info["pixel_size_y"]
voxel_info = px_info[["ID", "voxel_vol_um3"]]


def preprocess_merged_dataset(df_input):
    df = df_input.copy()
    df['Sample'] = df['Sample'].astype(str)
    df['bending'] = df['Sample'].str.extract(r'_bending_([^_]+)')
    df['grid'] = df['Sample'].str.extract(r'_grid_([^_]+)')
    df['smoothing'] = df['Sample'].str.extract(r'_smoothing_([^_]+)')
    df['atlas'] = df['Sample'].str.extract(r'_atlas_([^_]+)')
    return df


regions = ["snpc"]#, "vta"]

for region in regions: 
    savedir_path = P(r"<IMAGING>\Ishrat\JoM\final_count_results")
    os.makedirs(savedir_path, exist_ok=True)
    # --- Load region data ---
    counts = pd.read_csv(P(rf"<IMAGING>\Ishrat\JoM\final_count_results\all_samples_cell_counts_summary_{region}_{processing_status}.csv"))
    counts['Sample'] = counts['Sample'].str.replace(r'_cell_counts_python?$', '', regex=True)
    counts["ID"] = counts["Sample"]
    counts["Sample"] = counts["MaskFolder"]
    counts = counts.drop(columns="Unnamed: 0", errors="ignore")
    #counts["ID"] = counts["Sample"]
    cell_count_cols = [col for col in counts.columns if col.startswith("CellCount_")]
    cell_vol_cols = [col for col in counts.columns if col.startswith("RegionVolume")]
    # Sum across those columns for each row
    counts["CellCount_RegionSum"] = counts[cell_count_cols].sum(axis=1)
    counts["CellCount_Total"] = counts["TotalCellCount"]
    counts["Volume_RegionSum"] = counts[cell_vol_cols].sum(axis=1)
    # Merge with volumes and voxel info
    combined_df_volumes = pd.merge(counts, df_volumes, on=["Sample", "ID"], how="left")
    combined_df_volumes_processed = preprocess_merged_dataset(combined_df_volumes)
    combined_df_volumes_processed = pd.merge(combined_df_volumes_processed, voxel_info, on="ID", how="left")
    
    # Convert volume to mm³
    combined_df_volumes_processed = combined_df_volumes_processed.rename(columns={
        "dataset": "Sample",
        "CellCount_RegionSum": f"{region}_CellCount",
        "TotalCellCount": f"TotalCellCount_{region}_Image",
        "Volume_RegionSum": f"Volume_RegionSum_{region}_px3",
        "full_brain": "full_brain_mm3"
    })
    
    combined_df_volumes_processed[f"Volume_RegionSum_{region}_mm3"] = (
        combined_df_volumes_processed[f"Volume_RegionSum_{region}_px3"] * combined_df_volumes_processed["voxel_vol_um3"]
    ) / 1e9
    
    # --- Save processed data ---
    final_processed_df = combined_df_volumes_processed[
        [
            'Sample', 'ID', 'bending', 'grid', 'smoothing', 'atlas',
            'voxel_vol_um3', f'TotalCellCount_{region}_Image', f'{region}_CellCount',
            f'Volume_RegionSum_{region}_px3', f'Volume_RegionSum_{region}_mm3',
            'full_brain_mm3'
        ]
    ]
    
    final_processed_df.to_csv(
        os.path.join(savedir_path, rf"final_processed_data_{region}_{processing_status}.csv"),
        index=False
    )
    
    # --- Summary stats ---
    df = final_processed_df
    summary = df.groupby("ID").agg({
        f"TotalCellCount_{region}_Image": ["mean", "median", "std"],
        f"{region}_CellCount": ["mean", "median", "std"],
        f"Volume_RegionSum_{region}_mm3": ["mean", "median", "std"],
        "full_brain_mm3": ["mean", "median", "std"],
    })
    
    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(
        os.path.join(savedir_path, rf"final_processed_data_{region}_{processing_status}_summary.csv"),
        index=False
    )
    
    df = pd.read_csv(os.path.join(savedir_path,rf"final_processed_data_{region}_{processing_status}.csv"))
    # --- Plotting ---
    plot_configs = [
        (df, f'{region}_CellCount', f'Cell Counts - {region}', 'Cell Count', f'{region}_counts_{processing_status}'),
        (df, f'TotalCellCount_{region}_Image', f'Total Cell Counts - Full {region} Image', 'Cell Count', f'total_counts_{region}_image_{processing_status}'),
        (df, f'Volume_RegionSum_{region}_px3', f'Volume - {region} (px³)', 'Volume (px³)', f'{region}_volume_px3_{processing_status}'),
        (df, f'Volume_RegionSum_{region}_mm3', f'Volume - {region} (mm³)', 'Volume (mm³)', f'{region}_volume_mm3_{processing_status}'),
        (
            df,
            df[f"{region}_CellCount"] / df[f"Volume_RegionSum_{region}_mm3"],
            f'{region} Density (cells/mm³)',
            'Density (cells/mm³)',
            f'{region}_density_mm3_{processing_status}'
        ),
        (
            df,
            df[f"Volume_RegionSum_{region}_mm3"] / df["full_brain_mm3"],
            f'{region} Proportion of Full Brain',
            'Proportion (mm³)',
            f'{region}_vol_prop_{processing_status}'
        ),
    ]
    
    for config in plot_configs:
        metric, title, ylabel, fname = config[1:]
        if isinstance(metric, pd.Series):
            temp_df = df.copy()
            temp_df["metric"] = metric
            plot_metric(
                temp_df,
                metric="metric",
                title=title,
                ylabel=ylabel,
                filename=P(f"<IMAGING>\Ishrat\JoM/final_count_results/plots/{fname}.png")
            )
        else:
            plot_metric(
                df,
                metric=metric,
                title=title,
                ylabel=ylabel,
                filename=P(f"<IMAGING>\Ishrat\JoM/final_count_results/plots/{fname}.png")
            )
