# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:57:49 2026

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import os
import pandas as pd

# === CONFIG ===
parent_dirs = [
    P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_datasets\lightsheet_data\results\raw_tiff_stacks")
    
]

researcher_map = {
    P("<IMAGING>\Ishrat\JoM\post_review_analysis\validation_datasets\lightsheet_data\results\raw_tiff_stacks") : "lightsheet_validation"
    
}

batch_map = {
    P("<IMAGING>\Ishrat\JoM\post_review_analysis\validation_datasets\lightsheet_data\results\raw_tiff_stacks") : "batch_01"
}

output_csv = P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_datasets\lightsheet_data\metadata\analysis_manifest.csv")

rows = []

for parent_dir in parent_dirs:
    dims_csv = os.path.join(parent_dir, "001_image_dimensions_level1.csv")

    if os.path.exists(dims_csv):
        dims_df = pd.read_csv(dims_csv)
        dims_df["Folder"] = dims_df["Folder"].astype(str)
    else:
        print(f"Warning: no dimensions CSV found in {parent_dir}")
        dims_df = pd.DataFrame()

    for entry in sorted(os.scandir(parent_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue

        name = entry.name

        if "AF" not in name and "TH" not in name:
            continue

        animal_id = name.split("_")[0]

        if "AF" in name:
            #modality = "AF"
            channel_or_stain = "AF"
        elif "TH" in name:
            #modality = "TH"
            channel_or_stain = "TH"

        row = {
            "sample_id": name,
            "animal_id": animal_id,
            "researcher": researcher_map.get(parent_dir, ""),
            "batch": batch_map.get(parent_dir, ""),
            "group": "",
            "condition": "",
            "genotype": "",
            "treatment": "",
            "timepoint": "",
            #"modality": modality,
            "stain": channel_or_stain,
            "raw_stack_dir": entry.path,
            "include_in_combined_run": "yes",
            "exclude_reason": "",
            "notes": "",
        }

        # Append matching image dimensions
        if not dims_df.empty:
            match = dims_df[
                dims_df["Folder"]
                .astype(str)
                .str.contains(str(animal_id), case=False, na=False)
            ]

            if not match.empty:
                match = match.iloc[0]

                row["pixel_size_x"] = match.get("pixel_size_x", "")
                row["pixel_size_y"] = match.get("pixel_size_y", "")
                row["pixel_size_z"] = match.get("pixel_size_z", "")
                row["raw_x"] = match.get("x", "")
                row["raw_y"] = match.get("y", "")
                row["raw_z"] = match.get("z", "")
            else:
                row["pixel_size_x"] = ""
                row["pixel_size_y"] = ""
                row["pixel_size_z"] = ""
                row["raw_x"] = ""
                row["raw_y"] = ""
                row["raw_z"] = ""
                row["notes"] = f"No image dimensions match for animal_id={animal_id}"

        rows.append(row)

df = pd.DataFrame(rows)

df["raw_stack_dir"] = df["raw_stack_dir"].apply(os.path.normpath)

df["linux_raw_stack_dir"] = (
    df["raw_stack_dir"].map(to_cluster)
)

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)

print(f"Saved manifest with {len(df)} samples to:")
print(output_csv)

print("\nMissing pixel size rows:")
print(df[df["pixel_size_x"].isna() | (df["pixel_size_x"].astype(str) == "")][["sample_id", "animal_id", "raw_stack_dir", "notes"]])