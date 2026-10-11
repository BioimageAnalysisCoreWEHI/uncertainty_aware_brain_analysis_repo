# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 13:46:31 2025

@author: zaman.i
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import pandas as pd
import glob
import os

base_dir = P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_datasets\lightsheet_data\results\raw_tiff_stacks")
files = [f for f in os.listdir(base_dir) if "metadata" in f]

dfs = [pd.read_csv(os.path.join(base_dir,f)) for f in files]
merged = pd.concat(dfs, ignore_index=True)
merged["filename"] = merged["filename"].str.split("_").str[0]
merged = merged.rename(columns={"filename": "Folder"})
merged.to_csv(os.path.join(base_dir, "001_image_dimensions_level1.csv"), index=False)

print(f"Merged {len(files)} metadata files.")