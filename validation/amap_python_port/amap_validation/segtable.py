"""Port of helpers/getSegmentationSessionData.m: one row per manual segmentation TIFF."""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


def seg_idx(name: str) -> float:
    m = re.search(r"Seg(\d+)", name)
    return float(m.group(1)) if m else np.nan


def brain_name(name: str) -> str:
    # regexp(fileName,'[0-9][^0-9]') -> first digit/non-digit transition; up to '.tif'; drop last '_token'.
    start = re.search(r"[0-9][^0-9]", name).start() + 1
    end = name.index(".tif")
    stem = name[start:end]
    return stem[: stem.rindex("_")]


def region_name(name: str) -> str:
    start = re.search(r"\d\d\d-", name).start() + 4
    rest = name[start:]
    return rest[: rest.index("_")]


def slice_num(name: str) -> int:
    return int(re.search(r"Slice(\d+)\.", name).group(1))


def _duration(folder: Path, tif_name: str) -> float:
    # getDuration looks for '<tif name minus ".tif">_meta.txt' (with or without the doubled dot).
    base = tif_name[:-4]
    for cand in (base + "_meta.txt", base.rstrip(".") + "_meta.txt"):
        p = folder / cand
        if p.exists():
            lines = p.read_text(errors="replace").splitlines()
            for i, line in enumerate(lines):
                if line.strip() == "#duration" and i + 1 < len(lines):
                    try:
                        return float(lines[i + 1])
                    except ValueError:
                        return np.nan
    return np.nan


def build_segtable(seg_dir: os.PathLike) -> pd.DataFrame:
    """Table of all segmentations, ordered like MATLAB's dir() (sorted user dir, sorted file name)."""
    seg_dir = Path(seg_dir)
    rows = []
    for user_dir in sorted(p for p in seg_dir.iterdir() if p.is_dir()):
        for f in sorted(os.listdir(user_dir)):
            if not f.endswith(".tif") or f.startswith("._"):
                continue
            rows.append(dict(
                user=user_dir.name,
                idx=seg_idx(f),
                brain=brain_name(f),
                region=region_name(f),
                sliceNum=slice_num(f),
                duration=_duration(user_dir, f),
                path=str(user_dir / f),
            ))
    t = pd.DataFrame(rows)
    t["brainRegionComb"] = t["brain"] + t["region"]
    # Presentation order within a rater distinguishes first attempt (1) from the blind repeat (2).
    t["attempt"] = t.groupby(["user", "brainRegionComb"])["idx"].rank(method="first").astype(int)
    return t
