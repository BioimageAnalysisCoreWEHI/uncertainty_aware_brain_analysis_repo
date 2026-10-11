# -*- coding: utf-8 -*-
"""
region_utils.py

Shared helpers for resolving ARA2 (Allen Reference Atlas) region-tree IDs
into voxel masks / bounding boxes, and for running local SSIM QC against
those regions.

Used by both 03_compute_qc_metrics.py (batch) and run_qc_single.py
(per-job / HPC array task).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------------
# REGION TREE
# ---------------------------------------------------------------------

def load_region_tree(csv_path) -> pd.DataFrame:
    """
    Load the ARA2 region tree (id, parent_id, name, acronym, ...).
    """
    return pd.read_csv(csv_path)


def get_descendant_ids(
    region_id: int,
    tree_df: pd.DataFrame,
    include_self: bool = True,
) -> set[int]:
    """
    Recursively resolve a region id to the full set of atlas-volume ids
    that belong to it.

    - For a LEAF region (e.g. "Substantia nigra, compact part"), this is
      just {region_id}, since the atlas paints that id directly.
    - For a COMPOSITE/parent region (e.g. "Primary visual area"), the
      atlas never paints the parent's own id -- only its children/layers
      (VISp1, VISp2/3, ...) -- so this walks the whole subtree and
      returns every leaf id underneath it.

    This works identically regardless of where in the hierarchy
    `region_id` sits, which is the key fix for having a mix of leaf and
    higher-level regions.
    """
    ids = {region_id} if include_self else set()

    children = tree_df.loc[tree_df["parent_id"] == region_id, "id"].tolist()
    for child_id in children:
        ids |= get_descendant_ids(child_id, tree_df, include_self=True)

    return ids


# ---------------------------------------------------------------------
# REGION MASKS / BOUNDING BOXES
# ---------------------------------------------------------------------

def bbox_from_mask(mask: np.ndarray, padding: int = 5, min_size: int = 8) -> dict | None:
    """
    Compute a padded z/y/x bounding box around a boolean mask.

    `min_size` puts a floor on each dimension so the crop is always big
    enough for skimage's SSIM window (default win_size=7). Small/thin
    regions (e.g. a single cortical layer) would otherwise error out.

    Returns None if the mask is empty.
    """
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    mins = coords.min(axis=0).astype(int)
    maxs = coords.max(axis=0).astype(int) + 1

    mins = mins - padding
    maxs = maxs + padding

    for axis in range(3):
        size = maxs[axis] - mins[axis]
        if size < min_size:
            deficit = min_size - size
            mins[axis] -= deficit // 2
            maxs[axis] += deficit - deficit // 2

    mins = np.clip(mins, 0, None)
    maxs = np.minimum(maxs, np.array(mask.shape))

    return {
        "z0": int(mins[0]), "z1": int(maxs[0]),
        "y0": int(mins[1]), "y1": int(maxs[1]),
        "x0": int(mins[2]), "x1": int(maxs[2]),
    }


def split_mask_by_hemisphere(
    mask: np.ndarray,
    lr_axis: int,
    midline_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a boolean mask into two halves along the medio-lateral axis.

    Returns (mask_low, mask_high):
      - mask_low:  voxels with index < midline_index along lr_axis
      - mask_high: voxels with index >= midline_index along lr_axis

    Which physical side ("left" vs "right") corresponds to "low" vs
    "high" depends on your atlas's orientation convention -- verify
    once by eye against a known-asymmetric structure or the saved
    crops, then set roi.low_side_label / roi.high_side_label in the
    config accordingly.
    """
    shape_bc = [1] * mask.ndim
    shape_bc[lr_axis] = mask.shape[lr_axis]

    idx = np.arange(mask.shape[lr_axis]).reshape(shape_bc)

    mask_low = mask & (idx < midline_index)
    mask_high = mask & (idx >= midline_index)

    return mask_low, mask_high


def build_region_lookup(
    region_ids: list[int],
    tree_df: pd.DataFrame,
    annotation_volume: np.ndarray,
    padding: int = 5,
    split_hemispheres: bool = False,
    lr_axis: int = 2,
    midline_index: int | None = None,
    low_side_label: str = "right",
    high_side_label: str = "left",
) -> dict:
    """
    One-time precompute (do this ONCE per script run, not per-sample):
    for each requested region id, resolve its descendant ids, build a
    boolean mask against the atlas-space annotation volume, and derive
    padded bounding box(es) from that mask.

    `annotation_volume` MUST be the atlas ANNOTATION volume (region-id
    labels) in the same shape/space as `atlas.reference_image` /
    `downsampled_standard.tiff` -- NOT the per-sample registered_atlas.tiff.

    split_hemispheres:
      False (default) -- one bbox per region, covering both sides
        together. Simpler; a one-sided registration defect can get
        averaged out by a good result on the other side, so treat
        this as a starting point and check the SSIM distributions --
        if a region isn't discriminating well, that's the first thing
        to revisit.
      True -- mask is split in half along `lr_axis` at `midline_index`,
        producing one bbox per side (labels: low_side_label,
        high_side_label). Useful later for a "best of left / best of
        right" consensus rather than best-overall.

    Returns (structure is the same either way, so downstream code
    doesn't need to branch on this flag):
        {
          region_id: {
            "acronym", "name", "n_leaf_ids",
            "hemispheres": {
              "<side_label>": {"bbox": {...} or None},
              ...  # one entry ("both") if split_hemispheres=False,
                   # two entries otherwise
            }
          }
        }
    """
    if midline_index is None:
        midline_index = annotation_volume.shape[lr_axis] // 2

    lookup = {}

    for region_id in region_ids:
        row = tree_df.loc[tree_df["id"] == region_id]
        if row.empty:
            raise ValueError(f"Region id {region_id} not found in region tree CSV")

        name = row.iloc[0]["name"]
        acronym = row.iloc[0]["acronym"]

        descendant_ids = get_descendant_ids(region_id, tree_df, include_self=True)
        mask = np.isin(annotation_volume, list(descendant_ids))

        if not mask.any():
            raise ValueError(
                f"Region {region_id} ({name}) has zero voxels in annotation_volume "
                f"even after expanding to {len(descendant_ids)} descendant id(s). "
                f"Check that roi.annotation_image is the correct atlas/resolution "
                f"and matches atlas.reference_image."
            )

        if split_hemispheres:
            mask_low, mask_high = split_mask_by_hemisphere(mask, lr_axis, midline_index)
            hemispheres = {
                low_side_label: {"bbox": bbox_from_mask(mask_low, padding=padding)},
                high_side_label: {"bbox": bbox_from_mask(mask_high, padding=padding)},
            }
        else:
            hemispheres = {"both": {"bbox": bbox_from_mask(mask, padding=padding)}}

        lookup[region_id] = {
            "acronym": acronym,
            "name": name,
            "n_leaf_ids": len(descendant_ids),
            "hemispheres": hemispheres,
        }

        sides_str = " | ".join(f"{label} bbox={info['bbox']}" for label, info in hemispheres.items())
        print(f"  region {region_id} ({acronym} / {name}): {len(descendant_ids)} leaf id(s) | {sides_str}")

    return lookup


# ---------------------------------------------------------------------
# LOCAL SSIM
# ---------------------------------------------------------------------

def minmax_norm(arr: np.ndarray) -> np.ndarray:
    """
    Min-max normalise an array to 0-1.
    """
    arr = arr.astype(np.float32, copy=False)
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min + 1e-6)


def crop_bbox(arr: np.ndarray, bbox: dict) -> np.ndarray:
    """
    Crop an image using a z-y-x bounding box.
    """
    return arr[
        bbox["z0"]:bbox["z1"],
        bbox["y0"]:bbox["y1"],
        bbox["x0"]:bbox["x1"],
    ]


def compute_local_ssim(sample_img: np.ndarray, atlas_img: np.ndarray, bbox: dict) -> float:
    """
    Calculate SSIM between sample and atlas reference images, restricted
    to a single region's bounding box.
    """
    if bbox is None:
        return np.nan

    atlas_crop = minmax_norm(crop_bbox(atlas_img, bbox))
    sample_crop = minmax_norm(crop_bbox(sample_img, bbox))

    if atlas_crop.shape != sample_crop.shape:
        return np.nan

    return float(ssim(atlas_crop, sample_crop, data_range=1.0))


def apply_ssim_ranking(
    qc: pd.DataFrame,
    ssim_top_fraction: float,
    global_qc_column: str | None = "passes_global_qc",
    group_cols=("sample_id", "input_type"),
) -> pd.DataFrame:
    """
    Compute per-region-per-side SSIM rank/percentile/pass-fail columns.

    IMPORTANT: when global_qc_column is given (the default), ranking is
    computed ONLY among rows that already pass that column -- "top 50% of
    SSIM scores" means top 50% of the registrations that also passed
    global QC, not top 50% of everything. Rows that fail global_qc_column
    get passes_local_ssim_qc__* = False and NaN rank/percentile; they were
    never in the running.

    Pass global_qc_column=None to rank over every row regardless of global
    QC (the old behaviour) -- not recommended, kept only for cases where
    you deliberately want an unconditional ranking.

    Auto-discovers every "local_ssim__{suffix}" column already present in
    `qc` rather than needing a region_lookup passed in, so this can be
    called from anywhere a QC table has local_ssim columns in it -- the
    batch QC script, the HPC combine step, or a downstream consensus
    script. This is the ONE place ranking happens.
    """
    ssim_cols = [c for c in qc.columns if c.startswith("local_ssim__")]

    if global_qc_column is not None:
        eligible = qc[global_qc_column] == True  # noqa: E712
    else:
        eligible = pd.Series(True, index=qc.index)

    for col in ssim_cols:
        suffix = col[len("local_ssim__"):]
        rank_col = f"ssim_rank__{suffix}"
        pct_col = f"ssim_percentile__{suffix}"
        pass_col = f"passes_local_ssim_qc__{suffix}"

        qc[rank_col] = np.nan
        qc[pct_col] = np.nan
        qc[pass_col] = False

        grouped = qc.loc[eligible].groupby(list(group_cols))[col]

        qc.loc[eligible, rank_col] = grouped.rank(ascending=False, method="first")
        qc.loc[eligible, pct_col] = grouped.rank(ascending=True, pct=True)
        qc.loc[eligible, pass_col] = grouped.transform(
            lambda x: x >= x.quantile(1 - ssim_top_fraction)
        )

    return qc


def save_atlas_crops(atlas_img: np.ndarray, region_lookup: dict, out_dir) -> None:
    """
    Save the reference-atlas crop for every region x hemisphere, once.
    The atlas crop doesn't depend on the sample (same bbox, same atlas
    every time), so this is called once, not once per sample.

    Writes: {out_dir}/{acronym}_{side_label}__atlas_crop.tiff
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for region_id, info in region_lookup.items():
        for side_label, side_info in info["hemispheres"].items():
            bbox = side_info["bbox"]
            if bbox is None:
                continue
            atlas_crop = crop_bbox(atlas_img, bbox)
            tiff.imwrite(
                out_dir / f"{info['acronym']}_{side_label}__atlas_crop.tiff",
                atlas_crop,
            )


def save_sample_crop(
    sample_img: np.ndarray,
    bbox: dict,
    out_dir,
    prefix: str,
) -> None:
    """
    Save the raw (un-normalised) sample crop for a single region/sample,
    so you can compare it against the corresponding atlas crop from
    save_atlas_crops(). Gate this behind a flag when running at scale --
    unlike the atlas crop, this one is different for every sample.

    Writes: {out_dir}/{prefix}__sample_crop.tiff

    `prefix` should uniquely identify the sample/region, e.g.
    "sample01_dapi__VISp".
    """
    if bbox is None:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_crop = crop_bbox(sample_img, bbox)
    tiff.imwrite(out_dir / f"{prefix}__sample_crop.tiff", sample_crop)