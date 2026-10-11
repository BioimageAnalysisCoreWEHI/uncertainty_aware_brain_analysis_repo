# -*- coding: utf-8 -*-
"""Note on names: for this single-colour dataset the registration image is the labelled MARKER channel; file stems keep the historical "_AF_level1" name. "gt" here means the dataset authors' published (automated) registration, not a manual ground truth.


gt_utils.py

Shared helpers for validating the region consensus masks against the
independent ground-truth (GT) registered atlases (ScienceDB VTA projectome
dataset, DOI 10.57760/sciencedb.nb.00025).

GT atlases live in results/downsampled_25um_registered_atlas_ground_truth/
and are already on the same 25 um coronal-flipped grid as the consensus
masks (verified voxel-for-voxel for all 6 brains; no flip/transpose).

GT label values are NOT Allen IDs: value v is the 1-based rank of the
Allen structure ID among the atlas's 672 structures (0 = background).
Region masks are therefore built from
registration_label_mapping/allen_structure_to_registration_values.csv,
whose `registration_values` column already includes every descendant and
catch-all label at or below a structure. See
registration_label_mapping/README.md.

Used by 07a/07b/07c/07d/08 in this folder.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tifffile as tiff
import yaml
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from region_utils import load_region_tree, get_descendant_ids, bbox_from_mask  # noqa: E402
from pipeline_common import safe_value  # noqa: E402


# ---------------------------------------------------------------------
# PATHS / DATASET LAYOUT
# ---------------------------------------------------------------------

CONFIG_PATH = ROOT / "config" / "config.yaml"
CONSENSUS_ROOT = ROOT / "results" / "region_consensus"
GT_DIR = ROOT / "results" / "downsampled_25um_registered_atlas_ground_truth"
IMAGE_DIR = ROOT / "results" / "downsampled_25um_images"
MAPPING_DIR = ROOT / "registration_label_mapping"
OUT_DIR = ROOT / "results" / "validation_gt"

SAMPLES = ["GAD2", "Rep2-GAD2", "TH", "Rep2-TH", "vGlut2", "Rep2-vGlut2"]
INPUT_TYPES = ["raw", "preprocessed"]
SIDE = "both"
VOXEL_UM = 25.0
VOXEL_MM3 = (VOXEL_UM / 1000.0) ** 3


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


CFG = load_config()
THRESHOLDS = [float(t) for t in CFG["consensus"]["thresholds"]]
HEADLINE_THRESHOLD = 0.5
FINAL_QC_SUBSET = f"passes_final_qc_top_{safe_value(CFG['qc']['ssim_top_fraction'])}"
SUBSETS = ["all_registrations", "passes_global_qc", FINAL_QC_SUBSET]


def region_table() -> pd.DataFrame:
    """The 16 regions of interest (config roi.region_ids), in config order,
    with their acronyms from the ARA2 region tree."""
    tree = load_region_tree(ROOT / CFG["roi"]["region_tree_csv"])
    by_id = tree.set_index("id")
    ids = [int(i) for i in CFG["roi"]["region_ids"]]
    return pd.DataFrame({
        "region_id": ids,
        "acronym": [by_id.loc[i, "acronym"] for i in ids],
        "name": [by_id.loc[i, "name"] for i in ids],
    })


def region_dir(sample_id, input_type, acronym, subset) -> Path:
    return CONSENSUS_ROOT / sample_id / input_type / f"{acronym}_{SIDE}" / subset


def consensus_path(sample_id, input_type, acronym, subset, threshold) -> Path:
    return (
        region_dir(sample_id, input_type, acronym, subset)
        / f"consensus_thresh_{safe_value(threshold)}.tif"
    )


def gt_path(sample_id) -> Path:
    return GT_DIR / f"{sample_id}_AF_level1_registered_atlas_25um_coronal_flipped.tif"


def image_path(sample_id) -> Path:
    stem = f"{sample_id}_AF_level1"
    return IMAGE_DIR / stem / f"{stem}_downsampled_25um_coronal_flipped.tif"


# ---------------------------------------------------------------------
# GT ATLAS / REGION MASKS
# ---------------------------------------------------------------------

def load_gt_atlas(sample_id) -> np.ndarray:
    gt = read_tiff(gt_path(sample_id))
    if gt.dtype != np.uint16 or gt.max() > 672:
        raise ValueError(
            f"{gt_path(sample_id).name}: expected uint16 registration ranks 0-672, "
            f"got {gt.dtype} with max {gt.max()} (interpolated or wrong file?)"
        )
    return gt


def load_label_mapping() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(value -> Allen ID lut, Allen structure -> registration values)."""
    lut = pd.read_csv(MAPPING_DIR / "registration_value_to_allen_id.csv")
    agg = pd.read_csv(
        MAPPING_DIR / "allen_structure_to_registration_values.csv",
        keep_default_na=False,
    )
    return lut, agg


def gt_region_values(acronym, agg=None) -> list[int]:
    if agg is None:
        _, agg = load_label_mapping()
    row = agg.loc[agg["acronym"] == acronym]
    if len(row) != 1:
        raise KeyError(f"{acronym!r} not found exactly once in the aggregate mapping table")
    values = str(row["registration_values"].item())
    return [int(v) for v in values.split(";") if v != ""]


def gt_region_mask(gt, acronym, agg=None) -> np.ndarray:
    return np.isin(gt, gt_region_values(acronym, agg))


def read_tiff(path, attempts=4, wait_s=5):
    """tiff.imread with retries: reads from the V: network share
    occasionally fail transiently (OSError: [Errno 22] Invalid argument)."""
    import time

    for attempt in range(1, attempts + 1):
        try:
            return tiff.imread(path)
        except OSError:
            if attempt == attempts:
                raise
            print(f"  read failed ({attempt}/{attempts}), retrying: {path}", flush=True)
            time.sleep(wait_s)


def load_mask(path, expected_shape) -> np.ndarray:
    mask = read_tiff(path) != 0
    if mask.shape != tuple(expected_shape):
        raise ValueError(f"{path}: shape {mask.shape} != GT shape {tuple(expected_shape)}")
    return mask


def n_registrations(sample_id, input_type, acronym, subset) -> int:
    csv = region_dir(sample_id, input_type, acronym, subset) / "registrations_used.csv"
    return len(pd.read_csv(csv, usecols=["sample_id"]))


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def dice(a, b):
    total = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / total if total > 0 else np.nan


def _surface(mask):
    return mask & ~ndimage.binary_erosion(mask, border_value=0)


def surface_distances(pred, gt, spacing=VOXEL_UM):
    """Directed surface distances (um): pred surface -> nearest GT surface,
    and GT surface -> nearest pred surface. Computed on a padded crop around
    the union so distance transforms stay cheap."""
    bbox = bbox_from_mask(pred | gt, padding=3, min_size=1)
    sl = np.s_[bbox["z0"]:bbox["z1"], bbox["y0"]:bbox["y1"], bbox["x0"]:bbox["x1"]]
    p, g = pred[sl], gt[sl]
    sp, sg = _surface(p), _surface(g)
    dist_to_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dist_to_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    return dist_to_g[sp], dist_to_p[sg]


def seg_metrics(pred, gt, spacing=VOXEL_UM) -> dict:
    pred = pred.astype(bool, copy=False)
    gt = gt.astype(bool, copy=False)

    n_pred, n_gt = int(pred.sum()), int(gt.sum())
    tp = int(np.logical_and(pred, gt).sum())
    union = n_pred + n_gt - tp

    out = {
        "dice": 2 * tp / (n_pred + n_gt) if (n_pred + n_gt) else np.nan,
        "iou": tp / union if union else np.nan,
        "sensitivity": tp / n_gt if n_gt else np.nan,
        "precision": tp / n_pred if n_pred else np.nan,
        "vol_pred_mm3": n_pred * VOXEL_MM3,
        "vol_gt_mm3": n_gt * VOXEL_MM3,
        "vol_ratio": n_pred / n_gt if n_gt else np.nan,
        "hd95_um": np.nan,
        "assd_um": np.nan,
        "centroid_dist_um": np.nan,
    }
    if n_pred == 0 and n_gt > 0:
        out["dice"] = 0.0
        out["iou"] = 0.0

    if n_pred and n_gt:
        d_pg, d_gp = surface_distances(pred, gt, spacing)
        # Symmetric HD95 (max of the two directed 95th percentiles), as in MONAI.
        out["hd95_um"] = float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))
        out["assd_um"] = float(np.concatenate([d_pg, d_gp]).mean())
        c_pred = np.array(ndimage.center_of_mass(pred))
        c_gt = np.array(ndimage.center_of_mass(gt))
        out["centroid_dist_um"] = float(np.linalg.norm((c_pred - c_gt) * spacing))

    return out


# ---------------------------------------------------------------------
# PARALLEL HELPERS
# ---------------------------------------------------------------------

def groups():
    return [(s, t) for s in SAMPLES for t in INPUT_TYPES]


def _cached_group(worker, cache_dir, sample_id, input_type):
    """Run one group, or reuse its result if an earlier run already saved it."""
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{sample_id}__{input_type}.csv"
        if path.exists():
            print(f"cached: {sample_id} / {input_type}", flush=True)
            return pd.read_csv(path)
    df = worker(sample_id, input_type)
    if path is not None:
        tmp = path.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(path)
    return df


def run_groups(worker, n_workers=None, cache_name=None, fresh=False):
    """Run worker(sample_id, input_type) -> DataFrame over all 12 groups.

    With cache_name, each finished group is saved to
    results/validation_gt/_cache/<cache_name>/ so a crashed run (e.g. a
    network read error) resumes instead of starting over. fresh=True
    clears that cache first.
    """
    import shutil
    from multiprocessing import Pool

    cache_dir = None
    if cache_name is not None:
        cache_dir = OUT_DIR / "_cache" / cache_name
        if fresh and cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(worker, cache_dir, s, t) for s, t in groups()]
    n_workers = n_workers or min(len(jobs), 6)
    if n_workers == 1:
        frames = [_cached_group(*job) for job in jobs]
    else:
        with Pool(n_workers) as pool:
            frames = pool.starmap(_cached_group, jobs)
    return pd.concat(frames, ignore_index=True)
