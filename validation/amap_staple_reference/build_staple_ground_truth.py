# -*- coding: utf-8 -*-
"""
build_staple_ground_truth.py

Step 2 of the aMAP_validation ground-truth workflow: parse the raw
human-rater segmentation TIFFs you've already extracted from
data/userSegmentations.tar.bz2 into per-(brain, region) 4D NIfTI
stacks, and fuse them into a single STAPLE consensus segmentation per
(brain, region) using the bundled NiftySeg seg_LabFusion binary
(nonMatlab/niftyseg_build/seg-apps/seg_LabFusion -- runs directly on
Linux, no rebuild needed).

Mirrors, in Python, the logic of:
  - helpers/getSegmentationSessionData.m  (filename parsing)
  - classes/userRater.m: create4dNii()    (per-rater binarize/resize/stack)
  - classes/userRater.m: runStaple()      (seg_LabFusion -STAPLE call)

Filename convention (as produced by the original ImageJ ROI export --
note the filename contains ".tif" TWICE, once from the original raw
section image name and once as the true extension):
    Seg<idx><originalRawImageName>.tif<NNN>-<REGION>_Slice<N>..tif
e.g. Seg39MV_Ntsr1_165_sub.tif001-SUB_Slice40..tif
    -> idx=39, brain="MV_Ntsr1_165", region="SUB", sliceNum=40

Region tag -> Allen acronym mapping matches extract_region_geometry.py
exactly (both were built from the same file/data cross-reference) --
if you add regions, update both files together.

Usage:
    python build_staple_ground_truth.py \\
        --seg-dir /path/to/extracted/userSegmentations \\
        --seg-lab-fusion-bin /path/to/aMAP_validation/nonMatlab/niftyseg_build/seg-apps/seg_LabFusion \\
        --brain-shifts-yml /path/to/aMAP_validation/data/brainShifts.yml \\
        --out-dir results/staple_ground_truth \\
        --scale 0.25
"""
from pathlib import Path
import argparse
import re
import subprocess

import numpy as np
import pandas as pd
import tifffile as tiff
import nibabel as nib
from skimage.transform import resize

from extract_region_geometry import REGION_LOOKUP, load_geometry


# ---------------------------------------------------------------------
# FILENAME PARSING (mirrors helpers/getSegmentationSessionData.m)
# ---------------------------------------------------------------------

def get_seg_idx(fname: str):
    m = re.search(r"Seg(\d+)", fname)
    return int(m.group(1)) if m else None


def get_brain_name(fname: str) -> str:
    m = re.search(r"[0-9][^0-9]", fname)
    if not m:
        raise ValueError(f"Could not parse brain name from: {fname}")
    start = m.start() + 1
    end = fname.index(".tif")  # FIRST '.tif' -- that's the embedded original filename's extension
    name = fname[start:end]
    last_us = name.rfind("_")
    return name[:last_us]


def get_region_name(fname: str) -> str:
    m = re.search(r"\d{3}-", fname)
    if not m:
        raise ValueError(f"Could not parse region name from: {fname}")
    rest = fname[m.end():]
    end = rest.find("_")
    return rest[:end] if end != -1 else rest


def get_slice_num(fname: str) -> int:
    m = re.search(r"Slice(\d+)\.", fname)
    if not m:
        raise ValueError(f"Could not parse slice number from: {fname}")
    return int(m.group(1))


def is_segmentation_tif(path: Path) -> bool:
    """Real per-slice segmentation TIFFs only -- excludes _meta.txt, .roi,
    and macOS resource-fork artifacts (._*, .DS_Store)."""
    if path.name.startswith("."):
        return False
    return path.suffix.lower() == ".tif"


# ---------------------------------------------------------------------
# BUILD SEGMENTATION TABLE (one row per rater segmentation file)
# ---------------------------------------------------------------------

def build_segmentation_table(seg_dir: Path) -> pd.DataFrame:
    rows = []
    n_skipped = 0
    for user_dir in sorted(p for p in seg_dir.iterdir() if p.is_dir()):
        for f in sorted(user_dir.iterdir()):
            if not is_segmentation_tif(f):
                continue
            try:
                raw_tag = get_region_name(f.name)
                rows.append({
                    "user": user_dir.name,
                    "idx": get_seg_idx(f.name),
                    "brain": get_brain_name(f.name),
                    "raw_tag": raw_tag,
                    "region": REGION_LOOKUP.get(raw_tag, (None, raw_tag))[1],  # Allen acronym
                    "sliceNum": get_slice_num(f.name),
                    "path": str(f),
                })
            except ValueError as e:
                n_skipped += 1
                print(f"[WARN] Skipping unparsable file: {f.name} ({e})")

    if n_skipped:
        print(f"[WARN] {n_skipped} file(s) could not be parsed and were skipped.")

    table = pd.DataFrame(rows)
    if not table.empty:
        unmapped = sorted(set(table.loc[~table["raw_tag"].isin(REGION_LOOKUP), "raw_tag"]))
        if unmapped:
            print(f"[WARN] Region tag(s) with no REGION_LOOKUP entry (kept as-is): {unmapped}")

    return table


# ---------------------------------------------------------------------
# BUILD 4D STACK + RUN STAPLE (mirrors userRater.create4dNii/runStaple)
# ---------------------------------------------------------------------

def load_binary_resized(path: str, canvas_width: int, canvas_height: int, scale: float) -> np.ndarray:
    """Load one rater's binary TIFF, place it on the full-section canvas,
    transpose to (width, height) as the MATLAB code does
    (`imread(...)'`), binarize, and nearest-neighbour resize by `scale`.
    Returns a 2D uint8 array of shape
    (round(canvas_width*scale), round(canvas_height*scale))."""
    img = tiff.imread(path)  # (height, width), same as MATLAB's raw imread
    img_t = img.T            # -> (width, height)

    canvas = np.zeros((canvas_width, canvas_height), dtype=np.uint8)
    h = min(canvas_width, img_t.shape[0])
    w = min(canvas_height, img_t.shape[1])
    canvas[:h, :w] = (img_t[:h, :w] > 0).astype(np.uint8)

    if scale == 1.0:
        return canvas  # no-op: skip resize entirely, avoids needless interpolation cost at native resolution

    out_shape = (round(canvas_width * scale), round(canvas_height * scale))
    resized = resize(canvas, out_shape, order=0, preserve_range=True, anti_aliasing=False)
    return (resized > 0).astype(np.uint8)


def run_staple_for_group(
    brain: str,
    region: str,
    group: pd.DataFrame,
    canvas_width: int,
    canvas_height: int,
    scale: float,
    seg_lab_fusion_bin: Path,
    out_dir: Path,
    overwrite: bool = False,
):
    combo = f"{brain}{region}"

    staple_dir = out_dir / "staple"
    staple_dir.mkdir(parents=True, exist_ok=True)
    out_path = staple_dir / f"{combo}_{scale}_NiftySegSTAPLE.nii"
    log_path = staple_dir / f"{combo}_{scale}_NiftySegSTAPLE_log.txt"

    if out_path.exists() and not overwrite:
        print(f"[SKIP] Already exists: {out_path}")
        return

    print(f"[{combo}] {len(group)} rater segmentation(s)")

    slices = [
        load_binary_resized(row.path, canvas_width, canvas_height, scale)
        for row in group.itertuples()
    ]
    stack = np.stack(slices, axis=-1)           # (X, Y, N) -- N = number of rater segmentations
    stack = stack[:, :, np.newaxis, :]           # (X, Y, 1, N) -- seg_LabFusion wants a 4D label image

    pix_xy = 1.0 / scale  # native TIFFs are ~1um/px per the project wiki; scale=0.25 -> 4um/px
    affine = np.diag([pix_xy, pix_xy, 1.0, 1.0])

    nii_dir = out_dir / "nii4d"
    nii_dir.mkdir(parents=True, exist_ok=True)
    nii_4d_path = nii_dir / f"{combo}_{scale}_4DSegMathsMerged.nii.gz"
    nib.save(nib.Nifti1Image(stack.astype(np.uint8), affine), str(nii_4d_path))

    cmd = [str(seg_lab_fusion_bin), "-in", str(nii_4d_path), "-STAPLE", "-unc", "-out", str(out_path), "-v", "2"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(result.stdout + "\n" + result.stderr + "\n" + " ".join(cmd) + "\n"
                         + ";".join(f"{u}:{i}" for u, i in zip(group["user"], group["idx"])))

    if result.returncode != 0 or not out_path.exists():
        print(f"[ERROR] seg_LabFusion failed for {combo} (see {log_path}):\n{result.stderr[-500:]}")
    else:
        print(f"[OK] STAPLE ground truth -> {out_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seg-dir", required=True, help="Path to extracted userSegmentations/ (contains User_XX/ subfolders)")
    parser.add_argument("--seg-lab-fusion-bin", required=True, help="Path to the seg_LabFusion binary")
    parser.add_argument("--brain-shifts-yml", default=None,
                         help="Optional. No longer used for canvas sizing (brainShifts.yml's width/height do not "
                              "reliably match the real raw TIFF dimensions -- confirmed against real data, where "
                              "some regions' recorded width/height were smaller than the actual TIFFs, which would "
                              "silently truncate real ROI content if used as the canvas size). Canvas size is now "
                              "always read directly from the raw TIFFs. Passing this just enables a one-line note "
                              "confirming that substitution happened, for each (brain, region) that has a geometry "
                              "entry.")
    parser.add_argument("--out-dir", default="results/staple_ground_truth")
    parser.add_argument("--scale", type=float, default=1.0,
                         help="Downscale factor applied before STAPLE fusion. Default 1.0 = native resolution "
                              "(recommended): fuse at full precision, then downsample once, directly, to your "
                              "target resolution afterwards with downsample_staple_masks.py. The original paper's "
                              "own config used 0.25, but doing that AND a later downsample to 25um compounds two "
                              "separate nearest-neighbour resamplings unnecessarily -- pass --scale 0.25 only if "
                              "you specifically want to reproduce their exact intermediate representation.")
    parser.add_argument("--brain", default=None, help="Optional: only process this brain, e.g. MV_Ntsr1_169")
    parser.add_argument("--region", default=None, help="Optional: only process this Allen-acronym region, e.g. MV")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    seg_dir = Path(args.seg_dir)
    out_dir = Path(args.out_dir)
    seg_lab_fusion_bin = Path(args.seg_lab_fusion_bin)

    if not seg_lab_fusion_bin.exists():
        raise FileNotFoundError(f"seg_LabFusion binary not found: {seg_lab_fusion_bin}")

    table = build_segmentation_table(seg_dir)
    print(f"Parsed {len(table)} rater segmentation files across "
          f"{table['brain'].nunique()} brain(s) and {table['region'].nunique()} region(s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "segmentation_table.csv", index=False)

    if args.brain:
        table = table[table["brain"] == args.brain]
    if args.region:
        table = table[table["region"] == args.region]

    geometry = load_geometry(args.brain_shifts_yml) if args.brain_shifts_yml else None

    for (brain, region), group in table.groupby(["brain", "region"]):
        if geometry is not None and (brain, region) in geometry:
            print(f"[NOTE] {brain}/{region}: ignoring brainShifts.yml's width/height for canvas sizing -- "
                  f"confirmed these don't match the real raw TIFF dimensions for at least some regions "
                  f"(e.g. RSP), and using an undersized canvas would silently truncate real ROI data. "
                  f"Using the actual TIFF dimensions instead.")

        heights, widths = [], []
        for p in group["path"]:
            img = tiff.imread(p)
            heights.append(img.shape[0])
            widths.append(img.shape[1])
        canvas_width, canvas_height = max(widths), max(heights)

        run_staple_for_group(
            brain, region, group, canvas_width, canvas_height, args.scale,
            seg_lab_fusion_bin, out_dir, overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()