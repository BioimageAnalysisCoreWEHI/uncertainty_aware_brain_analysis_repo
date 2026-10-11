# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:22:05 2026

@author: zaman.i
"""

# -*- coding: utf-8 -*-
"""
downsample_staple_masks.py

Just the scaling step: takes STAPLE ground-truth .nii mask(s) -- built
at native-pixel-size/scale resolution (e.g. 1um native / 0.25 scale =
4um/px, matching build_staple_ground_truth.py's default) -- and
resamples them down to a target resolution (e.g. your pipeline's
25um), saving as .tif so you can quickly open and look at them.

No reorientation or alignment here -- just resolution matching, so you
can see what the ground truth looks like at your own working scale.

Usage (single file):
    python downsample_staple_masks.py \\
        --input results/staple_ground_truth/staple/CR_Syt6CreRfp_35MV_0.25_NiftySegSTAPLE.nii \\
        --out-dir results/staple_ground_truth/downsampled_25um

Usage (whole folder of STAPLE .nii files):
    python downsample_staple_masks.py \\
        --input-dir results/staple_ground_truth/staple \\
        --out-dir results/staple_ground_truth/downsampled_25um
"""
from pathlib import Path
import argparse

import numpy as np
import nibabel as nib
import tifffile as tiff
from skimage.transform import resize


def downsample_staple_mask(data: np.ndarray, scale: float, native_um: float, target_um: float) -> np.ndarray:
    """data is the STAPLE mask at (native_um/scale) resolution.
    Returns a binary mask resampled to target_um resolution."""
    factor = (native_um / scale) / target_um
    target_shape = tuple(round(s * factor) for s in data.shape)
    resized = resize(data, target_shape, order=0, preserve_range=True, anti_aliasing=False)
    return (resized > 0.5).astype(np.uint8)


def process_one(nii_path: Path, out_dir: Path, scale: float, native_um: float, target_um: float):
    img = nib.load(str(nii_path))
    data = img.get_fdata()

    downsampled = downsample_staple_mask(data, scale, native_um, target_um)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (nii_path.stem.replace(".nii", "") + f"_{target_um}um.tif")
    tiff.imwrite(out_path, downsampled)

    print(f"{nii_path.name}: {data.shape} @ {native_um/scale}um -> {downsampled.shape} @ {target_um}um "
          f"({downsampled.sum()} foreground px) -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", help="A single STAPLE .nii file")
    parser.add_argument("--input-dir", help="A directory containing STAPLE .nii files (matches *NiftySegSTAPLE*.nii)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scale", type=float, default=0.25, help="Downscale factor used when building the STAPLE masks")
    parser.add_argument("--native-um", type=float, default=1.0, help="Native raw-photomicrograph pixel size in microns")
    parser.add_argument("--target-um", type=float, default=25.0, help="Target resolution in microns")
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Provide either --input or --input-dir")

    out_dir = Path(args.out_dir)

    if args.input:
        process_one(Path(args.input), out_dir, args.scale, args.native_um, args.target_um)
    else:
        nii_files = sorted(Path(args.input_dir).glob("*NiftySegSTAPLE*.nii"))
        if not nii_files:
            raise FileNotFoundError(f"No *NiftySegSTAPLE*.nii files found in {args.input_dir}")
        print(f"Found {len(nii_files)} STAPLE file(s)")
        for f in nii_files:
            process_one(f, out_dir, args.scale, args.native_um, args.target_um)


if __name__ == "__main__":
    main()