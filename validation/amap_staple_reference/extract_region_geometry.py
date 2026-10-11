# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 10:31:35 2026

@author: zaman.i
"""

# -*- coding: utf-8 -*-
"""
extract_region_geometry.py

Step 3 of the aMAP_validation ground-truth workflow: pull the per-
(brain, region) geometry metadata out of aMAP_validation's
data/brainShifts.yml, and key it by Allen acronym so it lines up with
your own pipeline's region naming (MV, SUB, DG-sg, RSP, VISp, VISam,
VPM, AHN, SSp, ACA -- SNc isn't in their validated set).

This metadata is exactly what classes/pavelAtlas.m's
makeAtlasSegStack() used to resample their registered atlas into raw-
photomicrograph space (see zRange + shift + width/height usage there).
You'll use it -- inverted, and composed with your own 12.5um -> 25um
scale change -- to warp the STAPLE ground truth into your own
consensus volume's space instead of going the direction they did.

Region-tag cross-reference
---------------------------
Three different naming conventions exist across the repo's own files,
and none of them is your pipeline's Allen acronym directly:

  1. brainShifts.yml "section_name"  (e.g. "sub", "rspV1V2M", "adt")
  2. the region tag baked into each raw manual-segmentation TIFF's
     filename (e.g. "...001-SUB_Slice40..tif" -> "SUB")
  3. globalData.yml's regionsUsed "allenName" field

REGION_LOOKUP below was built by cross-referencing all three directly
against the real files in data/userSegmentations.tar.bz2 and
data/globalData.yml -- not guessed. Note that one section_name
("rspV1V2M") covers THREE Allen regions at once (RSP, VISp, VISam),
because all three are visible/segmented together in that same
posterior-cortex z-range and share one shift/zRange/canvas entry.

Caveat on zRange
-----------------
zRange indexes into dimension 2 of THEIR registered atlas volume
(their 12.5um `registered_atlas.tiff`-equivalent) -- see
pavelAtlas.makeAtlasSegStack's `obj.atlas.img(:,atlasPosList,:)`. It is
NOT an index into the raw photomicrograph stack directly. Before using
z0/z1 against your own registered_atlas.tiff, confirm your volume uses
the same axis order/orientation (their code explicitly permutes/flips
axes when going back to "TissueVision" space) -- do the visual sanity
check from the plan before trusting this blindly.
"""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import yaml


# raw_tag (from TIFF filenames) : (section_name in brainShifts.yml, Allen acronym)
REGION_LOOKUP = {
    "MV":   ("MV",       "MV"),
    "SUB":  ("sub",      "SUB"),
    "SG":   ("sg",       "DG-sg"),
    "RSP":  ("rspV1V2M", "RSP"),
    "V1":   ("rspV1V2M", "VISp"),
    "V2M":  ("rspV1V2M", "VISam"),
    "VPM":  ("vpm",      "VPM"),
    "AHN":  ("adt",      "AHN"),
    "S1BF": ("s1bf",     "SSp"),
    "CING": ("cingul",   "ACA"),
}


def parse_shift_matrix(shift_field) -> np.ndarray:
    """brainShifts.yml stores the affine as a MATLAB-style string, e.g.
    '1 0.00011 ...;-0.00011 1 ...;...;0 0 0 1'. PyYAML parses the
    surrounding `[...]` as a one-element list (no commas inside), so
    unwrap that first."""
    s = shift_field[0] if isinstance(shift_field, list) else shift_field
    rows = [row.strip() for row in s.split(";") if row.strip()]
    matrix = np.array([[float(v) for v in row.split()] for row in rows])
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 affine, got shape {matrix.shape} from: {s!r}")
    return matrix


def parse_z_range(z_field) -> tuple[int, int]:
    s = z_field[0] if isinstance(z_field, list) else z_field
    z0, z1 = (int(v) for v in str(s).split())
    return z0, z1


def load_geometry(brain_shifts_yml_path) -> dict:
    """Returns {(brain, allen_acronym): {section_name, z0, z1, shift, width, height}}"""
    with open(brain_shifts_yml_path, "r") as f:
        data = yaml.safe_load(f)

    geometry = {}
    for brain_entry in data["brainShifts"]:
        brain = brain_entry["name"]
        for shift_entry in brain_entry["shifts"]:
            section_name = shift_entry["name"]
            # A section_name like "rspV1V2M" maps to more than one Allen region.
            acronyms = [acro for raw, (sec, acro) in REGION_LOOKUP.items() if sec == section_name]
            if not acronyms:
                print(f"[WARN] No REGION_LOOKUP entry for section_name={section_name!r} (brain={brain}); skipping.")
                continue

            z0, z1 = parse_z_range(shift_entry["zRange"])
            shift = parse_shift_matrix(shift_entry["shift"])

            for acronym in acronyms:
                geometry[(brain, acronym)] = {
                    "section_name": section_name,
                    "z0": z0,
                    "z1": z1,
                    "shift": shift,
                    "width": shift_entry["width"],
                    "height": shift_entry["height"],
                }
    return geometry


def geometry_to_dataframe(geometry: dict) -> pd.DataFrame:
    rows = []
    for (brain, acronym), info in geometry.items():
        row = {
            "brain": brain,
            "region": acronym,
            "section_name": info["section_name"],
            "z0": info["z0"],
            "z1": info["z1"],
            "width": info["width"],
            "height": info["height"],
            "shift_flat": ";".join(str(v) for v in info["shift"].flatten()),
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["brain", "region"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brain-shifts-yml", default="aMAP_validation/data/brainShifts.yml")
    parser.add_argument("--out-csv", default="results/region_geometry.csv")
    parser.add_argument("--brain", default=None, help="Optional: only show this brain, e.g. MV_Ntsr1_169")
    args = parser.parse_args()

    geometry = load_geometry(args.brain_shifts_yml)
    df = geometry_to_dataframe(geometry)

    if args.brain:
        df = df[df["brain"] == args.brain]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(df.to_string(index=False))
    print(f"\nSaved {len(df)} (brain, region) geometry rows to: {out_path}")
    print("shift_flat is the 4x4 affine, row-major, ';'-joined -- reshape with "
          "np.array([float(v) for v in s.split(';')]).reshape(4, 4)")


if __name__ == "__main__":
    main()