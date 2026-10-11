"""Configuration: the YAML files in data/ plus constants hard-coded in the MATLAB classes."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

# Root of a clone of https://github.com/SainsburyWellcomeCentre/aMAP_validation (holds data/ and downscaledBrains/).
# Set AMAP_VALIDATION_ROOT when this package is not inside that clone (e.g. in the uncertainty-aware analysis repo).
REPO_ROOT = Path(os.environ.get("AMAP_VALIDATION_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = REPO_ROOT / "data"
DEFAULT_SEG_DIR = DATA_DIR / "userSegmentations" / "userSegmentations"
DEFAULT_BRAIN_DIR = REPO_ROOT / "downscaledBrains"
DEFAULT_OUT_DIR = REPO_ROOT.parent / "aMAP_validation_python_outputs"

# segAnalyser.removeUnsegmentedHemisphere (segAnalyser.m:68-86): 1-based nii x index; x <= midline is zeroed.
MIDLINES = {
    "MV_Ntsr1_165": 551,
    "MV_Ntsr1_169": 522,
    "CR_Syt6CreRfp_35": 503,
    "ER_Glt25d2Cre_9": 407,
    "AB_Gad67_223": 513,
    "MV131017_7": 530,
}

# segAnalyser.hemisphereNeedsFlipping (segAnalyser.m:360-366)
FLIPPED_HEMISPHERE = {("MV_Ntsr1_169", "SG")}

# pavelAtlas.makeAbbrevIndexTable (pavelAtlas.m:188-201): Allen ID -> region abbreviation used in filenames.
ALLEN_ID_TO_ABBREV = {202: "MV", 509: "SUB", 632: "SG", 254: "RSP", 394: "V2M", 385: "V1",
                      733: "VPM", 88: "AHN", 322: "S1BF", 31: "CING"}


def _flow_list(value):
    # readSimpleYAML treats "{a, b}" as a list; PyYAML parses it as a mapping with null values.
    if isinstance(value, dict):
        return list(value.keys())
    if isinstance(value, str):
        return [value]
    return list(value)


def _number_matrix(value) -> np.ndarray:
    # "[a b c d; e f g h]" is a MATLAB matrix literal; PyYAML yields a one-element list of that string.
    text = " ".join(str(v) for v in value) if isinstance(value, list) else str(value)
    rows = [r.split() for r in text.strip("[] ").split(";") if r.strip()]
    return np.array([[float(x) for x in r] for r in rows])


@dataclass(frozen=True)
class StackGeometry:
    brain: str
    stack: str
    z_range: tuple[int, int]
    matrix: np.ndarray = field(repr=False)
    width: int
    height: int


@dataclass(frozen=True)
class Region:
    name: str
    allen_id: int
    allen_name: str
    section_name: str

    @property
    def abbrev(self) -> str:
        return ALLEN_ID_TO_ABBREV[self.allen_id]


@dataclass
class Config:
    scale: float
    stack_size: int
    training_brains: list[str]
    brains: dict[str, dict]
    regions: list[Region]
    stacks: dict[tuple[str, str], StackGeometry]
    seg_dir: Path = DEFAULT_SEG_DIR
    brain_dir: Path = DEFAULT_BRAIN_DIR

    def stack(self, brain: str, stack: str) -> StackGeometry:
        return self.stacks[(brain, stack)]

    def region_by_abbrev(self, abbrev: str) -> Region:
        return next(r for r in self.regions if r.abbrev == abbrev)

    def brain_file(self, brain: str) -> Path:
        name = self.brains[brain]["brainFileName"]
        path = self.brain_dir / name
        if path.exists():
            return path
        # The published MV_Ntsr1_165 file carries a " (1)" download suffix.
        stem = name[:-4] if name.endswith(".nii") else name
        matches = sorted(glob.glob(str(self.brain_dir / (glob.escape(stem) + "*.nii"))))
        if len(matches) != 1:
            raise FileNotFoundError(f"cannot resolve downscaled brain for {brain}: {name}")
        return Path(matches[0])


def load_config(seg_dir: os.PathLike | None = None, brain_dir: os.PathLike | None = None) -> Config:
    g = yaml.safe_load((DATA_DIR / "globalData.yml").read_text())
    brains = {b["name"]: b for b in g["brains"]}
    regions = [Region(r["name"], int(r["allenID"]), r["allenName"], r["sectionName"]) for r in g["regionsUsed"]]

    s = yaml.safe_load((DATA_DIR / "brainShifts.yml").read_text())
    stacks = {}
    for b in s["brainShifts"]:
        for sh in b["shifts"]:
            z = _number_matrix(sh["zRange"]).ravel().astype(int)
            m = _number_matrix(sh["shift"])
            if m.shape != (4, 4):
                raise ValueError(f"bad shift matrix for {b['name']}/{sh['name']}: {m.shape}")
            stacks[(b["name"], sh["name"])] = StackGeometry(
                b["name"], sh["name"], (int(z[0]), int(z[1])), m, int(sh["width"]), int(sh["height"]))

    return Config(
        scale=float(g["scale"]),
        stack_size=int(g["stackSize"]),
        training_brains=_flow_list(g["trainingBrains"]),
        brains=brains,
        regions=regions,
        stacks=stacks,
        seg_dir=Path(seg_dir) if seg_dir else DEFAULT_SEG_DIR,
        brain_dir=Path(brain_dir) if brain_dir else DEFAULT_BRAIN_DIR,
    )


def matlab_num2str(x: float) -> str:
    """num2str for the scale factor as used in MATLAB file/dir names (0.25 -> '0.25')."""
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"
