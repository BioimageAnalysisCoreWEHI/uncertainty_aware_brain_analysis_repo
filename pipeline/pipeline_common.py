# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:54:20 2026

@author: zaman.i
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import to_local  # noqa: E402
# -*- coding: utf-8 -*-
"""
Shared utilities for the whole-brain registration / QC / region-export pipeline.

This module exists so that path handling, filename conventions, and
"which folders count as a registration set" logic are defined exactly once,
instead of being copy-pasted (and drifting) across scripts.

Used by:
    05_prepare_crop_spaces.py       (creates crop spaces + registration sets)
    08_build_consensus.py           (consensus masks from probability maps)
    export_parameter_masks.py       (per-registration parameter masks)
    build_final_results.py          (cell counts, volumes, final results tables)

run_qc_single.py (an earlier pipeline stage, before any of the above
concepts exist) still has its own local normalise_path rather than
importing this one - low-risk to fold in since it's a small, self-contained
function, but not yet done.

Changes
-------
- discover_registration_sets() now takes a crop_space_scope parameter
  ("shared" [default] or "per_input_type"), matching
  05_prepare_crop_spaces.py's own flag. The two layouts put sample_id/
  input_type at different relative depths below each crop space:

    shared (new default):
      <root>/<sample_id>/all_registrations_space/<input_type>/
      <root>/<sample_id>/global_qc_crop_space/<input_type>/passes_global_qc/
      <root>/<sample_id>/global_qc_crop_space/<input_type>/passes_final_qc_top_X/

    per_input_type (old/v5 behavior):
      <root>/<sample_id>/<input_type>/all_registrations_space/
      <root>/<sample_id>/<input_type>/global_qc_crop_space/passes_global_qc/
      <root>/<sample_id>/<input_type>/global_qc_crop_space/passes_final_qc_top_X/

  The previous version of this function only recognised the
  per_input_type layout - under the new shared layout it would silently
  return an empty list (every registrations_used.csv it found would fall
  through to the final "not a layout this pipeline recognises; skip"
  branch), which then surfaces downstream as a generic "no registration
  sets found" error rather than anything pointing at the real cause.

  set_name for the all_registrations set is "all_registrations_space" in
  BOTH scopes (matching the original per_input_type convention, where it
  equals the literal folder name) - kept consistent across scopes
  deliberately, since this value ends up in output CSVs
  (parameter_registration_results.csv's "registration_set" column) and
  having it differ by scope would make that column mean different things
  depending on how 05 happened to be run.
"""

from dataclasses import dataclass
from pathlib import Path
import json
import platform

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.transform import resize


# ---------------------------------------------------------------------
# PLOTTING STYLE (shared by the QC calibration / sensitivity figures)
# ---------------------------------------------------------------------

# Categorical sample colours (fixed order) + marker shapes as a second,
# colour-independent encoding of sample identity.
SAMPLE_COLOURS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
SAMPLE_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def make_sample_style(sample_labels) -> dict:
    """label -> (colour, marker), assigned in sorted-label order so the same
    sample gets the same style in every figure."""
    labels = sorted(sample_labels)

    if len(labels) > len(SAMPLE_COLOURS):
        print(
            f"WARNING: {len(labels)} samples but only {len(SAMPLE_COLOURS)} "
            "distinct colours; colours/markers will repeat."
        )

    return {
        label: (
            SAMPLE_COLOURS[i % len(SAMPLE_COLOURS)],
            SAMPLE_MARKERS[i % len(SAMPLE_MARKERS)],
        )
        for i, label in enumerate(labels)
    }


# ---------------------------------------------------------------------
# PATH / NAMING UTILITIES
# ---------------------------------------------------------------------

def normalise_path(path_str):
    """Map a stored path (placeholder form or a cluster path) to the local file system (see uaa_paths.py)."""
    if path_str is None:
        return None
    return Path(to_local(path_str))


def safe_value(value):
    """Make a numeric value filename-safe, e.g. -0.5 -> 'm0p5', 0.25 -> '0p25'."""
    return str(value).replace("-", "m").replace(".", "p")


def coronal_to_axial(array):
    return np.flip(np.transpose(array, (1, 0, 2)), axis=2)


def crop_bbox(array, bbox):
    return array[
        bbox["z0"]:bbox["z1"],
        bbox["y0"]:bbox["y1"],
        bbox["x0"]:bbox["x1"],
    ]


def resize_nearest(array, shape, dtype):
    """Nearest-neighbour resize. Safe to call on label/atlas arrays because
    it never blends values between neighbouring voxels."""
    return resize(
        array,
        output_shape=shape,
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(dtype)


def resize_probability(probability, output_shape):
    """Order=1 (linear) resize for continuous 0-1 probability data.

    Mirrors 05_prepare_crop_spaces.py's local helper of the same name;
    kept here too so other scripts (e.g. build_region_consensus_wholebrain.py)
    can reuse it without a second copy drifting from that one.
    """
    return resize(
        probability.astype(np.float32),
        output_shape=output_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.float32)


def make_output_name(row, prefix, label_id=None):
    """Build the standard parameter-tagged filename used for exported
    atlas crops and label masks."""
    suffix = (
        f"__bend_{safe_value(row.get('bending_energy_weight', 'NA'))}"
        f"__grid_{safe_value(row.get('grid_spacing', 'NA'))}"
        f"__smooth_{safe_value(row.get('smoothing_sigma_floating', 'NA'))}"
    )

    if label_id is None:
        return f"{prefix}{suffix}.tif"

    return f"{prefix}_{label_id}{suffix}.tif"


def load_crop_definition(crop_space_dir):
    """Read the bbox and full-resolution crop shape shared by every
    registration set inside one crop space (all_registrations_space/ or
    global_qc_crop_space/), as written by 05_prepare_crop_spaces.py.

    Returns (bbox_dict, crop_shape) or None if the crop space hasn't been
    prepared yet.
    """
    bbox_path = crop_space_dir / "bbox" / "bbox_downsampled_axial.json"
    crop_path = crop_space_dir / "fullres_crops" / "TH_crop_fullres_axial.tif"

    if not bbox_path.exists() or not crop_path.exists():
        return None

    with open(bbox_path, "r") as file:
        bbox = json.load(file)

    with tiff.TiffFile(crop_path) as tif:
        crop_shape = tuple(tif.series[0].shape)

    return bbox, crop_shape


# ---------------------------------------------------------------------
# REGISTRATION-SET DISCOVERY
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class RegistrationSet:
    """One exportable/countable registration subset."""
    sample_id: str
    input_type: str
    crop_space_name: str   # "all_registrations_space" or "global_qc_crop_space"
    crop_space_dir: Path   # holds bbox/, fullres_crops/, cellpose_outputs/
    set_name: str          # folder name of this specific set
    set_dir: Path          # holds registrations_used.csv + this set's outputs


_VALID_RUN_SETS = {"all", "all_registrations", "global_qc", "final_qc"}
_VALID_CROP_SPACE_SCOPES = {"shared", "per_input_type"}


def discover_registration_sets(
    region_assets_root, run_sets=("all",), crop_space_scope="shared", final_ssim_fraction=None,
):
    """
    Walk a region_assets root and return every registration-set folder
    produced by 05_prepare_crop_spaces.py, as a flat list of
    RegistrationSet records.

    This is the single source of truth for "which folders count as a
    registration set" so that 08, 09, and 10 can't silently drift apart
    from each other or from what 05 actually writes.

    Parameters
    ----------
    region_assets_root : path-like
        Root folder containing per-sample subfolders.
    run_sets : iterable of str
        Any of "all", "all_registrations", "global_qc", "final_qc".
        "all" expands to all three.
    crop_space_scope : {"shared", "per_input_type"}
        MUST match whichever scope 05_prepare_crop_spaces.py was actually
        run with - see module docstring for the two folder layouts.
    final_ssim_fraction : float, optional
        If given (and "final_qc" is requested), only match that specific
        fraction's folder (e.g. 0.5 -> passes_final_qc_top_0p5). If omitted,
        every passes_final_qc_top_* folder is matched.
    """
    region_assets_root = Path(region_assets_root)

    if crop_space_scope not in _VALID_CROP_SPACE_SCOPES:
        raise ValueError(
            f"Unknown crop_space_scope {crop_space_scope!r}. "
            f"Choose from: {sorted(_VALID_CROP_SPACE_SCOPES)}"
        )

    requested = set(run_sets)
    unknown = requested - _VALID_RUN_SETS
    if unknown:
        raise ValueError(f"Unknown run_sets value(s): {sorted(unknown)}")

    if "all" in requested:
        requested = {"all_registrations", "global_qc", "final_qc"}

    records = []

    for registrations_used in sorted(region_assets_root.rglob("registrations_used.csv")):
        set_dir = registrations_used.parent

        if crop_space_scope == "per_input_type":
            if set_dir.name == "all_registrations_space":
                if "all_registrations" not in requested:
                    continue
                crop_space_dir = set_dir
                crop_space_name = "all_registrations_space"
                set_name = "all_registrations_space"

            elif set_dir.parent.name == "global_qc_crop_space":
                crop_space_dir = set_dir.parent
                crop_space_name = "global_qc_crop_space"
                set_name = set_dir.name

                if set_name == "passes_global_qc":
                    if "global_qc" not in requested:
                        continue
                elif set_name.startswith("passes_final_qc_top_"):
                    if "final_qc" not in requested:
                        continue
                    if final_ssim_fraction is not None:
                        expected_name = "passes_final_qc_top_" + safe_value(final_ssim_fraction)
                        if set_name != expected_name:
                            continue
                else:
                    continue
            else:
                # Not a layout this pipeline recognises; skip.
                continue

            input_type = crop_space_dir.parent.name
            sample_id = crop_space_dir.parent.parent.name

        else:  # crop_space_scope == "shared"
            if set_dir.parent.name == "all_registrations_space":
                if "all_registrations" not in requested:
                    continue
                crop_space_dir = set_dir.parent
                crop_space_name = "all_registrations_space"
                # Kept identical to the per_input_type set_name (the literal
                # folder name in that layout) rather than the set_dir's own
                # name here (which is the input_type, e.g. "raw") - see
                # module docstring for why this is kept scope-independent.
                set_name = "all_registrations_space"
                input_type = set_dir.name
                sample_id = crop_space_dir.parent.name

            elif set_dir.parent.parent.name == "global_qc_crop_space":
                crop_space_dir = set_dir.parent.parent
                crop_space_name = "global_qc_crop_space"
                set_name = set_dir.name
                input_type = set_dir.parent.name
                sample_id = crop_space_dir.parent.name

                if set_name == "passes_global_qc":
                    if "global_qc" not in requested:
                        continue
                elif set_name.startswith("passes_final_qc_top_"):
                    if "final_qc" not in requested:
                        continue
                    if final_ssim_fraction is not None:
                        expected_name = "passes_final_qc_top_" + safe_value(final_ssim_fraction)
                        if set_name != expected_name:
                            continue
                else:
                    continue
            else:
                # Not a layout this pipeline recognises; skip.
                continue

        records.append(
            RegistrationSet(
                sample_id=sample_id,
                input_type=input_type,
                crop_space_name=crop_space_name,
                crop_space_dir=crop_space_dir,
                set_name=set_name,
                set_dir=set_dir,
            )
        )

    return records