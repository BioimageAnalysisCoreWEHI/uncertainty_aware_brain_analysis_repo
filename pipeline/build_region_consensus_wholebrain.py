# -*- coding: utf-8 -*-
"""
build_region_consensus_wholebrain.py

For each (sample_id, input_type), build THREE parallel registration
subsets -- matching the old crop-space pipeline's run sets -- and for
each one, build a per-voxel probability map for each region of interest,
directly in that sample's own FULL, UNCROPPED image space, then
threshold it into consensus masks:

    all_registrations       every registration, unfiltered
    passes_global_qc        registrations where qc[global_qc_column] is True
    passes_final_qc_top_X   the above, further narrowed to the top
                             ssim_top_fraction of local_ssim scores
                             *among those that already passed global QC*

Each of these is built per REGION (and per hemisphere side, if
roi.split_hemispheres is on) independently -- a registration can be in
passes_final_qc_top_X for one region and not another, since local SSIM
is region-specific.

raw vs preprocessed is not special-cased anywhere here: input_type is
just a column in the qc table, and grouping by (sample_id, input_type)
naturally produces separate outputs for each value already present.

Why this exists instead of reusing 05_prepare_crop_spaces.py /
08_build_consensus_v2.py / build_additional_region_consensus.py
------------------------------------------------------------------
Those scripts build a bbox-cropped, axial-transformed "crop space" per
sample, mainly so a single primary region's full-resolution Cellpose
segmentation output could be exported and hardlinked cheaply. That
machinery isn't needed here: we only want anatomical region
probability/consensus maps, and `registered_atlas.tiff` (one per
registration) is *already* the atlas label volume warped into that
sample's own native image space -- no crop, no coordinate transform, no
axial flip required. This script reads that file directly.

What "consensus" means here
----------------------------
Multiple registrations exist per (sample_id, input_type) because of the
brainreg parameter sweep (grid_spacing x bending_energy_weight x ...).
For a fixed region, stacking "does this voxel belong to region X" across
every registration in a given run set and averaging gives a per-voxel
probability that is robust to the choice of registration hyperparameters
for THAT sample -- not a cross-sample consensus.

Region resolution
------------------
Composite/parent regions (e.g. "Primary visual area") are expanded to
every descendant leaf id via region_utils.get_descendant_ids, same as
the QC step -- the atlas only ever paints leaf ids, so this is required
for anything above a leaf in the tree.

Resolution
-----------
Everything stays at native resolution (whatever registered_atlas.tiff is
on, e.g. atlas.resolution_um) -- no resampling for now.

Output layout (no crop-space concepts at all):
    {output_dir}/{sample_id}/{input_type}/{acronym}_{side}/
        all_registrations/
            individual_masks/mask__bend_X__grid_Y__smooth_Z.tif   (every registration in this set)
            probability.tif, consensus_thresh_*.tif, registrations_used.csv
        passes_global_qc/
            individual_masks/...    (same registrations that made it into this set)
            probability.tif, consensus_thresh_*.tif, registrations_used.csv
        passes_final_qc_top_{fraction}/
            individual_masks/...
            probability.tif, consensus_thresh_*.tif, registrations_used.csv

A registration that belongs to more than one set (e.g. passes both
passes_global_qc and passes_final_qc_top_X) gets its individual mask
duplicated into each set's own individual_masks/ folder -- deliberate,
so each gate's folder is self-contained, at the cost of some disk space.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import tifffile as tiff
import yaml

from region_utils import load_region_tree, get_descendant_ids, apply_ssim_ranking
from pipeline_common import normalise_path, safe_value, make_output_name


_VALID_RUN_SETS = {"all_registrations", "global_qc", "final_qc"}


def resolve_run_sets(requested) -> set:
    requested = set(requested)
    unknown = requested - _VALID_RUN_SETS - {"all"}
    if unknown:
        raise ValueError(f"Unknown run set(s): {sorted(unknown)}. Choose from: all, {sorted(_VALID_RUN_SETS)}")
    if "all" in requested:
        return set(_VALID_RUN_SETS)
    return requested


def resolve_side_labels(cfg: dict) -> list[str]:
    """Matches the column-naming convention used by 03_compute_qc_metrics.py /
    run_qc_single.py: "both" if split_hemispheres is off, else the two
    configured hemisphere labels."""
    roi_cfg = cfg.get("roi", {})
    if roi_cfg.get("split_hemispheres", False):
        return [roi_cfg.get("low_side_label", "right"), roi_cfg.get("high_side_label", "left")]
    return ["both"]


def build_region_id_lookup(region_ids: list[int], tree_df: pd.DataFrame) -> dict:
    """region_id -> {"acronym", "name", "descendant_ids"} (no masks/bboxes --
    those need the annotation volume, which isn't needed here since we mask
    directly against each registration's own registered_atlas.tiff)."""
    lookup = {}
    for region_id in region_ids:
        row = tree_df.loc[tree_df["id"] == region_id]
        if row.empty:
            raise ValueError(f"Region id {region_id} not found in region tree CSV")
        lookup[region_id] = {
            "acronym": row.iloc[0]["acronym"],
            "name": row.iloc[0]["name"],
            "descendant_ids": np.array(sorted(get_descendant_ids(region_id, tree_df, include_self=True))),
        }
    return lookup


def build_set_eligibility(group_df, global_qc_column, local_ssim_pass_col, run_sets) -> dict:
    """One boolean Series (aligned to group_df.index) per requested run set,
    for a single region/side."""
    eligibility = {}

    if "all_registrations" in run_sets:
        eligibility["all_registrations"] = pd.Series(True, index=group_df.index)

    global_qc_elig = group_df[global_qc_column] == True  # noqa: E712

    if "global_qc" in run_sets:
        eligibility["passes_global_qc"] = global_qc_elig

    if "final_qc" in run_sets:
        if local_ssim_pass_col in group_df.columns:
            eligibility["passes_final_qc"] = global_qc_elig & (group_df[local_ssim_pass_col] == True)  # noqa: E712
        else:
            eligibility["passes_final_qc"] = pd.Series(False, index=group_df.index)

    return eligibility


def process_group(
    sample_id, input_type, group_df, region_lookup, side_labels,
    global_qc_column, run_sets, ssim_top_fraction, thresholds, output_dir, overwrite,
    save_individual_masks,
):
    if "registered_atlas_path" not in group_df.columns:
        print(f"[SKIP] {sample_id} | {input_type}: qc table has no registered_atlas_path column")
        return

    final_qc_folder_name = f"passes_final_qc_top_{safe_value(ssim_top_fraction)}"
    set_folder_names = {
        "all_registrations": "all_registrations",
        "passes_global_qc": "passes_global_qc",
        "passes_final_qc": final_qc_folder_name,
    }

    # Per (region_id, side, set_name): eligibility + accumulator state.
    eligibility = {}
    accumulators = {}
    for region_id, info in region_lookup.items():
        for side in side_labels:
            local_ssim_pass_col = f"passes_local_ssim_qc__{info['acronym']}_{side}"
            set_elig = build_set_eligibility(group_df, global_qc_column, local_ssim_pass_col, run_sets)

            for set_name, elig in set_elig.items():
                key = (region_id, side, set_name)
                eligibility[key] = elig
                accumulators[key] = {"sum": None, "count": 0, "used_rows": []}

    if "final_qc" in run_sets:
        missing = {
            f"passes_local_ssim_qc__{info['acronym']}_{side}"
            for info in region_lookup.values() for side in side_labels
            if f"passes_local_ssim_qc__{info['acronym']}_{side}" not in group_df.columns
        }
        if missing:
            print(
                f"[WARN] {sample_id} | {input_type}: missing local-SSIM pass column(s) {sorted(missing)} -- "
                f"those region/side final_qc sets will be empty. Run region_utils.apply_ssim_ranking first."
            )

    # A registration's individual mask is now saved inside EVERY set folder
    # it belongs to (duplicated across all_registrations / passes_global_qc /
    # passes_final_qc_top_X where applicable) -- no longer deduped to one
    # shared location, per your call that duplicates are fine for now.
    #
    # One atlas load per registration, still shared across every region/side/
    # set that needs it -- not one load per (region, set) combination.
    for idx, row in group_df.iterrows():
        atlas_path = row["registered_atlas_path"]
        if pd.isna(atlas_path):
            continue

        needed_keys = [key for key, elig in eligibility.items() if elig.loc[idx]]
        if not needed_keys:
            continue

        atlas_path = normalise_path(atlas_path)
        if not atlas_path.exists():
            print(f"[WARN] Missing registered atlas, skipping registration: {atlas_path}")
            continue

        atlas = tiff.imread(atlas_path)

        # region/side pairs actually touched by this registration, deduped
        # across the (possibly multiple) sets it belongs to.
        region_sides_needed = {(region_id, side) for (region_id, side, _) in needed_keys}

        for region_id, side in region_sides_needed:
            descendant_ids = region_lookup[region_id]["descendant_ids"]
            mask = np.isin(atlas, descendant_ids)
            mask_u8 = mask.astype(np.uint8)  # native resolution, no resampling
            acronym = region_lookup[region_id]["acronym"]

            for set_name in ("all_registrations", "passes_global_qc", "passes_final_qc"):
                key = (region_id, side, set_name)
                if key not in accumulators:
                    continue
                if not eligibility[key].loc[idx]:
                    continue

                acc = accumulators[key]
                if acc["sum"] is None:
                    acc["sum"] = mask.astype(np.float32)
                elif acc["sum"].shape != mask.shape:
                    print(
                        f"[WARN] Shape mismatch for {acronym}_{side}/{set_folder_names[set_name]} "
                        f"at {atlas_path} ({mask.shape} vs {acc['sum'].shape}); skipping this registration."
                    )
                    continue
                else:
                    acc["sum"] += mask

                acc["count"] += 1
                acc["used_rows"].append(row)

                # Individual mask, saved into THIS set's own folder --
                # duplicated across every set this registration belongs to.
                if save_individual_masks:
                    folder_name = set_folder_names[set_name]
                    indiv_dir = (
                        Path(output_dir) / sample_id / input_type
                        / f"{acronym}_{side}" / folder_name / "individual_masks"
                    )
                    indiv_dir.mkdir(parents=True, exist_ok=True)

                    mask_out_path = indiv_dir / make_output_name(row, prefix="mask")
                    if overwrite or not mask_out_path.exists():
                        tiff.imwrite(mask_out_path, mask_u8, compression="zlib")

        del atlas

    # Save probability + consensus per region/side/set, at NATIVE
    # resolution -- no resampling for now.
    for (region_id, side, set_name), acc in accumulators.items():
        acronym = region_lookup[region_id]["acronym"]
        folder_name = set_folder_names[set_name]

        if acc["count"] == 0:
            print(f"[SKIP] {sample_id} | {input_type} | {acronym}_{side} | {folder_name}: no eligible registrations")
            continue

        set_out_dir = Path(output_dir) / sample_id / input_type / f"{acronym}_{side}" / folder_name
        probability_path = set_out_dir / "probability.tif"

        expected_outputs = [probability_path] + [
            set_out_dir / f"consensus_thresh_{safe_value(t)}.tif" for t in thresholds
        ]
        if not overwrite and all(p.exists() for p in expected_outputs):
            print(f"[SKIP] Already complete: {set_out_dir}")
            continue

        set_out_dir.mkdir(parents=True, exist_ok=True)

        probability_native = (acc["sum"] / acc["count"]).astype(np.float32)
        tiff.imwrite(probability_path, probability_native, compression="zlib")

        for threshold in thresholds:
            consensus_mask = (probability_native >= threshold).astype(np.uint8)
            tiff.imwrite(
                set_out_dir / f"consensus_thresh_{safe_value(threshold)}.tif",
                consensus_mask,
                compression="zlib",
            )

        pd.DataFrame(acc["used_rows"]).to_csv(set_out_dir / "registrations_used.csv", index=False)

        print(
            f"[SAVED] {sample_id} | {input_type} | {acronym}_{side} | {folder_name}: "
            f"{acc['count']} registrations | prob range "
            f"{probability_native.min():.3f}-{probability_native.max():.3f} | -> {set_out_dir}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--qc-table", default="results/qc_tables/qc_results.csv")
    parser.add_argument("--out-dir", default=None, help="Defaults to config region_consensus.output_dir.")
    parser.add_argument("--region-ids", nargs="+", type=int, default=None, help="Defaults to config roi.region_ids.")
    parser.add_argument("--thresholds", nargs="+", type=float, default=None, help="Defaults to config consensus.thresholds.")
    parser.add_argument("--global-qc-column", default=None, help="Defaults to config region_consensus.global_qc_column.")
    parser.add_argument(
        "--run-sets", nargs="+", choices=["all", "all_registrations", "global_qc", "final_qc"], default=None,
        help=(
            "Which registration subsets to build per region/side. Defaults to config "
            "region_consensus.run_sets (typically ['all']: all_registrations (unfiltered), "
            "global_qc (passes_global_qc only), and final_qc (passes_global_qc AND top-fraction "
            "local SSIM for that specific region/side)."
        ),
    )
    parser.add_argument(
        "--save-individual-masks", action=argparse.BooleanOptionalAction, default=None,
        help="Save each registration's own region mask (duplicated into every gate folder it belongs to), "
             "not just the aggregate probability/consensus. Defaults to config region_consensus.save_individual_masks.",
    )
    parser.add_argument("--sample-id", nargs="+", default=None)
    parser.add_argument("--input-type", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    region_ids = args.region_ids or cfg["roi"]["region_ids"]
    thresholds = sorted(set(args.thresholds or cfg["consensus"]["thresholds"]))
    output_dir = args.out_dir or cfg.get("region_consensus", {}).get("output_dir", "results/region_consensus")
    global_qc_column = args.global_qc_column or cfg.get("region_consensus", {}).get("global_qc_column", "passes_global_qc")
    run_sets = resolve_run_sets(args.run_sets or cfg.get("region_consensus", {}).get("run_sets", ["all"]))
    ssim_top_fraction = cfg["qc"]["ssim_top_fraction"]
    save_individual_masks = (
        args.save_individual_masks
        if args.save_individual_masks is not None
        else cfg.get("region_consensus", {}).get("save_individual_masks", True)
    )

    tree_df = load_region_tree(normalise_path(cfg["roi"]["region_tree_csv"]))
    region_lookup = build_region_id_lookup(region_ids, tree_df)
    side_labels = resolve_side_labels(cfg)

    print("Regions:")
    for region_id, info in region_lookup.items():
        print(f"  {region_id} ({info['acronym']} / {info['name']}): {len(info['descendant_ids'])} leaf id(s)")
    print(f"Sides: {side_labels}")
    print(f"Run sets: {sorted(run_sets)}")
    print(f"Global QC column: {global_qc_column}")
    print(f"SSIM top fraction (for final_qc): {ssim_top_fraction}")
    print(f"Save individual per-registration masks (duplicated per gate): {save_individual_masks}")
    print("Resolution: native (no resampling)")

    qc = pd.read_csv(args.qc_table)

    if global_qc_column not in qc.columns:
        raise ValueError(f"qc table {args.qc_table} has no column '{global_qc_column}'")

    if "final_qc" in run_sets:
        needed_cols = {f"passes_local_ssim_qc__{info['acronym']}_{side}" for info in region_lookup.values() for side in side_labels}
        if not needed_cols & set(qc.columns):
            print(
                "[INFO] No passes_local_ssim_qc__* columns found in qc table; "
                "computing them now via region_utils.apply_ssim_ranking "
                f"(ranked only among rows passing {global_qc_column})."
            )
            qc = apply_ssim_ranking(qc, ssim_top_fraction, global_qc_column=global_qc_column)

    if args.sample_id is not None:
        qc = qc[qc["sample_id"].astype(str).isin({str(v) for v in args.sample_id})]
    if args.input_type is not None:
        qc = qc[qc["input_type"].astype(str).isin({str(v) for v in args.input_type})]

    if qc.empty:
        raise RuntimeError("No rows left in qc table after filtering by --sample-id / --input-type.")

    for (sample_id, input_type), group_df in qc.groupby(["sample_id", "input_type"]):
        print()
        print("=" * 80)
        print(f"Group: sample_id={sample_id} | input_type={input_type} | {len(group_df)} registration(s)")
        print("=" * 80)

        process_group(
            sample_id=str(sample_id),
            input_type=str(input_type),
            group_df=group_df,
            region_lookup=region_lookup,
            side_labels=side_labels,
            global_qc_column=global_qc_column,
            run_sets=run_sets,
            ssim_top_fraction=ssim_top_fraction,
            thresholds=thresholds,
            output_dir=output_dir,
            overwrite=args.overwrite,
            save_individual_masks=save_individual_masks,
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()