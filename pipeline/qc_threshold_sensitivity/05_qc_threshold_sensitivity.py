# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 2026

@author: zaman.i

Global-QC threshold sensitivity analysis.

For every (sample_id, input_type) and every region of interest, sweeps a
grid of (IoU threshold, grad_p95 threshold) pairs and, for each pair,
rebuilds the passes_global_qc consensus exactly the way
build_region_consensus_wholebrain.py does:

    passing      = (iou >= t_iou) & (grad_p95 <= t_grad)       [same rule as 04_combine_qc_csvs.py]
    probability  = mean over passing registrations of that registration's
                   region mask (from its own registered_atlas.tiff, native
                   sample space, no resampling)
    consensus    = probability >= consensus threshold

and records how that consensus region changes:

    n_pass                       registrations passing the gate
    volume_mm3                   consensus volume
    volume_all_registrations_mm3 consensus volume with NO gating (reference)
    dice_vs_all_registrations    Dice(consensus, ungated consensus)
    dice_vs_reference            Dice(consensus, consensus at --reference-iou /
                                 --reference-grad), only if a reference is given
    uncertain_fraction           fraction of voxels with 0 < probability < 1
                                 among voxels with probability > 0 (how much
                                 the passing registrations disagree)

Gates that leave fewer than --min-registrations registrations are recorded
with n_pass but NaN metrics (a consensus of 1-2 registrations isn't one).

The threshold grid is derived from the WHOLE qc table (not per sample), so
every sample -- and every SLURM array task -- uses identical thresholds.
It spans min -> max of each metric, i.e. it includes the "no gating" edge.

Nothing here writes to, or depends on, passes_global_qc or the existing
region_consensus outputs; it only reads registered_atlas.tiff.

Caching: each registration's region masks are stored per sample in
{output_dir}/mask_cache/ (bit-packed), so re-running with a different
grid, reference or consensus threshold does not re-read the atlases.

Outputs
-------
    {output_dir}/threshold_grid.json
    {output_dir}/per_sample/{sample_id}__{input_type}.csv
    {output_dir}/mask_cache/{sample_id}__{input_type}.npz

Plot with 05b_plot_qc_threshold_sensitivity.py.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import tifffile as tiff
import yaml

from region_utils import load_region_tree
from pipeline_common import normalise_path
from build_region_consensus_wholebrain import build_region_id_lookup


# ---------------------------------------------------------------------
# THRESHOLD GRID
# ---------------------------------------------------------------------

def make_threshold_grid(values, n_steps, decimals, extra=None):
    """Evenly spaced thresholds from min to max of the observed values
    (rounded outwards so the extremes are always included), plus any extra
    values (e.g. the reference threshold) so they sit exactly on the grid."""

    scale = 10 ** decimals
    low = np.floor(np.nanmin(values) * scale) / scale
    high = np.ceil(np.nanmax(values) * scale) / scale

    grid = np.round(np.linspace(low, high, n_steps), decimals)

    if extra is not None:
        grid = np.append(grid, extra)

    return np.unique(grid)


# ---------------------------------------------------------------------
# REGION MASKS (one atlas read per registration)
# ---------------------------------------------------------------------

def load_region_membership(group_df, region_lookup):
    """
    Returns (save_dirs, membership) where membership[region_id] =
    (union_indices, members) and members is a bool array of shape
    (n_registrations, len(union_indices)): True where that registration's
    registered atlas puts the voxel inside the region.
    """

    region_indices = {region_id: [] for region_id in region_lookup}
    save_dirs = []
    volume_shape = None

    for i, (_, row) in enumerate(group_df.iterrows()):
        atlas_path = normalise_path(row["registered_atlas_path"])

        if atlas_path is None or not atlas_path.exists():
            print(f"  [WARN] Missing registered atlas, skipping: {atlas_path}")
            continue

        atlas = tiff.imread(atlas_path)

        if volume_shape is None:
            volume_shape = atlas.shape
        elif atlas.shape != volume_shape:
            print(f"  [WARN] Shape mismatch {atlas.shape} vs {volume_shape}, skipping: {atlas_path}")
            continue

        # Look labels up once per registration, then every region is a
        # cheap gather instead of a full-volume np.isin.
        labels, inverse = np.unique(atlas.ravel(), return_inverse=True)
        del atlas

        for region_id, info in region_lookup.items():
            in_region = np.isin(labels, info["descendant_ids"])
            region_indices[region_id].append(np.flatnonzero(in_region[inverse]))

        save_dirs.append(str(row["save_dir"]))
        print(f"  [{i + 1}/{len(group_df)}] loaded {atlas_path.parent.name}")

    membership = {}

    for region_id, index_lists in region_indices.items():
        if index_lists:
            union = np.unique(np.concatenate(index_lists))
        else:
            union = np.array([], dtype=np.int64)

        members = np.zeros((len(index_lists), len(union)), dtype=bool)
        for r, indices in enumerate(index_lists):
            members[r, np.searchsorted(union, indices)] = True

        membership[region_id] = (union, members)

    return save_dirs, membership


def save_membership_cache(path, save_dirs, membership):
    arrays = {"save_dirs": np.array(save_dirs)}

    for region_id, (union, members) in membership.items():
        arrays[f"union__{region_id}"] = union
        arrays[f"members__{region_id}"] = np.packbits(members, axis=1)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_membership_cache(path, region_lookup, expected_save_dirs):
    """Returns (save_dirs, membership) or None if the cache is missing, or
    doesn't cover the requested regions / registrations."""

    if not path.exists():
        return None

    with np.load(path) as cache:
        save_dirs = [str(s) for s in cache["save_dirs"]]

        if not set(save_dirs) <= set(expected_save_dirs):
            return None

        membership = {}
        for region_id in region_lookup:
            if f"union__{region_id}" not in cache:
                return None

            union = cache[f"union__{region_id}"]
            members = np.unpackbits(
                cache[f"members__{region_id}"], axis=1, count=len(union)
            ).astype(bool)
            membership[region_id] = (union, members)

    return save_dirs, membership


# ---------------------------------------------------------------------
# CONSENSUS METRICS
# ---------------------------------------------------------------------

def dice(a, b):
    total = a.sum() + b.sum()
    return 2 * np.logical_and(a, b).sum() / total if total > 0 else np.nan


def consensus_from(members, passing, consensus_thresholds):
    probability = members[passing].mean(axis=0)
    return probability, {t: probability >= t for t in consensus_thresholds}


def sweep_region(
    members, iou, grad, iou_grid, grad_grid, consensus_thresholds,
    min_registrations, voxel_volume_mm3, reference,
):
    rows = []

    everything = np.ones(len(iou), dtype=bool)
    _, all_consensus = consensus_from(members, everything, consensus_thresholds)

    reference_consensus = None
    if reference is not None:
        reference_passing = (iou >= reference[0]) & (grad <= reference[1])
        if reference_passing.sum() >= min_registrations:
            _, reference_consensus = consensus_from(members, reference_passing, consensus_thresholds)
        else:
            print(
                f"  [WARN] Reference gate keeps only {reference_passing.sum()} registration(s) "
                f"(< {min_registrations}); dice_vs_reference will be NaN for this sample."
            )

    # Many threshold pairs select the same registrations; compute each
    # distinct subset only once.
    subset_cache = {}

    for t_iou in iou_grid:
        for t_grad in grad_grid:
            passing = (iou >= t_iou) & (grad <= t_grad)
            n_pass = int(passing.sum())
            key = passing.tobytes()

            if key not in subset_cache:
                stats = {}

                if n_pass >= min_registrations:
                    probability, consensus = consensus_from(members, passing, consensus_thresholds)
                    nonzero = (probability > 0).sum()
                    uncertain = ((probability > 0) & (probability < 1)).sum()

                    for t in consensus_thresholds:
                        stats[t] = {
                            "volume_mm3": consensus[t].sum() * voxel_volume_mm3,
                            "dice_vs_all_registrations": dice(consensus[t], all_consensus[t]),
                            "dice_vs_reference": (
                                dice(consensus[t], reference_consensus[t])
                                if reference_consensus is not None else np.nan
                            ),
                            "uncertain_fraction": uncertain / nonzero if nonzero > 0 else np.nan,
                        }
                else:
                    for t in consensus_thresholds:
                        stats[t] = {
                            "volume_mm3": np.nan,
                            "dice_vs_all_registrations": np.nan,
                            "dice_vs_reference": np.nan,
                            "uncertain_fraction": np.nan,
                        }

                subset_cache[key] = stats

            for t in consensus_thresholds:
                rows.append({
                    "iou_threshold": t_iou,
                    "grad_threshold": t_grad,
                    "consensus_threshold": t,
                    "n_pass": n_pass,
                    "volume_all_registrations_mm3": all_consensus[t].sum() * voxel_volume_mm3,
                    **subset_cache[key][t],
                })

    return rows


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="config/config.yaml",
    )

    parser.add_argument(
        "--qc",
        default="results/qc_tables/qc_results.csv",
        help="Combined QC table from 04_combine_qc_csvs.py.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: results/qc_tables/threshold_sensitivity/<qc table name>.",
    )

    parser.add_argument("--iou-col", default="iou")
    parser.add_argument("--grad-col", default="grad_p95")

    parser.add_argument(
        "--n-steps",
        type=int,
        default=21,
        help="Number of thresholds per metric, spanning its observed min -> max.",
    )

    parser.add_argument(
        "--reference-iou",
        type=float,
        default=None,
        help="Provisional IoU threshold to compare every other gate against (optional).",
    )

    parser.add_argument(
        "--reference-grad",
        type=float,
        default=None,
        help="Provisional grad_p95 threshold to compare every other gate against (optional).",
    )

    parser.add_argument(
        "--min-registrations",
        type=int,
        default=3,
        help="Gates keeping fewer registrations than this get NaN metrics.",
    )

    parser.add_argument(
        "--region-ids",
        nargs="+",
        type=int,
        default=None,
        help="Defaults to config roi.region_ids.",
    )

    parser.add_argument(
        "--consensus-thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Defaults to config consensus.thresholds.",
    )

    parser.add_argument("--sample-id", nargs="+", default=None)
    parser.add_argument("--input-type", nargs="+", default=None)

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute per-sample CSVs (the mask cache is still reused).",
    )

    args = parser.parse_args()

    if (args.reference_iou is None) != (args.reference_grad is None):
        raise ValueError("Give both --reference-iou and --reference-grad, or neither.")

    reference = (
        (args.reference_iou, args.reference_grad)
        if args.reference_iou is not None else None
    )

    # -----------------------------------------------------------------
    # Load configuration and QC table
    # -----------------------------------------------------------------

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)

    qc_path = Path(args.qc)
    qc = pd.read_csv(qc_path)
    qc = qc.dropna(subset=[args.iou_col, args.grad_col]).reset_index(drop=True)

    output_dir = Path(
        args.output_dir or Path("results/qc_tables/threshold_sensitivity") / qc_path.stem
    )

    region_ids = args.region_ids or cfg["roi"]["region_ids"]
    consensus_thresholds = sorted(set(args.consensus_thresholds or cfg["consensus"]["thresholds"]))
    voxel_volume_mm3 = (cfg["atlas"]["resolution_um"] / 1000) ** 3

    tree_df = load_region_tree(normalise_path(cfg["roi"]["region_tree_csv"]))
    region_lookup = build_region_id_lookup(region_ids, tree_df)

    if cfg["roi"].get("split_hemispheres", False):
        print("[INFO] roi.split_hemispheres is on, but this sweep (like the consensus "
              "script) uses whole-region masks; side is reported as 'both'.")

    # -----------------------------------------------------------------
    # Threshold grid -- from the WHOLE table, before any sample filtering
    # -----------------------------------------------------------------

    iou_grid = make_threshold_grid(
        qc[args.iou_col], args.n_steps, decimals=3,
        extra=[args.reference_iou] if reference else None,
    )
    grad_grid = make_threshold_grid(
        qc[args.grad_col], args.n_steps, decimals=4,
        extra=[args.reference_grad] if reference else None,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "threshold_grid.json", "w") as file:
        json.dump(
            {
                "qc_table": str(qc_path),
                "iou_col": args.iou_col,
                "grad_col": args.grad_col,
                "iou_thresholds": iou_grid.tolist(),
                "grad_thresholds": grad_grid.tolist(),
                "reference_iou": args.reference_iou,
                "reference_grad": args.reference_grad,
                "min_registrations": args.min_registrations,
                "consensus_thresholds": consensus_thresholds,
                "pass_rule": f"{args.iou_col} >= iou_threshold AND {args.grad_col} <= grad_threshold",
            },
            file,
            indent=2,
        )

    print(f"IoU thresholds ({len(iou_grid)}): {iou_grid.min()} -> {iou_grid.max()}")
    print(f"grad thresholds ({len(grad_grid)}): {grad_grid.min()} -> {grad_grid.max()}")
    print(f"Reference gate: {reference}")
    print(f"Regions: {[info['acronym'] for info in region_lookup.values()]}")
    print(f"Consensus thresholds: {consensus_thresholds}")

    # -----------------------------------------------------------------
    # Sample filtering (after the grid, so array tasks share one grid)
    # -----------------------------------------------------------------

    if args.sample_id is not None:
        qc = qc[qc["sample_id"].astype(str).isin({str(v) for v in args.sample_id})]
    if args.input_type is not None:
        qc = qc[qc["input_type"].astype(str).isin({str(v) for v in args.input_type})]

    if qc.empty:
        raise RuntimeError("No rows left in qc table after filtering by --sample-id / --input-type.")

    for (sample_id, input_type), group_df in qc.groupby(["sample_id", "input_type"]):
        print()
        print("=" * 80)
        print(f"{sample_id} | {input_type} | {len(group_df)} registration(s)")
        print("=" * 80)

        tag = f"{sample_id}__{input_type}"
        out_csv = output_dir / "per_sample" / f"{tag}.csv"

        if out_csv.exists() and not args.overwrite:
            print(f"[SKIP] Already exists: {out_csv}")
            continue

        # -------------------------------------------------------------
        # Region masks (cached)
        # -------------------------------------------------------------

        cache_path = output_dir / "mask_cache" / f"{tag}.npz"
        cached = load_membership_cache(cache_path, region_lookup, group_df["save_dir"].astype(str).tolist())

        if cached is not None:
            print(f"Using cached region masks: {cache_path}")
            save_dirs, membership = cached
        else:
            save_dirs, membership = load_region_membership(group_df, region_lookup)
            save_membership_cache(cache_path, save_dirs, membership)
            print(f"Saved region-mask cache: {cache_path}")

        if not save_dirs:
            print("[SKIP] No registered atlases could be loaded.")
            continue

        metrics = group_df.set_index(group_df["save_dir"].astype(str)).loc[save_dirs]
        iou = metrics[args.iou_col].to_numpy()
        grad = metrics[args.grad_col].to_numpy()

        # -------------------------------------------------------------
        # Sweep
        # -------------------------------------------------------------

        rows = []

        for region_id, info in region_lookup.items():
            _, members = membership[region_id]

            region_rows = sweep_region(
                members, iou, grad, iou_grid, grad_grid, consensus_thresholds,
                args.min_registrations, voxel_volume_mm3, reference,
            )

            for row in region_rows:
                row.update({
                    "sample_id": sample_id,
                    "input_type": input_type,
                    "region_id": region_id,
                    "acronym": info["acronym"],
                    "side": "both",
                    "n_registrations_total": len(save_dirs),
                    "reference_iou": args.reference_iou,
                    "reference_grad": args.reference_grad,
                })

            rows.extend(region_rows)
            print(f"  swept {info['acronym']}")

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
