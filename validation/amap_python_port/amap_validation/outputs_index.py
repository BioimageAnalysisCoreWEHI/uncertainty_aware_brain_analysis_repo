"""Generate <out>/OUTPUTS_INDEX.md: every output folder with file counts and sizes, logs, and QC figures."""
from __future__ import annotations

import os
import time
from pathlib import Path

FOLDERS = [
    ("maskCache_0.25", "Stage 1: bit-packed 4 µm canvas masks (nearest for STAPLE, bicubic for scoring); safe to delete"),
    ("userStapleResults", "Stage 2: STAPLE/SBA consensus on the 4 µm canvas, sidecars, scores (top level only)"),
    ("userStapleResults/qc", "QC figures and the morphology table"),
    ("userStapleResults/inBrainSpace", "Stage 3: consensus, probability and label maps on the 12.5 µm brain grids (.nii.gz)"),
    ("userStapleResults/asr25um", "Stages 4-5: mirrored asr 25 µm brains, 2-D consensus, rater masks, logs"),
    ("pipeline_comparison", "Validation of the uncertainty-aware pipeline against the aMAP experts"),
]

LOGS = [
    ("userStapleResults/segTable_with_pq.csv", "every segmentation with its STAPLE sensitivity/specificity"),
    ("userStapleResults/staple_summary.csv", "one row per brain × structure: n, iterations, areas, STAPLE-vs-SBA Dice"),
    ("userStapleResults/manual_scores.csv", "rater Dice/Hausdorff vs STAPLE and SBA (4 µm)"),
    ("userStapleResults/qc/morphology.csv", "consensus morphology (components, holes, solidity, hemisphere)"),
    ("userStapleResults/inBrainSpace/inBrain_summary.csv", "brain-space placement checks per structure"),
    ("userStapleResults/asr25um/asr25um_slices.csv", "asr 25 µm slice index A of every consensus + rater-slice spread"),
    ("userStapleResults/asr25um/asr25um_raters.csv", "every rater mask at asr 25 µm with its slice index"),
    ("userStapleResults/asr25um/asr25um_all_masks.csv", "master index of every asr 25 µm image"),
    ("pipeline_comparison/results_summary.md", "validation results summary"),
]

FIGURES = [
    ("userStapleResults/qc/staple_outlines.png", "all rater outlines with STAPLE (solid) and SBA (dashed), 6 brains × 10 structures"),
    ("userStapleResults/qc/staple_outlines_{brain}.png", "the same, one brain, larger panels"),
    ("userStapleResults/qc/brain_{brain}.png", "consensus on the 12.5 µm downscaled brain, one plane per stack"),
    ("userStapleResults/qc/inBrain_AB_Gad67_223RSP.png", "brain-space sample: probability, STAPLE, raters' own slices, AP spread"),
    ("userStapleResults/qc/asr25_{brain}.png", "mirrored asr 25 µm: each structure on its slice with probability, STAPLE, rater outlines"),
]


def _size(path: Path):
    n, total = 0, 0
    for root, _, files in os.walk(path):
        for f in files:
            n += 1
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return n, total


def _top_level_size(path: Path):
    files = [p for p in path.iterdir() if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def write_index(out_dir: Path, brains, log=print) -> Path:
    out_dir = Path(out_dir)
    lines = [f"# aMAP pipeline outputs\n\nGenerated {time.strftime('%Y-%m-%d %H:%M')} by `python -m amap_validation index`. "
             "Paths are relative to this file. Documentation: `aMAP_validation/python/PIPELINE.md`.\n",
             "## Folders\n", "| Folder | Files | Size | Content |", "|---|---|---|---|"]
    missing = []
    for rel, what in FOLDERS:
        p = out_dir / rel
        if not p.exists():
            continue
        n, sz = _top_level_size(p) if rel == "userStapleResults" else _size(p)
        lines.append(f"| `{rel}/` | {n} | {sz / 1e6:,.0f} MB | {what} |")
    lines += ["\n## Logs and tables\n"]
    for rel, what in LOGS:
        if (out_dir / rel).exists():
            lines.append(f"* [`{rel}`]({rel}): {what}")
    lines += ["\n## QC figures\n"]
    for rel, what in FIGURES:
        paths = [rel.format(brain=b) for b in brains] if "{brain}" in rel else [rel]
        present = [r for r in paths if (out_dir / r).exists()]
        missing += [r for r in paths if not (out_dir / r).exists()]
        if present:
            links = ", ".join(f"[{Path(r).stem}]({r})" for r in present)
            lines.append(f"* {what}: {links}")
    fig_dir = out_dir / "pipeline_comparison" / "figures"
    if fig_dir.exists():
        figs = sorted(fig_dir.glob("*.png"))
        if figs:
            lines.append("* validation figures: " + ", ".join(
                f"[{f.stem}](pipeline_comparison/figures/{f.name})" for f in figs))
    path = out_dir / "OUTPUTS_INDEX.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"  wrote {path}" + (f" ({len(missing)} expected figures not present)" if missing else ""))
    return path
