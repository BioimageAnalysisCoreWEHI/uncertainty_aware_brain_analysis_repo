"""Pipeline overview chart: (A) the pipeline, HPC vs local steps; (B) how the code evolved; (C) validation branches.

    python docs/make_pipeline_overview.py      -> docs/pipeline_overview.{png,pdf,svg} (600 dpi)
"""
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
C_HPC, C_LOC, C_GPU = "#56B4E9", "#E69F00", "#CC79A7"
FILL = {C_HPC: "#e3f2fb", C_LOC: "#fcefd6", C_GPU: "#f6e3ee"}


def box(ax, x, y, w, h, title, body, col, mono=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03", fc=FILL[col], ec=col,
                                lw=1.1))
    ax.text(x + 0.03, y + h - 0.05, title, fontsize=7, fontweight="bold", va="top", ha="left")
    ax.text(x + 0.03, y + h - 0.2, body, fontsize=5.7, va="top", ha="left", linespacing=1.25)
    if mono:
        ax.text(x + 0.03, y + 0.05, mono, fontsize=4.7, va="bottom", ha="left", family="Consolas", color="0.25",
                linespacing=1.2)


def arrow(ax, x0, y0, x1, y1, txt=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="0.35", lw=0.9,
                                                                   mutation_scale=8))
    if txt:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.04, txt, fontsize=5.2, ha="center", va="bottom", color="0.35")


def panel_a(ax):
    ax.set_xlim(-0.08, 10.12)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    W, H = 2.3, 1.35
    top, bot = 1.95, 0.32
    xs = [0.0, 2.57, 5.14, 7.71]
    box(ax, xs[0], top, W, H, "0  Set-up", "downsample AF/TH to 20 um, tissue\nmask, optional preprocessing,\n"
        "samples.csv per dataset", C_LOC, "pipeline/00_setup/*\nconfig/datasets/<dataset>/")
    box(ax, xs[1], top, W, H, "1-2  Registration sweep", "brainreg (NiftyReg) for every bending\nenergy x grid "
        "spacing; one job per\nregistration (48 per brain and input)", C_HPC,
        "01_make_brainreg_jobs.py\n02_submit_brainreg_sweep.sh\nrun_brainreg_array.sh")
    box(ax, xs[2], top, W, H, "3  QC metrics", "tissue-atlas IoU, Jacobian-gradient\np95 (grad_p95), local SSIM per "
        "region;\none job per registration", C_HPC, "03_submit_qc_jobs.sh\nrun_qc_array.sh\nrun_qc_single.py")
    box(ax, xs[3], top, W, H, "4  Combine + select", "global QC (IoU > t, grad_p95 < t),\nthen keep the top 50 % by "
        "SSIM\n(passes_final_qc)", C_LOC, "04_combine_qc_csvs.py\n04b_plot_qc_calibration.py")
    box(ax, xs[0], bot, W, H, "5  Region consensus", "inclusion frequency of each voxel\nover retained registrations;"
        "\nconsensus at thresholds 0.01-1.0;\none job per sample x input x region", C_HPC,
        "05_build_region_consensus_jobs.py\nrun_region_consensus_array.sh\nbuild_region_consensus_wholebrain.py")
    box(ax, xs[1], bot, W, H, "6  Cell detection", "Cellpose (custom TH model) on the\nfull-resolution midbrain crop",
        C_GPU, "cells/06_make_cellpose_jobs.py\ncells/run_cellpose_array.sh\ncells/run_cellpose_one_crop.py")
    box(ax, xs[2], bot, W, H, "7-8  Counting", "overlap rule (a cell counts if any\nvoxel is in the mask); per-cell "
        "inclusion\nfrequency over the registrations", C_LOC,
        "cells/07_count_cells_in_masks.py\ncells/08_compute_cell_region_\n  probabilities.py")
    box(ax, xs[3], bot, W, H, "Analyses + figures", "count stability, QC-rule sensitivity,\nvalidation (aMAP, "
        "independent LSFM),\nmanuscript figures", C_LOC, "validation/analysis/scripts/*\nvalidation/independent_lsfm/*")
    for i in range(3):
        arrow(ax, xs[i] + W, top + H / 2, xs[i + 1], top + H / 2)
    arrow(ax, xs[3] + W / 2, top, xs[0] + W / 2, bot + H, None)
    for i in range(3):
        arrow(ax, xs[i] + W, bot + H / 2, xs[i + 1], bot + H / 2)
    ax.text(5.0, 0.08, "HPC SLURM arrays: 234 logged registrations took 92 h of compute "
            "but 3.0 h of wall time (31x)", ha="center", fontsize=5.8, color="0.3")
    for k, (c, t) in enumerate(((C_HPC, "HPC (SLURM array, CPU)"), (C_GPU, "HPC (GPU) or local"),
                                (C_LOC, "local workstation"))):
        ax.add_patch(FancyBboxPatch((0.3 + k * 2.6, 3.43), 0.18, 0.09, boxstyle="round,pad=0.005", fc=FILL[c], ec=c))
        ax.text(0.55 + k * 2.6, 3.475, t, fontsize=6, va="center")
    ax.text(-0.05, 3.55, "A", fontsize=10, fontweight="bold", va="top", ha="right", transform=ax.transData)


GENS = [  # (label, start, end, what it added, folder)
    ("Original manuscript (ad hoc)", date(2025, 8, 8), date(2026, 6, 1),
     "sweep, QC, SSIM, region extraction, Cellpose counts,\nconsensus and figures as separate scripts",
     "analysis_scripts/  ->  archive/"),
    ("Config-driven sweep", date(2026, 5, 22), date(2026, 5, 26),
     "config.yaml + job table for brainreg and QC metrics", "analysis_scripts/final_paper_scripts"),
    ("Consensus + cells refactor", date(2026, 5, 22), date(2026, 7, 30),
     "consensus maps, Cellpose jobs, counting, cell probabilities", "analysis_scripts/testing_scripts/final_paper_scripts"),
    ("Generic whole-brain pipeline", date(2026, 7, 16), date(2026, 8, 13),
     "crop spaces, registration sets, parameter masks, final tables", "analysis_of_whole_brain_projects/scripts"),
    ("Post-review pipeline", date(2026, 8, 24), date(2026, 9, 30),
     "SLURM arrays for every step, multi-region SSIM,\nwhole-brain region consensus, any atlas", "post_review_analysis/scripts  ->  pipeline/"),
    ("Validation forks + analyses", date(2026, 9, 1), date(2026, 10, 7),
     "aMAP and independent-LSFM configs, QC-rule sensitivity,\naMAP Python port, revision figures",
     "lightsheet_data/, validation_analysis/  ->  validation/"),
    ("Clean repository", date(2026, 10, 8), date(2026, 10, 20),
     "one copy of each module, configurable paths, environments", "uncertainty_aware_analysis_code_repo"),
]


def panel_b(ax):
    t0 = date(2025, 7, 1)
    days = lambda d: (d - t0).days  # noqa: E731
    cols = ["0.55", "#9ecae1", "#6baed6", "#4292c6", C_HPC, "#2171b5", "#7B3294"]
    for i, (lab, a, b, what, folder) in enumerate(GENS):
        y = len(GENS) - 1 - i
        ax.barh(y, max(days(b) - days(a), 6), left=days(a), height=0.55, color=cols[i])
        ax.text(-8, y, lab, ha="right", va="center", fontsize=6.3, fontweight="bold")
        ax.text(days(b) + 6, y + 0.13, what, va="center", fontsize=5.4, linespacing=1.1)
        ax.text(days(b) + 6, y - 0.33, folder, va="center", fontsize=5.0, family="Consolas", color="0.35")
    ticks = [date(2025, m, 1) for m in (8, 10, 12)] + [date(2026, m, 1) for m in (2, 4, 6, 8, 10)]
    ax.set_xticks([days(t) for t in ticks], [t.strftime("%b %Y") for t in ticks], fontsize=5.8)
    ax.set_yticks([])
    ax.set_xlim(-2, days(date(2027, 6, 1)))
    ax.set_ylim(-0.7, len(GENS) - 0.3)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.text(-0.3, 1.02, "B", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    ax.set_title("How the code evolved (file dates)", fontsize=7, loc="left", pad=4)


def panel_c(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.3)
    ax.axis("off")
    rows = [("aMAP (serial two-photon,\n6 brains, expert STAPLE\nreference)", "config/datasets/amap",
             "validation/amap_python_port (STAPLE, scoring)\nvalidation/analysis/scripts 01-04, 09-10, 34",
             "Figs 5-6, S1, S4; Tables S1-S2"),
            ("Independent LSFM (ScienceDB,\n6 brains, published\nregistration)", "config/datasets/independent_lsfm",
             "validation/independent_lsfm 07a-19",
             "Fig 7, S6"),
            ("QC-rule sensitivity\n(all three datasets)", "pipeline/qc_threshold_sensitivity",
             "validation/analysis/scripts 12, 14-17, 23, 30-33\n+ qcsens_worker.py (HPC array)",
             "Fig 8, S3, S5; Table S3")]
    for i, (d, cfg, code, figs) in enumerate(rows):
        y = 1.95 - i * 0.78
        ax.text(0.0, y, d, fontsize=6.2, va="center", fontweight="bold", linespacing=1.15)
        box(ax, 2.35, y - 0.3, 2.15, 0.6, "pipeline 01-05", "", C_HPC, cfg)
        arrow(ax, 4.55, y, 4.8, y)
        box(ax, 4.85, y - 0.3, 3.35, 0.6, "analysis", "", C_LOC, code)
        arrow(ax, 8.25, y, 8.55, y)
        ax.text(8.6, y, figs, fontsize=6, va="center")
    ax.text(-0.05, 2.35, "C", fontsize=10, fontweight="bold", va="top", ha="right")


def main():
    fig = plt.figure(figsize=(7.2, 8.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.55, 1.25, 1.0], hspace=0.22)
    panel_a(fig.add_subplot(gs[0]))
    axb = fig.add_subplot(gs[1])
    pos = axb.get_position()
    axb.set_position([pos.x0 + 0.2, pos.y0, pos.width - 0.2, pos.height])
    panel_b(axb)
    panel_c(fig.add_subplot(gs[2]))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"pipeline_overview.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    print("saved", OUT / "pipeline_overview.png")


if __name__ == "__main__":
    main()
