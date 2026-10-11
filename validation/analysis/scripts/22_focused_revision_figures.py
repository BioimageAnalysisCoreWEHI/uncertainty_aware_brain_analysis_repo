"""Simplified, manuscript-facing revision figures (see REVISION_OUTPUTS_FOCUSED/SIMPLIFIED_FIGURE_PLAN.md and
ANALYSIS_METHODS.md for how every panel is derived).

Uses completed outputs only: the verified aMAP validation cache and tables, the QC-sensitivity run sets and maps,
the independent-LSFM consensus and reference volumes, and the rerun original-LSFM registrations. Nothing upstream is
recomputed.

  Fig1  aMAP expert validation: five-panel rows (165 RSP, 165 MV) + per-structure comparison of annotators,
        single retained registrations and the consensus
  Fig2  QC: A-C IoU x grad_p95 per sample (original LSFM numbered Sample 1-5 as in the paper; aMAP / independent LSFM keep their IDs); D change in consensus
        Dice when the current thresholds are replaced by each alternative QC rule (aMAP, per case)
  Fig3  consensus vs single registrations (aMAP): per-case ranges grouped by structure and labelled by brain; four
        anatomical examples
  Fig4  independent LSFM vs an independent registration (not ground truth)
  FigS1 what the consensus threshold does (165 RSP)
  FigS2 partial dorsal imaging
  FigS3 consensus boundaries under the QC rules, annotated hemisphere only (a: aMAP; b: original LSFM SNpc)

QC rules (fixed; defined by retention, see ANALYSIS_METHODS.md): no global QC; untuned start (median -/+ 3 MAD);
very lenient (~95 % retained); lenient (~85 % retained); current; strict (~60 % of current's retention); very strict (~30 %); median (both
thresholds at the dataset medians of the raw sweep).

Colours: individual annotations orange (thin); expert STAPLE white; single registrations sky blue (thin); pipeline
consensus purple. Dice = aMAP protocol unless stated.
Output: <IMAGING>/Ishrat/JoM/post_review_analysis/REVISION_OUTPUTS_FOCUSED/figures/<name>.{pdf,svg,png}, source_data/*.csv
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from common import SHORT, STRUCTS_MAIN, TABLES, VA, amap_brain, log  # noqa: E402

PRA = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis"))
OUT = PRA / "REVISION_OUTPUTS_FOCUSED"
FIG, SRC = OUT / "figures", OUT / "source_data"
for d in (FIG, SRC):
    d.mkdir(parents=True, exist_ok=True)
QS = VA / "results" / "qc_sensitivity"
PAPER_REG = Path(P(r"<SCRATCH>\jom_review_sweep_original_data"))

C_ANN, C_REF, C_RUN, C_CONS, C_DEF = "#E69F00", "white", "#56B4E9", "#7B3294", "black"
C_ANN_BOX, C_RUN_BOX, C_CONS_BOX = "#fbe3b7", "#cde8f7", "#e3d3ec"
REGIMES = {  # name -> setting_id per dataset
    "No global QC": {"amap": "iounone_gradnone", "paper": "iounone_gradnone", "lsfm2": "iounone_gradnone"},
    "Untuned start (median ± 3·MAD)": {"amap": "iou0p8537_grad0p0528", "paper": "iou0p8107_grad0p0795",
                                       "lsfm2": "iou0p7455_grad0p0847"},
    "Very lenient": {"amap": "iou0p8_grad0p08", "paper": "iou0p84_grad0p08", "lsfm2": "iou0p5_grad0p13"},
    "Lenient": {"amap": "iou0p875_grad0p065", "paper": "iou0p86_grad0p07", "lsfm2": "iou0p6_grad0p1"},
    "Current": {"amap": "iou0p9_grad0p058", "paper": "iou0p9_grad0p045", "lsfm2": "iou0p8_grad0p08"},
    "Strict": {"amap": "iou0p9078_grad0p055", "paper": "iou0p91_grad0p042", "lsfm2": "iou0p82_grad0p07"},
    "Very strict": {"amap": "iou0p9078_grad0p047", "paper": "iou0p92_grad0p038", "lsfm2": "iou0p84_grad0p067"},
    "Median IoU and grad_p95": {"amap": "iou0p9078_grad0p0454", "paper": "iou0p9016_grad0p042",
                                "lsfm2": "iou0p8632_grad0p067"},
}
SAMPLE_LISTS = {
    "paper LSFM": PRA / "original_dataset" / "config" / "samples.csv",
    "aMAP": PRA / "config" / "samples_amap_validation_raw.csv",
    "validation LSFM": PRA / "validation_datasets" / "lightsheet_data" / "config" / "samples_raw.csv",
}
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
BOX = dict(facecolor="black", alpha=0.6, pad=1.2, lw=0)


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


f01 = _load("f01", VA / "focused" / "scripts" / "f01_focused_validation.py")
m3, nifti, LAY = f01.m3, f01.nifti, f01.LAY


# ------------------------------------------------------------------------------------------------------------
# sample labels: the original LSFM samples are numbered as in the paper (order of its pipeline sample list; mapping
# written to source_data); aMAP keeps its short brain IDs and the independent LSFM its own sample IDs
# ------------------------------------------------------------------------------------------------------------
def sample_numbers():
    rows = []
    for ds, p in SAMPLE_LISTS.items():
        ids = pd.read_csv(p).sample_id.astype(str).drop_duplicates().tolist()
        for k, sid in enumerate(ids, 1):
            key = amap_brain(sid) if ds == "aMAP" else sid
            rows.append(dict(dataset=ds, sample_number=k, label=f"Sample {k}", key=key, sample_id=sid))
    m = pd.DataFrame(rows)
    m[m.dataset == "paper LSFM"].drop(columns="key").to_csv(SRC / "sample_number_mapping.csv", index=False)
    return m


SN = sample_numbers()
NUM = {(r.dataset, r.key): r.sample_number for r in SN.itertuples()}   # also the display order within a dataset


def sname(ds, key):
    key = str(key)
    if ds == "paper LSFM":
        return f"Sample {NUM[(ds, key)]}"
    return SHORT.get(key, key) if ds == "aMAP" else key


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"  {name}")


def label(ax, s, x=-0.04, y=1.03):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def clean(ax):
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def contour(ax, m, color, lw, ls="-", alpha=1.0):
    if m is not None and np.asarray(m).any():
        ax.contour(np.asarray(m, float), [0.5], colors=[color], linewidths=lw, linestyles=[ls], alpha=alpha)


def scalebar(ax, h, px=20, txt="500 µm"):
    ax.plot([6, 6 + px], [h - 4, h - 4], color="white", lw=1.8, solid_capstyle="butt")
    ax.text(6, h - 6, txt, color="white", ha="left", va="bottom", fontsize=5.5)


def tag(ax, txt, fs=6):
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top", color="white", fontsize=fs, bbox=BOX)


def retention(ds, variant="raw"):
    lk = pd.read_csv(QS / ds / "setting_lookup.csv")
    g = lk[lk.variant == variant].drop_duplicates(["sample", "setting_id"])
    tot = 48 if ds == "paper" else 64
    return {name: 100 * g[g.setting_id == ids[ds]].n_global.sum() / (tot * g["sample"].nunique())
            for name, ids in REGIMES.items()}


def regime_style(ds):
    """Regimes ordered by retention (most lenient first); purple ramp light -> dark, stroke widths wide -> thin."""
    ret = retention(ds)
    order = sorted(REGIMES, key=lambda n: -ret[n])
    cols = ["#bdbdbd"] + list(plt.get_cmap("Purples")(np.linspace(0.3, 1.0, len(order) - 1)))
    lws = np.linspace(3.4, 0.6, len(order))
    return order, dict(zip(order, cols)), dict(zip(order, lws)), ret


# ------------------------------------------------------------------------------------------------------------
# aMAP helpers
# ------------------------------------------------------------------------------------------------------------
QC = None


def qc():
    global QC
    if QC is None:
        QC = m3.load_qc()
    return QC


def amap_case(brain, structure, runs=None):
    cs, final = f01.load(brain, structure, qc())
    used = runs if runs is not None else (final if final else cs.runs)
    cons = cs.consensus(used, 0.5)
    d = cs.slice_dice(cons)
    k = cs.slab_idx[int(np.nanargmax(d[cs.slab_idx]))]              # aMAP protocol: best section in the raters' stack
    return cs, used, k, float(d[k])


def amap_view(cs, used, k, extra=(), margin=8):
    cnt, n, keep = cs.probability(used)
    p = np.where(keep[k], cnt[k] / n, 0.0)
    cons = (cnt[k] >= 0.5 * n - 1e-6) & keep[k]
    union = cs.truth | (p > 0)
    for e in extra:
        union = union | e
    rr, cc = np.flatnonzero(union.any(1)), np.flatnonzero(union.any(0))
    s0, _, r0, _ = [int(v) for v in cs.bbox]
    brain = np.asarray(nifti.load(LAY.brain_img(cs.brain), mmap=True)[cs.A_list[k]], float)
    half = (max(rr[-1] - rr[0], cc[-1] - cc[0]) + 1) // 2 + margin
    cy, cx = s0 + (rr[0] + rr[-1]) // 2, r0 + (cc[0] + cc[-1]) // 2
    Y0 = int(np.clip(cy - half, 0, brain.shape[0] - 2 * half))
    X0 = int(np.clip(cx - half, 0, brain.shape[1] - 2 * half))
    img = brain[Y0:Y0 + 2 * half, X0:X0 + 2 * half]
    y0, x0 = Y0 - s0, X0 - r0
    H, W = img.shape

    def put(a):
        a = np.asarray(a)
        out = np.zeros(a.shape[:-2] + (H, W), a.dtype)
        ys, xs = max(0, -y0), max(0, -x0)
        ye, xe = min(H, a.shape[-2] - y0), min(W, a.shape[-1] - x0)
        out[..., ys:ye, xs:xe] = a[..., ys + y0:ye + y0, xs + x0:xe + x0]
        return out
    lo, hi = np.percentile(img, [1, 99.5])
    return dict(img=np.clip((img - lo) / (hi - lo + 1e-9), 0, 1), truth=put(cs.truth), p=put(p), cons=put(cons),
                put=put)


def base(ax, v):
    ax.imshow(v["img"], cmap="gray", interpolation="nearest")
    clean(ax)
    scalebar(ax, v["img"].shape[0])


def amap_selection():
    s = pd.read_csv(QS / "amap" / "selection.csv")
    s["brain"] = s["sample"].map(amap_brain)
    s["structure"] = s.region.str.replace("_both", "", regex=False)
    s["run_list"] = s.runs.map(lambda r: r.split(";") if isinstance(r, str) and r else [])
    return s


# ------------------------------------------------------------------------------------------------------------
def draw_row(axs, final, v, sc, n_users=None):
    """The five focused-figure panels (f01.draw_row) with the pipeline consensus in purple."""
    o = lambda ax, m, c, lw, a=1.0, ls="-": f01.outline(ax, m, c, lw, a, ls)
    n_r = len(v["raters"])
    f01.base(axs[0], v)
    for m in v["raters"]:
        o(axs[0], m, C_ANN, 0.5, 0.45)
    o(axs[0], v["truth"], C_REF, 1.3)
    rep = f"\n({n_users} raters × 2 attempts)" if n_users and n_r == 2 * n_users else ""
    axs[0].text(0.02, 0.98, f"{n_r} annotations{rep}", transform=axs[0].transAxes, color="white", va="top",
                fontsize=5.5, bbox=BOX)
    f01.base(axs[1], v, False)
    for m in v["runs"]:
        o(axs[1], m, C_RUN, 0.4, 0.35)
    axs[1].text(0.02, 0.98, f"n = {len(v['runs'])}", transform=axs[1].transAxes, color="white", va="top",
                fontsize=5.5, bbox=BOX)
    f01.base(axs[2], v, False)
    for r in final:
        o(axs[2], v["runs"][v["idx"][r]], C_RUN, 0.5, 0.7)
    o(axs[2], v["truth"], C_REF, 0.9, 0.9, "--")
    axs[2].text(0.02, 0.98, f"n = {len(final)} retained", transform=axs[2].transAxes, color="white", va="top",
                fontsize=5.5, bbox=BOX)
    f01.base(axs[3], v, False)
    ov = np.zeros(v["p"].shape + (4,))
    ov[..., :3] = matplotlib.colors.to_rgb(C_CONS)
    ov[..., 3] = 0.55 * v["p"]
    axs[3].imshow(ov, interpolation="nearest")
    o(axs[3], v["p"] >= 0.5, C_CONS, 1.2)
    o(axs[3], v["truth"], C_REF, 1.2)
    axs[3].text(0.02, 0.98, f"Dice {sc['dice_amap_protocol']:.2f} (aMAP protocol)", transform=axs[3].transAxes,
                color="white", va="top", fontsize=5, bbox=BOX)
    f01.base(axs[4], v, False)
    for m in v["raters"]:
        o(axs[4], m, C_ANN, 0.5, 0.45)
    o(axs[4], v["p"] >= 0.5, C_CONS, 1.3)
    axs[4].text(0.02, 0.98, f"consensus vs {n_r} annotations", transform=axs[4].transAxes, color="white", va="top",
                fontsize=5.5, bbox=BOX)


def view_at(cs, final, A, margin=10):
    """f01.view, but drawn at coronal section A (same index scale as cs.A0 / A_best) instead of the STAPLE section.
    The expert outlines and the STAPLE consensus are 2-D and section-free, so they are drawn unchanged; the brain
    image, the single registrations and the consensus probability are taken at section A."""
    i = cs.A_list.index(A)
    runs_m = np.stack([cs.run_mask(r)[i] for r in cs.runs])
    union = runs_m.any(0) | cs.truth | cs.raters.any(0)
    rr, cc = np.flatnonzero(union.any(1)), np.flatnonzero(union.any(0))
    s0, _, r0, _ = [int(v) for v in cs.bbox]
    brain = np.asarray(nifti.load(LAY.brain_img(cs.brain), mmap=True)[A], float)
    half = (max(rr[-1] - rr[0], cc[-1] - cc[0]) + 1) // 2 + margin
    cy, cx = s0 + (rr[0] + rr[-1]) // 2, r0 + (cc[0] + cc[-1]) // 2
    Y0 = int(np.clip(cy - half, 0, brain.shape[0] - 2 * half)); X0 = int(np.clip(cx - half, 0, brain.shape[1] - 2 * half))
    img = brain[Y0: Y0 + 2 * half, X0: X0 + 2 * half]
    y0, x0 = Y0 - s0, X0 - r0
    lo, hi = np.percentile(img, [1, 99.5])
    used = final if final else cs.runs
    cnt, n, keep = cs.probability(used)
    p = np.where(keep[i], cnt[i] / n, 0.0)
    H = img.shape[0]

    def put(a):
        a = np.asarray(a)
        out = np.zeros(a.shape[:-2] + (H, img.shape[1]), a.dtype)
        ys, xs = max(0, -y0), max(0, -x0)
        ye, xe = min(H, a.shape[-2] - y0), min(img.shape[1], a.shape[-1] - x0)
        out[..., ys:ye, xs:xe] = a[..., ys + y0: ye + y0, xs + x0: xe + x0]
        return out
    return dict(img=np.clip((img - lo) / (hi - lo + 1e-9), 0, 1), runs=put(runs_m),
                idx={r: k for k, r in enumerate(cs.runs)}, truth=put(cs.truth), raters=put(cs.raters), p=put(p))


def fig1(sections=None, name="Fig1_expert_anatomical_validation", src_name="Fig1_AE_examples.csv", sub_dorsal=False):
    """sections: optional {(brain, structure): coronal section A} to draw a row at a section other than the STAPLE
    section. Panel D still reports the case's aMAP-protocol Dice (best section in the stack), as in F."""
    sections = sections or {}
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.05], hspace=0.12, wspace=0.05)
    titles = ["Expert annotations\n+ STAPLE", "All registrations\n(before QC)", "Registrations\nretained after QC",
              "Pipeline consensus\nvs STAPLE", "Pipeline consensus\nvs annotations"]
    ex = []
    for row, (b, s) in enumerate(f01.EXAMPLES):
        cs, final = f01.load(b, s, qc())
        A = sections.get((b, s))
        v = view_at(cs, final, A) if A is not None else f01.view(cs, final)
        sc = f01.case_scores(cs, final)
        if A is not None:                                    # Dice of the consensus at the section actually drawn
            m_ = cs.consensus(final if final else cs.runs, 0.5)
            sc["section_drawn"] = A
            sc["dice_at_section_drawn"] = float(2 * (m_[cs.A_list.index(A)] & cs.truth).sum() /
                                                (m_[cs.A_list.index(A)].sum() + cs.truth.sum()))
        else:
            sc["section_drawn"] = cs.A0
        axs = [fig.add_subplot(gs[row, c]) for c in range(5)]
        draw_row(axs, final, v, sc, n_users=len(set(cs.rater_user)))
        axs[0].text(-0.08, 0.5, f"{sname('aMAP', b)}  {s}", transform=axs[0].transAxes, rotation=90, va="center",
                    ha="right", fontsize=7)
        if row == 0:
            for ax, t, l in zip(axs, titles, "ABCDE"):
                ax.set_title(t, fontsize=6.5)
                label(ax, l, x=-0.04, y=1.22)
        ex.append(dict(brain=b, sample=sname("aMAP", b), structure=s, **sc))
    cases = pd.read_csv(VA / "focused" / "tables" / "focused_case_scores.csv")
    cases = cases[cases.structure.isin(STRUCTS_MAIN)]
    raters = pd.read_csv(VA / "focused" / "tables" / "focused_rater_scores.csv")
    raters = raters[raters.structure.isin(STRUCTS_MAIN)]
    runs = pd.read_csv(TABLES / "amap_runs.csv")
    runs = runs[(runs.variant == "raw") & runs.structure.isin(STRUCTS_MAIN) & runs.passes_final_qc.astype(bool)]
    extra = None
    if sub_dorsal:   # sensitivity analysis: SUB scored on its dorsal part only (34_sub_dorsal_only.py), shown as an
        # extra group after the main structures; the main groups keep the atlas-defined SUB
        sd = pd.read_csv(SRC / "SUB_dorsal_only_per_case.csv")
        sg = pd.read_csv(SRC / "SUB_dorsal_only_singles.csv")
        extra = dict(ann=raters[raters.structure == "SUB"].dice, run=sg.dice_dorsal_only,
                     cons=sd.dice_consensus_dorsal_only, brains=sd.brain)
    ax = fig.add_subplot(gs[2, :])
    bxp = dict(showfliers=False, patch_artist=True, widths=0.22, medianprops=dict(color="k", lw=0.8),
               whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    isub = STRUCTS_MAIN.index("SUB")
    pos = {st: k + (1 if (extra is not None and k > isub) else 0) for k, st in enumerate(STRUCTS_MAIN)}
    for st in STRUCTS_MAIN:
        j = pos[st]
        c = cases[cases.structure == st].dice_amap_protocol
        for dx, vals, fc in ((-0.26, raters[raters.structure == st].dice, C_ANN_BOX),
                             (0.0, runs[runs.structure == st].dice_paper, C_RUN_BOX), (0.26, c, C_CONS_BOX)):
            bp = ax.boxplot(vals, positions=[j + dx], **bxp)
            bp["boxes"][0].set(facecolor=fc, edgecolor="0.4", lw=0.6)
        ax.scatter(np.full(len(c), j + 0.26) + np.linspace(-0.06, 0.06, len(c)), c, s=11, color=C_CONS,
                   edgecolor="white", lw=0.3, zorder=3)
    ticks = list(STRUCTS_MAIN)
    if extra is not None:
        j = isub + 1                                       # directly after the atlas-defined SUB
        for dx, vals, fc in ((-0.26, extra["ann"], C_ANN_BOX), (0.0, extra["run"], C_RUN_BOX),
                             (0.26, extra["cons"], C_CONS_BOX)):
            bp = ax.boxplot(vals, positions=[j + dx], **bxp)
            bp["boxes"][0].set(facecolor=fc, edgecolor="0.4", lw=0.6)
            bp["boxes"][0].set_hatch("////")
            bp["boxes"][0].set_linewidth(0.6)
        c = extra["cons"]
        ax.scatter(np.full(len(c), j + 0.26) + np.linspace(-0.06, 0.06, len(c)), c, s=11, color=C_CONS,
                   edgecolor="white", lw=0.3, zorder=3)
        ax.axvspan(j - 0.48, j + 0.48, color="0.95", zorder=0, lw=0)
        ax.text(j, 1.0, "dorsal\npart only", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=5.8, color="0.3")
        order = sorted(list(pos.items()) + [("SUB*", j)], key=lambda t: t[1])
        ax.set_xticks([t[1] for t in order], [t[0] for t in order])
        ax.set_xlim(-0.6, len(STRUCTS_MAIN) + 0.6)
        ax.text(0.5, -0.36, "* Sensitivity analysis: SUB scored on its dorsal part only. The ventral subiculum, which the experts did\n"
                "not annotate, was excluded for the consensus and the single registrations; expert scores unchanged.",
                transform=ax.transAxes, ha="center", va="top", fontsize=5.5, color="0.3", linespacing=1.2)
    else:
        ax.set_xticks(range(len(STRUCTS_MAIN)), ticks)
    if extra is None:
        ax.set_xlim(-0.6, len(STRUCTS_MAIN) - 0.4)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Dice with expert consensus (STAPLE)")
    per = raters.groupby(["brain", "structure"]).size()
    rep = sorted({SHORT.get(b, b) for (b, _), n in per.items() if n > per.min()})
    ann_lab = (f"individual annotations ({per.min()} per structure and sample; {per.max()} for "
               f"{', '.join(rep)}: two attempts)" if rep else f"individual annotations ({per.min()} per sample)")
    ax.legend(handles=[Patch(facecolor=C_ANN_BOX, edgecolor="0.4", label=ann_lab),
                       Patch(facecolor=C_RUN_BOX, edgecolor="0.4", label="single registrations retained after QC"),
                       Patch(facecolor=C_CONS_BOX, edgecolor="0.4", label="pipeline consensus (dots: one per sample)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=6)
    label(ax, "F", x=-0.045, y=1.0)
    handles = [Line2D([], [], color=C_ANN, lw=0.8, alpha=0.7, label="individual annotation"),
               Line2D([], [], color=C_REF, lw=1.3, label="expert consensus (STAPLE)"),
               Line2D([], [], color=C_RUN, lw=0.8, label="single registration"),
               Line2D([], [], color="#b88fd0", lw=1.3, label="pipeline consensus (p ≥ 0.5)")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True, fontsize=6, bbox_to_anchor=(0.5, 0.925))
    leg.get_frame().set_facecolor("#555555")
    for t in leg.get_texts():
        t.set_color("white")
    save(fig, name)
    pd.DataFrame(ex).to_csv(SRC / src_name, index=False)
    if name != "Fig1_expert_anatomical_validation":
        f = pd.concat([raters.assign(group="annotator")[["group", "brain", "structure", "dice"]],
                       runs.assign(group="single retained registration").rename(columns={"dice_paper": "dice"})[
                           ["group", "brain", "structure", "dice"]],
                       cases.assign(group="pipeline consensus").rename(columns={"dice_amap_protocol": "dice"})[
                           ["group", "brain", "structure", "dice"]]])
        if extra is not None:
            f = pd.concat([f,
                           raters[raters.structure == "SUB"].assign(group="annotator", structure="SUB (dorsal part only)")[
                               ["group", "brain", "structure", "dice"]],
                           sg.assign(group="single retained registration", structure="SUB (dorsal part only)",
                                     dice=sg.dice_dorsal_only)[["group", "brain", "structure", "dice"]],
                           sd.assign(group="pipeline consensus", structure="SUB (dorsal part only)",
                                     dice=sd.dice_consensus_dorsal_only)[["group", "brain", "structure", "dice"]]])
        f.to_csv(SRC / src_name.replace("AE_examples", "F_scores"), index=False)
        log(f"medians: consensus {cases.dice_amap_protocol.median():.3f}, annotators {raters.dice.median():.3f}, "
            f"retained registrations {runs.dice_paper.median():.3f}")
        return pd.DataFrame(ex)
    pd.concat([raters.assign(group="annotator")[["group", "brain", "structure", "dice"]],
               runs.assign(group="single retained registration").rename(columns={"dice_paper": "dice"})[
                   ["group", "brain", "structure", "dice"]],
               cases.assign(group="pipeline consensus").rename(columns={"dice_amap_protocol": "dice"})[
                   ["group", "brain", "structure", "dice"]]]).assign(
        sample=lambda d: d.brain.map(lambda b: sname("aMAP", b))).to_csv(SRC / "Fig1_F_scores.csv", index=False)
    return dict(consensus_median=cases.dice_amap_protocol.median(), annotator_median=raters.dice.median(),
                retained_registration_median=runs.dice_paper.median())


def median_rule_explained(allqc):
    """Why 'both thresholds at the median' retains so few: each median alone passes 50 %, and the joint pass rate
    depends on how IoU and grad_p95 co-vary (raw sweep; the two datasets with QC-sensitivity runs)."""
    rows = []
    for ds, cur in (("aMAP", (0.9, 0.058)), ("paper LSFM", (0.9, 0.045))):
        r = allqc[(allqc.dataset == ds) & (allqc.variant == "raw")]
        mi, mg = r.iou.median(), r.grad_p95.median()
        good = r[(r.iou > cur[0]) & (r.grad_p95 < cur[1])]
        pi, pg = (r.iou > mi).mean(), (r.grad_p95 < mg).mean()
        rows.append(dict(dataset=ds, n_registrations=len(r), median_iou=mi, median_grad_p95=mg,
                         pass_iou_above_median=pi, pass_grad_below_median=pg,
                         pass_both_medians=((r.iou > mi) & (r.grad_p95 < mg)).mean(), expected_if_independent=pi * pg,
                         spearman_iou_grad_all=r.iou.corr(r.grad_p95, method="spearman"),
                         spearman_iou_grad_current_passers=good.iou.corr(good.grad_p95, method="spearman"),
                         current_iou=cur[0], current_grad=cur[1], pass_current=len(good) / len(r),
                         pass_current_iou_only=(r.iou > cur[0]).mean(), pass_current_grad_only=(r.grad_p95 < cur[1]).mean()))
    pd.DataFrame(rows).to_csv(SRC / "Fig2_D_median_rule_explained.csv", index=False)


def fig2(split=False, same_x=False, fig_name="Fig2_QC_thresholds_and_sensitivity", lsfm=False):
    """split=False: the original layout (A-C, raw and preprocessed overlaid).
    split=True: A-C become 2 x 3 panels (top raw, bottom preprocessed); grad_p95 is always on one scale.
    same_x=True additionally puts the IoU axis of all six panels on one scale; otherwise each dataset keeps its own
    IoU range (shared by its raw and preprocessed panels). D is identical in every version."""
    f19 = _load("f19", VA / "scripts" / "19_figure3_qc.py")
    qc12 = _load("qc12", VA / "scripts" / "12_qc_threshold_procedure.py")
    import yaml
    allqc = qc12.load_all()
    median_rule_explained(allqc)
    allqc["sample"] = [sname(ds, s) for ds, s in zip(allqc.dataset, allqc["sample"].astype(str))]
    thr = {ds: tuple(yaml.safe_load(p.read_text())["qc"][k] for k in ("iou_threshold", "grad_p95_threshold"))
           for ds, p in f19.CONFIGS.items()}
    ylim = (0.028, allqc.grad_p95.max() * 1.04)
    dsets = ["paper LSFM", "aMAP", "validation LSFM"]
    if not split:
        fig = plt.figure(figsize=(6.85, 5.5))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.8, wspace=0.32)
        for k, ds in enumerate(dsets):
            ax = fig.add_subplot(gs[0, k])
            f19.scatter(ax, allqc[allqc.dataset == ds], thr[ds], ylim, show_marker_legend=(k == 0))
            ax.set_title(f19.TITLES[ds], fontsize=7, pad=3)
            if k:
                ax.set_ylabel("")
            label(ax, "ABC"[k], x=-0.22 if k == 0 else -0.12, y=1.06)
        gD = gs[1, :]
    else:
        # thresholds of the Current rule as applied to the pipeline outputs (qc_sensitivity settings.csv), not the
        # live config files: the independent-LSFM config.yaml was edited on 1 Oct to grad_p95 0.075, while every
        # pipeline output and analysis uses 0.08
        key = {"paper LSFM": "paper", "aMAP": "amap", "validation LSFM": "lsfm2"}
        for ds, kk in key.items():
            st = pd.read_csv(QS / kk / "settings.csv").set_index("setting_id").loc[REGIMES["Current"][kk]]
            thr[ds] = (float(st.iou_thr), float(st.grad_thr))
        if lsfm:
            fig = plt.figure(figsize=(6.85, 10.4))
            outer = fig.add_gridspec(3, 1, height_ratios=[2.15, 1.15, 1.2], hspace=0.62)
        else:
            fig = plt.figure(figsize=(6.85, 7.9))
            outer = fig.add_gridspec(2, 1, height_ratios=[2.15, 1.15], hspace=0.42)
        gs = outer[0].subgridspec(2, 3, hspace=0.3, wspace=0.32)
        pad = lambda lo, hi: (lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo))  # noqa: E731
        xall = pad(allqc.iou.min(), allqc.iou.max())
        for k, ds in enumerate(dsets):
            d = allqc[allqc.dataset == ds]
            xl = xall if same_x else pad(d.iou.min(), d.iou.max())
            for j, v in enumerate(("raw", "preprocessed")):
                ax = fig.add_subplot(gs[j, k])
                f19.scatter(ax, d[d.variant == v], thr[ds], ylim, show_marker_legend=False)
                ax.set_xlim(*xl)
                if j == 0:
                    ax.get_legend().remove()
                    ax.set_xlabel("")
                    ax.set_title(f19.TITLES[ds], fontsize=7, pad=3)
                    label(ax, "ABC"[k], x=-0.22 if k == 0 else -0.12, y=1.06)
                else:
                    ax.get_legend().set_bbox_to_anchor((0.5, -0.24))
                if k:
                    ax.set_ylabel("")
                else:
                    ax.set_ylabel(f"{v.capitalize()}\ngrad_p95")
                n_pass = int(((d[d.variant == v].iou > thr[ds][0]) & (d[d.variant == v].grad_p95 < thr[ds][1])).sum())
                ax.text(0.03 if same_x or ds != "paper LSFM" else 0.97, 0.97,
                        f"{v}: {n_pass}/{(d.variant == v).sum()} pass", transform=ax.transAxes, fontsize=5.5,
                        ha="left" if same_x or ds != "paper LSFM" else "right", va="top", color="0.3")
        gD = outer[1]
        pd.DataFrame([dict(dataset=ds, iou_threshold=t[0], grad_p95_threshold=t[1],
                           source="qc_sensitivity/<ds>/settings.csv, Current rule") for ds, t in thr.items()]).to_csv(
            SRC / "Fig2_split_ABC_thresholds.csv", index=False)
    allqc[["dataset", "variant", "sample", "bending_energy", "grid", "iou", "grad_p95"]].to_csv(
        SRC / "Fig2_ABC_qc_scatter.csv", index=False)
    per = pd.read_csv(TABLES / "qcsens_amap_per_case.csv")
    per = per[(per.variant == "raw") & per.structure.isin(STRUCTS_MAIN)]
    w = per.pivot_table(index=["brain", "structure"], columns="setting_id", values="dice_paper")
    order, cols, _, ret = regime_style("amap")
    alts = [n for n in order if n != "Current"]
    ax = fig.add_subplot(gD)
    rng = np.random.default_rng(0)
    rows = []
    for i, name in enumerate(alts):
        sid = REGIMES[name]["amap"]
        dd = (w[sid] - w[REGIMES["Current"]["amap"]]).dropna()
        n_missing = int(w[sid].isna().sum())
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(dd)), dd, s=8, color="0.45", lw=0, alpha=0.7)
        ax.plot([i - 0.28, i + 0.28], [dd.median()] * 2, color=C_CONS, lw=1.6)
        rows.append(dict(regime=name, setting_id=sid, retained_pct=ret[name], n_cases=len(dd),
                         n_cases_without_registrations=n_missing, median_delta=dd.median(),
                         max_abs_delta=dd.abs().max(), frac_within_0p02=float((dd.abs() <= 0.02).mean())))
    ax.axhline(0, color="k", lw=0.6)
    ax.axhspan(-0.02, 0.02, color="0.92", zorder=0)
    wrap = {"Untuned start (median ± 3·MAD)": "Untuned start\n(median ± 3·MAD)",
            "Median IoU and grad_p95": "Dataset\nmedians"}
    lab = [f"{wrap.get(n, n)}\n{ret[n]:.0f} % retained" + (f"\n({r['n_cases_without_registrations']} cases empty)"
                                              if r["n_cases_without_registrations"] else "") for n, r in zip(alts, rows)]
    ax.set_xticks(range(len(alts)), lab, fontsize=6)
    ax.set_ylabel("Change in consensus Dice\nvs current thresholds")
    ax.set_xlim(-0.6, len(alts) - 0.4)
    ax.text(len(alts) - 0.45, 0.021, "±0.02", fontsize=5.5, color="0.4", ha="right", va="bottom")
    ax.set_title(f"aMAP (expert STAPLE), 54 sample × structure cases; current thresholds retain {ret['Current']:.0f} %",
                 fontsize=6.5, pad=3)
    label(ax, "D", x=-0.07, y=1.02)
    if split and lsfm:
        lsfm_rule_panels(fig, outer[2])
    save(fig, fig_name)
    pd.DataFrame(rows).to_csv(SRC / "Fig2_D_summary.csv", index=False)
    (w.sub(w[REGIMES["Current"]["amap"]], axis=0)).reset_index().to_csv(SRC / "Fig2_D_delta_per_case.csv", index=False)
    return pd.DataFrame(rows)


SHORT_RULE = {"No global QC": "No QC", "Very lenient": "Very lenient", "Lenient": "Lenient",
              "Untuned start (median ± 3·MAD)": "Untuned start", "Current": "Current", "Strict": "Strict",
              "Very strict": "Very strict", "Median IoU and grad_p95": "Dataset medians"}


def lsfm_rule_panels(fig, g):
    """E-G: original-LSFM SNpc under each alternative rule vs the current rule, from the full-resolution consensus
    regions and Cellpose cells (QC_REGIME_FULLRES_AXIAL_MASKS; 30/31 scripts; = Supplementary Fig. S7A-C data).
    Rules ordered by this dataset's own raw retention. One point per sample (colour, as A) x input (circle raw,
    triangle preprocessed); the empty case (no registration passes) is annotated, not plotted."""
    ag = pd.read_csv(SRC / "FigS7_A-C_original_LSFM_vs_current.csv")
    order, _, _, ret = regime_style("paper")
    alts = [n for n in order if n != "Current"]
    labels = sorted(ag.sample_label.unique())
    okabe = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    colr = {b: okabe[k] for k, b in enumerate(labels)}
    gg = g.subgridspec(1, 3, wspace=0.32)
    specs = [("voxel_dice", "SNpc voxel Dice\nvs current rule", (0.4, 1.02), 1.0, "E"),
             ("count_change_pct", "SNpc TH+ count change\nvs current rule (%)", (-60, 20), 0.0, "F"),
             ("frac_current_cells_kept", "Fraction of current rule's\ncells still counted", (0.4, 1.02), 1.0, "G")]
    for k, (col, ylab, yl, ref, let) in enumerate(specs):
        ax = fig.add_subplot(gg[0, k])
        for i, r in enumerate(alts):
            for off, v, mk in ((-0.17, "raw", "o"), (0.17, "preprocessed", "^")):
                h = ag[(ag.rule == r) & (ag.input == v) & ~ag.is_empty].sort_values("sample_label")
                x = i + off + np.linspace(-0.06, 0.06, len(h))
                ax.scatter(x, h[col], s=10, marker=mk, c=[colr[b] for b in h.sample_label], lw=0.3,
                           edgecolor="white", zorder=3)
                for _, e in ag[(ag.rule == r) & (ag.input == v) & ag.is_empty].iterrows():
                    ax.annotate(f"{e.sample_label} {v}:\nnone passes", (i + off, yl[0]), xytext=(0, 3),
                                textcoords="offset points", fontsize=4.6, ha="center", va="bottom", color="0.3")
        ax.axhline(ref, color="0.6", lw=0.5, ls=(0, (3, 2)), zorder=1)
        ax.set_xticks(range(len(alts)), [f"{SHORT_RULE[n]} ({ret[n]:.0f}%)" for n in alts], fontsize=5.4,
                      rotation=40, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(alts) - 0.4)
        ax.set_ylim(*yl)
        ax.set_ylabel(ylab, fontsize=6.2)
        ax.tick_params(axis="y", labelsize=5.8)
        ax.grid(axis="y", color="0.93", lw=0.4, zorder=0)
        label(ax, let, x=-0.3, y=1.02)
        if k == 0:
            ax.text(-0.3, 1.24, f"Original LSFM SNpc (full-resolution consensus and TH+ cells), each rule vs the current "
                                f"rule (current retains {ret['Current']:.0f} %; x labels: % of raw registrations retained)",
                    transform=ax.transAxes, fontsize=6.5, va="bottom")
        if k == 0:
            h = [Line2D([], [], ls="", marker="o", ms=3.2, color=colr[b], label=b) for b in labels]
            h += [Line2D([], [], ls="", marker="o", ms=3.2, color="0.4", label="raw"),
                  Line2D([], [], ls="", marker="^", ms=3.2, color="0.4", label="preprocessed")]
            ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.3, 1.07), fontsize=5.4, ncol=7,
                      handletextpad=0.2, columnspacing=0.9, frameon=False)
    ag.assign(rule_retention_raw_pct=ag.rule.map(ret)).to_csv(SRC / "Fig2v2_EFG_original_LSFM_vs_current.csv",
                                                             index=False)


def fig3():
    runs = pd.read_csv(TABLES / "amap_runs.csv")
    runs = runs[(runs.variant == "raw") & runs.structure.isin(STRUCTS_MAIN) & runs.passes_global_qc.astype(bool)]
    ben = pd.read_csv(TABLES / "amap_benefit_cases.csv")
    ben = ben[(ben.variant == "raw") & ben.structure.isin(STRUCTS_MAIN)]
    rng_ = runs.groupby(["brain", "structure"]).dice_paper.agg(["min", "max", "count"]).reset_index()
    d = ben.merge(rng_, on=["brain", "structure"])
    d["sample_number"] = d.brain.map(lambda b: NUM[("aMAP", b)])
    d["spread"] = d["max"] - d["min"]
    order = d.groupby("structure").cons_dice.median().sort_values(ascending=False).index.tolist()
    d["sorder"] = d.structure.map(order.index)
    d = d.sort_values(["sorder", "sample_number"]).reset_index(drop=True)
    gap = 1.0
    d["x"] = d.index + d.sorder * gap
    # examples: widest spread in VISam and AHN (the most variable structures besides VPM) and in RSP or SUB; VISp is
    # shown in brain AB (author's choice, 1 Oct 2026; the widest-spread VISp case is ER)
    top = d.sort_values("spread", ascending=False).drop_duplicates("structure")
    visp = d[(d.structure == "VISp") & (d.brain == "AB_Gad67_223")]
    pick = pd.concat([top[top.structure == "VISam"], visp, top[top.structure == "AHN"]])
    extra = top[top.structure.isin(["RSP", "SUB"])].head(1)
    ex = pd.concat([pick, extra])
    fig = plt.figure(figsize=(7.0, 5.1))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.45, wspace=0.05)
    ax = fig.add_subplot(gs[0, :])
    ax.vlines(d.x, d["min"], d["max"], color=C_RUN, lw=1.8, zorder=1)
    ax.scatter(d.x, d.default_dice, marker="x", s=12, color=C_DEF, lw=0.8, zorder=2)
    ax.scatter(d.x, d.cons_dice, s=12, color=C_CONS, lw=0, zorder=3)
    for r in ex.itertuples():
        ax.scatter(r.x, r.cons_dice, s=45, facecolor="none", edgecolor="k", lw=0.8, zorder=4)
    ax.set_xticks(d.x, d.brain.map(SHORT), fontsize=4.5, rotation=90)
    for st in order:
        xs = d.x[d.structure == st]
        ax.text(xs.mean(), -0.2, st, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5)
        ax.axvline(xs.max() + (gap + 1) / 2, color="0.9", lw=0.6, zorder=0)
    ax.text(-0.01, -0.035, "brain", transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color="0.4")
    ax.set_xlim(d.x.min() - 1, d.x.max() + 1)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Dice with expert consensus")
    ax.legend(handles=[Line2D([], [], color=C_RUN, lw=2, label="range of single QC-passing registrations"),
                       Line2D([], [], ls="", marker="x", color=C_DEF, label="brainreg default registration"),
                       Line2D([], [], ls="", marker="o", ms=4, color=C_CONS, label="pipeline consensus"),
                       Line2D([], [], ls="", marker="o", ms=6, mfc="none", mec="k", label="examples in B")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6)
    label(ax, "A", x=-0.05, y=1.06)
    for j, r in enumerate(ex.itertuples()):
        cs, used, k, _ = amap_case(r.brain, r.structure)
        qcruns = runs[(runs.brain == r.brain) & (runs.structure == r.structure)].run.tolist()
        masks = [cs.run_mask(q)[k] for q in qcruns if q in cs.idx]
        v = amap_view(cs, used, k, extra=masks)
        axb = fig.add_subplot(gs[1, j])
        base(axb, v)
        for m in masks:
            contour(axb, v["put"](m), C_RUN, 0.4, alpha=0.7)
        contour(axb, v["truth"], C_REF, 1.2)
        contour(axb, v["cons"], C_CONS, 1.4)
        axb.set_title(f"{r.structure}, {sname('aMAP', r.brain)}", fontsize=7, pad=2)
        tag(axb, f"single {r._asdict()['min']:.2f}–{r._asdict()['max']:.2f}\nconsensus {r.cons_dice:.2f}", 5.5)
        if j == 0:
            label(axb, "B", x=-0.06, y=1.08)
    fig.legend(handles=[Line2D([], [], color=C_RUN, lw=0.8, label="single QC-passing registrations"),
                        Line2D([], [], color="0.6", lw=1.2, label="expert consensus (STAPLE, white)"),
                        Line2D([], [], color=C_CONS, lw=1.4, label="pipeline consensus")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.06), fontsize=6)
    save(fig, "Fig3_consensus_vs_single_registrations")
    d.assign(sample=d.brain.map(lambda b: sname("aMAP", b)))[
        ["structure", "brain", "sample", "min", "max", "count", "default_dice", "cons_dice", "spread"]].to_csv(
        SRC / "Fig3_A_cases.csv", index=False)
    ex[["structure", "brain", "min", "max", "cons_dice", "spread"]].to_csv(SRC / "Fig3_B_examples.csv", index=False)
    return ex[["brain", "structure", "min", "max", "cons_dice"]]


def figS1():
    b, s = "MV_Ntsr1_165", "RSP"
    cs, used, k, dk = amap_case(b, s)
    v = amap_view(cs, used, k)
    fig = plt.figure(figsize=(6.4, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.6)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(v["img"], cmap="gray", interpolation="nearest")
    im = ax.imshow(np.ma.masked_where(v["p"] <= 0, v["p"]), cmap="Purples", vmin=0, vmax=1, alpha=0.75,
                   interpolation="nearest")
    tcol = {0.25: "#c6b4de", 0.5: C_CONS, 0.75: "#3f007d"}
    for t, ls in ((0.25, ":"), (0.5, "-"), (0.75, "--")):
        contour(ax, v["p"] >= t - 1e-9, tcol[t], 1.1, ls)
    contour(ax, v["truth"], C_REF, 1.2)
    clean(ax)
    scalebar(ax, v["img"].shape[0])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("registration agreement", fontsize=6)
    ax.legend(handles=[Line2D([], [], color=tcol[0.25], ls=":", label="threshold 0.25"),
                       Line2D([], [], color=tcol[0.5], label="threshold 0.5"),
                       Line2D([], [], color=tcol[0.75], ls="--", label="threshold 0.75"),
                       Line2D([], [], color="0.6", lw=1.2, label="expert consensus (white)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=6)
    ax.set_title(f"{s}, {sname('aMAP', b)} (as in Fig. 1)", fontsize=7, pad=2)
    label(ax, "A")
    sets = pd.read_csv(TABLES / "amap_sets.csv")
    sets = sets[(sets.variant == "raw") & (sets.set_name == "passes_final_qc_top_0p5") &
                sets.structure.isin(STRUCTS_MAIN)]
    g = sets.groupby("threshold").dice_paper
    q = pd.DataFrame(dict(q1=g.quantile(0.25), med=g.median(), q3=g.quantile(0.75))).reset_index()
    ax = fig.add_subplot(gs[0, 1])
    ax.fill_between(q.threshold, q.q1, q.q3, color=C_CONS, alpha=0.15, lw=0)
    ax.plot(q.threshold, q.med, color=C_CONS, lw=1.5)
    ax.axvline(0.5, color="0.5", lw=0.7, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("consensus threshold")
    ax.set_ylabel("Dice with expert consensus\n(median and IQR, 54 cases)")
    label(ax, "B", x=-0.22)
    save(fig, "FigS1_consensus_threshold")
    q.to_csv(SRC / "FigS1_B_dice_vs_threshold.csv", index=False)


# ------------------------------------------------------------------------------------------------------------
def fig4():
    gtu = _load("gt_utils", PRA / "validation_datasets" / "lightsheet_data" / "scripts" / "validation" / "gt_utils.py")
    c = pd.read_csv(PRA / "validation_datasets" / "lightsheet_data" / "results" / "validation_gt" / "consensus_vs_gt.csv")
    c = c[(c.subset == gtu.FINAL_QC_SUBSET) & (c.threshold == 0.5) & (c.input_type == "raw")]
    stat = c.groupby("region").agg(dice=("dice", "median"), vol_ratio=("vol_ratio", "median"))
    a = stat.dice.idxmax()
    b_ = (stat.dice - stat.dice.median()).abs().drop(a).idxmin()
    cand = stat.drop([a, b_])
    cc_ = cand[cand.vol_ratio > 1].dice.idxmax()                  # consensus larger than the reference, best overlap
    regions = [a, b_, cc_]
    _, agg = gtu.load_label_mapping()
    fig = plt.figure(figsize=(7.0, 3.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.75, 1], hspace=0.22, wspace=0.05)
    src = []
    for j, reg in enumerate(regions):
        g = c[c.region == reg].sort_values("dice").reset_index(drop=True)
        r = g.iloc[(len(g) - 1) // 2]
        gt = gtu.load_gt_atlas(r.sample_id)
        ref = gtu.gt_region_mask(gt, reg, agg)
        cons = gtu.load_mask(gtu.consensus_path(r.sample_id, "raw", reg, gtu.FINAL_QC_SUBSET, 0.5), gt.shape)
        img = gtu.read_tiff(gtu.image_path(r.sample_id)).astype(float)
        z = int(np.argmax(ref.sum((1, 2)) + cons.sum((1, 2))))
        u = ref[z] | cons[z]
        rr, cc = np.flatnonzero(u.any(1)), np.flatnonzero(u.any(0))
        sl = (slice(max(rr[0] - 10, 0), rr[-1] + 11), slice(max(cc[0] - 10, 0), cc[-1] + 11))
        im = img[z][sl]
        lo, hi = np.percentile(im, [1, 99.5])
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(np.clip((im - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray", interpolation="nearest")
        contour(ax, ref[z][sl], C_REF, 1.1, "--")
        contour(ax, cons[z][sl], C_CONS, 1.3)
        clean(ax)
        scalebar(ax, im.shape[0])
        ax.set_title(f"{reg}, {sname('validation LSFM', r.sample_id)} (median sample)", fontsize=7, pad=2)
        tag(ax, f"Dice {r.dice:.2f}", 6.5)
        label(ax, "ABC"[j])
        src.append(dict(panel="ABC"[j], region=reg, sample_id=r.sample_id,
                        sample=sname("validation LSFM", r.sample_id), slice=z, dice=r.dice, vol_ratio=r.vol_ratio,
                        rule=["highest median Dice", "median Dice nearest the median of regions",
                              "consensus larger than reference (median volume ratio > 1), highest median Dice"][j]))
    ax = fig.add_subplot(gs[1, :])
    vol = c.groupby("region").vol_gt_mm3.median().sort_values(ascending=False)
    for i, reg in enumerate(vol.index):
        y = c[c.region == reg].dice
        ax.scatter(np.full(len(y), i) + np.linspace(-0.12, 0.12, len(y)), y, s=11, color=C_CONS, lw=0)
        ax.plot([i - 0.25, i + 0.25], [y.median()] * 2, color="0.3", lw=1)
    ax.set_xticks(range(len(vol)), vol.index)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dice with independent registration")
    ax.set_xlabel("regions, largest → smallest (one dot per sample)")
    label(ax, "D", x=-0.06)
    fig.legend(handles=[Line2D([], [], color="0.6", lw=1.1, ls="--",
                               label="independent registration by the dataset authors (white, dashed)"),
                        Line2D([], [], color=C_CONS, lw=1.3, label="pipeline consensus")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    save(fig, "Fig4_independent_LSFM_agreement")
    pd.DataFrame(src).to_csv(SRC / "Fig4_ABC_examples.csv", index=False)
    c.assign(sample=c.sample_id.map(lambda s: sname("validation LSFM", s)))[
        ["sample_id", "sample", "region", "dice", "vol_gt_mm3", "vol_pred_mm3", "vol_ratio", "hd95_um"]].to_csv(
        SRC / "Fig4_D_regions.csv", index=False)
    return regions


def figS2():
    brains = ["162", "527", "pp451", "xl353", "xl385"]
    fig, axes = plt.subplots(len(brains), 1, figsize=(4.0, 1.25 * len(brains)))
    src = []
    for ax, b in zip(axes, brains):
        d = PAPER_REG / b / "raw" / "bend_0p95__grid_m10__smooth_m1p0"
        atl = tifffile.imread(d / "registered_atlas.tiff")
        img = tifffile.imread(d / "downsampled.tiff").astype(float)
        snpc = atl == 374
        _, _, xx = np.nonzero(snpc)
        side = xx < atl.shape[2] / 2
        x = int(np.median(xx[side] if side.any() else xx))
        sag, a2, s2 = img[:, :, x].T, (atl[:, :, x] > 0).T, snpc[:, :, x].T
        lo, hi = np.percentile(sag, [1, 99.5])
        ax.imshow(np.clip((sag - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray", interpolation="nearest")
        contour(ax, a2, C_REF, 0.8, "--")
        contour(ax, s2, C_CONS, 1.3)
        clean(ax)
        ax.plot([6, 56], [sag.shape[0] - 5] * 2, color="white", lw=1.8, solid_capstyle="butt")   # 50 px x 20 um = 1 mm
        ax.text(6, sag.shape[0] - 7, "1 mm", color="white", ha="left", va="bottom", fontsize=5.5)
        tag(ax, sname("paper LSFM", b), 6.5)
        src.append(dict(sample_id=b, sample=sname("paper LSFM", b), sagittal_x=x, registration=str(d)))
    fig.legend(handles=[Line2D([], [], color="0.6", lw=0.8, ls="--", label="registered atlas outline (white)"),
                        Line2D([], [], color=C_CONS, lw=1.3, label="SNpc (registered atlas)")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.93))
    fig.subplots_adjust(hspace=0.08)
    save(fig, "FigS2_partial_dorsal_imaging")
    pd.DataFrame(src).to_csv(SRC / "FigS2_sections.csv", index=False)


# ------------------------------------------------------------------------------------------------------------
def regime_legend(fig, ds, y, extra=()):
    order, cols, lws, ret = regime_style(ds)
    h = [Line2D([], [], color=cols[n], lw=lws[n], label=f"{n} ({ret[n]:.0f} % retained)") for n in order] + list(extra)
    fig.legend(handles=h, loc="upper center", ncol=3, fontsize=6, bbox_to_anchor=(0.5, y), handlelength=2.2)


def figS3a():
    sel = amap_selection()
    per = pd.read_csv(TABLES / "qcsens_amap_per_case.csv")
    per = per[(per.variant == "raw") & per.structure.isin(STRUCTS_MAIN)]
    w = per.pivot_table(index=["brain", "structure"], columns="setting_id", values="dice_paper")
    order, cols, lws, _ = regime_style("amap")
    cur = REGIMES["Current"]["amap"]
    dev = w[[REGIMES[n]["amap"] for n in order]].sub(w[cur], axis=0).abs()
    reasonable = [REGIMES[n]["amap"] for n in ("No global QC", "Very lenient", "Untuned start (median ± 3·MAD)", "Lenient")]
    examples = [(("MV_Ntsr1_165", "RSP"), "typical"), (("MV_Ntsr1_165", "MV"), "typical"),
                (dev[reasonable].max(axis=1).idxmax(), "largest change, lenient rules"),
                (dev.max(axis=1).idxmax(), "largest change, any rule")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.3))
    src = []
    for ax, ((b, s), why) in zip(axes, examples):
        runsets = {}
        for n in order:
            r = sel[(sel.brain == b) & (sel.structure == s) & (sel.variant == "raw") &
                    (sel.setting_id == REGIMES[n]["amap"])]
            runsets[n] = r.run_list.iloc[0] if len(r) else []
        cs, used, k, _ = amap_case(b, s, runsets["Current"])
        cons = {n: (cs.consensus(rs, 0.5)[k] if rs else None) for n, rs in runsets.items()}
        v = amap_view(cs, used, k, extra=[m for m in cons.values() if m is not None])
        base(ax, v)
        for n in order:
            if cons[n] is not None:
                contour(ax, v["put"](cons[n]), cols[n], lws[n])
        contour(ax, v["truth"], C_REF, 0.9, "--")
        dv = [w.loc[(b, s), REGIMES[n]["amap"]] for n in order if not np.isnan(w.loc[(b, s), REGIMES[n]["amap"]])]
        ax.set_title(f"{s}, {sname('aMAP', b)}\n({why})", fontsize=6.2, pad=2)
        tag(ax, f"Dice {min(dv):.2f}–{max(dv):.2f}", 5.5)
        for n in order:
            src.append(dict(brain=b, sample=sname("aMAP", b), structure=s, example=why, regime=n,
                            setting_id=REGIMES[n]["amap"], n_registrations=len(runsets[n]),
                            dice=w.loc[(b, s), REGIMES[n]["amap"]]))
    regime_legend(fig, "amap", 0.22, [Line2D([], [], color="0.6", lw=0.9, ls="--", label="expert STAPLE (white)")])
    fig.subplots_adjust(wspace=0.05, bottom=0.27)
    save(fig, "FigS3a_QC_regimes_boundaries_aMAP")
    pd.DataFrame(src).to_csv(SRC / "FigS3a_examples.csv", index=False)


def figS3b():
    lk = pd.read_csv(QS / "paper" / "setting_lookup.csv")
    lk["sample"] = lk["sample"].astype(str)
    per = pd.read_csv(TABLES / "qcsens_paper_per_case.csv")
    per["sample"] = per["sample"].astype(str)
    order, cols, lws, _ = regime_style("paper")
    brains = sorted(lk["sample"].unique(), key=lambda s: NUM[("paper LSFM", s)])
    fig, axes = plt.subplots(2, len(brains), figsize=(7.0, 3.4))
    src = []
    for i, v_ in enumerate(("raw", "preprocessed")):
        for j, s in enumerate(brains):
            ax = axes[i, j]
            paths = {}
            for n in order:
                row = lk[(lk["sample"] == s) & (lk.variant == v_) & (lk.setting_id == REGIMES[n]["paper"])].iloc[0]
                paths[n] = (QS / "paper" / row.set_path / f"{s}_consensus_0.50_coronal.tif"
                            if isinstance(row.set_path, str) else None)
            vol = tifffile.imread(paths["Current"]) > 0
            half = vol.shape[2] // 2
            z = int(np.argmax(vol[:, :, :half].sum((1, 2))))
            cons = {}
            for n, p in paths.items():
                if p is None:
                    cons[n] = None
                    continue
                with tifffile.TiffFile(p) as t:
                    cons[n] = t.pages[z].asarray()[:, :half] > 0
            run = next(p for p in sorted((PAPER_REG / s / v_).iterdir()) if (p / "downsampled.tiff").exists())
            with tifffile.TiffFile(run / "downsampled.tiff") as t:
                img = t.pages[z].asarray()[:, :half].astype(float)
            u = np.logical_or.reduce([m for m in cons.values() if m is not None])
            rr, cc = np.flatnonzero(u.any(1)), np.flatnonzero(u.any(0))
            sl = (slice(max(rr[0] - 12, 0), rr[-1] + 13), slice(max(cc[0] - 12, 0), cc[-1] + 13))
            im = img[sl]
            lo, hi = np.percentile(im, [1, 99.5])
            ax.imshow(np.clip((im - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray", interpolation="nearest")
            for n in order:
                if cons[n] is not None:
                    contour(ax, cons[n][sl], cols[n], lws[n])
            clean(ax)
            scalebar(ax, im.shape[0], px=25)                                    # 25 px x 20 um = 500 um
            cnt = per[(per["sample"] == s) & (per.variant == v_)].set_index("setting_id")["count"]
            vals = [cnt.get(REGIMES[n]["paper"], np.nan) for n in order]
            ax.set_title((f"{v_}\n" if j == 0 else "\n") + sname("paper LSFM", s), fontsize=6.5, pad=2)
            tag(ax, f"{np.nanmin(vals):,.0f}–{np.nanmax(vals):,.0f} cells", 5.2)
            for n in order:
                src.append(dict(sample_id=s, sample=sname("paper LSFM", s), variant=v_, slice=z, regime=n,
                                setting_id=REGIMES[n]["paper"], count=cnt.get(REGIMES[n]["paper"], np.nan)))
    regime_legend(fig, "paper", 0.16)
    fig.subplots_adjust(wspace=0.05, hspace=0.3, bottom=0.22)
    save(fig, "FigS3b_QC_regimes_boundaries_LSFM")
    pd.DataFrame(src).to_csv(SRC / "FigS3b_counts.csv", index=False)


def copy_reused():
    g = VA / "focused" / "figures"
    (FIG / "FigS4_aMAP_per_brain_galleries").mkdir(exist_ok=True)
    for f in g.glob("FigS_focused_gallery_*.*"):
        shutil.copy2(f, FIG / "FigS4_aMAP_per_brain_galleries" / f.name)


if __name__ == "__main__":
    which = sys.argv[1:] or ["fig1", "fig2", "fig3", "figS1", "fig4", "figS2", "figS3", "copy"]
    if "fig1" in which:
        log(fig1())
    if "fig1v2" in which:                          # MV row drawn at section 487 instead of the STAPLE section
        log(fig1(sections={("MV_Ntsr1_165", "MV"): 487}, name="Fig1_expert_anatomical_validation_v2",
                 src_name="Fig1_AE_examples_v2.csv", sub_dorsal=True).round(3).to_string(index=False))
    if "fig2" in which:
        log(fig2().round(3).to_string(index=False))
    if "fig2split" in which:                       # raw / preprocessed rows; own IoU range and one shared IoU scale
        fig2(split=True, same_x=False, fig_name="Fig2_QC_thresholds_and_sensitivity_split_own_scale")
        fig2(split=True, same_x=True, fig_name="Fig2_QC_thresholds_and_sensitivity_split_same_scale")
    if "fig2v2" in which:                          # split same-scale A-C, D, plus original-LSFM E-G (S7A-C data)
        fig2(split=True, same_x=True, fig_name="Fig2_QC_thresholds_and_sensitivity_v2", lsfm=True)
    if "fig3" in which:
        log(fig3().round(3).to_string(index=False))
    if "figS1" in which:
        figS1()
    if "fig4" in which:
        log(fig4())
    if "figS2" in which:
        figS2()
    if "figS3" in which:
        figS3a()
        figS3b()
    if "copy" in which:
        copy_reused()
