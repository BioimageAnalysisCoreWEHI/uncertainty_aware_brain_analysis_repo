"""Focused aMAP validation: the pipeline consensus vs the aMAP authors' final STAPLE, with individual raters scored
against the same STAPLE. One main figure (<= 5 panels) and per-brain supplementary galleries.

Reference = the final STAPLE of all raters (the authors' consensus), for the pipeline and the raters alike.
Pipeline = raw input, global QC (IoU > 0.9, grad_p95 < 0.045) + SSIM top 50 %, majority vote (p >= 0.5), annotated
hemisphere from the registered hemisphere maps. Scored with the aMAP protocol (best coronal slice within the raters'
40-plane stack) and at the STAPLE plane itself. All data come from the verified cache of the main validation package
(../../cache); nothing upstream is modified.

Outputs: focused/tables, focused/figures (PDF + 600 dpi PNG), focused/source_data.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]                       # .../validation_analysis/focused
sys.path.insert(0, str(HERE.parent / "scripts"))
from common import (AMAP_OUT, BRAINS, CFG, QC_BRAINS, SHORT, STRUCTS, STRUCTS_MAIN, load_table, log)  # noqa: E402
from amap_validation import nifti  # noqa: E402
from amap_validation.reslice import AsrLayout  # noqa: E402

TAB, FIG, SRC = HERE / "tables", HERE / "figures", HERE / "source_data"
for d in (TAB, FIG, SRC):
    d.mkdir(parents=True, exist_ok=True)
DPI = 600
EXAMPLES = [("MV_Ntsr1_165", "RSP"), ("MV_Ntsr1_165", "MV")]
A = CFG["amap"]
OI = dict(blue="#0072B2", sky="#56B4E9", green="#009E73", orange="#E69F00", vermillion="#D55E00",
          purple="#CC79A7", yellow="#F0E442", grey="#8C8C8C", lightgrey="#D9D9D9")
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

spec = importlib.util.spec_from_file_location("m3", str(HERE.parent / "scripts" / "03_amap_metrics.py"))
m3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m3)
BOX = dict(facecolor="black", alpha=0.55, pad=1.0, lw=0)
LAY = AsrLayout(AMAP_OUT / "userStapleResults" / "asr25um")


def dice_iou(a, b):
    a, b = a > 0, b > 0
    inter, sa, sb = (a & b).sum(), a.sum(), b.sum()
    union = sa + sb - inter
    return (2 * inter / (sa + sb) if sa + sb else np.nan), (inter / union if union else np.nan)


def load(brain, structure, qc):
    cs = m3.Case(brain, structure, "raw")
    q = qc[(qc.brain == brain) & (qc.variant == "raw")]
    final = m3.qc_subset(q, structure, A["iou_threshold"], A["grad_p95_threshold"], A["ssim_top_fraction"])
    return cs, final


def case_scores(cs, final):
    used = final if final else cs.runs
    m = cs.consensus(used, 0.5)
    d = cs.slice_dice(m)
    k = cs.slab_idx[int(np.nanargmax(d[cs.slab_idx]))]
    dp, ip = dice_iou(m[k], cs.truth)
    df, if_ = dice_iou(m[cs.i0], cs.truth)
    return dict(n_registrations=len(cs.runs), n_retained=len(final), consensus_from=("final QC set" if final else
                "all registrations (QC-rejected brain)"), dice_amap_protocol=dp, iou_amap_protocol=ip,
                dice_staple_plane=df, iou_staple_plane=if_, A_best=cs.A_list[k], A_staple=cs.A0)


# ------------------------------------------------------------------------------------------------------------
# drawing
# ------------------------------------------------------------------------------------------------------------
def view(cs, final, margin=10):
    """Display crop (in cache-crop coordinates) around everything drawn at the STAPLE plane, plus the brain image."""
    i0 = cs.i0
    runs_m = np.stack([cs.run_mask(r)[i0] for r in cs.runs])
    union = runs_m.any(0) | cs.truth | cs.raters.any(0)
    rr, cc = np.flatnonzero(union.any(1)), np.flatnonzero(union.any(0))
    # square crop centred on everything drawn (in full-slice coordinates, so it may extend beyond the cache crop)
    s0, _, r0, _ = [int(v) for v in cs.bbox]
    brain = np.asarray(nifti.load(LAY.brain_img(cs.brain), mmap=True)[cs.A0], float)
    half = (max(rr[-1] - rr[0], cc[-1] - cc[0]) + 1) // 2 + margin
    cy, cx = s0 + (rr[0] + rr[-1]) // 2, r0 + (cc[0] + cc[-1]) // 2
    Y0 = int(np.clip(cy - half, 0, brain.shape[0] - 2 * half)); X0 = int(np.clip(cx - half, 0, brain.shape[1] - 2 * half))
    img = brain[Y0: Y0 + 2 * half, X0: X0 + 2 * half]
    y0, x0 = Y0 - s0, X0 - r0                      # crop origin in cache coordinates (may be negative)
    lo, hi = np.percentile(img, [1, 99.5])
    used = final if final else cs.runs
    cnt, n, keep = cs.probability(used)
    p = np.where(keep[i0], cnt[i0] / n, 0.0)
    H = img.shape[0]

    def put(a):  # cache-crop array(s) (..., h, w) -> display frame (..., H, H), zero outside the cache crop
        a = np.asarray(a)
        out = np.zeros(a.shape[:-2] + (H, img.shape[1]), a.dtype)
        ys, xs = max(0, -y0), max(0, -x0)
        ye, xe = min(H, a.shape[-2] - y0), min(img.shape[1], a.shape[-1] - x0)
        out[..., ys:ye, xs:xe] = a[..., ys + y0: ye + y0, xs + x0: xe + x0]
        return out
    return dict(img=np.clip((img - lo) / (hi - lo + 1e-9), 0, 1), runs=put(runs_m),
                idx={r: i for i, r in enumerate(cs.runs)}, truth=put(cs.truth), raters=put(cs.raters), p=put(p))


def outline(ax, mask, color, lw, alpha=1.0, ls="-"):
    if mask.any():
        ax.contour(mask.astype(float), [0.5], colors=[color], linewidths=lw, alpha=alpha, linestyles=ls)


def base(ax, v, scalebar=True):
    ax.imshow(v["img"], cmap="gray", interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if scalebar:
        h = v["img"].shape[0]
        ax.plot([2, 22], [h - 3, h - 3], color="white", lw=2.0, solid_capstyle="butt")          # 20 px x 25 um = 500 um
        ax.text(12, h - 5.5, "500 µm", color="white", ha="center", va="bottom", fontsize=5, bbox=BOX)


def draw_row(axs, cs, final, v, scores, first_row=False):
    n_r = len(v["raters"])
    # (a) raters + STAPLE
    base(axs[0], v)
    for m in v["raters"]:
        outline(axs[0], m, OI["orange"], 0.5, 0.45)
    outline(axs[0], v["truth"], "white", 1.3)
    axs[0].text(0.02, 0.98, f"{n_r} annotations", transform=axs[0].transAxes, color="white", va="top", fontsize=5.5, bbox=BOX)
    # (b) all registrations
    base(axs[1], v, False)
    for m in v["runs"]:
        outline(axs[1], m, OI["sky"], 0.4, 0.35)
    axs[1].text(0.02, 0.98, f"n = {len(v['runs'])}", transform=axs[1].transAxes, color="white", va="top", fontsize=5.5, bbox=BOX)
    # (c) retained after QC
    base(axs[2], v, False)
    if final:
        for r in final:
            outline(axs[2], v["runs"][v["idx"][r]], OI["sky"], 0.5, 0.7)
        txt = f"n = {len(final)} retained"
    else:
        txt = "no registration passed QC"
    outline(axs[2], v["truth"], "white", 0.9, 0.9, "--")
    axs[2].text(0.02, 0.98, txt, transform=axs[2].transAxes, color="white", va="top", fontsize=5.5, bbox=BOX)
    # (d) consensus vs STAPLE
    base(axs[3], v, False)
    ov = np.zeros(v["p"].shape + (4,))
    ov[..., :3] = matplotlib.colors.to_rgb(OI["vermillion"])
    ov[..., 3] = 0.55 * v["p"]
    axs[3].imshow(ov, interpolation="nearest")
    outline(axs[3], v["p"] >= 0.5, OI["vermillion"], 1.2)
    outline(axs[3], v["truth"], "white", 1.2)
    axs[3].text(0.02, 0.98, f"Dice {scores['dice_staple_plane']:.2f} / IoU {scores['iou_staple_plane']:.2f}\n"
                f"(aMAP protocol: {scores['dice_amap_protocol']:.2f} / {scores['iou_amap_protocol']:.2f})",
                transform=axs[3].transAxes, color="white", va="top", fontsize=5, bbox=BOX)
    # (e) consensus over the individual annotations
    if len(axs) > 4:
        base(axs[4], v, False)
        for m in v["raters"]:
            outline(axs[4], m, OI["orange"], 0.5, 0.45)
        outline(axs[4], v["p"] >= 0.5, OI["vermillion"], 1.3)
        axs[4].text(0.02, 0.98, f"consensus vs {n_r} annotations", transform=axs[4].transAxes, color="white",
                    va="top", fontsize=5.5, bbox=BOX)


def label(ax, s, x=-0.06, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log(f"  {name}")


# ------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    qc = m3.load_qc()
    case_rows, rater_rows, views = [], [], {}
    for brain in BRAINS:
        for s in STRUCTS:
            cs, final = load(brain, s, qc)
            sc = case_scores(cs, final)
            case_rows.append(dict(brain=brain, structure=s, **sc))
            for u, i, att, m in zip(cs.rater_user, cs.rater_idx, cs.rater_attempt, cs.raters):
                d, j = dice_iou(m, cs.truth)
                rater_rows.append(dict(brain=brain, structure=s, user=u, idx=int(i), attempt=int(att), dice=d, iou=j))
            views[(brain, s)] = (cs, final, view(cs, final), sc)
        log(f"{brain} done")
    cases = pd.DataFrame(case_rows)
    raters = pd.DataFrame(rater_rows)
    runs = load_table("amap_runs")
    r = runs[runs.variant == "raw"].groupby(["brain", "structure"]).agg(
        run_dice_min_all=("dice_paper", "min"), run_dice_max_all=("dice_paper", "max"))
    rq = runs[(runs.variant == "raw") & runs.passes_final_qc].groupby(["brain", "structure"]).agg(
        run_dice_min_retained=("dice_paper", "min"), run_dice_max_retained=("dice_paper", "max"))
    rs = raters.groupby(["brain", "structure"]).dice.agg(rater_dice_min="min", rater_dice_median="median",
                                                        rater_dice_max="max")
    cases = cases.merge(r, on=["brain", "structure"], how="left").merge(rq, on=["brain", "structure"], how="left") \
        .merge(rs, on=["brain", "structure"], how="left")
    cases["within_rater_range"] = cases.dice_amap_protocol >= cases.rater_dice_min
    cases["percentile_among_raters"] = [
        float((raters[(raters.brain == b) & (raters.structure == s)].dice < d).mean())
        for b, s, d in zip(cases.brain, cases.structure, cases.dice_amap_protocol)]
    cases.to_csv(TAB / "focused_case_scores.csv", index=False)
    raters.to_csv(TAB / "focused_rater_scores.csv", index=False)

    # verification against the verified main-package tables
    ag = load_table("amap_agreement_cases")
    ag = ag[ag.variant == "raw"][["brain", "structure", "dice_paper", "dice_fixed"]]
    v = cases.merge(ag, on=["brain", "structure"])
    ver = dict(n=len(v), max_diff_amap_protocol=float((v.dice_amap_protocol - v.dice_paper).abs().max()),
               max_diff_staple_plane=float((v.dice_staple_plane - v.dice_fixed).abs().max()))
    pd.DataFrame([ver]).to_csv(TAB / "focused_verification.csv", index=False)
    log(f"verification: {ver}")

    # Table 1: per structure (5 QC brains; DG-sg shown but excluded from ALL)
    rows = []
    q = cases[cases.brain.isin(QC_BRAINS)]
    rq_ = raters[raters.brain.isin(QC_BRAINS)]
    for st in STRUCTS + ["ALL (excl. DG-sg)"]:
        c = q[q.structure.isin(STRUCTS_MAIN)] if st.startswith("ALL") else q[q.structure == st]
        h = rq_[rq_.structure.isin(STRUCTS_MAIN)] if st.startswith("ALL") else rq_[rq_.structure == st]
        rows.append(dict(structure=st, n_cases=len(c),
                         consensus_dice_amap=c.dice_amap_protocol.median(), consensus_iou_amap=c.iou_amap_protocol.median(),
                         consensus_dice_staple_plane=c.dice_staple_plane.median(),
                         consensus_iou_staple_plane=c.iou_staple_plane.median(),
                         n_annotations=len(h), rater_dice_median=h.dice.median(),
                         rater_dice_q25=h.dice.quantile(.25), rater_dice_q75=h.dice.quantile(.75),
                         rater_iou_median=h.iou.median(),
                         cases_within_rater_range=int(c.within_rater_range.sum()),
                         median_percentile_among_raters=c.percentile_among_raters.median()))
    summ = pd.DataFrame(rows)
    summ.to_csv(TAB / "focused_structure_summary.csv", index=False)
    pd.set_option("display.width", 220)
    log(summ.round(3).to_string())
    log(cases[cases.brain.isin([b for b, _ in EXAMPLES])].round(3).to_string())

    # ---------------- main figure ----------------
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.25], hspace=0.14, wspace=0.05)
    titles = ["Expert annotations\n+ STAPLE", "All registrations\n(before QC)", "Registrations\nretained after QC",
              "Pipeline consensus\nvs STAPLE", "Pipeline consensus\nvs annotations"]
    src_rows = []
    for row, (b, s) in enumerate(EXAMPLES):
        cs, final, v_, sc = views[(b, s)]
        axs = [fig.add_subplot(gs[row, c]) for c in range(5)]
        draw_row(axs, cs, final, v_, sc)
        axs[0].text(-0.08, 0.5, f"{SHORT[b]}  {s}", transform=axs[0].transAxes, rotation=90, va="center", ha="right",
                    fontsize=7)
        if row == 0:
            for ax, t, l in zip(axs, titles, "abcde"):
                ax.set_title(t, fontsize=6.5)
                label(ax, l, x=-0.04, y=1.2)
        src_rows.append(dict(brain=b, structure=s, n_annotations=len(v_["raters"]), **sc))
    pd.DataFrame(src_rows).to_csv(SRC / "Fig_focused_a-e_examples.csv", index=False)
    # (e) scores
    ax = fig.add_subplot(gs[2, :])
    xs = np.arange(len(STRUCTS_MAIN))
    rng = np.random.default_rng(0)
    for j, st in enumerate(STRUCTS_MAIN):
        h = rq_[rq_.structure == st].dice
        bp = ax.boxplot(h, positions=[j - 0.2], widths=0.22, showfliers=False, patch_artist=True,
                        medianprops=dict(color="k", lw=0.8), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        bp["boxes"][0].set(facecolor=OI["lightgrey"], edgecolor=OI["grey"], lw=0.6)
        c = q[q.structure == st]
        jit = rng.uniform(-0.05, 0.05, len(c))
        ax.scatter(j + 0.12 + jit, c.dice_amap_protocol, s=12, color=OI["vermillion"], edgecolor="white", lw=0.3, zorder=3)
        ax.scatter(j + 0.27 + jit, c.dice_staple_plane, s=10, facecolor="none", edgecolor=OI["vermillion"], lw=0.7, zorder=3)
        for b, s in EXAMPLES:
            if s == st:
                e = c[c.brain == b]
                ax.scatter(j + 0.12, e.dice_amap_protocol, s=45, facecolor="none", edgecolor="k", lw=0.9, zorder=4)
    ax.set_xticks(xs, STRUCTS_MAIN)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Dice vs final STAPLE")
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor=OI["lightgrey"], edgecolor=OI["grey"],
                               label="individual annotations"))
    ax.scatter([], [], s=12, color=OI["vermillion"], label="pipeline consensus (aMAP protocol)")
    ax.scatter([], [], s=10, facecolor="none", edgecolor=OI["vermillion"], label="pipeline consensus (STAPLE plane)")
    ax.scatter([], [], s=45, facecolor="none", edgecolor="k", label="examples in a–e")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=6)
    label(ax, "f", x=-0.045, y=1.0)
    pd.concat([rq_[rq_.structure.isin(STRUCTS_MAIN)].assign(group="individual annotation"),
               q[q.structure.isin(STRUCTS_MAIN)].rename(columns={"dice_amap_protocol": "dice"})[
                   ["brain", "structure", "dice", "dice_staple_plane"]].assign(group="pipeline consensus")]) \
        .to_csv(SRC / "Fig_focused_f_scores.csv", index=False)
    # shared legend for the image panels
    handles = [plt.Line2D([], [], color=OI["orange"], lw=0.8, alpha=0.7, label="individual annotation"),
               plt.Line2D([], [], color="white", lw=1.3, label="final STAPLE (aMAP consensus)"),
               plt.Line2D([], [], color=OI["sky"], lw=0.8, label="single registration"),
               plt.Line2D([], [], color=OI["vermillion"], lw=1.3, label="pipeline consensus (p ≥ 0.5)")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True, fontsize=6, bbox_to_anchor=(0.5, 0.925))
    leg.get_frame().set_facecolor("#555555")
    for t in leg.get_texts():
        t.set_color("white")
    save(fig, "Fig_focused_validation")

    # ---------------- supplementary galleries (one per brain) ----------------
    for b in BRAINS:
        fig, axs = plt.subplots(len(STRUCTS), 5, figsize=(7.2, 1.42 * len(STRUCTS)),
                                gridspec_kw=dict(hspace=0.12, wspace=0.05))
        for i, s in enumerate(STRUCTS):
            cs, final, v_, sc = views[(b, s)]
            draw_row(axs[i], cs, final, v_, sc)
            axs[i, 0].text(-0.06, 0.5, s + (" *" if s == "DG-sg" else ""), transform=axs[i, 0].transAxes, rotation=90,
                           va="center", ha="right", fontsize=7)
        for ax, t in zip(axs[0], titles):
            ax.set_title(t, fontsize=5.8)
        note = "  (rejected by global QC: consensus from all registrations)" if not views[(b, STRUCTS[0])][1] else ""
        fig.subplots_adjust(top=0.955, bottom=0.035)
        fig.suptitle(f"{b}{note}", fontsize=8, y=0.985)
        lg = fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True, fontsize=6, bbox_to_anchor=(0.5, 0.0))
        lg.get_frame().set_facecolor("#555555")
        for tx in lg.get_texts():
            tx.set_color("white")
        save(fig, f"FigS_focused_gallery_{SHORT[b]}")
    log("done")
