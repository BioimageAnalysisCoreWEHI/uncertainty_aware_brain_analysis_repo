"""Publication figures (PDF + 300 dpi PNG) and one source-data CSV per panel.

Palette: Okabe-Ito (colour-blind safe); colour follows the entity (method), never its rank. Every panel's data are
written to source_data/<figure>_<panel>.csv before plotting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from common import (CFG, FIG_MAIN, FIG_SUPP, QC_BRAINS, REJECTED_BRAINS, SHORT, SRC, STRUCTS_MAIN, C, bwperim8, load_case,  # noqa: E402
                    load_table, log)

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 150})
OI = dict(blue="#0072B2", sky="#56B4E9", green="#009E73", orange="#E69F00", vermillion="#D55E00",
          purple="#CC79A7", yellow="#F0E442", black="#000000", grey="#8C8C8C", lightgrey="#D9D9D9")
COL = {"consensus": OI["blue"], "raw": OI["blue"], "preprocessed": OI["sky"], "human": OI["grey"],
       "default": OI["orange"], "top_ssim": OI["purple"], "lobo_top1": OI["green"], "lobo_top4_ensemble": "#66C2A5",
       "random_qc_run (expected)": OI["grey"], "qc_run_p10 (tail)": OI["black"], "oracle": OI["black"],
       "insample_best_param (optimistic)": OI["vermillion"], "insample_stiff_family (optimistic)": "#F4A582"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"])
DIV = LinearSegmentedColormap.from_list("div", [OI["vermillion"], "#F7F7F7", OI["blue"]])
TIE = float(CFG["tie_band"])
K = CFG["amap"]["lobo_ensemble_k"]


def src(df: pd.DataFrame, fig: str, panel: str):
    df.to_csv(SRC / f"{fig}_{panel}.csv", index=False)


def save(fig, name, main=True):
    d = FIG_MAIN if main else FIG_SUPP
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"  {name}")


def label(ax, s, x=-0.18, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def jitter(n, w=0.12, seed=0):
    return np.random.default_rng(seed).uniform(-w, w, n)


# =================================================================================================================
def fig1_agreement():
    ag = load_table("amap_agreement_cases")
    hum = load_table("human_loo")
    ag = ag[ag.structure.isin(STRUCTS_MAIN)]
    hum = hum[hum.structure.isin(STRUCTS_MAIN) & hum.brain.isin(QC_BRAINS)]
    fig = plt.figure(figsize=(7.1, 5.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[0.85, 1.0], hspace=0.5, wspace=0.9)
    # (a) examples: STAPLE vs consensus outline on the probability map, three structures
    ex = [("AB_Gad67_223", "RSP"), ("MV131017_7", "ACA"), ("CR_Syt6CreRfp_35", "VPM")]
    rows = []
    for i, (b, s) in enumerate(ex):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        cs = load_case(b, s, "raw")
        a = ag[(ag.brain == b) & (ag.structure == s) & (ag.variant == "raw")].iloc[0]
        k = cs["A_list"].index(int(a.A_best))
        runs_fin = load_table("amap_runs")
        rr = runs_fin[(runs_fin.brain == b) & (runs_fin.structure == s) & (runs_fin.variant == "raw") &
                      runs_fin.passes_final_qc].run.tolist() or cs["runs"]
        ids = [cs["runs"].index(r) for r in rr]
        hl = cs["h1"] if cs["label"] == 1 else cs["h2"]
        ho = cs["h2"] if cs["label"] == 1 else cs["h1"]
        keep = hl[ids, k].sum(0) > ho[ids, k].sum(0)
        p = np.where(keep, cs["masks"][ids, k].mean(0), 0)
        t = cs["truth"]
        yy, xx = np.nonzero((p > 0) | t)
        m = 6
        cy = slice(max(0, yy.min() - m), yy.max() + m + 1)
        cx = slice(max(0, xx.min() - m), xx.max() + m + 1)
        p, t = p[cy, cx], t[cy, cx]
        ax.imshow(p, cmap=SEQ, vmin=0, vmax=1, interpolation="nearest")
        ax.plot([1, 1 + 20], [p.shape[0] - 2] * 2, color="k", lw=1.2)  # 20 px = 500 um
        ax.text(11, p.shape[0] - 3, "500 µm", ha="center", va="bottom", fontsize=5.5)
        yy, xx = np.nonzero(bwperim8(t))
        ax.scatter(xx, yy, s=0.6, c=OI["orange"], marker="s", linewidths=0, label="Expert STAPLE")
        yy, xx = np.nonzero(bwperim8(p >= 0.5))
        ax.scatter(xx, yy, s=0.6, c=OI["black"], marker="s", linewidths=0, label="Pipeline consensus (p ≥ 0.5)")
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{SHORT[b]} {s}: Dice {a.dice_paper:.2f}\n(median human LOO {a.human_loo_median:.2f})")
        rows.append(dict(brain=b, structure=s, A=int(a.A_best), dice=a.dice_paper, human=a.human_loo_median))
        if i == 0:
            label(ax, "a", x=-0.12, y=1.18)
            ax.legend(loc="upper left", bbox_to_anchor=(0, -0.02), ncol=1, frameon=False, markerscale=6)
    src(pd.DataFrame(rows), "Fig1", "a_examples")
    # (b) Dice per structure
    ax = fig.add_subplot(gs[1, :4])
    xs = np.arange(len(STRUCTS_MAIN))
    d_rows = []
    for j, st in enumerate(STRUCTS_MAIN):
        h = hum[hum.structure == st].dice_loo
        bp = ax.boxplot(h, positions=[j - 0.25], widths=0.2, showfliers=False, patch_artist=True,
                        medianprops=dict(color="k", lw=0.8), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
        bp["boxes"][0].set(facecolor=OI["lightgrey"], edgecolor=OI["grey"], lw=0.6)
        for off, v in ((0.02, "raw"), (0.25, "preprocessed")):
            y = ag[(ag.structure == st) & (ag.variant == v) & ag.brain.isin(QC_BRAINS)].dice_paper
            ax.scatter(j + off + jitter(len(y), 0.05, j), y, s=9, color=COL[v], edgecolor="white", lw=0.3, zorder=3)
            er = ag[(ag.structure == st) & (ag.variant == v) & ag.brain.isin(REJECTED_BRAINS)].dice_paper
            ax.scatter(j + off + 0.0 * er, er, s=11, marker="x", color=COL[v], lw=0.7, zorder=3)
        d_rows += [dict(structure=st, group="human_LOO", value=x) for x in h]
    d_rows += [dict(structure=r.structure, group=f"pipeline_{r.variant}", brain=r.brain, value=r.dice_paper)
               for r in ag.itertuples()]
    src(pd.DataFrame(d_rows), "Fig1", "b_dice_by_structure")
    ax.set_xticks(xs, STRUCTS_MAIN)
    ax.set_ylabel("Dice vs expert consensus")
    ax.set_ylim(0, 1)
    ax.scatter([], [], s=9, color=COL["raw"], label="Pipeline, raw (QC brains)")
    ax.scatter([], [], s=9, color=COL["preprocessed"], label="Pipeline, preprocessed")
    if REJECTED_BRAINS:
        ax.scatter([], [], s=11, marker="x", color="k", lw=0.7, label="QC-rejected brain (all runs)")
    ax.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor=OI["lightgrey"], edgecolor=OI["grey"], label="Humans (LOO)"))
    ax.legend(loc="lower left", ncol=2, frameon=False)
    label(ax, "b", x=-0.08)
    # (c) pipeline percentile within the human distribution
    ax = fig.add_subplot(gs[1, 4:])
    a = ag[ag.brain.isin(QC_BRAINS) & (ag.variant == "raw")]
    order = a.groupby("structure").pipeline_percentile_in_humans.median().reindex(STRUCTS_MAIN)
    for j, st in enumerate(STRUCTS_MAIN):
        y = a[a.structure == st].pipeline_percentile_in_humans * 100
        ax.scatter(y, j + jitter(len(y), 0.12, j), s=8, color=COL["raw"], edgecolor="white", lw=0.3)
        ax.plot([order[st] * 100] * 2, [j - 0.3, j + 0.3], color="k", lw=0.9)
    ax.axvline(50, color=OI["grey"], lw=0.6, ls="--")
    ax.set_yticks(range(len(STRUCTS_MAIN)), STRUCTS_MAIN)
    ax.invert_yaxis()
    ax.set_xlabel("Pipeline percentile among\nhuman raters (LOO), raw")
    ax.set_xlim(-3, 103)
    src(a[["brain", "structure", "pipeline_percentile_in_humans"]], "Fig1", "c_percentile")
    label(ax, "c", x=-0.3)
    save(fig, "Fig1_agreement_with_experts")


# =================================================================================================================
def fig2_benefit():
    w = load_table("amap_benefit_cases")
    w = w[w.brain.isin(QC_BRAINS) & w.structure.isin(STRUCTS_MAIN)]
    s = load_table("amap_benefit_summary")
    comps = ["default", "top_ssim", "lobo_top1", f"lobo_top{K}_ensemble", "random_qc_run (expected)",
             "qc_run_p10 (tail)", "insample_best_param (optimistic)", "oracle_any_run (unattainable)"]
    names = ["brainreg default", "top-SSIM run", "LOBO parameter", f"LOBO top-{K} ensemble", "random QC run (mean)",
             "QC run, 10th pct", "in-sample best param*", "oracle run*"]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2.9), gridspec_kw=dict(width_ratios=[1.5, 1.0, 1.0], wspace=0.55))
    ax = axs[0]
    rows = []
    for j, (c, n) in enumerate(zip(comps, names)):
        for v, off in (("raw", -0.13), ("preprocessed", 0.13)):
            d = w[w.variant == v][f"delta__{c}"]
            ax.scatter(d, j + off + jitter(len(d), 0.06, j), s=5, color=COL[v], alpha=0.75, lw=0)
            m = s[(s.variant == v) & (s.comparator == c)].iloc[0]
            ax.errorbar(m.median_delta, j + off, xerr=[[m.median_delta - m.ci_lo], [m.ci_hi - m.median_delta]],
                        fmt="o", ms=3, color="k", mfc=COL[v], mec="k", mew=0.5, lw=0.8, capsize=0)
            rows += [dict(comparator=c, variant=v, brain=b, structure=st, delta=x)
                     for b, st, x in zip(w[w.variant == v].brain, w[w.variant == v].structure, d)]
    ax.axvspan(-TIE, TIE, color=OI["lightgrey"], lw=0, alpha=0.6)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlabel("Δ Dice = consensus − comparator\n(> 0: consensus better)")
    ax.set_xlim(-0.35, 0.6)
    ax.scatter([], [], s=6, color=COL["raw"], label="raw"), ax.scatter([], [], s=6, color=COL["preprocessed"], label="preprocessed")
    ax.legend(frameon=False, loc="lower right")
    src(pd.DataFrame(rows), "Fig2", "a_delta")
    label(ax, "a", x=-0.62)
    # (b) win / tie / loss (raw)
    ax = axs[1]
    ss = s[(s.variant == "raw") & s.comparator.isin(comps[:6])].set_index("comparator").reindex(comps[:6])
    y = np.arange(len(ss))
    ax.barh(y, ss.wins, color=OI["blue"], height=0.6, label="consensus better")
    ax.barh(y, ss.ties, left=ss.wins, color=OI["lightgrey"], height=0.6, label=f"tie (|Δ| ≤ {TIE})")
    ax.barh(y, ss.losses, left=ss.wins + ss.ties, color=OI["vermillion"], height=0.6, label="comparator better")
    ax.set_yticks(y, names[:6])
    ax.invert_yaxis()
    ax.set_xlabel("Cases (raw, n = 45)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.2), fontsize=6)
    src(ss.reset_index(), "Fig2", "b_win_tie_loss")
    label(ax, "b", x=-0.95)
    # (c) risk curves
    ax = axs[2]
    rc = load_table("amap_risk_curves")
    rc = rc[rc.variant == "raw"]
    styles = {"all runs": (OI["lightgrey"], "-"), "global-QC runs": (OI["grey"], "-"), "final-QC runs": ("#5A5A5A", "-"),
              "consensus": (OI["blue"], "-"), "default": (OI["orange"], "--"), "LOBO parameter": (OI["green"], ":")}
    for g, (c, ls) in styles.items():
        r = rc[rc.group == g]
        ax.plot(r.x, r.p_below, color=c, ls=ls, lw=1.2, label=g)
    ax.set_xlim(0.3, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Dice threshold x")
    ax.set_ylabel("P(Dice < x), raw")
    ax.legend(frameon=False, fontsize=5.5, loc="upper left", bbox_to_anchor=(-0.02, 1.02), handlelength=1.8)
    src(rc, "Fig2", "c_risk_curves")
    label(ax, "c", x=-0.3)
    save(fig, "Fig2_consensus_vs_single_registrations")


# =================================================================================================================
def heat(ax, piv, cmap, vmin, vmax, fmt="{:.2f}", cbar_label=None, fig=None):
    im = ax.imshow(piv.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    fmt_t = lambda c: f"{c:g}" if isinstance(c, (int, float, np.number)) else str(c)  # noqa: E731
    ax.set_xticks(range(piv.shape[1]), [fmt_t(c) for c in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [fmt_t(c) for c in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if not np.isnan(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=5.5 if piv.shape[1] <= 6 else 4.2,
                        color="white" if (v - vmin) / (vmax - vmin + 1e-12) > 0.6 else "black")
    if fig is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=5.5)
        if cbar_label:
            cb.set_label(cbar_label, fontsize=6)
    return im


def fig3_landscape():
    pooled = load_table("amap_landscape_pooled")
    land = load_table("amap_landscape_by_brain")
    wf = load_table("amap_winner_frequency")
    sets = load_table("amap_sets")
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6), gridspec_kw=dict(wspace=0.45, hspace=0.45))
    axs = axs.ravel()
    p = pooled[pooled.variant == "raw"].pivot_table(index="bending_energy", columns="grid", values="mean_dice")
    heat(axs[0], p, SEQ, 0.3, 0.85, fig=fig)
    axs[0].set_title("Mean Dice (raw, QC brains)")
    src(p.reset_index(), "Fig3", "a_mean_dice")
    q = pooled[pooled.variant == "raw"].pivot_table(index="bending_energy", columns="grid", values="frac_pass_global_qc")
    heat(axs[1], q, SEQ, 0, 1, fig=fig)
    axs[1].set_title("Fraction passing global QC")
    src(q.reset_index(), "Fig3", "b_frac_qc")
    f = wf[wf.variant == "raw"].pivot_table(index="bending_energy", columns="grid", values="n_cases_best").reindex(
        index=p.index, columns=p.columns).fillna(0)
    heat(axs[2], f, SEQ, 0, max(4, f.to_numpy().max()), fmt="{:.0f}", fig=fig)
    axs[2].set_title("Cases where run is best")
    src(f.reset_index(), "Fig3", "c_winners")
    for ax in axs[:3]:
        ax.set_xlabel("Grid spacing")
        ax.set_ylabel("Bending energy")
    # (d) LOBO: held-out brain, consensus vs LOBO parameter vs default vs oracle
    w = load_table("amap_benefit_cases")
    w = w[w.brain.isin(QC_BRAINS) & w.structure.isin(STRUCTS_MAIN) & (w.variant == "raw")]
    g = w.groupby("brain")[["cons_dice", "lobo_top1_dice", f"lobo_top{K}_ensemble_dice", "default_dice",
                            "oracle_dice"]].mean()
    ax = axs[3]
    xs = np.arange(len(g))
    for c, lab_, col, off in (("cons_dice", "consensus", OI["blue"], -0.24), ("lobo_top1_dice", "LOBO param", OI["green"], -0.08),
                              (f"lobo_top{K}_ensemble_dice", f"LOBO top-{K}", "#66C2A5", 0.08),
                              ("default_dice", "default", OI["orange"], 0.24)):
        ax.bar(xs + off, g[c], width=0.15, color=col, label=lab_)
    ax.scatter(xs, g.oracle_dice, marker="_", s=60, color="k", label="oracle*", zorder=4)
    ax.set_xticks(xs, [SHORT[b] for b in g.index])
    ax.set_ylim(0.5, 0.95)
    ax.set_ylabel("Mean Dice, held-out brain")
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="upper left")
    src(g.reset_index(), "Fig3", "d_lobo")
    for ax, l in zip(axs, "abcd"):
        label(ax, l, x=-0.35)
    save(fig, "Fig3_parameter_landscape_generalisation")


# =================================================================================================================
def fig4_qc():
    sc = load_table("amap_qc_scatter")
    qs = load_table("amap_qc_vs_dice_summary")
    gs = load_table("amap_qcgrid_summary")
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw=dict(wspace=0.95, width_ratios=[1.2, 0.8, 1.2]))
    ax = axs[0]
    s = sc[sc.variant == "raw"]
    oth = s[s.brain != "ER_Glt25d2Cre_9"]
    er = s[s.brain == "ER_Glt25d2Cre_9"]            # highlighted: rejected entirely at the earlier grad_p95 < 0.045
    im = ax.scatter(oth.iou, oth.grad_p95, c=oth.median_dice, cmap=SEQ, vmin=0.3, vmax=0.9, s=7, lw=0.2, edgecolor="grey")
    ax.scatter(er.iou, er.grad_p95, c=er.median_dice, cmap=SEQ, vmin=0.3, vmax=0.9, s=16, marker="D", lw=0.6,
               edgecolor=OI["vermillion"])
    ax.axvline(CFG["amap"]["iou_threshold"], color="k", ls="--", lw=0.6)
    ax.axhline(CFG["amap"]["grad_p95_threshold"], color="k", ls="--", lw=0.6)
    ax.set_xlabel("Tissue IoU")
    ax.set_ylabel("grad_p95")
    ax.scatter([], [], marker="D", s=14, facecolor="white", edgecolor=OI["vermillion"], label=f"ER ({'all rejected' if 'ER_Glt25d2Cre_9' in REJECTED_BRAINS else 'rejected at 0.045; ' + str(int(((er.iou > CFG['amap']['iou_threshold']) & (er.grad_p95 < CFG['amap']['grad_p95_threshold'])).sum())) + ' pass now'})")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("Median Dice over structures", fontsize=6)
    src(s, "Fig4", "a_qc_scatter")
    ax = axs[1]
    names = {"iou": "IoU", "grad_p95": "grad_p95", "ssim": "local SSIM", "bending_energy": "bending energy",
             "grid": "grid spacing"}
    q = qs[qs.metric.isin(names)]
    for v, off in (("raw", -0.12), ("preprocessed", 0.12)):
        qq = q[q.variant == v].set_index("metric").reindex(list(names))
        y = np.arange(len(qq)) + off
        ax.errorbar(qq.median_within_case, y, xerr=[qq.median_within_case - qq.ci_lo, qq.ci_hi - qq.median_within_case],
                    fmt="o", ms=3, color=COL[v], lw=0.8, label=v)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(range(len(names)), list(names.values()))
    ax.invert_yaxis()
    ax.set_xlabel("Within-case Spearman ρ with Dice\n(median, 95 % brain-bootstrap CI)")
    ax.legend(frameon=False, fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    src(q, "Fig4", "b_metric_correlations")
    ax = axs[2]
    g = gs[(gs.variant == "raw") & (gs.ssim_frac == 0.5) & (gs.grad_thr < 1)]
    piv = g.pivot_table(index="iou_thr", columns="grad_thr", values="median_dice_qc_brains")
    heat(ax, piv, SEQ, piv.min().min(), piv.max().max(), fmt="{:.3f}", fig=fig, cbar_label="Median consensus Dice")
    ax.tick_params(axis="x", labelrotation=45, labelsize=5.5)
    adm = g.pivot_table(index="iou_thr", columns="grad_thr", values="er_admitted").reindex_like(piv)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if adm.iat[i, j] > 0:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=OI["vermillion"], lw=1.0))
    ri = list(piv.index).index(CFG["amap"]["iou_threshold"])
    rj = list(piv.columns).index(CFG["amap"]["grad_p95_threshold"])
    ax.add_patch(plt.Rectangle((rj - 0.5, ri - 0.5), 1, 1, fill=False, ec="k", lw=1.4))
    ax.set_xlabel("grad_p95 threshold (<)")
    ax.set_ylabel("IoU threshold (>)")
    ax.set_title(f"black: production (grad_p95 < {CFG['amap']['grad_p95_threshold']:g}); red: ER admitted", fontsize=6)
    src(g, "Fig4", "c_qc_threshold_grid")
    for ax, l in zip(axs, "abc"):
        label(ax, l, x=-0.3)
    save(fig, "Fig4_qc_metrics_vs_accuracy")


# =================================================================================================================
def fig5_uncertainty():
    rel = load_table("amap_reliability_pooled")
    cal = load_table("amap_calib")
    tc = load_table("amap_threshold_curves")
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.1), gridspec_kw=dict(wspace=0.55))
    ax = axs[0]
    for v in ("raw", "preprocessed"):
        r = rel[(rel.variant == v) & (rel.p_bin_lo > 0)]  # p = 0 bin holds only missed expert pixels (always 1)
        ax.plot(r.mean_p, r.frac_expert, "o-", ms=2.5, lw=0.9, color=COL[v], label=v)
        ax.fill_between(r.mean_p, r.ci_lo, r.ci_hi, color=COL[v], alpha=0.2, lw=0)
    ax.plot([0, 1], [0, 1], color=OI["grey"], ls="--", lw=0.6)
    ax.set_xlabel("Pipeline probability (p > 0)")
    ax.set_ylabel("Fraction inside expert STAPLE")
    ax.set_xlim(0, 1.03), ax.set_ylim(0, 1.03)
    ax.legend(frameon=False, loc="upper left")
    src(rel, "Fig5", "a_reliability")
    ax = axs[1]
    c = cal[cal.brain.isin(QC_BRAINS) & cal.structure.isin(STRUCTS_MAIN)]
    for v in ("raw", "preprocessed"):
        cc = c[c.variant == v]
        ax.scatter(cc.human_band_width_um, cc.pipeline_band_width_um, s=7, color=COL[v], lw=0, label=v)
    m = max(c.human_band_width_um.max(), c.pipeline_band_width_um.max())
    ax.plot([0, m], [0, m], color=OI["grey"], ls="--", lw=0.6)
    ax.set_xlabel("Human boundary band (µm)")
    ax.set_ylabel("Pipeline boundary band (µm)")
    src(c, "Fig5", "b_band_width")
    ax = axs[2]
    t = tc[(tc.variant == "raw") & (tc.set_name == "passes_final_qc_top_0p5")]
    for st in STRUCTS_MAIN:
        r = t[t.structure == st]
        ax.plot(r.threshold, r.dice, lw=0.7, color=OI["lightgrey"])
    pool = t.groupby("threshold").dice.median()
    ax.plot(pool.index, pool.values, color=OI["blue"], lw=1.5, label="median of structures")
    ax.axvline(0.5, color="k", ls="--", lw=0.6)
    ax.set_xlabel("Consensus threshold")
    ax.set_ylabel("Median Dice (raw, final QC)")
    ax.legend(frameon=False, loc="lower center", fontsize=6)
    src(t, "Fig5", "c_threshold")
    ax = axs[3]
    pal = [OI["blue"], OI["orange"], OI["green"], OI["vermillion"], OI["purple"], OI["sky"], OI["black"], OI["yellow"],
           OI["grey"]]
    for st, c in zip(STRUCTS_MAIN, pal):
        r = t[t.structure == st]
        ax.plot(r.threshold, r.area_ratio, lw=0.8, label=st, color=c)
    ax.axhline(1, color="k", lw=0.6)
    ax.axvline(0.5, color="k", ls="--", lw=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("Consensus threshold")
    ax.set_ylabel("Area pipeline / expert")
    ax.legend(frameon=False, fontsize=4.8, ncol=2, loc="lower left")
    src(t[["structure", "threshold", "area_ratio"]], "Fig5", "d_volume_bias")
    for ax, l in zip(axs, "abcd"):
        label(ax, l, x=-0.35)
    save(fig, "Fig5_uncertainty_and_threshold")


# =================================================================================================================
def fig6_predictors():
    pr = load_table("amap_benefit_predictors")
    ui = load_table("amap_uncertainty_information")
    w = load_table("amap_benefit_cases")
    w = w[w.brain.isin(QC_BRAINS) & w.structure.isin(STRUCTS_MAIN) & (w.variant == "raw")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=1.1, width_ratios=[1.4, 1, 1]))
    ax = axs[0]
    p = pr[pr.variant == "raw"]
    comps = ["default", "top_ssim", "lobo_top1", "insample_best_param (optimistic)", "random_qc_run (expected)",
             "qc_run_p10 (tail)"]
    piv = p.pivot_table(index="predictor", columns="comparator", values="spearman").reindex(columns=comps)
    ph = p.pivot_table(index="predictor", columns="comparator", values="p_holm").reindex(columns=comps)
    piv = piv.reindex(sorted(piv.index, key=lambda s: (s[:3], s)))
    ph = ph.reindex(piv.index)
    im = ax.imshow(piv.to_numpy(), cmap=DIV, vmin=-0.8, vmax=0.8, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            ax.text(j, i, f"{v:.2f}" + ("*" if ph.iat[i, j] < 0.05 else ""), ha="center", va="center", fontsize=4.8)
    ax.set_xticks(range(len(comps)), ["default", "top-SSIM", "LOBO", "in-sample\nbest*", "random QC", "QC p10"],
                  rotation=45, ha="right")
    ax.set_yticks(range(piv.shape[0]), [s.replace("_", " ") for s in piv.index], fontsize=5.3)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Spearman ρ (predictor, Δ Dice)\n* Holm p < 0.05", fontsize=5.5)
    src(p, "Fig6", "a_predictor_heatmap")
    label(ax, "a", x=-0.9)
    ax = axs[1]
    ax.scatter(w.sd_dice_qc, w["delta__random_qc_run (expected)"], s=8, color=COL["raw"], lw=0)
    ax.scatter(w.sd_dice_qc, w["delta__default"], s=8, color=COL["default"], lw=0, marker="^")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("SD of Dice among QC runs (P4)")
    ax.set_ylabel("Δ Dice (consensus − comparator)")
    ax.scatter([], [], s=8, color=COL["raw"], label="vs random QC run")
    ax.scatter([], [], s=8, color=COL["default"], marker="^", label="vs default")
    ax.legend(frameon=False, fontsize=6)
    src(w[["brain", "structure", "sd_dice_qc", "delta__random_qc_run (expected)", "delta__default"]], "Fig6", "b_p4")
    label(ax, "b", x=-0.35)
    ax = axs[2]
    ax.scatter(w.gf_pairwise_dice_final, w.cons_dice, s=8, color=COL["raw"], lw=0)
    u = ui[(ui.variant == "raw") & (ui.signal == "P2_pairwise_dice_final") & (ui.target == "consensus Dice")].iloc[0]
    ax.set_xlabel("Agreement among retained runs\n(mean pairwise Dice; GT-free)")
    ax.set_ylabel("Consensus Dice vs experts")
    ax.set_title(f"ρ = {u.spearman:.2f} [{u.ci_lo:.2f}, {u.ci_hi:.2f}]", fontsize=6.5)
    src(w[["brain", "structure", "gf_pairwise_dice_final", "cons_dice"]], "Fig6", "c_gtfree_vs_error")
    label(ax, "c", x=-0.35)
    save(fig, "Fig6_when_does_consensus_help")


# =================================================================================================================
def fig7_lsfm():
    pr = load_table("lsfm_per_run")
    cons = load_table("lsfm_consensus")
    comp = load_table("lsfm_choices_vs_consensus")
    kn = load_table("lsfm_knob_sensitivity")
    rp = load_table("lsfm_raw_vs_preprocessed")
    var = load_table("lsfm_variability_by_stage")
    for d in (pr, cons, comp, kn, rp, var):
        d["sample"] = d["sample"].astype(str)
    samples = sorted(pr["sample"].unique())
    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 3, hspace=0.65, wspace=1.0)
    ax = fig.add_subplot(gs[0, :2])
    rows = []
    for i, s in enumerate(samples):
        for st, off in (("raw", -0.2), ("preprocessed", 0.2)):
            g = pr[(pr["sample"] == s) & (pr.status == st)]
            x = i * 2 + (0 if st == "raw" else 1)
            ax.scatter(x + jitter((~g.pass_qc).sum(), 0.18, i), g[~g.pass_qc].count_overlap, s=4, color=OI["lightgrey"], lw=0)
            ax.scatter(x + jitter(g.pass_qc.sum(), 0.18, i + 7), g[g.pass_qc].count_overlap, s=5, color=OI["grey"], lw=0)
            c = cons[(cons["sample"] == s) & (cons.status == st) & (cons.qc_filter == "passed_qc") &
                     (cons.ssim_filter == "top_50pct")].set_index("threshold")["count"]
            ax.errorbar(x + 0.28, c.loc[0.5], yerr=[[c.loc[0.5] - c.loc[0.75]], [c.loc[0.25] - c.loc[0.5]]], fmt="s",
                        ms=3.5, color=OI["blue"], lw=1.0, capsize=1.5)
            cm = comp[(comp["sample"] == s) & (comp.status == st)].set_index("method")
            ax.scatter(x + 0.28, cm.loc["default_0.95_-10", "count"], marker="^", s=14, color=OI["orange"], zorder=4)
            ax.scatter(x + 0.28, cm.loc["stiff_family_BE>=0.95_grid-30/-50", "count"], marker="v", s=12,
                       color=OI["vermillion"], zorder=4)
            rows.append(dict(sample=s, status=st, consensus_0p5=c.loc[0.5], consensus_0p25=c.loc[0.25],
                             consensus_0p75=c.loc[0.75], default=cm.loc["default_0.95_-10", "count"],
                             stiff_family=cm.loc["stiff_family_BE>=0.95_grid-30/-50", "count"]))
    ax.set_xticks([i * 2 + 0.5 for i in range(len(samples))], samples)
    for i in range(len(samples)):
        ax.text(i * 2, ax.get_ylim()[0], "R", fontsize=5, ha="center", va="bottom", color=OI["grey"])
        ax.text(i * 2 + 1, ax.get_ylim()[0], "P", fontsize=5, ha="center", va="bottom", color=OI["grey"])
    ax.set_ylabel("SNpc TH+ cells (overlap rule)")
    ax.scatter([], [], s=5, color=OI["lightgrey"], label="run failing global QC")
    ax.scatter([], [], s=5, color=OI["grey"], label="run passing global QC")
    ax.errorbar([], [], yerr=[], fmt="s", ms=3.5, color=OI["blue"], label="consensus p ≥ 0.5 (bars: 0.75–0.25)")
    ax.scatter([], [], marker="^", s=14, color=OI["orange"], label="brainreg default (0.95, −10)")
    ax.scatter([], [], marker="v", s=12, color=OI["vermillion"], label="stiff family (BE ≥ 0.95, grid −30/−50)")
    ax.legend(frameon=False, fontsize=5.5, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.03))
    src(pd.DataFrame(rows), "Fig7", "a_counts")
    src(pr, "Fig7", "a_counts_per_run")
    label(ax, "a", x=-0.1, y=1.2)
    ax = fig.add_subplot(gs[0, 2])
    order = ["single QC-run choice (IQR)", "single QC-run choice (range)", "QC thresholds IoU 0.85-0.95 x grad 0.040-0.055 (t=0.5)",
             "consensus QC/SSIM filter set (6 combos, t=0.5)", "consensus threshold 0.4-0.6", "consensus threshold 0.25-0.75"]
    nm = ["single QC run (IQR)", "single QC run (range)", "QC thresholds grid", "QC/SSIM filter set",
          "threshold 0.4–0.6", "threshold 0.25–0.75"]
    for j, k in enumerate(order):
        v = kn[kn.knob == k].rel_spread * 100
        ax.scatter(v, j + jitter(len(v), 0.12, j), s=6, color=OI["blue"] if "threshold" in k or "filter" in k else OI["grey"], lw=0)
        ax.plot([v.median()] * 2, [j - 0.3, j + 0.3], color="k", lw=0.9)
    ax.set_yticks(range(len(nm)), nm)
    ax.invert_yaxis()
    ax.set_xlabel("Spread of SNpc count\n(% of consensus count)")
    src(kn, "Fig7", "b_knobs")
    label(ax, "b", x=-0.95, y=1.2)
    ax = fig.add_subplot(gs[1, 0])
    mm = ["primary_consensus", "default_0.95_-10", "median_qc_run", "lobo_central_param",
          "same_param_single_run_(median over params passing QC in both)", "stiff_family_BE>=0.95_grid-30/-50"]
    nm2 = ["consensus", "default", "median QC run", "LOBO param", "same param\n(median)", "stiff family"]
    for j, m in enumerate(mm):
        v = rp[rp.method == m].abs_rel_diff * 100
        ax.scatter(v, j + jitter(len(v), 0.1, j), s=7, color=OI["blue"] if m == "primary_consensus" else OI["grey"], lw=0)
        ax.plot([v.median()] * 2, [j - 0.3, j + 0.3], color="k", lw=0.9)
    ax.set_yticks(range(len(nm2)), nm2)
    ax.invert_yaxis()
    ax.set_xlabel("|raw − preprocessed| count (%)")
    src(rp, "Fig7", "c_raw_vs_pre")
    label(ax, "c", x=-0.6)
    ax = fig.add_subplot(gs[1, 1])
    cc = comp[comp.method.isin(["default_0.95_-10", "median_qc_run", "lobo_central_param", "lobo_central_top4_ensemble",
                                "stiff_family_BE>=0.95_grid-30/-50", "affine_equivalent_BE1.0"])]
    nm3 = {"default_0.95_-10": "default", "median_qc_run": "median QC run", "lobo_central_param": "LOBO param",
           "lobo_central_top4_ensemble": "LOBO top-4", "stiff_family_BE>=0.95_grid-30/-50": "stiff family",
           "affine_equivalent_BE1.0": "BE 1.0"}
    for j, (m, n) in enumerate(nm3.items()):
        v = cc[cc.method == m].rel_dev * 100
        ok = cc[cc.method == m].runs_pass_qc.astype(bool)
        ax.scatter(v[ok], j + jitter(ok.sum(), 0.1, j), s=7, color=OI["grey"], lw=0)
        ax.scatter(v[~ok], j + jitter((~ok).sum(), 0.1, j + 3), s=9, color=OI["vermillion"], marker="x", lw=0.7)
    ax.axvline(0, color="k", lw=0.6)
    ax.axvspan(-5, 5, color=OI["lightgrey"], alpha=0.5, lw=0)
    ax.set_yticks(range(len(nm3)), list(nm3.values()))
    ax.invert_yaxis()
    ax.set_xlabel("Count vs consensus (%)\n(x: fails global QC)")
    src(cc, "Fig7", "d_choices")
    label(ax, "d", x=-0.55)
    ax = fig.add_subplot(gs[1, 2])
    v = var[var.rule == "count_table"]
    for st, col in (("raw", OI["blue"]), ("preprocessed", OI["sky"])):
        x = v[(v.status == st)].pivot_table(index="sample", columns="stage", values="cv")[["all", "global_qc", "ssim_top50"]]
        for s_, r in x.iterrows():
            ax.plot(range(3), r.values * 100, color=col, lw=0.8, marker="o", ms=2.5)
    ax.plot([], [], color=OI["blue"], label="raw"), ax.plot([], [], color=OI["sky"], label="preprocessed")
    ax.set_xticks(range(3), ["all", "global\nQC", "SSIM\ntop 50 %"])
    ax.set_ylabel("CV of single-run counts (%)")
    ax.legend(frameon=False, fontsize=6)
    src(v, "Fig7", "e_cv_by_stage")
    label(ax, "e", x=-0.35)
    save(fig, "Fig7_lsfm_snpc_counts")


# =================================================================================================================
def supp():
    sens = load_table("amap_sensitivity_cases")
    sens = sens[sens.structure.isin(STRUCTS_MAIN)]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.0), gridspec_kw=dict(wspace=0.55))
    s = sens[sens.brain.isin(QC_BRAINS) & (sens.variant == "raw")]
    for ax, (x, y, xl, yl) in zip(axs[:3], (("dice_amap_midline", "dice_paper", "aMAP midline", "registered hemispheres"),
                                             ("dice_fixed", "dice_paper", "fixed STAPLE slice", "paper protocol"),
                                             ("dice_paper", "dice_w10", "paper protocol", "window +10 slices"))):
        ax.scatter(s[x], s[y], s=7, color=OI["blue"], lw=0)
        ax.plot([0, 1], [0, 1], color=OI["grey"], ls="--", lw=0.6)
        ax.set_xlabel(f"Dice, {xl}")
        ax.set_ylabel(f"Dice, {yl}")
        ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax = axs[3]
    ax.hist(s.dA_um, bins=np.arange(-400, 425, 25), color=OI["blue"])
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Best slice − STAPLE slice (µm)\n(+ = posterior)")
    ax.set_ylabel("Cases")
    src(s, "FigS1", "sensitivities")
    for ax, l in zip(axs, "abcd"):
        label(ax, l, x=-0.35)
    save(fig, "FigS1_hemisphere_window_AP", main=False)

    ag = load_table("amap_agreement_cases")
    hum = load_table("human_loo")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2), gridspec_kw=dict(wspace=0.5))
    ax = axs[0]
    h = hum[hum.structure.isin(STRUCTS_MAIN)]
    ax.scatter(h.dice_self, h.dice_loo, s=3, color=OI["grey"], lw=0)
    ax.plot([0, 1], [0, 1], color="k", ls="--", lw=0.6)
    ax.set_xlabel("Rater Dice vs full STAPLE (self-inclusive)")
    ax.set_ylabel("Rater Dice vs LOO STAPLE")
    ax = axs[1]
    a = ag[ag.structure.isin(STRUCTS_MAIN) & ag.brain.isin(QC_BRAINS)]
    pv = a.pivot_table(index=["brain", "structure"], columns="variant", values="dice_paper")
    ax.scatter(pv.raw, pv.preprocessed, s=7, color=OI["blue"], lw=0)
    ax.plot([0, 1], [0, 1], color="k", ls="--", lw=0.6)
    ax.set_xlabel("Consensus Dice, raw")
    ax.set_ylabel("Consensus Dice, preprocessed")
    ax = axs[2]
    m = a[a.variant == "raw"].pivot_table(index="brain", columns="structure", values="dice_paper").reindex(columns=STRUCTS_MAIN)
    heat(ax, m.rename(index=SHORT), SEQ, 0.3, 1.0, fig=fig)
    ax.set_yticks(range(len(m)), [SHORT[b] for b in m.index])
    ax.set_xticks(range(len(STRUCTS_MAIN)), STRUCTS_MAIN, rotation=45, ha="right")
    src(h, "FigS2", "a_loo_vs_self")
    src(pv.reset_index(), "FigS2", "b_raw_vs_pre")
    src(m.reset_index(), "FigS2", "c_brain_structure")
    for ax, l in zip(axs, "abc"):
        label(ax, l, x=-0.3)
    save(fig, "FigS2_humans_preprocessing_brains", main=False)

    runs = load_table("amap_runs")
    er = runs[(runs.brain == "ER_Glt25d2Cre_9") & runs.structure.isin(STRUCTS_MAIN)]
    oth = runs[runs.brain.isin([b for b in QC_BRAINS if b != "ER_Glt25d2Cre_9"]) & runs.structure.isin(STRUCTS_MAIN)]
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.0), gridspec_kw=dict(wspace=0.5))
    for ax, v in zip(axs, ("raw", "preprocessed")):
        ax.hist(oth[oth.variant == v].dice_paper, bins=np.linspace(0, 1, 41), density=True, color=OI["lightgrey"],
                label="runs, other brains")
        ax.hist(er[er.variant == v].dice_paper, bins=np.linspace(0, 1, 41), density=True, histtype="step",
                color=OI["vermillion"], lw=1.0, label="runs, ER")
        ax.set_xlabel(f"Run Dice ({v})")
        ax.set_ylabel("Density")
    axs[0].legend(frameon=False, fontsize=6, loc="upper left")
    src(pd.concat([er.assign(group="ER"), oth.assign(group="QC brains")])[["group", "variant", "brain", "structure",
                                                                              "run", "dice_paper"]], "FigS3", "er")
    save(fig, "FigS3_ER_rejected_brain", main=False)

    mem = load_table("lsfm_membership_rule")
    mem["sample"] = mem["sample"].astype(str)
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    mm = mem[mem.source == "qc_ssim_filtered"].copy()
    x = np.arange(len(mm))
    ax.bar(x - 0.27, mm.overlap_consensus_t0p5, width=0.25, color=OI["blue"], label="overlap rule, consensus p ≥ 0.5")
    ax.bar(x, mm.count_p_ge_0p5, width=0.25, color=OI["sky"], label="centroid rule, P(SNc) ≥ 0.5")
    ax.bar(x + 0.27, mm.expected_count_sum_p, width=0.25, color=OI["grey"], label="centroid rule, Σ P(SNc)")
    ax.set_xticks(x, [f"{s}\n{st[:3]}" for s, st in zip(mm["sample"], mm.status)], fontsize=5.5)
    ax.set_ylabel("SNpc TH+ cells")
    ax.legend(frameon=False, fontsize=5.5, loc="upper left", bbox_to_anchor=(0, 1.25), ncol=2)
    src(mm, "FigS4", "membership_rule")
    save(fig, "FigS4_lsfm_membership_rule", main=False)


def fig8_scale():
    b = load_table("scale_selection_group_bias")
    a = load_table("scale_selection_freedom")
    sl = load_table("scale_sweep_lsfm")
    sa = load_table("scale_sweep_amap")
    tr = load_table("scale_tail_risk")
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 3, hspace=0.95, wspace=0.55)
    # (a) bias bound on a 2-vs-3 group comparison (raw)
    ax = fig.add_subplot(gs[0, :2])
    x0 = b[(b.status == "raw")].sort_values("consensus_diff")
    splits = x0[x0.accepted_set == "global_qc"].group_a.tolist()
    for j, g in enumerate(splits):
        for setname, off, col in (("global_qc", -0.15, OI["grey"]), ("ssim_top50", 0.15, OI["sky"])):
            r = x0[(x0.group_a == g) & (x0.accepted_set == setname)].iloc[0]
            ax.plot([j + off] * 2, [r.extreme_selection_lo * 100, r.extreme_selection_hi * 100], color=col, lw=2.2,
                    solid_capstyle="butt")
            ax.plot([j + off] * 2, [r.random_selection_p05 * 100, r.random_selection_p95 * 100], color="k", lw=0.8)
        ax.scatter(j, x0[(x0.group_a == g)].consensus_diff.iloc[0] * 100, marker="D", s=14, color=OI["blue"], zorder=4)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(range(len(splits)), [f"{g} vs rest" for g in splits], fontsize=5.5, rotation=30, ha="right")
    ax.set_ylabel("Apparent group difference (%)")
    ax.plot([], [], color=OI["grey"], lw=2.2, label="range reachable by selecting runs (global-QC set)")
    ax.plot([], [], color=OI["sky"], lw=2.2, label="range reachable (SSIM top-50 % set)")
    ax.plot([], [], color="k", lw=0.8, label="unbiased random selection (5–95 %)")
    ax.scatter([], [], marker="D", s=14, color=OI["blue"], label="consensus")
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    src(x0, "Fig8", "a_group_bias")
    label(ax, "a", x=-0.08, y=1.18)
    # (b) selection vs biological variance
    ax = fig.add_subplot(gs[0, 2])
    s = a[a["sample"] == "ALL"]
    xs = np.arange(2)
    for k, (st, col) in enumerate((("raw", OI["blue"]), ("preprocessed", OI["sky"]))):
        v = s[s.status == st].set_index("accepted_set").selection_to_between_variance.reindex(["global_qc", "ssim_top50"]) * 100
        ax.bar(xs + (k - 0.5) * 0.35, v, width=0.35, color=col, label=st)
    ax.set_xticks(xs, ["choose among\nglobal-QC runs", "choose among\nSSIM top 50 %"])
    ax.set_ylabel("Selection variance as % of\nbetween-animal variance")
    ax.legend(frameon=False, fontsize=6)
    src(s, "Fig8", "b_variance")
    label(ax, "b", x=-0.45, y=1.18)
    # (c) sweep size LSFM
    ax = fig.add_subplot(gs[1, 0])
    for st, col in (("raw", OI["blue"]), ("preprocessed", OI["sky"])):
        r = sl[(sl.status == st) & sl.design.str.startswith("random")].sort_values("k")
        ax.plot(r.k, r.median_abs_dev * 100, "o-", color=col, ms=2.5, lw=1, label=f"{st} median")
        ax.plot(r.k, r.p90_abs_dev * 100, "o--", color=col, ms=2, lw=0.8, label=f"{st} 90th pct")
        st9 = sl[(sl.status == st) & ~sl.design.str.startswith("random")]
        ax.scatter([9] * len(st9), st9.median_abs_dev * 100, marker="s", s=10, color=col, edgecolor="k", lw=0.4, zorder=4)
    ax.set_xlabel("Registrations per brain (k)")
    ax.set_ylabel("|count − full-sweep consensus| (%)")
    sec = ax.secondary_xaxis("top", functions=(lambda k: k * 23.05 / 60, lambda h: h * 60 / 23.05))
    sec.set_xlabel("CPU-hours per brain", fontsize=6)
    sec.tick_params(labelsize=5.5)
    ax.legend(frameon=False, fontsize=5, loc="upper right")
    src(sl, "Fig8", "c_sweep_lsfm")
    label(ax, "c", x=-0.3, y=1.25)
    # (d) sweep size aMAP
    ax = fig.add_subplot(gs[1, 1])
    for v, col in (("raw", OI["blue"]), ("preprocessed", OI["sky"])):
        r = sa[(sa.variant == v) & sa.design.str.startswith("random")].sort_values("k")
        ax.plot(r.k, r.median_delta, "o-", color=col, ms=2.5, lw=1, label=f"{v}: median")
        ax.plot(r.k, r.p10_delta, "o--", color=col, ms=2, lw=0.8, label=f"{v}: 10th pct")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Registrations per brain (k)")
    ax.set_ylabel("ΔDice vs full 64-run consensus")
    ax.legend(frameon=False, fontsize=5, loc="lower right")
    src(sa, "Fig8", "d_sweep_amap")
    label(ax, "d", x=-0.35, y=1.25)
    # (e) expected worst case in a cohort
    ax = fig.add_subplot(gs[1, 2])
    t = tr[tr.variant == "raw"]
    cols = {"consensus": OI["blue"], "default": OI["orange"], "top-SSIM run": OI["purple"], "LOBO parameter": OI["green"],
            "random QC-passing run": OI["grey"]}
    Ns = [1, 10, 50, 100]
    for m, c in cols.items():
        r = t[t.method == m].iloc[0]
        ax.plot(Ns, [r["median"], r.expected_worst_of_10, r.expected_worst_of_50, r.expected_worst_of_100], "o-",
                color=c, ms=2.5, lw=1, label=m)
    ax.set_xscale("log")
    ax.set_xticks(Ns, ["median", "10", "50", "100"])
    ax.set_xlabel("Cohort size (cases)")
    ax.set_ylabel("Expected worst-case Dice (raw)")
    ax.legend(frameon=False, fontsize=5, loc="lower left")
    src(t, "Fig8", "e_tail")
    label(ax, "e", x=-0.35, y=1.25)
    save(fig, "Fig8_robustness_at_scale")


def supp_qc_by_sample():
    """S5: IoU x grad_p95 per registration, coloured by sample (reviewer 2), both datasets."""
    pal = [OI["blue"], OI["orange"], OI["green"], OI["vermillion"], OI["purple"], OI["black"]]
    sc = load_table("amap_qc_scatter")
    pr = load_table("lsfm_per_run")
    pr["sample"] = pr["sample"].astype(str)
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.0), gridspec_kw=dict(wspace=0.75))
    panels = [("aMAP raw", sc[sc.variant == "raw"], "brain"), ("aMAP preprocessed", sc[sc.variant == "preprocessed"], "brain"),
              ("LSFM raw", pr[pr.status == "raw"], "sample"), ("LSFM preprocessed", pr[pr.status == "preprocessed"], "sample")]
    for ax, (title, d, key) in zip(axs, panels):
        for c, (k, g) in zip(pal, d.groupby(key)):
            ax.scatter(g.iou, g.grad_p95, s=5, color=c, lw=0, label=SHORT.get(k, k))
        ax.axvline(0.9, color="k", ls="--", lw=0.6)
        ax.axhline(0.045, color="k", ls="--", lw=0.6)
        ax.set_title(title)
        ax.set_xlabel("Tissue IoU")
        if ax in (axs[1], axs[3]):
            ax.legend(frameon=False, fontsize=5, markerscale=1.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axs[0].set_ylabel("grad_p95")
    src(pd.concat([sc.assign(dataset="aMAP"), pr.rename(columns={"sample": "brain", "status": "variant"}).assign(dataset="LSFM")])
        [["dataset", "brain", "variant", "iou", "grad_p95"]], "FigS5", "iou_grad_by_sample")
    save(fig, "FigS5_qc_metrics_by_sample", main=False)


def supp_residual():
    """S6: SNpc counts of registrations retained after all QC vs global-QC only, per brain, relative to consensus."""
    pr = load_table("lsfm_per_run")
    pr["sample"] = pr["sample"].astype(str)
    res = load_table("residual_ambiguity_per_brain")
    res["sample"] = res["sample"].astype(str)
    cons = res[res.set.str.startswith("all")].set_index(["sample", "status"]).consensus
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    x = 0
    ticks, labs = [], []
    for s in sorted(pr["sample"].unique()):
        for st in ("raw", "preprocessed"):
            g = pr[(pr["sample"] == s) & (pr.status == st)]
            P = cons[(s, st)]
            gq = g[g.pass_qc & ~g.in_ssim_top50.astype(bool)]
            aq = g[g.in_ssim_top50.astype(bool)]
            ax.scatter(x - 0.15 + jitter(len(gq), 0.08, x), (gq.count_overlap / P - 1) * 100, s=6, color=OI["lightgrey"], lw=0)
            ax.scatter(x + 0.15 + jitter(len(aq), 0.08, x + 50), (aq.count_overlap / P - 1) * 100, s=7, color=OI["blue"], lw=0)
            q1, q3 = np.percentile(aq.count_overlap / P - 1, [25, 75]) * 100
            ax.plot([x + 0.35] * 2, [q1, q3], color="k", lw=1.5)
            ticks.append(x)
            labs.append(f"{s}\n{st[:3]}")
            x += 1
    ax.axhline(0, color="k", lw=0.6)
    ax.axhspan(-5, 5, color=OI["lightgrey"], alpha=0.35, lw=0)
    ax.set_xticks(ticks, labs, fontsize=5.5)
    ax.set_ylabel("Count vs consensus (%)")
    ax.scatter([], [], s=6, color=OI["lightgrey"], label="passes global QC only")
    ax.scatter([], [], s=7, color=OI["blue"], label="passes all QC (retained)")
    ax.plot([], [], color="k", lw=1.5, label="IQR of retained")
    ax.legend(frameon=False, fontsize=5.5, ncol=3, loc="lower left")
    src(res.drop(columns="retained_params"), "FigS6", "residual_ambiguity")
    save(fig, "FigS6_residual_ambiguity_after_qc", main=False)


if __name__ == "__main__":
    for f in (fig1_agreement, fig2_benefit, fig3_landscape, fig4_qc, fig5_uncertainty, fig6_predictors, fig7_lsfm, fig8_scale, supp,
              supp_qc_by_sample, supp_residual):
        f()

