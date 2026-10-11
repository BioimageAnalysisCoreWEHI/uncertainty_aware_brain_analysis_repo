"""Figure 4 (v3): aMAP expert validation. Implements mockups/Fig4_v3_mockup.png from the existing revision panels.

  A-E  RSP, brain 165, through the pipeline (row 1 of the current Fig 5; validation_analysis 22_focused_revision_figures
       draw_row / f01.view, unchanged)
  F    Dice with the expert STAPLE per structure: annotations, retained single registrations, consensus; SUB* = dorsal
       SUB sensitivity analysis (as current Fig 5F)
  G    range of single QC-passing registrations, brainreg default and consensus for all 54 cases (current Fig 6A)
  H    four examples (current Fig 6B); the RSP example is the typical case shown in A-E
Supplementary: the MV row (brain 165, section 487) of the current Fig 5 -> FigS_aMAP_MV_example.

Output: manuscript_figures/Fig4_v3.{png,pdf,svg}, FigS_aMAP_MV_example.{png,pdf,svg}, source_data_fig4/*.csv
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P  # noqa: E402  (resolves <IMAGING>/<SCRATCH> placeholders)
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VA = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_analysis"))
_REPO = next(_p for _p in Path(__file__).resolve().parents if (_p / "uaa_paths.py").exists())
_CODE = _REPO / "validation" / "analysis" / "scripts"          # analysis scripts in this repository
sys.path.insert(0, str(_CODE))
os.chdir(_CODE)
_s = importlib.util.spec_from_file_location("m22", str(_CODE / "22_focused_revision_figures.py"))
m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m)
plt, Line2D, Patch = m.plt, m.Line2D, m.Patch

OUT = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\manuscript_figures"))
SRC = OUT / "source_data_fig4"
SRC.mkdir(exist_ok=True)
TITLES = ["Expert annotations\n+ STAPLE", "All registrations\n(before QC)", "Registrations\nretained after QC",
          "Consensus\nvs STAPLE", "Consensus\nvs annotations"]
CONS_LAB = "consensus (0.5)"


def row(fig, gsrow, b, s, section=None, letters=True):
    cs, final = m.f01.load(b, s, m.qc())
    v = m.view_at(cs, final, section) if section is not None else m.f01.view(cs, final)
    sc = m.f01.case_scores(cs, final)
    axs = [fig.add_subplot(gsrow[c]) for c in range(5)]
    m.draw_row(axs, final, v, sc, n_users=len(set(cs.rater_user)))
    axs[0].text(-0.08, 0.5, f"{m.sname('aMAP', b)}  {s}", transform=axs[0].transAxes, rotation=90, va="center",
                ha="right", fontsize=7)
    for ax, t, l in zip(axs, TITLES, "ABCDE"):
        ax.set_title(t, fontsize=6.5)
        if letters:
            m.label(ax, l, x=-0.04, y=1.22)
    return dict(brain=b, sample=m.sname("aMAP", b), structure=s, section=section if section is not None else cs.A0, **sc)


def row_legend(fig, y):
    h = [Line2D([], [], color=m.C_ANN, lw=0.8, alpha=0.7, label="individual annotation"),
         Line2D([], [], color=m.C_REF, lw=1.3, label="expert consensus (STAPLE)"),
         Line2D([], [], color=m.C_RUN, lw=0.8, label="single registration"),
         Line2D([], [], color="#b88fd0", lw=1.3, label=CONS_LAB)]
    leg = fig.legend(handles=h, loc="lower center", ncol=4, frameon=True, fontsize=6, bbox_to_anchor=(0.5, y))
    leg.get_frame().set_facecolor("#555555")
    for t in leg.get_texts():
        t.set_color("white")


def panel_f(ax):
    STRUCTS_MAIN = m.STRUCTS_MAIN
    cases = pd.read_csv(VA / "focused" / "tables" / "focused_case_scores.csv")
    cases = cases[cases.structure.isin(STRUCTS_MAIN)]
    raters = pd.read_csv(VA / "focused" / "tables" / "focused_rater_scores.csv")
    raters = raters[raters.structure.isin(STRUCTS_MAIN)]
    runs = pd.read_csv(m.TABLES / "amap_runs.csv")
    runs = runs[(runs.variant == "raw") & runs.structure.isin(STRUCTS_MAIN) & runs.passes_final_qc.astype(bool)]
    sd = pd.read_csv(m.SRC / "SUB_dorsal_only_per_case.csv")
    sg = pd.read_csv(m.SRC / "SUB_dorsal_only_singles.csv")
    extra = dict(ann=raters[raters.structure == "SUB"].dice, run=sg.dice_dorsal_only, cons=sd.dice_consensus_dorsal_only)
    bxp = dict(showfliers=False, patch_artist=True, widths=0.22, medianprops=dict(color="k", lw=0.8),
               whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    isub = STRUCTS_MAIN.index("SUB")
    pos = {st: k + (1 if k > isub else 0) for k, st in enumerate(STRUCTS_MAIN)}
    for st in STRUCTS_MAIN:
        j = pos[st]
        c = cases[cases.structure == st].dice_amap_protocol
        for dx, vals, fc in ((-0.26, raters[raters.structure == st].dice, m.C_ANN_BOX),
                             (0.0, runs[runs.structure == st].dice_paper, m.C_RUN_BOX), (0.26, c, m.C_CONS_BOX)):
            bp = ax.boxplot(vals, positions=[j + dx], **bxp)
            bp["boxes"][0].set(facecolor=fc, edgecolor="0.4", lw=0.6)
        ax.scatter(np.full(len(c), j + 0.26) + np.linspace(-0.06, 0.06, len(c)), c, s=11, color=m.C_CONS,
                   edgecolor="white", lw=0.3, zorder=3)
    j = isub + 1
    for dx, vals, fc in ((-0.26, extra["ann"], m.C_ANN_BOX), (0.0, extra["run"], m.C_RUN_BOX),
                         (0.26, extra["cons"], m.C_CONS_BOX)):
        bp = ax.boxplot(vals, positions=[j + dx], **bxp)
        bp["boxes"][0].set(facecolor=fc, edgecolor="0.4", lw=0.6)
        bp["boxes"][0].set_hatch("////")
    c = extra["cons"]
    ax.scatter(np.full(len(c), j + 0.26) + np.linspace(-0.06, 0.06, len(c)), c, s=11, color=m.C_CONS,
               edgecolor="white", lw=0.3, zorder=3)
    ax.axvspan(j - 0.48, j + 0.48, color="0.95", zorder=0, lw=0)
    ax.text(j, 1.0, "dorsal\npart only", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.8,
            color="0.3")
    order = sorted(list(pos.items()) + [("SUB*", j)], key=lambda t: t[1])
    ax.set_xticks([t[1] for t in order], [t[0] for t in order])
    ax.set_xlim(-0.6, len(STRUCTS_MAIN) + 0.6)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Dice with expert\nconsensus (STAPLE)")
    per = raters.groupby(["brain", "structure"]).size()
    rep = sorted({m.SHORT.get(b, b) for (b, _), n in per.items() if n > per.min()})
    ann_lab = (f"individual annotations ({per.min()} per structure and sample; {per.max()} for "
               f"{', '.join(rep)}: two attempts)")
    ax.legend(handles=[Patch(facecolor=m.C_ANN_BOX, edgecolor="0.4", label=ann_lab),
                       Patch(facecolor=m.C_RUN_BOX, edgecolor="0.4", label="single registrations retained after QC"),
                       Patch(facecolor=m.C_CONS_BOX, edgecolor="0.4", label="consensus (dots: one per sample)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=6)
    ax.text(0.5, -0.36, "* SUB scored on its dorsal part only (sensitivity analysis): the ventral subiculum, which the "
            "experts did not annotate, was excluded for the consensus and the single registrations.",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.5, color="0.3")
    m.label(ax, "F", x=-0.06, y=1.0)
    pd.concat([raters.assign(group="annotator")[["group", "brain", "structure", "dice"]],
               runs.assign(group="single retained registration").rename(columns={"dice_paper": "dice"})[
                   ["group", "brain", "structure", "dice"]],
               cases.assign(group="consensus").rename(columns={"dice_amap_protocol": "dice"})[
                   ["group", "brain", "structure", "dice"]]]).to_csv(SRC / "Fig4_F_scores.csv", index=False)


def panels_gh(fig, ax, gsh):
    runs = pd.read_csv(m.TABLES / "amap_runs.csv")
    runs = runs[(runs.variant == "raw") & runs.structure.isin(m.STRUCTS_MAIN) & runs.passes_global_qc.astype(bool)]
    ben = pd.read_csv(m.TABLES / "amap_benefit_cases.csv")
    ben = ben[(ben.variant == "raw") & ben.structure.isin(m.STRUCTS_MAIN)]
    rng_ = runs.groupby(["brain", "structure"]).dice_paper.agg(["min", "max", "count"]).reset_index()
    d = ben.merge(rng_, on=["brain", "structure"])
    d["sample_number"] = d.brain.map(lambda b: m.NUM[("aMAP", b)])
    d["spread"] = d["max"] - d["min"]
    order = d.groupby("structure").cons_dice.median().sort_values(ascending=False).index.tolist()
    d["sorder"] = d.structure.map(order.index)
    d = d.sort_values(["sorder", "sample_number"]).reset_index(drop=True)
    d["x"] = d.index + d.sorder * 1.0
    top = d.sort_values("spread", ascending=False).drop_duplicates("structure")
    visp = d[(d.structure == "VISp") & (d.brain == "AB_Gad67_223")]
    ex = pd.concat([top[top.structure == "VISam"], visp, top[top.structure == "AHN"],
                    top[top.structure.isin(["RSP", "SUB"])].head(1)])
    ax.vlines(d.x, d["min"], d["max"], color=m.C_RUN, lw=1.8, zorder=1)
    ax.scatter(d.x, d.default_dice, marker="x", s=12, color=m.C_DEF, lw=0.8, zorder=2)
    ax.scatter(d.x, d.cons_dice, s=12, color=m.C_CONS, lw=0, zorder=3)
    for r in ex.itertuples():
        ax.scatter(r.x, r.cons_dice, s=45, facecolor="none", edgecolor="k", lw=0.8, zorder=4)
    ax.set_xticks(d.x, d.brain.map(m.SHORT), fontsize=4.5, rotation=90)
    for st in order:
        xs = d.x[d.structure == st]
        ax.text(xs.mean(), -0.2, st, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5)
        ax.axvline(xs.max() + 1.0, color="0.9", lw=0.6, zorder=0)
    ax.text(-0.01, -0.035, "brain", transform=ax.transAxes, ha="right", va="top", fontsize=5.5, color="0.4")
    ax.set_xlim(d.x.min() - 1, d.x.max() + 1)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Dice with expert\nconsensus (STAPLE)")
    ax.legend(handles=[Line2D([], [], color=m.C_RUN, lw=2, label="range of single QC-passing registrations"),
                       Line2D([], [], ls="", marker="x", color=m.C_DEF, label="brainreg default registration"),
                       Line2D([], [], ls="", marker="o", ms=4, color=m.C_CONS, label="consensus"),
                       Line2D([], [], ls="", marker="o", ms=6, mfc="none", mec="k", label="examples in H")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=6)
    m.label(ax, "G", x=-0.06, y=1.06)
    for j, r in enumerate(ex.itertuples()):
        cs, used, k, _ = m.amap_case(r.brain, r.structure)
        qcruns = runs[(runs.brain == r.brain) & (runs.structure == r.structure)].run.tolist()
        masks = [cs.run_mask(q)[k] for q in qcruns if q in cs.idx]
        v = m.amap_view(cs, used, k, extra=masks)
        axb = fig.add_subplot(gsh[j])
        m.base(axb, v)
        for mk in masks:
            m.contour(axb, v["put"](mk), m.C_RUN, 0.4, alpha=0.7)
        m.contour(axb, v["truth"], m.C_REF, 1.2)
        m.contour(axb, v["cons"], m.C_CONS, 1.4)
        typical = " (typical case, as in A–E)" if r.structure == "RSP" else ""
        axb.set_title(f"{r.structure}, {m.sname('aMAP', r.brain)}{typical}", fontsize=6.5, pad=2)
        m.tag(axb, f"single {r._asdict()['min']:.2f}–{r._asdict()['max']:.2f}\nconsensus {r.cons_dice:.2f}", 5.5)
        if j == 0:
            m.label(axb, "H", x=-0.06, y=1.08)
    d[["structure", "brain", "min", "max", "count", "default_dice", "cons_dice", "spread"]].to_csv(
        SRC / "Fig4_G_cases.csv", index=False)
    ex[["structure", "brain", "min", "max", "cons_dice"]].to_csv(SRC / "Fig4_H_examples.csv", index=False)
    return [Line2D([], [], color=m.C_RUN, lw=0.8, label="single QC-passing registrations"),
            Line2D([], [], color="0.6", lw=1.2, label="expert consensus (STAPLE, white)"),
            Line2D([], [], color=m.C_CONS, lw=1.4, label="consensus")]


def main():
    rsp, mv = m.f01.EXAMPLES[0], m.f01.EXAMPLES[1]
    fig = plt.figure(figsize=(7.2, 9.4))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.72, 1.0, 1.15, 0.95], hspace=0.75, top=0.915, bottom=0.05)
    ex = [row(fig, gs[0].subgridspec(1, 5, wspace=0.05), *rsp)]
    row_legend(fig, 0.965)
    panel_f(fig.add_subplot(gs[1]))
    gg = gs[2:].__class__ if False else None
    ax_g = fig.add_subplot(gs[2])
    h = panels_gh(fig, ax_g, gs[3].subgridspec(1, 4, wspace=0.05))
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.015), fontsize=6)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"Fig4_v3.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # supplementary: the MV row
    fig = plt.figure(figsize=(7.2, 1.9))
    gs = fig.add_gridspec(1, 5, wspace=0.05, top=0.7)
    ex.append(row(fig, [gs[0, c] for c in range(5)], *mv, section=487))
    row_legend(fig, 0.9)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"FigS_aMAP_MV_example.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.DataFrame(ex).to_csv(SRC / "Fig4_AE_and_supp_rows.csv", index=False)
    print(pd.DataFrame(ex).T)


if __name__ == "__main__":
    main()
