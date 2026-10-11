"""Figure 6 (v4): as v3 with D and E on a shared y axis.
Figure 6 (v3): global QC metrics across datasets and sensitivity of the consensus to the global QC rules.
One layout, drawn for raw input (main figure) and preprocessed input (supplementary):

  A-C  tissue-atlas IoU vs grad_p95 for every registration: original LSFM, aMAP, independent LSFM (this input only)
  D    aMAP: change in consensus Dice with the expert STAPLE vs the current rule, 54 brain x structure cases
  E    independent LSFM (new): change in consensus 3-D Dice with the published registration vs the current rule,
       16 regions x 6 brains (validation_analysis results/tables/qcsens_lsfm2_per_case.csv, dice_vs_ref)
  F-H  original LSFM SNpc: voxel Dice, count change and fraction of cells kept vs the current rule, full-resolution
       consensus and Cellpose cells (REVISION_OUTPUTS_FOCUSED/source_data/FigS7_A-C_original_LSFM_vs_current.csv)
Rules, settings and drawing helpers from validation_analysis/scripts/22_focused_revision_figures.py (unchanged).
Output: manuscript_figures/Fig6_v3.{png,pdf,svg} (raw), FigS_QC_rules_preprocessed.{png,pdf,svg}, source_data_fig6/
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
import yaml

VA = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_analysis"))
_REPO = next(_p for _p in Path(__file__).resolve().parents if (_p / "uaa_paths.py").exists())
_CODE = _REPO / "validation" / "analysis" / "scripts"          # analysis scripts in this repository
sys.path.insert(0, str(_CODE))
os.chdir(_CODE)
_s = importlib.util.spec_from_file_location("m22", str(_CODE / "22_focused_revision_figures.py"))
m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m)
plt, Line2D = m.plt, m.Line2D

OUT = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\manuscript_figures"))
SRC = OUT / "source_data_fig6"
SRC.mkdir(exist_ok=True)
KEY = {"paper LSFM": "paper", "aMAP": "amap", "validation LSFM": "lsfm2"}
TITLE = {"paper LSFM": "Original LSFM", "aMAP": "aMAP (serial two-photon)", "validation LSFM": "Independent LSFM"}
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def retention(ds, variant):
    lk = pd.read_csv(m.QS / ds / "setting_lookup.csv")
    g = lk[lk.variant == variant].drop_duplicates(["sample", "setting_id"])
    tot = 48 if ds == "paper" else 64
    return {n: 100 * g[g.setting_id == ids[ds]].n_global.sum() / (tot * g["sample"].nunique())
            for n, ids in m.REGIMES.items()}


def rule_order(ds, variant):
    ret = retention(ds, variant)
    return sorted(m.REGIMES, key=lambda n: -ret[n]), ret


def scatters(fig, g, variant, src):
    f19 = m._load("f19", _CODE / "19_figure3_qc.py")
    qc12 = m._load("qc12", _CODE / "12_qc_threshold_procedure.py")
    allqc = qc12.load_all()
    allqc["sample"] = [m.sname(ds, s) for ds, s in zip(allqc.dataset, allqc["sample"].astype(str))]
    ylim = (0.028, allqc.grad_p95.max() * 1.04)
    gg = g.subgridspec(1, 3, wspace=0.32)
    pad = lambda lo, hi: (lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo))  # noqa: E731
    for k, ds in enumerate(KEY):
        st = pd.read_csv(m.QS / KEY[ds] / "settings.csv").set_index("setting_id").loc[m.REGIMES["Current"][KEY[ds]]]
        thr = (float(st.iou_thr), float(st.grad_thr))
        d = allqc[allqc.dataset == ds]
        dv = d[d.variant == variant]
        ax = fig.add_subplot(gg[k])
        f19.scatter(ax, dv, thr, ylim, show_marker_legend=False)
        ax.set_xlim(*pad(d.iou.min(), d.iou.max()))
        lg = ax.get_legend()
        if lg is not None:
            lg.set_bbox_to_anchor((0.5, -0.24))
        if k:
            ax.set_ylabel("")
        ax.set_title(TITLE[ds], fontsize=7, pad=3)
        n_pass = int(((dv.iou > thr[0]) & (dv.grad_p95 < thr[1])).sum())
        right = ds == "paper LSFM"
        ax.text(0.97 if right else 0.03, 0.97, f"{n_pass}/{len(dv)} pass", transform=ax.transAxes, fontsize=5.8,
                ha="right" if right else "left", va="top", color="0.3")
        m.label(ax, "ABC"[k], x=-0.22 if k == 0 else -0.12, y=1.06)
        src.append(dict(panel="ABC"[k], dataset=ds, input=variant, iou_threshold=thr[0], grad_threshold=thr[1],
                        n_pass=n_pass, n=len(dv)))
    dv = allqc[allqc.variant == variant]
    dv[["dataset", "sample", "bending_energy", "grid", "iou", "grad_p95"]].to_csv(
        SRC / f"Fig6_ABC_qc_scatter_{variant}.csv", index=False)


def delta_panel(ax, w, ds, variant, letter, title, ylab, src):
    order, ret = rule_order(ds, variant)
    alts = [n for n in order if n != "Current"]
    cur = m.REGIMES["Current"][ds]
    rng = np.random.default_rng(0)
    for i, name in enumerate(alts):
        sid = m.REGIMES[name][ds]
        dd = (w[sid] - w[cur]).dropna()
        n_empty = int(w[sid].isna().sum())
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(dd)), dd, s=6, color="0.45", lw=0, alpha=0.7)
        ax.plot([i - 0.28, i + 0.28], [dd.median()] * 2, color=m.C_CONS, lw=1.6)
        src.append(dict(panel=letter, dataset=ds, input=variant, rule=name, setting_id=sid, retained_pct=ret[name],
                        n_cases=len(dd), n_cases_without_registrations=n_empty, median_delta=dd.median(),
                        max_abs_delta=dd.abs().max(), frac_within_0p02=float((dd.abs() <= 0.02).mean())))
    rows = [r for r in src if r.get("panel") == letter]
    ax.axhline(0, color="k", lw=0.6)
    ax.axhspan(-0.02, 0.02, color="0.92", zorder=0)
    lab = [f"{m.SHORT_RULE[n]} ({ret[n]:.0f}%)" + (f"\n{r['n_cases_without_registrations']} empty"
                                                  if r["n_cases_without_registrations"] else "")
           for n, r in zip(alts, rows)]
    ax.set_xticks(range(len(alts)), lab, fontsize=5.4, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(alts) - 0.4)
    ax.set_ylabel(ylab, fontsize=6.2)
    ax.tick_params(axis="y", labelsize=5.8)
    ax.set_title(f"{title}; current rule retains {ret['Current']:.0f}%", fontsize=6.3, pad=3, loc="left")
    m.label(ax, letter, x=-0.16, y=1.02)


def panel_d(ax, variant, src):
    per = pd.read_csv(m.TABLES / "qcsens_amap_per_case.csv")
    per = per[(per.variant == variant) & per.structure.isin(m.STRUCTS_MAIN)]
    w = per.pivot_table(index=["brain", "structure"], columns="setting_id", values="dice_paper")
    delta_panel(ax, w, "amap", variant, "D", "aMAP vs expert STAPLE (54 cases)",
                "Change in consensus Dice\nvs current rule", src)
    w.sub(w[m.REGIMES["Current"]["amap"]], axis=0).reset_index().to_csv(SRC / f"Fig6_D_delta_{variant}.csv", index=False)


def panel_e(ax, variant, src):
    per = pd.read_csv(m.TABLES / "qcsens_lsfm2_per_case.csv")
    per = per[per.variant == variant]
    w = per.pivot_table(index=["sample", "region"], columns="setting_id", values="dice_vs_ref")
    delta_panel(ax, w, "lsfm2", variant, "E", f"Independent LSFM vs published registration ({len(w)} cases)",
                "Change in consensus 3-D Dice\nvs current rule", src)
    w.sub(w[m.REGIMES["Current"]["lsfm2"]], axis=0).reset_index().to_csv(SRC / f"Fig6_E_delta_{variant}.csv",
                                                                        index=False)


def panels_fgh(fig, g, variant):
    ag = pd.read_csv(m.SRC / "FigS7_A-C_original_LSFM_vs_current.csv")
    ag = ag[ag.input == variant]
    order, ret = rule_order("paper", variant)
    alts = [n for n in order if n != "Current"]
    labels = sorted(ag.sample_label.unique())
    colr = {b: OKABE[k] for k, b in enumerate(labels)}
    gg = g.subgridspec(1, 3, wspace=0.35)
    specs = [("voxel_dice", "SNpc voxel Dice\nvs current rule", (0.4, 1.02), 1.0, "F"),
             ("count_change_pct", "SNpc TH+ count change\nvs current rule (%)", (-60, 20), 0.0, "G"),
             ("frac_current_cells_kept", "Fraction of current rule's\ncells still counted", (0.4, 1.02), 1.0, "H")]
    for k, (col, ylab, yl, ref, let) in enumerate(specs):
        ax = fig.add_subplot(gg[k])
        for i, r in enumerate(alts):
            h = ag[(ag.rule == r) & ~ag.is_empty].sort_values("sample_label")
            x = i + np.linspace(-0.12, 0.12, len(h))
            ax.scatter(x, h[col], s=12, c=[colr[b] for b in h.sample_label], lw=0.3, edgecolor="white", zorder=3)
            for _, e in ag[(ag.rule == r) & ag.is_empty].iterrows():
                ax.annotate(f"{e.sample_label}:\nnone passes", (i, yl[0]), xytext=(0, 3), textcoords="offset points",
                            fontsize=4.6, ha="center", va="bottom", color="0.3")
        ax.axhline(ref, color="0.6", lw=0.5, ls=(0, (3, 2)), zorder=1)
        ax.set_xticks(range(len(alts)), [f"{m.SHORT_RULE[n]} ({ret[n]:.0f}%)" for n in alts], fontsize=5.4,
                      rotation=40, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(alts) - 0.4)
        ax.set_ylim(*yl)
        ax.set_ylabel(ylab, fontsize=6.2)
        ax.tick_params(axis="y", labelsize=5.8)
        ax.grid(axis="y", color="0.93", lw=0.4, zorder=0)
        m.label(ax, let, x=-0.3, y=1.02)
        if k == 0:
            ax.text(-0.3, 1.2, f"Original LSFM SNpc (full-resolution consensus and TH+ cells) vs the current rule "
                               f"(current retains {ret['Current']:.0f}%)", transform=ax.transAxes, fontsize=6.5,
                    va="bottom")
            ax.legend(handles=[Line2D([], [], ls="", marker="o", ms=3.4, color=colr[b], label=b) for b in labels],
                      loc="lower left", bbox_to_anchor=(1.0, 1.06), fontsize=5.6, ncol=5, handletextpad=0.2,
                      columnspacing=0.9, frameon=False)
    ag.assign(rule_retention_pct=ag.rule.map(ret)).to_csv(SRC / f"Fig6_FGH_original_LSFM_{variant}.csv", index=False)


def make(variant, name):
    src = []
    fig = plt.figure(figsize=(7.2, 9.4))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.95], hspace=0.95)
    scatters(fig, outer[0], variant, src)
    gde = outer[1].subgridspec(1, 2, wspace=0.28)
    ax_d = fig.add_subplot(gde[0])
    ax_e = fig.add_subplot(gde[1], sharey=ax_d)
    panel_d(ax_d, variant, src)
    panel_e(ax_e, variant, src)
    lo = min(ax_d.get_ylim()[0], ax_e.get_ylim()[0]); hi = max(ax_d.get_ylim()[1], ax_e.get_ylim()[1])
    ax_d.set_ylim(lo, hi)                                    # D and E on one y scale
    ax_e.set_ylabel("")
    panels_fgh(fig, outer[2], variant)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    out = pd.DataFrame(src)
    out.to_csv(SRC / f"{name}_summary.csv", index=False)
    print(name)
    print(out[out.panel.isin(["D", "E"])][["panel", "rule", "retained_pct", "n_cases", "n_cases_without_registrations",
                                          "median_delta", "max_abs_delta", "frac_within_0p02"]].round(3).to_string())


if __name__ == "__main__":
    make("raw", "Fig6_v4")
    make("preprocessed", "FigS_QC_rules_preprocessed_v2")
