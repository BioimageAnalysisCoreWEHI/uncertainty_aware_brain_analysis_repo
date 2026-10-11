"""Supplementary Fig. S7: where, what and how much the global-QC rule changes downstream results in the two LSFM datasets.

Uses the eight QC rules of the sensitivity analysis (REGIMES of 22_focused_revision_figures.py, parsed with ast, not
run) and outputs that already exist. Nothing is re-registered or re-thresholded, and no rule is chosen here.

  Original LSFM (SNpc, no ground truth):
    QC_REGIME_FULLRES_AXIAL_MASKS/agreement_vs_current.csv   (script 30: full-resolution voxel Dice, cell Jaccard,
                                                              fraction of Current's cells kept, count change)
    QC_REGIME_FULLRES_AXIAL_MASKS/counts.csv                  (script 24: cell count per sample x input x rule)
  Independent LSFM (16 regions, published registration as the independent reference; not ground truth):
    results/tables/qcsens_lsfm2_per_case.csv                  (script 17 aggregate: 3-D Dice / volume vs Current's
                                                              consensus and vs the published registration)

Outputs (REVISION_OUTPUTS_FOCUSED):
  figures/FigS7_LSFM_QC_rule_downstream.{pdf,svg,png} (600 dpi)
  source_data/FigS7_A-C_original_LSFM_vs_current.csv, FigS7_original_LSFM_counts_by_rule.csv,
  FigS7_original_LSFM_ranking.csv, FigS7_D-G_independent_LSFM_per_case.csv, FigS7_independent_LSFM_region_rule.csv,
  FigS7_independent_LSFM_rule_summary.csv, FigS7_verification.csv
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
from scipy.stats import kendalltau

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PRA = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis"))
VA = PRA / "validation_analysis"
OUT = PRA / "REVISION_OUTPUTS_FOCUSED"
FIG, SRC = OUT / "figures", OUT / "source_data"
FR = PRA / "QC_REGIME_FULLRES_AXIAL_MASKS"
QS = VA / "results" / "qc_sensitivity"
RULE_ORDER = ["No global QC", "Very lenient", "Lenient", "Untuned start (median ± 3·MAD)", "Current", "Strict",
              "Very strict", "Median IoU and grad_p95"]
SHORT = {"No global QC": "No QC", "Very lenient": "V. lenient", "Lenient": "Lenient",
         "Untuned start (median ± 3·MAD)": "Untuned", "Current": "Current", "Strict": "Strict",
         "Very strict": "V. strict", "Median IoU and grad_p95": "Median"}
FOLDER = {"No global QC": "no_global_QC", "Very lenient": "very_lenient", "Lenient": "lenient",
          "Untuned start (median ± 3·MAD)": "untuned_start", "Current": "current", "Strict": "strict",
          "Very strict": "very_strict", "Median IoU and grad_p95": "median"}
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 5.8,
                     "ytick.labelsize": 5.8, "legend.fontsize": 5.8, "axes.linewidth": 0.5, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none"})


def regimes():
    tree = ast.parse((VA / "scripts" / "22_focused_revision_figures.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "REGIMES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("REGIMES not found")


VER = []


def check(name, ok, detail=""):
    VER.append(dict(check=name, ok=bool(ok), detail=str(detail)))
    if not ok:
        raise AssertionError(f"{name}: {detail}")


# ------------------------------------------------------------------------------------------------------------
# original LSFM
# ------------------------------------------------------------------------------------------------------------
def original_lsfm(reg):
    mp = pd.read_csv(SRC / "sample_number_mapping.csv")
    lab = dict(zip(mp.sample_id.astype(str), mp.label))
    ag = pd.read_csv(FR / "agreement_vs_current.csv", dtype={"sample": str})
    inv = {v: k for k, v in FOLDER.items()}
    ag["rule"] = ag.rule.map(inv)
    cn = pd.read_csv(FR / "counts.csv", dtype={"sample": str}, encoding="utf-8")
    check("original LSFM: counts.csv rule names = REGIMES names", set(cn.rule) == set(RULE_ORDER), set(cn.rule))
    # rules that leave no registration have no mask folder (script 24 records them in counts.csv with 0 cells):
    # add them explicitly as an empty consensus (Dice 0, all cells lost) so they are not silently dropped
    have = set(zip(ag["sample"], ag.input, ag.rule))
    miss = cn[(cn.rule != "Current") & ~pd.Series(list(zip(cn["sample"], cn.input, cn.rule))).isin(have).to_numpy()]
    check("original LSFM: missing agreement rows are exactly the empty consensus cases",
          (miss.n_registrations.fillna(0) == 0).all() and (miss.cell_count.fillna(0) == 0).all(),
          miss[["sample", "input", "rule", "n_registrations", "cell_count"]].to_dict("records"))
    ag = pd.concat([ag, pd.DataFrame(dict(sample=miss["sample"], input=miss.input, rule=miss.rule, voxel_dice=0.0,
                                          volume_change_pct=-100.0, count_change_pct=-100.0, cell_jaccard=0.0,
                                          frac_current_cells_kept=0.0, is_empty=True))], ignore_index=True)
    ag["is_empty"] = ag["is_empty"].fillna(False).astype(bool)
    ag["sample_label"] = ag["sample"].map(lab)
    check("original LSFM: 7 rules x 10 cases vs Current", len(ag) == 70 and ag.rule.notna().all(), len(ag))
    cn["cell_count"] = cn.cell_count.fillna(0)
    cn["sample_label"] = cn["sample"].map(lab)
    check("original LSFM: counts.csv has 5 samples x 2 inputs x 8 rules", len(cn) == 80, len(cn))
    for r in RULE_ORDER:
        sid = cn[cn.rule == r].setting_id.unique()
        check(f"original LSFM {r}: setting id = REGIMES", list(sid) == [reg[r]["paper"]], sid)
    cur = cn[cn.rule == "Current"].set_index(["sample", "input"]).cell_count
    rec = ag.set_index(["sample", "input", "rule"]).count_change_pct
    for (s, v, r), x in rec.items():
        c = cn[(cn["sample"] == s) & (cn.input == v) & (cn.rule == r)].cell_count.iloc[0]
        cc = cur[(s, v)]
        ok = (np.isnan(x) and c == 0) or np.isclose(100 * (c / cc - 1), x)
        check(f"original LSFM {s} {v} {r}: count change equals counts.csv", ok, f"{x} vs {c}/{cc}")
    cn["count_change_pct_vs_current"] = cn.apply(
        lambda q: 100 * (q.cell_count / cur[(q["sample"], q.input)] - 1), axis=1)
    rk = []
    for v in ("raw", "preprocessed"):
        c0 = cn[(cn.input == v) & (cn.rule == "Current")].set_index("sample").cell_count
        for r in RULE_ORDER:
            c1 = cn[(cn.input == v) & (cn.rule == r)].set_index("sample").cell_count.reindex(c0.index)
            tau = kendalltau(c0, c1).statistic
            rk.append(dict(input=v, rule=r, kendall_tau_vs_current=tau,
                           same_order_as_current=bool((c0.rank() == c1.rank()).all()),
                           min_count=c1.min(), max_count=c1.max(),
                           cv_across_samples_pct=100 * c1.std() / c1.mean() if c1.mean() else np.nan,
                           n_empty=int((c1 == 0).sum())))
    rk = pd.DataFrame(rk)
    ag.to_csv(SRC / "FigS7_A-C_original_LSFM_vs_current.csv", index=False)
    cn.drop(columns=["source_consensus", "cell_labels", "union_mask"]).to_csv(
        SRC / "FigS7_original_LSFM_counts_by_rule.csv", index=False)
    rk.to_csv(SRC / "FigS7_original_LSFM_ranking.csv", index=False)
    return ag, cn, rk


# ------------------------------------------------------------------------------------------------------------
# independent LSFM
# ------------------------------------------------------------------------------------------------------------
def independent_lsfm(reg):
    pc = pd.read_csv(VA / "results" / "tables" / "qcsens_lsfm2_per_case.csv")
    sid2rule = {reg[r]["lsfm2"]: r for r in RULE_ORDER}
    d = pc[pc.setting_id.isin(sid2rule)].copy()
    d["rule"] = d.setting_id.map(sid2rule)
    d = d.drop_duplicates(["sample", "variant", "region", "rule"])
    check("independent LSFM: 8 rules x 192 cases", len(d) == 8 * 192, len(d))
    d["is_empty"] = d.n_final == 0
    cur = d[d.rule == "Current"].set_index(["sample", "variant", "region"])
    check("independent LSFM: Current dice vs itself = 1", np.allclose(cur.dice_vs_chosen.dropna(), 1.0))
    check("independent LSFM: Current never empty", (~cur.is_empty).all(), int(cur.is_empty.sum()))
    key = list(zip(d["sample"], d.variant, d.region))
    d["dice_vs_ref_current"] = cur.dice_vs_ref.reindex(key).to_numpy()
    d["volume_vox_current"] = cur.volume_vox.reindex(key).to_numpy()
    d["n_final_current"] = cur.n_final.reindex(key).to_numpy()
    # an empty consensus has Dice 0 with anything non-empty; keep it as 0 so empties are not silently dropped
    for c in ("dice_vs_chosen", "dice_vs_ref"):
        d[c + "_empty0"] = np.where(d.is_empty, 0.0, d[c])
    d["delta_dice_vs_ref"] = d.dice_vs_ref_empty0 - d.dice_vs_ref_current
    d["structure"] = d.region.str.replace("_both", "", regex=False)
    keep = ["sample", "variant", "structure", "rule", "setting_id", "iou_thr", "grad_thr", "n_global", "n_final",
            "n_final_current", "is_empty", "volume_vox", "volume_vox_current", "dice_vs_chosen", "dice_vs_chosen_empty0",
            "volume_ratio_vs_chosen", "hd95_vs_chosen_um", "dice_vs_ref", "dice_vs_ref_empty0",
            "dice_vs_ref_current", "delta_dice_vs_ref", "volume_ratio_vs_ref", "set_path"]
    d = d[keep].rename(columns={"dice_vs_chosen": "dice_vs_current", "dice_vs_chosen_empty0": "dice_vs_current_empty0",
                                "volume_ratio_vs_chosen": "volume_ratio_vs_current",
                                "hd95_vs_chosen_um": "hd95_vs_current_um",
                                "dice_vs_ref": "dice_vs_published", "dice_vs_ref_empty0": "dice_vs_published_empty0",
                                "dice_vs_ref_current": "dice_vs_published_current",
                                "delta_dice_vs_ref": "delta_dice_vs_published",
                                "volume_ratio_vs_ref": "volume_ratio_vs_published"})
    d.to_csv(SRC / "FigS7_D-G_independent_LSFM_per_case.csv", index=False)
    g = d.groupby(["variant", "structure", "rule"])
    rr = g.agg(median_dice_vs_current=("dice_vs_current_empty0", "median"),
               min_dice_vs_current=("dice_vs_current_empty0", "min"),
               median_delta_dice_vs_published=("delta_dice_vs_published", "median"),
               median_dice_vs_published=("dice_vs_published_empty0", "median"),
               median_volume_ratio_vs_current=("volume_ratio_vs_current", "median"),
               n_empty=("is_empty", "sum"), n_samples=("is_empty", "size"),
               median_n_final=("n_final", "median")).reset_index()
    rr.to_csv(SRC / "FigS7_independent_LSFM_region_rule.csv", index=False)
    rs = d.groupby(["variant", "rule"]).agg(
        median_dice_vs_current=("dice_vs_current_empty0", "median"),
        q10_dice_vs_current=("dice_vs_current_empty0", lambda x: x.quantile(0.1)),
        frac_cases_dice_vs_current_ge_0p9=("dice_vs_current_empty0", lambda x: (x >= 0.9).mean()),
        median_dice_vs_published=("dice_vs_published_empty0", "median"),
        median_delta_dice_vs_published=("delta_dice_vs_published", "median"),
        frac_cases_better_vs_published=("delta_dice_vs_published", lambda x: (x > 0.01).mean()),
        frac_cases_worse_vs_published=("delta_dice_vs_published", lambda x: (x < -0.01).mean()),
        n_empty=("is_empty", "sum"), n_cases=("is_empty", "size")).reset_index()
    rs.to_csv(SRC / "FigS7_independent_LSFM_rule_summary.csv", index=False)
    return d, rr, rs


# ------------------------------------------------------------------------------------------------------------
def strip(ax, df, col, ylabel, sample_col, ylim=None, ref=None):
    labels = sorted(df[sample_col].unique())
    colr = {b: OKABE[k % 6] for k, b in enumerate(labels)}
    rules = [r for r in RULE_ORDER if r != "Current"]
    for i, r in enumerate(rules):
        for j, (v, mk) in enumerate((("raw", "o"), ("preprocessed", "^"))):
            h = df[(df.rule == r) & (df.input == v)]
            x = i + (-0.17 if j == 0 else 0.17) + np.linspace(-0.06, 0.06, len(h))
            ax.scatter(x, h[col], s=9, marker=mk, c=[colr[b] for b in h[sample_col]], lw=0.3, edgecolor="white",
                       zorder=3)
    if ref is not None:
        ax.axhline(ref, color="0.6", lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.set_xticks(range(len(rules)), [SHORT[r] for r in rules], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(rules) - 0.4)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="0.92", lw=0.4, zorder=0)
    return colr


def heat(ax, m, cmap, norm, fmt, empties=None, title=""):
    im = ax.imshow(m.to_numpy(float), cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(m.shape[1]), [SHORT[r] for r in m.columns], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(m.shape[0]), m.index)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            x = m.iat[i, j]
            e = 0 if empties is None else int(empties.iat[i, j])
            t = fmt(x) + (f"\n{e} empty" if e else "")
            rgb = im.cmap(im.norm(x))[:3] if np.isfinite(x) else (1, 1, 1)
            dark = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 0.5
            ax.text(j, i, t, ha="center", va="center", fontsize=4.3, color="white" if dark else "0.1",
                    linespacing=0.9)
    ax.set_title(title, fontsize=6, pad=3)
    return im


def figure(ag, cn, rk, d, rr, rs):
    fig = plt.figure(figsize=(7.2, 9.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.9, 1.05], hspace=0.42, left=0.08, right=0.955, top=0.945,
                          bottom=0.06)
    # ---- row 1: original LSFM
    r1 = gs[0].subgridspec(1, 3, wspace=0.38)
    axA, axB, axC = (fig.add_subplot(r1[0, k]) for k in range(3))
    ne = ag[~ag.is_empty]
    lo = min(ne.voxel_dice.min(), ne.frac_current_cells_kept.min())
    lo = np.floor(lo * 20) / 20 - 0.05
    colr = strip(axA, ne, "voxel_dice", "Voxel Dice vs Current", "sample_label", (lo, 1.01), 1.0)
    strip(axB, ne, "count_change_pct", "Cell count change vs Current (%)", "sample_label", None, 0.0)
    strip(axC, ne, "frac_current_cells_kept", "Fraction of Current's cells kept", "sample_label", (lo, 1.01), 1.0)
    axB.set_ylim(np.floor(ne.count_change_pct.min() / 5) * 5 - 5, np.ceil(ne.count_change_pct.max() / 5) * 5 + 2)
    for ax, t in ((axA, "A"), (axB, "B"), (axC, "C")):
        ax.text(-0.3, 1.08, t, transform=ax.transAxes, fontsize=8, fontweight="bold")
    fig.text(0.01, 0.985, "Original LSFM (SNpc, full-resolution consensus masks and cellpose cells; "
                        "compared with the Current rule)", fontsize=6.8, fontweight="bold", va="top")
    for _, e in ag[ag.is_empty].iterrows():                               # plotted off-scale, annotated
        i = [r for r in RULE_ORDER if r != "Current"].index(e.rule)
        for ax in (axA, axB, axC):
            yl = ax.get_ylim()
            ax.annotate(f"{e.sample_label} {e.input}:\nno registration\npasses (empty)",
                        (i + (0.17 if e.input == "preprocessed" else -0.17), yl[0]), fontsize=4.2, ha="center",
                        va="bottom", xytext=(0, 2), textcoords="offset points", color="0.3", annotation_clip=False)
    h = [Line2D([], [], ls="", marker="o", ms=3.2, color=colr[b], label=b) for b in colr]
    h += [Line2D([], [], ls="", marker="o", ms=3.2, color="0.4", label="raw"),
          Line2D([], [], ls="", marker="^", ms=3.2, color="0.4", label="preprocessed")]
    axC.legend(handles=h, loc="lower left", fontsize=5, ncol=2, handletextpad=0.2, columnspacing=0.6,
               bbox_to_anchor=(0.0, 0.17))
    # ---- row 2: independent LSFM heatmaps (raw)
    r2 = gs[1].subgridspec(1, 2, wspace=0.32)
    axD, axE = fig.add_subplot(r2[0, 0]), fig.add_subplot(r2[0, 1])
    v = "raw"
    q = rr[rr.variant == v]
    size = d[(d.variant == v) & (d.rule == "Current")].groupby("structure").volume_vox_current.median()
    order = size.sort_values(ascending=False).index.tolist()
    cols = [r for r in RULE_ORDER if r != "Current"]
    md = q.pivot(index="structure", columns="rule", values="median_dice_vs_current").reindex(order)[cols]
    ne = q.pivot(index="structure", columns="rule", values="n_empty").reindex(order)[cols]
    imD = heat(axD, md, plt.get_cmap("Purples"), plt.Normalize(0.4, 1.0), lambda x: f"{x:.2f}", ne,
               "Median 3-D Dice of consensus vs Current's (raw)")
    cols2 = RULE_ORDER
    dd = q.pivot(index="structure", columns="rule", values="median_delta_dice_vs_published").reindex(order)[cols2]
    ne2 = q.pivot(index="structure", columns="rule", values="n_empty").reindex(order)[cols2]
    imE = heat(axE, dd, plt.get_cmap("RdBu"), TwoSlopeNorm(0, -0.3, 0.3), lambda x: f"{x:+.2f}", ne2,
               "Median change in Dice vs published registration (rule − Current, raw)")
    cb = fig.colorbar(imD, ax=axD, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=5)
    cb = fig.colorbar(imE, ax=axE, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=5)
    axD.set_ylabel("Region (largest to smallest consensus volume)")
    for ax, t in ((axD, "D"), (axE, "E")):
        ax.text(-0.16, 1.03, t, transform=ax.transAxes, fontsize=8, fontweight="bold")
    axD.text(-0.16, 1.08, "Independent LSFM (16 regions × 6 brains; published registration = independent reference, "
                          "not ground truth)", transform=axD.transAxes, fontsize=6.8, fontweight="bold")
    # ---- row 3: independent LSFM summaries
    r3 = gs[2].subgridspec(1, 3, wspace=0.42)
    axF, axG, axH = (fig.add_subplot(r3[0, k]) for k in range(3))
    for j, (vv, mk, off) in enumerate((("raw", "o", -0.17), ("preprocessed", "^", 0.17))):
        for i, r in enumerate(RULE_ORDER):
            x = d[(d.variant == vv) & (d.rule == r)].dice_vs_published_empty0
            bp = axF.boxplot(x, positions=[i + off], widths=0.28, showfliers=False, patch_artist=True,
                             medianprops=dict(color="black", lw=0.7), whiskerprops=dict(lw=0.5),
                             capprops=dict(lw=0.5), boxprops=dict(lw=0.5))
            bp["boxes"][0].set_facecolor("#cbc9e2" if vv == "raw" else "#6a51a3")
    axF.set_xticks(range(8), [SHORT[r] for r in RULE_ORDER], rotation=45, ha="right", rotation_mode="anchor")
    axF.set_ylabel("Dice vs published registration\n(all regions; empty = 0)")
    axF.grid(axis="y", color="0.92", lw=0.4, zorder=0)
    axF.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="#cbc9e2", lw=0.5, ec="k", label="raw"),
                        plt.Rectangle((0, 0), 1, 1, fc="#6a51a3", lw=0.5, ec="k", label="preprocessed")],
               loc="lower left", fontsize=5, ncol=2, bbox_to_anchor=(0.0, 0.99))
    # G: SNc per brain
    labels = sorted(d["sample"].unique())
    colr2 = {b: OKABE[k] for k, b in enumerate(labels)}
    for vv, mk, ls in (("raw", "o", "-"), ("preprocessed", "^", (0, (2, 1.5)))):
        for b in labels:
            h_ = d[(d.variant == vv) & (d.structure == "SNc") & (d["sample"] == b)].set_index("rule").reindex(
                RULE_ORDER)
            axG.plot(range(8), h_.dice_vs_published_empty0, ls=ls, marker=mk, ms=2.5, lw=0.6, color=colr2[b])
    axG.set_xticks(range(8), [SHORT[r] for r in RULE_ORDER], rotation=45, ha="right", rotation_mode="anchor")
    axG.set_ylabel("SNc Dice vs published registration")
    axG.grid(axis="y", color="0.92", lw=0.4, zorder=0)
    hh = [Line2D([], [], color=colr2[b], lw=0.8, label=b) for b in labels]
    hh += [Line2D([], [], color="0.4", lw=0.6, marker="o", ms=2.5, label="raw"),
           Line2D([], [], color="0.4", lw=0.6, ls=(0, (2, 1.5)), marker="^", ms=2.5, label="preprocessed")]
    axG.legend(handles=hh, loc="center left", fontsize=4.6, ncol=2, handletextpad=0.3, columnspacing=0.6,
               bbox_to_anchor=(0.0, 0.66))
    axG.set_ylim(-0.02, max(0.6, axG.get_ylim()[1]))
    # H: % cases empty / % cases with Dice vs Current >= 0.9
    for vv, mk, c in (("raw", "o", "#cbc9e2"), ("preprocessed", "^", "#6a51a3")):
        s_ = rs[rs.variant == vv].set_index("rule").reindex(RULE_ORDER)
        axH.plot(range(8), 100 * s_.frac_cases_dice_vs_current_ge_0p9, marker=mk, ms=3, lw=0.8, color=c,
                 mec="k", mew=0.3, label=f"Dice vs Current ≥ 0.9 ({vv})")
        axH.plot(range(8), 100 * s_.n_empty / s_.n_cases, marker=mk, ms=3, lw=0.8, ls=(0, (2, 1.5)), color=c,
                 mec="k", mew=0.3, label=f"empty consensus ({vv})")
    axH.set_xticks(range(8), [SHORT[r] for r in RULE_ORDER], rotation=45, ha="right", rotation_mode="anchor")
    axH.set_ylabel("% of region × brain cases")
    axH.set_ylim(-3, 103)
    axH.grid(axis="y", color="0.92", lw=0.4, zorder=0)
    axH.legend(loc="center left", fontsize=4.6, bbox_to_anchor=(0.0, 0.27))
    for ax, t in ((axF, "F"), (axG, "G"), (axH, "H")):
        ax.text(-0.3, 1.05, t, transform=ax.transAxes, fontsize=8, fontweight="bold")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"FigS7_LSFM_QC_rule_downstream.{ext}", dpi=600, facecolor="white")
    plt.close(fig)


def main():
    reg = regimes()
    check("REGIMES has lsfm2 ids for all 8 rules", all("lsfm2" in reg[r] for r in RULE_ORDER))
    ag, cn, rk = original_lsfm(reg)
    d, rr, rs = independent_lsfm(reg)
    figure(ag, cn, rk, d, rr, rs)
    pd.DataFrame(VER).to_csv(SRC / "FigS7_verification.csv", index=False)
    pd.set_option("display.width", 250)
    print("ORIGINAL LSFM vs Current\n", ag.groupby("rule")[["voxel_dice", "count_change_pct",
                                                            "frac_current_cells_kept"]].agg(["median", "min", "max"])
          .round(3).to_string())
    print(rk.round(3).to_string(index=False))
    print("INDEPENDENT LSFM\n", rs.round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
