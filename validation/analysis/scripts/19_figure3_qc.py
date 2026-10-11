"""Figure 3: global-QC landscape of the three datasets (A-C) and aMAP QC-threshold sensitivity (D).

Uses completed outputs only (no registration, QC or sensitivity analysis is rerun):
  A-C  per-registration tissue IoU and grad_p95: post_review_analysis/dataset_qc_comparison/*.csv
       (via 12_qc_threshold_procedure.load_all); reference lines = the QC thresholds in each dataset's pipeline
       config (original_dataset/config, config, validation_datasets/lightsheet_data/config).
  D    aMAP QC-threshold sensitivity (17_qc_sensitivity_consensus.py): results/tables/qcsens_amap_per_case.csv and
       results/qc_sensitivity/amap/settings.csv. Only the predefined named regimes are shown (not the grid):
       no global QC, median -/+ 3 MAD starting point, Tukey 1.5 IQR, current (manuscript), median. SSIM top-50 %
       selection and the 0.5 consensus are applied identically in every regime. Outcome = consensus Dice vs the
       expert STAPLE (aMAP protocol: best-matching section within the raters' stack), raw input; DG-sg excluded as
       elsewhere, SNc has no expert annotation. Preprocessed input is reported in the CSV and a supplementary panel.

Outputs: figures/main/Fig3_qc.{pdf,svg,png}, figures/main/Fig3{A,B,C,D}.{pdf,svg,png},
         figures/supplementary/FigS_Fig3D_preprocessed.{pdf,png}, results/tables/fig3_amap_qc_sensitivity_summary.csv,
         source_data/Fig3_*.csv
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import FIG_MAIN, FIG_SUPP, SRC, STRUCTS_MAIN, TABLES, VA, log, save_table  # noqa: E402

R = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis"))
CONFIGS = {"paper LSFM": R / "original_dataset" / "config" / "config.yaml",
           "aMAP": R / "config" / "config.yaml",
           "validation LSFM": R / "validation_datasets" / "lightsheet_data" / "config" / "config.yaml"}
TITLES = {"paper LSFM": "Original LSFM", "aMAP": "aMAP (serial two-photon)", "validation LSFM": "Independent LSFM"}
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
MARK = {"raw": "o", "preprocessed": "^"}
# predefined regimes, ordered from least to most stringent (display names; definitions come from settings.csv)
REGIMES = [("no global QC", "No global QC"), ("starting point (median ± 3·MAD)", "Median ± 3·MAD"),
           ("Tukey 1.5·IQR", "Tukey 1.5·IQR"), ("chosen", "Current (manuscript)"), ("median", "Median")]
RCOL = ["#BDBDBD", "#9ecae1", "#4292c6", "#2171b5", "#08306b"]  # sequential blue: ordered stringency, no "winner"


def _load(name, fname):
    s = importlib.util.spec_from_file_location(name, str(VA / "scripts" / fname))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def style():
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})


# ------------------------------------------------------------------------------------------------------------
def scatter(ax, d, thr, ylim, show_marker_legend):
    brains = sorted(d["sample"].astype(str).unique())
    for v in ("raw", "preprocessed"):
        for k, b in enumerate(brains):
            g = d[(d.variant == v) & (d["sample"].astype(str) == b)]
            ax.scatter(g.iou, g.grad_p95, s=7, marker=MARK[v], color=OKABE[k], alpha=0.75, lw=0)
    ax.axvline(thr[0], color="0.25", lw=0.7, ls="--", zorder=0)
    ax.axhline(thr[1], color="0.25", lw=0.7, ls="--", zorder=0)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Tissue–atlas IoU")
    ax.set_ylabel("grad_p95")
    h = [Line2D([], [], ls="", marker="o", ms=3.5, color=OKABE[k], label=b) for k, b in enumerate(brains)]
    if show_marker_legend:
        h += [Line2D([], [], ls="", marker=MARK[v], ms=3.5, color="0.35", label=v) for v in ("raw", "preprocessed")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.27), fontsize=5.5, handletextpad=0.1,
              labelspacing=0.2, columnspacing=0.8, ncol=2)


def regimes(settings, per):
    """-> regime table: definition + registrations retained by global QC (raw and preprocessed)."""
    rows = []
    reg = per.drop_duplicates(["sample", "variant", "setting_id"])  # n_global is per brain x input
    for lab, name in REGIMES:
        s = settings[settings.label == lab].iloc[0]
        g = reg[reg.setting_id == s.setting_id]
        defn = ("IoU and grad_p95 not applied" if pd.isna(s.iou_thr)
                else f"IoU > {s.iou_thr:g} and grad_p95 < {s.grad_thr:g}")
        r = dict(regime=name, setting_id=s.setting_id, definition=defn, iou_thr=s.iou_thr, grad_thr=s.grad_thr)
        for v in ("raw", "preprocessed"):
            gv = g[g.variant == v]
            n, tot = int(gv.n_global.sum()), 64 * gv["sample"].nunique()
            r.update({f"retained_{v}": n, f"total_{v}": tot, f"retained_pct_{v}": 100 * n / tot,
                      f"min_retained_per_brain_{v}": int(gv.n_global.min())})
        rows.append(r)
    return pd.DataFrame(rows)


def sensitivity_panel(ax, per, reg, variant, legend=True):
    d = per[(per.variant == variant) & per.structure.isin(STRUCTS_MAIN)]
    structs = sorted(STRUCTS_MAIN)
    off = np.linspace(-0.3, 0.3, len(REGIMES))
    for k, ((lab, name), col) in enumerate(zip(REGIMES, RCOL)):
        sid = reg.setting_id[reg.regime == name].iloc[0]
        g = d[d.setting_id == sid]
        for j, s in enumerate(structs):
            y = g[g.structure == s].dice_paper.dropna()
            ax.scatter(np.full(len(y), j + off[k]), y, s=5, color=col, alpha=0.45, lw=0, zorder=2)
            ax.plot(j + off[k], y.median(), marker="_", ms=7, mew=1.6, color=col, zorder=3)
        r = reg[reg.regime == name].iloc[0]
        pct = r[f"retained_pct_{variant}"]
        ax.plot([], [], ls="", marker="s", ms=4, color=col,
                label=f"{name} ({r.definition.replace('IoU and grad_p95 not applied', 'thresholds not applied')}; "
                      f"{pct:.0f} % retained)")
    ax.set_xticks(range(len(structs)), structs)
    ax.set_xlim(-0.6, len(structs) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Consensus Dice vs expert STAPLE")
    for j in range(1, len(structs)):
        ax.axvline(j - 0.5, color="0.92", lw=0.5, zorder=0)
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.1), fontsize=6, handletextpad=0.2, labelspacing=0.25,
                  ncol=2, columnspacing=1.2)


# ------------------------------------------------------------------------------------------------------------
def summary(per, reg):
    rows = []
    cur = reg.setting_id[reg.regime == "Current (manuscript)"].iloc[0]
    for v in ("raw", "preprocessed"):
        d = per[(per.variant == v) & per.structure.isin(STRUCTS_MAIN)]
        for s in sorted(STRUCTS_MAIN):
            ds_ = d[d.structure == s]
            med = {r.regime: ds_[ds_.setting_id == r.setting_id].dice_paper.median() for r in reg.itertuples()}
            case = ds_.pivot_table(index="brain", columns="setting_id", values="dice_paper")
            case_dev = case[[r.setting_id for r in reg.itertuples()]].sub(case[cur], axis=0).abs()
            core = [n for _, n in REGIMES if n != "Median"]      # the four regimes retaining most registrations
            rng = lambda names: max(med[n] for n in names) - min(med[n] for n in names)
            mx = lambda names: max(abs(med[n] - med["Current (manuscript)"]) for n in names)
            core_ids = reg.setting_id[reg.regime.isin(core)]
            for r in reg.itertuples():
                g = ds_[ds_.setting_id == r.setting_id]
                rows.append(dict(
                    variant=v, structure=s, regime=r.regime, definition=r.definition,
                    retained_pct=getattr(r, f"retained_pct_{v}"),
                    n_brains_with_consensus=int(g.dice_paper.notna().sum()), n_brains=int(len(g)),
                    median_n_final_in_consensus=g.n_final.median(),
                    dice_median_over_brains=med[r.regime],
                    abs_diff_from_current_median=abs(med[r.regime] - med["Current (manuscript)"]),
                    max_abs_diff_from_current_any_brain=float(np.nanmax(case_dev[r.setting_id].values)),
                    structure_dice_range_all5=rng(med), structure_max_abs_diff_from_current_all5=mx(med),
                    structure_dice_range_core4=rng(core), structure_max_abs_diff_from_current_core4=mx(core),
                    structure_max_abs_diff_any_brain_all5=float(np.nanmax(case_dev.values)),
                    structure_max_abs_diff_any_brain_core4=float(np.nanmax(case_dev[list(core_ids)].values))))
    return pd.DataFrame(rows)


def main():
    style()
    qc = _load("qc12", "12_qc_threshold_procedure.py")
    allqc = qc.load_all()
    thr = {ds: tuple(yaml.safe_load(p.read_text())["qc"][k] for k in ("iou_threshold", "grad_p95_threshold"))
           for ds, p in CONFIGS.items()}
    settings = pd.read_csv(VA / "results" / "qc_sensitivity" / "amap" / "settings.csv")
    per = pd.read_csv(TABLES / "qcsens_amap_per_case.csv")
    reg = regimes(settings, per)
    ylim = (0.028, allqc.grad_p95.max() * 1.04)

    # ---------------- assembled figure ----------------
    fig = plt.figure(figsize=(6.85, 6.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.78, wspace=0.32)
    order = ["paper LSFM", "aMAP", "validation LSFM"]
    for k, ds in enumerate(order):
        ax = fig.add_subplot(gs[0, k])
        scatter(ax, allqc[allqc.dataset == ds], thr[ds], ylim, show_marker_legend=(k == 0))
        ax.set_title(TITLES[ds], fontsize=7, pad=3)
        if k:
            ax.set_ylabel("")
        ax.text(-0.22 if k == 0 else -0.12, 1.06, "ABC"[k], transform=ax.transAxes, fontsize=9, fontweight="bold")
    axd = fig.add_subplot(gs[1, :])
    sensitivity_panel(axd, per, reg, "raw")
    axd.text(-0.07, 1.04, "D", transform=axd.transAxes, fontsize=9, fontweight="bold")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG_MAIN / f"Fig3_qc.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # ---------------- individual panels ----------------
    for k, ds in enumerate(order):
        f, ax = plt.subplots(figsize=(2.3, 2.0))
        scatter(ax, allqc[allqc.dataset == ds], thr[ds], ylim, show_marker_legend=True)
        ax.set_title(TITLES[ds], fontsize=7, pad=3)
        for ext in ("pdf", "svg", "png"):
            f.savefig(FIG_MAIN / f"Fig3{'ABC'[k]}.{ext}", dpi=600, bbox_inches="tight")
        plt.close(f)
    f, ax = plt.subplots(figsize=(6.85, 2.6))
    sensitivity_panel(ax, per, reg, "raw")
    for ext in ("pdf", "svg", "png"):
        f.savefig(FIG_MAIN / f"Fig3D.{ext}", dpi=600, bbox_inches="tight")
    plt.close(f)
    f, ax = plt.subplots(figsize=(6.85, 2.6))
    sensitivity_panel(ax, per, reg, "preprocessed")
    ax.set_title("aMAP, preprocessed input", fontsize=7)
    for ext in ("pdf", "png"):
        f.savefig(FIG_SUPP / f"FigS_Fig3D_preprocessed.{ext}", dpi=600, bbox_inches="tight")
    plt.close(f)

    # ---------------- tables ----------------
    summ = summary(per, reg)
    save_table(summ, "fig3_amap_qc_sensitivity_summary")
    save_table(reg, "fig3_amap_qc_regimes")
    allqc[allqc.dataset.isin(order)][["dataset", "variant", "sample", "bending_energy", "grid", "iou", "grad_p95"]] \
        .to_csv(SRC / "Fig3_ABC_qc_scatter.csv", index=False)
    pd.DataFrame([dict(dataset=ds, iou_threshold=t[0], grad_p95_threshold=t[1], config=str(CONFIGS[ds]))
                  for ds, t in thr.items()]).to_csv(SRC / "Fig3_ABC_thresholds.csv", index=False)
    per[per.setting_id.isin(reg.setting_id) & per.structure.isin(STRUCTS_MAIN)][
        ["brain", "structure", "variant", "setting_id", "n_global", "n_final", "dice_paper"]].merge(
        reg[["setting_id", "regime", "definition"]], on="setting_id").to_csv(SRC / "Fig3_D_dice.csv", index=False)

    # ---------------- console summary ----------------
    pd.set_option("display.width", 220)
    log(reg.round(1).to_string(index=False))
    for v in ("raw", "preprocessed"):
        s1 = summ[summ.variant == v].drop_duplicates("structure")
        for sc in ("all5", "core4"):
            c = f"structure_dice_range_{sc}"
            log(f"\n{v} [{sc}]: per-structure Dice range (medians over brains): median {s1[c].median():.3f}, max "
                f"{s1[c].max():.3f} ({s1.loc[s1[c].idxmax(), 'structure']}); median max|Δ| from current "
                f"{s1[f'structure_max_abs_diff_from_current_{sc}'].median():.3f}; largest brain-level |Δ| "
                f"{s1[f'structure_max_abs_diff_any_brain_{sc}'].max():.3f}")
        log(s1.sort_values("structure_dice_range_all5", ascending=False)[
            ["structure", "structure_dice_range_all5", "structure_dice_range_core4",
             "structure_max_abs_diff_any_brain_all5", "structure_max_abs_diff_any_brain_core4"]].round(3)
            .to_string(index=False))
        nb = summ[(summ.variant == v) & (summ.regime == "Median")]
        log(f"{v}: 'Median' regime - brains with a consensus per structure: {nb.n_brains_with_consensus.min()}-"
            f"{nb.n_brains_with_consensus.max()} of {nb.n_brains.max()}")
        t = summ[summ.variant == v].pivot_table(index="structure", columns="regime", values="dice_median_over_brains")
        log(t[[n for _, n in REGIMES]].round(3).to_string())


if __name__ == "__main__":
    main()
