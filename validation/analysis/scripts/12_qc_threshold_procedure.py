"""Global-QC threshold procedure across datasets (plan 2026-09-30), comparing untuned defaults.

Recommended starting point ("start", qc_start.py): median -/+ 3 MAD on the raw sweep, applied to both inputs
(IoU first, then grad_p95 among registrations passing the IoU start).
Earlier alternative, kept for comparison: Tukey's fences (conventional constant 1.5, nothing tuned):
  IoU default  = Q1 - 1.5 IQR of IoU;
  grad default = Q3 + 1.5 IQR of grad_p95 among registrations passing the IoU default.

Modes for computing the default:
  start        - median -/+ 3 MAD, set on raw and applied unchanged to preprocessed;
  pooled       - one default per dataset from raw + preprocessed registrations together, applied to both;
  separate     - a default computed on raw and a separate one on preprocessed registrations;
  raw_anchored - the default computed on raw registrations only and applied unchanged to preprocessed ones
                 (the user's working practice: calibrate on raw, then see whether preprocessing helps QC).

Chosen thresholds (visual calibration with 04b), one pair per dataset applied to both inputs:
  paper LSFM 0.9/0.045, aMAP 0.9/0.058, validation LSFM 0.8/0.08.

Per mode: summary + per-brain tables, downstream checks (aMAP consensus Dice vs expert STAPLE; paper LSFM SNpc
counts + ranking) at default vs chosen, standard-space table, and a figure. The raw_anchored mode adds the
preprocessing effect on QC pass rates (same thresholds for both inputs).

Inputs (read-only): post_review_analysis/dataset_qc_comparison/*.csv.
Outputs: results/tables/qcproc_<mode>_*.csv, qcproc_modes_comparison.csv, qcproc_preprocessing_effect.csv,
figures/main/Fig9_qc_calibration_<mode>.{pdf,png}, source_data/Fig9_<mode>_*.csv.
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
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIG_MAIN, SRC, STRUCTS_MAIN, VA, amap_brain, log, save_table  # noqa: E402
from qc_start import start_thresholds  # noqa: E402

QCDIR = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\dataset_qc_comparison"))
CHOSEN = {"paper LSFM": (0.9, 0.045), "aMAP": (0.9, 0.058), "validation LSFM": (0.8, 0.08)}
DATASETS = ["paper LSFM", "aMAP", "validation LSFM"]
VARIANTS = ["raw", "preprocessed"]
MODES = ["start", "pooled", "separate", "raw_anchored"]
MODE_LABEL = {"start": "median ± 3·MAD, set on raw, applied to preprocessed",
              "pooled": "Tukey, pooled raw + preprocessed", "separate": "Tukey, separate raw / preprocessed",
              "raw_anchored": "Tukey, set on raw, applied to preprocessed"}
MIN_PER_BRAIN = 10
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]


def _load(name, fname):
    s = importlib.util.spec_from_file_location(name, str(VA / "scripts" / fname))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def load_all() -> pd.DataFrame:
    """One long table: dataset, variant, sample, bending_energy, grid, iou, grad_p95, own pass flag."""
    frames = []
    o = pd.read_csv(QCDIR / "original_lsfm_full_final_results_with_ssim.csv")
    for v in VARIANTS:
        frames.append(pd.DataFrame(dict(dataset="paper LSFM", variant=v, sample=o["Sample ID"].astype(str),
                                        bending_energy=o.bending, grid=o.grid, iou=o[f"{v}_iou"],
                                        grad_p95=o[f"{v}_grad_p95"], grad_p99=o[f"{v}_grad_p99"],
                                        own_pass=o[f"{v}_pass_final"].astype(bool))))
    for ds, pre in (("aMAP", "amap_qc_results_"), ("validation LSFM", "validation_lsfm_qc_results_")):
        for v in VARIANTS:
            d = pd.read_csv(QCDIR / f"{pre}{v}.csv")
            smp = d.sample_id.map(amap_brain) if ds == "aMAP" else d.sample_id.astype(str)
            frames.append(pd.DataFrame(dict(dataset=ds, variant=v, sample=smp,
                                            bending_energy=d.bending_energy_weight, grid=d.grid_spacing, iou=d.iou,
                                            grad_p95=d.grad_p95, grad_p99=d.grad_p99,
                                            own_pass=d.passes_global_qc.astype(bool))))
    return pd.concat(frames, ignore_index=True)


def tukey(d: pd.DataFrame):
    q1, q3 = np.percentile(d.iou, [25, 75])
    t_iou = q1 - 1.5 * (q3 - q1)
    g = d[d.iou >= t_iou].grad_p95
    g1, g3 = np.percentile(g, [25, 75])
    return round(float(t_iou), 4), round(float(g3 + 1.5 * (g3 - g1)), 4)


def passes(d, t_iou, t_grad):
    """Pipeline convention: IoU > t and grad_p95 < t."""
    return (d.iou > t_iou) & (d.grad_p95 < t_grad)


def defaults(allqc: pd.DataFrame, mode: str) -> dict:
    """{(dataset, variant): (iou, grad)} Tukey default for the given mode."""
    out = {}
    for ds in DATASETS:
        d = allqc[allqc.dataset == ds]
        for v in VARIANTS:
            if mode == "pooled":
                out[(ds, v)] = tukey(d)
            elif mode == "separate":
                out[(ds, v)] = tukey(d[d.variant == v])
            elif mode == "start":  # recommended untuned starting point (qc_start.py)
                r = d[d.variant == "raw"]
                out[(ds, v)] = start_thresholds(r.iou, r.grad_p95)
            else:  # raw_anchored
                out[(ds, v)] = tukey(d[d.variant == "raw"])
    return out


# ------------------------------------------------------------------------------------------------------------
# downstream checks (computed once for every distinct threshold pair)
# ------------------------------------------------------------------------------------------------------------
def amap_downstream(pairs: dict) -> pd.DataFrame:
    """pairs: {variant: {(iou, grad), ...}} -> consensus Dice vs expert STAPLE (aMAP protocol) per case."""
    m3 = _load("m3", "03_amap_metrics.py")
    from common import BRAINS, case_path
    qc = m3.load_qc()
    rows = []
    for v in VARIANTS:
        for b in BRAINS:
            q = qc[(qc.brain == b) & (qc.variant == v)]
            for s in STRUCTS_MAIN:
                if not case_path(b, s, v).exists():
                    continue
                cs = m3.Case(b, s, v)
                for ti, tg in sorted(pairs[v]):
                    sel = m3.qc_subset(q, s, ti, tg, 0.5)
                    dice = cs.score(cs.consensus(sel, 0.5), full=False)["dice_paper"] if sel else np.nan
                    rows.append(dict(variant=v, brain=b, structure=s, iou_thr=ti, grad_thr=tg, n_selected=len(sel),
                                     dice=dice))
        log(f"  aMAP downstream {v}: {len(pairs[v])} threshold pairs")
    return pd.DataFrame(rows)


def lsfm_downstream(pairs: dict) -> pd.DataFrame:
    """pairs: {variant: {(iou, grad), ...}} -> SNpc consensus count (overlap rule, SSIM top 50 %, threshold 0.5)."""
    m7 = _load("m7", "07_lsfm_analysis.py")
    runs = pd.read_csv(VA / "cache" / "lsfm" / "runs.csv")
    runs["sample"] = runs["sample"].astype(str)
    rows = []
    for smp in sorted(runs["sample"].unique()):
        for v in VARIANTS:
            c = m7.Counter(smp, v)
            sub = runs[(runs["sample"] == smp) & (runs.status == v)]
            for ti, tg in sorted(pairs[v]):
                sel = m7.select(sub, "passed_qc", "top_50pct", ti, tg)
                rows.append(dict(variant=v, sample=smp, iou_thr=ti, grad_thr=tg, n_selected=len(sel),
                                 count=c.count(sel, 0.5) if sel else np.nan))
    return pd.DataFrame(rows)


def downstream_summary(mode_def, ad, ld) -> pd.DataFrame:
    rows = []
    for v in VARIANTS:
        ci, cg = CHOSEN["aMAP"]
        ti, tg = mode_def[("aMAP", v)]
        a = ad[ad.variant == v]
        ch = a[(a.iou_thr == ci) & (a.grad_thr == cg)].set_index(["brain", "structure"]).dice
        tu = a[(a.iou_thr == ti) & (a.grad_thr == tg)].set_index(["brain", "structure"]).dice
        rows.append(dict(dataset="aMAP", variant=v, metric="consensus Dice vs expert STAPLE",
                         chosen=ch.median(), default=tu.median(), median_abs_change=(ch - tu).abs().median(),
                         max_abs_change=(ch - tu).abs().max()))
        ci, cg = CHOSEN["paper LSFM"]
        ti, tg = mode_def[("paper LSFM", v)]
        l = ld[ld.variant == v]
        ch = l[(l.iou_thr == ci) & (l.grad_thr == cg)].set_index("sample")["count"]
        tu = l[(l.iou_thr == ti) & (l.grad_thr == tg)].set_index("sample")["count"].reindex(ch.index)
        rel = (tu / ch - 1).abs() * 100
        rows.append(dict(dataset="paper LSFM", variant=v, metric="SNpc consensus count", chosen=ch.median(),
                         default=tu.median(), median_abs_change_pct=rel.median(), max_abs_change_pct=rel.max(),
                         ranking_kendall_tau=stats.kendalltau(ch, tu).statistic))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------
def summarise(allqc, mode_def, mode):
    rows, per_brain = [], []
    for ds in DATASETS:
        for v in VARIANTS:
            d = allqc[(allqc.dataset == ds) & (allqc.variant == v)]
            ci, cg = CHOSEN[ds]
            ti, tg = mode_def[(ds, v)]
            p_c, p_t = passes(d, ci, cg), passes(d, ti, tg)
            rows.append(dict(
                mode=mode, dataset=ds, variant=v, n_registrations=len(d), n_brains=d["sample"].nunique(),
                chosen_iou=ci, chosen_grad=cg, default_iou=ti, default_grad=tg,
                chosen_stricter_iou=ci >= ti, chosen_stricter_grad=cg <= tg,
                pass_chosen=int(p_c.sum()), pass_default=int(p_t.sum()),
                min_per_brain_chosen=int(p_c.groupby(d["sample"]).sum().min()),
                min_per_brain_default=int(p_t.groupby(d["sample"]).sum().min()),
                own_pass_flag_reproduced=float((p_c == d.own_pass).mean()),
                chosen_iou_percentile=stats.percentileofscore(d.iou, ci),
                chosen_grad_percentile=stats.percentileofscore(d[d.iou > ci].grad_p95, cg),
                default_iou_percentile=stats.percentileofscore(d.iou, ti),
                default_grad_percentile=stats.percentileofscore(d[d.iou > ti].grad_p95, tg),
                chosen_iou_over_median=ci / d.iou.median(), chosen_grad_over_median=cg / d.grad_p95.median(),
                stiff_share_of_iou_rejected_chosen=float(((d.iou <= ci) & (d.bending_energy >= 0.99)).sum()
                                                         / max((d.iou <= ci).sum(), 1))))
            for smp, g in d.groupby("sample"):
                per_brain.append(dict(mode=mode, dataset=ds, variant=v, sample=smp, n=len(g),
                                      pass_chosen=int(passes(g, ci, cg).sum()),
                                      pass_default=int(passes(g, ti, tg).sum()),
                                      iou_median=g.iou.median(), grad_median=g.grad_p95.median()))
    pb = pd.DataFrame(per_brain)
    pb["below_min_chosen"] = pb.pass_chosen < MIN_PER_BRAIN
    pb["below_min_default"] = pb.pass_default < MIN_PER_BRAIN
    return pd.DataFrame(rows), pb


def preprocessing_effect(allqc, mode_def) -> pd.DataFrame:
    """Same thresholds for both inputs (raw-anchored default and chosen): per-brain pass counts raw vs preprocessed."""
    rows = []
    for ds in DATASETS:
        d = allqc[allqc.dataset == ds]
        for label, (ti, tg) in (("default", mode_def[(ds, "raw")]), ("chosen", CHOSEN[ds])):
            pr = passes(d, ti, tg).groupby([d["sample"], d.variant]).sum().unstack()
            nb = d.groupby([d["sample"], d.variant]).size().unstack()
            frac = pr / nb
            diff = frac["preprocessed"] - frac["raw"]
            w = stats.wilcoxon(diff).pvalue if (diff != 0).sum() >= 5 else np.nan
            rows.append(dict(dataset=ds, thresholds=label, iou_thr=ti, grad_thr=tg,
                             pass_frac_raw=float(frac["raw"].median()),
                             pass_frac_preprocessed=float(frac["preprocessed"].median()),
                             median_change_pct_points=float(diff.median() * 100),
                             brains_improved=int((diff > 0).sum()), brains_worse=int((diff < 0).sum()),
                             n_brains=len(diff), wilcoxon_p=w,
                             iou_median_raw=float(d[d.variant == "raw"].iou.median()),
                             iou_median_pre=float(d[d.variant == "preprocessed"].iou.median()),
                             grad_median_raw=float(d[d.variant == "raw"].grad_p95.median()),
                             grad_median_pre=float(d[d.variant == "preprocessed"].grad_p95.median())))
    return pd.DataFrame(rows)


def figure(allqc, summ, dsum, mode):
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.2, 7.0))
    gsp = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.95], hspace=0.7, wspace=0.45)
    src_rows = []
    for j, ds in enumerate(DATASETS):
        for i, v in enumerate(VARIANTS):
            ax = fig.add_subplot(gsp[i, j])
            d = allqc[(allqc.dataset == ds) & (allqc.variant == v)]
            s = summ[(summ.dataset == ds) & (summ.variant == v)].iloc[0]
            for k, (smp, g) in enumerate(d.groupby("sample")):
                ok = passes(g, s.chosen_iou, s.chosen_grad)
                ax.scatter(g.iou[ok], g.grad_p95[ok], s=6, color=OI[k % len(OI)], lw=0, label=smp)
                ax.scatter(g.iou[~ok], g.grad_p95[~ok], s=6, facecolor="none", edgecolor=OI[k % len(OI)], lw=0.5)
            ax.axvline(s.default_iou, color="0.5", ls="--", lw=0.8)
            ax.axhline(s.default_grad, color="0.5", ls="--", lw=0.8)
            ax.axvline(s.chosen_iou, color="k", lw=0.9)
            ax.axhline(s.chosen_grad, color="k", lw=0.9)
            ax.set_title(f"{ds} ({v})\nchosen {s.chosen_iou:g} / {s.chosen_grad:g}; default {s.default_iou:.3f} / "
                         f"{s.default_grad:.3f}", fontsize=6.3)
            ax.set_xlabel("Tissue IoU")
            if j == 0:
                ax.set_ylabel("grad_p95")
            if i == 0:
                ax.legend(frameon=False, fontsize=4.5, loc="upper left", markerscale=1.2, handletextpad=0.2)
            src_rows.append(d.assign(chosen_iou=s.chosen_iou, chosen_grad=s.chosen_grad,
                                     default_iou=s.default_iou, default_grad=s.default_grad))
            if i == 0 and j == 0:
                ax.text(-0.32, 1.28, "a", transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.suptitle(f"Global-QC calibration — default: {MODE_LABEL[mode]} (dashed grey); chosen thresholds (black)",
                 fontsize=7.5, y=0.995)
    pd.concat(src_rows).to_csv(SRC / f"Fig9_{mode}_a_qc_scatter.csv", index=False)
    # (b) standard space
    ax = fig.add_subplot(gsp[2, :2])
    x = np.arange(len(summ))
    ax.scatter(x - 0.1, summ.chosen_iou_percentile, marker="o", color="k", s=16, label="chosen IoU")
    ax.scatter(x - 0.1, summ.default_iou_percentile, marker="o", facecolor="none", edgecolor="k", s=16,
               label="default IoU")
    ax.scatter(x + 0.1, summ.chosen_grad_percentile, marker="s", color=OI[3], s=16, label="chosen grad_p95")
    ax.scatter(x + 0.1, summ.default_grad_percentile, marker="s", facecolor="none", edgecolor=OI[3], s=16,
               label="default grad_p95")
    ax.set_xticks(x, [f"{r.dataset}\n{r.variant}" for r in summ.itertuples()], fontsize=5.5)
    ax.set_ylabel("Percentile in own sweep\n(grad among IoU-passing runs)")
    ax.set_ylim(0, 102)
    ax.legend(frameon=False, fontsize=5, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.02))
    ax.text(-0.1, 1.22, "b", transform=ax.transAxes, fontsize=9, fontweight="bold")
    summ.to_csv(SRC / f"Fig9_{mode}_b_standard_space.csv", index=False)
    # (c) aMAP Dice change; (d) paper SNpc count change (separate units)
    sub = gsp[2, 2].subgridspec(2, 1, hspace=1.7)
    for k, (ds, col, xl, lab) in enumerate((("aMAP", "median_abs_change", "|Δ Dice|, median over cases", "c"),
                                            ("paper LSFM", "median_abs_change_pct", "|Δ SNpc count| %, median", "d"))):
        ax = fig.add_subplot(sub[k])
        r = dsum[dsum.dataset == ds]
        ax.barh(range(len(r)), r[col], color=["#0072B2", "#56B4E9"][: len(r)])
        ax.set_yticks(range(len(r)), list(r.variant), fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel(xl, fontsize=6)
        ax.set_title(f"{ds}: chosen vs default", fontsize=6.3)
        ax.text(-0.5, 1.2, lab, transform=ax.transAxes, fontsize=9, fontweight="bold")
    dsum.to_csv(SRC / f"Fig9_{mode}_cd_downstream.csv", index=False)
    for ext in ("pdf", "png"):
        fig.savefig(FIG_MAIN / f"Fig9_qc_calibration_{mode}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    log(f"  figure Fig9_qc_calibration_{mode}")


if __name__ == "__main__":
    allqc = load_all()
    mode_defs = {m: defaults(allqc, m) for m in MODES}
    pairs = {ds: {v: {CHOSEN[ds]} | {mode_defs[m][(ds, v)] for m in MODES} for v in VARIANTS}
             for ds in ("aMAP", "paper LSFM")}
    ad = amap_downstream(pairs["aMAP"])
    save_table(ad, "qcproc_amap_downstream")
    ld = lsfm_downstream(pairs["paper LSFM"])
    save_table(ld, "qcproc_lsfm_downstream")
    comp = []
    for mode in MODES:
        summ, pb = summarise(allqc, mode_defs[mode], mode)
        dsum = downstream_summary(mode_defs[mode], ad, ld)
        save_table(summ, f"qcproc_{mode}_summary")
        save_table(pb, f"qcproc_{mode}_per_brain")
        save_table(dsum, f"qcproc_{mode}_downstream_summary")
        figure(allqc, summ, dsum, mode)
        comp.append(summ[["mode", "dataset", "variant", "default_iou", "default_grad", "chosen_iou", "chosen_grad",
                          "chosen_stricter_iou", "chosen_stricter_grad", "pass_default", "pass_chosen",
                          "min_per_brain_default"]])
    comp = pd.concat(comp, ignore_index=True)
    save_table(comp, "qcproc_modes_comparison")
    pe = pd.concat([preprocessing_effect(allqc, mode_defs[m]).assign(rule=MODE_LABEL[m])
                    for m in ("start", "raw_anchored")], ignore_index=True)
    pe.loc[pe.thresholds == "chosen", "rule"] = "chosen (visual calibration)"
    pe = pe.drop_duplicates(["dataset", "rule"]).sort_values(["dataset", "thresholds"], kind="stable")
    save_table(pe, "qcproc_preprocessing_effect")
    # per-dataset thresholds for the viewer: chosen + default in every mode
    vt = []
    for ds in DATASETS:
        for v in VARIANTS:
            row = dict(dataset=ds, variant=v, chosen_iou=CHOSEN[ds][0], chosen_grad=CHOSEN[ds][1])
            for m in MODES:
                row[f"{m}_iou"], row[f"{m}_grad"] = mode_defs[m][(ds, v)]
            vt.append(row)
    save_table(pd.DataFrame(vt), "qcproc_viewer_thresholds")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    log(comp.round(4).to_string())
    log(pd.concat([pd.read_csv(VA / "results" / "tables" / f"qcproc_{m}_downstream_summary.csv").assign(mode=m)
                   for m in MODES]).round(4).to_string())
    log(pe.round(4).to_string())
