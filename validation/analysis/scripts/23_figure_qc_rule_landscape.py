"""Supplementary Fig. S5: where each global-QC rule of the QC-sensitivity analysis sits on the IoU x grad_p95 landscape.

Visualises the EXISTING rules only. Nothing is recomputed or redefined:
  * rule -> setting_id: the REGIMES dict of 22_focused_revision_figures.py (read from its source with ast, not run);
  * setting_id -> thresholds: results/qc_sensitivity/<ds>/settings.csv (written by 17_qc_sensitivity_consensus.py);
  * QC metrics: the same tables script 17 used (aMAP: pipeline qc_results{,_preprocessed}.csv; original LSFM:
    dataset_qc_comparison/original_lsfm_full_final_results_with_ssim.csv, raw_/preprocessed_ columns; independent
    LSFM: validation_datasets/lightsheet_data/results/qc_tables/qc_results{,_preprocessed}.csv).
As in script 17, one threshold pair per dataset (data-derived rules from the RAW sweep) is applied to raw and to
preprocessed registrations; pass = IoU > t_IoU and grad_p95 < t_grad.

Checks (written to source_data/FigS5_verification.csv, run aborts on failure):
  thresholds equal settings.csv; counts equal setting_lookup.csv n_global; Current equals the pipeline's own QC flags;
  No global QC keeps 100 %; Median = raw marginal medians and Untuned start = qc_start.start_thresholds(raw).

Outputs (REVISION_OUTPUTS_FOCUSED): figures/FigS5a_QC_rule_landscape_aMAP, FigS5b_QC_rule_landscape_LSFM,
FigS5c_QC_rule_landscape_independent_LSFM (.pdf/.svg/.png 600 dpi); source_data/FigS5_registrations.csv, FigS5_registration_by_rule.csv,
FigS5_rules_retention.csv, FigS5_verification.csv.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from common import PIPE, SHORT, VA, amap_brain, log  # noqa: E402
from qc_start import start_thresholds  # noqa: E402

PRA = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis"))
OUT = PRA / "REVISION_OUTPUTS_FOCUSED"
FIG, SRC = OUT / "figures", OUT / "source_data"
QS = VA / "results" / "qc_sensitivity"
PAPER_TABLE = PRA / "dataset_qc_comparison" / "original_lsfm_full_final_results_with_ssim.csv"
LS_PIPE = PRA / "validation_datasets" / "lightsheet_data" / "results"       # independent LSFM pipeline output
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]       # as Fig. 2A-C (19_figure3_qc.py)
PASS_FILL = "#e8f3e4"
RULE_ORDER = ["No global QC", "Very lenient", "Lenient", "Untuned start (median ± 3·MAD)", "Current", "Strict",
              "Very strict", "Median IoU and grad_p95"]
SHORT_NAME = {"Untuned start (median ± 3·MAD)": "Untuned start (median ± 3·MAD)",
              "Median IoU and grad_p95": "Median (marginal medians)"}
DERIVATION = {
    "No global QC": "thresholds switched off",
    "Very lenient": "grid setting keeping ≈95 % of raw registrations, both thresholds looser than Lenient",
    "Lenient": "grid setting keeping ≈85 % of raw registrations, both thresholds looser than Current",
    "Untuned start (median ± 3·MAD)": "raw sweep: IoU median − 3·MAD; grad_p95 median + 3·MAD among IoU passers",
    "Current": "manuscript rule",
    "Strict": "grid setting keeping ≈60 % of Current's raw retention, both thresholds stricter",
    "Very strict": "grid setting keeping ≈30 % of Current's raw retention",
    "Median IoU and grad_p95": "raw sweep: marginal median of IoU and, independently, of grad_p95",
}
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 5.5,
                     "ytick.labelsize": 5.5, "legend.fontsize": 6, "axes.linewidth": 0.5, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none"})


def regimes():
    """The REGIMES literal of script 22, parsed (not executed)."""
    tree = ast.parse((VA / "scripts" / "22_focused_revision_figures.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "REGIMES" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("REGIMES not found in script 22")


def load_metrics():
    """Registration-level QC metrics, exactly the tables script 17 read."""
    rows = []
    for v, f in (("raw", "qc_results.csv"), ("preprocessed", "qc_results_preprocessed.csv")):
        q = pd.read_csv(PIPE / "qc_tables" / f)
        q["registration_id"] = q.registration_dir.astype(str).str.rstrip("/").str.split("/").str[-1]
        q["brain"] = q.sample_id.map(amap_brain)
        rows.append(pd.DataFrame(dict(dataset="aMAP", input_type=v, sample_id=q.brain,
                                      sample_label=q.brain.map(lambda b: SHORT.get(b, b)),
                                      registration_id=q.registration_id, iou=q.iou, grad_p95=q.grad_p95,
                                      pipeline_passes_current=q.passes_global_qc.astype(bool))))
    t = pd.read_csv(PAPER_TABLE)
    t = t.loc[:, ~t.columns.str.contains("^Unnamed")]
    t["Sample ID"] = t["Sample ID"].astype(str)
    mp = pd.read_csv(SRC / "sample_number_mapping.csv")
    lab = dict(zip(mp.sample_id.astype(str), mp.label))
    for v in ("raw", "preprocessed"):
        rows.append(pd.DataFrame(dict(dataset="original LSFM", input_type=v, sample_id=t["Sample ID"],
                                      sample_label=t["Sample ID"].map(lab),
                                      registration_id=t.bending.map(lambda b: f"{b:g}") + "_" + t.grid.map(
                                          lambda g: f"{g:g}"),
                                      iou=t[f"{v}_iou"], grad_p95=t[f"{v}_grad_p95"],
                                      pipeline_passes_current=(t[f"{v}_pass_iou"].astype(bool)
                                                               & t[f"{v}_pass_grad_p95"].astype(bool)))))
    for v, f in (("raw", "qc_results.csv"), ("preprocessed", "qc_results_preprocessed.csv")):
        q = pd.read_csv(LS_PIPE / "qc_tables" / f)
        q["registration_id"] = q.registration_dir.astype(str).str.rstrip("/").str.split("/").str[-1]
        rows.append(pd.DataFrame(dict(dataset="independent LSFM", input_type=v, sample_id=q.sample_id.astype(str),
                                      sample_label=q.sample_id.astype(str), registration_id=q.registration_id,
                                      iou=q.iou, grad_p95=q.grad_p95,
                                      pipeline_passes_current=q.passes_global_qc.astype(bool))))
    return pd.concat(rows, ignore_index=True)


def main():
    reg = regimes()
    assert set(reg) == set(RULE_ORDER), sorted(reg)
    m = load_metrics()
    m.drop(columns="pipeline_passes_current").to_csv(SRC / "FigS5_registrations.csv", index=False)
    rule_rows, long_rows, ver = [], [], []

    def check(name, ok, detail=""):
        ver.append(dict(check=name, ok=bool(ok), detail=detail))
        if not ok:
            raise AssertionError(f"{name}: {detail}")

    for ds, key in (("aMAP", "amap"), ("original LSFM", "paper"), ("independent LSFM", "lsfm2")):
        st = pd.read_csv(QS / key / "settings.csv").set_index("setting_id")
        lk = pd.read_csv(QS / key / "setting_lookup.csv")
        lk = lk.drop_duplicates(["sample", "variant", "setting_id"])
        d = m[m.dataset == ds]
        raw = d[d.input_type == "raw"]
        med = (round(float(np.median(raw.iou)), 4), round(float(np.median(raw.grad_p95)), 4))
        sst = start_thresholds(raw.iou, raw.grad_p95)
        sid_med, sid_st = reg["Median IoU and grad_p95"][key], reg["Untuned start (median ± 3·MAD)"][key]
        check(f"{ds}: Median = raw marginal medians",
              np.isclose(st.loc[sid_med, "iou_thr"], med[0]) and np.isclose(st.loc[sid_med, "grad_thr"], med[1])
              and st.loc[sid_med, "label"] == "median",
              f"settings {st.loc[sid_med, 'iou_thr']}/{st.loc[sid_med, 'grad_thr']} vs medians {med}; "
              f"unrounded {np.median(raw.iou):.6f}/{np.median(raw.grad_p95):.6f}")
        check(f"{ds}: Untuned start = qc_start.start_thresholds(raw)",
              np.isclose(st.loc[sid_st, "iou_thr"], sst[0]) and np.isclose(st.loc[sid_st, "grad_thr"], sst[1]),
              f"{st.loc[sid_st, 'iou_thr']}/{st.loc[sid_st, 'grad_thr']} vs {sst}")
        for rule in RULE_ORDER:
            sid = reg[rule][key]
            ti, tg = st.loc[sid, "iou_thr"], st.loc[sid, "grad_thr"]
            for v in ("raw", "preprocessed"):
                g = d[d.input_type == v]
                p_i = (g.iou > ti).to_numpy() if pd.notna(ti) else np.ones(len(g), bool)
                p_g = (g.grad_p95 < tg).to_numpy() if pd.notna(tg) else np.ones(len(g), bool)
                both = p_i & p_g
                n, N = int(both.sum()), len(g)
                n_lookup = int(lk[(lk.setting_id == sid) & (lk.variant == v)].n_global.sum())
                check(f"{ds} {v} {rule}: n matches setting_lookup.csv", n == n_lookup, f"{n} vs {n_lookup}")
                if rule == "Current":
                    check(f"{ds} {v}: Current equals the pipeline's own global-QC flags",
                          (both == g.pipeline_passes_current.to_numpy()).all(),
                          f"{int((both != g.pipeline_passes_current.to_numpy()).sum())} mismatches")
                if rule == "No global QC":
                    check(f"{ds} {v}: No global QC keeps 100 %", n == N, f"{n}/{N}")
                rule_rows.append(dict(
                    dataset=ds, input_type=v, rule=rule, setting_id=sid, derivation=DERIVATION[rule],
                    thresholds_from="raw sweep of this dataset, applied unchanged to both input types",
                    iou_threshold=ti, grad_p95_threshold=tg, n_pass_iou=int(p_i.sum()), n_pass_grad=int(p_g.sum()),
                    n_retained=n, N_total=N, retention_pct=100 * n / N, pct_pass_iou=100 * p_i.mean(),
                    pct_pass_grad=100 * p_g.mean(), pct_expected_if_independent=100 * p_i.mean() * p_g.mean(),
                    n_global_setting_lookup=n_lookup))
                long_rows.append(pd.DataFrame(dict(
                    dataset=ds, input_type=v, sample_id=g.sample_id, registration_id=g.registration_id, iou=g.iou,
                    grad_p95=g.grad_p95, qc_rule=rule, iou_threshold=ti, grad_threshold=tg, passes_iou=p_i,
                    passes_grad=p_g, passes_both=both, n_retained=n, N_total=N, retention_pct=100 * n / N)))
    rules = pd.DataFrame(rule_rows)
    rules.to_csv(SRC / "FigS5_rules_retention.csv", index=False)
    pd.concat(long_rows, ignore_index=True).to_csv(SRC / "FigS5_registration_by_rule.csv", index=False)
    check("no Dice / STAPLE / count input used to place thresholds", True,
          "thresholds read from settings.csv only; metrics are IoU and grad_p95 only")
    pd.DataFrame(ver).to_csv(SRC / "FigS5_verification.csv", index=False)
    for ds, tag, title in (("aMAP", "a_QC_rule_landscape_aMAP", "aMAP (serial two-photon)"),
                           ("original LSFM", "b_QC_rule_landscape_LSFM", "Original LSFM"),
                           ("independent LSFM", "c_QC_rule_landscape_independent_LSFM", "Independent LSFM")):
        draw(m[m.dataset == ds], rules[rules.dataset == ds], f"FigS5{tag}", title)
    log(rules[["dataset", "input_type", "rule", "iou_threshold", "grad_p95_threshold", "n_retained", "N_total",
               "retention_pct", "pct_pass_iou", "pct_pass_grad"]].round(4).to_string(index=False))


def fmt(x, nd):
    return f"{x:.{nd}f}".rstrip("0").rstrip(".") if pd.notna(x) else "off"


def broken_scale(ax, brk, frac, xl):
    """Linear x axis whose part below `brk` is compressed to `frac` of the width (all points stay visible)."""
    lo = brk - xl[0]
    c = frac * (xl[1] - brk) / ((1 - frac) * lo)

    def fwd(x):
        x = np.asarray(x, float)
        return np.where(x < brk, brk - (brk - x) * c, x)

    def inv(y):
        y = np.asarray(y, float)
        return np.where(y < brk, brk - (brk - y) / c, y)
    ax.set_xscale("function", functions=(fwd, inv))
    ax.axvline(brk, color="0.75", lw=0.4, ls="-", zorder=1)
    ax.text(brk, 0.0, "//", transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=6, color="0.3",
            clip_on=False, bbox=dict(facecolor="white", lw=0, pad=0.2))


def draw(d, rules, name, title):
    labels = sorted(d.sample_label.unique())                      # same ordering -> same colours as Fig. 2A-C
    col = {b: OKABE[k] for k, b in enumerate(labels)}
    xi, xa = d.iou.min(), d.iou.max()
    yi, ya = d.grad_p95.min(), d.grad_p95.max()
    xl = (xi - 0.03 * (xa - xi), xa + 0.03 * (xa - xi))
    yl = (yi - 0.05 * (ya - yi), ya + 0.05 * (ya - yi))
    brk = 0.8 if xi < 0.7 else None                               # aMAP: long tail of failed registrations
    fig = plt.figure(figsize=(7.2, 9.0))
    outer = fig.add_gridspec(4, 2, hspace=0.62, wspace=0.16, left=0.07, right=0.99, top=0.925, bottom=0.07)
    for k, rule in enumerate(RULE_ORDER):
        inner = outer[k // 2, k % 2].subgridspec(1, 2, wspace=0.08)
        rr = rules[rules.rule == rule].set_index("input_type")
        ti, tg = rr.loc["raw", "iou_threshold"], rr.loc["raw", "grad_p95_threshold"]
        for j, v in enumerate(("raw", "preprocessed")):
            ax = fig.add_subplot(inner[0, j])
            if brk:
                broken_scale(ax, brk, 0.22, xl)
                ax.set_xticks([0.4, 0.8, 0.85, 0.9, 0.95], ["0.4", "0.8", "0.85", "0.9", "0.95"])
            ax.set_title(v, fontsize=5.8, color="0.3", pad=2)
            r = rr.loc[v]
            if pd.notna(ti) or pd.notna(tg):
                x0 = ti if pd.notna(ti) else xl[0]
                y1 = tg if pd.notna(tg) else yl[1]
                ax.add_patch(plt.Rectangle((x0, yl[0]), xl[1] - x0, y1 - yl[0], facecolor=PASS_FILL, lw=0, zorder=0))
            g = d[d.input_type == v]
            for b in labels:
                h = g[g.sample_label == b]
                ax.scatter(h.iou, h.grad_p95, s=3.2, color=col[b], alpha=0.8, lw=0, zorder=2)
            ismed = rule.startswith("Median")
            ls = (0, (4, 2)) if not ismed else (0, (1.5, 1.5))
            if pd.notna(ti):
                ax.axvline(ti, color="0.15", lw=0.7, ls=ls, zorder=3)
            if pd.notna(tg):
                ax.axhline(tg, color="0.15", lw=0.7, ls=ls, zorder=3)
            ax.set_xlim(*xl)
            ax.set_ylim(*yl)
            txt = f"Retained: {r.retention_pct:.0f}% ({int(r.n_retained)}/{int(r.N_total)})"
            if ismed:
                txt = (f"IoU > median alone: {r.pct_pass_iou:.0f}%\ngrad < median alone: {r.pct_pass_grad:.0f}%\n"
                       f"if independent: {r.pct_expected_if_independent:.0f}%\n" + txt)
            emptier_left = brk and not title.startswith("Independent")   # emptier upper corner of each dataset
            tx, ha = (0.03, "left") if emptier_left else (0.97, "right")
            ax.text(tx, 0.97, txt, transform=ax.transAxes, ha=ha, va="top", fontsize=5.2, linespacing=1.15,
                    zorder=1)                                     # beneath the points: no registration is hidden
            if j:
                ax.set_yticklabels([])
            elif k % 2 == 0:
                ax.set_ylabel("grad_p95")
            if k // 2 == 3:
                ax.set_xlabel("Tissue–atlas IoU")
        crit = ("no thresholds: every registration passes" if pd.isna(ti) and pd.isna(tg)
                else f"pass: IoU > {fmt(ti, 4)} and grad_p95 < {fmt(tg, 4)}")
        if rule.startswith("Median"):
            crit = (f"pass: IoU > {fmt(ti, 4)} and grad_p95 < {fmt(tg, 4)} "
                    "(independent marginal medians, raw sweep)")
        pos = outer[k // 2, k % 2].get_position(fig)
        fig.text(pos.x0 - 0.035, pos.y1 + 0.03, f"{chr(65 + k)}  {SHORT_NAME.get(rule, rule)}", fontsize=7,
                 fontweight="bold", va="bottom", ha="left")
        fig.text(pos.x0 - 0.035, pos.y1 + 0.016, crit, fontsize=5.4, va="bottom", ha="left", color="0.25")
    fig.suptitle(f"{title}: global-QC rules of the sensitivity analysis on the IoU × grad_p95 landscape",
                 fontsize=7.5, y=0.99)
    h = [Line2D([], [], ls="", marker="o", ms=3.5, color=col[b], label=b) for b in labels]
    h += [Patch(facecolor=PASS_FILL, edgecolor="none", label="passes global QC (IoU > t and grad_p95 < t)"),
          Line2D([], [], color="0.15", lw=0.7, ls=(0, (4, 2)), label="rule thresholds"),
          Line2D([], [], color="0.15", lw=0.7, ls=(0, (1.5, 1.5)), label="marginal medians (raw sweep)")]
    fig.legend(handles=h, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, 0.0), fontsize=5.8,
               handletextpad=0.3, columnspacing=1.0)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=600, facecolor="white")
    plt.close(fig)
    log(f"  {name}")


if __name__ == "__main__":
    sys.exit(main())
