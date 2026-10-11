"""Supplementary: QC-rule sensitivity of the consensus by region (raw input), as Fig 6D/E split by region.
  A  aMAP: change in consensus Dice with the expert STAPLE vs the current rule, one panel per structure
  B  independent LSFM: change in consensus 3-D Dice with the published registration vs the current rule, per region
Panels ordered largest -> smallest region (median volume under the current rule); in each panel the seven alternative
rules (most -> fewest registrations retained), one point per brain, purple bar = median, grey band = +/-0.02.
All panels share the y axis of Fig 6D/E. Same per-case values as Fig 6D/E.
Output: manuscript_figures/FigS_QC_sensitivity_by_region_v2.{png,pdf,svg}, source_data_fig6/FigS_QC_sensitivity_by_region_v2.csv
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

D6 = Path(__file__).resolve().parent / "fig6_and_figS6.py"
_s = importlib.util.spec_from_file_location("f6", str(D6))
f6 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(f6)
m = f6.m
plt = m.plt
OUT, SRC = f6.OUT, f6.SRC
NCOL = {"amap": 3, "lsfm2": 4}
YLIM = (-0.46, 0.23)                                         # as Fig 6D/E (shared axis)


def wide(ds):
    if ds == "amap":
        d = pd.read_csv(m.TABLES / "qcsens_amap_per_case.csv")
        d = d[(d.variant == "raw") & d.structure.isin(m.STRUCTS_MAIN)]
        w = d.pivot_table(index=["structure", "brain"], columns="setting_id", values="dice_paper")
        vcol = next((c for c in ("volume_vox", "volume") if c in d.columns), None)
        size = (d[d.setting_id == m.REGIMES["Current"]["amap"]].groupby("structure")[vcol].median()
                if vcol else pd.Series(range(len(m.STRUCTS_MAIN)), index=m.STRUCTS_MAIN))
    else:
        d = pd.read_csv(m.TABLES / "qcsens_lsfm2_per_case.csv")
        d = d[d.variant == "raw"]
        w = d.pivot_table(index=["region", "sample"], columns="setting_id", values="dice_vs_ref")
        size = d[d.setting_id == m.REGIMES["Current"]["lsfm2"]].groupby("region").volume_vox.median()
    return w, list(size.sort_values(ascending=False).index)


def section(fig, g, ds, title, letter, rows):
    w, order_r = wide(ds)
    order, ret = f6.rule_order(ds, "raw")
    alts = [n for n in order if n != "Current"]
    cur = m.REGIMES["Current"][ds]
    ncol = NCOL[ds]
    nrow = int(np.ceil(len(order_r) / ncol))
    gg = g.subgridspec(nrow, ncol, hspace=0.3, wspace=0.1)
    n_empty = {n: int((w[m.REGIMES[n][ds]].groupby(level=1).apply(lambda x: x.isna().all())).sum()) for n in alts}
    rng = np.random.default_rng(0)
    first = None
    for k, reg in enumerate(order_r):
        r, c = divmod(k, ncol)
        ax = fig.add_subplot(gg[r, c], sharey=first)
        first = first or ax
        sub = w.loc[reg]
        for i, n in enumerate(alts):
            dd = (sub[m.REGIMES[n][ds]] - sub[cur]).dropna()
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(dd)), dd, s=5, color="0.45", lw=0, alpha=0.8, zorder=3)
            if len(dd):
                ax.plot([i - 0.3, i + 0.3], [dd.median()] * 2, color=m.C_CONS, lw=1.4, zorder=4)
            rows.append(dict(dataset=ds, region=reg.replace("_both", ""), rule=n, retained_pct=ret[n],
                             median_delta=dd.median() if len(dd) else np.nan, max_abs_delta=dd.abs().max() if len(dd)
                             else np.nan, n_brains_changed_gt_0p02=int((dd.abs() > 0.02).sum()),
                             n_brains_evaluable=len(dd), n_brains=len(sub)))
        ax.axhline(0, color="k", lw=0.5, zorder=2)
        ax.axhspan(-0.02, 0.02, color="0.92", zorder=0)
        ax.set_ylim(*YLIM)
        ax.set_xlim(-0.6, len(alts) - 0.4)
        ax.set_title(reg.replace("_both", ""), fontsize=6.3, pad=2)
        ax.tick_params(axis="y", labelsize=5.2, length=2)
        last_in_col = k + ncol >= len(order_r)
        if last_in_col:
            ax.set_xticks(range(len(alts)), [f"{m.SHORT_RULE[n]} ({ret[n]:.0f}%)" + (f", {n_empty[n]} empty" if n_empty[n] else "") for n in alts], fontsize=4.8,
                          rotation=65, ha="right", rotation_mode="anchor")
        else:
            ax.set_xticks(range(len(alts)), [])
        ax.tick_params(axis="x", length=2)
        if c:
            plt.setp(ax.get_yticklabels(), visible=False)
        else:
            ax.set_ylabel("Δ Dice vs\ncurrent rule", fontsize=5.6)
        if k == 0:
            ax.text(0, 1.2, f"{letter}  {title}; current rule retains {ret['Current']:.0f}%", transform=ax.transAxes,
                    fontsize=7, fontweight="bold", va="bottom")
    return nrow


def main():
    rows = []
    fig = plt.figure(figsize=(7.2, 11.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[3, 4], hspace=0.4)
    section(fig, outer[0], "amap", "aMAP vs expert STAPLE (6 brains per structure)", "A", rows)
    section(fig, outer[1], "lsfm2", "Independent LSFM vs published registration (6 brains per region)", "B", rows)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"FigS_QC_sensitivity_by_region_v2.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(SRC / "FigS_QC_sensitivity_by_region_v2.csv", index=False)
    print("saved")


if __name__ == "__main__":
    main()
