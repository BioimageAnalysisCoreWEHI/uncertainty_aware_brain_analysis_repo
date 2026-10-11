"""How much the final 0.5 consensus boundary moves between global-QC regimes: overlays on the brain image.

Regimes: the five predefined regimes of Fig. 3D (no global QC, median -/+ 3 MAD, Tukey 1.5 IQR, current, median),
read per dataset from results/qc_sensitivity/<dataset>/settings.csv; consensus volumes from the completed sensitivity
analysis (setting_lookup.csv -> set folders). Nothing is recomputed.

aMAP (raw input): contours on the expert STAPLE section, placed on the STAPLE grid exactly as the validation cache
(C.to_our_grid, bbox crop), with the expert STAPLE contour. Examples, fixed in advance:
  - the two cases shown in the focused validation figure (MV_Ntsr1_165 RSP and MV), as typical examples;
  - the brain x structure with the largest |Dice change| from the current regime among the four regimes retaining
    most registrations, and the largest across all five regimes (worst cases, labelled as such).
Contours show the full (both-hemisphere) 0.5 consensus; the Dice values quoted are the paper-protocol scores
(annotated hemisphere) from qcsens_amap_per_case.csv.

Paper LSFM (SNpc; no ground truth): every brain, raw and preprocessed; the coronal slice with the largest area of the
current-regime consensus; background = that brain's brainreg downsampled.tiff (same grid as the masks).

Outputs: figures/supplementary/FigS_qc_boundaries_{amap,paper}.{pdf,svg,png}, source_data/FigS_qc_boundaries_*.csv
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
import tifffile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import C, FIG_SUPP, PIPE, SRC, STRUCTS_MAIN, TABLES, VA, load_case, log  # noqa: E402

_s = importlib.util.spec_from_file_location("f3", str(VA / "scripts" / "19_figure3_qc.py"))
f3 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(f3)
REGIMES, RCOL = f3.REGIMES, f3.RCOL
QS = Path(P("<IMAGING>/Ishrat/JoM/post_review_analysis/validation_analysis/results/qc_sensitivity"))
PAPER_REG = Path(P(r"<SCRATCH>\jom_review_sweep_original_data"))
STAPLE_COL = "#E69F00"
LWS = [3.0, 2.3, 1.7, 1.15, 0.65]  # stroke widths, least -> most stringent regime


def page(path, i):
    with tifffile.TiffFile(path) as t:
        return t.pages[int(i)].asarray()


def win(p):
    return to_local(p)


def gray(img, mask=None):
    v = img[mask] if mask is not None and mask.any() else img
    lo, hi = np.percentile(v, [1, 99.5])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def crop_box(masks, margin, shape):
    u = np.logical_or.reduce([m for m in masks if m is not None and m.any()])
    r, c = np.flatnonzero(u.any(1)), np.flatnonzero(u.any(0))
    return (max(r[0] - margin, 0), min(r[-1] + margin + 1, shape[0]), max(c[0] - margin, 0),
            min(c[-1] + margin + 1, shape[1]))


def draw(ax, img, cons, truth=None):
    ax.imshow(img, cmap="gray", interpolation="nearest")
    # nested strokes: least stringent widest underneath, each stricter regime thinner on top, so coinciding
    # boundaries appear as one layered band and diverging ones as separate lines
    for ((lab, name), col, lw) in zip(REGIMES, RCOL, LWS):
        m = cons.get(name)
        if m is not None and m.any():
            ax.contour(m.astype(float), levels=[0.5], colors=[col], linewidths=lw)
    if truth is not None:
        ax.contour(truth.astype(float), levels=[0.5], colors=[STAPLE_COL], linewidths=1.0, linestyles=[(0, (3, 1.5))])
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def regime_ids(ds):
    st = pd.read_csv(QS / ds / "settings.csv")
    return {name: st.setting_id[st.label == lab].iloc[0] for lab, name in REGIMES}, st


def legend(fig, with_staple, retained, y=-0.02):
    h = [Line2D([], [], color=c, lw=lw, label=f"{n} ({retained[n]:.0f} % of raw registrations retained)")
         for (_, n), c, lw in zip(REGIMES, RCOL, LWS)]
    if with_staple:
        h.append(Line2D([], [], color=STAPLE_COL, lw=1.4, ls="--", label="Expert STAPLE"))
    fig.legend(handles=h, loc="upper center", ncol=2, fontsize=6, frameon=False, bbox_to_anchor=(0.5, y),
               handlelength=2.2, columnspacing=1.5)


# ------------------------------------------------------------------------------------------------------------
def amap():
    ids, st = regime_ids("amap")
    lk = pd.read_csv(QS / "amap" / "setting_lookup.csv")
    per = pd.read_csv(TABLES / "qcsens_amap_per_case.csv")
    per = per[(per.variant == "raw") & per.structure.isin(STRUCTS_MAIN) & per.setting_id.isin(ids.values())]
    wide = per.pivot_table(index=["brain", "structure"], columns="setting_id", values="dice_paper")
    dev = wide.sub(wide[ids["Current (manuscript)"]], axis=0).abs()
    core = [ids[n] for _, n in REGIMES if n != "Median"]
    worst_core = dev[core].max(1).idxmax()
    worst_all = dev.max(1).idxmax()
    examples = [(("MV_Ntsr1_165", "RSP"), "typical"), (("MV_Ntsr1_165", "MV"), "typical"),
                (worst_core, "largest change, four main regimes"), (worst_all, "largest change, all regimes")]
    reg = f3.regimes(st, pd.read_csv(TABLES / "qcsens_amap_per_case.csv"))
    retained = dict(zip(reg.regime, reg.retained_pct_raw))
    q = pd.read_csv(PIPE / "qc_tables" / "qc_results.csv")
    fig, axes = plt.subplots(2, 2, figsize=(6.85, 4.6))
    src = []
    for ax, ((brain, s), why) in zip(axes.ravel(), examples):
        z = load_case(brain, s, "raw")
        full, (s0, s1, r0, r1), A0 = tuple(z["full_shape"]), z["bbox"], int(z["A0"])
        truth = z["truth"] if z["truth"].shape != full else z["truth"][s0:s1, r0:r1]
        sample = next(x for x in q.sample_id.unique() if x.startswith(C.SAMPLE_PREFIX[brain]))
        reg_dir = Path(win(q.registration_dir[q.sample_id == sample].iloc[0]))
        img = C.to_our_grid(page(reg_dir / "downsampled.tiff", A0).astype(np.float32), full)[s0:s1, r0:r1]
        cons, dice = {}, {}
        for (_, name) in REGIMES:
            row = lk[(lk.brain == brain) & (lk.variant == "raw") & (lk.region == f"{s}_both") &
                     (lk.setting_id == ids[name])].iloc[0]
            if isinstance(row.set_path, str):
                pg = page(QS / "amap" / row.set_path / "consensus_thresh_0p5.tif", A0) > 0
                cons[name] = C.to_our_grid(pg, full)[s0:s1, r0:r1]
            d = per[(per.brain == brain) & (per.structure == s) & (per.setting_id == ids[name])].dice_paper
            dice[name] = float(d.iloc[0]) if len(d) else np.nan
            src.append(dict(brain=brain, structure=s, example=why, regime=name, setting_id=ids[name],
                            n_final=row.n_final, dice_paper=dice[name]))
        b = crop_box(list(cons.values()) + [truth], 6, truth.shape)
        sl = (slice(b[0], b[1]), slice(b[2], b[3]))
        draw(ax, gray(img[sl]), {k: v[sl] for k, v in cons.items()}, truth[sl])
        dv = [v for v in dice.values() if not np.isnan(v)]
        ax.set_title(f"{s} · {brain} ({why})\nDice vs STAPLE across regimes {min(dv):.2f}–{max(dv):.2f}",
                     fontsize=6.5, pad=3)
    legend(fig, True, retained, y=0.04)
    fig.subplots_adjust(wspace=0.06, hspace=0.25, bottom=0.14)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG_SUPP / f"FigS_qc_boundaries_amap.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(src).to_csv(SRC / "FigS_qc_boundaries_amap.csv", index=False)
    log(f"aMAP examples: {examples}")


def paper():
    ids, st = regime_ids("paper")
    lk = pd.read_csv(QS / "paper" / "setting_lookup.csv")
    lk["sample"] = lk["sample"].astype(str)
    per = pd.read_csv(TABLES / "qcsens_paper_per_case.csv")
    per["sample"] = per["sample"].astype(str)
    brains = sorted(lk["sample"].unique())
    fig, axes = plt.subplots(len(brains), 2, figsize=(6.85, 7.0))
    src, retained = [], {}
    idx = lk.drop_duplicates(["sample", "variant", "setting_id"])
    for (_, name) in REGIMES:
        g = idx[(idx.setting_id == ids[name]) & (idx.variant == "raw")]
        retained[name] = 100 * g.n_global.sum() / (48 * len(brains))
    for j, v in enumerate(("raw", "preprocessed")):
        for i, s in enumerate(brains):
            ax = axes[i, j]
            paths = {}
            for (_, name) in REGIMES:
                row = lk[(lk["sample"] == s) & (lk.variant == v) & (lk.setting_id == ids[name])].iloc[0]
                paths[name] = (QS / "paper" / row.set_path / f"{s}_consensus_0.50_coronal.tif"
                               if isinstance(row.set_path, str) else None)
            vol = tifffile.imread(paths["Current (manuscript)"]) > 0
            z = int(np.argmax(vol.sum((1, 2))))
            cons = {n: (page(p, z) > 0 if p is not None else None) for n, p in paths.items()}
            run = next(p for p in sorted((PAPER_REG / s / v).iterdir()) if (p / "downsampled.tiff").exists())
            img = page(run / "downsampled.tiff", z).astype(np.float32)
            b = crop_box(list(cons.values()), 12, img.shape)
            sl = (slice(b[0], b[1]), slice(b[2], b[3]))
            draw(ax, gray(img[sl]), {k: (m[sl] if m is not None else None) for k, m in cons.items()})
            cnt = per[(per["sample"] == s) & (per.variant == v) & per.setting_id.isin(ids.values())]
            cc = cnt.set_index("setting_id")["count"]
            lo, hi = cc.min(), cc.max()
            ax.set_title((f"{v}\n" if i == 0 else "") + f"{s}: SNpc count across regimes {lo:,.0f}–{hi:,.0f}",
                         fontsize=6.5, pad=3)
            for (_, name) in REGIMES:
                src.append(dict(sample=s, variant=v, slice_index=z, regime=name, setting_id=ids[name],
                                count=cc.get(ids[name], np.nan)))
    legend(fig, False, retained, y=0.06)
    fig.subplots_adjust(wspace=0.05, hspace=0.3, bottom=0.1)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG_SUPP / f"FigS_qc_boundaries_paper.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(src).to_csv(SRC / "FigS_qc_boundaries_paper.csv", index=False)
    log("paper done")


if __name__ == "__main__":
    f3.style()
    amap()
    paper()
