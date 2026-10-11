"""Figure 3 (v9): A/B TH with consensus at > 0, 0.5, 1.0 (old 3F style; Sample 3 raw), C/D count vs threshold, E/F cell maps (0.5 boundary only); per-brain counts -> supplementary.
Figure 3 (v8): B = 13 retained registrations (global QC + top 50 % SSIM).
Figure 3 (v7): A = no QC, B = global QC (swapped); C all-48 curves; D retained; E per brain.
Figure 3 (v6): A/B reproduce scripts/plot_cell_prob_overlay.py (old Fig 3I/J).
Figure 3 (v5): what the consensus does to the cells and to the counts. Implements mockups/Fig3_v5_mockup.png.

  A/B Sample 1, raw input, full-resolution TH (max projection of planes 96-99, the slab of the old Fig 3I/J) with the
      Cellpose cells coloured by their assignment over (A) the registrations retained by global QC + SSIM and (B) all
      48 registrations (no QC):
        red        counted in the SNpc at inclusion >= 0.5 (overlap rule: any voxel inside the 0.5 consensus; these are
                   exactly the cells in the reported count)
        SNr / VTA / MB / other   region holding the cell centre in >= 50 % of the registrations
        uncertain  not counted and no region holds the cell centre in >= 50 % of the registrations
      Lines: SNpc consensus of that set at inclusion 0.25 (dotted), 0.5 (solid), 0.75 (dashed), max over the slab.
      Title %: uncertain cells among the cells that lie in the SNpc in at least one registration of that set (whole crop).
  C   consensus SNpc count vs inclusion threshold, retained registrations, raw (solid) and preprocessed (dotted)
  D   per brain: consensus count at 0.5 (diamond), range of counts for thresholds 0.25-0.75 (bar), retained singles (dots)
  Supplementary: C for all 48 registrations.

Counts: overlap rule from the voxel-vote cache (identical to final_count_results/consensus_threshold_cell_counts_overlap_
logic); singles: final_count_results/full_final_results_with_ssim.csv. Cell regions: fig3_cell_regions.py.
Output: manuscript_figures/Fig3_v6.{png,pdf,svg}, Fig3_v6_supp_all48.{png,pdf,svg}, source_data_fig3/*.csv
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P  # noqa: E402  (resolves <IMAGING>/<SCRATCH> placeholders)

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

VA = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_analysis"))
_REPO = next(_p for _p in Path(__file__).resolve().parents if (_p / "uaa_paths.py").exists())
_CODE = _REPO / "validation" / "analysis" / "scripts"          # analysis scripts in this repository
_s = importlib.util.spec_from_file_location("m35", str(_CODE / "35_new_main_figures.py"))
m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m)

OUT = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\manuscript_figures"))
SRC = OUT / "source_data_fig3"
SRC.mkdir(exist_ok=True)
REL = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\data_availability"))
S, V, SLAB = "162", "raw", (96, 99)
C_RAW, C_PRE = "#E08214", "#2C7BB6"
CAT_COL = {"SNpc": "#D7191C", "SNr": "#1a9850", "VTA": "#2C7BB6", "MB": "#7B3294", "other": "#BDBDBD",
           "uncertain": "#FFD92F"}
ORDER = ["SNpc", "SNr", "VTA", "MB", "other", "uncertain"]
CONF = 0.8                                   # old Fig 3I/J rule: assigned if one region holds the cell centre in >= 80 %


def L(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


# ------------------------------------------------------------------------------------------------------------
# A/B: exactly the method of scripts/plot_cell_prob_overlay.py (old Fig 3I/J)
# ------------------------------------------------------------------------------------------------------------
J = Path(P(r"<IMAGING>\Ishrat\JoM"))
AB_S, AB_V, AB_SLAB = "162", "preprocessed", (96, 99)
AB_IMAGE = (J / "results_preprocessed" / "extracted_regions_preprocessed_no_padding_union" /
            "162_AF_level1_preprocessed_freeform_bending_0.9_grid_-30_smoothing_-1_atlas_perens_lsfm_mouse_20um" /
            "162_snpc_image_region_scale.tif")
AB_LABELS = (J / "results_preprocessed" / "cellpose_masked_cells" / "for_cellpose" /
             "162_AF_level1_preprocessed_freeform_bending_0.0_grid_-2_smoothing_-1_atlas_perens_lsfm_mouse_20um" /
             "results_python" / "snpc" / "whole_segmentation_python.tif")
AB_SETS = {"A": (str(SRC / "162_preprocessed_retained_cell_region_grouped_probabilities_long.csv"), None),
           "B": ("cell_region_probability_outputs_all_runs", "overlay_clean_all_runs.tif")}
BOUNDARY = 0.8
BOUNDS = ((0.5, "-", 1.0),)
BCOL = {0.5: "white"}
OLD_COL = {"SNc": "#D7191C", "VTA": "#2C7BB6", "MB": "#7B3294", "Uncertain": "#FFD92F", "Other": "#D0D0D0"}
NEW_COL = dict(OLD_COL, SNr="#1a9850")                          # mockup: SNr shown separately (old: part of Other)


def dominant(folder):
    f = Path(folder)
    p = pd.read_csv(f if f.suffix == ".csv" else
                    J / "final_count_results" / folder / "ALL_cell_region_grouped_probabilities_long.csv",
                    low_memory=False)
    p = p[(p.sample_id.astype(str) == AB_S) & (p.processing_status == AB_V)]
    idx = p.groupby(["sample_id", "processing_status", "cell_id"])["probability"].idxmax()
    dom = p.loc[idx].rename(columns={"region_group": "dominant_region", "probability": "confidence"})
    snc = p[p.region_group == "SNc"].set_index("cell_id")["probability"]
    dom["snc_share"] = dom.cell_id.map(snc).fillna(0.0)
    return dom.reset_index(drop=True)


def plot_region(dom, with_snr):
    r = np.full(len(dom), "Other", dtype=object)
    conf = dom.confidence.values >= BOUNDARY
    keep = ["SNc", "VTA", "MB"] + (["SNr"] if with_snr else [])
    for k in keep:
        r[conf & (dom.dominant_region.values == k)] = k
    r[~conf] = "Uncertain"
    return r


def render(dom, cats, colours):
    with tifffile.TiffFile(AB_IMAGE) as t:
        img = np.stack([t.pages[z].asarray() for z in range(AB_SLAB[0], AB_SLAB[1] + 1)])
    img = img[..., 0] if img.ndim == 4 else img
    with tifffile.TiffFile(AB_LABELS) as t:
        lab = np.stack([t.pages[z].asarray() for z in range(AB_SLAB[0], AB_SLAB[1] + 1)])
    proj, lproj = img.max(0).astype(np.float32), lab.max(0)
    a, b = np.percentile(proj, [1, 99.8])
    base = np.clip((proj - a) / (b - a + 1e-8), 0, 1)
    rgb = np.dstack([base] * 3)
    lut = np.zeros((int(lproj.max()) + 1, 3), np.float32)
    has = np.zeros(int(lproj.max()) + 1, bool)
    for cid, c in zip(dom.cell_id.astype(int), cats):
        if cid <= lproj.max():
            lut[cid] = to_rgb(colours[c])
            has[cid] = True
    msk = has[lproj]
    out = rgb.copy()
    out[msk] = 0.35 * rgb[msk] + 0.65 * lut[lproj[msk]]
    return out


def ab_probability(keys):
    p = None
    for z in range(AB_SLAB[0], AB_SLAB[1] + 1):
        pz = np.mean([m.page(m.row(AB_S, AB_V, k).mask_axial_path, z) > 0 for k in keys], axis=0)
        p = pz if p is None else np.maximum(p, pz)
    return p


def panel_ab(fig, gs, src):
    runs = m.runs(AB_S, AB_V)
    sets = {"A": m.retained(AB_S, AB_V), "B": list(runs.run_key)}
    g = gs.subgridspec(1, 2, wspace=0.05)
    old_dir = OUT / "figures" / "Figure3 - Local QC + SSIM"
    disp = {"B": "E", "A": "F"}                               # shown: E = no QC (all 48), F = retained
    for k, letter in enumerate(("B", "A")):
        folder, old_png = AB_SETS[letter]
        dom = dominant(folder)
        # 1) exact reproduction of the old panel, checked against the saved image
        if old_png is not None:
            ref = render(dom, plot_region(dom, with_snr=False), OLD_COL)
            old = tifffile.imread(old_dir / old_png)
            same = np.abs((np.clip(ref, 0, 1) * 255).astype(np.uint8).astype(int) - old.astype(int))
        else:
            same = np.zeros((1, 1, 3), int)                  # no old image for the retained set
        # 2) the panel: same method, SNr separate, real boundaries, % uncertain
        cats = plot_region(dom, with_snr=True)
        rgb = render(dom, cats, NEW_COL)
        ax = fig.add_subplot(g[k])
        ax.imshow(rgb, interpolation="nearest")
        m.clean(ax)
        prob = ab_probability(sets[letter])
        for t, ls, lw in BOUNDS:
            m.contour(ax, prob >= t - 1e-6, BCOL[t], lw, ls)
        near = dom.snc_share.values > 0                          # (the TH image carries its own 500 um scale bar)
        pct = 100 * np.mean(cats[near] == "Uncertain")
        n = int(dom.n_registrations.iloc[0]) if "n_registrations" in dom else len(sets[letter])
        title = (f"Retained registrations ({len(sets[letter])})" if letter == "A"
                 else "No QC (all 48)")
        ax.set_title(f"{title}", fontsize=6.8, pad=3, loc="left")
        L(ax, disp[letter], x=-0.01, y=1.02)
        src.append(dict(panel=disp[letter], sample=m.LAB[AB_S], input=AB_V, n_registrations=len(sets[letter]),
                        probability_table=folder, cells_in_snpc_in_any_registration=int(near.sum()),
                        uncertain_among_them=int((cats[near] == "Uncertain").sum()), pct_uncertain=pct,
                        reproduction_vs_old_png_max_abs_diff=int(same.max()),
                        reproduction_pixels_differing=int((same.max(-1) > 1).sum())))
        dom.assign(plot_region=cats).to_csv(SRC / f"Fig3_v9_{disp[letter]}_cells.csv", index=False)
        if k == 0:
            h = [Patch(fc=NEW_COL[c], ec="0.3", lw=0.4, label={"SNc": "SNpc"}.get(c, c))
                 for c in ("SNc", "SNr", "VTA", "MB", "Other", "Uncertain")]
            h += [Line2D([], [], color="0.15", lw=1.1, label="SNpc consensus (0.5)")]
            ax.legend(handles=h, loc="upper center", bbox_to_anchor=(1.03, -0.02), ncol=9, fontsize=5.6,
                      handlelength=1.3, columnspacing=0.9, handletextpad=0.3)


def curves(ax, which, src, legend=True):
    thr = m.la.THR
    for s_ in m.SAMPLES:
        for v, ls in (("raw", "-"), ("preprocessed", ":")):
            keys = m.retained(s_, v) if which == "retained" else list(m.runs(s_, v).run_key)
            y = [m.counter(s_, v).count(keys, t) for t in thr]
            ax.plot(thr, y, color=m.SCOL[s_], ls=ls, lw=1.1)
            src += [dict(panel="C" if which == "all48" else "D", sample=m.LAB[s_], input=v, run_set=which,
                         n_registrations=len(keys), threshold=t, count=n) for t, n in zip(thr, y)]
    ax.axvspan(0.25, 0.75, color="0.94", lw=0, zorder=0)
    ax.axvline(0.5, color="0.25", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 7200)
    ax.set_xlabel("inclusion threshold")
    sec = ax.secondary_xaxis("top")
    sec.set_xticks([0.25, 0.5, 0.75])
    sec.tick_params(labelsize=5.8, length=2)
    if legend:
        h = [Line2D([], [], color=m.SCOL[s_], lw=1.3, label=m.LAB[s_].replace("Sample ", "S")) for s_ in m.SAMPLES]
        h += [Line2D([], [], color="0.3", lw=1.1, label="raw"),
              Line2D([], [], color="0.3", ls=":", lw=1.1, label="preprocessed")]
        ax.legend(handles=h, loc="lower left", fontsize=5.6, ncol=2, columnspacing=0.8, handlelength=1.6,
                  handletextpad=0.3)


def panel_d(ax, src):
    for i, s in enumerate(m.SAMPLES):
        for v, c, dx in (("raw", C_RAW, -0.17), ("preprocessed", C_PRE, 0.17)):
            keys = m.retained(s, v)
            cnt = m.counter(s, v)
            c25, c50, c75 = (cnt.count(keys, t) for t in (0.25, 0.5, 0.75))
            singles = m.runs(s, v).set_index("run_key").loc[keys, "snpc_CellCount"].values
            ax.plot([i + dx, i + dx], [c75, c25], color=c, lw=3.2, alpha=0.45, solid_capstyle="butt", zorder=2)
            rng = np.random.default_rng(i * 2 + (v == "raw"))
            ax.scatter(i + dx - 0.07 + rng.uniform(-0.025, 0.025, len(singles)), singles, s=4, color="0.6", lw=0,
                       zorder=2)
            ax.scatter(i + dx, c50, marker="D", s=26, color=c, edgecolor="black", lw=0.7, zorder=3)
            ax.text(i + dx, 1420, "raw" if v == "raw" else "pre", color=c, fontsize=5.6, ha="center")
            src.append(dict(panel="D", sample=m.LAB[s], input=v, n_retained=len(keys), count_0p25=c25,
                            count_0p5=c50, count_0p75=c75, singles_min=singles.min(), singles_median=np.median(singles),
                            singles_max=singles.max()))
    ax.set_xticks(range(5), [m.LAB[s].replace("Sample ", "S") for s in m.SAMPLES])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.55, 4.55)
    ax.set_ylim(1300, 6100)
    ax.set_ylabel("SNpc TH+ cell count")
    ax.grid(axis="y", color="0.93", lw=0.4)
    ax.set_axisbelow(True)
    h = [Line2D([], [], ls="", marker="D", ms=4.5, mfc=C_RAW, mec="black", label="consensus at 0.5, raw"),
         Line2D([], [], ls="", marker="D", ms=4.5, mfc=C_PRE, mec="black", label="consensus at 0.5, preprocessed"),
         Line2D([], [], color="0.6", lw=3, alpha=0.6, label="count at thresholds 0.25–0.75"),
         Line2D([], [], ls="", marker="o", ms=2.5, color="0.6", label="retained single registrations")]
    ax.legend(handles=h, loc="upper left", fontsize=5.8, ncol=2, columnspacing=1.0, handletextpad=0.3)


TH_S, TH_V = "pp451", "raw"                                  # Sample 3, raw: clearest difference no QC vs retained
REL_DA = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\data_availability"))
TH_LEVELS = ((0.0, "#FF5A4E", "> 0.0"), (0.5, "#A7BBD0", "≥ 0.5 (consensus)"),
             (1.0, "#FFE100", "1.0"))


def panel_th(fig, gs, src):
    """Old Fig 3F style: full-resolution TH (inferno) with the SNpc consensus at inclusion > 0, >= 0.5 and 1.0, for all
    48 registrations (A) and the retained registrations (B). Same brain, plane and crop in both panels."""
    lab = m.LAB[TH_S].replace(" ", "")
    d = REL_DA / lab / "fullres_midbrain" / TH_V
    p_all = tifffile.imread(d / "inclusion_frequency_fullres_all48.tif")
    p_ret = tifffile.imread(d / "inclusion_frequency_fullres.tif")
    z = int(np.argmax((p_ret >= 0.5).sum((1, 2))))
    th = m.page(m.row(TH_S, TH_V, m.retained(TH_S, TH_V)[0]).snpc_image_region_path, z).astype(np.float32)
    u = (p_all[z] >= 0.25) | (p_ret[z] > 0)                      # crop: ignore isolated single-registration outliers
    rr, cc = np.flatnonzero(u.any(1)), np.flatnonzero(u.any(0))
    sl = (slice(max(rr[0] - 90, 0), rr[-1] + 91), slice(max(cc[0] - 90, 0), cc[-1] + 91))
    a, b = np.percentile(th[sl], [1, 99.8])
    img = np.clip((th[sl] - a) / (b - a + 1e-9), 0, 1)
    py, px = m.fullres_um(TH_S, TH_V)
    g = gs.subgridspec(1, 2, wspace=0.04)
    for k, (letter, p, title, n) in enumerate((("A", p_all, "No QC (all 48 registrations)", 48),
                                               ("B", p_ret, "Retained registrations", len(m.retained(TH_S, TH_V))))):
        ax = fig.add_subplot(g[k])
        ax.imshow(img, cmap="inferno", interpolation="nearest")
        m.clean(ax)
        for t, col, _ in TH_LEVELS:
            msk = (p[z] > 0) if t == 0 else (p[z] >= t - 1e-6)
            m.contour(ax, msk[sl], col, 1.0)
        m.scalebar(ax, img.shape, 500, px, fs=5.5)
        ax.set_title(f"{title}: {m.LAB[TH_S]}, raw", fontsize=6.8, pad=3, loc="left")
        L(ax, letter, x=-0.01, y=1.02)
        cnt = m.counter(TH_S, TH_V)
        keys = list(m.runs(TH_S, TH_V).run_key) if n == 48 else m.retained(TH_S, TH_V)
        src.append(dict(panel=letter, sample=m.LAB[TH_S], input=TH_V, n_registrations=n, plane=int(z),
                        count_any=cnt.count(keys, 1e-6), count_0p5=cnt.count(keys, 0.5), count_all=cnt.count(keys, 1.0),
                        area_any_px=int((p[z] > 0).sum()), area_0p5_px=int((p[z] >= 0.5).sum()),
                        area_all_px=int((p[z] >= 1 - 1e-6).sum())))
        if k == 0:
            ax.legend(handles=[Line2D([], [], color=c, lw=1.4, label=l) for _, c, l in TH_LEVELS],
                      loc="upper center", bbox_to_anchor=(1.02, -0.02), ncol=3, fontsize=5.8,
                      title="SNpc inclusion frequency", title_fontsize=5.8)


def main():
    src, cur = [], []
    fig = plt.figure(figsize=(7.2, 8.3))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 1.05, 0.85], hspace=0.3)
    panel_th(fig, gs[0], src)
    gb = gs[1].subgridspec(1, 2, wspace=0.08)
    ac = fig.add_subplot(gb[0])
    ad = fig.add_subplot(gb[1], sharey=ac)
    curves(ac, "all48", cur, legend=True)
    ac.set_ylabel("consensus SNpc TH+ cell count")
    ac.set_title("All 48 registrations (no QC)", fontsize=6.8, pad=14, loc="left")
    L(ac, "C", x=-0.12, y=1.06)
    curves(ad, "retained", cur, legend=False)
    plt.setp(ad.get_yticklabels(), visible=False)
    ad.set_title("Retained registrations", fontsize=6.8, pad=14, loc="left")
    L(ad, "D", x=-0.03, y=1.06)
    panel_ab(fig, gs[2], src)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"Fig3_v9.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # supplementary: counts per brain (previous panel E)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    panel_d(ax, src)
    ax.set_title("SNpc TH+ count per brain (retained registrations)", fontsize=6.8, pad=3, loc="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"FigS_consensus_counts_per_brain.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    pd.DataFrame(src).to_csv(SRC / "Fig3_v9_summary.csv", index=False)
    pd.DataFrame(cur).replace({"panel": {"C": "C", "D": "D"}}).to_csv(SRC / "Fig3_v9_CD_curves.csv", index=False)
    print(pd.DataFrame([r for r in src if r.get("panel") in ("A", "B")]).to_string())


if __name__ == "__main__":
    main()
