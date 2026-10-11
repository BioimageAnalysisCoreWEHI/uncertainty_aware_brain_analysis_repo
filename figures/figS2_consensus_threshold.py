"""Supplementary Fig. S2 (v2): effect of the consensus threshold (aMAP), revised for Pradeep's comment C2426.
  A  aMAP brain 165, RSP: inclusion frequency of the retained registrations as a magma heatmap (truncated so that low
     frequencies stay visible); consensus boundaries at 0.25 (dotted), 0.5 (solid) and 0.75 (dashed) in cyan, and the
     expert STAPLE consensus in green, so that no outline shares a colour with the heatmap.
  B  Dice with the expert STAPLE consensus against the consensus threshold over the 54 brain x structure cases:
     median (solid) with interquartile range (dark band) and mean (dashed) with +/- 1 SD (light band).
Same data and case as 22_focused_revision_figures.figS1 (the v1 figure); helpers imported from that script.
Output: manuscript_figures/FigS2_consensus_threshold_v2.{png,pdf,svg}, source_data_figS2/FigS2_v2_B_dice_vs_threshold.csv
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P  # noqa: E402  (resolves <IMAGING>/<SCRATCH> placeholders)
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = next(_p for _p in Path(__file__).resolve().parents if (_p / "uaa_paths.py").exists())
_CODE = _REPO / "validation" / "analysis" / "scripts"          # analysis scripts in this repository
VS = _CODE
sys.path.insert(0, str(VS))
_s = importlib.util.spec_from_file_location("f22", str(VS / "22_focused_revision_figures.py"))
f22 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(f22)
plt = f22.plt
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\manuscript_figures"))
SRC = OUT / "source_data_figS2"
SRC.mkdir(exist_ok=True)
C_EXP, C_LINE = "#39D353", "#7B3294"
C_THR = {0.25: "#00E5FF", 0.5: "white", 0.75: "#3D7BFF"}       # none of these occur in the magma heatmap
HEAT = ListedColormap(plt.get_cmap("magma")(np.linspace(0.18, 0.97, 256)))


def main():
    b, s = "MV_Ntsr1_165", "RSP"
    cs, used, k, dk = f22.amap_case(b, s)
    v = f22.amap_view(cs, used, k)
    fig = plt.figure(figsize=(6.6, 2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.62)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(v["img"], cmap="gray", interpolation="nearest")
    im = ax.imshow(np.ma.masked_where(v["p"] <= 0, v["p"]), cmap=HEAT, vmin=0, vmax=1, alpha=0.85,
                   interpolation="nearest")
    for t, ls in ((0.75, "--"), (0.5, "-"), (0.25, ":")):          # 0.25 drawn last so it stays visible on top
        f22.contour(ax, v["p"] >= t - 1e-9, C_THR[t], 1.4 if t == 0.25 else 1.2, ls)
    f22.contour(ax, v["truth"], C_EXP, 1.3)
    f22.clean(ax)
    f22.scalebar(ax, v["img"].shape[0])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("inclusion frequency", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    leg = ax.legend(handles=[Line2D([], [], color=C_THR[0.25], ls=":", lw=1.4, label="threshold 0.25"),
                             Line2D([], [], color="0.25", lw=1.2, label="threshold 0.5 (used; white)"),
                             Line2D([], [], color=C_THR[0.75], ls="--", lw=1.2, label="threshold 0.75"),
                             Line2D([], [], color=C_EXP, lw=1.3, label="expert consensus (STAPLE)")],
                    loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=6, frameon=False,
                    handlelength=2.4)
    ax.set_title(f"{s}, {f22.sname('aMAP', b)} (as in Fig. 4)", fontsize=7, pad=2)
    f22.label(ax, "A")

    sets = pd.read_csv(f22.TABLES / "amap_sets.csv")
    sets = sets[(sets.variant == "raw") & (sets.set_name == "passes_final_qc_top_0p5") &
                sets.structure.isin(f22.STRUCTS_MAIN)]
    g = sets.groupby("threshold").dice_paper
    q = pd.DataFrame(dict(n=g.size(), q1=g.quantile(0.25), med=g.median(), q3=g.quantile(0.75),
                          mean=g.mean(), sd=g.std())).reset_index()
    ax = fig.add_subplot(gs[0, 1])
    ax.fill_between(q.threshold, (q["mean"] - q.sd).clip(0, 1), (q["mean"] + q.sd).clip(0, 1),
                    color=C_LINE, alpha=0.10, lw=0)
    ax.fill_between(q.threshold, q.q1, q.q3, color=C_LINE, alpha=0.28, lw=0)
    ax.plot(q.threshold, q.med, color=C_LINE, lw=1.5)
    ax.plot(q.threshold, q["mean"], color=C_LINE, lw=1.0, ls="--")
    ax.axvline(0.5, color="0.5", lw=0.7, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("consensus threshold (inclusion frequency)")
    ax.set_ylabel("Dice with expert consensus\n(54 brain × structure cases)")
    ax.legend(handles=[Line2D([], [], color=C_LINE, lw=1.5, label="median"),
                       Patch(color=C_LINE, alpha=0.28, lw=0, label="interquartile range"),
                       Line2D([], [], color=C_LINE, lw=1.0, ls="--", label="mean"),
                       Patch(color=C_LINE, alpha=0.10, lw=0, label="mean ± SD")],
              loc="lower left", bbox_to_anchor=(0.02, 0.02), fontsize=5.8, frameon=False, ncol=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    f22.label(ax, "B", x=-0.22)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"FigS2_consensus_threshold_v2.{ext}", dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    q.to_csv(SRC / "FigS2_v2_B_dice_vs_threshold.csv", index=False)
    print(q[q.threshold.round(2).isin([0.25, 0.5, 0.75])].round(3).to_string())


if __name__ == "__main__":
    main()
