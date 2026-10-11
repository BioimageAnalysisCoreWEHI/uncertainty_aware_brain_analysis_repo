"""Quality control of the consensus segmentations: morphology table and overlay figures.

* ``morphology.csv`` - per brain x structure: connected components, holes, solidity, area relative to the raters,
  containment in the rater union, centroid offset from the raters, and hemisphere of the centroid in the
  downscaled brain (via geometry.StackMapping at the median rater slice).
* ``staple_outlines.png`` - every rater outline (grey) with the STAPLE (solid) and SBA (dashed) consensus.
* ``brain_<brain>.png`` - STAPLE outlines mapped onto the downscaled brain, one coronal plane per stack.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import masks as M
from . import nifti
from .config import FLIPPED_HEMISPHERE, MIDLINES, Config, matlab_num2str
from .geometry import StackMapping, canvas_to_volume, matlab_round

STAPLE_COLOR = "#2a78d6"
SBA_COLOR = "#eb6834"
RATER_COLOR = "#9a9a9a"
PAPER_NAME = {"CING": "ACA", "S1BF": "SSp", "RSP": "RSP", "V1": "VISp", "V2M": "VISam", "SUB": "SUB",
              "AHN": "AHN", "MV": "MV", "VPM": "VPM", "SG": "DG-sg"}
REGION_ORDER = ["CING", "S1BF", "RSP", "V1", "V2M", "SUB", "AHN", "MV", "VPM", "SG"]


def _load_run(opts):
    out = opts.staple_dir()
    table = pd.read_csv(out / "segTable_with_pq.csv")
    canvas = tuple(json.loads((out / "canvas.json").read_text())["maxX_maxY"])
    return out, table[table["inStaple"]], canvas


def _stack_of(cur) -> str:
    # 'Seg37AB_Gad67_223_rspV1V2M.tif001-...' -> 'rspV1V2M'
    name = cur["path"].iloc[0].replace("\\", "/").split("/")[-1]
    return name[: name.index(".tif")].rsplit("_", 1)[1]


def morphology(cfg: Config, opts, log=print) -> pd.DataFrame:
    out, table, canvas = _load_run(opts)
    sc = matlab_num2str(cfg.scale)
    cache = opts.cache_dir(cfg.scale)
    rows = []
    for brc, cur in table.groupby("brainRegionComb"):
        st = nifti.load(out / f"{brc}_{sc}_NiftySegSTAPLE.nii") > 0
        raters = np.stack([M.cached_masks(p, canvas, cfg.scale, cache)[0] > 0 for p in cur["path"]])
        lab, ncomp = ndi.label(st, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())[1:] if ncomp else np.array([0])
        filled = ndi.binary_fill_holes(st)
        area = int(st.sum())
        rater_areas = raters.sum((1, 2))
        solidity = np.nan
        if area:
            from skimage.morphology import convex_hull_image
            sl = ndi.find_objects(st.astype(int))[0]
            solidity = area / convex_hull_image(st[sl]).sum()
        c_st = np.array(ndi.center_of_mass(st)) if area else np.array([np.nan, np.nan])
        c_r = np.median([ndi.center_of_mass(r) for r in raters if r.any()], axis=0)

        brain, region = cur["brain"].iloc[0], cur["region"].iloc[0]
        stack = _stack_of(cur)
        mapping = StackMapping(cfg.stack(brain, stack), cfg.scale)
        s_med = float(np.median(cur["sliceNum"]))
        nii_c = mapping.staple_to_nii(c_st[0], c_st[1], s_med) if area else [np.nan] * 3
        expected_high_x = (brain, region) not in FLIPPED_HEMISPHERE
        side_ok = (nii_c[0] + 1 > MIDLINES[brain]) == expected_high_x if area else False
        rows.append(dict(
            brainRegionComb=brc, brain=brain, region=region, structure=PAPER_NAME[region], stack=stack,
            n=len(cur), area_px=area, area_vs_median_rater=area / np.median(rater_areas),
            components=int(ncomp), largest_component_frac=float(sizes.max() / max(area, 1)),
            hole_px=int(filled.sum() - area), solidity=solidity,
            inside_rater_union=float((st & raters.any(0)).sum() / max(area, 1)),
            centroid_offset_um=float(np.hypot(*(c_st - c_r)) / cfg.scale),
            nii_centroid_xyz=tuple(np.round(nii_c, 1)), expected_hemisphere=side_ok,
        ))
    df = pd.DataFrame(rows)
    qc_dir = out / "qc"
    qc_dir.mkdir(exist_ok=True)
    df.to_csv(qc_dir / "morphology.csv", index=False)
    log(f"  wrote {qc_dir / 'morphology.csv'}")
    return df


def _contours(mask_xy, **kw):
    from skimage import measure
    return measure.find_contours(np.pad(mask_xy.T.astype(float), 1), 0.5, **kw)


def outline_figure(cfg: Config, opts, log=print, brain: str | None = None, dpi: int = 110):
    """All brains in one grid (default), or one brain on a larger 2 x 5 layout (``brain=...``)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out, table, canvas = _load_run(opts)
    sc = matlab_num2str(cfg.scale)
    cache = opts.cache_dir(cfg.scale)
    if brain is None:
        brains = list(cfg.brains)
        fig, axes = plt.subplots(len(brains), len(REGION_ORDER),
                                 figsize=(2.1 * len(REGION_ORDER), 2.2 * len(brains)), squeeze=False)
        panels = [(axes[bi, ri], b, r, bi == 0, ri == 0) for bi, b in enumerate(brains)
                  for ri, r in enumerate(REGION_ORDER)]
        title_fs, note_fs = 10, 6
    else:
        fig, axes = plt.subplots(2, 5, figsize=(22, 10.5))
        panels = [(ax, brain, r, True, False) for ax, r in zip(axes.ravel(), REGION_ORDER)]
        title_fs, note_fs = 13, 9
    for ax, brain_i, region, show_title, show_brain in panels:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#dddddd")
        brc = brain_i + region
        cur = table[table["brainRegionComb"] == brc]
        if show_title:
            ax.set_title(PAPER_NAME[region], fontsize=title_fs)
        if show_brain:
            ax.set_ylabel(brain_i, fontsize=8)
        if cur.empty:
            continue
        raters = [M.cached_masks(p, canvas, cfg.scale, cache)[0] > 0 for p in cur["path"]]
        st = nifti.load(out / f"{brc}_{sc}_NiftySegSTAPLE.nii") > 0
        sb = nifti.load(out / f"{brc}_{sc}_NiftySegSBA.nii") > 0
        # Crop to the consensus plus a margin so single aberrant outlines don't zoom the panel out.
        xs, ys = np.nonzero(st if st.any() else np.any(raters, axis=0))
        mx = max(25, int(0.5 * (xs.max() - xs.min())))
        my = max(25, int(0.5 * (ys.max() - ys.min())))
        x0, x1 = max(0, xs.min() - mx), min(st.shape[0], xs.max() + mx)
        y0, y1 = max(0, ys.min() - my), min(st.shape[1], ys.max() + my)
        clipped = sum(bool(r.any() and (r.sum() - r[x0:x1, y0:y1].sum()) > 0) for r in raters)
        # Contour a slightly larger window so outlines leaving the crop aren't closed along its edge.
        wx0, wy0 = max(0, x0 - 4), max(0, y0 - 4)
        wx1, wy1 = min(st.shape[0], x1 + 4), min(st.shape[1], y1 + 4)
        dx, dy = x0 - wx0 + 1, y0 - wy0 + 1  # +1: _contours pads by one pixel

        def draw(mask, **kw):
            for c in _contours(mask[wx0:wx1, wy0:wy1]):
                ax.plot(c[:, 1] - dx, c[:, 0] - dy, **kw)

        for r in raters:
            draw(r, color=RATER_COLOR, lw=0.5, alpha=0.8)
        draw(sb, color=SBA_COLOR, lw=1.2, ls=(0, (3, 2)))
        draw(st, color=STAPLE_COLOR, lw=1.6)
        ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0)  # image orientation: dorsal up
        ax.set_aspect("equal")
        note = f"n={len(cur)}, {(x1 - x0) / cfg.scale / 1000:.1f} mm"
        if clipped:
            note += f", {clipped} clipped"
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=note_fs, color="#555555")
    handles = [Line2D([], [], color=RATER_COLOR, lw=0.8, label="rater outlines"),
               Line2D([], [], color=STAPLE_COLOR, lw=1.6, label="STAPLE consensus"),
               Line2D([], [], color=SBA_COLOR, lw=1.2, ls=(0, (3, 2)), label="SBA consensus")]
    who = "" if brain is None else f"{brain}: "
    fig.suptitle(f"{who}Consensus vs rater outlines (4 µm canvas, dorsal up, each panel cropped around the "
                 f"consensus)", y=0.995, fontsize=title_fs + 1)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=3, frameon=False,
               fontsize=title_fs)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    name = "staple_outlines.png" if brain is None else f"staple_outlines_{brain}.png"
    path = out / "qc" / name
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    log(f"  wrote {path}")


def brain_figures(cfg: Config, opts, log=print):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out, table, _ = _load_run(opts)
    sc = matlab_num2str(cfg.scale)
    for brain in cfg.brains:
        vol = nifti.load(cfg.brain_file(brain), mmap=True)
        bt = table[table["brain"] == brain]
        stacks = sorted({_stack_of(g) for _, g in bt.groupby("brainRegionComb")})
        fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
        for ax, stack in zip(axes.ravel(), stacks):
            combos = [(brc, g) for brc, g in bt.groupby("brainRegionComb") if _stack_of(g) == stack]
            mapping = StackMapping(cfg.stack(brain, stack), cfg.scale)
            s_med = float(np.median(np.concatenate([g["sliceNum"].to_numpy() for _, g in combos])))
            # background: the nii plane the (slightly tilted) stack plane crosses at the brain's centre column
            y = int(matlab_round(mapping.plane_y(vol.shape[0] / 2, vol.shape[2] / 2, s_med)))
            plane = np.asarray(vol[:, y, :], dtype=float)
            lo, hi = np.percentile(plane, [1, 99.5])
            ax.imshow(plane.T, cmap="gray", origin="lower", vmin=lo, vmax=hi, interpolation="nearest")
            ax.axvline(MIDLINES[brain] - 1, color="#ffffff", lw=0.6, ls=(0, (2, 3)), alpha=0.7)
            for brc, g in combos:
                st = nifti.load(out / f"{brc}_{sc}_NiftySegSTAPLE.nii")
                vox, _ = canvas_to_volume(st, mapping, s_med, vol.shape)
                if not len(vox):
                    continue
                pm = np.zeros((vol.shape[0], vol.shape[2]), bool)  # (x, z) projection of the tilted plane
                pm[vox[:, 0], vox[:, 2]] = True
                ax.contour(pm.T.astype(float), levels=[0.5], colors=[STAPLE_COLOR], linewidths=1.4, origin="lower")
                zx = np.argwhere(pm).mean(0)
                ax.text(zx[0], zx[1], PAPER_NAME[g["region"].iloc[0]], color="#ffffff", fontsize=8,
                        ha="center", va="center", fontweight="bold")
            ax.set_title(f"{stack}  (nii plane y={y}, rater slice {s_med:g})", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        for ax in axes.ravel()[len(stacks):]:
            ax.axis("off")
        fig.suptitle(f"{brain}: STAPLE consensus mapped into the downscaled brain (12.5 µm, dorsal up; "
                     f"dashed = code midline)", fontsize=11)
        fig.tight_layout()
        path = out / "qc" / f"brain_{brain}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        log(f"  wrote {path}")


# Sequential blue ramp (steps 100 -> 700 of the reference palette) for probability maps.
PROB_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]


def inbrain_figure(cfg: Config, opts, brc: str, log=print):
    """Read back the brain-space files of one brain x structure and plot them on the downscaled brain."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    out = opts.staple_dir()
    dest = out / "inBrainSpace"
    sc = matlab_num2str(cfg.scale)
    info = json.loads((dest / f"{brc}_{sc}_STAPLE_inBrain.json").read_text())
    raters = json.loads((dest / "raters" / brc / "raters.json").read_text())["raters"]
    brain = info["brain"]
    vol = nifti.load(cfg.brain_file(brain), mmap=True)
    nx, ny, nz = vol.shape
    st_vox, _ = nifti.load_sparse(dest / f"{brc}_{sc}_STAPLE_inBrain.nii.gz")
    pr_vox, pr_val = nifti.load_sparse(dest / f"{brc}_{sc}_raterProbability_inBrain.nii.gz")
    y = int(np.median(st_vox[:, 1]))
    plane = np.asarray(vol[:, y, :], dtype=float)
    lo_i, hi_i = np.percentile(plane, [1, 99.5])
    st2 = np.zeros((nx, nz), bool)
    st2[st_vox[:, 0], st_vox[:, 2]] = True
    pr2 = np.zeros((nx, nz), float)
    pr2[pr_vox[:, 0], pr_vox[:, 2]] = pr_val
    own = []
    for r in raters:
        v, _ = nifti.load_sparse(dest / "raters" / brc / r["ownSlice_file"])
        m = np.zeros((nx, nz), bool)
        m[v[:, 0], v[:, 2]] = True
        own.append((r, m, float(np.median(v[:, 1])) if len(v) else np.nan))

    cmap = LinearSegmentedColormap.from_list("prob", PROB_RAMP)
    prm = np.ma.masked_where(pr2 <= 0, pr2)
    fig = plt.figure(figsize=(17, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.05, 0.8])
    ax_a, ax_b, ax_c = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    xs, zs = np.nonzero(st2 | (pr2 > 0))
    pad = 40
    x0, x1, z0, z1 = max(0, xs.min() - pad), min(nx, xs.max() + pad), max(0, zs.min() - pad), min(nz, zs.max() + pad)
    for ax, (a0, a1, b0, b1) in ((ax_a, (0, nx, 0, nz)), (ax_b, (x0, x1, z0, z1))):
        ax.imshow(plane.T, cmap="gray", origin="lower", vmin=lo_i, vmax=hi_i, interpolation="nearest")
        im = ax.imshow(prm.T, cmap=cmap, origin="lower", vmin=0, vmax=1, alpha=0.85, interpolation="nearest")
        ax.contour(st2.T.astype(float), levels=[0.5], colors=[SBA_COLOR], linewidths=1.6, origin="lower")
        ax.set_xlim(a0, a1); ax.set_ylim(b0, b1)
        ax.set_xticks([]); ax.set_yticks([])
    for _, m, _ in own:
        if m.any():
            ax_b.contour(m.T.astype(float), levels=[0.5], colors=[RATER_COLOR], linewidths=0.7, origin="lower")
    ax_a.set_title(f"coronal plane y={y} (12.5 µm, dorsal up)", fontsize=10)
    ax_b.set_title("zoom: + each rater on own slice (grey, projected)", fontsize=10)
    cb = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.02)
    cb.set_label("rater agreement (fraction of raters)", fontsize=9)
    ys = np.array([yy for _, _, yy in own])
    bins = np.arange(np.nanmin(ys) - 1.5, np.nanmax(ys) + 2.5, 1)
    ax_c.hist(ys, bins=bins, color=PROB_RAMP[4], edgecolor="#ffffff", linewidth=2)
    ax_c.axvline(y, color=SBA_COLOR, lw=2)
    q = info["slice_stats_nii_y"]
    ax_c.axvspan(q["q25"], q["q75"], color="#dddddd", zorder=0)
    ax_c.set_xlabel("nii plane y of each rater's own slice (12.5 µm)")
    ax_c.set_ylabel("raters")
    from matplotlib.ticker import MaxNLocator
    ax_c.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax_c.set_title("anterior-posterior spread of the raters", fontsize=10)
    for s in ("top", "right"):
        ax_c.spines[s].set_visible(False)
    handles = [Line2D([], [], color=SBA_COLOR, lw=1.6, label=f"STAPLE consensus (median slice {info['J0_median_slice']:g})"),
               Line2D([], [], color=RATER_COLOR, lw=0.8, label="rater outline (own slice)"),
               Line2D([], [], color="#dddddd", lw=8, label="rater slice IQR")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"{brc} in {info['brain_file']} space — read back from the written .nii.gz files", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    path = out / "qc" / f"inBrain_{brc}.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    log(f"  wrote {path}")


def asr25_figure(cfg: Config, opts, brain: str, gzip: bool = False, log=print):
    """Each structure on its asr 25 um coronal slice (read back from files): probability + STAPLE outline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from .reslice import AsrLayout
    lay = AsrLayout(opts.staple_dir() / "asr25um", gzip)
    img = nifti.load(lay.brain_img(brain), mmap=not gzip)
    rows = pd.read_csv(lay.slices_log)
    rows = rows[rows["brain"] == brain].sort_values("label_id")
    raters = pd.read_csv(lay.raters_log) if lay.raters_log.exists() else None
    cmap = LinearSegmentedColormap.from_list("prob", PROB_RAMP)
    fig, axes = plt.subplots(2, 5, figsize=(22, 9.5))
    for ax, row in zip(axes.ravel(), rows.itertuples()):
        sl = np.asarray(img[row.asr25_slice_A], dtype=float)  # (S, R): superior at the top
        m = nifti.load(lay.dest / row.staple_file)[0] > 0
        p = nifti.load(lay.dest / row.probability_file)[0]
        own_note = ""
        if raters is not None:
            rr_rows = raters[raters["brainRegionComb"] == row.brainRegionComb]
            for f in rr_rows[rr_rows["variant"] == "medianPlane"]["file"]:
                rm = nifti.load(lay.dest / f)[0] > 0
                if rm.any():
                    ax.contour(rm.astype(float), levels=[0.5], colors=[RATER_COLOR], linewidths=0.6)
            own = rr_rows[rr_rows["variant"] == "ownSlice"]["asr25_slice_A"]
            if len(own):
                own_note = f"; raters' own slices A={own.min()}–{own.max()}"
        ss, rr = np.nonzero(m | (p > 0))
        pad = 60
        s0, s1 = max(0, ss.min() - pad), min(sl.shape[0], ss.max() + pad)
        r0, r1 = max(0, rr.min() - pad), min(sl.shape[1], rr.max() + pad)
        lo, hi = np.percentile(sl, [1, 99.5])
        ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax.imshow(np.ma.masked_where(p <= 0, p), cmap=cmap, vmin=0, vmax=1, alpha=0.8, interpolation="nearest")
        ax.contour(m.astype(float), levels=[0.5], colors=[SBA_COLOR], linewidths=1.5)
        ax.set_xlim(r0, r1); ax.set_ylim(s1, s0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{row.paper_name} (ID {row.label_id}): slice A={row.asr25_slice_A}{own_note}", fontsize=10)
        ax.text(0.02, 0.02, f"area ratio {row.area_ratio:.3f}", transform=ax.transAxes, color="#ffffff", fontsize=8)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    fig.suptitle(f"{brain}: asr 25 µm coronal slices (superior up, R increasing to the right; left-right mirrored, "
                 f"segmented hemisphere on the left); probability = blue ramp, STAPLE = orange, "
                 f"raters on the STAPLE slice = grey", fontsize=12)
    fig.tight_layout()
    path = opts.staple_dir() / "qc" / f"asr25_{brain}.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    log(f"  wrote {path}")


RAW_COLOR, PRE_COLOR, EXPERT_COLOR, PIPE_OUTLINE = "#2a78d6", "#eb6834", "#1baf7a", "#104281"
STRUCT_ORDER = ["ACA", "SSp", "RSP", "VISp", "VISam", "SUB", "AHN", "MV", "VPM", "DG-sg"]


def comparison_figures(cfg: Config, opts, t: dict, fig_dir, lay, pipeline_root, log=print):
    """Figures for the pipeline-vs-expert validation (from the tables written by compare.run_compare)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from pathlib import Path
    from .compare import FINAL_SET, read_pages, set_dir, to_our_grid, hemisphere_keep
    from .reslice import AsrGrid

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    main, human, runs, rel = (t.get(k, pd.DataFrame()) for k in ("main", "human", "runs", "rel"))
    if not len(main):
        return
    rng = np.random.default_rng(0)
    structs = [s for s in STRUCT_ORDER if s in set(main.structure)]
    final = main[(main.qc_set == FINAL_SET) & (main.threshold == 0.5)]

    def style(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6", lw=0.6)
        ax.set_axisbelow(True)

    # 1. Dice per structure: humans vs pipeline (paper protocol)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for k, st in enumerate(structs):
        h = human[human.structure == st].dice
        ax.scatter(k - 0.25 + rng.uniform(-0.08, 0.08, len(h)), h, s=10, color="#9a9a9a", alpha=0.7, lw=0)
        if len(h):
            ax.plot([k - 0.37, k - 0.13], [h.median()] * 2, color="#555555", lw=2.5, solid_capstyle="round")
        for dx, variant, col, mk in ((0.05, "raw", RAW_COLOR, "o"), (0.25, "preprocessed", PRE_COLOR, "s")):
            v = final[(final.structure == st) & (final.variant == variant)].dice_paper
            ax.scatter(k + dx + rng.uniform(-0.04, 0.04, len(v)), v, s=34, color=col, marker=mk,
                       edgecolor="#ffffff", lw=1, zorder=3)
            if len(v):
                ax.plot([k + dx - 0.09, k + dx + 0.09], [v.median()] * 2, color=col, lw=2.5, solid_capstyle="round")
    ax.set_xticks(range(len(structs)))
    ax.set_xticklabels(structs)
    ax.set_ylabel("Dice vs expert STAPLE consensus")
    ax.set_ylim(0, 1.02)
    style(ax)
    ax.legend(handles=[Line2D([], [], color="#9a9a9a", marker="o", ls="", label="human raters (25 µm)"),
                       Line2D([], [], color=RAW_COLOR, marker="o", ls="", label="pipeline, raw input"),
                       Line2D([], [], color=PRE_COLOR, marker="s", ls="", label="pipeline, preprocessed input")],
              loc="lower left", frameon=False, ncol=3)
    ax.set_title("Pipeline consensus (final QC, threshold 0.5, best slice in the raters' stack) vs human raters; "
                 "bars = medians", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "dice_by_structure.png", dpi=130)
    plt.close(fig)

    # 2. threshold sweep
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5), sharex=True, sharey=True)
    fm = main[main.qc_set == FINAL_SET]
    for ax, st in zip(axes.ravel(), structs):
        for variant, col, ls in (("raw", RAW_COLOR, "-"), ("preprocessed", PRE_COLOR, "--")):
            for _, g in fm[(fm.structure == st) & (fm.variant == variant)].groupby("brain"):
                g = g.sort_values("threshold")
                ax.plot(g.threshold, g.dice_paper, color=col, ls=ls, lw=1.2, alpha=0.8, marker="o", ms=3)
        hm = human[human.structure == st].dice.median()
        ax.axhline(hm, color="#555555", lw=1, ls=":")
        ax.set_title(st, fontsize=10)
        style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("consensus threshold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Dice")
    fig.legend(handles=[Line2D([], [], color=RAW_COLOR, label="raw (one line per brain)"),
                        Line2D([], [], color=PRE_COLOR, ls="--", label="preprocessed"),
                        Line2D([], [], color="#555555", ls=":", label="human median")],
               loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(fig_dir / "threshold_sweep.png", dpi=120)
    plt.close(fig)

    # 3. individual registrations vs consensus, by QC stage (raw)
    if len(runs):
        fig, ax = plt.subplots(figsize=(12, 4.8))
        stage_cols = {"all_registrations": "#bdbdbd", "passes_global_qc": "#86b6ef", FINAL_SET: RAW_COLOR}
        for k, st in enumerate(structs):
            for dx, stage in zip((-0.25, 0, 0.25), stage_cols):
                v = runs[(runs.structure == st) & (runs.variant == "raw") & (runs.qc_set == stage)].dice_paper
                ax.scatter(k + dx + rng.uniform(-0.07, 0.07, len(v)), v, s=6, color=stage_cols[stage], lw=0)
            c = final[(final.structure == st) & (final.variant == "raw")].dice_paper
            ax.scatter(np.full(len(c), k + 0.4), c, marker="D", s=30, color="#0d366b", edgecolor="#ffffff", zorder=3)
        ax.set_xticks(range(len(structs)))
        ax.set_xticklabels(structs)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Dice vs expert STAPLE")
        style(ax)
        ax.legend(handles=[Line2D([], [], color=c, marker="o", ls="", label=s.replace("_", " ")) for s, c in stage_cols.items()]
                  + [Line2D([], [], color="#0d366b", marker="D", ls="", label="final consensus (0.5)")],
                  loc="lower left", frameon=False, ncol=4)
        ax.set_title("Individual registrations by QC stage vs the final consensus (raw input)", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / "qc_stages_individual_runs.png", dpi=130)
        plt.close(fig)

    # 4. calibration (reliability) pooled over brains and structures (excl. DG-sg)
    if len(rel):
        fig, ax = plt.subplots(figsize=(5.2, 5))
        ax.plot([0, 1], [0, 1], color="#bdbdbd", lw=1)
        r = rel[rel.structure != "DG-sg"]
        for variant, col, mk in (("raw", RAW_COLOR, "o"), ("preprocessed", PRE_COLOR, "s")):
            g = r[r.variant == variant].groupby(["p_bin_lo", "p_bin_hi"]).apply(
                lambda d: pd.Series(dict(mean_p=np.average(d.mean_p.fillna(0), weights=d.n_pixels) if d.n_pixels.sum() else np.nan,
                                         frac=np.average(d.frac_expert.fillna(0), weights=d.n_pixels) if d.n_pixels.sum() else np.nan)),
                include_groups=False).dropna()
            ax.plot(g.mean_p, g.frac, color=col, marker=mk, lw=1.5, label=variant)
        ax.set_xlabel("pipeline inclusion probability (bin mean)")
        ax.set_ylabel("fraction inside expert consensus")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        style(ax)
        ax.legend(frameon=False)
        ax.set_title("Calibration of the pipeline probability map", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / "calibration.png", dpi=130)
        plt.close(fig)

    # 5. overlays per brain (raw, final QC): brain slice at A*, probability, pipeline 0.5 outline, expert outline
    cmap = LinearSegmentedColormap.from_list("prob", PROB_RAMP)
    for brain in sorted(main.brain.unique()):
        h = nifti.read_header(cfg.brain_file(brain))
        grid = AsrGrid(h["shape"], h["sform"])
        img = nifti.load(lay.brain_img(brain), mmap=True)
        sl_rows = pd.read_csv(lay.slices_log)
        fig, axes = plt.subplots(2, 5, figsize=(22, 9.5))
        rows = final[(final.brain == brain) & (final.variant == "raw")]
        qc_set = FINAL_SET
        if not len(rows):  # ER: no final QC set
            rows = main[(main.brain == brain) & (main.variant == "raw") & (main.threshold == 0.5)
                        & (main.qc_set == "all_registrations")]
            qc_set = "all_registrations"
        for ax, st in zip(axes.ravel(), structs):
            r = rows[rows.structure == st]
            if not len(r):
                ax.axis("off")
                continue
            r = r.iloc[0]
            srow = sl_rows[(sl_rows.brain == brain) & (sl_rows.paper_name == st)].iloc[0]
            truth = nifti.load(lay.dest / srow.staple_file)[0] > 0
            A = int(r.A_best)
            d = set_dir(pipeline_root, brain, "raw", srow.region, qc_set)
            if r.get("hemisphere", "amap") == "registered":
                from .compare import RegisteredHemispheres, _run_dirs
                hem = RegisteredHemispheres(pipeline_root, brain, "raw", _run_dirs(d), [A], truth.shape)
                keep2d = hem.vote[0] == int(r.hemisphere_label)
            else:
                keep2d = np.broadcast_to(hemisphere_keep(grid, brain, srow.region)[None, :], truth.shape)
            p = to_our_grid(read_pages(d / "probability.tif", [A])[0], truth.shape) * keep2d
            sl = np.asarray(img[A], dtype=float)
            ss, rr = np.nonzero(truth | (p > 0))
            pad = 50
            s0, s1 = max(0, ss.min() - pad), min(sl.shape[0], ss.max() + pad)
            r0, r1 = max(0, rr.min() - pad), min(sl.shape[1], rr.max() + pad)
            lo, hi = np.percentile(sl, [1, 99.5])
            ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            ax.imshow(np.ma.masked_where(p <= 0, p), cmap=cmap, vmin=0, vmax=1, alpha=0.7, interpolation="nearest")
            ax.contour((p >= 0.5).astype(float), levels=[0.5], colors=[PIPE_OUTLINE], linewidths=1.4)
            ax.contour(truth.astype(float), levels=[0.5], colors=[EXPERT_COLOR], linewidths=1.6)
            ax.set_xlim(r0, r1); ax.set_ylim(s1, s0)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{st}: Dice {r.dice_paper:.3f}, best slice A={A} (expert A0={int(r.A0)})", fontsize=10)
        mode = "registered hemispheres" if "hemisphere" in rows and (rows.hemisphere == "registered").any() else "aMAP midline"
        fig.suptitle(f"{brain} ({qc_set}, raw input, hemisphere: {mode}): pipeline probability (blue ramp) and 0.5 "
                     f"consensus (dark blue) vs expert STAPLE (green) on the pipeline's best-matching coronal slice",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(fig_dir / f"overlay_{brain}.png", dpi=100)
        plt.close(fig)
    log(f"  wrote figures to {fig_dir}")


def run_qc(cfg: Config, opts, log=print):
    morphology(cfg, opts, log)
    outline_figure(cfg, opts, log)
    for brain in cfg.brains:
        outline_figure(cfg, opts, log, brain=brain, dpi=130)
    brain_figures(cfg, opts, log)
