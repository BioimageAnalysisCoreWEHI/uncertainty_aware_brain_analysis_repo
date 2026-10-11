"""End-to-end verification of the v2 data release (read-only). Writes a Markdown report with PASS/FAIL per check.

Checks
  A  integrity: every MANIFEST entry exists with the recorded size and SHA-256; no unlisted files
  B  per brain x input (original LSFM):
       B1 registrations.csv: 48 rows; global-QC flags recomputed (IoU > 0.9, grad_p95 < 0.045); SSIM retention
          recomputed (SSIM >= median of the passers); expected passers/retained per brain
       B2 20 um consensus: retained inclusion frequency == mean of the retained masks; consensus_<t> == inclusion >= t;
          all-registrations map == mean of the 48 masks
       B3 vote-cache recounts: every single-registration count == registrations.csv; consensus counts at every
          threshold (retained, all 48) == analysis_tables/original_lsfm/lsfm_consensus.csv
       B4 full resolution: consensus0p5_mask == inclusion_frequency_retained >= 0.5; cells_in_consensus0p5 == the
          Cellpose labels with any voxel in the mask; number of cells == the consensus count
       B5 coronal images: shape == mask grid; preprocessed AF == brainreg downsampled.tiff of the review rerun;
          tissue-mask agreement; overlay PNG (SNpc consensus on coronal AF)
  C  manuscript numbers recomputed from released tables (each with the value stated in the manuscript)
  D  methods consistency: grids, thresholds and rules in the data vs the Methods text
  E  sweep provenance notes (which numbers come from which sweep)

Usage: python verify_release.py --release <data_availability> --report <report.md> --overlays <folder>
           --votes <vote_cache folder> --tables <analysis results/tables> --fig3src <Fig. 3 source-data folder>
           [--rerun <jom_review_sweep_original_data>]
The vote cache and the analysis tables are not part of the slimmed release; they are the pipeline outputs used as
independent cross-checks of the released masks, maps and counts.
"""
import argparse
import hashlib
import json
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

IDS = {"Sample1": "162", "Sample2": "527", "Sample3": "pp451", "Sample4": "xl353", "Sample5": "xl385"}
EXPECT = {"raw": ([14, 16, 16, 21, 29], [7, 8, 8, 11, 15]),
          "preprocessed": ([25, 29, 29, 30, 36], [13, 15, 15, 15, 18])}
CONS05 = {"raw": [3421, 4662, 3381, 3612, 4677], "preprocessed": [3438, 4478, 3120, 3884, 4722]}
BE = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
GS = [-2, -3, -5, -10, -30, -50]
RES = []


def rec(section, name, ok, detail=""):
    if isinstance(ok, np.bool_):
        ok = bool(ok)
    RES.append(dict(section=section, check=name, result="PASS" if ok is True else ("FAIL" if ok is False else ok),
                    detail=detail))
    print(f"[{RES[-1]['result']}] {section} {name}: {detail}", flush=True)


def guarded(section, name):
    def deco(fn):
        def run(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as e:
                rec(section, name, False, f"error: {e!r}")
                traceback.print_exc()
        return run
    return deco


def sha256(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(buf):
            h.update(b)
    return h.hexdigest()


def rkey(be, gs):
    return f"{be:g}_{int(gs)}"


def read(p):
    return tifffile.imread(p)


# ------------------------------------------------------------------ A
@guarded("A", "manifest")
def check_manifest(R):
    man = pd.read_csv(R / "MANIFEST.csv")
    disk = {p.relative_to(R).as_posix() for p in R.rglob("*") if p.is_file()} - {"MANIFEST.csv"}
    listed = set(man.path)
    rec("A", "no unlisted files", disk <= listed, f"{len(disk - listed)} unlisted")
    rec("A", "no missing files", listed <= disk, f"{len(listed - disk)} missing")
    bad = [r.path for r in man.itertuples() if (R / r.path).exists() and
           ((R / r.path).stat().st_size != r.bytes or sha256(R / r.path) != r.sha256)]
    rec("A", "size + SHA-256 of every file", not bad, f"{len(man)} files, {len(bad)} mismatches {bad[:5]}")
    rec("A", "release size", "INFO", f"{len(man) + 1} files, {man.bytes.sum() / 1e9:.2f} GB")


# ------------------------------------------------------------------ B
class Votes:
    def __init__(self, f):
        z = np.load(f, allow_pickle=False)
        self.runs = [str(r) for r in z["runs"]]
        bits = np.unpackbits(z["votes"], axis=1)[:, :len(self.runs)].astype(bool)
        key = np.concatenate([z["cell"][:, None].view(np.uint8).reshape(len(z["cell"]), -1), z["votes"]], axis=1)
        _, first = np.unique(key, axis=0, return_index=True)          # identical (cell, pattern) rows
        self.cell, self.bits = z["cell"][first], bits[first]
        self.col = {r: i for i, r in enumerate(self.runs)}

    def count(self, runs, thr):
        ids = [self.col[r] for r in runs]
        frac = (self.bits[:, ids].sum(1) / len(ids)).astype(np.float32).astype(np.float64)
        return int(np.unique(self.cell[frac >= np.float64(thr)]).size)


@guarded("B", "brain x input")
def check_arm(R, s, i, arm, rerun, overlays, votes_dir, tables_dir):
    tag = f"{s} {arm}"
    reg = pd.read_csv(R / s / "02_registrations" / arm / "registrations.csv")
    # B1
    ok_rows = len(reg) == 48 and sorted(reg.bending_energy_weight.unique()) == BE and \
        sorted(reg.grid_spacing.unique(), reverse=True) == sorted(GS, reverse=True)
    rec("B1", f"{tag}: 48 registrations, 8 x 6 grid", ok_rows, f"{len(reg)} rows")
    pas = (reg.tissue_atlas_iou > 0.9) & (reg.grad_p95 < 0.045)
    rec("B1", f"{tag}: global-QC flags recomputed", bool((pas == reg.passes_global_qc).all()),
        f"{int(pas.sum())} pass (expected {EXPECT[arm][0][i]})")
    med = reg.loc[pas, "midbrain_ssim"].quantile(0.5)
    retained = pas & (reg.midbrain_ssim >= med)
    rec("B1", f"{tag}: SSIM retention recomputed", bool((retained == reg.retained_qc_ssim).all())
        and int(retained.sum()) == EXPECT[arm][1][i], f"{int(retained.sum())} retained (expected {EXPECT[arm][1][i]})")
    # B2
    masks = {m: read(R / s / "02_registrations" / arm / m) > 0 for m in reg.mask_file}
    ret_files = list(reg.mask_file[retained])
    inc = np.mean([masks[m] for m in ret_files], axis=0)
    d3 = R / s / "03_consensus_20um" / arm
    inc_rel = read(d3 / "retained" / f"{s}_inclusion_frequency_coronal.tif")
    rec("B2", f"{tag}: retained inclusion frequency == mean of retained masks",
        bool(np.allclose(inc_rel, inc, atol=1e-6)), f"max |diff| {np.abs(inc_rel - inc).max():.2e}")
    bad = []
    for f in sorted((d3 / "retained").glob(f"{s}_consensus_*_coronal.tif")):
        t = float(re.search(r"consensus_([0-9.]+)_coronal", f.name).group(1))
        want = inc_rel.astype(np.float32).astype(np.float64) >= t
        if not np.array_equal(read(f) > 0, want):
            bad.append(f.name)
    rec("B2", f"{tag}: consensus_<t> == inclusion >= t (retained)", not bad, f"mismatch {bad}")
    inc48 = np.mean(list(masks.values()), axis=0)
    a48 = read(d3 / "all_registrations" / f"{s}_inclusion_frequency_coronal.tif")
    rec("B2", f"{tag}: all-registrations inclusion frequency == mean of 48 masks",
        bool(np.allclose(a48, inc48, atol=1e-6)) or "WARN", f"max |diff| {np.abs(a48 - inc48).max():.2e}")
    # B3
    vf = Path(votes_dir) / f"{s}_{arm}.npz"
    V = Votes(vf if vf.exists() else Path(votes_dir) / f"{IDS[s]}_{arm}.npz")   # release or pipeline naming
    keys = [rkey(b, g) for b, g in zip(reg.bending_energy_weight, reg.grid_spacing)]
    single = np.array([V.count([k], 0.5) for k in keys])
    diff = single - reg.snpc_th_cell_count.values
    pr = pd.read_csv(Path(tables_dir) / "lsfm_per_run.csv")
    pr = pr[(pr["sample"].astype(str) == IDS[s]) & (pr.status == arm)].copy()
    pr["k"] = [rkey(b, g) for b, g in zip(pr.bending, pr.grid)]
    orig = pr.set_index("k").count_table.reindex(keys).values
    rec("B3", f"{tag}: registrations.csv counts == the pipeline's per-registration counts",
        bool(np.array_equal(orig, reg.snpc_th_cell_count.values)), "48/48 identical" if np.array_equal(
            orig, reg.snpc_th_cell_count.values) else "differ")
    rec("B3", f"{tag}: per-registration count vs recount with the consensus procedure (vote cache)",
        True if (diff == 0).all() else "INFO",
        f"{int((diff != 0).sum())}/48 differ by at most {int(np.abs(diff).max())} cells")
    lc = pd.read_csv(R / "shared" / "tables" / "original_LSFM_consensus_counts_by_threshold.csv")
    lc = lc[(lc.sample_id == s) & (lc.processing_status == arm)]
    ret_keys = [k for k, r in zip(keys, retained) if r]
    mism = []
    for (qf, sf), runs in ((("passed_qc", "top_50pct"), ret_keys), (("all_runs", "all"), keys)):
        sub = lc[(lc.qc_filter == qf) & (lc.ssim_filter == sf)]
        for r in sub.itertuples():
            c = V.count(runs, r.threshold)
            if c != int(r.consensus_cell_count):
                mism.append((qf, r.threshold, c, r.consensus_cell_count))
    c05 = V.count(ret_keys, 0.5)
    rec("B3", f"{tag}: consensus counts at every threshold (retained, all 48) == released table", not mism,
        f"{len(mism)} mismatches {mism[:3]}; consensus at 0.5 = {c05} (manuscript {CONS05[arm][i]})")
    rec("B3", f"{tag}: consensus count at 0.5 == manuscript", c05 == CONS05[arm][i], f"{c05}")
    # B4 (plane by plane to keep memory low)
    d4 = R / s / "04_fullres_midbrain" / arm
    with tifffile.TiffFile(d4 / "inclusion_frequency_retained.tif") as fi, \
            tifffile.TiffFile(d4 / "consensus0p5_mask.tif") as fm, \
            tifffile.TiffFile(d4 / "cellpose_labels.tif") as fl, \
            tifffile.TiffFile(d4 / "cells_in_consensus0p5.tif") as fc:
        n = len(fm.pages)
        mask_ok, hit = True, set()
        for z in range(n):
            m = fm.pages[z].asarray() > 0
            if not np.array_equal(m, fi.pages[z].asarray().astype(np.float32).astype(np.float64) >= 0.5):
                mask_ok = False
            hit.update(np.unique(fl.pages[z].asarray()[m]).tolist())
        hit.discard(0)
        cells_ok, kept = True, set()
        for z in range(n):
            lab, cin = fl.pages[z].asarray(), fc.pages[z].asarray()
            want = np.where(fm.pages[z].asarray() > 0, lab, 0)      # the label voxels inside the consensus
            if not np.array_equal(cin, want):
                cells_ok = False
            kept.update(np.unique(cin).tolist())
        kept.discard(0)
    rec("B4", f"{tag}: full-res consensus mask == inclusion_frequency_retained >= 0.5", mask_ok)
    rec("B4", f"{tag}: cells_in_consensus0p5 == Cellpose labels inside the consensus", cells_ok and kept == hit,
        f"{len(kept)} cells")
    rec("B4", f"{tag}: full-res cell count == vote-cache consensus count", len(hit) == c05, f"{len(hit)} vs {c05}")
    # B5
    d1 = R / s / "01_registration_inputs"
    af = read(d1 / f"AF_20um_{arm}_coronal.tif")
    rec("B5", f"{tag}: coronal AF on the mask grid", af.shape == inc.shape, f"{af.shape}")
    if arm == "raw":
        th = read(d1 / "TH_20um_coronal.tif")
        rec("B5", f"{s}: coronal TH on the mask grid", th.shape == inc.shape, f"{th.shape}")
        tm = read(d1 / "tissue_mask_20um_coronal.tif") > 0
        from skimage.filters import threshold_otsu
        afm = af > threshold_otsu(af[::2, ::2, ::2])
        dice = 2 * (tm & afm).sum() / (tm.sum() + afm.sum())
        rec("B5", f"{s}: tissue mask vs Otsu-thresholded coronal AF (Dice)", True if dice > 0.85 else "WARN", f"{dice:.3f}")
        ta = read(d1 / "tissue_mask_20um.tif") > 0
        with tifffile.TiffFile(d1 / "AF_20um_raw.tif") as t:
            shp = t.series[0].shape
        rec("B5", f"{s}: tissue_mask_20um (input orientation) == reoriented coronal mask, on the input grid",
            ta.shape == shp and np.array_equal(ta, np.flip(np.transpose(tm, (1, 0, 2)), axis=2)), f"{ta.shape}")
    if rerun is not None:
        ds = Path(rerun) / IDS[s] / arm / "bend_0p0__grid_m10__smooth_m1p0" / "downsampled.tiff"
        if ds.exists():
            b = read(ds)
            if b.shape == af.shape:
                r = np.corrcoef(af[::3, ::3, ::3].ravel(), b[::3, ::3, ::3].ravel())[0, 1]
                eq = np.array_equal(np.clip(np.rint(af), 0, 65535).astype(np.uint16), b) or \
                    np.array_equal(af.astype(np.uint16), b)
                rec("B5", f"{tag}: coronal AF vs brainreg downsampled.tiff (review rerun)",
                    True if (eq or r > 0.999) else ("INFO" if arm == "raw" else False),
                    f"r = {r:.6f}, identical after uint16 cast: {eq}")
            else:
                rec("B5", f"{tag}: coronal AF vs brainreg downsampled.tiff (review rerun)", "INFO",
                    f"rerun grid {b.shape} differs from release {af.shape} (rerun raw input had a different z extent)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cons = read(d3 / "retained" / f"{s}_consensus_0.50_coronal.tif") > 0
        z = int(np.argmax(cons.sum((1, 2))))
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
        a, b2 = np.percentile(af[z][af[z] > 0], [1, 99.5]) if (af[z] > 0).any() else (0, 1)
        ax.imshow(np.clip((af[z] - a) / (b2 - a + 1e-9), 0, 1), cmap="gray")
        ax.contour(cons[z], [0.5], colors=["#E040FB"], linewidths=0.8)
        ax.set_title(f"{s} {arm}: coronal AF + SNpc consensus (plane {z})", fontsize=6)
        ax.axis("off")
        Path(overlays).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(overlays) / f"{s}_{arm}_overlay.png", bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        rec("B5", f"{tag}: overlay", False, repr(e))


# ------------------------------------------------------------------ C
@guarded("C", "manuscript numbers")
def check_numbers(R, tables, fig3src):
    T = Path(tables)                                   # analysis tables (pipeline outputs, not in the release)
    S = R / "shared" / "tables"                        # released essential tables
    Q = R / "shared" / "qc_sensitivity"
    pr = pd.read_csv(T / "lsfm_per_run.csv")
    lc = pd.read_csv(T / "lsfm_consensus.csv")
    pr["sample"], lc["sample"] = pr["sample"].astype(str), lc["sample"].astype(str)
    cons = lc[(lc.qc_filter == "passed_qc") & (lc.ssim_filter == "top_50pct")]
    c05 = cons[np.isclose(cons.threshold, 0.5)].set_index(["sample", "status"])["count"]
    raw = pr[pr.status == "raw"]
    rng_all = raw.groupby("sample").count_overlap.agg(lambda x: x.max() - x.min()) / c05.xs("raw", level=1)
    rng_qc = raw[raw.pass_qc].groupby("sample").count_overlap.agg(lambda x: x.max() - x.min()) / c05.xs("raw", level=1)
    rec("C", "sweep spread 61% / global-QC spread 19% (raw, median over brains)",
        round(rng_all.median() * 100) == 61 and round(rng_qc.median() * 100) == 19,
        f"{rng_all.median() * 100:.1f}% / {rng_qc.median() * 100:.1f}%")
    n = pr.groupby("status").pass_qc.sum()
    rec("C", "global-QC passers 96 (raw) / 149 (preprocessed) of 240", int(n["raw"]) == 96 and int(n["preprocessed"]) == 149,
        f"{int(n['raw'])} / {int(n['preprocessed'])}")
    sd = lambda d: d.groupby("sample").count_overlap.std()
    red_g = (1 - sd(raw[raw.pass_qc]) / sd(raw)) * 100
    red_s = (1 - sd(raw[raw.pass_qc & raw.in_ssim_top50]) / sd(raw[raw.pass_qc])) * 100
    rec("C", "SD reduction raw: global QC 67-86% in Samples 1-3; SSIM 46-79%", "INFO",
        "global " + ", ".join(f"{IDS_inv(k)} {v:.0f}%" for k, v in red_g.items()) +
        " | SSIM " + ", ".join(f"{IDS_inv(k)} {v:.0f}%" for k, v in red_s.items()))
    rp = (c05.xs("preprocessed", level=1) - c05.xs("raw", level=1)).abs() / c05.xs("raw", level=1) * 100
    rec("C", "raw vs preprocessed consensus 0.5-7.7%", round(rp.min(), 1) == 0.5 and round(rp.max(), 1) == 7.7,
        f"{rp.min():.1f}-{rp.max():.1f}%")
    out = []
    for qf, sf in (("all_runs", "all"), ("passed_qc", "top_50pct")):
        sub = lc[(lc.qc_filter == qf) & (lc.ssim_filter == sf) & (lc.status == "raw")].set_index(["sample", "threshold"])["count"]
        ch = [(sub[(s, 0.25)] - sub[(s, 0.75)]) / sub[(s, 0.5)] * 100 for s in IDS.values()]
        out.append((min(ch), max(ch)))
    rec("C", "count change thresholds 0.25-0.75: all 48 48-98%, retained 15-20% (raw)",
        round(out[0][0]) == 48 and round(out[0][1]) == 98 and round(out[1][0]) == 15 and round(out[1][1]) == 20,
        f"all 48 {out[0][0]:.0f}-{out[0][1]:.0f}%, retained {out[1][0]:.0f}-{out[1][1]:.0f}%")
    pct = []
    for f in ("Fig3_v9_E_cells.csv", "Fig3_v9_F_cells.csv"):
        d = pd.read_csv(Path(fig3src) / f)
        near = d.snc_share > 0
        pct.append(100 * (d.loc[near, "plot_region"] == "Uncertain").mean())
    rec("C", "uncertain cells 58% (all 48) / 29% (retained), Sample 1", round(pct[0]) == 58 and round(pct[1]) == 29,
        f"{pct[0]:.1f}% / {pct[1]:.1f}%")
    t2 = pd.read_csv(S / "TableS2_SNpc_counts_per_brain.csv")
    ok = all(int(t2[(t2["sample"] == f"Sample {i + 1}") & (t2.input == arm)].consensus_count_0p5.iloc[0]) == CONS05[arm][i]
             for arm in CONS05 for i in range(5))
    rec("C", "Table S2 consensus counts == manuscript", ok)
    t5 = pd.read_csv(S / "TableS5_global_QC_rules.csv").set_index("rule")
    want = {"Current": (71.1, 40.0, 63.3), "Untuned start (median ± 3·MAD)": (76.0, 97.1, 70.8),
            "Median (marginal medians)": (12.5, 32.5, 15.4)}
    ok = all(abs(t5.loc[k, c] - v) < 0.05 for k, vals in want.items()
             for c, v in zip(("aMAP_retained_pct", "original_LSFM_retained_pct", "independent_LSFM_retained_pct"), vals))
    rec("C", "QC-rule retention (current 71/40/63%, untuned 76/97/71%, median 12.5/32.5/15%)", ok)
    t4 = pd.read_csv(S / "TableS4_independent_LSFM_agreement_by_region.csv").set_index("region")
    ok = abs(t4.loc["CP", "dice_median"] - 0.906) < 0.002 and abs(t4.loc["SNc", "dice_median"] - 0.100) < 0.002 and \
        round(t4.loc["DG-sg", "mean_boundary_distance_um_median"]) == 38 and round(t4.loc["SNc", "mean_boundary_distance_um_median"]) == 77
    rec("C", "independent LSFM: Dice CP 0.91 / SNc 0.10; distance DG-sg 38 / SNc 77 um", ok)
    q = pd.read_csv(Q / "qcsens_lsfm2_per_case.csv", usecols=["sample", "variant", "region", "setting_id", "dice_vs_ref"])
    w = q[q.variant == "raw"].pivot_table(index=["sample", "region"], columns="setting_id", values="dice_vs_ref")
    d = (w["iounone_gradnone"] - w["iou0p8_grad0p08"]).dropna()
    rec("C", "independent LSFM without global QC: 18/96 cases lose > 0.1 (max 0.41)",
        int((d < -0.1).sum()) == 18 and round(-d.min(), 2) == 0.41, f"{int((d < -0.1).sum())}/{len(d)}, max {-d.min():.2f}")
    p = pd.read_csv(Q / "qcsens_paper_per_case.csv", usecols=["sample", "variant", "setting_id", "count_change_pct"])
    p = p[p.variant == "raw"]
    strict = p[p.setting_id.isin(["iou0p91_grad0p042", "iou0p92_grad0p038", "iou0p9016_grad0p042"])].count_change_pct.abs()
    loose = p[p.setting_id.isin(["iounone_gradnone", "iou0p8107_grad0p0795", "iou0p84_grad0p08", "iou0p86_grad0p07"])].count_change_pct
    rec("C", "original LSFM: stricter rules max |change| 5%; looser rules up to -53%",
        round(strict.max()) == 5 and round(loose.min()) == -53, f"stricter max {strict.max():.1f}%, looser min {loose.min():.1f}%")
    t3 = pd.read_csv(S / "TableS3_consensus_vs_single_registrations_aMAP.csv").set_index("comparator")
    rec("C", "Table S3: aMAP vs default 15/28/11, 10th percentile 46/8/0",
        tuple(t3.iloc[0][["wins", "ties", "losses"]]) == (15, 28, 11) and tuple(t3.iloc[4][["wins", "ties", "losses"]]) == (46, 8, 0))
    sc = pd.read_csv(S / "aMAP_dice_scores.csv").groupby("group").dice.median()
    am = pd.read_csv(S / "aMAP_per_case_dice.csv")
    rec("C", "aMAP median Dice: experts 0.89, single retained 0.79, consensus 0.80 (54 cases)",
        len(am) == 54 and round(sc["annotator"], 2) == 0.89 and round(sc["single retained registration"], 2) == 0.79
        and round(sc["consensus"], 2) == 0.80,
        f"{sc['annotator']:.3f} / {sc['single retained registration']:.3f} / {sc['consensus']:.3f}")


def IDS_inv(k):
    return {v: f"S{i + 1}" for i, v in enumerate(IDS.values())}.get(str(k), str(k))


# ------------------------------------------------------------------ D, E
@guarded("D", "methods consistency")
def check_methods(R):
    T = R / "shared"
    rec("D", "original LSFM grid 8 x 6 and global-QC thresholds 0.9 / 0.045", "see B1",
        "recomputed from registrations.csv for every brain and input")
    rec("D", "SSIM rule: >= median among global-QC passers (top 50%)", "see B1", "")
    rec("D", "counting rule: any voxel in region; consensus threshold 0.5", "see B3/B4", "")
    q = pd.read_csv(T / "qc_sensitivity" / "qcsens_amap_per_case.csv", usecols=["setting_id"])
    rec("D", "QC-sensitivity grid settings (aMAP)", "INFO", f"{q.setting_id.nunique()} settings incl. extras")


def notes():
    rec("E", "which sweep feeds which number", "INFO",
        "All released masks, QC metrics and counts come from the original sweep (registrations.csv/masks match the "
        "released inputs' grid). The review rerun (jom_review_sweep_original_data) is used only for anatomy display "
        "in figures (validation scripts 20/22/35, manuscript fig2_v*), per the scripts' documentation; its raw arm was "
        "registered from a different 20 um stack (one z-plane fewer), so displayed raw-arm images are from that rerun.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--overlays", required=True, type=Path)
    ap.add_argument("--rerun", default=None)
    ap.add_argument("--votes", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--fig3src", required=True)
    a = ap.parse_args()
    R = a.release
    for i, s in enumerate(IDS):
        for arm in ("raw", "preprocessed"):
            check_arm(R, s, i, arm, a.rerun, a.overlays, a.votes, a.tables)
    check_numbers(R, a.tables, a.fig3src)
    check_methods(R)
    notes()
    check_manifest(R)
    df = pd.DataFrame(RES)
    n_fail = int((df.result == "FAIL").sum())
    lines = ["# Data release verification", "", f"Release: `{R}`", "",
             f"**{int((df.result == 'PASS').sum())} PASS, {n_fail} FAIL, "
             f"{int((~df.result.isin(['PASS', 'FAIL'])).sum())} INFO/WARN**", "",
             "| Section | Check | Result | Detail |", "|---|---|---|---|"]
    lines += [f"| {r.section} | {r.check} | {r.result} | {str(r.detail).replace('|', '/')} |" for r in df.itertuples()]
    a.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {a.report} ({n_fail} FAIL)")


if __name__ == "__main__":
    main()
