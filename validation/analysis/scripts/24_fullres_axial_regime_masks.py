"""Original LSFM SNpc: the QC-sensitivity consensus regions (Fig. S3b rules) mapped into the full-resolution axial
crop of the cellpose segmentation, the cellpose labels masked by them, and the cell counts recomputed from those files.

The mapping copies the manuscript pipeline exactly (<IMAGING>/Ishrat/JoM/analysis_scripts):
  union_region_extraction_parallel.py  coronal 20 um mask -> axial (np.flip(np.transpose(m, (1, 0, 2)), axis=2)) ->
                                        crop with the bbox of the union mask (<s>_consensus_0.0x_axial.tif, label 1,
                                        z padded by 10) -> skimage resize to the raw crop shape, order=0
  extract_counts_python_cp_parallel.py masked labels = cellpose labels * region mask (uint16)
  get_consensus_counts.py              a cell counts if ANY of its voxels is inside the mask
The consensus regions themselves are the existing QC-sensitivity outputs (results/qc_sensitivity/paper/consensus/...,
<s>_consensus_0.50_coronal.tif), one per rule (REGIMES of script 22). Nothing upstream is modified.

Checks (all must pass; written to the output folder):
  A  the transform reproduces the pipeline's own per-registration mask_region_downsampled / mask_region_resized files
     exactly (3 registrations per sample x input);
  B  Current: the resampled consensus equals the pipeline's own consensus (get_consensus_counts.py probability >= 0.5)
     voxel for voxel;
  C  every rule: the count from the masked labels equals the count in qcsens_paper_per_case.csv (Fig. S3b).

Output: <IMAGING>/Ishrat/JoM/post_review_analysis/QC_REGIME_FULLRES_AXIAL_MASKS/<sample>/<input>/<rule>/
   <sample>_<input>_<rule>_consensus0p5_mask_fullres_axial.tif   (uint8 0/1, zlib)
   <sample>_<input>_<rule>_masked_cells.tif                       (uint16 cellpose labels * mask, zlib)
 + counts.csv, checks.csv, README.md
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import ast
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.io
import skimage.measure
import skimage.transform
import tifffile

from common import VA

JOM = Path(P(r"<IMAGING>\Ishrat\JoM"))
OUT = JOM / "post_review_analysis" / "QC_REGIME_FULLRES_AXIAL_MASKS"
QSP = VA / "results" / "qc_sensitivity" / "paper"
TABLE = JOM / "final_count_results" / "full_final_results_with_ssim.csv"
OWN_CONS = JOM / "final_count_results" / "consensus_threshold_cell_counts_overlap_logic"
SAMPLES = ["162", "527", "pp451", "xl353", "xl385"]
INPUTS = ["raw", "preprocessed"]
SAFE = {"No global QC": "no_global_QC", "Very lenient": "very_lenient", "Lenient": "lenient",
        "Untuned start (median ± 3·MAD)": "untuned_start", "Current": "current", "Strict": "strict",
        "Very strict": "very_strict", "Median IoU and grad_p95": "median"}


def regimes():
    tree = ast.parse((VA / "scripts" / "22_focused_revision_figures.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "REGIMES" for t in node.targets):
            return ast.literal_eval(node.value)


def label_path(s, v):
    return (JOM / f"results_{v}" / "cellpose_masked_cells" / "for_cellpose" /
            f"{s}_AF_level1_preprocessed_freeform_bending_0.0_grid_-2_smoothing_-1_atlas_perens_lsfm_mouse_20um" /
            "results_python" / "snpc" / "whole_segmentation_python.tif")


def union_bbox(s, v):
    """union_region_extraction_parallel.py: bbox of (union == 1), z padded by 10."""
    d = JOM / f"results_{v}" / f"param_sweep_masks_{v}_no_padding_consensus" / f"{s}_snpc_consensus_masks"
    f = sorted(d.glob(f"{s}_consensus_0.0*_axial.tif"))
    assert len(f) == 1, f"union mask for {s} {v}: {f}"
    union = skimage.io.imread(f[0])
    props = skimage.measure.regionprops((union == 1).astype(np.uint8))
    bbox = list(props[0].bbox)
    bbox[0] = max(0, bbox[0] - 10)
    bbox[3] = bbox[3] + 10
    return bbox, f[0].name, union.shape


def to_fullres(coronal, bbox, crop_shape):
    """union_region_extraction_parallel.py steps 4 (axial, crop, resize)."""
    ax = np.flip(np.transpose(coronal, (1, 0, 2)), axis=2)
    cropped = ax[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
    resized = skimage.transform.resize(cropped, crop_shape, order=0, preserve_range=True).astype(np.uint8)
    return cropped, resized


def job(args):
    s, v = args
    t0 = time.time()
    reg = regimes()
    checks, counts = [], []
    labels = tifffile.imread(label_path(s, v))
    bbox, union_name, union_shape = union_bbox(s, v)
    tab = pd.read_csv(TABLE)
    tab = tab.loc[:, ~tab.columns.str.contains("^Unnamed")]
    tab["Sample ID"] = tab["Sample ID"].astype(str)
    sub = tab[tab["Sample ID"] == s].sort_values(["bending", "grid"]).reset_index(drop=True)
    # ---- A: reproduce the pipeline's own per-registration files
    for i in (0, len(sub) // 2, len(sub) - 1):
        r = sub.iloc[i]
        cor = skimage.io.imread(r[f"{v}_snpc_mask_path"])
        cropped, resized = to_fullres(cor, bbox, labels.shape)
        run_dir = Path(r[f"{v}_mask_axial_path"]).parent
        own_res = skimage.io.imread(r[f"{v}_mask_axial_path"])
        own_dn = skimage.io.imread(run_dir / f"{s}_snpc_mask_region_downsampled.tif")
        checks.append(dict(check="A transform reproduces own downsampled + resized", sample=s, input=v,
                           item=run_dir.name, ok=bool(np.array_equal(cropped, own_dn) and
                                                      np.array_equal(resized, own_res)),
                           detail=f"downsampled equal {np.array_equal(cropped, own_dn)}, resized equal "
                                  f"{np.array_equal(resized, own_res)}; union file {union_name}, bbox {bbox}"))
    # ---- rules
    lk = pd.read_csv(QSP / "setting_lookup.csv")
    lk["sample"] = lk["sample"].astype(str)
    per = pd.read_csv(VA / "results" / "tables" / "qcsens_paper_per_case.csv")
    per["sample"] = per["sample"].astype(str)
    for rule, ids in reg.items():
        sid = ids["paper"]
        row = lk[(lk["sample"] == s) & (lk.variant == v) & (lk.setting_id == sid)].iloc[0]
        ref_c = per[(per["sample"] == s) & (per.variant == v) & (per.setting_id == sid)]["count"].iloc[0]
        if not isinstance(row.set_path, str):                      # no registration passes this rule
            checks.append(dict(check="C count from masked labels = Fig. S3b count", sample=s, input=v, item=rule,
                               ok=bool(pd.isna(ref_c) and row.n_final == 0),
                               detail="no registration passes: no consensus, no count (also NaN in Fig. S3b)"))
            counts.append(dict(sample=s, input=v, rule=rule, setting_id=sid, n_registrations=0,
                               source_consensus="none (no registration passes)", cell_labels=str(label_path(s, v)),
                               union_mask=union_name, bbox_axial_20um=str(bbox), fullres_shape=str(labels.shape),
                               mask_voxels=0, cell_count=np.nan, fig_S3b_count=ref_c, folder="none"))
            continue
        cons_cor = skimage.io.imread(QSP / row.set_path / f"{s}_consensus_0.50_coronal.tif")
        cons_cor = (cons_cor > 0).astype(np.uint8)
        _, mask = to_fullres(cons_cor, bbox, labels.shape)
        masked = (labels * mask).astype(np.uint16)                 # extract_counts_python_cp_parallel.py
        ids_in = np.unique(masked)
        n = int((ids_in != 0).sum())                               # = get_consensus_counts.count_cells_in_mask
        ref = int(per[(per["sample"] == s) & (per.variant == v) & (per.setting_id == sid)]["count"].iloc[0])
        checks.append(dict(check="C count from masked labels = Fig. S3b count", sample=s, input=v, item=rule,
                           ok=n == ref, detail=f"{n} vs {ref}"))
        if rule == "Current":
            own = tifffile.imread(OWN_CONS / s / v / "passed_qc_top_50pct" /
                                  f"{s}_{v}_passed_qc_top_50pct_consensus_probability.tif")
            own_mask = (own >= 0.5).astype(np.uint8)
            diff = int((own_mask != mask).sum())
            checks.append(dict(check="B Current = own get_consensus_counts consensus (prob >= 0.5), voxelwise",
                               sample=s, input=v, item=rule, ok=diff == 0,
                               detail=f"{diff} differing voxels of {mask.size}; own count "
                                      f"{int((np.unique(labels[own_mask > 0]) != 0).sum())}"))
        d = OUT / s / v / SAFE[rule]
        d.mkdir(parents=True, exist_ok=True)
        stem = f"{s}_{v}_{SAFE[rule]}"
        tifffile.imwrite(d / f"{stem}_consensus0p5_mask_fullres_axial.tif", mask, compression="zlib")
        tifffile.imwrite(d / f"{stem}_masked_cells.tif", masked, compression="zlib")
        counts.append(dict(sample=s, input=v, rule=rule, setting_id=sid, n_registrations=int(row.n_final),
                           source_consensus=str(QSP / row.set_path / f"{s}_consensus_0.50_coronal.tif"),
                           cell_labels=str(label_path(s, v)), union_mask=union_name, bbox_axial_20um=str(bbox),
                           fullres_shape=str(labels.shape), mask_voxels=int(mask.sum()), cell_count=n,
                           fig_S3b_count=ref, folder=str(d)))
    print(f"{s} {v}: done ({time.time() - t0:.0f}s)", flush=True)
    return checks, counts


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, v) for s in SAMPLES for v in INPUTS]
    if len(sys.argv) > 1:                                          # e.g. "162 raw" for a single test
        jobs = [tuple(sys.argv[1:3])]
    with ProcessPoolExecutor(max_workers=3) as ex:
        res = list(ex.map(job, jobs))
    checks = pd.DataFrame([c for a, _ in res for c in a])
    counts = pd.DataFrame([c for _, b in res for c in b])
    tag = "" if len(jobs) == 10 else "_test"
    checks.to_csv(OUT / f"checks{tag}.csv", index=False)
    counts.to_csv(OUT / f"counts{tag}.csv", index=False)
    print(checks.groupby("check").ok.agg(["sum", "count"]).to_string())
    print(counts[["sample", "input", "rule", "cell_count", "fig_S3b_count"]].to_string(index=False))
    if not checks.ok.all():
        print(checks[~checks.ok].to_string(index=False))
        sys.exit(1)
