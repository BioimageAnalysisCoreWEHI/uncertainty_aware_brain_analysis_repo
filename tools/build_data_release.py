"""Build the registration-level data release for the 5 original LSFM brains (one folder, ready for Zenodo).

Sources are read only; nothing is modified or moved. Every released file gets one row in a provenance table that is
written OUTSIDE the release folder (it contains internal paths and the internal sample IDs).

    python tools/build_data_release.py --out <IMAGING>/Ishrat/JoM/post_review_analysis/data_availability --dry-run
    python tools/build_data_release.py --out ... [--only Sample1] [--skip-hash]

Release layout (see the README written into the folder):
    SampleN/inputs/                 20 um registration inputs of the original sweep + tissue mask used for IoU
    SampleN/fullres_midbrain/<arm>/ full-resolution TH crop, Cellpose labels, crop metadata
    SampleN/registrations/<arm>/    48 SNpc masks (20 um, coronal) + registrations.csv (QC metrics, counts, flags)
    SampleN/consensus/<arm>/        pipeline probability (inclusion-frequency) and consensus maps
    cellpose_model/, expert_annotations/, derived/ (tables), README.md, MANIFEST.csv, LICENSE, samples.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uaa_paths import P  # noqa: E402

JOM = Path(P("<IMAGING>/Ishrat/JoM"))
PRA = JOM / "post_review_analysis"
WBP = Path(P("<IMAGING>/Ishrat/analysis_of_whole_brain_projects/results"))
VA = PRA / "validation_analysis"
LSF = PRA / "validation_datasets" / "lightsheet_data"
IDS = ["162", "527", "pp451", "xl353", "xl385"]                        # manuscript Sample 1-5
LABEL = {s: f"Sample{i + 1}" for i, s in enumerate(IDS)}
ARMS = ("raw", "preprocessed")
IOU_T, GRAD_T, SSIM_FRAC = 0.9, 0.045, 0.5
# expert annotators are anonymised: annotator_1, annotator_2, ... in alphabetical order of their source folders
PATHLIKE = re.compile(r"^([A-Za-z]:[\\/]|/vast/|\\\\|/stornext/|<IMAGING>|<SCRATCH>)")


# ------------------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------------------
class Release:
    def __init__(self, out: Path, dry: bool, skip_hash: bool):
        self.out, self.dry, self.skip_hash = out, dry, skip_hash
        self.rows, self.est = [], 0

    def record(self, rel, sources, op, notes="", src_sha=""):
        sources = [str(s) for s in (sources if isinstance(sources, (list, tuple)) else [sources])]
        src_bytes = sum(Path(s).stat().st_size for s in sources if Path(s).is_file())
        dst = self.out / rel
        row = dict(release_path=rel, source_paths=" ; ".join(sources[:3]) + (f" ; ... ({len(sources)} files)"
                                                                               if len(sources) > 3 else ""),
                   n_source_files=len(sources), source_bytes=src_bytes, operation=op, source_sha256=src_sha,
                   release_bytes=dst.stat().st_size if dst.exists() else "", release_sha256="", notes=notes)
        if dst.exists() and not self.skip_hash:
            row["release_sha256"] = sha256(dst)
        self.rows.append(row)
        self.est += src_bytes

    def path(self, rel) -> Path:
        p = self.out / rel
        if not self.dry:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_stack(dst: Path, pages, shape, dtype, meta: dict):
    tifffile.imwrite(dst, data=pages, shape=shape, dtype=dtype, compression="zlib", bigtiff=True,
                     metadata=None, description=json.dumps(meta))


def series_pages(files):
    for f in files:
        yield tifffile.imread(f)


def tiff_pages(src):
    with tifffile.TiffFile(src) as t:
        for pg in t.pages:
            yield pg.asarray()


def tiff_info(src):
    with tifffile.TiffFile(src) as t:
        s = t.series[0]
        return tuple(s.shape), s.dtype


def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    """Drop path columns and replace internal sample IDs by SampleN."""
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")].copy()
    drop = [c for c in df.columns if df[c].dtype == object and df[c].astype(str).str.match(PATHLIKE).any()]
    drop += [c for c in df.columns if re.search(r"path|_dir$|folder", str(c), re.I)]
    df = df.drop(columns=sorted(set(drop)))
    idcols = [c for c in df.columns if str(c).lower() in ("sample", "sample_id", "sample id", "brain", "sample_key")]
    for c in idcols:
        df[c] = df[c].map(lambda v: LABEL.get(str(v), v))
    for c in df.columns:
        if df[c].dtype == object:
            for s, lab in LABEL.items():
                df[c] = df[c].astype(str).str.replace(rf"(?<![A-Za-z0-9]){re.escape(s)}(?=_|$)", lab, regex=True) \
                    if df[c].astype(str).str.contains(s, regex=False).any() else df[c]
    return df


def runs_table() -> pd.DataFrame:
    r = pd.read_csv(VA / "cache" / "lsfm" / "runs.csv")
    r["sample"] = r["sample"].astype(str)
    r["passes_global_qc"] = (r.iou > IOU_T) & (r.grad_p95 < GRAD_T)
    r["retained_qc_ssim"] = False
    for (s, v), g in r.groupby(["sample", "status"]):
        ok = g[g.passes_global_qc].dropna(subset=["ssim_value"]).sort_values("ssim_value", ascending=False)
        keep = ok.index[: max(1, math.ceil(len(ok) * SSIM_FRAC))] if len(ok) else []
        r.loc[keep, "retained_qc_ssim"] = True
    return r


def run_name(b, g):
    return f"be{float(b):g}_gs{int(g)}".replace(".", "p").replace("-", "m")


# ------------------------------------------------------------------------------------------------------------
# per-sample content
# ------------------------------------------------------------------------------------------------------------
def build_sample(R: Release, s: str, runs: pd.DataFrame, vox: dict):
    L = LABEL[s]
    t0 = time.time()
    # ---- inputs: exactly what the original sweep registered (analysis_scripts/submit_brainreg_*_extensive*.sh)
    inputs = [
        ("inputs/AF_20um_raw.tif", JOM / "final_preprocessed_stacks_raw" / f"{s}_AF_level1_preprocessed",
         "registration input, raw arm (brainreg main image)"),
        ("inputs/TH_20um.tif", JOM / "final_preprocessed_stacks_raw" / f"{s}_TH_level1_preprocessed",
         "TH channel at 20 um (brainreg additional image, raw arm)"),
        ("inputs/AF_20um_preprocessed.tif", JOM / "final_preprocessed_stacks" / f"{s}_AF_level1_preprocessed",
         "registration input, preprocessed arm (intensity normalisation and background correction)"),
    ]
    for rel, folder, note in inputs:
        files = sorted(folder.glob("*.tif*"))
        shape0 = tifffile.imread(files[0]).shape
        dt = tifffile.imread(files[0]).dtype
        dst = R.path(f"{L}/{rel}")
        if not R.dry and not dst.exists():
            write_stack(dst, series_pages(files), (len(files),) + shape0, dt,
                        dict(sample=L, voxel_size_um=[20, 20, 20], brainreg_orientation="sal", note=note))
        R.record(f"{L}/{rel}", files, "TIFF series -> single 3D TIFF (zlib)", note)
    m = JOM / "raw_tiff_stacks" / "downsampled" / f"{s}_AF_level1" / f"{s}_AF_level1_mask_coronal_flipped.tif"
    dst = R.path(f"{L}/inputs/tissue_mask_20um_coronal.tif")
    if not R.dry and not dst.exists():
        shp, dt = tiff_info(m)
        write_stack(dst, tiff_pages(m), shp, dt, dict(sample=L, voxel_size_um=[20, 20, 20],
                                                     note="tissue mask used for the tissue-atlas IoU (coronal)"))
    R.record(f"{L}/inputs/tissue_mask_20um_coronal.tif", m, "recompressed (zlib)",
             "tissue mask used by qc_tissue_overlap.py for the tissue-atlas IoU")

    for v in ARMS:
        g = runs[(runs["sample"] == s) & (runs.status == v)].sort_values(["bending", "grid"])
        # ---- full-resolution midbrain crop + Cellpose labels
        crop = Path(g.snpc_image_region_path.iloc[0])
        lab = (JOM / f"results_{v}" / "cellpose_masked_cells" / "for_cellpose" /
               f"{s}_AF_level1_preprocessed_freeform_bending_0.0_grid_-2_smoothing_-1_atlas_perens_lsfm_mouse_20um" /
               "results_python" / "snpc" / "whole_segmentation_python.tif")
        for src, rel, note in ((crop, "TH_crop_fullres.tif", "full-resolution TH, union midbrain crop (identical in "
                                "every registration folder of this arm)"),
                               (lab, "cellpose_labels.tif", "Cellpose TH+ cell labels used for all counts")):
            dst = R.path(f"{L}/fullres_midbrain/{v}/{rel}")
            if not R.dry and not dst.exists():
                shp, dt = tiff_info(src)
                write_stack(dst, tiff_pages(src), shp, dt,
                            dict(sample=L, arm=v, voxel_size_um=[vox[s]["z"], vox[s]["y"], vox[s]["x"]], note=note))
            R.record(f"{L}/fullres_midbrain/{v}/{rel}", src, "recompressed (zlib)", note)
        cm = R.path(f"{L}/fullres_midbrain/{v}/crop_metadata.json")
        if not R.dry:
            meta = crop_metadata(s, v, crop, vox[s])
            cm.write_text(json.dumps(meta, indent=2))
        R.record(f"{L}/fullres_midbrain/{v}/crop_metadata.json", [crop], "written (crop geometry)",
                 "bbox and transform of union_region_extraction_parallel.py, checked against the crop shape")

        # ---- 48 per-registration SNpc masks + table
        for r in g.itertuples():
            src = Path(r.snpc_mask_path)
            rel = f"{L}/registrations/{v}/snpc_mask_20um_coronal/{run_name(r.bending, r.grid)}.tif"
            dst = R.path(rel)
            if not R.dry and not dst.exists():
                shp, dt = tiff_info(src)
                write_stack(dst, tiff_pages(src), shp, dt, dict(sample=L, arm=v, bending_energy_weight=r.bending,
                                                               grid_spacing=r.grid, voxel_size_um=[20, 20, 20]))
            R.record(rel, src, "recompressed (zlib)", f"SNpc mask, bending {r.bending:g}, grid {int(r.grid)}")
        tab = g[["bending", "grid", "iou", "grad_p95", "ssim_value", "snpc_CellCount", "passes_global_qc",
                 "retained_qc_ssim"]].rename(columns={"bending": "bending_energy_weight", "grid": "grid_spacing",
                                                      "iou": "tissue_atlas_iou", "ssim_value": "midbrain_ssim",
                                                      "snpc_CellCount": "snpc_th_cell_count"})
        tab.insert(0, "mask_file", [f"snpc_mask_20um_coronal/{run_name(b, gg)}.tif" for b, gg in zip(g.bending, g.grid)])
        rel = f"{L}/registrations/{v}/registrations.csv"
        if not R.dry:
            tab.to_csv(R.path(rel), index=False)
        R.record(rel, [VA / "cache" / "lsfm" / "runs.csv"], "table (cleaned)",
                 f"global QC IoU>{IOU_T}, grad_p95<{GRAD_T}; retained = top ceil(n/2) by SSIM among passers")

        # ---- consensus: pipeline maps as stored (all masks in the folder) + the QC+SSIM consensus
        cdir = JOM / f"results_{v}" / f"param_sweep_masks_{v}_no_padding_consensus" / f"{s}_snpc_consensus_masks"
        n_dir = len([f for f in Path(g.snpc_mask_path.iloc[0]).parent.glob("*.tif") if "perens" in f.name])
        for f in sorted(cdir.glob("*.tif")):
            rel = f"{L}/consensus/{v}/all_registrations_in_sweep_folder/{f.name.replace(s, L, 1)}"
            dst = R.path(rel)
            if not R.dry and not dst.exists():
                shp, dt = tiff_info(f)
                write_stack(dst, tiff_pages(f), shp, dt, dict(sample=L, arm=v, voxel_size_um=[20, 20, 20],
                                                             n_masks=n_dir))
            R.record(rel, f, "recompressed (zlib)",
                     f"consensus_region_analysis_parallel.py output: mean of all {n_dir} masks in the sweep folder")
        lk = pd.read_csv(VA / "results" / "qc_sensitivity" / "paper" / "setting_lookup.csv")
        lk["sample"] = lk["sample"].astype(str)
        row = lk[(lk["sample"] == s) & (lk.variant == v) & (lk.setting_id == "iou0p9_grad0p045")].iloc[0]
        sdir = VA / "results" / "qc_sensitivity" / "paper" / row.set_path
        for f in sorted(sdir.glob("*.tif")):
            rel = f"{L}/consensus/{v}/qc_ssim_retained/{f.name.replace(s, L, 1)}"
            dst = R.path(rel)
            if not R.dry and not dst.exists():
                shp, dt = tiff_info(f)
                write_stack(dst, tiff_pages(f), shp, dt, dict(sample=L, arm=v, voxel_size_um=[20, 20, 20],
                                                             n_masks=int(row.n_final)))
            R.record(rel, f, "recompressed (zlib)",
                     f"inclusion frequency / 0.5 consensus of the {int(row.n_final)} retained registrations "
                     f"(global QC + top 50% SSIM), from the QC-sensitivity rebuild (17_qc_sensitivity_consensus.py)")
    print(f"  {L} done in {time.time() - t0:.0f}s", flush=True)


def crop_metadata(s, v, crop, vox):
    """Reproduce the crop geometry of union_region_extraction_parallel.py and check it against the stored crop."""
    union = JOM / f"results_{v}" / f"param_sweep_masks_{v}_no_padding_consensus" / f"{s}_snpc_consensus_masks" / \
        f"{s}_union_axial.tif"
    shp, _ = tiff_info(crop)
    meta = dict(sample=LABEL[s], arm=v, crop_shape_zyx=list(shp), fullres_voxel_size_um_zyx=[vox["z"], vox["y"], vox["x"]],
                orientation="full-resolution acquisition planes (horizontal); masks in this space are the 20 um "
                            "coronal masks transformed with np.flip(np.transpose(mask, (1, 0, 2)), axis=2), cropped "
                            "to bbox_20um_axial and resized (nearest neighbour) to crop_shape_zyx")
    if union.exists():
        u = tifffile.imread(union) > 0
        zz, yy, xx = np.nonzero(u)
        bb = [int(zz.min()), int(yy.min()), int(xx.min()), int(zz.max()) + 1, int(yy.max()) + 1, int(xx.max()) + 1]
        bb[0], bb[3] = max(0, bb[0] - 10), bb[3] + 10
        meta.update(bbox_20um_axial_z0y0x0z1y1x1=bb, axial_volume_shape=list(u.shape),
                    note="bbox = union of all SNpc masks of this arm (+/-10 slices in z), as in "
                         "union_region_extraction_parallel.py")
    return meta


# ------------------------------------------------------------------------------------------------------------
# shared content
# ------------------------------------------------------------------------------------------------------------
def build_shared(R: Release, runs: pd.DataFrame):
    m = JOM / "analysis_scripts" / "CP_epochs500_lr0.1_SGD_True_wd_1e-05"
    dst = R.path("cellpose_model/CP_epochs500_lr0.1_SGD_True_wd_1e-05")
    if not R.dry and not dst.exists():
        shutil.copy2(m, dst)
    R.record("cellpose_model/CP_epochs500_lr0.1_SGD_True_wd_1e-05", m, "copy", "custom Cellpose 2.x model (TH)",
             "" if R.skip_hash else sha256(m))
    # expert SNpc annotations (Sample 3), annotators anonymised
    seen = {}
    gt = JOM / "gt_mask"
    people = sorted({d.name.lower().split("_")[0] for d in gt.iterdir() if d.is_dir()})
    ANNOTATORS = {p: f"annotator_{i + 1}" for i, p in enumerate(people)}
    for f in sorted(gt.rglob("*.tif")):
        who = f.relative_to(gt).parts[0].lower().split("_")[0]
        h = sha256(f)
        if h in seen:
            continue
        seen[h] = f
        name = f.name.replace(" ", "_").replace("MASK_", "").replace(who, ANNOTATORS[who])
        rel = f"expert_annotations/{LABEL['pp451']}/{name}"
        dst = R.path(rel)
        if not R.dry and not dst.exists():
            shp, dt = tiff_info(f)
            write_stack(dst, tiff_pages(f), shp, dt, dict(sample=LABEL["pp451"], annotator=ANNOTATORS[who]))
        R.record(rel, f, "recompressed (zlib), annotator renamed", f"expert SNpc annotation ({ANNOTATORS[who]})", h)
    # derived tables
    tables = []
    tables += [(JOM / "final_count_results" / "consensus_threshold_cell_counts_overlap_logic" /
                "ALL_consensus_threshold_cell_counts_summary.csv", "derived/counts/consensus_counts_by_threshold.csv")]
    tables += [(VA / "cache" / "lsfm" / "runs.csv", "derived/counts/all_registrations_qc_and_counts.csv")]
    for f in sorted((VA / "results" / "tables").glob("*.csv")):
        sub = "amap_validation" if f.name.startswith(("amap_", "human_")) else (
            "qc_sensitivity" if f.name.startswith(("qcsens", "fig3_amap_qc")) else "original_lsfm_analysis")
        tables.append((f, f"derived/{sub}/{f.name}"))
    for f in sorted((PRA / "QC_REGIME_FULLRES_AXIAL_MASKS").glob("*.csv")):
        tables.append((f, f"derived/qc_sensitivity/fullres_qc_rules/{f.name}"))
    for f in sorted((LSF / "results" / "validation_gt").glob("*.csv")):
        tables.append((f, f"derived/independent_lsfm_validation/{f.name}"))
    for f in sorted((PRA / "REVISION_OUTPUTS_FOCUSED" / "source_data").glob("*.csv")):
        if "mapping" not in f.name:
            tables.append((f, f"derived/figure_source_data/revision/{f.name}"))
    for f in sorted((PRA / "REVISION_OUTPUTS_FOCUSED" / "new_main_figures" / "source_data").glob("*.csv")):
        tables.append((f, f"derived/figure_source_data/new_main_figures/{f.name}"))
    for src, rel in tables:
        if not src.exists():
            continue
        if not R.dry:
            clean_table(pd.read_csv(src, low_memory=False)).to_csv(R.path(rel), index=False)
        R.record(rel, src, "table (path columns dropped, sample IDs replaced)")
    # vote cache (exact overlap-rule counts for any run subset and threshold)
    for f in sorted((VA / "cache" / "lsfm").glob("*.npz")):
        s, v = f.stem.rsplit("_", 1)
        rel = f"derived/vote_cache/{LABEL[s]}_{v}.npz"
        if not R.dry and not R.path(rel).exists():
            shutil.copy2(f, R.path(rel))
        R.record(rel, f, "copy (renamed)", "per cell-voxel registration votes; see validation/analysis/scripts/"
                 "06_lsfm_cache.py", "" if R.skip_hash else sha256(f))


def write_docs(R: Release, vox: dict):
    if R.dry:
        return
    rows = [dict(sample=LABEL[s], manuscript_label=f"Sample {i + 1}", fullres_voxel_z_um=vox[s]["z"],
                 fullres_voxel_y_um=vox[s]["y"], fullres_voxel_x_um=vox[s]["x"], registration_voxel_um=20,
                 atlas="perens_lsfm_mouse_20um (BrainGlobe v1.2)", sex="TODO", age="TODO", strain="TODO",
                 treatment="TODO") for i, s in enumerate(IDS)]
    pd.DataFrame(rows).to_csv(R.path("samples.csv"), index=False)
    readme = Path(__file__).with_name("data_release_README.md").read_text(encoding="utf-8")
    R.path("README.md").write_text(readme, encoding="utf-8")
    R.path("LICENSE").write_text("Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
                                 "https://creativecommons.org/licenses/by/4.0/legalcode\n", encoding="utf-8")


FULLRES = PRA / "QC_REGIME_FULLRES_AXIAL_MASKS"           # made by validation/analysis/scripts/24_fullres_axial_regime_masks.py


def build_fullres_consensus(R: Release, s: str):
    """The manuscript consensus (current QC rule, inclusion >= 0.5) in the full-resolution crop where cells are counted,
    and the Cellpose labels inside it. Files are already zlib-compressed: copied byte for byte."""
    L = LABEL[s]
    for v in ARMS:
        d = FULLRES / s / v / "current"
        for src, name, note in (
                (d / f"{s}_{v}_current_consensus0p5_mask_fullres_axial.tif", "consensus0p5_mask_fullres.tif",
                 "0.5 consensus of the registrations retained by global QC + SSIM, mapped into this crop (uint8 0/1)"),
                (d / f"{s}_{v}_current_masked_cells.tif", "cellpose_labels_in_consensus0p5.tif",
                 "Cellpose labels x consensus mask (voxel-wise); a cell is counted if any of its voxels remains")):
            rel = f"{L}/fullres_midbrain/{v}/{name}"
            dst = R.path(rel)
            if not R.dry and not dst.exists():
                shutil.copy2(src, dst)
            R.record(rel, src, "copy", note, "" if R.skip_hash else sha256(src))


def build_fullres_probability(R: Release, s: str, runs: pd.DataFrame):
    """Inclusion frequency (probability map) of the retained registrations in the full-resolution crop: the mean of
    their full-resolution SNpc masks (mask_region_resized, made by union_region_extraction_parallel.py), written page
    by page as float32. Thresholding it at 0.5 gives consensus0p5_mask_fullres.tif."""
    L = LABEL[s]
    for v in ARMS:
        g = runs[(runs["sample"] == s) & (runs.status == v) & runs.retained_qc_ssim]
        srcs = [Path(p) for p in g.mask_axial_path]
        rel = f"{L}/fullres_midbrain/{v}/inclusion_frequency_fullres.tif"
        dst = R.path(rel)
        if not R.dry and not dst.exists():
            shp, _ = tiff_info(srcs[0])
            files = [tifffile.TiffFile(p) for p in srcs]
            try:
                def pages():
                    for z in range(shp[0]):
                        acc = np.zeros(shp[1:], np.uint16)
                        for f in files:
                            acc += f.pages[z].asarray() > 0
                        yield (acc / len(files)).astype(np.float32)
                write_stack(dst, pages(), shp, np.float32,
                            dict(sample=L, arm=v, n_registrations=len(srcs), registrations=list(g.run_key),
                                 note="fraction of the registrations retained by global QC + SSIM that include the voxel"))
            finally:
                for f in files:
                    f.close()
        R.record(rel, srcs, "mean of the retained registrations' full-resolution masks (float32, zlib)",
                 f"inclusion frequency over {len(srcs)} retained registrations (global QC + top 50% SSIM)")


def build_fullres_rules(R: Release):
    """All eight global-QC rules of the sensitivity analysis, full-resolution consensus masks and masked cells."""
    for f in sorted(FULLRES.rglob("*.tif")):
        s, v, rule = f.relative_to(FULLRES).parts[:3]
        rel = f"derived/qc_sensitivity/fullres_qc_rule_masks/{LABEL[s]}/{v}/{rule}/{f.name.replace(s, LABEL[s], 1)}"
        dst = R.path(rel)
        if not R.dry and not dst.exists():
            shutil.copy2(f, dst)
        R.record(rel, f, "copy (renamed)", f"QC rule '{rule}': 0.5 consensus / masked cells in the full-resolution crop",
                 "" if R.skip_hash else sha256(f))
    readme = FULLRES / "README.md"
    rel = "derived/qc_sensitivity/fullres_qc_rule_masks/README_source.md"
    if not R.dry:
        txt = readme.read_text(encoding="utf-8")
        for s in IDS:
            txt = re.sub(rf"(?<![A-Za-z0-9]){re.escape(s)}(?=[_ ,/)]|$)", LABEL[s], txt)
        R.path(rel).write_text(re.sub(r"[A-Z]:\\[^\s`]*|/vast/[^\s`]*", "<internal path>", txt), encoding="utf-8")
    R.record(rel, readme, "text (internal paths and IDs removed)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", choices=["all", "fullres_consensus"], default="all",
                    help="fullres_consensus: only add the full-resolution consensus and probability maps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--provenance", default=None, help="default: <out>_PROVENANCE.csv next to the release folder")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-hash", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="SampleN labels to build (default all) + 'shared'")
    a = ap.parse_args()
    out = Path(P(a.out))
    R = Release(out, a.dry_run, a.skip_hash)
    runs = runs_table()
    sm = pd.read_csv(PRA / "original_dataset" / "config" / "samples.csv")
    sm["sample_id"] = sm["sample_id"].astype(str)
    vox = {r.sample_id: dict(z=r.pixel_size_z, y=r.pixel_size_y, x=r.pixel_size_x) for r in sm.itertuples()}
    todo = a.only or [LABEL[s] for s in IDS] + ["shared"]
    for s in IDS:
        if LABEL[s] in todo:
            print("building", LABEL[s], flush=True)
            if a.parts == "all":
                build_sample(R, s, runs, vox)
            build_fullres_consensus(R, s)
            build_fullres_probability(R, s, runs)
    if "shared" in todo:
        if a.parts == "all":
            build_shared(R, runs)
            write_docs(R, vox)
        build_fullres_rules(R)
    prov = Path(P(a.provenance)) if a.provenance else out.parent / f"{out.name}_PROVENANCE.csv"
    df = pd.DataFrame(R.rows)
    if a.dry_run:
        print(df.groupby(df.release_path.str.split("/").str[0]).source_bytes.sum().div(1e9).round(2).to_string())
        print(f"source bytes to be read: {R.est / 1e9:.1f} GB (release will be smaller: zlib)")
        df.to_csv(prov.with_name(prov.stem + "_DRYRUN.csv"), index=False)
        return
    if prov.exists() and (a.only or a.parts != "all"):            # partial rebuild: merge rows
        old = pd.read_csv(prov)
        df = pd.concat([old[~old.release_path.isin(df.release_path)], df], ignore_index=True)
    df.sort_values("release_path").to_csv(prov, index=False)
    man = df[["release_path", "release_bytes", "release_sha256"]].rename(
        columns={"release_path": "path", "release_bytes": "bytes", "release_sha256": "sha256"})
    man.sort_values("path").to_csv(out / "MANIFEST.csv", index=False)
    print(f"release: {out}  ({df.release_bytes.fillna(0).astype(float).sum() / 1e9:.1f} GB, {len(df)} files)")
    print(f"provenance: {prov}")


if __name__ == "__main__":
    main()
