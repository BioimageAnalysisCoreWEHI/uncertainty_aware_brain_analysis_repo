"""Build the v2 data release (layout by sample, numbered pipeline steps) from the v1 release made by
build_data_release.py, plus figure source data and aMAP galleries. Read-only on all inputs: everything is copied.

Changes from v1:
  - SampleN/01_registration_inputs, 02_registrations, 03_consensus_20um, 04_fullres_midbrain; shared/ for the rest
  - NEW: AF/TH 20 um images in brainreg's coronal sample space (voxel-aligned with the SNpc masks):
         coronal = np.flip(np.transpose(acquisition, (1, 0, 2)), axis=2)   (inverse of crop_metadata.json's mapping)
  - "prob_map" -> "inclusion_frequency"; independent-LSFM "gt" -> "published" (file and column names)
  - samples.csv without animal metadata; no expert annotations
  - figure source data by final figure/table (from --figure-map), aMAP per-brain galleries (from --galleries)
  - MANIFEST.csv (all files, SHA-256) and a provenance table (written OUTSIDE the release)

Usage:
  python restructure_release_v2.py --v1 <data_availability_v1> --out <data_availability> --figure-map <csv>
         --galleries <folder with FigS9_aMAP_gallery_*.png/pdf> --provenance-v1 <v1 PROVENANCE.csv>
         --provenance-out <new PROVENANCE.csv> --readme <README.md for v2>
         --tables <validation analysis results/tables> --indep <consensus_vs_gt.csv>
         --supp-tables <folder of TableS*.csv from make_supplementary_tables.py> --fig4src <Fig. 4 source data>
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

ARMS = ("raw", "preprocessed")
DROP_SAMPLE_COLS = ["sex", "age", "strain", "treatment"]
GT = re.compile(r"(?<![a-z])gt(?![a-z])")


def sha256(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(buf):
            h.update(b)
    return h.hexdigest()


class Release:
    def __init__(self, v1, out):
        self.v1, self.out, self.rows = Path(v1), Path(out), []

    def _dst(self, rel):
        d = self.out / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        return d

    def copy(self, src, rel, op="copy", note=""):
        src = Path(src)
        shutil.copy2(src, self._dst(rel))
        self.rows.append(dict(release_path=rel, from_path=str(src), operation=op, note=note))

    def from_v1(self, v1rel, rel, op="copy", note=""):
        self.copy(self.v1 / v1rel, rel, op, note)

    def write_text(self, rel, text, src, op, note=""):
        # write bytes: pandas.to_csv() already uses os.linesep, and write_text would turn \r\n into \r\r\n on Windows
        self._dst(rel).write_bytes(text.replace("\r\n", "\n").encode("utf-8"))
        self.rows.append(dict(release_path=rel, from_path=str(src), operation=op, note=note))

    def coronal(self, v1rel, rel, sample, mask_shape):
        src = self.v1 / v1rel
        with tifffile.TiffFile(src) as t:
            a = t.series[0].asarray()
            desc = t.pages[0].description
        c = np.ascontiguousarray(np.flip(np.transpose(a, (1, 0, 2)), axis=2))
        assert c.shape == tuple(mask_shape), (src, c.shape, mask_shape)
        try:
            meta = json.loads(desc)
        except Exception:
            meta = {"source_description": desc}
        meta.update(space="brainreg coronal sample space (same grid as the 20 um SNpc masks and consensus maps)",
                    transform_from_acquisition="np.flip(np.transpose(a, (1, 0, 2)), axis=2)", sample=sample)
        tifffile.imwrite(self._dst(rel), c, compression="zlib", description=json.dumps(meta), metadata=None)
        self.rows.append(dict(release_path=rel, from_path=str(src), operation="reoriented to brainreg coronal space",
                              note="np.flip(np.transpose(a,(1,0,2)),axis=2); values unchanged"))


def all_registrations_maps(R, s, arm, reg_csv):
    """20 um maps over exactly the 48 registrations of the design (the pipeline's sweep-folder maps also averaged
    1-8 extra grid -1 registrations outside the design). Same conventions as the pipeline: float32 inclusion
    frequency; uint8 consensus_<t> = inclusion >= t (float32), consensus_0.00 = inclusion > 0 = union; axial =
    np.flip(np.transpose(coronal, (1, 0, 2)), axis=2)."""
    reg = pd.read_csv(reg_csv)
    mdir = Path(reg_csv).parent
    inc = np.mean([tifffile.imread(mdir / m) > 0 for m in reg.mask_file], axis=0).astype(np.float32)
    out = f"{s}/03_consensus_20um/{arm}/all_registrations"
    desc = lambda kind: json.dumps({"sample": s, "arm": arm, "voxel_size_um": [20, 20, 20], "n_masks": len(reg),
                                    "masks": "the 48 registrations of the parameter grid", "map": kind})
    ax = lambda a: np.ascontiguousarray(np.flip(np.transpose(a, (1, 0, 2)), axis=2))
    items = [(f"{s}_inclusion_frequency", inc, "inclusion frequency")]
    for t in np.round(np.arange(0, 1.0001, 0.1), 2):
        m = (inc > 0) if t == 0 else (inc.astype(np.float64) >= t)
        items.append((f"{s}_consensus_{t:.2f}", m.astype(np.uint8), f"consensus at inclusion >= {t:.2f}"
                      if t else "consensus at inclusion > 0"))
    items.append((f"{s}_union", (inc > 0).astype(np.uint8), "union of the 48 masks"))
    files = []
    for stem, arr, kind in items:
        for orient, a in (("coronal", arr), ("axial", ax(arr))):
            rel = f"{out}/{stem}_{orient}.tif"
            dst = R._dst(rel)
            tifffile.imwrite(dst, a, compression="zlib", description=desc(kind), metadata=None)
            files.append(rel)
            R.rows.append(dict(release_path=rel, from_path=str(mdir), operation="recomputed from the 48 released masks",
                               note=kind))
    return files


def build_sample(R, s):
    v = Path(s)
    # 01 inputs (+ coronal versions)
    mask_shape = tifffile.TiffFile(R.v1 / s / "inputs" / "tissue_mask_20um_coronal.tif").series[0].shape
    for f in ("AF_20um_raw.tif", "AF_20um_preprocessed.tif", "TH_20um.tif", "tissue_mask_20um_coronal.tif"):
        R.from_v1(v / "inputs" / f, f"{s}/01_registration_inputs/{f}")
    for f in ("AF_20um_raw", "AF_20um_preprocessed", "TH_20um"):
        R.coronal(v / "inputs" / f"{f}.tif", f"{s}/01_registration_inputs/{f}_coronal.tif", s, mask_shape)
    tissue_mask_acquisition(R, s)
    for arm in ARMS:
        # 02 registrations
        reg = R.v1 / s / "registrations" / arm
        df = pd.read_csv(reg / "registrations.csv")
        df["mask_file"] = df.mask_file.str.replace("snpc_mask_20um_coronal/", "snpc_masks_20um/", regex=False)
        R.write_text(f"{s}/02_registrations/{arm}/registrations.csv", df.to_csv(index=False),
                     reg / "registrations.csv", "copy; mask_file paths updated to the new folder name")
        for m in sorted((reg / "snpc_mask_20um_coronal").glob("*.tif")):
            R.copy(m, f"{s}/02_registrations/{arm}/snpc_masks_20um/{m.name}")
        # 03 consensus 20 um
        for m in sorted((R.v1 / s / "consensus" / arm / "qc_ssim_retained").glob("*.tif")):
            R.copy(m, f"{s}/03_consensus_20um/{arm}/retained/{m.name.replace('prob_map', 'inclusion_frequency')}",
                   "copy (renamed)" if "prob_map" in m.name else "copy")
        all_registrations_maps(R, s, arm, R.out / f"{s}/02_registrations/{arm}/registrations.csv")
        # 04 full-resolution midbrain
        fr = R.v1 / s / "fullres_midbrain" / arm
        for old, new in (("TH_crop_fullres.tif", "TH_crop_fullres.tif"), ("cellpose_labels.tif", "cellpose_labels.tif"),
                         ("inclusion_frequency_fullres.tif", "inclusion_frequency_retained.tif"),
                         ("inclusion_frequency_fullres_all48.tif", "inclusion_frequency_all48.tif"),
                         ("consensus0p5_mask_fullres.tif", "consensus0p5_mask.tif"),
                         ("cellpose_labels_in_consensus0p5.tif", "cells_in_consensus0p5.tif"),
                         ("crop_metadata.json", "crop_metadata.json")):
            R.copy(fr / old, f"{s}/04_fullres_midbrain/{arm}/{new}", "copy" if old == new else "copy (renamed)")


def tissue_mask_acquisition(R, s):
    """Tissue mask used for global QC, in the orientation of the released 20 um inputs (exact inverse of the coronal
    transform, which is its own inverse)."""
    src = R.v1 / s / "inputs" / "tissue_mask_20um_coronal.tif"
    a = tifffile.imread(src)
    m = np.ascontiguousarray(np.flip(np.transpose(a, (1, 0, 2)), axis=2))
    with tifffile.TiffFile(R.v1 / s / "inputs" / "AF_20um_raw.tif") as t:
        assert m.shape == t.series[0].shape, (s, m.shape, t.series[0].shape)
    rel = f"{s}/01_registration_inputs/tissue_mask_20um.tif"
    tifffile.imwrite(R._dst(rel), m, compression="zlib", metadata=None, description=json.dumps(
        {"sample": s, "space": "acquisition orientation of the 20 um inputs (brainreg orientation 'sal')",
         "note": "the tissue mask used for the tissue-atlas IoU; = np.flip(np.transpose(coronal, (1, 0, 2)), axis=2)"}))
    R.rows.append(dict(release_path=rel, from_path=str(src), operation="reoriented to the acquisition orientation",
                       note="exact inverse of the coronal transform"))


SAMPLE_LABEL = {"162": "Sample1", "527": "Sample2", "pp451": "Sample3", "xl353": "Sample4", "xl385": "Sample5"}


def essential_tables(R, tables_dir, indep_csv, supp_tables_dir, fig4_dir):
    """shared/tables: the essential result tables only (everything else is re-derivable with the code)."""
    T, out = Path(tables_dir), "shared/tables"
    R.copy(R.v1 / "derived" / "counts" / "all_registrations_qc_and_counts.csv",
           f"{out}/original_LSFM_registrations_qc_and_counts.csv", "copy (renamed)",
           "every registration (5 brains x 2 inputs x 48): parameters, QC metrics, SSIM, selection, SNpc TH+ count")
    # consensus counts: the complete vote-cache recount (identical to the pipeline's final_count_results table
    # wherever that table succeeded; it failed with out-of-memory errors for 6 unused all-runs/top-25 % sets)
    lc = pd.read_csv(T / "lsfm_consensus.csv")
    lc = pd.DataFrame({"sample_id": lc["sample"].astype(str).map(SAMPLE_LABEL), "processing_status": lc.status,
                       "qc_filter": lc.qc_filter, "ssim_filter": lc.ssim_filter, "threshold": lc.threshold,
                       "n_masks_used": lc.n_runs, "consensus_cell_count": lc["count"]})
    R.write_text(f"{out}/original_LSFM_consensus_counts_by_threshold.csv", lc.to_csv(index=False),
                 T / "lsfm_consensus.csv", "relabelled copy",
                 "consensus SNpc TH+ counts for every registration set and inclusion threshold (overlap rule)")
    F = Path(fig4_dir)
    sc = pd.read_csv(F / "Fig4_F_scores.csv")
    R.write_text(f"{out}/aMAP_dice_scores.csv", sc.to_csv(index=False), F / "Fig4_F_scores.csv", "copy",
                 "every Dice score with the expert STAPLE consensus (aMAP protocol, raw input): each expert "
                 "annotation, each retained single registration, and the consensus (Fig. 4F)")
    cs = pd.read_csv(F / "Fig4_G_cases.csv").rename(columns={
        "min": "single_qc_passing_min_dice", "max": "single_qc_passing_max_dice",
        "count": "n_single_qc_passing", "default_dice": "brainreg_default_dice", "cons_dice": "consensus_dice",
        "spread": "single_qc_passing_dice_range"})
    R.write_text(f"{out}/aMAP_per_case_dice.csv", cs.to_csv(index=False), F / "Fig4_G_cases.csv", "relabelled copy",
                 "per brain x structure (54 cases): consensus, brainreg default and range of QC-passing singles (Fig. 4G)")
    ind = pd.read_csv(indep_csv)
    ind.columns = [GT.sub("published", c) for c in ind.columns]
    R.write_text(f"{out}/independent_LSFM_consensus_vs_published.csv", ind.to_csv(index=False), indep_csv,
                 "copy; 'gt' -> 'published'", "consensus vs the published registration, every brain x region")
    for f in sorted(Path(supp_tables_dir).glob("TableS*.csv")):
        R.copy(f, f"{out}/{f.name}", "copy", "Supplementary Table")


def build_shared_slim(R, tables_dir, indep_csv, supp_tables_dir, galleries, fig4_dir):
    for f in sorted((R.v1 / "cellpose_model").rglob("*")):
        if f.is_file():
            R.copy(f, f"shared/cellpose_model/{f.relative_to(R.v1 / 'cellpose_model').as_posix()}")
    q = R.v1 / "derived" / "qc_sensitivity"
    for f in sorted(q.rglob("*")):
        if f.is_file():
            R.copy(f, f"shared/qc_sensitivity/{f.relative_to(q).as_posix()}")
    essential_tables(R, tables_dir, indep_csv, supp_tables_dir, fig4_dir)
    for g in sorted(Path(galleries).glob("FigS9_aMAP_gallery_*")):
        brain = g.stem.replace("FigS9_aMAP_gallery_", "").replace("brain", "")
        R.copy(g, f"shared/aMAP_per_brain_galleries/aMAP_gallery_{brain}{g.suffix}")


AMAP_BRAINS = {  # region_consensus / sweep folder prefix -> (aMAP brain ID used in the tables, resliced folder)
    "CR_Syt6_cre_rfp_35": ("CR_Syt6CreRfp_35", "CR_Syt6"), "ER_Glt25d2Cre_9": ("ER_Glt25d2Cre_9", "ER_Glt25d2Cre"),
    "GAD67GFP_223": ("AB_Gad67_223", "GAD67GFP_223"), "MV_131017_MV_7": ("MV131017_7", "MV_131017"),
    "MV_Ntsr1_165": ("MV_Ntsr1_165", "MV_Ntsr1"), "Ntsr1_169": ("MV_Ntsr1_169", "Ntsr1_169")}
AMAP_STRUCTS = ["ACA", "SSp", "RSP", "VISp", "VISam", "SUB", "AHN", "MV", "VPM", "DG-sg"]
STAPLE_CODE = {"CING": "ACA", "S1BF": "SSp", "SG": "DG-sg", "V1": "VISp", "V2M": "VISam", "AHN": "AHN", "MV": "MV",
               "RSP": "RSP", "SUB": "SUB", "VPM": "VPM"}
QC_COLS = ["bending_energy_weight", "grid_spacing", "iou", "grad_p95", "grad_p99", "jac_min", "jac_frac_lt_0",
           "passes_global_qc"]


def amap_consensus_regions(R, region_dir, sweep_dir, brains25_dir, staple_dir, hemisphere_cache, arm="raw"):
    """shared/aMAP_consensus_regions/<brain>/: brainreg's 25 um image (the space of all regions), the tissue mask,
    per structure the inclusion frequency of the retained registrations and the 0.5 consensus, and a QC table of
    all 64 registrations (parameters, global QC, local SSIM, retention per structure). Plus the expert STAPLE
    references (in the original aMAP stack grid) with their placement checks."""
    out = "shared/aMAP_consensus_regions"
    for d in sorted(Path(region_dir).iterdir()):
        key = next(k for k in AMAP_BRAINS if d.name.startswith(k))
        brain, short = AMAP_BRAINS[key]
        run = next(p for p in sorted((Path(sweep_dir) / d.name / arm).iterdir()) if (p / "downsampled.tiff").exists())
        img = tifffile.imread(run / "downsampled.tiff")
        rel = f"{out}/{brain}/{brain}_brain_25um.tif"
        tifffile.imwrite(R._dst(rel), img, compression="zlib", metadata=None, description=json.dumps(
            {"brain": brain, "space": "brainreg sample space at 25 um (grid of every region in this folder)",
             "source": "brainreg downsampled.tiff (identical for every registration of this brain)"}))
        R.rows.append(dict(release_path=rel, from_path=str(run / "downsampled.tiff"), operation="recompressed (zlib)",
                           note="aMAP brain in the registration space"))
        tm = next((Path(brains25_dir) / short).glob("*tissue_mask*_resliced.tif"))
        m = tifffile.imread(tm)
        assert m.shape == img.shape, (brain, m.shape, img.shape)
        rel = f"{out}/{brain}/{brain}_tissue_mask_25um.tif"
        tifffile.imwrite(R._dst(rel), (m > 0).astype(np.uint8), compression="zlib", metadata=None,
                         description=json.dumps({"brain": brain, "note": "tissue mask used for the tissue-atlas IoU"}))
        R.rows.append(dict(release_path=rel, from_path=str(tm), operation="binarised, recompressed", note="tissue mask"))
        qc = None
        for st in AMAP_STRUCTS:
            src = d / arm / f"{st}_both" / "passes_final_qc_top_0p5"
            p = tifffile.imread(src / "probability.tif").astype(np.float32)
            c = tifffile.imread(src / "consensus_thresh_0p5.tif") > 0
            assert p.shape == img.shape and np.array_equal(c, p >= 0.5), (brain, st)
            for name, arr, kind in ((f"{brain}_{st}_inclusion_frequency_25um.tif", p, "inclusion frequency of the retained registrations"),
                                    (f"{brain}_{st}_consensus0p5_25um.tif", c.astype(np.uint8), "consensus (inclusion >= 0.5)")):
                rel = f"{out}/{brain}/{name}"
                tifffile.imwrite(R._dst(rel), arr, compression="zlib", metadata=None, description=json.dumps(
                    {"brain": brain, "structure": st, "input": arm, "map": kind, "hemispheres": "both"}))
                R.rows.append(dict(release_path=rel, from_path=str(src), operation="recompressed (zlib)", note=kind))
            allr = pd.read_csv(d / arm / f"{st}_both" / "all_registrations" / "registrations_used.csv")
            cols = QC_COLS + [f"local_ssim__{st}_both", f"passes_local_ssim_qc__{st}_both"]
            t = allr[cols].rename(columns={f"local_ssim__{st}_both": f"ssim_{st}",
                                           f"passes_local_ssim_qc__{st}_both": f"retained_{st}"})
            qc = t if qc is None else qc.merge(t, on=QC_COLS, how="outer")
        assert len(qc) == 64, (brain, len(qc))
        R.write_text(f"{out}/{brain}/{brain}_registrations_qc.csv", qc.to_csv(index=False),
                     d / arm, "selected columns of registrations_used.csv (internal paths removed)",
                     "all 64 registrations: parameters, global QC, local SSIM and retention per structure")
    # expert STAPLE: 2-D sections on the region grid + the annotated-hemisphere masks used in the paper
    # (staple_dir = userStapleResults/asr25um; hemisphere_cache = validation/analysis cache/amap)
    import importlib.util
    spec = importlib.util.spec_from_file_location("st2", Path(__file__).with_name("amap_expert_staple_2d.py"))
    st2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st2)
    rel = f"{out}/expert_STAPLE"
    sec = st2.write_sections(R, staple_dir, rel, out)
    R.rows = [x for x in R.rows if not x["release_path"].endswith("expert_STAPLE_sections.csv")]
    sec = st2.write_hemispheres(R, hemisphere_cache, sec, rel, out)
    R.write_text(f"{rel}/expert_STAPLE_sections.csv", sec.to_csv(index=False), staple_dir, "generated table",
                 "coronal plane of every STAPLE section; raters' slab; annotated hemisphere label")


def staple_nifti_to_tiff(R, src, rel):
    """NIfTI-1 (x, y, z) -> 3-D TIFF (z, y, x): page k = NIfTI plane k, tiff[k, j, i] = nifti[i, j, k]. Reads
    only the non-zero voxels (the STAPLE is one section in a large grid) with the aMAP port's NIfTI reader, keeps
    values and dtype, records shape/voxel size/sform in the description and checks the round trip."""
    import importlib.util
    port = Path(__file__).resolve().parents[1] / "validation" / "amap_python_port" / "amap_validation" / "nifti.py"
    spec = importlib.util.spec_from_file_location("amap_nifti", port)
    nifti = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nifti)
    hdr = nifti.read_header(src)
    nx, ny, nz = hdr["shape"][:3]
    vox, vals = nifti.load_sparse(src)
    dtype = np.asarray(vals).dtype if len(vals) else hdr["dtype"].newbyteorder("=")
    vol = np.zeros((nz, ny, nx), dtype=dtype)
    vol[vox[:, 2], vox[:, 1], vox[:, 0]] = vals
    desc = {"source": Path(src).name, "content": "expert STAPLE consensus of the public aMAP annotations",
            "axes": "ZYX; tiff[k, j, i] = nifti[i, j, k]", "nifti_shape_xyz": [nx, ny, nz],
            "voxel_size_xyz": [float(v) for v in hdr["pixdim"][1:4]],
            "nifti_sform": np.asarray(hdr["sform"]).round(6).tolist(), "sform_code": int(hdr["sform_code"])}
    dst = R._dst(rel)
    tifffile.imwrite(dst, vol, compression="zlib", metadata=None, description=json.dumps(desc))
    back = tifffile.imread(dst)
    k, j, i = np.nonzero(back)
    order = np.lexsort((vox[:, 0], vox[:, 1], vox[:, 2]))
    assert back.shape == (nz, ny, nx) and len(k) == len(vox) and \
        np.array_equal(np.stack([i, j, k], 1), vox[order]) and np.array_equal(back[k, j, i], np.asarray(vals)[order]), src
    R.rows.append(dict(release_path=rel, from_path=str(src), operation="NIfTI -> TIFF (lossless, round-trip checked)",
                       note="expert STAPLE; axes ZYX, NIfTI header in the description"))


def build_shared(R, figure_map, galleries):
    """(Superseded 2026-10-10 by build_shared_slim: full set of analysis tables and figure source data.)"""
    d = R.v1 / "derived"
    for f in sorted((R.v1 / "cellpose_model").rglob("*")):
        if f.is_file():
            R.copy(f, f"shared/cellpose_model/{f.relative_to(R.v1 / 'cellpose_model').as_posix()}")
    for sub, new in (("counts", "analysis_tables/counts"), ("original_lsfm_analysis", "analysis_tables/original_lsfm"),
                     ("qc_sensitivity", "analysis_tables/qc_sensitivity"), ("vote_cache", "analysis_tables/vote_cache"),
                     ("amap_validation", "validation/aMAP")):
        for f in sorted((d / sub).rglob("*")):
            if f.is_file():
                R.copy(f, f"shared/{new}/{f.relative_to(d / sub).as_posix()}")
    for f in sorted((d / "independent_lsfm_validation").glob("*")):
        name = GT.sub("published", f.stem) + f.suffix
        if f.suffix == ".csv":
            df = pd.read_csv(f, low_memory=False)
            df.columns = [GT.sub("published", c) for c in df.columns]
            R.write_text(f"shared/validation/independent_LSFM/{name}", df.to_csv(index=False), f,
                         "copy; 'gt' -> 'published' in file and column names")
        else:
            R.copy(f, f"shared/validation/independent_LSFM/{name}")
    for g in sorted(Path(galleries).glob("FigS9_aMAP_gallery_*")):
        brain = g.stem.replace("FigS9_aMAP_gallery_", "").replace("brain", "")
        R.copy(g, f"shared/validation/aMAP/per_brain_galleries/aMAP_gallery_{brain}{g.suffix}")
    idx = [["folder", "file", "content"]]
    for row in csv.DictReader(open(figure_map, encoding="utf-8")):
        src = Path(row["source_path"])
        R.copy(src, f"shared/figure_source_data/{row['dest_folder']}/{src.name}", "copy (figure source data)")
        idx.append([row["dest_folder"], src.name, row["note"]])
    lines = ["| Folder | File | Content |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in idx[1:]]
    R.write_text("shared/figure_source_data/INDEX.md",
                 "# Figure and table source data\n\nOne folder per final figure or supplementary table.\n\n"
                 + "\n".join(lines) + "\n", figure_map, "generated index")


def main():
    ap = argparse.ArgumentParser()
    for a in ("--v1", "--out", "--galleries", "--provenance-v1", "--provenance-out", "--readme",
              "--tables", "--indep", "--supp-tables", "--fig4src"):
        ap.add_argument(a, required=True)
    a = ap.parse_args()
    R = Release(a.v1, a.out)
    assert not R.out.exists() or not any(R.out.iterdir()), "output folder must be empty"
    for s in ("Sample1", "Sample2", "Sample3", "Sample4", "Sample5"):
        build_sample(R, s)
        print("built", s, flush=True)
    build_shared_slim(R, a.tables, a.indep, a.supp_tables, a.galleries, a.fig4src)
    sm = pd.read_csv(R.v1 / "samples.csv").drop(columns=DROP_SAMPLE_COLS, errors="ignore")
    R.write_text("samples.csv", sm.to_csv(index=False), R.v1 / "samples.csv", "copy without animal metadata")
    R.copy(R.v1 / "LICENSE", "LICENSE")
    R.copy(a.readme, "README.md", "written for v2")
    # provenance: chain each file back to its original source through the v1 provenance table
    v1p = pd.read_csv(a.provenance_v1).set_index("release_path")
    rows = []
    for r in R.rows:
        p = Path(r["from_path"])
        try:
            v1rel = p.relative_to(R.v1).as_posix()
        except ValueError:
            v1rel = None
        orig = v1p.loc[v1rel, "source_paths"] if v1rel in v1p.index else r["from_path"]
        rows.append(dict(release_path=r["release_path"], v1_release_path=v1rel or "", original_source=orig,
                         operation=r["operation"], note=r["note"],
                         bytes=(R.out / r["release_path"]).stat().st_size, sha256=sha256(R.out / r["release_path"])))
    prov = pd.DataFrame(rows)
    prov.to_csv(a.provenance_out, index=False)
    man = prov[["release_path", "bytes", "sha256"]].rename(columns={"release_path": "path"})
    man.to_csv(R.out / "MANIFEST.csv", index=False)
    print(f"files: {len(prov)} ({prov.bytes.sum() / 1e9:.2f} GB) + MANIFEST.csv")


if __name__ == "__main__":
    main()
