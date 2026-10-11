"""Build the interactive global-QC viewers and their registration thumbnails.

Two static pages (no server, no internet), sharing data.js and thumbs/:
  index.html - IoU x grad_p95 scatter with the untuned starting point (median -/+ 3 MAD on raw, qc_start.py) and,
               for the JoM datasets, the chosen thresholds; drag or type thresholds, click a point to see the
               registration; optionally overlay a labels CSV exported from blind.html.
  blind.html - blinded rating: registrations shown one at a time in a seeded random order (stratified by brain),
               without metrics or parameters; rate pass / fail / unsure (+ failure mode) and export a CSV.
               Turn ratings into thresholds with 16_thresholds_from_labels.py (the page also shows a preview).

Thumbnail: three coronal slices (35 / 50 / 65 % along the anterior-posterior axis) of brainreg's downsampled.tiff
with that registration's boundaries.tiff in cyan, cropped to the tissue, JPEG, in
<out>/thumbs/<dataset>_<variant>/<sample>__<run>.jpg. Existing thumbnails are skipped.

Usage:
  JoM datasets (thresholds from script 12's qcproc_viewer_thresholds.csv), output validation_analysis/qc_viewer:
      python 13_build_qc_viewer.py [--skip-thumbs] [--include-paper]
  Any pipeline QC table(s) (columns sample_id, bending_energy_weight, grid_spacing, iou, grad_p95, save_dir):
      python 13_build_qc_viewer.py --qc NAME=qc_results.csv[,qc_results_preprocessed.csv] [--qc ...] --out DIR
  The first table of each --qc is treated as raw: the starting point is computed on it and applied to the second.
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from common import VA, amap_brain, log, load_table
from qc_start import start_thresholds

QCDIR = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\dataset_qc_comparison"))
TEMPLATES = Path(__file__).resolve().parent / "viewer_templates"
OUT = VA / "qc_viewer"
PAPER_ROOT = Path(P(r"<SCRATCH>\jom_review_sweep_original_data"))
SLICES = (0.35, 0.50, 0.65)
H = 200  # thumbnail height per slice (px)
VARIANTS = ("raw", "preprocessed")


def win(p: str) -> str:
    return to_local(p)


def fmt(x):  # 0.95 -> 0p95, 1.0 -> 1p0, 0.0 -> 0p0
    s = f"{float(x):g}"
    return (s if "." in s else s + ".0").replace(".", "p")


def pipeline_rows(ds: str, v: str, d: pd.DataFrame, sample_fn=str) -> list[dict]:
    rows = []
    for r in d.itertuples():
        reg = win(str(r.save_dir))
        rows.append(dict(dataset=ds, variant=v, sample=sample_fn(r.sample_id), be=float(r.bending_energy_weight),
                         grid=int(r.grid_spacing), iou=float(r.iou), grad=float(r.grad_p95), run=Path(reg).name,
                         reg_dir=reg))
    return rows


def finish(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["key"] = df.dataset.str.replace(" ", "_") + "_" + df.variant
    df["thumb"] = "thumbs/" + df.key + "/" + df["sample"].astype(str) + "__" + df.run + ".jpg"
    df["id"] = df.key + "|" + df["sample"].astype(str) + "|" + df.run
    return df


def jom_registrations() -> pd.DataFrame:
    rows = []
    o = pd.read_csv(QCDIR / "original_lsfm_full_final_results_with_ssim.csv")
    for v in VARIANTS:
        for r in o.itertuples():
            run = f"bend_{fmt(r.bending)}__grid_m{abs(int(r.grid))}__smooth_m1p0"
            smp = str(getattr(r, "_2"))
            rows.append(dict(dataset="paper LSFM", variant=v, sample=smp, be=float(r.bending), grid=int(r.grid),
                             iou=float(getattr(r, f"{v}_iou")), grad=float(getattr(r, f"{v}_grad_p95")),
                             run=run, reg_dir=str(PAPER_ROOT / smp / v / run)))
    for ds, pre, fn in (("aMAP", "amap_qc_results_", amap_brain), ("validation LSFM", "validation_lsfm_qc_results_", str)):
        for v in VARIANTS:
            rows += pipeline_rows(ds, v, pd.read_csv(QCDIR / f"{pre}{v}.csv"), fn)
    return finish(rows)


def jom_thresholds() -> dict:
    thr = {}
    for r in load_table("qcproc_viewer_thresholds").itertuples():
        k = f"{r.dataset.replace(' ', '_')}_{r.variant}"
        thr[k] = dict(label=f"{r.dataset} — {r.variant}", chosen=[r.chosen_iou, r.chosen_grad],
                      defaults={m: [getattr(r, f"{m}_iou"), getattr(r, f"{m}_grad")]
                                for m in ("start", "pooled", "separate", "raw_anchored")})
        if r.dataset == "paper LSFM":
            thr[k]["note"] = PAPER_NOTE
    return thr


PAPER_NOTE = ("Paper LSFM: the images and boundaries come from a rerun of the parameter sweep "
              "(jom_review_sweep_original_data, same inputs and parameters); IoU and grad_p95 are from the original "
              "paper sweep. The two runs agree closely but not exactly (SNpc Dice 0.93–1.00 in spot checks).")


def generic(specs: list[str]):
    """--qc NAME=raw.csv[,pre.csv] -> registrations + starting-point thresholds (set on raw, applied to both)."""
    rows, thr = [], {}
    for spec in specs:
        name, files = spec.split("=", 1)
        tables = [pd.read_csv(f) for f in files.split(",")]
        start = start_thresholds(tables[0].iou, tables[0].grad_p95)
        for v, d in zip(VARIANTS, tables):
            rows += pipeline_rows(name, v, d)
            thr[f"{name.replace(' ', '_')}_{v}"] = dict(label=f"{name} — {v}", chosen=None, defaults={"start": list(start)})
    return finish(rows), thr


def read_pages(path, idx):
    with tifffile.TiffFile(path) as t:
        n = len(t.pages)
        if n > 1:
            return [t.pages[i].asarray() for i in idx], n
        arr = t.asarray()
        return [arr[i] for i in idx], arr.shape[0]


_img_cache = {}


def brain_slices(reg_dir):
    """downsampled.tiff is the same for every registration of a brain x variant: read once."""
    key = str(Path(reg_dir).parent)
    if key not in _img_cache:
        p = Path(reg_dir) / "downsampled.tiff"
        with tifffile.TiffFile(p) as t:
            n = len(t.pages)
        idx = [int(round(f * (n - 1))) for f in SLICES]
        imgs, _ = read_pages(p, idx)
        _img_cache[key] = (idx, imgs)
    return _img_cache[key]


def make_thumb(row) -> str:
    dest = OUT / row.thumb
    if dest.exists():
        return "cached"
    reg = Path(row.reg_dir)
    if not (reg / "boundaries.tiff").exists() or not (reg / "downsampled.tiff").exists():
        return "missing"
    try:
        idx, imgs = brain_slices(row.reg_dir)
        bnds, _ = read_pages(reg / "boundaries.tiff", idx)
        tiles = []
        for img, b in zip(imgs, bnds):
            img = img.astype(np.float32)
            lo, hi = np.percentile(img, [1, 99.5])
            g = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
            rgb = np.stack([g, g, g], -1)
            m = b != 0
            rgb[m] = 0.35 * rgb[m] + 0.65 * np.array([0.0, 0.9, 1.0])
            keep = (g > 0.08) | m
            rr, cc = np.flatnonzero(keep.any(1)), np.flatnonzero(keep.any(0))
            if len(rr) and len(cc):
                rgb = rgb[max(rr[0] - 4, 0): rr[-1] + 5, max(cc[0] - 4, 0): cc[-1] + 5]
            im = Image.fromarray((rgb * 255).astype(np.uint8))
            im = im.resize((max(1, int(im.width * H / im.height)), H), Image.BILINEAR)
            tiles.append(im)
        W = sum(t.width for t in tiles) + 4 * (len(tiles) - 1)
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        x = 0
        for t in tiles:
            canvas.paste(t, (x, 0))
            x += t.width + 4
        dest.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(dest, quality=85)
        return "written"
    except Exception as e:  # keep going; report at the end
        return f"error: {e}"


def build_thumbs(df: pd.DataFrame, include_paper: bool):
    todo = df[~df.thumb.map(lambda t: (OUT / t).exists())]
    if not include_paper:  # paper sweep is being rewritten; add it later with --include-paper
        todo = todo[todo.dataset != "paper LSFM"]
    log(f"{len(todo)} thumbnails to build ({len(df)} registrations)")
    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:  # one brain x variant at a time: downsampled.tiff read once
        for _, g in todo.groupby(["key", "sample"]):
            results += list(ex.map(make_thumb, g.itertuples()))
            _img_cache.clear()
    if results:
        log(pd.Series(results).str.split(":").str[0].value_counts().to_string())


def build_html(df: pd.DataFrame, thr: dict):
    df = df.copy()
    df["has_thumb"] = df.thumb.map(lambda t: (OUT / t).exists())
    recs = df[["id", "key", "dataset", "variant", "sample", "be", "grid", "iou", "grad", "run", "thumb", "has_thumb"]]
    data = dict(records=recs.to_dict("records"), thresholds=thr)
    (OUT / "data.js").write_text("window.QC_DATA = " + json.dumps(data) + ";", encoding="utf-8")
    for page in ("index.html", "blind.html"):
        shutil.copyfile(TEMPLATES / page, OUT / page)
    log(f"viewer: {len(recs)} registrations, {int(df.has_thumb.sum())} with thumbnails -> {OUT / 'index.html'}, "
        f"{OUT / 'blind.html'}")


if __name__ == "__main__":
    a = sys.argv[1:]
    specs = [a[i + 1] for i, x in enumerate(a) if x == "--qc"]
    if specs:
        OUT = Path(a[a.index("--out") + 1]) if "--out" in a else Path.cwd() / "qc_viewer"
        df, thr = generic(specs)
    else:
        df, thr = jom_registrations(), jom_thresholds()
    OUT.mkdir(parents=True, exist_ok=True)
    if "--skip-thumbs" not in a:
        build_thumbs(df, include_paper=bool(specs) or "--include-paper" in a)
    build_html(df, thr)
