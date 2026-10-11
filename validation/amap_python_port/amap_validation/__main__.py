"""Command line: python -m amap_validation <step> [options]  (run from python/). See PIPELINE.md.

Steps: agreement, score, all (= agreement + score), qc, tobrain, asr25, asr25-raters, unzip, index,
full (= the whole pipeline in order), compare (validation against the user's pipeline).
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P, to_cluster, to_local  # noqa: E402,F401  (resolves <IMAGING>/<SCRATCH> placeholders)

import argparse
from pathlib import Path

from .config import DEFAULT_OUT_DIR, load_config
from .pipeline import RunOptions, run_agreement, run_manual_scoring

FULL_SEQUENCE = [["agreement"], ["score"], ["qc"], ["tobrain", "--no-raters"], ["asr25", "--stamp", "--figures"],
                 ["asr25-raters", "--figures"], ["index"]]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="amap_validation")
    ap.add_argument("step", choices=["agreement", "score", "all", "qc", "tobrain", "asr25", "asr25-raters", "unzip",
                                     "index", "full", "compare"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="output directory (outside the repo)")
    ap.add_argument("--mode", choices=["", "range", "quantiles"], default="",
                    help="z-window: '' = all slices (default), 'range' (with --range), 'quantiles' (25-75%%)")
    ap.add_argument("--range", type=int, default=7)
    ap.add_argument("--repeats", choices=["both", "first"], default="both",
                    help="'both' = MATLAB code (default); 'first' = paper's first-attempt-only variant")
    ap.add_argument("--no-unc", action="store_true", help="estimate STAPLE on all voxels (no -unc)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=(), help="brainRegionComb values, e.g. AB_Gad67_223RSP")
    ap.add_argument("--no-raters", action="store_true", help="tobrain: skip the individual rater volumes")
    ap.add_argument("--figures", action="store_true",
                    help="tobrain/asr25: also write QC figures (read back from the written files)")
    ap.add_argument("--gzip", action="store_true", help="asr25: write .nii.gz instead of plain .nii")
    ap.add_argument("--stamp", action="store_true", help="asr25: also write a 3-D asr 25 um label volume per brain")
    ap.add_argument("--pipeline-root", type=Path, default=Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\results")),
                    help="compare: the uncertainty-aware pipeline's results folder (read-only)")
    ap.add_argument("--hemisphere", choices=["amap", "registered"], default="amap",
                    help="compare: restrict pipeline masks with aMAP's midline or the pipeline's registered_hemispheres")
    a = ap.parse_args(argv)

    if a.step == "full":
        common = ["--out", str(a.out), "--workers", str(a.workers), "--repeats", a.repeats]
        if a.mode:
            common += ["--mode", a.mode, "--range", str(a.range)]
        if a.no_unc:
            common.append("--no-unc")
        for sub in FULL_SEQUENCE:
            print(f"\n===== full: {' '.join(sub)} =====", flush=True)
            main(sub + common)
        return

    arg = a.range if a.mode == "range" else ([0.25, 0.75] if a.mode == "quantiles" else None)
    opts = RunOptions(out_dir=a.out, mode=a.mode, arg=arg, repeats=a.repeats, unc=not a.no_unc,
                      workers=a.workers, only=tuple(a.only))
    cfg = load_config()
    if a.step in ("agreement", "all"):
        run_agreement(cfg, opts)
    if a.step in ("score", "all"):
        run_manual_scoring(cfg, opts)
    if a.step == "qc":
        from .qc import run_qc
        run_qc(cfg, opts)
    if a.step == "tobrain":
        from .tobrain import run_to_brain
        summary = run_to_brain(cfg, opts, with_raters=not a.no_raters)
        if a.figures and not a.no_raters:
            from .qc import inbrain_figure
            for brc in summary["brainRegionComb"]:
                if not opts.only or brc in opts.only:
                    inbrain_figure(cfg, opts, brc)
    if a.step in ("asr25", "asr25-raters"):
        from .reslice import migrate_legacy_layout
        migrate_legacy_layout(opts.staple_dir() / "asr25um")
    if a.step == "asr25-raters":
        from .reslice import run_raters_asr25
        brains = [b for b in cfg.brains if not a.only or b in a.only]
        run_raters_asr25(cfg, opts, brains=brains, gzip=a.gzip)
        if a.figures:
            from .qc import asr25_figure
            for b in brains:
                asr25_figure(cfg, opts, b, gzip=a.gzip)
    if a.step == "asr25":
        import json
        from .reslice import run_asr25, verify_asr25
        brains = [b for b in cfg.brains if not a.only or b in a.only]
        run_asr25(cfg, opts, brains=brains, gzip=a.gzip, stamp=a.stamp)
        for b in brains:
            res = verify_asr25(cfg, opts, b, gzip=a.gzip)
            print(json.dumps({k: v for k, v in res.items() if k != "structures"}, default=float))
            if a.figures:
                from .qc import asr25_figure
                asr25_figure(cfg, opts, b, gzip=a.gzip)
    if a.step == "unzip":
        from .config import matlab_num2str
        from .reslice import decompress
        src = opts.staple_dir() / "inBrainSpace"
        sc = matlab_num2str(cfg.scale)
        files = [src / f"{brc}_{sc}_{kind}_inBrain.nii.gz" for brc in a.only for kind in ("STAPLE", "raterProbability")]
        decompress([f for f in files if f.exists()], src / "uncompressed")
    if a.step == "index":
        from .outputs_index import write_index
        write_index(a.out, list(cfg.brains))
    if a.step == "compare":
        from .compare import run_compare
        brains = [b for b in cfg.brains if not a.only or b in a.only]
        run_compare(cfg, opts, a.pipeline_root, brains=brains, hemisphere=a.hemisphere)
    print(f"outputs in {opts.staple_dir()}")


if __name__ == "__main__":
    main()
