"""Add the all-48-registration (no QC) inclusion-frequency maps in the full-resolution midbrain crop to the data release:
SampleN/fullres_midbrain/<arm>/inclusion_frequency_fullres_all48.tif (float32, zlib).

Source: get_consensus_counts.py output final_count_results/consensus_threshold_cell_counts_overlap_logic/<id>/<arm>/
all_runs_all/<id>_<arm>_all_runs_all_consensus_probability.tif (recompressed, values unchanged). Where that file is
missing it is computed as the mean of the 48 full-resolution masks (same definition). Provenance and MANIFEST updated.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_data_release as B  # noqa: E402

OUT = B.PRA / "data_availability"
PROV = B.PRA / "data_availability_PROVENANCE.csv"


def main():
    runs = B.runs_table()
    R = B.Release(OUT, dry=False, skip_hash=False)
    for s in B.IDS:
        for v in B.ARMS:
            rel = f"{B.LABEL[s]}/fullres_midbrain/{v}/inclusion_frequency_fullres_all48.tif"
            dst = R.path(rel)
            src = (B.JOM / "final_count_results" / "consensus_threshold_cell_counts_overlap_logic" / s / v / "all_runs_all"
                   / f"{s}_{v}_all_runs_all_consensus_probability.tif")
            g = runs[(runs["sample"] == s) & (runs.status == v)]
            meta = dict(sample=B.LABEL[s], arm=v, n_registrations=48,
                        note="fraction of all 48 registrations (no QC) that include the voxel")
            if src.exists():
                if not dst.exists():
                    shp, dt = B.tiff_info(src)
                    B.write_stack(dst, B.tiff_pages(src), shp, np.float32, meta)
                R.record(rel, src, "recompressed (zlib)", "all-48 inclusion frequency from get_consensus_counts.py")
            else:
                srcs = [Path(p) for p in g.mask_axial_path]
                if not dst.exists():
                    shp, _ = B.tiff_info(srcs[0])
                    files = [tifffile.TiffFile(p) for p in srcs]
                    try:
                        def pages():
                            for z in range(shp[0]):
                                acc = np.zeros(shp[1:], np.uint16)
                                for f in files:
                                    acc += f.pages[z].asarray() > 0
                                yield (acc / len(files)).astype(np.float32)
                        B.write_stack(dst, pages(), shp, np.float32, meta)
                    finally:
                        for f in files:
                            f.close()
                R.record(rel, srcs, "mean of the 48 full-resolution masks (float32, zlib)",
                         "all-48 inclusion frequency; computed because the get_consensus_counts.py output is missing")
            print("done", rel, flush=True)
    old = pd.read_csv(PROV)
    df = pd.concat([old[~old.release_path.isin([r["release_path"] for r in R.rows])], pd.DataFrame(R.rows)],
                   ignore_index=True).sort_values("release_path")
    df.to_csv(PROV, index=False)
    df[["release_path", "release_bytes", "release_sha256"]].rename(
        columns={"release_path": "path", "release_bytes": "bytes", "release_sha256": "sha256"}).to_csv(
        OUT / "MANIFEST.csv", index=False)
    print("provenance and manifest updated")


if __name__ == "__main__":
    main()
