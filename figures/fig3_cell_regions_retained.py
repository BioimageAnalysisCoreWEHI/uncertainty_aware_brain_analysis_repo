"""Per-cell regional assignment for Fig 3 B (Sample 1 = 162, preprocessed), exactly as
analysis_scripts/cell_region_probabilities_and_visualiser.py: cell centroids (regionprops, rounded) looked up in each
registration's <id>_snpc_atlas_region_full.tif; per cell, the fraction of registrations per region group
(SNc, SNr, VTA, MRN, RR, MB, other).

Checked first: with the 25 global-QC passers it must reproduce the stored table
final_count_results/cell_region_probability_outputs/162/preprocessed/cell_region_grouped_probabilities_long.csv.
Then written for the 13 retained registrations (global QC + top 50 % SSIM), in the same long format.
Output: source_data_fig3/162_preprocessed_retained_cell_region_grouped_probabilities_long.csv
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(_p for _p in _Path(__file__).resolve().parents if (_p / 'uaa_paths.py').exists())))
from uaa_paths import P  # noqa: E402  (resolves <IMAGING>/<SCRATCH> placeholders)
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops

VA = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\validation_analysis"))
_REPO = next(_p for _p in Path(__file__).resolve().parents if (_p / "uaa_paths.py").exists())
_CODE = _REPO / "validation" / "analysis" / "scripts"          # analysis scripts in this repository
_s = importlib.util.spec_from_file_location("m35", str(_CODE / "35_new_main_figures.py"))
m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(m)

J = Path(P(r"<IMAGING>\Ishrat\JoM"))
OUT = Path(P(r"<IMAGING>\Ishrat\JoM\post_review_analysis\manuscript_figures\source_data_fig3"))
S, V = "162", "preprocessed"
ROI = ["SNc", "SNr", "VTA", "MRN", "RR", "MB"]


def main():
    onto = pd.read_csv(J / "ARA2_annotation_info_avail_regions.csv")
    acr = dict(zip(onto["id"].astype(int), onto["acronym"]))
    labels = tifffile.imread(m.la.lc.label_path(S, V))
    props = regionprops(labels)
    ids = np.array([p.label for p in props], dtype=int)
    cz, cy, cx = np.round(np.array([p.centroid for p in props])).astype(int).T
    shape = labels.shape
    del labels
    runs = m.runs(S, V)
    glob_keys = list(runs[runs.passes_global].run_key)
    ret = m.retained(S, V)
    groups = {}
    for k in glob_keys:
        a = tifffile.imread(m.row(S, V, k).region_labels_path)
        ok = (cz < a.shape[0]) & (cy < a.shape[1]) & (cx < a.shape[2])
        rid = np.full(len(ids), -1)
        rid[ok] = a[cz[ok], cy[ok], cx[ok]].astype(int)
        del a
        g = np.array([acr.get(r, f"unknown_{r}") for r in rid], dtype=object)
        groups[k] = np.where(np.isin(g, ROI), g, "other")
        print("read", k, flush=True)

    def long(keys):
        rows = []
        stack = np.stack([groups[k] for k in keys])
        for grp in ROI + ["other"]:
            p = (stack == grp).mean(0)
            nz = p > 0
            rows.append(pd.DataFrame({"sample_id": S, "processing_status": V, "cell_id": ids[nz], "region_group": grp,
                                      "probability": p[nz], "z": cz[nz], "y": cy[nz], "x": cx[nz]}))
        return pd.concat(rows, ignore_index=True)

    chk = long(glob_keys).set_index(["cell_id", "region_group"]).probability
    old = pd.read_csv(J / "final_count_results" / "cell_region_probability_outputs" / S / V /
                      "cell_region_grouped_probabilities_long.csv").set_index(["cell_id", "region_group"]).probability
    j = pd.concat([chk.rename("new"), old.rename("old")], axis=1).fillna(0)
    print(f"check vs stored global-QC table: {len(j)} cell-group rows, max |diff| {(j.new - j.old).abs().max():.6f}")
    out = long(ret)
    out["n_registrations"] = len(ret)
    out.to_csv(OUT / "162_preprocessed_retained_cell_region_grouped_probabilities_long.csv", index=False)
    print("retained registrations:", len(ret), ret)


if __name__ == "__main__":
    main()
