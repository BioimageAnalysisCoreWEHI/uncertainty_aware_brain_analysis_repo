# aMAP validation — Python reproduction

A Python port of the human-rater part of the aMAP validation pipeline in this repository
(Niedworok et al. 2016, *Nat Commun* 7:11879, "aMAP is a validated pipeline for registration and segmentation of
high-resolution mouse brain data"). The MATLAB code in the parent directory is the reference: where the paper
and the code disagree, this port follows the code and records the difference (see
[Paper vs code](#known-paper--code-discrepancies)).

**Start here:**
* [`PIPELINE.md`](PIPELINE.md): the whole pipeline end to end, from rater outlines to STAPLE, then the downscaled
  brains, then mirrored BrainGlobe `asr` at 25 µm. Includes commands, outputs, checks, decisions and a transformation
  table.
* [`RECONSTRUCTION.md`](RECONSTRUCTION.md): the investigation of the original MATLAB workflow (evidence for every
  convention).
* [`PLAN_inBrainSpace.md`](PLAN_inBrainSpace.md), [`PLAN_asr25um.md`](PLAN_asr25um.md): stage designs and sample
  results.
* `<out>/OUTPUTS_INDEX.md`: generated index of every output folder, log and QC figure (`python -m amap_validation index`).
* Reproduce everything: `.\run_full_pipeline.ps1` (= `python -m amap_validation full`).

## What it implements

| Step | Python | Original MATLAB |
|---|---|---|
| Segmentation inventory: user, presentation index, brain, structure, slice, duration | `segtable.py` | `helpers/getSegmentationSessionData.m` |
| Mask loading: `imread(tif)'`, `>0`, zero-pad to the global canvas | `masks.read_mask_xy` | `userRater.create4dNii`, `userRater.resizeSegImgsToMax` |
| STAPLE input downsample: `imresize(…, 0.25, 'nearest')` | `masks.imresize_nearest` | `userRater.m:106` |
| Scoring downsample: `imresize(logical, 0.25)` (bicubic + antialias) | `masks.imresize_logical_bicubic` | `segAnalyser.m:156` |
| Consensus per brain × structure: STAPLE (`-unc`) and SBA | `staple.staple`, `staple.sba` | `seg_LabFusion -STAPLE -unc` / `-SBA` (`userRater.runStaple`) |
| z-window modes: all slices / 7-slice window / quartiles | `staple.valid_slices` | `userRater.create4dNii` |
| Manual scoring: Dice and Hausdorff vs STAPLE and SBA | `scoring.py`, `pipeline.run_manual_scoring` | `helpers/diceScore.m`, `hausdorffFromImages.m`, `padToLarger.m`, `segAnalyser.scoreManualSegs` |
| STAPLE canvas ↔ downscaled-brain geometry | `geometry.py` | derived from `pavelAtlas.makeAtlasSegStack` (inverted) |
| Config (`globalData.yml`, `brainShifts.yml`) and constants from the classes | `config.py` | `sharedParams`, `segAnalyser`, `pavelAtlas` |
| NIfTI I/O with MATLAB `make_nii`/`save_nii` header conventions (+ `.nii.gz`, streaming sparse writer) | `nifti.py` | NIfTI toolbox (`externalModules/NIfTI_20140122`) |
| Consensus, raters and rater agreement placed in downscaled-brain space | `tobrain.py`, `geometry.canvas_to_volume` | new (inverse of `pavelAtlas.makeAtlasSegStack`) |
| QC figures and morphology table | `qc.py` | — |
| ImageJ `.roi` reader (tests and overlays only) | `roi.py` | — |

**Not implemented:** scoring of the automated aMAP segmentation, i.e. resampling a registered atlas into each
rater stack and taking the max Dice over planes (`segAnalyser.scoreNiftySegs`). It needs the registered atlases,
which are not in the repository. `geometry.py` contains the mapping that step needs.

## Environment

Developed and run with the existing conda env `spatialdata_env` (Python 3, numpy 2.4.6), with **no extra
installs**. Required: `numpy`, `scipy`, `pandas`, `tifffile` (+ `imagecodecs` for PackBits TIFFs), `PyYAML`.
Optional: `SimpleITK` (one cross-check test, skipped when absent). nibabel is not needed.

> On this machine `analysis_env` has a broken BLAS: any numpy matrix multiply or `linalg` call kills the process
> (Windows error 0xC06D007F). Don't use it for this code.

Run from PowerShell, in this `python/` directory:

```powershell
$py = "python"
Set-Location <repo>\python
```

## Inputs

All read from the repository:

* `data/userSegmentations/userSegmentations/User_XX/` — 880 filled masks (`*..tif`, 12109 × 9997, {0,255}),
  ImageJ `.roi` outlines and `_meta.txt`. Filenames encode presentation index, brain, stack, structure and slice.
* `data/globalData.yml` (scale 0.25, brains, structures) and `data/brainShifts.yml` (per-stack rigid matrix,
  z-range, stack width/height).
* `downscaledBrains/*.nii` — only for the geometry tests and for mapping results into brain space.

## Running

```powershell
& $py -m amap_validation all                       # agreement segmentations + manual scoring (defaults = MATLAB code)
& $py -m amap_validation agreement                 # STAPLE/SBA only
& $py -m amap_validation score                     # scoring only (needs a previous agreement run)
& $py -m amap_validation all --only AB_Gad67_223RSP   # one brain x structure
& $py -m amap_validation qc                        # morphology table + QC figures for an existing run
```

Options:

| Flag | Meaning | Default |
|---|---|---|
| `--out DIR` | output directory | `..\..\aMAP_validation_python_outputs` (sibling of the repo) |
| `--mode range --range 7` / `--mode quantiles` | z-window variants of `createAgreementSegs.m` | all slices |
| `--repeats first` | paper variant: first attempt only for the repeated brains | `both` (as in the code) |
| `--no-unc` | STAPLE statistics over all voxels | `-unc` behaviour |
| `--workers N` | processes for mask decoding | 6 |

Every variant writes to its own subfolder: `userStapleResults[_zRange7|_zRangeQuartiles][_firstAttemptOnly][_noUnc]`.

**Full analysis:** `& $py -m amap_validation all` reproduces everything reported below. The first run decodes all
880 TIFFs (≈7.5 min with 6 workers) and caches the downsampled masks; later runs reuse the cache.

## Outputs

In `<out>/userStapleResults*/`:

| File | Content |
|---|---|
| `<brain><REGION>_0.25_NiftySegSTAPLE.nii` | STAPLE consensus mask, uint8 {0,1}, 3028 × 2500 |
| `<brain><REGION>_0.25_NiftySegSBA.nii` | SBA consensus mask, same geometry |
| `<brain><REGION>_0.25_NiftySegSTAPLE.json` | raters (`user:idx`), slices, per-rater sensitivity p / specificity q, prior, iterations, convergence, areas, STAPLE-vs-SBA Dice |
| `segTable_with_pq.csv` | one row per segmentation, with p, q and whether it went into STAPLE |
| `staple_summary.csv` | one row per brain × structure |
| `manual_scores.csv` | per segmentation: `diceSTAPLE`, `diceSBA`, `hdSTAPLE`/`hdSBA` (4 µm px) and `_um` |
| `canvas.json` | global canvas size (maxX, maxY) |

`<out>/maskCache_0.25/` holds the bit-packed downsampled masks (safe to delete).

The `qc` step (`qc.py`) adds `<out>/userStapleResults*/qc/`:

| File | Content |
|---|---|
| `morphology.csv` | per brain × structure: connected components, hole pixels, solidity, area relative to the median rater, fraction inside the rater union, centroid offset from the raters (µm), and whether the centroid falls in the expected hemisphere of the downscaled brain |
| `staple_outlines.png` | all rater outlines (grey) with the STAPLE (solid) and SBA (dashed) consensus, one panel per brain × structure, cropped around the consensus; outlines leaving the crop are counted |
| `staple_outlines_<brain>.png` | the same for one brain, larger panels (2 × 5) for close inspection |
| `brain_<brain>.png` | STAPLE consensus mapped with `geometry.py` onto the downscaled brain, one coronal plane per stack (at the median rater slice), code midline dashed |

QC of the default run: all 60 consensus centroids fall in the expected hemisphere; 100 % of every consensus lies
inside the rater union; consensus area / median rater area is 0.83–1.37 (median 1.00); 55/60 are one connected
region, and the rest have only negligible extra specks. Visually, every structure lands on the correct anatomy in all
six downscaled brains (e.g. RSP → VISam → VISp medial to lateral in dorsal cortex, MV beside the 4th ventricle,
AHN in the ventral hypothalamus). This is the cross-brain placement check of the geometry.

**Mask geometry.** Arrays and `.nii` files use MATLAB orientation: index `[x, y]` = [TIFF column, TIFF row],
4 µm pixels, top-left aligned with the rater-stack plane. As in the original, the NIfTI header has a placeholder
1 mm pixel size (sform = identity + 1 offset), so physical geometry is **not** in the header. Transpose (`.T`) for
image-style display. The consensus is 2-D and pools slices 1–40, so it has no single z position.
`geometry.StackMapping` maps canvas pixels ↔ downscaled-brain voxels for a chosen stack plane.

## Consensus and raters in downscaled-brain space

`& $py -m amap_validation tobrain [--only <brc> ...] [--no-raters] [--figures]` places the 2-D results of a finished
run into the grid of each brain's `downscaledBrains/*.nii`: same dimensions, same index order and a copy of its sform.
They overlay directly on the downscaled brain in ITK-SNAP, napari or Fiji. Design and checks:
`PLAN_inBrainSpace.md`.

**Method** (`geometry.canvas_to_volume`, `tobrain.py`):
* **Plane.** The consensus pools rater slices 1–40 and has no z of its own. It is placed on the **median rater slice**
  J0 (the authors' display convention, `segAnalyser.m:282`); the sidecar records the full slice spread. Each rater
  annotation is written twice: on its **own slice** (J = SliceNN, its true position) and on J0.
* **Tilt.** The stack plane is slightly tilted in the brain grid (shift-matrix rotations ≤ 1.7e-3 rad, i.e. up to
  ~1.7 voxels in y across a whole brain). For each (x, z) voxel column, the voxel nearest the plane is chosen, giving
  exactly one voxel per column.
* **Sampling.** Nearest neighbour from the 4 µm canvas: the value at the canvas pixel nearest the voxel centre, as
  with `reg_resample -inter 0` in the code. The rater masks are the same nearest-downsampled canvas masks STAPLE was
  given.
* **Probability.** The fraction of raters covering each voxel, pooled on the J0 plane, i.e. exactly the input STAPLE
  sees. It equals the mean of the median-plane rater volumes.
* **Label map.** IDs 1–10 in `regionsUsed` order: 1 MV, 2 SUB, 3 DG-sg (SG), 4 RSP, 5 VISp (V1), 6 VISam (V2M), 7 VPM,
  8 AHN, 9 SSp (S1BF), 10 ACA (CING). Where structures of one stack overlap, the lowest ID wins; overlaps are counted
  in the JSON.
* Files are written from sparse voxel lists by a streaming gzip writer (`nifti.save_sparse`), so ~1 G-voxel grids
  never exist densely in memory (~3.7 MB per file).

**Outputs** in `<out>/userStapleResults/inBrainSpace/`:

| File | Content |
|---|---|
| `<brc>_0.25_STAPLE_inBrain.nii.gz` | STAPLE consensus, uint8 {0,1}, on J0 |
| `<brc>_0.25_raterProbability_inBrain.nii.gz` | rater agreement, float32 0–1 (steps of 1/n), on J0 |
| `<brc>_0.25_STAPLE_inBrain.json` | brain file, stack, n and raters, J0, slice stats and their nii y, voxel count, areas, centroid (voxel, mm), plane y-range, and all check results |
| `raters/<brc>/<user>_Seg<idx>_slice<s>_ownSlice_inBrain.nii.gz` | one rater's annotation on its own slice |
| `raters/<brc>/<user>_Seg<idx>_slice<s>_medianPlane_inBrain.nii.gz` | the same annotation on J0 |
| `raters/<brc>/raters.json` | per rater: user, idx, attempt, slice, J, voxel count, plane y-range |
| `<brain>_STAPLE_labels.nii.gz` / `.json` | per-brain label map, label table, overlap count, whether complete |
| `inBrain_summary.csv` | one row per brain × structure with every check |
| `tobrain.log` | progress log |

With `--figures`, `<out>/userStapleResults/qc/inBrain_<brc>.png` is written from the saved files: the coronal
plane with the probability map and STAPLE outline, a zoom with every rater's own-slice outline, and the AP spread of
the raters' planes.

**Checks** (recorded per structure in the JSON and `inBrain_summary.csv`):
1. Header and grid equal the brain file's; the read-back file equals the in-memory result.
2. One voxel per column.
3. Area preserved.
4. Round trip back through an emulation of the code's own forward path (`geometry.stack_plane_from_volume` =
   `reg_resample -inter 0` + `permute` + `flip`) vs the STAPLE canvas.
5. Rater ROI vertices land on the consensus boundary.
7. Own-slice rater masks lie on their own planes, and their vertices on their boundaries.
8. Per-rater median-plane round trips.
9. Probability exactness.

Sample `AB_Gad67_223RSP`:
- area ratio 1.0003, round-trip Dice 0.994, per-rater round trips ≥ 0.993;
- rater vertices lie a median 10 µm from the consensus boundary and 7 µm from their own boundary;
- probability exact.

Full table in `PLAN_inBrainSpace.md`.

## Coronal BrainGlobe `asr` at 25 µm

`& $py -m amap_validation asr25 [--only <brain> ...] [--stamp] [--figures] [--gzip]` reslices each downscaled brain
and its brain-space consensus products (previous section) to the BrainGlobe **`asr`** orientation of
`allen_mouse_25um`:
* axis 0 = anterior → posterior (coronal slices);
* axis 1 = superior → inferior;
* axis 2 is **mirrored left–right** relative to the original x (user request, 2026-09-23). It increases *away from*
  the segmented hemisphere, so the segmented hemisphere is on the left of each coronal image.

The mirror is a display/matching choice. The anatomical L/R labelling of the original x is only inferred (paper
Fig. 1a: structures in the left hemisphere). `AsrGrid(..., mirror_lr=False)` gives the unmirrored variant.

It then downsamples to **25 µm**. Design and checks: `PLAN_asr25um.md`. The 12.5 µm products in `inBrainSpace/` are
not mirrored; they stay aligned with the original `downscaledBrains` files.

**Transform** (`reslice.AsrGrid`, shared by every product):
* a = ny−1−y, s = nz−1−z, **r = nx−1−x** (mirror), then A = a//2, S = s//2, R = r//2.
* This is a pure permutation + flips, because the brains were already aligned to the Allen average, plus an exact
  2×2×2 block reduction.
* Odd dimensions give a last block of one voxel, and means use only the valid voxels.
* The written affine = original sform ∘ (block centre → original voxel), so the 25 µm files overlay the originals in
  world space.
* `brains/<brain>_transform.json` holds all index maps and matrices.

**Steps:**
* `asr25 --stamp --figures`: brain stacks, STAPLE and probability images, 3-D label volumes and the slice log.
* `asr25-raters --figures`: every rater mask, in two variants:
  - **ownSlice**, on the 25 µm slice of the plane the rater actually drew on;
  - **medianPlane**, on the STAPLE slice.

  They come from the same cached 4 µm masks STAPLE used, via the same plane placement and 2×2 reduction.
  880 segmentations × 2 = 1,760 masks, both attempts included for MV_Ntsr1_165/169; filter by `attempt` for the
  paper's first-attempt-only version.

**Outputs** in `<out>/userStapleResults/asr25um/` (plain `.nii` by default, `--gzip` for `.nii.gz`; every image
also as `.tif`):

```
asr25um/
  asr25um_slices.csv/.json      one row per brain × structure: label ID, names, n raters, rater-slice stats,
                                asr slice index A (+ slices touched), AP position (mm), area ratio, file paths
  asr25um_raters.csv/.json      one row per rater mask: brain, structure, user, idx, attempt, rater stack slice,
                                variant, asr slice A, A − STAPLE A, area ratio, file path
  asr25um_all_masks.csv/.json   master index of every image (brain, labels_3d, STAPLE, probability, rater_*)
  <brain>/
    <brain>_asr_25um.nii/.tif                 brain (2×2×2 block mean, original dtype; TIFF pages = coronal slices)
    <brain>_STAPLE_labels_asr_25um.nii/.tif   3-D label volume (IDs 1–10), each structure on its slice A
    <brain>_transform.json                    orientation (mirror), shapes, index maps, P, affines
    asr25um_checks_<brain>.json               checks
    <ACA|SSp|RSP|VISp|VISam|SUB|AHN|MV|VPM|DG-sg>/
      <brc>_STAPLE_asr25um.nii/.tif           2-D STAPLE mask (uint8), 25 µm pixel kept if ≥ 50 % covered
      <brc>_raterProbability_asr25um.nii/.tif 2-D rater agreement (float32), 2×2 block mean
      raters/ownSlice/<user>_Seg<idx>_att<k>_slice<s>_ownSlice_asr25um.nii/.tif
      raters/medianPlane/<user>_Seg<idx>_att<k>_slice<s>_medianPlane_asr25um.nii/.tif
```

`slice<s>` is the rater's plane in the original 40-plane stack. The slice index A in the logs is 0-based along
axis 0; the Fiji page = A + 1. All paths in the logs are relative to `asr25um/`. The first `asr25` layout
(`brains/`, `structures/`) is migrated automatically (`reslice.migrate_legacy_layout`).

The 2-D NIfTIs have shape (1, S, R) with an affine placing them at slice A, so they overlay the 25 µm brain. The TIFFs
are single images.

The 12.5 µm brain-space volumes of the previous section stay `.nii.gz`: uncompressed they would total ~290 GB.
`& $py -m amap_validation unzip --only <brc> ...` writes plain `.nii` copies of chosen ones to
`inBrainSpace/uncompressed/`.

**Checks, full mirrored run (all six brains pass):**
- Block means exact at 4,000 random blocks per brain.
- Affine vs child world positions ≤ 4.4e-7 mm.
- ACA anterior of MV and RSP superior of AHN in every brain.
- Every structure's centroid on the expected (mirrored) side of the code midline, including the other-hemisphere
  MV_Ntsr1_169 DG-sg.
- Raters: all 1,760 masks non-empty; every medianPlane mask exactly on its STAPLE slice; ownSlice masks within
  −15…+16 slices of it (the raters' AP spread); area ratio 0.87–1.11 (median 1.02).
- All 1,772 files in the master index exist.

Figures: `qc/asr25_<brain>.png` (STAPLE, probability and the medianPlane rater outlines, with the own-slice range).

Earlier sample check (AB_Gad67_223, before the mirror; unaffected by it), all pass:
- Block means exact at 4,000 random blocks.
- Affine vs child world positions ≤ 3e-7 mm.
- Directions: ACA anterior of MV, RSP superior of AHN.
- Slice index consistent with the 12.5 µm plane for all ten structures.
- Area ratio 1.006–1.039 (≈ +1.6 % from keeping half-covered blocks).
- Upsampled round trip Dice 0.98–0.99, and 0.94 for the thin DG-sg.

Figure: `qc/asr25_<brain>.png`.

## Tests

```powershell
& $py -m unittest discover -s tests -t . -v
```

47 tests (1 skipped without SimpleITK), about 45–80 s; they need the repo data and `downscaledBrains/`.

* **asr 25 µm** (`test_reslice.py`): the reorientation is an exact index permutation with the right directions; block
  means match a reference loop, with odd dimensions; the affine puts each 25 µm centre at the mean world position of
  its children, with 25 µm spacing; the 2-D sheet path matches a dense reference, including a sheet crossing two y
  levels; the TIFF round trip keeps the spacing metadata.

* **Brain space** (`test_tobrain.py`): one voxel per column and |J − J0| ≤ half a voxel, also under a deliberately
  strong 0.05 rad tilt; area preservation; round trip through the code's forward path; probability = mean of rater
  volumes; streaming writer byte-identical to a dense write, `.nii.gz` round trip; label-map overlap rule.

* **Geometry:** toy check of the code's `permute`/`flip` indexing; rigid, near-identity shift matrices; exact
  round trip over all 880 ROIs' vertices (error < 1e-6 µm); discretised round trip through the 4 µm canvas and
  12.5 µm voxels (≤ 12.4 µm in-plane, slice always recovered); every ROI centroid maps inside its brain volume.
* **Masks:** nearest-index rule and mirror boundary, output sizes, bicubic weight normalisation and area preservation.
* **STAPLE / scoring:** recovery of synthetic truth and p/q, `-unc` consensus handling, optional SimpleITK
  agreement, SBA, z-window selection, MATLAB `quantile`, Dice, top-left padding, Hausdorff, 8-connected perimeter.
* **Segmentation table:** filename parsing, 880 rows, 11 per brain × structure (22 for the repeated brains),
  attempt numbering.
* **NIfTI:** write/read round trip with the MATLAB-style header; real brain headers.

## Reproduced results vs the paper

Default run (MATLAB-code behaviour); human Dice vs STAPLE consensus, DG-sg excluded as in the paper.

| Statistic | This reproduction | Paper |
|---|---|---|
| Median Dice, 4 scoring brains (n = 495) | 0.888 | 0.91 (Fig. 2b–d) |
| Median inter-rater Dice, repeated brains, attempts 1+2 (n = 396) | 0.890 | 0.916 (Fig. 3e) |
| SD, pooled scoring brains | 0.175 | 0.16 |
| ACA median (scoring brains) | 0.925 | 0.952 |
| SSp median (scoring brains) | 0.922 | 0.950 (inter-rater) |

Per structure (medians):

| Structure | 4 scoring brains | Repeated brains (inter-rater) | All 6 brains | SBA, all 6 | Hausdorff µm |
|---|---|---|---|---|---|
| ACA (CING) | 0.925 | 0.933 | 0.925 | 0.917 | 112 |
| SSp (S1BF) | 0.922 | 0.931 | 0.915 | 0.913 | 298 |
| RSP | 0.943 | 0.930 | 0.931 | 0.922 | 185 |
| VISp (V1) | 0.905 | 0.900 | 0.891 | 0.860 | 198 |
| VISam (V2M) | 0.903 | 0.883 | 0.889 | 0.875 | 196 |
| SUB | 0.859 | 0.890 | 0.872 | 0.848 | 165 |
| AHN | 0.837 | 0.864 | 0.856 | 0.842 | 132 |
| MV | 0.836 | 0.850 | 0.839 | 0.818 | 180 |
| VPM | 0.672 | 0.581 | 0.640 | 0.517 | 232 |

All 60 STAPLE runs converged (≤ 61 iterations). The ranking of structures matches the paper (RSP/ACA/SSp best,
VPM worst).

### Unresolved: Dice about 0.02–0.03 below the paper
The reproduced Dice scores are systematically lower than the published ones. We have **not** tuned the
implementation to close the gap. Checks done on the four scoring brains:

| Variant | Median Dice |
|---|---|
| Default (`-unc`, prior = mean decision over uncertain voxels) | 0.888 |
| `-unc`, prior 0.5 | 0.889 |
| Majority vote instead of STAPLE | 0.888 |
| First attempt only for the repeated brain (paper wording) | 0.885 |
| No `-unc` (consensus 25 % larger than the average rater) | 0.861 |

So the STAPLE settings and the repeat policy don't explain it. Plausible, unverified causes: the paper's manual
cleaning and 5 excluded segmentations (neither is in the repository); the published figures being computed from
data or scoring code that differ from this repository; and residual differences between MATLAB/NiftySeg internals
and this port (below). Resolving it needs the authors' STAPLE outputs or logs (possibly on GIN
`brainglobe/amap_validation`).

## Known paper / code discrepancies

1. **Repeated attempts.** The paper: "for inter-rater analyses, only the first segmentation of the repeated brain was
   used". The code (`userRater.m:49`) feeds **both** attempts of MV_Ntsr1_165 and MV_Ntsr1_169 into STAPLE (22 inputs)
   and scores all of them. Default here = code; `--repeats first` = paper.
2. **Manual cleaning and 5 excluded segmentations.** The paper says outlines were cleaned ("isolated touches") and
   5 wrong-structure/hemisphere segmentations were excluded. Neither step is in the code, and there is no exclusion
   list. The TIFFs still contain aberrant shapes, e.g. CR VPM User_10 Seg11, ER SUB User_02 Seg22,
   ER V2M User_13 Seg24, 169 MV User_19 Seg17, 169 SUB User_05 Seg32. Nothing is excluded here.
3. **DG-sg** is computed but excluded from summaries (as the code does via `meanDiceStapleNoSg`). For MV_Ntsr1_169 most
   raters outlined the opposite hemisphere; its SBA consensus is empty.
4. The paper names `seg_maths` for STAPLE/SBA; the code uses `seg_LabFusion` (`seg_maths` only merges).
5. The code downsamples masks differently for STAPLE (nearest) and for scoring (bicubic). Both are reproduced.

## Exact, inferred, approximate

**Follows the code exactly** (same operations, same order; verified by tests or by construction):
segmentation parsing and grouping; global canvas; transpose to `[x, y]`; binarisation; which segmentations enter
each STAPLE (all slices by default, both attempts); z-window rules; output names and NIfTI header convention;
Dice; top-left `padToLarger`; symmetric Hausdorff between 8-connected perimeters; the geometry formulas in
`pavelAtlas.makeAtlasSegStack` (the `permute`/`flip` indexing is checked on a toy array).

**Reimplemented from MATLAB behaviour, not checked against MATLAB** (no MATLAB here):
* `imresize` nearest: 1-based input index `4u − 1` with mirror clamping at the edge.
* `imresize` on a logical: uint8 0/255, bicubic with antialiasing per dimension (dim 1 first), rounding after each
  pass, then `> 128`.
* `bwperim` border handling (image edge counts as background).

**Approximate (NiftySeg internals not in the repo):**
* STAPLE: the EM follows Warfield 2004 with NiftySeg's documented `-unc` behaviour. Initialisation (0.95), prior
  (fixed, from the uncertain voxels), convergence test (max parameter change < 1e-5) and the 0.5 threshold are our
  choices. The outputs are insensitive to them (see the table above).
* p/q are written directly instead of being parsed from a NiftySeg log. With `-unc` they describe raters only over
  the disputed pixels, so they are much lower than whole-image values.
* SBA: mean signed Euclidean distance < 0 (Rohlfing & Maurer 2007); NiftySeg's distance definition may differ.

**Inferred geometry** (strong evidence, not stated anywhere):
* TIFF masks are top-left aligned with the rater-stack plane at 1 µm/px.
* The NiftyReg affine maps reference → floating world.
* Rater slice *s* = reference plane *J*.
* Presentation index order = attempt order.

These were checked on one brain/stack by overlaying real outlines on the downscaled brain. Cross-brain empirical
validation was deliberately not pursued. Raw STP data cannot be reached from the repository: the Allen-alignment
matrices and the 40-plane rater stacks are not included.

## Downstream: JoM validation analysis
The reproduction's outputs feed the validation package at
`<IMAGING>\Ishrat\JoM\post_review_analysis\validation_analysis\`. See its `validation_report.md` §8 for the current state
and next steps.
