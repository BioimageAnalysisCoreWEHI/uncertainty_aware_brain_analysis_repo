# Next stage: reslice to coronal BrainGlobe `asr` and downsample to 25 µm (plan, 2026-09-23)

Status: approved 2026-09-23. **Implemented; sample AB_Gad67_223 passed all checks (below).**

**Done 2026-09-23:** mirrored rewrite of all six brains, then all 1,760 individual rater masks (ownSlice +
medianPlane). Everything is in the per-brain/per-structure layout with three logs (`asr25um_slices`,
`asr25um_raters`, `asr25um_all_masks`). All checks pass; see the README section "Coronal BrainGlobe asr at 25 µm".

**Change 2026-09-23 (user request): all asr 25 µm outputs are mirrored left–right**: r = nx − 1 − x instead of
r = x, applied in `AsrGrid` so brain, structures and labels flip together. Axis 2 now increases away from the
segmented hemisphere. This is a display/matching choice; the anatomical L/R labelling stays inferred.
The earlier unmirrored outputs were deleted and all six brains are being rewritten; a hemisphere check (structure
centroid on the expected side of the code midline in R) was added to `verify_asr25`. The sample numbers below were
measured before the mirror. A mirror does not change areas, slice indices or round-trip values; they are re-checked
in the rewrite.

## Sample results: AB_Gad67_223

* Brain: 1078 × 1218 × 729 (12.5 µm) → `asr` (609, 365, 539) at 25 µm, uint16.
  - `.nii` and TIFF shapes match; TIFF spacing is 25 µm.
  - Block means are exact at 4,000 random blocks against an independent reference (max error 0).
  - Affine vs the mean world position of the children: max error 3e-7 mm (float32 header precision).
* Directions: ACA (A = 232) is anterior of MV (A = 536); RSP is superior of AHN.
* Structures:

| Structure | A | Area ratio (25 µm / canvas) | Dice, upsampled vs 12.5 µm sheet |
|---|---|---|---|
| MV | 536 | 1.018 | 0.984 |
| SUB | 412 | 1.016 | 0.985 |
| DG-sg | 412 | 1.039 | 0.937 |
| RSP / VISp / VISam | 389 | 1.016 / 1.015 / 1.017 | 0.989 / 0.989 / 0.988 |
| VPM | 341 | 1.018 | 0.981 |
| AHN | 327 | 1.018 | 0.983 |
| SSp | 269 | 1.006 | 0.994 |
| ACA | 232 | 1.016 | 0.987 |

* A matches (ny − 1 − y)//2 from §10 for all ten, and each structure's slice affine is correct. Every sheet falls in a
  single 25 µm slice, i.e. the tilt never straddled two slices here.
* The ~+1.6 % area is the expected bias of the "≥ half of the 2×2 block" rule (half-covered blocks are kept).
* Figure `qc/asr25_AB_Gad67_223.png`: all ten on correct anatomy, superior up, segmented hemisphere on the right.
* Runtime ~4 min per brain. Sizes: 686 MB for the brain `.nii` + `.tif` + stamped label volume; 19 MB for the 20
  structure files.

## Context
The downscaled brains (and everything placed in their grid in §10) are stored as [x, y = AP (↑ anterior),
z = DV (↑ dorsal)] at 12.5 µm. The user wants them resliced to **coronal BrainGlobe `asr`**: axis 0 anterior →
posterior (coronal slices), axis 1 superior → inferior, axis 2 right → left. That is the convention of
`allen_mouse_25um` (brainreg/brainglobe). They also want them **downsampled to 25 µm**, with brains and labels moved
by the identical transform.

The brains were rigidly aligned to the Allen average before downscaling, so the coronal axis is exactly nii y. The
reslice is therefore a pure axis permutation + flips (no interpolation), and 12.5 → 25 µm is an exact 2× block
reduction.

User decisions:
* `asr`; NIfTI + TIFF outputs.
* Scope = brain images, label information, per-structure STAPLE masks and probability maps.
* **The structures become 2-D coronal images in the `asr` 25 µm grid plus a log of their slice index**, instead of
  3-D one-voxel sheets (user's suggestion). This avoids the sheet-downsampling problem and label overlaps.

## Index transform (defined once, shared by every product)
Original 0-based voxel (x, y, z), dims (nx, ny, nz):
* `asr` at 12.5 µm: a = ny − 1 − y (anterior first), s = nz − 1 − z (superior first), r = x (↑ toward the
  segmented = left hemisphere per paper Fig. 1a) [I: L/R labelling is inferred, as in §1].
* 25 µm: A = a // 2, S = s // 2, R = r // 2. Blocks start at the anterior/superior/right edge. An odd dimension
  leaves a last block of 1, and means are taken over the valid voxels only, so no padding bias.
* 25 µm affine = original sform ∘ P, where P maps the block centre (A, S, R) → (x = 2R + 0.5,
  y = ny − 1.5 − 2A, z = nz − 1.5 − 2S). So the 25 µm files overlay the originals in world space in any viewer.
  The sform is written with this (permuted, flipped) matrix; the TIFF carries 25 µm spacing and the `asr` string.
* A per-brain `transform.json` records dims, the index maps, P, both affines and the `asr` orientation, so any
  coordinate can be mapped back and forth.

## File format: uncompressed by default (user request)
All `asr` 25 µm NIfTI outputs are plain, **uncompressed `.nii`** (normal NIfTI-1, readable everywhere and
memory-mappable), alongside the TIFFs. `--gzip` optionally writes `.nii.gz` instead. File names below use `.nii`.

Sizes are fine: 25 µm brains are ~240 MB each (~1.4 GB for six); the 2-D structure images are ~0.2–0.8 MB each.

The §10 12.5 µm full-grid volumes stay `.nii.gz`. Uncompressed they would be ~1 GB (uint8) / ~3.9 GB (float32)
each, ~290 GB for all 120. They are not converted wholesale. A small helper `decompress(path)` (new step
`unzip --only <brc>`) can write a plain `.nii` copy of chosen files on demand.

## Products (in `<out>/userStapleResults/asr25um/`)
* **Brains:** `brains/<brain>_asr_25um.nii` + `.tif` (multi-page, one coronal slice per page, ImageJ metadata
  spacing 25 µm, axes ZYX = A, S, R). Values = 2×2×2 block mean of the original intensities, rounded to the original
  dtype (uint16, ER uint8). Streamed by z-pairs of the original file (sequential read, ~2 input planes in memory).
  AB e.g. 1078 × 1218 × 729 → `asr` shape (609, 365, 539).
* **Per structure (60), 2-D coronal images on the 25 µm grid of their brain:**
  - `structures/<brc>_STAPLE_asr25um.nii` + `.tif`: uint8 {0,1}. From the §10 12.5 µm volume: map voxels to
    (a, s, r), project out a (the sheet), 2×2 block in (s, r), and keep the pixel if at least half the block is
    inside. The NIfTI is shape (1, nS, nR), with an affine that places it at its slice so it overlays the 25 µm brain.
  - `structures/<brc>_raterProbability_asr25um.nii` + `.tif`: float32, 2×2 in-plane block mean (no AP
    dilution).
* **Log `asr25um_slices.csv` (+ JSON):** one row per brain × structure: label ID/abbrev/paper name, n raters, J0 and
  the rater-slice stats, 12.5 µm y range of the sheet, `asr` 25 µm slice index A (median) and its range (the tilt can
  straddle two slices), AP world position (mm), area at 25 µm vs the STAPLE canvas area, and file names.
* **Labels:** the per-structure 2-D images + the log replace the 3-D label maps. A helper `stamp_labels(brain)` can
  rebuild a 3-D 25 µm label volume from them on demand (not written by default). The existing 12.5 µm 3-D label maps
  from §10 stay as they are.

## Code (under `python/`; nothing else touched)
* `amap_validation/reslice.py` (new):
  - `AsrGrid(shape12, sform12)`: index maps (to/from `asr` 12.5 and 25 µm), P, the 25 µm affine, `transform.json`.
  - `downsample_brain(cfg, brain, dest)`: streaming 2×2×2 block mean → `.nii.gz` + `.tif`.
  - `structure_to_asr25(...)`: STAPLE and probability sheets → 2-D images + log row.
  - `run_asr25(cfg, opts)`: all brains/structures, log, figures.
* Reuses `nifti.load(mmap=True)`, `nifti.load_sparse`, `nifti.save` / `save_sparse`, `config.Config.brain_file`,
  `tobrain.label_ids`, `qc.PAPER_NAME`; TIFF via `tifffile` (already in spatialdata_env).
* `amap_validation/__main__.py`: new step `asr25` (respects `--only` brains/structures).
* `amap_validation/qc.py`: `asr25_figure(brain)` shows, for each structure, the 25 µm coronal slice A with the
  STAPLE outline and probability, read back from the files.
* `tests/test_reslice.py` (new): see Verification.

## Sample first: AB_Gad67_223 (brain + its 10 structures), then the other five brains after review.

## Verification
* **Unit tests** (synthetic volumes):
  - reorientation is an exact index permutation (values identical at mapped indices, `asr` directions correct);
  - the block mean equals a reference loop, including odd dims;
  - the world position of each 25 µm voxel centre (from the written affine) = mean world position of its 8 children;
  - the 2-D mask/probability path gives the same result as a dense reference;
  - the TIFF round trip keeps shape, dtype and spacing metadata;
  - `stamp_labels` puts each structure only on its slice.
* **Sample (AB):**
  1. Shapes and dtypes; the affine maps the 25 µm grid onto the original world extent.
  2. Sampling the original volume at 25 µm voxel centres matches the downsampled volume (correlation ≳ 0.99).
     Direction check: dorsal is at low S and anterior at low A. Using structure positions: ACA (anterior) should get
     a lower A than MV (posterior), and cortex a lower S than AHN.
  3. Structure areas at 25 µm / canvas area ≈ 1 (±5 %). Upsampling each 2-D mask back to the 12.5 µm sheet gives
     Dice ≳ 0.9.
  4. Slice indices are consistent with §10's y (A = (ny − 1 − y) // 2).
  5. Visual: the `asr25` figure shows every structure on correct anatomy, matching the earlier `brain_<brain>.png`.
* Then all six brains; summary table in the README; `python/PLAN_asr25um.md` saved on approval and updated with
  results.
