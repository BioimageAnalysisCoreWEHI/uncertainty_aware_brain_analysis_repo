# Next stage: STAPLE consensus → downscaled-brain space (plan, 2026-09-23)

Status: approved 2026-09-23. **Implemented; sample `AB_Gad67_223RSP` passed all checks (below).** The full run
(all 60 brain × structure, 1,760 rater volumes, 6 label maps) is pending the user's go-ahead.

## Sample results: AB_Gad67_223RSP (n = 11 raters, stack rspV1V2M)

| Check | Result |
|---|---|
| 1 Grid/header (STAPLE + probability files; read back = in-memory) | pass: 1078 × 1218 × 729, brain sform, sform_code 1 |
| 2 One voxel per (x, z) column; tilt spread | pass; plane at nii y = 438 (spread 0, tilt < 0.5 voxel over RSP) |
| 3 Area (volume / canvas) | 1.0003 (1,455,469 vs 1,455,088 µm²); 0 canvas pixels off-grid |
| 4 Round trip through the code's forward path (Dice) | 0.994 |
| 5 Rater ROI vertices (own slice) → STAPLE boundary, median | 10.3 µm (< 1 voxel) |
| 7 Own-slice raters: vertices → own boundary / off own plane, median | 7.2 µm / 2.9 µm; all 22 rater headers OK |
| 7 Own-slice planes | y 419–440 for slices 6–24 (1.2 voxels/slice); User_03 chose slice 6, ~250 µm posterior of the others |
| 8 Median-plane raters, round trip Dice (min / median) | 0.993 / 0.994 |
| 9 Probability = mean of median-plane rater volumes; max; Dice(p > 0.5, STAPLE) | exact; 1.0; 0.993 |
| Runtime / size | 96 s for this structure (22 rater files); ~3.7 MB per file |

Figure: `qc/inBrain_AB_Gad67_223RSP.png`. Full-run estimate: ~1,880 files × ~4 s ≈ 2 h, ~7 GB.

## Context
The Python reproduction (`python/amap_validation`) produces 60 STAPLE masks on the 2-D rater canvas
(3028 × 2500, 4 µm, MATLAB [x = TIFF col, y = TIFF row], placeholder 1 mm header). The goal is to place each
consensus in the grid, orientation and header of its `downscaledBrains/*.nii` volume (12.5 µm, dims x,y,z, sform =
diag(0.0125) + 0.0125). It goes on a single coronal plane at a "rough z", so the masks can be overlaid directly in any
viewer and used downstream. The geometry is already verified: `geometry.StackMapping` passes exact/discretised
round-trip tests over all ROI vertices, and the `qc` brain figures show correct anatomy in all six brains.

User decisions: **z = one plane at the median rater slice** (the authors' display convention, segAnalyser.m:282;
the slice spread is recorded in a sidecar); **nearest-neighbour** sampling (matches `reg_resample -inter 0` in
pavelAtlas.m:94); outputs = **one binary mask per brain × structure + one label map per brain**. Additionally,
**every individual rater annotation in two variants**: (a) on the rater's **own slice** (J = SliceNN, their true 3-D
position) and (b) on the **STAPLE median plane**. Also a **rater-agreement probability map**: the fraction of raters
covering each voxel, all raters pooled onto the median plane, i.e. exactly the input STAPLE sees. Start with one
sample, check it, then run all.

## Design
**Plane selection (exact, handles the small rotations).** Rotations in the shift matrices reach 1.7e-3 rad. Across
a ~1000-voxel-wide brain that tilts the stack plane by up to ~1.7 voxels in y, so a single fixed nii plane is not
exact. Instead, the composite map is affine: stack plane coordinate J = a·x + b·y + c·z + d (from
`StackMapping.nii_to_staple`). For each (x, z) column, solve for the y where J = J0 and round half-up. That gives
exactly one voxel per column, lying on the tilted plane.

**Sampling (nearest).** For each selected voxel centre (x, y*, z) → `nii_to_staple` → (i0, j0) → round → take the
STAPLE canvas pixel if inside the canvas, else 0. Only columns inside the STAPLE bounding box (mapped into the brain,
plus a margin) are evaluated, for speed.

**J0.** MATLAB-style median of the included segmentations' `sliceNum` (a .5 value stays continuous). The sidecar
records min / q25 / median / q75 / max slices and the corresponding nii y at the mask centroid.

**Rater masks** use the same canvas masks STAPLE was given (the cached nearest-downsampled 4 µm masks,
`masks.cached_masks(...)[0]`), mapped with the same plane/sampling routine at J = SliceNN (own slice) or J = J0
(median plane). **Probability** = mean of the n rater canvas masks, sampled nearest onto the J0 voxels
(float32, 0–1). By construction it equals the voxel-wise mean of the median-plane rater volumes. The rater set and n
are the same segmentations that went into STAPLE: both attempts for MV_Ntsr1_165/169 under the default policy, and
fewer under `--repeats first` / z-window variants.

**Writing full-grid volumes without dense arrays.** A brain grid is ~0.95–1.0 G voxels (≈1 GB as uint8), and there
are 1,760 rater volumes plus 120 STAPLE/probability volumes. So all volumes are held as sparse voxel lists and written
by a streaming NIfTI writer: header, then one z-plane (X × Y, F-order) at a time through `gzip` level 1. Peak memory
is one plane. Zero-dominated planes compress to near nothing, so expect a few MB per file and ~2–3 s each; the full
run is roughly an hour, in the background with a log. The per-brain label map uses the same writer.

**Outputs** in `<out>/userStapleResults/inBrainSpace/` (outside the repo):
* `<brc>_0.25_STAPLE_inBrain.nii.gz`: uint8 {0,1}, **same dims as the brain file**, sform copied from it
  (sform_code 1, qform 0). Opens aligned over the downscaled brain in ITK-SNAP, napari or Fiji.
* `<brc>_0.25_STAPLE_inBrain.json`: brain file, stack, n, J0 and slice stats with their nii y, voxel count,
  area µm² vs the STAPLE canvas area, centroid (voxel and world mm), y-range of the tilted plane, pixels lost off-grid.
* `<brain>_STAPLE_labels.nii.gz` + `<brain>_STAPLE_labels.json`: uint8 label map, IDs 1–10 in `regionsUsed`
  order (MV, SUB, SG, RSP, V1, V2M, VPM, AHN, S1BF, CING), with abbrev / paper name / Allen ID. Structures in the
  same stack (RSP/V1/V2M) can share voxels. Rule: the first ID in that order wins, and the overlap voxel count is
  reported in the JSON.
* `<brc>_0.25_raterProbability_inBrain.nii.gz`: float32 0–1 on the STAPLE (J0) plane, same grid/header. The sidecar
  (`<brc>_0.25_STAPLE_inBrain.json`) gains n and the rater list.
* `raters/<brc>/<user>_Seg<idx>_slice<s>_ownSlice_inBrain.nii.gz` and `…_medianPlane_inBrain.nii.gz`: uint8 {0,1},
  same grid/header; 880 × 2 files. `raters/<brc>/raters.json` lists, per file: user, idx, attempt, slice, J used,
  nii y at its centroid, voxel count, and whether it went into STAPLE.

## Code changes (all under `python/`; MATLAB untouched)
* `amap_validation/geometry.py`
  - `StackMapping.plane_y(x, z, J)`: exact y of the J-plane for voxel columns.
  - `staple_mask_to_volume(mask_xy, mapping, J, nii_shape)`: returns voxel indices (x, y, z) of the mask on the tilted
    plane, via nearest sampling.
  - `stack_plane_from_volume(vol, mapping, J, canvas_shape)`: emulates the code's forward path. That is
    `reg_resample -inter 0` from the brain into the reference grid at plane J, then `permute([1 3 2])` + `flip(,2)`,
    giving the canvas-oriented [x, y] 4 µm mask. Used for verification now and reusable for later aMAP scoring.
  - Remove `staple_mask_to_nii_plane` (the nearest-fixed-plane shortcut) and switch `qc.brain_figures` to the new
    function, so only one convention remains.
* `amap_validation/nifti.py`: `.nii.gz` read/write via stdlib `gzip` (no memmap for .gz). New
  `save_sparse(path, shape, dtype, voxels, values, affine)`, a streaming writer (header + one z-plane at a time, gzip
  level 1), and `load_sparse(path)` (plane-by-plane read → nonzero voxels/values), so checks never need dense volumes.
* `amap_validation/tobrain.py` (new): `run_to_brain(cfg, opts, raters=True)` writes the STAPLE masks, probability
  maps, rater volumes (both variants), sidecars and label maps. It reuses `pipeline.RunOptions`, `select_rows`,
  `masks.cached_masks`, `config.Config.brain_file`, `nifti.read_header`. Logging and progress go to
  `<out>/…/inBrainSpace/tobrain.log`.
* `amap_validation/__main__.py`: new step `tobrain` (respects `--only`; `--no-raters` skips the 1,760 rater files).
* `tests/test_tobrain.py` (new): synthetic mask → volume → `stack_plane_from_volume` round trip; one voxel per column;
  J at selected voxels within ±0.42 of J0 (half a voxel / 1.2 voxels per slice); `.nii.gz` dense and sparse
  (streaming) round trip, identical bytes between dense and streaming writes of a small volume; label-map overlap
  rule; probability = mean of median-plane rater volumes on synthetic raters.
* `python/README.md`: new section "Consensus in downscaled-brain space" (method, conventions, outputs, checks),
  written after the sample passes and updated after the full run.

## Sample first: `AB_Gad67_223RSP` (then the rest only after review)
`& $py -m amap_validation tobrain --only AB_Gad67_223RSP`, then check:
1. **Grid/header:** output dims and sform equal the brain file's; it reads back with `nifti.load`.
2. **Plane:** at most one voxel per (x, z) column; report the y spread from the tilt.
3. **Area:** voxels × 156.25 µm² ≈ STAPLE px × 16 µm² (expect within a few %).
4. **Code-convention round trip:** `stack_plane_from_volume` of the written mask at J0 vs the original STAPLE canvas.
   Expect Dice ≳ 0.9 (limited by 12.5 µm quantisation). This is exactly the comparison `segAnalyser.scoreNiftySegs`
   would make.
5. **Rater consistency:** the RSP ROI vertices mapped with `tiff_to_nii` at each rater's own slice lie within the
   recorded slice spread in y, and near the mask boundary in-plane (median distance in µm).
6. **Visual:** brain plane at the mask's y with the outline read back from the `.nii.gz` (a `qc` figure). The
   neighbouring ±5 planes should contain no mask voxels.
7. **Raters, own slice:** each rater volume sits only on its own plane (y consistent with its SliceNN). Its own ROI
   vertices mapped with `tiff_to_nii` at that slice lie on its mask boundary (median distance ≲ one voxel, 12.5 µm).
   The per-rater y values spread along AP as the recorded slice spread predicts.
8. **Raters, median plane:** all on the STAPLE plane; `stack_plane_from_volume` round trip per rater (Dice ≳ 0.9).
9. **Probability:** values in [0, 1] in steps of 1/n; equals the voxel-wise mean of the median-plane rater volumes
   (exact). Dice of {p > 0.5} vs the STAPLE volume mask is high (canvas majority vote vs STAPLE had area ratio 0.98).
   Figure: probability heat map on the brain plane with the STAPLE outline (qc/`inBrain_<brc>.png`).

Then run all 60 structures plus 6 label maps and 1,760 rater volumes in the background with a log. Repeat the
automatic checks (1–4, 7–9) for all and add a summary table to the README. Spot-check the label maps and a few
rater/probability volumes visually.

## Documentation
* On approval, save this §10 plan as `python/PLAN_inBrainSpace.md`. Keep it updated with the sample findings, then
  mark it done after the full run.
* The README section "Consensus and raters in downscaled-brain space" is written after the sample passes: method,
  conventions (median-plane choice, nearest sampling, tilt handling, label IDs, overlap rule), outputs, how to run,
  and the check results. It is updated with the full-run summary table.

## Verification
* `& $py -m unittest discover -s tests -t .` (spatialdata_env; all existing and new tests pass).
* Sample checks 1–6 above; the full-run summary table (area ratio, round-trip Dice, overlap counts) goes in the README.
