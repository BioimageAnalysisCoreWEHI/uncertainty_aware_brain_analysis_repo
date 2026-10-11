# aMAP manual-annotation pipeline: from rater outlines to BrainGlobe `asr` 25 µm

This document records the whole pipeline built in `python/amap_validation`. It takes the aMAP human rater annotations
(Niedworok et al. 2016, *Nat Commun* 7:11879) to STAPLE consensus segmentations, and then into the space of the user's
brainreg outputs: each brain's own grid, BrainGlobe `asr` orientation, 25 µm, mirrored left–right. It was built to
validate the uncertainty-aware registration pipeline of the JoM manuscript against expert annotations.

Related documents:
* `RECONSTRUCTION.md`: the read-only investigation of the original MATLAB workflow (evidence for every convention).
* `README.md`: module-level documentation, options, outputs and test descriptions.
* `PLAN_inBrainSpace.md`, `PLAN_asr25um.md`: the approved design of stages 3 and 4, with their sample results.
* `<out>/OUTPUTS_INDEX.md`: generated index of every output folder and QC figure (step `index`).

## Reproducing everything

```powershell
Set-Location <repo>\python
.\run_full_pipeline.ps1              # = spatialdata_env python -m amap_validation full
```

`full` runs, in order: `agreement` → `score` → `qc` → `tobrain --no-raters` → `asr25 --stamp --figures` →
`asr25-raters --figures` → `index`. Roughly 1.5 h on the V: share: ~8 min of mask decoding (cached afterwards),
~20 min brain space, ~25 min `asr` brains, ~10 min raters.

Environment: `<HOME>\.conda\envs\spatialdata_env` (numpy 2, scipy, tifffile, pandas, PyYAML), no installs.
`analysis_env` must not be used: its BLAS crashes on any matrix operation. Run from PowerShell; Git-Bash heredocs to
conda python hang. All outputs go to `…\amap\aMAP_validation_python_outputs\` (outside the repo). Tests:
`python -m unittest discover -s tests -t .` (47 tests, 1 skipped without SimpleITK).

## Data in

| Input | Content |
|---|---|
| `data/userSegmentations/userSegmentations/User_XX/` | 880 rater segmentations: filled mask TIFFs (12109 × 9997, {0,255}), ImageJ `.roi` outlines, `_meta.txt`. 22 raters × 40 (10 structures × 3 brains + a blind repeat of one brain) |
| `data/globalData.yml`, `data/brainShifts.yml` | scale 0.25, brains, structures; per brain × stack: rigid matrix, z-range, stack width/height |
| `downscaledBrains/*.nii` | the six brains at 12.5 µm (x, y ↑anterior, z ↑dorsal), sform = diag(0.0125) + 0.0125 |

Brains: MV_Ntsr1_165, CR_Syt6CreRfp_35 (aMAP parameter-search brains); ER_Glt25d2Cre_9, AB_Gad67_223, MV131017_7,
MV_Ntsr1_169 (aMAP scoring brains). MV_Ntsr1_165/169 are the repeated brains (22 segmentations per structure).
Structures (file tag → Allen name): CING → ACA, S1BF → SSp, RSP, V1 → VISp, V2M → VISam, SUB, AHN, MV, VPM, SG → DG-sg.

## Stage 1: rater masks on the analysis canvas (`segtable.py`, `masks.py`)
* `build_segtable` ports `getSegmentationSessionData.m`: user, presentation index, brain, structure, slice number
  and attempt, all parsed from filenames.
* Each TIFF → `imread(tif)'` (MATLAB [x = column, y = row]) → `>0`, on the global canvas 12109 × 9997
  (`userRater.m:27-31`).
* Two downsamplings, exactly as the code uses them:
  - STAPLE input: `imresize(…, 0.25, 'nearest')` (0-based sample index 4u+2, mirror edge);
  - scoring: `imresize(logical, 0.25)` (bicubic + antialias, uint8 0/255, per-dimension rounding, > 128).
  - Canvas 3028 × 2500 at 4 µm.
* Cached as bit-packed `.npz` in `<out>/maskCache_0.25/` (880 files, ~7.5 min once).

## Stage 2: STAPLE / SBA consensus and scoring (`staple.py`, `scoring.py`, `pipeline.py`; steps `agreement`, `score`)
* One consensus per brain × structure (60), in 2-D. All raters are pooled onto one canvas whatever slice they chose.
* By default both attempts of the repeated brains are included (22 inputs), as in `userRater.m:49`. `--repeats first`
  gives the paper's first-attempt-only variant.
* STAPLE is binary EM (Warfield 2004) with NiftySeg `-unc` semantics: unanimous pixels are excluded from estimation,
  and the output is the hard label (posterior > 0.5). SBA = mean signed distance < 0. All 60 converged (≤ 61
  iterations).
* Scoring (`segAnalyser.scoreManualSegs`): Dice and 8-connected-perimeter Hausdorff, each rater vs its consensus.
  - Median Dice 0.888 (4 scoring brains) vs 0.91 in the paper. The ~0.02–0.03 gap is unresolved; it is insensitive
    to STAPLE prior, majority vote and repeat policy, and was deliberately not tuned.
  - Structure ranking matches the paper (RSP/ACA/SSp best, VPM worst).
* Outputs `<out>/userStapleResults/`: `<brc>_0.25_NiftySegSTAPLE.nii`/`…SBA.nii` + `.json`, `segTable_with_pq.csv`,
  `staple_summary.csv`, `manual_scores.csv`.

## Stage 3: into the downscaled brains' grid, 12.5 µm (`geometry.py`, `tobrain.py`; step `tobrain`)
* The mapping is the inverse of the code's own forward path (`pavelAtlas.makeAtlasSegStack`):
  - `r = [0.004·i1, 0.015·J, 0.004·(Hs+1−j1)]`, then `f = M·r` (NiftyReg ref → flo), then
    `nii = f/0.0125 + [0, zRange(1)−1, 0]`;
  - M, zRange and H per stack come from `brainShifts.yml`; J = stack plane (the rater's slice number).
* Each consensus is placed on the **median rater slice** J0 (the authors' display convention); the slice spread is
  logged.
* The tilted plane is handled exactly: one voxel per (x, z) column, nearest-neighbour sampling (like
  `reg_resample -inter 0`).
* Also written: a rater-agreement probability map (fraction of raters, on J0) and a label map per brain (IDs 1–10).
* Checks over all 60:
  - header = brain's; one voxel per column; area 0.995–1.006;
  - round trip back through an emulation of the code's forward path, Dice 0.960–0.997;
  - probability = mean of rater masks exactly;
  - rater outlines within 7 µm of their own mask edge.
* Outputs `<out>/userStapleResults/inBrainSpace/` (`.nii.gz`, sparse-streamed; uncompressed they would be ~290 GB).
* Individual 12.5 µm rater volumes were deferred; the only ones written are for the AB RSP sample.

## Stage 4: mirrored BrainGlobe `asr` at 25 µm (`reslice.py`; step `asr25`)
* Index map shared by every product (`AsrGrid`):
  - a = ny−1−y, s = nz−1−z, **r = nx−1−x** (left–right mirror), then A = a//2, S = s//2, R = r//2;
  - a pure permutation + flips (the brains were already aligned to the Allen average), plus an exact 2×2×2 block
    reduction.
* This is the grid of the user's brainreg outputs. brainreg's `ipl` → `asr` conversion reverses all three axes, and
  the shapes match (AB is one S slice longer here: 365 vs brainreg's 364).
* The mirror is a display/matching choice; the anatomical L/R labelling stays inferred.
* Brains: block mean in the original dtype, `.nii` (affine keeps world alignment with the originals) + ImageJ TIFF
  (pages = coronal slices, 25 µm).
* Structures: 2-D coronal images, not 3-D sheets:
  - STAPLE (25 µm pixel kept if ≥ 50 % covered) and probability (2×2 mean);
  - the **slice index A** of each is logged in `asr25um_slices.csv`;
  - a 3-D label volume is stamped from them.
* Checks (all six brains pass):
  - block means exact at 4,000 random blocks per brain;
  - affine ≤ 4.4e-7 mm;
  - ACA anterior of MV, RSP superior of AHN;
  - every structure on the mirrored side of the midline (MV_Ntsr1_169 DG-sg on the other side, as its raters
    chose);
  - area ratio ≈ 1.00–1.04; upsampled round trip Dice 0.94–0.99.

## Stage 5: every rater mask at `asr` 25 µm (step `asr25-raters`)
* 880 segmentations × 2 variants = 1,760 masks, from the same cached canvas masks STAPLE used:
  - **ownSlice**: on the rater's own plane;
  - **medianPlane**: on the STAPLE slice.
* The log `asr25um_raters.csv` gives user, idx, attempt, original stack slice, variant, slice A and A − STAPLE A.
* Checks: none empty; every medianPlane mask exactly on its STAPLE slice; ownSlice within −15…+16 slices of it
  (±375–400 µm, the raters' AP disagreement); area ratio 0.87–1.11.

## Output layout (`<out>/userStapleResults/`)

```
segTable_with_pq.csv, staple_summary.csv, manual_scores.csv, canvas.json
<brc>_0.25_NiftySegSTAPLE.nii/.json, <brc>_0.25_NiftySegSBA.nii              stage 2 (4 µm canvas)
qc/  staple_outlines*.png, brain_<brain>.png, inBrain_<brc>.png, asr25_<brain>.png, morphology.csv
inBrainSpace/  <brc>_0.25_{STAPLE,raterProbability}_inBrain.nii.gz/.json, <brain>_STAPLE_labels.nii.gz,
               inBrain_summary.csv, tobrain.log                                  stage 3 (12.5 µm)
asr25um/  asr25um_slices.csv, asr25um_raters.csv, asr25um_all_masks.csv (+ .json)
          <brain>/  <brain>_asr_25um.nii/.tif, <brain>_STAPLE_labels_asr_25um.nii/.tif, <brain>_transform.json,
                    asr25um_checks_<brain>.json
                    <ACA|SSp|RSP|VISp|VISam|SUB|AHN|MV|VPM|DG-sg>/  <brc>_STAPLE_asr25um.nii/.tif,
                         <brc>_raterProbability_asr25um.nii/.tif, raters/{ownSlice,medianPlane}/*.nii/.tif   stages 4–5
```

## End-to-end transformation table

| Stage | Grid | Voxel | Axes | Operation | Code |
|---|---|---|---|---|---|
| rater TIFF | 12109 × 9997 (W × H), plane = slice s of a 40-plane stack | 1 µm (15 µm between planes) | rows ↓ ventral, cols ↑ segmented side | ROI → filled mask; crop/pad top-left to the global canvas (done by the authors) | data; `userRater.resizeSegImgsToMax` |
| canvas | 3028 × 2500, MATLAB [x, y] | 4 µm | transposed | `'`, `>0`, `imresize 0.25` (nearest for STAPLE, bicubic for scoring) | `masks.py` |
| STAPLE | same | 4 µm | same | STAPLE `-unc` / SBA over all raters | `staple.py` |
| brain space | brain dims (x, y, z) | 12.5 µm | x = image col, y ↑ anterior, z ↑ dorsal | `f = M·r` inverse, plane J0 / own slice, one voxel per column, nearest | `geometry.canvas_to_volume` |
| `asr` 25 µm | (A, S, R), e.g. AB 609 × 365 × 539 | 25 µm | A ↑ posterior, S ↑ inferior, R ↑ away from segmented side | a = ny−1−y, s = nz−1−z, r = nx−1−x; 2×2×2 mean (brains) or 2×2 (2-D structures) | `reslice.AsrGrid` |
| brainreg outputs | `registered_atlas.tiff` grid | 25 µm | same as `asr` above | brainreg `--orientation ipl -v 12.5` → asr | user's `scripts/run_brainreg_array.sh` |

## Decision log

| Decision | Choice | Why |
|---|---|---|
| Reference implementation | the MATLAB code, not the paper text | it is what exists; discrepancies are documented |
| Repeat attempts in STAPLE | both (22), `--repeats first` available | matches `userRater.m:49`; the paper says first only (consensus Dice 22 vs 11: median 0.964) |
| STAPLE engine | NumPy re-implementation of NiftySeg `-unc` | bundled binaries are Linux-only |
| Brain-space plane | median rater slice J0, slice spread logged | the consensus pools slices; the authors display it at the median |
| Brain-space sampling | nearest neighbour | as `reg_resample -inter 0` in the code |
| Structures at 25 µm | 2-D coronal images + slice index log | avoids losing one-voxel sheets when downsampling (user's suggestion) |
| Orientation | BrainGlobe `asr`, mirrored left–right | matches the user's brainreg outputs (user request) |
| File format | plain `.nii` (+ TIFF) at 25 µm; `.nii.gz` for 12.5 µm full grids | user preference; the 12.5 µm grids would be ~290 GB uncompressed |
| Environment | `spatialdata_env`, no installs | `analysis_env` BLAS is broken |

## Known limitations and open points
* Paper vs code: repeat attempts; the 5 excluded / manually cleaned segmentations are not in the repo (aberrant masks
  remain in the data); DG-sg excluded from the paper's analysis.
* The Dice ~0.02–0.03 below the paper is unresolved.
* Inferred (not stated anywhere):
  - TIFF masks top-left aligned with the stack;
  - NiftyReg direction ref → flo;
  - rater slice = reference plane J;
  - L/R labelling of the image columns.

  All of these are consistent with every check, and the anatomy is correct in all six brains.
* Not recoverable from the repo: the raw STP data and its alignment to the Allen average, and the 40-plane rater stacks.
* Approximate: NiftySeg STAPLE internals and SBA distance definition; MATLAB `imresize` internals were reimplemented
  from their documented behaviour, not run against MATLAB.
