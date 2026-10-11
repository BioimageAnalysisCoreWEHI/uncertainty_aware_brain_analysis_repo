# aMAP validation: reconstruction of the authors' MATLAB workflow

> Investigation carried out 2026-09-23 before any Python was written (read-only). It is the evidence base for the
> Python reproduction; see `PIPELINE.md` for what was then built. Two points were revised later:
> the empirical geometry check was extended to all six brains (QC figures `brain_<brain>.png`, `asr25_<brain>.png`),
> and the analysis environment is `spatialdata_env` (analysis_env's BLAS is broken).

## Context
Goal: reproduce the aMAP validation (Niedworok et al. 2016, *Nat Commun* 7:11879) in Python faithfully. Before
writing any code, this document reconstructs exactly what the MATLAB pipeline does to the manual segmentations,
how STAPLE is fed and used, and whether the STAPLE consensus can be mapped back onto the source images.
No repository files were modified. Header/TIFF/ROI facts come from inline stdlib-Python parsing of the data.

Evidence labels: **[E]** explicitly demonstrated (code, data, paper or wiki states or shows it); **[I]** strongly
inferred (several independent pieces of evidence agree); **[U]** unknown / not recoverable from the repo.
Where the paper or wiki conflicts with the code, the code is taken as the implementation being reproduced.

---

## 0. Bottom line
* STAPLE is run **per brain × structure, in 2-D**, on one binary mask per segmentation attempt. Every rater's mask
  is placed on **one shared 2-D canvas**, whichever of the 40 stack planes they drew on, so **the consensus has no
  z position**. [E] `userRater.create4dNii` (userRater.m:65-125)
* STAPLE-canvas pixel (i, j) = the rater-stack plane at **4 µm/px, top-left aligned, x = column, y = row (down)**.
  Only axis permutation: a MATLAB transpose to [col,row]. No crop offset, rotation or flip happens before STAPLE.
  [E] for the operations, [I] for "top-left aligned to the stack" (see §2).
* **Back onto the rater stack plane (1 µm):** trivial: a ×4 upscale from the top-left. [I]
* **Back onto the downscaled validation brains in `downscaledBrains/` (12.5 µm):** fully specified by the repo
  (per-stack rigid matrix + z-window in `data/brainShifts.yml`, the stack height, plus a vertical flip). The code
  does this mapping in the forward direction to score aMAP (`pavelAtlas.makeAtlasSegStack`), and I checked it
  empirically on real ROIs (§6.4). The only free parameter is **which coronal plane**, because the consensus is 2-D.
* **Back onto the raw STP microscopy (1×1×5 µm, pre-alignment): not possible from the repo.** The rigid
  "align to Allen average" matrices, the full-res aligned volumes and the 40-plane rater stacks are not included. [U]

---

## 1. Source images

| Item | Finding | Evidence |
|---|---|---|
| Brains segmented | 6 STP brains: MV_Ntsr1_165, CR_Syt6CreRfp_35 (parameter-search/"training"); ER_Glt25d2Cre_9, AB_Gad67_223, MV131017_7, MV_Ntsr1_169 (scoring) | [E] wiki Home.md:7-15; globalData.yml:2,12-36; segmentation filenames |
| Raw acquisition | Coronal STP, voxel 1 µm (x) × 1 µm (y) × 5 µm (z), ×10 objective | [E] paper Methods "Imaging" |
| Alignment | Each brain rigidly aligned to the Allen average brain. Matrix estimated on z-smoothed (Gaussian σ = 5 voxels) data downscaled to 12.5 µm (reg_aladin), then applied to full-res data in MATLAB | [E] paper Methods "Imaging", Discussion ¶"In our study…". Matrices/code **not in repo** [U] |
| Rater stacks | Per brain, 8 stacks (`MV, sub, sg, rspV1V2M, vpm, adt, s1bf, cingul`); RSP/V1/V2M share one stack. Each has 40 coronal images at a **15 µm step**, 8-bit, 1 µm px | [E] paper Methods "Segmentation task"; meta.txt `#File …/8BitStacks/<brain>/single8Bit/<stack>.tif001.tif`, "Segmentation 1 of 40"; globalData.yml:10 `stackSize: 40`; brainShifts.yml section names |
| Rater-stack size | W × H per stack = `width`/`height` in brainShifts.yml (13518–14052 × 9323–10133 px). **Stacks not in repo** | [I]: `segAnalyser` builds the stack-frame reference grid from exactly these values (segAnalyser.m:181-184, 319-321; pavelAtlas.m:87) |
| `downscaledBrains/*.nii` | NIfTI-1, **12.5 µm isotropic**, uint16 (ER: uint8), `qform_code=0`, `sform_code=1`, `srow = diag(0.0125)` with offset **+0.0125** on each axis, so world = (1-based index) × 0.0125 mm. Dims (x,y,z): CR 1082×959×736, ER 860×1280×604, AB 1078×1218×729, MV131017 1074×1220×728, 165 1080×1220×732, 169 1074×1220×730 | [E] header parse. dim3 equals `height` in globalData.yml:16-36 [E] |
| Axis meaning of the `.nii` | dim1 = image column (TissueVision x), dim2 = coronal-plane index (**increases anteriorly**: MV zRange ≈ 127–291, cingul ≈ 730–922), dim3 = image row **flipped** (increases dorsally) | [E] atlasDistanceScorer.m:7-9,46-47 ("TissueVision-XYZ…y axis inverted compared to .nii", `nii(x+1, z, brainHeight−y)`); tiff2nii.m:40-55 (coronal permute [2 3 1], then flip dim 3). AP sign [I] from zRanges |
| Are these the validation images? | They are the **registration targets** (aMAP atlas → brain) and the space the automated segmentation is sampled from before being resampled into the stack frame. Raters **did not** draw on them: raters drew on the full-res 1 µm / 15 µm stacks | [E] wiki Home.md:5-17; registrations.yml atlasNames = `<brain>…_MAP.nii`; NiftyRegMultiRun.py:45,53 (brain = `-ref`); paper p.2-3 "downsampled versions of our STP data sets (12.5 µm isotropic)" |
| Relation of downscaled ↔ raw | Z-smoothed, downscaled 12.5 µm version of the aligned data. The filename suffix `AffRegNoTrans`/`AffNoTrans` suggests "affine-registered, translation dropped"; exact procedure [U] | paper Methods; filenames |

Hemisphere: paper Fig. 1a says structures are in the **left** hemisphere. In code/data the segmented side is
**high x** (large image columns / nii dim1 > midline): `removeUnsegmentedHemisphere` zeroes `x ≤ midline`
(segAnalyser.m:68-86), and all ROIs lie at x ≈ 5400–12100 px. [E] for the code side; anatomical L/R labelling of
image columns [I]. Exception: MV_Ntsr1_169 SG, where most raters used the other hemisphere (median x ≈ 4300) and the
code switches hemisphere (`hemisphereNeedsFlipping`, segAnalyser.m:360-366).

---

## 2. Manual segmentations

**Production** [E] paper Methods: 22 raters in 2 groups of 11. Each rater segmented 10 structures on 3 brains,
plus a blind repeat of one brain. A custom Fiji plugin with a Wacom Cintiq was used. For each structure the rater
picked the one plane (of 40) best matching an Allen plate and outlined the structure. Outlines were "converted to
filled binary images".

**What is stored** (`data/userSegmentations/userSegmentations/User_XX/`, 880 sets) [E]:
* `Seg<idx><brain>_<stack>.tif001-<REGION>_Slice<s>..tif`: filled mask, **12109 × 9997 px (W×H) for all 880
  files**, 8-bit, PackBits, values {0, 255}, no physical calibration (72 dpi default).
* `.roi`: ImageJ ROI (v226; freehand, polygon or composite). `_meta.txt`: source stack path, "Segmentation k of 40",
  confidence, timestamps, `#duration`. Optional `_comments.txt`.
* Counts: 11 per brain×structure; **22 for MV_Ntsr1_165 and MV_Ntsr1_169** (the repeated brains, 11 raters × 2
  attempts). Group A = {AB, CR, 169}, group B = {ER, MV131017, 165}. Slice numbers 1–40.

**Coordinate space** [E]/[I]:
* The TIFF's nonzero bounding box equals the ROI's bounding box pixel-for-pixel (checked on 3 files, ROI
  right/bottom exclusive), so ROI coordinates and the TIFF share one frame. [E]
* That frame is the rater-stack plane at 1 µm/px, origin top-left, rows increasing ventrally. [I] Evidence:
  (a) All TIFFs are 12109×9997, smaller than every stack (W ≥ 13518). Across all 880 ROIs, max right = 12108 and
  max bottom = 9996, i.e. exactly the canvas edge. So each mask was cropped to its content at the bottom/right and
  then zero-padded top-left-aligned to the global maximum. `segAnalyser.cropBottomRight` (segAnalyser.m:390-395) and
  `userRater.resizeSegImgsToMax` (userRater.m:36-46, "done already", createAgreementSegs.m:1-3) are exactly these two
  operations. Both keep the top-left origin.
  (b) The empirical overlay in §6.4 puts these pixel coordinates on the correct anatomy.
* Plane: the `SliceNN` token only. The TIFF itself is 2-D. [E]

**Association** [E] `getSegmentationSessionData.m`:
* Walks `manSegBaseDirs/*/` and takes files ending `.tif` (lines 10-34). The table is cached in a `persistent`
  (lines 3-7, 40).
* `idx` = digits after `Seg` (43-61), the per-rater presentation index (distinguishes attempt 1 and 2).
* `brain` = text from the first digit→non-digit transition to `.tif`, minus the last `_token` (63-72). For example
  `Seg10MV_Ntsr1_169_sg.tif…` gives `MV_Ntsr1_169`.
* `region` = text after `\d\d\d-` up to the next `_` (74-82), e.g. `SG`. `sliceNum` = digits after `Slice` (84-88).
* `brainRegionComb = brain ∥ region`, e.g. `MV_Ntsr1_169SG` (35-37). This is the STAPLE grouping key.
  `user` = folder name.

---

## 3. Preprocessing: every operation between the TIFF and STAPLE
All in `userRater` (called from createAgreementSegs.m:27,31). `scale = 0.25` (globalData.yml:11 via sharedParams.m:64-66).

| # | Operation | Code | Notes |
|---|---|---|---|
| P0 | Global canvas size `maxX = max(Width) = 12109`, `maxY = max(Height) = 9997` over **all 880** TIFFs (all brains) | userRater.m:27-31 | [E] Canvas is the same for every brain/structure |
| P1 | (already applied to the data) zero-pad each TIFF to maxY×maxX, **top-left aligned** | userRater.m:36-46 | [E]; commented out in createAgreementSegs.m:3 |
| P2 | Optional z-window: default `range=1e7` → all slices `min..max`; `'range',7` → 7-slice window with most segmentations; `'quantiles',[.25 .75]` → `floor(q25)..floor(q75)` | userRater.m:51-90, 96 | [E]. Default run = all slices |
| P3 | `imread(tif)'`: **transpose**, so the array becomes `[x = column, y = row]`, 12109×9997 | userRater.m:103-104 | [E]. Only axis permutation in the chain |
| P4 | binarise `uint8(img>0)` (255 → 1) | userRater.m:105 | [E] |
| P5 | `imresize(img, 0.25, 'nearest')` → **3028 × 2500** (ceil). No antialiasing for nearest. The MATLAB mapping picks input index `4u−1` (1-based), i.e. 0-based `[2::4, 2::4]`; the last index is mirror-clamped | userRater.m:106 | Size [E]; exact sampling phase [I] (MATLAB-internal) |
| P6 | `make_nii(img)`: uint8, **pixdim 1 mm (placeholder)**, originator 0. `save_nii` → `sform_code=1`, `srow=diag(1,1,1)` with offset +1 | userRater.m:107-109; make_nii.m:52-53, 236-253; save_nii_hdr.m:15-28 | [E]. The STAPLE files carry **no real physical geometry** |
| P7 | `seg_maths first.nii -merge N−1 4 others… out.nii.gz` → 4-D **[3028, 2500, 1, N]** | userRater.m:112-116, 123 | [E]. Order = segTable row order (directory listing); the rater list is written to `*_4DSegMathsMergeCommand.txt` (117-122) |
| P8 | Temporary per-seg `.nii` files deleted | userRater.m:124 | [E] |

There is **no** cropping to a region of interest, rotation, flip, slice-order reversal, offset or resampling
beyond P3–P5 before STAPLE. [E]

---

## 4. STAPLE

**Call chain** [E]: createAgreementSegs.m:27 `userRater(manSegBaseDirs, getManSegNiiDir(''), getStapleDir(''), 0.25)`
→ `getSegmentationSessionData`, `grp2idx(brainRegionComb)`, maxX/maxY → :31 `create4dNii('', [])` (§3)
→ :33 `runStaple` → :34 `parseStapleResults` → `getRaterScore`.

**Input**: one 4-D NIfTI per brain×structure, e.g. `MV_Ntsr1_169SG_0.25_4DSegMathsMerged.nii.gz`, shape
[x=3028, y=2500, z=1, rater=N], uint8 {0,1}. **N = 11** (AB, CR, ER, MV131017) or **N = 22** (165, 169: both attempts
of each rater, per userRater.m:49 "use both segmentations from one rater as individual data point"). All slices are
pooled onto one plane. [E]

**Command** (userRater.m:156-162) [E]:
`seg_LabFusion -in <4D> -STAPLE -unc -out <brainRegionComb>_0.25_NiftySegSTAPLE.nii -v 2`, plus the SBA analogue
(:159, 168). From the bundled binary's help strings [E]: `-out` = "integer segmented image" (hard labels; `-outProb`
was **not** used); `-unc` = "only consider non-consensus voxels to calculate statistics". Convergence default ε = 1e-5.
NiftySeg source/commit is not in the repo; the build dir points at `niftyseg-git`, Eigen ffa86ffb5570. [U] for the
exact algorithm version.

**Output → final agreement segmentation**: the fused hard label map [3028, 2500, (1)] = the final consensus. It is
the binary argmax of the STAPLE posterior (for 2 labels, effectively P(fg) > 0.5). [I] from NiftySeg semantics.
No thresholding, smoothing or morphology is added in MATLAB. It is loaded with `load_nii`
(AverageSegmentationLoader.m:19-29). Diagonal-positive sform means `xform_nii` does not reorient, so array indices
equal the P3 indices. Scoring binarises with `>0` (diceScore.m:10-11). [E]/[I]

**Per-rater parameters**: stdout is logged to `*_NiftySegSTAPLE_log.txt` with the rater list appended.
`getRaterScore` takes tokens after the last `*******************************` line; for each `[k]` it reads `q` at
the token after `[k]=` and `p` 3 tokens later (getRaterScore.m:10-37). These go into `segTable.p/.q`
(userRater.m:171-202). [E] Parsing depends on NiftySeg's log format [U].

**Downstream use (scoring, not part of STAPLE)** [E] segAnalyser.m:129-171: each manual mask is re-read,
`newImg'>0`, then **`imresize(…, 0.25)` with the default bicubic + antialias on a logical** (line 156). That is a
different downsampling from the STAPLE input (nearest). The mask is `padToLarger` (top-left zero-pad) and then
Dice/Hausdorff are computed. A reproduction of the Dice scores must copy both kernels.

---

## 5. Paper/wiki vs code discrepancies
1. **Repeat attempts in STAPLE**: paper Methods says "for inter-rater analyses, only the first segmentation of the
   repeated brain was used". The code feeds **both** attempts (N = 22) into STAPLE for 165/169 and scores all rows.
   Paper Fig. 3e ("first and second attempt versus STAPLE") partly matches the code. [E] conflict
2. **Excluded "wrong structure/hemisphere" cases (5)**: no exclusion list exists in code or data. The stored TIFFs
   still contain large aberrant shapes: CR VPM User_10 Seg11, ER SUB User_02 Seg22, ER V2M User_13 Seg24, 169 MV
   User_19 Seg17, 169 SUB User_05 Seg32 and others found by a centroid-outlier scan. So the "manual cleaning" is
   either already baked in or was done outside the repo. [U]
3. **DG-sg**: the paper excludes it from the analysis. The code still builds its STAPLE; the exclusion happens only
   in summaries (`meanDiceStapleNoSg`, segAnalyser.m:44-46). [E]
4. Paper says the consensus was computed with "seg_maths"; the code uses `seg_LabFusion` (seg_maths only merges). Minor. [E]
5. Wiki refers to a "registrationRunners" folder; the actual folder is `nonMatlab/pythonRegRunners`. Minor. [E]
6. `scoreParameterSearches.m`/`parameterSearchResult` are broken (`error('…defunkt…')`, parameterSearchResult.m:32). [E]

---

## 6. Spatial mapping: STAPLE → source images

### 6.1 Onto the rater-stack plane (1 µm, aligned full-res frame) [I]
STAPLE pixel (i0, j0) (0-based, array = [x,y]) ↔ stack columns `4·i0 … 4·i0+3`, rows `4·j0 … 4·j0+3`. There is no
offset because of the top-left alignment (§2); the value was sampled at (4·i0+2, 4·j0+2) (P5). The canvas covers stack
cols 0–12111 and rows 0–9999. Stacks are ≥13518 wide. H ranges 9323–10133, so for 165/169 the canvas extends
past the stack bottom (zeros), and for CR it clips the bottom ~100 rows (no segmentations there).
**Plane**: undefined. The consensus pools slices 1–40. The authors' own display puts it at
`round(median(sliceNum))` (segAnalyser.m:281-285). The stacks themselves are not in the repo [U].

### 6.2 Onto the downscaled brain `.nii` (12.5 µm): derived from the code's forward path [E]/[I]
Forward path used to score aMAP (pavelAtlas.m:52-104, called from segAnalyser.m:178-184):
1. Slab: `smallAtlas = atlas(:, zRange(1):zRange(2), :)`, a coronal slab along nii dim2 (line 78). Saved with the
   brain pixdim 0.0125 (line 80) → world = (slab 1-based idx)·0.0125.
2. Reference grid = stack frame: dims `[Ws, n, Hs] = [round(W·0.25), numel(zRange), round(H·0.25)]`, pixdim
   `[0.004, 0.015, 0.004]` mm (segAnalyser.m:181-184 → pavelAtlas.m:87-89), so world = (1-based idx)·pixdim
   (save_nii_hdr.m:15-28). The 2×2×2 data trick keeps the header dims (save_nii.m write_nii uses `hdr` as-is).
3. `reg_resample -trans shift.txt -inter 0` (lines 85, 95). The NiftyReg affine convention maps
   **reference world → floating world**: `f = M · r`. [I] NiftyReg convention, not in repo.
   `bash ~/poltergeist.sh` wrapper is not in the repo [U].
4. `permute([1 3 2])` → [x, z, plane]; `flip(…, 2)` → rows top-down (lines 99-100; comment "bring back to
   TissueVision coordinate space").
5. `padToLarger` top-left against the STAPLE canvas; max Dice over planes (segAnalyser.m:193-204).

**Inverse (what you need):** for STAPLE pixel (i1, j1) (1-based; i = x/column, j = row) of brain b, stack k, at
stack-plane index J (1-based):
```
Hs  = round(H_k · 0.25)                       # H_k, W_k, M_k, zRange_k from brainShifts.yml[b][k]
r   = [0.004·i1, 0.015·J, 0.004·(Hs + 1 − j1)]  # stack/ref world (mm); vertical flip via Hs
f   = M_k[0:3,0:3] · r + M_k[0:3,3]           # floating (slab) world (mm)
nii = [f_x/0.0125,  f_y/0.0125 + zRange_k(1) − 1,  f_z/0.0125]   # 1-based voxel in downscaledBrains/<b>.nii
```
* `J ↔ rater slice s`: **J = s** [I]. The ref y spacing (15 µm) equals the stack step. M is almost the identity
  (|R−I| ≤ 1.7e-3, det = 1.000000, t_y ≈ 0). 40 × 1.2 = 48 slab planes ≈ numel(zRange) = 46–52. Planes J > 40 fall
  outside the slab (zero padding). One plane step = 1.2 nii voxels.
* The consensus is 2-D. To place it in the volume, choose J (e.g. median slice, or each rater's own s) [U: not defined
  by the authors]. The atlas-vs-STAPLE scoring sidesteps this by taking the **max Dice over J**.
* Rotations in M are ≤ 0.1°. Translations vary per stack (e.g. CR t_x −0.02 → +1.16 mm, t_z −1.18 → −2.06 mm), which
  suggests per-stack crop origins plus a small rigid residual. How M was estimated is [U].
* Sub-pixel caveat: the ref grid's world convention (4 µm pixel i1 at 0.004·i1) and imresize's pixel-centre convention
  (centre at input 4·i0+1.5) differ by ~1.5 µm (0.12 voxel at 12.5 µm). Negligible, but fix one convention in Python.
* Since M is rigid and the permute/flip/pad are bijective, the mapping is **invertible**. The repo has everything
  needed for this space: brainShifts.yml, globalData.yml, headers, and code conventions.

### 6.3 Onto the raw STP microscopy — not recoverable [U]
Needs (a) per-brain rigid alignment matrices to the Allen average (paper Methods; not in repo); (b) how the
8-bit 40-plane stacks were cut from the aligned full-res volume (start plane, crop origin per stack, 8-bit scaling;
not in repo); (c) the stacks or aligned volumes themselves. With the downscaled brains alone you can go STAPLE →
12.5 µm aligned space (§6.2), **not** back to pre-alignment raw coordinates.

### 6.4 Empirical check performed (AB_Gad67_223, stack rspV1V2M, nii plane 438)
I pushed two real ROI polygons (User_01 RSP slice 23, User_22 V1 slice 18) through the §6.2 formula and rendered
them over the `.nii` plane.
* **With the code's flip**: RSP runs along the midline fissure from the dorsal pial surface (z ≈ 632–644)
  ventrally. V1 sits dorsolaterally on the cortical surface. Anatomically correct.
* **Without the flip**: RSP lands on the ventral brainstem surface (z ≈ 56–212). Wrong.

This confirms the vertical flip, no left/right flip, the top-left anchoring, and the scale factors. [E] (empirical)
Suggested follow-up: repeat the check for every brain×stack; I tested one brain/stack only.

---

## 7. Transformation table

| Stage | Dims | Pixel / voxel size | Axis order / orientation | Transform into this stage | Code / source |
|---|---|---|---|---|---|
| **Raw STP** | unknown | 1×1×5 µm | coronal sections | acquisition | paper Methods [E]; data **not in repo** [U] |
| **Aligned full-res volume** | unknown | 1×1×5 µm (assumed) | coronal, aligned to Allen average | rigid (reg_aladin at 12.5 µm, applied in MATLAB) | paper Methods [E]; matrices **not in repo** [U] |
| **Downscaled validation brain** (`downscaledBrains/*.nii`) | e.g. AB 1078×1218×729 (x,y,z) | 12.5 µm iso; sform diag(0.0125)+0.0125 | dim1 = image col; dim2 = coronal plane (↑ anterior); dim3 = image row flipped (↑ dorsal) | z-Gaussian σ = 5 vox → downscale (details [U]) | headers [E]; atlasDistanceScorer.m:7-9,47 [E]; tiff2nii.m:37-55 |
| **Rater stack** (per brain×stack) | W×H×40 (e.g. AB rsp 13747×9587×40) | 1 µm × 1 µm × 15 µm step | ImageJ [row down (ventral), col right (segmented hemisphere)], plane 1 = posterior-most [I] | cut from aligned full-res; stack ↔ nii: `f = M·r`, slab `zRange`, vertical flip w.r.t. H | paper [E]; brainShifts.yml [E]; pavelAtlas.m:76-100 [E]; stacks **not in repo** |
| **Manual seg TIFF** | 12109 (W) × 9997 (H), 2-D; plane = `SliceNN` | 1 µm (uncalibrated) | same as stack plane, top-left aligned | ROI → filled mask {0,255} → crop bottom/right → zero-pad to global max (top-left) | data [E]; userRater.m:36-46, segAnalyser.m:390-395 [I] |
| **STAPLE input (per seg)** | 3028 × 2500 (MATLAB array [x, y]) | 4 µm (header says 1 mm) | transposed: dim1 = column, dim2 = row | `'` transpose; `>0`; `imresize 0.25 nearest` (samples 4i+2, 4j+2) | userRater.m:103-106 [E] |
| **STAPLE 4-D input** | 3028 × 2500 × 1 × N (N = 11 or 22) | same | dim4 = rater/attempt (segTable order) | `make_nii` / `save_nii` / `seg_maths -merge` along dim 4 | userRater.m:107-123 [E] |
| **STAPLE output** | 3028 × 2500 (×1) | same | same as input | `seg_LabFusion -STAPLE -unc -out` → hard label {0,1} (no `-outProb`) | userRater.m:156-162 [E]; binary help [E] |
| **Back to stack plane** | 12112 × 10000 (canvas) within W×H | 1 µm | stack frame | ×4 upsample, top-left; plane undefined (pooled); authors display at median slice | §6.1 [I]; segAnalyser.m:281-285 [E] |
| **Back to downscaled `.nii`** | point/voxel map | 12.5 µm | nii frame | `r=[.004 i, .015 J, .004(Hs+1−j)]`, `f=M r`, `nii=f/.0125 + [0, zRange(1)−1, 0]`; J = slice (choose) | pavelAtlas.m:52-104 inverted [E]/[I]; §6.4 check [E] |
| **Back to raw STP** | — | — | — | needs alignment matrices + stack-cutting parameters | **not possible from repo** [U] |

---

## 7b. Validation status (after the aborted cross-brain attempt)
* The empirical geometry evidence is still only the single overlay in §6.4 (AB_Gad67_223 / rspV1V2M). The
  cross-brain overlay, numerical round-trip and explicit alternative-hypothesis tests were **not run**.
  The user chose to proceed on the current reconstruction.
* Environment for the port: the base Python has no numpy. `<HOME>\.conda\envs\analysis_env` has
  numpy 1.26.4 and scipy 1.15.3. The V: share reads at about 200 MB/s. Run the conda interpreter via PowerShell,
  not Git-Bash heredocs.
* The round-trip test needs no image data, so it becomes a unit test of the mapping module (see §9).

## 8. Open questions and things to check before the Python port
1. Get the authors' reference outputs if available (GIN `brainglobe/amap_validation`): STAPLE/SBA `.nii`, 4-D merges,
   logs, and ideally the 8-bit stacks. The repo holds no STAPLE outputs to regress against.
2. Decide on the repeat-attempt policy (code: both attempts; paper: first only) and the 5 excluded segmentations
   (unidentified).
3. Pin down NiftySeg's STAPLE semantics (`-unc`, prior, convergence, log format) against the NiftySeg source.
4. Confirm MATLAB `imresize` sampling/edge rules for both 'nearest' (STAPLE input) and bicubic-logical (scoring).
5. (Deferred by user) Extend the §6.4 overlay check to all 6 brains × 8 stacks.

---
