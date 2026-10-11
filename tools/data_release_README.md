# Uncertainty-aware atlas registration of whole-brain LSFM: registration-level data (5 mouse brains)

This deposit contains the data needed to reproduce the SNpc analyses of the manuscript from the registration step onwards, for all five brains of the original LSFM dataset:
- the inputs to atlas registration;
- the outcome of every registration in the parameter sweep;
- the full-resolution midbrain TH images and cell segmentations used for counting;
- the consensus maps;
- the derived tables behind the figures.

Whole-brain full-resolution images (about 0.5 TB) are not included and are available from the corresponding author on request.

- **Code:** <GitHub URL / Zenodo DOI of the code release> (the `uncertainty_aware_analysis` repository).
- **Licence:** CC BY 4.0 (see LICENSE).
- **Manuscript:** <citation>.

## Samples
`samples.csv` gives:
- the sample labels (Sample1–5 = "Sample 1"–"Sample 5" in the manuscript);
- the voxel sizes;
- animal metadata.

| Item | Value |
|---|---|
| Full-resolution imaging (level 1 of the acquisition pyramid) | about 2.15–2.20 µm in-plane, 6.96–7.03 µm between planes |
| Registration | 20 µm isotropic, Perens LSFM mouse atlas (BrainGlobe `perens_lsfm_mouse_20um` v1.2), brainreg orientation `sal` |

## Layout
```
SampleN/
  inputs/
    AF_20um_raw.tif              autofluorescence at 20 um: registration input, raw arm
    AF_20um_preprocessed.tif     autofluorescence after intensity normalisation / background correction: preprocessed arm
    TH_20um.tif                  TH channel at 20 um (registered as an additional channel)
    tissue_mask_20um_coronal.tif tissue mask used for the tissue-atlas IoU (global QC)
  fullres_midbrain/<raw|preprocessed>/
    TH_crop_fullres.tif          full-resolution TH crop around the midbrain (union of all SNpc masks of that arm)
    cellpose_labels.tif          Cellpose TH+ cell labels in the same space (model: cellpose_model/)
    inclusion_frequency_fullres.tif         probability map in this crop: the fraction of the registrations retained by
                                            global QC + SSIM that include each voxel (float32)
    consensus0p5_mask_fullres.tif           the manuscript consensus SNpc (inclusion >= 0.5 over the registrations
                                            retained by global QC + SSIM) mapped into this crop: the region in which
                                            cells are counted
    cellpose_labels_in_consensus0p5.tif     Cellpose labels inside that region (a cell counts if any voxel remains)
    crop_metadata.json           crop geometry and how 20 um masks map into this space
  registrations/<raw|preprocessed>/
    registrations.csv            one row per registration (48): bending energy weight, grid spacing, tissue-atlas
                                 IoU, deformation (grad_p95), midbrain SSIM, SNpc TH+ count, global-QC pass,
                                 retained after SSIM ranking
    snpc_mask_20um_coronal/      SNpc mask of each registration (20 um, coronal)
  consensus/<raw|preprocessed>/
    qc_ssim_retained/            inclusion frequency (prob_map) and consensus masks at thresholds 0.01-1.0 from the
                                 registrations retained by global QC + SSIM ranking (the manuscript's consensus)
    all_registrations_in_sweep_folder/
                                 pipeline maps averaged over every SNpc mask in the sweep folder (all registrations),
                                 coronal and axial; used to define the full-resolution crop
cellpose_model/                  custom Cellpose 2.x model for TH+ cells
expert_annotations/Sample3/      expert SNpc annotations (20 um) of Sample 3 by two annotators
derived/                         tables: counts by threshold, per-registration QC and counts, QC-rule sensitivity,
                                 aMAP and independent-LSFM validation results, figure source data, vote cache;
                                 qc_sensitivity/fullres_qc_rule_masks/: the full-resolution consensus masks and masked
                                 cells for all eight global-QC rules of the sensitivity analysis
MANIFEST.csv                     every file with size and SHA-256
```

All images are single 3D TIFFs with zlib compression; a JSON description in each file records sample, arm and voxel size.

The two arms are:
- **raw**: registration of the autofluorescence image as acquired, downsampled to 20 µm.
- **preprocessed**: registration after intensity normalisation and background correction.

## Reproducing the counts
A cell is counted in a region if any of its voxels lies in the region mask. The consensus mask is `inclusion frequency >= threshold`; the manuscript uses 0.5.

`derived/vote_cache/` stores, for every voxel of every segmented cell, which registrations include it. With these votes, counts for any set of registrations and any threshold can be reproduced exactly with the code repository (`validation/analysis/scripts/07_lsfm_analysis.py`, class `Counter`).

## Notes
- **Data selection and processing** are described in the manuscript Methods, and the code that produced every file is in the code repository.
- **Released data come from the original parameter sweep.** The registration inputs and all per-registration masks, QC metrics and counts are from the sweep reported in the manuscript.
