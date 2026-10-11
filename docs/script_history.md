# How the scripts evolved

This repository consolidates several generations of code. Dates are file modification dates.

| Generation | Original location | Dates | What it did / added | In this repository |
|---|---|---|---|---|
| Original manuscript | `JoM/analysis_scripts` (about 115 scripts) | Aug 2025 – Jun 2026 | the published sweep (`submit_brainreg_*_extensive*.sh`), tissue IoU and Jacobian QC (`qc_tissue_overlap.py`, `qc_jacobian_metric.py`), midbrain SSIM (`ssim_all.py`), SNpc mask extraction, all-registration consensus maps, union full-resolution crops, Cellpose counting, results tables, consensus counts by threshold (`get_consensus_counts.py`), cell probabilities and figures | `archive/original_manuscript_scripts/` (the scripts that wrote the folders used for the published numbers; paths made configurable, logic unchanged) |
| Config-driven sweep | `JoM/analysis_scripts/final_paper_scripts` | May 2026 | `config.yaml` and a job table for the brainreg sweep and QC metrics | `archive/final_paper_scripts_2026-05/` |
| Consensus and cells refactor | `JoM/analysis_scripts/testing_scripts/final_paper_scripts` | May – Jul 2026 | consensus maps, Cellpose jobs, counting, cell-region probabilities | `pipeline/cells/` |
| Generic whole-brain pipeline | `analysis_of_whole_brain_projects/scripts/final_pipeline` | Jul – Aug 2026 | crop spaces, registration sets (`pipeline_common.discover_registration_sets`), parameter masks, final result tables for other projects | not included; its shared helpers continue in `pipeline/pipeline_common.py` |
| Post-review pipeline | `JoM/post_review_analysis/scripts` | Aug – Sep 2026 | SLURM arrays for every step, multi-region local SSIM with region trees, whole-brain region consensus for any atlas and region (`build_region_consensus_wholebrain.py`) | `pipeline/` (canonical) |
| Dataset forks | `post_review_analysis/original_dataset/scripts`, `validation_datasets/lightsheet_data/scripts` | Sep – Oct 2026 | the same scripts, with dataset paths and configs; QC-threshold sensitivity (lightsheet) | one copy in `pipeline/`; configs in `config/datasets/`; sensitivity in `pipeline/qc_threshold_sensitivity/` |
| Validation analyses | `post_review_analysis/validation_analysis`, `lightsheet_data/scripts/validation`, `aMAP_validation/python` | Sep – Oct 2026 | aMAP Python port, expert-reference validation, QC-rule sensitivity, independent-LSFM comparison, revision figures | `validation/` |

## Duplicated modules
These files were byte-identical in the post-review, original-dataset and independent-LSFM copies, and are kept once:
- `region_utils.py`
- `pipeline_common.py`
- `run_qc_single.py`
- `03_compute_qc_metrics.py`
- `build_region_consensus_wholebrain.py`
- `05_build_region_consensus_jobs.py`

The other copies (`01`, `02`, `03` and `05` submit and array scripts) differed only in dataset paths, sample tables and job-table names. These now come from `config/` and the environment.

Two files differ in more than paths. The post-review copy is the one kept:
- `04b_plot_qc_calibration.py`
- `automated_preprocessing.py`

## Not included
- Scripts that write manuscript or response documents (`validation_analysis/scripts/25–29`).
- The internal collector of revision outputs (`21`).
- Exploratory and testing scripts in `analysis_scripts`.
- Atlas volumes (downloaded by BrainGlobe).
- All results.
