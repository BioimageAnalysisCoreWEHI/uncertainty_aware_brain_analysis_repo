# Running the pipeline on a SLURM cluster

## Environment
Set these before submitting jobs:
```
export IMAGING_ROOT=/path/to/project/storage     # where inputs and results live
export SCRATCH=/path/to/your/scratch             # where brainreg writes the sweep (large: ~7-9 GB per registration)
export CONDA_ENV=brainreg_hpc                     # or the path of the environment
export PROJECT_DIR=/path/to/your/project          # folder holding config/config.yaml and samples*.csv
```
The submit scripts use `module load miniconda3` and `conda activate ${CONDA_ENV}`; adapt these to your cluster. The `#SBATCH` resources in each `run_*_array.sh` (time, memory, CPUs) are the values we used.

## Order of commands
| # | Step | Command |
|---|---|---|
| 1 | Samples table | `pipeline/00_setup/build_samples_csv.py` (raw input), `build_samples_csv_preprocessed.py` (preprocessed input) |
| 2 | Job table | `python pipeline/01_make_brainreg_jobs.py`: one row per sample × input × parameter combination |
| 3 | Registration sweep | `sbatch pipeline/02_submit_brainreg_sweep.sh` (and `_preprocessed.sh`): submits `run_brainreg_array.sh` as an array over the job table |
| 4 | QC metrics | `sbatch pipeline/03_submit_qc_jobs.sh`: one task per registration (`run_qc_single.py`) |
| 5 | Combine and select | `python pipeline/04_combine_qc_csvs.py` (global QC flags, SSIM ranking, final selection) and optionally `04b_plot_qc_calibration.py` |
| 6 | Region consensus | `python pipeline/05_build_region_consensus_jobs.py`, then `sbatch pipeline/05_submit_region_consensus_sweep.sh` |
| 7 | Cells | `python pipeline/cells/06_make_cellpose_jobs.py`, `sbatch pipeline/cells/06_submit_cellpose_jobs.sh`, then `07_count_cells_in_masks.py` and `08_compute_cell_region_probabilities.py` |

Array sizes over 1000 tasks are submitted in chunks by the submit scripts. Every array task writes its own output, and finished tasks are skipped on resubmission.

## What runs where
| Where | Steps |
|---|---|
| HPC (CPU arrays) | registration (`run_brainreg_array.sh`), QC metrics (`run_qc_array.sh`), region consensus (`run_region_consensus_array.sh`), QC-threshold sensitivity (`run_threshold_sensitivity_array.sh`, `validation/analysis/scripts/hpc/run_qcsens_array.sh`) |
| HPC (GPU) or local | Cellpose |
| Local workstation | set-up, combining QC tables, counting, all validation analyses and figures |
