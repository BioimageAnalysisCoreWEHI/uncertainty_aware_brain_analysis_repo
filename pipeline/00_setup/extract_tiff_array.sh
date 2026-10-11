#!/bin/bash
#SBATCH --job-name=extract_ims_array
#SBATCH --time=10:00:00
#SBATCH --array=0-5
#SBATCH --output=${IMAGING_ROOT}/Ishrat/hpc_processing/logs/am_extract_tiff_%A_%a.out
#SBATCH --cpus-per-task=50
#SBATCH --mem=120G
#SBATCH --mail-type=END
##SBATCH --mail-user=you@example.org   # optional
# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"

module load miniconda3
conda activate ${CONDA_ENV}

python ${IMAGING_ROOT}/Ishrat/JoM/post_review_analysis/validation_datasets/lightsheet_data/scripts/setting_up/extract_tiff_array.py

wait