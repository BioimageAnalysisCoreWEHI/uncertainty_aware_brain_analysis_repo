#!/bin/bash
#SBATCH --job-name=extract_mask
#SBATCH --output=${IMAGING_ROOT}/Ishrat/hpc_processing/logs/reg_extraction_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=132G
#SBATCH --time=01:00:00
#SBATCH --array=0-269 
# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"

module load miniconda3
conda activate ${CONDA_ENV}

SAMPLE=$1

PARENT_DIR=${SCRATCH}/jom_v2_raw

DATASETS=($(ls $PARENT_DIR))

DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

python region_extraction_same_image_parallel.py $SAMPLE $DATASET