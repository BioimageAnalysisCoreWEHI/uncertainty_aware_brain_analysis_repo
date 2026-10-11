#!/bin/bash
#SBATCH --job-name=cellpose_array
#SBATCH --output=${IMAGING_ROOT}/Ishrat/hpc_processing/logs/cellpose_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=05:00:00
# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"


module load miniconda3
conda activate ${CONDA_ENV}

JOB_TABLE="results/job_tables/cellpose_jobs.csv"
SLURM_SCRIPT="scripts/run_cellpose_array.sh"

N_JOBS=$(($(wc -l < "$JOB_TABLE") - 1))
ARRAY_MAX=$((N_JOBS - 1))

echo "Submitting $N_JOBS Cellpose jobs"
sbatch --array=0-${ARRAY_MAX} "$SLURM_SCRIPT"