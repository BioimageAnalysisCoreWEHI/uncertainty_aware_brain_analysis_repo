#!/bin/bash
#SBATCH --job-name=consensus
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --output=${IMAGING_ROOT}/Ishrat/hpc_processing/logs/consensus_py_%A_%a.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --array=0-4
#SBATCH --mail-type=END
##SBATCH --mail-user=you@example.org   # optional
# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"

module load miniconda3
conda activate ${CONDA_ENV}

python consensus_region_analysis_parallel.py
