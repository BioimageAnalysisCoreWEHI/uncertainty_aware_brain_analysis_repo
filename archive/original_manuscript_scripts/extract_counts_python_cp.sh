#!/bin/bash
#SBATCH --job-name=cellpose_array
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --output=${IMAGING_ROOT}/Ishrat/hpc_processing/logs/extract_counts_py_%A_%a.out
#SBATCH --cpus-per-task=10
#SBATCH --mem=400G
#SBATCH --array=0-1
#SBATCH --mail-type=END
##SBATCH --mail-user=you@example.org   # optional
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A30:1
# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"

PARENT_DIR="${IMAGING_ROOT}/Ishrat/JoM/extracted_regions_raw_consensus/for_cellpose/rerun"
FOLDERS=( $(ls "$PARENT_DIR") )
FOLDER_NAME=${FOLDERS[$SLURM_ARRAY_TASK_ID]}

echo "Processing $FOLDER_NAME"
module load miniconda3
conda activate ${CONDA_ENV}

python extract_counts_python_cp_parallel.py --parent_dir "$PARENT_DIR" --folder "$FOLDER_NAME"
