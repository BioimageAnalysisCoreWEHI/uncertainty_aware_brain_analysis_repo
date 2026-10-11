#!/bin/bash
#SBATCH --job-name=downsample_25um
#SBATCH --output=${IMAGING_ROOT}/Ishrat/hpc_processing/logs/downsample_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"

module load miniconda3
conda activate ${CONDA_ENV}

# === CONFIG ===
MANIFEST="${IMAGING_ROOT}/Ishrat/JoM/post_review_analysis/config/samples_amap_validation_raw.csv"
SCRIPT="/${IMAGING_ROOT}/Ishrat/JoM/post_review_analysis/scripts/setting_up/downsample_and_reslice.py"

# Count rows excluding header
NUM_SAMPLES=$(($(wc -l < "$MANIFEST") - 1))

if [ "$NUM_SAMPLES" -le 0 ]; then
    echo "No samples found in manifest: $MANIFEST"
    exit 1
fi

echo "Found $NUM_SAMPLES samples in manifest"

# === RE-SUBMIT AS ARRAY IF NOT ALREADY ARRAY JOB ===
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    sbatch --array=0-$((NUM_SAMPLES-1)) "$0"
    exit 0
fi

# === RUN PYTHON SCRIPT ===
python "$SCRIPT"
