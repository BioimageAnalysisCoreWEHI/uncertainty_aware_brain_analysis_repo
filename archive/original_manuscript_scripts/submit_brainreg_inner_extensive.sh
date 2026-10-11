#!/bin/bash
#SBATCH --job-name=brainreg_inner
#SBATCH --output=${IMAGING_ROOT}/Ishrat/hpc_processing/logs/brainreg_inner_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=0-05:00:00
# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"


module load miniconda3
conda activate ${CONDA_ENV}

# Read the combination for this task
IFS=',' read -r bending grid smoothing atlas < <(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" combinations_extensive.csv)

atlas=$(echo "$atlas" | tr -d '\r')  # Remove carriage returns
bending=$(echo "$bending" | xargs)
grid=$(echo "$grid" | xargs)
smoothing=$(echo "$smoothing" | xargs)
atlas=$(echo "$atlas" | xargs)

SAVE_DIR="${SCRATCH}/jom_v2_preprocessed/${SAMPLE_NAME}_freeform_bending_${bending}_grid_${grid}_smoothing_${smoothing}_atlas_${atlas}"


echo "Running brainreg for $SAMPLE_NAME, combination $SLURM_ARRAY_TASK_ID"
mkdir -p "$SAVE_DIR"

brainreg "$AF_DIR" "$SAVE_DIR" -v 20 20 20 --orientation sal --atlas "$atlas" --debug --save-original-orientation \
  -a "$TH_DIR" "$UNPROCESSED_DIR" --n-free-cpus 0 \
  --bending-energy-weight "$bending" --grid-spacing "$grid" --smoothing-sigma-floating "$smoothing" \
  --affine-n-steps 3 --affine-use-n-steps 2 --freeform-n-steps 4 --freeform-use-n-steps 4

echo "✅ Done: $SAMPLE_NAME, combination $SLURM_ARRAY_TASK_ID"