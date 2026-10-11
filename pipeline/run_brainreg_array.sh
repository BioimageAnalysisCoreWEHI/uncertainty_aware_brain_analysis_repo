#!/bin/bash
#SBATCH --job-name=brainreg_sweep
#SBATCH --output=${IMAGING_ROOT}/Ishrat/hpc_processing/logs/brainreg_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=05:00:00
# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"

module load miniconda3
conda activate ${CONDA_ENV}

JOB_TABLE=${JOB_TABLE}

JOB_OFFSET=${JOB_OFFSET:-0}
GLOBAL_TASK_ID=$((JOB_OFFSET + SLURM_ARRAY_TASK_ID))

echo "Running SLURM array task: $SLURM_ARRAY_TASK_ID"
echo "JOB_OFFSET: $JOB_OFFSET"
echo "Global job index: $GLOBAL_TASK_ID"
echo "Using job table: $JOB_TABLE"

if [ ! -f "$JOB_TABLE" ]; then
    echo "ERROR: Job table not found: $JOB_TABLE"
    exit 1
fi

ROW_NUM=$((GLOBAL_TASK_ID + 2))
ROW=$(sed -n "${ROW_NUM}p" "$JOB_TABLE")

IFS=',' read -r \
    sample_id \
    input_type \
    output_group \
    AF_DIR \
    UNPROCESSED_DIR \
    SAVE_DIR \
    atlas \
    vx \
    vy \
    vz \
    orientation \
    bending \
    grid \
    smoothing \
    affine_n_steps \
    affine_use_n_steps \
    freeform_n_steps \
    freeform_use_n_steps \
    <<< "$ROW"

echo "-----------------------------------"
echo "sample_id       = $sample_id"
echo "input_type      = $input_type"
echo "output_group    = $output_group"
echo "AF_DIR          = $AF_DIR"
#echo "TH_DIR          = $TH_DIR"
echo "UNPROCESSED_DIR = $UNPROCESSED_DIR"
echo "SAVE_DIR        = $SAVE_DIR"
echo "atlas           = $atlas"
echo "voxel size      = $vx $vy $vz"
echo "orientation     = $orientation"
echo "bending         = $bending"
echo "grid            = $grid"
echo "smoothing       = $smoothing"
echo "-----------------------------------"

# === SKIP IF THIS JOB ALREADY COMPLETED ===
# Skip if output directory already exists and contains files
if [ -d "$SAVE_DIR" ] && [ "$(ls -A "$SAVE_DIR")" ]; then
    echo "Output directory already exists and is not empty."
    echo "Skipping: $SAVE_DIR"
    exit 0
fi


mkdir -p "$SAVE_DIR"

cat <<EOF > "${SAVE_DIR}/params.txt"
sample_id=$sample_id
input_type=$input_type
output_group=$output_group
AF_DIR=$AF_DIR
UNPROCESSED_DIR=$UNPROCESSED_DIR
SAVE_DIR=$SAVE_DIR
atlas=$atlas
voxel_size=$vx,$vy,$vz
orientation=$orientation
bending_energy_weight=$bending
grid_spacing=$grid
smoothing_sigma_floating=$smoothing
affine_n_steps=$affine_n_steps
affine_use_n_steps=$affine_use_n_steps
freeform_n_steps=$freeform_n_steps
freeform_use_n_steps=$freeform_use_n_steps
EOF

brainreg "$AF_DIR" "$SAVE_DIR" \
  -v "$vx" "$vy" "$vz" \
  --orientation "$orientation" \
  --atlas "$atlas" \
  --debug \
  --save-original-orientation \
  -a "$UNPROCESSED_DIR" \
  --n-free-cpus 0 \
  --bending-energy-weight "$bending" \
  --grid-spacing "$grid" \
  --smoothing-sigma-floating "$smoothing" \
  --affine-n-steps "$affine_n_steps" \
  --affine-use-n-steps "$affine_use_n_steps" \
  --freeform-n-steps "$freeform_n_steps" \
  --freeform-use-n-steps "$freeform_use_n_steps"