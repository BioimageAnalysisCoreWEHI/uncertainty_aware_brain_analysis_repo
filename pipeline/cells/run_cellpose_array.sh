#!/bin/bash
#SBATCH --job-name=cellpose_array
#SBATCH --output=${IMAGING_ROOT}/Ishrat/hpc_processing/logs/cellpose_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A30:1
# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"

module load miniconda3
conda activate ${CONDA_ENV}


JOB_TABLE="results/job_tables/cellpose_jobs.csv"
CONFIG="config/config.yaml"

ROW_NUM=$((SLURM_ARRAY_TASK_ID + 2))
ROW=$(sed -n "${ROW_NUM}p" "$JOB_TABLE")

IFS=',' read -r sample_id input_type th_crop out_dir <<< "$ROW"

echo "Sample: $sample_id"
echo "Input type: $input_type"
echo "TH crop: $th_crop"
echo "Output: $out_dir"

mkdir -p "$out_dir"

python scripts/05_run_cellpose_one_crop.py \
  --config "$CONFIG" \
  --image "$th_crop" \
  --out-dir "$out_dir"