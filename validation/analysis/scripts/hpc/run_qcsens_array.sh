#!/bin/bash
#SBATCH --job-name=qcsens
#SBATCH --output=${IMAGING_ROOT}/Ishrat/hpc_processing/logs/qcsens_%A_%a.out
#SBATCH --ntasks=1
# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"

# Global-QC sensitivity analysis: one array task = one case (aMAP: brain x input x region; paper: brain x input).
# Resources (--cpus-per-task, --mem, --time) are set per dataset by submit_qcsens.sh - edit there, not here.

module load miniconda3
conda activate ${CONDA_ENV}

VA_DIR="${IMAGING_ROOT}/Ishrat/JoM/post_review_analysis/validation_analysis"
JOB_TABLE=${JOB_TABLE}
JOB_OFFSET=${JOB_OFFSET:-0}
GLOBAL_TASK_ID=$((JOB_OFFSET + SLURM_ARRAY_TASK_ID))

echo "SLURM array job ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Dataset: ${DATASET}"
echo "Job table: ${JOB_TABLE}"
echo "Global task index: ${GLOBAL_TASK_ID}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo

if [ ! -f "${JOB_TABLE}" ]; then
    echo "ERROR: Job table not found: ${JOB_TABLE}"
    exit 1
fi

python -u "${VA_DIR}/scripts/qcsens_worker.py" \
    --job-table "${JOB_TABLE}" \
    --task "${GLOBAL_TASK_ID}" \
    --threads "${SLURM_CPUS_PER_TASK:-4}"
STATUS=$?

echo
echo "Finished: $(date) (exit ${STATUS})"
exit ${STATUS}
