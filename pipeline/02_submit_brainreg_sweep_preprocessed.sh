#!/bin/bash
#SBATCH --job-name=brainreg_sweep
#SBATCH --output=${IMAGING_ROOT}/Ishrat/hpc_processing/logs/v2_brainreg_%A_%a.out
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

JOB_TABLE="${IMAGING_ROOT}/Ishrat/JoM/post_review_analysis/results/job_tables/brainreg_jobs_preprocessed_proper.csv"
SLURM_SCRIPT="${IMAGING_ROOT}/Ishrat/JoM/post_review_analysis/scripts/run_brainreg_array.sh"


N_JOBS=$(($(wc -l < "$JOB_TABLE") - 1))

if [ "$N_JOBS" -le 0 ]; then
    echo "ERROR: No jobs found in $JOB_TABLE"
    exit 1
fi

echo "Submitting $N_JOBS brainreg jobs"

MAX_ARRAY_SIZE=1001
CHUNK_SIZE=1000

START=0

while [ "$START" -lt "$N_JOBS" ]; do
    REMAINING=$((N_JOBS - START))

    if [ "$REMAINING" -lt "$CHUNK_SIZE" ]; then
        THIS_CHUNK_SIZE=$REMAINING
    else
        THIS_CHUNK_SIZE=$CHUNK_SIZE
    fi

    ARRAY_END=$((THIS_CHUNK_SIZE - 1))

    echo "Submitting global jobs ${START}-$((START + THIS_CHUNK_SIZE - 1)) as array 0-${ARRAY_END} with offset ${START}"

    sbatch \
    --array=0-${ARRAY_END} \
    --export=ALL,JOB_OFFSET=${START},JOB_TABLE="${JOB_TABLE}" \
    "$SLURM_SCRIPT"

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to submit chunk starting at ${START}"
        exit 1
    fi

    START=$((START + CHUNK_SIZE))
done

echo "All chunks submitted."