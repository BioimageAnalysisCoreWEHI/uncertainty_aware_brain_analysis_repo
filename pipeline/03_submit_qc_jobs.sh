#!/bin/bash
#SBATCH --job-name=single_reg_qc
#SBATCH --output=${IMAGING_ROOT}/Ishrat/hpc_processing/logs/qc_%A_%a.out
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=00:30:00
# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"


module load miniconda3
conda activate ${CONDA_ENV}

PROJECT_DIR="${IMAGING_ROOT}/Ishrat/JoM/post_review_analysis"

JOB_TABLE="${PROJECT_DIR}/results/job_tables/brainreg_jobs.csv"

SLURM_SCRIPT="${PROJECT_DIR}/scripts/run_qc_array.sh"

OUTPUT_DIR="${PROJECT_DIR}/results/qc_tables/individual_registration_qc_hpc"


# ---------------------------------------------------------------------
# CHECK INPUT FILES
# ---------------------------------------------------------------------

if [ ! -f "${JOB_TABLE}" ]; then
    echo "ERROR: Job table not found:"
    echo "${JOB_TABLE}"
    exit 1
fi

if [ ! -f "${SLURM_SCRIPT}" ]; then
    echo "ERROR: SLURM worker script not found:"
    echo "${SLURM_SCRIPT}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"


# ---------------------------------------------------------------------
# COUNT REGISTRATIONS
# ---------------------------------------------------------------------

# Count the CSV rows and subtract one for the header.
N_JOBS=$(( $(wc -l < "${JOB_TABLE}") - 1 ))

if [ "${N_JOBS}" -le 0 ]; then
    echo "ERROR: No registrations found in:"
    echo "${JOB_TABLE}"
    exit 1
fi

echo "Submitting ${N_JOBS} QC jobs"
echo


# ---------------------------------------------------------------------
# CHUNK SETTINGS
# ---------------------------------------------------------------------

CHUNK_SIZE=1000

START=0
CHUNK_NUMBER=1

while [ "${START}" -lt "${N_JOBS}" ]; do

    REMAINING=$((N_JOBS - START))

    if [ "${REMAINING}" -lt "${CHUNK_SIZE}" ]; then
        THIS_CHUNK_SIZE="${REMAINING}"
    else
        THIS_CHUNK_SIZE="${CHUNK_SIZE}"
    fi

    ARRAY_END=$((THIS_CHUNK_SIZE - 1))

    GLOBAL_END=$((START + THIS_CHUNK_SIZE - 1))

    echo "Chunk ${CHUNK_NUMBER}"
    echo "  Global registration indices: ${START}-${GLOBAL_END}"
    echo "  SLURM array indices: 0-${ARRAY_END}"
    echo "  Job offset: ${START}"

    SUBMISSION_OUTPUT=$(
        sbatch \
            --array="0-${ARRAY_END}" \
            --export="ALL,JOB_OFFSET=${START},JOB_TABLE=${JOB_TABLE},OUTPUT_DIR=${OUTPUT_DIR}" \
            "${SLURM_SCRIPT}"
    )

    echo "  ${SUBMISSION_OUTPUT}"
    echo

    START=$((START + THIS_CHUNK_SIZE))

    CHUNK_NUMBER=$((CHUNK_NUMBER + 1))

done

echo "All QC chunks submitted."