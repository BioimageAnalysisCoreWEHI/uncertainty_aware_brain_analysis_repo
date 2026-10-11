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

PYTHON_EXE="${CONDA_ENV}/bin/python"

PYTHON_SCRIPT="${PROJECT_DIR}/scripts/run_qc_single.py"

CONFIG_FILE="${PROJECT_DIR}/config/config.yaml"

SAMPLES_FILE="${PROJECT_DIR}/config/samples_amap_validation_raw.csv"

JOBS_FILE=${JOB_TABLE}

OUTPUT_DIR=${OUTPUT_DIR}


# ---------------------------------------------------------------------
# CHECKS
# ---------------------------------------------------------------------

if [ ! -x "${PYTHON_EXE}" ]; then
    echo "ERROR: Python executable not found:"
    echo "${PYTHON_EXE}"
    exit 1
fi

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "ERROR: Python QC script not found:"
    echo "${PYTHON_SCRIPT}"
    exit 1
fi

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found:"
    echo "${CONFIG_FILE}"
    exit 1
fi

if [ ! -f "${SAMPLES_FILE}" ]; then
    echo "ERROR: Samples file not found:"
    echo "${SAMPLES_FILE}"
    exit 1
fi

if [ ! -f "${JOBS_FILE}" ]; then
    echo "ERROR: Jobs file not found:"
    echo "${JOBS_FILE}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p ${IMAGING_ROOT}/Ishrat/hpc_processing/logs


# ---------------------------------------------------------------------
# DETERMINE REGISTRATION INDEX
# ---------------------------------------------------------------------

JOB_OFFSET="${JOB_OFFSET:-0}"

GLOBAL_JOB_INDEX=$((SLURM_ARRAY_TASK_ID + JOB_OFFSET))


echo "SLURM array job ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Job offset: ${JOB_OFFSET}"
echo "Global registration index: ${GLOBAL_JOB_INDEX}"

echo "Using job table: $JOB_TABLE"
echo "Using output directory: $OUTPUT_DIR"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo

echo "Python executable: ${PYTHON_EXE}"
echo "Python script: ${PYTHON_SCRIPT}"
echo


# ---------------------------------------------------------------------
# RUN ONE REGISTRATION
# ---------------------------------------------------------------------

"${PYTHON_EXE}" "${PYTHON_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --samples "${SAMPLES_FILE}" \
    --jobs "${JOBS_FILE}" \
    --job-index "${GLOBAL_JOB_INDEX}" \
    --output-dir "${OUTPUT_DIR}"


echo
echo "Finished: $(date)"