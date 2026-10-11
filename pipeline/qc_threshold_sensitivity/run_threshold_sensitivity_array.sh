#!/bin/bash
#SBATCH --job-name=qc_threshold_sensitivity
#SBATCH --output=${IMAGING_ROOT}/Ishrat/hpc_processing/logs/threshold_sensitivity_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"


module load miniconda3
conda activate ${CONDA_ENV}

PROJECT_DIR="${IMAGING_ROOT}/Ishrat/JoM/post_review_analysis/validation_datasets/lightsheet_data"

PYTHON_EXE="${CONDA_ENV}/bin/python"

PYTHON_SCRIPT="${PROJECT_DIR}/scripts/05_qc_threshold_sensitivity.py"

CONFIG_FILE="${PROJECT_DIR}/config/config.yaml"

# All set by 05_submit_threshold_sensitivity*.sh via --export - edit there,
# not here.
JOB_TABLE=${JOB_TABLE}
QC_TABLE=${QC_TABLE}
JOB_OFFSET=${JOB_OFFSET:-0}
REFERENCE_IOU=${REFERENCE_IOU:-}
REFERENCE_GRAD=${REFERENCE_GRAD:-}
OVERWRITE=${OVERWRITE:-0}

REFERENCE_FLAGS=""
if [ -n "${REFERENCE_IOU}" ] && [ -n "${REFERENCE_GRAD}" ]; then
    REFERENCE_FLAGS="--reference-iou ${REFERENCE_IOU} --reference-grad ${REFERENCE_GRAD}"
fi

OVERWRITE_FLAG=""
if [ "${OVERWRITE}" = "1" ]; then
    OVERWRITE_FLAG="--overwrite"
fi


# ---------------------------------------------------------------------
# CHECKS
# ---------------------------------------------------------------------

if [ ! -x "${PYTHON_EXE}" ]; then
    echo "ERROR: Python executable not found:"
    echo "${PYTHON_EXE}"
    exit 1
fi

for FILE in "${PYTHON_SCRIPT}" "${CONFIG_FILE}" "${JOB_TABLE}" "${QC_TABLE}"; do
    if [ ! -f "${FILE}" ]; then
        echo "ERROR: File not found:"
        echo "${FILE}"
        exit 1
    fi
done


# ---------------------------------------------------------------------
# DETERMINE SAMPLE
# ---------------------------------------------------------------------

GLOBAL_JOB_INDEX=$((SLURM_ARRAY_TASK_ID + JOB_OFFSET))

ROW_NUM=$((GLOBAL_JOB_INDEX + 2))
ROW=$(sed -n "${ROW_NUM}p" "${JOB_TABLE}")
IFS=',' read -r sample_id <<< "${ROW}"

if [ -z "${sample_id}" ]; then
    echo "ERROR: Could not read a sample_id from row ${ROW_NUM} of ${JOB_TABLE}"
    exit 1
fi

echo "SLURM array job ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Global job index: ${GLOBAL_JOB_INDEX}"
echo "Sample: ${sample_id}"
echo "QC table: ${QC_TABLE}"
echo "Reference gate: ${REFERENCE_IOU:-none} / ${REFERENCE_GRAD:-none}"
echo "Overwrite: ${OVERWRITE}"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo


# ---------------------------------------------------------------------
# RUN ONE SAMPLE
# ---------------------------------------------------------------------

# Config paths (atlas, region tree, output dirs) are relative to the
# project root.
cd "${PROJECT_DIR}"

"${PYTHON_EXE}" -u "${PYTHON_SCRIPT}" \
    --config "${CONFIG_FILE}" \
    --qc "${QC_TABLE}" \
    --sample-id "${sample_id}" \
    ${REFERENCE_FLAGS} \
    ${OVERWRITE_FLAG}


echo
echo "Finished: $(date)"
