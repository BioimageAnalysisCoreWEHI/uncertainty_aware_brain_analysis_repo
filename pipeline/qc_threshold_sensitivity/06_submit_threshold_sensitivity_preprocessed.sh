#!/bin/bash
# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"
#
# Submit the global-QC threshold sensitivity sweep (05_qc_threshold_sensitivity.py),
# one SLURM array task per sample_id, for the PREPROCESSED qc table.
#
# Optional provisional reference gate (compared against every other gate):
#   REFERENCE_IOU=0.85 REFERENCE_GRAD=0.08 bash 05_submit_threshold_sensitivity_preprocessed.sh
# Re-running with a new reference needs OVERWRITE=1 (region masks are cached,
# so this is quick):
#   OVERWRITE=1 REFERENCE_IOU=0.85 REFERENCE_GRAD=0.08 bash 05_submit_threshold_sensitivity_preprocessed.sh

module load miniconda3
conda activate ${CONDA_ENV}

PROJECT_DIR="${IMAGING_ROOT}/Ishrat/JoM/post_review_analysis/validation_datasets/lightsheet_data"

PYTHON_EXE="${CONDA_ENV}/bin/python"

QC_TABLE="${PROJECT_DIR}/results/qc_tables/qc_results_preprocessed.csv"

JOB_TABLE="${PROJECT_DIR}/results/job_tables/region_consensus_jobs_preprocessed.csv"

SLURM_SCRIPT="${PROJECT_DIR}/scripts/run_threshold_sensitivity_array.sh"

REFERENCE_IOU=${REFERENCE_IOU:-}
REFERENCE_GRAD=${REFERENCE_GRAD:-}
OVERWRITE=${OVERWRITE:-0}


# ---------------------------------------------------------------------
# CHECK INPUT FILES
# ---------------------------------------------------------------------

if [ ! -f "${QC_TABLE}" ]; then
    echo "ERROR: QC table not found:"
    echo "${QC_TABLE}"
    exit 1
fi

if [ ! -f "${SLURM_SCRIPT}" ]; then
    echo "ERROR: SLURM worker script not found:"
    echo "${SLURM_SCRIPT}"
    exit 1
fi

# One row per sample_id -- same job table as the region-consensus step.
if [ ! -f "${JOB_TABLE}" ]; then
    echo "Job table not found; building it from the QC table."
    "${PYTHON_EXE}" "${PROJECT_DIR}/scripts/build_region_consensus_jobs.py" \
        --qc-table "${QC_TABLE}" \
        --out "${JOB_TABLE}" || exit 1
fi


# ---------------------------------------------------------------------
# SUBMIT
# ---------------------------------------------------------------------

N_JOBS=$(( $(wc -l < "${JOB_TABLE}") - 1 ))

if [ "${N_JOBS}" -le 0 ]; then
    echo "ERROR: No samples found in:"
    echo "${JOB_TABLE}"
    exit 1
fi

echo "Submitting ${N_JOBS} threshold-sensitivity jobs (one per sample_id)"
echo "QC table: ${QC_TABLE}"
echo "Reference gate: ${REFERENCE_IOU:-none} / ${REFERENCE_GRAD:-none}"
echo "Overwrite: ${OVERWRITE}"
echo

sbatch \
    --array="0-$((N_JOBS - 1))" \
    --export="ALL,JOB_OFFSET=0,JOB_TABLE=${JOB_TABLE},QC_TABLE=${QC_TABLE},REFERENCE_IOU=${REFERENCE_IOU},REFERENCE_GRAD=${REFERENCE_GRAD},OVERWRITE=${OVERWRITE}" \
    "${SLURM_SCRIPT}"
