#!/bin/bash
#SBATCH --job-name=region_consensus
#SBATCH --output=${IMAGING_ROOT}/Ishrat/hpc_processing/logs/region_consensus_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"

module load miniconda3
conda activate ${CONDA_ENV}

PROJECT_DIR="${IMAGING_ROOT}/Ishrat/JoM/post_review_analysis"
CONFIG="${PROJECT_DIR}/config/config.yaml"
QC_TABLE="${PROJECT_DIR}/results/qc_tables/qc_results_preprocessed.csv"

# All set by submit_region_consensus_sweep.sh via --export - edit there,
# not here.
JOB_TABLE=${JOB_TABLE}
JOB_OFFSET=${JOB_OFFSET:-0}
RUN_SETS=${RUN_SETS:-all}
GLOBAL_QC_COLUMN=${GLOBAL_QC_COLUMN:-passes_global_qc}

OVERWRITE=${OVERWRITE:-0}
OVERWRITE_FLAG=""
if [ "${OVERWRITE}" = "1" ]; then
    OVERWRITE_FLAG="--overwrite"
fi

GLOBAL_TASK_ID=$((JOB_OFFSET + SLURM_ARRAY_TASK_ID))

echo "SLURM array job ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Job offset: ${JOB_OFFSET}"
echo "Global job index: ${GLOBAL_TASK_ID}"
echo "Using job table: ${JOB_TABLE}"
echo "Run sets: ${RUN_SETS}"
echo "Global QC column: ${GLOBAL_QC_COLUMN}"
echo "Overwrite mode: ${OVERWRITE} (flag: '${OVERWRITE_FLAG}')"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo

if [ ! -f "${JOB_TABLE}" ]; then
    echo "ERROR: Job table not found: ${JOB_TABLE}"
    exit 1
fi

ROW_NUM=$((GLOBAL_TASK_ID + 2))
ROW=$(sed -n "${ROW_NUM}p" "${JOB_TABLE}")
IFS=',' read -r sample_id <<< "${ROW}"

echo "-----------------------------------"
echo "sample_id = ${sample_id}"
echo "-----------------------------------"

if [ -z "${sample_id}" ]; then
    echo "ERROR: Could not read a sample_id from row ${ROW_NUM} of ${JOB_TABLE}"
    exit 1
fi

# --sample-id scopes this task to just one sample; every input_type present
# for that sample_id in the qc table is still covered within this single
# task (matches how the job table was built -- one row per unique
# sample_id, see build_region_consensus_jobs.py).
python -u "${PROJECT_DIR}/scripts/build_region_consensus_wholebrain.py" \
    --config "${CONFIG}" \
    --qc-table "${QC_TABLE}" \
    --global-qc-column "${GLOBAL_QC_COLUMN}" \
    --run-sets ${RUN_SETS} \
    --sample-id "${sample_id}" \
    ${OVERWRITE_FLAG}

echo
echo "Finished: $(date)"