#!/bin/bash
#SBATCH --job-name=region_consensus_sweep
#SBATCH --output=${IMAGING_ROOT}/Ishrat/hpc_processing/logs/region_consensus_submit_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"

module load miniconda3
conda activate ${CONDA_ENV}

PROJECT_DIR="${IMAGING_ROOT}/Ishrat/JoM/post_review_analysis"

JOB_TABLE="${PROJECT_DIR}/results/job_tables/region_consensus_jobs.csv"
SLURM_SCRIPT="${PROJECT_DIR}/scripts/run_region_consensus_array.sh"

# Off by default. Set via: OVERWRITE=1 sbatch submit_region_consensus_sweep.sh
OVERWRITE=${OVERWRITE:-0}
# 'all' (default) builds all_registrations + passes_global_qc +
# passes_final_qc_top_X per region/side. Narrow it, e.g.
# RUN_SETS="global_qc final_qc" sbatch submit_region_consensus_sweep.sh,
# to skip all_registrations if you don't need it.
RUN_SETS=${RUN_SETS:-all}
# Matches build_region_consensus_wholebrain.py's own default. Set to
# passes_global_qc_effective when using a manual-QC-override-merged
# table, e.g.: GLOBAL_QC_COLUMN=passes_global_qc_effective sbatch submit_region_consensus_sweep.sh
GLOBAL_QC_COLUMN=${GLOBAL_QC_COLUMN:-passes_global_qc}

if [ ! -f "${JOB_TABLE}" ]; then
    echo "ERROR: Job table not found:"
    echo "${JOB_TABLE}"
    echo "Run build_region_consensus_jobs.py first (this array reuses that table)."
    exit 1
fi

if [ ! -f "${SLURM_SCRIPT}" ]; then
    echo "ERROR: SLURM worker script not found:"
    echo "${SLURM_SCRIPT}"
    exit 1
fi

N_JOBS=$(( $(wc -l < "${JOB_TABLE}") - 1 ))
if [ "${N_JOBS}" -le 0 ]; then
    echo "ERROR: No jobs found in ${JOB_TABLE}"
    exit 1
fi

echo "Submitting ${N_JOBS} region-consensus jobs (one per sample_id)"
echo "Run sets: ${RUN_SETS}"
echo "Global QC column: ${GLOBAL_QC_COLUMN}"
echo "Overwrite mode: ${OVERWRITE}"
echo

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
    echo "  Global job indices: ${START}-${GLOBAL_END}"
    echo "  SLURM array indices: 0-${ARRAY_END}"
    echo "  Job offset: ${START}"

    SUBMISSION_OUTPUT=$(
        sbatch \
            --array="0-${ARRAY_END}" \
            --export="ALL,JOB_OFFSET=${START},JOB_TABLE=${JOB_TABLE},OVERWRITE=${OVERWRITE},RUN_SETS=${RUN_SETS},GLOBAL_QC_COLUMN=${GLOBAL_QC_COLUMN}" \
            "${SLURM_SCRIPT}"
    )
    echo "  ${SUBMISSION_OUTPUT}"
    echo

    START=$((START + THIS_CHUNK_SIZE))
    CHUNK_NUMBER=$((CHUNK_NUMBER + 1))
done

echo "All region-consensus chunks submitted."