#!/bin/bash
# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"
# Submit the global-QC sensitivity array for one dataset.
#
#   DATASET=amap  bash submit_qcsens.sh      (132 tasks: brain x input x region)
#   DATASET=paper bash submit_qcsens.sh      (10 tasks: brain x input)
#   DATASET=lsfm2 bash submit_qcsens.sh      (192 tasks: independent LSFM brain x input x region)
#
# Run `python 17_qc_sensitivity_consensus.py prepare --dataset <amap|paper>` first (it writes the job table).
# Tasks that already have a done-marker exit immediately, so resubmitting is safe. To rerun only some tasks:
#   DATASET=paper ARRAY="3,7" bash submit_qcsens.sh
# After all tasks finish: `python 17_qc_sensitivity_consensus.py aggregate --dataset <amap|paper>`.

DATASET=${DATASET:?set DATASET=amap or DATASET=paper}
VA_DIR="${IMAGING_ROOT}/Ishrat/JoM/post_review_analysis/validation_analysis"
JOB_TABLE="${VA_DIR}/results/qc_sensitivity/${DATASET}/jobs.csv"
SLURM_SCRIPT="${VA_DIR}/scripts/hpc/run_qcsens_array.sh"
mkdir -p ${IMAGING_ROOT}/Ishrat/hpc_processing/logs

case "${DATASET}" in
    amap)  RES="--cpus-per-task=8 --mem=24G --time=02:00:00" ;;
    paper) RES="--cpus-per-task=8 --mem=48G --time=04:00:00" ;;
    lsfm2) RES="--cpus-per-task=8 --mem=32G --time=02:00:00" ;;
    *) echo "ERROR: DATASET must be amap, paper or lsfm2"; exit 1 ;;
esac

if [ ! -f "${JOB_TABLE}" ]; then
    echo "ERROR: Job table not found: ${JOB_TABLE}"
    echo "Run: python 17_qc_sensitivity_consensus.py prepare --dataset ${DATASET}"
    exit 1
fi
if [ ! -f "${SLURM_SCRIPT}" ]; then
    echo "ERROR: worker script not found: ${SLURM_SCRIPT}"
    exit 1
fi

N_JOBS=$(( $(wc -l < "${JOB_TABLE}") - 1 ))
if [ "${N_JOBS}" -le 0 ]; then
    echo "ERROR: No jobs in ${JOB_TABLE}"
    exit 1
fi

if [ -n "${ARRAY}" ]; then
    echo "Submitting selected tasks ${ARRAY} of ${N_JOBS} (${DATASET})"
    sbatch --array="${ARRAY}" ${RES} \
        --export="ALL,JOB_OFFSET=0,JOB_TABLE=${JOB_TABLE},DATASET=${DATASET}" "${SLURM_SCRIPT}"
    exit $?
fi

echo "Submitting ${N_JOBS} ${DATASET} tasks (${RES})"
CHUNK_SIZE=1000
START=0
while [ "${START}" -lt "${N_JOBS}" ]; do
    REMAINING=$((N_JOBS - START))
    THIS_CHUNK=$(( REMAINING < CHUNK_SIZE ? REMAINING : CHUNK_SIZE ))
    echo "  tasks ${START}-$((START + THIS_CHUNK - 1))"
    sbatch --array="0-$((THIS_CHUNK - 1))" ${RES} \
        --export="ALL,JOB_OFFSET=${START},JOB_TABLE=${JOB_TABLE},DATASET=${DATASET}" "${SLURM_SCRIPT}"
    START=$((START + THIS_CHUNK))
done
echo "Submitted. Check progress with: DATASET=${DATASET} bash ${VA_DIR}/scripts/hpc/check_qcsens.sh"
