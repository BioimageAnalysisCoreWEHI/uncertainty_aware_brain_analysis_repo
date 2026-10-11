#!/bin/bash
# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"
# List finished / unfinished tasks of the global-QC sensitivity array.
#   DATASET=amap bash check_qcsens.sh
# Prints the ARRAY string to resubmit unfinished tasks:  DATASET=amap ARRAY="..." bash submit_qcsens.sh

DATASET=${DATASET:?set DATASET=amap or DATASET=paper}
ROOT="${IMAGING_ROOT}/Ishrat/JoM/post_review_analysis/validation_analysis/results/qc_sensitivity/${DATASET}"
N_JOBS=$(( $(wc -l < "${ROOT}/jobs.csv") - 1 ))
MISSING=()
for ((t = 0; t < N_JOBS; t++)); do
    [ -f "${ROOT}/done/${t}.done" ] || MISSING+=("${t}")
done
echo "${DATASET}: $((N_JOBS - ${#MISSING[@]})) / ${N_JOBS} tasks done"
if [ "${#MISSING[@]}" -gt 0 ]; then
    ARRAY=$(IFS=,; echo "${MISSING[*]}")
    echo "unfinished: ${ARRAY}"
    echo "resubmit:   DATASET=${DATASET} ARRAY=\"${ARRAY}\" bash submit_qcsens.sh"
fi
