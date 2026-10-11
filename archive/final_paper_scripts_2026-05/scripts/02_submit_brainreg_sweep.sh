#!/bin/bash

JOB_TABLE="results/job_tables/brainreg_jobs.csv"
SLURM_SCRIPT="scripts/slurm/run_brainreg_array.sh"

if [ ! -f "$JOB_TABLE" ]; then
    echo "ERROR: Job table not found: $JOB_TABLE"
    echo "Run: python scripts/01_make_brainreg_jobs.py"
    exit 1
fi

N_JOBS=$(($(wc -l < "$JOB_TABLE") - 1))

if [ "$N_JOBS" -le 0 ]; then
    echo "ERROR: No jobs found in $JOB_TABLE"
    exit 1
fi

ARRAY_MAX=$((N_JOBS - 1))

echo "Submitting $N_JOBS brainreg jobs"
echo "Array range: 0-$ARRAY_MAX"

sbatch --array=0-${ARRAY_MAX} "$SLURM_SCRIPT"