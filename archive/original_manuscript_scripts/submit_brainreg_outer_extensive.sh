# Storage roots (set these for your system; see config/local_paths.example.yaml):
: "${IMAGING_ROOT:?export IMAGING_ROOT=/path/to/project/storage}"
: "${SCRATCH:?export SCRATCH=/path/to/your/scratch}"
: "${SCRATCH_ROOT:=${SCRATCH}}"
: "${CONDA_ENV:=brainreg_hpc}"
 #!/bin/bash
#SBATCH --job-name=brainreg_outer
#SBATCH --output=${IMAGING_ROOT}/Ishrat/hpc_processing/logs/brainreg_outer_%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --array=0-4

module load miniconda3
conda activate ${CONDA_ENV}

# Get the AF folder for this task
AF_FOLDERS=($(ls -d ${IMAGING_ROOT}/Ishrat/JoM/final_preprocessed_stacks/*AF*))
AF_DIR=${AF_FOLDERS[$SLURM_ARRAY_TASK_ID]}
SAMPLE_NAME=$(basename $AF_DIR)
TH_DIR=$(echo $AF_DIR | sed 's/AF/TH/')  # match your pattern
UNPROCESSED_DIR=$(echo $AF_DIR | sed 's/AF/unpreprocessed/')

echo "Submitting sub-array job for sample: $SAMPLE_NAME"

echo "AF_DIR          = $AF_DIR"
echo "TH_DIR          = $TH_DIR"
echo "UNPROCESSED_DIR = $UNPROCESSED_DIR"


# Submit sub-array job for the 48 parameter combinations
#sbatch --export=AF_DIR="$AF_DIR",TH_DIR="$TH_DIR",UNPROCESSED_DIR="$UNPROCESSED_DIR",SAMPLE_NAME="$SAMPLE_NAME" --array=0-7 submit_brainreg_inner_extensive.sh

sbatch --export=AF_DIR="$AF_DIR",TH_DIR="$TH_DIR",SAMPLE_NAME="$SAMPLE_NAME" \
       --array=0-47 submit_brainreg_inner_extensive.sh