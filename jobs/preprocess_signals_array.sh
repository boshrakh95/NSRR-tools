#!/bin/bash
#SBATCH --job-name=preprocess_array
#SBATCH --account=def-YOUR_ACCOUNT  # CHANGE THIS
#SBATCH --time=02:00:00             # 2 hours per task
#SBATCH --cpus-per-task=4           # 4 cores per task
#SBATCH --mem-per-cpu=4G            # 4GB per core = 16GB total
#SBATCH --array=0-999%50            # Process 1000 subjects, max 50 concurrent
#SBATCH --output=logs/array/preprocess_%A_%a.out
#SBATCH --error=logs/array/preprocess_%A_%a.err

# SLURM Array Job for massive parallelization
# Each array task processes one subject
# 
# Usage:
#   1. First, generate subject list:
#      python scripts/generate_subject_list.py --dataset stages --output jobs/subject_lists/stages_subjects.txt
#   
#   2. Then submit array job:
#      sbatch jobs/preprocess_signals_array.sh stages

set -e

# Configuration
DATASET=${1:-stages}
SUBJECT_LIST_FILE="jobs/subject_lists/${DATASET}_subjects.txt"

# Setup paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$SCRIPT_DIR"

# Create logs directory
mkdir -p logs/array

# Load modules
module load python/3.11
module load scipy-stack
module load gcc/9.3.0

# Activate environment
source "$HOME/sleepfm_env/bin/activate"

echo "========================================================================"
echo "NSRR Signal Preprocessing - Array Job"
echo "========================================================================"
echo "Array Job ID:  $SLURM_ARRAY_JOB_ID"
echo "Task ID:       $SLURM_ARRAY_TASK_ID"
echo "Node:          $SLURM_NODELIST"
echo "Dataset:       $DATASET"
echo "========================================================================"

# Check if subject list exists
if [ ! -f "$SUBJECT_LIST_FILE" ]; then
    echo "ERROR: Subject list not found: $SUBJECT_LIST_FILE"
    echo "Please run: python scripts/generate_subject_list.py --dataset $DATASET"
    exit 1
fi

# Get total number of subjects
N_SUBJECTS=$(wc -l < "$SUBJECT_LIST_FILE")
echo "Total subjects in list: $N_SUBJECTS"

# Get this task's subject (1-indexed line number = TASK_ID + 1)
TASK_LINE=$((SLURM_ARRAY_TASK_ID + 1))

if [ $TASK_LINE -gt $N_SUBJECTS ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID exceeds number of subjects ($N_SUBJECTS)"
    echo "Nothing to process for this task."
    exit 0
fi

# Read subject info from list (format: subject_id edf_path)
SUBJECT_INFO=$(sed -n "${TASK_LINE}p" "$SUBJECT_LIST_FILE")
SUBJECT_ID=$(echo "$SUBJECT_INFO" | awk '{print $1}')
EDF_PATH=$(echo "$SUBJECT_INFO" | awk '{print $2}')

echo "Processing: $SUBJECT_ID"
echo "EDF: $EDF_PATH"
echo "------------------------------------------------------------------------"

# Process this subject
python scripts/preprocess_single_subject.py \
    --dataset "$DATASET" \
    --subject-id "$SUBJECT_ID" \
    --edf-path "$EDF_PATH" \
    --skip-existing

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Completed: $SUBJECT_ID"
else
    echo "✗ Failed: $SUBJECT_ID (exit code: $EXIT_CODE)"
fi

echo "========================================================================"

deactivate

exit $EXIT_CODE
