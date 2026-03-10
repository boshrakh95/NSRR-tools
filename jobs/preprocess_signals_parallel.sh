#!/bin/bash
#SBATCH --job-name=preprocess_signals
#SBATCH --account=def-forouzan
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=9000M
#SBATCH --output=logs/preprocess_%x_%j.out
#SBATCH --error=logs/preprocess_%x_%j.err

# Preprocess NSRR EDF signals to HDF5 format
# Usage: sbatch jobs/preprocess_signals_parallel.sh <dataset> [max_subjects] [--no-skip-existing] [--reprocess-annotations] [--log-level LEVEL] [--config PATH]
#   dataset: stages | shhs | apples | mros | all (required)
#   max_subjects: limit number of subjects (optional, e.g., 10 for testing)
#   --no-skip-existing: reprocess existing files (optional)
#   --reprocess-annotations: reprocess annotations only, keep HDF5 (optional)
#   --log-level: DEBUG | INFO | WARNING | ERROR (optional, default: INFO)
#   --config: path to custom preprocessing_params.yaml (optional)
#
# Examples:
#   sbatch jobs/preprocess_signals_parallel.sh stages
#   sbatch jobs/preprocess_signals_parallel.sh stages 10
#   sbatch jobs/preprocess_signals_parallel.sh stages 10 --log-level DEBUG
#   sbatch jobs/preprocess_signals_parallel.sh shhs --reprocess-annotations  # Keep HDF5, redo annotations
#   sbatch jobs/preprocess_signals_parallel.sh all --no-skip-existing

set -e

# Parse arguments
DATASET=${1:-stages}           # Dataset: stages | shhs | apples | mros | all
MAX_SUBJECTS=""
SKIP_EXISTING="--skip-existing"
REPROCESS_ANNOTATIONS=""       # "" (default: off) | "--reprocess-annotations" (uncomment to enable)
LOG_LEVEL="INFO"
CONFIG_PATH=""

# Parse remaining arguments
shift 1  # Remove dataset argument
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-existing)
            SKIP_EXISTING="--skip-existing"
            shift
            ;;
        --no-skip-existing)
            SKIP_EXISTING=""
            shift
            ;;
        --reprocess-annotations)
            REPROCESS_ANNOTATIONS="--reprocess-annotations"
            shift
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        [0-9]*)
            MAX_SUBJECTS="$1"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Setup paths
cd /home/boshra95/NSRR-tools
mkdir -p logs

# Activate virtual environment
source .venv/bin/activate

echo "========================================================================"
echo "NSRR Signal Preprocessing"
echo "========================================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Dataset:       $DATASET"
if [ -n "$MAX_SUBJECTS" ]; then
    echo "Max subjects:  $MAX_SUBJECTS"
fi
echo "Skip existing: $([ -n "$SKIP_EXISTING" ] && echo 'Yes' || echo 'No')"
echo "Reprocess annot: $([ -n "$REPROCESS_ANNOTATIONS" ] && echo 'Yes' || echo 'No')"
echo "Log level:     $LOG_LEVEL"
if [ -n "$CONFIG_PATH" ]; then
    echo "Config:        $CONFIG_PATH"
fi
echo "Start time:    $(date)"
echo "========================================================================"
echo ""

# Build Python command
CMD="python scripts/preprocess_signals.py --dataset $DATASET"

if [ -n "$MAX_SUBJECTS" ]; then
    CMD="$CMD --max-subjects $MAX_SUBJECTS"
fi

if [ -n "$SKIP_EXISTING" ]; then
    CMD="$CMD $SKIP_EXISTING"
fi

if [ -n "$REPROCESS_ANNOTATIONS" ]; then
    CMD="$CMD $REPROCESS_ANNOTATIONS"
fi

if [ -n "$LOG_LEVEL" ]; then
    CMD="$CMD --log-level $LOG_LEVEL"
fi

if [ -n "$CONFIG_PATH" ]; then
    CMD="$CMD --config $CONFIG_PATH"
fi

# Run preprocessing
echo "Running: $CMD"
echo ""

eval $CMD

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "End time:      $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Status:        SUCCESS"
else
    echo "Status:        FAILED (exit code: $EXIT_CODE)"
fi
echo "========================================================================"

deactivate

exit $EXIT_CODE
