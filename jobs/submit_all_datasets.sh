#!/bin/bash
# Submit preprocessing jobs for all datasets
# Usage: ./jobs/submit_all_datasets.sh [max_subjects] [--log-level LEVEL]
#
# Examples:
#   ./jobs/submit_all_datasets.sh              # Process all subjects
#   ./jobs/submit_all_datasets.sh 100          # Process first 100 of each
#   ./jobs/submit_all_datasets.sh 10 --log-level DEBUG  # Debug mode

set -e

MAX_SUBJECTS=${1:-}
LOG_LEVEL_OPT=""

# Parse optional arguments
shift 1 2>/dev/null || shift 0
while [[ $# -gt 0 ]]; do
    case $1 in
        --log-level)
            LOG_LEVEL_OPT="--log-level $2"
            shift 2
            ;;\n        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

cd "$(dirname "$0")/.."

# Create necessary directories
mkdir -p logs

echo "========================================================================"
echo "NSRR Preprocessing - Submit All Datasets"
echo "========================================================================"
if [ -n "$MAX_SUBJECTS" ]; then
    echo "Max subjects per dataset: $MAX_SUBJECTS"
else
    echo "Processing all subjects"
fi
if [ -n "$LOG_LEVEL_OPT" ]; then
    echo "Log level: $LOG_LEVEL_OPT"
fi
echo ""

echo "Submitting jobs..."
for dataset in stages shhs apples mros; do
    echo "  - $dataset"
    if [ -n "$MAX_SUBJECTS" ]; then
        sbatch --job-name="preprocess_${dataset}" \
            jobs/preprocess_signals_parallel.sh "$dataset" "$MAX_SUBJECTS" $LOG_LEVEL_OPT
    else
        sbatch --job-name="preprocess_${dataset}" \
            jobs/preprocess_signals_parallel.sh "$dataset" $LOG_LEVEL_OPT
    fi
done

echo ""
echo "========================================================================"
echo "Jobs submitted! Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  logs/preprocess_*.out"
echo "========================================================================"
if [ "$MODE" = "array" ]; then
    echo "  logs/array/preprocess_*.out"
else
    echo "  logs/preprocess_*.out"
fi
echo "========================================================================"
