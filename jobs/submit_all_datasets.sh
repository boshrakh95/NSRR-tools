#!/bin/bash
# Submit preprocessing jobs for all datasets
# Usage: ./jobs/submit_all_datasets.sh [parallel|array]

set -e

MODE=${1:-parallel}  # Default to parallel mode
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "$SCRIPT_DIR/.."

# Create necessary directories
mkdir -p logs logs/array jobs/subject_lists

echo "========================================================================"
echo "NSRR Preprocessing - Submit All Datasets"
echo "========================================================================"
echo "Mode: $MODE"
echo ""

if [ "$MODE" = "array" ]; then
    echo "Generating subject lists..."
    for dataset in stages shhs apples mros; do
        echo "  - $dataset"
        python scripts/generate_subject_list.py \
            --dataset "$dataset" \
            --output "jobs/subject_lists/${dataset}_subjects.txt"
    done
    
    echo ""
    echo "Submitting array jobs..."
    for dataset in stages shhs apples mros; do
        n_subjects=$(wc -l < "jobs/subject_lists/${dataset}_subjects.txt")
        max_array=$((n_subjects - 1))
        
        echo "  - $dataset: $n_subjects subjects (array 0-$max_array)"
        sbatch --array="0-${max_array}%50" \
            --job-name="preprocess_${dataset}" \
            jobs/preprocess_signals_array.sh "$dataset"
    done
    
elif [ "$MODE" = "parallel" ]; then
    echo "Submitting parallel processing jobs..."
    for dataset in stages shhs apples mros; do
        echo "  - $dataset"
        sbatch --job-name="preprocess_${dataset}" \
            jobs/preprocess_signals_parallel.sh "$dataset"
    done
    
else
    echo "ERROR: Invalid mode '$MODE'. Use 'parallel' or 'array'"
    exit 1
fi

echo ""
echo "========================================================================"
echo "Jobs submitted! Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
if [ "$MODE" = "array" ]; then
    echo "  logs/array/preprocess_*.out"
else
    echo "  logs/preprocess_*.out"
fi
echo "========================================================================"
