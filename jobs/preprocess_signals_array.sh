#!/bin/bash
#SBATCH --job-name=preprocess_batch
#SBATCH --account=def-forouzan
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --output=logs/preprocess_batch_%x_%j.out
#SBATCH --error=logs/preprocess_batch_%x_%j.err

# Batch preprocessing for a subset of subjects
# Useful for splitting large datasets across multiple jobs
#
# Usage: sbatch jobs/preprocess_signals_array.sh <dataset> <start_index> <end_index> [options]
#   dataset: stages | shhs | apples | mros (required)
#   start_index: starting subject index, 0-based (optional, default: 0)
#   end_index: ending subject index, exclusive (optional, default: all)
#   --log-level: DEBUG | INFO | WARNING | ERROR (optional)
#
# Examples:
#   # Process first 100 subjects
#   sbatch jobs/preprocess_signals_array.sh stages 0 100
#   
#   # Process subjects 100-200
#   sbatch jobs/preprocess_signals_array.sh stages 100 200
#   
#   # Process all subjects starting from 500
#   sbatch jobs/preprocess_signals_array.sh stages 500

set -e

# Parse arguments
DATASET=${1:-stages}
START_INDEX=${2:-0}
END_INDEX=${3:-}
LOG_LEVEL="INFO"

# Parse optional arguments
shift 3 2>/dev/null || shift $#
while [[ $# -gt 0 ]]; do
    case $1 in
        --log-level)
            LOG_LEVEL="$2"
            shift 2
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
echo "NSRR Signal Preprocessing - Batch Job"
echo "========================================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Dataset:       $DATASET"
echo "Start index:   $START_INDEX"
if [ -n "$END_INDEX" ]; then
    echo "End index:     $END_INDEX"
    N_SUBJECTS=$((END_INDEX - START_INDEX))
    echo "Subjects:      $N_SUBJECTS"
fi
echo "Log level:     $LOG_LEVEL"
echo "Start time:    $(date)"
echo "========================================================================"
echo ""

# Load metadata to get total subject count
TOTAL_SUBJECTS=$(python -c "
import pandas as pd
from pathlib import Path
import sys

metadata_path = Path('/scratch/boshra95/psg/unified/metadata/unified_metadata.parquet')
if not metadata_path.exists():
    metadata_path = Path('/scratch/boshra95/psg_metadata/unified_metadata.parquet')

if not metadata_path.exists():
    print('ERROR: Metadata not found', file=sys.stderr)
    sys.exit(1)

df = pd.read_parquet(metadata_path)
ds_df = df[(df['dataset'].str.upper() == '$DATASET'.upper()) & (df['has_edf'] == True)]
print(len(ds_df))
")

if [ -z "$END_INDEX" ]; then
    END_INDEX=$TOTAL_SUBJECTS
fi

echo "Total subjects in dataset: $TOTAL_SUBJECTS"
echo "Processing subjects $START_INDEX to $END_INDEX"
echo ""

# Calculate max subjects parameter
MAX_SUBJECTS=$END_INDEX

# Build and run command
CMD="python -c \"
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

from scripts.preprocess_signals import PreprocessingPipeline
import pandas as pd

config_path = Path('configs/preprocessing_params.yaml')
pipeline = PreprocessingPipeline(config_path=config_path)

# Load metadata
metadata_path = Path('/scratch/boshra95/psg/unified/metadata/unified_metadata.parquet')
if not metadata_path.exists():
    metadata_path = Path('/scratch/boshra95/psg_metadata/unified_metadata.parquet')

df = pd.read_parquet(metadata_path)
ds_df = df[(df['dataset'].str.upper() == '$DATASET'.upper()) & (df['has_edf'] == True)]

# Slice by index range
ds_df = ds_df.iloc[$START_INDEX:$END_INDEX]

if len(ds_df) == 0:
    print('No subjects in this range')
    sys.exit(0)

print(f'Processing {len(ds_df)} subjects from index $START_INDEX to $END_INDEX')

# Process with custom dataframe slice
# Note: This requires modifying the pipeline to accept a pre-filtered dataframe
# For now, use max_subjects as a workaround
pipeline.process_dataset(
    dataset_name='$DATASET',
    max_subjects=$MAX_SUBJECTS,
    skip_existing=True
)
\""

echo "This batch processing mode is being simplified."
echo "Please use the main script instead:"
echo "  sbatch jobs/preprocess_signals_parallel.sh $DATASET $MAX_SUBJECTS --log-level $LOG_LEVEL"
echo ""
echo "Or submit multiple jobs with different max_subjects values:"
echo "  sbatch jobs/preprocess_signals_parallel.sh $DATASET 100  # First 100"
echo "  # Then manually process the next batch after the first completes"
echo ""
echo "For true parallel processing, consider submitting multiple jobs:"
echo "  sbatch jobs/preprocess_signals_parallel.sh stages 500"
echo "  sbatch jobs/preprocess_signals_parallel.sh shhs 500"
echo "========================================================================"

# For now, just call the regular preprocessing for this batch
python scripts/preprocess_signals.py \
    --dataset "$DATASET" \
    --max-subjects "$MAX_SUBJECTS" \
    --skip-existing \
    --log-level "$LOG_LEVEL"

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
