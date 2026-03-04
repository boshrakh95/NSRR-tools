#!/bin/bash
#SBATCH --job-name=preprocess_signals
#SBATCH --account=def-YOUR_ACCOUNT  # CHANGE THIS to your CC account
#SBATCH --time=12:00:00             # Max 12 hours per job
#SBATCH --cpus-per-task=16          # 16 cores for parallel processing
#SBATCH --mem=64G                   # 64GB RAM (adjust based on needs)
#SBATCH --output=logs/preprocess_%x_%j.out
#SBATCH --error=logs/preprocess_%x_%j.err

# Preprocess NSRR EDF signals to HDF5 format with GNU parallel
# Usage: sbatch jobs/preprocess_signals_parallel.sh <dataset>
#   where dataset = stages | shhs | apples | mros | all

set -e

# Configuration
DATASET=${1:-all}               # Dataset to process (default: all)
N_WORKERS=${SLURM_CPUS_PER_TASK:-8}  # Number of parallel workers
MAX_SUBJECTS=${2:-}             # Optional: limit number of subjects

# Setup paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$SCRIPT_DIR"

# Create logs directory
mkdir -p logs

# Load modules (Compute Canada)
module load python/3.11
module load scipy-stack
module load gcc/9.3.0

# Activate conda environment
source "$HOME/sleepfm_env/bin/activate"

# Install GNU parallel if not available (session-local)
if ! command -v parallel &> /dev/null; then
    echo "Installing GNU parallel..."
    mkdir -p "$HOME/.local/bin"
    wget -O "$HOME/.local/bin/parallel" https://raw.githubusercontent.com/martinda/gnu-parallel/master/src/parallel
    chmod +x "$HOME/.local/bin/parallel"
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "========================================================================"
echo "NSRR Signal Preprocessing - Parallel Processing"
echo "========================================================================"
echo "Dataset:       $DATASET"
echo "Workers:       $N_WORKERS"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Memory:        $SLURM_MEM_PER_NODE MB"
if [ -n "$MAX_SUBJECTS" ]; then
    echo "Max subjects:  $MAX_SUBJECTS"
fi
echo "========================================================================"

# Function to process a single subject
process_subject() {
    local dataset=$1
    local subject_id=$2
    local edf_path=$3
    
    echo "[$(date +%H:%M:%S)] Processing $dataset: $subject_id"
    
    # Run preprocessing for this subject
    python scripts/preprocess_single_subject.py \
        --dataset "$dataset" \
        --subject-id "$subject_id" \
        --edf-path "$edf_path" \
        --skip-existing \
        2>&1 | grep -E "(SUCCESS|ERROR|WARNING)" || true
}

export -f process_subject

# Get list of subjects to process
if [ "$DATASET" = "all" ]; then
    DATASETS="stages shhs apples mros"
else
    DATASETS="$DATASET"
fi

# Process each dataset
for ds in $DATASETS; do
    echo ""
    echo "Processing dataset: ${ds^^}"
    echo "------------------------------------------------------------------------"
    
    # Get subject list from metadata
    SUBJECT_LIST=$(python -c "
import pandas as pd
from pathlib import Path
import sys

# Load metadata
metadata_path = Path('/scratch/boshra95/psg/unified/metadata/unified_metadata.parquet')
if not metadata_path.exists():
    metadata_path = Path('/scratch/boshra95/psg_metadata/unified_metadata.parquet')

if not metadata_path.exists():
    print('ERROR: Metadata not found', file=sys.stderr)
    sys.exit(1)

df = pd.read_parquet(metadata_path)
ds_df = df[(df['dataset'].str.upper() == '$ds'.upper()) & (df['has_edf'] == True)]

if len(ds_df) == 0:
    print('ERROR: No subjects found', file=sys.stderr)
    sys.exit(1)

# Limit if requested
max_subjects = '$MAX_SUBJECTS'
if max_subjects:
    ds_df = ds_df.head(int(max_subjects))

# Output: dataset subject_id edf_path (one per line)
for idx, row in ds_df.iterrows():
    subject_id = row.get('subject_id', row.get('nsrrid', ''))
    edf_path = row.get('edf_path', '')
    if edf_path:
        print(f'$ds {subject_id} {edf_path}')
" 2>&1)
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to read metadata for $ds"
        continue
    fi
    
    # Count subjects
    N_SUBJECTS=$(echo "$SUBJECT_LIST" | wc -l)
    echo "Found $N_SUBJECTS subjects to process"
    
    if [ $N_SUBJECTS -eq 0 ]; then
        echo "No subjects to process for $ds"
        continue
    fi
    
    # Process in parallel using GNU parallel
    echo "$SUBJECT_LIST" | parallel -j "$N_WORKERS" --colsep ' ' \
        process_subject {1} {2} {3}
    
    echo "Completed $ds: $N_SUBJECTS subjects"
done

echo ""
echo "========================================================================"
echo "All datasets completed!"
echo "========================================================================"

# Deactivate environment
deactivate
