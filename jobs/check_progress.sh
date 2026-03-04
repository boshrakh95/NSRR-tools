#!/bin/bash
# Monitor preprocessing job progress
# Usage: ./jobs/check_progress.sh [dataset]

DATASET=${1:-all}
SCRATCH_BASE="/scratch/$USER/psg"

echo "========================================================================"
echo "NSRR Preprocessing Progress Report"
echo "========================================================================"
echo "Date: $(date)"
echo ""

# Function to count files for a dataset
check_dataset() {
    local ds=$1
    local ds_upper=$(echo "$ds" | tr '[:lower:]' '[:upper:]')
    
    # Get expected count from metadata
    local expected=$(python3 -c "
import pandas as pd
from pathlib import Path
import sys

try:
    metadata_path = Path('$SCRATCH_BASE/unified/metadata/unified_metadata.parquet')
    if not metadata_path.exists():
        metadata_path = Path('/scratch/$USER/psg_metadata/unified_metadata.parquet')
    
    df = pd.read_parquet(metadata_path)
    ds_df = df[(df['dataset'].str.upper() == '$ds_upper') & (df['has_edf'] == True)]
    print(len(ds_df))
except Exception as e:
    print('0')
" 2>/dev/null)
    
    # Count processed files
    local hdf5_dir="$SCRATCH_BASE/$ds/derived/hdf5_signals"
    local annot_dir="$SCRATCH_BASE/$ds/derived/annotations"
    
    if [ -d "$hdf5_dir" ]; then
        local n_hdf5=$(find "$hdf5_dir" -name "*.h5" 2>/dev/null | wc -l)
    else
        local n_hdf5=0
    fi
    
    if [ -d "$annot_dir" ]; then
        local n_annot=$(find "$annot_dir" -name "*_annotations.json" 2>/dev/null | wc -l)
    else
        local n_annot=0
    fi
    
    # Calculate percentages
    local pct_hdf5=0
    local pct_annot=0
    if [ $expected -gt 0 ]; then
        pct_hdf5=$((100 * n_hdf5 / expected))
        pct_annot=$((100 * n_annot / expected))
    fi
    
    # Get total size
    local size="0"
    if [ -d "$hdf5_dir" ]; then
        size=$(du -sh "$hdf5_dir" 2>/dev/null | awk '{print $1}')
    fi
    
    # Print status
    printf "%-10s: %4d/%-4d HDF5 (%3d%%) | %4d/%-4d Annot (%3d%%) | Size: %s\n" \
        "$ds_upper" $n_hdf5 $expected $pct_hdf5 $n_annot $expected $pct_annot "$size"
}

# Check datasets
if [ "$DATASET" = "all" ]; then
    for ds in stages shhs apples mros; do
        check_dataset "$ds"
    done
else
    check_dataset "$DATASET"
fi

echo ""
echo "------------------------------------------------------------------------"
echo "SLURM Jobs:"
echo "------------------------------------------------------------------------"
squeue -u $USER -o "%.10i %.9P %.30j %.8u %.2t %.10M %.6D %R" 2>/dev/null || echo "No active jobs"

echo ""
echo "------------------------------------------------------------------------"
echo "Recent Errors (last 10):"
echo "------------------------------------------------------------------------"
if [ -d "logs" ]; then
    find logs -name "*.err" -type f -exec grep -l "ERROR\|Failed" {} \; 2>/dev/null | tail -10 | while read errfile; do
        echo "  $errfile:"
        grep -m 1 "ERROR\|Failed" "$errfile" 2>/dev/null | sed 's/^/    /'
    done
else
    echo "  No log directory found"
fi

echo ""
echo "========================================================================"
echo "Quick Commands:"
echo "  Watch progress:     watch -n 30 ./jobs/check_progress.sh"
echo "  Cancel all jobs:    scancel -u \$USER"
echo "  Resubmit failed:    sbatch jobs/preprocess_signals_parallel.sh <dataset>"
echo "========================================================================"
