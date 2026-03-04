#!/bin/bash
# NSRR Preprocessing - Quick Start Cheat Sheet
# ============================================

# ============================================
# BEFORE FIRST RUN
# ============================================

# 1. Verify metadata exists
ls -lh /scratch/$USER/psg/unified/metadata/unified_metadata.parquet

# 2. Create log directories
mkdir -p logs logs/array jobs/subject_lists

# ============================================
# BASIC USAGE
# ============================================

# Process single dataset (all subjects)
sbatch jobs/preprocess_signals_parallel.sh stages

# Process single dataset (first 10 subjects for testing)
sbatch jobs/preprocess_signals_parallel.sh stages 10

# Process all datasets
bash jobs/submit_all_datasets.sh

# Process all datasets (first 10 of each for testing)
bash jobs/submit_all_datasets.sh 10

# ============================================
# ADVANCED USAGE
# ============================================

# With debug logging
sbatch jobs/preprocess_signals_parallel.sh stages 10 --log-level DEBUG

# Reprocess existing files (no skip)
sbatch jobs/preprocess_signals_parallel.sh stages --no-skip-existing

# Custom config file
sbatch jobs/preprocess_signals_parallel.sh stages --config configs/custom_preprocessing.yaml

# Combine options
sbatch jobs/preprocess_signals_parallel.sh stages 50 --log-level DEBUG

# All datasets with debug logging
bash jobs/submit_all_datasets.sh 10 --log-level DEBUG

# ============================================
# RESOURCE CUSTOMIZATION
# ============================================

# More memory (for large files)
sbatch --mem=32000M jobs/preprocess_signals_parallel.sh stages

# More time (for large datasets)
sbatch --time=24:00:00 jobs/preprocess_signals_parallel.sh shhs

# Different account
sbatch --account=def-OTHERACCOUNT jobs/preprocess_signals_parallel.sh stages

# ============================================
# MONITORING
# ============================================

# Check job status
squeue -u $USER

# Check progress
./jobs/check_progress.sh

# Watch progress in real-time
watch -n 30 './jobs/check_progress.sh'

# Check specific job logs (replace JOBID)
tail -f logs/preprocess_stages_JOBID.out

# Check for errors
grep -r "ERROR" logs/*.err

# ============================================
# TROUBLESHOOTING
# ============================================

# Job fails immediately
cat logs/preprocess_*.err | head -20

# Out of memory
sbatch --mem=32000M jobs/preprocess_signals_parallel.sh <dataset>

# Too slow (check if channels are reasonable)
# Edit configs/preprocessing_params.yaml:
#   channel_selection:
#     strategy: minimal  # Use 4 channels instead of 8

# Cancel jobs
scancel -u $USER

# Cancel specific job
scancel <job_id>

# ============================================
# VALIDATION
# ============================================

# Count output files
find /scratch/$USER/psg/stages/derived/hdf5_signals -name "*.h5" | wc -l

# Check specific output
h5ls /scratch/$USER/psg/stages/derived/hdf5_signals/subject_001.h5

# Validate HDF5 file
python scripts/validate_hdf5.py /scratch/$USER/psg/stages/derived/hdf5_signals/subject_001.h5

# Check total size
du -sh /scratch/$USER/psg/*/derived/hdf5_signals

# Validate HDF5 files (check for corruption)
python -c "
import h5py
from pathlib import Path
files = list(Path('/scratch/$USER/psg/stages/derived/hdf5_signals').glob('*.h5'))
bad = []
for f in files:
    try:
        with h5py.File(f, 'r') as h:
            pass
    except:
        bad.append(f)
print(f'Corrupted: {len(bad)}/{len(files)}')
"

# ============================================
# REPROCESSING FAILED SUBJECTS
# ============================================

# Identify failed subjects from logs
grep "Failed" logs/preprocess_*.out | grep -oP "(?<=: )[^ ]+" > failed_subjects.txt

# Just rerun - script will skip existing
sbatch jobs/preprocess_signals_parallel.sh stages

# ============================================
# RESOURCE OPTIMIZATION
# ============================================

# For small test run
sbatch jobs/preprocess_signals_parallel.sh stages 10

# For large dataset with more resources  
sbatch --cpus-per-task=32 --mem=128G --time=24:00:00 \
    jobs/preprocess_signals_parallel.sh shhs

# For quick dataset (APPLES)
sbatch --time=4:00:00 jobs/preprocess_signals_parallel.sh apples

# ============================================
# EXPECTED COMPLETION TIMES
# ============================================
# Parallel mode (16 cores):
#   STAGES (1700):  3-4 hours
#   APPLES (1100):  2-3 hours
#   MROS (2900):    6-8 hours
#   SHHS (8400):    15-20 hours
#
# Array mode (50 concurrent, 200 cores total):
#   STAGES: 1 hour
#   APPLES: 40 min
#   MROS:   2 hours
#   SHHS:   4-5 hours
