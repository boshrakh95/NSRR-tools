#!/bin/bash
# NSRR Preprocessing - Quick Start Cheat Sheet
# ============================================

# ============================================
# BEFORE FIRST RUN
# ============================================

# 1. Update SLURM account in job scripts
sed -i 's/def-YOUR_ACCOUNT/def-ACTUAL_ACCOUNT/g' jobs/preprocess_signals_*.sh

# 2. Verify metadata exists
ls -lh /scratch/$USER/psg/unified/metadata/unified_metadata.parquet

# 3. Create log directories
mkdir -p logs logs/array jobs/subject_lists

# ============================================
# OPTION A: PARALLEL PROCESSING (Recommended)
# ============================================
# Good for: 100-1000 subjects per dataset
# Resources: 16 cores, 64GB RAM, single node

# Submit single dataset
sbatch jobs/preprocess_signals_parallel.sh stages

# Submit all datasets
bash jobs/submit_all_datasets.sh parallel

# With limited subjects (testing)
sbatch jobs/preprocess_signals_parallel.sh stages 10

# Monitor progress
watch -n 30 ./jobs/check_progress.sh

# ============================================
# OPTION B: ARRAY JOBS (For massive scale)
# ============================================
# Good for: 1000+ subjects, distributed across nodes
# Resources: 4 cores, 16GB per task, scales to 100s of nodes

# Step 1: Generate subject list
python scripts/generate_subject_list.py \
    --dataset stages \
    --output jobs/subject_lists/stages_subjects.txt

# Step 2: Submit array job
N=$(wc -l < jobs/subject_lists/stages_subjects.txt)
sbatch --array=0-$((N-1))%50 jobs/preprocess_signals_array.sh stages

# Or use helper (all datasets)
bash jobs/submit_all_datasets.sh array

# ============================================
# MONITORING
# ============================================

# Check job status
squeue -u $USER

# Check progress
./jobs/check_progress.sh

# Watch progress in real-time
watch -n 30 './jobs/check_progress.sh'

# Check specific job logs
tail -f logs/preprocess_stages_<jobid>.out

# Check for errors
grep -r "ERROR" logs/*.err

# ============================================
# TROUBLESHOOTING
# ============================================

# Job fails immediately
cat logs/preprocess_*.err | head -20

# Out of memory
sbatch --mem=128G jobs/preprocess_signals_parallel.sh <dataset>

# Too slow
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
