# NSRR Signal Preprocessing Jobs for Compute Canada

Parallel and distributed processing scripts for converting NSRR EDF files to SleepFM HDF5 format on Compute Canada clusters.

## Quick Start

### 1. Update Configuration

Edit job scripts and replace `def-YOUR_ACCOUNT` with your Compute Canada allocation:

```bash
# Edit these files:
vim jobs/preprocess_signals_parallel.sh  # Line 3
vim jobs/preprocess_signals_array.sh     # Line 3
```

### 2. Choose Processing Mode

**Option A: Parallel Processing** (Recommended for moderate datasets)
- Processes multiple subjects in parallel on a single node
- Good for: 100-1000 subjects per dataset
- Uses GNU parallel with 16 cores

```bash
# Submit single dataset
sbatch jobs/preprocess_signals_parallel.sh stages

# Or all datasets at once
bash jobs/submit_all_datasets.sh parallel
```

**Option B: Array Jobs** (For massive datasets)
- Each subject gets its own SLURM task
- Good for: 1000+ subjects, can scale to 10,000+
- Tasks run across multiple nodes

```bash
# Generate subject lists
python scripts/generate_subject_list.py --dataset stages --output jobs/subject_lists/stages_subjects.txt

# Submit array job  
sbatch --array=0-1669%50 jobs/preprocess_signals_array.sh stages

# Or all datasets
bash jobs/submit_all_datasets.sh array
```

## Detailed Usage

### Parallel Processing Mode

**Single Dataset:**
```bash
# Basic usage
sbatch jobs/preprocess_signals_parallel.sh stages

# With limited subjects (testing)
sbatch jobs/preprocess_signals_parallel.sh stages 10

# Custom resources
sbatch --cpus-per-task=32 --mem=128G jobs/preprocess_signals_parallel.sh shhs
```

**All Datasets:**
```bash
bash jobs/submit_all_datasets.sh parallel
```

**Features:**
- 16 concurrent workers by default
- Automatic subject list generation from metadata
- GNU parallel for efficient CPU usage
- Single log file per dataset

**Resource Requirements:**
- CPUs: 16 (configurable with `--cpus-per-task`)
- Memory: 64GB (16GB for data + 48GB for parallel processing)
- Time: 12 hours (sufficient for 500-1000 subjects)

### Array Job Mode

**Step 1: Generate Subject Lists**
```bash
# Single dataset
python scripts/generate_subject_list.py \
    --dataset stages \
    --output jobs/subject_lists/stages_subjects.txt

# With annotations only
python scripts/generate_subject_list.py \
    --dataset stages \
    --require-annotations \
    --output jobs/subject_lists/stages_annotated.txt

# Limited for testing
python scripts/generate_subject_list.py \
    --dataset stages \
    --max-subjects 100 \
    --output jobs/subject_lists/stages_test.txt
```

**Step 2: Submit Array Job**
```bash
# Check subject count
N_SUBJECTS=$(wc -l < jobs/subject_lists/stages_subjects.txt)
MAX_ARRAY=$((N_SUBJECTS - 1))

# Submit with throttling (max 50 concurrent tasks)
sbatch --array=0-${MAX_ARRAY}%50 jobs/preprocess_signals_array.sh stages

# Or use the helper script
bash jobs/submit_all_datasets.sh array
```

**Features:**
- One SLURM task per subject
- Scales across multiple nodes
- Individual log per task
- Easy to restart failed tasks

**Resource Requirements per Task:**
- CPUs: 4
- Memory: 16GB (4GB per CPU)
- Time: 2 hours (sufficient for most subjects)

**Managing Array Jobs:**
```bash
# Check status
squeue -u $USER

# Cancel all tasks
scancel <job_id>

# Cancel specific tasks
scancel <job_id>_[100-200]

# Check failed tasks
grep "exit code" logs/array/*.err | grep -v "exit code: 0"

# Resubmit failed tasks only
# ... create new subject list with failed subjects ...
sbatch --array=<failed_task_ids> jobs/preprocess_signals_array.sh stages
```

## Monitoring

### Check Job Status
```bash
# All jobs
squeue -u $USER

# Specific job
squeue -j <job_id>

# Job details
scontrol show job <job_id>
```

### Check Logs
```bash
# Parallel mode
tail -f logs/preprocess_preprocess_signals_<jobid>.out

# Array mode - find failed tasks
grep -l "ERROR\|Failed" logs/array/*.err

# Count completed vs failed
echo "Success: $(grep -l "Completed" logs/array/*.out | wc -l)"
echo "Failed:  $(grep -l "Failed\|ERROR" logs/array/*.err | wc -l)"
```

### Check Output Files
```bash
# Count processed files
find /scratch/$USER/psg/stages/derived/hdf5_signals -name "*.h5" | wc -l

# Check file sizes
du -sh /scratch/$USER/psg/*/derived/hdf5_signals

# Validate HDF5 files
python -c "
import h5py
from pathlib import Path
from tqdm import tqdm

h5_dir = Path('/scratch/$USER/psg/stages/derived/hdf5_signals')
files = list(h5_dir.glob('*.h5'))

corrupted = []
for h5_file in tqdm(files):
    try:
        with h5py.File(h5_file, 'r') as f:
            pass
    except Exception as e:
        corrupted.append(h5_file)
        
print(f'Corrupted files: {len(corrupted)}/{len(files)}')
if corrupted:
    for f in corrupted[:10]:
        print(f'  {f.name}')
"
```

## Resource Optimization

### Adjust for Dataset Size

**Small datasets (STAGES: ~1700 subjects):**
```bash
# Parallel mode is efficient
sbatch --cpus-per-task=16 --mem=64G --time=8:00:00 \
    jobs/preprocess_signals_parallel.sh stages
```

**Large datasets (SHHS: ~8400 subjects):**
```bash
# Array job is better
# Process in batches of 2000
python scripts/generate_subject_list.py --dataset shhs --max-subjects 2000 --output jobs/subject_lists/shhs_batch1.txt
sbatch --array=0-1999%50 jobs/preprocess_signals_array.sh shhs
```

**For quick testing:**
```bash
# Process just 10 subjects
sbatch jobs/preprocess_signals_parallel.sh stages 10
```

### Memory Requirements

Typical memory usage per subject:
- EDF loading: 0.5-2 GB
- Processing: 1-3 GB  
- Peak: 2-5 GB

Safe allocations:
- **Parallel mode**: 64GB for 16 workers (4GB per worker)
- **Array mode**: 16GB per task (4 cores × 4GB)

If you get OOM errors:
```bash
# Increase memory
sbatch --mem=128G jobs/preprocess_signals_parallel.sh shhs

# Or reduce workers
sbatch --cpus-per-task=8 --mem=32G jobs/preprocess_signals_parallel.sh shhs
```

## Troubleshooting

### Job Fails Immediately
```bash
# Check error log
cat logs/preprocess_*.err

# Common issues:
# 1. Account name not set: Edit #SBATCH --account=...
# 2. Metadata not found: Run metadata extraction first
# 3. Environment issues: Check module load commands
```

### Out of Memory
```bash
# Increase memory allocation
sbatch --mem=128G jobs/preprocess_signals_parallel.sh <dataset>

# Or reduce parallel workers
sbatch --cpus-per-task=8 jobs/preprocess_signals_parallel.sh <dataset>
```

### Slow Processing
```bash
# Check if preloading is working (should see "Preloading N channels")
grep "Preloading" logs/*.out

# Check filter method (should use MNE FFT)
grep "filter" logs/*.out

# If slow, try minimal channel strategy
# Edit configs/preprocessing_params.yaml:
#   channel_selection:
#     strategy: minimal  # Only 4 channels
```

### Some Subjects Fail
```bash
# Find failed subjects
grep "ERROR\|Failed" logs/*.out | grep -oP "(?<=subjects: )[^ ]+"

# Check specific error
grep "<subject_id>" logs/*.out

# Skip and continue
# The scripts already use --skip-existing, so just rerun
```

## Expected Performance

### Processing Speed (with optimizations)
- **Per subject**: 1-3 minutes (depending on recording length)
- **10 hours of data**: ~1.5 minutes
- **Speed ratio**: 3-4x realtime

### Dataset Completion Times

**Parallel Mode (16 cores):**
- STAGES (1700 subjects): ~3-4 hours
- APPLES (1100 subjects): ~2-3 hours  
- MrOS (2900 subjects): ~6-8 hours
- SHHS (8400 subjects): ~15-20 hours

**Array Mode (50 concurrent tasks, 4 cores each = 200 cores):**
- STAGES: ~1 hour
- APPLES: ~40 minutes
- MrOS: ~2 hours
- SHHS: ~4-5 hours

### Output Size
Expected HDF5 file sizes (with float16 compression):
- 8-10 hour recording: 15-25 MB
- Total per dataset:
  - STAGES: ~30 GB
  - APPLES: ~20 GB
  - MrOS: ~50 GB
  - SHHS: ~150 GB

## Files Created

```
NSRR-tools/
├── jobs/
│   ├── preprocess_signals_parallel.sh  # Main parallel processing script
│   ├── preprocess_signals_array.sh     # Array job script
│   ├── submit_all_datasets.sh          # Helper to submit all
│   └── subject_lists/                  # Generated subject lists
│       ├── stages_subjects.txt
│       ├── shhs_subjects.txt
│       ├── apples_subjects.txt
│       └── mros_subjects.txt
├── scripts/
│   ├── preprocess_single_subject.py    # Single subject processor
│   └── generate_subject_list.py        # Subject list generator
└── logs/
    ├── preprocess_*.out                # Parallel job logs
    ├── preprocess_*.err
    └── array/
        ├── preprocess_<jobid>_<taskid>.out
        └── preprocess_<jobid>_<taskid>.err
```

## Next Steps

After preprocessing completes:

1. **Validate outputs**:
   ```bash
   python scripts/validate_preprocessing.py --dataset stages
   ```

2. **Generate train/val/test splits**:
   ```bash
   python scripts/generate_splits.py
   ```

3. **Start training** (if using SleepFM):
   ```bash
   python train_sleepfm.py --config configs/sleepfm_config.yaml
   ```
