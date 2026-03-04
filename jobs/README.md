# NSRR Signal Preprocessing Jobs for Compute Canada

Parallel and distributed processing scripts for converting NSRR EDF files to SleepFM HDF5 format on Compute Canada clusters.

## Quick Start

### 1. Verify Setup

Ensure metadata and virtual environment are ready:

```bash
# Check metadata exists
ls -lh /scratch/$USER/psg/unified/metadata/unified_metadata.parquet

# Check virtual environment
ls -lh /home/boshra95/NSRR-tools/.venv/bin/activate

# Create log directories
mkdir -p logs
```

### 2. Run Preprocessing

**Single Dataset:**
```bash
# Process all subjects
sbatch jobs/preprocess_signals_parallel.sh stages

# Process first 10 subjects (testing)
sbatch jobs/preprocess_signals_parallel.sh stages 10

# With debug logging
sbatch jobs/preprocess_signals_parallel.sh stages 10 --log-level DEBUG
```

**All Datasets:**
```bash
# Process all subjects in all datasets
bash jobs/submit_all_datasets.sh

# Process first 10 of each dataset
bash jobs/submit_all_datasets.sh 10
```

## Detailed Usage

### Command-Line Parameters

The main preprocessing script accepts these parameters:

```bash
sbatch jobs/preprocess_signals_parallel.sh <dataset> [max_subjects] [--no-skip-existing] [--log-level LEVEL] [--config PATH]
```

**Parameters:**
- `<dataset>` (required): Dataset to process
  - Options: `stages`, `shhs`, `apples`, `mros`, `all`
- `[max_subjects]` (optional): Limit number of subjects
  - Example: `10` for testing, `100` for partial run
  - Omit to process all subjects
- `[--no-skip-existing]` (optional): Reprocess existing files
  - Default: skips files that already exist
- `[--log-level LEVEL]` (optional): Logging verbosity
  - Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`
  - Default: `INFO`
- `[--config PATH]` (optional): Custom preprocessing config
  - Default: `configs/preprocessing_params.yaml`

### Examples

```bash
# Basic: process all STAGES subjects
sbatch jobs/preprocess_signals_parallel.sh stages

# Testing: first 10 subjects with debug logging
sbatch jobs/preprocess_signals_parallel.sh stages 10 --log-level DEBUG

# Reprocess: force reprocessing existing files
sbatch jobs/preprocess_signals_parallel.sh stages --no-skip-existing

# Custom: use custom configuration
sbatch jobs/preprocess_signals_parallel.sh stages --config my_config.yaml

# All datasets: process all with 100 subject limit per dataset
bash jobs/submit_all_datasets.sh 100

# All datasets with debug logging
bash jobs/submit_all_datasets.sh 10 --log-level DEBUG
```

### Resource Requirements

**Default Resources:**
- **CPUs:** 4 cores per job
- **Memory:** 16GB RAM
- **Time:** 12 hours max
- **Expected performance:**  ~1.5 min per subject
  - 100 subjects = ~2.5 hours
  - 500 subjects = ~12 hours

**Customize Resources:**
```bash
# More memory
sbatch --mem=32000M jobs/preprocess_signals_parallel.sh stages

# More time
sbatch --time=24:00:00 jobs/preprocess_signals_parallel.sh shhs

# Different account
sbatch --account=def-OTHERACCOUNT jobs/preprocess_signals_parallel.sh stages
```

## Monitoring Progress

### Check Job Status

```bash
# View all your jobs
squeue -u $USER

# View specific job details
scontrol show job <JOBID>

# Check job efficiency after completion
seff <JOBID>
```

### Check Processing Progress

```bash
# Run once
./jobs/check_progress.sh

# Watch in real-time (updates every 30 seconds)
watch -n 30 './jobs/check_progress.sh'

# Check specific dataset
./jobs/check_progress.sh stages
```

### View Logs

```bash
# Follow log for running job (replace JOBID)
tail -f logs/preprocess_stages_JOBID.out

# Check for errors
grep -i error logs/preprocess_*.err

# View summary
cat logs/preprocess_*.out | grep "SUMMARY"
```

## Output Structure

Processed files are saved to `/scratch/$USER/psg/<dataset>/derived/`:

```
/scratch/$USER/psg/
├── stages/
│   └── derived/
│       ├── hdf5_signals/          # Signal data in HDF5 format
│       │   ├── GSSA00001.h5
│       │   ├── GSSA00002.h5
│       │   └── ...
│       ├── annotations/           # Sleep stage annotations
│       │   ├── GSSA00001_stages.npy
│       │   ├── GSSA00002_stages.npy
│       │   └── ...
│       └── logs/                  # Processing summaries
│           └── preprocessing_summary_stages.csv
├── shhs/
│   └── derived/...
├── apples/
│   └── derived/...
└── mros/
    └── derived/...
```

### HDF5 File Contents

Each HDF5 file contains:
- **Signal data:** Multi-channel PSG signals (128 Hz, float16, gzip-compressed)
- **Metadata:** Channel names, sampling rate, subject ID, duration
- **Attributes:** Processing parameters, data version

Example inspection:
```bash
# List contents
h5ls /scratch/$USER/psg/stages/derived/hdf5_signals/GSSA00001.h5

# View details
h5dump -H /scratch/$USER/psg/stages/derived/hdf5_signals/GSSA00001.h5

# Validate with script
python scripts/validate_hdf5.py /scratch/$USER/psg/stages/derived/hdf5_signals/GSSA00001.h5
```

## Troubleshooting

### Common Issues

**1. Job fails immediately**
```bash
# Check error log
cat logs/preprocess_*.err | head -20

# Check if metadata exists
ls -lh /scratch/$USER/psg/unified/metadata/unified_metadata.parquet
```

**2. Out of memory errors**
```bash
# Increase memory allocation
sbatch --mem=32000M jobs/preprocess_signals_parallel.sh stages
```

**3. Job times out**
```bash
# Increase time limit
sbatch --time=24:00:00 jobs/preprocess_signals_parallel.sh shhs

# Or process in batches
sbatch jobs/preprocess_signals_parallel.sh shhs 500
# Wait for completion, then:
# (manually re-run with skip-existing to continue)
```

**4. Slow processing**
```bash
# Check if using too many channels
# Edit configs/preprocessing_params.yaml:
#   channel_selection:
#     strategy: minimal  # Reduces to 4 channels
```

**5. Python module not found**
```bash
# Ensure virtual environment is activated
cd /home/boshra95/NSRR-tools
source .venv/bin/activate
python -c "import nsrr_tools; print('OK')"
```

### Cancel Jobs

```bash
# Cancel all your jobs
scancel -u $USER

# Cancel specific job
scancel <JOBID>

# Cancel job array
scancel <ARRAY_JOB_ID>
```

## Validation

### Check Output Statistics

```bash
# Count processed subjects
find /scratch/$USER/psg/stages/derived/hdf5_signals -name "*.h5" | wc -l

# Check total size
du -sh /scratch/$USER/psg/stages/derived/hdf5_signals

# View processing summary
column -t -s, /scratch/$USER/psg/stages/derived/logs/preprocessing_summary_stages.csv | less -S
```

### Verify File Integrity

```bash
# Check specific file
h5ls /scratch/$USER/psg/stages/derived/hdf5_signals/GSSA00001.h5

# Validate file
python scripts/validate_hdf5.py /scratch/$USER/psg/stages/derived/hdf5_signals/GSSA00001.h5

# Batch validate (first 10 files)
for file in $(ls /scratch/$USER/psg/stages/derived/hdf5_signals/*.h5 | head -10); do
    echo "Validating $file..."
    python scripts/validate_hdf5.py "$file"
done
```

## Advanced Configuration

### Custom Preprocessing Parameters

Edit `configs/preprocessing_params.yaml` to customize:

```yaml
signal_processing:
  target_sampling_rate: 128  # Hz
  data_type: float16
  
channel_selection:
  strategy: standard  # or 'minimal', 'extended'
  
compression:
  method: gzip
  level: 4
  
filters:
  eeg:
    high_pass: 0.3
    low_pass: 35.0
  # ... more channel types
```

Then use custom config:
```bash
sbatch jobs/preprocess_signals_parallel.sh stages --config my_config.yaml
```

## Performance Tips

1. **Start with small test:**
   ```bash
   sbatch jobs/preprocess_signals_parallel.sh stages 10 --log-level DEBUG
   ```

2. **Check timing:**  
   - Expected: ~1.5 min per 10-hour recording
   - If slower, check channel selection strategy

3. **Monitor resources:**
   ```bash
   seff <JOBID>  # After job completes
   ```

4. **Process in batches** for very large datasets:
   ```bash
   sbatch jobs/preprocess_signals_parallel.sh shhs 500  # First 500
   # After completion, skip-existing handles the rest automatically
   sbatch jobs/preprocess_signals_parallel.sh shhs      # Continues from 501+
   ```

## Support

For issues:
1. Check logs: `logs/preprocess_*.err`
2. Verify setup: `./jobs/check_progress.sh`
3. Test with small batch: `sbatch jobs/preprocess_signals_parallel.sh stages 1 --log-level DEBUG`

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
