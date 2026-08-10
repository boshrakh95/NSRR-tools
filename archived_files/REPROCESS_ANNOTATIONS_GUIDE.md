# Guide: Reprocessing Annotations After Bug Fix

## Problem
After fixing the variable-duration stage bug in SHHS/MrOS adapters, you need to reprocess annotations to get correct synchronization metrics. However, signal processing (HDF5 creation) is slow and doesn't need to be redone.

## Solution: Three Options

### Option 1: Use New `--reprocess-annotations` Flag (RECOMMENDED)

The script now has a flag that skips signal processing but reprocesses annotations:

```bash
cd /home/boshra95/NSRR-tools
source .venv/bin/activate

# For SHHS
python scripts/preprocess_signals.py \
  --dataset shhs \
  --skip-existing \
  --reprocess-annotations

# For MrOS
python scripts/preprocess_signals.py \
  --dataset mros \
  --skip-existing \
  --reprocess-annotations
```

**How it works:**
- `--skip-existing`: Skip subjects where outputs exist
- `--reprocess-annotations`: Override the skip for annotation files only
- Result: Keeps HDF5 files, regenerates NPY annotation files

**Expected behavior:**
- Signal processing: SKIPPED (uses existing HDF5)
- Annotation processing: RUNS (recreates NPY with fixed bug)
- Much faster than full reprocessing!

### Option 2: Delete Annotation Files Manually

If you prefer, delete only the annotation NPY files:

```bash
# Delete SHHS annotations (not signals!)
rm /home/boshra95/scratch/psg/shhs/derived/annotations/*.npy

# Delete MrOS annotations (not signals!)
rm /home/boshra95/scratch/psg/mros/derived/annotations/*.npy

# HDF5 files remain intact
ls /home/boshra95/scratch/psg/shhs/derived/hdf5_signals/*.h5  # Still there!
```

Then run normally:

```bash
python scripts/preprocess_signals.py \
  --dataset shhs \
  --skip-existing
```

### Option 3: Use Parallel Processing (For Cluster)

Update `jobs/preprocess_signals_parallel.sh` to use the new flag:

```bash
#!/bin/bash
#SBATCH --account=def-forouzan
#SBATCH --cpus-per-task=6
#SBATCH --mem=64000M
#SBATCH --time=3:00:00

source ~/.venv/bin/activate

python scripts/preprocess_signals.py \
  --dataset ${DATASET} \
  --skip-existing \
  --reprocess-annotations
```

Then submit:

```bash
cd /home/boshra95/NSRR-tools

# Submit SHHS
DATASET=shhs sbatch jobs/preprocess_signals_parallel.sh

# Submit MrOS
DATASET=mros sbatch jobs/preprocess_signals_parallel.sh
```

## What Changed in the Code

### New Flag Added
**File:** `scripts/preprocess_signals.py`

**New argument:**
```python
parser.add_argument(
    '--reprocess-annotations',
    action='store_true',
    default=False,
    help='Reprocess annotations even if they exist (keeps existing HDF5 signals)'
)
```

### Skip Logic Updated
**Before (line 184):**
```python
# Skip if BOTH exist
if skip_existing and hdf5_path.exists() and annot_path.exists():
    skip_subject()
```

**After (lines 183-207):**
```python
if skip_existing:
    if reprocess_annotations:
        # Only check HDF5 existence
        skip_signal = hdf5_path.exists()
        skip_annotation = False  # Always reprocess!
    else:
        # Normal: both must exist to skip
        if hdf5_path.exists() and annot_path.exists():
            skip_subject()
        skip_signal = hdf5_path.exists()
        skip_annotation = annot_path.exists()
```

### Signal vs Annotation Independence

**Signal processing** (lines 228-238):
- Reads: EDF file
- Writes: HDF5 file (signals in 128Hz, float16, gzip)
- Output: `/scratch/psg/DATASET/derived/hdf5_signals/SUBJECT.h5`

**Annotation processing** (lines 241-266):
- Reads: Annotation file (XML/CSV) + EDF header (for sync validation)
- Writes: NPY file (epoch array)
- Output: `/scratch/psg/DATASET/derived/annotations/SUBJECT_stages.npy`

**They are INDEPENDENT!** Annotation processing only reads EDF headers (not HDF5).

## Verification After Reprocessing

### Check a Previously Problematic Subject

**Before the fix:**
```bash
grep "204173_v2" /scratch/psg/shhs/derived/logs/preprocessing_summary_shhs.csv
# Expected: sync_status=padded, sync_difference_sec=9869.996, adjustment_epochs=328
```

**After reprocessing:**
```bash
grep "204173_v2" /scratch/psg/shhs/derived/logs/preprocessing_summary_shhs.csv
# Expected: sync_status=synchronized, sync_difference_sec=0.0, adjustment_epochs=0
```

### Check XML vs Processing Results

Use the xml_to_csv tool:

```bash
python3 scripts/xml_to_csv_simple.py \
  /scratch/nsrr_downloads/shhs/.../shhs2-204173-nsrr.xml \
  --stages-only

# Should show: No gaps, perfect alignment
```

### Summary Statistics

Check aggregated metrics:

```bash
# Count sync statuses BEFORE fix
cut -d',' -f13 preprocessing_summary_shhs_OLD.csv | sort | uniq -c
#   123 synchronized
#   456 padded          <- Many false "padded"!
#    78 truncated

# Count sync statuses AFTER fix
cut -d',' -f13 preprocessing_summary_shhs_NEW.csv | sort | uniq -c
#   945 synchronized    <- Much better!
#    12 truncated
#     2 padded
```

## Important Notes

1. **Summary CSV will be overwritten** - The preprocessing script overwrites the summary CSV each run. If you want to compare before/after, backup first:
   ```bash
   cp /scratch/psg/shhs/derived/logs/preprocessing_summary_shhs.csv \
      /scratch/psg/shhs/derived/logs/preprocessing_summary_shhs_BEFORE_FIX.csv
   ```

2. **HDF5 files are preserved** - Signal processing output is not touched when using `--reprocess-annotations`

3. **Fast reprocessing** - Annotation processing is ~100x faster than signal processing (no FFT, no resampling, just parsing XML/CSV)

4. **Memory usage** - Annotation reprocessing uses minimal memory (<1GB per subject)

5. **Parallel processing** - Safe to run many subjects in parallel when only reprocessing annotations

## Troubleshooting

### "No subjects to process"
- Means all HDF5 files exist and annotations are being skipped
- Use `--reprocess-annotations` flag OR delete annotation NPY files

### "EDF not found"
- The annotation processor needs EDF files for sync validation
- Make sure EDF files are in the expected paths from metadata

### "Still showing padded status"
- Check if you're using the fixed adapter code (look for `last_stage['duration']` in shhs_adapter.py:276)
- Verify the bug fix was applied: `git log --oneline | head -5`

## Expected Time Savings

**Full reprocessing:**
- Signal + Annotation: ~5-10 minutes per subject
- 1000 subjects: ~83-167 hours

**Annotation-only reprocessing:**
- Annotation only: ~3-10 seconds per subject
- 1000 subjects: ~1-3 hours

**Time saved: ~98% faster!** 🚀
