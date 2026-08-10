# STAGES ID Mismatch - Debug Guide

## The Problem

**EDF filenames** use site codes like: `GSSA00001`, `GSSA00008`, `GSSA00013`, etc.

**Metadata files** use different codes: `BOGN00001`, `BOGN00003`, `BOGN00004`, etc.

**PSG Key file** has correct IDs: `BOGN00001`, `BOGN00003`, etc. (matches metadata)

## Where the Mismatch is Detected

### 1. In `find_annotation_file()` - Line 186
**File**: `/home/boshra95/NSRR-tools/src/nsrr_tools/datasets/stages_adapter.py`

```python
def find_annotation_file(self, subject_id: str) -> Optional[Path]:
    # Line 181: Tries to find annotation using EDF-based subject_id (GSSA00001)
    usable_dir = original_path / subject_id / 'usable'  # Looks for GSSA00001/usable/
    
    if not usable_dir.exists():
        # Line 186: THIS IS WHERE THE ERROR IS LOGGED
        logger.warning(f"Usable directory not found for subject {subject_id}")
        return None
```

**What happens**: When subject_id is `GSSA00001`, it looks for `/scratch/.../original/GSSA00001/usable/` but the actual directory is probably `/scratch/.../original/BOGN00001/usable/`

### 2. In `extract_subject_metadata()` - Line 387
**File**: `/home/boshra95/NSRR-tools/src/nsrr_tools/datasets/stages_adapter.py`

```python
def extract_subject_metadata(self, subject_id: str, metadata_df: pd.DataFrame) -> Dict[str, Any]:
    # Line 384: Looks for subject_id (GSSA00001) in metadata
    subject_row = metadata_df[metadata_df[self.subject_id_col] == subject_id]
    
    if len(subject_row) == 0:
        # Line 387: THIS IS WHERE THE MISMATCH IS DETECTED
        logger.warning(f"Subject {subject_id} not found in metadata")
        return {'subject_id': subject_id, 'found': False}
```

**What happens**: Metadata has `subject_code = BOGN00001` but we're searching for `GSSA00001`, so no match is found.

### 3. In test output - Line ~50 of test script
**File**: `/home/boshra95/NSRR-tools/scripts/test_stages_adapter.py`

When the test runs, you see:
```
Subject: GSSA00001
WARNING: No annotation found
Extracting subject metadata...
  Found: False
```

## The Missing Link: PSG Key File

**Location**: `/scratch/boshra95/psg/nsrr/stages/original/De-identified Data/PSG SRBD Variables/STAGESPSGKeySRBDVariables2020-08-29 Deidentified.xlsx`

**Contents**:
- Column `s_code`: Has the CORRECT IDs (BOGN00001, BOGN00003, etc.)
- Column `ahi`: Has the AHI values
- Shape: 1687 subjects, 19 columns
- Other columns: sleep_time, sex, age, bmi, n_obs, n_cen, n_mix, etc.

**This file should be**:
1. Added to `metadata_files` dict in stages_adapter.py
2. Merged with the other metadata CSVs
3. Used to get AHI values

## What You Need to Debug

### Find the Mapping File
There must be a file that maps EDF filenames (GSSA00001) to metadata IDs (BOGN00001).

**Places to check**:
```bash
# Check for any mapping file
find /scratch/boshra95/psg/nsrr/stages -name "*map*" -o -name "*link*" -o -name "*key*"

# Check if EDF headers contain the BOGN ID
# You can use MNE to read the patient ID from EDF header

# Check the De-identified Data directory
ls -la "/scratch/boshra95/psg/nsrr/stages/original/De-identified Data/"

# Check if BOGN directories exist in original
ls -la /scratch/boshra95/psg/nsrr/stages/original/ | grep BOGN

# Check if GSSA directories exist
ls -la /scratch/boshra95/psg/nsrr/stages/original/ | grep GSSA
```

### Option 1: EDF Headers Contain Mapping
The EDF file header might contain the subject ID. You can read it with MNE:

```python
import mne
edf_path = "/scratch/boshra95/psg/nsrr/stages/sample_extraction/STAGES PSGs/GSSA/GSSA00001.edf"
raw = mne.io.read_raw_edf(edf_path, preload=False)
print("Patient ID:", raw.info.get('subject_info'))
print("Recording ID:", raw.info.get('meas_id'))
```

### Option 2: Filename Pattern Decoding
Maybe there's a pattern: GSSA + number = BOGN + number?
- GSSA00001 → BOGN00001?
- Check if the numbers correspond

### Option 3: Separate Mapping File
There might be a CSV/Excel file with columns [edf_filename, subject_code]

## Quick Fix Options

### Temporary: Use PSG Key File s_code
If the PSG key file has all the subjects you care about, you could:
1. Load the xlsx file in stages_adapter
2. Use `s_code` column as the subject IDs
3. This gives you the BOGN codes which match metadata

### Permanent: Find the Real Mapping
The real solution is finding how GSSA codes map to BOGN codes.

## How to Debug Yourself

### 1. Add print statements in stages_adapter.py

Around **line 78-90** in `find_edf_files()`:
```python
for edf_path in sample_edfs:
    subject_id = self._extract_base_subject_id(edf_path.stem)
    print(f"DEBUG: EDF filename: {edf_path.name}, extracted ID: {subject_id}")
    edf_files.append((subject_id, edf_path))
```

Around **line 384-387** in `extract_subject_metadata()`:
```python
print(f"DEBUG: Looking for subject_id '{subject_id}' in metadata")
print(f"DEBUG: Available subject_codes (first 5): {metadata_df[self.subject_id_col].head().tolist()}")
subject_row = metadata_df[metadata_df[self.subject_id_col] == subject_id]
```

### 2. Check EDF headers directly

Run this in Python:
```python
import mne
edf_path = "/scratch/boshra95/psg/nsrr/stages/sample_extraction/STAGES PSGs/GSSA/GSSA00001.edf"
raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

print("EDF Info:")
print(f"  Filename: {edf_path}")
print(f"  Subject info: {raw.info.get('subject_info')}")
print(f"  Description: {raw.info.get('description')}")
print(f"  Device info: {raw.info.get('device_info')}")

# Check the raw EDF header
with open(edf_path, 'rb') as f:
    header = f.read(256)  # EDF header is 256 bytes
    patient_id = header[8:88].decode('ascii').strip()  # Bytes 8-87: patient ID
    print(f"  Patient ID from header: '{patient_id}'")
```

### 3. Check directory structure

```bash
# See what subject directories exist
ls /scratch/boshra95/psg/nsrr/stages/original/ | head -20

# Check if sample directory has different structure
ls "/scratch/boshra95/psg/nsrr/stages/sample_extraction/STAGES PSGs/"
```

### 4. Load PSG Key xlsx and check

```python
import pandas as pd
df = pd.read_excel('/scratch/boshra95/psg/nsrr/stages/original/De-identified Data/PSG SRBD Variables/STAGESPSGKeySRBDVariables2020-08-29 Deidentified.xlsx')
print(df[['s_code', 'ahi', 'sleep_time']].head(20))

# Check if there are any GSSA codes
print("Any GSSA codes?", df['s_code'].str.contains('GSSA').sum())
print("Any BOGN codes?", df['s_code'].str.contains('BOGN').sum())
```

## Files to Modify After Finding Solution

Once you find the mapping, update:

1. **`stages_adapter.py`** - Add mapping logic in `find_edf_files()` or create a `_map_edf_to_metadata_id()` method

2. **`stages_adapter.py`** - Add PSG key file to `metadata_files`:
```python
self.metadata_files = {
    'main': 'stages-dataset-0.3.0.csv',
    'harmonized': 'stages-harmonized-dataset-0.3.0.csv',
    'psg_key': 'STAGESPSGKeySRBDVariables2020-08-29 Deidentified.xlsx',  # ADD THIS
}
```

3. **`stages_adapter.py`** - Update `load_metadata()` to handle xlsx files and merge on `s_code`

## Summary

**ID Mismatch Detected At**:
- Line 186: `find_annotation_file()` - Can't find usable directory
- Line 387: `extract_subject_metadata()` - Can't find subject in metadata

**Root Cause**: EDF filenames (GSSA) don't match metadata subject_codes (BOGN)

**Next Steps**: 
1. Check EDF headers for hidden BOGN ID
2. Look for mapping file in De-identified Data directory
3. Check if GSSA/BOGN codes follow a pattern
4. Add PSG key xlsx file to metadata loading

**The xlsx file IS there** (you were right!):
`/scratch/boshra95/psg/nsrr/stages/original/De-identified Data/PSG SRBD Variables/STAGESPSGKeySRBDVariables2020-08-29 Deidentified.xlsx`
