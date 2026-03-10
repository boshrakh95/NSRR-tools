# Target Extraction Implementation Plan

**Date:** March 10, 2026  
**Scope:** Tier 1 + Tier 2 + rested_morning (Tier 3)  
**Goal:** Extract classification targets from CSV files and create efficient storage structure

---

## Task List

### Tier 1 (Priority: CRITICAL)
1. ✅ **apnea_binary** - All 4 datasets (~13,900 subjects)
2. ✅ **depression_binary** - STAGES + APPLES (~2,600 subjects)
3. ✅ **sleepiness_binary** - APPLES (~1,100 subjects)

### Tier 2 (Priority: HIGH)
4. ✅ **anxiety_binary** - STAGES (~1,513 subjects)
5. ✅ **insomnia_binary** - STAGES + MrOS (~4,400 subjects)
6. ✅ **cvd_binary** - SHHS (~8,444 subjects)

### Tier 3 (Priority: MEDIUM - after Tier 2)
7. ✅ **rested_morning** - STAGES, SHHS, MrOS (variable, subjective)

---

## Storage Architecture

### Proposed Structure

```
/scratch/boshra95/psg/
├── unified/
│   ├── metadata/
│   │   ├── unified_metadata.parquet           # ✅ Existing
│   │   ├── unified_targets.parquet            # 🆕 Master targets file (all datasets)
│   │   └── target_extraction.log              # 🆕 Extraction logs
│   └── targets/
│       ├── stages_targets.csv                 # 🆕 Per-dataset files
│       ├── shhs_targets.csv
│       ├── apples_targets.csv
│       ├── mros_targets.csv
│       ├── task_subject_lists.json            # 🆕 Fast lookup: task -> list of subject IDs
│       ├── task_statistics.json               # 🆕 Class distributions per task
│       └── README.md                          # Documentation
```

### File Formats

#### 1. Per-Dataset CSV Files (e.g., `stages_targets.csv`)

**Columns:**
```python
{
    'subject_id': str,              # Original subject ID
    'dataset': str,                 # 'stages' (constant in file)
    'visit': int,                   # Visit number (0 for single-visit datasets)
    
    # Binary targets (0, 1, or empty string '' for missing)
    'apnea_binary': str,
    'depression_binary': str,
    'sleepiness_binary': str,
    'anxiety_binary': str,
    'insomnia_binary': str,
    'cvd_binary': str,
    'rested_morning': str,
    
    # Continuous scores (numeric or empty string '' for missing)
    'ahi_score': str,
    'phq9_score': str,
    'bdi_score': str,
    'gad7_score': str,
    'ess_score': str,
    'isi_score': str,
    'fss_score': str,
    
    # Source tracking
    'extraction_date': str,         # ISO format date
    'source_files': str,            # JSON string of {score_name: csv_file}
}
```

**Why CSV with empty strings:**
- Easy to read with pandas: `pd.read_csv()` treats empty as NaN
- Human-readable for inspection
- Easy to join with other CSVs
- Small file size

#### 2. Master Parquet File (`unified_targets.parquet`)

**Schema:**
```python
{
    'unified_id': str,              # {dataset}_{subject_id}_{visit}
    'dataset': str,
    'subject_id': str,
    'visit': int,
    
    # Binary targets (0, 1, or -1 for missing)
    'apnea_binary': int,
    'depression_binary': int,
    'sleepiness_binary': int,
    'anxiety_binary': int,
    'insomnia_binary': int,
    'cvd_binary': int,
    'rested_morning': int,
    
    # Continuous scores (float or NaN)
    'ahi_score': float,
    'phq9_score': float,
    'bdi_score': float,
    'gad7_score': float,
    'ess_score': float,
    'isi_score': float,
    'fss_score': float,
    
    # Metadata
    'extraction_date': datetime,
}
```

**Why Parquet:**
- Fast loading for training
- Efficient storage with compression
- Type-safe (int vs float vs string)
- Easy joins with unified_metadata.parquet

#### 3. Task Subject Lists (`task_subject_lists.json`)

**Format:**
```json
{
  "apnea_binary": {
    "stages": ["BOGN00004", "BOGN00007", ...],
    "shhs": [1, 2, 3, ...],
    "apples": [100001, 100002, ...],
    "mros": [1001, 1002, ...]
  },
  "depression_binary": {
    "stages": ["BOGN00004", ...],
    "apples": [100001, ...]
  },
  "sleepiness_binary": {
    "apples": [100001, ...]
  },
  ...
}
```

**Why JSON:**
- Quick filtering: "Give me all APPLES subjects with apnea labels"
- Small file size (just lists of IDs)
- Easy to load in Python/PyTorch dataloaders

#### 4. Task Statistics (`task_statistics.json`)

**Format:**
```json
{
  "apnea_binary": {
    "total": 13968,
    "by_dataset": {
      "stages": {"total": 1513, "positive": 621, "negative": 892, "prevalence": 0.41},
      "shhs": {"total": 8444, "positive": 2533, "negative": 5911, "prevalence": 0.30},
      "apples": {"total": 1104, "positive": 442, "negative": 662, "prevalence": 0.40},
      "mros": {"total": 2907, "positive": 1163, "negative": 1744, "prevalence": 0.40}
    }
  },
  ...
}
```

**Why Statistics:**
- Quick sanity checks during experiments
- No need to reload data to check class balance
- Track extraction quality

---

## Implementation Steps

### Phase 1: Setup and Data Discovery (Step 1-2)

#### Step 1: Create Directory Structure
```bash
mkdir -p /scratch/boshra95/psg/unified/targets
```

**Script:** `scripts/setup_target_directories.sh`

---

#### Step 2: CSV File Verification
**Script:** `scripts/verify_csv_files.py`

**Purpose:** Check that all required CSV files exist and have expected columns

**Tasks:**
1. Verify STAGES files:
   - `stages-dataset-0.3.0.csv` (phq_1000, gad_0800, fss_1000, ess_0900?)
   - `stages-harmonized-dataset-0.3.0.csv`
   
2. Verify APPLES files:
   - `apples-dataset-0.1.0.csv` (bditotalscore, esstotalscoreqc, mmsetotalscore)
   - `apples-harmonized-dataset-0.1.0.csv` (nsrr_ahi_chicago1999)
   
3. Verify SHHS files:
   - `shhs1-dataset-0.21.0.csv` (rdi3p, rest10)
   - `shhs2-dataset-0.21.0.csv` (rdi3p, ms204c)
   - `shhs-cvd-summary-dataset-0.21.0.csv` (any_cvd)
   
4. Verify MrOS files:
   - `mros-visit1-harmonized-0.6.0.csv` (nsrr_ahi_hp3r_aasm15)
   - `mros-visit1-dataset-0.6.0.csv` (slisiscr, poxqual3, epepwort)
   - `mros-visit2-dataset-0.6.0.csv` (slisiscr, poxqual3)

**Output:** Log file listing all found/missing columns

**Questions for User:**
1. Are there external XLSX files for STAGES ISI? Where are they located?
2. Should we extract STAGES AHI from XML now or skip initially?

---

### Phase 2: Dataset-by-Dataset Extraction (Step 3-6)

#### Step 3: Extract APPLES Targets (EASIEST - START HERE)
**Script:** `scripts/extract_targets_apples.py`

**Extraction Details:**

**a) Apnea Binary (apnea_binary)**
```python
# From: apples-harmonized-dataset-0.1.0.csv
# Column: nsrr_ahi_chicago1999
# Filter: visitn == 3 (DX visit - when PSG was done)
# Threshold: >= 15 → 1, < 15 → 0, missing/NaN → ''

df = pd.read_csv('apples-harmonized-dataset-0.1.0.csv')
df_dx = df[df['visitn'] == 3]  # Diagnostic visit only

df_dx['apnea_binary'] = df_dx['nsrr_ahi_chicago1999'].apply(
    lambda x: 1 if x >= 15 else 0 if pd.notna(x) else ''
)
df_dx['ahi_score'] = df_dx['nsrr_ahi_chicago1999']
```

**b) Depression Binary (depression_binary)**
```python
# From: apples-dataset-0.1.0.csv
# Column: bditotalscore
# Filter: visitn == 1 (Baseline visit)
# Threshold: >= 14 → 1, < 14 → 0, missing/NaN → ''
# Alternative: >= 20 for moderate depression

df = pd.read_csv('apples-dataset-0.1.0.csv')
df_bl = df[df['visitn'] == 1]  # Baseline visit

df_bl['depression_binary'] = df_bl['bditotalscore'].apply(
    lambda x: 1 if x >= 14 else 0 if pd.notna(x) else ''
)
df_bl['bdi_score'] = df_bl['bditotalscore']
```

**c) Sleepiness Binary (sleepiness_binary)**
```python
# From: apples-dataset-0.1.0.csv
# Column: esstotalscoreqc
# Filter: visitn == 3 (DX visit - PSG-matched)
# Threshold: >= 11 → 1, < 11 → 0, missing/NaN → ''

df_dx = df[df['visitn'] == 3]

df_dx['sleepiness_binary'] = df_dx['esstotalscoreqc'].apply(
    lambda x: 1 if x >= 11 else 0 if pd.notna(x) else ''
)
df_dx['ess_score'] = df_dx['esstotalscoreqc']
```

**Join Strategy:**
```python
# Merge baseline (BL) and diagnostic (DX) visit data on subject_id
# Most subjects should have both visits, but handle missing gracefully

targets_bl = df_bl[['nsrrid', 'depression_binary', 'bdi_score']]
targets_dx = df_dx[['nsrrid', 'apnea_binary', 'ahi_score', 'sleepiness_binary', 'ess_score']]

targets = targets_bl.merge(targets_dx, on='nsrrid', how='outer')
targets.rename(columns={'nsrrid': 'subject_id'}, inplace=True)
targets['dataset'] = 'apples'
targets['visit'] = 0  # Collapsed across visits
```

**Output:** `apples_targets.csv`

**Validation Checks:**
- Total subjects: ~1,104
- Apnea prevalence: ~35-45%
- Depression prevalence: ~15-25% (at threshold ≥14)
- Sleepiness prevalence: ~25-35%

---

#### Step 4: Extract SHHS Targets
**Script:** `scripts/extract_targets_shhs.py`

**Extraction Details:**

**a) Apnea Binary (apnea_binary)**
```python
# From: shhs1-dataset-0.21.0.csv OR shhs2-dataset-0.21.0.csv
# Column: rdi3p (Respiratory Disturbance Index with ≥3% desaturation)
# Strategy: Extract both visits separately
# Threshold: >= 15 → 1, < 15 → 0, missing/NaN → ''

# Visit 1
df1 = pd.read_csv('shhs1-dataset-0.21.0.csv')
df1['apnea_binary'] = df1['rdi3p'].apply(lambda x: 1 if x >= 15 else 0 if pd.notna(x) else '')
df1['ahi_score'] = df1['rdi3p']
df1['visit'] = 1

# Visit 2
df2 = pd.read_csv('shhs2-dataset-0.21.0.csv')
df2['apnea_binary'] = df2['rdi3p'].apply(lambda x: 1 if x >= 15 else 0 if pd.notna(x) else '')
df2['ahi_score'] = df2['rdi3p']
df2['visit'] = 2

# Combine (keep both visits as separate samples)
targets = pd.concat([df1, df2], ignore_index=True)
```

**b) CVD Binary (cvd_binary)**
```python
# From: shhs-cvd-summary-dataset-0.21.0.csv
# Column: any_cvd (incident CVD during follow-up)
# Threshold: == 1 → 1, == 0 → 0, missing/NaN → ''
# Note: This is a subject-level outcome, not visit-specific

df_cvd = pd.read_csv('shhs-cvd-summary-dataset-0.21.0.csv')
df_cvd['cvd_binary'] = df_cvd['any_cvd'].apply(lambda x: 1 if x == 1 else 0 if pd.notna(x) else '')

# Merge with visit data
targets = targets.merge(df_cvd[['nsrrid', 'cvd_binary']], on='nsrrid', how='left')
```

**c) Rested Morning (rested_morning) - LATER**
```python
# Visit 1: rest10 (1-5 scale)
# Visit 2: ms204c (1-5 scale)
# Mapping: >= 4 → 1 (good), <= 2 → 0 (poor), 3 or missing → ''

def map_rested(score):
    if pd.isna(score):
        return ''
    elif score >= 4:
        return 1
    elif score <= 2:
        return 0
    else:
        return ''  # Middle value = ambiguous

df1['rested_morning'] = df1['rest10'].apply(map_rested)
df2['rested_morning'] = df2['ms204c'].apply(map_rested)
```

**Output:** `shhs_targets.csv`

**Validation Checks:**
- Total records: ~8,444 visit 1 + ~4,000 visit 2 = ~12,000+ (many subjects in both)
- Apnea prevalence: ~25-35%
- CVD prevalence: ~10-20%

**Question for User:**
- Should we create separate records for each visit, or use only Visit 1 (baseline)?
- **Recommendation:** Keep both visits as separate samples for maximum data

---

#### Step 5: Extract MrOS Targets
**Script:** `scripts/extract_targets_mros.py`

**Extraction Details:**

**a) Apnea Binary (apnea_binary)**
```python
# From: mros-visit1-harmonized-0.6.0.csv
# Column: nsrr_ahi_hp3r_aasm15
# Threshold: >= 15 → 1, < 15 → 0, missing/NaN → ''

df1 = pd.read_csv('mros-visit1-harmonized-0.6.0.csv')
df1['apnea_binary'] = df1['nsrr_ahi_hp3r_aasm15'].apply(
    lambda x: 1 if x >= 15 else 0 if pd.notna(x) else ''
)
df1['ahi_score'] = df1['nsrr_ahi_hp3r_aasm15']
df1['visit'] = 1

# Visit 2 (if has AHI - check)
df2 = pd.read_csv('mros-visit2-dataset-0.6.0.csv')
# Check if visit 2 has AHI...
```

**b) Insomnia Binary (insomnia_binary)**
```python
# From: mros-visit1-dataset-0.6.0.csv
# Column: slisiscr (ISI total score)
# Threshold: >= 15 → 1, < 15 → 0, missing/NaN → ''

df1_main = pd.read_csv('mros-visit1-dataset-0.6.0.csv')
df1_main['insomnia_binary'] = df1_main['slisiscr'].apply(
    lambda x: 1 if x >= 15 else 0 if pd.notna(x) else ''
)
df1_main['isi_score'] = df1_main['slisiscr']

# Merge with harmonized data
df1 = df1.merge(df1_main[['nsrrid', 'insomnia_binary', 'isi_score']], on='nsrrid', how='left')
```

**c) Sleepiness Binary (sleepiness_binary)**
```python
# From: mros-visit1-dataset-0.6.0.csv
# Column: epepwort (ESS total)
# Threshold: >= 11 → 1, < 11 → 0, missing/NaN → ''

df1_main['sleepiness_binary'] = df1_main['epepwort'].apply(
    lambda x: 1 if x >= 11 else 0 if pd.notna(x) else ''
)
df1_main['ess_score'] = df1_main['epepwort']
```

**d) Rested Morning (rested_morning) - LATER**
```python
# From: mros-visit1-dataset-0.6.0.csv
# Column: poxqual3 (1-5 scale)
# Mapping: >= 4 → 1, <= 2 → 0, else → ''

df1_main['rested_morning'] = df1_main['poxqual3'].apply(map_rested)
```

**Output:** `mros_targets.csv`

**Validation Checks:**
- Total subjects: ~2,907 (Visit 1)
- Apnea prevalence: ~35-45%
- Insomnia prevalence: ~20-30%
- Sleepiness prevalence: ~25-35%

---

#### Step 6: Extract STAGES Targets
**Script:** `scripts/extract_targets_stages.py`

**Extraction Details:**

**a) Depression Binary (depression_binary)**
```python
# From: stages-dataset-0.3.0.csv
# Column: phq_1000 (PHQ-9 total)
# Threshold: >= 10 → 1, < 10 → 0, missing/NaN → ''

df = pd.read_csv('stages-dataset-0.3.0.csv')
df['depression_binary'] = df['phq_1000'].apply(
    lambda x: 1 if x >= 10 else 0 if pd.notna(x) else ''
)
df['phq9_score'] = df['phq_1000']
```

**b) Anxiety Binary (anxiety_binary)**
```python
# From: stages-dataset-0.3.0.csv
# Column: gad_0800 (GAD-7 total)
# Threshold: >= 10 → 1, < 10 → 0, missing/NaN → ''

df['anxiety_binary'] = df['gad_0800'].apply(
    lambda x: 1 if x >= 10 else 0 if pd.notna(x) else ''
)
df['gad7_score'] = df['gad_0800']
```

**c) Fatigue Binary (fatigue_binary) - OPTIONAL FOR NOW**
```python
# From: stages-dataset-0.3.0.csv
# Column: fss_1000 (FSS total, 9-63 range)
# Threshold: >= 36 → 1, < 36 → 0, missing/NaN → ''
# Note: Some datasets use mean >= 4.0 instead

df['fatigue_binary'] = df['fss_1000'].apply(
    lambda x: 1 if x >= 36 else 0 if pd.notna(x) else ''
)
df['fss_score'] = df['fss_1000'] / 9.0  # Convert to mean
```

**d) Sleepiness Binary (sleepiness_binary) - IF AVAILABLE**
```python
# From: stages-dataset-0.3.0.csv
# Column: ess_0900 (ESS total) - CHECK IF EXISTS
# Threshold: >= 11 → 1, < 11 → 0, missing/NaN → ''

if 'ess_0900' in df.columns:
    df['sleepiness_binary'] = df['ess_0900'].apply(
        lambda x: 1 if x >= 11 else 0 if pd.notna(x) else ''
    )
    df['ess_score'] = df['ess_0900']
```

**e) Insomnia Binary (insomnia_binary) - EXTERNAL FILE**
```python
# From: STAGES ASQ ISI to DIET 20200513 Final deidentified.xlsx
# Column: isi_score OR total ISI
# Threshold: >= 15 → 1, < 15 → 0, missing/NaN → ''
# NOTE: Need to locate this file first

# Pseudocode:
# df_isi = pd.read_excel('path/to/STAGES_ASQ_ISI.xlsx')
# df_isi['insomnia_binary'] = df_isi['isi_score'].apply(...)
# df = df.merge(df_isi[['subject_id', 'insomnia_binary', 'isi_score']], ...)
```

**f) Apnea Binary (apnea_binary) - SKIP FOR NOW OR EXTRACT FROM XML**
```python
# Option 1: Skip STAGES apnea for Phase 1 (focus on easier datasets)
# Option 2: Extract from XML annotations (medium effort)
# Option 3: Extract from STAGESPSGKeySRBDVariables.xlsx if available

# Recommendation: SKIP for Phase 1, add later
```

**Output:** `stages_targets.csv`

**Validation Checks:**
- Total subjects: ~1,513
- Depression prevalence: ~15-25%
- Anxiety prevalence: ~15-20%
- Insomnia prevalence: ~20-30% (if we get external file)

**Key Question for User:**
1. Where are STAGES external XLSX files located?
   - `STAGES ASQ ISI to DIET 20200513 Final deidentified.xlsx`
   - `STAGESPSGKeySRBDVariables.xlsx`
   - `STAGES post sleep questionnaire 2020-09-06 deidentified.xlsx`
2. Should we skip STAGES apnea for now?

---

### Phase 3: Consolidation and Validation (Step 7-9)

#### Step 7: Create Master Targets File
**Script:** `scripts/create_master_targets.py`

**Tasks:**
1. Load all per-dataset CSV files
2. Standardize columns (ensure same schema)
3. Create unified_id: `{dataset}_{subject_id}_{visit}`
4. Convert empty strings to -1 for binary, NaN for continuous
5. Save as `unified_targets.parquet`

```python
import pandas as pd
from pathlib import Path

# Load per-dataset CSVs
stages = pd.read_csv('/scratch/boshra95/psg/unified/targets/stages_targets.csv')
shhs = pd.read_csv('/scratch/boshra95/psg/unified/targets/shhs_targets.csv')
apples = pd.read_csv('/scratch/boshra95/psg/unified/targets/apples_targets.csv')
mros = pd.read_csv('/scratch/boshra95/psg/unified/targets/mros_targets.csv')

# Concatenate
all_targets = pd.concat([stages, shhs, apples, mros], ignore_index=True)

# Create unified_id
all_targets['unified_id'] = (
    all_targets['dataset'] + '_' + 
    all_targets['subject_id'].astype(str) + '_' + 
    all_targets['visit'].astype(str)
)

# Convert empty strings to -1 (missing) for binary targets
binary_cols = ['apnea_binary', 'depression_binary', 'sleepiness_binary', 
               'anxiety_binary', 'insomnia_binary', 'cvd_binary', 'rested_morning']

for col in binary_cols:
    if col in all_targets.columns:
        all_targets[col] = all_targets[col].replace('', -1).astype(int)

# Convert empty strings to NaN for continuous scores
score_cols = ['ahi_score', 'phq9_score', 'bdi_score', 'gad7_score', 
              'ess_score', 'isi_score', 'fss_score']

for col in score_cols:
    if col in all_targets.columns:
        all_targets[col] = pd.to_numeric(all_targets[col], errors='coerce')

# Add extraction metadata
all_targets['extraction_date'] = pd.Timestamp.now()

# Save as parquet
all_targets.to_parquet('/scratch/boshra95/psg/unified/metadata/unified_targets.parquet', 
                       index=False)

print(f"Saved {len(all_targets)} records to unified_targets.parquet")
```

---

#### Step 8: Create Task Subject Lists
**Script:** `scripts/create_task_subject_lists.py`

**Purpose:** Fast lookup of which subjects have which tasks

```python
import json

task_subject_lists = {}

# For each task
for task in binary_cols:
    task_subject_lists[task] = {}
    
    # For each dataset
    for dataset in ['stages', 'shhs', 'apples', 'mros']:
        # Get subjects with non-missing values for this task
        subset = all_targets[
            (all_targets['dataset'] == dataset) & 
            (all_targets[task] >= 0)  # 0 or 1, exclude -1
        ]
        
        subject_ids = subset['subject_id'].tolist()
        task_subject_lists[task][dataset] = subject_ids
        
        print(f"{task} - {dataset}: {len(subject_ids)} subjects")

# Save as JSON
with open('/scratch/boshra95/psg/unified/targets/task_subject_lists.json', 'w') as f:
    json.dump(task_subject_lists, f, indent=2)
```

**Output Format:**
```json
{
  "apnea_binary": {
    "stages": [],
    "shhs": [1, 2, 3, ...],
    "apples": [100001, 100002, ...],
    "mros": [1001, 1002, ...]
  },
  "depression_binary": {
    "stages": ["BOGN00004", "BOGN00007", ...],
    "apples": [100001, ...]
  }
}
```

---

#### Step 9: Create Task Statistics
**Script:** `scripts/create_task_statistics.py`

**Purpose:** Class distributions and prevalence for each task

```python
task_stats = {}

for task in binary_cols:
    # Get all non-missing values
    valid = all_targets[all_targets[task] >= 0]
    
    total = len(valid)
    positive = (valid[task] == 1).sum()
    negative = (valid[task] == 0).sum()
    prevalence = positive / total if total > 0 else 0
    
    task_stats[task] = {
        'total': int(total),
        'positive': int(positive),
        'negative': int(negative),
        'prevalence': round(prevalence, 3),
        'by_dataset': {}
    }
    
    # Per-dataset statistics
    for dataset in ['stages', 'shhs', 'apples', 'mros']:
        subset = valid[valid['dataset'] == dataset]
        if len(subset) > 0:
            ds_total = len(subset)
            ds_pos = (subset[task] == 1).sum()
            ds_neg = (subset[task] == 0).sum()
            ds_prev = ds_pos / ds_total
            
            task_stats[task]['by_dataset'][dataset] = {
                'total': int(ds_total),
                'positive': int(ds_pos),
                'negative': int(ds_neg),
                'prevalence': round(ds_prev, 3)
            }

# Save as JSON
with open('/scratch/boshra95/psg/unified/targets/task_statistics.json', 'w') as f:
    json.dump(task_stats, f, indent=2)

# Also print to console
print("\n" + "="*80)
print("TASK STATISTICS SUMMARY")
print("="*80)
for task, stats in task_stats.items():
    print(f"\n{task}:")
    print(f"  Total: {stats['total']}")
    print(f"  Prevalence: {stats['prevalence']:.1%}")
    print(f"  By Dataset:")
    for ds, ds_stats in stats['by_dataset'].items():
        print(f"    {ds}: {ds_stats['total']} subjects ({ds_stats['prevalence']:.1%} positive)")
```

---

### Phase 4: Quality Validation (Step 10)

#### Step 10: Validation and QC Report
**Script:** `scripts/validate_targets.py`

**Validation Checks:**

1. **Column Presence Check**
   - All expected columns exist in each file
   - No unexpected columns

2. **Value Range Check**
   - Binary targets: Only -1, 0, 1
   - AHI scores: 0-120 (physiologically valid range)
   - PHQ-9: 0-27
   - BDI-II: 0-63
   - GAD-7: 0-21
   - ESS: 0-24
   - ISI: 0-28
   - FSS: 1-7 (mean)

3. **Prevalence Check (Expected Ranges)**
   - Apnea: 25-45% (warn if outside)
   - Depression: 10-30%
   - Sleepiness: 20-40%
   - Anxiety: 10-25%
   - Insomnia: 15-35%
   - CVD: 8-20%

4. **Missing Pattern Check**
   - Report % missing per task per dataset
   - Flag if > 50% missing

5. **Cross-Task Consistency**
   - Check subjects with multiple tasks
   - Look for suspicious patterns (e.g., all same value)

6. **Join Check with Unified Metadata**
   - Verify all subject_ids in targets match unified_metadata
   - Report orphaned subjects (in targets but not metadata)

**Output:**
- `logs/target_validation_report.txt`
- `logs/target_validation_warnings.txt` (issues to review)
- Console summary

---

### Phase 5: Tier 3 - Rested Morning (Step 11)

#### Step 11: Extract Rested Morning (After Tier 2 Complete)
**Script:** `scripts/extract_rested_morning.py`

**Extraction by Dataset:**

**SHHS:**
```python
# Visit 1: rest10
# Visit 2: ms204c
# Already extracted in Step 4, just verify
```

**MrOS:**
```python
# Visit 1 & 2: poxqual3
# Already extracted in Step 5, just verify
```

**STAGES:**
```python
# From: STAGES post sleep questionnaire 2020-09-06 deidentified.xlsx
# Columns: isq_0500 or compared_usual_feel_upon_awakening
# Need to map subjective responses to 0/1/-1

# This requires manual inspection of the XLSX file first
```

**Note:** This task is more complex due to subjective mapping and external files.
Save for after Tier 1 & 2 are validated.

---

## Execution Plan

### Week 1: Tier 1 Tasks

**Day 1:**
- [ ] Step 1: Setup directories
- [ ] Step 2: Verify CSV files (all datasets)
- [ ] Step 3: Extract APPLES targets (complete)

**Day 2:**
- [ ] Step 4: Extract SHHS targets (complete)
- [ ] Step 5: Extract MrOS targets (partial - apnea + sleepiness)

**Day 3:**
- [ ] Step 6: Extract STAGES targets (depression only)
- [ ] Step 7: Create master targets file (Tier 1 only)
- [ ] Step 8: Create task subject lists
- [ ] Step 9: Create task statistics
- [ ] Step 10: Validation report

**Expected Output after Day 3:**
- ✅ apnea_binary: SHHS, APPLES, MrOS (~12,000 subjects)
- ✅ depression_binary: STAGES, APPLES (~2,600 subjects)
- ✅ sleepiness_binary: APPLES (~1,100 subjects)

### Week 2: Tier 2 Tasks

**Day 4:**
- [ ] Rerun Step 6: Add STAGES anxiety
- [ ] Rerun Step 5: Add MrOS insomnia
- [ ] Update master file

**Day 5:**
- [ ] Step 6: Handle STAGES insomnia (external XLSX)
- [ ] Step 4: Verify SHHS CVD extraction
- [ ] Update all outputs

**Day 6:**
- [ ] Validation and testing
- [ ] Fix any issues
- [ ] Final QC report

**Expected Output after Day 6:**
- ✅ All Tier 1 + Tier 2 tasks extracted
- ✅ ~14,000+ total samples across all tasks

### Week 3: Tier 3 (Optional)

**Day 7:**
- [ ] Step 11: Extract rested_morning (all datasets)
- [ ] Final validation

---

## Key Questions for User

### Critical (Need answers before starting):
1. **STAGES External Files:**
   - Where are these XLSX files located?
   - `STAGES ASQ ISI to DIET 20200513 Final deidentified.xlsx`
   - `STAGESPSGKeySRBDVariables.xlsx`
   - `STAGES post sleep questionnaire 2020-09-06 deidentified.xlsx`

2. **STAGES AHI:**
   - Skip for Phase 1 and focus on SHHS/APPLES/MrOS?
   - Or extract from XML now?
   - **Recommendation:** Skip for Phase 1

3. **Multi-Visit Strategy for SHHS/MrOS:**
   - Keep both visits as separate samples (more data)?
   - Or use only Visit 1 (simpler)?
   - **Recommendation:** Keep both as separate samples

4. **BDI-II Threshold:**
   - Use ≥14 (sensitive, ~20-30% positive)?
   - Or ≥20 (specific, ~10-15% positive)?
   - **Recommendation:** Start with ≥14, can threshold later

### Nice to Have (Can decide later):
5. Should we extract fatigue_binary from STAGES (FSS)?
6. Do you want cognition (MMSE) from APPLES despite ceiling effect?
7. Should we track which visit each target came from for APPLES (BL vs DX)?

---

## Success Metrics

After full implementation, we should have:

### Data Coverage:
- ✅ **Tier 1:** ~15,000+ samples across 3 tasks
- ✅ **Tier 2:** Added ~4,000+ for insomnia, ~8,000+ for CVD
- ✅ **Overall:** ~20,000+ total task-subject pairs

### Quality:
- ✅ Prevalence within expected clinical ranges (15-40%)
- ✅ < 10% invalid values (out of range)
- ✅ > 90% join success with unified_metadata
- ✅ No duplicate subjects within dataset (except multi-visit)

### Usability:
- ✅ Fast loading: task_subject_lists.json < 1 MB
- ✅ Easy filtering: Can select subjects for any task in < 1 second
- ✅ Documentation: README explains all columns and thresholds

---

## File Delivery Schedule

### After Day 3 (Tier 1):
```
/scratch/boshra95/psg/unified/targets/
├── apples_targets.csv            # ✅
├── shhs_targets.csv              # ✅ (partial - apnea only)
├── mros_targets.csv              # ✅ (partial - apnea, sleepiness)
├── stages_targets.csv            # ✅ (partial - depression only)
├── task_subject_lists.json       # ✅ (Tier 1 tasks)
└── task_statistics.json          # ✅ (Tier 1 tasks)

/scratch/boshra95/psg/unified/metadata/
└── unified_targets.parquet       # ✅ (Tier 1 tasks)
```

### After Day 6 (Tier 1 + Tier 2):
```
All files updated with:
- anxiety_binary (STAGES)
- insomnia_binary (STAGES, MrOS)
- cvd_binary (SHHS)
```

### After Day 7 (All tasks):
```
All files updated with:
- rested_morning (STAGES, SHHS, MrOS)
```

---

## Next Steps

Please review this plan and let me know:

1. ✅ Approve overall approach?
2. 📁 Location of STAGES external XLSX files?
3. 🎯 Any threshold changes (especially BDI-II: ≥14 vs ≥20)?
4. 📊 Multi-visit: Keep both visits or Visit 1 only?
5. 🚀 Ready to start with Step 1-3 (APPLES extraction)?

Once approved, I'll start with:
- Create directory structure
- CSV verification script
- APPLES extraction (easiest, will serve as template)
