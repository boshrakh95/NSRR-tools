# STAGES Data Reality Check

## Summary
The STAGES dataset structure differs from expected:

### EDF Files
- **Expected**: Extracted EDFs in `original/*/usable/*.edf`
- **Reality**: Raw EDFs are compressed in `raw_tar/stages_raw.tar.zst`
- **Action needed**: Extract the tar.zst file first

### Metadata Files (Multi-CSV Pattern)
Following the nocturn project structure, STAGES metadata is split across multiple CSV files:

**Main dataset** (`stages-dataset-0.3.0.csv`):
- Subject ID: `subject_code` (not `nsrrid`)
- Columns: Clinical questionnaires
  - `phq_1000`: PHQ-8 depression score
  - `gad_0800`: GAD-7 anxiety score
  - `isi_score`: Insomnia Severity Index
  - `ess_0900`: Epworth Sleepiness Scale
  - `fss_1000`: Fatigue Severity Scale
  - Medical history: `mdhx_*` columns
  - Sleep schedules: `sched_*` columns

**Harmonized dataset** (`stages-harmonized-dataset-0.3.0.csv`):
- Subject ID: `subject_code`
- Columns: NSRR-standardized demographics
  - `nsrr_age`: Age
  - `nsrr_sex`: Sex (M/F)
  - `nsrr_race`: Race category
  - `nsrr_bmi`: Body Mass Index
  - `nsrr_current_smoker`: Smoking status

**PSG-derived metrics**:
- **AHI** (Apnea-Hypopnea Index): Extracted from XML annotations, not CSV
- Sleep stages: From XML files
- Events: From XML files

### Implementation
The adapter now:
1. ✅ Loads both main and harmonized CSV files
2. ✅ Merges them on `subject_code` (STAGES) or `nsrrid` (other datasets)
3. ✅ No more missing column warnings
4. ✅ Increased from 1,881 to 2,103 subjects (outer merge captures all subjects)

### Next Steps
1. Extract `stages_raw.tar.zst` to get EDFs
2. ~~Update adapter to use `subject_code`~~ ✅ Done
3. ~~Merge main + harmonized datasets~~ ✅ Done  
4. Resolve subject ID mismatch between EDF filenames and metadata codes
5. Extract PSG metrics (AHI, sleep stages) from XML annotations
