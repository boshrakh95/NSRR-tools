# STAGES Data Reality Check

## Summary
The STAGES dataset structure differs from expected:

### EDF Files
- **Expected**: Extracted EDFs in `original/*/usable/*.edf`
- **Reality**: Raw EDFs are compressed in `raw_tar/stages_raw.tar.zst`
- **Action needed**: Extract the tar.zst file first

### Metadata Columns
**Main dataset** (`stages-dataset-0.3.0.csv`):
- Subject ID: `subject_code` (not `nsrrid`)
- Has: `isi_score`, `phq_1000`, `gad_0800` ✓
- Missing: `ahi` from PSG - only has questionnaire-based apnea scores

**Harmonized dataset** (`stages-harmonized-dataset-0.3.0.csv`):
- Has demographics: `nsrr_age`, `nsrr_sex`, `nsrr_bmi` ✓
- Subject ID: `subject_code`

### Next Steps
1. Extract `stages_raw.tar.zst` to get EDFs
2. Update adapter to use `subject_code` instead of `nsrrid`
3. Merge main + harmonized datasets for complete metadata
4. PSG-derived AHI will come from XML annotations, not CSV metadata
