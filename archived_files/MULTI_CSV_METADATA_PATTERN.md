# Multi-CSV Metadata Loading Pattern

## Overview
NSRR datasets distribute metadata across multiple CSV files based on data type and standardization level. Following the pattern from the nocturn project, metadata should be loaded from multiple sources and merged.

## Common File Types

### 1. Harmonized Dataset (NSRR-standardized)
**Filename pattern**: `{dataset}-harmonized-dataset-{version}.csv`

**Common columns**:
- `nsrr_age`: Age in years
- `nsrr_sex`: Sex (male/female/M/F)
- `nsrr_race`: Race category
- `nsrr_ethnicity`: Ethnicity
- `nsrr_bmi`: Body Mass Index
- `nsrr_current_smoker`: Current smoking status (0/1/yes/no)
- `nsrr_ever_smoker`: Ever smoking status
- `nsrr_bp_systolic`: Systolic blood pressure
- `nsrr_bp_diastolic`: Diastolic blood pressure

**PSG metrics** (when available):
- `nsrr_ahi_*`: Apnea-Hypopnea Index (various definitions)
- `nsrr_ttleffsp_f1`: Total sleep efficiency
- `nsrr_phrnumar_f1`: Arousal index
- `nsrr_pctdursp_sr`: REM percentage
- `nsrr_pctdursp_s3`: N3 percentage
- `nsrr_ttldursp_f1`: Total sleep time

### 2. Main Dataset (Dataset-specific questionnaires)
**Filename pattern**: `{dataset}-dataset-{version}.csv`

**Dataset-specific columns**:
- Clinical questionnaires (PHQ, GAD, ISI, ESS, FSS, FOSQ, etc.)
- Medical history
- Social history
- Sleep schedules
- Cognitive assessments

### 3. Specialized Files (Optional)
- CVD summary: Cardiovascular outcomes
- HRV summary: Heart rate variability metrics
- PSG key variables: Additional PSG-derived metrics
- Cognitive assessments: CNB, MMSE scores
- Post-sleep questionnaires

## Implementation Pattern

### Step 1: Define Metadata Files
```python
class DatasetAdapter(BaseNSRRAdapter):
    def __init__(self, config):
        super().__init__(config, 'dataset_name')
        
        # Define all available metadata files
        self.metadata_files = {
            'main': 'dataset-dataset-X.Y.Z.csv',
            'harmonized': 'dataset-harmonized-dataset-X.Y.Z.csv',
            # Add optional files as needed
            # 'cvd': 'dataset-cvd-summary-X.Y.Z.csv',
            # 'hrv': 'dataset-hrv-summary-X.Y.Z.csv',
        }
```

### Step 2: Implement Multi-CSV Loading
```python
def load_metadata(self) -> pd.DataFrame:
    """Load and merge metadata from multiple CSV files."""
    datasets_path = self.dataset_paths['datasets']
    dfs_to_merge = []
    
    # Load each available file
    for file_key, filename in self.metadata_files.items():
        file_path = datasets_path / filename
        
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {file_key}: {len(df)} subjects, {len(df.columns)} columns")
                dfs_to_merge.append(df)
            except Exception as e:
                logger.error(f"Error loading {file_key}: {e}")
        else:
            logger.warning(f"{file_key} not found: {file_path}")
    
    # Merge on subject ID
    if not dfs_to_merge:
        logger.error("No metadata files found")
        return pd.DataFrame()
    
    if len(dfs_to_merge) == 1:
        df = dfs_to_merge[0]
    else:
        df = dfs_to_merge[0]
        for df_next in dfs_to_merge[1:]:
            # Use dataset-specific ID column
            merge_key = self.subject_id_col
            
            df = df.merge(df_next, on=merge_key, how='outer', suffixes=('', '_dup'))
            
            # Remove duplicate columns
            dup_cols = [col for col in df.columns if col.endswith('_dup')]
            if dup_cols:
                df = df.drop(columns=dup_cols)
    
    logger.info(f"Merged metadata: {len(df)} subjects, {len(df.columns)} columns")
    return df
```

### Step 3: Specify Expected Columns
```python
# Group columns by source for documentation
self.phenotype_cols = [
    # From harmonized
    'nsrr_age', 'nsrr_sex', 'nsrr_race', 'nsrr_bmi',
    
    # From main dataset
    'phq_score', 'ess_score', 'isi_score',
    
    # Note: AHI usually from annotations or harmonized file
]
```

## Dataset-Specific Examples

### STAGES
```python
metadata_files = {
    'main': 'stages-dataset-0.3.0.csv',
    'harmonized': 'stages-harmonized-dataset-0.3.0.csv',
}
# Merge on: subject_code (not nsrrid!)
# AHI: From XML annotations (not CSV)
```

### SHHS
```python
metadata_files = {
    'main': 'shhs1-dataset-0.21.0.csv',  # or shhs2-...
    'harmonized': 'shhs-harmonized-dataset-0.21.0.csv',
    'cvd': 'shhs-cvd-summary-dataset-0.21.0.csv',
    # 'hrv': 'shhs-hrv-summary-0.21.0.csv',  # optional
}
# Merge on: nsrrid
# Visit-specific: visitnumber column (1 or 2)
# AHI: rdi3p from main CSV
```

### APPLES
```python
metadata_files = {
    'main': 'apples-dataset-0.1.0.csv',
    'harmonized': 'apples-harmonized-dataset-0.1.0.csv',
}
# Merge on: nsrrid
# Visit-specific: visitn column (1=BL, 3=DX, 4=CPAP)
# AHI: nsrr_ahi_chicago1999 from harmonized
```

### MrOS
```python
metadata_files = {
    'main': 'mros-visit1-dataset-0.6.0.csv',
    'harmonized': 'mros-visit1-harmonized-0.6.0.csv',
}
# Merge on: nsrrid
# Visit: visit1 or visit2
```

## Subject ID Column Mapping

| Dataset | ID Column    | Notes                           |
|---------|-------------|---------------------------------|
| STAGES  | subject_code | **Not nsrrid!**                |
| SHHS    | nsrrid      | Standard                        |
| APPLES  | nsrrid      | Standard                        |
| MrOS    | nsrrid      | Standard                        |

## Merge Strategy

### Outer Join
Use `how='outer'` to capture all subjects across files:
- Some subjects may have demographics but no questionnaires
- Some may have questionnaires but missing harmonized data
- Preserves maximum data availability

### Handle Duplicates
When merging, duplicate columns may appear:
- Use `suffixes=('', '_dup')` to mark duplicates
- Drop `*_dup` columns after merge
- Prioritize harmonized standardized columns over dataset-specific versions

## Testing
After implementing multi-CSV loading:

1. **Check subject count**: Should increase after merging
   ```python
   # Before: Only subjects in main CSV
   # After: Union of all subjects across files
   ```

2. **Check column availability**:
   ```python
   missing_cols = [col for col in phenotype_cols if col not in df.columns]
   # Should be empty or minimal
   ```

3. **Verify merge quality**:
   ```python
   # Check for unexpected NaN patterns
   print(df[phenotype_cols].isnull().sum())
   ```

## References
- Nocturn project: `/home/boshra95/nocturn/`
- Config files: `/home/boshra95/nocturn/configs/datasets/*.yaml`
- Ontology mappings: `/home/boshra95/nocturn/configs/ontology-datasets.yaml`
- Sanity scripts: `/home/boshra95/nocturn/src/nocturn/cli/sanity_*.py`
