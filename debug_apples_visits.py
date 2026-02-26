#!/usr/bin/env python3
"""Debug APPLES visit structure."""

import pandas as pd
from pathlib import Path

# Load APPLES CSVs
datasets_path = Path('/scratch/boshra95/nsrr_downloads/apples/datasets')
harm_file = datasets_path / 'apples-harmonized-dataset-0.1.0.csv'
main_file = datasets_path / 'apples-dataset-0.1.0.csv'

print("Loading APPLES CSVs...")
harm = pd.read_csv(harm_file, low_memory=False)
main = pd.read_csv(main_file, low_memory=False)

# Rename appleid to nsrrid for consistency
if 'appleid' in main.columns:
    main = main.rename(columns={'appleid': 'nsrrid'})

print("\n" + "="*80)
print("HARMONIZED FILE")
print("="*80)
print(f"Total rows: {len(harm)}")
print(f"Unique nsrrids: {harm['nsrrid'].nunique()}")

if 'visitn' in harm.columns:
    print(f"\nVisit values: {sorted(harm['visitn'].dropna().unique())}")
    print("\nVisit counts:")
    print(harm['visitn'].value_counts().sort_index())

if 'fileid' in harm.columns:
    print(f"\nFileids (non-null): {harm['fileid'].notna().sum()}")
    print(f"Unique fileids: {harm['fileid'].nunique()}")
    
    # Check which visits have EDFs
    with_edf = harm[harm['fileid'].notna()]
    print(f"\n{len(with_edf)} rows have EDF (fileid not null)")
    if 'visitn' in with_edf.columns:
        print("Visits with EDFs:")
        print(with_edf['visitn'].value_counts().sort_index())
        
        # Show sample of visit 1 vs visit 3
        print("\n" + "="*80)
        print("SAMPLE DATA COMPARISON")
        print("="*80)
        
        # Take first subject with multiple visits
        subj_with_multiple = harm.groupby('nsrrid').size()
        subj_with_multiple = subj_with_multiple[subj_with_multiple > 1].index[0]
        
        sample = harm[harm['nsrrid'] == subj_with_multiple][['nsrrid', 'visitn', 'fileid', 
                                                               'nsrr_age', 'nsrr_sex', 'nsrr_bmi',
                                                               'nsrr_ahi_chicago1999']].copy()
        print(f"\nSample subject {subj_with_multiple} (multiple visits):")
        print(sample.to_string())
        
print("\n" + "="*80)
print("MAIN FILE")
print("="*80)
print(f"Total rows: {len(main)}")
print(f"Unique nsrrids: {main['nsrrid'].nunique()}")

if 'visitn' in main.columns:
    print(f"\nVisit values: {sorted(main['visitn'].dropna().unique())}")
    print("\nVisit counts:")
    print(main['visitn'].value_counts().sort_index())

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\nStrategy:")
print("1. Filter csvs to only DX visit (visit 3) - this is where PSG was recorded")
print("2. Create separate BL visit (visit 1) demographics dataframe")
print("3. Merge DX data with channel data on fileid")
print("4. Overlay BL demographics (age, sex, bmi) onto DX data")
print("5. Result: ONE row per subject with EDF")
