#!/usr/bin/env python3
"""Quick script to inspect current preprocessing state."""

import pandas as pd
from pathlib import Path

# Check unified metadata
print("="*80)
print("UNIFIED METADATA INSPECTION")
print("="*80)

metadata_path = Path('/scratch/boshra95/psg/unified/metadata/unified_metadata.parquet')
if metadata_path.exists():
    df = pd.read_parquet(metadata_path)
    print(f"\nTotal subjects: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nFirst 20 columns: {list(df.columns[:20])}")
    print(f"\nDataset distribution:")
    print(df['dataset'].value_counts())
    print(f"\nHas EDF distribution:")
    print(df['has_edf'].value_counts())
    if 'has_sleep_staging' in df.columns:
        print(f"\nHas sleep staging:")
        print(df['has_sleep_staging'].value_counts())
else:
    print(f"NOT FOUND: {metadata_path}")

# Check preprocessing summaries
print("\n" + "="*80)
print("PREPROCESSING SUMMARIES")
print("="*80)

for dataset in ['stages', 'shhs', 'apples', 'mros']:
    summary_path = Path(f'/scratch/boshra95/psg/{dataset}/derived/logs/preprocessing_summary_{dataset}.csv')
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        print(f"\n{dataset.upper()}:")
        print(f"  Total records: {len(df)}")
        print(f"  Status distribution:")
        for status, count in df['status'].value_counts().items():
            print(f"    {status}: {count}")
        if 'has_annotations' in df.columns:
            print(f"  With annotations: {df['has_annotations'].sum()} / {len(df)}")
    else:
        print(f"\n{dataset.upper()}: No summary found")

print("\n" + "="*80)
