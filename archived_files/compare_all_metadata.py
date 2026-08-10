#!/usr/bin/env python3
"""Compare all unified_metadata.parquet files to find the most complete one."""

import pandas as pd
from pathlib import Path
import sys

# All found metadata files
metadata_files = [
    '/scratch/boshra95/psg/unified/unified_metadata.parquet',
    '/scratch/boshra95/psg/unified/metadata/unified_metadata.parquet',
    '/scratch/boshra95/psg_metadata/unified_metadata.parquet',
    '/scratch/boshra95/psg_metadata/mros/unified_metadata.parquet',
]

print("=" * 100)
print("COMPREHENSIVE METADATA FILE COMPARISON")
print("=" * 100)

results = []

for idx, filepath in enumerate(metadata_files, 1):
    path = Path(filepath)
    print(f"\n{'=' * 100}")
    print(f"FILE {idx}: {filepath}")
    print(f"{'=' * 100}")
    
    if not path.exists():
        print("   ✗ FILE DOES NOT EXIST")
        results.append({
            'file': filepath,
            'exists': False,
            'total_subjects': 0,
            'datasets': [],
            'size_mb': 0,
            'last_modified': None
        })
        continue
    
    try:
        # Read metadata
        df = pd.read_parquet(path)
        
        # Get file stats
        file_size_mb = path.stat().st_size / 1024 / 1024
        last_modified = pd.Timestamp.fromtimestamp(path.stat().st_mtime)
        
        print(f"✓ EXISTS")
        print(f"File size: {file_size_mb:.2f} MB")
        print(f"Last modified: {last_modified}")
        print(f"\nTotal rows: {len(df)}")
        print(f"Total columns: {len(df.columns)}")
        
        # Check datasets
        if 'dataset' in df.columns:
            datasets = df['dataset'].unique()
            print(f"\nDatasets present: {list(datasets)}")
            print(f"\nBreakdown by dataset:")
            dataset_counts = df.groupby('dataset').size()
            for dataset, count in dataset_counts.items():
                print(f"  {dataset:10s}: {count:5d} subjects")
            
            # Check EDF availability
            if 'has_edf' in df.columns:
                edf_counts = df.groupby('dataset')['has_edf'].sum()
                print(f"\nSubjects with EDF files:")
                for dataset in datasets:
                    if dataset in edf_counts.index:
                        print(f"  {dataset:10s}: {edf_counts[dataset]:5d} / {dataset_counts[dataset]:5d}")
            
            # Check data completeness
            if 'has_staging' in df.columns:
                staging_counts = df.groupby('dataset')['has_staging'].sum()
                print(f"\nSubjects with staging annotations:")
                for dataset in datasets:
                    if dataset in staging_counts.index:
                        print(f"  {dataset:10s}: {staging_counts[dataset]:5d}")
            
            results.append({
                'file': filepath,
                'exists': True,
                'total_subjects': len(df),
                'datasets': list(datasets),
                'num_datasets': len(datasets),
                'size_mb': file_size_mb,
                'last_modified': last_modified,
                'with_edf': df['has_edf'].sum() if 'has_edf' in df.columns else 0
            })
        else:
            print("\n⚠ No 'dataset' column found")
            results.append({
                'file': filepath,
                'exists': True,
                'total_subjects': len(df),
                'datasets': [],
                'num_datasets': 0,
                'size_mb': file_size_mb,
                'last_modified': last_modified,
                'with_edf': 0
            })
            
    except Exception as e:
        print(f"✗ ERROR reading file: {e}")
        results.append({
            'file': filepath,
            'exists': True,
            'error': str(e),
            'total_subjects': 0,
            'datasets': [],
            'size_mb': 0,
        })

# Summary comparison
print(f"\n{'=' * 100}")
print("SUMMARY COMPARISON")
print(f"{'=' * 100}\n")

print(f"{'File':<50s} {'Subjects':<10s} {'Datasets':<30s} {'Size (MB)':<10s}")
print("-" * 100)

for r in results:
    if r['exists'] and 'error' not in r:
        datasets_str = ', '.join(r['datasets']) if r['datasets'] else 'None'
        print(f"{Path(r['file']).name:<50s} {r['total_subjects']:<10d} {datasets_str:<30s} {r['size_mb']:<10.2f}")

# Recommendation
print(f"\n{'=' * 100}")
print("RECOMMENDATION")
print(f"{'=' * 100}\n")

# Find most complete file
valid_results = [r for r in results if r['exists'] and 'error' not in r]

if valid_results:
    # Sort by: 1) number of datasets, 2) total subjects, 3) last modified
    most_complete = sorted(valid_results, 
                          key=lambda x: (x.get('num_datasets', 0), 
                                       x['total_subjects'], 
                                       x.get('last_modified', pd.Timestamp(0))),
                          reverse=True)[0]
    
    print(f"✓ MOST COMPLETE FILE:")
    print(f"  Path: {most_complete['file']}")
    print(f"  Subjects: {most_complete['total_subjects']}")
    print(f"  Datasets: {', '.join(most_complete['datasets'])}")
    print(f"  With EDF: {most_complete.get('with_edf', 0)}")
    print(f"  Size: {most_complete['size_mb']:.2f} MB")
    if most_complete.get('last_modified'):
        print(f"  Last modified: {most_complete['last_modified']}")
    
    # Expected location
    expected_path = '/scratch/boshra95/psg/unified/metadata/unified_metadata.parquet'
    
    if most_complete['file'] == expected_path:
        print(f"\n✓✓ This file is already in the correct location!")
        print(f"   Your preprocessing script will find it.")
    else:
        print(f"\n⚠ This file needs to be copied to the expected location:")
        print(f"   {expected_path}")
        print(f"\nRun this command:")
        print(f"   mkdir -p {Path(expected_path).parent}")
        print(f"   cp {most_complete['file']} {expected_path}")
        
        # Check if target already exists
        if Path(expected_path).exists():
            existing = [r for r in results if r['file'] == expected_path][0]
            if existing['total_subjects'] < most_complete['total_subjects']:
                print(f"\n   Note: Current file at target location has FEWER subjects ({existing['total_subjects']} vs {most_complete['total_subjects']})")
                print(f"   Overwriting is recommended.")
            else:
                print(f"\n   Note: Current file at target location already has {existing['total_subjects']} subjects")
else:
    print("✗ No valid metadata files found!")

print(f"\n{'=' * 100}\n")
