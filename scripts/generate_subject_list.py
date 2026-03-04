#!/usr/bin/env python3
"""Generate subject list for SLURM array jobs.

Creates a text file with one line per subject:
    subject_id edf_path

This file is used by the array job script to distribute work.
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description='Generate subject list for array jobs')
    parser.add_argument('--dataset', required=True, choices=['stages', 'shhs', 'apples', 'mros'],
                        help='Dataset to generate list for')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output file path')
    parser.add_argument('--max-subjects', type=int, default=None,
                        help='Limit number of subjects (for testing)')
    parser.add_argument('--require-annotations', action='store_true',
                        help='Only include subjects with annotations')
    args = parser.parse_args()
    
    # Load metadata
    metadata_path = Path('/scratch/boshra95/psg/unified/metadata/unified_metadata.parquet')
    if not metadata_path.exists():
        metadata_path = Path('/scratch/boshra95/psg_metadata/unified_metadata.parquet')
    
    if not metadata_path.exists():
        print(f"ERROR: Metadata not found", file=sys.stderr)
        print(f"  Tried: /scratch/boshra95/psg/unified/metadata/unified_metadata.parquet", file=sys.stderr)
        print(f"  Tried: /scratch/boshra95/psg_metadata/unified_metadata.parquet", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_parquet(metadata_path)
    
    # Filter to dataset
    ds_df = df[(df['dataset'].str.upper() == args.dataset.upper()) & (df['has_edf'] == True)].copy()
    
    if len(ds_df) == 0:
        print(f"ERROR: No subjects found for {args.dataset}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(ds_df)} subjects with EDFs")
    
    # Filter to subjects with annotations if requested
    if args.require_annotations:
        ds_df = ds_df[ds_df['annotation_path'].notna()]
        print(f"  {len(ds_df)} have annotations")
    
    # Limit if requested
    if args.max_subjects:
        ds_df = ds_df.head(args.max_subjects)
        print(f"  Limited to {args.max_subjects} subjects")
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Write subject list
    with open(args.output, 'w') as f:
        for idx, row in ds_df.iterrows():
            subject_id = row.get('subject_id', row.get('nsrrid', ''))
            edf_path = row.get('edf_path', '')
            
            if not subject_id or not edf_path:
                print(f"WARNING: Skipping row with missing data: {idx}", file=sys.stderr)
                continue
            
            f.write(f"{subject_id} {edf_path}\n")
    
    print(f"\n✓ Subject list saved to: {args.output}")
    print(f"  Total subjects: {len(ds_df)}")
    print(f"\nTo submit array job:")
    print(f"  sbatch --array=0-{len(ds_df)-1}%50 jobs/preprocess_signals_array.sh {args.dataset}")

if __name__ == '__main__':
    main()
