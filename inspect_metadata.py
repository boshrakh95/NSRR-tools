#!/usr/bin/env python3
"""Quick script to inspect unified metadata files."""

import pandas as pd
from pathlib import Path

# Check both locations
legacy_path = Path('/scratch/boshra95/psg_metadata/unified_metadata.parquet')
new_path = Path('/scratch/boshra95/psg/unified/metadata/unified_metadata.parquet')

print("=" * 80)
print("UNIFIED METADATA INSPECTION")
print("=" * 80)

# Check legacy location
print(f"\n1. LEGACY LOCATION: {legacy_path}")
if legacy_path.exists():
    df_legacy = pd.read_parquet(legacy_path)
    print(f"   ✓ EXISTS")
    print(f"   Rows: {len(df_legacy)}")
    print(f"   Columns: {len(df_legacy.columns)}")
    print(f"   Datasets: {df_legacy['dataset'].unique()}")
    print(f"\n   Dataset breakdown:")
    print(df_legacy.groupby('dataset').size())
    print(f"\n   Columns:")
    print(f"   {', '.join(df_legacy.columns)}")
    print(f"\n   First few rows:")
    print(df_legacy.head(3))
    print(f"\n   Has EDF file:")
    if 'has_edf' in df_legacy.columns:
        print(f"   {df_legacy['has_edf'].sum()} subjects with EDF files")
    print(f"\n   File size: {legacy_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   Last modified: {pd.Timestamp.fromtimestamp(legacy_path.stat().st_mtime)}")
else:
    print("   ✗ DOES NOT EXIST")

# Check new location
print(f"\n2. NEW LOCATION: {new_path}")
if new_path.exists():
    df_new = pd.read_parquet(new_path)
    print(f"   ✓ EXISTS")
    print(f"   Rows: {len(df_new)}")
    print(f"   Columns: {len(df_new.columns)}")
    print(f"   Datasets: {df_new['dataset'].unique()}")
    print(f"\n   Dataset breakdown:")
    print(df_new.groupby('dataset').size())
    print(f"\n   File size: {new_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   Last modified: {pd.Timestamp.fromtimestamp(new_path.stat().st_mtime)}")
else:
    print("   ✗ DOES NOT EXIST")

print(f"\n{'=' * 80}")
print("RECOMMENDATION")
print("=" * 80)

if legacy_path.exists() and not new_path.exists():
    print("\n✓ Your metadata is in the LEGACY location")
    print("\nOption A: Copy to new location (RECOMMENDED)")
    print(f"  mkdir -p {new_path.parent}")
    print(f"  cp {legacy_path} {new_path}")
    print("\nOption B: Re-run extract_metadata.py without --output")
    print("  cd /home/boshra95/NSRR-tools")
    print("  python scripts/extract_metadata.py --datasets stages shhs apples mros")
    print("  (This will save to the config-based location)")

elif legacy_path.exists() and new_path.exists():
    print("\n☑ Both files exist! You need to decide which one to keep.")
    print("\nCompare file sizes and dates above.")
    print("Recommended: Use the newer/larger one and delete the other.")

else:
    print("\n⚠ No metadata files found. You need to run extract_metadata.py")
