#!/usr/bin/env python3
"""Debug SHHS merge issue."""

import pandas as pd
from pathlib import Path

# Load CSV
csv_path = Path('/scratch/boshra95/nsrr_downloads/shhs/datasets/shhs-harmonized-dataset-0.21.0.csv')
csv = pd.read_csv(csv_path)

print("CSV STRUCTURE")
print("="*80)
print(f"nsrrid type: {csv['nsrrid'].dtype}")
print(f"nsrrid range: {csv['nsrrid'].min()} - {csv['nsrrid'].max()}")
print(f"Unique nsrrids: {csv['nsrrid'].nunique()}")

# Check specific IDs
test_ids = [200077, 200078, 200001, 205804]
print("\nChecking specific IDs:")
for nid in test_ids:
    rows = csv[csv['nsrrid'] == nid]
    if len(rows) > 0:
        visits = rows['visitnumber'].tolist()
        print(f"  {nid}: FOUND, visits={visits}")
    else:
        print(f"  {nid}: NOT FOUND")

# Check type issue in adapter
print("\n" + "="*80)
print("TYPE MISMATCH TEST")
print("="*80)
print("When adapter extracts '200077' from filename, it's a STRING")
print("When CSV loads nsrrid, it's an INT")
print("\nTesting:")
print(f"  '200077' in csv['nsrrid'].values: {'200077' in csv['nsrrid'].values}")
print(f"  200077 in csv['nsrrid'].values: {200077 in csv['nsrrid'].values}")
print(f"  int('200077') in csv['nsrrid'].values: {int('200077') in csv['nsrrid'].values}")
