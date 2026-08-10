#!/usr/bin/env python3
"""Quick test of annotation path finding."""

import sys
from pathlib import Path
sys.path.insert(0, '/home/boshra95/NSRR-tools/src')

from nsrr_tools.utils.config import Config
from nsrr_tools.datasets.stages_adapter import STAGESAdapter
from nsrr_tools.datasets.apples_adapter import APPLESAdapter

print("=" * 60)
print("Testing Annotation Path Finding")
print("=" * 60)

# Test STAGES
print("\n[STAGES]")
config = Config()
adapter = STAGESAdapter(config)
edf_files = adapter.find_edf_files()[:3]
print(f"Found {len(edf_files)} EDFs")
for subject_id, edf_path in edf_files:
    annotation = adapter.find_annotation_file(subject_id, edf_path=edf_path)
    status = "✓" if annotation else "✗"
    print(f"  {status} {subject_id}: {annotation.name if annotation else 'NOT FOUND'}")

# Test APPLES
print("\n[APPLES]")
adapter = APPLESAdapter(config)
edf_files = adapter.find_edf_files()[:3]
print(f"Found {len(edf_files)} EDFs")
for subject_id, edf_path in edf_files:
    annotation = adapter.find_annotation_file(subject_id, edf_path=edf_path)
    status = "✓" if annotation else "✗"
    print(f"  {status} {subject_id}: {annotation.name if annotation else 'NOT FOUND'}")

print("\n" + "=" * 60)
print("Test Complete")
print("=" * 60)
