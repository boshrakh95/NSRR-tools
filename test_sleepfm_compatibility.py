#!/usr/bin/env python
"""
Quick test to check if our HDF5 files are compatible with sleepfm-clinical
"""

import h5py
import sys
import os

# Add sleepfm-clinical to path
sys.path.insert(0, '/home/boshra95/sleepfm-clinical')
sys.path.insert(0, '/home/boshra95/sleepfm-clinical/sleepfm')

from models.dataset import SetTransformerDataset
from utils import load_config, load_data

print("=" * 80)
print("Testing HDF5 Compatibility with SleepFM-Clinical")
print("=" * 80)

# Test 1: Check file extension compatibility (.h5 vs .hdf5)
print("\n[Test 1] File Extension Check")
print("-" * 80)
print("Our files use: .h5")
print("SleepFM demo uses: .hdf5")
print("Result: ✓ Both .h5 and .hdf5 are identical HDF5 format, just different extensions")

# Test 2: Inspect our HDF5 file structure
print("\n[Test 2] Our HDF5 File Structure")
print("-" * 80)
test_file = "/home/boshra95/scratch/psg/mros/derived/hdf5_signals/AA0001.h5"

if not os.path.exists(test_file):
    print(f"ERROR: Test file not found: {test_file}")
    sys.exit(1)

print(f"Opening file: {test_file}")
with h5py.File(test_file, 'r') as hf:
    print(f"\nDatasets in file:")
    for key in list(hf.keys())[:10]:  # Show first 10
        dataset = hf[key]
        print(f"  - {key:30s} | shape: {dataset.shape} | dtype: {dataset.dtype}")
    
    if len(hf.keys()) > 10:
        print(f"  ... and {len(hf.keys()) - 10} more datasets")
    
    print(f"\nTotal datasets: {len(hf.keys())}")

# Test 3: Check what SleepFM expects
print("\n[Test 3] SleepFM Expected Structure")
print("-" * 80)

# Load SleepFM config and channel groups
model_path = "/home/boshra95/sleepfm-clinical/sleepfm/checkpoints/model_base"
config_path = os.path.join(model_path, "config.json")
channel_groups_path = "/home/boshra95/sleepfm-clinical/sleepfm/configs/channel_groups.json"

print(f"Loading config from: {config_path}")
print(f"Loading channel groups from: {channel_groups_path}")

config = load_config(config_path)
channel_groups = load_data(channel_groups_path)

print(f"\nModality types: {config['modality_types']}")
print(f"\nChannel groups:")
for modality, channels in channel_groups.items():
    print(f"  {modality}: {len(channels)} channels")
    print(f"    Examples: {channels[:5]}")

# Test 4: Try loading our file with their dataset class
print("\n[Test 4] Loading Our File with SleepFM Dataset Class")
print("-" * 80)

try:
    print("Creating SetTransformerDataset with our file...")
    dataset = SetTransformerDataset(
        config, 
        channel_groups, 
        hdf5_paths=[test_file], 
        split="test"
    )
    
    print(f"✓ Dataset created successfully!")
    print(f"  Total samples: {len(dataset)}")
    
    if len(dataset) > 0:
        print("\n  Attempting to load first sample...")
        sample = dataset[0]
        
        if sample is None:
            print("  ✗ Sample returned None (might be all artifacts or missing channels)")
        else:
            data, file_path, dset_names, chunk_start, modalities_length = sample
            print(f"  ✓ Sample loaded successfully!")
            print(f"    File: {file_path}")
            print(f"    Available channels: {dset_names}")
            print(f"    Chunk start: {chunk_start}")
            print(f"    Number of modalities: {len(data)}")
            for i, modality_data in enumerate(data):
                print(f"    Modality {i} shape: {modality_data.shape}")
    else:
        print("  No samples available (file might not have required channels)")
        
except Exception as e:
    print(f"  ✗ Error loading dataset: {str(e)}")
    import traceback
    traceback.print_exc()

# Test 5: Direct comparison with SleepFM's preprocessed file
print("\n[Test 5] Compare with SleepFM Demo File Structure")
print("-" * 80)

sleepfm_demo_file = "/home/boshra95/sleepfm-clinical/notebooks/demo_data/demo_psg.hdf5"
if os.path.exists(sleepfm_demo_file):
    print(f"Opening SleepFM demo file: {sleepfm_demo_file}")
    with h5py.File(sleepfm_demo_file, 'r') as hf:
        print(f"\nDatasets in SleepFM demo file:")
        for key in list(hf.keys())[:10]:
            dataset = hf[key]
            print(f"  - {key:30s} | shape: {dataset.shape} | dtype: {dataset.dtype}")
        print(f"\nTotal datasets: {len(hf.keys())}")
else:
    print(f"SleepFM demo file not found: {sleepfm_demo_file}")
    print("(Run their demo.py to generate it for comparison)")

print("\n" + "=" * 80)
print("Test Complete!")
print("=" * 80)
