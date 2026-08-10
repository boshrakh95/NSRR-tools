#!/usr/bin/env python3
"""Debug EEG detection for SHHS and APPLES."""

import pyedflib
from pathlib import Path
from nsrr_tools.utils.config import Config
from nsrr_tools.datasets.shhs_adapter import SHHSAdapter
from nsrr_tools.datasets.apples_adapter import APPLESAdapter  
from nsrr_tools.core.channel_mapper import ChannelMapper
from nsrr_tools.core.modality_detector import ModalityDetector

config = Config()
mapper = ChannelMapper(config)
detector = ModalityDetector(config)

print("=" * 70)
print("SHHS EEG Detection Test")
print("=" * 70)

shhs_adapter = SHHSAdapter(config)
edf_files = shhs_adapter.find_edf_files()[:3]

for subject_id, edf_path in edf_files:
    print(f"\nSubject: {subject_id}")
    print(f"EDF: {edf_path.name}")
    
    with pyedflib.EdfReader(str(edf_path)) as edf:
        ch_names = edf.getSignalLabels()
        print(f"  Total channels: {len(ch_names)}")
        print(f"  Raw channels: {ch_names[:5]}")
        
        detected = mapper.detect_channels_from_list(ch_names)
        print(f"  Detected: {len(detected)} standardized channels")
        
        modality_groups = detector.group_channels_by_modality(detected)
        print(f"  Modalities: {list(modality_groups.keys())}")
        
        if 'EEG' in modality_groups:
            print(f"  ✓ EEG channels: {len(modality_groups['EEG'])}")
            print(f"    {list(modality_groups['EEG'].items())[:3]}")
        else:
            print(f"  ✗ NO EEG DETECTED!")
            print(f"    All detected: {list(detected.items())[:10]}")

print("\n" + "=" * 70)
print("APPLES EEG Detection Test")
print("=" * 70)

apples_adapter = APPLESAdapter(config)
edf_files = apples_adapter.find_edf_files()[:3]

for subject_id, edf_path in edf_files:
    print(f"\nSubject: {subject_id}")
    print(f"EDF: {edf_path.name}")
    
    with pyedflib.EdfReader(str(edf_path)) as edf:
        ch_names = edf.getSignalLabels()
        print(f"  Total channels: {len(ch_names)}")
        print(f"  Raw channels: {ch_names[:5]}")
        
        detected = mapper.detect_channels_from_list(ch_names)
        print(f"  Detected: {len(detected)} standardized channels")
        
        modality_groups = detector.group_channels_by_modality(detected)
        print(f"  Modalities: {list(modality_groups.keys())}")
        
        if 'EEG' in modality_groups:
            print(f"  ✓ EEG channels: {len(modality_groups['EEG'])}")
            print(f"    {list(modality_groups['EEG'].items())[:3]}")
        else:
            print(f"  ✗ NO EEG DETECTED!")
            print(f"    All detected: {list(detected.items())[:10]}")

print("\n" + "=" * 70)
