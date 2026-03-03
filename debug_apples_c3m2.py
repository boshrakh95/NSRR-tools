#!/usr/bin/env python3
"""Debug channel detection for APPLES C3_M2."""

import pyedflib
from nsrr_tools.utils.config import Config
from nsrr_tools.datasets.apples_adapter import APPLESAdapter  
from nsrr_tools.core.channel_mapper import ChannelMapper

config = Config()
mapper = ChannelMapper(config)

apples_adapter = APPLESAdapter(config)
edf_files = apples_adapter.find_edf_files()[:1]

for subject_id, edf_path in edf_files:
    print(f"Subject: {subject_id}")
    print(f"EDF: {edf_path}")
    
    with pyedflib.EdfReader(str(edf_path)) as edf:
        ch_names = edf.getSignalLabels()
        print(f"\nAll {len(ch_names)} raw channels:")
        for i, ch in enumerate(ch_names, 1):
            print(f"  {i:2d}. {ch}")
        
        detected = mapper.detect_channels_from_list(ch_names)
        print(f"\nAll {len(detected)} detected (standardized) channels:")
        for std, raw in detected.items():
            print(f"  {std:15s} <- {raw}")
        
        # Check if C3_M2 was detected
        has_c3m2 = any('C3' in std for std in detected.keys())
        print(f"\nHas C3-related channels: {has_c3m2}")
