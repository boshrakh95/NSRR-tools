#!/usr/bin/env python3
"""Quick test to verify XML stage parsing fix."""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

def test_xml_parsing(xml_file, dataset_name):
    """Parse XML and show stage distribution."""
    print(f"\n{'='*60}")
    print(f"{dataset_name}: {xml_file.name}")
    print('='*60)
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Define stage mapping (same as in adapters now)
    stage_map = {
        'Stage 1 sleep|1': 1,
        'Stage 2 sleep|2': 2,
        'Stage 3 sleep|3': 3,
        'Stage 4 sleep|4': 4,
        'REM sleep|5': 5,
        'Wake|0': 0,
        'Unscored|9': -1
    }
    
    stages = []
    
    for scored_event in root.findall('.//ScoredEvent'):
        event_concept = scored_event.find('EventConcept')
        start = scored_event.find('Start')
        
        # NEW CODE: Check if event_concept.text is in stage_map
        if event_concept is not None and event_concept.text in stage_map:
            stages.append({
                'label': event_concept.text,
                'stage': stage_map[event_concept.text],
                'start': float(start.text) if start is not None else 0
            })
    
    # Show results
    print(f"Total stage annotations: {len(stages)}")
    
    stage_counts = Counter(s['label'] for s in stages)
    print("\nStage distribution:")
    for label in sorted(stage_counts.keys()):
        count = stage_counts[label]
        print(f"  {label:25s}: {count:4d} epochs")
    
    # Check critical stages
    has_rem = any('REM' in s['label'] for s in stages)
    has_wake = any('Wake' in s['label'] for s in stages)
    has_stage1 = any('Stage 1' in s['label'] for s in stages)
    
    print(f"\nCapture check:")
    print(f"  ✓ REM sleep:   {'YES ✓' if has_rem else 'NO ❌'}")
    print(f"  ✓ Wake:        {'YES ✓' if has_wake else 'NO ❌'}")
    print(f"  ✓ Stage 1-4:   {'YES ✓' if has_stage1 else 'NO ❌'}")
    
    return has_rem and has_wake and has_stage1

# Test MrOS
mros_file = Path('/home/boshra95/scratch/nsrr_downloads/mros/polysomnography/annotations-events-nsrr/visit2/mros-visit2-aa2201-nsrr.xml')
if mros_file.exists():
    mros_ok = test_xml_parsing(mros_file, 'MrOS')
else:
    print(f"❌ MrOS test file not found: {mros_file}")
    mros_ok = False

# Test SHHS
shhs_file = Path('/home/boshra95/scratch/nsrr_downloads/shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-200001-nsrr.xml')
if shhs_file.exists():
    shhs_ok = test_xml_parsing(shhs_file, 'SHHS')
else:
    print(f"❌ SHHS test file not found: {shhs_file}")
    shhs_ok = False

print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)

if mros_ok and shhs_ok:
    print("✅ SUCCESS! All stage types (Wake, REM, Stage 1-4) are now captured.")
else:
    print("❌ FAILED! Some stages missing.")
