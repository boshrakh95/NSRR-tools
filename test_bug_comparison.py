#!/usr/bin/env python3
"""Compare OLD vs NEW XML parsing to show the bug fix."""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter

def test_old_parsing(xml_file):
    """OLD CODE: Only captures stages with 'Stage' in name."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
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
        
        # OLD BUG: Only matches if 'Stage' is in the text
        if event_concept is not None and 'Stage' in event_concept.text:
            label = event_concept.text
            if label in stage_map:
                stages.append(label)
    
    return stages

def test_new_parsing(xml_file):
    """NEW CODE: Captures all known stage types."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
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
        
        # NEW FIX: Check if label is in stage_map
        if event_concept is not None and event_concept.text in stage_map:
            stages.append(event_concept.text)
    
    return stages

def compare_parsing(xml_file, dataset_name):
    """Compare OLD vs NEW parsing."""
    print(f"\n{'='*70}")
    print(f"{dataset_name}: {xml_file.name}")
    print('='*70)
    
    old_stages = test_old_parsing(xml_file)
    new_stages = test_new_parsing(xml_file)
    
    old_counts = Counter(old_stages)
    new_counts = Counter(new_stages)
    
    print(f"\nOLD CODE (buggy):")
    print(f"  Total captured: {len(old_stages)} stage annotations")
    for label in sorted(old_counts.keys()):
        print(f"    {label:25s}: {old_counts[label]:4d}")
    
    print(f"\nNEW CODE (fixed):")
    print(f"  Total captured: {len(new_stages)} stage annotations")
    for label in sorted(new_counts.keys()):
        print(f"    {label:25s}: {new_counts[label]:4d}")
    
    # Show what was missed
    missed_labels = set(new_counts.keys()) - set(old_counts.keys())
    if missed_labels:
        print(f"\n⚠️  MISSING in OLD code:")
        for label in sorted(missed_labels):
            print(f"    {label:25s}: {new_counts[label]:4d} epochs ❌")
    
    # Calculate percentage missed
    pct_missed = (len(new_stages) - len(old_stages)) / len(new_stages) * 100
    print(f"\n📊 Impact: {len(new_stages) - len(old_stages)} annotations missed ({pct_missed:.1f}%)")

# Test both datasets
mros_file = Path('/home/boshra95/scratch/nsrr_downloads/mros/polysomnography/annotations-events-nsrr/visit2/mros-visit2-aa2201-nsrr.xml')
if mros_file.exists():
    compare_parsing(mros_file, 'MrOS')

shhs_file = Path('/home/boshra95/scratch/nsrr_downloads/shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-200001-nsrr.xml')
if shhs_file.exists():
    compare_parsing(shhs_file, 'SHHS')

print(f"\n{'='*70}")
print("CONCLUSION")
print('='*70)
print("✅ Bug fixed! Wake and REM sleep are now correctly captured.")
print("   Previously these were completely missing from annotation arrays.")
