#!/usr/bin/env python3
"""Test that Duration fields are now parsed correctly."""

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

def test_duration_parsing(xml_file, dataset_name):
    """Parse XML with durations and show epoch coverage."""
    print(f"\n{'='*70}")
    print(f"{dataset_name}: {xml_file.name}")
    print('='*70)
    
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
        duration_elem = scored_event.find('Duration')
        
        if event_concept is not None and event_concept.text in stage_map:
            start_time = float(start.text) if start is not None else 0
            duration = float(duration_elem.text) if duration_elem is not None else 30.0
            
            stages.append({
                'start': start_time,
                'duration': duration,
                'stage': stage_map[event_concept.text],
                'label': event_concept.text
            })
    
    print(f"Parsed {len(stages)} stage annotations from XML")
    
    # Calculate without durations (OLD BUG)
    if stages:
        last_start = stages[-1]['start']
        total_duration_old = last_start + 30
        epochs_old = int(np.ceil(total_duration_old / 30))
        coverage_old = len(stages)  # Only counts annotations, not epochs filled
    
    # Calculate WITH durations (NEW FIX)
    if stages:
        last_stage = stages[-1]
        last_start = last_stage['start']
        last_duration = last_stage['duration']
        total_duration_new = last_start + last_duration
        epochs_new = int(np.ceil(total_duration_new / 30))
        
        # Count actual epochs filled
        filled_epochs = sum(int(np.ceil(s['duration'] / 30)) for s in stages)
    
    print(f"\nOLD CODE (without Duration parsing):")
    print(f"  Epoch array size: {epochs_old} epochs")
    print(f"  Filled epochs: {len(stages)} (treated each annotation as 1 epoch)")
    print(f"  Coverage: {len(stages)}/{epochs_old} = {len(stages)/epochs_old*100:.1f}%")
    
    print(f"\nNEW CODE (with Duration parsing):")
    print(f"  Epoch array size: {epochs_new} epochs")
    print(f"  Filled epochs: {filled_epochs} (using actual durations)")
    print(f"  Coverage: {filled_epochs}/{epochs_new} = {filled_epochs/epochs_new*100:.1f}%")
    
    # Show some examples
    print(f"\nExample annotations with durations:")
    for i, stage in enumerate(stages[:5]):
        num_epochs = int(np.ceil(stage['duration'] / 30))
        print(f"  {stage['label']:25s} at {stage['start']:7.1f}s: {stage['duration']:6.1f}s = {num_epochs} epochs")
    
    return filled_epochs, epochs_new

# Test SHHS
shhs_file = Path('/home/boshra95/scratch/nsrr_downloads/shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-200001-nsrr.xml')
if shhs_file.exists():
    filled, total = test_duration_parsing(shhs_file, 'SHHS Subject 200001')
    
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print('='*70)
    if filled == total:
        print("✅ PERFECT! All epochs in array are now scored (no gaps).")
    elif filled / total > 0.9:
        print(f"✅ GOOD! {filled/total*100:.1f}% coverage (nearly continuous).")
    else:
        print(f"⚠️  {filled/total*100:.1f}% coverage (still some gaps).")
else:
    print(f"Test file not found: {shhs_file}")
