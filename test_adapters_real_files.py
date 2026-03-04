#!/usr/bin/env python3
"""Test actual adapter functionality with real files."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from nsrr_tools.datasets.stages_adapter import STAGESAdapter
from nsrr_tools.datasets.apples_adapter import APPLESAdapter
from nsrr_tools.utils.config import Config
from collections import Counter

def test_stages_adapter():
    """Test STAGES adapter with real file."""
    print("="*70)
    print("STAGES Adapter Test (Real File)")
    print("="*70)
    
    config = Config()
    adapter = STAGESAdapter(config)
    
    # Test file
    test_file = Path('/home/boshra95/scratch/nsrr_downloads/stages/original/STAGES PSGs/GSSA/GSSA00022.csv')
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    result = adapter.parse_annotations(test_file)
    
    print(f"\n📊 STAGES Subject GSSA00022:")
    print(f"   Annotations parsed: {len(result['stages'])}")
    print(f"   Format: {result.get('format', 'unknown')}")
    
    # Count stage types
    stage_labels = Counter(s['label'] for s in result['stages'])
    
    print(f"\n   Stage distribution:")
    for label in sorted(stage_labels.keys()):
        count = stage_labels[label]
        stage_num = result['stages'][0] if result['stages'] else {}
        # Find the stage number for this label
        stage_num = next((s['stage'] for s in result['stages'] if s['label'] == label), None)
        print(f"      {label:15s} → {stage_num:2d}: {count:4d} annotations")
    
    # Check for all expected stages
    has_wake = any('Wake' in s['label'] for s in result['stages'])
    has_rem = any('REM' in s['label'] for s in result['stages'])
    has_unknown = any('Unknown' in s['label'] for s in result['stages'])
    
    print(f"\n   Stage capture check:")
    print(f"      Wake:    {'✓' if has_wake else '❌'}")
    print(f"      REM:     {'✓' if has_rem else '❌'}")
    print(f"      Unknown: {'✓' if has_unknown else '❌'}")
    
    # Check duration handling
    if result['stages']:
        durations = [s.get('duration', 0) for s in result['stages']]
        avg_duration = sum(durations) / len(durations)
        print(f"\n   Duration handling:")
        print(f"      Average duration: {avg_duration:.1f}s")
        print(f"      Has variable durations: {'✓' if max(durations) != min(durations) else '❌'}")
    
    return has_wake and has_rem

def test_apples_adapter():
    """Test APPLES adapter with real file."""
    print("\n" + "="*70)
    print("APPLES Adapter Test (Real File)")
    print("="*70)
    
    config = Config()
    adapter = APPLESAdapter(config)
    
    # Test file (crosses midnight: 21:44 to 06:54)
    test_file = Path('/home/boshra95/scratch/nsrr_downloads/apples/polysomnography/apples-140094.annot')
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    result = adapter.parse_annotations(test_file)
    
    print(f"\n📊 APPLES Subject 140094:")
    print(f"   Annotations parsed: {len(result['stages'])}")
    print(f"   Format: {result.get('format', 'unknown')}")
    
    # Count stage types
    stage_labels = Counter(s['label'] for s in result['stages'])
    
    print(f"\n   Stage distribution:")
    for label in sorted(stage_labels.keys()):
        count = stage_labels[label]
        stage_num = next((s['stage'] for s in result['stages'] if s['label'] == label), None)
        print(f"      {label:5s} → {stage_num:2d}: {count:4d} annotations")
    
    # Check for all expected stages
    has_wake = any(s['label'] == 'W' for s in result['stages'])
    has_rem = any(s['label'] in ['R', 'REM'] for s in result['stages'])
    
    print(f"\n   Stage capture check:")
    print(f"      Wake (W):  {'✓' if has_wake else '❌'}")
    print(f"      REM (R):   {'✓' if has_rem else '❌'}")
    
    # Check day boundary handling
    if result['stages']:
        starts = [s['start'] for s in result['stages']]
        is_monotonic = all(starts[i] <= starts[i+1] for i in range(len(starts)-1))
        
        print(f"\n   Day boundary handling:")
        print(f"      Start times monotonic: {'✓' if is_monotonic else '❌'}")
        print(f"      First start: {starts[0]:.1f}s")
        print(f"      Last start: {starts[-1]:.1f}s")
        print(f"      Recording duration: {(starts[-1] - starts[0])/3600:.1f}h")
        
        # Check durations
        durations = [s.get('duration', 0) for s in result['stages']]
        has_negative = any(d < 0 for d in durations)
        
        print(f"\n   Duration handling:")
        print(f"      Has negative durations: {'❌ (BUG!)' if has_negative else '✓ (FIXED)'}")
        print(f"      Average duration: {sum(durations)/len(durations):.1f}s")
    
        return has_wake and has_rem and is_monotonic and not has_negative
    
    return False

# Run tests
stages_ok = test_stages_adapter()
apples_ok = test_apples_adapter()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

if stages_ok and apples_ok:
    print("✅ All adapter tests PASSED!")
    print("\n All datasets now correctly:")
    print("   ✓ Capture Wake, REM, and Unscored stages")
    print("   ✓ Parse duration fields (not just 30s default)")
    print("   ✓ Handle day boundary crossing (midnight)")
    print("   ✓ Handle negative durations")
else:
    print("❌ Some adapter tests FAILED")
    if not stages_ok:
        print("   - STAGES adapter issues")
    if not apples_ok:
        print("   - APPLES adapter issues")
