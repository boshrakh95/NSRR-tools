#!/usr/bin/env python3
"""Test script to verify XML annotation parsing now captures all stages."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from nsrr_tools.datasets.mros_adapter import MrOSAdapter
from nsrr_tools.datasets.shhs_adapter import SHHSAdapter
from nsrr_tools.utils.config import Config

def test_mros_parsing():
    """Test MrOS XML parsing."""
    print("=" * 80)
    print("Testing MrOS Annotation Parsing")
    print("=" * 80)
    
    config = Config()
    adapter = MrOSAdapter(config, visit=2)
    
    # Test file
    test_file = Path('/home/boshra95/scratch/nsrr_downloads/mros/polysomnography/annotations-events-nsrr/visit2/mros-visit2-aa2201-nsrr.xml')
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    result = adapter.parse_annotations(test_file)
    
    print(f"\n📊 MrOS Subject aa2201:")
    print(f"   Total annotations parsed: {len(result['stages'])}")
    print(f"   Format: {result.get('format', 'unknown')}")
    
    # Count stage types
    stage_counts = {}
    for stage in result['stages']:
        label = stage['label']
        stage_counts[label] = stage_counts.get(label, 0) + 1
    
    print(f"\n   Stage distribution:")
    for label, count in sorted(stage_counts.items()):
        print(f"      {label}: {count} epochs")
    
    # Check for critical stages
    has_rem = any('REM' in s['label'] for s in result['stages'])
    has_wake = any('Wake' in s['label'] for s in result['stages'])
    
    print(f"\n   ✓ REM sleep captured: {'YES ✓' if has_rem else 'NO ❌'}")
    print(f"   ✓ Wake captured: {'YES ✓' if has_wake else 'NO ❌'}")
    
    return has_rem and has_wake

def test_shhs_parsing():
    """Test SHHS XML parsing."""
    print("\n" + "=" * 80)
    print("Testing SHHS Annotation Parsing")
    print("=" * 80)
    
    config = Config()
    adapter = SHHSAdapter(config)
    
    # Test file
    test_file = Path('/home/boshra95/scratch/nsrr_downloads/shhs/polysomnography/annotations-events-nsrr/shhs1/shhs1-200001-nsrr.xml')
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    result = adapter.parse_annotations(test_file)
    
    print(f"\n📊 SHHS Subject 200001:")
    print(f"   Total annotations parsed: {len(result['stages'])}")
    print(f"   Format: {result.get('format', 'unknown')}")
    
    # Count stage types
    stage_counts = {}
    for stage in result['stages']:
        label = stage['label']
        stage_counts[label] = stage_counts.get(label, 0) + 1
    
    print(f"\n   Stage distribution:")
    for label, count in sorted(stage_counts.items()):
        print(f"      {label}: {count} epochs")
    
    # Check for critical stages
    has_rem = any('REM' in s['label'] for s in result['stages'])
    has_wake = any('Wake' in s['label'] for s in result['stages'])
    
    print(f"\n   ✓ REM sleep captured: {'YES ✓' if has_rem else 'NO ❌'}")
    print(f"   ✓ Wake captured: {'YES ✓' if has_wake else 'NO ❌'}")
    
    return has_rem and has_wake

if __name__ == '__main__':
    mros_ok = test_mros_parsing()
    shhs_ok = test_shhs_parsing()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if mros_ok and shhs_ok:
        print("✅ All tests passed! Wake and REM sleep are now captured correctly.")
        sys.exit(0)
    else:
        print("❌ Tests failed! Some stages are still not being captured.")
        sys.exit(1)
