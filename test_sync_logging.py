#!/usr/bin/env python3
"""Test improved logging for STAGES annotation synchronization."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from nsrr_tools.datasets.stages_adapter import STAGESAdapter
from nsrr_tools.core.annotation_processor import AnnotationProcessor
from nsrr_tools.utils.config import Config

def test_bogn00004():
    """Test BOGN00004 which has annotations shorter than signal."""
    print("="*70)
    print("Testing Improved Synchronization Logging")
    print("="*70)
    
    config = Config()
    adapter = STAGESAdapter(config)
    
    # Parse annotations
    csv_file = Path('/home/boshra95/scratch/nsrr_downloads/stages/original/STAGES PSGs/BOGN/BOGN00004.csv')
    edf_file = Path('/home/boshra95/scratch/nsrr_downloads/stages/original/STAGES PSGs/BOGN/BOGN00004.edf')
    
    if not csv_file.exists() or not edf_file.exists():
        print("Test files not found")
        return
    
    print(f"\nSubject: BOGN00004")
    print("-" * 70)
    
    # Parse annotations
    annot_data = adapter.parse_annotations(csv_file)
    print(f"\n1. Parsed {len(annot_data['stages'])} stage annotations from CSV")
    
    # Show time range
    if annot_data['stages']:
        first_start = annot_data['stages'][0]['start']
        last_stage = annot_data['stages'][-1]
        last_end = last_stage['start'] + last_stage.get('duration', 30)
        
        print(f"   Annotation timespan: {first_start:.0f}s to {last_end:.0f}s")
        print(f"   Coverage: {(last_end - first_start)/3600:.1f} hours")
        print(f"   Expected epochs: {int(np.ceil((last_end - first_start) / 30))}")
    
    # Process with annotation processor (includes sync checking)
    annotation_processor = AnnotationProcessor(adapter)
    output_path = Path('/tmp/test_bogn00004.npy')
    
    print(f"\n2. Processing with synchronization check...")
    print("-" * 70)
    
    result = annotation_processor.process_annotations(
        annotation_path=csv_file,
        edf_path=edf_file,
        output_path=output_path,
        validate_sync=True
    )
    
    print("\n" + "-" * 70)
    print("3. Final Result:")
    print(f"   Total epochs: {result['num_epochs']}")
    print(f"   Scored epochs: {result['scored_epochs']}")
    print(f"   Unscored epochs: {result['unscored_epochs']}")
    print(f"   Coverage: {result['scoring_coverage']:.1%}")
    print(f"   Sync status: {result.get('sync_status', 'unknown')}")
    
    # Cleanup
    if output_path.exists():
        output_path.unlink()
    
    print("\n" + "="*70)
    print("Explanation:")
    print("="*70)
    print("The annotations cover the actual sleep period (~7.6 hours),")
    print("but the recording continued beyond that (~10.6 hours total).")
    print("The padding adds unscored (-1) epochs for the uncovered time.")
    print("This is NORMAL and expected for clinical sleep recordings.")

if __name__ == '__main__':
    test_bogn00004()
