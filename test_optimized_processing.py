#!/usr/bin/env python3
"""Test optimized signal processing performance."""

import time
from pathlib import Path
import sys

src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from nsrr_tools.utils.config import Config
from nsrr_tools.core.signal_processor import SignalProcessor
from nsrr_tools.datasets.stages_adapter import STAGESAdapter

def test_processing_speed():
    """Test processing speed with optimizations."""
    
    config = Config()
    processor = SignalProcessor(config)
    adapter = STAGESAdapter(config)
    
    # Get first EDF file
    edf_files = adapter.find_edf_files()
    if not edf_files:
        print("No EDF files found")
        return
    
    subject_id, edf_path = edf_files[0]
    output_path = Path("/tmp/test_signal_processing.h5")
    
    print(f"\nTesting optimized processing:")
    print(f"  File: {edf_path.name}")
    print(f"  Strategy: {processor.channel_strategy}")
    print(f"  Max channels: {sum(processor.channel_limits.values())}")
    print("="*70)
    
    # Process
    start_time = time.time()
    result = processor.process_edf(edf_path, output_path)
    elapsed = time.time() - start_time
    
    print("\nResults:")
    print(f"  Success: {result['success']}")
    print(f"  Channels processed: {result.get('channels_processed', 0)}")
    print(f"  Duration: {result.get('duration_hours', 0):.2f} hours of data")
    print(f"  Output size: {result.get('output_size_mb', 0):.2f} MB")
    print(f"\n  Processing time: {elapsed:.2f} seconds")
    print(f"  Speed: {result.get('duration_hours', 0) * 3600 / elapsed:.1f}x realtime")
    print("="*70)
    
    # Cleanup
    if output_path.exists():
        output_path.unlink()
    
    return elapsed

if __name__ == "__main__":
    test_processing_speed()
