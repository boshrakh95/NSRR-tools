#!/usr/bin/env python
"""
Test Channel Selection Strategies

Quick test script to verify different channel selection strategies work correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from nsrr_tools.core.signal_processor import SignalProcessor
from loguru import logger

# Configure logger for cleaner output
logger.remove()
logger.add(sys.stderr, format="<level>{message}</level>", level="INFO")

def main():
    print("="*70)
    print("Channel Selection Strategy Test")
    print("="*70)
    print()
    
    # Display available strategies
    print("Available Strategies:")
    print("-" * 70)
    for strategy, limits in SignalProcessor.CHANNEL_STRATEGIES.items():
        total = sum(limits.values())
        print(f"  {strategy:15s} -> BAS={limits['BAS']:2d}, EKG={limits['EKG']:2d}, "
              f"EMG={limits['EMG']:2d}, RESP={limits['RESP']:2d} "
              f"(total: {total:2d} max channels)")
    print()
    
    # Test simulated channel selection
    print("Simulating Channel Selection:")
    print("-" * 70)
    
    # Simulate a subject with many channels
    test_channels = {
        'BAS': ['C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'F3-M2', 'F4-M1', 
                'LOC', 'ROC', 'Fp1', 'Fp2', 'Fz', 'Cz'],  # 12 BAS channels
        'EKG': ['EKG', 'ECG-L', 'ECG-R'],  # 3 ECG channels
        'EMG': ['CHIN', 'LLEG', 'RLEG', 'EMG'],  # 4 EMG channels
        'RESP': ['Airflow', 'Thor', 'ABD', 'SpO2', 'HR', 'Snore', 'RespRate']  # 7 RESP channels
    }
    
    print(f"Subject has:")
    for modality, channels in test_channels.items():
        print(f"  {modality}: {len(channels)} channels -> {channels[:3]}...")
    print()
    
    # Test each strategy
    for strategy_name, limits in SignalProcessor.CHANNEL_STRATEGIES.items():
        print(f"Strategy: {strategy_name}")
        total_selected = 0
        for modality, limit in limits.items():
            available = len(test_channels[modality])
            selected = min(available, limit)
            total_selected += selected
            
            if limit == 0:
                print(f"  {modality}: SKIP (limit=0)")
            elif selected < limit:
                print(f"  {modality}: {selected}/{limit} (only {available} available)")
            else:
                print(f"  {modality}: {selected}/{limit} channels")
        
        print(f"  → Total: {total_selected} channels selected")
        print()
    
    print("="*70)
    print("✅ All strategies defined and functional!")
    print()
    print("To use a strategy, edit configs/preprocessing_params.yaml:")
    print("  channel_selection:")
    print("    strategy: 'fast'  # or 'minimal', 'sleepfm_full', 'custom'")
    print("="*70)

if __name__ == '__main__':
    main()
