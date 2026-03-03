#!/usr/bin/env python3
"""Profile signal processing to identify bottlenecks."""

import time
import numpy as np
from pathlib import Path
import sys

src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from nsrr_tools.utils.config import Config
from nsrr_tools.core.signal_processor import SignalProcessor
from nsrr_tools.datasets.stages_adapter import STAGESAdapter
import mne

def profile_processing_steps():
    """Profile each step of signal processing."""
    
    config = Config()
    processor = SignalProcessor(config)
    adapter = STAGESAdapter(config)
    
    # Get first EDF file
    edf_files = adapter.find_edf_files()
    if not edf_files:
        print("No EDF files found")
        return
    
    subject_id, edf_path = edf_files[0]
    print(f"\nProfiling: {edf_path.name}")
    print("="*80)
    
    # Step 1: Load EDF
    t0 = time.time()
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose='ERROR')
    t_load_lazy = time.time() - t0
    print(f"1. Load EDF (lazy):        {t_load_lazy:.3f}s")
    
    # Step 2: Load EDF with preload
    t0 = time.time()
    raw_preload = mne.io.read_raw_edf(str(edf_path), preload=True, verbose='ERROR')
    t_load_preload = time.time() - t0
    print(f"2. Load EDF (preload):     {t_load_preload:.3f}s")
    
    # Step 3: Channel detection
    t0 = time.time()
    channel_mapping = processor.channel_mapper.detect_channels_from_list(raw.ch_names)
    t_channel_detect = time.time() - t0
    print(f"3. Channel detection:      {t_channel_detect:.3f}s")
    
    # Step 4: Modality grouping
    t0 = time.time()
    modality_groups = processor.modality_detector.group_channels_by_modality(channel_mapping)
    t_modality = time.time() - t0
    print(f"4. Modality grouping:      {t_modality:.3f}s")
    
    # Step 5: Apply SleepFM limits
    t0 = time.time()
    channel_mapping = processor._apply_sleepfm_limits(channel_mapping, modality_groups)
    t_limits = time.time() - t0
    print(f"5. Apply SleepFM limits:   {t_limits:.3f}s")
    
    print(f"\nSelected {len(channel_mapping)} channels for processing")
    print("="*80)
    
    # Profile per-channel processing with lazy loading
    print("\nPer-channel processing (lazy loading):")
    for i, (std_name, raw_name) in enumerate(list(channel_mapping.items())[:3]):
        modality = processor._get_channel_modality(std_name, modality_groups)
        
        print(f"\n  Channel {i+1}: {std_name} ({raw_name}) - {modality}")
        
        # Get raw signal
        t0 = time.time()
        ch_idx = raw.ch_names.index(raw_name)
        signal_data, _ = raw[ch_idx, :]
        signal_data = signal_data.flatten()
        original_sr = raw.info['sfreq']
        t_read = time.time() - t0
        print(f"    Read signal:           {t_read:.3f}s ({len(signal_data):,} samples @ {original_sr} Hz)")
        
        # Filtering
        t0 = time.time()
        if modality in processor.FILTER_PARAMS:
            filter_params = processor.FILTER_PARAMS[modality]
            signal_filtered = processor._bandpass_filter(
                signal_data,
                original_sr,
                filter_params['low'],
                filter_params['high'],
                filter_params['order']
            )
        else:
            signal_filtered = signal_data
        t_filter = time.time() - t0
        print(f"    Bandpass filter:       {t_filter:.3f}s")
        
        # Resampling
        t0 = time.time()
        if original_sr != processor.TARGET_SR:
            signal_resampled = processor._resample_signal(signal_filtered, original_sr, processor.TARGET_SR)
        else:
            signal_resampled = signal_filtered
        t_resample = time.time() - t0
        samples_ratio = len(signal_resampled) / len(signal_data)
        print(f"    Resample:              {t_resample:.3f}s ({original_sr}Hz → {processor.TARGET_SR}Hz, ratio={samples_ratio:.3f})")
        
        # Normalization
        t0 = time.time()
        signal_normalized, stats = processor._normalize_signal(signal_resampled)
        t_normalize = time.time() - t0
        print(f"    Z-score normalize:     {t_normalize:.3f}s")
        
        # Convert dtype
        t0 = time.time()
        signal_final = signal_normalized.astype(processor.OUTPUT_DTYPE)
        t_dtype = time.time() - t0
        print(f"    Convert to float16:    {t_dtype:.3f}s")
        
        total = t_read + t_filter + t_resample + t_normalize + t_dtype
        print(f"    Total:                 {total:.3f}s")
        print(f"    Breakdown: Read={t_read/total*100:.1f}%, Filter={t_filter/total*100:.1f}%, "
              f"Resample={t_resample/total*100:.1f}%, Normalize={t_normalize/total*100:.1f}%")
    
    # Compare with preloaded raw
    print("\n" + "="*80)
    print("\nPer-channel processing (preloaded):")
    for i, (std_name, raw_name) in enumerate(list(channel_mapping.items())[:3]):
        modality = processor._get_channel_modality(std_name, modality_groups)
        
        print(f"\n  Channel {i+1}: {std_name} ({raw_name}) - {modality}")
        
        # Get raw signal from preloaded
        t0 = time.time()
        ch_idx = raw_preload.ch_names.index(raw_name)
        signal_data, _ = raw_preload[ch_idx, :]
        signal_data = signal_data.flatten()
        original_sr = raw_preload.info['sfreq']
        t_read = time.time() - t0
        print(f"    Read signal:           {t_read:.3f}s ({len(signal_data):,} samples)")
        
        # Rest of processing
        t0 = time.time()
        if modality in processor.FILTER_PARAMS:
            filter_params = processor.FILTER_PARAMS[modality]
            signal_filtered = processor._bandpass_filter(
                signal_data,
                original_sr,
                filter_params['low'],
                filter_params['high'],
                filter_params['order']
            )
        else:
            signal_filtered = signal_data
        t_filter = time.time() - t0
        
        t0 = time.time()
        if original_sr != processor.TARGET_SR:
            signal_resampled = processor._resample_signal(signal_filtered, original_sr, processor.TARGET_SR)
        else:
            signal_resampled = signal_filtered
        t_resample = time.time() - t0
        
        total = t_read + t_filter + t_resample
        print(f"    Filter + Resample:     {t_filter + t_resample:.3f}s")
        print(f"    Total:                 {total:.3f}s")
    
    print("\n" + "="*80)
    print("\nSummary:")
    print(f"  Lazy loading overhead:   ~{t_load_lazy:.3f}s initial + per-channel disk reads")
    print(f"  Preload all at once:     {t_load_preload:.3f}s upfront")
    print(f"  Number of channels:      {len(channel_mapping)}")
    print(f"\n  Recommendation: {'Preload if < 20 channels' if len(channel_mapping) < 20 else 'Lazy load for memory efficiency'}")

if __name__ == "__main__":
    profile_processing_steps()
