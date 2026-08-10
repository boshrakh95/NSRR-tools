"""
Signal Processing Optimization Recommendations
================================================

Based on profiling STAGES/SHHS/APPLES EDF files (typically 8-12 hours, 200-256 Hz).

Current Bottlenecks:
--------------------
1. Per-channel disk reads (lazy loading): ~3.4s per channel
2. Bandpass filtering (scipy.signal.butter): ~6.0s per channel  
3. Resampling (interp1d): ~2-4s per channel (estimated)
4. Total per file: ~80-120 seconds for 8 channels

Optimization Strategy:
----------------------

PRIORITY 1: Load EDF Once (CRITICAL - 10x speedup)
---------------------------------------------------
Problem: Lazy loading causes 8 separate disk reads (3.4s × 8 = 27s)
Solution: Selective preloading

Change in signal_processor.py _load_edf():
```python
def _load_edf(self, edf_path: Path, channels: Optional[List[str]] = None) -> mne.io.Raw:
    '''Load EDF with selective channel preloading.'''
    if channels and len(channels) <= 20:
        # For small channel sets, preload selected channels only
        raw = mne.io.read_raw_edf(
            str(edf_path), 
            preload=False, 
            verbose='ERROR'
        )
        # Pick only needed channels, then load
        if channels:
            try:
                raw.pick_channels(channels, ordered=False)
            except ValueError:
                pass  # Some channels may not exist
        raw.load_data()  # Load ONLY selected channels
    else:
        # For many channels, use lazy loading
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose='ERROR')
    return raw
```

Expected improvement: 27s → 8s (3x faster) for 8-channel case
Reason: Single disk read instead of 8 separate reads


PRIORITY 2: Faster Bandpass Filtering (HIGH - 5x speedup)
---------------------------------------------------------
Problem: scipy.signal.butter applies filter to entire signal (millions of samples)
Solutions:

Option A: Use MNE's built-in filtering (RECOMMENDED)
```python
def _bandpass_filter(self, signal_data, sr, low, high, order=4):
    '''Use MNE's optimized filtering (uses FFT for long signals).'''
    # MNE automatically uses FFT for long signals
    import mne
    signal_filtered = mne.filter.filter_data(
        signal_data,
        sr,
        l_freq=low,
        h_freq=high,
        method='fir',  # or 'iir' for Butterworth
        fir_design='firwin',
        verbose=False
    )
    return signal_filtered
```
Expected improvement: 6s → 1-1.5s per channel (4-6x faster)

Option B: Apply filter in chunks (if memory is an issue)
```python
def _bandpass_filter_chunked(self, signal_data, sr, low, high, order=4, chunk_size=None):
    '''Process signal in chunks to reduce memory usage.'''
    if chunk_size is None:
        chunk_size = int(sr * 300)  # 5-minute chunks
    
    # Design filter once
    sos = scipy_signal.butter(order, [low, high], btype='band', fs=sr, output='sos')
    
    # Process in overlapping chunks
    hop_size = chunk_size // 2
    filtered = np.zeros_like(signal_data)
    
    for start in range(0, len(signal_data), hop_size):
        end = min(start + chunk_size, len(signal_data))
        chunk = signal_data[start:end]
        
        # Apply filter
        filtered_chunk = scipy_signal.sosfilt(sos, chunk)
        
        # Overlap-add
        if start == 0:
            filtered[start:end] = filtered_chunk
        else:
            # Blend overlapping region
            blend_len = min(hop_size, len(filtered_chunk))
            filtered[start:start+blend_len] = (
                filtered[start:start+blend_len] * 0.5 +
                filtered_chunk[:blend_len] * 0.5
            )
            filtered[start+blend_len:end] = filtered_chunk[blend_len:]
    
    return filtered
```


PRIORITY 3: Faster Resampling (MEDIUM - 2x speedup)
---------------------------------------------------
Problem: interp1d creates function object, slow for millions of samples
Solutions:

Option A: scipy.signal.resample_poly (RECOMMENDED for integer ratios)
```python
def _resample_signal(self, signal_data, original_sr, target_sr):
    '''Use polyphase resampling for integer rate ratios.'''
    # Check if ratio is close to integer
    ratio = original_sr / target_sr
    
    if abs(ratio - round(ratio)) < 0.01:
        # Use faster polyphase resampling
        from scipy.signal import resample_poly
        down = int(round(ratio))
        up = 1
        resampled = resample_poly(signal_data, up, down, padtype='line')
        
        # Adjust length to match expected
        target_length = int(len(signal_data) * target_sr / original_sr)
        if len(resampled) != target_length:
            resampled = resampled[:target_length]
        
        return resampled
    else:
        # Fall back to linear interpolation for non-integer ratios
        # But use np.interp instead of interp1d (faster, no function object)
        original_length = len(signal_data)
        target_length = int(original_length * target_sr / original_sr)
        
        original_idx = np.arange(original_length)
        target_idx = np.linspace(0, original_length - 1, target_length)
        
        resampled = np.interp(target_idx, original_idx, signal_data)
        return resampled
```

Option B: For exact integer ratios (e.g., 200Hz → 128Hz is NOT integer)
Use scipy.signal.decimate for downsampling:
```python
from scipy.signal import decimate

# For 256Hz → 128Hz (factor of 2)
if original_sr == 256 and target_sr == 128:
    return decimate(signal_data, 2, ftype='fir')

# For 512Hz → 128Hz (factor of 4)  
if original_sr == 512 and target_sr == 128:
    return decimate(signal_data, 4, ftype='fir')
```


PRIORITY 4: Parallel Channel Processing (HIGH - Nx speedup)
----------------------------------------------------------
Problem: Channels are processed sequentially
Solution: Use multiprocessing or concurrent.futures

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def process_edf(self, edf_path, output_path, channel_mapping=None):
    '''Process EDF with parallel channel processing.'''
    # ... (channel detection code same as before)
    
    # Preload selected channels once
    raw = self._load_edf(edf_path, channels=list(channel_mapping.values()))
    
    # Process channels in parallel (use N-1 cores)
    n_workers = max(1, multiprocessing.cpu_count() - 1)
    
    processed_channels = {}
    normalization_stats = {}
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all channels
        future_to_channel = {}
        for std_name, raw_name in channel_mapping.items():
            modality = self._get_channel_modality(std_name, modality_groups)
            
            # Extract signal once
            ch_idx = raw.ch_names.index(raw_name)
            signal_data, _ = raw[ch_idx, :]
            signal_data = signal_data.flatten()
            original_sr = raw.info['sfreq']
            
            # Submit processing task
            future = executor.submit(
                self._process_channel_data,
                signal_data,
                original_sr,
                modality
            )
            future_to_channel[future] = std_name
        
        # Collect results
        for future in as_completed(future_to_channel):
            std_name = future_to_channel[future]
            try:
                processed_signal, stats = future.result()
                processed_channels[std_name] = processed_signal
                normalization_stats[std_name] = stats
            except Exception as e:
                logger.warning(f"Failed to process {std_name}: {e}")
    
    # ... (save HDF5 code same as before)

def _process_channel_data(self, signal_data, original_sr, modality):
    '''Process signal data (designed for parallel execution).'''
    # Filter
    if modality in self.FILTER_PARAMS:
        filter_params = self.FILTER_PARAMS[modality]
        signal_data = self._bandpass_filter(
            signal_data, original_sr,
            filter_params['low'], filter_params['high'],
            filter_params['order']
        )
    
    # Resample
    if original_sr != self.TARGET_SR:
        signal_data = self._resample_signal(signal_data, original_sr, self.TARGET_SR)
    
    # Normalize
    signal_data, norm_stats = self._normalize_signal(signal_data)
    
    # Convert dtype
    signal_data = signal_data.astype(self.OUTPUT_DTYPE)
    
    return signal_data, norm_stats
```

Expected improvement with 8 cores: 80s → 15-20s (4-5x faster)


PRIORITY 5: Batch Processing (OPTIONAL)
---------------------------------------
Process multiple EDF files in parallel (at process_signals.py level):
```python
from concurrent.futures import ProcessPoolExecutor

def process_multiple_files(edf_paths, output_dir, n_workers=4):
    '''Process multiple EDFs in parallel.'''
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for edf_path in edf_paths:
            output_path = output_dir / f"{edf_path.stem}.h5"
            future = executor.submit(processor.process_edf, edf_path, output_path)
            futures.append(future)
        
        for future in as_completed(futures):
            result = future.result()
            # Handle result
```


Expected Overall Improvement:
------------------------------
Current: ~90 seconds per file (8 channels)
After optimizations:
- Selective preload: 27s → 8s (-19s)
- MNE filtering: 48s → 12s (-36s)
- Faster resample: 16s → 8s (-8s)  
- Parallel processing (8 cores): 28s → 7s (-21s)

**Final: 90s → 7-10s per file (9-13x speedup!)**

For 1000 files:
- Current: 25 hours
- Optimized: 2-3 hours


Implementation Priority:
------------------------
1. Selective preload (IMMEDIATE - easiest, big impact)
2. MNE filtering (IMMEDIATE - one-line change)
3. Better resampling (MEDIUM - moderate complexity)
4. Parallel processing (ADVANCED - requires careful testing)

Quick Win Code:
--------------
Just 2 changes for 5-6x speedup:

1. In _load_edf():
```python
def _load_edf(self, edf_path: Path) -> mne.io.Raw:
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose='ERROR')  # Change False → True
    return raw
```

2. In _bandpass_filter():
```python
def _bandpass_filter(self, signal_data, sr, low, high, order=4):
    import mne
    return mne.filter.filter_data(
        signal_data, sr, l_freq=low, h_freq=high,
        method='fir', verbose=False
    )
```

Note: Full preload may use 1-2GB RAM per file, but eliminates disk reads.
"""
