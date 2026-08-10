# HDF5 Format Verification for SleepFM Compatibility

**Date**: March 2, 2026  
**Status**: ✅ **VERIFIED COMPATIBLE**

---

## Summary

**Our NSRR-tools preprocessing IS CORRECT** - channels are saved separately (not grouped), exactly as SleepFM expects!

---

## SleepFM Expected Format (Verified from Codebase)

### 1. **HDF5 File Structure**
```
subject001.hdf5
├── C3-M2 (dataset): shape=(N,), dtype=float16, chunks=(38400,)
├── C4-M1 (dataset): shape=(N,), dtype=float16, chunks=(38400,)
├── O1-M2 (dataset): shape=(N,), dtype=float16, chunks=(38400,)
├── EOG(L) (dataset): shape=(N,), dtype=float16, chunks=(38400,)
├── EKG (dataset): shape=(N,), dtype=float16, chunks=(38400,)
├── Flow (dataset): shape=(N,), dtype=float16, chunks=(38400,)
├── Thor (dataset): shape=(N,), dtype=float16, chunks=(38400,)
├── CHIN (dataset): shape=(N,), dtype=float16, chunks=(38400,)
└── ... (all other channels)
```

**Key Points**:
- ✅ Each channel = **separate dataset** (NOT grouped into modalities)
- ✅ Shape = **1D array** `[total_samples]` (continuous time series, NOT segmented)
- ✅ Chunks = **38,400 samples** (5 minutes × 60 seconds × 128 Hz)
- ✅ Dtype = **float16**
- ✅ Compression = **gzip**
- ✅ Normalization = **Already applied** (mean≈0, std≈1)

---

## How SleepFM Reads the Data

### From `sleepfm/models/dataset.py`:

```python
# 1. Open HDF5 file and read all channel names
with h5py.File(file_path, 'r') as hf:
    dset_names = list(hf.keys())  # ['C3-M2', 'C4-M1', 'EKG', ...]
    
    # 2. Group channels by modality using channel_groups.json
    for dset_name in dset_names:
        if dset_name in channel_groups["BAS"]:
            modality_to_channels["BAS"].append(dset_name)
        if dset_name in channel_groups["EKG"]:
            modality_to_channels["EKG"].append(dset_name)
        # ... etc for RESP, EMG
    
    # 3. Load 5-minute chunks (38,400 samples)
    for ds_name in modality_to_channels[modality_type]:
        signal = hf[ds_name][chunk_start:chunk_start+38400]
        data[idx] = signal  # shape: (num_channels, 38400)
```

**Process**:
1. Reads HDF5 file with **separate channel datasets**
2. **Groups channels dynamically** using `channel_groups.json` lookup
3. Loads **5-minute chunks** (38,400 samples at 128 Hz)
4. Stacks channels by modality → `[num_channels, 38400]`
5. Pads to max channels per modality (BAS=10, RESP=7, EKG=2, EMG=4)
6. Creates attention masks (0=real channel, 1=padding)

---

## Model Configuration (from checkpoint)

```json
{
    "sampling_freq": 128,           // Hz
    "sampling_duration": 5,         // minutes
    "samples_per_chunk": 38400,     // 5 * 60 * 128
    "patch_size": 640,              // embeddings per chunk
    "embed_dim": 128,
    "BAS_CHANNELS": 10,
    "RESP_CHANNELS": 7,
    "EKG_CHANNELS": 2,
    "EMG_CHANNELS": 4,
    "modality_types": ["BAS", "RESP", "EKG", "EMG"]
}
```

---

## Our NSRR-tools Implementation (Verified)

### From `src/nsrr_tools/core/signal_processor.py`:

```python
def _save_hdf5(self, output_path, channels, norm_stats, raw):
    with h5py.File(output_path, 'w') as hf:
        # Save each channel as a dataset
        for channel_name, signal_data in channels.items():
            chunk_size = min(5 * 60 * self.TARGET_SR, len(signal_data))  # 38,400
            
            hf.create_dataset(
                channel_name,                # ✅ Separate dataset per channel
                data=signal_data,            # ✅ 1D array [N,]
                dtype=self.OUTPUT_DTYPE,     # ✅ float16
                chunks=(chunk_size,),        # ✅ (38400,)
                compression=self.COMPRESSION,# ✅ gzip
                compression_opts=self.COMPRESSION_LEVEL  # ✅ level 4
            )
```

### Constants:
```python
TARGET_SR = 128              # ✅ 128 Hz
OUTPUT_DTYPE = np.float16    # ✅ float16
COMPRESSION = 'gzip'         # ✅ gzip
COMPRESSION_LEVEL = 4        # ✅ level 4
```

**Result**: ✅ **Perfect match with SleepFM requirements!**

---

## Dimensionality & Segmentation

### What the Paper Says:
> "We use 5-minute segments from overnight PSG recordings..."
> "Each 5-minute segment is processed independently"

### How It's Implemented:
1. **HDF5 files**: Store **continuous** time series (NOT pre-segmented)
   - Example: 6 hours = 2,764,800 samples at 128 Hz
   
2. **Chunking for I/O**: HDF5 chunks are 38,400 samples (5 minutes)
   - Optimizes read performance
   - Does NOT mean data is segmented
   
3. **Model input**: Dataloader **dynamically** extracts 5-minute chunks
   - From `dataset.py`: `signal = hf[ds_name][chunk_start:chunk_start+38400]`
   - Slides through the continuous recording

### Embedding Generation:
- **Input**: 5-minute chunk → `[num_channels, 38400]`
- **Patch embedding**: 38400 samples ÷ 640 patch_size = **60 patches**
- **Output (aggregated)**: `[1, 128]` per modality (5-minute level)
- **Output (granular)**: `[60, 128]` per modality (5-second level)

---

## Verification Results

### ✅ Checked: SleepFM Demo Data
```
File: sleepfm-clinical/notebooks/demo_data/demo_psg.hdf5
Channels: ['Airflow', 'Arm EMG', 'C3-A2', 'EKG']
Shape: (1117312,) = 1D continuous array [145.5 minutes]
Dtype: float16
Chunks: (38400,)
Compression: gzip
Normalization: mean=-0.001, std=0.995 (normalized)
```

### ✅ Checked: NSRR-tools Output
```python
chunk_size = 5 * 60 * 128 = 38400 samples
dtype = float16
compression = gzip (level 4)
shape = (N,) - 1D continuous array
channels = separate datasets (C3-M2, C4-M1, EKG, ...)
```

---

## Compatibility Matrix

| Requirement | SleepFM Expects | NSRR-tools Produces | Status |
|-------------|-----------------|---------------------|--------|
| File format | HDF5 | HDF5 | ✅ |
| Channel storage | Separate datasets | Separate datasets | ✅ |
| Data shape | 1D `[N,]` continuous | 1D `[N,]` continuous | ✅ |
| Sampling rate | 128 Hz | 128 Hz | ✅ |
| Dtype | float16 | float16 | ✅ |
| Chunk size | 38,400 samples | 38,400 samples | ✅ |
| Compression | gzip | gzip (level 4) | ✅ |
| Normalization | z-score inline | z-score inline | ✅ |
| Channel naming | Standard names | Standard names | ✅ |
| Modality grouping | Dynamic (via config) | Dynamic (via config) | ✅ |
| Max channels | BAS=10, ECG=2, EMG=4, RESP=7 | Same limits enforced | ✅ |

---

## Conclusion

### ✅ **NO CHANGES NEEDED**

Your NSRR-tools preprocessing output **matches SleepFM format exactly**:

1. ✅ Channels saved as **separate HDF5 datasets** (not grouped)
2. ✅ 1D continuous arrays `[total_samples]`
3. ✅ 128 Hz sampling rate
4. ✅ float16 dtype
5. ✅ 38,400-sample chunks (5 minutes)
6. ✅ gzip compression
7. ✅ z-score normalization applied
8. ✅ Standard channel names compatible with `channel_groups.json`

### How SleepFM Will Use It:

1. Load HDF5 file → read all channel dataset names
2. Group channels by modality using `channel_groups.json`
3. Extract 5-minute chunks dynamically (38,400 samples)
4. Stack channels per modality → `[num_channels, 38400]`
5. Process through SetTransformer → generate embeddings

**Your data is ready for SleepFM!** 🎉

---

## References

- SleepFM codebase: `sleepfm-clinical/sleepfm/models/dataset.py` lines 22-180
- Demo HDF5: `sleepfm-clinical/notebooks/demo_data/demo_psg.hdf5`
- Model config: `sleepfm-clinical/sleepfm/checkpoints/model_base/config.json`
- Your implementation: `NSRR-tools/src/nsrr_tools/core/signal_processor.py`
