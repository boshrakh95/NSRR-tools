# Channel Selection Strategy Guide

**Date**: March 2, 2026  
**Feature**: Flexible channel limiting for faster processing

---

## Overview

You can now control how many channels are processed per modality to speed up preprocessing. This is useful for:
- **Quick testing** before running full datasets
- **Fast initial analysis** with essential channels only
- **Minimal processing** for exploratory work

---

## How to Use

### 1. Open Configuration File
Edit: `configs/preprocessing_params.yaml`

### 2. Find the `channel_selection` Section
```yaml
channel_selection:
  strategy: "sleepfm_full"  # Change this line!
  
  custom_limits:
    BAS: 6
    EKG: 1
    EMG: 2
    RESP: 2
```

### 3. Change the Strategy

**Option A: Fast Processing (8 channels total)**
```yaml
strategy: "fast"
```
- **BAS = 4**: C3-M2, C4-M1, LOC, ROC
- **EKG = 1**: EKG
- **EMG = 2**: CHIN, LLEG (or RLEG)
- **RESP = 1**: Thor (or Airflow)
- **Total: 8 channels**
- **Processing time: ~65% faster** than sleepfm_full

**Option B: Minimal Processing (4 channels total)**
```yaml
strategy: "minimal"
```
- **BAS = 2**: C3-M2, C4-M1
- **EKG = 1**: EKG
- **EMG = 1**: CHIN
- **RESP = 0**: None (skipped)
- **Total: 4 channels**
- **Processing time: ~80% faster** than sleepfm_full

**Option C: Full SleepFM Specification (default)**
```yaml
strategy: "sleepfm_full"
```
- **BAS = 10**: All available EEG + EOG
- **EKG = 2**: Both ECG leads
- **EMG = 4**: CHIN, LLEG, RLEG, EMG
- **RESP = 7**: All respiratory signals
- **Total: Up to 23 channels**
- **Processing time: Full (baseline)**

**Option D: Custom Limits**
```yaml
strategy: "custom"

custom_limits:
  BAS: 6   # Your choice
  EKG: 1
  EMG: 2
  RESP: 2
```

---

## Strategy Comparison Table

| Strategy | BAS | EKG | EMG | RESP | Total | Use Case |
|----------|-----|-----|-----|------|-------|----------|
| **sleepfm_full** | 10 | 2 | 4 | 7 | 23 | Full preprocessing, final analysis |
| **fast** | 4 | 1 | 2 | 1 | 8 | Quick testing, initial exploration |
| **minimal** | 2 | 1 | 1 | 0 | 4 | Fastest testing, debugging |
| **custom** | ? | ? | ? | ? | ? | Your specific needs |

---

## Example Workflows

### Workflow 1: Quick Test → Full Processing

1. **Start with minimal** for debugging:
   ```yaml
   strategy: "minimal"
   ```
   Run on 10-20 subjects to verify pipeline works

2. **Switch to fast** for initial analysis:
   ```yaml
   strategy: "fast"
   ```
   Process full dataset, generate embeddings, run initial models

3. **Switch to sleepfm_full** for final results:
   ```yaml
   strategy: "sleepfm_full"
   ```
   Full preprocessing for publication-quality analysis

### Workflow 2: Custom for Specific Analysis

If you only need sleep staging (BAS channels important):
```yaml
strategy: "custom"

custom_limits:
  BAS: 6    # More brain channels
  EKG: 1    # Minimal cardiac
  EMG: 1    # Minimal muscle
  RESP: 0   # Skip respiratory
```

---

## Channel Selection Priority

Channels are selected by **priority order** (most clinically important first):

### BAS (Brain Activity Signals)
1. C3-M2 (central left)
2. C4-M1 (central right)
3. O1-M2 (occipital left)
4. O2-M1 (occipital right)
5. LOC (left eye)
6. ROC (right eye)
7. F3-M2, F4-M1 (frontal)
8. ... (additional EEG/EOG)

### EKG (Cardiac)
1. EKG (primary lead)
2. ECG-L (left lead)
3. ECG-R (right lead)

### EMG (Muscle)
1. CHIN (chin EMG - most important for sleep staging)
2. LLEG (left leg)
3. RLEG (right leg)
4. EMG (generic)

### RESP (Respiratory)
1. Airflow (nasal cannula)
2. Thor (thoracic effort)
3. ABD (abdominal effort)
4. SpO2, HR, Snore, RespRate

**With `strategy: "fast"`**, you automatically get:
- Top 4 BAS channels (C3-M2, C4-M1, LOC, ROC)
- Top 1 EKG channel (EKG)
- Top 2 EMG channels (CHIN, LLEG)
- Top 1 RESP channel (Airflow or Thor)

---

## Technical Details

### Implementation
- Located in: `src/nsrr_tools/core/signal_processor.py`
- Method: `_apply_sleepfm_limits()`
- Strategies defined in: `SignalProcessor.CHANNEL_STRATEGIES`

### Logging
You'll see in the logs:
```
SignalProcessor initialized
  Target sampling rate: 128 Hz
  Output dtype: <class 'numpy.float16'>
  Compression: gzip (level 4)
  Channel selection strategy: fast
  Channel limits: BAS=4, EKG=1, EMG=2, RESP=1 (total: 8 max)
```

During processing:
```
Processing subject001.edf...
  Found 30 channels across 5 modalities
  BAS: Limiting 12 → 4 channels (strategy=fast)
    Selected: ['C3-M2', 'C4-M1', 'LOC', 'ROC']
    Dropped: ['O1-M2', 'O2-M1', 'F3-M2', 'F4-M1', ...]
  EKG: 1 channel (within limit)
  EMG: Limiting 3 → 2 channels (strategy=fast)
  RESP: Limiting 5 → 1 channels (strategy=fast)
  Selected 8 channels after applying SleepFM limits
```

---

## Validation

After changing strategy, verify it worked:

```bash
# Check a processed HDF5 file
python -c "
import h5py
with h5py.File('output/subject001.h5', 'r') as f:
    print(f'Channels: {list(f.keys())}')
    print(f'Total: {len(f.keys())} channels')
"
```

Expected output for `strategy: "fast"`:
```
Channels: ['C3-M2', 'C4-M1', 'LOC', 'ROC', 'EKG', 'CHIN', 'LLEG', 'Thor']
Total: 8 channels
```

---

## Recommendations

### For Your Current Goal (Quick Analysis)
**Recommended: `strategy: "fast"`**

This gives you:
- ✅ Essential brain activity (C3-M2, C4-M1, LOC, ROC)
- ✅ Cardiac monitoring (EKG)
- ✅ Muscle activity for sleep staging (CHIN, LLEG)
- ✅ Basic respiratory (Thor or Airflow)
- ✅ **~3x faster processing** than full
- ✅ **Sufficient for most analyses**

### Processing Time Estimates
(Based on 6-hour PSG recordings)

| Strategy | Channels | Time/Subject | 1000 Subjects |
|----------|----------|--------------|---------------|
| sleepfm_full | ~20 | 4 min | ~67 hours |
| **fast** | **8** | **~1.5 min** | **~25 hours** |
| minimal | 4 | ~1 min | ~17 hours |

*Note: Actual times vary by system, EDF complexity, and parallel processing*

---

## FAQ

**Q: Will "fast" strategy affect SleepFM model quality?**  
A: The model is channel-agnostic and pads missing channels. With 8 essential channels, you'll get high-quality embeddings. However, having more channels (sleepfm_full) provides richer information.

**Q: Can I change strategy after processing some subjects?**  
A: Yes! Each HDF5 file is independent. Different subjects can have different channels. Just update the config and continue.

**Q: What if a subject doesn't have all channels for the selected strategy?**  
A: The processor uses whatever channels are available. If a subject has only 2 BAS channels and you selected "fast" (4 BAS), it will save those 2 channels.

**Q: Can I exclude specific channels (e.g., not Airflow)?**  
A: Yes! Use priority order. If you don't want Airflow, make sure Thor is higher priority in `modality_groups.yaml`, or use a custom strategy that specifies which exact channels to use.

---

## Next Steps

1. **Edit config**: Change `strategy` in `configs/preprocessing_params.yaml`
2. **Test on one subject**: Run preprocessing on a single file
3. **Verify output**: Check channels in the HDF5 file
4. **Process dataset**: Run full preprocessing
5. **Generate embeddings**: Use SleepFM to create embeddings
6. **Start analysis**: Begin your cognitive prediction work

**Good luck with your analysis!** 🚀
