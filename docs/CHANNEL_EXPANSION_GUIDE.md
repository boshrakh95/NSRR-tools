# Channel Expansion Guide: Fast (7) → SleepFM Full (23)

**Goal:** Re-preprocess all NSRR datasets with the full SleepFM channel set (up to 23 channels),
re-extract embeddings, and retrain all v2 experiments.

---

## 1. How Our Pipeline Actually Uses Channels

This is the complete channel data flow. Understanding it is essential for the paper.

```
EDFs (raw signals)
      │
      │  preprocess_signals.py
      │  reads: preprocessing_params.yaml   ← controls HOW MANY channels per modality
      │          channel_definitions.yaml   ← maps raw EDF names → standardised names
      │          modality_groups.yaml       ← defines which standardised names = which modality
      ▼
HDF5 files  (e.g. C3-M2, C4-M1, LOC, ROC, EKG, CHIN, Airflow)
      │
      │  extract_sleepfm_embeddings.py
      │  reads: phase0_v2_config.yaml  data.channel_priority  ← which HDF5 keys to pick per modality
      │                                data.max_channels       ← how many max per modality
      │  Does NOT read: sleepfm-clinical/sleepfm/configs/channel_groups.json
      │                  (our HDF5 already uses standardised names — no remapping needed)
      │  For each subject: intersection( HDF5 keys, channel_priority ) → load → zero-pad → SleepFM encoder
      ▼
.npy files  shape [T, 4, 128]
            T   = number of 5-sec patches in the recording
            4   = modalities: BAS, RESP, EKG, EMG   (always exactly 4)
            128 = embedding dimension per modality
            Channel identity is fully encoded here — there are no channel names downstream.
      │
      │  train_context_sweep.py  /  infer_subject_windows.py
      │  reads: .npy files only (via ContextWindowDataset)
      │  Knows nothing about original channels.
      ▼
Trained heads  (LSTM / Transformer / MeanPool)
```

### What each config file controls

| File | Where used | Controls |
|------|-----------|---------|
| `configs/preprocessing_params.yaml` | `preprocess_signals.py` | How many channels per modality to extract from EDF → HDF5. Key field: `channel_selection.strategy` |
| `configs/channel_definitions.yaml` | `preprocess_signals.py` | Maps raw EDF channel names (e.g., `"EEG C3-A2"`, `"EKG L"`) to standardised HDF5 keys (e.g., `"C3-M2"`, `"EKG"`) |
| `configs/modality_groups.yaml` | `preprocess_signals.py` | Defines which standardised names belong to EEG/EOG/ECG/EMG/RESP; sets SleepFM BAS/RESP/EKG/EMG group membership |
| `configs/phase0_v2_config.yaml` → `data.channel_priority` | `extract_sleepfm_embeddings.py` | Priority-ordered list of HDF5 key names to feed to SleepFM per modality. Channels are selected by intersection with what the HDF5 actually contains. |
| `configs/phase0_v2_config.yaml` → `data.max_channels` | `extract_sleepfm_embeddings.py` | Hard cap on channels per modality fed to SleepFM (BAS≤10, RESP≤7, EKG≤2, EMG≤4). Zero-padding fills unused slots. |
| `sleepfm-clinical/.../channel_groups.json` | **NOT used** | SleepFM's own channel name registry. Irrelevant for our pipeline because our HDF5 files already store standardised names that `channel_priority` references directly. |

---

## 2. Current State vs Target

### What is in the HDF5 files right now (`strategy: "fast"`)

| Dataset | Channels stored | BAS | RESP | EKG | EMG |
|---------|----------------|-----|------|-----|-----|
| STAGES  | C3-M2, C4-M1, LOC, ROC, EKG, CHIN, LLEG, Airflow | 4 | 1 | 1 | 2 |
| SHHS    | EEG, LOC, ROC, EKG, EMG, Airflow, Thor | 3 | 2 | 1 | 1 |
| APPLES  | C3-M2, C4-M1, LOC, ROC, EKG, EMG, LLEG, Airflow | 4 | 1 | 1 | 2 |
| MrOS    | C3-M2, C4-M1, LOC, ROC, EKG, CHIN, LLEG, Airflow | 4 | 1 | 1 | 2 |

### What the SleepFM encoder actually received during embedding extraction

`phase0_v2_config.yaml` already specifies all 23 channels in `channel_priority`.
`select_channels_from_hdf5()` in our embedding script takes the **intersection** of that
priority list with what the HDF5 file actually contains.
So the encoder received exactly the 7–8 columns above — zero-padded to 10/7/2/4 for the missing slots.

### What SleepFM can accept (`strategy: "sleepfm_full"`)

| Modality | Max | Priority channels (HDF5 key name written by preprocessing) |
|----------|-----|--------------------------------------------------------------|
| BAS      | 10  | C3-M2, C4-M1, LOC, ROC, O1-M2, O2-M1, F3-M2, F4-M1, A1, A2 |
| RESP     | 7   | Airflow, Thor, ABD, SpO2, HR, Snore, RespRate |
| EKG      | 2   | EKG, ECG-L |
| EMG      | 4   | CHIN, LLEG, RLEG, EMG |

**Note on SHHS:** SHHS EDFs use generic `EEG`, `EMG` labels (no per-electrode names),
so it will not gain extra BAS channels — but it will gain RESP channels (Thor, ABD, SpO2).

---

## 3. For the Paper: What to Report

### Channels fed to SleepFM per dataset (current fast strategy)

| Dataset | BAS channels | RESP channels | EKG channels | EMG channels | Total |
|---------|-------------|--------------|-------------|-------------|-------|
| STAGES  | C3-M2, C4-M1, LOC, ROC | Airflow | EKG | CHIN, LLEG | 8 |
| SHHS    | EEG, LOC, ROC | Airflow, Thor | EKG | EMG | 7 |
| APPLES  | C3-M2, C4-M1, LOC, ROC | Airflow | EKG | EMG, LLEG | 8 |
| MrOS    | C3-M2, C4-M1, LOC, ROC | Airflow | EKG | CHIN, LLEG | 8 |

Missing channels are zero-padded by the SleepFM encoder (binary mask tells the model
which slots contain real signal vs padding).

### After re-processing with `sleepfm_full` strategy (typical, subject-dependent)

| Dataset | Typical BAS | Typical RESP | EKG | EMG | Total typical |
|---------|------------|-------------|-----|-----|---------------|
| STAGES  | 8–10 (adds O1-M2, O2-M1, F3-M2, F4-M1, A1, A2) | 3–5 (adds Thor, ABD, SpO2, HR) | 1–2 | 3–4 | 15–21 |
| SHHS    | 3 (EEG, LOC, ROC — no per-electrode labels) | 2–4 (adds Thor, ABD, SpO2) | 1 | 1 | 7–9 |
| APPLES  | 6–8 (adds O1-M2, O2-M1, F3-M2, F4-M1) | 3–5 | 1–2 | 3–4 | 13–19 |
| MrOS    | 6–8 | 3–5 | 1 | 3–4 | 13–18 |

---

## 4. No Config Changes Needed for Training or Embedding Extraction

`configs/phase0_v2_config.yaml` already specifies the full 23-channel priority list:

```yaml
data:
  max_channels:
    BAS: 10
    RESP: 7
    EKG: 2
    EMG: 4
  channel_priority:
    BAS:  [C3-M2, C4-M1, LOC, ROC, O1-M2, O2-M1, F3-M2, F4-M1, A1, A2]
    RESP: [Airflow, Thor, ABD, SpO2, HR, Snore, RespRate]
    EKG:  [EKG, ECG-L, ECG-R]
    EMG:  [CHIN, LLEG, RLEG, EMG]
```

`extract_sleepfm_embeddings.py` uses this list to pick channels from each HDF5 file.
Currently the HDF5 files only have 7–8 of these, so the rest are zero-padded.
After re-preprocessing, the same config automatically picks up the new channels.
**No edits to `phase0_v2_config.yaml` or any other config are required.**

---

## 5. Only One Pre-flight Change: Preprocessing Strategy

Edit `configs/preprocessing_params.yaml`, line ~35:

```yaml
# BEFORE:
strategy: "fast"

# AFTER:
strategy: "sleepfm_full"
```

```bash
sed -i 's/strategy: "fast"/strategy: "sleepfm_full"/' \
    /home/boshra95/NSRR-tools/configs/preprocessing_params.yaml

# Verify:
grep "strategy:" /home/boshra95/NSRR-tools/configs/preprocessing_params.yaml
```

---

## 6. Step-by-Step Run Order

### Step 1 — Re-preprocess HDF5 (4 parallel CPU jobs, ~26 h each)

```bash
cd /home/boshra95/NSRR-tools

sbatch jobs/preprocess_signals_parallel.sh stages --no-skip-existing
sbatch jobs/preprocess_signals_parallel.sh shhs   --no-skip-existing
sbatch jobs/preprocess_signals_parallel.sh apples --no-skip-existing
sbatch jobs/preprocess_signals_parallel.sh mros   --no-skip-existing
```

Monitor:
```bash
squeue -u boshra95
tail -f logs/preprocess_stages_*.out
```

**Optional: test on 5 subjects first**
```bash
sbatch jobs/preprocess_signals_parallel.sh stages 5 --no-skip-existing --log-level DEBUG
```
Then verify:
```bash
python3 -c "
import h5py, glob
for f in sorted(glob.glob('/scratch/boshra95/psg/stages/derived/hdf5_signals/*.h5'))[:3]:
    with h5py.File(f) as h:
        print(f.split('/')[-1], '->', sorted(h.keys()))
"
```

Verify channel counts when done:
```bash
python3 - <<'EOF'
import h5py, glob
from collections import Counter
datasets = {
    'stages': '/scratch/boshra95/psg/stages/derived/hdf5_signals/',
    'shhs':   '/scratch/boshra95/psg/shhs/derived/hdf5_signals/',
    'apples': '/scratch/boshra95/psg/apples/derived/hdf5_signals/',
    'mros':   '/scratch/boshra95/psg/mros/derived/hdf5_signals/',
}
for name, path in datasets.items():
    files = sorted(glob.glob(path + '*.h5'))[:10]
    ch_freq = Counter()
    for f in files:
        with h5py.File(f) as h:
            for ch in h.keys(): ch_freq[ch] += 1
    print(f'\n{name}:')
    for ch, cnt in sorted(ch_freq.items()):
        print(f'  {ch:<20} {cnt}/{len(files)} files')
EOF
```

### Step 2 — Re-extract SleepFM Embeddings (6 parallel GPU jobs, ~4 h each)

The existing `.npy` files were built from 7-channel HDF5. They must be regenerated.

```bash
cd /home/boshra95/NSRR-tools

# 6 parallel jobs covering ~2500 subjects each:
sbatch --export=ALL,START=0,END=2500      jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,START=2500,END=5000   jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,START=5000,END=7500   jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,START=7500,END=10000  jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,START=10000,END=12500 jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,START=12500,END=15000 jobs/extract_embeddings_gpu.sh
```

**Important:** Delete the shape cache before re-extracting — it encodes the old file count.
```bash
rm -f /scratch/boshra95/psg/unified/embeddings/sleepfm_5sec/shape_cache.json
```

Verify when done:
```bash
find /scratch/boshra95/psg/unified/embeddings/sleepfm_5sec -name '*.npy' | wc -l
# Expected: ~14,992

python3 -c "
import numpy as np
a = np.load('/scratch/boshra95/psg/unified/embeddings/sleepfm_5sec/apples/APL0473.npy', mmap_mode='r')
print('shape:', a.shape)   # e.g. (6120, 4, 128) — shape unchanged
# Shape is always [T, 4, 128] regardless of channel count.
# More channels → richer embeddings in the same 128-dim space.
"
```

Note: the `.npy` shape `[T, 4, 128]` is identical before and after — the channel
count difference is encoded inside the 128-dim modality embeddings, not in the
array dimensions.

### Step 3 — Retrain All Experiments

Because the `.npy` embeddings now encode more signal, all existing trained models
are stale. Retrain from scratch using `gen_commands.py`:

```bash
cd /home/boshra95/NSRR-tools

# Tier 1 (3 heads each)
python scripts/gen_commands.py train sex_binary_lstm         | bash
python scripts/gen_commands.py train sex_binary_transformer  | bash
python scripts/gen_commands.py train sex_binary_mean_pool    | bash

python scripts/gen_commands.py train sleep_efficiency_binary_lstm        | bash
python scripts/gen_commands.py train sleep_efficiency_binary_transformer | bash
python scripts/gen_commands.py train sleep_efficiency_binary_mean_pool   | bash

python scripts/gen_commands.py train bmi_binary_lstm         | bash
python scripts/gen_commands.py train bmi_binary_transformer  | bash
python scripts/gen_commands.py train bmi_binary_mean_pool    | bash

python scripts/gen_commands.py train age_class_lstm          | bash
python scripts/gen_commands.py train age_class_transformer   | bash
python scripts/gen_commands.py train age_class_mean_pool     | bash

# Tier 2
python scripts/gen_commands.py train psqi_binary_lstm               | bash
python scripts/gen_commands.py train depression_extreme_binary_lstm | bash
python scripts/gen_commands.py train osa_binary_apples_postqc_lstm  | bash
python scripts/gen_commands.py train osa_severity_apples_lstm       | bash
```

### Step 4 — Re-run Inference and Analysis

```bash
# Per experiment, after training completes:
python scripts/gen_commands.py infer sex_binary_lstm | bash
python scripts/gen_commands.py analyze sex_binary_lstm --plot | bash
python scripts/gen_commands.py analyze sex_binary_lstm --k-dense | bash
python scripts/gen_commands.py build-heatmap sex_binary_lstm | bash
python scripts/gen_commands.py iso-plots sex_binary_lstm | bash

# Once all 3 heads done for a task:
python scripts/gen_commands.py saturation sex_binary \
    --heads lstm transformer mean_pool | bash
```

---

## 7. Summary Checklist

```
Pre-flight (one-time, before any jobs):
  ☐  Edit preprocessing_params.yaml: strategy "fast" → "sleepfm_full"
  ☐  Optional: test on 5 STAGES subjects and verify HDF5 output

Step 1 — Preprocessing (4 parallel CPU jobs, ~26 h):
  ☐  sbatch stages --no-skip-existing
  ☐  sbatch shhs   --no-skip-existing
  ☐  sbatch apples --no-skip-existing
  ☐  sbatch mros   --no-skip-existing
  ☐  Verify HDF5 channel counts per dataset

Step 2 — Embedding Extraction (6 parallel GPU jobs, ~4 h):
  ☐  Delete shape_cache.json
  ☐  Submit 6 jobs covering START=0..15000 in increments of 2500
  ☐  Verify ~14,992 .npy files exist

Step 3 — Training (many parallel GPU jobs):
  ☐  Tier 1: sex_binary, sleep_efficiency, bmi_binary, age_class (3 heads each)
  ☐  Tier 2: psqi, depression, osa tasks (lstm only)

Step 4 — Inference + Analysis (after each task's training):
  ☐  infer → analyze --plot → analyze --k-dense → build-heatmap → iso-plots
  ☐  saturation (once all 3 heads done per task)
```
