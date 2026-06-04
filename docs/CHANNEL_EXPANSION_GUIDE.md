# Channel Expansion Guide: Fast-Channel → Full-Channel Run

This guide explains how to run the full 23-channel SleepFM pipeline **without
overwriting any existing data or results**.  Everything new goes under
`/scratch/boshra95/psg_full/` — a completely separate directory tree.

---

## 1. What Exists vs What Will Be Created

```
EXISTING — do NOT touch                     NEW — channel expansion run
─────────────────────────────────────────   ──────────────────────────────────────────
/scratch/boshra95/psg/                      /scratch/boshra95/psg_full/
  {dataset}/derived/hdf5_signals/   (7ch)     {dataset}/derived/hdf5_signals/  (23ch)
  unified/embeddings/sleepfm_5sec/             unified/embeddings/sleepfm_5sec/
  unified/results/phase0_v3/                   unified/results/phase0_v3_full/
  unified/targets_v2/               ◄──── SHARED (read-only, not duplicated)

configs/phase0_v3_config.yaml               configs/phase0_v3_full_config.yaml  ✅ created
configs/preprocessing_params.yaml           configs/preprocessing_params_full.yaml  ✅ created
experiments/v2_registry.yaml                experiments/v2_full_registry.yaml  ✅ created
logs_v3/                                    logs_v3_full/
```

Targets (`targets_v2/`) and sleep-stage annotations are **shared** — they don't
depend on channel count and will not be duplicated.

---

## 2. New Files Created (Already Done)

| File | What changed |
|------|-------------|
| `configs/preprocessing_params_full.yaml` | `strategy: "sleepfm_full"` + `base_output: /scratch/boshra95/psg_full` |
| `configs/phase0_v3_full_config.yaml` | All paths point to `psg_full/`; targets/annotations still point to `psg/` |
| `experiments/v2_full_registry.yaml` | Points to `phase0_v3_full_config.yaml` and `psg_full` results/inference dirs |
| `jobs/extract_embeddings_gpu.sh` | `CONFIG` is now an overridable env variable (default still `phase0_v3_config.yaml`) |

---

## 3. How Our Pipeline Uses Channels

```
EDFs
  │  preprocess_signals.py
  │  config: preprocessing_params_full.yaml   ← controls HOW MANY channels per modality
  │           channel_definitions.yaml        ← maps raw EDF names → standardised names
  │           modality_groups.yaml            ← assigns standardised names to modalities
  ▼
HDF5 files  [~14–21 channels each, standardised names e.g. C3-M2, LOC, Airflow]
  │  extract_sleepfm_embeddings.py
  │  config: phase0_v3_full_config.yaml
  │          data.channel_priority  ← which HDF5 keys to pick per modality (priority order)
  │          data.max_channels      ← BAS≤10, RESP≤7, EKG≤2, EMG≤4
  │  Does NOT use sleepfm-clinical/sleepfm/configs/channel_groups.json
  ▼
.npy files  [T, 4, 128]   — channels encoded inside; no channel names downstream
  │  train_context_sweep.py / infer_subject_windows.py
  │  config: phase0_v3_full_config.yaml   (only reads dataset.embedding_dir)
  ▼
Trained models  →  results/phase0_v3_full/
```

---

## 4. Channels Available per Dataset After Full-Channel Preprocessing

Not every dataset will have every channel — the preprocessing silently skips
missing ones. The SleepFM encoder zero-pads to the modality maximum.

| Dataset | Typical BAS (max 10) | Typical RESP (max 7) | EKG (max 2) | EMG (max 4) | Typical total |
|---------|---------------------|---------------------|-------------|-------------|--------------|
| STAGES  | C3-M2, C4-M1, LOC, ROC, O1-M2, O2-M1, F3-M2, F4-M1, A1, A2 (8–10) | Airflow, Thor, ABD, SpO2, HR (3–5) | EKG, ECG-L (1–2) | CHIN, LLEG, RLEG (2–4) | 15–21 |
| SHHS    | EEG, LOC, ROC (3 — no per-electrode names) | Airflow, Thor, ABD, SpO2 (2–4) | EKG (1) | EMG (1) | 7–9 |
| APPLES  | C3-M2, C4-M1, LOC, ROC, O1-M2, O2-M1 (6–8) | Airflow, Thor, ABD, SpO2 (3–5) | EKG, ECG-L (1–2) | EMG, LLEG, RLEG (2–4) | 13–19 |
| MrOS    | C3-M2, C4-M1, LOC, ROC, O1-M2, O2-M1 (6–8) | Airflow, Thor, ABD, SpO2, HR (3–5) | EKG (1) | CHIN, LLEG, RLEG (3–4) | 13–18 |

**Current (fast) run for comparison:**
All datasets: 7–8 channels (BAS=4, RESP=1, EKG=1, EMG=1–2).

---

## 5. Step-by-Step Run

### Step 1 — Re-preprocess HDF5 into `psg_full/` (4 parallel CPU jobs, ~26 h each)

These write to `/scratch/boshra95/psg_full/{dataset}/derived/hdf5_signals/`.
**Existing HDF5 in `psg/` are not touched.**

```bash
cd /home/boshra95/NSRR-tools

sbatch jobs/preprocess_signals_parallel.sh stages --config configs/preprocessing_params_full.yaml
sbatch jobs/preprocess_signals_parallel.sh shhs   --config configs/preprocessing_params_full.yaml
sbatch jobs/preprocess_signals_parallel.sh apples --config configs/preprocessing_params_full.yaml
sbatch jobs/preprocess_signals_parallel.sh mros   --config configs/preprocessing_params_full.yaml
```

Note: no `--no-skip-existing` needed — these are brand-new output directories,
so every file is new. If you want to be explicit:
```bash
sbatch jobs/preprocess_signals_parallel.sh stages \
    --config configs/preprocessing_params_full.yaml --no-skip-existing
```

**Optional: test on 5 subjects first**
```bash
sbatch jobs/preprocess_signals_parallel.sh stages 5 \
    --config configs/preprocessing_params_full.yaml --log-level DEBUG
```

Verify output:
```bash
python3 - <<'EOF'
import h5py, glob
for f in sorted(glob.glob('/scratch/boshra95/psg_full/stages/derived/hdf5_signals/*.h5'))[:3]:
    with h5py.File(f) as h:
        print(f.split('/')[-1], '->', sorted(h.keys()))
EOF
# Expected e.g.: STNF00032.h5 -> ['ABD', 'Airflow', 'C3-M2', 'C4-M1', 'CHIN',
#   'EKG', 'F3-M2', 'F4-M1', 'LLEG', 'LOC', 'O1-M2', 'O2-M1', 'ROC', 'SpO2', 'Thor']
```

Confirm file counts when done:
```bash
for ds in stages shhs apples mros; do
    n=$(ls /scratch/boshra95/psg_full/${ds}/derived/hdf5_signals/*.h5 2>/dev/null | wc -l)
    echo "$ds: $n files (expected: stages=1513 shhs=8444 apples=1104 mros=3933)"
done
```

---

### Step 2 — Extract SleepFM Embeddings into `psg_full/` (6 parallel GPU jobs, ~4 h each)

These write to `/scratch/boshra95/psg_full/unified/embeddings/sleepfm_5sec/`.
**Existing `.npy` files in `psg/unified/embeddings/` are not touched.**

```bash
cd /home/boshra95/NSRR-tools

NEW_CFG=configs/phase0_v3_full_config.yaml

sbatch --export=ALL,CONFIG=$NEW_CFG,START=0,END=2500      jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,CONFIG=$NEW_CFG,START=2500,END=5000   jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,CONFIG=$NEW_CFG,START=5000,END=7500   jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,CONFIG=$NEW_CFG,START=7500,END=10000  jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,CONFIG=$NEW_CFG,START=10000,END=12500 jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,CONFIG=$NEW_CFG,START=12500,END=15000 jobs/extract_embeddings_gpu.sh
```

The job log will print `Config: configs/phase0_v3_full_config.yaml` so you can confirm
the correct config was used.

Verify when done:
```bash
find /scratch/boshra95/psg_full/unified/embeddings/sleepfm_5sec -name '*.npy' | wc -l
# Expected: ~14,992

python3 -c "
import numpy as np
a = np.load('/scratch/boshra95/psg_full/unified/embeddings/sleepfm_5sec/apples/APL0473.npy', mmap_mode='r')
print('shape:', a.shape)  # still [T, 4, 128] — same dimensions, richer content
"
```

---

### Step 3 — Train All Experiments (using full-channel registry)

Pass `--registry experiments/v2_full_registry.yaml` to `gen_commands.py`.
Results go to `/scratch/boshra95/psg_full/unified/results/phase0_v3_full/`.
**Nothing in `phase0_v3/` is touched.**

```bash
cd /home/boshra95/NSRR-tools
REG="--registry experiments/v2_full_registry.yaml"

# Tier 1
python scripts/gen_commands.py $REG train sex_binary_lstm         | bash
python scripts/gen_commands.py $REG train sex_binary_transformer  | bash
python scripts/gen_commands.py $REG train sex_binary_mean_pool    | bash

python scripts/gen_commands.py $REG train sleep_efficiency_binary_lstm        | bash
python scripts/gen_commands.py $REG train sleep_efficiency_binary_transformer | bash
python scripts/gen_commands.py $REG train sleep_efficiency_binary_mean_pool   | bash

python scripts/gen_commands.py $REG train bmi_binary_lstm        | bash
python scripts/gen_commands.py $REG train bmi_binary_transformer | bash
python scripts/gen_commands.py $REG train bmi_binary_mean_pool   | bash

python scripts/gen_commands.py $REG train age_class_lstm         | bash
python scripts/gen_commands.py $REG train age_class_transformer  | bash
python scripts/gen_commands.py $REG train age_class_mean_pool    | bash

# Tier 2
python scripts/gen_commands.py $REG train psqi_binary_lstm               | bash
python scripts/gen_commands.py $REG train depression_extreme_binary_lstm | bash
python scripts/gen_commands.py $REG train osa_binary_apples_postqc_lstm  | bash
python scripts/gen_commands.py $REG train osa_severity_apples_lstm       | bash
```

Monitor:
```bash
python scripts/gen_commands.py $REG runs sex_binary_lstm
python scripts/gen_commands.py $REG status
```

---

### Step 4 — Inference and Analysis

```bash
REG="--registry experiments/v2_full_registry.yaml"

# Per experiment after training:
python scripts/gen_commands.py $REG infer sex_binary_lstm | bash
python scripts/gen_commands.py $REG analyze sex_binary_lstm --plot | bash
python scripts/gen_commands.py $REG analyze sex_binary_lstm --k-dense | bash
python scripts/gen_commands.py $REG build-heatmap sex_binary_lstm | bash
python scripts/gen_commands.py $REG iso-plots sex_binary_lstm | bash

# Saturation curve once all 3 heads done:
python scripts/gen_commands.py $REG saturation sex_binary \
    --heads lstm transformer mean_pool | bash
```

---

## 6. How to Compare Fast-Channel vs Full-Channel Results

Both experiment sets coexist on disk. To compare them side by side:

```bash
# Fast-channel (current):
python scripts/gen_commands.py --registry experiments/v2_registry.yaml \
    saturation sex_binary --heads lstm transformer mean_pool | bash

# Full-channel (new):
python scripts/gen_commands.py --registry experiments/v2_full_registry.yaml \
    saturation sex_binary --heads lstm transformer mean_pool | bash
```

Or load both `summary.csv` files directly:
```python
import pandas as pd
fast = pd.read_csv('/scratch/boshra95/psg/unified/results/phase0_v3/sex_binary_lstm/summary.csv')
full = pd.read_csv('/scratch/boshra95/psg_full/unified/results/phase0_v3_full/sex_binary_lstm/summary.csv')
fast['run'] = 'fast_7ch'; full['run'] = 'full_23ch'
combined = pd.concat([fast, full])
print(combined[['run','context_length','test_auroc']])
```

---

## 7. Complete Checklist

```
Step 0 — Config files (ALREADY DONE — no action needed):
  ✅  configs/preprocessing_params_full.yaml    (strategy=sleepfm_full, base=psg_full)
  ✅  configs/phase0_v3_full_config.yaml        (all paths → psg_full; targets → psg)
  ✅  experiments/v2_full_registry.yaml         (config + results → full variants)
  ✅  jobs/extract_embeddings_gpu.sh            (CONFIG now overridable via env var)

Step 1 — Preprocessing (4 parallel CPU jobs, ~26 h, no GPU):
  ☐  sbatch stages --config configs/preprocessing_params_full.yaml
  ☐  sbatch shhs   --config configs/preprocessing_params_full.yaml
  ☐  sbatch apples --config configs/preprocessing_params_full.yaml
  ☐  sbatch mros   --config configs/preprocessing_params_full.yaml
  ☐  Verify: ~14,992 HDF5 files in psg_full/, 14–21 channels each

Step 2 — Embedding Extraction (6 parallel GPU jobs, ~4 h each):
  ☐  Submit 6 jobs: CONFIG=configs/phase0_v3_full_config.yaml, START/END 0–15000
  ☐  Verify: ~14,992 .npy files in psg_full/unified/embeddings/sleepfm_5sec/

Step 3 — Training (--registry experiments/v2_full_registry.yaml):
  ☐  Tier 1: sex_binary, sleep_efficiency, bmi_binary, age_class (3 heads each)
  ☐  Tier 2: psqi, depression, osa (lstm only)
  ☐  Monitor with: gen_commands.py $REG runs / status

Step 4 — Inference + Analysis:
  ☐  infer → analyze --plot → analyze --k-dense → build-heatmap → iso-plots
  ☐  saturation (once all 3 heads done per task)

Existing data — NEVER TOUCHED:
  ✓  /scratch/boshra95/psg/{dataset}/derived/hdf5_signals/
  ✓  /scratch/boshra95/psg/unified/embeddings/sleepfm_5sec/
  ✓  /scratch/boshra95/psg/unified/results/phase0_v3/
  ✓  /scratch/boshra95/psg/unified/targets_v2/
```
