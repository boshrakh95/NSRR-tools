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
logs_v3/                                    logs_v3_expand_channel/  (preprocess + embed)
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
| `configs/phase0_v3_full_config.yaml` | All paths → `psg_full/`; targets → `psg/`; **`hidden_dim=128, num_layers=1`** (seq2label, matches fast-channel baseline) |
| `configs/phase0_v3_full_staging_config.yaml` | Same as above but **`hidden_dim=256, num_layers=2`** for sleep staging; used by sleep staging experiments via per-experiment `config:` field in the registry |
| `experiments/v2_full_registry.yaml` | Points to `phase0_v3_full_config.yaml` as default; sleep staging entries have `config: configs/phase0_v3_full_staging_config.yaml` to override |
| `scripts/gen_commands.py` | `build_train_cmd` and `build_infer_cmd` now check `exp.get("config")` first, falling back to `registry["config"]` — enables per-experiment config overrides |
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

### Step 1 — Re-preprocess HDF5 into `psg_full/` (parallel CPU jobs)

These write to `/scratch/boshra95/psg_full/{dataset}/derived/hdf5_signals/`.
**Existing HDF5 in `psg/` are not touched.**

**Which script to use:**
- Small/medium datasets (STAGES=1513, APPLES=1104, MrOS=3933): use `preprocess_signals_parallel.sh` (single 26 h job each).
- SHHS (8444 subjects): split across ~6 jobs using `preprocess_signals_array.sh` (8 h per batch).

```bash
cd /home/boshra95/NSRR-tools
CFG=--config configs/preprocessing_params_full.yaml

# Small/medium datasets — one job each
sbatch jobs/preprocess_signals_parallel.sh stages $CFG
sbatch jobs/preprocess_signals_parallel.sh apples $CFG
sbatch jobs/preprocess_signals_parallel.sh mros   $CFG

# SHHS — split into 6 batches of ~1400 subjects (8 h each, all run in parallel)
sbatch jobs/preprocess_signals_array.sh shhs    0  1500 $CFG
sbatch jobs/preprocess_signals_array.sh shhs 1500  3000 $CFG
sbatch jobs/preprocess_signals_array.sh shhs 3000  4500 $CFG
sbatch jobs/preprocess_signals_array.sh shhs 4500  6000 $CFG
sbatch jobs/preprocess_signals_array.sh shhs 6000  7500 $CFG
sbatch jobs/preprocess_signals_array.sh shhs 7500  9000 $CFG
```

All 9 jobs can run simultaneously. Safe to re-submit any that fail — existing
HDF5 files are skipped by default.

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
  ✅  configs/preprocessing_params_full.yaml         (strategy=sleepfm_full, base=psg_full)
  ✅  configs/phase0_v3_full_config.yaml             (all paths → psg_full; hidden=128/layers=1 for seq2label)
  ✅  configs/phase0_v3_full_staging_config.yaml     (same paths; hidden=256/layers=2 for sleep staging)
  ✅  experiments/v2_full_registry.yaml              (default config=full; staging exps override to staging config)
  ✅  scripts/gen_commands.py                        (per-experiment config override: exp["config"] > registry["config"])
  ✅  jobs/extract_embeddings_gpu.sh                 (CONFIG overridable; logs → logs_v3_expand_channel)
  ✅  jobs/preprocess_signals_parallel.sh            (logs → logs_v3_expand_channel, auto-requeue)
  ✅  jobs/preprocess_signals_array.sh               (logs → logs_v3_expand_channel, auto-requeue)

  ⬜  TODO: Make train_context_sweep_gpu.sh and infer_subject_windows_gpu.sh use a
           LOGS_DIR env var (currently hardcoded to logs_v3/).
           Until then, training/inference SLURM .out/.err files go to logs_v3_full/
           (controlled by v2_full_registry.yaml logs_dir via gen_commands.py), but
           the internal status JSONL and persistent .log files still go to logs_v3/.
           Track by looking at logs_v3/status/train_* and logs_v3/train_* for the
           expand-channel runs as well — they will be mixed with the existing run logs.

Step 1 — Preprocessing (9 parallel CPU jobs total, no GPU):
  ☐  parallel.sh stages  --config configs/preprocessing_params_full.yaml  (~26 h)
  ☐  parallel.sh apples  --config configs/preprocessing_params_full.yaml  (~26 h)
  ☐  parallel.sh mros    --config configs/preprocessing_params_full.yaml  (~26 h)
  ☐  array.sh shhs 0    1500  --config configs/preprocessing_params_full.yaml  (~8 h × 6)
  ☐  array.sh shhs 1500 3000  --config configs/preprocessing_params_full.yaml
  ☐  array.sh shhs 3000 4500  --config configs/preprocessing_params_full.yaml
  ☐  array.sh shhs 4500 6000  --config configs/preprocessing_params_full.yaml
  ☐  array.sh shhs 6000 7500  --config configs/preprocessing_params_full.yaml
  ☐  array.sh shhs 7500 9000  --config configs/preprocessing_params_full.yaml
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

---

## 8. Head Architecture and Fast→Full Comparability

### Design decision: matched architectures

The full-channel run uses **the same head configs as the fast-channel baseline** for each
task type. This is deliberate: any AUROC difference fast→full is attributable solely to the
richer channel set, not to a larger model.

| Run | Config file | seq2label tasks | Sleep staging |
|---|---|---|---|
| Fast-channel (v3 baseline) | `phase0_v3_config.yaml` | hidden=128, layers=1 (~658K LSTM) | hidden=256, layers=2 (~3.16M LSTM) |
| Full-channel (v3_full) | `phase0_v3_full_config.yaml` | hidden=128, layers=1 (~658K LSTM) | hidden=256, layers=2 (~3.16M LSTM) |

Sleep staging uses a larger head in **both** runs because phase0 showed kappa dropping
0.62 → 0.54 at 10m with the smaller 128/1 config; this is documented in `EXPERIMENTS_GUIDE.md`
§ "Width vs depth rationale". The staging comparison across channel counts is still clean.

### The `hidden_dim` config key

A single yaml value `model.hidden_dim` controls both LSTM hidden size and Transformer d_model:

```yaml
# phase0_v3_full_config.yaml — seq2label section
model:
  input_dim: 512    # fixed: 4 modalities × 128 SleepFM dims
  hidden_dim: 128   # LSTM hidden-state size AND Transformer d_model
  num_layers: 1
  num_heads: 8
  dropout: 0.3
```

Sleep staging uses `configs/phase0_v3_full_staging_config.yaml` (identical to the above but
with `hidden_dim: 256, num_layers: 2`). Each sleep staging entry in `v2_full_registry.yaml`
has an explicit `config:` field pointing to this file; `gen_commands.py` checks
`exp.get("config")` before the registry-level default, so no manual flag is needed — just run
`python scripts/gen_commands.py --registry experiments/v2_full_registry.yaml train sleep_staging_lstm`
as normal and the correct config is used automatically.

### Parameter counts

| Head | Config | Parameters |
|---|---|---|
| LSTMHead (seq2label) | hidden=128, layers=1, BiLSTM | ~658K |
| TransformerHead (seq2label) | d_model=128, heads=8, ff=512, layers=1 | ~264K |
| LSTMHead (sleep staging) | hidden=256, layers=2, BiLSTM | ~3.16M |
| TransformerHead (sleep staging) | d_model=256, heads=8, ff=1024, layers=2 | ~1.7M |
| MeanPoolHead | any | ~1K |

Note: the TransformerHead for sleep staging is ~1.7M params — the config comment saying
"~1M" is incorrect.

### Paper implications

The paper can state cleanly: *"All models in the full-channel experiment use identical
architectures to their fast-channel counterparts. Any performance difference reflects the
richer channel set alone."* No footnote or caveat is needed about head capacity differences.

**Do not change `hidden_dim` or `num_layers` in `phase0_v3_full_config.yaml` after any
full-channel runs have started** — changing them mid-run would make results incomparable
within the full-channel set itself.
