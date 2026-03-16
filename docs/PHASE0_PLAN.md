# Phase 0: Context-Length Sweep — Implementation Plan

**Goal**: Quantify how PSG-based classification performance changes as a function of context length
(number of consecutive SleepFM embeddings fed to the sequence head).

---

## Checkpoints & Encoder

There are **3 checkpoints** in `sleepfm-clinical/sleepfm/checkpoints/`:

| Checkpoint | Architecture | Role | patch_size |
|---|---|---|---|
| `model_base/best.pt` | SetTransformer | **Frozen encoder for ALL tasks** | 640 (= 5 sec @ 128 Hz) |
| `model_sleep_staging/best.pth` | SleepEventLSTMClassifier | Pre-trained staging head (reference only) | N/A |
| `model_diagnosis/best.pth` | DiagnosisFinetuneFullLSTMCOXPH | Pre-trained disease head (reference only) | N/A |

**`model_base` is the only encoder.** The existing task heads are not used — we train our own heads from scratch. Both existing heads take model_base embeddings as input, confirming model_base is the right choice for feature extraction across all tasks.

With patch_size=640 (5-sec patches), achievable context lengths are multiples of 5 sec:

| Duration | N patches |
|---|---|
| 30 sec | 6 |
| 2 min | 24 |
| 5 min | 60 |
| 10 min | 120 |
| 20 min | 240 |
| 40 min | 480 |
| 80 min | 960 |

---

## Task Types

Two fundamentally different classification modes must both be supported:

| Mode | Tasks | Output shape | Label per |
|---|---|---|---|
| **seq→label** | apnea, depression, anxiety, insomnia, sleepiness, fatigue, CVD, rested | `[batch, num_classes]` | Subject/night |
| **seq→seq** | sleep staging (5-class) | `[batch, N, 5]` | 5-sec patch |

Sleep staging uses the same encoder and context-window dataset, but the head outputs per-timestep logits and the loss is computed patch-by-patch (weighted cross-entropy; class weights roughly {Wake:1, N1:4, N2:2, N3:4, REM:3} to handle imbalance).

---

## Architecture Overview

```
Raw PSG (HDF5)
     │
     ▼
model_base SetTransformer (frozen, patch_size=640)
     │
     └─ e[1]: per-patch embeddings  →  [S, 128]   S = T_total × 60  (60 patches per 5-min chunk)
                                                    Saved to disk as [T_total, 128] per subject
                              │
              Context window of N patches  [N, 128]
              (N chosen to match target duration; e.g. N=120 for 10 min)
                              │
              ┌───────────────┴──────────────────┐
              │                                  │
    seq→label head                       seq→seq head
    LSTM / Transformer / Mamba           LSTM (per-patch output)
    [batch, num_classes]                 [batch, N, num_classes]
              │                                  │
    single classification              per-patch classification
    (apnea, depression, ...)             (sleep staging)
```

---

## Step-by-Step Implementation

### Step 1 — Embedding Extraction  (`scripts/extract_sleepfm_embeddings.py`)

**What it does:**
- Loads frozen `model_base` SetTransformer
- Iterates over subjects in `master_targets.parquet`
- For each subject's HDF5 PSG file: processes all 5-min chunks, takes `e[1]` (per-patch embeddings),
  concatenates across the full recording → saves as one `.npy` per subject
- Logs skipped subjects (missing HDF5) in a summary

**Output:**
```
scratch/psg/unified/embeddings/sleepfm_5sec/
  apples/{subject_id}_v{visit}.npy    # shape: [T, 128]  T = total 5-sec patches in recording
  shhs/
  mros/
  stages/
```

**Config:**
```yaml
embedding:
  checkpoint: "/home/boshra95/sleepfm-clinical/sleepfm/checkpoints/model_base"
  channel_groups: "/home/boshra95/sleepfm-clinical/sleepfm/configs/channel_groups.json"
  output_dir: "/home/boshra95/scratch/psg/unified/embeddings/sleepfm_5sec"
  batch_size: 16
  num_workers: 4
```

---

### Step 2 — Context-Window Dataset  (`src/nsrr_tools/datasets/context_window_dataset.py`)

**What it does:**
- Loads `.npy` embeddings for a subject → full-night `[T, 128]`
- Converts context duration string (e.g. `"10m"`) to number of patches N
- Samples N consecutive patches:
  - **Training**: random start; **Val/Test**: center of recording
- Pads shorter recordings (zeros + mask)
- Supports both classification modes via `task_type: "seq2label" | "seq2seq"`
  - For seq2seq (sleep staging): also loads and aligns per-patch stage labels from CSV

**Returns:**
```python
{
  "embeddings": Tensor [N, 128],
  "mask":       Tensor [N],         # 1=real, 0=pad
  "label":      int or Tensor [N],  # int for seq2label; [N] for seq2seq
  "subject_id": str,
  "dataset":    str,
  "visit":      int,
}
```

**Config:**
```yaml
dataset:
  context_lengths: ["5m", "10m", "20m"]    # durations — start here, extend later
  embedding_dir: ".../embeddings/sleepfm_5sec"
  label_source: ".../targets/master_targets.parquet"   # for seq2label tasks
  sleep_stage_dir: ".../targets/sleep_stages"          # for seq2seq (staging)
  task: "apnea_binary"        # any task column, or "sleep_staging"
  task_type: "seq2label"      # "seq2label" | "seq2seq"
  datasets: ["apples", "shhs", "mros", "stages"]
  val_strategy: "center"
  train_split: 0.7
  val_split:   0.15
  test_split:  0.15
  split_seed:  42
```

---

### Step 3 — Sequence Head Models  (`src/nsrr_tools/models/sequence_head.py`)

**Single `SequenceHead` factory, two output modes:**

| Head | seq2label output | seq2seq output |
|---|---|---|
| `MeanPool` | average pool → linear | not applicable |
| `LSTMHead` | last hidden state → linear | all hidden states → linear |
| `TransformerHead` | CLS token → linear | all tokens → linear |
| `MambaHead` | last state → linear | all states → linear |

Input always `[batch, N, 128]`. Mode is set by `task_type`.

**Config:**
```yaml
model:
  head_type:   "lstm"          # "mean_pool" | "lstm" | "transformer" | "mamba"
  task_type:   "seq2label"     # "seq2label" | "seq2seq"
  hidden_dim:  256
  num_layers:  2
  dropout:     0.3
  num_classes: 2               # set automatically from task
```

---

### Step 4 — Training Script  (`scripts/train_context_sweep.py`)

**What it does:**
- Reads `phase0_config.yaml`
- For each context length: train → eval → save checkpoint + metrics
- Metrics for seq2label: AUROC, balanced accuracy, F1, per-class recall
- Metrics for seq2seq (staging): per-class accuracy, macro F1, Cohen's kappa
- Writes `summary.csv` across all context lengths

**Output:**
```
scratch/psg/unified/results/phase0/
  {task}_{head_type}/
    context_5m/
      best_model.pt
      metrics.csv
    context_10m/
      ...
    summary.csv
```

**Config:**
```yaml
training:
  epochs: 30
  lr: 1e-3
  weight_decay: 1e-4
  optimizer: "adam"
  scheduler: "cosine"
  early_stopping_patience: 5
  device: "cuda"
  mixed_precision: true
  # for seq2seq (sleep staging):
  class_weights: [1, 4, 2, 4, 3]   # Wake, N1, N2, N3, REM

logging:
  results_dir: "/home/boshra95/scratch/psg/unified/results/phase0"
  log_every_n_steps: 10
```

---

## Master Config: `configs/phase0_config.yaml`

```yaml
# NOTE: comment/uncomment path blocks when switching between cluster and laptop

embedding:
  checkpoint: "/home/boshra95/sleepfm-clinical/sleepfm/checkpoints/model_base"
  channel_groups: "/home/boshra95/sleepfm-clinical/sleepfm/configs/channel_groups.json"
  output_dir: "/home/boshra95/scratch/psg/unified/embeddings/sleepfm_5sec"
  # Local Mac:
  # checkpoint: "/Users/boshra/NSRR-workspace/sleepfm-clinical/sleepfm/checkpoints/model_base"
  # channel_groups: "/Users/boshra/NSRR-workspace/sleepfm-clinical/sleepfm/configs/channel_groups.json"
  # output_dir: "/Users/boshra/NSRR-workspace/cc_scratch/psg/unified/embeddings/sleepfm_5sec"
  batch_size: 16
  num_workers: 4

dataset:
  context_lengths: ["5m", "10m", "20m"]
  embedding_dir: "/home/boshra95/scratch/psg/unified/embeddings/sleepfm_5sec"
  label_source: "/home/boshra95/scratch/psg/unified/targets/master_targets.parquet"
  task_subject_dir: "/home/boshra95/scratch/psg/unified/targets/task_subjects"
  sleep_stage_dir: "/home/boshra95/scratch/psg/unified/targets/sleep_stages"
  hdf5_dir: "/home/boshra95/scratch/psg"
  # Local Mac:
  # embedding_dir: "/Users/boshra/NSRR-workspace/cc_scratch/psg/unified/embeddings/sleepfm_5sec"
  # label_source: "/Users/boshra/NSRR-workspace/cc_scratch/psg/unified/targets/master_targets.parquet"
  # task_subject_dir: "/Users/boshra/NSRR-workspace/cc_scratch/psg/unified/targets/task_subjects"
  # sleep_stage_dir: "/Users/boshra/NSRR-workspace/cc_scratch/psg/unified/targets/sleep_stages"
  # hdf5_dir: "/Users/boshra/NSRR-workspace/cc_scratch/psg"
  task: "apnea_binary"          # change per experiment
  task_type: "seq2label"        # "seq2label" | "seq2seq"
  datasets: ["apples", "shhs", "mros", "stages"]
  val_strategy: "center"
  train_split: 0.7
  val_split:   0.15
  test_split:  0.15
  split_seed:  42

model:
  head_type:  "lstm"
  task_type:  "seq2label"
  hidden_dim: 256
  num_layers: 2
  dropout:    0.3

training:
  epochs: 30
  lr: 1e-3
  weight_decay: 1e-4
  optimizer: "adam"
  scheduler: "cosine"
  early_stopping_patience: 5
  device: "cuda"
  mixed_precision: true
  class_weights: null           # null = uniform; [1,4,2,4,3] for sleep staging

logging:
  results_dir: "/home/boshra95/scratch/psg/unified/results/phase0"
  # Local Mac:
  # results_dir: "/Users/boshra/NSRR-workspace/cc_scratch/psg/unified/results/phase0"
  log_every_n_steps: 10
```

---

## Implementation Order

| # | File | Depends on | Debug needed |
|---|------|-----------|--------------|
| 1 | `configs/phase0_config.yaml` | — | No |
| 2 | `scripts/extract_sleepfm_embeddings.py` | model_base checkpoint + HDF5 | Yes |
| 3 | `src/nsrr_tools/datasets/context_window_dataset.py` | Step 2 output | Yes |
| 4 | `src/nsrr_tools/models/sequence_head.py` | — | Unit test |
| 5 | `scripts/train_context_sweep.py` | Steps 2–4 | Yes (short run) |

---

## Open Questions / Decisions

- [ ] **Sleep staging labels**: confirm where per-subject stage CSV files are stored on the cluster
- [ ] **Test-time sampling**: center-of-night window vs. majority vote over multiple windows?
- [ ] **Multi-visit subjects** (SHHS, MrOS, APPLES): treat each visit independently, or aggregate across visits?
- [ ] **Class imbalance (seq2label)**: weighted loss, oversampling, or both?
- [ ] **Multiclass tasks**: `num_classes` is the only change needed — confirm no other differences
