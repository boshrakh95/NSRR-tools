# Experiment Execution Guide — Phase 0 / V2 Tasks

This document is the definitive reference for running training, inference, and analysis experiments for both the original Phase 0 tasks and the new V2 task definitions.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Steps](#pipeline-steps)
3. [Config Files](#config-files)
4. [Experiment Registry and Command Generator](#experiment-registry-and-command-generator)
5. [V2 Experiment Plan](#v2-experiment-plan)
6. [Step-by-Step Commands](#step-by-step-commands)
7. [Results Directory Structure](#results-directory-structure)
8. [Expected Runtimes](#expected-runtimes)
9. [Tracking and Status Checks](#tracking-and-status-checks)
10. [Adding New Experiments](#adding-new-experiments)
11. [Regression Tasks (Deferred)](#regression-tasks-deferred)

---

## Overview

The pipeline takes frozen SleepFM embeddings and trains lightweight sequence heads (LSTM, Transformer, MeanPool) to predict clinical labels from PSG signals. Each experiment sweeps over multiple context lengths (how much of the night the model sees) to measure how performance scales with context.

**Three phases per experiment:**
1. **Train** — fit the head on the training split, checkpoint the best model by val_auroc
2. **Infer** — run the best model on every window of every test subject, save per-window probabilities
3. **Analyze** — sweep K=1,5,10,20,50,all windows per subject, compute metrics, write markdown tables and optional plots

---

## Pipeline Steps

### Step 1 — Training

**Script:** `scripts/train_context_sweep.py`  
**Submit via:** `jobs/train_context_sweep_gpu.sh`  
**One job per (experiment, context_length).** Each job trains one head on one context.

Output (per context):
```
results/{phase}/{task}_{head}{_tag}/
  context_{L}/
    best_model.pt        # best checkpoint (by val_auroc)
    metrics.json         # train/val/test metrics at best epoch
  summary.csv            # one row per completed context
```

### Step 2 — Inference

**Script:** `scripts/infer_subject_windows.py`  
**Submit via:** `jobs/infer_subject_windows_gpu.sh`  
**One job per experiment** (auto-discovers all trained contexts).

Output (per context):
```
results/{phase}/inference/{task}_{head}{_tag}/
  context_{L}/
    test_windows.parquet    # per-window: subject_id, true_label, pred_label, prob_class*
    test_subjects.parquet   # per-subject aggregations
```

### Step 3 — Window Analysis

**Script:** `scripts/analyze_windows.py`  
**Run locally** (no GPU needed, fast).

Output (per experiment):
```
results/{phase}/inference/{task}_{head}{_tag}/
  window_analysis.csv      # K-sweep metrics table
  window_analysis.md       # formatted markdown
  window_analysis_auroc.png  # optional plot (--plot flag)
```

---

## Config Files

| Config | Target Dir | Results Dir | Use for |
|--------|-----------|------------|---------|
| `configs/phase0_config.yaml` | `targets/` | `results/phase0` | Original v1 tasks |
| `configs/phase0_v2_config.yaml` | `targets_v2/` | `results/phase0_v2` | New v2 tasks |

The two configs are identical except for the three paths above. Both use the same embeddings, model architecture, and training hyperparameters.

---

## Experiment Registry and Command Generator

All V2 experiments are defined in `experiments/v2_registry.yaml`. Never type experiment parameters manually — always generate commands from the registry.

### Registry format

```yaml
experiments:
  sex_binary_lstm:
    task: sex_binary
    task_type: seq2label
    num_classes: 2
    head: lstm
    datasets: [apples, shhs, stages]
    contexts: [30s, 10m, 40m, 80m]
    batch_size: 32
    lr: 1.0e-4
    run_tag: ""
    tier: 1
    notes: "..."
```

### Command generator: `scripts/gen_commands.py`

```bash
# List all experiments (with status)
python scripts/gen_commands.py list
python scripts/gen_commands.py list --tier 1

# Generate train commands (one per context)
python scripts/gen_commands.py train sex_binary_lstm
python scripts/gen_commands.py train sex_binary_lstm --context 30s 10m

# Generate inference command (auto-uses trained contexts)
python scripts/gen_commands.py infer sex_binary_lstm
python scripts/gen_commands.py infer sex_binary_lstm --split val

# Generate analysis command
python scripts/gen_commands.py analyze sex_binary_lstm
python scripts/gen_commands.py analyze sex_binary_lstm --plot

# Check status for all or one experiment
python scripts/gen_commands.py status
python scripts/gen_commands.py status sex_binary_lstm
```

The `status` command checks for `best_model.pt`, `test_windows.parquet`, and `window_analysis.md` on disk and reports which steps are done without requiring any manual tracking.

### Typical workflow for one experiment

```bash
# 1. Generate and submit training (one sbatch per context)
python scripts/gen_commands.py train sex_binary_lstm
# → copy-paste each printed command, or pipe to bash:
python scripts/gen_commands.py train sex_binary_lstm | bash

# 2. After training completes, generate and submit inference
python scripts/gen_commands.py infer sex_binary_lstm | bash

# 3. After inference, run analysis locally
python scripts/gen_commands.py analyze sex_binary_lstm --plot | bash

# 4. Check what's done
python scripts/gen_commands.py status sex_binary_lstm
```

---

## V2 Experiment Plan

### Tasks overview

| Task | Type | Classes | N total | Datasets | Tier |
|------|------|---------|---------|----------|------|
| `sex_binary` | seq2label | 2 | 13,163 | APPLES, SHHS, STAGES | 1 |
| `sleep_efficiency_binary` | seq2label | 2 | 13,615 | APPLES, SHHS, MrOS | 1 |
| `bmi_binary` | seq2label | 2 | 15,532 | all | 1 |
| `age_class` | seq2label | 3 | 16,007 | all | 1 |
| `psqi_binary` | seq2label | 2 | 3,933 | MrOS | 2 |
| `depression_extreme_binary` | seq2label | 2 | 1,761 | APPLES, STAGES | 2 |
| `osa_binary_apples_postqc` | seq2label | 2 | 1,516 | APPLES | 2 |
| `osa_severity_apples` | seq2label | 4 | 1,516 | APPLES | 2 |
| `age_regression` | regression | — | 16,007 | all | deferred |
| `bmi_regression` | regression | — | 15,532 | all | deferred |

### Experiments per task

**Tier 1** — run all three heads:

| Experiment ID | Head | Contexts | Notes |
|--------------|------|---------|-------|
| `sex_binary_lstm` | lstm | 30s, 10m, 40m, 80m | |
| `sex_binary_transformer` | transformer | 30s, 10m, 40m | 80m skipped (memory) |
| `sex_binary_mean_pool` | mean_pool | 30s, 10m, 40m, 80m | |
| `sleep_efficiency_binary_lstm` | lstm | 30s, 10m, 40m, 80m | |
| `sleep_efficiency_binary_transformer` | transformer | 30s, 10m, 40m | |
| `sleep_efficiency_binary_mean_pool` | mean_pool | 30s, 10m, 40m, 80m | |
| `bmi_binary_lstm` | lstm | 30s, 10m, 40m, 80m | |
| `bmi_binary_transformer` | transformer | 30s, 10m, 40m | |
| `bmi_binary_mean_pool` | mean_pool | 30s, 10m, 40m, 80m | |
| `age_class_lstm` | lstm | 30s, 10m, 40m, 80m | |
| `age_class_transformer` | transformer | 30s, 10m, 40m | |
| `age_class_mean_pool` | mean_pool | 30s, 10m, 40m, 80m | |

**Tier 2** — lstm only initially:

| Experiment ID | Head | Contexts | Notes |
|--------------|------|---------|-------|
| `psqi_binary_lstm` | lstm | 30s, 10m, 40m, 80m | MrOS only |
| `depression_extreme_binary_lstm` | lstm | 30s, 10m, 40m | small N, skip 80m |
| `osa_binary_apples_postqc_lstm` | lstm | 30s, 10m, 40m | APPLES only |
| `osa_severity_apples_lstm` | lstm | 30s, 10m, 40m | APPLES only, 4-class |

**Total training jobs:** 47 (Tier 1: 36, Tier 2: 11)  
**Total inference jobs:** 16 (one per experiment)  
**Total analysis commands:** 16

### Suggested run order

1. Submit Tier 1 lstm experiments first (largest N, fastest to validate)
2. Submit Tier 1 transformer and mean_pool in parallel
3. After Tier 1 trains, run inference + analysis for Tier 1
4. Submit Tier 2 experiments
5. Run inference + analysis for Tier 2

---

## Step-by-Step Commands

### Manual sbatch syntax (if not using gen_commands.py)

**Training:**
```bash
cd /home/boshra95/NSRR-tools

TASK=sex_binary \
TASK_TYPE=seq2label \
HEAD=lstm \
CONTEXT=30s \
DATASETS="apples shhs stages" \
BATCH_SIZE=32 \
LR=1e-4 \
CONFIG=configs/phase0_v2_config.yaml \
sbatch jobs/train_context_sweep_gpu.sh
```

**Inference:**
```bash
TASK=sex_binary \
TASK_TYPE=seq2label \
HEAD=lstm \
CONTEXTS="30s 10m 40m 80m" \
SPLIT=test \
DATASETS="apples shhs stages" \
CONFIG=configs/phase0_v2_config.yaml \
sbatch jobs/infer_subject_windows_gpu.sh
```

**Window analysis:**
```bash
python scripts/analyze_windows.py \
  --task sex_binary \
  --head lstm \
  --results-dir /scratch/boshra95/psg/unified/results/phase0_v2/inference \
  --plot
```

**Collect all results into master CSV (after multiple experiments):**
```bash
python scripts/collect_results.py \
  --results-dir /scratch/boshra95/psg/unified/results/phase0_v2
```

---

## Results Directory Structure

```
/scratch/boshra95/psg/unified/results/phase0_v2/
│
├── master_results.csv              # all training metrics, one row per (task, head, context)
│
├── {task}_{head}/                  # e.g. sex_binary_lstm/
│   ├── summary.csv                 # one row per context (val/test metrics)
│   └── context_{L}/               # e.g. context_30s/
│       ├── best_model.pt
│       └── metrics.json
│
└── inference/
    └── {task}_{head}/             # e.g. sex_binary_lstm/
        ├── window_analysis.csv
        ├── window_analysis.md
        ├── window_analysis_auroc.png   (if --plot)
        └── context_{L}/
            ├── test_windows.parquet
            └── test_subjects.parquet
```

---

## Expected Runtimes

Approximate wall-clock times on H100 (10GB slice):

| Dataset size | Head | Context | Train | Infer |
|-------------|------|---------|-------|-------|
| Large (N~13k) | lstm | 30s | ~1–2 h | ~30 min |
| Large (N~13k) | lstm | 10m | ~2–3 h | ~45 min |
| Large (N~13k) | lstm | 40m | ~3–5 h | ~1 h |
| Large (N~13k) | lstm | 80m | ~5–8 h | ~1.5 h |
| Small (N~1.5k) | lstm | any | ~30 min–1 h | ~15 min |
| Any | transformer | 30s–40m | similar to lstm | similar |
| Any | mean_pool | any | ~30–60 min | ~15 min |

SLURM time limits in job scripts: training=24h, inference=5h. These are safe margins.

---

## Tracking and Status Checks

Use `gen_commands.py status` to check progress without opening any result files:

```bash
python scripts/gen_commands.py status
```

Sample output:
```
============================================================
  sex_binary_lstm  [tier 1]
  task=sex_binary  head=lstm  contexts=['30s', '10m', '40m', '80m']
  Trained:   ['30s', '10m']
  Inferred:  []
  Analyzed:  no

============================================================
  sex_binary_transformer  [tier 1]
  ...
```

The status is determined by presence of files on disk — no manual log required.

For a summary table:
```bash
python scripts/gen_commands.py list
```

---

## Adding New Experiments

1. Add an entry to `experiments/v2_registry.yaml` following the existing format.
2. Use `gen_commands.py` to generate commands — no other changes needed.

If you want a different LR for an existing task (e.g., `lr=3e-4`):
```yaml
sex_binary_lstm_lr3e4:
  task: sex_binary
  ...
  lr: 3.0e-4
  run_tag: "lr3e4"   # creates folder sex_binary_lstm_lr3e4/
```

The `run_tag` is appended to the folder name so results don't overwrite the base run.

---

## Regression Tasks (Deferred)

`age_regression` and `bmi_regression` require a regression head (MSE loss, float output) which is not yet implemented in `train_context_sweep.py`. When implemented:

1. Add `task_type: regression` support to `train_context_sweep.py` and `infer_subject_windows.py`
2. Uncomment the deferred entries in `experiments/v2_registry.yaml`
3. Metrics will differ from classification: use RMSE, MAE, R² instead of AUROC

Labels are already prepared: `age_value` and `bmi_value` float columns in `targets_v2/master_targets.parquet`.

---

## Notes on Specific Tasks

**`sex_binary`**: MrOS excluded — all-male cohort, zero variance. Model trained on APPLES+SHHS+STAGES only.

**`sleep_efficiency_binary`**: STAGES excluded — no sleep efficiency score available. Model trained on APPLES+SHHS+MrOS.

**`age_class`**: 3-class (<50=0, 50–64=1, ≥65=2). MrOS subjects are all class 2 (cohort is 65+). Include MrOS anyway to test generalization.

**`bmi_binary`**: WHO obesity threshold (BMI≥30=1). MrOS labels are visit-1 only (no harmonized v2 BMI); MrOS visit-2 rows are excluded from training labels.

**`depression_extreme_binary`**: Extreme-group design — middle range subjects are excluded from subject lists. Effectively reduces N from 2,794 (all depression subjects) to 1,761 (extreme groups only).

**`osa_binary_apples_postqc`** and **`osa_severity_apples`**: APPLES-only. Relatively small N (~1,516). Use 30s/10m/40m contexts; skip 80m.

**`psqi_binary`**: MrOS-only. Good N for a single dataset (N=3,933 across v1+v2 visits). Both visits contribute (PSQI is visit-specific for MrOS).
