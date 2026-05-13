# Experiment Execution Guide — Phase 0 / V2 Tasks

This document is the definitive reference for running training, inference, and analysis experiments for Phase 0 V2 task definitions.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Steps](#pipeline-steps)
3. [Config Files](#config-files)
4. [Experiment Registry and Command Generator](#experiment-registry-and-command-generator)
5. [Submitting Jobs](#submitting-jobs)
6. [Checkpoint Resume and Auto-Requeue](#checkpoint-resume-and-auto-requeue)
7. [Job Run History and Tracking](#job-run-history-and-tracking)
8. [V2 Experiment Plan](#v2-experiment-plan)
9. [Results Directory Structure](#results-directory-structure)
10. [Expected Runtimes](#expected-runtimes)
11. [Configurable Training K Strategy](#configurable-training-k-strategy)
12. [Adding New Experiments](#adding-new-experiments)
13. [Regression Tasks (Deferred)](#regression-tasks-deferred)
14. [Notes on Specific Tasks](#notes-on-specific-tasks)
15. [Results Collection](#results-collection)

---

## Overview

The pipeline takes frozen SleepFM embeddings and trains lightweight sequence heads (LSTM, Transformer, MeanPool) to predict clinical labels from PSG signals. Each experiment sweeps over multiple context lengths (how much of the night the model sees) to measure how performance scales with context.

**Five phases per experiment:**
1. **Train** — fit the head on the training split, checkpoint the best model by val_auroc
2. **Infer** — run the best model on every window of every test subject, save per-window probabilities
3. **Analyze** — sweep K=1,5,10,20,50,all windows per subject, compute metrics, write markdown tables and optional plots
4. **Iso-compute analysis** — dense K sweep (`--k-dense`, ~25 values), build heatmap DataFrame, produce 7 iso-compute plots: heatmap, metric-vs-k, metric-vs-total-context, Pareto front, min-cost frontier, marginal gain, double-tradeoff
5. **Saturation curve** — AUROC/balanced_accuracy vs context length per head; the primary "Figure 1" for the paper

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
    best_model.pt           # best checkpoint (by val_auroc)
    resume.pt               # per-epoch resume checkpoint (deleted on success)
    metrics.json            # train/val/test metrics at best epoch
    training_curves.csv     # per-epoch: loss, bal_acc, val monitor (written on completion)
  summary.csv            # one row per completed context length (val + test metrics)
```

### Step 2 — Inference

**Script:** `scripts/infer_subject_windows.py`  
**Submit via:** `jobs/infer_subject_windows_gpu.sh`  
**One job per experiment** (auto-discovers all trained contexts; already-done contexts are skipped).

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
  window_analysis_{split}.csv   # K-sweep metrics table
  window_analysis.md            # formatted markdown tables (all splits)
  {task}_{head}_{split}_window_sweep_{metric}.png  # optional plot (--plot)
```

### Step 4 — Iso-Compute Analysis (dense K sweep → plots)

**Scripts:** `scripts/analyze_windows.py --k-dense`, `scripts/build_heatmap_df.py`, `scripts/plot_iso_compute.py`  
**Run locally** (no GPU needed). Steps must run in order.

```bash
# 4a. Dense K sweep (~25 K values per context; used for heatmap/iso-compute)
python scripts/gen_commands.py analyze sex_binary_lstm --k-dense | bash

# 4b. Build heatmap DataFrame (parses context strings, renames columns, adds total_compute_min)
python scripts/gen_commands.py build-heatmap sex_binary_lstm | bash

# 4c. Produce all 7 iso-compute plots
python scripts/gen_commands.py iso-plots sex_binary_lstm | bash
```

Output:
```
results/{phase}/inference/{task}_{head}/
  window_analysis_{split}.csv       # updated with dense K values
  heatmap_df_{split}.csv            # heatmap-ready: context_length_min, k, auroc, total_compute_min, ...

results/{phase}/figures/{task}_{head}/{metric}_{split}/
  heatmap_{metric}.{png,pdf}
  metric_vs_k_{metric}.{png,pdf}
  metric_vs_total_{metric}.{png,pdf}
  pareto_front_{metric}.{png,pdf}
  min_cost_frontier_{metric}.{png,pdf}
  marginal_gain_{metric}.{png,pdf}
  double_tradeoff_{metric}.{png,pdf}
```

### Step 5 — Saturation Curve (Figure 1)

**Script:** `scripts/plot_saturation.py`  
**Run locally.** Reads `summary.csv` per head — no dense K sweep needed.

```bash
python scripts/gen_commands.py saturation sex_binary --heads lstm transformer mean_pool | bash
```

Output:
```
results/{phase}/figures/
  saturation_{task}_{metric}_{split}.{png,pdf}   # one line per head, x=context length
```

---

## Config Files

| Config | Target Dir | Results Dir | Use for |
|--------|-----------|------------|---------|
| `configs/phase0_v2_config.yaml` | `targets_v2/` | `results/phase0_v2` | V2 tasks (current) |
| `configs/phase0_config.yaml` | `targets/` | `results/phase0` | Original v1 tasks |

---

## Experiment Registry and Command Generator

All experiments are defined in `experiments/v2_registry.yaml`. **Always generate commands from the registry** — never type parameters manually.

### Registry format

```yaml
config: configs/phase0_v2_config.yaml
results_dir: /scratch/boshra95/psg/unified/results/phase0_v2
inference_dir: /scratch/boshra95/psg/unified/results/phase0_v2/inference
logs_dir: /home/boshra95/NSRR-tools/logs_v2

experiments:
  sex_binary_lstm:
    task: sex_binary
    task_type: seq2label
    num_classes: 2
    head: lstm
    datasets: [apples, shhs, stages]
    contexts: [30s, 10m, 40m, 80m, 120m]
    batch_size: 32
    lr: 1.0e-4
    run_tag: ""
    n_size: large       # large (N>10k) | medium (N~3-5k) | small (N<2k)
    tier: 1
```

### Command generator: `scripts/gen_commands.py`

```bash
# List all experiments (with training/inference/analysis status)
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
python scripts/gen_commands.py analyze sex_binary_lstm --k-dense   # dense K sweep for iso-compute

# Iso-compute analysis pipeline (Steps 4a–4c above)
python scripts/gen_commands.py build-heatmap sex_binary_lstm
python scripts/gen_commands.py build-heatmap sex_binary_lstm --split val
python scripts/gen_commands.py iso-plots sex_binary_lstm
python scripts/gen_commands.py iso-plots sex_binary_lstm --metric auroc --budget 240

# Saturation curve (Figure 1: metric vs context length per head)
python scripts/gen_commands.py saturation sex_binary
python scripts/gen_commands.py saturation sex_binary --heads lstm transformer mean_pool
python scripts/gen_commands.py saturation sex_binary --metric auroc balanced_accuracy

# Check file-level status (trained / inferred / analyzed)
python scripts/gen_commands.py status
python scripts/gen_commands.py status sex_binary_lstm

# Check job run history (from JSONL tracking files)
python scripts/gen_commands.py runs
python scripts/gen_commands.py runs sex_binary_lstm
python scripts/gen_commands.py runs sex_binary_lstm -v   # verbose: show full history
```

---

## Submitting Jobs

### Using gen_commands.py (recommended)

The generator produces complete sbatch commands with wall-time estimates, log paths, and `--requeue` already included. You do not need to edit any bash file.

```bash
cd /home/boshra95/NSRR-tools

# Print commands (review before submitting)
python scripts/gen_commands.py train sex_binary_lstm

# Submit all contexts for one experiment
python scripts/gen_commands.py train sex_binary_lstm | bash

# Submit a single specific context (copy-paste the one line you want)
python scripts/gen_commands.py train sex_binary_lstm --context 30s | bash

# Submit inference after training
python scripts/gen_commands.py infer sex_binary_lstm | bash

# Run analysis locally (no GPU needed)
python scripts/gen_commands.py analyze sex_binary_lstm --plot | bash
```

A generated train command looks like:
```bash
TASK=sex_binary TASK_TYPE=seq2label HEAD=lstm CONTEXT=30s \
  DATASETS="apples shhs" BATCH_SIZE=32 LR=0.0001 \
  CONFIG=configs/phase0_v2_config.yaml \
  sbatch --requeue \
    --time=04:00:00 \
    --output=/home/boshra95/NSRR-tools/logs_v2/train_sex_binary_lstm_30s_lr1e-4_%j.out \
    --error=/home/boshra95/NSRR-tools/logs_v2/train_sex_binary_lstm_30s_lr1e-4_%j.err \
    /home/boshra95/NSRR-tools/jobs/train_context_sweep_gpu.sh
```

Key things that are set automatically:
- `--requeue` — SLURM auto-requeues on timeout; training resumes from last saved epoch
- `--time` — estimated from the `n_size` and head/context in the registry lookup table
- `--output` / `--error` — go to `logs_v2/` with filename encoding task/head/context/lr/job-id
- `CONFIG` — points to the v2 config file

### Manual sbatch (when not using gen_commands.py)

If you must submit manually, include these flags:

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
sbatch --requeue \
  --time=04:00:00 \
  --output=logs_v2/train_sex_binary_lstm_30s_lr1e-4_%j.out \
  --error=logs_v2/train_sex_binary_lstm_30s_lr1e-4_%j.err \
  jobs/train_context_sweep_gpu.sh
```

> **Important:** `--time`, `--output`, `--error` on the command line override the `#SBATCH` defaults inside the script. The bash scripts default to `logs_v2/` and a 24h time limit — the generator tightens the time limit per-context.

### Typical workflow for one experiment

```bash
# 1. Train all contexts (jobs run in parallel on cluster)
python scripts/gen_commands.py train sex_binary_lstm | bash

# 2. Monitor progress
python scripts/gen_commands.py runs sex_binary_lstm      # job history
python scripts/gen_commands.py status sex_binary_lstm    # file-level progress

# 3. After training, infer
python scripts/gen_commands.py infer sex_binary_lstm | bash

# 4. After inference: standard analysis (sparse K, markdown table + plots)
python scripts/gen_commands.py analyze sex_binary_lstm --plot | bash

# 5. Iso-compute analysis (dense K sweep → 7 plots per metric)
python scripts/gen_commands.py analyze sex_binary_lstm --k-dense | bash
python scripts/gen_commands.py build-heatmap sex_binary_lstm | bash
python scripts/gen_commands.py iso-plots sex_binary_lstm | bash

# 6. Saturation curve — once all three heads are trained for this task
python scripts/gen_commands.py saturation sex_binary --heads lstm transformer mean_pool | bash

# 7. Collect all results into flat CSVs (run after any new training or inference)
python scripts/collect_results_v2.py
git add results/collected/ && git commit -m "collect results" && git push

# 8. Check all experiments at once
python scripts/gen_commands.py list
```

---

## Checkpoint Resume and Auto-Requeue

Training jobs automatically handle timeouts without losing work.

### How it works

**Per-epoch checkpoint:** After every epoch, `train_context_sweep.py` saves a `resume.pt` file in the same `context_{L}/` directory as `best_model.pt`. This file contains:
- Current epoch number
- Model, optimizer, and scheduler state dicts
- Best monitor metric seen so far (`best_monitor`)
- Patience counter (`no_improve`)
- Full epoch history
- W&B run ID (for seamless run continuation in W&B)
- Accumulated training time across all restarts

**Auto-requeue on timeout:** All jobs are submitted with `--requeue`. When SLURM kills a job at its time limit, SLURM automatically requeues it with the same parameters. The requeued job restarts the bash script from the top, but the Python training code detects `resume.pt` and picks up from the last completed epoch.

**W&B continuity:** When resuming, W&B is initialized with `id=saved_run_id, resume="must"` so the resumed run appends metrics to the same W&B run — no duplicate entries.

**Cleanup on success:** When training completes and `metrics.json` is written successfully, `resume.pt` is deleted. This is the signal that the context is fully done — the existing `best_model.pt` skips logic at the outer loop is still the primary completion check.

### Scenarios

| Scenario | What happens |
|----------|-------------|
| Job completes normally | `metrics.json` written, `resume.pt` deleted, context skipped on any resubmit |
| Job times out | SLURM requeues automatically; next run resumes from last epoch |
| Node fails mid-epoch | SLURM requeues; epoch not committed → resumes from previous epoch |
| You resubmit manually | `resume.pt` found → resumes; `metrics.json` found → skips entirely |
| You want a forced fresh restart | Delete both `best_model.pt` and `resume.pt` from that `context_{L}/` directory |

### Force a fresh restart

```bash
# For one context:
rm /scratch/boshra95/psg/unified/results/phase0_v2/sex_binary_lstm/context_30s/best_model.pt
rm /scratch/boshra95/psg/unified/results/phase0_v2/sex_binary_lstm/context_30s/resume.pt

# Then resubmit normally
python scripts/gen_commands.py train sex_binary_lstm --context 30s | bash
```

---

## Job Run History and Tracking

Every job writes structured events to a JSONL file in `logs_v2/status/`. These files are separate from SLURM log output.

### Status files

- **Train:** `logs_v2/status/train_{task}_{head}_{context}_lr{lr}.jsonl`
- **Infer:** `logs_v2/status/infer_{task}_{head}_{split}.jsonl`

Each line is one event (one JSON object):
```json
{"ts":"2026-05-03T09:20:07-07:00","job_id":"38373667","restart":0,"node":"fc11004","status":"STARTED","task":"sex_binary","head":"lstm","context":"10m","lr":"0.0001","datasets":"apples shhs"}
{"ts":"2026-05-03T10:42:24-07:00","job_id":"38373667","restart":0,"node":"fc11004","status":"SUCCESS","task":"sex_binary","head":"lstm","context":"10m","lr":"0.0001","datasets":"apples shhs"}
```

Event statuses: `STARTED`, `REQUEUED`, `TIMEOUT_REQUEUED`, `SUCCESS`, `FAILED`

### Querying history

```bash
# All jobs
python scripts/gen_commands.py runs

# Filter to one experiment (all contexts and splits)
python scripts/gen_commands.py runs sex_binary_lstm

# Verbose: always show full event list
python scripts/gen_commands.py runs sex_binary_lstm -v
```

Sample output:
```
────────────────────────────────────────────────────────────────────────
  train_sex_binary_lstm_30s_lr1e-4
  Latest status : SUCCESS
  Attempts      : 1  |  Events: 2
  Latest job    : 38373667  node=fc11004  ts=2026-05-03T10:42:24-07:00
  History:
    [2026-05-03T09:20:07-07:00] status=STARTED              job=38373667  node=fc11004
    [2026-05-03T10:42:24-07:00] status=SUCCESS              job=38373667  node=fc11004

────────────────────────────────────────────────────────────────────────
  train_sex_binary_lstm_10m_lr1e-4
  Latest status : TIMEOUT_REQUEUED
  Attempts      : 2  |  Events: 4
  Latest job    : 38400001  node=fc11006  ts=2026-05-04T12:00:00-07:00
```

### Log files

Raw SLURM stdout/stderr go to `logs_v2/` with descriptive filenames:

```
logs_v2/train_sex_binary_lstm_30s_lr1e-4_38373667.out
logs_v2/train_sex_binary_lstm_30s_lr1e-4_38373667.err
logs_v2/infer_sex_binary_lstm_lr1e-4_38400002.out
```

---

## V2 Experiment Plan

### Tasks overview

| Task | Type | Classes | N total | Datasets | Tier |
|------|------|---------|---------|----------|------|
| `sex_binary` | seq2label | 2 | ~13k | APPLES, SHHS | 1 |
| `sleep_efficiency_binary` | seq2label | 2 | ~13k | APPLES, SHHS, MrOS | 1 |
| `bmi_binary` | seq2label | 2 | ~15k | APPLES, SHHS, MrOS | 1 |
| `age_class` | seq2label | 3 | ~16k | APPLES, SHHS, MrOS | 1 |
| `psqi_binary` | seq2label | 2 | ~4k | MrOS | 2 |
| `depression_extreme_binary` | seq2label | 2 | ~1.8k | APPLES | 2 |
| `osa_binary_apples_postqc` | seq2label | 2 | ~1.5k | APPLES | 2 |
| `osa_severity_apples` | seq2label | 4 | ~1.5k | APPLES | 2 |

Context lengths:
- **Tier 1:** `30s, 10m, 40m, 80m, 120m, 240m`
- **Tier 2:** `30s, 10m, 40m, 80m, 120m` (smaller N; 240m not added)

### Experiments per task

**Tier 1** — all three heads:

| Experiment ID | Head | Notes |
|--------------|------|-------|
| `sex_binary_lstm` | lstm | |
| `sex_binary_transformer` | transformer | |
| `sex_binary_mean_pool` | mean_pool | batch_size=64 |
| `sleep_efficiency_binary_lstm` | lstm | |
| `sleep_efficiency_binary_transformer` | transformer | |
| `sleep_efficiency_binary_mean_pool` | mean_pool | batch_size=128 |
| `bmi_binary_lstm` | lstm | |
| `bmi_binary_transformer` | transformer | |
| `bmi_binary_mean_pool` | mean_pool | batch_size=128 |
| `age_class_lstm` | lstm | 3-class |
| `age_class_transformer` | transformer | 3-class |
| `age_class_mean_pool` | mean_pool | 3-class, batch_size=128 |

**Tier 2** — lstm only:

| Experiment ID | Head | Notes |
|--------------|------|-------|
| `psqi_binary_lstm` | lstm | MrOS only |
| `depression_extreme_binary_lstm` | lstm | APPLES only, extreme-group design |
| `osa_binary_apples_postqc_lstm` | lstm | APPLES only |
| `osa_severity_apples_lstm` | lstm | APPLES only, 4-class |

### Suggested run order

1. Submit Tier 1 lstm across all contexts first (validate pipeline end-to-end)
2. Submit Tier 1 transformer and mean_pool in parallel
3. After Tier 1 trains → run inference + analysis
4. Submit Tier 2
5. Run inference + analysis for Tier 2

---

## Results Directory Structure

```
/scratch/boshra95/psg/unified/results/phase0_v2/
│
├── {task}_{head}/                    # e.g. sex_binary_lstm/
│   ├── summary.csv                   # one row per context (val/test metrics)
│   └── context_{L}/                 # e.g. context_30s/
│       ├── best_model.pt             # best checkpoint by val_auroc
│       ├── resume.pt                 # per-epoch resume checkpoint (deleted on success)
│       ├── metrics.json              # final metrics (presence = context is done)
│       └── training_curves.csv       # per-epoch loss/bal_acc/monitor (written on completion)
│
├── inference/
│   └── {task}_{head}/
│       ├── window_analysis_{split}.csv   # K-sweep metrics (sparse; or dense with --k-dense)
│       ├── window_analysis.md            # markdown tables (all splits)
│       ├── heatmap_df_{split}.csv        # iso-compute-ready: context_min, k, auroc, total_compute_min
│       └── context_{L}/
│           ├── {split}_windows.parquet   # per-window predictions
│           └── {split}_subjects.parquet  # per-subject aggregations
│
└── figures/
    ├── saturation_{task}_{metric}_{split}.{png,pdf}   # Step 5: head comparison
    └── {task}_{head}/
        └── {metric}_{split}/           # Step 4: iso-compute plots
            ├── heatmap_{metric}.{png,pdf}
            ├── metric_vs_k_{metric}.{png,pdf}
            ├── metric_vs_total_{metric}.{png,pdf}
            ├── pareto_front_{metric}.{png,pdf}
            ├── min_cost_frontier_{metric}.{png,pdf}
            ├── marginal_gain_{metric}.{png,pdf}
            └── double_tradeoff_{metric}.{png,pdf}

/home/boshra95/NSRR-tools/logs_v2/
├── train_sex_binary_lstm_30s_lr1e-4_{job_id}.out    # SLURM stdout
├── train_sex_binary_lstm_30s_lr1e-4_{job_id}.err    # SLURM stderr
└── status/
    ├── train_sex_binary_lstm_30s_lr1e-4.jsonl       # event log for this context
    ├── train_sex_binary_lstm_10m_lr1e-4.jsonl
    └── infer_sex_binary_lstm_test.jsonl
```

---

## Expected Runtimes

Wall-time estimates on H100 (10 GB slice). The generator uses these to set `--time` automatically — no manual adjustment needed unless you change contexts or n_size.

### Training (per context, approximate)

| n_size | Head | 30s | 10m | 40m | 80m | 120m | 240m |
|--------|------|-----|-----|-----|-----|------|------|
| large (N>10k) | lstm | 2h | 3h | 3h | 4h | 6h | 12h |
| large | transformer | 2h | 3h | 4h | 8h | 12h | 24h |
| large | mean_pool | 1h | 1h | 1h | 1h | 1h | 2h |
| medium (N~3-5k) | lstm | 1h | 2h | 2h | 3h | 4h | 8h |
| small (N<2k) | lstm | 1h | 1h | 1h | 2h | 2h | 4h |

Calibrated from observed sex_binary_lstm runs on H100 (large, K=5, batch=32): 30s=31min, 10m=46min, 40m=80min, 80m=111min, 120m=185min. Estimates include ~50% safety margin over 30 epochs worst case. Checkpoint resume means a timeout just causes one requeue — no data loss.

### Inference (one job runs all contexts)

Total time scales roughly linearly with number of contexts. The generator estimates this from the trained-context list automatically.

### Observed actual times (H100, sex_binary_lstm)

| Context | Epochs | Train time | Test AUROC |
|---------|--------|------------|-----------|
| 30s | 18 | 31 min | 0.741 |
| 10m | 17 | 46 min | 0.757 |
| 80m | — | in progress | — |

---

## Configurable Training K Strategy

Training K (windows sampled per subject per epoch) can be controlled via `configs/phase0_v2_config.yaml`:

```yaml
training:
  windows_strategy: "fixed"        # "fixed" | "token_budget"
  token_budget_minutes: 80         # used only when windows_strategy = "token_budget"
```

- **`fixed`** (default): K = `dataset.windows_per_subject` (currently 5) for all context lengths.
- **`token_budget`**: K = `max(1, floor(token_budget_minutes / ctx_minutes))` so that the total signal seen per subject is approximately constant across context lengths (e.g., 80min budget → K=160 for 30s context, K=2 for 40m context).

The logic is applied in `train_context_sweep.py` before dataset construction and prints the K value at training time. Switching from `"fixed"` to `"token_budget"` requires retraining (and changing the config). To run a token-budget ablation without overwriting the K=5 baseline, use a separate `run_tag` and a separate config file — see `docs/context_length_experiment_design.md` §12 for details.

---

## Adding New Experiments

### New task or head

Add an entry to `experiments/v2_registry.yaml` following the existing format:

```yaml
my_new_task_lstm:
  task: my_new_task
  task_type: seq2label
  num_classes: 2
  head: lstm
  datasets: [apples, shhs]
  contexts: [30s, 10m, 40m, 80m]
  batch_size: 32
  lr: 1.0e-4
  run_tag: ""
  n_size: large
  tier: 1
  notes: ""
```

Then use `gen_commands.py` as normal. No other changes needed.

### New context length

Add the new length to the `contexts` list in the registry. If the length is not in the wall-time lookup table in `scripts/gen_commands.py`, also add it to `_TRAIN_HOURS` and `_INFER_HOURS_PER_CTX`. Otherwise it defaults to 24h, which is safe but wasteful.

### Different LR for an existing task

```yaml
sex_binary_lstm_lr3e4:
  task: sex_binary
  ...
  lr: 3.0e-4
  run_tag: "lr3e4"   # creates folder sex_binary_lstm_lr3e4/ — won't overwrite base run
```

---

## Regression Tasks (Deferred)

`age_regression` and `bmi_regression` require a regression head (MSE loss, float output) not yet implemented in `train_context_sweep.py`. When implemented:

1. Add `task_type: regression` support to `train_context_sweep.py` and `infer_subject_windows.py`
2. Uncomment the deferred entries in `experiments/v2_registry.yaml`
3. Metrics will differ: use RMSE, MAE, R² instead of AUROC

Labels are already prepared: `age_value` and `bmi_value` float columns in `targets_v2/master_targets.parquet`.

---

## Notes on Specific Tasks

**`sex_binary`**: MrOS excluded — all-male cohort, zero variance. Trained on APPLES+SHHS only.

**`sleep_efficiency_binary`**: STAGES excluded — no sleep efficiency score. Trained on APPLES+SHHS+MrOS.

**`age_class`**: 3-class (<50=0, 50–64=1, ≥65=2). MrOS subjects are all class 2 (cohort is 65+). Included anyway to test generalization.

**`bmi_binary`**: WHO obesity threshold (BMI≥30=1). MrOS visit-2 rows excluded from training (no harmonized v2 BMI available).

**`depression_extreme_binary`**: Extreme-group design — middle BDI/PHQ-9 range subjects dropped. Reduces effective N from ~2.8k to ~1.8k.

**`osa_binary_apples_postqc`** and **`osa_severity_apples`**: APPLES-only. Small N (~1,516). Context lengths up to 120m included; be cautious with 80m+ given small per-split N.

**`psqi_binary`**: MrOS-only. Both MrOS visits contribute (PSQI is visit-specific). N=~3,933 across both visits.

---

## Results Collection

**Script:** `scripts/collect_results_v2.py`  
**Run from:** either cluster, any time after new training or inference results are available.

```bash
cd /home/boshra95/NSRR-tools
python scripts/collect_results_v2.py
```

The script scans the scratch results directory and appends new rows to three output files:

| File | Content | Location |
|------|---------|----------|
| `results/collected/training.csv` | One row per (task, head, context, epoch) | Repo + scratch |
| `results/collected/analysis.csv` | One row per (task, head, context, K, split) | Repo + scratch |
| `collected/predictions/{task}_{head}_{ctx}_{split}.parquet` | Per-window probabilities | Scratch only |

The CSVs are committed to the repo and synced across clusters via git. The parquets are scratch-only (too large for git) and accumulate independently on each cluster.

**Sync workflow:**
```bash
# After collecting results on one cluster:
git add results/collected/ && git commit -m "collect results" && git push

# Before collecting on the other cluster:
git pull
python scripts/collect_results_v2.py
git add results/collected/ && git commit -m "collect results" && git push
```

**Using the collected files for analysis:**
- `training.csv` filtered by `is_best_epoch == True` → paper performance tables
- `training.csv` all rows → learning curve plots
- `analysis.csv` with `k == "all"` → saturation curve (Figure 1)
- `analysis.csv` with `total_compute_min` column → iso-compute heatmap and Pareto plots
- `predictions/*.parquet` → custom aggregations, ROC at iso-compute, per-dataset breakdowns

See `docs/RESULTS_COLLECTION.md` for the full column schemas and detailed usage examples.
