# Experiment Execution Guide — Phase 0 / V3 Protocol

This document is the definitive reference for running training, inference, and analysis experiments for Phase 0 V2 task definitions.

> **V3 protocol (current):** Results are written to `phase0_v3/`, logs to `logs_v3/`. Training uses overlapping-window fixed-K sampling (K=5 per subject at all context lengths); **val/test/inference use non-overlapping stride-N windows** (T//N positions) to avoid redundant windows at evaluation time. Context-specific LR at 120m/240m and varying batch size recorded in metrics.json. Use `configs/phase0_v3_config.yaml` and `experiments/v2_registry.yaml` (already updated). Do NOT mix v2 and v3 results in the same comparison figure. See [TRAINING_PROTOCOL_FIXES.md](TRAINING_PROTOCOL_FIXES.md) for the rationale behind each change.
>
> **Note on paths:** All examples in this document use V3 paths (`phase0_v3/`, `logs_v3/`). The archived V2 config (`phase0_v2_config.yaml`) is shown only in the Config Files table.

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
12. [K Windows: Training vs Val vs Inference](#k-windows-training-vs-val-vs-inference)
13. [Batch Size Protocol](#batch-size-protocol)
14. [Adding New Experiments](#adding-new-experiments)
15. [Regression Tasks (Deferred)](#regression-tasks-deferred)
16. [Notes on Specific Tasks](#notes-on-specific-tasks)
17. [Results Collection](#results-collection)

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
| `configs/phase0_v3_config.yaml` | `targets_v2/` | `results/phase0_v3` | **V3 tasks (current)** |
| `configs/phase0_v2_config.yaml` | `targets_v2/` | `results/phase0_v2` | V2 tasks (archived) |
| `configs/phase0_config.yaml` | `targets/` | `results/phase0` | Original v1 tasks |

---

## Experiment Registry and Command Generator

All experiments are defined in `experiments/v2_registry.yaml`. **Always generate commands from the registry** — never type parameters manually.

### Registry format

```yaml
config: configs/phase0_v3_config.yaml
results_dir: /scratch/boshra95/psg/unified/results/phase0_v3
inference_dir: /scratch/boshra95/psg/unified/results/phase0_v3/inference
logs_dir: /home/boshra95/NSRR-tools/logs_v3

experiments:
  sex_binary_lstm:
    task: sex_binary
    task_type: seq2label
    num_classes: 2
    head: lstm
    datasets: [apples, shhs]
    contexts: [30s, 10m, 40m, 80m, 120m, 240m]
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
python scripts/gen_commands.py analyze sex_binary_lstm --k-dense                 # dense K sweep for iso-compute
python scripts/gen_commands.py analyze sex_binary_lstm --k-dense --bootstrap 1000  # + bootstrap 95% CIs

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
  DATASETS="apples shhs" BATCH_SIZE=32 ACCUM_STEPS=1 LR=1e-4 \
  CONFIG=configs/phase0_v3_config.yaml \
  sbatch --requeue \
    --time=01:30:00 \
    --output=/home/boshra95/NSRR-tools/logs_v3/train_sex_binary_lstm_30s_lr1e-4_%j.out \
    --error=/home/boshra95/NSRR-tools/logs_v3/train_sex_binary_lstm_30s_lr1e-4_%j.err \
    /home/boshra95/NSRR-tools/jobs/train_context_sweep_gpu_rorqual.sh
```

Key things that are set automatically:
- `--requeue` — SLURM auto-requeues on node failure; wall-time timeouts are handled by the USR1 handler
- `--time` — estimated from the `n_size` and head/context in the registry lookup table
- `--output` / `--error` — go to `logs_v3/` with filename encoding task/head/context/lr/job-id
- `ACCUM_STEPS` — set per-context so `BATCH_SIZE × ACCUM_STEPS = 32` (gradient accumulation mode)
- `CONFIG` — points to the v3 config file

### Manual sbatch (when not using gen_commands.py)

If you must submit manually, include these flags:

```bash
cd /home/boshra95/NSRR-tools

TASK=sex_binary \
TASK_TYPE=seq2label \
HEAD=lstm \
CONTEXT=30s \
DATASETS="apples shhs" \
BATCH_SIZE=32 \
ACCUM_STEPS=1 \
LR=1e-4 \
CONFIG=configs/phase0_v3_config.yaml \
sbatch --requeue \
  --time=01:30:00 \
  --output=logs_v3/train_sex_binary_lstm_30s_lr1e-4_%j.out \
  --error=logs_v3/train_sex_binary_lstm_30s_lr1e-4_%j.err \
  jobs/train_context_sweep_gpu_rorqual.sh
```

> **Important:** `--time`, `--output`, `--error` on the command line override the `#SBATCH` defaults inside the script. The bash scripts default to `logs_v3/` and a 24h time limit — the generator tightens the time limit per-context.

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

# 5. Iso-compute analysis (dense K sweep + bootstrap CIs → 7 plots per metric)
python scripts/gen_commands.py analyze sex_binary_lstm --k-dense --bootstrap 1000 | bash
python scripts/gen_commands.py build-heatmap sex_binary_lstm | bash
python scripts/gen_commands.py iso-plots sex_binary_lstm | bash

# 6. Saturation curve — once all three heads are trained for this task
python scripts/gen_commands.py saturation sex_binary --heads lstm transformer mean_pool | bash

# 7. Collect all results into flat CSVs (prerequisite for scaling-laws and task-comparison)
python scripts/gen_commands.py collect sex_binary_lstm sex_binary_transformer sex_binary_mean_pool | bash

# 8. Extended plots (run after collect; or use run_analysis.sh to do steps 4–8 in one command)
python scripts/gen_commands.py scaling-laws sex_binary --heads lstm transformer mean_pool | bash
python scripts/gen_commands.py calibration sex_binary_lstm | bash
# ... see scripts/run_analysis.sh for the full 13-step pipeline

# 9. Check all experiments at once
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

**Auto-requeue on timeout (two mechanisms):**

- **Wall-time handler (`--signal=B:USR1@120`):** SLURM sends SIGUSR1 to the bash process 120 seconds before the wall-time limit. The bash script's `_timeout_handler` trap fires, kills Python cleanly (SIGTERM), then manually calls `sbatch` to resubmit the same script with `--export=ALL` and an explicit `--output`/`--error` that preserves the descriptive log filename (e.g. `train_sex_binary_lstm_30s_lr1e-4_%j.out`). The new job picks up from `resume.pt`.

- **Node-failure requeue (`--requeue`):** All jobs are also submitted with `--requeue`. If SLURM cancels a job due to a node failure or preemption (not wall-time), SLURM automatically requeues it. The requeued job resumes from `resume.pt` as usual.

Both mechanisms land in the same persistent `.log` file (via `tee -a`) and write a `TIMEOUT_REQUEUED` or `REQUEUED` event to the JSONL status file.

**W&B continuity:** When resuming, W&B is initialized with `id=saved_run_id, resume="must"` so the resumed run appends metrics to the same W&B run — no duplicate entries.

**Cleanup on success:** When training completes and `metrics.json` is written successfully, `resume.pt` is deleted. This is the signal that the context is fully done — the existing `best_model.pt` skips logic at the outer loop is still the primary completion check.

### Scenarios

| Scenario | What happens |
|----------|-------------|
| Job completes normally | `metrics.json` written, `resume.pt` deleted, context skipped on any resubmit |
| Job times out | USR1 handler fires 120s early → kills Python → sbatch resubmit → resumes from last epoch |
| Node fails mid-epoch | SLURM `--requeue` fires; epoch not committed → resumes from previous epoch |
| You resubmit manually | `resume.pt` found → resumes; `metrics.json` found → skips entirely |
| You want a forced fresh restart | Delete both `best_model.pt` and `resume.pt` from that `context_{L}/` directory |

### Force a fresh restart

```bash
# For one context (substitute phase0_v3 for current runs):
rm /scratch/boshra95/psg/unified/results/phase0_v3/sex_binary_lstm/context_30s/best_model.pt
rm /scratch/boshra95/psg/unified/results/phase0_v3/sex_binary_lstm/context_30s/resume.pt

# Then resubmit normally
python scripts/gen_commands.py train sex_binary_lstm --context 30s | bash
```

---

## Job Run History and Tracking

Every job writes structured events to a JSONL file in `logs_v3/status/` (v3 protocol). These files are separate from SLURM log output.

### Status files

- **Train:** `logs_v3/status/train_{task}_{head}_{context}_lr{lr}.jsonl`
- **Infer:** `logs_v3/status/infer_{task}_{head}_{split}.jsonl`

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

Raw SLURM stdout/stderr go to `logs_v3/` with descriptive filenames. Both the original submission and any resubmitted jobs (after timeout or node failure) share the same log stem — the job ID suffix differs, but the prefix encodes the experiment:

```
logs_v3/train_sex_binary_lstm_30s_lr1e-4_38373667.out
logs_v3/train_sex_binary_lstm_30s_lr1e-4_38373667.err
logs_v3/infer_sex_binary_lstm_test_38400002.out
logs_v3/infer_sex_binary_lstm_test_38400002.err
```

Stem format: `{step}_{task}_{head}_{context}_{lr_or_split}` (train includes context+lr; infer includes split). Resubmitted jobs get a new job ID suffix but the same descriptive stem, because the `_timeout_handler` passes `--output`/`--error` explicitly to the resubmission `sbatch` call.

---

## Cohort Consistency Filter

To ensure a valid context-length comparison, subjects whose full-night PSG recording is **shorter than the longest context window (240m = 2880 patches)** are excluded from **all** context lengths — not just from 240m. This prevents the cohort from silently shifting as context length grows, which would confound performance differences with population differences.

- **Config key:** `dataset.min_recording_patches: 2880` in `configs/phase0_v3_config.yaml`
- **Effect:** `ContextWindowDataset` drops subjects with `T < 2880` before building the window index, at every context length and every split.
- **Excluded subjects:** 20 subjects for Tier 1 tasks (≤ 0.21%), 15–19 for APPLES-only Tier 2 tasks (≤ 1.72%). Full list: [`docs/excluded_subjects_T_lt_2880.csv`](excluded_subjects_T_lt_2880.csv).
- **Full rationale and paper language:** [`docs/cohort_filter.md`](cohort_filter.md)

This filter also eliminates the OOM root cause for the Transformer head at 240m (zero-padded windows triggered O(N²) Math attention; see `cohort_filter.md` for the technical explanation).

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
/scratch/boshra95/psg/unified/results/phase0_v3/   # V3 (current); v2 uses phase0_v2/
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
│           ├── {split}_windows.parquet   # per-window predictions (non-overlapping T//N windows)
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

/home/boshra95/NSRR-tools/logs_v3/   # V3 (current); v2 uses logs_v2/
├── train_sex_binary_lstm_30s_lr1e-4_{job_id}.out    # SLURM stdout
├── train_sex_binary_lstm_30s_lr1e-4_{job_id}.err    # SLURM stderr
└── status/
    ├── train_sex_binary_lstm_30s_lr1e-4.jsonl   # event log for this context
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

Training K (windows sampled per subject per epoch) can be controlled via `configs/phase0_v3_config.yaml` (or the relevant config for your run):

```yaml
training:
  windows_strategy: "fixed"        # "fixed" | "token_budget"
  token_budget_minutes: 80         # used only when windows_strategy = "token_budget"
```

- **`fixed`** (default): K = `dataset.windows_per_subject` (currently 5) for all context lengths.
- **`token_budget`**: K = `max(1, floor(token_budget_minutes / ctx_minutes))` so that the total signal seen per subject is approximately constant across context lengths (e.g., 80min budget → K=160 for 30s context, K=2 for 40m context).

The logic is applied in `train_context_sweep.py` before dataset construction and prints the K value at training time. Switching from `"fixed"` to `"token_budget"` requires retraining (and changing the config). To run a token-budget ablation without overwriting the K=5 baseline, use a separate `run_tag` and a separate config file — see `docs/context_length_experiment_design.md` §12 for details.

---

## K Windows: Training vs Val vs Inference

There are three distinct K values in the pipeline. They use different sources and different windowing pools:

| K concept | Where set | Pool | Typical value |
|-----------|-----------|------|---------------|
| **K_train** (windows/epoch) | `dataset.windows_per_subject` in config (or token_budget) | **Overlapping** — any start in [0, T−N] | 5 |
| **K_val** (training-time val evaluation) | same `windows_per_subject` config | **Non-overlapping** stride-N positions | min(5, T//N) |
| **K_infer** (inference, all windows) | `--all-windows` flag → K_max=99,999 | **Non-overlapping** stride-N positions | T//N (all) |
| **K_analysis** (post-hoc sweep) | `analyze_windows.py --k-values` | Subsampled from K_infer parquet | 1,5,10,20,50,all |

Key points:
- **K_val during training is often < K_max.** For very long contexts (e.g. 240m with T//N = 2), val evaluation uses only 2 windows per subject — the early-stopping AUROC signal is noisier than for short contexts. This is inherent to long-context training on typical recording lengths.
- **K_infer recovers all T//N windows** because `infer_subject_windows.py` overrides `windows_per_subject` to 99,999, then the val/test branch of the dataset returns all non-overlapping positions.
- **K_analysis is completely separate** — it subsamples from the already-saved parquet rows in `analyze_windows.py`. No model or GPU needed.
- K_train and K_val share the same config value (`windows_per_subject`) but use different windowing pools (overlapping vs non-overlapping). This is by design: training benefits from diverse window positions, while val evaluation needs deterministic, non-redundant coverage.

---

## Batch Size Protocol

**Single protocol — batch size 32, accum_steps 1, identical at all context lengths.**

This was confirmed after fixing two issues that previously caused CUDA OOM at 240m on the Transformer head:

1. **Cohort filter** (`dataset.min_recording_patches=2880`): removes subjects shorter than 240m from all splits and context lengths, ensuring all padding masks are always all-False. See `docs/cohort_filter.md`.
2. **Mask fix** (`TransformerHead.forward()`): passes `src_key_padding_mask=None` when the mask is all-False, so PyTorch selects Flash attention (O(N) memory). The previous code passed a float tensor even when all-zeros, which forced O(N²) Math attention — trying to allocate ~42 GB at N=2881, batch=168 on a 9.75 GiB H100 MIG slice.

With both fixes, batch=32 fits at every context length. Training logs now confirm:
```
[Attn] SDPA backends — flash=True mem_eff=True math=True  |  mask=None expected=True
[Attn] Flash (mask=None, O(N) memory) | dtype=float32 | mode=train | N=2880 | any_padding=False
```

**Paper claim:** "All models were trained with batch size 32, identical across all context lengths."

**How it's controlled:** `gradient_accumulation.context_micro_batch` in `experiments/v2_registry.yaml` is set to 32 for every context, with `effective_batch=32`. `gen_commands.py` computes `accum_steps = 32/32 = 1` for all contexts automatically.

### Adjusting if memory constraints arise on a different GPU/head

Lower the relevant `context_micro_batch` entry; the gradient accumulation infrastructure handles the rest automatically:

```yaml
# Example: if 240m requires micro_batch=8 to fit in memory:
gradient_accumulation:
  effective_batch: 32
  context_micro_batch:
    "240m": 8   # gen_commands.py computes accum_steps = 32/8 = 4 automatically
```

`gen_commands.py` will set `BATCH_SIZE=8 ACCUM_STEPS=4` for that context. The effective gradient update is mathematically identical to batch=32 with accum=1.

**Paper wording if accumulation is needed:** "All models were trained with effective batch size 32, achieved via gradient accumulation at context lengths where GPU memory required a smaller micro-batch."

### About find_batch_size.py

`scripts/find_batch_size.py` (run via `gen_commands.py probe-batch <exp_id>`) probes the maximum batch size that fits on the GPU for each context length. It is available but not needed for the standard batch=32 protocol. It may be useful to determine GPU headroom, or if a new task/head/GPU combination requires revisiting memory limits.

The probe tests both train mode (backward pass) and eval mode (forward-only), since PyTorch selects different attention kernel paths in each. With `min_recording_patches=2880` and the mask fix in place, both modes use Flash attention and the probe results are reliable. Starting from `--starting-batch-size 256`, the probe for 6 context lengths completes in under 10 minutes.

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

---

## Extended Analysis Features

These features were added to support the deeper analyses described in `docs/ANALYSIS_IDEAS.md` — analyses beyond the core H1–H4 hypotheses that strengthen the paper with uncertainty quantification, overfitting characterisation, and neural scaling-law results.

### Overfitting phase (`overfit_epochs`)

**Config (`configs/phase0_v3_config.yaml`, under `training:`):**
```yaml
overfit_epochs: 0   # 0 = disabled (default); set e.g. to 20 to enable
```

**What it does:** After early stopping fires (val_auroc stops improving for `patience` epochs), if `overfit_epochs > 0`, training continues for that many additional epochs without updating `best_model.pt`. These extra epochs are tagged `is_overfit_epoch: True` in `training_curves.csv` and in the collected `training.csv`.

**Why it was added:** Normal early-stopped training only exposes the left and flat portions of the U-shaped val_loss curve. To study overfitting dynamics and fit neural scaling laws (see ANALYSIS_IDEAS §7 and §8) you need the right arm — the regime where val_loss rises while train_loss keeps falling. Concretely:

- **U-shape / overfitting curves:** Plot `train_loss` and `val_loss` vs epoch for all rows (including overfit rows). The width of the gap at the right characterises how well model capacity matches the dataset for each context length and head.
- **Scaling laws:** Compute cumulative FLOPs at each epoch (from `seq_len × steps_per_epoch × FLOPs_per_token × epoch_number`) and plot test AUROC vs FLOPs. Running past early stopping gives additional (compute, performance) datapoints on the overfit side.

**SLURM safety:** `resume.pt` is updated with `in_overfit_phase: True` and `overfit_start_epoch` before the main-loop break, so if the job times out mid-overfit-phase the resumed job picks up from the correct epoch.

**Using the data:** In `training.csv`, filter `is_overfit_epoch == False` for paper performance tables; include all rows (both normal and overfit) for learning-curve and U-shape plots.

---

### Snapshot checkpoints (`save_snapshots`, `snapshot_interval`)

**Config:**
```yaml
save_snapshots: false       # true to enable (disabled by default to save disk)
snapshot_interval: 5        # save a snapshot every N epochs (only when save_snapshots: true)
```

**What it does:** When enabled, saves `context_{L}/snapshots/epoch_{NNNN}.pt` every N epochs. These are model state dicts only (smaller than full resume checkpoints). The existing `best_model.pt` and `resume.pt` are unaffected.

**Why it was added:** Scaling-law analysis (ANALYSIS_IDEAS §8) requires measuring test AUROC at multiple training compute budgets — not just at the best epoch. Snapshots enable running inference at intermediate epochs (e.g., at 5, 10, 15, 20 epochs) to produce a (FLOPs, AUROC) curve and fit a power-law. Combined with `seq_len`, `steps_per_epoch`, and `n_trainable_params` now recorded in `metrics.json`, you can compute FLOPs analytically post-hoc without any additional training.

Default is off because snapshots are rarely needed and add disk usage (~model_size × n_snapshots per context). Enable only for scaling-law experiments.

---

### Bootstrap confidence intervals (`bootstrap_samples`)

**Config (`analysis:` section):**
```yaml
analysis:
  bootstrap_samples: 0   # 0 = disabled; 1000 recommended for paper tables
```

**What it does:** When > 0, `analyze_windows.py` performs subject-level bootstrap resampling N times after computing point estimates. The 2.5th and 97.5th percentiles form 95% CIs. Four new columns appear in `window_analysis_{split}.csv` (and in the collected `analysis.csv`):
- `mean_prob_auroc_ci_lo`, `mean_prob_auroc_ci_hi`
- `mean_prob_bal_acc_ci_lo`, `mean_prob_bal_acc_ci_hi`

**Why subject-level resampling:** Subjects are the independent unit, not windows. Resampling at the window level would underestimate variance because windows within a subject are correlated.

**Why it was added:** Paper tables and saturation curves need uncertainty quantification (ANALYSIS_IDEAS §5). CIs are especially important when comparing across context lengths — a small AUROC difference between L=80m and L=120m may not be meaningful without CIs.

**gen_commands.py integration:** The `analyze` subcommand reads `analysis.bootstrap_samples` from the config yaml automatically. If > 0, `--bootstrap N` is appended to the generated command — no manual editing required.

---

### New columns in `training.csv` and `analysis.csv`

`collect_results_v2.py` now writes additional columns to support the new analyses:

**`training.csv` — new columns:**

| Column | Source | Purpose |
|--------|--------|---------|
| `is_overfit_epoch` | `training_curves.csv` | Flag; include these rows in U-shape plots, exclude from paper tables |
| `n_overfit_epochs` | `metrics.json` | How many overfit epochs ran (0 if disabled) |
| `batch_size` | `metrics.json` | Needed to compute FLOPs per step |
| `seq_len` | `metrics.json` | Sequence length seen by the head (= N patches in context window) |
| `steps_per_epoch` | `metrics.json` | Number of gradient steps per epoch |
| `windows_per_subject_train` | `metrics.json` | K_train (for reproducibility and fairness docs) |
| `n_trainable_params` | `metrics.json` | Model-size axis for scaling-law plots |
| `input_dim`, `hidden_dim` | `metrics.json` | Needed for analytical FLOPs computation by head type |
| `save_snapshots`, `snapshot_interval` | `metrics.json` | Know which runs have snapshots available for snapshot-based AUROC curves |

FLOPs per step can be computed analytically post-hoc: for LSTM `∝ seq_len × hidden_dim²`; for Transformer `∝ seq_len² × hidden_dim`. Total training FLOPs to epoch E = `steps_per_epoch × FLOPs_per_step × E`.

**`analysis.csv` — new columns:**

| Column | Source | Purpose |
|--------|--------|---------|
| `mean_prob_auroc_ci_lo/hi` | `window_analysis_*.csv` | 95% CI bounds for error bars on saturation and K-sweep plots |
| `mean_prob_bal_acc_ci_lo/hi` | `window_analysis_*.csv` | Same for balanced accuracy |

All CI columns are `NaN` when `bootstrap_samples: 0` — backward compatible with existing plotting code.

**Note on best-epoch detection:** `collect_training()` now filters out `is_overfit_epoch=True` rows before calling `idxmax()` to find the best epoch. This means `is_best_epoch=True` always points to the early-stopping optimum, even when overfit rows are present in the same CSV.

---

### Extended analysis plot scripts ✅ Implemented

All scripts described in `docs/ANALYSIS_IDEAS.md` are now implemented. Use `gen_commands.py` to generate the exact commands — you never need to remember the flags.

#### collect (prerequisite for §1 and §6)

**Script:** `scripts/collect_results_v2.py`  
**gen_commands.py:** `python scripts/gen_commands.py collect [<exp_id> ...]`

Gathers results from all or selected experiments into:
- `{results_dir}/collected/training.csv` — per-epoch training curves including overfit rows
- `{results_dir}/collected/analysis.csv` — per-(task, head, context, k) metrics with optional bootstrap CI columns

Must be run before `scaling-laws` and `task-comparison`.

---

#### §1 — Overfitting curves and scaling laws

**Script:** `scripts/plot_scaling_laws.py`  
**gen_commands.py:** `python scripts/gen_commands.py scaling-laws sex_binary --heads lstm transformer mean_pool`

Reads `{collected_dir}/training.csv`. Produces three plots:

| Plot | Description |
|------|-------------|
| `1A` (U-shape) | train_loss + val_loss vs epoch; overfit rows shown dotted; fill_between for generalisation gap |
| `1B` (scaling law) | best val AUROC vs estimated training FLOPs (log-log); power-law fit `y = a - A × FLOPs^(-b)` via scipy |
| `1C` (optimal epoch) | bar chart of early-stop epoch per (head, context length) |

FLOPs computation: analytical from `seq_len × steps_per_epoch × FLOPs_per_step` where FLOPs formulas are:
- LSTM: `3 × seq_len × 4 × hidden_dim × (input_dim + hidden_dim)`
- Transformer: `3 × seq_len × (seq_len × hidden_dim + 4 × hidden_dim²)`
- MeanPool: `3 × seq_len × input_dim`

**Output:** `{results_dir}/figures/scaling_laws/{task}_{head}_{1A|1B|1C}.{png,pdf}`

---

#### §2 — Model calibration and reliability

**Script:** `scripts/plot_calibration.py`  
**gen_commands.py:** `python scripts/gen_commands.py calibration sex_binary_lstm [--heads lstm transformer mean_pool]`

Reads per-window parquets. Produces three plots:

| Plot | Description |
|------|-------------|
| `2A` (reliability diagrams) | 10-bin calibration curves for 3 representative contexts; ECE annotated |
| `2B` (ECE vs context) | Log-x saturation-style curve, one line per head; use `--heads` for multi-head |
| `2C` (ECE vs K) | ECE as a function of K (windows aggregated), one line per context |

**Output:** `{results_dir}/figures/{task}_{head}/calibration_2A_reliability.{png,pdf}` etc.

---

#### §4 — Window position and temporal structure

**Script:** `scripts/plot_window_position.py`  
**gen_commands.py:** `python scripts/gen_commands.py window-position sex_binary_lstm`

Reads per-window parquets. Normalises `window_idx` to [0, 1] within each subject (0=night start, 1=night end), bins into 20 equal bins.

| Plot | Description |
|------|-------------|
| `4A` (position profiles) | Mean prob_class1 ± 1 SD vs normalised position; separate panels for positive and negative subjects |
| `4B` (variance vs position) | std(prob_class1) at each position bin, one line per context |

**Output:** `{results_dir}/figures/{task}_{head}/window_position_4A_profiles.{png,pdf}` etc.

---

#### §5 — Per-subject consistency and prediction stability

**Script:** `scripts/plot_subject_consistency.py`  
**gen_commands.py:** `python scripts/gen_commands.py subject-consistency sex_binary_lstm`

Reads per-window parquets.

| Plot | Description |
|------|-------------|
| `5A` (variance distribution) | Violin plots of within-subject std(prob_class1) for correct vs incorrect subjects at 3 representative contexts |
| `5B` (variance vs K) | std of per-subject K-window mean across subjects vs K; confirms aggregation stabilises predictions |
| `5C` (hard subjects) | Histogram: x = number of context lengths at which subject is correctly classified; subjects at x=0 are never correctly classified |

**Output:** `{results_dir}/figures/{task}_{head}/subject_consistency_5A_variance.{png,pdf}` etc.

---

#### §6 — Task × context sensitivity matrix

**Script:** `scripts/plot_task_comparison.py`  
**gen_commands.py:** `python scripts/gen_commands.py task-comparison --head lstm [--tasks sex_binary bmi_binary ...]`

Reads `{collected_dir}/analysis.csv`. **Prerequisite: run `collect` first.**

| Plot | Description |
|------|-------------|
| `6A` (sensitivity scatter) | Each task = 1 point; x = baseline difficulty (1 − AUROC@30s); y = context sensitivity (ΔAUROC 30s → best); reference lines at medians |
| `6B` (AUROC bars) | Grouped bars per context per task; tasks sorted by context sensitivity ascending |
| `6C` (L* per task) | Dot chart on log-x axis; L* = smallest context within 0.5% of best AUROC |

**Output:** `{results_dir}/figures/task_comparison_6A_scatter.{png,pdf}` etc.

---

#### §7 — Cohort-stratified saturation

**Script:** `scripts/plot_cohort_saturation.py`  
**gen_commands.py:** `python scripts/gen_commands.py cohort-saturation sex_binary_lstm [--datasets apples shhs mros]`

Reads per-window parquets, filters by `dataset` column.

| Plot | Description |
|------|-------------|
| `7A` (per-cohort saturation) | One line per dataset on log-x context axis; N annotated on each point |
| `7B` (per-cohort N) | Grouped bars showing subject count per (dataset, context_length) |

**Output:** `{results_dir}/figures/{task}_{head}/cohort_saturation_7A.{png,pdf}` etc.

---

#### §8 — Precision-recall analysis

**Script:** `scripts/plot_precision_recall.py`  
**gen_commands.py:** `python scripts/gen_commands.py precision-recall sex_binary_lstm [--heads lstm transformer mean_pool]`

Reads per-window parquets.

| Plot | Description |
|------|-------------|
| `8A` (PR curves) | One PR curve per context at K=all; AP annotated in legend; chance line at prevalence |
| `8B` (AUC-PR vs context) | Log-x axis, one line per head; use `--heads` for multi-head |
| `8C` (vote sweep) | Majority-vote threshold sweep: t=1..K, t windows must vote positive; one curve per context showing PR tradeoff via vote threshold |

**Output:** `{results_dir}/figures/{task}_{head}/pr_8A_curves.{png,pdf}` etc.

---

#### §9 — Per-subject K* (minimum windows to correct classification)

**Script:** `scripts/plot_subject_kstar.py`  
**gen_commands.py:** `python scripts/gen_commands.py subject-kstar sex_binary_lstm [--kmax 30] [--reps 20]`

Reads per-window parquets. For each subject, estimates K* = minimum k such that a random subset of k windows produces a correct prediction (mean prob_class1 > 0.5 matches true_label). If never correct at any k ≤ kmax: K* = ∞.

| Plot | Description |
|------|-------------|
| `9A` (K* histogram) | Up to 4 representative contexts side-by-side; "never correct" fraction annotated |
| `9B` (coverage curves) | Fraction of subjects correctly classified using ≤K windows vs K; one line per context |

**Note:** Runtime scales as O(n_subjects × kmax × reps). Start with `--kmax 15 --reps 10` to validate. Default is kmax=30, reps=20.

**Output:** `{results_dir}/figures/{task}_{head}/kstar_9A_histogram.{png,pdf}` etc.

---

#### Saturation curves with CI bands (updated)

`scripts/plot_saturation.py` now accepts `--collected-dir` to overlay bootstrap 95% CI bands:

```bash
python scripts/gen_commands.py saturation sex_binary \
    --heads lstm transformer mean_pool \
    --collected-dir results/collected
```

CI bands are drawn as shaded regions (alpha=0.15) around each line when `mean_prob_auroc_ci_lo/hi` columns are present in `analysis.csv`. The y-axis label notes "shading = 95% bootstrap CI".
