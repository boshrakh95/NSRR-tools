# Experiment Execution Guide — Phase 0 / V3 Protocol

This document is the definitive reference for running training, inference, and analysis experiments for Phase 0 V2 task definitions.

> **V3 protocol (current):** Results are written to `phase0_v3/`, logs to `logs_v3/`. Training uses overlapping-window fixed-K sampling (K=5 per subject at all context lengths); **val/test during training also use K=5 overlapping windows** (evenly spaced, deterministic) so the early-stopping signal is equally reliable at all context lengths; **inference uses non-overlapping stride-N windows** (T//N positions) for systematic, non-redundant night coverage. Context-specific LR at 120m/240m. Use `configs/phase0_v3_config.yaml` and `experiments/v2_registry.yaml` (already updated). Do NOT mix v2 and v3 results in the same comparison figure. See [TRAINING_PROTOCOL_FIXES.md](TRAINING_PROTOCOL_FIXES.md) for the rationale behind each change.
>
> **Note on paths:** All examples in this document use V3 paths (`phase0_v3/`, `logs_v3/`). The archived V2 config (`phase0_v2_config.yaml`) is shown only in the Config Files table.

---

## Table of Contents

1. [Overview](#overview)
2. [**End-to-End Run Playbooks**](#end-to-end-run-playbooks) ← start here
   - [Run identity quick-reference](#run-identity-quick-reference)
   - [Fast-channel run (v3 baseline)](#fast-channel-run-v3-baseline)
   - [Full-channel run (channel expansion)](#full-channel-run-channel-expansion)
   - [Modality ablation run (channel importance)](#modality-ablation-run-channel-importance)
   - [**Figure Generation — All Paper Figures**](#figure-generation--all-paper-figures) ← figures only (analyze already done)
3. [Pipeline Steps](#pipeline-steps)
4. [Config Files](#config-files)
5. [Experiment Registry and Command Generator](#experiment-registry-and-command-generator)
6. [Submitting Jobs](#submitting-jobs)
7. [Checkpoint Resume and Auto-Requeue](#checkpoint-resume-and-auto-requeue)
8. [Job Run History and Tracking](#job-run-history-and-tracking)
9. [Experiment Plan](#experiment-plan)
10. [Results Directory Structure](#results-directory-structure)
11. [Expected Runtimes](#expected-runtimes)
12. [Configurable Training K Strategy](#configurable-training-k-strategy)
13. [K Windows: Training vs Val vs Inference](#k-windows-training-vs-val-vs-inference)
14. [Batch Size Protocol](#batch-size-protocol)
15. [Model Architecture Reference](#model-architecture-reference)
16. [Adding New Experiments](#adding-new-experiments)
17. [Regression Tasks (Deferred)](#regression-tasks-deferred)
18. [Notes on Specific Tasks](#notes-on-specific-tasks)
19. [Results Collection](#results-collection)

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

## End-to-End Run Playbooks

Two parallel experiment sets live on disk simultaneously and never overwrite each other.

### Run identity quick-reference

| | Fast-channel (v3 baseline) | Full-channel (channel expansion) | Modality ablation |
|---|---|---|---|
| **Channels/subject** | 7–8 (BAS=3, RESP=1, EKG=1, EMG=1–2) | Up to 23 (BAS≤10, RESP≤7, EKG≤2, EMG≤4) | Same as fast-channel; selected groups zeroed in embedding |
| **Preprocessing config** | `configs/preprocessing_params.yaml` | `configs/preprocessing_params_full.yaml` | **None — reuses fast-channel embeddings** |
| **HDF5 root** | `/scratch/boshra95/psg/` | `/scratch/boshra95/psg_full/` | **Reused** `/scratch/boshra95/psg/` |
| **Embeddings dir** | `/scratch/boshra95/psg/unified/embeddings/sleepfm_5sec/` | `/scratch/boshra95/psg_full/unified/embeddings/sleepfm_5sec/` | **Reused** `/scratch/boshra95/psg/unified/embeddings/sleepfm_5sec/` |
| **Embedding config** | `configs/phase0_v3_config.yaml` | `configs/phase0_v3_full_config.yaml` | `configs/phase0_v3_abl_config.yaml` |
| **Train config** | `configs/phase0_v3_config.yaml` (hidden=128, layers=1) | `configs/phase0_v3_full_config.yaml` | `configs/phase0_v3_abl_config.yaml` (hidden=128, layers=1) |
| **Sleep staging config** | `configs/phase0_v3_staging_config.yaml` (hidden=256, layers=2, val_kappa) | `configs/phase0_v3_full_staging_config.yaml` | N/A — no sleep staging in ablation |
| **Registry** | `experiments/v2_registry.yaml` | `experiments/v2_full_registry.yaml` | `experiments/v2_ablation_registry.yaml` |
| **Results root** | `/scratch/boshra95/psg/unified/results/phase0_v3/` | `/scratch/boshra95/psg_full/unified/results/phase0_v3_full/` | `/scratch/boshra95/psg/unified/results/phase0_v3_abl/` |
| **Inference root** | `/scratch/boshra95/psg/unified/results/phase0_v3/inference/` | `/scratch/boshra95/psg_full/unified/results/phase0_v3_full/inference/` | `/scratch/boshra95/psg/unified/results/phase0_v3_abl/inference/` |
| **Training logs** | `logs_v3/` | `logs_v3_full/` | `logs_v3_abl/` |
| **Preprocessing/embedding logs** | `logs_v3/` (historical) | `logs_v3_full/` | **Not applicable** |
| **Status** | **DONE** — all steps complete | **IN PROGRESS** — preprocessing/embedding running | **PENDING** — ready to train |

---

### Fast-channel run (v3 baseline)

> All steps for the fast-channel run are **already complete**. This section documents
> the commands that were used, for reproducibility and reference.

**Working directory for all commands:** `cd /home/boshra95/NSRR-tools`

#### Step 0 — Preprocessing (EDF → HDF5)

Outputs to `/scratch/boshra95/psg/{dataset}/derived/hdf5_signals/`.
Logs to `logs_v3/` (historical; the full-channel run uses `logs_v3_full/` for all steps).

```bash
CFG=configs/preprocessing_params.yaml

# Small/medium datasets (one job each, ~26h wall time)
DATASET=stages CONFIG_PATH=$CFG sbatch jobs/preprocess_signals_parallel.sh
DATASET=apples CONFIG_PATH=$CFG sbatch jobs/preprocess_signals_parallel.sh
DATASET=mros   CONFIG_PATH=$CFG sbatch jobs/preprocess_signals_parallel.sh

# SHHS (8444 subjects — split into 6 parallel array jobs, 8h each)
sbatch jobs/preprocess_signals_array.sh shhs    0  1500 --config $CFG
sbatch jobs/preprocess_signals_array.sh shhs 1500  3000 --config $CFG
sbatch jobs/preprocess_signals_array.sh shhs 3000  4500 --config $CFG
sbatch jobs/preprocess_signals_array.sh shhs 4500  6000 --config $CFG
sbatch jobs/preprocess_signals_array.sh shhs 6000  7500 --config $CFG
sbatch jobs/preprocess_signals_array.sh shhs 7500  9000 --config $CFG
```

**Output:** `.h5` files per subject, 7–8 channels, 128 Hz.
**Verify:** `ls /scratch/boshra95/psg/shhs/derived/hdf5_signals/ | wc -l`  → expected ~8444.

#### Step 1 — Embedding Extraction (HDF5 → .npy)

Outputs to `/scratch/boshra95/psg/unified/embeddings/sleepfm_5sec/{dataset}/{subject}.npy`.
Each `.npy` has shape `[T, 4, 128]`.

```bash
# 4 parallel GPU jobs covering all ~15,000 subjects
# Subject global index order (from phase0_v3_config.yaml datasets list):
#   apples(0–1103), shhs(1104–9547), mros(9548–13480), stages(13481–14993)

sbatch --export=ALL,START=0,END=2500    jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,START=2500,END=5000 jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,START=5000,END=7500 jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,START=7500,END=9600 jobs/extract_embeddings_gpu.sh
# (mros+stages submitted after their preprocessing completed)
sbatch --export=ALL,START=9600,END=12500  jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,START=12500,END=15100 jobs/extract_embeddings_gpu.sh
```

Note: `jobs/extract_embeddings_gpu.sh` defaults to `CONFIG=configs/phase0_v3_config.yaml`.
Logs go to `logs_v3/embeddings_*.out`.

**Verify:** `find /scratch/boshra95/psg/unified/embeddings/sleepfm_5sec -name '*.npy' | wc -l`  → expected ~14,992.

#### Step 2 — Training: seq2label tasks

Registry: `experiments/v2_registry.yaml` (no `--registry` flag needed — it's the default).
Config: `configs/phase0_v3_config.yaml` (hidden=128, layers=1, i.e., ~658K LSTM params).
Results: `/scratch/boshra95/psg/unified/results/phase0_v3/{task}_{head}/context_{L}/`.
Logs: `logs_v3/train_{task}_{head}_{context}_lr{lr}_{jobid}.out`.

```bash
# Tier 1 — all three heads (run all contexts in parallel per experiment)
python scripts/gen_commands.py train sex_binary_lstm              | bash
python scripts/gen_commands.py train sex_binary_transformer       | bash
python scripts/gen_commands.py train sex_binary_mean_pool         | bash

python scripts/gen_commands.py train sleep_efficiency_binary_lstm        | bash
python scripts/gen_commands.py train sleep_efficiency_binary_transformer | bash
python scripts/gen_commands.py train sleep_efficiency_binary_mean_pool   | bash

python scripts/gen_commands.py train bmi_binary_lstm        | bash
python scripts/gen_commands.py train bmi_binary_transformer | bash
python scripts/gen_commands.py train bmi_binary_mean_pool   | bash

python scripts/gen_commands.py train age_class_lstm         | bash
python scripts/gen_commands.py train age_class_transformer  | bash
python scripts/gen_commands.py train age_class_mean_pool    | bash

python scripts/gen_commands.py train apnea_binary_lstm        | bash
python scripts/gen_commands.py train apnea_binary_transformer | bash
python scripts/gen_commands.py train apnea_binary_mean_pool   | bash

# Tier 2 — lstm only
python scripts/gen_commands.py train psqi_binary_lstm               | bash
python scripts/gen_commands.py train depression_extreme_binary_lstm | bash
python scripts/gen_commands.py train osa_binary_apples_postqc_lstm  | bash
python scripts/gen_commands.py train osa_severity_apples_lstm       | bash
python scripts/gen_commands.py train cvd_binary_lstm                | bash
python scripts/gen_commands.py train cvd_binary_transformer         | bash
python scripts/gen_commands.py train sleepiness_binary_lstm         | bash
python scripts/gen_commands.py train sleepiness_binary_transformer  | bash
```

**Check status:** `python scripts/gen_commands.py status`
**Check job history:** `python scripts/gen_commands.py runs sex_binary_lstm`

#### Step 3 — Training: sleep staging

Sleep staging uses `task_type: seq2seq` with a **centered** context window and the 256/2
head. Registry: `v2_registry.yaml`. Each sleep staging entry has an explicit
`config: configs/phase0_v3_staging_config.yaml` (hidden=256, layers=2, **`val_kappa` monitor**,
epochs=60). `gen_commands.py` picks `exp["config"]` before `registry["config"]` automatically.

**Why val_kappa for sleep staging:** val_auroc for 5-class OvR macro is slow to plateau and
caused 120m to hit the 40-epoch ceiling without converging. val_kappa directly optimises the
primary reported metric. See `docs/sleep_staging_design.md §10.3` for the full analysis.

Results: `/scratch/boshra95/psg/unified/results/phase0_v3/sleep_staging_{head}/context_{L}/`.
Primary metric: Cohen's κ (logged as `val_kappa` and `test_kappa` in `metrics.json`).

```bash
# Primary run: shhs + mros + apples (no STAGES — STAGES dominates data and hurts kappa)
python scripts/gen_commands.py train sleep_staging_lstm        | bash
python scripts/gen_commands.py train sleep_staging_transformer | bash
python scripts/gen_commands.py train sleep_staging_mean_pool   | bash

# Comparison run: includes STAGES (for ablation only)
python scripts/gen_commands.py train sleep_staging_lstm_with_stages        | bash
python scripts/gen_commands.py train sleep_staging_transformer_with_stages | bash
```

#### Step 4 — Inference (all trained contexts → per-window parquets)

One GPU job per experiment auto-discovers all trained contexts and skips any already inferred.

Results: `/scratch/boshra95/psg/unified/results/phase0_v3/inference/{task}_{head}/context_{L}/test_windows.parquet`.

```bash
# Run after training is complete for each experiment
python scripts/gen_commands.py infer sex_binary_lstm | bash
python scripts/gen_commands.py infer sex_binary_transformer | bash
python scripts/gen_commands.py infer sex_binary_mean_pool | bash
# ... repeat for each experiment

# For val split (needed for threshold tuning):
python scripts/gen_commands.py infer sex_binary_lstm --split val | bash

# Sleep staging inference (same command):
python scripts/gen_commands.py infer sleep_staging_lstm | bash
```

#### Step 5 — Analysis and Plotting (local, no GPU)

> **Registry rule:** No `--registry` flag = fast-channel; all outputs go to `scratch/psg/…`.
> Passing `--registry experiments/v2_full_registry.yaml` redirects to `scratch/psg_full/…` — only
> do that in the full-channel section below.

**Multi-task pipeline (recommended):** runs all 13 steps (window sweep with dense K, collect,
heatmap, iso-plots, saturation, scaling-laws, calibration, window-position,
subject-consistency, cohort-saturation, precision-recall, subject-kstar, task-comparison).

Step 1 — Full pipeline, no bootstrap (~10–20 min, gets all plots without CI bands):
```bash
source /home/boshra95/sleepfm_env/bin/activate
bash scripts/run_analysis.sh \
  age_class_lstm age_class_transformer \
  apnea_binary_lstm apnea_binary_transformer \
  bmi_binary_lstm bmi_binary_transformer \
  cvd_binary_lstm cvd_binary_transformer \
  depression_extreme_binary_lstm \
  osa_binary_apples_postqc_lstm \
  psqi_binary_lstm \
  sex_binary_lstm sex_binary_transformer \
  sleep_efficiency_binary_lstm sleep_efficiency_binary_transformer \
  sleepiness_binary_lstm sleepiness_binary_transformer \
  --heads lstm transformer \
  2>&1 | tee analysis_run.log
# No --registry → fast channel (scratch/psg/unified/results/phase0_v3/)
```

Step 2 — Add bootstrap CIs to window-sweep plots only (~2–3 hrs, run in tmux).
Reruns only the analyze step; all other plots (Steps 2–13) are untouched:
```bash
bash scripts/run_analysis.sh \
  age_class_lstm age_class_transformer \
  apnea_binary_lstm apnea_binary_transformer \
  bmi_binary_lstm bmi_binary_transformer \
  cvd_binary_lstm cvd_binary_transformer \
  depression_extreme_binary_lstm \
  osa_binary_apples_postqc_lstm \
  psqi_binary_lstm \
  sex_binary_lstm sex_binary_transformer \
  sleep_efficiency_binary_lstm sleep_efficiency_binary_transformer \
  sleepiness_binary_lstm sleepiness_binary_transformer \
  --heads lstm transformer \
  --bootstrap 1000 --analyze-only \
  2>&1 | tee bootstrap_run.log
```

Step 3 — After Step 2, re-collect so `results/collected/phase0_v3/analysis.csv` picks up the
new bootstrap CI columns (collect uses key-dedup by default, so `--force` is required):
```bash
python scripts/collect_results_v2.py --force
```

**Single experiment / a la carte:**
```bash
# Dense K sweep + plots
python scripts/gen_commands.py analyze sex_binary_lstm --k-dense --plot | bash

# Heatmap DataFrame (after dense K sweep)
python scripts/gen_commands.py build-heatmap sex_binary_lstm | bash

# Post-hoc threshold tuning (binary tasks only, after val inference)
python scripts/gen_commands.py threshold-tuning sex_binary_lstm | bash
```

Output: `inference/{task}_{head}/window_analysis_{split}.csv`, `window_analysis.md`,
`heatmap_df_{split}.csv`, `threshold_tuning.csv`.
Collected CSVs: `results/collected/phase0_v3/analysis.csv` and `training.csv` (in git repo).

#### Step 6 — Plotting (a la carte, local, no GPU)

`run_analysis.sh` Step 1 above runs all plots automatically. Use individual commands below to
re-run a specific plot type. For the full paper figure set, use `scripts/run_figures.sh` instead
(see [Figure Generation — All Paper Figures](#figure-generation--all-paper-figures) below).

```bash
# ── Per-experiment plots (run for each task × head combination) ──────────────

# Fig 2 + S-Fig 3: iso-compute plots (heatmap, metric_vs_k, metric_vs_total, pareto,
#   min_cost_frontier, marginal_gain) — double_tradeoff is blacklisted
python scripts/gen_commands.py iso-plots sex_binary_transformer | bash
python scripts/gen_commands.py iso-plots sex_binary_lstm | bash   # repeat for all 21 exps

# S-Fig 4a + 4b: calibration — 2A reliability diagram, 2B ECE vs context
#   2C (ECE vs K) is blacklisted from default --plots
python scripts/gen_commands.py calibration sex_binary_lstm | bash  # repeat per exp

# S-Fig 7: window position profiles (4A + 4B)
python scripts/gen_commands.py window-position sleep_efficiency_binary_lstm | bash
python scripts/gen_commands.py window-position sex_binary_lstm | bash

# S-Fig 6a + 6b: subject consistency (5A variance violins, 5C hard-subject bars)
#   5B (variance vs K) is blacklisted from default --plots; 5C redesigned (cumulative line)
python scripts/gen_commands.py subject-consistency sex_binary_transformer | bash  # repeat per exp

# S-Fig 5: per-cohort saturation (7A only; 7B is blacklisted)
python scripts/gen_commands.py cohort-saturation sex_binary_lstm | bash  # repeat per exp

# S-Fig 10: precision-recall (8A PR curves, 8B AUC-PR vs context; 8C vote sweep blacklisted)
python scripts/gen_commands.py precision-recall sex_binary_lstm | bash  # repeat per exp

# S-Fig 9: K* histograms (9A only; 9B is blacklisted)
python scripts/gen_commands.py subject-kstar sex_binary_transformer | bash  # repeat per exp

# ── Per-task plots ───────────────────────────────────────────────────────────

# Fig 1: saturation curves — AUROC vs context length, 3 heads per panel
python scripts/gen_commands.py saturation sex_binary \
    --heads lstm transformer mean_pool | bash   # repeat for all 7 tasks

# S-Fig 8: compute scaling law — 1B FLOPs vs AUROC scatter (1C blacklisted)
python scripts/gen_commands.py scaling-laws sex_binary \
    --heads lstm transformer mean_pool --plots 1B | bash   # repeat for all 7 tasks

# ── Multi-task plot ──────────────────────────────────────────────────────────

# Fig 3: task landscape — 6A scatter + 6C L* lollipop (6B bars blacklisted)
python scripts/gen_commands.py task-comparison \
    --tasks sex_binary apnea_binary sleep_efficiency_binary bmi_binary age_class \
            depression_extreme_binary osa_binary_apples_postqc \
    --head lstm | bash

# ── Cross-round figures (no gen_commands.py — direct script calls) ───────────

# Fig 4: modality ablation bar chart (v3_abl + v3 baseline + v3_full reference)
python scripts/plot_modality_bar.py

# S-Fig 2: fast vs full channel saturation overlay (v3 + v3_full, Transformer)
python scripts/plot_channel_comparison.py

# S-Fig 12 / Fig 5 (TBD): aggregate context-length scaling analysis
python scripts/plot_aggregate_scaling.py \
    --collected-dir results/collected/phase0_v3 \
    --results-dir /scratch/boshra95/psg/unified/results/phase0_v3
```

Output: `/scratch/boshra95/psg/unified/results/phase0_v3/figures/`.

---

### Full-channel run (channel expansion)

All commands use `--registry experiments/v2_full_registry.yaml`.
Working directory: `cd /home/boshra95/NSRR-tools`

#### Path summary

```
Preprocessing output  →  /scratch/boshra95/psg_full/{dataset}/derived/hdf5_signals/
Embeddings            →  /scratch/boshra95/psg_full/unified/embeddings/sleepfm_5sec/{dataset}/
Targets/annotations   →  /scratch/boshra95/psg/unified/targets_v2/  (SHARED — do not duplicate)
Seq2label results     →  /scratch/boshra95/psg_full/unified/results/phase0_v3_full/{task}_{head}/
Inference results     →  /scratch/boshra95/psg_full/unified/results/phase0_v3_full/inference/
Figures               →  /scratch/boshra95/psg_full/unified/results/phase0_v3_full/figures/
All logs              →  /home/boshra95/NSRR-tools/logs_v3_full/
```

#### Step 0 — Preprocessing (EDF → HDF5)

Outputs to `/scratch/boshra95/psg_full/{dataset}/derived/hdf5_signals/`.
Logs to `logs_v3_full/preprocess_*.out`.
Config `preprocessing_params_full.yaml` sets `strategy: "sleepfm_full"` (full channel set).

```bash
CFG=configs/preprocessing_params_full.yaml

# Small/medium datasets (one job each, up to 26h)
DATASET=stages CONFIG_PATH=$CFG sbatch jobs/preprocess_signals_parallel.sh
DATASET=apples CONFIG_PATH=$CFG sbatch jobs/preprocess_signals_parallel.sh
DATASET=mros   CONFIG_PATH=$CFG sbatch jobs/preprocess_signals_parallel.sh

# SHHS (8444 subjects — 6 parallel array jobs, 8h each)
sbatch jobs/preprocess_signals_array.sh shhs    0  1500 --config $CFG
sbatch jobs/preprocess_signals_array.sh shhs 1500  3000 --config $CFG
sbatch jobs/preprocess_signals_array.sh shhs 3000  4500 --config $CFG
sbatch jobs/preprocess_signals_array.sh shhs 4500  6000 --config $CFG
sbatch jobs/preprocess_signals_array.sh shhs 6000  7500 --config $CFG
sbatch jobs/preprocess_signals_array.sh shhs 7500  9000 --config $CFG
```

**All 9 jobs can run simultaneously.** Safe to re-submit any that fail — existing HDF5 files are
skipped (`--skip-existing` is the default).

**Verify when done:**
```bash
for ds in stages shhs apples mros; do
    n=$(ls /scratch/boshra95/psg_full/${ds}/derived/hdf5_signals/*.h5 2>/dev/null | wc -l)
    echo "$ds: $n (expected: stages≈1513  shhs≈8444  apples≈1104  mros≈3933)"
done

# Spot-check channel count (should be 14–21 per subject)
python3 - <<'EOF'
import h5py, glob
for f in sorted(glob.glob('/scratch/boshra95/psg_full/stages/derived/hdf5_signals/*.h5'))[:3]:
    with h5py.File(f) as h:
        print(f.split('/')[-1], '->', sorted(h.keys()))
EOF
```

#### Step 1 — Embedding Extraction (HDF5 → .npy)

Outputs to `/scratch/boshra95/psg_full/unified/embeddings/sleepfm_5sec/{dataset}/{subject}.npy`.
Each `.npy` shape: `[T, 4, 128]` (same format as fast-channel; content richer due to more channels).
Logs to `logs_v3_full/embeddings_*.out`.

```bash
# Subject global index order (from phase0_v3_full_config.yaml datasets list):
#   apples(0–1103), shhs(1104–9547), mros(9548–13480), stages(13481–14993)
#
# 6 parallel GPU jobs (~4h each on H100 MIG)
# Submit first 4 immediately (apples + shhs complete);
# submit last 2 after mros and stages preprocessing finishes.

NEW_CFG=configs/phase0_v3_full_config.yaml

sbatch --export=ALL,CONFIG=$NEW_CFG,START=0,END=2500       jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,CONFIG=$NEW_CFG,START=2500,END=5000    jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,CONFIG=$NEW_CFG,START=5000,END=7500    jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,CONFIG=$NEW_CFG,START=7500,END=9600    jobs/extract_embeddings_gpu.sh
# Submit below AFTER mros (9548–13480) and stages (13481–14993) preprocessing is done:
sbatch --export=ALL,CONFIG=$NEW_CFG,START=9600,END=12500   jobs/extract_embeddings_gpu.sh
sbatch --export=ALL,CONFIG=$NEW_CFG,START=12500,END=15100  jobs/extract_embeddings_gpu.sh
```

`--skip-existing` is enabled by default — safe to resubmit any range.

**Verify when done:**
```bash
find /scratch/boshra95/psg_full/unified/embeddings/sleepfm_5sec -name '*.npy' | wc -l
# Expected: ~14,992

python3 -c "
import numpy as np
a = np.load('/scratch/boshra95/psg_full/unified/embeddings/sleepfm_5sec/stages/STNF00032.npy', mmap_mode='r')
print('shape:', a.shape)   # [T, 4, 128] — T varies by recording length
"
```

#### Step 2 — Training: seq2label tasks

Config: `configs/phase0_v3_full_config.yaml` (hidden=128, layers=1 — matches fast-channel baseline).
Registry: `experiments/v2_full_registry.yaml`.
Results: `/scratch/boshra95/psg_full/unified/results/phase0_v3_full/{task}_{head}/context_{L}/`.
Logs: `logs_v3_full/train_{task}_{head}_{context}_lr{lr}_{jobid}.out`.

```bash
REG="--registry experiments/v2_full_registry.yaml"

# Tier 1 — all three heads
python scripts/gen_commands.py $REG train sex_binary_lstm              | bash
python scripts/gen_commands.py $REG train sex_binary_transformer       | bash
python scripts/gen_commands.py $REG train sex_binary_mean_pool         | bash

python scripts/gen_commands.py $REG train sleep_efficiency_binary_lstm        | bash
python scripts/gen_commands.py $REG train sleep_efficiency_binary_transformer | bash
python scripts/gen_commands.py $REG train sleep_efficiency_binary_mean_pool   | bash

python scripts/gen_commands.py $REG train bmi_binary_lstm        | bash
python scripts/gen_commands.py $REG train bmi_binary_transformer | bash
python scripts/gen_commands.py $REG train bmi_binary_mean_pool   | bash

python scripts/gen_commands.py $REG train age_class_lstm         | bash
python scripts/gen_commands.py $REG train age_class_transformer  | bash
python scripts/gen_commands.py $REG train age_class_mean_pool    | bash

python scripts/gen_commands.py $REG train apnea_binary_lstm        | bash
python scripts/gen_commands.py $REG train apnea_binary_transformer | bash
python scripts/gen_commands.py $REG train apnea_binary_mean_pool   | bash

# Tier 2 — lstm only
python scripts/gen_commands.py $REG train psqi_binary_lstm               | bash
python scripts/gen_commands.py $REG train depression_extreme_binary_lstm | bash
python scripts/gen_commands.py $REG train osa_binary_apples_postqc_lstm  | bash
python scripts/gen_commands.py $REG train osa_severity_apples_lstm       | bash
python scripts/gen_commands.py $REG train cvd_binary_lstm                | bash
python scripts/gen_commands.py $REG train cvd_binary_transformer         | bash
python scripts/gen_commands.py $REG train sleepiness_binary_lstm         | bash
python scripts/gen_commands.py $REG train sleepiness_binary_transformer  | bash
```

**Check status:** `python scripts/gen_commands.py $REG status`

#### Step 3 — Training: sleep staging

Sleep staging automatically uses `configs/phase0_v3_full_staging_config.yaml` (hidden=256,
layers=2, num_classes=5) because each sleep staging entry in `v2_full_registry.yaml` has an
explicit `config:` field. `gen_commands.py` picks `exp["config"]` before `registry["config"]`
— no special flag needed.

Results: `/scratch/boshra95/psg_full/unified/results/phase0_v3_full/sleep_staging_{head}/context_{L}/`.
Primary metric: Cohen's κ (field `test_kappa` in `metrics.json`).

```bash
REG="--registry experiments/v2_full_registry.yaml"

# Primary run: shhs + mros + apples (no STAGES)
python scripts/gen_commands.py $REG train sleep_staging_lstm        | bash
python scripts/gen_commands.py $REG train sleep_staging_transformer | bash
python scripts/gen_commands.py $REG train sleep_staging_mean_pool   | bash

# Comparison run: includes STAGES (for ablation — expected lower kappa)
python scripts/gen_commands.py $REG train sleep_staging_lstm_with_stages        | bash
python scripts/gen_commands.py $REG train sleep_staging_transformer_with_stages | bash
```

**Verify the correct config is being used** by inspecting the generated command:
```bash
python scripts/gen_commands.py $REG train sleep_staging_lstm --context 10m
# Should show: CONFIG=configs/phase0_v3_full_staging_config.yaml
```

#### Step 4 — Inference

Results: `/scratch/boshra95/psg_full/unified/results/phase0_v3_full/inference/{task}_{head}/context_{L}/test_windows.parquet`.

```bash
REG="--registry experiments/v2_full_registry.yaml"

python scripts/gen_commands.py $REG infer sex_binary_lstm        | bash
python scripts/gen_commands.py $REG infer sex_binary_transformer | bash
python scripts/gen_commands.py $REG infer sex_binary_mean_pool   | bash
# ... repeat for each experiment

# Sleep staging inference (same command — context auto-discovered)
python scripts/gen_commands.py $REG infer sleep_staging_lstm | bash

# Val split for threshold tuning
python scripts/gen_commands.py $REG infer sex_binary_lstm --split val | bash
```

#### Step 5 — Analysis and Plotting (local, no GPU)

**Multi-task pipeline (recommended):** runs all 13 steps. Always pass
`--registry experiments/v2_full_registry.yaml` — outputs go to `scratch/psg_full/…`.

Step 1 — Full pipeline, no bootstrap (~10–20 min, gets all plots without CI bands):
```bash
source /home/boshra95/sleepfm_env/bin/activate
bash scripts/run_analysis.sh \
  age_class_lstm age_class_transformer \
  apnea_binary_lstm apnea_binary_transformer \
  bmi_binary_lstm bmi_binary_transformer \
  cvd_binary_lstm cvd_binary_transformer \
  depression_extreme_binary_lstm \
  osa_binary_apples_postqc_lstm \
  psqi_binary_lstm \
  sex_binary_lstm sex_binary_transformer \
  sleep_efficiency_binary_lstm sleep_efficiency_binary_transformer \
  sleepiness_binary_lstm sleepiness_binary_transformer \
  --registry experiments/v2_full_registry.yaml \
  --heads lstm transformer \
  2>&1 | tee analysis_full_run.log
# --registry → full channel (scratch/psg_full/unified/results/phase0_v3_full/)
```

Step 2 — Add bootstrap CIs to window-sweep plots only (~2–3 hrs, run in tmux).
Reruns only the analyze step; all other plots (Steps 2–13) are untouched:
```bash
bash scripts/run_analysis.sh \
  age_class_lstm age_class_transformer \
  apnea_binary_lstm apnea_binary_transformer \
  bmi_binary_lstm bmi_binary_transformer \
  cvd_binary_lstm cvd_binary_transformer \
  depression_extreme_binary_lstm \
  osa_binary_apples_postqc_lstm \
  psqi_binary_lstm \
  sex_binary_lstm sex_binary_transformer \
  sleep_efficiency_binary_lstm sleep_efficiency_binary_transformer \
  sleepiness_binary_lstm sleepiness_binary_transformer \
  --registry experiments/v2_full_registry.yaml \
  --heads lstm transformer \
  --bootstrap 1000 --analyze-only \
  2>&1 | tee bootstrap_full_run.log
```

Step 3 — After Step 2, re-collect so `results/collected/phase0_v3_full/analysis.csv` picks up
the new bootstrap CI columns (`--force` bypasses key-dedup):
```bash
python scripts/collect_results_v2.py \
  --results-dir /scratch/boshra95/psg_full/unified/results/phase0_v3_full \
  --force
```

**Single experiment / a la carte:**
```bash
REG="--registry experiments/v2_full_registry.yaml"

# Dense K sweep + plots
python scripts/gen_commands.py $REG analyze sex_binary_lstm --k-dense --plot | bash

# Heatmap DataFrame (after dense K sweep)
python scripts/gen_commands.py $REG build-heatmap sex_binary_lstm | bash

# Post-hoc threshold tuning (binary tasks only, after val inference)
python scripts/gen_commands.py $REG threshold-tuning sex_binary_lstm | bash
```

Output: `inference/{task}_{head}/window_analysis_{split}.csv`, `heatmap_df_{split}.csv`,
`threshold_tuning.csv`.
Collected CSVs: `results/collected/phase0_v3_full/analysis.csv` and `training.csv` (in git repo).

#### Step 6 — Plotting (a la carte, local, no GPU)

`run_analysis.sh` Step 1 above runs all plots automatically. The v3_full run contributes one
paper figure: **S-Fig 2** (fast vs full channel overlay), generated by `plot_channel_comparison.py`
which reads both collected CSVs directly — no per-experiment steps are needed for that figure.
Use the commands below only if you need to regenerate the full v3_full per-experiment plot set.

```bash
REG="--registry experiments/v2_full_registry.yaml"

# Iso-compute plots (runs for v3_full inference outputs)
python scripts/gen_commands.py $REG iso-plots sex_binary_lstm | bash

# Saturation curves (v3_full; useful for debugging full-channel L* estimates)
python scripts/gen_commands.py $REG saturation sex_binary \
    --heads lstm transformer mean_pool | bash

# Sleep staging saturation (kappa vs context)
python scripts/gen_commands.py $REG saturation sleep_staging \
    --heads lstm transformer | bash

# S-Fig 2 (the only v3_full paper figure — no registry flag needed):
python scripts/plot_channel_comparison.py
# Output: /scratch/boshra95/psg_full/unified/results/phase0_v3_full/figures/
#          phase0_v3_full/channel_comparison.{png,pdf}
```

Output: `/scratch/boshra95/psg_full/unified/results/phase0_v3_full/figures/`.

#### Comparing fast-channel vs full-channel results

**Quick comparison table for all seq2label tasks (paper-ready):**

`scripts/summarize_results.py` reads `window_analysis_test.csv` directly from both inference
directories and produces a best-AUROC-per-task comparison table. It never depends on collect
having been run — always reflects the latest `analyze` outputs.

```bash
source /home/boshra95/sleepfm_env/bin/activate

# Console table (k=5 primary, k=all ceiling)
python scripts/summarize_results.py --compare

# Save to CSV
python scripts/summarize_results.py --compare --out results_summary_fast_vs_full.csv

# LaTeX table
python scripts/summarize_results.py --compare --latex

# Fast-channel only (no comparison)
python scripts/summarize_results.py
```

Columns: task display name, head, N subjects, fast AUROC @ k=5 (best context),
full AUROC @ k=5 (best context), Δ. Also prints k=all (ceiling) table below.

**Saturation plots (per head, per channel, via gen_commands.py):**
```bash
# Fast-channel
python scripts/gen_commands.py saturation sex_binary --heads lstm transformer | bash
# Full-channel
python scripts/gen_commands.py --registry experiments/v2_full_registry.yaml \
    saturation sex_binary --heads lstm transformer | bash
```

---

### Modality ablation run (channel importance)

> **⚠️ 2026-06-27: the completed run below used the wrong model architecture for all 25
> experiments** (`hidden_dim: 256, num_layers: 2` — the sleep-staging arch — instead of
> `hidden_dim: 128, num_layers: 1`, matching `v3`/`v3_full`'s seq2label arch). The playbook steps
> below are still correct and will be reused as-is for the rerun — only
> `configs/phase0_v3_abl_config.yaml`'s `model:` section needs fixing first. Nothing in this
> section has been deleted; it documents what was run and remains the reference for how to run it
> again correctly. **Full archive-then-rerun plan: `docs/REMAINING_TRAINING_CHECKLIST.md` §
> Later: re-running v3_abl analysis after the architecture fix.**

**Goal:** Measure how much each SleepFM modality group contributes to each clinical task by
retraining the head with one or more groups permanently zeroed. The head never sees the absent
modality during training — this measures **peak capability** of each channel subset, not
robustness of the baseline model (see §4G of `SOTA_COMPARISON_AND_ABLATIONS.md` for the
inference-only robustness variant).

**Key properties:**
- **No new preprocessing or embedding extraction.** Reuses the fast-channel `.npy` files
  exactly. The zeroing happens in `ContextWindowDataset.__getitem__()` via `zero_modality_indices`,
  applied to the float32 copy before the model sees it — the `.npy` files on disk are never touched.
- **Completely separate outputs.** Results go to `phase0_v3_abl/`, logs to `logs_v3_abl/`.
  There is zero overlap with `phase0_v3/` or `phase0_v3_full/`.
- **Status:** ✅ all 25 experiments completed 2026-06-17, but ⚠️ with the wrong architecture (see
  warning above) — being rerun. Once retrained, status will start at "pending" again for the
  fresh run (the registry points to a dedicated results directory, archived and recreated empty
  — see the rerun plan).

#### Why these five ablation conditions?

| Condition (`run_tag`) | Groups zeroed | Groups active | Active embedding ranges | Answers |
|---|---|---|---|---|
| `abl_no_bas` | BAS (EEG+EOG) | RESP+EKG+EMG | [128:512] | Can we drop the neural sensor entirely? |
| `abl_no_resp` | RESP | BAS+EKG+EMG | [0:128], [256:512] | Does removing respiratory signal alone hurt? |
| `abl_no_ekg` | EKG | BAS+RESP+EMG | [0:256], [384:512] | Does removing the cardiac signal alone hurt? |
| `abl_cardio` | BAS+EMG | RESP+EKG | [128:256], [256:384] | SleepFounder comparison: cardio-only performance |
| `abl_bas_only` | RESP+EKG+EMG | BAS only | [0:128] | How much do brain signals alone explain? |

The baseline (all channels active) is the existing `phase0_v3` result for each task at the
matched context length — no additional "full" training run is needed.

**`no_resp` and `no_ekg` were added 2026-06-17**, after reviewing OSF (arXiv:2603.00190). OSF's
own flagship missing-channel ablation is exactly a single-group leave-one-out: they zero
Respiratory channels alone and evaluate on Hypopnea/Oxygen Desaturation — the direct analogue of
our `apnea_binary`. Without `no_resp`, our design could only infer RESP's necessity indirectly
(via `no_bas`/`cardio`/`bas_only`, all of which conflate RESP with EKG/EMG). `no_ekg` completes
the single-knockout matrix for the three non-EMG groups OSF varies in their secondary
"realistic settings" table. EMG is still not isolated — none of our 5 ablation tasks are
EMG/PLM-driven the way sleep staging would be. See `SOTA_COMPARISON_AND_ABLATIONS.md` §A.1 for
the full rationale.

**SleepFM embedding layout (512 dim = 4 × 128):**
```
[  0:128]  BAS  — EEG leads + EOG (brain activity)
[128:256]  RESP — Airflow, Thor, ABD, SpO2, HR, Snore, RespRate
[256:384]  EKG  — cardiac
[384:512]  EMG  — chin + leg muscles
```

#### Tasks and context choices

| Task | Context | Reason |
|---|---|---|
| `sex_binary` | 120m | Well-powered (N≈13k), L*=120m; all groups expected to contribute |
| `apnea_binary` | 120m | L*=120m; RESP expected dominant (OSA is a respiratory disorder) |
| `sleep_efficiency_binary` | 120m | Highest context benefit; BAS expected crucial (SE encodes staging) |
| `age_class` | 120m | Large N, physiological aging visible in all modalities |
| `bmi_binary` | 40m | L*=10m; 40m avoids sparse-window issues at 120m, uses default LR=1e-4 |

25 experiments total: 5 tasks × 5 conditions. ✅ All complete as of 2026-06-17. Results in
`SOTA_COMPARISON_AND_ABLATIONS.md` §A.6.1 and `results/tables/table6_modality.{csv,md,tex}`.

#### Path summary

```
Config          →  configs/phase0_v3_abl_config.yaml
Registry        →  experiments/v2_ablation_registry.yaml
Embeddings      →  /scratch/boshra95/psg/unified/embeddings/sleepfm_5sec/  (SHARED, read-only)
Targets         →  /scratch/boshra95/psg/unified/targets_v2/               (SHARED, read-only)
Training results→  /scratch/boshra95/psg/unified/results/phase0_v3_abl/{task}_{head}_{tag}/context_{L}/
Inference       →  /scratch/boshra95/psg/unified/results/phase0_v3_abl/inference/{task}_{head}_{tag}/
All logs        →  /home/boshra95/NSRR-tools/logs_v3_abl/
```

Working directory for all commands: `cd /home/boshra95/NSRR-tools`

```bash
REG="--registry experiments/v2_ablation_registry.yaml"
```

#### Step 0 — No preprocessing or embedding extraction

> The modality ablation **reuses the existing fast-channel embeddings** unchanged.
> Steps 0 and 1 of the other playbooks (EDF→HDF5, HDF5→.npy) do not need to be repeated.
> Start directly at training (Step 1 below).

Verify the embeddings are available before submitting training jobs:
```bash
find /scratch/boshra95/psg/unified/embeddings/sleepfm_5sec -name '*.npy' | wc -l
# Expected: ~14,992
```

#### Step 1 — Training (25 jobs: 5 tasks × 5 conditions) ✅ completed 2026-06-17, ⚠️ wrong arch, rerunning

Config: `configs/phase0_v3_abl_config.yaml` — **intended** hidden=128, layers=1 (LSTM head only,
matching the seq2label arch used by `v3`/`v3_full`). The file as actually used for the 2026-06-17
run had `hidden_dim: 256, num_layers: 2` instead (a copy-paste bug from the staging config) — see
the warning at the top of this section. Being fixed and rerun; this line will be updated once the
corrected run completes.
Results: `/scratch/boshra95/psg/unified/results/phase0_v3_abl/{task}_lstm_{tag}/context_{L}/`.
Logs: `logs_v3_abl/train_{task}_lstm_{tag}_{context}_lr{lr}_{jobid}.out`.

The `zero_modalities` field in the registry is read by `gen_commands.py` and forwarded as the
`ZERO_MODALITIES` env var to the SLURM job, which passes `--zero-modalities BAS` etc. to
`train_context_sweep.py`. No manual editing of env vars is needed.

```bash
REG="--registry experiments/v2_ablation_registry.yaml"

# ── Condition 1: no BAS (RESP+EKG+EMG active) ────────────────────────────────
python scripts/gen_commands.py $REG train sex_binary_lstm_abl_no_bas              | bash
python scripts/gen_commands.py $REG train apnea_binary_lstm_abl_no_bas            | bash
python scripts/gen_commands.py $REG train sleep_efficiency_binary_lstm_abl_no_bas | bash
python scripts/gen_commands.py $REG train age_class_lstm_abl_no_bas               | bash
python scripts/gen_commands.py $REG train bmi_binary_lstm_abl_no_bas              | bash

# ── Condition 2: no RESP (BAS+EKG+EMG active) ────────────────────────────────
python scripts/gen_commands.py $REG train sex_binary_lstm_abl_no_resp              | bash
python scripts/gen_commands.py $REG train apnea_binary_lstm_abl_no_resp            | bash
python scripts/gen_commands.py $REG train sleep_efficiency_binary_lstm_abl_no_resp | bash
python scripts/gen_commands.py $REG train age_class_lstm_abl_no_resp               | bash
python scripts/gen_commands.py $REG train bmi_binary_lstm_abl_no_resp              | bash

# ── Condition 3: no EKG (BAS+RESP+EMG active) ────────────────────────────────
python scripts/gen_commands.py $REG train sex_binary_lstm_abl_no_ekg              | bash
python scripts/gen_commands.py $REG train apnea_binary_lstm_abl_no_ekg            | bash
python scripts/gen_commands.py $REG train sleep_efficiency_binary_lstm_abl_no_ekg | bash
python scripts/gen_commands.py $REG train age_class_lstm_abl_no_ekg               | bash
python scripts/gen_commands.py $REG train bmi_binary_lstm_abl_no_ekg              | bash

# ── Condition 4: cardio only (BAS+EMG zeroed, RESP+EKG active) ──────────────
python scripts/gen_commands.py $REG train sex_binary_lstm_abl_cardio              | bash
python scripts/gen_commands.py $REG train apnea_binary_lstm_abl_cardio            | bash
python scripts/gen_commands.py $REG train sleep_efficiency_binary_lstm_abl_cardio | bash
python scripts/gen_commands.py $REG train age_class_lstm_abl_cardio               | bash
python scripts/gen_commands.py $REG train bmi_binary_lstm_abl_cardio              | bash

# ── Condition 5: BAS only (RESP+EKG+EMG zeroed) ──────────────────────────────
python scripts/gen_commands.py $REG train sex_binary_lstm_abl_bas_only              | bash
python scripts/gen_commands.py $REG train apnea_binary_lstm_abl_bas_only            | bash
python scripts/gen_commands.py $REG train sleep_efficiency_binary_lstm_abl_bas_only | bash
python scripts/gen_commands.py $REG train age_class_lstm_abl_bas_only               | bash
python scripts/gen_commands.py $REG train bmi_binary_lstm_abl_bas_only              | bash
```

**Status as of 2026-06-29 (post architecture fix):** the 2026-06-17 run above is archived (see
the warning at the top of this section) — the live `phase0_v3_abl/` directory was emptied and is
being repopulated with the corrected 128/1 architecture. `sex_binary_lstm_abl_no_bas` was already
retrained and verified (the Step 4 sanity check in `REMAINING_TRAINING_CHECKLIST.md`) — running
the commands above now will correctly skip that one (its `best_model.pt` already exists with the
fixed arch) and train the other 24. The list above is otherwise unchanged and reusable as-is.

**Check status:**
```bash
python scripts/gen_commands.py $REG status
python scripts/gen_commands.py $REG list
```

**Verify a generated command before submitting** (should show `ZERO_MODALITIES="BAS"`):
```bash
python scripts/gen_commands.py $REG train sex_binary_lstm_abl_no_bas
```

Expected output includes:
```
TASK=sex_binary TASK_TYPE=seq2label HEAD=lstm CONTEXT=120m \
  DATASETS="apples shhs" BATCH_SIZE=32 ACCUM_STEPS=1 LR=5e-5 \
  RUN_TAG="abl_no_bas" ZERO_MODALITIES="BAS" \
  CONFIG=configs/phase0_v3_abl_config.yaml \
  sbatch --requeue --time=06:00:00 ...
```

**Expected wall times** (H100 MIG, LSTM, n_size=large):
- 120m context: ~6h (same as v3 120m, K=5 fixed, same N subjects)
- 40m context (bmi_binary): ~3h

#### Step 2 — Inference

One GPU job per experiment. Auto-discovers the trained context and skips any already-inferred.
Results: `.../phase0_v3_abl/inference/{task}_lstm_{tag}/context_{L}/test_windows.parquet`.

```bash
REG="--registry experiments/v2_ablation_registry.yaml"

# All five conditions, all five tasks (run after each condition's training finishes)
for exp in \
  sex_binary_lstm_abl_no_bas              \
  apnea_binary_lstm_abl_no_bas            \
  sleep_efficiency_binary_lstm_abl_no_bas \
  age_class_lstm_abl_no_bas               \
  bmi_binary_lstm_abl_no_bas              \
  sex_binary_lstm_abl_no_resp              \
  apnea_binary_lstm_abl_no_resp            \
  sleep_efficiency_binary_lstm_abl_no_resp \
  age_class_lstm_abl_no_resp               \
  bmi_binary_lstm_abl_no_resp              \
  sex_binary_lstm_abl_no_ekg              \
  apnea_binary_lstm_abl_no_ekg            \
  sleep_efficiency_binary_lstm_abl_no_ekg \
  age_class_lstm_abl_no_ekg               \
  bmi_binary_lstm_abl_no_ekg              \
  sex_binary_lstm_abl_cardio              \
  apnea_binary_lstm_abl_cardio            \
  sleep_efficiency_binary_lstm_abl_cardio \
  age_class_lstm_abl_cardio               \
  bmi_binary_lstm_abl_cardio              \
  sex_binary_lstm_abl_bas_only              \
  apnea_binary_lstm_abl_bas_only            \
  sleep_efficiency_binary_lstm_abl_bas_only \
  age_class_lstm_abl_bas_only               \
  bmi_binary_lstm_abl_bas_only; do
    python scripts/gen_commands.py $REG infer $exp | bash
done
```

**Important:** the inference job reads `RUN_TAG` and `ZERO_MODALITIES` from the generated command,
ensuring it applies the same zeroing as training. A checkpoint trained with `abl_no_bas` will
be inferred with `ZERO_MODALITIES="BAS"` automatically — no manual matching is required.

#### Step 3 — Analysis (window sweep, no GPU)

Runs locally. Computes K-sweep AUROC at the standard K values (1, 5, 10, 20, 50, all).
No bootstrap or dense K needed — Table 6 only uses AUROC at K=5 and K=all.

> **Why not `run_analysis.sh`?** That script is designed for the full v3/v3_full pipeline
> (13 steps, multi-head, iso-compute heatmaps, task-comparison plots). Ablation is LSTM-only
> and only needs per-experiment analyze → collect → Table 6. Use the loop below instead.

```bash
REG="--registry experiments/v2_ablation_registry.yaml"
source /home/boshra95/sleepfm_env/bin/activate

ABL_EXPS="
  sex_binary_lstm_abl_no_bas
  apnea_binary_lstm_abl_no_bas
  sleep_efficiency_binary_lstm_abl_no_bas
  age_class_lstm_abl_no_bas
  bmi_binary_lstm_abl_no_bas
  sex_binary_lstm_abl_no_resp
  apnea_binary_lstm_abl_no_resp
  sleep_efficiency_binary_lstm_abl_no_resp
  age_class_lstm_abl_no_resp
  bmi_binary_lstm_abl_no_resp
  sex_binary_lstm_abl_no_ekg
  apnea_binary_lstm_abl_no_ekg
  sleep_efficiency_binary_lstm_abl_no_ekg
  age_class_lstm_abl_no_ekg
  bmi_binary_lstm_abl_no_ekg
  sex_binary_lstm_abl_cardio
  apnea_binary_lstm_abl_cardio
  sleep_efficiency_binary_lstm_abl_cardio
  age_class_lstm_abl_cardio
  bmi_binary_lstm_abl_cardio
  sex_binary_lstm_abl_bas_only
  apnea_binary_lstm_abl_bas_only
  sleep_efficiency_binary_lstm_abl_bas_only
  age_class_lstm_abl_bas_only
  bmi_binary_lstm_abl_bas_only
"

for exp in $ABL_EXPS; do
  echo "=== START $exp $(date) ==="
  python scripts/gen_commands.py $REG analyze $exp | bash
  echo "=== END $exp $(date) ==="
done 2>&1 | tee analysis_abl_step5.log
```

Output per experiment:
```
.../phase0_v3_abl/inference/{task}_lstm_{tag}/
  window_analysis_test.csv
  window_analysis.md
```

#### Step 4 — Collect results

```bash
REG="--registry experiments/v2_ablation_registry.yaml"

python scripts/gen_commands.py $REG collect | bash
```

> **Note:** `collect_results_v2.py`'s experiment-folder parser originally matched
> only `{task}_{head}` (suffix `_lstm`/`_transformer`/`_mean_pool`). Ablation folders
> are named `{task}_{head}_{run_tag}` (e.g. `sex_binary_lstm_abl_no_bas`), so the
> parser silently dropped every ablation row. This has been fixed — `parse_exp_dir()`
> now finds the head as a substring and captures everything after it as `run_tag`,
> and `run_tag` was added to the dedup keys (`TRAIN_KEY`, `ANALYSIS_KEY`) so the three
> ablation conditions for the same task/context don't collide. The fix is backward
> compatible: old CSVs without a `run_tag` column are treated as `run_tag=""` on
> read. If you collected ablation results before this fix and got 0 new rows,
> re-run the command above — it will now pick them up.

Outputs to:
```
/scratch/boshra95/psg/unified/results/phase0_v3_abl/collected/analysis.csv
/scratch/boshra95/psg/unified/results/phase0_v3_abl/collected/training.csv
```

These CSVs are committed to the repo (same as v3 collect) and appear under
`results/collected/phase0_v3_abl/` in the working tree.

#### Step 5 — Table 6 generation

**Script:** `scripts/make_table6_modality.py` ✅ implemented.
Reads `results/collected/phase0_v3/analysis.csv` (the "Full" baseline) and
`results/collected/phase0_v3_abl/analysis.csv` (the five ablation conditions, keyed
by `run_tag`), and joins them into one task × condition AUROC table with deltas.

```bash
source /home/boshra95/sleepfm_env/bin/activate

# Default: all 5 tasks, lstm head, K=all (full-night ceiling), test split
python scripts/make_table6_modality.py

# K=5 deployment scenario instead of K=all
python scripts/make_table6_modality.py --k 5

# Subset of tasks
python scripts/make_table6_modality.py --tasks sex_binary apnea_binary

# LaTeX output (printed to stdout in addition to saved files)
python scripts/make_table6_modality.py --latex
```

Output:
```
results/tables/table6_modality.csv
results/tables/table6_modality.md
results/tables/table6_modality.tex
```

Columns: `Task`, `Context`, `N_test`, `Full`, `No BAS`, `Δ(No BAS)`, `No RESP`, `Δ(No RESP)`,
`No EKG`, `Δ(No EKG)`, `Cardio only`, `Δ(Cardio only)`, `BAS only`, `Δ(BAS only)`. A `—` in
any condition column means that (task, condition) hasn't finished training/inference/analysis
yet — the script prints a reminder to check `gen_commands.py … status` when this happens.

If `results/collected/phase0_v3_abl/analysis.csv` is missing or stale (e.g. you trained
more conditions since the last collect), re-run Step 4 first — the script will tell you
which file is missing rather than failing silently.

#### Interpreting the results

| Pattern | Interpretation |
|---|---|
| `abl_no_bas` ≈ full AUROC | Task is independent of EEG/EOG; wrist-worn or contactless PSG may suffice |
| `abl_no_bas` large drop | Brain signals are uniquely informative; EEG/EOG cannot be dropped |
| `abl_no_resp` large drop, esp. for `apnea_binary` | Direct confirmation that respiratory signal alone is necessary — the OSF-style leave-one-out test |
| `abl_no_resp` ≈ full | Task does not need respiratory channels specifically (information is redundant with BAS/EKG/EMG) |
| `abl_no_ekg` ≈ full | Cardiac signal is not uniquely necessary for this task (no task in our 5-task set is hypothesized EKG-dominant) |
| `abl_no_ekg` notable drop | Unexpected — would suggest EKG/HRV carries independent signal even for non-cardiac tasks |
| `abl_cardio` ≈ `abl_no_bas` | EMG adds little; RESP+EKG drives the cardiorespiratory information |
| `abl_cardio` > SleepFounder AUROC | Full PSG superiority is not purely from extra cardiorespiratory channels |
| `abl_bas_only` ≈ full AUROC | Task is primarily encoded in EEG (e.g., sleep efficiency) |
| `abl_bas_only` near chance | No task information survives without RESP+EKG (e.g., apnea) |
| `abl_no_resp` ≪ `abl_no_bas`/`abl_no_ekg` for the same task | RESP is the single most necessary group — strongest, cleanest evidence in the table |

For `apnea_binary`: we expect `abl_no_bas` ≈ full and `abl_bas_only` near chance (OSA is a
respiratory disorder); **`abl_no_resp` is the cleanest test of this and should show the
largest drop of any condition for this task** — this is the direct OSF-style leave-one-out
analogue of their hypopnea/oxygen-desaturation finding (see `SOTA_COMPARISON_AND_ABLATIONS.md`
§A.1). For `sleep_efficiency_binary`: we expect `abl_bas_only` > `abl_cardio`, and `abl_no_resp`
to show a moderate drop too (sleep efficiency correlates with apnea). For `sex_binary`,
`age_class`, and `bmi_binary`: no strong prior on `abl_no_resp`/`abl_no_ekg` — these two
conditions are genuinely exploratory for these three tasks (no task in our 5-task set is
hypothesized to be EKG-dominant; that would be `cvd_binary`, which is not in the ablation set).

---

### Figure Generation — All Paper Figures

This section covers the complete set of figure and table commands needed to regenerate
every figure in the TBME paper (main + supplementary), after analyze/collect/build-heatmap
have already been run. For the full pipeline in one command, use `scripts/run_figures.sh`.

#### Quick start — full paper figure set

```bash
source /home/boshra95/sleepfm_env/bin/activate
cd /home/boshra95/NSRR-tools

# Dry-run first to review all commands:
bash scripts/run_figures.sh --dry-run

# Run everything (iso-plots + figures + cross-round + tables):
bash scripts/run_figures.sh 2>&1 | tee figures_run.log

# Skip the slow iso-compute step (if you only need saturation / task-comparison / etc.):
bash scripts/run_figures.sh --skip-iso 2>&1 | tee figures_run.log

# Skip cross-round figures and tables (v3 figures only):
bash scripts/run_figures.sh --skip-cross-round --skip-tables 2>&1 | tee figures_run.log
```

#### What `run_figures.sh` does (step by step)

| Step | Script / subcommand | Paper location | Experiments |
|---|---|---|---|
| 1 | `gen_commands.py iso-plots` | Fig 2, S-Fig 3 | All 21 (7 tasks × 3 heads) |
| 2 | `gen_commands.py saturation` | **Fig 1** | 7 tasks, all 3 heads |
| 3 | `gen_commands.py scaling-laws --plots 1B` | S-Fig 8 | 7 tasks |
| 4 | `gen_commands.py calibration` | S-Fig 4a, 4b | All 21 experiments |
| 5 | `gen_commands.py window-position` | S-Fig 7 | 7 LSTM experiments |
| 6 | `gen_commands.py subject-consistency` | S-Fig 6a, 6b | 7 Transformer experiments |
| 7 | `gen_commands.py cohort-saturation` | S-Fig 5 | 7 LSTM experiments |
| 8 | `gen_commands.py precision-recall` | S-Fig 10 | All 21 experiments |
| 9 | `gen_commands.py subject-kstar` | S-Fig 9 | 7 Transformer experiments |
| 10 | `gen_commands.py task-comparison` | **Fig 3** | 7 tasks, LSTM head |
| 11 | `plot_modality_bar.py` | **Fig 4** | v3_abl + v3 + v3_full (cross-round) |
| 12 | `plot_channel_comparison.py` | S-Fig 2 | v3 + v3_full (cross-round) |
| 13 | `plot_aggregate_scaling.py` | S-Fig 12 / Fig 5 TBD | v3 only |
| 14 | `make_table1_peak_auroc.py` (fast-ch) | **paper Table II** — peak AUROC fast-ch cols | v3 |
| 14b | `make_table1_peak_auroc.py` (full-ch) | **paper Table II** — peak AUROC full-ch cols | v3_full |
| 14c | `make_table2_lstar.py` | **paper Table III** — L* per task | v3 |
| 15 | `make_table4_sensitivity.py` | supp sensitivity ranking | v3 |
| 16 | `make_table9_cohort.py` × 3 | supp cohort breakdown (sex/bmi/apnea) | v3 (reads parquets) |
| 17 | `make_table10_ci.py` | supp bootstrap CIs | v3 |
| 18 | `make_table3_kgrid.py sex_binary_lstm` | supp K-grid | v3 |
| 19 | `make_table5_heads.py` | **paper Table IV** — heads at L* | v3 |
| 20 | `make_table6_modality.py` | **paper Table V** — modality Δ | v3 + v3_abl |

#### Cross-round figures — individual commands

These three scripts do not go through `gen_commands.py`; they read the collected CSVs
directly and work across multiple rounds.

```bash
source /home/boshra95/sleepfm_env/bin/activate
cd /home/boshra95/NSRR-tools

# Fig 4 — Modality contribution bar chart
#   Reads:  results/collected/phase0_v3_abl/analysis.csv  (ablation)
#           results/collected/phase0_v3/analysis.csv       (fast-ch baseline)
#           results/collected/phase0_v3_full/analysis.csv  (full-ch reference)
#   Output: /scratch/boshra95/psg/unified/results/phase0_v3_abl/figures/
#            phase0_v3_abl/modality_ablation_bar.{png,pdf}
python scripts/plot_modality_bar.py

# S-Fig 2 — Fast vs full channel saturation overlay (Transformer, 6 tasks)
#   Reads:  results/collected/phase0_v3/analysis.csv
#           results/collected/phase0_v3_full/analysis.csv
#   Output: /scratch/boshra95/psg_full/unified/results/phase0_v3_full/figures/
#            phase0_v3_full/channel_comparison.{png,pdf}
python scripts/plot_channel_comparison.py

# S-Fig 12 / Fig 5 (TBD) — Aggregate context-length scaling
#   Reads:  results/collected/phase0_v3/analysis.csv
#   Output: /scratch/boshra95/psg/unified/results/phase0_v3/figures/
#            aggregate/aggregate_scaling.{png,pdf}
python scripts/plot_aggregate_scaling.py \
    --collected-dir results/collected/phase0_v3 \
    --results-dir /scratch/boshra95/psg/unified/results/phase0_v3

# Robustness check without non-monotonic outlier:
python scripts/plot_aggregate_scaling.py \
    --collected-dir results/collected/phase0_v3 \
    --results-dir /scratch/boshra95/psg/unified/results/phase0_v3 \
    --exclude-tasks depression_extreme_binary
```

#### Table regeneration — individual commands

All scripts default to `results/collected/phase0_v3/analysis.csv` (v3 fast-channel).

```bash
source /home/boshra95/sleepfm_env/bin/activate
cd /home/boshra95/NSRR-tools

# ── Primary paper tables ─────────────────────────────────────────────────────

# Table 1 script → paper Table II — Peak AUROC per task × head at best context
# paper_figures.md: "Peak AUROC: fast-ch + full-ch columns" → run BOTH:
python scripts/make_table1_peak_auroc.py --latex
# Output: results/tables/table1_peak_auroc_fast.{csv,md,tex}

python scripts/make_table1_peak_auroc.py \
    --collected-dir results/collected/phase0_v3_full \
    --channel full \
    --latex
# Output: results/tables/table1_peak_auroc_full.{csv,md,tex}

# Table 2 script → paper Table III — L* saturation context + ΔAUROC from 30s
python scripts/make_table2_lstar.py --latex
# Output: results/tables/table2_lstar_fast.{csv,md,tex}

# Table 5 script → paper Table IV — Head comparison at LSTM's L* (K=5 and K=all)
python scripts/make_table5_heads.py --latex
# Output: results/tables/table5_heads_fast.{csv,md,tex}
# Note: Transformer/MeanPool values may be lower than Table II — they're evaluated
# at the LSTM's L*, not each head's own best context.

# Table 6 script → paper Table V — Modality ablation ΔAUROC (v3 + v3_abl)
python scripts/make_table6_modality.py --latex
# Output: results/tables/table6_modality.{csv,md,tex}

# ── Supplementary tables ─────────────────────────────────────────────────────

# Table 3 — K-grid (K × context AUROC pivot; sex_binary_lstm as paper example)
python scripts/make_table3_kgrid.py sex_binary_lstm --latex
# Output: results/tables/table3_kgrid_sex_binary_lstm_fast.{csv,md,tex}

# Table 4 — Context sensitivity ranking (AUROC gain from 30s to best L)
python scripts/make_table4_sensitivity.py --latex
# Output: results/tables/table4_sensitivity_fast_lstm.{csv,md,tex}

# Table 9 — Per-cohort AUROC breakdown at L* (reads inference parquets directly)
# PAPER_TABLES.md: "representative tasks: sex_binary, bmi_binary, apnea_binary"
for exp_id in sex_binary_lstm bmi_binary_lstm apnea_binary_lstm; do
    python scripts/make_table9_cohort.py "$exp_id" --latex
done
# Output: results/tables/table9_cohort_{exp_id}_fast.{csv,md,tex}

# Table 10 — Bootstrap CI summary (requires --bootstrap N in analyze step)
python scripts/make_table10_ci.py --latex
# Output: results/tables/table10_ci_fast.{csv,md,tex}
```

#### Blacklisted outputs (never include in paper)

These plot functions exist in the scripts but are excluded from default `--plots`
and must not appear in the paper:

| Blacklisted output | Script | Reason |
|---|---|---|
| `*_calibration_2C_ece_vs_k` | `plot_calibration.py` | Excluded per design |
| `*_subject_consistency_5B_variance_vs_k` | `plot_subject_consistency.py` | Excluded |
| `task_comparison_6B_bars` | `plot_task_comparison.py` | Redundant with Fig 1 + Table II |
| `*_pr_8C_vote_sweep` | `plot_precision_recall.py` | Majority-vote removed from paper |
| `*_kstar_9B_coverage` | `plot_subject_kstar.py` | Excluded |
| `*_cohort_saturation_7B_n` | `plot_cohort_saturation.py` | N in Methods only |
| `double_tradeoff` | `plot_iso_compute.py` | Redundant with heatmap + pareto |
| `*_1C_optimal_epoch` | `plot_scaling_laws.py` | Non-monotonic |

#### Pending figures (not yet generated)

| Figure | Status | Blocker |
|---|---|---|
| S-Fig 11 (1A uShape) | PENDING | Requires rerun with balanced-accuracy metric |
| S-Fig 12 / Fig 5 placement | TBD | Decide main vs. supp after viewing std bands |

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

### Training / head configs

| Config | Run | Head config | Monitor | Results dir | Registry |
|--------|-----|-------------|---------|-------------|---------|
| `configs/phase0_v3_config.yaml` | Fast-channel, **seq2label** tasks | hidden=128, layers=1 (seq2label ran with this) | `val_auroc` | `phase0_v3/` | `v2_registry.yaml` |
| `configs/phase0_v3_staging_config.yaml` | Fast-channel, **sleep staging** only | hidden=256, layers=2, epochs=60 | `val_kappa` | `phase0_v3/` | `v2_registry.yaml` (per-exp override) |
| `configs/phase0_v3_full_config.yaml` | Full-channel, **seq2label** tasks | hidden=128, layers=1 | `val_auroc` | `phase0_v3_full/` | `v2_full_registry.yaml` |
| `configs/phase0_v3_full_staging_config.yaml` | Full-channel, **sleep staging** only | hidden=256, layers=2, epochs=60 | `val_kappa` | `phase0_v3_full/` | `v2_full_registry.yaml` (per-exp override) |
| `configs/phase0_v2_config.yaml` | Archived | — | — | `phase0_v2/` | — |

### Preprocessing configs

| Config | Channels | Output root | Strategy |
|--------|----------|-------------|---------|
| `configs/preprocessing_params.yaml` | 7–8 (fast-channel baseline) | `/scratch/boshra95/psg/` | `sleepfm` (minimal) |
| `configs/preprocessing_params_full.yaml` | Up to 23 (channel expansion) | `/scratch/boshra95/psg_full/` | `sleepfm_full` |

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

## Experiment Plan

### Tasks overview

All tasks run under v3 protocol and write results to `phase0_v3/`. Tasks originally from phase0 (legacy) have been added to the registry and re-run here for a consistent comparison baseline.

| Task | Type | Classes | N | Datasets | Tier | Phase0? |
|------|------|---------|---|----------|------|---------|
| `sex_binary` | seq2label | 2 | ~13k | APPLES, SHHS | 1 | No |
| `sleep_efficiency_binary` | seq2label | 2 | ~13k | APPLES, SHHS, MrOS | 1 | No |
| `bmi_binary` | seq2label | 2 | ~15k | APPLES, SHHS, MrOS | 1 | No |
| `age_class` | seq2label | 3 | ~16k | APPLES, SHHS, MrOS | 1 | No |
| `apnea_binary` | seq2label | 2 | 14,097 | APPLES, SHHS, MrOS, STAGES | 1 | ✓ |
| `sleep_staging` | seq2seq | 5 | 14,960 | SHHS, MrOS, STAGES, APPLES | 1 | ✓ |
| `psqi_binary` | seq2label | 2 | ~4k | MrOS | 2 | No |
| `depression_extreme_binary` | seq2label | 2 | ~1.8k | APPLES, STAGES | 2 | No |
| `osa_binary_apples_postqc` | seq2label | 2 | ~1.5k | APPLES | 2 | No |
| `osa_severity_apples` | seq2label | 4 | ~1.5k | APPLES | 2 | No |
| `cvd_binary` | seq2label | 2 | 13,045 | SHHS, MrOS | 2 | ✓ |
| `sleepiness_binary` | seq2label | 2 | 16,431 | APPLES, SHHS, MrOS, STAGES | 2 | ✓ |
| `insomnia_binary` | seq2label | 2 | 1,710 | STAGES | deferred | ✓ |
| `rested_morning` | seq2label | 2 | 3,934 | MrOS | deferred | ✓ |
| `anxiety_binary` | seq2label | 2 | 1,698 | STAGES | deferred | ✓ |

Context lengths:
- **Tier 1:** `30s, 10m, 40m, 80m, 120m, 240m` (all 6)
- **Tier 2:** `30s, 10m, 40m, 80m, 120m` (small N) or all 6 for large-N Tier 2 tasks (cvd, sleepiness)
- **Deferred:** `30s, 10m, 40m, 80m, 120m`

### Experiments per task

**Tier 1** — all three heads:

| Experiment ID | Head | Notes |
|--------------|------|-------|
| `sex_binary_lstm/transformer/mean_pool` | all | |
| `sleep_efficiency_binary_lstm/transformer/mean_pool` | all | |
| `bmi_binary_lstm/transformer/mean_pool` | all | |
| `age_class_lstm/transformer/mean_pool` | all | 3-class |
| `apnea_binary_lstm/transformer/mean_pool` | all | legacy re-run |
| `sleep_staging_lstm/transformer/mean_pool` | all | seq2seq, 5-class; primary metric: Cohen's κ |

**Tier 2** — lstm only (add more heads if results warrant):

| Experiment ID | Head | Notes |
|--------------|------|-------|
| `psqi_binary_lstm` | lstm | MrOS only |
| `depression_extreme_binary_lstm` | lstm | APPLES+STAGES, extreme-group design |
| `osa_binary_apples_postqc_lstm` | lstm | APPLES only |
| `osa_severity_apples_lstm` | lstm | APPLES only, 4-class |
| `cvd_binary_lstm` | lstm | legacy re-run; mixed CVD definition (see Notes) |
| `sleepiness_binary_lstm` | lstm | legacy re-run; borderline AUROC in phase0 |

**Deferred** — run last, poor phase0 signal:

| Experiment ID | Notes |
|--------------|-------|
| `insomnia_binary_lstm` | STAGES only; phase0 AUROC ~0.58 |
| `rested_morning_lstm` | MrOS only; phase0 AUROC ~0.53 (near chance) |
| `anxiety_binary_lstm` | STAGES only; phase0 AUROC ~0.57 |

### Suggested run order

1. Submit Tier 1 lstm across all contexts first (validate pipeline end-to-end)
2. Submit Tier 1 transformer and mean_pool in parallel
3. After Tier 1 trains → run inference + analysis
4. Submit Tier 2 (cvd and sleepiness are large N, schedule like Tier 1)
5. Run inference + analysis for Tier 2
6. Submit deferred tasks only if GPU budget allows

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

### Why K=5 fixed is the right default (paper defence)

Reviewers sometimes argue that K=5 at 30s "wastes available data" because only 5 of ~960 possible windows are used per epoch. This conflates two distinct notions of fairness:

| Fairness criterion | K=5 fixed | Token budget (K×L = const) |
|---|---|---|
| Equal gradient updates/subject/epoch | ✅ identical across all L | ✗ 160× more updates at 30s |
| Equal information/subject/epoch | ✗ 30s sees far less signal | ✅ constant across all L |

**K=5 fixed is fair in the gradient-update sense**, which is the right criterion for comparing context lengths. K=all would give the 30s model 240× more gradient updates per epoch than the 120m model — the comparison would measure how many training iterations each model received, not whether longer context helps.

Token budget is fair in the information sense — but it gives the 30s model 160× more gradient updates per epoch, introducing a different confound (higher effective learning rate, stronger regularisation through repetition).

Crucially, at long contexts (80m, 120m) both strategies converge: K_token_budget ≈ 1 ≈ K=5. The only meaningful difference is at short contexts (30s, 10m). With sufficient epochs and random window sampling, the K=5 model still covers most of the 30s window space over training — K=5 controls the per-epoch exposure, not the total data seen across the full training run.

**Recommended paper wording (Methods):**
> "At each context length, K=5 randomly-sampled non-overlapping windows were drawn per subject per training epoch, keeping the number of gradient updates per subject constant across context lengths. This controls for training compute when comparing models at different context lengths. As a sensitivity analysis, we verified that training with a token-budget schedule (K × L = 80 min, giving K=160 at 30s) yields the same qualitative saturation curves (Supplementary Table X)."

**Recommended ablation:** Run `sex_binary_lstm` (or `bmi_binary_lstm`) with `windows_strategy: "token_budget"` for all 6 contexts using a separate `run_tag: "kbudget"`. If the saturation curve shape is the same (expected, since at 80m+ both are already K≈1), report as a supplementary sensitivity check and close the reviewer concern entirely. See `context_length_experiment_design.md` §12–13 for the ready-to-run registry entry and config.

---

## K Windows: Training vs Val vs Inference

There are three distinct K values in the pipeline. They use different sources and different windowing pools:

| K concept | Where set | Pool | Typical value |
|-----------|-----------|------|---------------|
| **K_train** (windows/epoch) | `dataset.windows_per_subject` in config (or token_budget) | **Overlapping** — any start in [0, T−N], random | 5 |
| **K_val** (training-time val/test) | same `windows_per_subject` config | **Overlapping** — evenly spaced across [0, T−N], deterministic | 5 at all context lengths |
| **K_infer** (inference, all windows) | `--all-windows` flag → K_max=99,999 | **Non-overlapping** stride-N positions | T//N (all) |
| **K_analysis** (post-hoc sweep) | `analyze_windows.py --k-values` | Subsampled from K_infer parquet | 1,5,10,20,50,all |

Key points:
- **K_val = K_max = 5 at all context lengths.** Val and test during training use the overlapping pool (K_max ≤ 100 branch), so K=5 is always achieved regardless of context length. The 5 positions are evenly spaced across [0, T−N] and fixed per subject, giving a deterministic, stable early-stopping signal.
- **K_infer recovers all T//N windows** because `infer_subject_windows.py` overrides `windows_per_subject` to 99,999, routing to the non-overlapping stride-N branch. Inference is unaffected by the val windowing change.
- **K_analysis is completely separate** — it subsamples from the already-saved parquet rows in `analyze_windows.py`. No model or GPU needed.
- The distinction between K_val and K_infer is implemented via K_max: ≤100 → overlapping (val/test), >100 → non-overlapping (inference).

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

## Model Architecture Reference

Documents the downstream head hyperparameters, their rationale, and implications for
cross-run comparisons. Relevant for paper Methods §III-E and for any decision to change
the config.

### Input representation

After SleepFM embedding extraction each subject is a `[T, 4, 128]` array (T = number of
5-second patches, 4 modalities, 128-dim SleepFM embedding per modality). At training time
a context window of N patches is flattened to `[N, 512]` as head input.

```
input_dim = 4 modalities × 128 SleepFM dims = 512    ← fixed, not a tunable hyperparameter
```

### The `hidden_dim` config key controls both LSTM and Transformer

A single yaml value `model.hidden_dim` drives both head architectures:

| Head | Role of `hidden_dim` |
|---|---|
| LSTMHead | LSTM hidden-state size; BiLSTM output = `2 × hidden_dim` |
| TransformerHead | `d_model` (token representation width); `dim_feedforward = 4 × hidden_dim` |
| MeanPoolHead | Unused — goes straight from `input_dim=512` to `Linear(512, C)` |

So `hidden_dim: 128` in the seq2label config means both the LSTM hidden size and the Transformer
d_model are 128; sleep staging uses a separate config with `hidden_dim: 256`.

### Architecture configs per run

Head configs are **intentionally matched** between fast-channel and full-channel so that the
only variable in the fast→full comparison is the number of PSG channels, not model capacity:

| Run | seq2label config | Sleep staging config | seq2label head | Sleep staging head |
|---|---|---|---|---|
| Fast-channel (v3) | `phase0_v3_config.yaml` (val_auroc) | `phase0_v3_staging_config.yaml` (val_kappa, epochs=60) | hidden=128, layers=1 (~658K LSTM) | hidden=256, layers=2 (~3.16M LSTM) |
| Full-channel (v3_full) | `phase0_v3_full_config.yaml` (val_auroc) | `phase0_v3_full_staging_config.yaml` (val_kappa, epochs=60) | hidden=128, layers=1 (~658K LSTM) | hidden=256, layers=2 (~3.16M LSTM) |

Both sleep staging configs use `val_kappa` as `early_stopping_monitor` (changed from `val_auroc`
on 2026-06-08). Sleep staging entries in both registries have an explicit `config:` field;
`gen_commands.py` picks `exp.get("config")` before `registry["config"]` automatically.

This is the correct design: any AUROC or kappa difference fast→full is attributable solely to
richer channels. A reviewer asking "is the gain from the bigger model?" can be answered "no —
architectures are identical in both runs."

**Parameter counts:**

| Head | Config | Parameters |
|---|---|---|
| LSTMHead (seq2label) | hidden=128, layers=1, BiLSTM | ~658K |
| TransformerHead (seq2label) | d_model=128, heads=8, ff=512, layers=1 | ~264K |
| LSTMHead (sleep staging) | hidden=256, layers=2, BiLSTM | ~3.16M |
| TransformerHead (sleep staging) | d_model=256, heads=8, ff=1024, layers=2 | ~1.7M |
| MeanPoolHead | any | ~1K |

### Width vs depth rationale

Hidden=128 with 1 BiLSTM layer is not underpowered for seq2label tasks:
- The fast-channel results (e.g., sex_binary AUROC 0.741–0.757) demonstrate the head extracts
  real signal; the full-channel run will add real channel information on top of the same head.
- Sleep staging uses 256/2 because phase0 showed kappa dropping from 0.62 → 0.54 at 10m
  when using 128/1; the 5-class seq2seq task needs substantially more capacity than binary tasks.
- For seq2label, going 256/2 would add ~2.5M params over 658K with diminishing returns at
  N=1k–16k subjects, and would confound the channel comparison.

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

**`depression_extreme_binary`**: Extreme-group design — middle BDI/PHQ-9 range subjects dropped. Reduces effective N from ~2.8k to ~1.8k. Now includes STAGES (PHQ-9 ≤4 → 0, ≥15 → 1) alongside APPLES (BDI ≤9 → 0, ≥20 → 1). APPLES-only training failed (only 27 class-1 subjects in APPLES alone).

**`osa_binary_apples_postqc`** and **`osa_severity_apples`**: APPLES-only. Small N (~1,516). Context lengths up to 120m included; be cautious with 80m+ given small per-split N.

**`psqi_binary`**: MrOS-only. Both MrOS visits contribute (PSQI is visit-specific). N=~3,933 across both visits.

**`apnea_binary`** (legacy re-run): AHI≥15 standard clinical threshold. 4 datasets, ~1:1 class balance, N=14,097. Phase0 AUROC reached 0.73 at 40m with lstm. This is the primary OSA result for the paper; `osa_binary_apples_postqc` is a supplementary ablation using clinician-adjudicated labels.

**`sleep_staging`** (legacy re-run): Anchor-based seq2seq — model sees a context window and predicts the label of the centre epoch. 5 classes: Wake=0, N1=1, N2=2, N3=3, REM=4. Primary metric: Cohen's κ and per-stage F1 (AUROC also logged for reference). Phase0 κ=0.58–0.63 at 10–40m. N1 is minority (~5–8% of epochs) and will have lowest per-stage F1. mean_pool head loses position info and may underperform on this anchor task.

**`cvd_binary`** (legacy re-run): SHHS uses composite any_cvd (CHD + stroke + heart failure + peripheral vascular); MrOS uses cvchd (coronary heart disease only) — these are different definitions. Merged here for consistency with phase0. For publication, consider reporting SHHS-only and MrOS-only separately. Phase0 AUROC 0.64–0.67.

**`sleepiness_binary`** (legacy re-run): ESS≥11 threshold. N=16,431 across all 4 datasets. Phase0 AUROC 0.59–0.61 — borderline. Included as Tier 2 to test whether v3 overlapping-window protocol and 240m context improve the signal.

**`insomnia_binary`**, **`rested_morning`**, **`anxiety_binary`** (deferred): All had AUROC ≤0.60 in phase0 (near-chance for rested_morning at 0.52–0.54). Run only if GPU budget allows; results are unlikely to change the paper's conclusions.

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

---

## seq2seq Window Design (Sleep Staging)

Sleep staging uses `task_type: seq2seq`. The context window and anchor filtering logic
differs from seq2label and is controlled by three config params under `dataset:`.

### seq2seq_context_mode

**`"causal"` (original):** Window covers the N patches immediately before (and including)
the anchor epoch. Past-only; no future signal.

**`"centered"` (recommended for sleep staging):** Window is symmetric around the anchor
epoch: `[(N-6)//2 past patches] + [6 anchor patches] + [(N-6)//2 future patches]`.
Uses both past and future context. Better absolute performance; standard in the sleep
staging literature.

At 30s (N=6): centered == causal (no room for extra context — window IS the anchor epoch).

### seq2seq_padding_policy

Controls which anchor epochs are included in the dataset index:

| Policy | Included anchors | Padding |
|--------|-----------------|---------|
| `"allow_all"` | All epochs; causal mode applies legacy `min_past` filter | Possible |
| `"max_fraction"` | Epochs where `padding/N <= seq2seq_max_padding_fraction` | Bounded |
| `"complete_only"` | Only epochs where full context fits within recording | None |

`"complete_only"` eliminates all padding → Flash attention fires at all context lengths
→ 2–5× faster training. Anchors excluded are those within `(N-6)//2` patches of the
recording boundary (centered) or within the first N patches (causal).

### Results history and config used

| Results directory | context_mode | padding_policy | hidden_dim | num_layers |
|---|---|---|---|---|
| `sleep_staging_lstm_old_arch128` | `causal` | `allow_all` | 128 | 1 |
| `sleep_staging_transformer_old_arch128` | `causal` | `allow_all` | 128 | 1 |
| `sleep_staging_lstm` | `centered` | `complete_only` | 256 | 2 |
| `sleep_staging_transformer` | `centered` | `complete_only` | 256 | 2 |

The old `_old_arch128` results used the default config values (causal, allow_all, hidden=128).
New results use the sleep-stage-redesign branch config (centered, complete_only, hidden=256).

### Paper Methods wording

> "For sleep staging we adopt a symmetric context window of length L centred on each anchor
> epoch, comprising ⌊(L−30s)/2⌋ seconds of past signal, the 30-second anchor epoch, and
> ⌊(L−30s)/2⌋ seconds of future signal. We include only anchor epochs for which the full
> context window lies within the recording, ensuring no zero-padding is introduced and
> enabling Flash attention throughout. At context length L, this excludes the first and
> last ⌊L/2⌋ − 15 seconds of each recording. At 30s the window reduces to the anchor epoch
> alone; at 240m approximately the central 4 hours of each recording are evaluated."

---

## Post-hoc Decision Threshold Tuning (binary tasks only)

For class-imbalanced binary tasks the default t=0.5 threshold under-predicts the minority
class. This section describes the post-hoc threshold tuning step that should be run after
inference is complete for all seq2label binary tasks.

**Does not require retraining.** AUROC is unchanged. Only balanced accuracy and per-class
recall change. See `docs/POSTHOC_THRESHOLD_TUNING.md` for full background and task table.

### Which tasks need it (v3 results — COMPLETED 2026-05-30)

Val inference and tuning have been run for all binary tasks. Results in
`inference/{exp_id}/threshold_tuning.csv`. See `docs/POSTHOC_THRESHOLD_TUNING.md`
for the full per-context numbers and surprises. Summary:

**Use tuned BA in paper (positive gain):**

| Task | Best gain | Note |
|---|---|---|
| `osa_binary_apples_postqc_lstm` | **+0.22 at 10m** | MUST use — t=0.5 predicts class 1 for ~98% of subjects |
| `depression_extreme_binary_lstm` | **+0.065 at 80m** | Surprise — balanced at short ctx, biased at long ctx |
| `bmi_binary_transformer` | +0.027 avg +0.013 | Consistently positive across contexts |
| `bmi_binary_lstm` | +0.013 avg +0.006 | Smaller than predicted (val-based threshold more conservative) |
| `sex_binary_lstm` | +0.020 avg +0.009 | Small but real |
| `sleepiness_binary_lstm` | avg +0.006 | Small; include for consistency |
| `sleepiness_binary_transformer` | avg +0.006 | Small; include for consistency |

**Keep t=0.5 (tuning zero or negative):**

| Task | Reason |
|---|---|
| `cvd_binary_lstm` | Tuning HURTS (avg −0.005) — val too small to reliably generalise |
| `cvd_binary_transformer` | Near zero or negative |
| `sex_binary_transformer` | Near zero |
| `apnea_binary_lstm/transformer` | Near zero |
| `sleep_efficiency_binary_*` | Near zero, some negative |

**Skip (not binary):** `age_class` (3-class), `sleep_staging` (5-class seq2seq), `psqi_binary_lstm` (AUROC=0.525, near chance).

### Step 1 — Run val inference

Val parquets do not exist yet (inference was only run on test). Re-run inference with
`--split val` for each binary experiment — reuses trained model, no GPU needed:

```bash
python scripts/gen_commands.py infer <exp_id> --split val | bash
```

For all binary seq2label experiments at once (adjust list as needed):
```bash
for exp in bmi_binary_lstm bmi_binary_transformer \
           sleep_efficiency_binary_lstm sleep_efficiency_binary_transformer \
           sex_binary_lstm sex_binary_transformer \
           depression_extreme_binary_lstm \
           osa_binary_apples_postqc_lstm \
           apnea_binary_lstm apnea_binary_transformer \
           cvd_binary_lstm cvd_binary_transformer \
           sleepiness_binary_lstm sleepiness_binary_transformer; do
  python scripts/gen_commands.py infer $exp --split val | bash
done
```

### Step 2 — Run threshold tuning

```bash
source /home/boshra95/sleepfm_env/bin/activate

for exp in bmi_binary_lstm bmi_binary_transformer \
           sleep_efficiency_binary_lstm sleep_efficiency_binary_transformer \
           sex_binary_lstm sex_binary_transformer \
           depression_extreme_binary_lstm \
           osa_binary_apples_postqc_lstm \
           apnea_binary_lstm apnea_binary_transformer \
           cvd_binary_lstm cvd_binary_transformer \
           sleepiness_binary_lstm sleepiness_binary_transformer; do
  python scripts/gen_commands.py threshold-tuning $exp | bash
done
```

Or for a single experiment with check output:
```bash
python scripts/gen_commands.py threshold-tuning bmi_binary_lstm
# Shows command + warns if val parquet is missing
python scripts/gen_commands.py threshold-tuning bmi_binary_lstm | bash
```

### Output

Each experiment gets a new file (existing results untouched):

```
results/phase0_v3/inference/{exp_id}/threshold_tuning.csv
```

Contains both `orig_*` (t=0.5) and `tuned_*` (t_opt) metrics side by side.
`auroc` is included and is identical in both — threshold-free.

### Paper reporting

Use tuned metrics as primary for balanced accuracy and recall in paper tables.
Keep original t=0.5 results available in `threshold_tuning.csv` for supplementary.
Include a single footnote: *"Balanced accuracy at t∗ selected on validation set
(Youden's Index); AUROC is unaffected."*
