# Supervisor Meeting Notes
*Phase 0 Progress Update — April 2026*

---

## Overview: Four Research Ideas

My work focuses on two core problems in PSG-based sleep modeling, and I have four research ideas (two per problem) to address them.

### Problem A: Context Usage Uncertainty
> *How much temporal context does a sleep model actually need, and can we learn that automatically?*

The optimal amount of past PSG signal is **task-dependent** — staging a single 30-second epoch may need only local context, while predicting AHI severity may benefit from longer overnight structure. Currently this is a manual hyperparameter, which is brittle and doesn't generalize.

- **Idea 1 — AMTA + CSL (Adaptive Multi-Timescale Adapter + Context Sufficiency Learning)**:
  Build a lightweight long-context module with three temporal pathways (short/medium/long), each contributing to a final prediction via a learned mixing gate. Train it simultaneously on short and long views of the same sample, using a *consistency loss* that pushes the short-context prediction to match the long-context one — but only when the long context actually helps (gated by confidence or loss improvement). This turns context-length sensitivity from a design choice into a measurable, interpretable property of the model.

- **Idea 2 — Adaptive Early-Exit Length Selection**:
  Train a model with multiple exit heads at different context lengths (e.g., 2m, 5m, 10m, 20m, 40m), plus a small selector network that, at inference time, chooses the *earliest exit within a small margin of the best exit* for each sample. The selector is supervised using oracle labels (derived during training from which exit was closest to optimal) — no reinforcement learning needed. At test time, the selector provides a per-sample length decision without requiring labels.

### Problem B: Distribution Change Robustness
> *How do we maintain performance under montage changes, missing channels, artifacts, and cross-site shifts — without labeled target data?*

- **Idea 3 — Safe Test-Time Adaptation (TTA) with Safety Gates**:
  After training on labeled source data, adapt a small subset of parameters (normalization layers, small adapters) at test time using unlabeled objectives (entropy minimization, augmentation consistency). Three safety gates (entropy, prototype margin, drift rollback) prevent adaptation collapse.

- **Idea 4 — Modality-Aware Mixture-of-Experts (MoE) Routing**:
  Multiple specialist sub-models (experts), each suited to a different modality regime (EEG-dominant, respiration-dominant, missing-EEG, etc.), with a router that selects and mixes experts based on the observed missingness mask.

---

## What I Started With: Phase 0

Before developing any of these methods, I am running **Phase 0** — a systematic validation study to confirm that context length actually matters, how much it matters, and whether the effect is task-specific. This serves two purposes:

1. **Confirms the problem is real** — if all tasks perform the same regardless of context, Ideas 1–2 are unnecessary.
2. **Creates oracle reference curves** — the best-performing context length per task, against which learned methods (Ideas 1–2) can be validated later ("does the model learn to prefer the same length a human would choose?").

---

## Phase 0: Context-Length Sweep — Full Pipeline

### Core Idea

For each task and context length L ∈ {30s, 10m, 40m, 80m, 120m, …}, train an independent lightweight head on top of **frozen SleepFM embeddings**, evaluate on the held-out test set, and compare performance across L.

The frozen encoder means all context-length heads see the same representation — the only variable is how much temporal history is given to the head.

---

### Encoder: SleepFM `model_base`

- **Architecture**: SetTransformer (frozen throughout all Phase 0 experiments)
- **Input**: raw PSG HDF5 files chunked into 5-minute segments, with channels zero-padded/masked if missing
- **Channel groups** (following SleepFM's montage-agnostic design):
  - `BAS` (Brain Activity Signals): EEG + EOG, up to 10 channels
  - `RESP` (Respiratory): Airflow, Thorax, Abdomen, SpO2, HR, Snore, RespRate — up to 7 channels
  - `EKG`: up to 2 channels
  - `EMG`: CHIN, LLEG, RLEG — up to 4 channels
  - Channels are selected by a priority list; if a channel is missing from a recording, its slots are zero-padded with `mask=True`
- **Output used**: `e[1]` — per-5-second-patch embeddings, shape `[T, 4, 128]` per subject
  - T = total 5-sec patches in the recording
  - 4 modality families, each 128-dim
  - At model input time: flattened to `[T, 512]`
- **Storage**: ~5.6 MB per subject (float16); ~22 GB total across ~4000 subjects

Embeddings are **pre-extracted once and cached** as `.npy` files on disk. Training reads slices from memory-mapped files — only the requested context window is loaded per sample.

---

### Datasets

Four NSRR datasets are used:

| Dataset | Subjects (approx) | Notes |
|---------|-------------------|-------|
| **SHHS** (Sleep Heart Health Study) | ~5000 | Large community cohort; 2 visits |
| **MrOS** (Osteoporotic Fractures in Men) | ~3000 | Older men; 2 visits |
| **APPLES** | ~1100 | OSA-focused; rich cognitive + questionnaire data |
| **STAGES** | ~1500 | Multi-site; ISI, GAD-7, PHQ questionnaires |

Not all tasks have labels in all datasets. For example, ISI (insomnia) and GAD-7 (anxiety) were only collected in STAGES. ESS (sleepiness) is available in all four; AHI is available in all four (though with slight definitional differences between AASM 2015 and Chicago 1999 scoring).

**Known data issues handled**:
- 152 subjects have NaN embeddings (1 in APPLES, 151 in STAGES/STLK cohort — likely a signal processing issue). These are automatically excluded via a `nan_blocklist.txt` at dataset load time.
- APPLES: 412 of 1516 subjects have no PSG recording on NSRR (withdrawals, failed recordings). 1103 subjects have usable embeddings.

---

### Tasks

All tasks are formulated as classification problems. Labels come from questionnaires, PSG-derived metrics, or medical history in the dataset CSV files. A unified `master_targets.parquet` is built by running dataset adapters (one per dataset) that extract, harmonise, and merge the relevant columns.

| Task | Type | Datasets | Description | Label definition |
|------|------|----------|-------------|-----------------|
| `sleep_staging` | 5-class | all 4 | Predict sleep stage per 30-sec epoch | Wake / N1 / N2 / N3 / REM |
| `apnea_binary` | binary | all 4 | OSA presence | AHI ≥ 15 |
| `cvd_binary` | binary | shhs, mros, apples | Cardiovascular disease history | any_cvd / cvchd |
| `sleepiness_binary` | binary | all 4 | Excessive daytime sleepiness | ESS ≥ 11 |
| `depression_binary` | binary | apples, stages | Depression | BDI / PHQ threshold |
| `insomnia_binary` | binary | stages only | Insomnia | ISI score threshold |
| `anxiety_binary` | binary | stages only | Anxiety | GAD-7 threshold |
| `rested_morning` | binary | mros | Morning restedness | Single-question item |

**Important caveat**: The same construct (e.g., "sleepiness") may be operationalised differently across datasets — different questionnaire versions, different thresholds, different cohort characteristics. For some tasks (AHI, ESS), the definitions are close enough to merge; for others (depression, insomnia, restedness), merging is inappropriate and each dataset contributes its own subjects.

---

### Context-Window Dataset

For each subject, a context window of N consecutive 5-second patches is sampled:

**seq2seq mode (sleep staging)**:
- Index = (subject, anchor_patch_idx) where anchor = a specific 30-sec epoch
- Input: N patches *ending at* the anchor (causal, past-only window)
- Label: scalar stage of the anchor epoch
- Early-recording anchors where past < N → zero-padded with mask
- This answers: *"does more past context improve staging of this specific epoch?"*

**seq2label mode (all other tasks)**:
- Index = (subject, window_start) — K non-overlapping windows per subject
- K = min(K_max=5, available windows) — **kept constant across all context lengths** for fair comparison
  - If K varied with L, longer contexts would have fewer windows and different gradient counts
- Training: K windows at random positions; Val/Test: K windows at evenly-spaced positions (deterministic)
- Label: same scalar for all windows of a subject (night-level label)

Context lengths supported: any duration string `"Xs"` or `"Xm"` — e.g., `"30s"` (6 patches), `"10m"` (120 patches), `"80m"` (960 patches), `"120m"` (1440 patches). `"full_night"` is a special sentinel (all available patches).

---

### Sequence Head Models

Three lightweight architectures are compared, all operating on `[batch, N, 512]` input:

| Head | Mechanism | Parameters | Notes |
|------|-----------|-----------|-------|
| **MeanPool** | Masked mean of all N patches → Linear | ~1K | Ignores temporal order; strong baseline |
| **LSTM** | 2-layer BiLSTM; last valid hidden → Linear | ~3.2M | Handles variable-length via pack_padded_sequence |
| **Transformer** | CLS token + sinusoidal PE → encoder → Linear | ~530K | Skipped for very long contexts (O(N²) memory) |

All heads are trained from scratch on each task × context-length combination.

---

### Training

- **Optimizer**: AdamW, learning rate 1e-4, cosine scheduler
- **Early stopping**: patience = 10 epochs on `val_auroc` (threshold-independent, robust to class imbalance)
  - *Why AUROC, not val_loss?* Early in training, val loss may decrease while val AUROC continues improving. Using val_loss caused checkpoints to be saved at epoch 3 while the model was still underfitting — AUROC-based monitoring fixes this.
- **Class imbalance**:
  - `class_weights: "auto"` — inverse-frequency weights passed to CrossEntropyLoss, normalised so mean = 1
  - `weighted_sampler: false` — WeightedRandomSampler was tested but caused recall collapse (the model predicted only the minority class) at the 2–3:1 imbalance ratios present in most tasks; disabled in favor of loss weighting only
- **Checkpointing**: `best_model.pt` saved whenever val AUROC improves; `metrics.json` written at end of training; safe to resubmit (already-completed contexts are skipped)
- **Infrastructure**: SLURM jobs on Compute Canada `fir` cluster, H100 MIG GPU (10 GB slice); W&B logging

---

### Evaluation Pipeline

After training, evaluation is a three-step pipeline:

#### Step 5a — All-windows inference

Instead of the K=5 training windows, inference runs on **all available non-overlapping windows** per subject (K_max overridden to 99,999). This saves a parquet of per-window predictions:

```
subject_id | dataset | window_idx | true_label | pred_label | prob_class0 | prob_class1
```

This is done once (GPU job, ~5–30 min depending on context length), then all downstream aggregation is done on CPU from the saved parquet — no GPU needed for analysis.

#### Step 5b — Subject-level aggregation

Groups windows per subject and computes two aggregation strategies:

- **Mean-probability (mean_prob)**: average softmax probabilities across all windows → argmax → predicted label. Soft aggregation; preserves confidence.
- **Majority vote**: mode of per-window hard predictions. Hard aggregation; more robust to outlier windows.

**Why these two?** A single 30-second window may be unrepresentative (artifact, anomaly). Aggregating across the full night produces a subject-level prediction that better reflects the overall physiology. Mean-prob is used for AUROC (requires soft scores); majority vote provides an alternative hard-decision metric. Comparing them reveals whether uncertain windows are a problem.

#### Step 5c — K-window sweep analysis (`analyze_windows.py`)

Reads the all-windows parquet and sweeps K ∈ {1, 5, 10, 20, 50, "all"} on CPU:
- For each K: select K windows per subject using a configured strategy (evenly-spaced, first, last, random)
- Compute segment-level, mean-prob, and majority-vote metrics

**Outputs**:
- CSV per (task, head, context, split, K) with all metrics
- Markdown table showing how AUROC / balanced accuracy / macro-F1 change as K increases

This answers: *"how many windows per subject are needed before performance saturates?"* — directly relevant to efficient clinical deployment.

---

### Metrics Reported

| Metric | Used for |
|--------|---------|
| AUROC (macro OvR) | Primary — threshold-independent; used for early stopping and comparisons |
| Balanced accuracy | Secondary — mean per-class recall; robust to imbalance |
| Macro F1 | Supporting metric |
| Per-class recall | Important for staging (N1 recall is typically lowest) |
| Cohen's κ | Sleep staging only |

---

## Current Results (Summary)

Early results across tasks and context lengths:

| Task | Head | Best test AUROC | Best context | Status |
|------|------|----------------|-------------|--------|
| Sleep staging | LSTM | **0.939** | 10m | ✓ inference + analysis done |
| Apnea binary | Transformer (lr=1e-4) | **0.800** | 120m | inference in progress |
| Apnea binary | LSTM (lr=1e-4) | 0.771 | 120m | partial inference |
| CVD binary | LSTM | ~0.665 | 120m | inference in progress |
| Depression binary | LSTM | ~0.643 | 120m | inference done (4 contexts) |
| Sleepiness binary | LSTM | ~0.611 | varies | training done |
| Insomnia binary | LSTM | ~0.583 | 40m | inference pending |
| Rested morning | LSTM | ~0.540 | 10m | inference done (3 contexts) |

**Key observations so far**:

- Sleep staging shows a clear improvement from 30s → 10m (0.927 → 0.939) then plateaus — suggesting that ~10 minutes of past context is sufficient for epoch-level staging.
- Apnea shows a consistent upward trend from 30s → 120m (0.685 → 0.800), with no plateau yet — longer context keeps helping.
- CVD binary is relatively flat across contexts (~0.64–0.67), suggesting either the PSG signal doesn't encode much incremental CVD information with longer context, or the label quality is limiting.
- Weaker tasks (sleepiness, insomnia, rested morning, depression) show limited improvement with context and generally lower peak AUROC — suggesting noisy labels or a mismatch between the PSG signal and the questionnaire-derived label.
- This **task-specific variation is exactly the Phase 0 finding we expected**: different tasks need different amounts of context, and some tasks may not benefit from longer context at all.

---

## Task Definitions: Known Issues and Planned Improvements

The current task definitions are imperfect and may explain some of the weaker results. After completing the current Phase 0 sweep, I plan to revisit several tasks:

### Problems identified

**Sleepiness (ESS)**:
- ESS ≥ 11 is the standard clinical threshold; current results (~0.61 AUROC) are weak. The threshold and/or the cohort mix may be contributing — ESS is validated but still a subjective scale. Considering also trying ESS as a 3-class task.

**Depression**:
- Currently using BDI (APPLES) and PHQ (STAGES) with different scales, merged. Should use **extreme groups only** (e.g., BDI ≤ 9 vs ≥ 20) to remove the ambiguous middle range. No cross-dataset merge — each dataset contributes independently.

**Insomnia**:
- Currently uses ISI from STAGES only. MrOS has ISI at visit 2 (`slisiscr`) but was not originally included. Adding MrOS visit 2 would substantially increase the dataset size for this task.

**Rested morning**:
- Near-random AUROC (~0.52–0.54). This is a single-question subjective item (`RefreshInAMHP` in APPLES, `poxqual3` in MrOS) — likely too noisy a signal. May remove or deprioritize.

**CVD binary**:
- Current labels (`any_cvd`, `cvchd`) are prevalent disease history, not incident events. A survival analysis formulation (time-to-event, using SHHS longitudinal data) would be more appropriate but requires a different head (Cox PH or survival regression).

### Planned new tasks

- **Sleep efficiency (binary)**: objective PSG-derived metric (TST/TIB), available across SHHS, APPLES, MrOS. Much cleaner label than subjective scales.
- **Sleep quality (PSQI)**: gold-standard scale, available in MrOS. PSQI > 5 → poor sleeper.
- **Cognitive performance (APPLES)**: Mean reaction time and lapse count from psychomotor vigilance testing — objective, directly sleep-affected, unique to APPLES.
- **AHI as 4-class**: extend binary apnea to 4-class severity (<5 / 5–15 / 15–30 / ≥30 events/hour).

The general principle: **prefer PSG-derived or validated-scale labels; avoid single-question subjective items; use extreme groups for noisy continuous scales**.

---

## Next Steps: From Phase 0 to Phase 1

Once Phase 0 context-sweep results are complete across all tasks, the plan is:

1. **Summarize the oracle reference curves**: for each task, what is the best context length, and is the curve monotonically increasing, humped, or flat?

2. **Refine task definitions** (as above) and re-run the sweeps for the revised tasks — this will likely improve weak results and validate that the method works when label quality is reasonable.

3. **Begin Idea 1 (AMTA + CSL)**:
   - Use the Phase 0 curves as the ground truth target behavior
   - Train AMTA+CSL on the same tasks — the learned mixing gates and sufficiency curves should align with the manually observed task-specific optima
   - Key validation: does the model learn to weight long-path features more for apnea, and plateau earlier for staging?

4. **Optionally Idea 2 (Early-exit selector)** as a complementary or alternative approach.

The Phase 0 pipeline is already built and running. The infrastructure (frozen embeddings, context-window dataset, multi-head training, subject-level aggregation) transfers directly to Ideas 1–2: the backbone and data pipeline stay the same; only the head architecture changes.

---

*Note: All code is in `NSRR-tools/`. Key scripts: `scripts/train_context_sweep.py`, `scripts/infer_subject_windows.py`, `scripts/analyze_windows.py`. Config: `configs/phase0_config.yaml`.*
