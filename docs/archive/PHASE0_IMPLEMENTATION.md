# Phase 0 Implementation Notes

> **Purpose**: Authoritative record of what was built, why, and how. Intended for paper writing,
> supervisor explanation, and future self-reference. `PHASE0_PLAN.md` contains earlier design
> notes and is kept as a historical record only.

---

## Overview

Phase 0 quantifies how PSG-based classification performance changes as a function of **context
length** — the number of consecutive SleepFM 5-second embeddings fed to a lightweight sequence
head. All experiments share a single frozen encoder (`model_base`); only the head architecture
and context length vary.

**Research question**: Does giving the model more temporal context (30 seconds → full night) of
past PSG signal improve classification on downstream tasks, and where does it plateau?

---

## File Map

### Configuration
| File | Purpose |
|---|---|
| `configs/phase0_config.yaml` | Master config — all paths, hyperparams, context lengths |

### Data pipeline
| File | Purpose |
|---|---|
| `scripts/extract_sleepfm_embeddings.py` | Step 1 — extract frozen embeddings from HDF5 PSG |
| `scripts/scan_nan_embeddings.py` | Utility — scan all .npy files for NaN; writes blocklist |
| `src/nsrr_tools/datasets/context_window_dataset.py` | Step 2 — PyTorch dataset for context windows |

### Model
| File | Purpose |
|---|---|
| `src/nsrr_tools/models/sequence_head.py` | Step 3 — LSTM / Transformer / MeanPool heads |

### Training
| File | Purpose |
|---|---|
| `scripts/train_context_sweep.py` | Step 4 — training loop, checkpointing, W&B logging |
| `jobs/train_context_sweep_gpu.sh` | SLURM job script for training (H100 GPU) |

### Evaluation
| File | Purpose |
|---|---|
| `scripts/infer_subject_windows.py` | Step 5a — run inference on all windows per subject |
| `jobs/infer_subject_windows_gpu.sh` | SLURM job script for inference |
| `scripts/compute_subject_metrics.py` | Step 5b — aggregate per-subject predictions → metrics |
| `scripts/analyze_windows.py` | Step 5c — sweep K windows/subject from saved parquets (CPU) |
| `scripts/collect_results.py` | Step 6 — collect all summary.csv → master CSV + RESULTS.md |
| `scripts/plot_context_comparison.py` | Step 6 — plot context-length vs metric curves |

---

## Encoder: SleepFM `model_base`

The frozen encoder is a **SetTransformer** from the SleepFM codebase
(`sleepfm-clinical/sleepfm/checkpoints/model_base`).

- Input per chunk: `(B, C_max, 38400)` where 38400 = 5 min × 60 s × 128 Hz
- `C_max` per modality: BAS=10 (EEG/EOG), RESP=7, EKG=2, EMG=4
- Channels not available in a recording → zero-padded with `mask=True`
- Output used: `e[1]` → shape `(B, 60, 128)` — per-5-sec-patch embeddings (60 patches per 5-min chunk)
- Output NOT used: `e[0]` — pooled 5-min chunk embedding

The three existing SleepFM checkpoints (`model_base`, `model_sleep_staging`, `model_diagnosis`)
all share `model_base` as their encoder; we use only `model_base` and train new heads from scratch.

---

## Step 1 — Embedding Extraction

`scripts/extract_sleepfm_embeddings.py`

For each subject, chunks the full-night HDF5 PSG into 5-minute segments, runs each batch through
the frozen encoder, extracts `e[1]`, and saves the concatenated result:

```
{embedding_dir}/{dataset}/{subject_id}.npy
  dtype : float16
  shape : [T, 4, 128]
    T   = total 5-second patches in the recording
    4   = modalities in order: BAS, RESP, EKG, EMG
  128 = SleepFM embed_dim per modality
```

Storage: an 8-hour MrOS recording → ~5760 patches → 5760 × 4 × 128 × 2 bytes ≈ 5.6 MB per subject.
All ~4000 subjects ≈ 22 GB total.

### Channel mapping

HDF5 keys use our standardised names (C3-M2, EKG, CHIN, Airflow, etc.). These are assigned to
SleepFM modalities by `data.channel_priority` in `phase0_config.yaml`. Channels not present in a
given recording → zero-padded input slots with `mask=True`.

### Resumability

Already-extracted `.npy` files are skipped. Use `--no-skip` to force re-extraction.

### NaN blocklist

`scripts/scan_nan_embeddings.py` scans all `.npy` files using `mmap_mode='r'` (memory-mapped, no
full file load) and parallel workers. Found **152 bad files**:

- `apples/APL1027` — 1 subject with NaN values throughout
- `stages/STLK*` — 151 STLK subjects (root cause under investigation; possibly a signal
  processing issue during extraction)

Result written to `{embedding_dir}/nan_blocklist.txt`. `ContextWindowDataset` reads this blocklist
at startup and silently excludes these subjects. This resolved NaN training loss that persisted
even after fixing AMP (Automatic Mixed Precision).

---

## Step 2 — Context-Window Dataset

`src/nsrr_tools/datasets/context_window_dataset.py`

### Embedding loading

- Loads `[T, 4, 128]` `.npy` → flattens modality dimension → `[T, 512]` at item-load time
- Uses `mmap_mode='r'` — only pages accessed slices are read from disk (efficient for large files)
- `shape_cache.json` in the embedding dir caches `T` per subject (avoids re-opening every file
  during index building)

### Context length parsing

`parse_context_length(s)` converts duration strings to patch counts:

```python
re.fullmatch(r"(\d+(?:\.\d+)?)(s|m)", s)   # accepts ANY "Xs" or "Xm" value
# Examples: "30s"→6, "10m"→120, "80m"→960, "120m"→1440, "2.5m"→30
PATCH_SECONDS = 5   # each SleepFM patch = 5 seconds
```

The config `context_lengths` list is only the default for sweeps; `--context` CLI overrides it
with any value, including custom ones like `120m`.

`"full_night"` is a special sentinel (value = -1) meaning all available patches.

### Train/val/test split

Subjects shuffled with `split_seed=42`, then sliced 70/15/15 by ratio. Split is **subject-level**
(a subject's data appears in only one split). Multi-visit subjects (SHHS visit 1 & 2, MrOS) are
treated as independent entries.

### seq2seq mode (sleep staging — anchor-based)

- Index = `(subject_row, anchor_patch_idx)` — one anchor per 30-second epoch per subject
- `anchor_patch_idx` = multiples of 6 (6 patches × 5 sec = 30 sec)
- Unknown/artefact stage epochs (label = -1 after remapping) excluded from anchors
- Input tensor: N patches ending at anchor (causal, past-only window)
- Left-padding with zeros + `mask=True` for early-recording anchors where past < N
- Label: scalar stage of the anchor epoch only (not the whole sequence)
- Stage remapping: SleepFM uses {0=Wake, 1=N1, 2=N2, 3=N3, 5=REM} → remapped to {0,1,2,3,4}

This formulation answers: *"does more past context improve prediction of this specific epoch?"*
Every context length is evaluated on the same set of anchor epochs — a fair comparison.

### seq2label mode (night-level tasks — K-window sampling)

- Index = `(subject_row, window_start)` — K non-overlapping windows per subject
- `K = min(K_max, n_available_windows)` where `K_max=5` (config: `windows_per_subject`)
- **Train**: K windows placed at random positions (seeded)
- **Val/Test**: K windows placed at evenly-spaced positions (deterministic, reproducible)
- Input tensor: N patches from `window_start` (right-padded if recording too short)
- Label: scalar from subject CSV `label` column (same label for all windows of a subject)

Keeping K=5 across all context lengths ensures `__len__` is approximately equal regardless of L,
making training time and gradient count comparable — a prerequisite for fair context comparison.

### Returns

`(x [N,512], mask [N,bool], label scalar)` — identical interface for both task types.

---

## Step 3 — Sequence Head Models

`src/nsrr_tools/models/sequence_head.py`

Input: `[batch, N, 512]`, mask `[batch, N]` bool (True=padded). Output: `(B, num_classes)`.

| Head | Mechanism | full_night support |
|---|---|---|
| `MeanPoolHead` | Masked mean of all N patch embeddings → Linear(512, C) | Yes |
| `LSTMHead` | 2-layer BiLSTM, last **valid** hidden state → Linear(512, C) | Yes |
| `TransformerHead` | CLS token + sinusoidal PE → Transformer encoder → Linear(512, C) | No (O(N²)) |

**LSTMHead detail**: Uses `pack_padded_sequence` to skip padding. The "last valid hidden state"
is extracted by indexing with the true sequence length — it sees exactly the non-padded context.
`hidden_dim=256` per direction; concatenated forward+backward = 512.

**TransformerHead**: Automatically skipped for `full_night` context (memory would be O(N²) for
N ≈ 5000+ patches). Fine for fixed-length contexts up to 80m.

**Note on seq2seq**: Sleep staging is implemented as anchor-prediction (scalar label per anchor),
so all heads output `(B, C)` — no seq2seq output mode needed. The "causal" nature comes from the
dataset (past-only window), not the head.

---

## Step 4 — Training Pipeline

`scripts/train_context_sweep.py`, launched via `jobs/train_context_sweep_gpu.sh`

### Loop structure

For each context length in the sweep:
1. Build `ContextWindowDataset` for train/val/test
2. Compute class weights from training labels
3. Optionally create `WeightedRandomSampler`
4. Create DataLoaders
5. Instantiate head model
6. Train with early stopping; save best checkpoint
7. Load best checkpoint; evaluate train/val/test; save `metrics.json`
8. Append val+test metrics row to `summary.csv`

Already-completed context lengths (`metrics.json` exists) are skipped → safe to resubmit.

### Class imbalance strategy

Three complementary mechanisms, all configurable in `phase0_config.yaml`:

**`class_weights`** (config: `training.class_weights`):
- `"auto"`: inverse-frequency weights computed from training labels → `w_i = N / (n_classes × n_i)`, normalised so mean = 1. Passed to `CrossEntropyLoss(weight=w)`.
- `null`: uniform loss (for naturally balanced tasks)
- `[w0, w1, ...]`: manual per-class weights

**`weighted_sampler`** (config: `training.weighted_sampler`, default `false`):
- When `true`: `WeightedRandomSampler` resamples training items so each batch is approximately class-balanced. Uses the same per-class weights as the loss. Complements class_weights; can be combined for severe imbalance.
- Automatically disabled for `full_night` (1 window per subject; resampling adds no value).

**`early_stopping_monitor`** (config: `training.early_stopping_monitor`, default `"val_auroc"`):
- Determines which metric selects the best checkpoint and triggers early stopping.
- `"val_auroc"` (recommended): threshold-independent, robust to class imbalance. Higher = better.
- `"val_balanced_accuracy"`: direct average recall across classes. Higher = better.
- `"val_macro_f1"`: macro-averaged F1. Higher = better.
- `"val_loss""`: original behavior. Lower = better (not recommended for imbalanced tasks — the best-loss epoch is often very early in training, before the model has learned anything useful).

**Why this matters**: Without AUROC-based checkpointing, we observed the best checkpoint being
saved at epoch 3 (lowest val loss) while val AUROC continued improving through epoch 11+. The
model at epoch 3 was barely trained. Switching to `val_auroc` ensures the saved model is actually
the best discriminator.

### Per-epoch logging

Each epoch prints:
```
Epoch   4/30 | loss: train=0.628  val=0.779 | bal_acc: train=0.623  val=0.701 | auroc: val=0.641  best=0.631* | patience=0/10
```
- `bal_acc` = balanced accuracy (not raw accuracy — more informative for imbalanced tasks)
- `auroc: val=...  best=...` = current val monitor value and running best; `*` = checkpoint updated

### W&B integration

Each context-length run is one W&B run: `{task}_{head}_{context}` in group `{task}_{head}`.
Per-epoch: `train/loss`, `val/loss`, `train/bal_acc`, `val/bal_acc`, `val/{monitor_metric}`.
Final summary: all train/val/test metrics on the best checkpoint.

### Output structure

```
{results_dir}/{task}_{head_type}/
  context_{L}/
    best_model.pt     — checkpoint with best val monitor metric
    metrics.json      — train/val/test full metrics + early_stopping_monitor used
  summary.csv         — one row per context (val + test metrics, appended)
```

`metrics.json` structure:
```json
{
  "context_length": "10m",
  "task": "apnea_binary",
  "early_stopping_monitor": "val_auroc",
  "best_val_monitor": 0.7812,
  "train": {"auroc": ..., "balanced_accuracy": ..., ...},
  "val":   {"auroc": ..., ...},
  "test":  {"auroc": ..., ...}
}
```

### Mixed precision

Disabled by default (`mixed_precision: false`). LSTMs are numerically unstable with float16
(gradient overflow), which caused NaN loss in early runs before the root cause (NaN embeddings)
was also identified. AMP is safe for Transformer and MeanPool heads.

### Cluster notes (fir cluster, Compute Canada)

- GPU request: full H100 80GB (`nvidia_h100_80gb_hbm3:1`). MIG slices (`1g.10gb`) caused silent
  CPU fallback on nodes fc11006 and fc11013 — `torch.cuda.is_available()` returned False despite
  GPU being allocated. Fixed with a fail-fast check at job start.
- Node fc11006 excluded via `--exclude=fc11006` (persistent GPU detection issue).
- No `cuda` environment module exists on fir; CUDA is available natively on GPU nodes.
- W&B API key: use classic 40-char key from wandb.ai/authorize, stored in `~/.wandb_key`.

---

## Step 5 — Evaluation Pipeline

### Step 5a — All-windows inference

`scripts/infer_subject_windows.py`, launched via `jobs/infer_subject_windows_gpu.sh`

Loads a trained checkpoint and runs inference on **all available windows** per subject (overrides
`K_max=99999` in a copy of cfg, so every non-overlapping window position is scored).

Output parquet per split:
```
{results_dir}/inference/{task}_{head}/context_{ctx}/{split}_windows.parquet
  columns: subject_id, dataset, window_idx, true_label, pred_label, prob_class0, prob_class1, ...
```

Subject IDs are recovered post-hoc from `dataset._index` — no changes to `__getitem__` needed.

### Step 5b — Subject-level aggregation

`scripts/compute_subject_metrics.py`

Groups the all-windows parquet by `(subject_id, dataset)` and computes two aggregation methods:

- **mean_prob**: average softmax probabilities across windows → `argmax` for the predicted class.
  Soft aggregation; preserves confidence information.
- **majority_vote**: mode of per-window hard predictions. Hard aggregation; more robust to
  outlier windows.

AUROC is always computed from mean_prob probabilities (requires soft scores, not hard labels).

Saves: `subject_metrics.json` and `{split}_subjects.parquet` alongside the windows parquet.

### Step 5c — K-window sweep analysis

`scripts/analyze_windows.py`

Reads the all-windows parquets (GPU job done once) and sweeps K values [1, 5, 10, 20, 50, "all"]
on CPU (fast). For each K: selects K windows per subject using a configurable strategy
(evenly-spaced, first, last, random), then computes segment-level, mean-prob, and majority-vote
metrics. Produces per-split CSVs and a combined markdown table.

This answers: *"how many windows per subject are needed before performance plateaus?"*

Window selection strategies:
- `evenly-spaced` (default, matches val/test eval during training)
- `first` / `last` — tests whether early or late night windows are more informative
- `random` — control for position effects

---

## Step 6 — Results Collection

`scripts/collect_results.py`

Scans all `summary.csv` files under `results_dir` and produces:
- `master_results.csv` — flat table of all runs
- `RESULTS.md` — markdown tables grouped by task → head → context length, showing train/val/test
  metrics for all splits. Subject-level section added if inference has been run.

`scripts/plot_context_comparison.py`

Plots context-length vs. metric curves, with three lines per panel:
- Segment K=5 (dashed) — matches training evaluation
- Subject mean-prob (solid)
- Subject majority-vote (dotted)

---

## Tasks

| Task | Type | Datasets with real labels | Class ratio (approx) |
|---|---|---|---|
| `sleep_staging` | seq2seq, 5-class | apples, shhs, mros, stages | Imbalanced (N1 rare) |
| `apnea_binary` | seq2label | apples, shhs, mros, stages | ~1:1 (threshold-defined) |
| `apnea_class` | seq2label, 4-class | apples, shhs, mros, stages | Imbalanced |
| `cvd_binary` | seq2label | apples, shhs, mros, stages | Imbalanced |
| `sleepiness_binary` | seq2label | apples, shhs, mros, stages | Imbalanced |
| `sleepiness_class` | seq2label, 3-class | apples, shhs, mros, stages | Imbalanced |
| `depression_binary` | seq2label | apples, shhs, mros, stages | Imbalanced |
| `depression_class` | seq2label | apples, shhs, mros, stages | Imbalanced |
| `rested_morning` | seq2label | apples, shhs, mros, stages | Near-balanced |
| `insomnia_binary` | seq2label | **stages only** | ~55:45 |
| `anxiety_binary` | seq2label | **stages only** | ~4.5:1 |

Note: `insomnia_binary` and `anxiety_binary` use stages-only because only the STAGES dataset
collected ISI (Insomnia Severity Index) and GAD-7 questionnaires. All other datasets have `-1`
(missing) for these columns in `master_targets.parquet`.

---

## Metrics

All metrics computed on the **best checkpoint** (selected by `early_stopping_monitor`):

| Metric | All tasks | Note |
|---|---|---|
| `auroc` | ✓ | Macro OvR for multiclass; threshold-independent |
| `balanced_accuracy` | ✓ | Mean per-class recall; robust to imbalance |
| `macro_f1` | ✓ | Unweighted mean F1 across classes |
| `accuracy` | ✓ | Raw accuracy (reported but not primary metric) |
| `recall_classN` | ✓ | Per-class recall (important for staging) |
| `cohen_kappa` | Sleep staging only | Agreement corrected for chance |

---

## Known Data Issues

**APPLES embedding coverage**:
- 1104 HDF5 files available out of 1516 subjects in metadata (~412 subjects have no recording
  on NSRR — withdrawals, failed recordings, or excluded before distribution)
- APL0419 skipped: recording too short (~3.2 min < one 5-min chunk)
- Final: 1103 subjects with embeddings

**NaN embeddings**:
- `apples/APL1027`: all-NaN embedding file. Root cause unknown (possibly corrupt source EDF).
- `stages/STLK*` (151 subjects): all-NaN embeddings. Root cause under investigation — possibly
  related to the STLK cohort's recording equipment or preprocessing pipeline.
- All 152 subjects listed in `{embedding_dir}/nan_blocklist.txt` and excluded at dataset load time.
- Impact: ~8% of stages subjects excluded; apnea/shhs/mros unaffected.

**MROS subject count**:
- Dataset reports slightly different counts between visits; both treated as independent subjects.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Encoder | Frozen `model_base` SetTransformer | Transfer learning; avoids overfitting on small datasets |
| Embedding dim | 512 (4 modalities × 128) | Full multimodal; can zero-out modalities for ablation |
| Patch size | 5 sec (640 samples @ 128 Hz) | Fixed by SleepFM architecture |
| Split strategy | Random 70/15/15 subject-level | Simple; consistent across all tasks |
| Sleep staging formulation | Anchor-prediction (scalar label) | Enables same head/loss for all tasks; causal window |
| seq2label sampling | K=5 non-overlapping windows | Equal `__len__` across context lengths → fair comparison |
| Val/test window placement | Evenly-spaced (deterministic) | Reproducible; covers full recording |
| Early stopping monitor | `val_auroc` (configurable) | Robust to imbalance; selects truly best discriminator |
| Class weights | Auto inverse-frequency | Standard approach; accounts for label distribution |
| Mixed precision | Disabled (LSTM), optional | LSTMs overflow in float16 |
| Multi-visit subjects | Independent entries | Simplest approach; each night is a separate data point |

---

## Open Questions / Future Work

- [ ] **STLK NaN root cause**: investigate why 151 stages/STLK subjects have all-NaN embeddings
- [ ] **Stratified splits**: pre-compute and save split JSONs (current random split may not be balanced by label for small tasks like insomnia/anxiety)
- [ ] **full_night context**: currently works for LSTM/MeanPool; Transformer needs efficient attention (Flash Attention / Linformer) for N>960
- [ ] **Modality ablations (Phase 0B)**: zero-out modality embeddings at dataset time (post-extraction, no GPU needed)
- [ ] **Coverage analysis**: does using early-night vs. late-night windows matter? (analyze_windows.py `--window-strategy first/last`)
- [ ] **Rerun imbalanced tasks with val_auroc checkpointing**: earlier runs (anxiety_binary, insomnia_binary) used val_loss — should be rerun for fair final comparison
- [ ] **K sweep results**: run analyze_windows.py once inference parquets are available to determine optimal K
