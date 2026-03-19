# Phase 0 Implementation Notes

## Overview

Phase 0 quantifies how PSG-based classification performance changes as a function
of context length (number of consecutive SleepFM embeddings fed to a sequence head).
All experiments share a single frozen encoder; only the lightweight sequence head
and context length vary.

---

## Files

| File | Purpose |
|---|---|
| `configs/phase0_config.yaml` | Master config — all paths, hyperparams, channel priority |
| `scripts/extract_sleepfm_embeddings.py` | Step 1 — extract embeddings from HDF5 |
| `src/nsrr_tools/datasets/context_window_dataset.py` | Step 2 — PyTorch dataset for context windows |
| `src/nsrr_tools/models/sequence_head.py` | Step 3 — LSTM/Transformer/MeanPool heads |
| `scripts/train_context_sweep.py` | Step 4 — training loop over all context lengths |

---

## Step 1 — Embedding Extraction (`extract_sleepfm_embeddings.py`)

### What it does

Loads the frozen `model_base` SetTransformer and produces one `.npy` file per
subject. The full-night recording is processed in 5-minute chunks; per-patch
embeddings (`e[1]`) are extracted for all 4 SleepFM modalities and stacked.

### Output

```
{output_dir}/{dataset}/{subject_id}.npy
dtype : float16
shape : [T, 4, 128]
  T   = floor(n_signal_samples / 640)  — total 5-second patches
  4   = modalities: BAS, RESP, EKG, EMG  (always in this order)
  128 = SleepFM embed_dim
```

Storage estimate: ~8-hour MrOS recording → ~6 360 patches → 6360 × 4 × 128 × 2 bytes ≈ 6.5 MB per subject.
All ~4 000 subjects ≈ 26 GB total.

### SleepFM model interface

```
model_base SetTransformer:
  input  x    : (B, C_max, 38400)   B=chunks, C_max=modality max channels, T=38400
  input  mask : (B, C_max)           True = padded channel slot
  output e[0] : (B, 128)             pooled 5-min embedding  [not used here]
  output e[1] : (B, 60, 128)         per-5-sec-patch embeddings  [saved]
```

- `C_max` per modality: BAS=10, RESP=7, EKG=2, EMG=4 (from `model_base/config.json`)
- `patch_size=640` → 640 samples / 128 Hz = **5 seconds per patch**
- `chunk_size=38400` = 5 min × 60 s × 128 Hz → **60 patches per chunk**

### Channel mapping

Our HDF5 files use standardised channel names (C3-M2, C4-M1, LOC, ROC, EKG,
CHIN, LLEG, Airflow). These are grouped into SleepFM modalities using the
`data.channel_priority` table in `phase0_config.yaml`.  No mapping against
SleepFM's `channel_groups.json` is needed because our names already match.

Channels not present in a given HDF5 file become zero-padded slots in the model
input, indicated by the corresponding mask bit = True.

### Batching strategy

For each subject, all signal data is loaded into RAM once. Then chunks are batched
`chunk_batch_size=16` at a time, calling the model once per modality per batch.
For a typical 8h recording (96 chunks): 6 batches × 4 modalities = 24 GPU calls.

### Resumability

Already-extracted `.npy` files are skipped by default. Use `--no-skip` to force
re-extraction.

### Python environment

Uses `sleepfm_env` (has torch, einops, h5py, loguru). The NSRR-tools `.venv`
does NOT have torch. Launch configs in `launch.json` set `PYTHONPATH` so the
SleepFM repo modules can be imported without installing them as a package.

---

## Step 2 — Context-Window Dataset ✓

`src/nsrr_tools/datasets/context_window_dataset.py`

- Loads `[T, 4, 128]` embedding → flattens modality dim → `[T, 512]` at item-load time
- Parses context length string via `parse_context_length()`: `"10m"` → 120 patches
- Filters subjects to those that have an embedding `.npy` file (warns on missing)
- Train/val/test split: shuffle with `split_seed`, then slice by ratio (70/15/15)
- Window sampling:
  - Train: random start (seeded per sample for reproducibility)
  - Val/Test: center window `start = (T - N) // 2`
- Recordings shorter than N: zero-pad on the right; mask `True` for padded positions
- Two modes:
  - `seq2label`: reads `label` column from subject CSV; returns `(x [N,512], mask [N], label scalar)`
  - `seq2seq`: loads annotation `.npy` (30-sec epochs), expands to 5-sec resolution (repeat×6),
    truncates to `min(T_embedding, T_annotation)`, returns `(x [N,512], mask [N], stages [N])`
- Stage remapping: 5 (original NSRR REM) → 4; padded positions labeled -1

### Smoke-test
```
python scripts/test_context_window_dataset.py \
    --config configs/phase0_config.yaml \
    --task apnea_binary --context 10m --datasets apples
```
Or use the `👽 Phase0 Step2` launch configs in `launch.json`.

---

## Step 3 — Sequence Head ✓

`src/nsrr_tools/models/sequence_head.py`

Input always `[batch, N, 512]`, mask `[batch, N]` bool (True=padded). Output:

| Head | seq2label | seq2seq |
|---|---|---|
| `MeanPoolHead` | masked mean → linear → (B, C) | N/A |
| `LSTMHead` | BiLSTM, last valid state → linear → (B, C) | all states → linear → (B, N, C) |
| `TransformerHead` | CLS token → linear → (B, C) | patch tokens → linear → (B, N, C) |

Key design choices:
- `LSTMHead`: uses `pack_padded_sequence` to skip padding efficiently
- `TransformerHead`: Pre-LN (`norm_first=True`), sinusoidal positional encoding, CLS prepended
- `build_head(cfg)` factory reads `cfg["model"]` and returns the configured head

Config fields used: `head_type`, `task_type`, `input_dim`, `hidden_dim`, `num_layers`,
`num_heads` (transformer only), `dropout`, `num_classes`.

---

## Step 4 — Training Script ✓

`scripts/train_context_sweep.py`

Loops over all context lengths in config; trains head; saves metrics. Skips already-done
context lengths on resubmit. GPU job: `jobs/train_context_sweep_gpu.sh`.

Metrics:
- seq2label: AUROC, balanced accuracy, macro F1, per-class recall
- seq2seq: per-class accuracy, macro F1, Cohen's kappa (ignores padded label=-1)

Output:
```
{results_dir}/{task}_{head_type}/
  context_30s/  best_model.pt  metrics.json
  context_2m/   ...
  context_5m/   ...
  ...
  summary.csv   — one row per context length
```

Key design:
- Early stopping on val loss (patience=5)
- Cosine LR scheduler
- Mixed precision (AMP) on GPU
- `ignore_index=-1` in CrossEntropyLoss for seq2seq padded positions
- `--context 5m 10m` to run specific lengths only
- `--datasets apples` to restrict datasets (for debugging)

---

## Open Questions

- [ ] Multi-visit subjects (SHHS, MrOS): treat each visit as independent subject (current plan)
- [ ] Class imbalance (seq2label): try weighted cross-entropy first, then compare to oversampling
- [ ] Modality ablations (Phase 0B): drop one modality at extraction time or zero out at dataset time?
