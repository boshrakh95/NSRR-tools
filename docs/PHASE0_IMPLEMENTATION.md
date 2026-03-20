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

## Design Decisions (Finalised after discussion)

### Context lengths
```
L ∈ {30s, 2m, 5m, 10m, 20m, 40m, 80m, full_night}
```
`full_night` = all available patches up to anchor t (staging) or the whole night
(seq2label). This tests whether the plateau observed at 80m holds for the full night.

The Transformer head is capped at **80m** (960 patches) due to O(N²) attention memory.
MeanPool and LSTM handle `full_night` without issue (O(N)).

### Train/val/test split strategy
Random subject-level shuffle using `split_seed` (70/15/15).
**Future improvement**: pre-compute stratified splits (by label distribution) and save
to JSON files for full reproducibility, as in the original SleepFM codebase.

### Window sampling — the key methodological decision

The goal of Phase 0 is to produce a **performance-vs-context-length curve** where
every point on the curve is a fair comparison. This requires:

> **Same set of predictions at every context length; only the amount of input context differs.**

#### seq2seq (sleep staging) — anchor-based sampling
- Index = **(subject, anchor_epoch_t)**, one per 30-sec epoch per subject
- `__len__` = total anchor epochs across all subjects (~960 per 8h night)
- This is **fixed regardless of context length L** — fair comparison guaranteed
- Input: L patches ending at anchor t (past-only causal window)
- Label: **scalar** stage of epoch t (not the whole sequence)
- For `full_night`: input = all patches from recording start up to t
- Early-in-night anchors with L > available past: zero-pad the beginning

This directly answers: *"does giving more past context improve prediction of this epoch?"*

#### seq2label (night-level tasks) — K-window sampling
- Index = **(subject, window_k)**, K random non-overlapping windows per subject
- `K = min(K_max, n_available_windows)` where `K_max=5` by default
- For `full_night`: K naturally = 1 (only one window exists); this is accepted
- This keeps training set size **approximately equal** across context lengths:
  - 30s → K=5 per subject (out of ~960 available windows)
  - 80m → K=5 per subject (out of ~5 available windows)
  - full_night → K=1 per subject
- **Fairness note**: full_night gets 5x fewer gradient updates per epoch than shorter
  contexts. For Phase 0 this is acceptable (paper note: train full_night for 5x more
  epochs or equal total steps if needed for fair comparison).
- Val/Test: same K windows but with deterministic positions (evenly spaced)

---

## Step 2 — Context-Window Dataset (redesigned)

`src/nsrr_tools/datasets/context_window_dataset.py`

- Loads `[T, 4, 128]` embedding → flattens modality dim → `[T, 512]` at item-load time
- Parses context length string via `parse_context_length()`: `"10m"` → 120 patches;
  `"full_night"` → special sentinel value -1 (resolved per-subject at load time)
- Filters subjects to those that have an embedding `.npy` file (warns on missing)
- Train/val/test split: shuffle with `split_seed`, then slice by ratio (70/15/15)

**seq2seq mode (anchor-based)**:
- Builds index of `(subject_row, anchor_patch_idx)` pairs during `__init__`
- `anchor_patch_idx` = multiples of 6 (one per 30-sec epoch); excludes epochs with
  unknown labels (-1 after remapping)
- Input: N patches ending at anchor (zero-padded at start if recording too short)
- Mask: True for left-pad positions
- Label: scalar stage of anchor epoch (int64)
- Stage remapping: 5 → 4; unknown/artefact stages skipped as valid anchors

**seq2label mode (K-window)**:
- Builds index of `(subject_row, window_start)` pairs during `__init__`
- Windows are non-overlapping; K = min(K_max, n_available_windows)
- Train: K windows randomly placed (seeded); Val/Test: K windows evenly spaced
- Input: N patches from window_start (zero-padded if recording shorter than N)
- Mask: True for pad positions
- Label: scalar from subject CSV `label` column

Returns: `(x [N,512], mask [N], label scalar)` for both modes.

---

## Step 3 — Sequence Head (to be revised)

`src/nsrr_tools/models/sequence_head.py`

Both task types now return a **scalar label** (seq2seq becomes anchor-predicts-one-epoch).
So all heads output `(B, C)` — no `(B, N, C)` seq2seq mode needed.

Input always `[batch, N, 512]`, mask `[batch, N]` bool (True=padded). Output `(B, C)`.

| Head | Mechanism | full_night support |
|---|---|---|
| `MeanPoolHead` | masked mean of all N patches → linear | yes |
| `LSTMHead` | causal LSTM, last valid hidden state → linear | yes |
| `TransformerHead` | CLS token + sinusoidal PE → transformer → linear | capped at 80m |

For staging (causal task): `LSTMHead` is most appropriate (processes patches left-to-right,
last state sees all context). `TransformerHead` uses full self-attention which technically
leaks future context — fine for Phase 0 sweep but not for final model.

Config fields: `head_type`, `input_dim`, `hidden_dim`, `num_layers`, `num_heads`,
`dropout`, `num_classes`.

---

## Step 4 — Training Script (to be revised)

`scripts/train_context_sweep.py`

Same structure as before; simplified because both task types now produce scalar labels.
Loss: `CrossEntropyLoss()` (no ignore_index needed — anchor selection already excludes
unknown-stage epochs).

Metrics (same for both tasks, since both are now classification with scalar labels):
- AUROC (binary) or macro OvR AUROC (multi-class)
- Balanced accuracy, macro F1, per-class recall
- For staging: additionally Cohen's kappa

---

## Known Data Notes

**APPLES: 1104 HDF5 files out of 1516 subjects in metadata**
- The NSRR APPLES archive contains only 1104 EDFs, not 1516.
- The remaining ~412 subjects appear in clinical metadata but have no recording on NSRR
  (withdrawals, failed recordings, or excluded before distribution).
- Of the 1104 with HDF5: 1103 produced embeddings; APL0419 was skipped (recording too
  short: ~3.2 min < one 5-min chunk).
- For apnea_binary: 1103/1223 eligible subjects have embeddings; the 120-subject warning
  in ContextWindowDataset is expected and correct — not a pipeline bug.

---

## Open Questions

- [ ] Multi-visit subjects (SHHS, MrOS): treat each visit as independent subject (current plan)
- [ ] Class imbalance: weighted cross-entropy vs oversampling
- [ ] Modality ablations (Phase 0B): zero out modality at dataset time (post-embedding)
- [ ] Transformer head for full_night: skip entirely or use efficient attention (linformer/flash)?
- [ ] Coverage sweep for seq2label (start/middle/end/full night region): Phase 0 extension
