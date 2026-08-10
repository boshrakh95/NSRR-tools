# OSF Implementation Plan — Code-Level, Cluster-Runnable

Detailed step-by-step plan for adding **OSF** (On Pre-training and Scaling of
Sleep Foundation Models, ICML 2026) as the first TSFM baseline compared
against SleepFM. Written for an agent picking this up on the Compute Canada
cluster with `NSRR-tools`, `OSF-Open-Sleep-FM`, and
`npj_digital_medicine_submission` all cloned as sibling directories (see
`CLAUDE.md` → "Cluster Execution Guidance").

**Status: planning only, nothing implemented yet.** This is model #1 of 3;
PhysioOmni and MOMENT will get their own plan docs later, reusing whatever
this pass validates.

**Read before this doc:** `CLAUDE.md` (repo map + code reuse assessment),
`docs/TSFM_BASELINE_CANDIDATES.md` §2.1 (OSF's public research findings),
`docs/EXPERIMENTS_GUIDE.md` (the SleepFM pipeline this mirrors).

---

## 0. Decisions already made (do not re-litigate without asking)

Confirmed with the user on 2026-08-10:

1. **LoRA fine-tuning at long context: attempt full end-to-end first.** If it
   hits GPU memory limits, the fallback ladder (in order) is: (a) enable
   gradient checkpointing, (b) request a larger memory allocation on
   Compute Canada, (c) only if both of those fail, cap the LoRA condition
   at a shorter max context length and report the frozen-only condition at
   the longer ones with the limitation stated explicitly. **Do not skip
   straight to (c)** — try (a) and (b) first.
2. **Comparison baseline is `phase0_v3_full`** (the existing full-channel
   SleepFM run), not `phase0_v3` (paper-primary, fast-channel) — because
   OSF needs channels (snore, full thoracic/abdominal/airflow) that only
   the full-channel config carries. State this explicitly in any results
   writeup: OSF is compared against SleepFM's full-channel numbers, not the
   paper's primary fast-channel headline numbers.
3. **First pass covers Tier 1 tasks only**: `sex_binary`,
   `sleep_efficiency_binary`, `bmi_binary`, `age_class`, `apnea_binary`.
   Sleep staging and Tier 2 tasks (depression, PSQI, etc.) are a later
   pass, after this pipeline is validated end-to-end.
4. **Embedding storage: CLS token + mean-pooled patch tokens**, both
   768-dim, saved per epoch (not the full 90-token patch sequence, and not
   CLS-only). See §3 for the exact shape.

---

## 1. Channel mapping (our full-channel HDF5 → OSF's 12-channel input)

OSF's `ViT` expects exactly 12 channels, in this order (from
`train_config.py` `TRAIN_EDF_COLS_UNI_ENC` in the OSF repo):
`ECG, EMG_Chin, EMG_LLeg, EMG_RLeg, ABD, THX, NP, SN, EOG_E1_A2, EOG_E2_A1,
EEG_C3_A2, EEG_C4_A1`.

Our full-channel config (`configs/modality_groups.yaml` /
`configs/phase0_v3_full_config.yaml`'s `data.channel_priority`) maps onto
this **cleanly, channel-for-channel** — this was the key open question
going in, and it resolves favorably:

| OSF channel | Our channel (full-channel HDF5) | Confidence |
|---|---|---|
| `EEG_C3_A2` | `C3-M2` | High — both are C3 referenced to the contralateral mastoid (M2≡A2) |
| `EEG_C4_A1` | `C4-M1` | High — same, M1≡A1 |
| `EOG_E1_A2` | `LOC` | **Verify on cluster** — need to confirm LOC is stored referenced to A2/M2 in our preprocessing, not to Fpz or left unreferenced. Check `src/nsrr_tools/core/channel_mapper.py` provenance for a couple of subjects, or just compare embedding sanity (no NaN/degenerate output) once extraction runs. |
| `EOG_E2_A1` | `ROC` | Same caveat as above (contralateral) |
| `ECG` | `EKG` | High — single lead, direct match |
| `EMG_Chin` | `CHIN` | High |
| `EMG_LLeg` | `LLEG` | High |
| `EMG_RLeg` | `RLEG` | High |
| `ABD` | `ABD` | High |
| `THX` | `Thor` | High |
| `NP` (nasal pressure/airflow) | `Airflow` | High — closest available proxy; not necessarily the identical sensor modality (nasal pressure vs. thermistor/generic airflow depends on cohort), acceptable approximation |
| `SN` (snore) | `Snore` | **Verify on cluster** — full-channel RESP priority list includes Snore but not every cohort/subject necessarily has a recorded snore channel; check per-cohort availability (Step 0 below) |

**Conclusion: no raw EDF reprocessing needed.** All 12 channels are
already present (or near-equivalent) in our existing full-channel HDF5s at
`/scratch/boshra95/psg_full/{dataset}/derived/hdf5_signals/`. The
remaining work is a resample + reorder + rename + rechunk adapter, not new
preprocessing from raw signal.

**Normalization compatibility:** OSF's own `SleepEpochDataset` applies no
additional per-channel normalization to any of these 12 channels beyond a
final `clamp(-6, 6)` (`NEED_NORM_COL = [HR, SPO2, OX]` in OSF's
`train_config.py` — none of which are in our 12-channel list). This means
OSF expects its 12 input channels to already be roughly zero-mean,
unit-variance (z-scored) before being fed in. Our own preprocessing
(`signal_processor.py`'s `_normalize_signal`) already does per-channel
z-score normalization — **compatible by construction**, but confirm
empirically (§9 sanity checks) rather than assuming.

**Missing channels:** OSF's own demo code zero-fills any channel absent
from the input. Do the same in our extraction script — don't skip subjects
just because one of the 12 channels is missing, unless a dataset is
missing so many channels that the input becomes mostly zeros (flag in the
per-dataset channel-availability report from Step 0, decide then whether
to exclude that dataset from the OSF comparison specifically).

---

## 2. Preprocessing decision (confirmed)

**Reuse existing full-channel HDF5s. Do not reprocess raw EDFs.** Sampling
rate and epoch length are the only real mismatches, and both are handled
in the extraction script (§3), not by re-running the EDF→HDF5 pipeline:

| | Our full-channel HDF5 | OSF's expected input |
|---|---|---|
| Sampling rate | 128 Hz | 64 Hz |
| Chunking | 5-second patches (SleepFM convention) | 30-second epochs |
| Channel set | Harmonized names, already referenced/z-scored | 12 specific channels, same referencing convention |

Resampling 128→64 Hz: OSF's own `_resample_df` does linear interpolation
to the target rate; replicate this exactly (or use `scipy.signal.resample`
/ simple 2:1 decimation with an anti-alias filter — linear interpolation
is what OSF's authors used on their own pretraining data, so matching it
exactly is the lower-risk choice for staying in-distribution).

---

## 3. Stage 1 (frozen encoder) — components

### 3.1 New script: `scripts/extract_osf_embeddings.py`

Mirror `scripts/extract_sleepfm_embeddings.py`'s structure (SIGTERM
handling, sys.path insertion for the sibling model repo, GPU batching,
`--datasets`/`--limit`/`--no-skip` flags) with these changes:

- **Model import**: `sys.path.insert(0, "../OSF-Open-Sleep-FM")` (sibling
  repo, same pattern as the existing script's
  `sys.path.insert(..., "../sleepfm-clinical")`). Import the `ViT` class
  from `osf.backbone.vit1d_cls` and its `vit_base(...)` factory.
- **Checkpoint loading**: download `yang-ai-lab/OSF-Base` from HuggingFace
  first (§5) — do this once, not per-job. Verify the actual checkpoint
  filename after download (README says `osf_backbone.pth`, `demo.ipynb`
  says `dino_vit_base_backbone.pth` — these disagree, confirm which is
  real before hardcoding a path).
- **Per-subject processing loop**:
  1. Load the 12 mapped channels (§1) from the full-channel HDF5 at 128 Hz.
  2. Resample each channel to 64 Hz.
  3. Zero-fill any missing channel (log which channels were zero-filled,
     per subject, for the Step 0 availability report).
  4. Chunk into contiguous, non-overlapping 30-second (1920-sample)
     epochs; drop any incomplete trailing epoch (same convention as
     SleepFM's incomplete-chunk handling).
  5. Batch epochs through `ViT.forward_encoding(x, return_sequence=True)`
     → `cls: [B, 768]`, `patches: [B, 90, 768]`.
  6. Mean-pool `patches` over the 90-token axis → `[B, 768]`.
  7. Stack `[cls, mean_pooled_patches]` → `[B, 2, 768]` per epoch.
- **Output**: `{output_dir}/{dataset}/{subject_id}.npy`, dtype float16,
  shape `[T_epochs, 2, 768]` (`T_epochs` = number of complete 30s epochs in
  the recording). Output dir:
  `/scratch/boshra95/psg_full/unified/embeddings/osf_30sec/` (naming
  convention mirrors the existing `sleepfm_5sec/`).
- **GPU batching**: batch across epochs (not subjects), same pattern as
  the SleepFM script's chunk batching, for GPU utilization.

### 3.2 New dataset class: `src/nsrr_tools/datasets/osf_context_window_dataset.py`

Per the Code Reuse Assessment in `CLAUDE.md`, `ContextWindowDataset` is
SleepFM-shape-hardcoded and not reusable unmodified. Create
`OSFContextWindowDataset` as a **parallel class**, copied from
`ContextWindowDataset` with these constants changed:

```python
N_SUBTOKENS = 2       # was N_MODALITIES = 4  (index 0 = CLS, index 1 = mean-pooled patches)
EMBED_DIM   = 768     # was 128
FLAT_DIM    = 1536    # was 512  (= N_SUBTOKENS * EMBED_DIM)
PATCH_SECONDS = 30    # was 5
```

**Context-length → epoch-count mapping (recompute, do not reuse
`parse_context_length`'s 5-second-patch arithmetic as-is):**

| Context | SleepFM (5s patches) | OSF (30s epochs) |
|---|---|---|
| 30s | 6 | **1** |
| 10m | 120 | **20** |
| 40m | 480 | **80** |
| 80m | 960 | **160** |
| 120m | 1440 | **240** |
| 240m | 2880 | **480** |

**Cohort consistency filter — recompute `min_recording_patches`.** The
existing filter (`dataset.min_recording_patches: 2880` in
`phase0_v3_full_config.yaml`) is calibrated to 5-second patches (240m ×
60s/m ÷ 5s = 2880). For OSF's 30-second epochs the equivalent value is
**480** (240m × 60s/m ÷ 30s = 480), not 2880. Using 2880 unmodified against
epoch counts would incorrectly exclude almost every subject. **This is
the single easiest place to introduce a silent bug — double check it.**

There is no `zero_modality_indices` equivalent needed — OSF has no
4-modality-group structure to ablate (drop this parameter/feature from the
forked class entirely rather than keeping dead code).

Everything else (K-sampling logic — overlapping train/val/test vs.
non-overlapping inference at K_max>100, `SubjectGroupedSampler`, seq2label
window building, padding/collate) is shape-parametric once the above
constants are updated — copy as-is.

### 3.3 New training/inference scripts

Per the Code Reuse Assessment, `train_context_sweep.py` and
`infer_subject_windows.py` are mostly backbone-agnostic (they delegate
embedding I/O entirely to the dataset class and only import
`build_head`/`ContextWindowDataset`). Fork them as:

- `scripts/train_osf_context_sweep.py` — same as `train_context_sweep.py`
  except: import `OSFContextWindowDataset` instead of
  `ContextWindowDataset`; drop the `--zero-modalities` CLI flag and
  `_MODALITY_INDICES` dict (not applicable, no modality groups); otherwise
  identical (checkpoint/resume, early stopping, overfit-phase, snapshots,
  bootstrap-CI-adjacent flags all carry over unchanged since they're
  head/optimizer-level, not backbone-level).
- `scripts/infer_osf_subject_windows.py` — same relationship to
  `infer_subject_windows.py`.

**Model architecture: keep `hidden_dim=128, num_layers=1` (matching
`phase0_v3_full_config.yaml`'s seq2label head), only `input_dim` changes**
(1536 instead of 512). This preserves the existing paper's "architecture
held constant, only the encoder/channels change" comparison principle —
do not tune the head size differently for OSF without a specific reason.

### 3.4 New config: `configs/phase0_osf_config.yaml`

Copy `configs/phase0_v3_full_config.yaml` as the template. Changes:

```yaml
embedding:
  checkpoint_dir: "<path to downloaded OSF-Base checkpoint>"
  output_dir: "/scratch/boshra95/psg_full/unified/embeddings/osf_30sec"
  # chunk_batch_size: tune empirically — start at 16, matching SleepFM's default

data:
  hdf5_dir: "/scratch/boshra95/psg_full"     # same source HDF5s as SleepFM full-channel
  sampling_freq: 64                            # OSF's expected rate, not 128
  epoch_seconds: 30                            # was chunk_seconds: 300 / patch_size: 640
  channel_order: [ECG, EMG_Chin, EMG_LLeg, EMG_RLeg, ABD, THX, NP, SN,
                   EOG_E1_A2, EOG_E2_A1, EEG_C3_A2, EEG_C4_A1]
  channel_mapping:                              # our name -> OSF name, from §1
    C3-M2: EEG_C3_A2
    C4-M1: EEG_C4_A1
    LOC: EOG_E1_A2
    ROC: EOG_E2_A1
    EKG: ECG
    CHIN: EMG_Chin
    LLEG: EMG_LLeg
    RLEG: EMG_RLeg
    ABD: ABD
    Thor: THX
    Airflow: NP
    Snore: SN

dataset:
  embedding_dir: "/scratch/boshra95/psg_full/unified/embeddings/osf_30sec"
  label_source: "/scratch/boshra95/psg/unified/targets_v2/master_targets.parquet"   # SAME as SleepFM — reuse, do not regenerate
  task_subject_dir: "/scratch/boshra95/psg/unified/targets_v2/task_subjects"         # SAME as SleepFM
  context_lengths: ["30s", "10m", "40m", "80m", "120m", "240m"]
  datasets: [apples, shhs, mros, stages]
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
  split_seed: 42          # SAME seed as SleepFM runs — required for a fair comparison on identical splits
  windows_per_subject: 5
  min_recording_patches: 480    # NOTE: 480, not 2880 — see §3.2 cohort-filter recompute

model:
  input_dim: 1536     # 2 * 768, not 512
  head_type: "lstm"   # sweep lstm/transformer/mean_pool same as SleepFM
  hidden_dim: 128
  num_layers: 1
  num_heads: 8
  dropout: 0.3
  num_classes: 2       # per-task, same as existing registries

training:
  epochs: 40
  lr: 1.0e-4
  weight_decay: 1.0e-3
  optimizer: "adamw"
  scheduler: "cosine"
  early_stopping_patience: 10
  device: "cuda"
```

**Label/split reuse is a hard requirement, not just an optimization**: OSF
must be trained/evaluated on the **exact same subjects and the exact same
train/val/test split** as the SleepFM `phase0_v3_full` runs for the
comparison to mean anything — hence pointing `label_source`,
`task_subject_dir`, and `split_seed` at the identical existing files
rather than regenerating them.

### 3.5 New registry + command generation

Per the Code Reuse Assessment, `gen_commands.py` has no backbone hook and
retrofitting it is higher-risk than a parallel generator. For this first
pass (5 tasks × 3 heads × 6 contexts = 90 training runs, manageable),
create:

- `experiments/v2_osf_registry.yaml` — same schema as `v2_registry.yaml`,
  restricted to the 5 Tier-1 tasks × 3 heads, pointing at
  `configs/phase0_osf_config.yaml`, `results_dir:
  /scratch/boshra95/psg_full/unified/results/phase0_osf`,
  `logs_dir: /home/boshra95/NSRR-tools/logs_osf`.
- `scripts/gen_commands_osf.py` — fork of `gen_commands.py` trimmed to
  the subcommands actually needed for this pass (`train`, `infer`,
  `analyze`, `status`, `runs`, `collect` — the plotting/table
  subcommands can wait, or better: once results land in the same
  `metrics.json`/`summary.csv`/parquet schema, try pointing the *existing*
  `analyze`/`collect`/plotting code at the new results dir directly rather
  than forking those too — they're plausibly reusable as-is per the Code
  Reuse Assessment). New wall-time lookup tables (`_TRAIN_HOURS`,
  `_INFER_HOURS_PER_CTX`) will need fresh calibration — OSF's per-epoch
  ViT forward pass has a different cost profile than SleepFM's frozen
  512-dim embeddings, so do **not** copy SleepFM's wall-time numbers
  as a starting assumption. Start with generous `--time` estimates for the
  first few runs, then tighten the table once actual wall-clock times are
  observed (mirrors how the original SleepFM table was calibrated —
  see `docs/EXPERIMENTS_GUIDE.md` §"Expected Runtimes").

### 3.6 New job scripts

`jobs/extract_osf_embeddings_gpu.sh`, `jobs/train_osf_context_sweep_gpu.sh`,
`jobs/infer_osf_subject_windows_gpu.sh` — copy the existing
`*_gpu.sh`/`*_gpu_rorqual.sh` pairs (same SLURM account, GPU slice,
`--signal=B:USR1@120` + `--requeue` auto-resume pattern), pointing at the
new Python scripts and a **new virtual environment** (§4) instead of
`sleepfm_env`.

---

## 4. Environment

**Do not reuse `sleepfm_env`.** OSF's `requirements.txt` pins
`torch==2.5.1`, `transformers==4.47.0`, `peft==0.14.0`,
`pytorch-lightning==2.4.0`, `timm==1.0.12`, plus several git-based
dependencies — risking a version conflict with whatever `sleepfm_env` has
pinned for the existing pipeline. Create a fresh environment:

```bash
python3.10 -m venv /home/boshra95/osf_env
source /home/boshra95/osf_env/bin/activate
pip install -r /home/boshra95/OSF-Open-Sleep-FM/requirements.txt
# Also install our own package in editable mode so scripts/ can import src/nsrr_tools:
pip install -e /home/boshra95/NSRR-tools
```

Note: `environment.yml` in the OSF repo is an **aarch64 (ARM) conda lock
file** — not usable on a Compute Canada x86_64 cluster. Use
`requirements.txt` (pip, architecture-independent) instead, as above.

Some of OSF's declared dependencies (`albumentations`, `efficientnet_pytorch`,
`segmentation_models_pytorch`) look unrelated to the sleep encoder itself
(likely shared from a broader team requirements file) — install the full
list first to avoid import-time failures from unexpected internal imports,
and only trim later if there's a real reason to (e.g. a genuine install
conflict).

---

## 5. Checkpoint download

```bash
source /home/boshra95/osf_env/bin/activate
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='yang-ai-lab/OSF-Base',
                   local_dir='/home/boshra95/OSF-Open-Sleep-FM/pretrained_weights')
"
ls /home/boshra95/OSF-Open-Sleep-FM/pretrained_weights/
# Confirm the actual filename here (osf_backbone.pth vs dino_vit_base_backbone.pth —
# README and demo.ipynb disagree, see §3.1) and hardcode whichever is real into
# configs/phase0_osf_config.yaml's embedding.checkpoint_dir / the extraction script.
```

---

## 6. Stage 2 (LoRA fine-tuning) — components

This is architecturally different from Stage 1, not just a flag flip:
Stage 1 precomputes embeddings once and reuses them across all training
runs; Stage 2 needs the OSF encoder inside the trainable graph, so
embeddings can no longer be precomputed — raw signal has to be loaded and
encoded on the fly, every training step.

### 6.1 New script: `scripts/train_osf_lora.py`

A genuinely new end-to-end training script, not a fork of
`train_context_sweep.py`. Structure:

- **New raw-epoch dataset** (e.g. `OSFRawEpochWindowDataset`): same
  windowing/K-sampling logic as `OSFContextWindowDataset` (§3.2), but
  `__getitem__` returns the raw `[N_epochs, 12, 1920]` signal tensor for
  the window (built from the full-channel HDF5 via the §1 channel mapping
  + resample, same as the extraction script) instead of a precomputed
  embedding array.
- **Combined model**: wrap OSF's `ViT` with LoRA
  (`peft.get_peft_model(vit, LoraConfig(target_modules=["to_qkv", "to_out.0"], r=..., lora_alpha=...))`
  — `to_qkv` and `to_out.0` are the actual Linear-layer attribute names in
  OSF's `Attention` class, `osf/backbone/vit1d_cls.py`, confirmed by
  reading the source directly), then wrap `(lora_vit, sequence_head)`
  together in one `nn.Module` so a single optimizer and a single
  `modules_to_save`-style mechanism cover both. Concretely: build the
  combined module first, *then* call `get_peft_model` on the whole thing
  with `modules_to_save=["sequence_head"]`, rather than LoRA-wrapping the
  ViT in isolation and bolting the head on after — this way PEFT's
  save/load and gradient-freezing logic treats the head as a first-class
  fully-trainable submodule, matching the `modules_to_save=["classifier"]`
  pattern already documented in `docs/TSFM_BASELINE_CANDIDATES.md` §6.
- **Warm start from Stage 1**: load the Stage-1-trained sequence head's
  weights into the combined module's head submodule before starting Stage
  2 training (per the staged LP-FT procedure in `CLAUDE.md` → "Frozen vs.
  LoRA-fine-tuned conditions") — don't start LoRA fine-tuning from a
  randomly-initialized head.
- **Forward pass per training step**: for each window, run all `N_epochs`
  raw epochs through the LoRA-adapted ViT (batched across epochs, same
  batching pattern as the extraction script) to get `[N_epochs, 2, 768]`
  embeddings, flatten to `[N_epochs, 1536]`, feed through the sequence
  head, compute loss, backprop through **both** the LoRA adapters and the
  head.
- **Checkpoint/resume**: replicate `train_context_sweep.py`'s
  `resume.pt`/`best_model.pt` pattern (save combined-module + optimizer +
  scheduler state every epoch; SIGUSR1/timeout handling via the same job
  script pattern) — don't skip this, LoRA runs at long context will be the
  slowest, most timeout-prone jobs in the whole project.

### 6.2 Memory mitigation ladder (apply in this order — per §0 decision)

1. **Gradient checkpointing** (`torch.utils.checkpoint`) through the ViT's
   transformer blocks — standard, should be the first thing tried, cheap
   to implement, no change to results.
2. **Request a larger GPU memory allocation** on Compute Canada (bigger
   MIG slice or full H100) if checkpointing alone isn't enough.
3. **Only if both of the above are insufficient**: cap the LoRA condition
   at the longest context length that fits (e.g. stop at 80m or 120m
   instead of 240m), keep the frozen-embedding condition (Stage 1) at all
   6 context lengths as usual, and state the compute ceiling explicitly in
   any results table/caption — do not silently omit the long-context LoRA
   points.

### 6.3 Wall-time / compute budget — unknown, do not assume

Unlike Stage 1 (which reuses the well-calibrated SleepFM-style
checkpoint/resume/wall-time infrastructure), Stage 2's per-step cost is
new and uncalibrated: it does a full ViT forward+backward pass per epoch
in the window, at every training step, versus Stage 1's cheap
precomputed-embedding lookup. **Run a short pilot (a handful of epochs at
the smallest context length, e.g. 30s or 10m) before submitting the full
sweep**, to get real wall-clock numbers and set realistic `--time` values
— do not extrapolate from SleepFM's training-time table.

---

## 7. Task/label reuse recap

No new target extraction is needed. `master_targets.parquet` and
`task_subjects/*.csv` under `/scratch/boshra95/psg/unified/targets_v2/`
already exist from the SleepFM pipeline and cover all 7 tasks (including
the 5 Tier-1 tasks used here) — point the new config at them directly
(§3.4). The only new label-adjacent work would be if a future pass needs a
task OSF can't fairly support (not the case for any of the 5 Tier-1 tasks
— the apnea/no-respiratory-pathway problem is specific to PhysioOmni, not
OSF, which does have a respiratory channel).

---

## 8. Honest reporting reminders (recap from `CLAUDE.md`, OSF-specific)

- **SHHS is confirmed in OSF's pretraining set** — any SHHS-inclusive
  AUROC comparison against SleepFM needs an explicit contamination
  caveat. **STAGES is very likely also in pretraining** (numeric-ID
  pattern match) — confirm by cross-checking a handful of numeric IDs
  from OSF's `osf/splits/patient_pretrain_*.csv` against our local STAGES
  `subject_code` list before treating as certain (this is a 10-minute
  cluster task, do it early). **MrOS is downstream/eval-only in OSF's own
  splits** (lower risk). **APPLES is clean.**
- Report APPLES (and, once confirmed, STAGES-excluded) results as the
  primary honest comparison; report the full 4-cohort numbers too but with
  the contamination caveat stated alongside, not buried in a footnote.
- If a run doesn't complete (frozen or LoRA, any context length), say so
  explicitly in the results table rather than leaving a blank/implied-zero
  cell.

---

## 9. Step 0 — verification checklist (run these before the full sweep)

1. **Channel availability per cohort**: for a handful of subjects per
   dataset, check which of the 12 mapped channels are actually present in
   the full-channel HDF5 (`h5py` keys) vs. need zero-filling. Flag any
   cohort where Snore or the EOG channels are missing for most subjects.
2. **EOG referencing check**: confirm `LOC`/`ROC` in our HDF5s are
   referenced the way OSF expects (contralateral mastoid) — check
   `channel_mapper.py`/original NSRR channel-label provenance, or just
   inspect embedding output sanity (no NaN, no degenerate all-zero CLS
   vectors) once extraction runs on a small pilot batch.
3. **Checkpoint filename resolution** (§3.1/§5) — confirm which of the two
   candidate filenames is real after download.
4. **Small-scale pilot**: run `extract_osf_embeddings.py --limit 5` on one
   dataset, inspect the output `.npy` shape/values, then a tiny Stage-1
   training run (`--context 30s`, one task, one head) end-to-end before
   submitting the full 90-run sweep.
5. **STAGES-in-pretraining confirmation** (§8) — cross-check numeric IDs.
6. **Cohort filter unit check** (§3.2) — confirm `min_recording_patches:
   480` is actually being applied in epoch units, not silently
   misinterpreted as 5-second-patch units anywhere downstream.

---

## 10. Suggested execution order

1. Set up `osf_env` (§4), download the checkpoint (§5).
2. Run Step 0 verification checklist (§9) — do not skip this, several
   items here are cheap and catch expensive mistakes early.
3. Implement `extract_osf_embeddings.py` (§3.1), run on a small pilot,
   then the full extraction for all 4 datasets.
4. Implement `OSFContextWindowDataset` (§3.2), `phase0_osf_config.yaml`
   (§3.4), forked train/infer scripts (§3.3), registry + generator (§3.5),
   job scripts (§3.6).
5. Run Stage 1 (frozen) for all 5 Tier-1 tasks × 3 heads × 6 contexts = 90
   training runs, then inference, then analysis (reuse existing
   `analyze_windows.py`/`collect_results_v2.py` pointed at the new results
   dir if the schema lines up, per the Code Reuse Assessment).
6. Implement `train_osf_lora.py` (§6), pilot at short context first (§6.3),
   then run Stage 2 (LoRA) across context lengths per the memory
   mitigation ladder (§6.2).
7. Compile results against `phase0_v3_full` (§0.2), applying the
   contamination caveats (§8) honestly.
8. Report back before starting PhysioOmni or MOMENT — do not start the
   next model's plan unprompted.

---

## 11. Open items not fully resolved (flagged, not blocking)

- Exact checkpoint filename (osf_backbone.pth vs. dino_vit_base_backbone.pth).
- Whether LOC/ROC referencing exactly matches OSF's E1-A2/E2-A1 convention.
- Per-cohort Snore/EOG channel availability in the full-channel HDF5s.
- Real wall-clock cost of Stage 2 (LoRA) training — no calibrated estimate
  exists yet, unlike Stage 1 which can reuse SleepFM-style estimation
  logic once its own pilot numbers are in.
- Whether `analyze`/`collect`/plotting code can be pointed at the new
  results directory unmodified, or needs its own fork — plausible per the
  Code Reuse Assessment but not verified against real OSF results yet.
