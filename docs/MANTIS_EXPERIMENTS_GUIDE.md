# Experiment Execution Guide — Mantis Baseline

This is the Mantis-baseline counterpart to `docs/EXPERIMENTS_GUIDE.md` (the
SleepFM pipeline's execution guide), `docs/OSF_EXPERIMENTS_GUIDE.md` (the
OSF baseline's), and `docs/PHYSIOOMNI_EXPERIMENTS_GUIDE.md` (the
PhysioOmni baseline's). **Being filled in incrementally as the Mantis
pipeline is implemented and run**, per the user's explicit instruction
(2026-09-07) to start this guide once real submittable/runnable steps
exist — see `docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md`'s Implementation
Checklist for current progress; this doc records the concrete
commands/paths/verified outputs for each step once that step is actually
built and tested, so a future session (human or Claude) can run/monitor/
debug the Mantis pipeline without re-deriving anything. Sections below
marked **(not yet implemented)** are placeholders, not real content —
don't treat them as instructions until the corresponding checklist item is
done. **Keep this file updated alongside the implementation plan as jobs
run and things get fixed/changed** — same living-document convention as
the plan doc.

Same operational conventions as `EXPERIMENTS_GUIDE.md`/
`OSF_EXPERIMENTS_GUIDE.md`/`PHYSIOOMNI_EXPERIMENTS_GUIDE.md` throughout:
`gen_commands`-generated commands, the same status-JSONL/log-directory
pattern, the same checkpoint/resume mechanism — different folder names
(`mantis_env`, `logs_mantis`, `phase0_mantis`) but the same shape. Anywhere
Mantis genuinely needs something different from SleepFM/OSF/PhysioOmni (a
parameter, a default, a step, an included/excluded task), it's flagged
explicitly — see `docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md`'s checklist for
the complete reasoning behind each; everything not listed there is
intentionally identical.

**Scope note (unlike SleepFM):** SleepFM is our main model, so it gets the
full protocol — modality/channel ablations, full vs. fast-channel rounds,
sleep staging, all task tiers. Mantis (like OSF and PhysioOmni) is a
*comparison baseline*, not the main model — it gets **frozen-backbone
Stage 1** (this guide's main content), plus a planned **MantisPlus
ablation** (same registry shape, different checkpoint — plan §5.1) and a
possible future **Stage 2 (LoRA)** round, not yet started. No channel
ablation, no full-channel round, no sleep staging (out of scope for all
three TSFM baselines, plan §5.8) — just the 5 Tier-1 seq2label tasks × 3
heads × 6 context lengths, compared against SleepFM's `phase0_v3`
(paper-primary) results.

---

## Table of Contents

1. [Overview](#overview)
2. [Run identity quick-reference (Mantis vs. OSF vs. PhysioOmni vs. SleepFM)](#run-identity-quick-reference-mantis-vs-osf-vs-physioomni-vs-sleepfm)
3. [Step 0 — Environment + Checkpoint](#step-0--environment--checkpoint)
4. [Step 1 — Embedding Extraction](#step-1--embedding-extraction)
5. [Step 2 — Channel Loader + Dataset Smoke Tests](#step-2--channel-loader--dataset-smoke-tests)
6. [Step 3 — Config](#step-3--config)
7. [Step 4 — Windowing/Layer Pilots (Pilots 1+2)](#step-4--windowinglayer-pilots-pilots-12)
8. [Step 5 — Training](#step-5--training)
9. [Step 6 — Inference](#step-6--inference)
10. [Step 7 — Command Generator (`gen_commands_mantis.py`)](#step-7--command-generator-gen_commands_mantispy)
11. [Step 8 — Running the Full Stage 1 Sweep](#step-8--running-the-full-stage-1-sweep)
12. [Checkpoint Resume and Auto-Requeue](#checkpoint-resume-and-auto-requeue)
13. [Job Run History and Tracking](#job-run-history-and-tracking)

---

## Overview

The Mantis pipeline mirrors the SleepFM `phase0_v3` (fast-channel,
paper-primary), OSF, and PhysioOmni pipelines exactly in shape: precompute
frozen encoder embeddings once, then sweep lightweight sequence heads
(LSTM / Transformer / MeanPool) over context lengths on top of those
embeddings. Same five phases per experiment (train → infer → analyze →
iso-compute → saturation curve), same subjects/splits/K-sampling method,
same training hyperparameters — only the encoder and its native input
format differ.

**What's genuinely different from OSF's/PhysioOmni's version of this
pipeline** (see `docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md` for the full
technical reasoning behind each):
- **Mantis is ONE channel-agnostic encoder**, not a fixed 12-channel input
  (OSF) or 4 independent per-modality encoders (PhysioOmni). Every one of
  the 6 canonical channel slots (EEG, EOG_L, EOG_R, ECG, EMG, RESP) goes
  through the SAME frozen weights, batched together as one forward call.
- **Embedding shape**: `[T, 6, 512]` — a real 3-D shape with a channel
  axis, like OSF's `[T, 2, 768]` but 6 "subtokens" instead of 2, and those
  6 subtokens are literally the 6 channel slots (not CLS+mean like OSF's).
  `combined @ last layer` (cls+mean concatenated, full 6-layer depth) —
  empirically confirmed, not just decided (Pilots 1+2, Step 4 below).
- **Task scope**: 5 Tier-1 tasks, same as OSF, **not** PhysioOmni's 4 —
  `apnea_binary` IS included, since Mantis's RESP slot goes through the
  same encoder as every other channel (no missing-modality gap the way
  PhysioOmni has). Makes Mantis the only one of the three baselines
  directly comparable to OSF on apnea.
- **Windowing decision (Option D)**: feeds the model a full 30-second
  epoch at its pretrained shape (`seq_len=3840, num_patches=240`,
  `patch_window_size=16` unchanged from pretraining) rather than chopping
  into sub-windows at the model's native 512-sample shape — chosen for
  cross-model fairness (every baseline sees a full 30s epoch as one unit)
  and empirically confirmed not to matter much either way (gap vs. the
  alternative: -0.0096 weighted F1, well under the 0.15 escape-hatch
  threshold).
- **No autocast/fp16 — TF32 instead.** `torch.backends.cuda.matmul.allow_tf32`
  is enabled at the top of every GPU-touching Mantis script; config sets
  `training.mixed_precision: false` deliberately (plan §4.2).
- **Achieved-TFLOP/s instrumented everywhere compute happens** (extraction,
  training) — a real, measured percentage of the allocated GPU's peak,
  not a computed-once-and-forgotten estimate, per plan §4.1's "the single
  most expensive mistake so far was never checking this" lesson.

See `docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md` for the full technical plan
this guide operationalizes.

---

## Run identity quick-reference (Mantis vs. OSF vs. PhysioOmni vs. SleepFM)

| | SleepFM (`phase0_v3`) | OSF | PhysioOmni | Mantis |
|---|---|---|---|---|
| **Python env** | `/home/boshra95/sleepfm_env` | `/home/boshra95/osf_env` | `/home/boshra95/physioomni_env` | `/home/boshra95/mantis_env` |
| **Checkpoint** | `sleepfm-clinical/sleepfm/checkpoints/model_base` | `OSF-Open-Sleep-FM/pretrained_weights/osf_backbone.pth` | `/home/boshra95/PhysioOmni/checkpoints/PhysioOmni.pt` | `/home/boshra95/mantis_checkpoints/Mantis-8M` (`paris-noah/Mantis-8M`) |
| **Embedding config** | `configs/phase0_v3_config.yaml` | `configs/phase0_osf_config.yaml` | `configs/phase0_physioomni_config.yaml` | `configs/phase0_mantis_config.yaml` |
| **HDF5 root (source)** | `/scratch/boshra95/psg/` (fast-channel) | `/scratch/boshra95/psg_full/` | `/scratch/boshra95/psg/` (fast-channel) | `/scratch/boshra95/psg/` (fast-channel) |
| **Embeddings dir** | `.../unified/embeddings/sleepfm_5sec/` | `.../unified/embeddings/osf_30sec/` | `.../unified/embeddings/physioomni_30sec/` | `.../unified/embeddings/mantis_30sec/` |
| **Embedding shape** | `[T, 4, 128]` (5s patches) | `[T, 2, 768]` (30s epochs) | `[T, 500]` (30s epochs, 2D) | `[T, 6, 512]` (30s epochs — 6 channel slots) |
| **Train config** | `phase0_v3_config.yaml` (hidden=128, layers=1) | `phase0_osf_config.yaml` (hidden=128, layers=1) | `phase0_physioomni_config.yaml` (hidden=128, layers=1) | `phase0_mantis_config.yaml` (hidden=128, layers=1 — same parity decision) |
| **Registry** | `experiments/v2_registry.yaml` (`gen_commands.py`) | `experiments/v2_osf_registry.yaml` (`gen_commands_osf.py`) | `experiments/v2_physioomni_registry.yaml` (`gen_commands_physioomni.py`) | `experiments/v2_mantis_registry.yaml` (`gen_commands_mantis.py`) |
| **Results root** | `.../results/phase0_v3/` | `.../results/phase0_osf/` | `.../results/phase0_physioomni/` | `.../results/phase0_mantis/` |
| **Inference root** | `.../results/phase0_v3/inference/` | `.../results/phase0_osf/inference/` | `.../results/phase0_physioomni/inference/` | `.../results/phase0_mantis/inference/` |
| **Training/job logs** | `logs_v3/` | `logs_osf/` | `logs_physioomni/` | `logs_mantis/` |
| **Labels/splits** | `/scratch/boshra95/psg/unified/targets_v2/` | Same | Same | Same |
| **W&B project** | `nsrr-phase0` | `nsrr-phase0-osf` | `nsrr-phase0-physioomni` | `nsrr-phase0-mantis` (installed status in `mantis_env` not yet confirmed — falls back gracefully with a warning, not a crash, same as OSF's/PhysioOmni's gap) |
| **Task scope** | All Tier 1/2 tasks | Tier 1: sex, sleep efficiency, BMI, age, apnea (5 tasks) | Tier 1 minus apnea (4 tasks) | **Tier 1, same as OSF (5 tasks, apnea included)** |
| **Channel ablation** | Yes (main model) | No | No | No (comparison baseline) |
| **Rounds** | Full protocol | Frozen + LoRA | Frozen + LoRA (code done) | Frozen (this guide) + MantisPlus ablation planned; LoRA not yet started |
| **Status** | **DONE** | Frozen done, LoRA in progress | Frozen done, LoRA code done | **Frozen round: code + smoke tests done, full sweep not yet run (checklist 1.11-1.12)** |

---

## Step 0 — Environment + Checkpoint

**Status: DONE.** See `docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md` checklist
0.1-0.3 for the full build log.

```bash
module load python/3.10.13
source /home/boshra95/mantis_env/bin/activate
```

Checkpoint downloaded to `/home/boshra95/mantis_checkpoints/Mantis-8M`
(`paris-noah/Mantis-8M`, `MantisV1`, 8,112,384 params in the file /
8,037,632 live after dropping the two shape-mismatched buffers,
`scripts/verify_mantis_checkpoint.py`, PASSING). Manual loading (NOT
`.from_pretrained()`, which rebuilds from the repo's own `config.json`
sized for a different `seq_len`/`num_patches`) via
`safetensors.torch.load_file` + `strict=False`, dropping `pos_encoder.pe`
and `prj.{0,1}.{weight,bias}` before loading (two independent
shape-mismatch causes — see plan §3.4).

---

## Step 1 — Embedding Extraction

**Status: DONE (script + GPU job script), real-GPU-verified via Pilots
1-3. Full population not yet extracted — this is the actual next step,
checklist 1.11.**

```bash
cd /home/boshra95/NSRR-tools-mantis
source /home/boshra95/mantis_env/bin/activate

# Small debug run (CPU — slow, real measured ~172s/subject for a
# full-length APPLES recording; expect high variance by recording length)
python scripts/extract_mantis_embeddings.py --config configs/phase0_mantis_config.yaml \
    --datasets apples --limit 3 --cpu

# Minimal real GPU verification (small subject count, quick GPU pilot
# account per the project's standing job-submission convention — NOT
# the login node, NOT the def-forouzan production account for a quick test):
sbatch --account=def-egranger_gpu --time=00:15:00 \
    --export=ALL,END=2,DATASETS=apples \
    jobs/extract_mantis_embeddings_gpu.sh

# GPU job, full population, sharded (subject order = concatenated
# apples+shhs+mros+stages list per configs/phase0_mantis_config.yaml's
# embedding.datasets order, sliced globally — see job script header):
sbatch --export=ALL,START=0,END=2500       jobs/extract_mantis_embeddings_gpu.sh
sbatch --export=ALL,START=2500,END=5000    jobs/extract_mantis_embeddings_gpu.sh
sbatch --export=ALL,START=5000,END=7500    jobs/extract_mantis_embeddings_gpu.sh
sbatch --export=ALL,START=7500,END=9600    jobs/extract_mantis_embeddings_gpu.sh
sbatch --export=ALL,START=9600,END=12500   jobs/extract_mantis_embeddings_gpu.sh
sbatch --export=ALL,START=12500,END=15000  jobs/extract_mantis_embeddings_gpu.sh
```

**Output:** `/scratch/boshra95/psg/unified/embeddings/mantis_30sec/{dataset}/{subject_id}.npy`,
shape `[T, 6, 512]` float16 — `T` = complete 30s epochs, `6` = SLOT_ORDER
(EEG, EOG_L, EOG_R, ECG, EMG, RESP), `512` = combined (cls+mean) @ last
layer (the confirmed setting, Step 4 below).

**Verify:** `find /scratch/boshra95/psg/unified/embeddings/mantis_30sec -name '*.npy' | wc -l`
(should eventually read ~14,994 — same population as OSF's/PhysioOmni's,
since Mantis reads the same fast-channel HDF5s and, unlike PhysioOmni,
does need STAGES for `apnea_binary`).

**Real GPU throughput, measured 2026-09-06/07 (Pilot 3 + the Pilot 1/2
100-subject run, H100 MIG `1g.10gb` slice, Option D windowing — the
decided production config): ~8.0s/subject.** At that rate the full
~14,994-subject population is ~33.3h serial on one GPU — shard into
parallel jobs (~2500 subjects/job, ~5.6h each) rather than running it in
one job, per the examples above. Achieved TFLOP/s during Pilot 3: 6.18% of
the actually-allocated 1g.10gb slice's own peak (~44× PhysioOmni's
historical 0.14% on a similar slice) — the `--gpu-fraction` flag exists
specifically so this percentage is computed against the real allocated
slice, not a device query that can't be trusted on this cluster (`nvidia-smi`
reports the full card's memory even inside a MIG job).

**Atomic writes (2026-09-07)**: `extract_mantis_embeddings.py` writes to a
per-process temp file (`{subject_id}.tmp{pid}.npy`) then `os.replace()`s
it into place — a killed/timed-out job can never leave a truncated `.npy`
that a later resume would mistake for "done." Verified with a real
single-subject CPU extraction (no leftover temp files, correct final
shape). Output filenames are per-subject, so concurrent sharded jobs can
never overwrite each other's files even without separate output
directories — the `--start-idx`/`--end-idx` ranges just need to stay
disjoint (they do, by construction, in the sharding examples above).

**Not yet done**: the full population extraction itself (checklist 1.11).
Once it's running, this section should be updated with real completion
counts and any per-cohort gaps found (mirroring OSF's/PhysioOmni's own
"Full extraction complete" notes), plus the §13.4-12 split-population
audit and the §5.5 `resp_source` tally the plan calls for.

---

## Step 2 — Channel Loader + Dataset Smoke Tests

**Status: DONE.**

```bash
# Channel loader (against real HDF5s across 4 cohorts)
python scripts/test_mantis_channel_loader.py

# Checkpoint verification (both Mantis-8M and MantisPlus)
python scripts/verify_mantis_checkpoint.py

# Context-window dataset (against the real 100-subject Pilot 1/2 population,
# since production extraction hasn't run yet — --embedding-dir points at it)
python scripts/test_mantis_context_window_dataset.py \
    --config configs/phase0_mantis_config.yaml \
    --embedding-dir /scratch/boshra95/psg/unified/embeddings/mantis_pilot12/D_Llast_combined \
    --task sex_binary --context 30s 10m 240m full_night --datasets apples shhs
```

Verified (2026-09-07, against 100 real Pilot 1/2 subjects — a
population large enough to actually exercise K-sampling and the padding
branch, unlike PhysioOmni's original 3-subject test): correct
69/14/15 train/val/test split, correct shapes/dtypes at every tested
context length, `SubjectGroupedSampler` keeps each subject's items
consecutive, `full_night` collation pads correctly to the batch's longest
subject. The seq2label right-padding branch structurally never fires on
real data at 240m (`min_recording_patches=480` guarantees every kept
subject has enough epochs) — covered separately by a synthetic unit test
against `_get_seq2label_window`/`_get_causal_window` directly.

---

## Step 3 — Config

**Status: DONE.** `configs/phase0_mantis_config.yaml` — key sections:
- `embedding.*` — extraction params: `seq_len: 3840, num_patches: 240`
  (Option D, matches pretraining exactly), `return_transf_layer: -1`
  (last layer), `output_token: "combined"`, `embed_dim: 512`,
  `chunk_batch_size: 192` (32 epochs × 6 channels)
- `data.channel_candidates` — the 6-slot canonical map (EEG, EOG_L,
  EOG_R, ECG, EMG, RESP), measured per cohort 2026-08-27
- `dataset.*` — same `task_subject_dir`/`split_seed: 42`/split ratios as
  SleepFM/OSF/PhysioOmni, `min_recording_patches: 480` (240m in 30s-epoch
  units — NOT SleepFM's 2880, which is 5s-patch units)
- `model.*` — `input_dim: 3072` (6 × 512, asserted at runtime against a
  real `.npy`'s shape), `hidden_dim: 128`, `num_layers: 1` (architecture
  held constant vs. SleepFM/OSF/PhysioOmni)
- `training.*` — `mixed_precision: false` (TF32 instead, plan §4.2),
  same placeholder hyperparameters as OSF's/PhysioOmni's starting point
  otherwise (`epochs: 40`, `lr: 1e-4`, `early_stopping_patience: 10`)
- `logging.results_dir` — `/scratch/boshra95/psg/unified/results/phase0_mantis`

---

## Step 4 — Windowing/Layer Pilots (Pilots 1+2)

**Status: DONE, both decisions empirically confirmed** (not just decided
on paper) — 2026-09-07, real 100-subject GPU run (50 APPLES + 50 SHHS,
`def-egranger_gpu`, 1g.10gb MIG slice, ~61 min total).

```bash
python scripts/pilot_mantis_windowing_layer.py \
    --output-root /scratch/boshra95/psg/unified/embeddings/mantis_pilot12 \
    --datasets apples shhs --limit-per-dataset 50
```

Real embeddings for all 100 subjects × 12 variants (windowing ∈
{D, Dinterp, B} × layer ∈ {2, last} × token ∈ {cls, combined}) kept at
`/scratch/boshra95/psg/unified/embeddings/mantis_pilot12/` — real GPU
compute, and the 11 not-taken rows are the paper's supplementary "what we
gave up" numbers.

**Decisions**: Windowing — D vs B (combined@last): gap -0.0096, escape
hatch needs >0.15 → **Option D CONFIRMED**. Output layer — @2 vs @last
(Option D): gap 0.0033, escape hatch needs >0.08 → **combined @ last
CONFIRMED**. All 12 variants span only 0.0415 weighted F1 — none of these
implementation choices moved the needle much; the cross-model fairness
reasoning behind both decisions was never fighting the data. See
`MANTIS_CLAUDE.md`'s 2026-09-07 entries for the full 12-row results table.

This pilot's output (`mantis_pilot12/D_Llast_combined/`) is also the
stand-in real-data population used for Steps 2/5/6's own smoke tests below,
since production extraction (Step 1) hasn't produced a large enough
population yet.

---

## Step 5 — Training

**Status: DONE (script + GPU job script), CPU-smoke-tested end-to-end on
real data; full sweep not yet run (checklist 1.12).**

```bash
# Tiny CPU debug run (mirrors the VSCode "🦗 Mantis Phase1 Step8: Train
# Context Sweep smoke test" launch config), against the Pilot 1/2 population
python scripts/train_mantis_context_sweep.py \
    --config configs/phase0_mantis_config.yaml \
    --embedding-dir /scratch/boshra95/psg/unified/embeddings/mantis_pilot12/D_Llast_combined \
    --task sex_binary --task-type seq2label --head lstm \
    --context 30s --datasets apples shhs \
    --cpu --no-wandb
```

**Output:** `{results_dir}/{task}_{head_type}/context_{L}/{best_model.pt,
metrics.json,training_curves.csv}`, `{results_dir}/{task}_{head_type}/summary.csv`
— `results_dir` = `/scratch/boshra95/psg/unified/results/phase0_mantis`.
Identical schema to SleepFM's/OSF's/PhysioOmni's, minus
`zero_modality_indices` (no modality-group ablation here either).

**Verified 2026-09-07** against the full 98-subject Pilot 1/2 population
(69/14/15 split): `Items — train: 345 | val: 70 | test: 75`, ran to
`Status: SUCCESS — all context lengths completed` in ~14 seconds
wall-clock (30s context is the cheapest point in the sweep — subject count
alone is not a good proxy for CPU feasibility, context length matters far
more). Also verified the Transformer head at 10m context on a 10-subject
limit (exercises the CLS-token + positional-encoding + key-padding-mask
path).

**Two deliberate deviations from OSF's/PhysioOmni's training scripts**
(plan §4, both required, not optional style choices): TF32 enabled at
startup instead of autocast/fp16 (`scaler` hardcoded `None`, matching
`training.mixed_precision: false`), and achieved-TFLOP/s logged every
epoch via a parameter-count-based head-FLOP estimator — verified correct
(not just plausible) by confirming its output for the LSTM head exactly
equals `2 × N × total_weight_params` against the real printed
trainable-param count.

**Real bug found and fixed here, inherited from OSF's/PhysioOmni's own
scripts (byte-identical, confirmed by diff)**: if the early-stopping
monitor (`val_auroc`) is `NaN` for an entire run — happens when a tiny/
degenerate validation split has only one class present — `NaN > -inf` is
always `False` in Python, so `best_model.pt` was never saved, and
evaluation crashed with `FileNotFoundError` at the end. Fixed in
`train_mantis_context_sweep.py` with a safety-net checkpoint save that
doesn't fake progress or reset early-stopping patience. **Not fixed in
OSF's/PhysioOmni's own scripts** (worktree isolation rule) — hits in
practice only with unusually small `--limit` debug runs, never with real
production-scale splits.

**`--embedding-dir` override**: both `train_mantis_context_sweep.py` and
`infer_mantis_subject_windows.py` accept this flag to point at the Pilot
1/2 population instead of the production `embedding_dir` — useful until
Step 1's full extraction lands. Not part of the production registry
commands (Step 7 below), which always use the config's real
`embedding_dir`.

**GPU job:**
```bash
sbatch --export=ALL,TASK=sex_binary,HEAD=lstm jobs/train_mantis_context_sweep_gpu.sh
```
Same auto-resume mechanism as SleepFM's/OSF's/PhysioOmni's job scripts
(`--signal=B:USR1@120` bash trap + `resume.pt`), same status-JSONL
convention, logs to `logs_mantis/`. Not yet submitted for real as of
2026-09-07 — underlying training logic is CPU-verified (above); the SLURM
wrapper itself has only been syntax-checked (`bash -n`), not run through a
real `sbatch` job yet. Structurally identical to OSF's/PhysioOmni's
already-battle-tested job scripts, so no issues expected.

---

## Step 6 — Inference

**Status: DONE (script + GPU job script), CPU-smoke-tested end-to-end
against a real trained checkpoint; full sweep not yet run (depends on
1.12).**

```bash
# CPU debug run against a checkpoint trained in Step 5
python scripts/infer_mantis_subject_windows.py \
    --config configs/phase0_mantis_config.yaml \
    --embedding-dir /scratch/boshra95/psg/unified/embeddings/mantis_pilot12/D_Llast_combined \
    --task sex_binary --task-type seq2label --head lstm --context 30s \
    --datasets apples shhs --split test --batch-size 64 --cpu
```

**Output:** `{results_dir}/inference/{task}_{head_type}/context_{L}/{split}_windows.parquet`
— columns `subject_id, dataset, window_idx, true_label, pred_label,
prob_class0…prob_classN` (plus `anchor_patch_end` for seq2seq tasks — none
currently in scope). Identical schema to SleepFM's/OSF's/PhysioOmni's
inference output.

**Verified 2026-09-07**: trained a real LSTM/30s checkpoint on the full
98-subject Pilot 1/2 split, ran inference against it (15 test subjects,
all-windows mode → 14,712 rows, ~981 windows/subject). Parquet inspected
directly, not just eyeballed: exactly the 7 documented columns, correct
dtypes (`int16`/`float32`/`int32`), **zero NaN in every column**,
`prob_class0+prob_class1` sums to 1.0 every row, `window_idx` restarts at
0 per subject.

**GPU job:**
```bash
sbatch --export=ALL,TASK=sex_binary,TASK_TYPE=seq2label,HEAD=lstm,CONTEXTS="30s 10m 40m 80m 120m 240m" \
    jobs/infer_mantis_subject_windows_gpu.sh
```
Same auto-resume mechanism, same status-JSONL convention as training's job
script, logs to `logs_mantis/`. Not yet submitted for real — same caveat
as Step 5's GPU job.

**Batch-size auto-scaling note:** `_ref_bs=64`/`_ref_N=480` copied
unchanged from OSF's/PhysioOmni's own scripts (same 30s-epoch token unit
transfers directly). Not yet GPU-verified for Mantis specifically — same
open caveat those scripts' own comments carry, now carried a third time
rather than pretending it's newly solved.

---

## Step 7 — Command Generator (`gen_commands_mantis.py`)

**Status: DONE**, smoke-tested against a real Step 5/6 checkpoint; no real
GPU sweep has run yet (depends on Step 1's full extraction).

Same role as SleepFM's `scripts/gen_commands.py` / OSF's
`gen_commands_osf.py` / PhysioOmni's `gen_commands_physioomni.py`: reads
`experiments/v2_mantis_registry.yaml` and prints ready-to-run
`sbatch`/`python` commands, rather than hand-writing them. Structural fork
of `gen_commands_physioomni.py`.

```bash
# List all 15 tier-1 experiments (5 tasks × 3 heads) and their status
python scripts/gen_commands_mantis.py list

# Train sbatch command for one context (auto-computes wall time, batch/accum)
python scripts/gen_commands_mantis.py train sex_binary_lstm --context 30s

# Inference sbatch command (auto-discovers trained contexts)
python scripts/gen_commands_mantis.py infer sex_binary_lstm --split val

# File-level status / job history for one experiment
python scripts/gen_commands_mantis.py status sex_binary_lstm
python scripts/gen_commands_mantis.py runs sex_binary_lstm
```

**Registry:** `experiments/v2_mantis_registry.yaml` — 15 tier-1 entries,
fields copied verbatim from `v2_osf_registry.yaml` (same
datasets/contexts/batch_size/lr/n_size/notes per task) since Mantis is
compared against OSF's exact task/dataset scope, with
`results_dir`/`inference_dir` pointed at `.../results/phase0_mantis` and
`python_bin: /home/boshra95/mantis_env/bin/python`. **`apnea_binary` IS
present** (unlike PhysioOmni's registry) — Mantis has a RESP pathway.
`sleep_staging` is out of scope for all three baselines.

**Verified 2026-09-07** against a real checkpoint trained at the exact
production path (`{results_dir}/sex_binary_lstm/context_30s/best_model.pt`,
no `run_tag`): `list` correctly showed `trained (1/6 contexts), not
inferred`, `status` correctly listed `Trained: ['30s']`, `infer` correctly
auto-discovered `['30s']` and generated the matching sbatch command with
the right wall-time estimate. Test checkpoint removed afterward.

**Subcommands kept** (same as `gen_commands_physioomni.py`): `list,
probe-batch, train, infer, analyze, build-heatmap, collect,
threshold-tuning, status, runs`. `analyze`/`build-heatmap`/`collect`/
`threshold-tuning` call the same underlying scripts as SleepFM's/OSF's/
PhysioOmni's pipelines (`analyze_windows.py`, `build_heatmap_df.py`,
`collect_results_v2.py`, `apply_threshold_tuning.py`) **unmodified** —
backbone-agnostic.

**Subcommands deliberately dropped**: all figure/table subcommands
(`iso-plots, saturation, scaling-laws, calibration, window-position,
subject-consistency, task-comparison, cohort-saturation, precision-recall,
subject-kstar, table-1..table-10`) — same reasoning as OSF's/PhysioOmni's
generators: these wrap `plot_*.py`/`make_table*.py` scripts, superseded by
notebooks for the current paper. Once Mantis results exist, feed them into
notebooks the same way SleepFM's/OSF's/PhysioOmni's are.

**Known gaps** (same shape as OSF's/PhysioOmni's generators):
- `probe-batch` is schema parity only — no Mantis experiment uses
  `batch_mode: memory_bounded` yet, and `jobs/find_batch_size_mantis_gpu.sh`
  doesn't exist.
- Wall-time lookup tables (`_TRAIN_HOURS`/`_INFER_HOURS_PER_CTX`) are
  seeded from Pilot 3's single real GPU measurement, not a full Mantis
  training sweep yet — revisit after checklist 1.12.
- No rorqual variant — only Fir job scripts exist for Mantis so far.
- MantisPlus ablation's own registry (`v2_mantis_plus_registry.yaml`) and
  config don't exist yet (plan §5.1/§11) — `--registry` already supports
  pointing at a different registry file once they do.

---

## Step 8 — Running the Full Stage 1 Sweep

**Not yet run — this is where the pipeline goes once Step 1's full
extraction (checklist 1.11) is done.** Mirrors
`docs/EXPERIMENTS_GUIDE.md`'s "Submitting Jobs" pattern exactly, pointed
at `gen_commands_mantis.py` and looped over all 15 tier-1 experiments.

```bash
cd /home/boshra95/NSRR-tools-mantis

MANTIS_EXPS="
  sex_binary_lstm sex_binary_transformer sex_binary_mean_pool
  sleep_efficiency_binary_lstm sleep_efficiency_binary_transformer sleep_efficiency_binary_mean_pool
  bmi_binary_lstm bmi_binary_transformer bmi_binary_mean_pool
  age_class_lstm age_class_transformer age_class_mean_pool
  apnea_binary_lstm apnea_binary_transformer apnea_binary_mean_pool
"

# 1. Train — submits one sbatch job per (experiment, context), up to 90 total.
#    Submit a few at a time and watch the queue if worried about hitting a
#    job-count/priority limit — no need to fire all 90 at once.
for exp in $MANTIS_EXPS; do
    python scripts/gen_commands_mantis.py train $exp | bash
done

# 2. Monitor
python scripts/gen_commands_mantis.py list                     # status of all 15 at a glance
python scripts/gen_commands_mantis.py runs sex_binary_lstm      # job history for one
python scripts/gen_commands_mantis.py status sex_binary_lstm    # file-level detail for one
sq                                                               # raw queue state

# 3. Inference — once an experiment's contexts finish training
for exp in $MANTIS_EXPS; do
    python scripts/gen_commands_mantis.py infer $exp | bash
done

# 4. Analysis (local, no GPU — activate mantis_env first)
source /home/boshra95/mantis_env/bin/activate
for exp in $MANTIS_EXPS; do
    echo "=== START $exp $(date) ==="
    python scripts/gen_commands_mantis.py analyze $exp --plot | bash
    echo "=== END $exp $(date) ==="
done 2>&1 | tee analysis_mantis_stage1.log

# 5. Collect all results into flat CSVs
python scripts/gen_commands_mantis.py collect $MANTIS_EXPS | bash
```

**A generated train command looks like** (same shape as SleepFM's/OSF's/
PhysioOmni's — see `EXPERIMENTS_GUIDE.md`'s "Submitting Jobs" section for
the field reference):
```bash
TASK=sex_binary TASK_TYPE=seq2label HEAD=lstm CONTEXT=30s \
  DATASETS="apples shhs" BATCH_SIZE=32 ACCUM_STEPS=1 LR=1e-4 \
  CONFIG=configs/phase0_mantis_config.yaml LOGS_DIR=/home/boshra95/NSRR-tools-mantis/logs_mantis \
  sbatch --requeue --time=01:30:00 \
    --output=.../logs_mantis/train_sex_binary_lstm_30s_lr1e-4_%j.out \
    --error=.../logs_mantis/train_sex_binary_lstm_30s_lr1e-4_%j.err \
    /home/boshra95/NSRR-tools-mantis/jobs/train_mantis_context_sweep_gpu.sh
```

**Not included** (deliberately, per Step 7's design decision): saturation
curves, iso-compute plots, scaling-laws, calibration, and the other
figure/table subcommands. Once `collect` has produced `training.csv`/
`analysis.csv` under `.../phase0_mantis/collected/`, feed those into
notebooks the same way SleepFM's/OSF's/PhysioOmni's collected results are.

---

## Checkpoint Resume and Auto-Requeue

Identical mechanism to SleepFM's/OSF's/PhysioOmni's (see
`EXPERIMENTS_GUIDE.md`'s section of the same name for the full
explanation) — two distinct layers:
1. **Wall-time-triggered resume**: `#SBATCH --signal=B:USR1@120` + a bash
   trap in `jobs/train_mantis_context_sweep_gpu.sh` /
   `jobs/extract_mantis_embeddings_gpu.sh` that kills Python cleanly and
   resubmits itself via `sbatch`, picking up from `resume.pt` (training)
   or the skip-logic over already-written `.npy` files (extraction).
2. **Node-failure requeue**: SLURM-native `--requeue`, supplied at the
   *initial* `sbatch` invocation by `gen_commands_mantis.py` (see Step 7
   above) for train/infer jobs, not baked into the job script itself.

---

## Job Run History and Tracking

Same JSONL status-file convention as SleepFM/OSF/PhysioOmni:
`logs_mantis/status/train_{task}_{head}[_{run_tag}]_{context}_lr{lr}.jsonl`,
one line per `STARTED`/`TIMEOUT_REQUEUED`/`SUCCESS`/`FAILED` event. Query
via `python scripts/gen_commands_mantis.py status [<exp_id>]`/`runs
[<exp_id>]` (see Step 7 above) — mirrors `python scripts/gen_commands.py
status`/`runs` exactly, just pointed at
`experiments/v2_mantis_registry.yaml`. No real GPU training/inference jobs
have run yet as of 2026-09-07 (only debug-scale extraction verification
has), so `logs_mantis/status/train_*`/`infer_*` is still empty.
