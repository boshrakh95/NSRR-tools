# Experiment Execution Guide — OSF Baseline

This is the OSF-baseline counterpart to `docs/EXPERIMENTS_GUIDE.md` (the
SleepFM pipeline's execution guide). **Being filled in incrementally as the
OSF pipeline is implemented** — see
`docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s Implementation Checklist for
current progress; this doc records the concrete commands/paths/verified
outputs for each step once that step is actually built and tested, so a
future session (human or Claude) can run/monitor/debug the OSF pipeline
without re-deriving anything. Sections below marked **(not yet
implemented)** are placeholders, not real content — don't treat them as
instructions until the corresponding checklist item is done.

Same operational conventions as `EXPERIMENTS_GUIDE.md` throughout:
`gen_commands`-generated commands, the same status-JSONL/log-directory
pattern, the same checkpoint/resume mechanism — different folder names
(`osf_env`, `logs_osf`, `phase0_osf`) but the same shape, per the user's
explicit instruction to keep the operational pattern comparable. Anywhere
OSF genuinely needs something different from SleepFM (a parameter, a
default, a step), it's flagged explicitly — see
`docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s Key Decisions table for the
complete list; everything not listed there is intentionally identical.

---

## Table of Contents

1. [Overview](#overview)
2. [Run identity quick-reference (OSF vs. SleepFM)](#run-identity-quick-reference-osf-vs-sleepfm)
3. [Step 0 — Environment + Checkpoint](#step-0--environment--checkpoint)
4. [Step 1 — Embedding Extraction](#step-1--embedding-extraction)
5. [Step 2 — Dataset Smoke Test](#step-2--dataset-smoke-test)
6. [Step 3 — Config](#step-3--config)
7. [Step 4 — Training](#step-4--training)
8. [Step 5 — Inference](#step-5--inference)
9. [Step 6 — Command Generator (`gen_commands_osf.py`)](#step-6--command-generator-gen_commands_osfpy)
10. [Step 7 — Running the Full Stage 1 Sweep](#step-7--running-the-full-stage-1-sweep)
11. [Step 8 — Stage 2 (LoRA fine-tuning)](#step-8--stage-2-lora-fine-tuning)
12. [Checkpoint Resume and Auto-Requeue](#checkpoint-resume-and-auto-requeue)
13. [Job Run History and Tracking](#job-run-history-and-tracking)

---

## Overview

The OSF pipeline mirrors the SleepFM `phase0_v3_full` pipeline exactly in
shape: precompute frozen encoder embeddings once, then sweep lightweight
sequence heads (LSTM / Transformer / MeanPool) over context lengths on top
of those embeddings. Same five phases per experiment (train → infer →
analyze → iso-compute → saturation curve), same subjects/splits/K-sampling
method, same training hyperparameters — only the encoder and its native
input format differ. See `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` for the
full technical plan this guide operationalizes.

---

## Run identity quick-reference (OSF vs. SleepFM)

| | SleepFM (`phase0_v3_full`) | OSF |
|---|---|---|
| **Python env** | `/home/boshra95/sleepfm_env` | `/home/boshra95/osf_env` |
| **Checkpoint** | `sleepfm-clinical/sleepfm/checkpoints/model_base` | `OSF-Open-Sleep-FM/pretrained_weights/osf_backbone.pth` |
| **Embedding config** | `configs/phase0_v3_full_config.yaml` | `configs/phase0_osf_config.yaml` |
| **HDF5 root (source)** | `/scratch/boshra95/psg_full/` | **Same** — `/scratch/boshra95/psg_full/` |
| **Embeddings dir** | `.../unified/embeddings/sleepfm_5sec/` | `.../unified/embeddings/osf_30sec/` |
| **Embedding shape** | `[T, 4, 128]` (5s patches) | `[T, 2, 768]` (30s epochs) |
| **Train config** | `configs/phase0_v3_full_config.yaml` (hidden=128, layers=1) | `configs/phase0_osf_config.yaml` (hidden=128, layers=1 — same, per parity decision) |
| **Registry** | `experiments/v2_full_registry.yaml` (`gen_commands.py`) | `experiments/v2_osf_registry.yaml` (`gen_commands_osf.py`) |
| **Results root** | `.../results/phase0_v3_full/` | `.../results/phase0_osf/` |
| **Inference root** | `.../results/phase0_v3_full/inference/` | `.../results/phase0_osf/inference/` |
| **Training/job logs** | `logs_v3_full/` | `logs_osf/` |
| **Labels/splits** | `/scratch/boshra95/psg/unified/targets_v2/` | **Same** — required for a fair comparison |
| **W&B project** | `nsrr-phase0` (or per-script default) | `nsrr-phase0-osf` (kept separate, OSF-specific difference) |
| **Task scope (first pass)** | All Tier 1/2 tasks | Tier 1 only: `sex_binary`, `sleep_efficiency_binary`, `bmi_binary`, `age_class`, `apnea_binary` |
| **Status** | **DONE** | **IN PROGRESS** — see checklist in `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` |

---

## Step 0 — Environment + Checkpoint

**Status: DONE.** See `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` §4/§5 (Appendix)
for the full build log (dependency trims, version relaxations, etc.).

```bash
module load python/3.10.13
source /home/boshra95/osf_env/bin/activate
```

Checkpoint already downloaded to
`/home/boshra95/OSF-Open-Sleep-FM/pretrained_weights/osf_backbone.pth`
(325MB, MIT license, strict-load-verified).

---

## Step 1 — Embedding Extraction

**Status: DONE (script + job script), extraction only run on small subject
counts so far (not the full population — that's checklist item 1.9).**
`jobs/extract_osf_embeddings_gpu.sh` exists but has **not been verified via
a real GPU allocation yet** (see plan doc checklist 1.8) — test with a
small `END` range before trusting it for the full run.

```bash
cd /home/boshra95/NSRR-tools
source /home/boshra95/osf_env/bin/activate

# Small debug run (CPU, ~200-400s/subject — slow, CPU-only)
python scripts/extract_osf_embeddings.py --config configs/phase0_osf_config.yaml \
    --datasets apples --limit 10 --cpu
python scripts/extract_osf_embeddings.py --config configs/phase0_osf_config.yaml \
    --datasets shhs --limit 10 --cpu

# GPU job (small test range — verify this works before the full run):
sbatch --export=ALL,END=5 jobs/extract_osf_embeddings_gpu.sh

# GPU job, full population, sharded (subject order = config's datasets list):
sbatch --export=ALL,START=0,END=2500      jobs/extract_osf_embeddings_gpu.sh
sbatch --export=ALL,START=2500,END=5000   jobs/extract_osf_embeddings_gpu.sh
# ... see the job script's header comment for the full 6-shard breakdown
```

**Output:** `/scratch/boshra95/psg_full/unified/embeddings/osf_30sec/{dataset}/{subject_id}.npy`,
shape `[T, 2, 768]`, plus a per-dataset `_channel_fill_log.jsonl` recording
which OSF input channels were zero-filled/substituted per subject (see
`docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s Channel Mapping section for the
expected per-cohort pattern to sanity-check against).

**Verify:** `find /scratch/boshra95/psg_full/unified/embeddings/osf_30sec -name '*.npy' | wc -l`

As of 2026-08-11: 10 subjects each for `apples`/`shhs` (20 total) — enough
for dataset-class/training smoke tests, not yet the full ~15,000-subject
population.

---

## Step 2 — Dataset Smoke Test

**Status: DONE.**

```bash
python scripts/test_osf_context_window_dataset.py \
    --config configs/phase0_osf_config.yaml \
    --task apnea_binary --task-type seq2label \
    --context 30s 10m 80m \
    --datasets apples shhs
```

Verified (both by Claude and the user independently, 2026-08-11): correct
`[B, N, 1536]` shapes at 30s/10m/80m contexts and correct variable-length
`full_night` collation, against a real 14/3/3 train/val/test split.

---

## Step 3 — Config

**Status: DONE.** `configs/phase0_osf_config.yaml` — see
`docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` §3.4 (Appendix) for the fully
annotated template and the two footnotes about config keys that look
load-bearing but aren't (`training.optimizer`/`scheduler`/`device`).

---

## Step 4 — Training

**Status: DONE (script + job script), smoke-tested on real data; full
sweep not yet run (checklist item 1.10).**

```bash
# Tiny CPU debug run (mirrors the VSCode "🎯 OSF Step4: Train Sweep DEBUG" config)
python scripts/train_osf_context_sweep.py \
    --config configs/phase0_osf_config.yaml \
    --task apnea_binary --task-type seq2label --head lstm \
    --context 30s 10m --datasets apples shhs \
    --max-items 50 --no-wandb --cpu
```

**Output:** `{results_dir}/{task}_{head_type}/context_{L}/{best_model.pt,metrics.json,training_curves.csv}`,
`{results_dir}/{task}_{head_type}/summary.csv` — `results_dir` =
`/scratch/boshra95/psg_full/unified/results/phase0_osf` (per
`configs/phase0_osf_config.yaml`'s `logging.results_dir`). Identical
schema to SleepFM's `metrics.json`/`summary.csv`, minus the
`zero_modality_indices` field (OSF has no modality-group ablation feature —
dropped intentionally, see plan doc §3.2/§3.3).

**GPU job** (once ready to actually submit — not done as of 2026-08-11):
```bash
sbatch --export=ALL,TASK=apnea_binary,HEAD=lstm jobs/train_osf_context_sweep_gpu.sh
```
Same auto-resume mechanism as SleepFM's job script (`--signal=B:USR1@120`
bash trap + `resume.pt`), same status-JSONL convention, logs to
`logs_osf/` instead of `logs_v3_full/`.

---

## Step 5 — Inference

**Status: DONE (script + job script), smoke-tested on real data; full
sweep not yet run (depends on the real Stage 1 sweep, checklist item 1.10).**

```bash
# CPU debug run against the checkpoints from the Step 4 debug run
# (mirrors the VSCode "🎯 OSF Step5: Infer DEBUG" config)
python scripts/infer_osf_subject_windows.py \
    --config configs/phase0_osf_config.yaml \
    --task apnea_binary --task-type seq2label --head lstm \
    --context 30s 10m 40m --datasets apples shhs \
    --split val --cpu
```

**Output:** `{results_dir}/inference/{task}_{head_type}/context_{L}/{split}_windows.parquet`
— columns `subject_id, dataset, window_idx, true_label, pred_label,
prob_class0…prob_classN` (plus `anchor_patch_end` for seq2seq tasks only).
Identical schema to SleepFM's inference output, verified by inspection
2026-08-11.

**GPU job** (once ready — not done as of 2026-08-11):
```bash
sbatch --export=ALL,TASK=apnea_binary,TASK_TYPE=seq2label,HEAD=lstm,CONTEXTS="30s 10m 40m 80m 120m 240m" \
    jobs/infer_osf_subject_windows_gpu.sh
```
Same auto-resume mechanism, same status-JSONL convention as training's job
script, logs to `logs_osf/`.

**Environment note (found 2026-08-11):** `osf_env` needed `pyarrow` for
`df.to_parquet()`, which wasn't installed (it was dropped from
`osf_env`'s dependencies during initial setup — nothing in OSF's own
model code needs it, so this wasn't caught until inference actually ran).
Compute Canada's `pip install pyarrow` doesn't work in an isolated venv
(hits a deliberate dummy package pointing at the Arrow environment module,
whose injection mechanism doesn't reach a `--system-site-packages=false`
venv). Fixed via a `.pth` file in `osf_env`'s site-packages pointing
directly at `arrow/18.1.0`'s Python 3.10 build (the default `arrow/25.0.0`
module only ships Python 3.11+ bindings) — same trick already used for
`nsrr_tools_src.pth`. No `module load` needed at runtime; verified in a
clean shell. See plan doc's Key Decisions table for the full reasoning
(including why `fastparquet` was tried and rejected as an alternative).

**Batch-size auto-scaling note:** the inference script auto-scales its
batch size down for longer contexts to avoid OOM, same formula as
SleepFM's script, but the reference point was recalibrated from SleepFM's
`_ref_N=2880` (5s-patch units) to `_ref_N=480` (OSF's own 240m in 30s-epoch
units) — reusing SleepFM's literal number would have been dimensionally
wrong. Not yet GPU-verified either way (same caveat SleepFM's own script
comment carries) — check actual memory headroom once the Stage 1 GPU sweep
runs.

---

## Step 6 — Command Generator (`gen_commands_osf.py`)

**Status: DONE**, smoke-tested against the real Step 4/5 debug checkpoints;
no real GPU sweep has run yet (depends on 1.9's full extraction).

Same role as SleepFM's `scripts/gen_commands.py`: reads
`experiments/v2_osf_registry.yaml` and prints ready-to-run `sbatch`/`python`
commands, rather than hand-writing them. **Not a `gen_commands.py`
retrofit** — a separate script, per `CLAUDE.md`'s Code-reuse-assessment
decision (the original registry/wall-time-table schema has no
backbone-family hook).

```bash
# List all 15 tier-1 experiments (5 tasks × 3 heads) and their status
python scripts/gen_commands_osf.py list

# Train sbatch command for one context (auto-computes wall time, batch/accum)
python scripts/gen_commands_osf.py train apnea_binary_lstm --context 30s

# Inference sbatch command (auto-discovers trained contexts)
python scripts/gen_commands_osf.py infer apnea_binary_lstm --split val

# File-level status / job history for one experiment
python scripts/gen_commands_osf.py status apnea_binary_lstm
python scripts/gen_commands_osf.py runs apnea_binary_lstm
```

**Registry:** `experiments/v2_osf_registry.yaml` — same 15 tier-1 entries as
`experiments/v2_full_registry.yaml` (identical datasets/contexts/batch_size/
lr, since OSF is compared against `phase0_v3_full`), with
`results_dir`/`inference_dir` pointed at `.../results/phase0_osf` and
`python_bin: /home/boshra95/osf_env/bin/python`. `sleep_staging` is
deferred (not yet ported to `OSFContextWindowDataset`).

**Subcommands kept** (same as `gen_commands.py`): `list, probe-batch, train,
infer, analyze, build-heatmap, collect, threshold-tuning, status, runs`.
`analyze`/`build-heatmap`/`collect`/`threshold-tuning` call the same
underlying scripts as SleepFM's pipeline (`analyze_windows.py`,
`build_heatmap_df.py`, `collect_results_v2.py`,
`apply_threshold_tuning.py`) **unmodified** — confirmed backbone-agnostic
by smoke test (ran with `--help` in `osf_env`, accept the exact flags
`gen_commands_osf.py` generates).

**Subcommands deliberately dropped**: all figure/table subcommands
(`iso-plots, saturation, scaling-laws, calibration, window-position,
subject-consistency, task-comparison, cohort-saturation, precision-recall,
subject-kstar, table-1..table-10`) — these wrap `plot_*.py`/`make_table*.py`
scripts, which `CLAUDE.md` already documents as superseded by notebooks for
the current paper. Once OSF results exist, feed them into notebooks the
same way SleepFM's results are, rather than building a second
figure-generation code path.

**Known gaps:**
- `probe-batch` is schema parity only — no OSF experiment uses
  `batch_mode: memory_bounded` yet, and `jobs/find_batch_size_osf_gpu.sh`
  doesn't exist.
- Wall-time lookup tables (`_TRAIN_HOURS`/`_INFER_HOURS_PER_CTX`) are
  placeholder copies of SleepFM's — not GPU-calibrated for OSF. An
  underestimate just costs one extra auto-requeue, not lost work.
- No rorqual variant — only Fir job scripts (`train_osf_context_sweep_gpu.sh`,
  `infer_osf_subject_windows_gpu.sh`) exist for OSF so far.

---

## Step 7 — Running the Full Stage 1 Sweep

**This is the actual next step once 1.9 (full extraction) is done — pure
job submission from here, no code left to write for Stage 1.** Mirrors
`docs/EXPERIMENTS_GUIDE.md`'s "Submitting Jobs" / "Typical workflow for one
experiment" pattern exactly, just pointed at `gen_commands_osf.py` and
looped over all 15 tier-1 experiments instead of one.

**Prerequisite:** checklist 1.9 done (full embeddings extracted for all 4
datasets — verify with `find .../osf_30sec -name '*.npy' | wc -l`).

```bash
cd /home/boshra95/NSRR-tools

OSF_EXPS="
  sex_binary_lstm sex_binary_transformer sex_binary_mean_pool
  sleep_efficiency_binary_lstm sleep_efficiency_binary_transformer sleep_efficiency_binary_mean_pool
  bmi_binary_lstm bmi_binary_transformer bmi_binary_mean_pool
  age_class_lstm age_class_transformer age_class_mean_pool
  apnea_binary_lstm apnea_binary_transformer apnea_binary_mean_pool
"

# 1. Train — submits one sbatch job per (experiment, context), up to 90 total.
#    Submit a few at a time and watch the queue if you're worried about
#    hitting a job-count/priority limit — no need to fire all 90 at once.
for exp in $OSF_EXPS; do
    python scripts/gen_commands_osf.py train $exp | bash
done

# 2. Monitor
python scripts/gen_commands_osf.py list                # status of all 15 at a glance
python scripts/gen_commands_osf.py runs apnea_binary_lstm    # job history for one
python scripts/gen_commands_osf.py status apnea_binary_lstm  # file-level detail for one
sq                                                       # raw queue state

# 3. Inference — once an experiment's contexts finish training
for exp in $OSF_EXPS; do
    python scripts/gen_commands_osf.py infer $exp | bash
done

# 4. Analysis (local, no GPU — activate osf_env first)
source /home/boshra95/osf_env/bin/activate
for exp in $OSF_EXPS; do
    echo "=== START $exp $(date) ==="
    python scripts/gen_commands_osf.py analyze $exp --plot | bash
    echo "=== END $exp $(date) ==="
done 2>&1 | tee analysis_osf_stage1.log

# 5. Collect all results into flat CSVs
python scripts/gen_commands_osf.py collect $OSF_EXPS | bash
```

**A generated train command looks like** (same shape as SleepFM's, see
`EXPERIMENTS_GUIDE.md`'s "Submitting Jobs" section for the field reference):
```bash
TASK=apnea_binary TASK_TYPE=seq2label HEAD=lstm CONTEXT=30s \
  DATASETS="apples shhs mros stages" BATCH_SIZE=32 ACCUM_STEPS=1 LR=1e-4 \
  CONFIG=configs/phase0_osf_config.yaml LOGS_DIR=/home/boshra95/NSRR-tools/logs_osf \
  sbatch --requeue --time=01:30:00 \
    --output=.../logs_osf/train_apnea_binary_lstm_30s_lr1e-4_%j.out \
    --error=.../logs_osf/train_apnea_binary_lstm_30s_lr1e-4_%j.err \
    /home/boshra95/NSRR-tools/jobs/train_osf_context_sweep_gpu.sh
```

**Not included** (deliberately, per Step 6's design decision): saturation
curves, iso-compute plots, scaling-laws, calibration, and the other
figure/table subcommands. Once `collect` has produced `training.csv`/
`analysis.csv` under `.../phase0_osf/collected/`, feed those into notebooks
the same way SleepFM's collected results are — don't build a parallel
`plot_*.py` pipeline for OSF.

---

## Step 8 — Stage 2 (LoRA fine-tuning)

**Status: implementation in progress.** Checklist 2.1–2.5 done (peft
verification, config + raw-signal dataset, training script, inference
script, registry + command generator) — see
`docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s Phase 2 checklist for full
technical detail; this section is the operational how-to-run counterpart,
same relationship Steps 0–7 above have to Phase 1. **Not yet done:**
checklist 2.6 (real GPU wall-time calibration), 2.7 (the full sweep,
🛑 user-gated), 2.8 (analysis).

### What's different from Stage 1

Stage 1 precomputes embeddings once, then trains lightweight heads on
cached `.npy` files — cheap, and the OSF backbone never appears in the
trainable graph. **Stage 2 puts the OSF backbone itself in the trainable
graph** (wrapped with LoRA adapters via `peft`) and trains it jointly with
the sequence head, on raw signal loaded live every step — there is no
precomputed-embeddings step at all. Concretely:

| | Stage 1 (frozen encoder) | Stage 2 (LoRA) |
|---|---|---|
| **Input to training** | Precomputed `.npy` embeddings `[T,2,768]` | Raw signal `[N,12,1920]`, loaded live per window |
| **What's trainable** | Sequence head only (~130K–2.4M params) | LoRA adapters (~442K, 0.52% of backbone) + sequence head (~2.1M) |
| **Config** | `configs/phase0_osf_config.yaml` | `configs/phase0_osf_lora_config.yaml` |
| **Training script** | `scripts/train_osf_context_sweep.py` | `scripts/train_osf_lora.py` |
| **Job script** | `jobs/train_osf_context_sweep_gpu.sh` | `jobs/train_osf_lora_gpu.sh` |
| **Inference script** | `scripts/infer_osf_subject_windows.py` (reads `.npy`) | `scripts/infer_osf_lora_subject_windows.py` (live backbone forward pass) |
| **Inference job script** | `jobs/infer_osf_subject_windows_gpu.sh` | `jobs/infer_osf_lora_subject_windows_gpu.sh` |
| **Registry** | `experiments/v2_osf_registry.yaml` | `experiments/v2_osf_lora_registry.yaml` |
| **Command generator** | `scripts/gen_commands_osf.py` | `scripts/gen_commands_osf_lora.py` |
| **Results dir** | `.../results/phase0_osf/` | `.../results/phase0_osf_lora/` |
| **Logs dir** | `logs_osf/` | `logs_osf_lora/` |
| **Checkpoint contents** | Full sequence-head `state_dict` | `peft`'s `get_peft_model_state_dict()` — LoRA deltas + head only, NOT the 85M-param frozen backbone (always reloadable from `embedding.checkpoint_dir`) |
| **Batch size** | 32 (lstm/transformer), 64–128 (mean_pool), grad-accum to an effective batch of 32 | 32 for all heads, grad-accum to effective batch 32 (same convention — **not yet verified to fit on GPU**, lower `context_micro_batch` + raise `accum_steps` if it OOMs) |
| **Training procedure** | Single stage | **Staged LP-FT**: warm-starts the sequence head from the matching Stage 1 checkpoint, then fine-tunes LoRA + head together (Kumar et al. 2022 justification — see `CLAUDE.md`) |
| **Wall-time tables** | Placeholder (uncalibrated) | Placeholder × 6 (uncalibrated, qualitative multiplier — see checklist 2.6) |

Everything not listed above (labels/splits, `task_subject_dir`,
`split_seed`, channel mapping, sequence-head architecture) is identical to
Stage 1 by design, for a fair comparison.

### Step 8.0 — Environment (same as Stage 1)

Same `osf_env`, same checkpoint. No separate setup needed — see Step 0
above.

### Step 8.1 — peft/LoRA wiring verification

**Status: DONE.** Live-verified against the real OSF checkpoint (not
assumed from docs): `LoraConfig(target_modules=["to_qkv", "to_out.0"],
r=8, lora_alpha=16)` correctly injects LoRA adapters into all 12
transformer blocks (96 LoRA-parameter submodules, 442,368 / 85,767,936
base params trainable ≈ 0.52%), `peft`'s custom-method delegation
(`forward_encoding`) forwards correctly through the wrapped model, and
gradients flow to LoRA parameters from a loss computed through it. See
`docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` checklist 2.1 for the verification
script/output.

### Step 8.2 — Config + raw-signal dataset

**Status: DONE.** `configs/phase0_osf_lora_config.yaml` — forked from
Stage 1's config, same channel mapping / labels / splits / head
architecture, new `lora:` section (`r=8, lora_alpha=16, lora_dropout=0.05,
target_modules=["to_qkv","to_out.0"], modules_to_save=["sequence_head"]`
— not yet tuned). No precomputed *embeddings* (that's the whole point of
LoRA fine-tuning the backbone) — but the raw signal itself IS precomputed,
see Step 8.2b below, added after this step was originally written.

```bash
# Dataset smoke test (mirrors Step 2 above, but for raw signal windows)
python scripts/test_osf_raw_epoch_dataset.py \
    --config configs/phase0_osf_lora_config.yaml \
    --task apnea_binary --context 30s 10m 240m \
    --datasets apples --limit 5
```

Verified (2026-08-13): correct `[B,N,12,1920]` shapes at 30s/10m/240m
context extremes, no NaN/Inf, correct padding, against real APPLES
subjects. VSCode config: `🔬 OSF-LoRA Step1`.

### Step 8.2b — Raw signal cache precompute (offline, CPU-only)

**Status: DONE, APPLES fully cached and live-verified; SHHS/MrOS/STAGES
not yet run.** **Run this before any real training or inference job** —
`OSFRawEpochWindowDataset` will raise a clear `FileNotFoundError` at
construction time if any selected subject's cache is missing, rather than
falling back to a slow raw-HDF5 read or silently dropping subjects.

**Why this exists**: a real GPU training job (`54716906`,
`apnea_binary_lstm`/30s) stalled for 2+ hours with the GPU essentially
idle before printing "Epoch 1" — root cause was `OSFRawEpochWindowDataset`
reading and resampling raw signal from HDF5 live, on every item, for
every job, repeating identical work across every task/head/context
combination. This script precomputes it once. See
`docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` checklist 2.5b for the full
diagnosis, including a related split-matching bug found and fixed in the
same pass (Stage 2 now filters subjects by Stage 1 embedding-file
existence, not raw HDF5, so `OSFRawEpochWindowDataset`'s train/val/test
split is guaranteed to exactly match `OSFContextWindowDataset`'s —
**required, since Stage 1 and Stage 2 must be compared on identical
subjects/splits**; SleepFM's own split is subtly different from OSF's,
flagged separately, not something this fix touches).

```bash
# Small test (a few subjects, CPU, seconds)
python scripts/precompute_osf_raw_signal_cache.py \
    --config configs/phase0_osf_lora_config.yaml \
    --datasets apples --limit 5

# One full dataset, locally/interactively (real numbers: 1104 APPLES
# subjects in 5.1 min at --num-workers 8, ~3.6 subjects/s, CPU-only)
python scripts/precompute_osf_raw_signal_cache.py \
    --config configs/phase0_osf_lora_config.yaml \
    --datasets apples --num-workers 8

# Full population, sharded, CPU-only SLURM job (no GPU allocation used —
# ~15,000 subjects total; at the APPLES rate, expect well under an hour
# per shard at --num-workers 16, comfortably inside the 8h default):
sbatch --export=ALL,START=0,END=4000       jobs/precompute_osf_raw_signal_cache.sh
sbatch --export=ALL,START=4000,END=8000    jobs/precompute_osf_raw_signal_cache.sh
sbatch --export=ALL,START=8000,END=12000   jobs/precompute_osf_raw_signal_cache.sh
sbatch --export=ALL,START=12000,END=15100  jobs/precompute_osf_raw_signal_cache.sh

# Or one dataset at a time:
sbatch --export=ALL,DATASETS=shhs   jobs/precompute_osf_raw_signal_cache.sh
sbatch --export=ALL,DATASETS=mros   jobs/precompute_osf_raw_signal_cache.sh
sbatch --export=ALL,DATASETS=stages jobs/precompute_osf_raw_signal_cache.sh

# Re-run any time to fill gaps — already-cached subjects are skipped
# automatically (idempotent, safe to re-submit or re-run interactively):
python scripts/precompute_osf_raw_signal_cache.py --config configs/phase0_osf_lora_config.yaml
```

**Output:** `{raw_signal_cache_dir}/{dataset}/{subject_id}.npy` — float16,
shape `[12, n_samples_64]` (resampled, NOT epoch-chunked), plus a
per-dataset `_channel_fill_log.jsonl` (same schema/purpose as Stage 1's
extraction log). `raw_signal_cache_dir` =
`/scratch/boshra95/psg_full/unified/osf_raw_signal_64hz` (per
`configs/phase0_osf_lora_config.yaml`'s `data.raw_signal_cache_dir`).

**Verify:** `find /scratch/boshra95/psg_full/unified/osf_raw_signal_64hz -name '*.npy' | wc -l`
— compare against `find /scratch/boshra95/psg_full/{dataset}/derived/hdf5_signals -name '*.h5' | wc -l`
per dataset (apples=1104, shhs=8444, mros=3933, stages=1513).

**Estimated total size**: ~0.8TB (float16, all ~15,000 subjects) — well
within scratch headroom (12.6TB free at time of writing, 6.4TB/19TB used).

**As of 2026-08-14**: APPLES fully cached (1104/1104) and live-verified
(dataset construction: 172s cold for train+val+test combined, vs. the
~47-min-per-job cost this replaces; zero train/val/test subject overlap
confirmed; a real, non-`--limit` training run against the full cache
produced numerically identical training behavior to earlier raw-HDF5-
backed pilots). SHHS/MrOS/STAGES not yet run — do that before any real
sweep involving those cohorts (checklist 2.7).

### Step 8.3 — Training

**Status: DONE (script + job script), CPU pilot successful end-to-end;
full sweep not yet run (depends on 2.6/2.7).**

**Levers available if `batch_size=32` doesn't fit on GPU** (checklist
2.5c — a pre-submission audit found these were silently unwired despite
Stage 1 having working versions; both default OFF, matching Stage 1/
SleepFM, so nothing changes unless you flip them in
`configs/phase0_osf_lora_config.yaml`):
- `training.mixed_precision: true` — enables AMP (autocast + GradScaler),
  reduces backbone activation memory. Try this before lowering batch size.
- `training.weighted_sampler: true` — switches the train loader from
  `SubjectGroupedSampler` to `WeightedRandomSampler`; trades away the
  cache-locality speed benefit for class-balanced sampling. Usually not
  needed since `class_weights: "auto"` already reweights the loss.
- Lower `context_micro_batch` in `experiments/v2_osf_lora_registry.yaml`
  and let `accum_steps` rise to compensate (keeps `effective_batch=32`) —
  see Step 8.5 below.

```bash
# Tiny CPU debug pilot (mirrors the VSCode "🔬 OSF-LoRA Step2" config).
# --limit/--max-items keep this fast — DEBUG ONLY, omit for real runs.
# NOTE: training.epochs comes from the config file, not a CLI flag (same
# as Stage 1) — edit configs/phase0_osf_lora_config.yaml's training.epochs
# temporarily (e.g. to 3) for a faster debug run, then set it back.
python scripts/train_osf_lora.py \
    --config configs/phase0_osf_lora_config.yaml \
    --task apnea_binary --head lstm --context 30s \
    --datasets apples --limit 24 --max-items 12 \
    --batch-size 2 --run-tag pilot_test --cpu
```

**Output:** `{results_dir}/{task}_{head_type}/context_{L}/{best_model.pt,metrics.json,training_curves.csv}`,
`{results_dir}/{task}_{head_type}/summary.csv` — `results_dir` =
`/scratch/boshra95/psg_full/unified/results/phase0_osf_lora`.
`best_model.pt` is a `peft` state dict (LoRA deltas + `sequence_head`),
**not** the full backbone — much smaller than Stage 1's checkpoint.

**⚠️ Required run order: 30s before any other context, per (task, head)**
(checklist 2.5d — full reasoning there; this is the operational summary).
Compute scales ~linearly with context length, so independently
fine-tuning all 6 lengths isn't feasible. The fix: **only 30s
warm-starts from Stage 1** (unchanged); **every other context length
warm-starts from that same (task, head)'s own 30s LoRA checkpoint**
instead — auto-detected at `{results_dir}/{task}_{head}/context_30s/best_model.pt`,
always the **plain, untagged path**, regardless of what `--run-tag` the
*current* run uses. Each context length still trains fully independently
to its own convergence — only the starting point changes.
- If you run a non-30s context before that task/head's 30s run exists
  (and don't pass an explicit override), it fails immediately with a
  clear error telling you to run 30s first — it does **not** silently
  fall back to Stage 1 or random init.
- Overrides, for edge cases (e.g. ablating warm-start source): `--stage2-30s-checkpoint <path>`
  (point at a specific Stage 2 30s checkpoint) or `--stage1-checkpoint <path>`
  (force the old Stage-1-warm-start behavior for a non-30s context).
- `gen_commands_osf_lora.py train` also checks this at command-generation
  time and prints a `⚠ WARNING`/`⚠ will fail` tag if a 30s checkpoint
  isn't discoverable yet — see Step 8.5.

**Run-tag policy for the current rollout**: only the very first
`apnea_binary_lstm` 30s comparison (old `lr=1e-4` pilot vs. the
re-tuned `run_tag=v2` pilot, see Step 8.3's config-revision note below)
uses a `--run-tag`. Once you've picked the better of the two, consolidate
it into the plain untagged path (`apnea_binary_lstm/context_30s/`) —
rename directories, don't leave both — and delete the losing pilot's
output entirely. **Do not use `--run-tag` for any other task or context**
— every other 30s run (the other 4 tasks) and every longer-context run
goes straight to the untagged path, so the auto-detection above finds it
with zero ambiguity.

**Training budget revised 2026-08-15** (checklist 2.5d, grounded in a
real observed overfitting curve, not guessed): `epochs=18,
early_stopping_patience=5, lr=5e-5` (`context_lr_overrides` at 2.5e-5 for
120m/240m) — down from Stage 1's original placeholder values
(`40/10/1e-4`). These are now the config defaults; no CLI flags needed to
get them.

Without either a resume state or a warm-start checkpoint at all, the head
starts from random init — a loud `warnings.warn` fires, and this is only
meant for quick architecture-correctness pilots, never a real run.

**Expected pilot behavior — severe overfitting, not a bug:** with only a
handful of debug items, train loss collapses toward 0 within a few epochs
while val loss climbs and val accuracy stays at chance. This is expected —
the pilot's purpose is "does the pipeline run and do gradients flow,"
not "is the model good." Don't read anything into pilot-scale metrics.

**GPU job — go through `gen_commands_osf_lora.py`, not a hand-written
`sbatch` call.** Calling `jobs/train_osf_lora_gpu.sh` directly without
`--context` runs ALL requested context lengths sequentially inside one
job against a single wall-time budget — easy to under-provision.
`gen_commands_osf_lora.py` submits one correctly-time-boxed job per
context instead:
```bash
# One job per context, each with its own wall-time estimate + resolved batch/accum
python scripts/gen_commands_osf_lora.py train apnea_binary_lstm | bash

# Or a single context to start
python scripts/gen_commands_osf_lora.py train apnea_binary_lstm --context 30s | bash
```
Same auto-resume mechanism as Stage 1 (`--signal=B:USR1@120` + `resume.pt`,
saved every epoch as a `peft` state dict + optimizer/scheduler state),
same status-JSONL convention, logs to `logs_osf_lora/`.

**GPU size**: `jobs/train_osf_lora_gpu.sh` requests `3g.40gb` (upgraded
2026-08-15 from `1g.10gb` — MIG partitions compute proportionally to
memory, so this is a real ~3× throughput gain). `3g.40gb` is the largest
MIG slice this cluster offers (no full/unpartitioned H100 available).

⚠️ **Wall-time NOT calibrated at `3g.40gb`** — every estimate
`gen_commands_osf_lora.py` prints is still a placeholder. One real data
point exists at the *old* `1g.10gb` size: `apnea_binary/lstm/30s` ran at
~60 min/epoch (measured from `resume.pt`, not guessed). **Compute scales
~linearly with context length, not sub-linearly** (see checklist 2.5d) —
240m would be ~480× that at the same GPU size. Auto-requeue means an
underestimate just costs one resubmission, not lost work, but confirm a
context length is actually tractable before submitting a long one.

### Step 8.4 — Inference

**Status: DONE (script + job script), CPU pilot successful end-to-end
against a real Stage-2-trained checkpoint; full sweep not yet run.**

```bash
# CPU debug run against a Step 8.3 pilot checkpoint (mirrors the VSCode
# "🔬 OSF-LoRA Step3" config). --limit is DEBUG ONLY (not in Stage 1's
# inference script at all) — Stage 2 inference runs a live backbone
# forward pass per window, so full-scope CPU debugging is impractically
# slow without it; omit for real runs.
python scripts/infer_osf_lora_subject_windows.py \
    --config configs/phase0_osf_lora_config.yaml \
    --task apnea_binary --head lstm --context 30s \
    --datasets apples --split test --limit 6 --no-all-windows \
    --batch-size 2 --run-tag pilot_test --cpu
```

**Output:** `{inference_dir}/{task}_{head_type}/context_{L}/{split}_windows.parquet`
— identical schema to Stage 1's inference output (`subject_id, dataset,
window_idx, true_label, pred_label, prob_class0…prob_classN`) for
downstream `analyze`/`collect` compatibility. No `anchor_patch_end`
column — Stage 2 is seq2label-only for now (matches
`OSFRawEpochWindowDataset`'s scope).

**GPU job — again, prefer `gen_commands_osf_lora.py infer` over a
hand-written `sbatch` call**, same reason as training: it auto-discovers
trained contexts and sizes the wall-time estimate to match, rather than
using the job script's flat `05:00:00` default regardless of how many
contexts are listed (not yet run for real):
```bash
python scripts/gen_commands_osf_lora.py infer apnea_binary_lstm --split test | bash
```
Same auto-resume mechanism, same status-JSONL convention, logs to
`logs_osf_lora/`.

**Batch-size note (deliberately simpler than Stage 1's):** does NOT reuse
Stage 1's context-length auto-scaling formula — that formula assumes a
cheap embedding lookup, and every Stage 2 item still runs a full
LoRA-adapted backbone forward pass. Uses a fixed `--batch-size` (default
4) instead of inventing an unverified scaling formula.

### Step 8.5 — Command Generator (`gen_commands_osf_lora.py`)

**Status: DONE**, smoke-tested against the real registry (`list`,
`train`, `infer`, `status`, `runs`, `analyze`, `collect` all verified to
produce correct commands); no real GPU sweep has run yet.

```bash
# List all 15 tier-1 LoRA experiments and their status
python scripts/gen_commands_osf_lora.py list

# Train sbatch command for one context — warm-start source is
# auto-detected (Stage 1 checkpoint for 30s, this task/head's own 30s
# LoRA checkpoint for everything else, checklist 2.5d); no need to specify it
python scripts/gen_commands_osf_lora.py train apnea_binary_lstm --context 30s

# Inference sbatch command (auto-discovers trained contexts)
python scripts/gen_commands_osf_lora.py infer apnea_binary_lstm --split val

# File-level status / job history for one experiment
python scripts/gen_commands_osf_lora.py status apnea_binary_lstm
python scripts/gen_commands_osf_lora.py runs apnea_binary_lstm

# Analysis / collection — SAME underlying scripts as Stage 1, reused
# unmodified, just pointed at phase0_osf_lora's results dir
python scripts/gen_commands_osf_lora.py analyze apnea_binary_lstm --plot
python scripts/gen_commands_osf_lora.py collect
```

**Registry:** `experiments/v2_osf_lora_registry.yaml` — same 15 tier-1
entries as `v2_osf_registry.yaml` (identical tasks/datasets/contexts,
required for a fair comparison), `results_dir`/`inference_dir` pointed at
`.../results/phase0_osf_lora`, **same `gradient_accumulation` convention
as Stage 1/SleepFM**: `batch_size: 32` per entry + a `context_micro_batch`
table (all contexts start at 32, `accum_steps=1`) — required for
comparable training dynamics, not just comparable results. **32 is NOT
yet verified to fit on GPU for Stage 2** — if a context OOMs, lower that
context's `context_micro_batch` value and `accum_steps` rises
automatically to keep `effective_batch=32`.

**Real schema differences from `gen_commands_osf.py`** (not just path
renames — found while double-checking against Stage 1, see
`docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` checklist 2.5 for the full
reasoning):
- **No `probe-batch` subcommand** — dropped entirely (not kept for schema
  parity like Stage 1's is), since this registry has no `batch_mode`
  concept to probe (only the `grad_accum` mode is implemented, no
  `memory_bounded` branch — no probing script exists for Stage 2).
- **No `TASK_TYPE` env var** — neither Stage 2 script has a `--task-type`
  flag.
- Added an optional `--stage1-checkpoint` override for `train` (normally
  omitted so auto-detection handles it).
- `train` prints a `⚠ WARNING`/`⚠ will fail` pre-flight tag (checklist
  2.5d) if a non-30s context is requested before that (task, head) has a
  discoverable 30s Stage 2 checkpoint — catches the ordering requirement
  at command-generation time rather than only at job runtime.

**Subcommands kept**: `list, train, infer, analyze, build-heatmap,
collect, threshold-tuning, status, runs` — same figure/table subcommands
dropped as Stage 1's generator, same reasoning (superseded by notebooks).

**Two real bugs found and fixed while building this step:**
1. `train_osf_lora.py` never read `context_lr_overrides` from the config
   (Stage 1's script does — `phase0_osf_lora_config.yaml` sets a lower LR
   for 120m/240m, but nothing was applying it). Fixed by porting Stage 1's
   exact `cli_lr_set`-gated override logic into `train_osf_lora.py`. If
   you ran any 120m/240m Stage 2 training before 2026-08-14, it used the
   base LR, not the override — re-run if that matters for your results.
2. **This step originally shipped with NO gradient accumulation at all**
   (flat `batch_size: 4`, reasoned as "Stage 2's cost is dominated by the
   backbone forward pass so effective-batch parity doesn't obviously
   matter") — **the user correctly rejected this**: batch size affects
   optimization dynamics regardless of what dominates per-item compute,
   and the standing rule is to match SleepFM/Stage 1's options unless the
   model genuinely requires otherwise (it doesn't here). Fixed the same
   day: `run_epoch()` (already imported from `train_osf_context_sweep.py`)
   already had working `accum_steps` support — it just wasn't wired
   through `train_osf_lora.py`'s CLI, the job script, or the generator.
   Added `--accum-steps`, the `args.batch_size or 32` fallback, and
   `resolve_batch_accum()`, all matching Stage 1 exactly. If you ran any
   Stage 2 training before 2026-08-14, it trained at effective batch 4,
   not 32 — re-run if that matters for your results.

**Audit note**: a full three-way CLI comparison (SleepFM vs. Stage 1 vs.
Stage 2 training/inference scripts) also confirmed `--task-type`/
`--full-night-epochs` are genuinely, deliberately absent from Stage 2
(seq2label-only scope, no `full_night` support — not oversights), and
`--wandb-project`/`--wandb-entity`/`--no-wandb` are absent too, left that
way since `wandb` isn't installed in `osf_env` for *either* stage (Stage
1's own W&B flags are already non-functional) — flagged, not fixed, since
it has zero effect on training.

### Not yet done

- **Checklist 2.6** — real wall-time calibration pilot on GPU (current
  wall-time tables are an unverified 6× placeholder multiplier on Stage
  1's own already-placeholder numbers).
- **Checklist 2.7** — 🛑 the full Stage 2 sweep (same 5-task ×
  lstm/transformer scope as Stage 1's current progress; `mean_pool`
  deferred, matching Stage 1). Applies the memory-mitigation ladder from
  `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md` Appendix §6.2 if GPU memory
  pressure appears (gradient checkpointing → larger GPU allocation →
  capped max context, in that order).
- **Checklist 2.8** — analyze + collect Stage 2 results, compare against
  Stage 1 + SleepFM, same contamination-aware methodology as Stage 1's
  results section.

---

## Checkpoint Resume and Auto-Requeue

Identical mechanism to SleepFM's (see `EXPERIMENTS_GUIDE.md`'s section of
the same name for the full explanation) — two distinct layers:
1. **Wall-time-triggered resume**: `#SBATCH --signal=B:USR1@120` + a bash
   trap in `jobs/train_osf_context_sweep_gpu.sh` that kills Python cleanly
   and resubmits itself via `sbatch`, picking up from `resume.pt` (saved
   every epoch by `train_osf_context_sweep.py`).
2. **Node-failure requeue**: SLURM-native `--requeue`, supplied at the
   *initial* `sbatch` invocation by `gen_commands_osf.py` (see Step 6
   above), not baked into the job script itself.

---

## Job Run History and Tracking

Same JSONL status-file convention as SleepFM:
`logs_osf/status/train_{task}_{head}[_{run_tag}]_{context}_lr{lr}.jsonl`,
one line per `STARTED`/`TIMEOUT_REQUEUED`/`SUCCESS`/`FAILED` event. Query
via `python scripts/gen_commands_osf.py status [<exp_id>]`/`runs [<exp_id>]`
(see Step 6 above) — mirrors `python scripts/gen_commands.py status`/`runs`
exactly, just pointed at `experiments/v2_osf_registry.yaml`. No real GPU
jobs have run yet as of 2026-08-12, so `logs_osf/status/` is still empty
(`runs` correctly reports "No status directory found yet").
