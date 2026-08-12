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
9. Step 6 — Command Generator (`gen_commands_osf.py`) **(not yet implemented)**
10. Step 7 — Analysis and Plotting **(not yet implemented)**
11. Step 8 — Stage 2 (LoRA fine-tuning) **(not yet implemented)**
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
| **Registry** | `experiments/v2_full_registry.yaml` | `experiments/v2_osf_registry.yaml` **(not yet implemented)** |
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

**Status: DONE (script), extraction only run on small subject counts so
far (not the full population — that's checklist item 1.9).**

```bash
cd /home/boshra95/NSRR-tools
source /home/boshra95/osf_env/bin/activate

# Small debug run (CPU, ~200-400s/subject — slow, CPU-only; GPU job TBD, checklist 1.9)
python scripts/extract_osf_embeddings.py --config configs/phase0_osf_config.yaml \
    --datasets apples --limit 10 --cpu
python scripts/extract_osf_embeddings.py --config configs/phase0_osf_config.yaml \
    --datasets shhs --limit 10 --cpu
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

## Checkpoint Resume and Auto-Requeue

Identical mechanism to SleepFM's (see `EXPERIMENTS_GUIDE.md`'s section of
the same name for the full explanation) — two distinct layers:
1. **Wall-time-triggered resume**: `#SBATCH --signal=B:USR1@120` + a bash
   trap in `jobs/train_osf_context_sweep_gpu.sh` that kills Python cleanly
   and resubmits itself via `sbatch`, picking up from `resume.pt` (saved
   every epoch by `train_osf_context_sweep.py`).
2. **Node-failure requeue**: SLURM-native `--requeue`, supplied at the
   *initial* `sbatch` invocation by `gen_commands_osf.py`
   **(not yet implemented — checklist item 1.7)**, not baked into the job
   script itself.

---

## Job Run History and Tracking

Same JSONL status-file convention as SleepFM:
`logs_osf/status/train_{task}_{head}[_{run_tag}]_{context}_lr{lr}.jsonl`,
one line per `STARTED`/`TIMEOUT_REQUEUED`/`SUCCESS`/`FAILED` event.
Status-checking commands **(not yet implemented — depends on
`gen_commands_osf.py`'s `status`/`runs` subcommands, checklist item 1.7)**
will mirror `python scripts/gen_commands.py status`/`runs` exactly, just
pointed at `experiments/v2_osf_registry.yaml`.
