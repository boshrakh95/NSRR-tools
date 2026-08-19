# Experiment Execution Guide — PhysioOmni Baseline

This is the PhysioOmni-baseline counterpart to `docs/EXPERIMENTS_GUIDE.md`
(the SleepFM pipeline's execution guide) and `docs/OSF_EXPERIMENTS_GUIDE.md`
(the OSF baseline's). **Being filled in incrementally as the PhysioOmni
pipeline is implemented and run** — see
`docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md`'s Implementation Checklist
for current progress; this doc records the concrete commands/paths/verified
outputs for each step once that step is actually built and tested, so a
future session (human or Claude) can run/monitor/debug the PhysioOmni
pipeline without re-deriving anything. Sections below marked **(not yet
implemented)** are placeholders, not real content — don't treat them as
instructions until the corresponding checklist item is done. **Keep this
file updated alongside the implementation plan as jobs run and things get
fixed/changed** — same living-document convention as the plan doc.

Same operational conventions as `EXPERIMENTS_GUIDE.md`/`OSF_EXPERIMENTS_GUIDE.md`
throughout: `gen_commands`-generated commands, the same status-JSONL/log-directory
pattern, the same checkpoint/resume mechanism — different folder names
(`physioomni_env`, `logs_physioomni`, `phase0_physioomni`) but the same
shape. Anywhere PhysioOmni genuinely needs something different from
SleepFM/OSF (a parameter, a default, a step, an excluded task), it's flagged
explicitly — see `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md`'s checklist
for the complete reasoning behind each; everything not listed there is
intentionally identical.

**Scope note (unlike SleepFM):** SleepFM is our main model, so it gets the
full protocol — modality/channel ablations, full vs. fast-channel rounds,
sleep staging, all task tiers. PhysioOmni (like OSF, and like MOMENT
whenever that starts) is a *comparison baseline*, not the main model — it
gets exactly **two rounds**: one frozen-backbone round (this guide, Steps
0–7) and one LoRA-fine-tuned round (Step 8, not yet started). No channel
ablation, no full-channel round, no sleep staging (not yet ported to any of
the TSFM baselines' dataset classes) — just the 4 Tier-1 seq2label tasks ×
3 heads × 6 context lengths, frozen then LoRA, compared against SleepFM's
`phase0_v3` (paper-primary) results.

---

## Table of Contents

1. [Overview](#overview)
2. [Run identity quick-reference (PhysioOmni vs. OSF vs. SleepFM)](#run-identity-quick-reference-physioomni-vs-osf-vs-sleepfm)
3. [Step 0 — Environment + Checkpoint](#step-0--environment--checkpoint)
4. [Step 1 — Embedding Extraction](#step-1--embedding-extraction)
5. [Step 2 — Dataset Smoke Test](#step-2--dataset-smoke-test)
6. [Step 3 — Config](#step-3--config)
7. [Step 4 — Training](#step-4--training)
8. [Step 5 — Inference](#step-5--inference)
9. [Step 6 — Command Generator (`gen_commands_physioomni.py`)](#step-6--command-generator-gen_commands_physioomnipy)
10. [Step 7 — Running the Full Stage 1 Sweep](#step-7--running-the-full-stage-1-sweep)
11. [Step 8 — Stage 2 (LoRA fine-tuning)](#step-8--stage-2-lora-fine-tuning-not-yet-implemented)
12. [Checkpoint Resume and Auto-Requeue](#checkpoint-resume-and-auto-requeue)
13. [Job Run History and Tracking](#job-run-history-and-tracking)

---

## Overview

The PhysioOmni pipeline mirrors the SleepFM `phase0_v3` (fast-channel,
paper-primary) and OSF pipelines exactly in shape: precompute frozen
encoder embeddings once, then sweep lightweight sequence heads (LSTM /
Transformer / MeanPool) over context lengths on top of those embeddings.
Same five phases per experiment (train → infer → analyze → iso-compute →
saturation curve), same subjects/splits/K-sampling method, same training
hyperparameters — only the encoder and its native input format differ.

**What's genuinely different from OSF's version of this pipeline** (see
`docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` for the full technical
reasoning behind each):
- **Source HDF5 tree**: fast-channel `psg/` (same tree SleepFM's
  `phase0_v3` uses), not `psg_full/` — PhysioOmni needs only
  EEG/EOG/ECG/EMG (no respiratory pathway at all), so the fast-channel
  tree already carries everything needed. Compared against `phase0_v3`,
  not `phase0_v3_full`.
- **Embedding shape**: genuinely 2D `[T, 500]` (200 EEG + 100 EOG + 100
  ECG + 100 EMG, already flat, concatenated from 4 *independent* frozen
  encoders — no unified cross-modality fusion in the released checkpoint,
  no `N_SUBTOKENS` middle dimension the way OSF's `[T, 2, 768]` needs).
- **Task scope**: 4 Tier-1 tasks, not 5 — **`apnea_binary` is excluded**,
  since PhysioOmni has no respiratory/airflow pathway anywhere in the
  model or its pretraining data (confirmed at 4 independent code
  locations — see `CLAUDE.md`'s PhysioOmni section).
- **SHHS gets 1 real EEG channel, not 2** — SHHS's generic `'EEG'` HDF5
  key maps to `EEG_C3` only; `EEG_C4` is omitted entirely (not
  duplicated, not zero-filled) — see plan §4.5.
- **Normalization inversion is self-calibrating, not a hardcoded
  per-channel table** — `invert_normalization()` checks `abs(stats["std"])
  < 1.0` per channel at runtime, since the raw unit (V vs. µV) turned out
  to be file/cohort-dependent, not a fixed convention.

See `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` for the full technical
plan this guide operationalizes, including §19–§20 (native context ceiling
reasoning and the 3-way SleepFM/OSF/PhysioOmni comparison table).

---

## Run identity quick-reference (PhysioOmni vs. OSF vs. SleepFM)

| | SleepFM (`phase0_v3`) | OSF | PhysioOmni |
|---|---|---|---|
| **Python env** | `/home/boshra95/sleepfm_env` | `/home/boshra95/osf_env` | `/home/boshra95/physioomni_env` |
| **Checkpoint** | `sleepfm-clinical/sleepfm/checkpoints/model_base` | `OSF-Open-Sleep-FM/pretrained_weights/osf_backbone.pth` | `/home/boshra95/PhysioOmni/checkpoints/PhysioOmni.pt` |
| **Embedding config** | `configs/phase0_v3_config.yaml` | `configs/phase0_osf_config.yaml` | `configs/phase0_physioomni_config.yaml` |
| **HDF5 root (source)** | `/scratch/boshra95/psg/` (fast-channel) | `/scratch/boshra95/psg_full/` | **Same as SleepFM** — `/scratch/boshra95/psg/` (fast-channel) |
| **Embeddings dir** | `.../unified/embeddings/sleepfm_5sec/` | `.../unified/embeddings/osf_30sec/` | `.../unified/embeddings/physioomni_30sec/` |
| **Embedding shape** | `[T, 4, 128]` (5s patches) | `[T, 2, 768]` (30s epochs) | `[T, 500]` (30s epochs, genuinely 2D — no subtoken dim) |
| **Train config** | `configs/phase0_v3_config.yaml` (hidden=128, layers=1) | `configs/phase0_osf_config.yaml` (hidden=128, layers=1) | `configs/phase0_physioomni_config.yaml` (hidden=128, layers=1 — same, per parity decision) |
| **Registry** | `experiments/v2_registry.yaml` (`gen_commands.py`) | `experiments/v2_osf_registry.yaml` (`gen_commands_osf.py`) | `experiments/v2_physioomni_registry.yaml` (`gen_commands_physioomni.py`) |
| **Results root** | `.../results/phase0_v3/` | `.../results/phase0_osf/` | `.../results/phase0_physioomni/` |
| **Inference root** | `.../results/phase0_v3/inference/` | `.../results/phase0_osf/inference/` | `.../results/phase0_physioomni/inference/` |
| **Training/job logs** | `logs_v3/` | `logs_osf/` | `logs_physioomni/` |
| **Labels/splits** | `/scratch/boshra95/psg/unified/targets_v2/` | **Same** | **Same** — required for a fair comparison |
| **W&B project** | `nsrr-phase0` (or per-script default) | `nsrr-phase0-osf` | `nsrr-phase0-physioomni` — **but wandb is not installed in `physioomni_env`** (same known gap as `osf_env`); training/inference scripts fall back gracefully with a warning, not a crash, so `--no-wandb` is optional, not required. |
| **Task scope** | All Tier 1/2 tasks | Tier 1 only: sex, sleep efficiency, BMI, age, apnea (5 tasks) | Tier 1 minus apnea: sex, sleep efficiency, BMI, age (**4 tasks** — apnea excluded, no respiratory pathway) |
| **Channel ablation** | Yes (main model, full protocol) | No (comparison baseline) | No (comparison baseline) |
| **Rounds** | Full protocol | Frozen + LoRA (2 rounds) | Frozen (this guide) + LoRA (not yet started) — 2 rounds, same as OSF |
| **Status** | **DONE** | Frozen round in progress, LoRA in progress | **Frozen round in progress** — see checklist in `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` |

---

## Step 0 — Environment + Checkpoint

**Status: DONE.** See `docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md`
checklist 0.1/0.2 for the full build log.

```bash
module load python/3.10.13
source /home/boshra95/physioomni_env/bin/activate
```

Checkpoint already downloaded to
`/home/boshra95/PhysioOmni/checkpoints/PhysioOmni.pt`, strict-load-verified
against the 4 independent `NeuralTransformer` encoders (EEG n_embd=200,
patch=200; EOG/ECG/EMG n_embd=100, patch=100). License is split: the GitHub
code repo has no LICENSE file, but the HuggingFace weights repo
(`Weibang/PhysioOmni`) declares CC-BY-4.0 explicitly (verified live via the
HF API) — see `CLAUDE.md`'s PhysioOmni section for the full note.

**Environment fix applied 2026-08-18** (found while building Step 5/inference):
`physioomni_env` had a non-functional `pyarrow` (Compute Canada ships it as
a "dummy" stub wheel — the real compiled package lives under the `arrow`
environment module's own site-packages). Fixed by adding a `.pth` file:
```bash
echo "/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v4/Compiler/gcccore/arrow/18.1.0/lib/python3.10/site-packages" \
    > /home/boshra95/physioomni_env/lib/python3.10/site-packages/pyarrow_arrow_module.pth
```
No `module load` needed at runtime (the `.pth` path is absolute) — same
trick already used for `osf_env`'s identical gap.

---

## Step 1 — Embedding Extraction

**Status: DONE (script + both job scripts), real-GPU-verified. Full
population not yet extracted (checklist 1.10, next step).**

```bash
cd /home/boshra95/NSRR-tools-omni
source /home/boshra95/physioomni_env/bin/activate

# Small debug run (CPU — slow, 50-450s/subject, highly variable by recording length)
python scripts/extract_physioomni_embeddings.py --config configs/phase0_physioomni_config.yaml \
    --datasets apples --limit 8 --cpu
python scripts/extract_physioomni_embeddings.py --config configs/phase0_physioomni_config.yaml \
    --datasets shhs --limit 8 --cpu

# CPU-only pilot/debug SLURM job (for a handful of subjects — NEVER run
# sustained extraction directly on the login node):
sbatch --export=ALL,DATASETS="apples",LIMIT=8 jobs/extract_physioomni_embeddings_cpu.sh

# GPU job (small test range — verify before the full run):
sbatch --export=ALL,END=20,DATASETS="apples shhs" jobs/extract_physioomni_embeddings_gpu.sh

# GPU job, full population, sharded (subject order = concatenated
# apples+shhs+mros+stages list, sliced globally — see job script header):
sbatch --export=ALL,START=0,END=2500       jobs/extract_physioomni_embeddings_gpu.sh
sbatch --export=ALL,START=2500,END=5000    jobs/extract_physioomni_embeddings_gpu.sh
sbatch --export=ALL,START=5000,END=7500    jobs/extract_physioomni_embeddings_gpu.sh
sbatch --export=ALL,START=7500,END=9600    jobs/extract_physioomni_embeddings_gpu.sh
sbatch --export=ALL,START=9600,END=12500   jobs/extract_physioomni_embeddings_gpu.sh
sbatch --export=ALL,START=12500,END=15000  jobs/extract_physioomni_embeddings_gpu.sh
```

**Output:** `/scratch/boshra95/psg/unified/embeddings/physioomni_30sec/{dataset}/{subject_id}.npy`,
shape `[T, 500]` float16 — genuinely 2D, no subtoken dimension.

**Verify:** `find /scratch/boshra95/psg/unified/embeddings/physioomni_30sec -name '*.npy' | wc -l`

**Real GPU throughput, measured 2026-08-18 (checklist 1.9, 3 real `sbatch`
jobs on an H100 MIG `1g.10gb` slice):** ~4.1s/subject — 15-100x faster than
the CPU path. A `chunk_batch_size` A/B (16 vs. 64, matched 20-subject shhs
batches) found **no meaningful difference** here (4.05s vs. 4.15s/subject)
— unlike OSF, where this knob was the real bottleneck. Kept at 16 (the
config default). At this rate, the full ~14,994-subject population
(apples 1104 + shhs 8444 + mros 3933 + stages 1513) is ~17h serial on one
GPU — shard into parallel jobs (~2500 subjects/job, ~2.85h each) rather
than running it in one job, per the examples above.

**Full extraction complete as of 2026-08-19** (checklist 1.10): apples
1104/1104, shhs 8444/8444, mros 3933/3933, stages 1512/1513 —
14,993/14,994 (99.99%), zero errors elsewhere. The 1 gap
(`stages/STLK00096`) has no PhysioOmni-relevant channels at all in its
HDF5 — a known outlier, already flagged in `CLAUDE.md`'s OSF section too
(same subject is SleepFM-only there), not a pipeline bug. **Ready for the
full Stage 1 sweep (Step 7 below).**

---

## Step 2 — Dataset Smoke Test

**Status: DONE**, with an honest population-size caveat.

```bash
python scripts/test_physioomni_context_window_dataset.py \
    --config configs/phase0_physioomni_config.yaml
```

VSCode debug config: **"🫀 PhysioOmni Phase1 Step3: Test ContextWindowDataset
(sex_binary, apples+shhs)"**. Verified (2026-08-18,
against the 3 subjects extracted at that point): correct 2/0/1
train/val/test split (val=0 is arithmetic at that population size —
`int(3*0.15)==0` — not a bug), correct `(N, 500)` shapes at
`30s`/`10m`/`full_night`, zero NaN, zero unexpected padding. **Known gap**:
that population didn't exercise the padding branch or realistic
K-sampling — re-test with more extracted subjects before trusting fully at
sweep scale (the Step 4 training smoke test below, run against 16
subjects, is a stronger real-world check of the same dataset class).

---

## Step 3 — Config

**Status: DONE.** `configs/phase0_physioomni_config.yaml` — key sections:
- `embedding.*` — extraction params (checkpoint, output dir, `chunk_batch_size: 16`, datasets)
- `data.channel_candidates` — PhysioOmni slot → HDF5 channel name priority order, including SHHS's 1-EEG-channel special case (handled in code, not config)
- `dataset.*` — same `task_subject_dir`/`split_seed: 42`/split ratios as SleepFM/OSF, `min_recording_patches: 480` (240m in 30s-epoch units)
- `model.*` — `input_dim: 500`, `hidden_dim: 128`, `num_layers: 1` (architecture held constant vs. SleepFM/OSF — only `input_dim` changes, preserving the "only encoder changes" comparison principle)
- `training.*` — same placeholder hyperparameters as OSF's starting point (`epochs: 40`, `lr: 1e-4`, `early_stopping_patience: 10`) — not yet tuned from a real pilot
- `logging.results_dir` — `/scratch/boshra95/psg/unified/results/phase0_physioomni`

---

## Step 4 — Training

**Status: DONE (script + job script), CPU-smoke-tested end-to-end on real
data; full sweep not yet run (checklist 1.11).**

```bash
# Tiny CPU debug run (mirrors the VSCode "🫀 PhysioOmni Phase1 Step4:
# Train Sweep DEBUG (sex_binary, lstm, 30s, CPU)" config)
python scripts/train_physioomni_context_sweep.py \
    --config configs/phase0_physioomni_config.yaml \
    --context 30s --datasets apples shhs \
    --batch-size 2 --no-wandb --cpu
```

**Output:** `{results_dir}/{task}_{head_type}/context_{L}/{best_model.pt,metrics.json,training_curves.csv}`,
`{results_dir}/{task}_{head_type}/summary.csv` — `results_dir` =
`/scratch/boshra95/psg/unified/results/phase0_physioomni`. Identical schema
to SleepFM's/OSF's `metrics.json`/`summary.csv`, minus
`zero_modality_indices` (no modality-group ablation feature here either).

**Verified 2026-08-18** against an 8-apples + 8-shhs population (16
subjects total — the minimum found by simulating the exact split logic to
guarantee both classes land in val for `sex_binary`): `Items — train: 55 |
val: 10 | test: 15`, val AUROC a real number (0.52, not NaN), so
`best_model.pt` saved correctly. Checkpoint resume was also exercised live
(picked up a stale checkpoint and continued to early-stop correctly). Ran
to `Status: SUCCESS — all context lengths completed.` **Test-split metrics
are degenerate at this tiny population (expected, not a bug)** — will
resolve naturally once the full extraction (checklist 1.10) is done.

**Known pre-existing failure mode** (also seen in OSF's own Stage 1 build):
if a val split has too few subjects to include both classes, `roc_auc_score`
returns NaN, and since `NaN > -inf` is `False` in Python, the `improved`
flag never triggers, so `best_model.pt` is never saved — later causing a
`FileNotFoundError` at inference time. **Fix is always "extract enough
subjects for that task's datasets," never a code change.**

**GPU job:**
```bash
sbatch --export=ALL,TASK=sex_binary,HEAD=lstm jobs/train_physioomni_context_sweep_gpu.sh
```
Same auto-resume mechanism as SleepFM's/OSF's job scripts
(`--signal=B:USR1@120` bash trap + `resume.pt`), same status-JSONL
convention, logs to `logs_physioomni/`. **Not yet submitted for real** as
of 2026-08-18 — the underlying training logic is verified (above), but the
SLURM wrapper itself (auto-resume trap, status logging) hasn't been
exercised through an actual `sbatch` job yet, only syntax-checked
(`bash -n`). Structurally identical to OSF's already-battle-tested job
script, so no issues expected, but worth watching the first real run.

---

## Step 5 — Inference

**Status: DONE (script + job script), CPU-smoke-tested end-to-end against
a real trained checkpoint; full sweep not yet run (depends on 1.11).**

```bash
# CPU debug run against the checkpoint from the Step 4 debug run
python scripts/infer_physioomni_subject_windows.py \
    --config configs/phase0_physioomni_config.yaml \
    --task sex_binary --task-type seq2label --head lstm --context 30s \
    --datasets apples shhs --split val --cpu
```

**Output:** `{results_dir}/inference/{task}_{head_type}/context_{L}/{split}_windows.parquet`
— columns `subject_id, dataset, window_idx, true_label, pred_label,
prob_class0…prob_classN` (plus `anchor_patch_end` for seq2seq tasks —
none currently in scope). Identical schema to SleepFM's/OSF's inference
output.

**Verified 2026-08-18**: `Dataset items: 1,796 (subjects: 2)` →
`Saved 1,796 rows → .../val_windows.parquet`, `Segment accuracy: 50.84%`.
Parquet inspected directly: correct 7-column schema, zero NaNs.

**GPU job:**
```bash
sbatch --export=ALL,TASK=sex_binary,TASK_TYPE=seq2label,HEAD=lstm,CONTEXTS="30s 10m 40m 80m 120m 240m" \
    jobs/infer_physioomni_subject_windows_gpu.sh
```
Same auto-resume mechanism, same status-JSONL convention as training's job
script, logs to `logs_physioomni/`. **Not yet submitted for real** — same
caveat as Step 4's GPU job (underlying logic verified, SLURM wrapper only
syntax-checked so far).

**Batch-size auto-scaling note:** kept OSF's own formula and reference
point unchanged (`_ref_bs=64`, `_ref_N=480`) rather than re-deriving it —
PhysioOmni's dataset uses the exact same token unit as OSF's (one row per
30s epoch), so `_ref_N=480` (240m) transfers directly. Not yet GPU-verified
either way (same open caveat OSF's own script comment carries).

---

## Step 6 — Command Generator (`gen_commands_physioomni.py`)

**Status: DONE**, smoke-tested against the real Step 4/5 checkpoints; no
real GPU sweep has run yet (depends on 1.10's full extraction).

Same role as SleepFM's `scripts/gen_commands.py` / OSF's
`scripts/gen_commands_osf.py`: reads `experiments/v2_physioomni_registry.yaml`
and prints ready-to-run `sbatch`/`python` commands, rather than
hand-writing them. Structural fork of `gen_commands_osf.py`, same pipeline
logic.

```bash
# List all 12 tier-1 experiments (4 tasks × 3 heads) and their status
python scripts/gen_commands_physioomni.py list

# Train sbatch command for one context (auto-computes wall time, batch/accum)
python scripts/gen_commands_physioomni.py train sex_binary_lstm --context 30s

# Inference sbatch command (auto-discovers trained contexts)
python scripts/gen_commands_physioomni.py infer sex_binary_lstm --split val

# File-level status / job history for one experiment
python scripts/gen_commands_physioomni.py status sex_binary_lstm
python scripts/gen_commands_physioomni.py runs sex_binary_lstm
```

**Registry:** `experiments/v2_physioomni_registry.yaml` — 12 tier-1
entries (same datasets/contexts/batch_size/lr as `v2_registry.yaml`, since
PhysioOmni is compared against `phase0_v3`), with
`results_dir`/`inference_dir` pointed at `.../results/phase0_physioomni`
and `python_bin: /home/boshra95/physioomni_env/bin/python`.
**`apnea_binary` is deliberately absent** — no respiratory pathway.
`sleep_staging` is deferred, same as OSF.

**Subcommands kept** (same as `gen_commands_osf.py`): `list, probe-batch,
train, infer, analyze, build-heatmap, collect, threshold-tuning, status,
runs`. `analyze`/`build-heatmap`/`collect`/`threshold-tuning` call the
same underlying scripts as SleepFM's/OSF's pipelines
(`analyze_windows.py`, `build_heatmap_df.py`, `collect_results_v2.py`,
`apply_threshold_tuning.py`) **unmodified** — backbone-agnostic, same as
for OSF.

**Subcommands deliberately dropped**: all figure/table subcommands
(`iso-plots, saturation, scaling-laws, calibration, window-position,
subject-consistency, task-comparison, cohort-saturation, precision-recall,
subject-kstar, table-1..table-10`) — same reasoning as OSF's generator:
these wrap `plot_*.py`/`make_table*.py` scripts, superseded by notebooks
for the current paper. Once PhysioOmni results exist, feed them into
notebooks the same way SleepFM's/OSF's are.

**Known gaps** (same shape as OSF's generator):
- `probe-batch` is schema parity only — no PhysioOmni experiment uses
  `batch_mode: memory_bounded` yet, and `jobs/find_batch_size_physioomni_gpu.sh`
  doesn't exist.
- Wall-time lookup tables (`_TRAIN_HOURS`/`_INFER_HOURS_PER_CTX`) are
  placeholder copies of OSF's own (themselves not yet calibrated) — not
  GPU-calibrated for PhysioOmni. An underestimate just costs one extra
  auto-requeue, not lost work.
- No rorqual variant — only Fir job scripts exist for PhysioOmni so far.

---

## Step 7 — Running the Full Stage 1 Sweep

**Prerequisite done (checklist 1.10, 2026-08-19) — this is the actual next
step, pure job submission from here, no code left to write for the frozen
round.** Mirrors `docs/EXPERIMENTS_GUIDE.md`'s "Submitting Jobs" pattern
exactly, just pointed at `gen_commands_physioomni.py` and looped over all
12 tier-1 experiments instead of one.

Full embeddings extracted for all 4 datasets — 14,993/14,994 subjects
(99.99%, see Step 1 above for the one known gap). Verify any time with
`find /scratch/boshra95/psg/unified/embeddings/physioomni_30sec -name '*.npy' | wc -l`
(should read 14,993).

```bash
cd /home/boshra95/NSRR-tools-omni

PHYSIOOMNI_EXPS="
  sex_binary_lstm sex_binary_transformer sex_binary_mean_pool
  sleep_efficiency_binary_lstm sleep_efficiency_binary_transformer sleep_efficiency_binary_mean_pool
  bmi_binary_lstm bmi_binary_transformer bmi_binary_mean_pool
  age_class_lstm age_class_transformer age_class_mean_pool
"

# 1. Train — submits one sbatch job per (experiment, context), up to 72 total.
#    Submit a few at a time and watch the queue if worried about hitting a
#    job-count/priority limit — no need to fire all 72 at once.
for exp in $PHYSIOOMNI_EXPS; do
    python scripts/gen_commands_physioomni.py train $exp | bash
done

# 2. Monitor
python scripts/gen_commands_physioomni.py list                     # status of all 12 at a glance
python scripts/gen_commands_physioomni.py runs sex_binary_lstm      # job history for one
python scripts/gen_commands_physioomni.py status sex_binary_lstm    # file-level detail for one
sq                                                                  # raw queue state

# 3. Inference — once an experiment's contexts finish training
for exp in $PHYSIOOMNI_EXPS; do
    python scripts/gen_commands_physioomni.py infer $exp | bash
done

# 4. Analysis (local, no GPU — activate physioomni_env first)
source /home/boshra95/physioomni_env/bin/activate
for exp in $PHYSIOOMNI_EXPS; do
    echo "=== START $exp $(date) ==="
    python scripts/gen_commands_physioomni.py analyze $exp --plot | bash
    echo "=== END $exp $(date) ==="
done 2>&1 | tee analysis_physioomni_stage1.log

# 5. Collect all results into flat CSVs
python scripts/gen_commands_physioomni.py collect $PHYSIOOMNI_EXPS | bash
```

**A generated train command looks like** (same shape as SleepFM's/OSF's —
see `EXPERIMENTS_GUIDE.md`'s "Submitting Jobs" section for the field
reference):
```bash
TASK=sex_binary TASK_TYPE=seq2label HEAD=lstm CONTEXT=30s \
  DATASETS="apples shhs" BATCH_SIZE=32 ACCUM_STEPS=1 LR=1e-4 \
  CONFIG=configs/phase0_physioomni_config.yaml LOGS_DIR=/home/boshra95/NSRR-tools-omni/logs_physioomni \
  sbatch --requeue --time=01:30:00 \
    --output=.../logs_physioomni/train_sex_binary_lstm_30s_lr1e-4_%j.out \
    --error=.../logs_physioomni/train_sex_binary_lstm_30s_lr1e-4_%j.err \
    /home/boshra95/NSRR-tools-omni/jobs/train_physioomni_context_sweep_gpu.sh
```

**Not included** (deliberately, per Step 6's design decision): saturation
curves, iso-compute plots, scaling-laws, calibration, and the other
figure/table subcommands. Once `collect` has produced `training.csv`/
`analysis.csv` under `.../phase0_physioomni/collected/`, feed those into
notebooks the same way SleepFM's/OSF's collected results are.

---

## Step 8 — Stage 2 (LoRA fine-tuning) (not yet implemented)

**Status: not started.** Corresponds to
`docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md`'s Phase 2 checklist
(2.1–2.6), all still unchecked as of 2026-08-18. Do not start this without
being asked — the frozen round (Steps 0–7 above) needs to finish first.

**Expected shape, by analogy with OSF's own Stage 2**
(`docs/OSF_EXPERIMENTS_GUIDE.md` Step 8 — read that section for the full
detail of what this will look like): wrap PhysioOmni's 4 encoders with
`peft` LoRA adapters, train them jointly with the sequence head on raw
signal loaded live per window (no precomputed-embeddings step), staged
LP-FT (warm-start head from Stage 1, then fine-tune LoRA+head together).
**Open design question not yet resolved** (plan checklist 2.1): PhysioOmni
has 4 *independent* encoders, not OSF's single backbone — how LoRA wraps
across 4 separate `NeuralTransformer` instances needs its own design pass,
not a direct copy of OSF's single-encoder wrapping.

This section will be filled in with real commands/paths/verified output
once Phase 2 implementation actually starts, same as every other section
in this guide.

---

## Checkpoint Resume and Auto-Requeue

Identical mechanism to SleepFM's/OSF's (see `EXPERIMENTS_GUIDE.md`'s
section of the same name for the full explanation) — two distinct layers:
1. **Wall-time-triggered resume**: `#SBATCH --signal=B:USR1@120` + a bash
   trap in `jobs/train_physioomni_context_sweep_gpu.sh` that kills Python
   cleanly and resubmits itself via `sbatch`, picking up from `resume.pt`
   (saved every epoch by `train_physioomni_context_sweep.py`).
2. **Node-failure requeue**: SLURM-native `--requeue`, supplied at the
   *initial* `sbatch` invocation by `gen_commands_physioomni.py` (see Step
   6 above), not baked into the job script itself.

---

## Job Run History and Tracking

Same JSONL status-file convention as SleepFM/OSF:
`logs_physioomni/status/train_{task}_{head}[_{run_tag}]_{context}_lr{lr}.jsonl`,
one line per `STARTED`/`TIMEOUT_REQUEUED`/`SUCCESS`/`FAILED` event. Query
via `python scripts/gen_commands_physioomni.py status [<exp_id>]`/`runs
[<exp_id>]` (see Step 6 above) — mirrors `python scripts/gen_commands.py
status`/`runs` exactly, just pointed at
`experiments/v2_physioomni_registry.yaml`. No real GPU training/inference
jobs have run yet as of 2026-08-18 (only the extraction job has), so
`logs_physioomni/status/train_*`/`infer_*` is still empty.
