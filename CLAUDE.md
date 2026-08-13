# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

**Never add `Co-Authored-By: Claude ...` (or any Claude/Anthropic attribution) to commit
messages in this repo.** The user does not want Claude listed as a collaborator on
GitHub. This applies to every commit, not just ones the user is looking at right now.

**Keep this file updated.** Whenever a significant decision is made about the
TSFM baseline comparison — model selection, a Plan A/B/C usage mode
finalized for a specific model, a code-level finding that changes
integration effort, a run started or completed on the cluster — update the
relevant section here so a future session (local or on-cluster) starts with
accurate context.

---

## What This Repo Is

`NSRR-tools` is the experiment/analysis codebase for an npj Digital Medicine
paper studying how much overnight PSG temporal context a frozen SleepFM
foundation-model encoder needs for near-optimal clinical prediction, across
seven tasks (sex, sleep efficiency, BMI, age group, apnea severity, sleep
staging, depression) and four NSRR cohorts (SHHS, MrOS, APPLES, STAGES,
~16,000 subjects). The paper itself lives in a separate repo,
`npj_digital_medicine_submission/` (see that repo's own `CLAUDE.md` for
paper-specific conventions — section order, acronym rules, figure
numbering, etc.).

**Always read `docs/EXPERIMENTS_GUIDE.md` first** for anything about the
existing SleepFM-based training/inference/analysis pipeline (the V3
protocol, config files, job submission, results collection). This
`CLAUDE.md` does not repeat that guide — it exists specifically to
document the newer TSFM baseline comparison effort described below.

**Data/code split:** raw PSG signals and all derived HDF5/embedding data
live only on the Compute Canada cluster, under `/scratch/boshra95/...`
(never locally). Code repos are cloned both locally (for planning/editing)
and on the cluster (for anything that needs to touch signal data). See
"Cluster Execution Guidance" below.

---

## Repository Map

This section is a code-verified orientation map (read directly from source,
not inferred from docstrings) — it complements `docs/EXPERIMENTS_GUIDE.md`
rather than duplicating it. Read the guide for *how to run things*; read
this section for *where things live and how the pieces fit together*.

### Housekeeping: archived content (as of 2026-08-10)

The repo root and `docs/` used to be cluttered with early-implementation
(Feb–Mar 2026) debug scripts and status docs. These have been moved:

- **`archived_files/`** — old root-level `.md`/`.py` files from the initial
  EDF-adapter/annotation-debugging phase (e.g. `debug_shhs_merge.py`,
  `STAGES_ID_MISMATCH_DEBUG_GUIDE.md`). Historical only — nothing here is
  imported by current code or referenced by current docs. Safe to ignore
  unless specifically researching early preprocessing decisions.
- **`docs/archive/`** — a *larger* and more mixed set: some genuinely
  obsolete planning docs, but also several docs that `docs/EXPERIMENTS_GUIDE.md`
  itself treats as authoritative reference material and still links to by
  bare filename (e.g. `TRAINING_PROTOCOL_FIXES.md`, `RESULTS_COLLECTION.md`,
  `SOTA_COMPARISON_AND_ABLATIONS.md`, `cohort_filter.md`,
  `POSTHOC_THRESHOLD_TUNING.md`, `sleep_staging_design.md`, `PAPER_TABLES.md`,
  `PAPER_PLAN.md`). **Those links are now stale** — the content is still
  correct and worth reading, it just moved. If `docs/EXPERIMENTS_GUIDE.md`
  or this file references `docs/SOMETHING.md` and it isn't in `docs/`,
  check `docs/archive/SOMETHING.md` before assuming it's gone.
- A handful of loose root-level files (`test_preprocessing.py`,
  `test_sleepfm_compatibility.py`, `profile_signal_processing.py`, a few
  `test_*.csv` fixtures, `assemble_figures.log`) weren't archived but are
  also superseded by `scripts/` and `docs/EXPERIMENTS_GUIDE.md` — harmless,
  no action needed, just don't treat them as current guidance.

### `src/nsrr_tools/` — the installable package

```
core/
  signal_processor.py      EDF → HDF5 preprocessing (resampling, filtering, channel extraction)
  annotation_processor.py  Sleep-stage/event XML annotation parsing → aligned label arrays
  channel_mapper.py        Canonical channel-name resolution (reads configs/channel_definitions.yaml)
  modality_detector.py     Maps resolved channels to the 4 SleepFM modality groups (BAS/RESP/EKG/EMG)
  metadata_builder.py      Builds unified per-subject metadata across datasets
datasets/
  base_adapter.py          Abstract per-dataset adapter interface
  {apples,shhs,mros,stages}_adapter.py   One adapter per NSRR cohort (raw file layout → common schema)
  context_window_dataset.py   ContextWindowDataset — the PyTorch Dataset every training/inference
                               script uses. SEE "Code reuse assessment" below — this class is
                               SleepFM-embedding-shape-specific ([T,4,128], hardcoded constants
                               N_MODALITIES=4/EMBED_DIM=128/FLAT_DIM=512), not generic.
models/
  sequence_head.py          MeanPoolHead, LSTMHead, TransformerHead + build_head(cfg) factory.
                             Dim-agnostic (input_dim is just a constructor arg) — fully reusable
                             for any backbone's embeddings, see "Code reuse assessment" below.
targets/
  extraction_utils.py       Shared helpers for the per-dataset extract_targets_*.py scripts
utils/
  config.py                 YAML config loading/merging for experiment configs
  mount_utils.py             Cluster filesystem path helpers (SCRATCH/HOME expansion, etc.)
```

### `scripts/` — ~70 files, grouped by purpose

```
Preprocessing / metadata (cluster, GPU not needed):
  preprocess_signals.py, preprocess_single_subject.py   EDF→HDF5 (called by jobs/preprocess_*.sh)
  extract_metadata.py, extract_nsrr_channels.py          Per-dataset metadata/channel inventory
  xml_to_csv.py, xml_to_csv_simple.py                    Annotation XML → CSV
  validate_hdf5.py, extract_sample_edfs.sh                Spot-checking utilities

Target extraction (labels):
  extract_targets_{apples,mros,shhs,stages}.py            Per-dataset clinical label extraction
  create_master_targets.py, create_task_subject_lists.py  Unify into targets_v2/, build per-task subject lists

Embeddings (GPU, cluster):
  extract_sleepfm_embeddings.py                           HDF5 → per-subject [T,4,128] .npy (SleepFM forward pass)
  scan_nan_embeddings.py                                  Sanity-check embeddings for NaNs

Core pipeline (GPU for train/infer, CPU for analyze):
  gen_commands.py            THE command generator — see subcommand list below. Always go through
                              this rather than hand-writing sbatch/python invocations.
  train_context_sweep.py     Training entry point (see "Code reuse assessment" below)
  infer_subject_windows.py   Inference entry point (see "Code reuse assessment" below)
  analyze_windows.py         K-sweep metrics (local, no GPU)
  build_heatmap_df.py        Iso-compute heatmap DataFrame construction
  eval_checkpoint.py         Ad hoc single-checkpoint evaluation
  find_batch_size.py         Batch-size probing (gen_commands.py probe-batch)
  collect_results.py (v1, superseded), collect_results_v2.py (current — use this one)
  apply_threshold_tuning.py  Post-hoc decision-threshold tuning (binary tasks)
  analyze_common_eval_set.py, analyze_windows.py --k-dense   Dense-K / cross-context comparisons
  debug_nan.py                NaN debugging for training runs

Plotting (local, no GPU) — ⚠️ **NOT the current source of the paper's actual
figures, see caveat below**:
  plot_saturation.py, plot_iso_compute.py, plot_scaling_laws.py, plot_calibration.py,
  plot_window_position.py, plot_subject_consistency.py, plot_cohort_saturation.py,
  plot_precision_recall.py, plot_subject_kstar.py, plot_task_comparison.py
  plot_modality_bar.py, plot_channel_comparison.py, plot_aggregate_scaling.py
      (these 3 are cross-round: called directly, not via gen_commands.py — read collected CSVs)

Paper tables (local, no GPU) — ⚠️ **used only to double-check numbers, see
caveat below**:
  make_table1_peak_auroc.py, make_table2_lstar.py, make_table3_kgrid.py,
  make_table4_sensitivity.py, make_table5_heads.py, make_table6_modality.py,
  make_table9_cohort.py, make_table10_ci.py
      (also reachable via gen_commands.py table-1 .. table-10 subcommands)

Orchestration shell wrappers:
  run_analysis.sh    Full 13-step analysis+plot pipeline for a task/head list (see guide §"Analysis and Plotting")
  run_figures.sh     ⚠️ Documented in the guide but NOT how current figures are made — see caveat below
  gen_tables.sh      Regenerates the make_table*.py outputs (double-check reference only, see caveat below)
  assemble_figures.py   Composite multi-panel figure assembly — also superseded, see caveat below
  repo_sync.py       Cross-cluster git sync helper

Dataset-adapter unit tests (current, not archived debug scripts):
  test_apples_adapter.py, test_mros_adapter.py, test_shhs_adapter.py, test_stages_adapter.py,
  test_channel_config.py, test_context_window_dataset.py
```

**⚠️ Important workflow caveat (confirmed by the user, 2026-08-10): the
`plot_*.py` scripts, `run_figures.sh`, and `assemble_figures.py` are
documented in `docs/EXPERIMENTS_GUIDE.md` as the figure-generation path,
but that is no longer how the actual paper figures are produced.** The
user now maintains figure-generation code directly in notebooks at
`results/paper_figures/notebooks_npj/` (one notebook per figure, e.g.
`main_fig2_kvsk.ipynb`, `sfig14_task_landscape.ipynb`) — the plotting
logic was originally copied out of the `plot_*.py` scripts into these
notebooks, and **the notebooks, not the scripts, are what's edited now**.
When paper figure order changes, the user renames/edits the notebooks
directly to keep filenames and numbering correct — `scripts/plot_*.py`
and `scripts/run_figures.sh` are not kept in sync with this and should be
treated as historical/reference only, not as the current figure source.
Outputs land in `results/paper_figures/final_npj/`. (There's also an older
`results/paper_figures/notebooks/` + `final/` pair from the TBME
submission, and `results/paper_figures/FIGURE_PIPELINE.md` documents that
older TBME-era layout, not `notebooks_npj/` — don't follow it for current
figure work.) **Similarly, the `make_table*.py` scripts / `table-N`
subcommands / `gen_tables.sh` do not produce the paper's actual table
content** — the tables in `npj_main.tex` are written and edited by hand;
these scripts are used only to spot-check specific numbers against the
collected results CSVs, not as a generation pipeline to run and paste
from. If asked to "regenerate the tables/figures for the paper," these
scripts are the wrong target — ask where the notebook/tex edits should
happen instead, or check `notebooks_npj/` and `npj_main.tex` directly.

### `configs/` — YAML configs

| File | Purpose |
|---|---|
| `phase0_v3_config.yaml` | Fast-channel seq2label training (hidden=128, layers=1, val_auroc monitor) — **the active baseline config** |
| `phase0_v3_staging_config.yaml` | Fast-channel sleep staging (hidden=256, layers=2, val_kappa monitor) |
| `phase0_v3_full_config.yaml` / `phase0_v3_full_staging_config.yaml` | Full-channel counterparts |
| `phase0_v3_abl_config.yaml` | Modality-ablation config (reuses fast-channel embeddings) |
| `preprocessing_params.yaml` / `preprocessing_params_full.yaml` | EDF→HDF5 channel-set strategy (`sleepfm` vs `sleepfm_full`) |
| `channel_definitions.yaml` | Canonical channel-name alias resolution (read by `channel_mapper.py`) — `channel_definitions_old.yaml` is a superseded copy, kept for reference only |
| `modality_groups.yaml` | BAS/RESP/EKG/EMG channel priority lists per group |
| `paths.yaml` | Canonical `${SCRATCH}`/`${HOME}`-relative path templates |
| `target_extraction.yaml` / `target_extraction_v2.yaml` | Per-task label thresholds/source-column mapping for `extract_targets_*.py` |
| `phase0_config.yaml` / `phase0_v2_config.yaml` | Archived protocol versions — do not use for new work |

### `experiments/` — registries (read by `gen_commands.py`)

- `v2_registry.yaml` — fast-channel, the default (no `--registry` flag needed)
- `v2_full_registry.yaml` — full-channel (`--registry experiments/v2_full_registry.yaml`)
- `v2_ablation_registry.yaml` — modality ablation (`--registry experiments/v2_ablation_registry.yaml`)

Each entry maps one `{task}_{head}` (or `{task}_{head}_{run_tag}`) experiment
ID to `task, task_type, num_classes, head, datasets, contexts, batch_size,
lr, run_tag, n_size, tier`. See `docs/EXPERIMENTS_GUIDE.md` §"Experiment
Registry and Command Generator" for the full field reference and how to add
new entries.

### `jobs/` — SLURM submission scripts (Compute Canada)

Two cluster targets exist with slightly different SLURM directives — **check
which cluster you're on before submitting**:
- **Default (Fir)**: `jobs/{train_context_sweep,infer_subject_windows}_gpu.sh` — uses `--exclude=fc11006,fc11013,fc11010` (bad nodes), no `--partition` needed.
- **Rorqual**: `jobs/{train_context_sweep,infer_subject_windows,find_batch_size}_gpu_rorqual.sh` — requires `--partition=gpubase_bygpu_b3`, no `--exclude` (different node names).

Common SLURM settings across GPU jobs: `--account=def-forouzan_gpu`,
1×H100 MIG 10GB slice (`nvidia_h100_80gb_hbm3_1g.10gb:1`), Python env
activated via `source /home/boshra95/sleepfm_env/bin/activate` (**not**
this repo's local `.venv` — that's for local/editing use only, never on
the cluster for training/inference). Auto-requeue via
`--signal=B:USR1@120` + `--requeue`; see `docs/EXPERIMENTS_GUIDE.md`
§"Checkpoint Resume and Auto-Requeue" for the mechanism.

`jobs/README.md` covers preprocessing job usage specifically (already
completed for the existing SleepFM pipeline; relevant again if a new
backbone needs its own preprocessing pass).

### Logs

`logs_v3/` (fast-channel), `logs_v3_full/` (full-channel), `logs_v3_abl*/`
(ablation, including one archived-run directory
`logs_v3_abl_arch256_20260627/` from the wrong-architecture rerun mentioned
in the guide), `logs_v2/` (archived protocol). Each has SLURM
`.out`/`.err` files plus a `status/*.jsonl` structured event log per
train/infer job (`STARTED/REQUEUED/TIMEOUT_REQUEUED/SUCCESS/FAILED`) —
query via `python scripts/gen_commands.py runs [<exp_id>]`.

### `gen_commands.py` — complete subcommand list (code-verified, not just the guide's prose)

`list, probe-batch, train, infer, analyze, build-heatmap, iso-plots,
saturation, collect, scaling-laws, calibration, window-position,
subject-consistency, task-comparison, cohort-saturation, precision-recall,
subject-kstar, threshold-tuning, table-1, table-2, table-3, table-4,
table-5, table-9, table-10, status, runs`. The `table-N` subcommands
(direct wrappers around the `make_tableN_*.py` scripts) aren't called out
in the guide's usage examples but are real and working.

### Paper table numbering caveat

`docs/EXPERIMENTS_GUIDE.md` cross-references "paper Table II/III/IV/V"
using **TBME-era roman numerals**. The active paper is now
`npj_digital_medicine_submission/npj_main.tex`, which uses **arabic
numbers in a different order** (6 tables total, in appearance order:
`tab:sweep, tab:saturation, tab:heads, tab:isocompute, tab:modality,
tab:tasks`). Don't trust the guide's roman-numeral table references for
precision — `grep '\label{tab:' npj_main.tex` in the paper repo to get the
current mapping if it matters.

---

## TSFM Baseline Model Comparison (in progress)

### Why this exists

The supervisor asked why SleepFM specifically was chosen as the frozen
encoder, and why we hadn't compared against recent general-purpose
time-series foundation models (TSFMs, e.g. Chronos) or other pretrained
physiological-signal foundation models. We agreed to add real baseline
comparisons rather than only defend the choice narratively.

**The full candidate survey, selection reasoning, and all code-verified
findings live in [`docs/TSFM_BASELINE_CANDIDATES.md`](docs/TSFM_BASELINE_CANDIDATES.md).
Read that file for the actual technical detail — this section is the
top-level plan and pointer, not a duplicate.**

### Selected models (first three, from the candidates doc's shortlist)

In priority order, cloned and being investigated:

1. **OSF** (On Pre-training and Scaling of Sleep Foundation Models, ICML
   2026) — the only other sleep-PSG-specific foundation model with a public
   checkpoint. Real contamination risk: OSF's pretraining data includes
   SHHS and MrOS, two of our four test cohorts — must be handled honestly
   in the paper (see candidates doc §2.1 for the exact caveat and mitigation).
2. **PhysioOmni** — the strongest genuinely multimodal *general
   physiological* (not sleep-specific) FM found, explicitly designed for
   missing-modality robustness, which conceptually parallels our own
   reduced/full-channel framing. Open question: whether it has a
   respiratory/airflow pathway at all (critical for the apnea task).
3. **MOMENT** — the strongest general-purpose TSFM for this use case
   (classification-native, unlike Chronos/TimesFM/Moirai, which are
   forecasting-native). MIT-licensed, documented multichannel-classification
   API, has its own PEFT/LoRA ECG tutorial.

If more models are wanted later (Chronos-2, TimesFM, UniShape, Lag-Llama,
single-modality EEG/ECG fallbacks, etc.), see the candidates doc — do not
start on any of them without being asked; the user said "if I want more,
I'll tell you."

**Repos cloned locally at `/Users/boshra/NSRR-workspace/`:**
- `OSF-Open-Sleep-FM/` — `git clone https://github.com/yang-ai-lab/OSF-Open-Sleep-FM.git`
- `PhysioOmni/` — `git clone https://github.com/935963004/PhysioOmni.git`
- `moment/` — `git clone https://github.com/moment-timeseries-foundation-model/moment.git`

These are read-only reference clones for planning and code inspection —
not modified, not part of the NSRR-tools git history.

### The three usage modes (per model, where applicable)

Every candidate model is evaluated against up to three usage modes,
matched to how well its native context length compares to our context-length
sweep (30s to 240min):

- **Plan A — native long context, no sequence head.** If a model can
  natively ingest a long raw (or lightly-downsampled) signal sequence
  directly, feed it the sequence at each of our context lengths and put a
  classification head straight on its output — no LSTM/Transformer/
  MeanPool sequence head needed on top. This is the "fairest" comparison
  mode when available, because it tests the backbone's own long-context
  capability rather than our aggregation machinery.
- **Plan B — short-segment embedder + our sequence head.** For models that
  only accept short segments (seconds to a couple of minutes), extract
  embeddings per segment and train our existing sequence heads (Bi-LSTM,
  Transformer, MeanPool — `src/nsrr_tools/models/sequence_head.py`) on top,
  exactly like we currently do with SleepFM. This is the fallback whenever
  Plan A isn't possible, and it's how SleepFM itself is used throughout the
  paper (SleepFM's own chunking is 5-minute, 300s segments).
  **Report the drawback explicitly in the paper**: whatever context length
  the model was natively evaluable at, plus the fact that a from-scratch
  sequence head was needed to go longer.
- **Plan C — feed pre-extracted embeddings as a long multivariate series.**
  For models that are classification/sequence-native but not built for raw
  signal, take SleepFM's own per-patch embeddings (or the candidate
  model's own short-segment embeddings) and feed the resulting embedding
  sequence into the candidate model treated as a generic multivariate time
  series over hours. This turns the "foundation model" into a pretrained
  sequence head over pre-extracted features rather than a raw-signal
  encoder — a different, but still informative, experiment ("does a
  general-purpose pretrained sequence model out-aggregate our own
  LSTM/Transformer heads over the same embeddings?").

  If a model runs out of memory/compute before reaching a target context
  length under Plan A or Plan C, **report the longest context that
  actually ran and state the resource ceiling explicitly** — do not
  silently omit longer points or extrapolate.

**Confirmed status (code-verified 2026-08-05, all three repos cloned and
read): none of OSF, PhysioOmni, or MOMENT support Plan A as released.**
- **OSF**: strictly per-30s-epoch, no cross-epoch/whole-night attention
  anywhere in the code (`osf/backbone/vit1d_cls.py` — positional-embedding
  table is sized to exactly one epoch). Plan B only.
- **PhysioOmni**: hard ~512-second (~8.5 min) positional-embedding ceiling
  per modality, bidirectional non-causal attention (masked-pretraining
  style, not built for long-context generation). Plan B only.
- **MOMENT**: native context is **512 timesteps ≈ 4 seconds at 128 Hz**
  (hard-fixed everywhere in the code — patch length 8, stride 8), far
  shorter than even our own 5-second SleepFM patches. Plan B only. Plan C
  (pre-extracted embeddings as input) is also unsupported for MOMENT — its
  `embed()` mode uses the same 512-timestep raw-amplitude pipeline as
  classification, not a generic multivariate-series mode.

**Practical consequence: all three models will be evaluated via Plan B for
every context length in the sweep** — there is no "native long context"
condition to report for any of the three first-round models. This should be
stated plainly in the paper (the comparison is "SleepFM vs. baseline-FM
embeddings + our sequence head," not "SleepFM vs. baseline-FM's own native
context handling") rather than implied to be a fairer test than it is. See
`docs/TSFM_BASELINE_CANDIDATES.md` §2.1-2.3 for the full code evidence
behind each of these three findings.

### Code reuse assessment (code-verified 2026-08-10) — what to reuse vs. reimplement

Since all three baselines are Plan-B-only (short-segment embedder + our own
sequence head), the natural question is how much of the *existing* SleepFM
pipeline (`src/nsrr_tools/`, `scripts/train_context_sweep.py`,
`scripts/infer_subject_windows.py`, `scripts/gen_commands.py`) can be
reused directly rather than rebuilt per backbone. Read directly from source:

- **`src/nsrr_tools/models/sequence_head.py` — fully reusable, as-is.**
  `MeanPoolHead`/`LSTMHead`/`TransformerHead` and the `build_head(cfg)`
  factory are dim-agnostic: `input_dim` is just a constructor argument, no
  SleepFM-specific assumption anywhere in the file. Any new backbone's
  embeddings, reshaped to `(B, N, D)` with a `(B, N)` padding mask, can go
  straight into these heads by setting `input_dim=D` in the config.
- **`src/nsrr_tools/datasets/context_window_dataset.py`
  (`ContextWindowDataset`) — NOT reusable unmodified.** It hardcodes
  SleepFM's embedding shape throughout: `N_MODALITIES=4`, `EMBED_DIM=128`,
  `FLAT_DIM=512` module-level constants, `.npy` files assumed shape
  `[T, 4, 128]`, reshape/pad logic built around that exact 4×128 layout.
  A backbone whose embeddings aren't shaped `[T, 4, 128]` (which is all
  three of OSF/PhysioOmni/MOMENT — each has its own embedding dim and no
  4-modality-group structure) needs either a parallel dataset class or a
  parameterized fork of this one (replace the three hardcoded constants and
  the reshape calls with values read from config/embedding metadata). This
  is the central adapter-engineering task for each new backbone.
- **`scripts/train_context_sweep.py` and `scripts/infer_subject_windows.py`
  — mostly backbone-agnostic, delegate embedding I/O entirely to
  `ContextWindowDataset`.** Neither script touches raw `.npy` shapes
  directly — both import `ContextWindowDataset`/`build_head` and otherwise
  just orchestrate training/checkpointing/inference. The **only**
  SleepFM-specific artifact in either file is the `_MODALITY_INDICES =
  {"BAS":0,"RESP":1,"EKG":2,"EMG":3}` dict used for the `--zero-modalities`
  flag (modality ablation) — irrelevant to the TSFM baselines, which have
  no 4-group structure to ablate. **Practical implication: once a
  backbone-appropriate dataset class exists, these two scripts should work
  with little to no modification** — checkpoint/resume, early stopping,
  overfit-phase, snapshot, and bootstrap-CI machinery all come for free.
- **`scripts/gen_commands.py` — no existing hook for a different backbone;
  a parallel command generator is the lower-risk path, not a retrofit.**
  The registry schema (`experiments/v2_registry.yaml` et al.) has no
  `backbone`/`model_family` field, and the wall-time lookup tables
  (`_TRAIN_HOURS`, `_INFER_HOURS_PER_CTX`) are calibrated specifically to
  SleepFM's compute profile per `(n_size, head)`. Retrofitting this file to
  support multiple backbones would touch a lot of load-bearing, working
  code for uncertain benefit. **Recommendation: write a small parallel
  generator (or a new registry file + a thin backbone-aware wrapper) for
  the TSFM runs**, rather than extending `gen_commands.py` itself — this
  should be a concrete decision point in the implementation plan, not
  something to improvise mid-implementation. The downstream `analyze`,
  `collect`, plotting, and table subcommands, which just read
  `results/collected/*.csv` and per-window parquets in a fixed schema, are
  more plausibly reusable once a new backbone's results land in the same
  `metrics.json`/`summary.csv`/parquet format `train_context_sweep.py`/
  `infer_subject_windows.py` already produce.

### Frozen vs. LoRA-fine-tuned conditions

Every model is run in two conditions: **frozen backbone** (embeddings only,
new head trained from scratch) and **LoRA fine-tuned** (backbone adapted
via low-rank adapters, head trained alongside). The **training procedure is
staged, not joint** — see `docs/TSFM_BASELINE_CANDIDATES.md` §6 for the full
justification (LP-FT literature, Kumar et al. 2022):

1. **Stage 1** — freeze the backbone, train only the new classification
   head. This *is* the frozen-embedding condition.
2. **Stage 2** — wrap the (Stage-1-warmed) model with
   `peft.get_peft_model(model, LoraConfig(target_modules=[...],
   modules_to_save=["classifier"]))`, continue training LoRA + head
   together. This is the LoRA condition.

`peft` is not currently a dependency anywhere in NSRR-tools — add it when
implementation starts (`pip install peft`, pin a version, add to
`requirements.txt`/`pyproject.toml` next to the other training deps).

### Honest comparison framing (do not soften these in the paper)

- **OSF**: pretraining-set contamination is real, precisely quantified,
  and highly non-uniform across cohorts — **2026-08-13 update: verified
  directly from the OSF paper text (Fig. 2's explicit 9-dataset list) and
  by exact subject-ID matching against OSF's own shipped
  `osf/splits/patient_pretrain_{train,valid,test}_ids.csv`, not just an ID
  pattern guess.**
  - **SHHS: severe, quantified contamination.** OSF's paper lists SHHS as
    one of five **in-domain/pretraining** cohorts (SHHS, NCHSDB, WSC,
    CCSHS, CFS). Exact-ID overlap check against our own SHHS test split
    (apnea_binary/80m, N=1,271): **990 (77.9%) were in OSF's pretrain
    *train* split, 124 (9.8%) in OSF's pretrain *valid* split — 87.7%
    total direct pretraining exposure.** Only 125 (9.8%) were in OSF's own
    held-out `pretrain_test` split (genuinely unseen even by OSF's own
    authors) and 32 (2.5%) weren't found in any OSF split. **Any SHHS
    AUROC comparison against SleepFM is not a fair generalization test as
    currently computed.** Re-running AUROC on just the 125+32=157
    individually-unseen subjects did *not* shrink OSF's advantage on SHHS
    (if anything it grew slightly for the two tasks checked) — meaning
    subject-level exclusion alone doesn't fully resolve this: OSF's
    encoder was still trained on ~1,114 *other* SHHS subjects, so even
    "clean" individual subjects likely still benefit from the encoder
    having deeply learned that cohort's specific recording/device signal
    characteristics (a cohort-level, not just subject-level, contamination
    concern). Treat SHHS as **not a clean comparison cohort for OSF**,
    full stop — report it separately with this caveat, don't blend it into
    headline numbers.
  - **STAGES: confirmed clean, corrects an earlier weaker claim.** The
    earlier "very likely also in pretraining" note (based on a numeric-ID
    pattern guess) is **wrong** — OSF's paper explicitly lists only 9
    datasets total (SHHS, NCHSDB, WSC, CCSHS, CFS pretrain + MROS, MESA,
    CHAT, SOF out-of-domain); **STAGES appears nowhere in OSF's training
    corpus.** Confirmed a second way: searched all 6 of OSF's shipped
    split files for STAGES's actual site-code naming (`STNF`, `MSTR`,
    `GSDV`, `MAYO`, `MSNF`, `GSSW`, `GSBB`, `BOGN`, etc.) — zero matches
    anywhere. STAGES can be reported as directly as APPLES.
  - **MrOS: confirmed clean, same as before, now re-verified by exact-ID
    match (zero overlap) rather than inference from OSF's dataset list
    alone.** OSF's own paper explicitly holds MrOS/MESA/CHAT/SOF out of
    pretraining entirely, using them only for out-of-domain downstream
    evaluation — the same methodology we're implicitly relying on by
    treating MrOS as a fair cohort.
  - **APPLES: confirmed clean** — no mention anywhere in OSF's
    config/splits/README, re-verified with zero exact-ID matches.
  - **Practical implication for the npj comparison**: APPLES, MrOS, and
    STAGES are all genuinely fair comparison cohorts (zero subject- or
    even cohort-level OSF exposure). SHHS is not, at any granularity —
    report it separately, caveated, never blended into a pooled/headline
    AUROC number. See `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s "Stage 1
    Results" section for the full per-cohort breakdown this produced.
- **PhysioOmni**: **confirmed, not just suspected — no respiratory/airflow
  pathway exists anywhere in the model or its pretraining data** (verified
  via `dataset.yaml` schema, `dataset.py` modality keys, and two dataset-prep
  scripts that actively discard respiratory channels from source EDFs even
  when available). **Apnea must be excluded from the PhysioOmni comparison
  with a stated reason** — do not attempt a workaround (adding a
  respiratory pathway would require new-modality pretraining, not a
  fine-tuning adapter). Also has no LICENSE file anywhere in the repo —
  flag as an open question before any redistribution/derivative-use claim.
- **MOMENT** (and any other general-purpose TSFM): native context is 512
  timesteps ≈ **4 seconds** at 128 Hz (confirmed in code, not "tens of
  seconds" as first estimated) — Plan A is not available, and Plan C
  (pre-extracted embeddings as input) is also not a supported MOMENT
  feature. State which context lengths were actually reachable and via
  which plan in any results table. Separately: MOMENT's own reference LoRA
  tutorial omits `modules_to_save` for the classification head — using it
  as-is likely freezes the head during LoRA fine-tuning; fix this before
  trusting any LoRA-condition results from that recipe.
- General rule for all three: if frozen-embedding performance is reported
  without LoRA (or vice versa) for a given context length because a run
  didn't finish or wasn't attempted, say so in the table/caption — do not
  leave a cell that reads like a completed, unremarkable result. (This
  mirrors the standing instruction already applied throughout the paper's
  supplementary review: never report incomplete results as if complete.)

### Status

Model selection is done. All three repos are cloned locally and
code-verified — `docs/TSFM_BASELINE_CANDIDATES.md` §2.1-2.3 now reflect
confirmed (not web-search-assumed) input formats, checkpoint locations,
LoRA support/gaps, and contamination/modality-support findings; the
"Confirmed status" and "Honest comparison framing" subsections above are
current as of that pass.

**The detailed, code-level, cluster-runnable implementation plan for
OSF (model #1 of 3) is written: [`docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`](docs/TSFM_OSF_IMPLEMENTATION_PLAN.md).**
It covers the confirmed channel mapping (our full-channel HDF5 → OSF's
12-channel input — no raw EDF reprocessing needed), the decision to reuse
existing full-channel HDF5s and compare against `phase0_v3_full` (not the
paper-primary `phase0_v3`), new scripts/config/registry needed for Stage 1
(frozen encoder + our sequence heads) and Stage 2 (LoRA fine-tuning,
end-to-end, with a memory-mitigation fallback ladder), and a Step 0
verification checklist to run before the full sweep. First pass scope is
the 5 Tier-1 tasks (sex, sleep efficiency, BMI, age, apnea).

**PhysioOmni (model #2 of 3) also has a written plan now, code-verified
2026-08-13: [`docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md`](docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md)
— plan only, nothing implemented.** Verdict: proceed, but with real
caveats stated up front, not softened — the paper is arXiv-only (never
peer-reviewed), its sleep-relevant pretraining slice (CAP + Sleep-EDF,
~305 recordings) is tiny next to our ~16,000-subject cohort and next to
OSF's own pretraining scale, and **on its own best-fit downstream task
(HMC sleep staging), PhysioOmni's own reported number (0.7377 balanced
accuracy) does not beat its paper's own non-foundation-model baseline
(FeatFusion, 0.7478)** — a real reason to keep expectations modest, not a
reason to skip the comparison (a mixed/negative result here is still
informative for a paper about *why* SleepFM was chosen). Apnea is excluded
(no respiratory pathway anywhere in the model, confirmed at 4 independent
code locations). License is split, not simply "missing": the GitHub code
repo has no LICENSE file, but the HuggingFace weights repo
(`Weibang/PhysioOmni`) declares **CC-BY-4.0** explicitly (verified live via
the HF API) — both facts should be stated if this ships in the paper.
Channel-coverage news is better than OSF's: PhysioOmni needs only
EEG/EOG/ECG/EMG (no RESP), and the existing **fast-channel** `psg/` HDF5s
(the paper-primary tree) already carry everything needed — confirmed both
from `configs/preprocessing_params.yaml`/`modality_groups.yaml`'s
priority-order caps and from real HDF5 key listings for all 4 cohorts, not
assumed by analogy to OSF — so no reprocessing is needed, and the
comparison baseline is `phase0_v3` (paper-primary), not `phase0_v3_full`.
Normalization is the one place PhysioOmni is harder than OSF: it expects
raw amplitude scaled by `/100`, not z-scored data, so extraction needs to
invert our stored per-channel `normalization_stats` back to raw scale
first — mechanically free (the stats are already saved in every HDF5) but
**not a uniform V→µV conversion**: reading real stats shows `LOC`/`ROC`'s
recovered scale looks like volts while other channels already look
µV-scale, so the unit-correction has to be checked per channel, not
applied as one flat rule. Not yet empirically validated either way.
**Do not start implementing PhysioOmni until OSF's Stage 1 sweep and Stage
2 LoRA are done** — MOMENT (model #3) still doesn't have a plan doc yet;
write that only once asked, same rule.

---

## Cluster Execution Guidance

Raw PSG signals, our processed HDF5s, and all embeddings live only on the
Compute Canada cluster at `/scratch/boshra95/...` (see below) — nothing
touching real signal data can run locally. Any implementation work for the
TSFM baselines has to happen on the cluster once the detailed plan (next
step) is written. For an agent picking this up on the cluster:

**Repos to clone on the cluster, all at the same level (mirroring the local
layout at `/Users/boshra/NSRR-workspace/`):**
```
NSRR-tools/                        # this repo — the experiment/analysis codebase
OSF-Open-Sleep-FM/                 # github.com/yang-ai-lab/OSF-Open-Sleep-FM
PhysioOmni/                        # github.com/935963004/PhysioOmni
moment/                            # github.com/moment-timeseries-foundation-model/moment
npj_digital_medicine_submission/   # the paper repo — read for framing/consistency,
                                    # not for code; has its own CLAUDE.md
```

**Read in this order before writing any adapter/training code:**
1. `NSRR-tools/docs/EXPERIMENTS_GUIDE.md` — the existing SleepFM pipeline
   (V3 protocol, configs, job submission) that any new backbone's
   experiment should mirror as closely as possible for a fair comparison.
2. This file's "Repository Map" section above — code-verified map of
   `src/`, `scripts/`, `configs/`, `experiments/`, `jobs/`, and (critically)
   the "Code reuse assessment" under the TSFM section below, which says
   precisely what can be reused unmodified (`sequence_head.py`, largely
   `train_context_sweep.py`/`infer_subject_windows.py`) versus what needs a
   new implementation per backbone (`ContextWindowDataset`'s SleepFM-shape
   assumption, `gen_commands.py`'s lack of a multi-backbone hook).
3. `NSRR-tools/docs/TSFM_BASELINE_CANDIDATES.md` — per-model technical
   detail: confirmed input format, checkpoint location, license,
   classification/LoRA support, integration effort.
4. This file's "TSFM Baseline Model Comparison" section above — the Plan
   A/B/C usage modes and staged frozen/LoRA training procedure to follow.
5. **[`NSRR-tools/docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`](docs/TSFM_OSF_IMPLEMENTATION_PLAN.md)
   — the actual step-by-step plan to execute, for OSF (model #1).** Start
   here for concrete implementation work; the docs above are the context
   that plan was written from. PhysioOmni and MOMENT don't have plan docs
   yet — write those (following the same research-then-plan process) only
   once asked, after OSF's pipeline is validated.

**Cluster data paths (existing convention, from `docs/EXPERIMENTS_GUIDE.md`
and `docs/archive/RESULTS_COLLECTION.md` — moved during the 2026-08-10
docs cleanup, see "Repository Map" → "Housekeeping" above):**
- Raw downloads: `/scratch/boshra95/nsrr_downloads/{stages,shhs,apples,mros}/`
- Processed PSG (fast/reduced channels): `/scratch/boshra95/psg/{stages,shhs,apples,mros}/derived/`
- Processed PSG (full channels): `/scratch/boshra95/psg_full/...` (same structure)
- Unified metadata/targets: `/scratch/boshra95/psg/unified/{metadata,targets}/`
- SleepFM results (V3, the only valid protocol): `/scratch/boshra95/psg/unified/results/phase0_v3/`

New TSFM baseline results should follow the same `results/<protocol_or_model>/`
convention rather than inventing a new directory scheme — propose a
sibling directory name (e.g. `/scratch/boshra95/psg/unified/results/tsfm_osf/`)
in the implementation plan rather than deciding ad hoc mid-run.

**Compute note:** the cluster is the only place with GPU access and the
actual signal data — do not attempt to prototype adapter code against real
data locally. Local work is limited to repo cloning, code reading, and
planning-doc updates (as in this session).
