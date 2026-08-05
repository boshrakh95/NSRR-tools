# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

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

- **OSF**: pretraining-set contamination is real but not uniform across
  cohorts — code-verified from OSF's own shipped patient-ID splits. **SHHS
  is confirmed in OSF's pretraining set** (high risk, any SHHS AUROC
  comparison against SleepFM is not fair). **STAGES is very likely also in
  pretraining** (numeric-ID pattern match; needs one more confirmation
  step — cross-check a handful of numeric IDs against our local STAGES
  `subject_code` list before treating as certain). **MrOS was only in OSF's
  downstream/eval split, not pretraining** — lower risk than SHHS/STAGES,
  but OSF's own authors already benchmarked on it. **APPLES has no mention
  anywhere in OSF's config/splits/README — clean.** Report SHHS (and likely
  STAGES) results with an explicit contamination caveat; APPLES (and
  probably MrOS) can be reported more directly.
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
current as of that pass. **A detailed, step-by-step implementation plan
(one per model, code-level, cluster-runnable) has not been written yet** —
that is the next request, not something to start unprompted.

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
2. `NSRR-tools/docs/TSFM_BASELINE_CANDIDATES.md` — per-model technical
   detail: confirmed input format, checkpoint location, license,
   classification/LoRA support, integration effort.
3. This file's "TSFM Baseline Model Comparison" section above — the Plan
   A/B/C usage modes and staged frozen/LoRA training procedure to follow.
4. The detailed step-by-step implementation plan doc (once written — not
   yet present as of this section's writing).

**Cluster data paths (existing convention, from `docs/EXPERIMENTS_GUIDE.md`
and `docs/RESULTS_COLLECTION.md`):**
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
