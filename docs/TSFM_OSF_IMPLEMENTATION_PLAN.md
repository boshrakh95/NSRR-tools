# OSF Implementation Plan

> **Purpose**: Authoritative record of what's being built for OSF as TSFM
> baseline #1 (of 3 — PhysioOmni and MOMENT follow later), why, and current
> status. Read this top section for the plan and current checklist; the
> **Appendix** below (old §0-§12) has the full research/verification detail
> behind every choice here — consult it when you need the reasoning, not as
> your first read. Format mirrors `docs/archive/PHASE0_IMPLEMENTATION.md`
> (the SleepFM pipeline this is being compared against) — same spirit, not
> an identical structure, since the codebase differs.

---

## Overview

OSF ("On Pre-training and Scaling of Sleep Foundation Models", ICML 2026)
is being added as a frozen-then-LoRA-fine-tuned encoder baseline, compared
against SleepFM's existing `phase0_v3_full` context-length-sweep results.
Same research question as the SleepFM pipeline: does more temporal context
(30s → 240m) improve Tier-1 clinical prediction tasks, and how does OSF's
encoder compare to SleepFM's at matched context lengths and matched
subjects/splits? Implementation lives in **`NSRR-tools`** (this repo, on
the `osf-implementation` branch) — `OSF-Open-Sleep-FM` is a read-only
reference clone, not where we write code.

**Reference materials** — check before guessing about a model detail:
- **`/home/boshra95/related_work/OSF.pdf`** — the OSF paper (duplicate at
  `NSRR-tools/papers/2603.00190v1_OSF.pdf`). Code is primary source of
  truth for implementation details; the paper is the place to check *why*
  (pretraining objective, patchification rationale) when code alone
  doesn't explain it.
- **`NSRR-tools/output/channel_analysis/{apples,shhs,mros,stages}_channels.csv`**
  — raw per-subject EDF channel-label dumps behind
  `configs/channel_definitions.yaml`'s alias tables. Useful alongside (not
  instead of) directly sampling the real HDF5s at
  `/scratch/boshra95/psg_full/{dataset}/derived/hdf5_signals/*.h5`.
- `CLAUDE.md` (repo map), `docs/TSFM_BASELINE_CANDIDATES.md` §2.1 (OSF
  research background), `docs/EXPERIMENTS_GUIDE.md` (the SleepFM pipeline
  this mirrors), `docs/archive/PHASE0_IMPLEMENTATION.md` (SleepFM's
  finished version of this same doc — useful for comparison).
- **`docs/OSF_EXPERIMENTS_GUIDE.md`** — the OSF counterpart to
  `docs/EXPERIMENTS_GUIDE.md`, being filled in incrementally alongside this
  implementation (started 2026-08-11) — has the concrete
  commands/paths/verified-output-shapes for each step as it's built, so a
  future session can run/monitor the OSF pipeline without re-deriving
  anything. This plan doc is the "why," that one is the "how to actually
  run it."

## Status (2026-08-12)

**All of Phase 1's code is implemented — nothing left to build for Stage 1.**
From here it's job submission/monitoring, not implementation: a small GPU
test job (`54342713`) is running for checklist 1.8's verification; once
confirmed good, 1.9 (full extraction, 6 sharded GPU jobs) and 1.10 (the 90-run
Stage 1 sweep) are both pure `sbatch`/`gen_commands_osf.py` operations with
copy-pasteable commands now written into the checklist below and
`docs/OSF_EXPERIMENTS_GUIDE.md`'s Step 7 — no further code changes are
needed until Phase 2 (LoRA, `train_osf_lora.py`, not started).

---

## File Map

### Environment / Checkpoint
| Path | Purpose | Status |
|---|---|---|
| `/home/boshra95/osf_env` | Python 3.10 venv, OSF's trimmed/relaxed dependencies | ✅ DONE |
| `/home/boshra95/OSF-Open-Sleep-FM/pretrained_weights/osf_backbone.pth` | OSF-Base checkpoint (325MB, MIT license) | ✅ DONE |

### Configuration
| File | Purpose | Status |
|---|---|---|
| `configs/phase0_osf_config.yaml` | Master config — paths, channel mapping, hyperparams | ✅ DONE |

### Data pipeline
| File | Purpose | Status |
|---|---|---|
| `scripts/extract_osf_embeddings.py` | Step 1 — extract frozen embeddings from HDF5 PSG | ✅ DONE |
| `jobs/extract_osf_embeddings_gpu.sh` | SLURM job script for embedding extraction | ✅ DONE |
| `src/nsrr_tools/datasets/osf_context_window_dataset.py` | Step 2 — PyTorch dataset for context windows | ✅ DONE |

### Model
| File | Purpose | Status |
|---|---|---|
| `src/nsrr_tools/models/sequence_head.py` | LSTM/Transformer/MeanPool heads | Reused unmodified — no new file needed |

### Training
| File | Purpose | Status |
|---|---|---|
| `scripts/train_osf_context_sweep.py` | Step 4 — training loop, checkpointing | ✅ DONE |
| `jobs/train_osf_context_sweep_gpu.sh` | SLURM job script for training | ✅ DONE |

### Evaluation
| File | Purpose | Status |
|---|---|---|
| `scripts/infer_osf_subject_windows.py` | Step 5 — inference on all windows per subject | ✅ DONE |
| `jobs/infer_osf_subject_windows_gpu.sh` | SLURM job script for inference | ✅ DONE |

### Command generation
| File | Purpose | Status |
|---|---|---|
| `experiments/v2_osf_registry.yaml` | Experiment registry (5 tasks × 3 heads × 6 contexts) | ✅ DONE |
| `scripts/gen_commands_osf.py` | Generates train/infer/analyze/collect/status/runs commands from the registry | ✅ DONE |

### Stage 2 (LoRA)
| File | Purpose | Status |
|---|---|---|
| `scripts/train_osf_lora.py` | New end-to-end script, OSF encoder in the trainable graph | ⬜ TODO (checklist 2.1) |

---

## Encoder: OSF `vit_base`

- **Input per epoch**: `(B, 12, 1920)` — 12 fixed-order channels, 30s @ 64Hz.
- **Patchify**: `lead_wise=1` (2D `Conv2d`), `patch_size_ch=4`,
  `patch_size_time=64` → 3 channel-groups × 30 time-patches = **90 tokens**
  + 1 CLS token (checkpoint's `pos_embedding` shape `(1,91,768)` confirms
  this exactly).
- **Output used**: CLS `[B,768]` + mean-pooled patch tokens `[B,768]`,
  stacked → `[B,2,768]` per epoch — **not** the full 90-token sequence,
  and not CLS-only (decision made 2026-08-10, see Appendix §0).
- **Output NOT used**: the raw 91-token undivided sequence
  (`return_sequence=True` — do not pass this; it returns the wrong thing).
- **Checkpoint**: `osf_backbone.pth`, MIT license, 85,325,568 params,
  strict-load-verified (zero missing/unexpected `state_dict` keys).
- **No cross-epoch attention** — every forward pass sees exactly one 30s
  epoch; there is no windowing/chunking mechanism inside OSF itself (that's
  what `OSFContextWindowDataset` + the sequence head are for, same
  division of labor as the SleepFM pipeline).

---

## Channel Mapping

OSF expects exactly 12 channels, fixed order:
`ECG, EMG_Chin, EMG_LLeg, EMG_RLeg, ABD, THX, NP, SN, EOG_E1_A2, EOG_E2_A1, EEG_C3_A2, EEG_C4_A1`.

| OSF channel | Our HDF5 source (primary → fallback) |
|---|---|
| `EEG_C3_A2` | `C3-M2` (SHHS: generic `EEG`, duplicated — see below) |
| `EEG_C4_A1` | `C4-M1` (SHHS: generic `EEG`, duplicated — see below) |
| `EOG_E1_A2` | `LOC` |
| `EOG_E2_A1` | `ROC` |
| `ECG` | `EKG` → `ECG-L` |
| `EMG_Chin` | `CHIN` → generic `EMG` |
| `EMG_LLeg` | `LLEG` |
| `EMG_RLeg` | `RLEG` |
| `ABD` | `ABD` |
| `THX` | `Thor` |
| `NP` | `Airflow` |
| `SN` | `Snore` |

**Real per-cohort availability** (50-subject audit, 2026-08-10 — see
Appendix §1 for the full derivation):

| OSF slot | APPLES | SHHS | MrOS | STAGES |
|---|---|---|---|---|
| `ECG` | 100% | 100% | 100% | 90% |
| `EMG_Chin` | 100%† | 100%† | 100% | 100% |
| `EMG_LLeg` | 78% | **0%** | 100% | 56% |
| `EMG_RLeg` | **0%** | **0%** | 100% | 56% |
| `ABD` | 100% | 100% | **0%** | 100% |
| `THX` | 100% | 100% | 100% | 100% |
| `NP` | 100% | 68% | 100% | 100% |
| `SN` | 100% | **0%** | **0%** | 100% |
| `EOG_E1_A2`/`EOG_E2_A1` | 100% | 100% | 100% | 100% |
| `EEG_C3_A2`/`EEG_C4_A1` | 100% | **0%** | 100% | 100% |

†generic `EMG` fallback, not a channel literally named `CHIN`.

**SHHS decision (confirmed with the user 2026-08-10, provisional)**:
SHHS has no distinguishable C3/C4 — duplicate its single generic `EEG`
channel into both `EEG_C3_A2`/`EEG_C4_A1` slots, zero-fill
`EMG_LLeg`/`EMG_RLeg`/`SN`. If this hurts SHHS's OSF results too much,
revisit with a targeted SHHS-specific reprocessing pass later (not
committed to yet). Implemented in `extract_osf_embeddings.py`'s
`build_channel_candidates()` and **verified working** in the 2026-08-10
smoke test (§ Status above).

**EOG referencing**: STAGES's dominant raw label (`EOG_LOC-A2`,
`EOG_ROC-A1`) confirms OSF's expected contralateral-mastoid convention;
not uniformly guaranteed for every subject/cohort per
`channel_definitions.yaml`'s broader alias table. No NaNs in any smoke-test
subject so far — encouraging, not exhaustive proof. See Appendix §1.

---

## Stage 1 Results (partial, in progress — 2026-08-13)

**Scope so far**: 3 of 5 Tier-1 tasks (`sex_binary`, `sleep_efficiency_binary`,
`apnea_binary`), 2 of 3 heads (`lstm`, `transformer`) — trained, inferred,
and analyzed (`gen_commands_osf.py analyze`, K-sweep, test split). `bmi_binary`,
`age_class`, and `mean_pool` not run yet.

### Pooled AUROC, K=5, mean-prob aggregation, test split

| Task | Head | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|---|
| sex_binary | lstm | **OSF 0.906** / SF 0.840 | **0.932** / 0.871 | **0.941** / 0.894 | **0.950** / 0.885 | **0.943** / 0.894 | — |
| sex_binary | transformer | **0.907** / 0.825 | **0.933** / 0.854 | **0.946** / 0.892 | **0.958** / 0.921 | **0.963** / 0.929 | — |
| sleep_efficiency_binary | lstm | **0.714** / 0.704 | **0.739** / 0.695 | **0.754** / 0.722 | **0.779** / 0.751 | **0.794** / 0.767 | **0.824** / 0.810 |
| sleep_efficiency_binary | transformer | **0.706** / 0.694 | **0.731** / 0.709 | **0.765** / 0.751 | **0.790** / 0.783 | **0.801** / 0.798 | **0.841** / 0.825 |
| apnea_binary | lstm | **0.789** / 0.768 | **0.824** / 0.782 | **0.851** / 0.827 | **0.875** / 0.857 | **0.882** / 0.874 | — |
| apnea_binary | transformer | **0.787** / 0.774 | **0.832** / 0.799 | **0.858** / 0.857 | **0.902** / 0.888 | **0.910** / 0.900 | — |

(SF = SleepFM `phase0_v3_full`, same split protocol, same test subjects
where N matches — see caveat below. `sex_binary`/`apnea_binary` have no
240m row in either pipeline's collected results as of this writing.)

**Read naively, OSF beats SleepFM at every single context/task/head
combination above, by 0.4–8 pp AUROC.** That headline is misleading on its
own — see the per-cohort breakdown immediately below, which is the actual
basis for the verdict in each task's row of the summary table further down.

### Critical caveat: OSF's pretraining contamination risk (already flagged in `CLAUDE.md`)

OSF's own pretraining data includes **SHHS** (confirmed) and **very likely
STAGES** (not yet 100% confirmed — see Known Issues). **MrOS** was only in
OSF's downstream/eval split (lower risk). **APPLES is clean** (no mention
anywhere in OSF's own splits/config). A pooled, all-cohort AUROC number
cannot distinguish "OSF's encoder genuinely generalizes better" from "OSF
already memorized some of these subjects during pretraining" — it has to
be broken down per cohort. Did this at context=80m (mid-sweep, both
heads) directly from `test_windows.parquet` (mean-prob aggregation):

| Task | Head | Cohort | OSF AUROC | SleepFM AUROC | Δ | Contamination risk |
|---|---|---|---|---|---|---|
| sex_binary | lstm | apples | 0.957 | 0.783 | **+17.4pp** | clean |
| sex_binary | lstm | shhs | 0.949 | 0.899 | +5.0pp | high |
| sex_binary | transformer | apples | 0.970 | 0.846 | **+12.4pp** | clean |
| sex_binary | transformer | shhs | 0.957 | 0.933 | +2.4pp | high |
| sleep_efficiency_binary | lstm | apples | 0.755 | 0.701 | +5.5pp | clean |
| sleep_efficiency_binary | lstm | mros | 0.638 | 0.637 | ~0pp | low |
| sleep_efficiency_binary | lstm | shhs | 0.800 | 0.765 | +3.5pp | high |
| sleep_efficiency_binary | transformer | apples | 0.775 | 0.813 | **−3.8pp** | clean |
| sleep_efficiency_binary | transformer | mros | 0.660 | 0.733 | **−7.3pp** | low |
| sleep_efficiency_binary | transformer | shhs | 0.807 | 0.789 | +1.8pp | high |
| apnea_binary | lstm | apples | 0.849 | 0.865 | **−1.7pp** | clean |
| apnea_binary | lstm | mros | 0.791 | 0.871 | **−7.9pp** | low |
| apnea_binary | lstm | shhs | 0.896 | 0.869 | +2.7pp | high |
| apnea_binary | lstm | stages | 0.856 | 0.766 | +9.0pp | none (confirmed clean) |
| apnea_binary | transformer | apples | 0.862 | 0.916 | **−5.4pp** | clean |
| apnea_binary | transformer | mros | 0.832 | 0.897 | **−6.5pp** | low |
| apnea_binary | transformer | shhs | 0.920 | 0.899 | +2.1pp | high |
| apnea_binary | transformer | stages | 0.874 | 0.843 | +3.1pp | none (confirmed clean) |

### Precise contamination quantification (2026-08-13, supersedes the earlier general risk statement)

Cross-checked our own test-split subject IDs against OSF's own shipped
`osf/splits/patient_pretrain_{train,valid,test}_ids.csv` by exact ID
match (not inference from OSF's dataset list alone) — full detail in
`CLAUDE.md`'s "Honest comparison framing" section, summarized here:

- **SHHS is severely contaminated, precisely: 87.7% of our SHHS test
  subjects (1,114 of 1,271, apnea_binary/80m) were directly in OSF's own
  pretraining train+valid splits.** Only 157 (12.3%) were genuinely unseen
  by OSF (125 in OSF's own held-out `pretrain_test` split + 32 not found
  in any OSF split). Recomputed AUROC on just those 157 clean-subject
  subjects — **OSF's advantage over SleepFM did not shrink** (if anything
  grew slightly: sex_binary/lstm +5.0pp full → +7.1pp clean-only;
  apnea_binary/lstm +2.7pp full → +3.2pp clean-only). This means
  individual-subject filtering alone doesn't fully rescue SHHS as a fair
  comparison cohort — OSF's encoder was still trained on ~1,114 *other*
  SHHS subjects, so even individually-unseen SHHS subjects likely still
  benefit from the encoder having deeply learned that cohort's specific
  recording/device signal characteristics. **Treat SHHS as not usable for
  a fair OSF-vs-SleepFM comparison, at any subject-filtering granularity.**
- **STAGES is confirmed clean — this corrects the earlier "very likely
  also in pretraining" note, which was wrong.** OSF's paper explicitly
  names its 9 training datasets (SHHS/NCHSDB/WSC/CCSHS/CFS pretrain +
  MROS/MESA/CHAT/SOF out-of-domain) — STAGES isn't one of them. Also
  searched all 6 of OSF's split files for STAGES's real site-code naming
  (`STNF`, `MSTR`, `GSDV`, `MAYO`, etc.) — zero matches. STAGES is exactly
  as usable as APPLES for a fair comparison.
- **APPLES and MrOS: re-confirmed clean by exact-ID match** (previously
  inferred from OSF's dataset list; now directly verified, zero overlap).

**Why this isn't just a supervised-label-leakage question**: even though
OSF's pretraining is self-supervised (no downstream task labels seen), the
encoder still directly updates its weights on the raw SHHS signal for
these subjects — the objective doesn't need to see a "sex" or "apnea"
label to potentially memorize subject-specific or cohort-specific signal
idiosyncrasies that a downstream linear/LSTM head can then exploit. This
is a well-established concern in foundation-model contamination
discussions generally, and **OSF's own authors clearly treat it as real**:
they explicitly partition their benchmark into disjoint in-domain
(pretrain) vs. out-of-domain cohorts, and even hold out a `pretrain_test`
subject split *within* their own pretraining cohorts specifically so they
have a genuinely-unseen slice to evaluate on. If the risk were negligible,
that design effort wouldn't be necessary.

**Does OSF's own paper report downstream results on SHHS?** Yes, but via
their own carefully disjoint `pretrain_test` split — never the subjects
they actually trained on. Our current SHHS numbers, by contrast, are 87.7%
computed on subjects OSF *did* train on — not comparable to how OSF's own
authors validated the model.

**Recommendation — no retraining needed to answer this.** Retraining our
sequence heads without SHHS in the training mix wouldn't fix anything (the
contamination is in OSF's *encoder* pretraining, not in how we train our
downstream head) and would diverge from SleepFM's own comparison protocol
unnecessarily. Instead: **keep SHHS in the training/registry mix for
protocol consistency with SleepFM, but never report its AUROC as part of
a pooled/headline number — always report it separately with this
caveat**, optionally alongside the clean-157-subject-subset number for
context (computed post-hoc from already-existing inference parquets, no
rerun required — see the snippet used for the numbers above, filterable
by cross-referencing `osf/splits/patient_pretrain_{train,valid}_ids.csv`).

### Context-length pattern: partially similar, with real per-task differences

Checked whether OSF and SleepFM show the same *shape* of AUROC-vs-context
curve, using the fully-clean APPLES cohort only (lstm head, all 6
contexts) to avoid any contamination confound:

- **`sex_binary`**: both models improve with context, but the *pattern*
  differs — SleepFM is fairly flat/noisy (0.79–0.82, even dipping toward
  120m/240m), while OSF climbs steadily (0.939→0.960). The gap actually
  *widens* with context (+14.6pp at 30s → +20.1pp at 240m). OSF appears to
  make more effective use of longer context for this task.
- **`sleep_efficiency_binary`**: both broadly increase with context —
  similar shape — but cross over at 240m (SleepFM edges ahead by 1.3pp,
  the only context where it does on this cohort).
- **`apnea_binary`**: both curves are noisy on this cohort (N=168) with
  multiple sign flips across the sweep (SleepFM ahead at 30s/40m/80m,
  roughly tied at 120m, OSF ahead at 240m) — no confident claim about
  pattern similarity here; treat as within-noise until a larger cohort or
  bootstrap CIs are available.

### Honest per-task verdict

- **`sex_binary` — real, credible win.** The clean cohort (APPLES) shows
  an *even bigger* gap (+12–17pp) than the contaminated cohort (SHHS,
  +2–5pp on the full set, +7pp even on SHHS's clean-subject subset). If
  contamination were driving this, the pattern would run the other way.
  This is genuine evidence OSF's encoder is substantially better at sex
  classification, not a memorization artifact — and it appears to use
  additional context more effectively too. Holds for both heads.
- **`sleep_efficiency_binary` — mixed, inconclusive.** APPLES flips sign
  between heads (lstm +5.5pp, transformer −3.8pp); MrOS favors SleepFM in
  both heads (near-tied to −7.3pp). Only SHHS (contaminated, and
  unreliable even after subject-filtering — see above) consistently
  favors OSF. **Do not claim an OSF win here** — the clean-cohort evidence
  doesn't support it either way with confidence yet.
- **`apnea_binary` — genuinely mixed on clean data, not a clean
  "contamination explains it" story.** ~~Earlier draft of this section
  claimed OSF only won on contamination-risk cohorts~~ — **that was wrong,
  based on STAGES being incorrectly flagged as contamination-risk (now
  corrected above, STAGES is confirmed clean).** The real picture: OSF
  **loses** on APPLES (−1.7 to −5.4pp) and MrOS (−6.5 to −7.9pp) — both
  clean — but **wins** on STAGES (+3.1 to +9.0pp, also clean) and SHHS
  (+2.1 to +2.7pp, contaminated). Two clean cohorts favor SleepFM, one
  clean cohort favors OSF by a large margin. **Contamination explains
  SHHS's result but not the pooled picture as a whole** — this is a
  genuinely mixed, cohort-dependent result that needs more investigation
  (e.g., does STAGES's known channel-completeness profile — see
  `docs/OSF_CHANNEL_REPROCESSING_PLAN.md` — interact with this somehow?)
  before drawing any conclusion, rather than a tidy narrative either way.

### Known caveat not yet resolved: subject-count mismatches between OSF and SleepFM

N counts are close but not always identical between the two pipelines at
the same context/task (e.g. apnea_binary/80m: OSF N=2,077 vs SleepFM
N=2,054, driven mostly by STAGES: 227 vs 200 — a ~13% difference for that
cohort specifically). Both pipelines are supposed to draw from the same
`task_subject_dir`/`split_seed` (confirmed identical in the configs), so
this is most likely explained by embedding-extraction coverage
differences (not every subject may have had an OSF embedding ready when a
given inference run happened) rather than an actual split mismatch — but
this hasn't been root-caused yet. **Worth verifying before trusting the
comparison too precisely for STAGES-heavy tasks**, though the direction
and rough magnitude of the per-cohort findings above are unlikely to
flip from a few percent of subjects being added/removed.

### What this means for next steps

- **`sex_binary` is worth pursuing further** (remaining head — mean_pool —
  and eventually LoRA) — the strongest, cleanest signal so far.
- **`apnea_binary` needs the per-cohort breakdown front-and-center** if it
  goes in the paper — not just an SHHS caveat, since STAGES (clean) also
  drives part of the pooled number; never present the pooled AUROC alone.
  Worth investigating *why* STAGES and SHHS favor OSF while APPLES/MrOS
  don't before concluding anything about apnea specifically.
- **`sleep_efficiency_binary` needs more data before a verdict** — remaining
  head, and possibly the other two Tier-1 tasks (`bmi_binary`, `age_class`)
  for a fuller picture of whether OSF's advantage is task-specific or
  broader.
- Per the user's explicit sequencing: **frozen (Stage 1) and LoRA
  (Stage 2) results should both be in before drawing final conclusions** —
  these frozen-encoder numbers are informative but not the final word on
  whether OSF is a strong baseline; do not start Stage 2 implementation
  until explicitly asked.

---

## Implementation Checklist

**Work through a few unchecked items, commit after each, check the box,
then stop for a user checkpoint (🛑) before continuing.** Each item links
to its Appendix section for full detail — read that section if you need
the "why," not to know what to do next.

### Phase 0 — Setup — ✅ ALL DONE
- [x] 0.1 Build `osf_env`, verify imports (Appendix §4)
- [x] 0.2 Download + strict-load-verify the OSF checkpoint (Appendix §5)
- [x] 0.3 Verify this plan against real OSF/NSRR-tools source + a real
      channel audit (this whole doc)
- [x] 0.4 Locate reference materials (channel CSVs, OSF paper — see above)
- [x] 0.5 Resolve the SHHS channel-completeness decision (see Channel
      Mapping above)
- [x] 0.6 Find evidence for the EOG-referencing question (see Channel
      Mapping above; Appendix §1)
- [x] 0.7 Create the `osf-implementation` branch

### Phase 1 — Stage 1 (frozen encoder)
- [x] 1.1 Implement `scripts/extract_osf_embeddings.py` +
      `configs/phase0_osf_config.yaml` (Appendix §3.1, §3.4)
- [x] 1.2 Add VSCode debug configs, smoke-test on real APPLES + SHHS
      subjects — no NaNs, correct shapes, fill-logs match the audit table
      above (Appendix §12). 🛑 **User re-verified independently, 2026-08-11
      — passed** (ran `--limit 2` for both cohorts).
- [x] 1.3 Implement `src/nsrr_tools/datasets/osf_context_window_dataset.py`
      (`OSFContextWindowDataset` — Appendix §3.2) — done 2026-08-11. Also
      resolved the previously-open `PATCHES_PER_EPOCH` question while
      forking: OSF's embeddings are already epoch-granularity (one row per
      30s), so `PATCHES_PER_EPOCH=1` (was 6 for SleepFM's 5s sub-epoch
      patches). Caught and fixed a matching units bug in
      `configs/phase0_osf_config.yaml`'s `max_min_past_patches` (was 240,
      copied verbatim from the SleepFM config; needed to be 40 to represent
      the same 20-minute cap in 30s-epoch units — only affects seq2seq/
      sleep-staging, not the Tier-1 seq2label tasks, but fixed now while
      touching this code).
- [x] 1.4 Implement `scripts/test_osf_context_window_dataset.py` + debug
      config, smoke-test 🛑 (Appendix §12 item 2) — done 2026-08-11.
      Extracted 10 real subjects each for apples/shhs (CPU, ~50 min) to get
      a real train/val/test split (14/3/3 subjects — too few subjects
      earlier would have left val empty and crashed the test). All smoke
      tests passed: 30s (`N=1`), 10m (`N=20`), and `full_night` (variable-
      length collation) all produced correct `[B,N,1536]` shapes, correct
      dtypes, zero shape-mismatch errors, and `full_night`'s per-sample
      valid lengths matched each subject's actual recording length exactly.
      🛑 **User re-verified independently, 2026-08-11 — passed** (ran
      30s/10m/80m, all three passed).
- [x] 1.5 Implement `scripts/train_osf_context_sweep.py` + debug config,
      smoke-test a tiny CPU run 🛑 (Appendix §3.3) — done 2026-08-11. Per
      the user's explicit instruction, training parameters/subjects/
      splitting/sweeping method/heads are kept identical to SleepFM's
      pipeline wherever not genuinely model-specific — the only two
      differences from `train_context_sweep.py` are dropping the
      `--zero-modalities` flag (OSF has no modality groups) and defaulting
      `--wandb-project` to `nsrr-phase0-osf` instead of `nsrr-phase0`
      (kept separate so OSF runs don't mix with SleepFM's in the same W&B
      dashboard) — both flagged in this doc's Key Decisions table.
      **Bundled `jobs/train_osf_context_sweep_gpu.sh` into this same step**
      per the user's explicit request (moved up from item 1.8) — same
      SLURM directives, same SIGUSR1 auto-resume trap, same status-JSONL
      convention as `train_context_sweep_gpu.sh`, pointing at `osf_env`
      and `logs_osf/` instead of `sleepfm_env`/`logs_v3_full/`.
      Smoke-tested end-to-end on real data (CPU, `--max-items 50`, 30s +
      10m context, apnea_binary/lstm): full training loop ran correctly
      (early stopping, checkpointing, snapshots), and the output directory
      structure/`metrics.json`/`summary.csv` schema exactly matches
      SleepFM's (verified by inspection), minus the `zero_modality_indices`
      field (intentionally absent). **Known gap, not blocking**: `wandb`
      isn't installed in `osf_env` (dropped during env setup — needs a Go
      toolchain to build, wasn't used by OSF's own model code) — W&B
      tracking is currently unavailable for OSF runs even though the code
      path is fully implemented; `--no-wandb` works around it for now.
      **USER CHECKPOINT** — re-verify via the `🎯 OSF Step4: Train Sweep
      DEBUG` config in `~/.vscode/launch.json` before continuing to item 1.6.
- [x] 1.6 Implement `scripts/infer_osf_subject_windows.py` **+
      `jobs/infer_osf_subject_windows_gpu.sh` bundled together** (same
      pattern as 1.5's training script + job script), debug config,
      smoke-test 🛑 (Appendix §3.3/§3.6) — done 2026-08-11. Backbone-
      agnostic fork, identical to `infer_subject_windows.py` except the
      dataset import and one deliberate recalibration: the batch-size
      auto-scaling reference point (`_ref_N`) was `2880` (SleepFM's 240m
      in 5s-patch units) — reusing that literal number for OSF would be
      dimensionally wrong, since OSF's own 240m is 480 (30s-epoch units).
      Changed to `_ref_N=480`; `_ref_bs=64` kept as an unverified starting
      assumption, same caveat as the original script's own comment (not
      GPU-tested yet either way) — flagged in Key Decisions table.
      **Found and fixed a real gap while smoke-testing**: `osf_env` was
      missing `pyarrow` (dropped during initial env setup since nothing in
      OSF's own code needed it — but `infer_osf_subject_windows.py`'s
      `df.to_parquet()` call does). Compute Canada's `pip install pyarrow`
      hits the same "dummy package, use the Arrow module" wall documented
      in Appendix §4, and the Arrow module's `EBPYTHONPREFIXES` injection
      mechanism doesn't reach an isolated (`--system-site-packages=false`)
      venv. Tried `fastparquet` as an alternative parquet engine instead —
      it wanted `numpy<2.0`, which would have downgraded numpy and risked
      breaking already-tested code, so abandoned that path without
      completing the downgrade. Fixed cleanly instead: `arrow/25.0.0` (the
      default module) only ships Python 3.11+ bindings, but `arrow/18.1.0`
      still has a Python 3.10 build — added a `.pth` file in `osf_env`'s
      site-packages pointing directly at
      `.../arrow/18.1.0/lib/python3.10/site-packages` (same mechanism as
      the existing `nsrr_tools_src.pth`). Verified this works with no
      `module load` needed at runtime (tested in a fully clean shell) —
      safe for job scripts. Full smoke test (real trained checkpoints from
      1.5, `30s`/`10m`/`40m`, val split): all three contexts succeeded,
      parquet schema exactly matches SleepFM's documented columns
      (`subject_id, dataset, window_idx, true_label, pred_label,
      prob_class0…N`).
      **USER CHECKPOINT** — re-verify via the `🎯 OSF Step5: Infer DEBUG`
      config in `~/.vscode/launch.json` before continuing to item 1.7.
- [x] 1.7 Implement `experiments/v2_osf_registry.yaml` +
      `scripts/gen_commands_osf.py` (Appendix §3.5) — done 2026-08-12.
      Registry ports the exact same 15 tier-1 entries (5 tasks × 3 heads:
      sex_binary, sleep_efficiency_binary, bmi_binary, age_class,
      apnea_binary) field-for-field from `v2_full_registry.yaml` (identical
      datasets/contexts/batch_size/lr — full-channel registry, since OSF
      compares against `phase0_v3_full`), with `results_dir`/
      `inference_dir` pointed at `phase0_osf` and `python_bin:
      /home/boshra95/osf_env/bin/python` as required. `sleep_staging`
      (seq2seq) deferred — not yet ported to `OSFContextWindowDataset`.
      `gen_commands_osf.py` is a parallel generator (not a `gen_commands.py`
      retrofit, per the Code-reuse-assessment decision in `CLAUDE.md`),
      keeping `list/probe-batch/train/infer/analyze/build-heatmap/collect/
      threshold-tuning/status/runs` — same JSONL status format and
      `logs_osf/status/` convention as `gen_commands.py`. **Deliberately
      dropped**: `iso-plots/saturation/scaling-laws/calibration/
      window-position/subject-consistency/task-comparison/
      cohort-saturation/precision-recall/subject-kstar/table-1..table-10`
      — all of these wrap `plot_*.py`/`make_table*.py` scripts, which
      `CLAUDE.md` already documents as superseded by notebooks
      (`results/paper_figures/notebooks_npj/`) for the current paper; no
      reason to build a second figure-generation path for OSF. `probe-batch`
      is kept for schema parity only — no OSF experiment uses
      `batch_mode: memory_bounded` yet and `jobs/find_batch_size_osf_gpu.sh`
      doesn't exist. Wall-time lookup tables are placeholder copies of
      SleepFM's — **not GPU-calibrated for OSF** (only CPU-debugged so
      far); revisit after 1.10's real sweep. Smoke-tested: `list` correctly
      picked up the item-1.5 debug checkpoints
      (`apnea_binary_lstm` → `trained (3/6 contexts)`); `train`, `infer`,
      `status`, `runs`, `collect`, `threshold-tuning`, `build-heatmap`,
      `analyze` all generate correct commands, and the four reused
      downstream scripts (`analyze_windows.py`, `build_heatmap_df.py`,
      `collect_results_v2.py`, `apply_threshold_tuning.py`) were confirmed
      to run in `osf_env` and accept the exact flags generated — no forking
      needed for those, confirming `CLAUDE.md`'s reuse assessment.
- [x] 1.8 Implement `jobs/extract_osf_embeddings_gpu.sh` (Appendix §3.6) —
      done 2026-08-12. Forked from `jobs/extract_embeddings_gpu.sh`, same
      `--start-idx`/`--end-idx` sharding pattern and SIGUSR1 auto-resume
      trap as the other OSF job scripts (Fir only — no rorqual variant
      exists for OSF yet). Points at `osf_env`/`logs_osf/`, calls
      `extract_osf_embeddings.py` with `--config`/`--start-idx`/`--end-idx`/
      `--datasets`/`--no-skip`. `bash -n` syntax-checked clean; file
      permissions matched to the other job scripts (`640`).
      **Not fully smoke-tested — flagging honestly rather than claiming
      more than was verified**: this interactive session's node appears to
      have no GPU (`sq` shows a `vsc-proxy-jump` job, not a GPU allocation),
      and running the script's `torch.cuda.is_available()` fail-fast check
      directly did not return within ~90s (vs. the near-instant activation
      step) — inconclusive whether that's this node lacking CUDA/NVML
      entirely (plausible explanation, not a code bug) or something worth
      investigating. The identical check pattern already exists unmodified
      in the already-committed `train_osf_context_sweep_gpu.sh` /
      `infer_osf_subject_windows_gpu.sh`, so this isn't new/untested logic
      — but neither of those has been run through an actual `sbatch` GPU
      allocation this session either (only their underlying `.py` scripts,
      directly, with `--cpu`). **Test this specific job script via a real
      small GPU allocation (e.g. `--export=ALL,END=5`) before trusting it
      for the full run in 1.9.**
- [ ] 1.9 Run full embedding extraction, all 4 datasets, GPU job. **Small
      test job submitted 2026-08-12, job `54342713`
      (`--export=ALL,END=5`, first 5 subjects) — check it succeeded
      (`sacct -j 54342713`, then verify `.npy` files/`_channel_fill_log.jsonl`
      under `osf_30sec/apples/`) before submitting the full sharded run
      below.** From here this is pure job submission — no more code to
      write for this step:
      ```bash
      cd /home/boshra95/NSRR-tools
      # 6 shards, ~2500 subjects each (same subject order as SleepFM's
      # phase0_v3_full extraction — see job script header comment):
      sbatch --export=ALL,START=0,END=2500       jobs/extract_osf_embeddings_gpu.sh
      sbatch --export=ALL,START=2500,END=5000    jobs/extract_osf_embeddings_gpu.sh
      sbatch --export=ALL,START=5000,END=7500    jobs/extract_osf_embeddings_gpu.sh
      sbatch --export=ALL,START=7500,END=9600    jobs/extract_osf_embeddings_gpu.sh
      sbatch --export=ALL,START=9600,END=12500   jobs/extract_osf_embeddings_gpu.sh
      sbatch --export=ALL,START=12500,END=15100  jobs/extract_osf_embeddings_gpu.sh
      # Monitor: sq   (or squeue -u $USER)
      # Verify counts once done:
      find /scratch/boshra95/psg_full/unified/embeddings/osf_30sec -name '*.npy' | wc -l
      ```
      Per-subject GPU cost is unknown (checklist 1.8) — watch the first
      shard's early progress in its `.out` log before assuming all 6 will
      finish within the 4h `#SBATCH --time` default; bump `--time` on
      resubmit if needed (auto-requeue handles timeouts either way).
- [~] 1.10 Run the Stage 1 sweep (5 tasks × 3 heads × 6 contexts = up to 90
      training runs, generated via `gen_commands_osf.py` from 1.7), then
      inference, then analysis. Also pure job submission from here —
      mirrors `docs/EXPERIMENTS_GUIDE.md`'s "Submitting Jobs" /
      "Typical workflow" pattern exactly, just pointed at
      `gen_commands_osf.py`. **See `docs/OSF_EXPERIMENTS_GUIDE.md`'s new
      "Step 7 — Running the Full Stage 1 Sweep" section for the complete
      copy-pasteable loop** (train all 15 experiments → monitor → infer →
      analyze → collect). Requires 1.9 to be done first.
      **IN PROGRESS as of 2026-08-13**: `sex_binary`/`sleep_efficiency_binary`/
      `apnea_binary` × `lstm`/`transformer` (6 of 15 registry entries) are
      trained + inferred + analyzed — see the new "Stage 1 Results" section
      above for the comparison against SleepFM and the honest per-task
      verdict. `bmi_binary`, `age_class`, and all `mean_pool` runs remain.
- [ ] 1.11 Re-run the channel-completeness audit against real (not
      50-subject-preview) extraction output; update the table above

### Phase 2 — Stage 2 (LoRA fine-tuning)
- [ ] 2.1 Implement `scripts/train_osf_lora.py` (Appendix §6.1)
- [ ] 2.2 Add debug config, run the short wall-time pilot 🛑 (Appendix §6.3)
- [ ] 2.3 Run the full Stage 2 sweep, applying the memory-mitigation
      ladder as needed (Appendix §6.2)

### Phase 3 — Results
- [ ] 3.1 Compile Stage 1 + Stage 2 results against `phase0_v3_full`,
      applying contamination + SHHS channel-completeness caveats honestly
      (Appendix §8)
- [ ] 3.2 Report back before starting PhysioOmni or MOMENT — do not start
      the next model's plan unprompted

---

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Comparison baseline | `phase0_v3_full` (full-channel SleepFM) | OSF needs channels (snore, thoracic/abdominal/airflow) only the full-channel config carries |
| Task scope (first pass) | Tier-1 only: `sex_binary`, `sleep_efficiency_binary`, `bmi_binary`, `age_class`, `apnea_binary` | Validate the pipeline end-to-end before adding sleep staging / Tier-2 |
| Embedding storage | CLS + mean-pooled patches, `[T,2,768]`, per epoch | Matches SleepFM's "save full-resolution, slice windows at train time" pattern |
| SHHS EEG handling | Duplicate generic `EEG` into both C3/C4 slots | No distinguishable C3/C4 exists in our SHHS HDF5s; provisional, see Channel Mapping |
| Architecture parity | `hidden_dim=128, num_layers=1`, only `input_dim` changes (1536 vs 512) | Preserves "encoder/channels change, architecture held constant" comparison principle |
| Resampling 128→64Hz | Exact decimation (`x[::2]`) | Mathematically identical to OSF's own linear-interpolation resample for this exact 2:1 rate ratio |
| Environment | Fresh `osf_env`, not `sleepfm_env` | Conflicting pins (`torch==2.5.1` vs whatever `sleepfm_env` has) |
| LoRA target modules | `to_qkv`, `to_out.0` | The only two Linear layers in OSF's `Attention` block; PEFT suffix-matches through the `PreNorm` wrapper |
| Label/split reuse | Same `task_subject_dir` + `split_seed` as SleepFM | Required for a fair comparison on identical subjects/splits — not just an optimization |
| W&B project | `nsrr-phase0-osf`, not `nsrr-phase0` | Keeps OSF runs separate from SleepFM's dashboard; currently inert (`wandb` not installed in `osf_env`, see Known Issues) |
| Inference batch-size reference point | `_ref_N=480` (OSF's 240m), not SleepFM's literal `2880` | Same formula/intent as `infer_subject_windows.py`, but `2880` is 5s-patch units — reusing it verbatim would be dimensionally wrong for OSF's 30s-epoch units |
| `pyarrow` in `osf_env` | `.pth` file pointing at `arrow/18.1.0`'s Python 3.10 site-packages (not `pip install`) | CC's `pip install pyarrow` always hits a dummy package; the Arrow module's own injection mechanism doesn't reach an isolated venv; `fastparquet` (the alternative) wanted `numpy<2.0`, an unacceptable downgrade — see checklist 1.6 |
| `gen_commands_osf.py` scope | Separate script from `gen_commands.py`; kept `list/probe-batch/train/infer/analyze/build-heatmap/collect/threshold-tuning/status/runs`, dropped all figure/table subcommands (`iso-plots`, `saturation`, `scaling-laws`, `calibration`, `window-position`, `subject-consistency`, `task-comparison`, `cohort-saturation`, `precision-recall`, `subject-kstar`, `table-1..10`) | No backbone hook in the original registry/wall-time-table schema (see `CLAUDE.md` Code-reuse-assessment); the dropped subcommands wrap `plot_*.py`/`make_table*.py` scripts already superseded by notebooks for the current paper — no reason to build a second figure path for OSF |
| OSF wall-time tables | Placeholder copies of SleepFM's `_TRAIN_HOURS`/`_INFER_HOURS_PER_CTX` | No real GPU sweep has run yet to calibrate against (only CPU-debugged); auto-requeue means an underestimate just costs one resubmission — revisit after checklist 1.10 |

---

## Known Issues / Open Questions

- [x] ~~**Does `extract_osf_embeddings.py`'s channel mapping correctly use
  everything available, or is it missing an unrecognized channel-name
  variant?**~~ **✅ VERIFIED 2026-08-12 against the real full-population
  extraction run (checklist 1.9, in progress).** Cross-checked two ways:
  (1) exhaustively scanned all distinct HDF5 keys across 200 random
  subjects/dataset — no leg-EMG/abdominal/snore-like channel name exists
  under any other name that the candidate lists are missing; (2) aggregated
  the real `_channel_fill_log.jsonl` files from the running extraction
  (716 APPLES / 936 SHHS / 267 MrOS subjects so far) — every zero-fill rate
  is a clean 100% or ~0%, never partial/flaky, confirming these are
  structural per-cohort absences (the channel was never recorded for that
  cohort), not a naming-match bug: APPLES `EMG_RLeg` 100%, `EMG_LLeg` 19%;
  SHHS `EMG_LLeg`/`EMG_RLeg`/`SN` 100%, `NP` 15%; MrOS `ABD`/`SN` 100%. This
  matches the pre-registered 50-subject audit above almost exactly, and
  matches OSF's own reference implementation's missing-channel handling
  (zero-fill, per §0 Appendix). **No fix needed in
  `extract_osf_embeddings.py` itself** — it correctly uses everything
  present in `psg_full/`. SHHS remains the cohort with the most degraded
  input (3 slots always zero + 1 partial + the EEG C3/C4 duplication
  approximation) — **impact on SHHS's actual downstream OSF results is
  still unknown** (revisit once Stage 1 training results exist).
  **Important follow-up finding (2026-08-12, deeper dive into the raw EDF
  channel inventories, not just `psg_full/`'s existing HDF5s):**
  `psg_full/` itself — the *upstream* preprocessing, shared with SleepFM —
  has three confirmed, quantified, fixable gaps: MrOS's `ABD` is a genuine
  code bug (100% missing despite being ~100% present in raw data), and
  STAGES `LLEG`/`RLEG` (+23.2pp available) and SHHS `NP` (+22.9pp
  available) are simple alias-list gaps. **Full investigation, root-cause
  traces, and a fork-not-edit fix plan (does not touch any SleepFM file)
  are in [`docs/OSF_CHANNEL_REPROCESSING_PLAN.md`](OSF_CHANNEL_REPROCESSING_PLAN.md)
  — deliberately not acted on now** (re-preprocessing ~14,000 subjects is
  a bigger undertaking than anything else in this implementation, needs an
  explicit decision to spend the compute). Come back to that doc if
  MrOS/STAGES/SHHS results look degraded enough to be worth the re-run.
- [ ] **EOG referencing** — encouraging (no NaNs across 20+ smoke-tested
  subjects so far) but not exhaustively confirmed across every cohort.
- [ ] **STAGES-in-pretraining confirmation** — cross-check numeric IDs
  against OSF's `osf/splits/patient_pretrain_*.csv` (Appendix §8) — not
  yet done.
- [x] ~~`ContextWindowDataset`'s `PATCHES_PER_EPOCH` constant~~ **✅
  RESOLVED 2026-08-11** — `PATCHES_PER_EPOCH=1` for OSF (embeddings are
  already epoch-granularity), see Implementation Checklist item 1.3.
- [ ] **`wandb` not installed in `osf_env`** — dropped during env setup
  (needs a Go toolchain to build `wandb-core`, wasn't used by OSF's own
  model code — see Appendix §4). `train_osf_context_sweep.py` fully
  implements W&B tracking (matching SleepFM's pipeline exactly, per the
  parity requirement), but it's inert until `wandb` is actually installed.
  Not blocking for CPU/GPU debug runs (`--no-wandb`), but worth revisiting
  before the real Stage 1 sweep if W&B tracking parity with SleepFM
  matters for that run.
- [ ] **Stage 2 (LoRA) wall-clock cost** — no calibrated estimate yet,
  unlike Stage 1's SleepFM-style estimation (Appendix §6.3).
- [ ] **MrOS's raw EDFs do have an "ABD" channel** (per
  `channel_analysis/mros_channels.csv`) **but it's absent from our
  preprocessed full-channel HDF5s** — worth a quick look at whether this is
  a fixable preprocessing gap rather than a true sensor absence, though not
  investigated further this session (out of scope for the OSF integration
  itself).

---

# Appendix: Detailed Verification & Design Notes

**Everything below this point is supporting detail for the plan above —
the full research trail, code citations, and reasoning behind every
decision and checklist item.** It's organized as §0-§12 (numbering
preserved from earlier drafts so existing cross-references throughout this
doc still resolve correctly) and is dense by design — it exists so nothing
had to be re-derived or re-verified from scratch, not for a linear read.
Jump to a specific §N via the cross-references above rather than reading
top to bottom.

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
`train_config.py` `TRAIN_EDF_COLS_UNI_ENC` in the OSF repo — **code-verified
2026-08-10, exact match confirmed against `train_config.py`, `demo.ipynb`'s
`CHANNEL_NAMES`, and `README.md`'s "Input Format" section, all three
independently agree**):
`ECG, EMG_Chin, EMG_LLeg, EMG_RLeg, ABD, THX, NP, SN, EOG_E1_A2, EOG_E2_A1,
EEG_C3_A2, EEG_C4_A1`. Order matters — it's fed positionally into a
`Conv2d` patch-embedding layer indexed by channel position, not by name.

**Channel-name mapping (which raw HDF5 key fills which OSF slot) — no
changes from the original draft, still accurate:**

| OSF channel | Our channel (full-channel HDF5) |
|---|---|
| `EEG_C3_A2` | `C3-M2` |
| `EEG_C4_A1` | `C4-M1` |
| `EOG_E1_A2` | `LOC` |
| `EOG_E2_A1` | `ROC` |
| `ECG` | `EKG` (fallback `ECG-L` where `EKG` absent, MrOS/STAGES only) |
| `EMG_Chin` | `CHIN` (fallback generic `EMG` where `CHIN` absent — SHHS, APPLES) |
| `EMG_LLeg` | `LLEG` |
| `EMG_RLeg` | `RLEG` |
| `ABD` | `ABD` |
| `THX` | `Thor` |
| `NP` (nasal pressure/airflow) | `Airflow` |
| `SN` (snore) | `Snore` |

**What changed since the original draft: per-cohort channel *availability*
is far less uniform than "clean, channel-for-channel, all High confidence"
implied.** The original table's confidence ratings were not backed by
measurement — they were plausibility judgments. **Code-verified 2026-08-10**
by sampling 50 random subjects per cohort from the real full-channel HDF5s
at `/scratch/boshra95/psg_full/{dataset}/derived/hdf5_signals/` and checking
raw key presence:

| OSF slot | APPLES (n=50) | SHHS (n=50) | MrOS (n=50) | STAGES (n=50) |
|---|---|---|---|---|
| `ECG` | 100% | 100% | 100% | 90% |
| `EMG_Chin` | 100%† | 100%† | 100% | 100% |
| `EMG_LLeg` | 78% | **0%** | 100% | 56% |
| `EMG_RLeg` | **0%** | **0%** | 100% | 56% |
| `ABD` | 100% | 100% | **0%** | 100% |
| `THX` | 100% | 100% | 100% | 100% |
| `NP` (Airflow) | 100% | 68% | 100% | 100% |
| `SN` (Snore) | 100% | **0%** | **0%** | 100% |
| `EOG_E1_A2` (LOC) | 100% | 100% | 100% | 100% |
| `EOG_E2_A1` (ROC) | 100% | 100% | 100% | 100% |
| `EEG_C3_A2` | 100% | **0%** | 100% | 100% |
| `EEG_C4_A1` | 100% | **0%** | 100% | 100% |

†APPLES's and SHHS's 100% "EMG_Chin" is the generic `EMG` fallback, not a
channel literally named `CHIN` — see the SHHS caveat below.

**This is structural, not per-subject noise** — the 0%/100% entries are
consistent across the full 50-subject sample per cohort, meaning these are
cohort-wide recording/preprocessing characteristics, not something a larger
sample would smooth out. Three findings change the plan materially:

1. **SHHS cannot supply distinct C3/C4 EEG at all.** SHHS's full-channel
   HDF5s store a single generic `EEG` channel (0% `C3-M2`, 0% `C4-M1`, in
   every sampled subject) — SHHS's original PSG montage does not preserve a
   left/right central-electrode distinction in this repo's preprocessing.
   Combined with SHHS also having 0% `LLEG`/`RLEG`/`Snore` and only 68%
   `Airflow`, **5 of OSF's 12 input slots are either always zero or
   majority-zero for every SHHS subject** (`EMG_LLeg`, `EMG_RLeg`, `SN`,
   and effectively both `EEG_C3_A2`/`EEG_C4_A1` unless the generic `EEG`
   channel is duplicated into both slots as an approximation).
   **✅ DECIDED 2026-08-10 (confirmed with the user): duplicate SHHS's
   single generic `EEG` channel into both `EEG_C3_A2` and `EEG_C4_A1`
   slots** (same spirit as the already-accepted `Airflow`-for-`NP`
   approximation), **zero-fill `EMG_LLeg`/`EMG_RLeg`/`SN`.** Explicitly
   flagged by the user as provisional — track in a future results pass
   whether SHHS's OSF numbers look meaningfully degraded by this
   approximation, and revisit with a **targeted re-preprocessing pass for
   SHHS specifically** if so (see Master Implementation Checklist item 0.5
   for what that would involve). Not implementing that reprocessing now —
   only the duplication logic in `extract_osf_embeddings.py`.
2. **MrOS has zero `ABD` channels** (0/50, not partial) despite `ABD` being
   rated "High confidence" in the original table and despite MrOS otherwise
   having excellent coverage (100% on 9 of the other 11 slots, including
   `EMG_LLeg`/`EMG_RLeg`/`CHIN` — better than APPLES or STAGES on those).
   MrOS also has 0% `Snore`. **2 of 12 slots always zero-filled for MrOS**;
   otherwise MrOS is the cleanest cohort for OSF's specific 12-channel
   requirement, which matters since the plan's §0.2 baseline-choice
   reasoning ("OSF needs channels... abdominal... that only the full-channel
   config carries") assumed `ABD` availability that MrOS does not actually
   have.
3. **APPLES has zero `EMG_RLeg`** (0/50, structural — no right-leg EMG
   channel exists in APPLES's full-channel HDF5s at all) and only 78%
   `EMG_LLeg`. **STAGES** has only 56% `EMG_LLeg`/`EMG_RLeg` and 90% `ECG`
   (not the 100% "High confidence" the original table implied for either).

**Missing-channel handling (unchanged decision, now with real numbers to
apply it to):** OSF's own demo code zero-fills any channel absent from the
input — do the same in the extraction script, and **log which channels
were zero-filled per subject** (not just per dataset) so the per-cohort
completeness table above can be regenerated from real extraction-run data,
not just this 50-subject preview sample, before results are reported.
Given the numbers above, no cohort is "mostly zeros" in the sense that
would justify outright exclusion on channel-completeness grounds alone
(APPLES/MrOS/STAGES are all ≥83% complete by slot-count; SHHS is the
outlier at ~58% complete, 7/12 slots at ≥68%, but that's a judgment call,
not an automatic exclusion — see decision point above).

**Conclusion: no raw EDF reprocessing needed** — this part of the original
draft holds. All 12 channels are either present, present via a documented
approximation (generic `EMG`→chin, `Airflow`→`NP`), or absent and
zero-filled, using data already in the existing full-channel HDF5s at
`/scratch/boshra95/psg_full/{dataset}/derived/hdf5_signals/`. The
remaining work is a resample + reorder + rename + rechunk adapter, not new
preprocessing from raw signal.

**Normalization compatibility:** OSF's own `SleepEpochDataset` applies no
additional per-channel normalization to any of these 12 channels beyond a
final `clamp(-6, 6)` (`NEED_NORM_COL = [HR, SPO2, OX]` in OSF's
`train_config.py` — none of which are in our 12-channel list; **code-verified
2026-08-10** directly in `pretrain_dataset.py`'s `SleepEpochDataset.__getitem__`,
both the pretrain and downstream branches). This means OSF expects its 12
input channels to already be roughly zero-mean, unit-variance (z-scored)
before being fed in. Our own preprocessing (`signal_processor.py`'s
`_normalize_signal`) already does per-channel z-score normalization —
**compatible by construction**, but confirm empirically (§9 sanity checks)
rather than assuming.

**EOG referencing — encouraging evidence found 2026-08-10, not yet fully
closed.** The channel-presence audit above only checked whether the
`LOC`/`ROC` *keys* exist, not what they're referenced to. Checking
`configs/channel_definitions.yaml`'s alias tables and the raw per-subject
channel labels in `output/channel_analysis/*_channels.csv` (see "Reference
materials" at the top of this doc) gives a real (if partial) answer:
**STAGES's dominant raw alias is literally `EOG_LOC-A2`/`EOG_ROC-A1`** —
i.e. the raw EDF label itself spells out contralateral-mastoid referencing
(A2 for the left/LOC channel, A1 for the right/ROC channel), which is
*exactly* OSF's expected `EOG_E1_A2`/`EOG_E2_A1` convention, not just a
plausible guess. **However, `channel_definitions.yaml`'s full `LOC` alias
list also folds in non-contralateral variants for other cohorts/subjects**
(`E1:M1` — ipsilateral if M1 is the same-side mastoid, `E1-Cz`, `E1:E2` —
referenced to the other EOG channel, not a mastoid at all), meaning the
referencing convention behind our generic "LOC" standard channel name is
**not uniformly guaranteed across every subject/cohort**, just confirmed
correct for STAGES's dominant case and very likely correct for SHHS/APPLES
(both use `EOG(L)`/`EOG(R)` or bare `LOC`/`ROC` raw labels, which by
standard AASM/clinical PSG convention also imply contralateral mastoid
referencing, though this wasn't verified with the same "the raw label
itself says A2" level of certainty as STAGES). **Net: likely fine for the
large majority of subjects, worth the empirical no-NaN/no-degenerate-CLS
sanity check in §9 item 2 as final confirmation rather than treating this
as fully closed.**

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

**Code-verified 2026-08-10 — exact OSF-native pipeline order to replicate**
(from `osf/datasets/pretrain_dataset.py`'s `SleepEpochDataset`, confirmed
identical in `demo.ipynb`'s inline reimplementation): **resample (pandas
`reindex` + `.interpolate(method="linear", limit_direction="both")`,
`_resample_df` at `pretrain_dataset.py:251-263`) → select/reorder to the
fixed 12-channel list, zero-filling any missing → apply `to_pm1` only to
`NEED_NORM_COL` channels (none of our 12, so this step is a no-op for us)
→ pad/truncate to exactly `sample_rate * window_size` = 1920 samples →
`clamp(-6, 6)`.** There is no separate "chunk" step distinct from this —
chunking into 30s epochs *is* windowing at read time, one epoch per forward
pass (see §6.1's per-epoch-only confirmation). `_resample_df`'s literal
implementation, for exact replication:
```python
def _resample_df(self, df, target_hz):
    if not np.issubdtype(df.index.dtype, np.number):
        t = np.arange(len(df)) / float(target_hz)
        df = df.copy(); df.index = t
    t0, t1 = float(df.index.min()), float(df.index.max())
    t_target = np.arange(t0, t0 + self.window_size, 1.0 / target_hz)
    if t_target[-1] > t1:
        t_target = t_target[t_target <= t1 + 1e-9]
    return df.reindex(t_target).interpolate(method="linear", limit_direction="both").fillna(0.0)
```
Note `torch.load(..., weights_only=False)` is required when loading the
checkpoint in §5/§3.1 — the payload is a plain dict of `state_dict` +
`metadata`, which newer torch's default `weights_only=True` safety check
can reject depending on version; this is confirmed in both `demo.ipynb`
and `README.md`'s own loading snippet.

---

## 3. Stage 1 (frozen encoder) — components

### 3.1 New script: `scripts/extract_osf_embeddings.py`

**Environment/checkpoint status (done 2026-08-10, ahead of this section
being implemented): `osf_env` is fully built and verified, and the
checkpoint is downloaded and verified-loadable — see §4/§5, both now
"done" not "planned."**

Mirror `scripts/extract_sleepfm_embeddings.py`'s structure, but **not
uniformly** — one piece of its structure (SIGTERM handling) should
specifically *not* be copied verbatim; see below. Code-verified 2026-08-10
against the real file (`scripts/extract_sleepfm_embeddings.py`):

- **Model import — exact `sys.path` pattern to match** (not a literal
  `"../OSF-Open-Sleep-FM"` string): the existing script computes the
  sibling-repo path via `_REPO = Path(__file__).resolve().parent.parent.parent
  / "sleepfm-clinical"` (three `.parent` calls: `scripts/` → `NSRR-tools/` →
  sibling level) then does **two** insertions —
  `sys.path.insert(0, str(_REPO))` and
  `sys.path.insert(0, str(_REPO / "sleepfm"))`. For OSF, mirror this exactly
  with `_REPO = Path(__file__).resolve().parent.parent.parent /
  "OSF-Open-Sleep-FM"` and a single `sys.path.insert(0, str(_REPO))` (OSF
  has no nested-package layer analogous to `sleepfm-clinical/sleepfm`).
  Import the `ViT` class from `osf.backbone.vit1d_cls` and its
  `vit_base(...)` factory.
- **SIGTERM handling — mirror `extract_sleepfm_embeddings.py`'s pattern,
  NOT `train_context_sweep.py`'s.** These two existing scripts use
  *different* patterns and the embedding-extraction script should follow
  its own sibling, not the training script: `extract_sleepfm_embeddings.py`
  sets a module-level flag (`_stop_requested = False`) in the SIGTERM
  handler and the per-subject loop checks it *after* finishing the current
  subject (graceful finish-current-item, relying on `out_path.exists()`
  skip-logic for resumability — there's no per-subject `resume.pt`).
  `train_context_sweep.py` instead calls `sys.exit(0)` immediately from the
  handler (safe there because it flushes `resume.pt` every epoch). OSF's
  extraction script has no epoch-level checkpoint either, so it needs the
  *finish-current-subject-then-stop* pattern, exactly like
  `extract_sleepfm_embeddings.py` — copy that one's SIGTERM code, not
  `train_context_sweep.py`'s.
- **Checkpoint loading — resolved, no longer an open question.** The real
  HuggingFace Hub filename is **`osf_backbone.pth`** (confirmed both by
  downloading it — see §5 — and by disassembling the archive: its internal
  folder name is `dino_vit_base_backbone/`, meaning the file was originally
  `torch.save`'d as `dino_vit_base_backbone.pth` — the name `demo.ipynb`
  still references — and later renamed to `osf_backbone.pth` for the public
  HF upload. One checkpoint, two historical names; use `osf_backbone.pth`
  since that's what `snapshot_download`/`hf_hub_download` actually produce).
  Load with `torch.load(path, map_location=..., weights_only=False)` — the
  payload is a plain `{"state_dict": ..., "metadata": ...}` dict (**verified
  by actually loading it**: `metadata = {'model_name': 'dino',
  'encoder_name': 'vit_base', 'num_leads': 12, 'patch_size_time': 64,
  'patch_size_ch': 4, 'lead_wise': 1, 'sample_rate': 64,
  'window_size_sec': 30, 'seq_len': 1920, 'width': 768, 'depth': 12}`).
  Instantiate with **all five of these kwargs read from `metadata`, not
  guessed** — `vit_base(num_leads=meta['num_leads'], seq_len=meta['seq_len'],
  patch_size=meta['patch_size_time'], patch_size_ch=meta['patch_size_ch'],
  lead_wise=meta['lead_wise'])` — then `model.load_state_dict(state_dict,
  strict=True)`. **Verified 2026-08-10: this loads with zero missing/unexpected
  keys, 85,325,568 params.** `vit_base(...)`'s own defaults
  (`patch_size=50`, no `lead_wise`/`patch_size_ch`) do NOT match the released
  checkpoint and will raise `assert seq_len % patch_size == 0` if you omit
  `patch_size`/call with defaults — this bit us once already when testing.
  **Patch geometry, now pinned down** (previously left implicit): the
  checkpoint uses `lead_wise=1` (2D `Conv2d` patchify, not the 1D `Conv1d`
  path), giving `Lr = num_leads/patch_size_ch = 12/4 = 3` channel-groups ×
  `Nt = seq_len/patch_size_time = 1920/64 = 30` time-patches = **90 tokens**
  (matches the checkpoint's `pos_embedding` shape `(1, 91, 768)` = 90+CLS
  exactly).
- **Per-subject processing loop**:
  1. Load the 12 mapped channels (§1) from the full-channel HDF5 at 128 Hz.
  2. Resample each channel to 64 Hz (§2's `_resample_df`-equivalent).
  3. Zero-fill any missing channel (log which channels were zero-filled,
     per subject — §1's per-cohort completeness table should be
     regenerated from these logs once extraction actually runs, not left
     as the 50-subject preview sample it currently is).
  4. Chunk into contiguous, non-overlapping 30-second (1920-sample)
     epochs; drop any incomplete trailing epoch (same convention as
     SleepFM's incomplete-chunk handling).
  5. Batch epochs through **`ViT.forward_encoding(x, return_sequence=False)`**
     (⚠️ **corrected from the original draft, which had this boolean
     backwards** — `return_sequence=False` is both the default and the
     value that actually returns the `(cls, patches)` 2-tuple the next step
     needs; `return_sequence=True` returns the raw undivided `[B, 91, 768]`
     sequence with CLS and patches still concatenated together, which would
     silently break the next line's unpacking. Verified directly against
     `demo.ipynb`'s own inference cell, which calls with `return_sequence=False`.)
     → `cls: [B, 768]`, `patches: [B, 90, 768]`.
  6. Mean-pool `patches` over the 90-token axis → `[B, 768]`.
  7. Stack `[cls, mean_pooled_patches]` → `[B, 2, 768]` per epoch.
- **Output**: `{output_dir}/{dataset}/{subject_id}.npy`, dtype float16,
  shape `[T_epochs, 2, 768]` (`T_epochs` = number of complete 30s epochs in
  the recording). Output dir:
  `/scratch/boshra95/psg_full/unified/embeddings/osf_30sec/` (naming
  convention mirrors the existing `sleepfm_5sec/`).
- **GPU batching**: batch across epochs (not subjects), same pattern as
  the SleepFM script's chunk batching, for GPU utilization. Note
  `extract_sleepfm_embeddings.py` has **no `--batch-size` CLI flag** — chunk
  batch size comes only from a config field (`embedding.chunk_batch_size`,
  default 16). Mirror that (config-driven, not CLI-driven) rather than
  adding a new flag.

### 3.2 New dataset class: `src/nsrr_tools/datasets/osf_context_window_dataset.py`

Per the Code Reuse Assessment in `CLAUDE.md`, `ContextWindowDataset` is
SleepFM-shape-hardcoded and not reusable unmodified. **Code-verified
2026-08-10** directly against `context_window_dataset.py`: the real
module-level constants (`:111-118`) are `PATCH_SECONDS = 5`,
`PATCHES_PER_EPOCH = 6`, `EMBED_DIM = 128`, `N_MODALITIES = 4`,
`FLAT_DIM = N_MODALITIES * EMBED_DIM`, `FULL_NIGHT_SENTINEL = -1` — one
more constant (`PATCHES_PER_EPOCH`) than the original draft listed. Its
exact role wasn't fully traced during this verification pass — check its
call sites when actually forking the file, since it may or may not need an
OSF-side equivalent (OSF has no sub-epoch patch/epoch distinction the way
SleepFM does 6×5s-patches-per-30s-epoch, so this constant may simply not
apply, or may need to become `1`; don't assume either way without reading
its usage first). Create `OSFContextWindowDataset` as a **parallel class**,
copied from `ContextWindowDataset` with these constants changed:

```python
N_SUBTOKENS = 2       # was N_MODALITIES = 4  (index 0 = CLS, index 1 = mean-pooled patches)
EMBED_DIM   = 768     # was 128
FLAT_DIM    = 1536    # was 512  (= N_SUBTOKENS * EMBED_DIM)
PATCH_SECONDS = 30    # was 5
```

These are used directly (not shape-introspected from the loaded array) in
every `np.zeros((..., N_MODALITIES, EMBED_DIM))` pad-block allocation and
every `.reshape(N, FLAT_DIM)` call across the three window-builder methods
(causal/centered/seq2label) — confirmed 6 call sites total. If a real
`.npy` had OSF's `(T, 2, 768)` shape but the constants were left at
SleepFM's values, the mismatched pad-block concatenation would crash
loudly (shape mismatch) rather than silently misbehave — a reasonably
safe failure mode if the constants are missed, but still worth getting
right the first time.

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
**Code-verified 2026-08-10: the threshold comparison itself
(`context_window_dataset.py:395-413`) is genuinely unit-agnostic** — it
reads `T` directly from the `.npy`'s on-disk shape (`np.load(...,
mmap_mode="r").shape[0]`) and compares it to `min_recording_patches` as a
plain integer, with no embedded assumption about what a "patch" is, so
`480` will work correctly as-is. **One cosmetic bug to fix while forking,
not a logic bug:** the warning message printed when a subject is excluded
computes a "minutes" figure via `min_min = self._min_recording_patches * 5
// 60` (hardcoded `5`, i.e. seconds-per-patch, baked into the display
string only) — this will print a wrong number for OSF (e.g. `480*5//60=40`
displayed instead of the correct `480*30//60=240`) unless updated to `* 30`
in the fork. Purely cosmetic (doesn't affect which subjects get excluded),
but worth fixing so debug output isn't misleading.

There is no `zero_modality_indices` equivalent needed — OSF has no
4-modality-group structure to ablate. **Code-verified 2026-08-10: this is
a dataset-class-level feature (not training-script-level), cleanly
separable but not centralized** — it's the constructor param
`zero_modality_indices` (stored as `self._zero_modality_indices`), the
`_apply_modality_zeroing()` method, and **three separate call sites** (one
in each of the causal/centered/seq2label window-builder methods). Forking
means dropping the constructor param, the stored attribute, the method,
and all three call sites — mechanically simple, but four distinct edits,
not one.

Everything else (K-sampling logic — overlapping train/val/test vs.
non-overlapping inference at K_max>100, `SubjectGroupedSampler`, seq2label
window building, padding/collate) is shape-parametric once the above
constants are updated — copy as-is. **Code-verified 2026-08-10**: the
window-index builder (`_build_seq2label_index`) and `SubjectGroupedSampler`
operate purely on integers (`T` from the shape cache, `N` from
`parse_context_length`) with zero references to `N_MODALITIES`/`EMBED_DIM`
anywhere in their logic — confirming this reuse claim holds exactly as
stated, not just approximately.

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
  head/optimizer-level, not backbone-level). **Code-verified 2026-08-10:
  `_MODALITY_INDICES = {"BAS":0,"RESP":1,"EKG":2,"EMG":3}` is real
  (`train_context_sweep.py:117`, `infer_subject_windows.py:70`) and is
  correctly the only *shape/channel-semantic* SleepFM hardcode in either
  file** — but not literally the only SleepFM-*calibrated* thing worth
  knowing about (see the batch-size note below). Checkpoint/resume for
  reference when replicating in the fork: `resume.pt` (full training state
  — epoch, step, model/optimizer/scheduler state, history, etc.) is
  rewritten every epoch; a `SIGTERM` handler calls `sys.exit(0)`
  immediately (safe because `resume.pt` is already flushed); `best_model.pt`
  (lighter, state_dict only) saves whenever the monitored val metric
  improves; `resume.pt` is deleted on successful completion so a later
  resubmit isn't mistaken for a mid-run resume.
- `scripts/infer_osf_subject_windows.py` — same relationship to
  `infer_subject_windows.py`. **One SleepFM-calibrated (not
  SleepFM-hardcoded) detail worth checking rather than blindly trusting**:
  `infer_subject_windows.py` auto-scales its inference batch size against a
  reference point tuned for SleepFM's context arithmetic (`_ref_bs=64`,
  `_ref_N=2880`, i.e. "training used batch=32 at 240m/2880 patches") via
  `eff_bs = min(args.batch_size, max(_ref_bs, int(_ref_bs*_ref_N/N_patches)))`.
  This is generic index arithmetic (won't crash or misbehave for OSF's much
  smaller epoch counts and much larger 1536-dim vectors) but it also isn't
  *tuned* for OSF's memory profile — don't assume the auto-scaled batch
  size is well-chosen for OSF without checking GPU memory headroom
  empirically (§9's pilot run is the place to catch this).

**Model architecture: keep `hidden_dim=128, num_layers=1` (matching
`phase0_v3_full_config.yaml`'s seq2label head), only `input_dim` changes**
(1536 instead of 512). This preserves the existing paper's "architecture
held constant, only the encoder/channels change" comparison principle —
do not tune the head size differently for OSF without a specific reason.

### 3.4 New config: `configs/phase0_osf_config.yaml`

Copy `configs/phase0_v3_full_config.yaml` as the template. **⚠️ Correction
(code-verified 2026-08-10): the template below was originally missing a
required `logging:` section — `train_context_sweep.py:998` and
`infer_subject_windows.py:215` both do `Path(cfg["logging"]["results_dir"])`
via plain bracket access with no default, and the job scripts also read
`['logging']['results_dir']` for failure-reason lookup. Without this
section, the very first invocation of either script would `KeyError`
immediately.** Now added below. Changes from the real
`phase0_v3_full_config.yaml`:

```yaml
embedding:
  checkpoint_dir: "/home/boshra95/OSF-Open-Sleep-FM/pretrained_weights/osf_backbone.pth"  # resolved filename, see §3.1/§5
  output_dir: "/scratch/boshra95/psg_full/unified/embeddings/osf_30sec"
  # chunk_batch_size: tune empirically — start at 16, matching SleepFM's default

data:
  hdf5_dir: "/scratch/boshra95/psg_full"     # same source HDF5s as SleepFM full-channel
  sampling_freq: 64                            # OSF's expected rate, not 128
  epoch_seconds: 30                            # was chunk_seconds: 300 / patch_size: 640
  channel_order: [ECG, EMG_Chin, EMG_LLeg, EMG_RLeg, ABD, THX, NP, SN,
                   EOG_E1_A2, EOG_E2_A1, EEG_C3_A2, EEG_C4_A1]
  channel_mapping:                              # our name -> OSF name; primary alternative only, see §1's
                                                 # per-cohort table for fallback names (e.g. generic EMG for
                                                 # CHIN, ECG-L for EKG) and the SHHS EEG-duplication decision
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
  label_source: "/scratch/boshra95/psg/unified/targets_v2/master_targets.parquet"   # documentation only, see note below
  task_subject_dir: "/scratch/boshra95/psg/unified/targets_v2/task_subjects"         # SAME as SleepFM — this one is load-bearing
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
  optimizer: "adamw"   # NOTE: not actually read — see caveat below
  scheduler: "cosine"  # NOTE: not actually read — see caveat below
  early_stopping_patience: 10
  device: "cuda"        # NOTE: not actually read — see caveat below

logging:                # ⚠️ REQUIRED — missing from the original draft, would KeyError without it
  results_dir: "/scratch/boshra95/psg_full/unified/results/phase0_osf"
```

**Two footnotes, both code-verified 2026-08-10, that materially change how
to read this template — neither is an OSF-specific bug, both are
pre-existing behavior in the SleepFM config/scripts too, just easy to miss
when copying the template for a new backbone:**

1. **`training.optimizer`/`training.scheduler`/`training.device` are dead
   config keys — `train_context_sweep.py` never reads any of the three.**
   The optimizer is hardcoded to plain `torch.optim.Adam` (not AdamW,
   despite the field commonly being set to `"adamw"` in existing configs —
   weight decay is applied Adam-style, not decoupled), the scheduler is
   hardcoded to `CosineAnnealingLR`, and device selection comes from the
   `--cpu` CLI flag / `torch.cuda.is_available()`, not this config field.
   This is pre-existing cruft already present in
   `phase0_v3_full_config.yaml` itself (not introduced by this template) —
   don't assume changing these three keys in the OSF config does anything;
   if a different optimizer/scheduler is ever wanted for OSF specifically,
   it requires an actual code change to `train_osf_context_sweep.py`, not
   a config edit.
2. **`dataset.label_source` is documentation-only — never read anywhere in
   `src/` or `scripts/` (grepped, zero hits).** The actual runtime
   label/subject source is `{task_subject_dir}/{task}_subjects.csv`
   (already-materialized per-task CSVs), plus `train_split`/`val_split`/
   `test_split`/`split_seed` (all read directly by `ContextWindowDataset`).
   The "reuse the same labels/splits as SleepFM" requirement below is still
   correct and still matters — it just actually depends on
   `task_subject_dir` + `split_seed` (+ the split ratios), not
   `label_source`, which can be left pointing at the same file purely for
   human documentation value.

**Label/split reuse is a hard requirement, not just an optimization**: OSF
must be trained/evaluated on the **exact same subjects and the exact same
train/val/test split** as the SleepFM `phase0_v3_full` runs for the
comparison to mean anything — hence pointing `task_subject_dir` and
`split_seed` (the fields that actually matter at runtime, see footnote 2
above) at the identical existing files rather than regenerating them.

### 3.5 New registry + command generation

Per the Code Reuse Assessment, `gen_commands.py` has no backbone hook and
retrofitting it is higher-risk than a parallel generator. For this first
pass (5 tasks × 3 heads × 6 contexts = 90 training runs, manageable),
create:

- `experiments/v2_osf_registry.yaml` — same schema as `v2_registry.yaml`,
  restricted to the 5 Tier-1 tasks × 3 heads, pointing at
  `configs/phase0_osf_config.yaml`, `results_dir:
  /scratch/boshra95/psg_full/unified/results/phase0_osf`,
  `logs_dir: /home/boshra95/NSRR-tools/logs_osf`. **Two more top-level
  fields the plan's original sketch didn't mention, code-verified
  2026-08-10 as required/load-bearing:**
  - **`inference_dir`** — required, not optional: `gen_commands.py:199`
    does `Path(registry["inference_dir"])` via plain bracket access (no
    default), so `infer`-subcommand generation will `KeyError` without it.
    Set to `/scratch/boshra95/psg_full/unified/results/phase0_osf/inference`
    (mirrors `v2_full_registry.yaml`'s `.../phase0_v3_full/inference`
    convention).
  - **`python_bin`** — technically optional (has a fallback default), but
    **the fallback default is `/home/boshra95/sleepfm_env/bin/python`**
    (hardcoded in `gen_commands.py` at every one of its ~15 call sites via
    `registry.get("python_bin", "/home/boshra95/sleepfm_env/bin/python")`).
    **If the OSF registry omits `python_bin`, any command generated by the
    unmodified `gen_commands.py` (or a careless fork that keeps this
    default) would silently use `sleepfm_env`'s interpreter — which has
    none of OSF's `torch`/`peft`/`transformers` pins — rather than
    `osf_env`.** Set explicitly to `/home/boshra95/osf_env/bin/python` in
    the registry, and if `gen_commands_osf.py` is forked (next bullet),
    update its own copy of that fallback default too, not just the
    registry.
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
`*_gpu.sh`/`*_gpu_rorqual.sh` pairs, pointing at the new Python scripts and
`osf_env` (§4, already built) instead of `sleepfm_env`.

**Auto-resume mechanism — two distinct mechanisms, at two different
levels, both real (code-verified 2026-08-10 directly against
`jobs/train_context_sweep_gpu.sh` and `scripts/gen_commands.py`, after an
initial pass of this verification incorrectly concluded `--requeue` was
unused — correcting that here since it was double-checked and found
wrong):**
1. **Wall-time-triggered resume** (bash-level, baked into the `.sh` file
   itself): `#SBATCH --signal=B:USR1@120` sends `SIGUSR1` to the **bash**
   script (the `B:` prefix) 120s before the wall-time limit; Python runs in
   the background (`eval "$CMD" & ; _PYTHON_PID=$! ; wait $_PYTHON_PID`) so
   `wait` returns immediately when the trap fires instead of blocking until
   Python exits on its own; a `trap '_timeout_handler' USR1` handler writes
   a `TIMEOUT_REQUEUED` status line, `SIGTERM`s the Python process, reads
   the job's original time limit via `scontrol show job`, and **calls
   `sbatch` again on its own script path** (`sbatch --export=ALL
   --time="$_TIME_LIMIT" --output=... --error=... "$_SCRIPT_PATH"` — this
   particular resubmit call does **not** include `--requeue`, confirmed by
   reading the trap's exact `sbatch` invocation) — i.e. it resubmits itself
   as a brand-new SLURM job, which then finds `resume.pt` on disk and
   continues.
2. **Node-failure requeue** (SLURM-native, supplied at the *initial*
   `sbatch` invocation, not inside the `.sh` file): `scripts/gen_commands.py`
   appends a literal `--requeue` flag when it constructs the `sbatch`
   command line for both `train`/`infer` subcommands (confirmed at two call
   sites). This is what handles a node failure/preemption mid-run (SLURM's
   own requeue mechanism) — a genuinely different failure mode from #1
   (wall-time approaching), and it's why `CLAUDE.md`'s existing one-line
   summary ("Auto-requeue via `--signal=B:USR1@120` + `--requeue`") is
   correct as a list of the two mechanisms in play, even though neither
   `--signal` nor `--requeue` alone would be a complete description — they
   apply at different levels (script pragma vs. `sbatch` CLI flag) for
   different failure modes, which is easy to miss from a `jobs/*.sh`-only
   grep (that's the mistake this verification pass initially made and is
   correcting here).

**Practical implication for a new `jobs/train_osf_context_sweep_gpu.sh`**:
copy the `#SBATCH --signal=B:USR1@120` + bash trap pattern verbatim into
the `.sh` file (mechanism #1); separately, whatever generates the `sbatch`
invocation for OSF runs (§3.5's `gen_commands_osf.py`, or manual
submission during early debugging) needs to append `--requeue` itself at
submission time (mechanism #2) — don't expect the `.sh` file alone to
provide node-failure resilience. Other structural details to copy exactly
from the existing job scripts: SLURM directives (`--account=def-forouzan_gpu`,
`--gpus=nvidia_h100_80gb_hbm3_1g.10gb:1`, `--cpus-per-task=4`,
`--mem=32000M`, `--exclude=fc11006[,...]`), a CUDA fail-fast check
(`python -c "import torch; assert torch.cuda.is_available()..."`), and a
per-job JSONL status log under `{logs_dir}/status/*.jsonl` with
`STARTED/TIMEOUT_REQUEUED/SUCCESS/FAILED` events. **One nuance specific to
the new scripts**: `train_context_sweep.py`/`infer_subject_windows.py`
never import SleepFM code directly (only the embedding-extraction script
does), so the `PYTHONPATH` export pointing at the sibling model repo is
only actually needed in `extract_osf_embeddings_gpu.sh`, not in
`train_osf_context_sweep_gpu.sh`/`infer_osf_subject_windows_gpu.sh` — copy
it there too for consistency if it's harmless, but don't assume it's
load-bearing in the train/infer job scripts.

---

## 4. Environment — ✅ DONE (2026-08-10), built and verified on-cluster

**Do not reuse `sleepfm_env`.** OSF's `requirements.txt` pins
`torch==2.5.1`, `transformers==4.47.0`, `peft==0.14.0`,
`pytorch-lightning==2.4.0`, `timm==1.0.12`, plus several git-based
dependencies — risking a version conflict with whatever `sleepfm_env` has
pinned for the existing pipeline. `/home/boshra95/osf_env` (Python 3.10.13,
via the cluster's `python/3.10.13` module) now exists and has all 187
packages installed and verified: `import nsrr_tools` and
`from osf.backbone.vit1d_cls import ViT, vit_base` both import cleanly, and
the real checkpoint loads with zero missing/unexpected `state_dict` keys
(§5). **Several real obstacles came up that the original draft's simple
`pip install -r requirements.txt` command doesn't survive unmodified —
recorded here so no one repeats the debugging:**

1. **This cluster's Python doesn't self-report as `manylinux`-compatible
   to pip** (`pip debug --verbose` shows only plain `linux_x86_64` tags,
   no `manylinux*` variants) — standard PyPI wheels for many packages are
   therefore unusable here, and pip falls back to building from source
   (or fails outright, e.g. for `onnxruntime`, which ships no sdist).
   Compute Canada's own wheel cache (`/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/`,
   auto-configured via `$PIP_CONFIG_FILE`) has plain-`linux_x86_64`-tagged
   builds and is the actual working path — but it doesn't carry every
   exact version OSF's `requirements.txt` pins. **Practical rule that
   worked:** for any package CC's wheelhouse has *some* version of, relax
   the exact pin to the closest version CC carries rather than fighting to
   get the literal pinned version from PyPI; only fall back to building
   from source (needs a `module load` of a matching toolchain — `gcc`,
   `rust/1.85.0`+ for Rust extensions, etc.) for packages CC's wheelhouse
   doesn't have at all. From-source C++ builds are also memory-constrained
   on this node — `matplotlib` and `xgboost`'s parallel C++ compiles both
   got OOM-killed (`cc1plus`/`cython` subprocess `SIGKILL`) before being
   relaxed to CC wheelhouse versions instead.
2. **~20 packages in `requirements.txt` are genuinely unused by OSF's own
   code** (confirmed via `grep -rl "import X" --include="*.py"` across the
   whole repo, zero hits) — an artifact of a shared team requirements file,
   exactly as suspected below. Safely dropped rather than fought:
   `kornia`/`kornia_rs` (also: broken sdist packaging bug on PyPI,
   confirmed even with Rust installed), `opencv-python`/`opencv-python-headless`,
   `albumentations`/`albucore`, `insightface` (the actual source of the
   `albumentations`/`opencv` chain — dropping `albumentations` alone wasn't
   enough since `insightface` pulls it back in transitively), `wandb`
   (needs a Go toolchain to build `wandb-core` from source; also confirmed
   used only in OSF's own `main_pretrain.py`/`main_finetune.py`, which this
   integration never runs), `xgboost`, `streamlit` (the source of a
   `pyarrow` build failure — CC deliberately ships a dummy `pyarrow`
   package on this cluster instructing `module load arrow` instead of pip;
   dropping `streamlit`/`datasets`/`pyarrow` — all confirmed unused —
   avoided that entirely rather than fighting the module-load requirement),
   `datasets`, `scikit-image`, `onnx`/`onnx2torch`, `numba`/`llvmlite`
   (came back transitively via `pynndescent`, a real dependency — but
   resolved fine from CC's wheelhouse once not exact-pinned), `ml_dtypes`.
   A working copy of the trimmed requirements lives at
   `/home/boshra95/osf_env_requirements.txt` (the original
   `OSF-Open-Sleep-FM/requirements.txt` is untouched).
3. **Version-relaxed (CC wheelhouse's closest available, not the exact
   pin), all confirmed working:** `numpy` 2.1.2→2.1.1, `h5py` 3.14.0→3.12.0,
   `onnxruntime` 1.23.1→1.17.3 (also unused by OSF's own code, kept anyway
   since it installs cleanly), `safetensors` 0.6.2→0.4.5, `matplotlib`
   3.9.3→3.9.2, `grpcio` — dropped as an exact top-level pin (unused, only
   a transitive dependency of `tensorboard`/`wandb`, which don't need the
   specific version) and resolved fine from CC's wheelhouse once
   unconstrained.
4. **`pip install -e /home/boshra95/NSRR-tools` does not work as the
   original draft assumed — NSRR-tools' `pyproject.toml` declares
   `requires-python = ">=3.11"`**, incompatible with `osf_env`'s Python 3.10
   (deliberately chosen to match OSF's own pinned stack). Modern pip has no
   flag to bypass a `requires-python` gate for an editable install.
   **Actual working substitute**: a `.pth` file
   (`osf_env/lib/python3.10/site-packages/nsrr_tools_src.pth`, containing
   the single line `/home/boshra95/NSRR-tools/src`) achieves the identical
   practical effect (`import nsrr_tools` works) without needing to satisfy
   the packaging metadata gate. Use this instead of `pip install -e` when
   setting up `osf_env`.

```bash
module load python/3.10.13
python3.10 -m venv /home/boshra95/osf_env
source /home/boshra95/osf_env/bin/activate
pip install -r /home/boshra95/osf_env_requirements.txt   # the trimmed/relaxed copy, not the repo's own requirements.txt
echo "/home/boshra95/NSRR-tools/src" > "$(python -c 'import site; print(site.getsitepackages()[0])')/nsrr_tools_src.pth"
python -c "import nsrr_tools; from osf.backbone.vit1d_cls import ViT, vit_base" \
  # (run from /home/boshra95/OSF-Open-Sleep-FM, or with it on sys.path — both import cleanly once there)
```

Note: `environment.yml` in the OSF repo is an **aarch64 (ARM) conda lock
file** — not usable on a Compute Canada x86_64 cluster. Use
`requirements.txt` (pip, architecture-independent) instead, as above.
**Also code-verified 2026-08-10**: `README.md`'s own "Dependencies"
section states `PyTorch >= 2.9.0` and `PyTorch Lightning >= 2.5.5` — both
*above* what `requirements.txt` actually pins (`torch==2.5.1`,
`pytorch-lightning==2.4.0`). Trust `requirements.txt` (the real, working,
pinned environment), not the README prose.

The dependencies that looked unrelated to the sleep encoder itself turned
out to genuinely be unrelated (see point 2 above) — the original
"install everything first, trim only on a real conflict" instinct was
right in spirit, and every trim made was backed by both a grep-confirmed
zero-usage check and an actual install failure, not a guess.

---

## 5. Checkpoint download — ✅ DONE (2026-08-10), downloaded and verified-loadable

```bash
source /home/boshra95/osf_env/bin/activate
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='yang-ai-lab/OSF-Base',
                   local_dir='/home/boshra95/OSF-Open-Sleep-FM/pretrained_weights')
"
```

**Filename resolved: it's `osf_backbone.pth`** (325MB / 341,360,652 bytes),
not `dino_vit_base_backbone.pth` — confirmed both by actually downloading
it and by disassembling the archive (its internal folder name is
`dino_vit_base_backbone/`, meaning the checkpoint was originally
`torch.save`'d under that name and later renamed to `osf_backbone.pth` for
the public HF upload; one file, two historical names — `demo.ipynb` just
has the stale one). Use `osf_backbone.pth`, matching what
`snapshot_download`/`hf_hub_download` actually produce.

**End-to-end load verified**, not just downloaded:
```python
import torch
from osf.backbone.vit1d_cls import vit_base
ckpt = torch.load('pretrained_weights/osf_backbone.pth', map_location='cpu', weights_only=False)
meta = ckpt['metadata']   # {'model_name':'dino','encoder_name':'vit_base','num_leads':12,
                           #  'patch_size_time':64,'patch_size_ch':4,'lead_wise':1,
                           #  'sample_rate':64,'window_size_sec':30,'seq_len':1920,'width':768,'depth':12}
model = vit_base(num_leads=meta['num_leads'], seq_len=meta['seq_len'],
                  patch_size=meta['patch_size_time'], patch_size_ch=meta['patch_size_ch'],
                  lead_wise=meta['lead_wise'])
missing, unexpected = model.load_state_dict(ckpt['state_dict'], strict=False)
# missing: [], unexpected: [] — full strict match, 85,325,568 params
```
**License: MIT** (verified — `LICENSE` file, UCLA Health Intelligence Lab,
standard unmodified boilerplate, no additional restrictions) — a clean,
permissive baseline, notably better-documented than PhysioOmni's situation
(no LICENSE file at all, per `CLAUDE.md`'s existing note on that model).

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
  reading the source directly, **and independently confirmed 2026-08-10
  from the real checkpoint's own `state_dict` key strings** (e.g.
  `block0.attn.fn.to_qkv.weight`, `block0.attn.fn.to_out.0.weight`, through
  `block11...`). **One nuance PEFT's config hides but worth knowing if
  anyone ever needs the exact dotted path** (e.g. manual freezing/inspection
  outside PEFT): `Attention` is wrapped in a `PreNorm` module
  (`self.attn = PreNorm(dim=..., fn=attn)`), so the *fully-qualified*
  module path is actually `block{i}.attn.fn.to_qkv` /
  `block{i}.attn.fn.to_out.0`, not `block{i}.attn.to_qkv`. This doesn't
  change the `LoraConfig` above — PEFT matches `target_modules` by name
  *suffix*, so plain `"to_qkv"`/`"to_out.0"` still correctly hits all 12
  blocks — but don't be surprised by the `.fn.` segment if inspecting
  `named_modules()` directly.), then wrap `(lora_vit, sequence_head)`
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
  batching pattern as the extraction script, calling
  `forward_encoding(x, return_sequence=False)` per §3.1's corrected call —
  same fix applies here) to get `(cls, patches)` → mean-pool patches →
  stack to `[N_epochs, 2, 768]` embeddings, flatten to `[N_epochs, 1536]`,
  feed through the sequence head, compute loss, backprop through **both**
  the LoRA adapters and the head.
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

1. **Channel availability per cohort — ✅ DONE (2026-08-10), see §1.**
   50-subject-per-cohort audit complete, with exact percentages per OSF
   slot per cohort. Headline results: SHHS has 0% `EEG_C3_A2`/`EEG_C4_A1`/
   `EMG_LLeg`/`EMG_RLeg`/`SN` and 68% `NP` — **a decision on how to handle
   SHHS's missing EEG-channel distinction is still needed before
   implementation** (§1 lists 4 options; not yet chosen). MrOS has 0%
   `ABD`/`SN`. APPLES has 0% `EMG_RLeg`. STAGES has only 56%
   `EMG_LLeg`/`EMG_RLeg` and 90% `ECG`. **Re-run this audit against the
   real extraction output once `extract_osf_embeddings.py` exists and has
   run on all subjects** (not just the 50-per-cohort preview sample) to
   confirm the full-population numbers match this preview.
2. **EOG referencing check — still open, not resolved by the channel-count
   audit above** (that only checked whether `LOC`/`ROC` *keys* exist, not
   their reference electrode). Confirm `LOC`/`ROC` in our HDF5s are
   referenced the way OSF expects (contralateral mastoid) — check
   `channel_mapper.py`/original NSRR channel-label provenance, or inspect
   embedding output sanity (no NaN, no degenerate all-zero CLS vectors)
   once extraction runs on a small pilot batch.
3. **Checkpoint filename resolution — ✅ DONE (2026-08-10), see §5.**
   Confirmed `osf_backbone.pth`; checkpoint downloaded and
   strict-load-verified (85.3M params, zero missing/unexpected keys).
4. **Small-scale pilot** — still to do once `extract_osf_embeddings.py`
   exists: run `--limit 5` on one dataset, inspect the output `.npy`
   shape/values, then a tiny Stage-1 training run (`--context 30s`, one
   task, one head) end-to-end before submitting the full 90-run sweep.
   This is where a VSCode debug config earns its keep — see §12.
5. **STAGES-in-pretraining confirmation** (§8) — cross-check numeric IDs.
   Not yet done.
6. **Cohort filter unit check** (§3.2) — confirm `min_recording_patches:
   480` is actually being applied in epoch units. **Partially resolved by
   code reading** (2026-08-10): the threshold comparison itself is
   genuinely unit-agnostic (reads `T` from the `.npy`'s on-disk shape, no
   embedded patch-duration assumption) — so `480` will work correctly
   once the file exists. Still needs an *empirical* check once real OSF
   `.npy` files exist (confirm the printed subject-exclusion counts look
   sane, not that every subject gets excluded or none do).
7. **NEW — Environment/checkpoint sanity, already covered by §4/§5's own
   verification, listed here for completeness**: `osf_env` package install
   ✅, `import nsrr_tools` + `from osf.backbone.vit1d_cls import ViT,
   vit_base` ✅, checkpoint strict-load ✅ — all done 2026-08-10, ahead of
   the rest of this checklist.

---

## 10. Suggested execution order

**Superseded by the "Master Implementation Checklist" near the top of this
doc (added 2026-08-10) — that's the actively-maintained, checkable list;
this section is kept only for the narrative reasoning behind the
ordering.**

1. ~~Set up `osf_env` (§4), download the checkpoint (§5).~~ **✅ DONE
   2026-08-10.**
2. **Decide the SHHS channel-completeness question (§1)** — this blocks a
   concrete decision inside `extract_osf_embeddings.py`'s channel-mapping
   logic, so resolve it before writing that script, not during.
3. Run the remaining Step 0 verification checklist items (§9: EOG
   referencing, STAGES-in-pretraining ID cross-check) — cheap, catch
   expensive mistakes early.
4. Implement `extract_osf_embeddings.py` (§3.1) with a VSCode debug config
   (§12) from the start, run on a small pilot, then the full extraction
   for all 4 datasets.
5. Implement `OSFContextWindowDataset` (§3.2), `phase0_osf_config.yaml`
   (§3.4), forked train/infer scripts (§3.3), registry + generator (§3.5),
   job scripts (§3.6) — each with a debug config (§12) added as it's
   written, mirroring the existing NSRR-tools pattern.
6. Run Stage 1 (frozen) for all 5 Tier-1 tasks × 3 heads × 6 contexts = 90
   training runs, then inference, then analysis (reuse existing
   `analyze_windows.py`/`collect_results_v2.py` pointed at the new results
   dir if the schema lines up, per the Code Reuse Assessment — plausible,
   not yet verified against real OSF results).
7. Implement `train_osf_lora.py` (§6), pilot at short context first (§6.3),
   then run Stage 2 (LoRA) across context lengths per the memory
   mitigation ladder (§6.2).
8. Compile results against `phase0_v3_full` (§0.2), applying the
   contamination caveats (§8) honestly.
9. Report back before starting PhysioOmni or MOMENT — do not start the
   next model's plan unprompted.

---

## 11. Open items not fully resolved (flagged, not blocking)

- ~~Exact checkpoint filename~~ **✅ RESOLVED 2026-08-10 — `osf_backbone.pth`,
  see §5.**
- **SHHS channel-completeness handling — new, code-verified 2026-08-10,
  needs a decision before `extract_osf_embeddings.py` is written.** See §1
  for the 4 options (duplicate generic EEG into both slots / zero-fill one
  slot / report with an explicit caveat / exclude SHHS entirely). Not
  something to decide unilaterally given it stacks on top of SHHS's
  existing contamination caveat.
- Whether LOC/ROC referencing exactly matches OSF's E1-A2/E2-A1 convention
  — still open.
- ~~Per-cohort Snore/EOG channel availability~~ **✅ largely resolved
  2026-08-10 — see §1's full per-cohort, per-slot table** (50 subjects/
  cohort). What remains open specifically: re-confirming these percentages
  against the *full* population once real extraction runs (the table is
  currently a 50-subject preview, not exhaustive), and the EOG-referencing
  question above (a different kind of check — reference electrode, not
  presence).
- Real wall-clock cost of Stage 2 (LoRA) training — no calibrated estimate
  exists yet, unlike Stage 1 which can reuse SleepFM-style estimation
  logic once its own pilot numbers are in.
- Whether `analyze`/`collect`/plotting code can be pointed at the new
  results directory unmodified, or needs its own fork — plausible per the
  Code Reuse Assessment (their code has zero hardcoded references to
  SleepFM's embedding shape, confirmed 2026-08-10) but not verified
  against real OSF results yet.
- **NEW — `ContextWindowDataset`'s `PATCHES_PER_EPOCH` constant** (§3.2):
  its exact role wasn't traced during this verification pass; check
  whether `OSFContextWindowDataset` needs an equivalent before assuming it
  doesn't.

---

## 12. VSCode debug workflow — a launch.json config per implementation step

**New section, added 2026-08-10** per explicit request: every script below
should get a debug config added to `/home/boshra95/.vscode/launch.json`
(the **workspace-root** launch.json — `~/.vscode/launch.json`, not a
per-repo one; **code-verified 2026-08-10: `NSRR-tools/.vscode/` itself has
never contained a `launch.json`** — the actual established pattern lives
at the home-directory workspace level, since it already contains configs
spanning multiple sibling repos/interpreters, e.g. `sleepfm_env` vs.
`NSRR-tools/.venv`) at the point that script is written, using a small
`--limit`/`--max-items`/`--cpu` debug run — mirroring the existing pattern
already used for every SleepFM-pipeline script (e.g. "Test: Extract
Channels (5 files)", "🎾 Phase0 Step4: Train Sweep DEBUG (apnea_binary,
lstm, CPU)"). **Do not implement any of these scripts yet** — this section
is the plan for the configs, to be added incrementally as each script is
actually written (§10's execution order), not all at once now.

**Interpreter/env-var convention to follow** (matches the existing
pattern's per-python-env grouping):

| Script | `python` | `PYTHONPATH` needed? |
|---|---|---|
| `extract_osf_embeddings.py` | `/home/boshra95/osf_env/bin/python` | Yes — `/home/boshra95/OSF-Open-Sleep-FM` (only script that imports OSF directly) |
| `test_osf_context_window_dataset.py` | `/home/boshra95/osf_env/bin/python` | No |
| `train_osf_context_sweep.py` | `/home/boshra95/osf_env/bin/python` | No (confirmed 2026-08-10: neither the SleepFM nor the OSF train/infer scripts import the backbone repo directly — only the extraction script does) |
| `infer_osf_subject_windows.py` | `/home/boshra95/osf_env/bin/python` | No |
| `train_osf_lora.py` | `/home/boshra95/osf_env/bin/python` | Yes — same reason as extraction (loads the raw ViT for the trainable graph) |

All configs need `"env": {"SCRATCH": "/scratch/boshra95", "HOME":
"/home/boshra95"}` (matches every existing entry) and `"justMyCode":
false` (needed to step into `nsrr_tools`/`osf` library code, not just the
top-level script).

**Planned configs, one per script, added when that script exists:**

1. **`extract_osf_embeddings.py`** — two configs mirroring the existing
   "👽 Phase0 Step1: Extract Embeddings" pair (CPU-debug + GPU-full):
   ```jsonc
   {
     "name": "🧬 OSF Step1: Extract Embeddings (5 subjects, CPU debug)",
     "type": "debugpy", "request": "launch",
     "program": "/home/boshra95/NSRR-tools/scripts/extract_osf_embeddings.py",
     "args": ["--config", "/home/boshra95/NSRR-tools/configs/phase0_osf_config.yaml",
               "--datasets", "apples", "--limit", "5", "--cpu"],
     "console": "integratedTerminal", "justMyCode": false,
     "python": "/home/boshra95/osf_env/bin/python",
     "cwd": "/home/boshra95/NSRR-tools",
     "env": {"SCRATCH": "/scratch/boshra95", "HOME": "/home/boshra95",
              "PYTHONPATH": "/home/boshra95/OSF-Open-Sleep-FM"}
   }
   ```
   This is the very first thing to debug once the script exists — it's
   also where §9 item 1's real-population channel-completeness re-check
   and item 2's EOG-referencing sanity check both happen in practice
   (inspect the saved `.npy` for NaNs / degenerate all-zero CLS vectors).
2. **`test_osf_context_window_dataset.py`** — a genuinely new debug script
   (there is no existing one to copy the launch.json entry from, since
   `test_context_window_dataset.py` itself predates the workspace
   `launch.json`'s current entries — see the correction below). Fork
   `scripts/test_context_window_dataset.py`'s CLI/structure (it's a solid,
   reusable template — instantiates the dataset for train/val/test,
   pulls one batch, asserts dtypes/shapes) with **one required
   edit**: the hardcoded `x.shape[-1] == 512` assertion must become
   `1536` for OSF. Two configs mirroring "🈂️ Phase0 Step2: Test
   ContextWindowDataset":
   ```jsonc
   {
     "name": "🧪 OSF Step2: Test OSFContextWindowDataset (apnea_binary, apples)",
     "type": "debugpy", "request": "launch",
     "program": "/home/boshra95/NSRR-tools/scripts/test_osf_context_window_dataset.py",
     "args": ["--config", "/home/boshra95/NSRR-tools/configs/phase0_osf_config.yaml",
               "--task", "apnea_binary", "--task-type", "seq2label",
               "--context", "30s", "10m", "--datasets", "apples"],
     "console": "integratedTerminal", "justMyCode": false,
     "python": "/home/boshra95/osf_env/bin/python",
     "cwd": "/home/boshra95/NSRR-tools"
   }
   ```
3. **`train_osf_context_sweep.py`** — mirrors "🎾 Phase0 Step4: Train Sweep
   DEBUG", using `--max-items`/`--cpu` for a fast smoke test:
   ```jsonc
   {
     "name": "🎯 OSF Step4: Train Sweep DEBUG (apnea_binary, lstm, CPU)",
     "type": "debugpy", "request": "launch",
     "program": "/home/boshra95/NSRR-tools/scripts/train_osf_context_sweep.py",
     "args": ["--config", "/home/boshra95/NSRR-tools/configs/phase0_osf_config.yaml",
               "--task", "apnea_binary", "--task-type", "seq2label", "--head", "lstm",
               "--context", "30s", "10m", "--datasets", "apples",
               "--max-items", "200", "--cpu"],
     "console": "integratedTerminal", "justMyCode": false,
     "python": "/home/boshra95/osf_env/bin/python",
     "cwd": "/home/boshra95/NSRR-tools"
   }
   ```
4. **`infer_osf_subject_windows.py`** — **new**, no existing SleepFM
   equivalent currently in `launch.json` to mirror (worth adding one for
   *both* pipelines while at it, since inference debugging has no config
   today even for the SleepFM path):
   ```jsonc
   {
     "name": "🎯 OSF Step5: Infer DEBUG (apnea_binary, lstm, one context, CPU)",
     "type": "debugpy", "request": "launch",
     "program": "/home/boshra95/NSRR-tools/scripts/infer_osf_subject_windows.py",
     "args": ["--config", "/home/boshra95/NSRR-tools/configs/phase0_osf_config.yaml",
               "--task", "apnea_binary", "--task-type", "seq2label", "--head", "lstm",
               "--context", "30s", "--datasets", "apples", "--split", "val", "--cpu"],
     "console": "integratedTerminal", "justMyCode": false,
     "python": "/home/boshra95/osf_env/bin/python",
     "cwd": "/home/boshra95/NSRR-tools"
   }
   ```
5. **`train_osf_lora.py`** — Stage 2's own pilot config (§6.3 already calls
   for a short pilot at the smallest context before the full sweep; this
   is that pilot, runnable under the debugger too):
   ```jsonc
   {
     "name": "🔬 OSF Step6: LoRA Pilot DEBUG (apnea_binary, 30s, few epochs)",
     "type": "debugpy", "request": "launch",
     "program": "/home/boshra95/NSRR-tools/scripts/train_osf_lora.py",
     "args": ["--config", "/home/boshra95/NSRR-tools/configs/phase0_osf_config.yaml",
               "--task", "apnea_binary", "--task-type", "seq2label", "--head", "lstm",
               "--context", "30s", "--datasets", "apples",
               "--max-items", "50", "--epochs", "2", "--cpu"],
     "console": "integratedTerminal", "justMyCode": false,
     "python": "/home/boshra95/osf_env/bin/python",
     "cwd": "/home/boshra95/NSRR-tools",
     "env": {"SCRATCH": "/scratch/boshra95", "HOME": "/home/boshra95",
              "PYTHONPATH": "/home/boshra95/OSF-Open-Sleep-FM"}
   }
   ```
   (`--epochs`/`--max-items` flag names here are provisional — `train_osf_lora.py`
   is genuinely new code, not a fork, so its actual CLI will be whatever
   is implemented; update this config to match once written, don't treat
   these flag names as settled.)

**Correction to a premise worth flagging explicitly**: earlier in this
planning process it was assumed `test_context_window_dataset.py` already
had an established `launch.json` debug-config precedent to point at inside
the NSRR-tools *git repo itself*. **Code-verified 2026-08-10: that's not
where the precedent lives** — `NSRR-tools/.vscode/launch.json` was briefly
added and then deleted early in the project (2026-02-23, well before
`test_context_window_dataset.py` existed) and was never re-created inside
the repo; the real, current, working set of ~30 debug configs lives at
`/home/boshra95/.vscode/launch.json` (workspace root), which does already
include working entries for `test_context_window_dataset.py`
("🈂️ Phase0 Step2: Test ContextWindowDataset..."). The plan above targets
that same file, correctly.
