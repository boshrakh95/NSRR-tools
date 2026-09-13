# TSFM Model Comparison: SleepFM vs. OSF vs. PhysioOmni

**Purpose of this document**: synthesize the OSF and PhysioOmni baseline results into
a single comparison, with real numbers pulled directly from
`results/collected/*/analysis.csv` (not copied from planning docs without
re-verification), an honest accounting of why the LoRA sweeps stop where they stop,
an architecture/input-handling reference table, and concrete suggestions for how (and
whether) this belongs in `npj_main.tex`. **This document does not modify the paper** —
it is a proposal for review.

**Scope**: SleepFM (this paper's own baseline), OSF, and PhysioOmni only. **Mantis is
intentionally excluded** — it is a third TSFM baseline still being implemented on its
own branch, not yet merged. Every table below leaves a placeholder row/column for it
so it drops in later without restructuring (see §5.4).

All numbers below are **directly re-extracted from the collected CSVs** as part of
writing this document (`phase0_v3`, `phase0_v3_full`, `phase0_osf`,
`phase0_osf_lora`, `phase0_physioomni`, `phase0_physioomni_lora` — all six directories
confirmed to exist with populated `training.csv`/`analysis.csv` before writing
anything here), not transcribed from `CLAUDE.md` or the implementation plans. Where a
planning doc's number is quoted, it is the same number as this direct extraction — no
discrepancies were found. Metric: `mean_prob_auroc`, test split, `k = "all"` (i.e.
$K_{\max}$), Transformer head (matching `npj_main.tex`'s own convention: "Unless
otherwise stated, all primary results use the Transformer head").

---

## 0. Executive summary

- **OSF's frozen encoder is a genuinely strong baseline for sex, age, and BMI** — real,
  credible wins over SleepFM even on OSF's contamination-clean cohorts. It is
  **inconclusive-to-mixed for sleep efficiency and apnea**, the two tasks tied to
  dynamic physiological events rather than static subject characteristics. This
  split should be the headline framing, not "OSF beats SleepFM."
- **PhysioOmni's frozen encoder underperforms SleepFM at every single context and
  task tested**, by a wide margin at short context (e.g. sex classification, 30s: 0.754
  vs. 0.832) narrowing but not closing at long context (240m: 0.877 vs. 0.910). This
  is a real, unglamorous finding worth stating plainly, not omitting.
- **LoRA fine-tuning's value is itself task-dependent** — mirroring the main paper's
  own thesis that context value is task-dependent. For OSF, LoRA clearly helps apnea
  and sex/age (up to +5pp), is flat-to-mixed for BMI, and **actually *hurts*
  sleep efficiency at every context tested** (e.g. Transformer, 40m: frozen 0.773 →
  LoRA 0.753). This is a genuine, somewhat surprising result, not a smoothed-over
  narrative — worth a sentence in the paper if this material is included at all.
- **The two models' LoRA sweeps are incomplete in fundamentally different ways.** OSF's
  is a clean, deliberate, uniform stop (all 5 tasks × 2 heads trained through exactly
  `30s→120m`, `240m` withheld as a documented compute/timeline decision). PhysioOmni's
  is a jagged, partial stop (9 of 48 task×head×context cells; two tasks — BMI, age —
  have **zero** LoRA cells) driven by a mix of a genuine architectural compute-efficiency
  ceiling *and* a real 15-day operational GPU-billing mistake. These should not be
  described with the same language in the paper (see §3).
- **Recommendation on framing**: present this as a **generalization/robustness check**
  ("does the paper's central finding — that context value is task-specific — hold under
  a different frozen encoder?"), not a head-to-head leaderboard. The uneven LoRA
  coverage, the two different SleepFM baselines being compared against (full- vs.
  reduced-channel), and the OSF contamination issue make a clean leaderboard claim
  indefensible; a robustness/generalization framing is both accurate and still a real
  contribution. See §5 for suggested section placement and wording.

---

> **⚠ Update 2026-09-10 — read before finalizing anything from this document.**
> Two secondary (Tier 2) tasks were added and run for Phase 1 (frozen encoder) only:
> `depression_extreme_binary` (OSF + PhysioOmni) and `osa_binary_apples_postqc` (OSF
> only). Everything above this notice is the **original analysis, unchanged** — kept
> exactly as first written so you can see what the verdict was *before* these two
> tasks existed. **§7 (end of document) is the new material and states plainly where
> it confirms, adds to, or complicates the original verdict.** The single most
> important new finding is in §7.2 — a real tension between this document's own
> `apnea_binary` finding and the new `osa_binary_apples_postqc` result, on
> essentially the same clinical question and cohort. Read that before deciding
> whether either new task ships in the paper.

---

> **⚠ Update 2026-09-13 — Mantis Stage 1 (frozen) results now in this document;
> Stage 2 (LoRA) is not.** Mantis is no longer excluded (§0's "Mantis is
> intentionally excluded" text above describes this document's original state,
> before this update — left as-is; see §6.5 for the current framing). New/updated
> content: §1's architecture table (Mantis column filled in), §2.4 (the full
> results comparison — **read this first**: Mantis's frozen encoder beats SleepFM
> on sex, age, and the secondary OSA task, contradicting a real, pre-registered
> "expect a weak frozen result" prediction), §3.3/§3.4 (Stage 2 LoRA's honest,
> more-incomplete-than-OSF-or-PhysioOmni status, plus a real metric-comparability
> problem: no comparable frozen-vs-LoRA number exists for Mantis yet, for reasons
> beyond coverage), §4's Mantis subsection (real measured GPU-efficiency data), §5
> (new caveats), and §6.5 (updated framing recommendation). As with §7's own
> precedent, everything above this notice describing Mantis as excluded or
> placeholder-only is left unchanged, not fixed in place.

---

## 1. Architecture & Input-Handling Comparison

Adapted and extended from the three-way table already drafted in
`docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` §20 (code-verified there; reused here,
not re-derived) plus checkpoint/license facts from `docs/TSFM_BASELINE_CANDIDATES.md`
§2.1-2.2. **The Mantis column below is now filled in (2026-09-13, Stage 1/frozen
results only) — it was originally left as a placeholder** (see §2.4 for the full
results comparison this table's facts support).

> **⚠ Verification note, added 2026-09-13.** The original placeholder text pointed
> to "§5.4" for more on Mantis's synthetic-pretraining framing. No §5.4 exists in
> this document's current numbering (§5 has no subsections) — a stale cross-reference
> from an earlier draft, not fixed at the time. The content that reference was
> presumably pointing to now lives in §2.4 and §6.5 below.

| | SleepFM | OSF | PhysioOmni | Mantis |
|---|---|---|---|---|
| **Role in study** | This paper's primary encoder | Sleep-PSG-specific FM baseline | General physiological (non-sleep-specific) FM baseline | General-purpose, non-physiological TSFM baseline — classification-native, pretrained on generic time-series (not physiological signal at all) |
| **Checkpoint / license** | (used throughout paper; not itself under license review here) | HF `yang-ai-lab/OSF-Base`; **MIT confirmed** | HF `Weibang/PhysioOmni`; **code repo has no LICENSE file; HF weights repo declares CC-BY-4.0** — state both facts if shipped | HF `paris-noah/Mantis-8M`, real-time-series pretraining — **this is the checkpoint all Stage 1 results below use**, correcting this table's own original placeholder text (below), which assumed the *synthetic*-only `MantisPlus` checkpoint would be the one reported. `MantisPlus` (CauKer-synthetic-only pretrain, architecturally identical, differs by exactly 2 buffer tensors) exists as a planned internal ablation and has **not been run** — deferred, not dropped. License: **Apache-2.0 confirmed** (repo `LICENSE` + all three HF model cards) — the cleanest license of the three baselines |
| **Native per-call window** | Exactly 300s (5 min); incomplete trailing chunks dropped | Exactly 30s; no cross-epoch attention anywhere in the model | No fixed requirement — variable-length token sequence; real per-modality ceiling: EEG 512s / EOG 256s / ECG 102s / EMG 102s at its own reference resample rates | 512 samples (32 patches × 16) as released. We regenerate the sinusoidal positional buffer to cover 240 patches (3840 samples) instead, so a full 30-s epoch at 128 Hz is fed **natively**, not interpolated down to the released 512-sample window (3840→512 would give ~17 Hz effective resolution, below Nyquist for spindles/beta/EMG — rejected outright, not just judged suboptimal) |
| **What our pipeline actually uses per call** | 300s (5s sub-patches within) | 30s (architecturally forced) | 30s (a **deliberate choice**, not architecturally forced — see below) | 30s (3840 samples, 240 patches) — a full scoring epoch, via the buffer-regeneration above, not the architecturally-forced ceiling OSF has |
| **Channels used** | 4 modality groups (BAS/RESP/EKG/EMG), reduced (7-8ch) or full (≤23ch) config | 12-channel fixed input incl. snore + full thoracic/abdominal/airflow — **requires our full-channel HDF5s**, not the reduced/fast config | EEG/EOG/ECG/EMG only — **no respiratory pathway at all** (confirmed at 4 independent code locations); fast-channel HDF5s suffice | 6-slot canonical map (2 EEG-adjacent + LOC + ROC + EKG + EMG/CHIN + RESP, with per-slot candidate lists per cohort — our fast-channel data is **not** a uniform 6-channel set across cohorts, see `MANTIS_CLAUDE.md`) — same fast/reduced-channel HDF5 tree PhysioOmni uses, not OSF's full-channel tree |
| **Apnea comparable?** | Yes (reference) | Yes | **No — excluded**, stated reason (no RESP pathway; adding one would require new-modality pretraining, not fine-tuning) | **Yes** — channel-independent by construction (`Conv1d(in_channels=1)` per channel), so the RESP slot is processed by the identical encoder as every other channel; no architectural exclusion needed, unlike PhysioOmni |
| **Output / embedding shape (this project's convention)** | `[T,4,128]`, flat dim 512 | `[T,2,768]` (CLS ⊕ mean-pooled patches), flat dim 1536 | `[T,500]` (concatenated per-modality CLS, 2D, no sub-token axis) | `[T,6,512]` per channel — `combined` token = concat(CLS, mean-pooled patches) from the model's **last** transformer layer (empirically confirmed choice, not the authors' own per-checkpoint "optimal layer" recipe — see §2.4), flat dim 3072 |
| **Sequence-head `input_dim`** | 512 | 1536 | 500 | 3072 |
| **Encoder parameter count** | ~4.4M (Supplementary §"Model Parameter Counts") | 85,325,568 (strict-load-verified) | 13,871,304 total across 4 independent encoders (EEG 7.84M; EOG/ECG/EMG ~2.01M each) | 8,037,632 live params (identical for Mantis-8M and MantisPlus) — the "~8.1M" figure usually quoted for this model is the checkpoint-file total, which also includes a non-trainable positional buffer and a `prj` head that is dead weight at inference |
| **Fusion in released weights** | Contrastive alignment across 4 modality encoders; each still outputs its own vector | Single unified ViT — full fusion | **None** — 4 fully independent tokenizers/encoders, no cross-modal attention in the checkpoint; any fusion is downstream-constructed by us | **None** — channel-independent by construction, the same structural pattern as PhysioOmni (not a fused single-tensor design like SleepFM/OSF); channels are combined only downstream, by our own sequence heads |
| **Usage mode (Plan A/B/C)** | B (only option) | B (only option — architecture leaves no other choice) | **B (chosen)** — architecture could theoretically support up to ~1.7–8.5 min native, still short of every sweep point past 30s | B (only option) — `self.seq_len` is never referenced in `forward()`; no native long-context path exists at all |
| **SleepFM baseline used for comparison** | — | `phase0_v3_full` (full-channel) | `phase0_v3` (reduced/fast-channel, paper-primary) | `phase0_v3` (reduced/fast-channel, paper-primary) — same as PhysioOmni, not OSF's `phase0_v3_full` |
| **Pretraining-cohort overlap with our 4 test cohorts** | N/A | **SHHS: severe (confirmed, exact-ID match, 87.7% of our SHHS test subjects were in OSF's own pretrain train/valid split). STAGES: confirmed clean. MrOS: confirmed clean (downstream/eval-only in OSF, not pretraining). APPLES: confirmed clean.** | None of our 4 cohorts are in PhysioOmni's pretraining corpus (TUH/CAP/Sleep-EDF/DEAP — none are NSRR) | **Provably zero** — Mantis-8M's pretraining corpus is generic real time-series archives, not physiological signal of any kind; no NSRR cohort could appear in it by construction. This rests on the published pretraining-corpus description, not an exact-ID check the way OSF's overlap was quantified — no ID-level check is possible (or needed) when the pretraining data was never PSG in the first place |
| **Peer review status** | Published (ICML 2024 + Nat. Med. extension) | Published (ICML 2026) | **arXiv only, v3 dated 2026-03 — never peer-reviewed** | arXiv only (`arXiv:2502.15637`); peer-review status not otherwise confirmed in our own docs — treat as unreviewed until directly checked, same category as PhysioOmni |
| **A caveat worth stating if PhysioOmni is cited on its own merits** | — | — | On its own best-fit downstream task (HMC sleep staging) in its own paper, PhysioOmni's reported number (0.7377 balanced accuracy) **does not beat its own paper's non-foundation-model baseline** (FeatFusion, 0.7478) | — |
| **A caveat worth stating if Mantis's frozen result is cited on its own merits** | — | — | — | The opposite direction of caveat from PhysioOmni's: `docs/TSFM_THIRD_MODEL_DECISION.md` explicitly pre-committed to expecting **a weak frozen result** ("Expect — and pre-commit to reporting — a weak frozen result... general-purpose pretraining transfers to sleep PSG only with adaptation"), citing published evidence that freezing this class of model "leads to a huge decrease in performance" on EEG. **That prediction did not hold** — see §2.4. Worth stating precisely because it was a real, pre-registered expectation that the data contradicted, not a post-hoc framing choice |

**Why OSF and PhysioOmni are each compared against a different SleepFM variant, not the
same one**: OSF needs full-channel signal (thoracic/abdominal/airflow/snore), which
only exists in the full-channel (`phase0_v3_full`) HDF5 tree; PhysioOmni needs only
EEG/EOG/ECG/EMG, which the paper-primary reduced/fast-channel tree
(`phase0_v3`) already carries. **Do not cross-compare OSF's numbers against
`phase0_v3` or PhysioOmni's against `phase0_v3_full`** — those are not the SleepFM
baselines each was actually run against, and the reduced- vs. full-channel gap itself
is non-trivial (Section 5.7 of `npj_main.tex`, "Full-channel configuration helps
cardiorespiratory task": +0.03–0.05 AUROC for apnea/BMI, smaller for sex).

**Mantis uses the fast/reduced-channel tree and compares against `phase0_v3`, the
same baseline as PhysioOmni** — the same warning applies: do not cross-compare
Mantis's numbers against `phase0_v3_full`.

---

## 2. Results Comparison

All tables: Transformer head, `mean_prob_auroc`, test split, $K = K_{\max}$ (matching
the npj paper's own headline-metric convention). LSTM-head numbers were pulled too and
show the same qualitative ranking between models at every context (available in
`results/collected/*/analysis.csv` if needed) — omitted here to avoid duplicating
tables that add no new pattern, following the main paper's own convention of
relegating the LSTM counterpart to supplementary.

**— denotes a cell that was not run**, not a zero and not "no benefit." Never present a
blank/missing LoRA cell as if it were a completed, unremarkable result.

### 2.1 OSF vs. SleepFM (full-channel)

**sex_binary** ("sex classification")

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (full-ch.) | 0.843 | 0.867 | 0.899 | 0.925 | 0.929 | 0.920 |
| OSF-frozen | 0.929 | 0.938 | 0.950 | 0.959 | 0.963 | 0.954 |
| OSF-LoRA | 0.938 | 0.950 | 0.952 | 0.953 | 0.954 | — |

**sleep_efficiency_binary**

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (full-ch.) | 0.707 | 0.715 | 0.761 | 0.791 | 0.798 | 0.825 |
| OSF-frozen | 0.714 | 0.740 | 0.773 | 0.794 | 0.802 | 0.841 |
| OSF-LoRA | 0.707 | 0.744 | 0.753 | 0.787 | 0.803 | — |

**apnea_binary** ("apnea detection")

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (full-ch.) | 0.800 | 0.826 | 0.870 | 0.895 | 0.900 | 0.901 |
| OSF-frozen | 0.813 | 0.853 | 0.870 | 0.905 | 0.910 | 0.911 |
| OSF-LoRA | 0.816 | 0.869 | 0.893 | 0.917 | 0.923 | — |

**age_class** ("age-group prediction")

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (full-ch.) | 0.859 | 0.869 | 0.887 | 0.905 | 0.908 | 0.911 |
| OSF-frozen | 0.907 | 0.922 | 0.931 | 0.941 | 0.942 | 0.942 |
| OSF-LoRA | 0.909 | 0.937 | 0.946 | 0.950 | 0.951 | — |

**bmi_binary** ("BMI (obese)")

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (full-ch.) | 0.777 | 0.788 | 0.801 | 0.811 | 0.812 | 0.816 |
| OSF-frozen | 0.823 | 0.841 | 0.842 | 0.847 | 0.848 | 0.845 |
| OSF-LoRA | 0.816 | 0.837 | 0.847 | 0.839 | 0.839 | — |

**Read naively, OSF beats SleepFM at every single cell above — this is misleading on
its own.** OSF's pretraining set includes SHHS (confirmed, severe: 87.7% of our SHHS
test subjects were directly in OSF's pretrain train/valid split by exact-ID match) and
excludes APPLES, MrOS, STAGES (all three confirmed clean by exact-ID match against
OSF's own shipped splits). When the pooled numbers above are broken down by
cohort — done previously in `docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s "Stage 1
Results" using the underlying per-window parquets, not re-derived here — the picture
splits cleanly by task category:

- **Real, credible OSF wins, surviving the contamination check**: `sex_binary`,
  `bmi_binary`, `age_class`. On all three, the **clean** APPLES cohort shows a
  *larger* OSF advantage (e.g. age_class: +11.5 to +15.2pp) than the **contaminated**
  SHHS cohort (+4.3 to +5.2pp) — the opposite of what contamination alone would
  produce, so this is genuine encoder-quality evidence, not memorization.
- **Inconclusive**: `sleep_efficiency_binary` — clean cohorts split by head (APPLES:
  lstm +5.5pp, transformer −3.8pp; MrOS favors SleepFM in both heads); only the
  contaminated SHHS cohort consistently favors OSF.
- **Genuinely mixed, not explained by contamination**: `apnea_binary` — OSF **loses**
  on two clean cohorts (APPLES, MrOS) and **wins** on one clean cohort (STAGES) plus
  the contaminated one (SHHS). Never present a single pooled apnea AUROC for OSF
  without this breakdown.
  > **⚠ Update 2026-09-10**: the new `osa_binary_apples_postqc` task — essentially
  > the same clinical question (AHI-based OSA severity), evaluated on the same
  > APPLES cohort, trained as its own single-cohort model rather than as part of
  > this pooled 4-cohort one — shows OSF **winning by the largest margin of any
  > task in this entire document** (+6 to +11.5pp). That is the opposite direction
  > from "OSF loses on APPLES" stated just above. See §7.2 for the full analysis —
  > there is a real, defensible explanation (training-set size/composition, not an
  > inconsistent model), but the optics next to this exact sentence are a genuine
  > risk worth reading before including either task in the paper.

**LoRA vs. frozen, task-by-task (a second, independent finding, not previously
headlined this way)**: LoRA clearly helps `apnea_binary` (+2 to +5pp at matched
context) and modestly helps `sex_binary`/`age_class`; it is flat-to-slightly-negative
for `bmi_binary`; and it **consistently underperforms the frozen encoder for
`sleep_efficiency_binary`** at every context tested (e.g. Transformer 40m: frozen
0.773 → LoRA 0.753; 80m: 0.794 → 0.787). This is a genuine, non-obvious result — LoRA
fine-tuning is not a uniform improvement, and its value is itself task-dependent, in
the same spirit as the main paper's own thesis about context length. Worth a sentence
if this material ships.

### 2.2 PhysioOmni vs. SleepFM (reduced/fast-channel)

`apnea_binary` is excluded throughout — PhysioOmni has no respiratory pathway.

**sex_binary**

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (reduced-ch.) | 0.832 | 0.851 | 0.872 | 0.897 | 0.905 | 0.910 |
| PhysioOmni-frozen | 0.754 | 0.817 | 0.864 | 0.879 | 0.884 | 0.877 |
| PhysioOmni-LoRA | 0.790 | 0.851 | — | — | — | — |

**sleep_efficiency_binary**

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (reduced-ch.) | 0.707 | 0.711 | 0.760 | 0.796 | 0.815 | 0.831 |
| PhysioOmni-frozen | 0.693 | 0.709 | 0.738 | 0.766 | 0.788 | 0.816 |
| PhysioOmni-LoRA | 0.691 | 0.711 | — | — | — | — |

**age_class**

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (reduced-ch.) | 0.854 | 0.870 | 0.877 | 0.900 | 0.902 | 0.905 |
| PhysioOmni-frozen | 0.807 | 0.832 | 0.845 | 0.852 | 0.852 | 0.854 |
| PhysioOmni-LoRA | — | — | — | — | — | — |

**bmi_binary**

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (reduced-ch.) | 0.747 | 0.755 | 0.755 | 0.769 | 0.766 | 0.777 |
| PhysioOmni-frozen | 0.702 | 0.728 | 0.735 | 0.735 | 0.737 | 0.746 |
| PhysioOmni-LoRA | — | — | — | — | — | — |

**PhysioOmni's frozen encoder underperforms SleepFM at every context, on every
comparable task**, with no contamination confound to explain it away (PhysioOmni's
pretraining corpus contains none of our four NSRR cohorts). The gap is largest at
short context (e.g. sex, 30s: −7.8pp) and narrows but never closes at long context
(sex, 240m: −3.3pp). This is consistent with a real, structural difference: PhysioOmni
was pretrained on general clinical/BCI-scale EEG corpora (TUH, CAP, Sleep-EDF, DEAP;
per-modality context ceiling ≤512s) rather than full-night NSRR-scale PSG, and (as
already flagged in `CLAUDE.md`) even in its own paper's best-fit downstream task it
did not beat a non-foundation-model baseline. Where LoRA was actually run (`sex_binary`
through 40m; `sleep_efficiency_binary` through 10m only), it is a real but modest
improvement over frozen at every matched context — but **`bmi_binary` and `age_class`
have zero LoRA cells**, and no comparison table should imply otherwise.

### 2.3 Cross-cutting caveats

- **Split mismatch (OSF only)**: SleepFM and OSF Stage 1 use subtly different
  train/val/test splits — both filter by "has embedding file" using the identical RNG
  shuffle pattern, but against different embedding directories with slightly different
  per-subject coverage (e.g. APPLES has exactly 1 subject with an OSF embedding but no
  SleepFM embedding). Because the shuffle's entire output permutation changes when the
  input list differs by even one subject, **individual subjects can land in different
  splits between the already-published SleepFM numbers and the OSF numbers above** —
  not a rounding-level discrepancy. Not fixed (fixing it would invalidate one of two
  already-completed, already-analyzed result sets) — state this plainly if the OSF
  comparison ships, rather than implying subject-identical splits.
- **PhysioOmni has no analogous split-mismatch investigation on record** — not
  because it was checked and found fine, but because it hasn't been checked. Flag as
  an open item before trusting the PhysioOmni-vs-SleepFM gap to the same precision as
  the OSF one.
- **LSTM head**: pulled directly from the same CSVs; shows the same task-category
  ranking (OSF strong on sex/age/BMI, mixed on apnea/sleep-efficiency; PhysioOmni
  below SleepFM throughout) at every context. Not tabulated here to avoid duplicating
  a pattern that adds no new information, consistent with how `npj_main.tex` itself
  treats the LSTM head as secondary evidence.

### 2.4 Mantis (Stage 1, frozen) vs. SleepFM (reduced/fast-channel) — added 2026-09-13

**Scope of this subsection: Stage 1 (frozen encoder) only.** Stage 2 (LoRA) is
actively running as of this writing and is nowhere near comparable coverage —
see §3.4 for its own honest accounting, with no results table, for reasons
explained there. Numbers below are pulled directly from
`results/collected/phase0_mantis/analysis.csv` (already collected; matches
`MANTIS_CLAUDE.md`'s own "14/14 task×head done" status note) against
`results/collected/phase0_v3/analysis.csv` — same metric convention as the rest
of this document (`mean_prob_auroc`, test split, `k = "all"`, Transformer head).
The checkpoint is **`Mantis-8M`** (real-time-series pretraining), not the
synthetic-only `MantisPlus` ablation (deferred, not run) — see §1's corrected
table entry. Unlike PhysioOmni, **apnea is in scope** for Mantis (channel-independent
architecture, no respiratory-pathway exclusion needed), so all seven tasks this
document covers for SleepFM/OSF are covered here too.

**sex_binary**

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (reduced-ch.) | 0.832 | 0.851 | 0.872 | 0.897 | 0.905 | 0.910 |
| Mantis-frozen | 0.863 | 0.891 | 0.916 | 0.927 | 0.935 | 0.923 |

**age_class**

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (reduced-ch.) | 0.854 | 0.870 | 0.877 | 0.900 | 0.902 | 0.905 |
| Mantis-frozen | 0.856 | 0.885 | 0.912 | 0.918 | 0.923 | 0.919 |

**apnea_binary**

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (reduced-ch.) | 0.753 | 0.793 | 0.825 | 0.847 | 0.857 | 0.854 |
| Mantis-frozen | 0.733 | 0.792 | 0.832 | 0.851 | 0.857 | 0.842 |

**bmi_binary**

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (reduced-ch.) | 0.747 | 0.755 | 0.755 | 0.769 | 0.766 | 0.777 |
| Mantis-frozen | 0.746 | 0.774 | 0.782 | 0.778 | 0.781 | 0.770 |

**sleep_efficiency_binary**

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (reduced-ch.) | 0.707 | 0.711 | 0.760 | 0.796 | 0.815 | 0.831 |
| Mantis-frozen | 0.707 | 0.720 | 0.764 | 0.787 | 0.807 | 0.827 |

**depression_extreme_binary** (secondary; APPLES + STAGES)

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (reduced-ch.) | 0.756 | 0.739 | 0.750 | 0.749 | 0.754 | 0.746 |
| Mantis-frozen | 0.751 | 0.748 | 0.755 | 0.746 | 0.760 | 0.755 |

**osa_binary_apples_postqc** (secondary; APPLES only)

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (reduced-ch.) | 0.789 | 0.804 | 0.853 | 0.888 | 0.856 | 0.861 |
| Mantis-frozen | 0.818 | 0.837 | 0.872 | 0.887 | 0.879 | 0.903 |

**Headline finding, stated plainly per this project's own standing instruction not
to soften a result either way: Mantis's frozen encoder beats SleepFM outright on
sex (+1.3 to +4.4pp at every single context) and on age (+0.2 to +3.5pp at every
context, smallest at 30s), and wins at five of six contexts on the secondary OSA
task (+1.9 to +4.2pp, with an essential tie at 80m: −0.1pp)** — the largest margin
of any comparison in this document, at its largest point (240m: +4.2pp). This is a
general-purpose, non-physiological time-series model — pretrained on generic
time-series archives, never on a physiological signal, let alone PSG — outperforming
a domain-specific encoder pretrained on 585,000 hours of real PSG, **frozen, with no
fine-tuning at all.**
[All percentage-point figures independently recomputed from the tables above,
`round((mantis - sleepfm) * 100, 1)` per context — verify against the raw tables
before quoting a different range.]

**This directly contradicts a real, pre-registered prediction, not a strawman.**
`docs/TSFM_THIRD_MODEL_DECISION.md` explicitly committed, before any Mantis code
was written, to "expect — and pre-commit to reporting — a weak frozen result,"
citing published evidence that freezing this class of model "leads to a huge
decrease in performance" on EEG, and framed a good Stage 1 result as the unlikely
outcome. That expectation did not hold for sex, age, and OSA. Report this as the
genuine surprise it is, not as a foregone conclusion dressed up after the fact.

Apnea, sleep efficiency, and BMI are closer calls, not clean Mantis wins, and none
should be folded into the headline above:
- **Apnea** is closely matched at every context (within 2.0pp either direction),
  essentially tied at its own saturation point (0.857 vs. 0.857 at 120m), with no
  consistent direction — Mantis trails at 30s/10m/240m (−2.0, −0.1, −1.2pp) and
  edges ahead at 40m/80m (+0.7, +0.4pp).
- **Sleep efficiency** is nearly tied through 40m (0.0 to +0.9pp) and runs slightly
  *behind* SleepFM from 80m on (−0.4 to −0.9pp) — the one task where Mantis is
  consistently, if narrowly, the weaker model at long context.
- **BMI** is close throughout with no consistent direction (−0.7 to +2.7pp; Mantis
  ahead at four of six contexts, SleepFM ahead at 30s and 240m).
- **Depression** is flat and noisy for both models, as established elsewhere in
  this document.

#### 2.4.1 Context-sensitivity comparison — the most important finding in this section

The paper's and this document's central claim is that context value is
task-specific, not that any particular AUROC number replicates exactly. The
question that matters most for Mantis is whether *which* tasks are context-sensitive
looks the same under a completely different, non-physiological encoder.
**It does, almost exactly.**

| Task | Mantis $L^*$ | Mantis $\Delta$ (30s→best) | SleepFM $L^*$ | SleepFM $\Delta$ (30s→best) |
|---|---|---|---|---|
| Apnea detection | 120m | +0.123 | 120m | +0.103 |
| Sleep efficiency | 240m (still rising) | +0.120 | 240m (still rising) | +0.124 |
| Sex classification | 120m | +0.072 | 240m (still rising) | +0.079 |
| Age-group prediction | 80m | +0.066 | 120m | +0.051 |
| BMI (obese) | 40m | +0.036 | 240m (nominal; flat) | +0.030 |
| OSA severity (secondary) | 240m | +0.085 | 80m | +0.098 |
| Depression (secondary) | 120m | +0.009 | 30s (no benefit) | +0.000 |

($L^*$: smallest context within 0.005 AUROC of the sweep peak, same definition as
`npj_main.tex` equation (7); $\Delta$: peak minus 30s, both at $K=K_{\max}$,
Transformer head, independently computed from the collected CSVs above, not copied
from any planning doc.)

**The qualitative split replicates cleanly**: sleep efficiency and apnea are the
two most context-sensitive tasks under both encoders (their exact rank order swaps
by a hair — apnea edges ahead for Mantis, sleep efficiency for SleepFM — but both
remain the top two by a wide margin over everything else); sex and age form a
context-sensitive-but-more-moderate middle tier under both; BMI is the least
context-sensitive primary task under both, with closely matched $\Delta$ (+0.036
vs. +0.030). This is a genuine generalization result: a completely different,
non-physiological pretraining source produces the same qualitative task ordering
the paper's central SleepFM-based finding rests on.

**One real divergence, flagged rather than resolved, per this document's own
standing practice (§7.2's precedent)**: SleepFM's sex AUROC rises monotonically
through every context tested, still climbing at 240m, the paper's own basis for
calling sex "still rising at the longest context evaluated." **Mantis's sex AUROC
peaks at 120m (0.935) and then *declines* at 240m (0.923, Transformer; 0.925→0.922,
LSTM)** — the same shape SleepFM shows for no primary task. Two things are true at
once here: **no bootstrap confidence interval has been computed for any Mantis
number** (the CI columns in `phase0_mantis/analysis.csv` are entirely `NaN` —
confirmed directly, not assumed), so there is no statistical basis yet to call this
dip real rather than sampling noise; and the test-subject count is stable and large
at both contexts (1,431 subjects at every context for sex, no cohort-dropout
confound). This is reported as an open, unresolved divergence, not a claim that
Mantis's context-value story differs from SleepFM's — the weight of the other six
tasks argues the split replicates. Whether this is real requires a bootstrap CI
pass this document does not yet have.

#### 2.4.2 Caveats specific to the Mantis comparison

- **No bootstrap confidence intervals exist for any Mantis number** — every
  `mean_prob_auroc_ci_lo`/`_hi` cell in `phase0_mantis/analysis.csv` is `NaN`,
  confirmed directly. Every point estimate above, not just the sex 240m dip, should
  be read with this in mind; OSF's and PhysioOmni's own CI status is not
  independently re-verified here, but Mantis's absence is a real, checked fact,
  not an assumption.
- **A small test-population mismatch exists between Mantis and SleepFM, the same
  class of issue already documented for OSF in §2.3** (though smaller in absolute
  size here): test-subject counts differ by single-digit-to-a-few-dozen subjects
  per task (e.g. apnea_binary: 2,077 vs. 2,054; depression_extreme_binary: 241 vs.
  229), consistent with a slightly different per-subject embedding-extraction
  success population shifting `rng.shuffle()`'s output the same way OSF's did.
  **Not independently investigated to OSF's level of rigor** (no exact-ID
  cross-check was run) — flagged as an open item, not a resolved non-issue.
- **`mean_pool` head was never run for Mantis, matching OSF's and PhysioOmni's own
  status** — no H3-style "does the MeanPool-vs-temporal-head gap replicate" claim
  can be made from this data.
- **Val-split threshold-tuning was never run**, same standing caveat as §5.

---

## 3. Honest computational-cost narrative

**Why isn't there a full 6-context × 2-head × 5-task LoRA sweep for every model?**
Grounded in `docs/LORA_GPU_THROUGHPUT_INVESTIGATION.md`'s real, measured findings, not
a schedule excuse.

### 3.1 The underlying mechanism (applies to any model fine-tuned this way)

LoRA fine-tuning here means running the **raw signal**, not pre-extracted embeddings,
through the (LoRA-adapted) backbone for every 30-second epoch inside a training
window, because a frozen-encoder-only pipeline cannot backpropagate into the backbone.
Concretely, `CombinedOSFLoRAModel.forward()` runs every epoch in a window through the
full backbone individually, so a 240-minute window (480 raw epochs) costs roughly
480× what a 30-second window (1 epoch) costs per training step — this scaling is
**architectural, not a tuning artifact**, and it is the same mechanism for any model
fine-tuned the same way (explicitly flagged as relevant to Mantis's eventual LoRA
stage too, not just OSF/PhysioOmni).

Measured on OSF (`osf_lora`, single-segment, verified timings):

| Context | Raw epochs/window | min/epoch | Est. TFLOP/s |
|---|---|---|---|
| 30s | 1 | 18.8 | ~2.0 |
| 10m | 20 | 58.9 | ~12.8 |
| 40m | 80 | 112.3 | ~18.5 |
| 80m | 160 | 217.2 | ~19.2 |

**The key finding: short contexts are overhead-bound, long contexts are
compute-bound.** GPU utilization jumps ~6.4× from 30s to 10m, then plateaus (+3% only)
from 40m to 80m — meaning a 1g.10gb→3g.40gb MIG upgrade (3× more compute) measured
**zero** speedup at 30s (overhead-dominated regime) but real gains at 40m+ where
compute genuinely dominates wall time. This is why generic fixes (mixed precision,
bigger GPU allocation) that seem obviously helpful for "cost scales with context"
measured **zero** speedup when tried at short contexts — they were tested in the wrong
regime. The actual lever that worked was `chunk_batch_size` (how many raw epochs get
batched per backbone forward call): raising it 16→64 gave a confirmed 3.28× speedup by
amortizing per-call overhead — a GPU-scheduling fix, not a numerics or hardware one.

### 3.2 What this means concretely for scope decisions, per model

**OSF — a clean, deliberate, uniform stopping rule.** All 5 Tier-1 tasks × 2 heads (10
runs) were trained through `30s → 10m → 40m → 80m → 120m`; `240m` was never started for
any of the 10. This is a genuine methodological choice, made once the per-context cost
curve above was measured and understood: extending to 240m would cost roughly
2× the 120m wall-time per run (480 vs. 240 raw epochs/window) across all 10 runs, for
a comparison-completeness benefit judged not worth it for a first LoRA-vs-frozen pass.
**Every task/head combination in the OSF-LoRA table is missing the exact same cell
(240m) for the exact same documented reason** — a rectangular, explainable gap, not an
arbitrary one.

**PhysioOmni — a jagged, partial stop, for a materially different and less clean set
of reasons.** Two of the four comparable tasks (`bmi_binary`, `age_class`) have **zero**
LoRA cells; the other two are complete only through 10m (sleep efficiency) or 40m
(sex). This did not happen for one clean reason the way OSF's did:
1. **A genuine architectural compute-efficiency ceiling.** PhysioOmni's four encoders
   have small hidden dimensions (100–200, and specifically not multiples of 8 for
   three of the four), which use GPU tensor cores poorly. Measured throughput:
   **~0.69 TFLOP/s, ~3.6% of this account's realistic fp32 ceiling on a 3g.40gb
   slice** — a training-script change cannot fix this; it is a property of the
   released checkpoint's architecture.
2. **A real 15-day operational mistake, not an architectural one.** A GPU-allocation
   misconfiguration (`2g.20gb`→whole-H100 request) left an 80m job `PENDING` for 15
   real days because whole-card jobs bill 2.3× and starve under the account's
   fairshare policy — caught only when progress had visibly stalled, not by design.
3. Additional real bugs consumed further time before any long-context run could
   proceed: an OOM on a mostly-MrOS shard, a non-atomic cache-write bug that produced
   silently-corrupt cache files, and a `chunk_batch_size`-independent OOM specific to
   PhysioOmni's per-encoder memory profile that required gradient checkpointing to
   resolve for 240m specifically.

**Do not describe these two situations with the same language in the paper.** OSF's
gap is a single, uniform, planned stopping point applied identically across every run
— defensible as "we measured the cost curve and chose not to extend past 120m for this
comparison round." PhysioOmni's gap is a combination of a real architectural
limitation (worth stating — it is a genuine property of the model) and an operational
setback (worth stating separately — it is not a property of the model, and shouldn't be
allowed to make the model look categorically worse at fine-tuning than it may actually
be). If PhysioOmni's LoRA numbers are reported at all, say explicitly that its sweep
coverage is a fraction of OSF's for two compounding reasons, only one of which reflects
something inherent to PhysioOmni itself.

### 3.3 Stopping-criteria summary table

| Model | Stage 1 (frozen) coverage | Stage 2 (LoRA) coverage | Stopping rule |
|---|---|---|---|
| SleepFM | Complete: all tasks × 3 heads × 6 contexts | N/A (SleepFM used frozen throughout the paper) | — |
| OSF | Complete: 5 Tier-1 tasks × 2 heads × 6 contexts (`mean_pool` not run) | 5 tasks × 2 heads × **5 of 6 contexts** (`30s`–`120m`; `240m` withheld) | Deliberate, uniform, cost-curve-informed; documented before the fact |
| PhysioOmni | Complete: 4 tasks (apnea excluded) × 2 heads × 6 contexts (`mean_pool` not run) | **9 of 48** task×head×context cells; 2 of 4 tasks have zero LoRA cells | Architectural compute ceiling + a real 15-day operational GPU-billing incident + a sequence of real bugs; not a single clean rule |
| Mantis | Complete: 7 tasks (apnea included) × 2 heads × 6 contexts (`mean_pool` not run) | **Training completed for 5 of 30 LSTM cells** (30s only, for `age_class`/`apnea_binary`/`bmi_binary`/`sex_binary`/`depression_extreme_binary`; `depression_extreme_binary` additionally has 10m). Zero cells for `sleep_efficiency_binary`, `osa_binary_apples_postqc`, or any Transformer-head run. **Subject-level K-aggregated inference has never been run for any cell** — see §3.4 | Actively running as of this writing (2026-09-13); GPU memory/queue calibration still in progress, not a deliberate stop — see §3.4 |

---

### 3.4 Mantis Stage 2 (LoRA) — actively running, more incomplete than the table above suggests, added 2026-09-13

**This is not a results section — no comparable frozen-vs-LoRA AUROC number exists
for Mantis yet, for a reason beyond incomplete coverage.** State this plainly before
anything else: `infer_mantis_lora_subject_windows.py` (the subject-level,
K-aggregated inference script — the one that produces the `mean_prob_auroc,
k="all"` metric this entire document is built on) **has not been run for any
Mantis LoRA checkpoint, at any context.** The `test_auroc` value currently sitting
in each `summary.csv` on disk is computed by the training script's own held-out
evaluation over a small fixed set of windows per subject (the same `w=5`-style
window-sampling convention used for training/validation, not the full,
non-overlapping, subject-aggregated $K_{\max}$ protocol). **These numbers are not
directly comparable to any other figure in this document, including Mantis's own
Stage 1 numbers in §2.4, and are deliberately not reproduced here** — showing them
side-by-side with §2.4's table would repeat exactly the mistake this document
elsewhere warns against (§2.3's convention statement: never present numbers under
different metrics as if they were the same measurement). A real Stage 1-vs-Stage 2
comparison for Mantis requires that inference script to be run first.

**Exactly which cells have completed training, checked directly against
`/scratch/boshra95/psg/unified/results/phase0_mantis_lora/` as of 2026-09-13**
(a `metrics.json` alongside `best_model.pt` means done; a lone `resume.pt` with no
`metrics.json` means still training):

| Task | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| `age_class` (lstm) | done | in progress | — | — | — | — |
| `apnea_binary` (lstm) | done | in progress | — | — | — | — |
| `bmi_binary` (lstm) | done | in progress | — | — | — | — |
| `sex_binary` (lstm) | done | in progress | — | — | — | — |
| `depression_extreme_binary` (lstm) | done | done | in progress | in progress | — | — |
| `sleep_efficiency_binary` | — | — | — | — | — | — |
| `osa_binary_apples_postqc` | — | — | — | — | — | — |

No Transformer-head LoRA run has been started for any task. `mean_pool` is
deferred, matching PhysioOmni's own LoRA registry precedent. Several of the
"in progress" cells above are running on the cluster right now, so this table is a
snapshot, not a fixed state — re-check the results directory directly rather than
trusting this table if it matters later.

**Real, measured per-training-epoch cost at 30s** (from each completed run's own
`summary.csv`, `training_time_min / n_epochs_run` — a real measurement, not
inferred):

| Task | Epochs run | Wall time (min) | Min/epoch | Achieved TFLOP/s |
|---|---|---|---|---|
| `apnea_binary` | 16 | 24.72 | **1.55** | 6.63 (1.34% of a whole-H100 TF32 peak) |
| `bmi_binary` | 10 | 11.06 | 1.11 | 6.63 |
| `sex_binary` | 17 | 17.03 | 1.00 | 6.62 |
| `depression_extreme_binary` | 8 | 11.52 | 1.44 | 6.62 |
| `age_class` | 15 | 165.77 | **11.05** | 6.64 |

`age_class`'s wall-clock time is 7-10× every other task's, despite **essentially
identical achieved TFLOP/s** across all five rows (~6.6, all within 0.3%) — the
anomaly is in wall-clock time, not compute throughput, which points toward
something external (all four 30s jobs ran concurrently on the same shared node,
`g34`) rather than a genuine per-task compute difference. **Not investigated
further, per explicit decision** — `apnea_binary`'s clean 1.55 min/epoch is used
below as the representative 30s baseline; `age_class`'s number is reported as a
real measurement but flagged as an unexplained outlier, not used for any
extrapolation.

**One real longer-context data point exists**: `depression_extreme_binary`'s 10m
run completed (9 epochs, 85.48 min → **9.50 min/epoch**). Its own 10m/30s ratio is
**6.6×** — well below the ~20× this codebase's own documented mechanism for OSF
would predict from raw-epoch-count scaling alone (10m has 20 raw 30-s epochs per
training window vs. 30s's 1; §3.1's finding that LoRA compute scales
~linearly in raw epochs per window was established on OSF and explicitly flagged
in `MANTIS_CLAUDE.md` as "relevant to Mantis's eventual LoRA stage too," not yet
confirmed for Mantis specifically until now). Plausible explanation, not yet
verified by a controlled A/B: Mantis had TF32 and a large `chunk_batch_size` (192,
confirmed via Pilot 3 to show no further sensitivity to this knob) built in from
day one, unlike OSF and PhysioOmni, which discovered these fixes only after
weeks of running unoptimized — so Mantis's 30s baseline may already be closer to
its own compute-bound ceiling than OSF's 30s number was, making the jump to 10m
proportionally smaller. **This is one data point, from the smallest task in the
comparison (5,615 training windows vs. 43,000-48,000 for the four larger Tier-1
tasks) — it may not transfer directly to the larger tasks' own 10m cost, which is
why the estimate below is still labeled an estimate, not deflated by this factor.**

**ESTIMATE, explicitly not a measurement, for the four larger Tier-1 tasks'
longer contexts** — using the same naive linear-in-raw-epochs assumption this
document already applies to OSF in §3.1, anchored to `apnea_binary`'s measured
30s baseline (1.55 min/epoch):

| Context | Raw epochs/window (vs. 30s) | **ESTIMATED** min/epoch |
|---|---|---|
| 10m | 20× | **~31** |
| 40m | 80× | **~124** |
| 80m | 160× | **~248** |
| 120m | 240× | **~371** |
| 240m | 480× | **~742** |

**This estimate is very likely an overstatement, based on the one real longer-context
measurement available** (depression's 10m came in at 6.6× its own 30s cost, not the
naive 20× used to build this table) — but that correction factor is not applied
here, because it comes from a much smaller task and has not been confirmed for
`apnea_binary`/`bmi_binary`/`sex_binary`/`age_class` specifically. Treat the table
above as a conservative upper bound pending a real 10m measurement on one of the
four larger tasks (already running as of this writing).

**Why this is worth stating as a finding, not just an apology for missing cells**:
fine-tuning a raw-signal backbone end-to-end at PSG-scale context lengths is
genuinely, structurally expensive — even a small, 8M-parameter, efficiently-batched
model whose 30s condition already reaches double-digit-percent of a whole H100's
TF32 peak faces a real per-epoch cost that plausibly reaches several hours by
240m for the larger tasks. This is a real, honest constraint on what "just fine-tune
the backbone at every context length" costs in practice for long PSG recordings,
independent of which specific model is used, and is itself part of the answer to
why this comparison's LoRA stage — across all three baselines, not only Mantis —
is the least complete part of this document.

**Other real, concrete incompleteness causes, not generic scheduling color**: the
registry's `context_micro_batch` was a flat, uncalibrated `32` at every context
(the registry's own comment already flagged this as unresolved, dated before this
session) — this OOM'd every context past 30s on a **whole 80GB H100**, not a small
slice, once real jobs were submitted. Separately, a whole-H100 request queued for
approximately 44 hours on the Nibi cluster specifically (772 pending vs. 90 running
GPU-wide at the time) — directly contradicting this project's own Fir-cluster-derived
assumption that a whole-card request costs nothing extra in queue time; that finding
was real for Fir and does not transfer to Nibi's current load. A memory-calibration
pilot to fix the `context_micro_batch` values (rather than the flat placeholder) is
in progress as of this writing.

---

## 4. Intrinsic architectural sources of GPU inefficiency (honest per-model take)

**Draft for review — not yet proposed for any specific location in the paper.** This
is a candidate short paragraph (or a compact table + a few sentences) for wherever the
computational-cost narrative in §3 lands, if this material ships at all. It answers a
narrower question than §3: not "why did the sweep stop where it stopped" but "what,
specifically, about each model's own architecture — as opposed to how carefully we
drove it — makes it waste GPU throughput." Only claims backed by a real measurement
are stated as fact; everything else is flagged as structural inference.

**Why this only matters for the LoRA condition.** In the frozen condition, a backbone
is run once per subject to produce a saved embedding, then never touched again — a
saved embedding is a saved embedding, and the compute-cost story in `npj_main.tex`
(Extended Data Figure, the FLOPs-vs-AUROC power-law fit) is entirely about the
lightweight downstream head, not the backbone. Backbone-level architectural
inefficiency only becomes visible once a backbone is put through backpropagation —
the LoRA condition in this comparison — because now the backbone's own native call
size, tensor dimensions, and single-vs-multi-encoder structure determine how much of
the GPU's actual tensor-core throughput is reachable.

### SleepFM — no measured data; structurally the best-positioned of the three, unverified

SleepFM is used **frozen throughout this entire project** — it was never LoRA
fine-tuned here, so there is no measured TFLOP/s number for it to compare against
OSF's ~2–19 or PhysioOmni's ~0.69. Anything said about its efficiency is a structural
inference from its published design, not a measurement, and should be labeled as such
if it goes in the paper:

- One joint tensor across all 4 modality groups per call (same efficient
  single-backbone pattern as OSF, not PhysioOmni's 4-way split).
- 128-dim per-modality embedding — a power of 2, tensor-core-friendly, unlike
  PhysioOmni's d=100.
- A 300-second native chunk — 10× larger than OSF's 30-second unit, meaning each
  backbone call does 10× more work before returning, which on the mechanism found for
  OSF (short calls are overhead-bound, longer ones are compute-bound) would suggest
  SleepFM reaches good GPU utilization at a much shorter position in a context sweep
  than OSF does, if it were ever fine-tuned the same way. This is a plausible
  prediction from the same mechanism that explained OSF's numbers, not a result.
- One real, if minor, design quirk worth a sentence for completeness: incomplete
  trailing 300-second chunks are dropped entirely (`extract_sleepfm_embeddings.py`),
  so a small amount of every recording's tail is simply discarded — a data-completeness
  cost, not a GPU-throughput one.

### OSF — measured, and diagnosed: a granularity/batching problem, not an architecture problem

- Single joint tensor across its 12 channels per call, well-shaped hidden dim
  (d=768, a multiple of 8, plenty large) — neither of these is the source of the
  measured inefficiency.
- The actual issue is **granularity**: one native call spans only 30 seconds
  (~46.8 GFLOP/epoch, a hand estimate), small enough that fixed per-call overhead
  (kernel launch, the Python-side chunking loop) dominates wall time unless many
  epochs are batched into one call. Measured: ~2.0 TFLOP/s at 30s (1 epoch/window)
  vs. ~19.2 TFLOP/s at 80m (160 epochs/window) — GPU utilization jumps ~6.4× just from
  having enough raw epochs available to batch together, no architecture change
  involved.
- **This is a real fault in the sense that the model provides no automatic path to
  good utilization** — the default batching granularity (`chunk_batch_size=16`) left
  a measured 3.28× of throughput on the table until a human noticed and raised it to
  64. Nothing in OSF's own reference code surfaces this as a tunable a user should
  check.
- No cross-epoch attention anywhere in the backbone means all temporal aggregation
  over a long context is 100% external (our own sequence head) — the backbone itself
  never does anything smarter with more context than "run once per epoch, N times."
  Every doubling of context is a literal doubling of backbone calls, with no internal
  amortization the architecture provides on its own.

### PhysioOmni — measured, and diagnosed: closer to a structural ceiling than a tuning problem

- **No joint tensor** — four fully separate per-modality encoders (EEG/EOG/ECG/EMG),
  each its own forward call, no shared computation and no cross-modal attention
  anywhere in the pretrained weights. This is the most fragmented of the three designs:
  even a perfectly-tuned batching scheme still pays for 4 separate kernel-launch groups
  per window instead of 1.
- **Hidden dims are small in absolute terms**: d=200 for EEG, d=100 for
  EOG/ECG/EMG. Regardless of divisibility, matmuls this small struggle to fill an
  H100's streaming multiprocessors or amortize kernel-launch latency — d=100 is also
  not a multiple of 8, a minor additional misalignment stacked on top of the more
  fundamental "just too small" problem.
- **Measured consequence**: PhysioOmni's LoRA stage ran at **~0.69 TFLOP/s, ~3.6% of a
  realistic fp32 ceiling** on the same class of GPU slice OSF was measured against —
  roughly an order of magnitude below OSF's own already-imperfect ~19 TFLOP/s at long
  context.
- **The critical difference from OSF: the batching trick that fixed OSF's problem did
  not fix PhysioOmni's.** A controlled `chunk_batch_size` 16-vs-64 A/B (run during
  PhysioOmni's own embedding extraction, on matched SHHS batches) found **no
  meaningful difference** — unlike OSF's confirmed 3.28×. That is real, if indirect,
  evidence that PhysioOmni's bottleneck is the matmul size itself, not call-launch
  overhead, and therefore not something batching alone can amortize away. This is a
  materially harder problem to engineer around than OSF's.
- A second, smaller compounding factor: missing-modality handling (a batch-level
  present-mask, per-modality conditional branches) adds control-flow overhead that a
  single-tensor model like OSF or SleepFM simply doesn't pay.
- Separately from architecture: a real 15-day stall came from an operational
  GPU-billing misconfiguration, not from anything above — see §3.2. Don't let that
  incident get folded into "the architecture is inefficient"; it's a distinct,
  non-architectural cause that happened to compound with the real architectural one.

### Mantis — measured (partial), and diagnosed: efficient from day one, not yet a completed picture

**Updated 2026-09-13, real measurements, superseding the "no data yet" placeholder
above.** Two of the two things flagged as unknowns when this section was a stub are
now partially answered:

- **Its architecture is the best-shaped of the three fine-tuned backbones, and this
  now shows up in real numbers, not just structural inference.** One joint tensor
  per channel call (`Conv1d(in_channels=1)`, applied identically to every channel —
  channel-independent, but each call is still a single, well-shaped op, not
  PhysioOmni's four-way split), `hidden_dim=256` (attention inner dim 1024) — a
  clean power of 2, unlike PhysioOmni's d=100. At 30s, Mantis's LoRA condition
  already reaches **~6.6 TFLOP/s, ~1.34% of a whole H100's TF32 peak** (§3.4) —
  roughly **10× PhysioOmni's own measured 30s-adjacent regime** (~0.69 TFLOP/s,
  ~3.6% of a smaller `3g.40gb` slice's peak; the two percentages are against
  different-sized GPU allocations, so compare the raw TFLOP/s, not the percentages,
  when reading these side by side) and in the same rough range as OSF's own
  30s number (~2.0 TFLOP/s on a smaller slice).
- **It was pretrained on "general time series" real data (Mantis-8M) for the
  Stage 1 sweep, not the synthetic-only checkpoint** — correcting this section's
  own earlier placeholder text, which assumed the opposite (synthetic pretraining)
  and speculated about hidden dims never being "scoped against real PSG signal
  characteristics." That framing was about the wrong checkpoint. The synthetic
  (`MantisPlus`) ablation remains unrun, deferred, not dropped (§1).

**What is genuinely still unknown, not yet measured**: the "granularity vs.
batching" question this section originally flagged is only half-answered. Mantis
was built with TF32 and a large `chunk_batch_size` (192) from day one (unlike OSF
and PhysioOmni, which discovered these only after weeks of unoptimized running),
and Pilot 3 confirmed no further sensitivity to `chunk_batch_size` at the
embedding-extraction stage — but that pilot measured **frozen-encoder extraction**,
not the LoRA (backprop) condition this section is actually about. The one real
LoRA-stage longer-context data point (`depression_extreme_binary`'s 10m run, §3.4)
shows a 6.6× cost increase over 30s, well below the ~20× naive raw-epoch-count
scaling — consistent with Mantis already being closer to compute-bound at 30s than
OSF was, but from a single data point on the smallest task in the comparison, not
yet confirmed on any of the four larger Tier-1 tasks. **The honest state of this
question is: better-shaped than PhysioOmni, plausibly better-behaved than OSF's
initial (pre-fix) numbers, but not yet proven at the scale (larger tasks, longer
contexts) that would settle it.**

### Honest opinion, one paragraph, for the paper if this ships

None of these backbones were designed with this project's specific fine-tuning
workload — many short raw epochs, batched into long windows — in mind, and all three
leave real throughput on the table if driven naively. But the failure modes are not
equivalent, and the paper should say so rather than treating "the LoRA sweep is
incomplete" as one uniform story. **OSF's inefficiency is a granularity/batching
problem**: real, and it cost real project time to discover, but it is fixable by
tuning, and once fixed reaches respectable GPU utilization at long context.
**PhysioOmni's inefficiency looks structural**: its four-encoder, small-hidden-dim
design has a utilization ceiling that the same batching fix does not move, which
points to a property of the released architecture rather than of how carefully it was
driven. **Mantis, so far, looks like neither problem** — its measured 30s TFLOP/s
already exceeds PhysioOmni's by roughly an order of magnitude, and its one real
longer-context measurement scaled better than OSF's own initial, unoptimized numbers
did — but "so far" is doing real work in that sentence: the sweep is far less
complete than either of the other two models' LoRA stages were at a comparable point
in their own timelines (§3.4), and this section's earlier optimism about OSF's own
fixability looked equally reasonable before PhysioOmni's structural ceiling was
found, so Mantis's apparent efficiency should be read as encouraging, not settled.
If the paper wants one sentence: OSF's GPU-cost problem was an engineering problem;
PhysioOmni's looks like an architecture problem; Mantis's, on the evidence so far,
looks like neither — but the evidence so far is thin.

---

## 5. What NOT to over-interpret

- Neither model's `mean_pool` head was ever run (Stage 1 or Stage 2), so H3-style
  "does the temporal-head advantage over MeanPool replicate under a different
  encoder" cannot be answered from this data yet.
- Val-split threshold-tuning was never run for either model (Stage 1 or 2) — no
  balanced-accuracy/threshold-based comparison should be built from this data without
  first running it, mirroring the standing "never report incomplete results as
  complete" rule already applied throughout the paper.
- The OSF/PhysioOmni comparisons use different SleepFM baselines (§1) — there is no
  valid "OSF vs. PhysioOmni" number in this document, only "OSF vs. its own SleepFM
  baseline" and "PhysioOmni vs. its own (different) SleepFM baseline." A reader should
  not walk away thinking OSF beats PhysioOmni head-to-head; that comparison was never
  run under matched conditions.
- **(Added 2026-09-13) Mantis's frozen-encoder wins (sex, age, OSA — §2.4) have no
  bootstrap confidence interval behind them yet** — every CI cell for Mantis in
  `phase0_mantis/analysis.csv` is `NaN`. Treat every Mantis point estimate,
  especially the sex 240m decline, as provisional until a CI pass exists.
- **(Added 2026-09-13) Mantis Stage 2 (LoRA) produces no valid frozen-vs-LoRA
  comparison at all yet** — not because of coverage alone, but because the
  subject-level, $K_{\max}$-aggregated inference script has never been run against
  any Mantis LoRA checkpoint (§3.4). The window-level numbers currently on disk
  should not be compared to any `mean_prob_auroc` figure elsewhere in this document.
- **(Added 2026-09-13) A three-way "SleepFM vs. OSF vs. PhysioOmni vs. Mantis"
  leaderboard is not supportable from this data**, for the same reason §1 already
  gives for OSF vs. PhysioOmni: Mantis and PhysioOmni share a SleepFM baseline
  (`phase0_v3`), but OSF does not (`phase0_v3_full`) — any apparent "Mantis beats
  OSF" or "OSF beats Mantis" comparison built from this document's tables would be
  comparing each model against a different reference point, not against each other.

---

## 6. Suggestions for the paper

### 6.1 Framing: robustness/generalization check, not a leaderboard

`npj_supplementary.tex` already has a section
(`sec:supp-sota`, "Scope of Comparisons with Prior Work") explaining why the paper
does **not** report head-to-head AUROC against externally-published numbers: cohort
overlap risk, inconsistent task definitions, lightweight-vs-full-fine-tune head
mismatch, and fundamentally different task families. **That reasoning does not apply
here** — OSF and PhysioOmni were run through the *identical* pipeline (same tasks,
same lightweight heads, same cohorts, same evaluation protocol) that SleepFM itself
uses, which is exactly the controlled setup that section says is missing from prior
comparisons. This is worth stating explicitly if this material is added: it is the
controlled comparison the existing supplementary section argues for, not another
instance of what it argues against.

That said, avoid overclaiming a full leaderboard, precisely because of what's in §2-3:
uneven LoRA coverage, two different SleepFM baselines, and OSF's contamination issue.
**Recommended framing**: "does the paper's central finding — that context-length value
is highly task-specific — replicate under a different frozen encoder, and does encoder
choice change *which* tasks are context-sensitive?" This is answerable from the data
in hand, is a real contribution, and doesn't require the LoRA sweeps to be complete to
say something true.

**Suggested subsection title** (avoid "benchmark," "comparison," or "leaderboard"
language that implies more completeness than exists): something like *"Robustness to
encoder choice"* or *"Generalization across frozen encoders."*

### 6.2 Where it goes

- **Main text**: a short new Results subsection, placed after
  §"Full-channel configuration helps cardiorespiratory task" /
  §"Modality group ablation" (the paper's existing secondary-analyses block) and before
  the Discussion. Keep it brief — a few sentences plus one compact figure or table
  showing the sex/age/BMI-vs-sleep-efficiency/apnea split for OSF, and the
  frozen-underperformance finding for PhysioOmni — mirroring how the paper already
  keeps secondary analyses (channel comparison, modality ablation) compact and
  supplementary-heavy.
- **Discussion**: the existing Limitations paragraph (`npj_main.tex`, in the
  "Several factors constrain..." paragraph) already names this exact gap: *"testing
  this directly against alternative frozen encoders (OSF, SleepMaMi, SleepFounder) ...
  is a natural next step this design does not itself resolve."* Once this lands,
  that sentence should be rewritten to say OSF and PhysioOmni **were** tested (with a
  forward-reference to the new subsection/supplementary section), while SleepMaMi and
  SleepFounder remain cited-only (no public checkpoint, per
  `docs/TSFM_BASELINE_CANDIDATES.md` §4).
- **Supplementary**: full per-context tables (§2 above), the architecture/input-handling
  table (§1), the contamination quantification, and the computational-cost narrative
  (§3) belong in supplementary, extending the existing `sec:supp-sota` section rather
  than creating a disconnected new one — that section is already the paper's designated
  place for "how do we relate to other models" discussion.

### 6.3 Which caveat must appear next to which number

| Number | Required caveat |
|---|---|
| Any OSF number that includes SHHS in a pooled/headline figure | Contamination: 87.7% of SHHS test subjects were in OSF's own pretraining set; report SHHS separately, never blended in |
| Any OSF-vs-SleepFM comparison, any cohort | Split mismatch: SleepFM and OSF Stage 1 use subtly different train/val/test splits (§2.3) |
| Any PhysioOmni number, anywhere it appears | Apnea excluded (no respiratory pathway); arXiv-only, never peer-reviewed; license split between code repo (none) and HF weights (CC-BY-4.0) |
| Any PhysioOmni-LoRA number | Only 9/48 cells exist; `bmi_binary`/`age_class` have zero LoRA cells — state coverage explicitly, do not imply a completed sweep |
| Any OSF-LoRA number at 240m | Does not exist — do not show an empty/interpolated cell |
| Any cross-model "OSF vs. PhysioOmni" framing | Invalid — they were compared against different SleepFM baselines (full- vs. reduced-channel), never against each other under matched conditions |
| Any Mantis Stage 1 number | No bootstrap CI computed yet (all `NaN` in the collected CSV); checkpoint is `Mantis-8M` (real-time-series pretraining), not the synthetic-only `MantisPlus` ablation, which has not been run |
| Any Mantis sex_binary number at 240m specifically | AUROC declines from its 120m peak (0.935→0.923, Transformer) rather than continuing to rise like SleepFM's — flagged as unresolved, not statistical noise vs. real effect, pending a CI pass (§2.4.1) |
| Any Mantis Stage 2 (LoRA) number | **Do not report at all as a comparison to anything else in this document** — subject-level $K_{\max}$-aggregated inference has never been run for any Mantis LoRA checkpoint; the on-disk numbers are a different, window-level metric (§3.4) |
| Any cross-model "Mantis vs. OSF" or "Mantis vs. PhysioOmni" framing | Mantis and PhysioOmni share a SleepFM baseline (`phase0_v3`); OSF does not (`phase0_v3_full`) — same invalidity as the OSF-vs-PhysioOmni row above |

### 6.4 Where Mantis slots in later

Every table in §1 and the stopping-criteria table in §3.3 has a placeholder
column/row ready for Mantis. Two things worth deciding now so Mantis drops in cleanly:

1. **Mantis is a different flavor of baseline than OSF/PhysioOmni**, and should be
   caveated differently, not folded into the same "sleep/physiological FM" bucket:
   it is classification-native and lightweight (8M params, real public checkpoint),
   but pretrained **exclusively on synthetic data**, carrying no physiological prior at
   all. Per `docs/TSFM_BASELINE_CANDIDATES.md` §4, it is "closer to a strong generic
   classifier architecture than a foundation model in the sense the supervisor's
   question is really asking about." Frame its eventual inclusion as answering a
   different question — "does *any* physiological pretraining matter, or does a
   generic strong classifier do just as well?" — not as a third data point on the same
   "which sleep FM is best" axis as OSF/PhysioOmni.
2. Once Mantis's results exist, redo the §0 executive summary and §5.1 framing
   decision together, rather than just appending a column — a 3-model comparison may
   change which framing (robustness check vs. something stronger) is actually
   supportable.

> **⚠ Verification note, added 2026-09-13.** Point 1 above describes Mantis as
> "pretrained exclusively on synthetic data." **This is not the checkpoint the
> actual Stage 1 sweep used.** The production results in §2.4 use `Mantis-8M`
> (real-time-series pretraining); the synthetic-only `MantisPlus` ablation this
> paragraph describes has not been run (§1). Left as originally written per this
> document's own "don't fix in place" convention — see §6.5 for the corrected,
> results-informed version of this framing decision.

### 6.5 Mantis results now exist — updated framing (added 2026-09-13)

**Point 2 above asked for this once results existed — here it is.**

**Does Mantis change the §0/§6.1 "generalization/robustness check, not a
leaderboard" recommendation, or reinforce it? It reinforces it, more strongly than
OSF or PhysioOmni did individually.** §2.4's central finding — the qualitative
context-sensitivity split (sleep efficiency and apnea most sensitive; sex and age
moderately so; BMI least) replicates almost exactly under an encoder with **zero
physiological pretraining at all** — is a stronger generalization result than
either OSF's (a sleep-specific encoder, contamination-confounded on SHHS) or
PhysioOmni's (a physiological-but-non-sleep encoder that underperformed SleepFM
throughout, so its agreement on task ranking was less informative — a weaker model
that happens to rank tasks the same way is a less surprising confirmation than a
*stronger*, unrelated-domain model that does). If the paper adopts one framing
sentence for the three-model result together, it should lead with this: *the
task-specific value of context length is not an artifact of SleepFM's specific
pretraining — it replicates under a sleep-specific encoder, a general physiological
encoder, and a non-physiological, general-time-series encoder alike.*

**This does not mean Mantis is "just another confirmation" to fold in quietly.**
Two things distinguish it from OSF/PhysioOmni's own additions and should not be
smoothed over:

1. **Mantis's frozen encoder beats SleepFM on multiple tasks (sex, age, OSA) —
   OSF only won this cleanly on the static/structural tasks (sex, age, BMI) with a
   contamination caveat attached to its strongest cohort, and PhysioOmni never won
   anywhere.** A non-physiological encoder outright beating a 585,000-hour
   PSG-specific one, with a completely clean contamination story (§1: "provably
   zero" overlap, no exact-ID check even needed), is the single most attention-worthy
   number this document now contains, not a footnote.
2. **This result directly falsified a real, pre-registered prediction**
   (`docs/TSFM_THIRD_MODEL_DECISION.md`'s "expect a weak frozen result"). If this
   material ships, that pre-registration is worth keeping in the write-up
   explicitly — it is what turns "Mantis did surprisingly well" from a
   post-hoc-sounding claim into a genuinely falsifiable one that was, in fact,
   falsified.

**Placement, extending §6.2's plan rather than replacing it**: the "why these three
baselines, why now" framing already in this task's brief — SleepFM was the only
model available when the paper's core methodology was designed; OSF, PhysioOmni,
and Mantis were added later, close to submission, specifically because the
supervisor asked why no comparison against recent TSFMs existed — is honest and
should appear plainly in whichever paper section introduces this material, not
softened or over-apologized for. Concretely:

- **Main text**: extend the same short Results subsection §6.2 proposes for
  OSF/PhysioOmni (after "Modality group ablation," before Discussion) with Mantis's
  headline finding — the context-sensitivity replication, stated as this
  subsection's lead sentence per the framing above, with the sex/age/OSA wins as a
  second, clearly-flagged sentence. **Stage 2 (LoRA) is not ready for the main text
  in any form** — §3.4's metric-comparability problem means there is nothing to
  show yet, not just an incomplete table.
- **Discussion**: the same Limitations-paragraph rewrite §6.2 proposes for
  OSF/PhysioOmni should now say all three were tested, with Mantis explicitly
  named as the one carrying zero physiological pretraining.
- **Supplementary**: §1's completed architecture table, §2.4's full per-task tables,
  and §3.4's honest LoRA-incompleteness accounting (including the per-epoch-cost
  finding, which is a genuine methodological contribution about the cost of
  fine-tuning raw-signal TSFMs at PSG scale, worth keeping even if the LoRA
  results themselves aren't ready) all belong in the same extended `sec:supp-sota`
  location §6.2 already designates.
- **Do not add Mantis's Stage 2 to any table yet** — not because of low coverage
  alone (OSF's own LoRA table has real, accepted gaps at 240m), but because §3.4
  found there is currently no metric on disk for Mantis Stage 2 that is comparable
  to anything else in this document. Coverage and comparability are separate
  problems here, and only coverage would resolve on its own with more compute time.

---

## 7. Update 2026-09-10 — `depression_extreme_binary` and `osa_binary_apples_postqc` (Phase 1 only)

**Everything above this line is unchanged from the original document.** This section
is purely additive. Both tasks are **frozen-encoder (Phase 1) only** — no LoRA was run
for either, on either model — so every number below is a frozen-vs-frozen comparison;
treat any `—` accordingly, same convention as §2. `osa_binary_apples_postqc` was added
to OSF only, not PhysioOmni (no respiratory pathway — see §1's existing row on this).
Numbers pulled the same way as §2 (`mean_prob_auroc`, test, `k="all"`, Transformer),
re-verified directly from the collected CSVs, not estimated.

### 7.1 The numbers

**depression_extreme_binary** (APPLES + STAGES — both confirmed contamination-clean,
see §1's pretraining-overlap row; no SHHS caveat needed for this task, unlike sex/age/BMI/apnea)

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (full-ch.) | 0.738 | 0.742 | 0.752 | 0.756 | 0.740 | 0.741 |
| SleepFM (reduced-ch.) | 0.756 | 0.739 | 0.750 | 0.749 | 0.754 | 0.746 |
| OSF-frozen | 0.762 | 0.770 | 0.777 | 0.781 | 0.776 | 0.765 |
| PhysioOmni-frozen | 0.718 | 0.729 | 0.725 | 0.722 | 0.702 | 0.726 |

**osa_binary_apples_postqc** (APPLES only, post-QC filtered — confirmed clean)

| Model | 30s | 10m | 40m | 80m | 120m | 240m |
|---|---|---|---|---|---|---|
| SleepFM (full-ch.) | 0.755 | 0.762 | 0.791 | 0.814 | 0.814 | 0.818 |
| OSF-frozen | 0.867 | 0.868 | 0.906 | 0.877 | 0.914 | 0.887 |

### 7.2 The important finding: a real tension with the existing `apnea_binary` result

**`osa_binary_apples_postqc` and `apnea_binary` use essentially the same clinical
threshold** — confirmed directly from the registries, not assumed: `apnea_binary`'s
notes say *"AHI >= 15 = moderate/severe OSA. Standard clinical threshold"*;
`osa_binary_apples_postqc`'s say *"Non-rand+Mild->0, Moderate+Severe->1"* — the same
moderate/severe cutoff, just APPLES-only with an added post-QC subject filter. On
`apnea_binary` (pooled across apples+shhs+mros+stages, §2.1), **OSF loses to SleepFM
specifically on the APPLES cohort**. On `osa_binary_apples_postqc` — same threshold,
same cohort, but trained as its own single-cohort model — **OSF beats SleepFM by
+6 to +11.5pp, the largest margin of any task/model pair in this entire document**,
bigger than any of the "real, credible" sex/age/BMI wins in §2.1.

**A real, defensible explanation exists — this is not an inconsistent or broken
result**: verified directly (`training.csv`), `osa_binary_apples_postqc` trains on
3,795 subjects (APPLES only) vs. `apnea_binary`'s pooled 48,380 (~12.75× larger,
4 heterogeneous cohorts/sites/devices). Single-cohort, homogeneous-site training
producing a different performance profile than pooled multi-site training is a
well-understood, unsurprising ML phenomenon, not evidence of a flawed comparison.
Worth noting too: SleepFM's **own** `osa_binary_apples_postqc` numbers (0.755-0.818)
are themselves lower than SleepFM's own `apnea_binary` numbers at matched context
(0.800-0.901) — so the smaller/harder single-cohort task is harder for *both* models,
just disproportionately less so for OSF.

**Why this is still a real risk if it ships uncaveated**: a reviewer who reads both
numbers side by side — same threshold, same cohort, opposite directional finding for
the same encoder — has an obvious, fair question, and "different training-set size"
is a real answer but requires the paper to actually make that argument explicitly, not
just present both numbers and hope it isn't noticed. There is currently **no
controlled test of the training-set-size hypothesis** — e.g. retraining `apnea_binary`
on APPLES-only data with the same post-QC filter, to see whether *that* also flips
positive for OSF, which would directly confirm (or refute) the explanation above
rather than leaving it as a plausible-but-unverified story.

**One more thing worth naming plainly, not glossed over**: subject-ID-level
contamination is confirmed clean for APPLES (§1), but that only rules out *direct*
subject overlap — it does not rule out OSF's pretraining corpus (heavily SHHS-weighted)
sharing *device/protocol/site-level* characteristics with APPLES that a subject-ID
match can't detect. This is speculative, not evidence of anything — flagged here as an
open uncertainty, not a finding, precisely because it's the kind of thing worth having
an answer ready for if asked, not because there's reason to believe it's true.

### 7.3 depression_extreme_binary: complicates the picture less, but not a clean fit either

Unlike `osa_binary_apples_postqc`, this task **does not contradict anything already in
this document** — OSF wins consistently (+2.4 to +3.6pp at every context, both
models notably flat across context length, unlike sex/age/BMI's clear monotonic
climb) and PhysioOmni underperforms SleepFM consistently (-1.0 to -5.2pp), which
*replicates and strengthens* both of §2's existing headline patterns rather than
complicating them. It is also the cleanest task in the whole document from a
contamination standpoint — both its cohorts (APPLES, STAGES) are confirmed clean, so
unlike sex/age/BMI/apnea it needs no SHHS caveat at all.

That said, two things are worth naming honestly before treating it as a fourth
"OSF wins" data point on the same footing as sex/age/BMI:

- **It doesn't actually belong in the "static/structural" bucket §0/§2.1 built for
  sex/age/BMI.** Those are simple, well-defined, objectively-measured biological/
  demographic traits with established physiological correlates in sleep macrostructure.
  Depression is a psychological/clinical construct — noisier, more subjectively
  defined (the task's own "extreme-group design," dropping the middle group entirely,
  is itself a tell that the raw signal is weak enough to need this to get a usable
  classifier at all), and its link to PSG-observable sleep macrostructure is far less
  established in the literature than sex/age/BMI's. It happens to *pattern* like the
  "OSF wins" bucket (flat-with-context, OSF ahead throughout), but that's a
  resemblance in results, not a claim that the same underlying mechanism (encoder
  captures a stable structural signal better) is what's actually happening.
- **Absolute performance is meaningfully lower than the tasks it would sit next to.**
  0.70-0.78 AUROC, vs. 0.83-0.96 for sex/age/BMI — a real, weaker signal. Reporting it
  in the same table/figure as those without flagging the gap in absolute performance
  would read as implying comparable task difficulty/validity, which isn't accurate.

### 7.4 Recommendation

**Include `depression_extreme_binary`, with the two caveats in §7.3 stated
explicitly** (different task category than sex/age/BMI; weaker absolute AUROC). It's a
clean, contamination-free, internally-consistent result that strengthens both of the
document's existing headline findings rather than complicating either.

**Hold `osa_binary_apples_postqc` back from the paper for now**, or at minimum do not
place it anywhere near the existing `apnea_binary` discussion without directly
addressing the tension in §7.2. My honest opinion: the training-set-size explanation
is real and probably correct, but it is currently a plausible story, not a verified
one — and the size of the discrepancy (a task that's the single biggest OSF win in the
whole document, on the exact clinical question where the exact same encoder was
previously reported losing on the exact same cohort) is large enough that I would not
ship it without either (a) the controlled APPLES-only-`apnea_binary` retrain that
would actually test the explanation, or (b) an explicit, well-argued paragraph in the
paper itself making the training-composition case, written knowing a reviewer will
likely ask about it directly. Either is real, scoped work — not a small edit — so this
is a genuine decision point, not something to default into either direction.
