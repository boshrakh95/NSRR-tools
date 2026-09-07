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

## 1. Architecture & Input-Handling Comparison

Adapted and extended from the three-way table already drafted in
`docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md` §20 (code-verified there; reused here,
not re-derived) plus checkpoint/license facts from `docs/TSFM_BASELINE_CANDIDATES.md`
§2.1-2.2. A Mantis row is left as a placeholder.

| | SleepFM | OSF | PhysioOmni | Mantis *(placeholder)* |
|---|---|---|---|---|
| **Role in study** | This paper's primary encoder | Sleep-PSG-specific FM baseline | General physiological (non-sleep-specific) FM baseline | General-purpose classification-native TSFM baseline |
| **Checkpoint / license** | (used throughout paper; not itself under license review here) | HF `yang-ai-lab/OSF-Base`; **MIT confirmed** | HF `Weibang/PhysioOmni`; **code repo has no LICENSE file; HF weights repo declares CC-BY-4.0** — state both facts if shipped | `paris-noah/Mantis-8M`; pretrained **exclusively on synthetic data**, not physiological signal — a different comparison in kind (see §5.4) |
| **Native per-call window** | Exactly 300s (5 min); incomplete trailing chunks dropped | Exactly 30s; no cross-epoch attention anywhere in the model | No fixed requirement — variable-length token sequence; real per-modality ceiling: EEG 512s / EOG 256s / ECG 102s / EMG 102s at its own reference resample rates | *TBD once implemented* |
| **What our pipeline actually uses per call** | 300s (5s sub-patches within) | 30s (architecturally forced) | 30s (a **deliberate choice**, not architecturally forced — see below) | *TBD* |
| **Channels used** | 4 modality groups (BAS/RESP/EKG/EMG), reduced (7-8ch) or full (≤23ch) config | 12-channel fixed input incl. snore + full thoracic/abdominal/airflow — **requires our full-channel HDF5s**, not the reduced/fast config | EEG/EOG/ECG/EMG only — **no respiratory pathway at all** (confirmed at 4 independent code locations); fast-channel HDF5s suffice | *TBD* |
| **Apnea comparable?** | Yes (reference) | Yes | **No — excluded**, stated reason (no RESP pathway; adding one would require new-modality pretraining, not fine-tuning) | *TBD* |
| **Output / embedding shape (this project's convention)** | `[T,4,128]`, flat dim 512 | `[T,2,768]` (CLS ⊕ mean-pooled patches), flat dim 1536 | `[T,500]` (concatenated per-modality CLS, 2D, no sub-token axis) | *TBD* |
| **Sequence-head `input_dim`** | 512 | 1536 | 500 | *TBD* |
| **Encoder parameter count** | ~4.4M (Supplementary §"Model Parameter Counts") | 85,325,568 (strict-load-verified) | 13,871,304 total across 4 independent encoders (EEG 7.84M; EOG/ECG/EMG ~2.01M each) | 8M |
| **Fusion in released weights** | Contrastive alignment across 4 modality encoders; each still outputs its own vector | Single unified ViT — full fusion | **None** — 4 fully independent tokenizers/encoders, no cross-modal attention in the checkpoint; any fusion is downstream-constructed by us | *TBD* |
| **Usage mode (Plan A/B/C)** | B (only option) | B (only option — architecture leaves no other choice) | **B (chosen)** — architecture could theoretically support up to ~1.7–8.5 min native, still short of every sweep point past 30s | *TBD* |
| **SleepFM baseline used for comparison** | — | `phase0_v3_full` (full-channel) | `phase0_v3` (reduced/fast-channel, paper-primary) | *TBD* |
| **Pretraining-cohort overlap with our 4 test cohorts** | N/A | **SHHS: severe (confirmed, exact-ID match, 87.7% of our SHHS test subjects were in OSF's own pretrain train/valid split). STAGES: confirmed clean. MrOS: confirmed clean (downstream/eval-only in OSF, not pretraining). APPLES: confirmed clean.** | None of our 4 cohorts are in PhysioOmni's pretraining corpus (TUH/CAP/Sleep-EDF/DEAP — none are NSRR) | *TBD* |
| **Peer review status** | Published (ICML 2024 + Nat. Med. extension) | Published (ICML 2026) | **arXiv only, v3 dated 2026-03 — never peer-reviewed** | *TBD* |
| **A caveat worth stating if PhysioOmni is cited on its own merits** | — | — | On its own best-fit downstream task (HMC sleep staging) in its own paper, PhysioOmni's reported number (0.7377 balanced accuracy) **does not beat its own paper's non-foundation-model baseline** (FeatFusion, 0.7478) | — |

**Why OSF and PhysioOmni are each compared against a different SleepFM variant, not the
same one**: OSF needs full-channel signal (thoracic/abdominal/airflow/snore), which
only exists in the full-channel (`phase0_v3_full`) HDF5 tree; PhysioOmni needs only
EEG/EOG/ECG/EMG, which the paper-primary reduced/fast-channel tree
(`phase0_v3`) already carries. **Do not cross-compare OSF's numbers against
`phase0_v3` or PhysioOmni's against `phase0_v3_full`** — those are not the SleepFM
baselines each was actually run against, and the reduced- vs. full-channel gap itself
is non-trivial (Section 5.7 of `npj_main.tex`, "Full-channel configuration helps
cardiorespiratory task": +0.03–0.05 AUROC for apnea/BMI, smaller for sex).

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

### Mantis — no data yet, flagged for when it lands

Not implemented in this repo yet. Two things worth checking once it is, for
consistency with the above: (1) its native context length is short (built and
pretrained as a lightweight, ~8M-parameter classification-native model), so it will
likely face the same "granularity vs. batching" question as OSF once fine-tuned —
worth measuring rather than assuming either way; (2) since it is pretrained on
synthetic, not physiological, data, its useful hidden dims/tensor shapes weren't
scoped against real PSG signal characteristics at all — an unknown, not a predicted
problem, until measured.

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
driven. If the paper wants one sentence: OSF's GPU-cost problem is an engineering
problem; PhysioOmni's looks like an architecture problem.

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
