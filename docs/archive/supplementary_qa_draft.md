# Supplementary Q&A — Design Decisions Transparency Document
## IEEE TBME Submission: Context-Length Dependence in Overnight PSG-Based Clinical Prediction

*Drafted from codebase and documentation analysis. Sections flagged `[NEEDS AUTHOR INPUT]` require
confirmation or narrative content that cannot be recovered from the code alone.*

---

## Table of Contents

1. [Window / Context Definition](#1-window--context-definition)
2. [Training Window Sampling](#2-training-window-sampling)
3. [Inference Aggregation (K)](#3-inference-aggregation-k)
4. [Architecture Decisions](#4-architecture-decisions)
5. [Training Procedure](#5-training-procedure)
6. [Dataset and Preprocessing](#6-dataset-and-preprocessing)
7. [Train / Val / Test Splits](#7-train--val--test-splits)
8. [Task Definition and Label Choices](#8-task-definition-and-label-choices)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Comparison with State of the Art](#10-comparison-with-state-of-the-art)
11. [Versioning — v3 vs v2 vs v1](#11-versioning--v3-vs-v2-vs-v1)
12. [Compute and FLOPs Analysis](#12-compute-and-flops-analysis)
13. [Things That Did Not Work or Were Surprising](#13-things-that-did-not-work-or-were-surprising)

---

## 1. Window / Context Definition

**Q1.1 What is a "context window" (L) in this paper?**

A context window of length L is a contiguous sequence of N consecutive 5-second SleepFM patch
embeddings extracted from a subject's overnight PSG recording. Each patch is produced by the frozen
SleepFM SetTransformer encoder applied to 640 raw signal samples (5 seconds at 128 Hz) across all
four modality groups (BAS, RESP, EKG, EMG), yielding a 512-dimensional embedding vector
(4 modalities × 128 dimensions). A context window of length L therefore has shape `[N, 512]`
where N = L / 5 seconds.

**Q1.2 What specific context lengths were evaluated, and why those values?**

Six context lengths were evaluated: **30 s, 10 min, 40 min, 80 min, 120 min, and 240 min**
(corresponding to N = 6, 120, 480, 960, 1440, and 2880 patches respectively).

The rationale is threefold. First, 30 seconds corresponds to the standard sleep-staging epoch — the
smallest clinically meaningful unit of PSG. Second, 240 minutes corresponds to the typical full-night
PSG recording duration — the longest context that can be consistently provided across all subjects in
the cohort. Third, the intermediate values (10, 40, 80, 120 min) are roughly log-spaced across this
range, covering approximately three decades (0.5 to 240 minutes). This spacing provides adequate
density on the saturation curve while keeping total training compute tractable.

One acknowledged limitation: the gap between 30 s and 10 min (a 20× jump) is the largest on the
scale, yet it is likely the regime of steepest AUROC ascent for most tasks. Adding a 2 m or 5 m
context was considered but deferred to resource constraints.

**Q1.3 How are windows extracted from an overnight recording?**

There are two extraction modes, used for different stages of the pipeline:

- **Training (overlapping pool):** Any integer window start position in `[0, T − N]` is valid,
  where T is the total number of 5-second patches in the recording. Windows may overlap
  arbitrarily. Specifically, for context 240 min (N = 2880) in a typical 8-hour night (T ≈ 5760
  patches), there are 2881 valid start positions. K = 5 windows are sampled uniformly without
  replacement from these positions.

- **Validation during training (overlapping pool, deterministic):** K = 5 positions are drawn by
  evenly spacing across `[0, T − N]` via `np.linspace`, giving a deterministic and reproducible
  early-stopping signal.

- **Inference (non-overlapping, stride-N):** Positions `0, N, 2N, ...,` up to `floor(T/N) × N`.
  This gives `floor(T/N)` non-redundant, non-overlapping windows covering the full recording
  systematically.

**Q1.4 What context mode is used for sleep staging (seq2seq)?**

Sleep staging uses a **centered** context window: the N patches are arranged symmetrically around
the anchor epoch (the epoch whose stage label is predicted): `floor((N−6)/2)` past patches, 6
anchor-epoch patches, and `floor((N−6)/2)` future patches. At 30 s (N = 6), the window reduces
to the anchor epoch alone (degenerate: centered equals causal). Centered context is preferred over
causal (past-only) context because it enables bidirectional integration of surrounding sleep
structure, which is standard in the sleep-staging literature.

**Q1.5 What happens at the recording boundaries?**

For seq2label tasks, subjects whose total recording length T < 2880 patches (< 240 minutes) are
excluded entirely from all context lengths — not only from 240 m. This "cohort consistency filter"
(`min_recording_patches: 2880` in the config) ensures that the subject pool is identical across the
entire context-length sweep, preventing a silent cohort shift that would otherwise confound
comparisons. Subjects shorter than their context window would receive one zero-padded window; the
filter avoids this entirely. 20 subjects (≤ 0.21% for Tier 1 tasks) were removed; these had
recording lengths from 5 to 230 minutes and appear to be truncated acquisitions (full list in
`docs/excluded_subjects_T_lt_2880.csv`).

For sleep staging (seq2seq), only anchor epochs where the full centered window fits inside the
recording are included (`complete_only` padding policy). No zero-padding is introduced for sleep
staging either.

---

## 2. Training Window Sampling

**Q2.1 How many windows are used per subject per training epoch?**

K\_train = **5 windows per subject per epoch**, fixed at all context lengths. The code parameter is
`dataset.windows_per_subject: 5` in `configs/phase0_v3_config.yaml`. The actual number of windows
included in the training index is `min(5, n_valid)` where `n_valid = T − N + 1`. After the cohort
consistency filter (T ≥ 2880 at all contexts), K\_train = 5 is always achieved.

**Q2.2 Why K = 5 specifically, and not K = all or a token-budget schedule?**

The choice of K = 5 is motivated by a specific fairness criterion: **equal gradient updates per
subject per epoch across all context lengths**. This is the correct criterion for isolating the
effect of context length L, because gradient-update asymmetry would introduce a training-compute
confound.

Two alternative strategies were explicitly considered and rejected:

| Strategy | Gradient updates/subject/epoch | Verdict |
|---|---|---|
| **K = 5 (chosen)** | Identical at all L | ✅ Correct criterion for context-length comparison |
| **K = all** | ~960 at 30 s vs ~2 at 240 m (480× asymmetry) | ✗ Measures training iterations, not context quality |
| **Token-budget (K × L = 80 min)** | ~160× more at 30 s than at 120 m | ✗ Asymmetric effective learning rate |

Importantly, the K-budget and fixed-K strategies converge at long contexts (L ≥ 80 min, where both
yield K ≈ 1). Any difference in the saturation curve between the two strategies would only appear
at the short-context end (30 s, 10 min), and a one-task sensitivity ablation (`sex_binary_lstm`
with `windows_strategy: "token_budget"`, run tag `"kbudget"`) is planned to verify that the
saturation curve shape is unchanged — closing the reviewer concern definitively.

**Q2.3 Are training windows overlapping?**

Yes. Since v3, training windows are drawn from the **overlapping pool** (any integer start in
`[0, T − N]`). This was a deliberate fix from v2: the old v2 code sampled only from non-overlapping
N-aligned positions (`0, N, 2N, ...`), which meant at 240 m context only 1–2 positions existed per
typical night — K = 5 was silently degraded to K = 1 or K = 2. The fix ensures that K = 5 is
always achievable. Overlapping windows are not identical training examples: for an 8-hour night with
240 m context, the expected spacing between the 5 sampled start positions is ~48 minutes.

**Q2.4 How is the starting position of a training window sampled?**

Uniformly at random without replacement from the set `{0, 1, 2, ..., T − N}`. The result is
sorted ascending to preserve temporal order in the mini-batch. The random seed varies per epoch
(epoch index is used as the RNG seed component), so different windows are seen at different epochs.
Across a full training run (up to 40 epochs × 5 windows = 200 window samples per subject), the
model sees a representative sample of the window space for all context lengths.

---

## 3. Inference Aggregation (K)

**Q3.1 What is K in the inference context?**

K is the number of context windows drawn from a subject's recording at test time, whose
probabilities are averaged (mean-probability aggregation) to produce a single subject-level
prediction. It is a post-hoc analysis parameter: after inference saves per-window probabilities for
all non-overlapping windows, any value of K from 1 to K\_max can be evaluated without touching the
GPU.

**Q3.2 How are the K windows positioned during inference?**

During inference, `infer_subject_windows.py` sets `windows_per_subject = 99,999`, which routes to
the **non-overlapping stride-N branch**: window starts at `0, N, 2N, ..., floor(T/N) × N`. This
gives `floor(T/N)` windows per subject — the complete, non-redundant coverage of the full overnight
recording. When a post-hoc K sweep samples K < K\_max windows, it draws them from this precomputed
set of `floor(T/N)` positions.

**Q3.3 Why mean-probability aggregation and not majority vote?**

Mean-probability aggregation (soft voting) preserves calibration information from each window's
output probabilities, whereas majority vote (hard voting at t = 0.5) discards the magnitude of
model confidence. At small K (K = 1, 5), soft voting consistently outperforms majority vote, with
the gap narrowing as K increases. The analysis pipeline computes both; majority vote is reported in
supplementary for completeness but is not the primary aggregation method. This choice is consistent
with standard practice in multiple-instance learning.

**Q3.4 What is K\_max, and how is it determined?**

K\_max is the total number of non-overlapping stride-N windows that fit in the recording:
`K_max = floor(T / N)`. For a typical 8-hour night (T ≈ 5760 patches):

| Context L | N patches | K\_max |
|---|---|---|
| 30 s | 6 | 960 |
| 10 min | 120 | 48 |
| 40 min | 480 | 12 |
| 80 min | 960 | 6 |
| 120 min | 1440 | 4 |
| 240 min | 2880 | 2 |

In the sparse K sweep used for paper tables, K ∈ {1, 5, 10, 20, 50, all}; in the dense K sweep for
iso-compute analysis, K sweeps approximately 25 values per context for the heatmap plots. "K = all"
uses all `floor(T/N)` windows for each subject, giving the full-night aggregation ceiling.

**Q3.5 What is the iso-compute analysis and what does it add?**

The iso-compute analysis places each (L, K) configuration on a 2D grid where one axis is context
length (train-time) and the other is aggregation count (inference-time). On this grid, an iso-compute
contour line connects all (L, K) pairs with the same total per-subject signal budget at inference:
K × L\_min = constant. This answers Hypothesis H2: for a fixed inference budget, is it better to
use one long-context window (large L, K = 1) or many short-context windows (small L, large K)? If
AUROC is higher along the large-L end of an iso-compute contour, longer-context training captures
something that aggregation alone cannot recover.

---

## 4. Architecture Decisions

**Q4.1 What is SleepFM, and what does the frozen encoder do?**

SleepFM is a pre-trained sleep foundation model whose architecture is a SetTransformer with
contrastive self-supervised pre-training using a leave-one-modality-out strategy among four modality
groups. The encoder takes raw PSG signals chunked into 5-second (640-sample at 128 Hz) patches and
produces a 128-dimensional embedding per patch per modality group. With four groups (BAS: EEG+EOG,
RESP: respiratory, EKG: cardiac, EMG: muscle), the per-patch output is a `[4, 128]` tensor,
concatenated to `[512]` before downstream use. The encoder is kept **frozen throughout all
experiments**: no gradients flow through it. This serves two purposes: (a) it enables fair
comparison across context lengths because the same representation is used at all L, and (b) it
dramatically reduces compute by pre-extracting all embeddings once and memory-mapping them at
training time.

**Q4.2 Why were LSTM, Transformer, and MeanPool chosen as the three downstream heads?**

The three heads represent a principled ablation of temporal modeling capacity:

- **MeanPool** (no temporal order): temporal mean-pooling followed by a linear classifier. This
  baseline establishes how much the model can achieve without any temporal integration. If MeanPool
  performs on par with the sequence heads, the task does not require temporal reasoning — the
  bag-of-patches representation is sufficient. Parameter count: ~1K (a single linear layer).

- **LSTM** (sequential, inductive bias for time series): a 1-layer bidirectional LSTM that
  processes the context window sequentially and uses the final hidden state for classification.
  BiLSTM is preferred over unidirectional LSTM because the full window is available at inference
  time (non-causal setting). ~658K parameters for seq2label tasks (hidden=128).

- **Transformer** (attention-based, no positional inductive bias): a 1-layer Transformer encoder
  with a learned CLS token and sinusoidal positional encoding. The Transformer can model arbitrary
  long-range dependencies within a context window, while the LSTM processes information
  sequentially. ~264K parameters for seq2label tasks (d\_model=128).

Together, these three architectures form an ordering by modeling assumptions: MeanPool < LSTM <
Transformer, from least to most flexible temporal integration. If results agree across heads, the
finding is robust to architecture choice. Where they diverge, the divergence itself is informative
(e.g., if Transformer shows stronger context saturation effects, this suggests long-range
dependencies that BiLSTM's sequential processing cannot exploit as effectively).

**Q4.3 Why are the heads trained from scratch?**

The downstream heads contain only task-specific parameters (initialized randomly) and are trained
entirely on the labeled subset for each task. The SleepFM encoder is frozen. This design is
intentional: it makes the context-length experiment a controlled study of what a lightweight head
can learn from embeddings at different temporal scales — not a study of how much fine-tuning the
encoder helps. It also ensures that any AUROC difference between context lengths reflects
representational quality, not encoder adaptation.

**Q4.4 Was a Mamba head considered?**

The SleepMaMi paper (arXiv:2602.07628) uses a bidirectional Mamba macro-encoder for full-night
temporal aggregation and reports improved performance over Transformer baselines. Adding Mamba as a
fourth head is noted as motivated future work in the project documentation (`PAPER_PLAN.md`,
`SOTA_COMPARISON_AND_ABLATIONS.md` §3.3). It was not included in the current study to keep the
comparison clean (three architectures are already sufficient to test the temporal-modeling spectrum)
and because Mamba's advantage is expected to appear primarily at very long contexts where quadratic
attention is expensive — a regime addressed here via the cohort filter and Flash attention.

**Q4.5 Why does sleep staging use a different (larger) head architecture?**

Sleep staging (seq2seq, 5-class) uses hidden\_dim = 256 and 2 LSTM layers (~3.16M parameters),
compared to hidden\_dim = 128 and 1 layer (~658K) for all seq2label tasks. This was driven by
empirical evidence: during development, the 128/1 architecture achieved Cohen's κ = 0.54 at 10 m
context, while the 256/2 architecture achieved κ = 0.62–0.63. The jump is attributable to the
greater per-step complexity of 5-class sequential prediction (each patch must encode both local
stage features and their temporal context) versus scalar binary prediction (which can leverage the
aggregated context summary). The full-channel run uses the same 256/2 staging architecture as the
fast-channel run, ensuring the channel-comparison is a fair head-to-head.

---

## 5. Training Procedure

**Q5.1 What optimizer was used?**

**AdamW** with `weight_decay = 1e-3`. The L2 regularization in AdamW is appropriate for the
moderately-sized head (658K – 3.16M parameters) trained on datasets of 1,500 – 16,000 subjects.

**Q5.2 What learning rate and schedule were used?**

Base learning rate: **1e-4** for context lengths 30 s, 10 min, 40 min, and 80 min. For context
lengths 120 min and 240 min the learning rate is halved to **5e-5**. The rationale for the
per-context LR reduction is that at longer context lengths, the K = 5 overlapping windows sampled
per subject per epoch are more correlated (average window spacing is ~48 min at 240 m vs ~96 min at
30 s), reducing effective gradient diversity per step. A lower learning rate compensates for this
reduced diversity.

The schedule is **cosine annealing** over the training epochs (from `lr` to 0). No warmup is used.

**Q5.3 What is the batch size?**

**Batch size = 32** at all context lengths. No gradient accumulation is required after applying (a)
the cohort consistency filter (removing subjects shorter than 240 min, guaranteeing all-False
padding masks) and (b) the Transformer mask fix (passing `src_key_padding_mask = None` when the
mask is all-False, enabling Flash attention's O(N) memory path). Before these fixes, the
Transformer head at 240 m with batch = 32 triggered CUDA OOM by allocating ~42 GB for O(N²) Math
attention on an H100 MIG slice with 9.75 GB of GPU memory.

For gradient accumulation infrastructure: the effective batch is always 32. If a different GPU or
head requires a smaller micro-batch (e.g., micro-batch = 8 with accum\_steps = 4 for batch = 32),
`gen_commands.py` computes the accumulation steps automatically from the per-context
`context_micro_batch` entries in `experiments/v2_registry.yaml`.

**Q5.4 How was early stopping determined?**

For all seq2label tasks: early stopping monitors **validation AUROC** (macro OvR for multi-class),
with patience = 10 epochs and a maximum of 40 epochs. The best checkpoint (by val\_auroc) is saved
as `best_model.pt`.

For sleep staging (seq2seq): early stopping monitors **validation Cohen's κ** (patience = 10,
maximum 60 epochs). The switch from val\_auroc to val\_kappa for staging was made because val\_auroc
for 5-class OvR is slow to plateau and caused the 120 m staging run to hit the 40-epoch ceiling
without converging. Val\_kappa directly optimizes the primary reported metric and yields faster,
more reliable convergence.

**Q5.5 Were any regularization techniques used beyond weight decay?**

Yes. Each head applies a **Dropout** layer before the final linear classifier (dropout probability
specified in the head implementation; typically p = 0.1–0.2 for the lightweight heads). Dropout is
applied during training only (standard `model.train()` / `model.eval()` toggling).

Additionally, **inverse-frequency class weights** are applied in the CrossEntropyLoss for all
tasks. This is used instead of WeightedRandomSampler so that per-subject window sampling remains
stratified by subject identity (not by label), preserving the K = 5 fixed windows per subject
invariant. The class weights are computed from the training split label distribution before each
training run and stored in `metrics.json`.

**Q5.6 How many epochs were trained, and what was the typical stopping epoch?**

Maximum epochs: 40 for seq2label; 60 for sleep staging. Early stopping typically fires between
epochs 15–30 for seq2label tasks. Observed stopping epochs for `sex_binary_lstm`: 18 epochs at
30 s, 17 epochs at 10 m. Sleep staging converged at higher epoch counts due to the larger
architecture and 5-class complexity.

---

## 6. Dataset and Preprocessing

**Q6.1 Why were these four specific datasets chosen (SHHS, APPLES, MrOS, STAGES)?**

The four datasets were chosen to maximize task coverage, sample size, and demographic diversity
within the NSRR (National Sleep Research Resource) data ecosystem:

- **SHHS** (Sleep Heart Health Study, N ≈ 8,444): the largest epidemiological sleep cohort in
  NSRR. Provides CVD labels, ESS, AHI, sex, age, BMI, sleep efficiency. Two visits (SHHS1,
  SHHS2) contributing independently.
- **MrOS** (Osteoporotic Fractures in Men Sleep Study, N ≈ 3,933): older men (≥ 65), two visits.
  Uniquely provides PSQI (Pittsburgh Sleep Quality Index) and CVD (coronary heart disease).
  All-male, so excluded from `sex_binary`.
- **APPLES** (Apnea Positive Pressure Long-Term Efficacy Study, N ≈ 1,104): OSA-enriched cohort
  with rich questionnaire data including BDI (depression), clinician-adjudicated OSA severity,
  ESS, and sleep efficiency. The OSA-enriched design makes APPLES the only dataset with a
  reasonable class balance for severe-OSA prediction.
- **STAGES** (Stanford Technology Analytics and Genomics in Sleep, N ≈ 1,513): multi-site
  academic sleep clinic cohort with unique psychiatric instruments: PHQ-9 (depression), GAD-7
  (anxiety), and ISI (insomnia). Younger and more diverse than MrOS and SHHS.

No other NSRR dataset had usable HDF5 preprocessing pipelines and label extraction completed
within the project timeline.

**Q6.2 What is the "fast-channel" channel subset?**

The fast-channel strategy (used for all v3 primary results) selects up to **7–8 channels** per
subject: BAS up to 4 (C3-M2, C4-M1, LOC, ROC), RESP = 1 (Airflow), EKG = 1, EMG = 1–2 (CHIN,
LLEG when available). This is the minimum channel set sufficient for sleep staging (two central EEG
leads plus both EOG channels) and for providing signal to all four SleepFM modality groups.

Not all channels are available in all datasets: SHHS uses generic EEG channel names ("EEG",
"EEG sec") rather than electrode-specific names, limiting the number of distinguishable BAS
channels to 1–2 in practice.

**Q6.3 What does the "full-channel" set include, and why is it a separate run?**

The full-channel strategy (v3\_full, in progress) selects up to **23 channels**: BAS up to 10
(C3-M2, C4-M1, LOC, ROC, O1-M2, O2-M1, F3-M2, F4-M1, A1, A2), RESP up to 7 (Airflow, Thor,
ABD, SpO2, HR, Snore, RespRate), EKG up to 2, EMG up to 4. The full-channel run is a separate
preprocessing and embedding extraction pass, using a separate HDF5 directory
(`/scratch/boshra95/psg_full/`) and separate embeddings directory. Head architectures are identical
between fast and full runs (hidden = 128, layers = 1) so any AUROC improvement is attributable
solely to richer channel information, not to a larger model.

**Q6.4 Why 128 Hz resampling?**

128 Hz is SleepFM's expected input sampling rate. The original SleepFM model was pre-trained on
signals at 128 Hz, so all channels must be resampled to this rate before embedding extraction.
The patch size of 640 samples per patch equals exactly 5 seconds at 128 Hz.

**Q6.5 What bandpass filters were applied and why?**

Per-channel FIR bandpass filters are applied before resampling, using MNE-Python's `filter_data`
(linear-phase, zero-phase, FFT-based):

| Modality | Band | Rationale |
|---|---|---|
| EEG / EOG (BAS) | 0.3 – 35 Hz | Standard sleep-EEG band; removes DC drift, power-line noise, and high-frequency EMG contamination |
| EKG / ECG | 0.5 – 45 Hz | Captures QRS complex and T-wave; rejects DC drift |
| EMG | 10 – 100 Hz | Captures muscle burst activity; rejects low-frequency motion artifacts |
| RESP | 0.05 – 2.0 Hz | Captures breathing at 0.1–0.5 Hz; rejects faster oscillations irrelevant to respiratory effort |

These bands follow standard clinical PSG processing conventions and match SleepFM's pre-training
preprocessing (inferred from the SleepFM paper/codebase).

**Q6.6 What normalization is applied?**

After bandpass filtering and resampling, each channel is **z-score normalized** over the full
overnight recording (mean subtracted, divided by standard deviation). Channels with standard
deviation = 0 (flat channels) receive only mean-centering. NaN values are set to 0 and Inf values
are clipped to ±10 after normalization. HDF5 files are stored as float16 with gzip compression
(level 4).

---

## 7. Train / Val / Test Splits

**Q7.1 How were subjects split into train, val, and test?**

**70% train / 15% validation / 15% test**, stratified by subject identity (not by window), with
random seed = 42. Stratification is performed at the subject level so that all windows from a given
subject appear exclusively in one split. No subject appears in more than one split. For multi-visit
datasets (SHHS with two visits, MrOS with two visits), both visit rows from a single subject are
assigned to the same split together — visit data is not split independently.

**Q7.2 Was there any dataset-level or cohort-level stratification?**

The split is stratified by task-specific label (binary or multi-class) within the combined
multi-dataset subject pool, not stratified separately per dataset. This means the dataset
composition in each split approximately mirrors the overall dataset composition. No explicit
dataset-level balancing was applied: if SHHS contributes 60% of subjects for a task, it contributes
approximately 60% of each split's subjects. Per-cohort AUROC breakdowns (S-Fig 5) are provided in
supplementary to characterize within-cohort and cross-cohort performance.

**Q7.3 Are SHHS1 and SHHS2 visits treated independently?**

Yes, for subject-count and label purposes. SHHS Visit 1 (shhs1) and Visit 2 (shhs2) subjects each
contribute a separate row to the master targets file. Both visits contribute independently to the
training pool. However, both visits from the same subject are kept in the same split (train, val,
or test) to prevent data leakage. MrOS Visit 1 and Visit 2 are treated the same way. APPLES and
STAGES are single-visit cohorts.

---

## 8. Task Definition and Label Choices

**Q8.1 Why use an "extreme-group" design for depression (BDI ≤ 9 vs ≥ 20)?**

Standard depression screening typically uses BDI ≥ 14 (mild) or ≥ 10 (PHQ-9) as a positive
threshold. The extreme-group design (positive = BDI ≥ 20 or PHQ-9 ≥ 15; negative = BDI ≤ 9 or
PHQ-9 ≤ 4; middle range excluded) was chosen for two reasons: (a) it maximizes the effect size by
contrasting clearly depressed and clearly non-depressed subjects, and (b) the APPLES cohort has
only 27 subjects with BDI ≥ 10 at a binary threshold — far too few for a well-powered classifier.
With the extreme-group design, the positive class in APPLES is 234 subjects. The cost is a reduced
effective N (from ~2,800 to ~1,761) and the explicit exclusion of the middle range.

An important design note: combining APPLES (BDI instrument) and STAGES (PHQ-9 instrument) required
mapping both instruments to a common binary label. The thresholds used (BDI ≤ 9 → negative, ≥ 20
→ positive; PHQ-9 ≤ 4 → negative, ≥ 15 → positive) were chosen to represent comparable severity
bands on each instrument.

**Q8.2 Why AHI ≥ 15 as the apnea threshold?**

AHI ≥ 15 (moderate-to-severe OSA) is the standard clinical threshold for intervention
recommendation in US and international sleep medicine guidelines. It produces near-balanced classes
across the combined dataset (N = 14,097; 48.9% positive, 51.1% negative), which is favorable for
both AUROC optimization and clinical interpretability. The 4-class AHI severity categorization
(< 5, 5–15, 15–30, ≥ 30) was computed in parallel and provides a more granular label but was not
the primary analysis target.

**Q8.3 Why is `osa_binary_apples_postqc` treated as a separate task from `apnea_binary`?**

`osa_binary_apples_postqc` uses the clinician-adjudicated post-quality-control severity rating from
the APPLES study, which assigns each subject to a four-level severity category (Non-randomized,
Mild, Moderate, Severe) based on polysomnographic and clinical review. The binary version collapses
Non-randomized+Mild to negative and Moderate+Severe to positive. This is distinct from the
AHI-threshold-based `apnea_binary` label because: (a) it incorporates clinical judgment beyond a
single AHI threshold, (b) it is APPLES-specific (no equivalent exists in SHHS, MrOS, or STAGES),
and (c) it provides a methodological cross-validation: do PSG-derived embeddings predict
clinician-adjudicated severity as well as they predict AHI-derived labels?

**Q8.4 What tasks were considered but dropped or deferred?**

- **`insomnia_binary`** (STAGES only, ISI ≥ 15): AUROC ~0.58 in phase0, near chance. Deferred.
- **`rested_morning`** (MrOS only, subjective restedness): AUROC 0.52–0.54 in phase0, effectively
  chance. Deferred; unlikely to change.
- **`anxiety_binary`** (STAGES only, GAD-7 ≥ 10): AUROC ~0.57 in phase0. Deferred.
- **`fatigue_binary`** (FSS ≥ 36): disabled in config — zero valid subjects after filtering.
- **`age_regression`** and **`bmi_regression`** (continuous targets): deferred pending
  implementation of a regression head (MSE loss). Labels are prepared; code changes needed.
- **`cognition_regression`** (MMSE from APPLES): has ceiling effects (most subjects score 24–30)
  and no strong a priori reason to expect a PSG signal. Not added.

**Q8.5 Why is CVD defined differently across datasets?**

In SHHS, `cvd_binary` uses a composite variable `any_cvd` that aggregates coronary heart disease,
stroke, heart failure, and peripheral vascular disease. In MrOS, `cvd_binary` uses `cvchd`
(coronary heart disease only — a stricter definition). The mismatch is acknowledged explicitly in
the experiment notes and should be stated in the paper. Per-dataset CVD results are reported
separately in the cohort saturation analysis; the merged definition is used only for a combined
AUROC because a merged label is needed for an adequately powered training set.

---

## 9. Evaluation Metrics

**Q9.1 Why is AUROC the primary metric?**

AUROC (area under the receiver operating characteristic curve) is chosen as the primary metric
because it is threshold-free: it measures discriminability across the entire operating range without
requiring a fixed decision cutoff. This is important for two reasons: (a) the optimal clinical
threshold may differ across deployment contexts, and (b) comparing models at different context
lengths using a single threshold could introduce a systematic bias if the threshold calibration
changes with context. AUROC is also the standard primary metric in most clinical prediction
benchmarks, making results comparable to prior work. For multi-class tasks (age\_class,
osa\_severity\_apples, sleep\_staging), macro one-vs-rest AUROC is used.

Early stopping also uses validation AUROC (except sleep staging), ensuring that the selected
checkpoint maximizes the same quantity used for reporting.

**Q9.2 Why balanced accuracy as a secondary metric for imbalanced tasks?**

For imbalanced binary tasks, standard accuracy over-reports performance on the majority class.
Balanced accuracy (arithmetic mean of per-class recall) is robust to class imbalance and is
interpretable as "how well does the model do on each class independently." It is reported alongside
AUROC in paper tables at the optimal decision threshold selected on the validation set via Youden's
Index (threshold that maximizes balanced accuracy). AUROC is unchanged by this threshold choice.

Post-hoc threshold tuning produced meaningful improvements for some tasks: `osa_binary_apples_postqc`
gained +0.22 in balanced accuracy at 10 m (t = 0.5 was predicting class 1 for ~98% of subjects);
`depression_extreme_binary` gained +0.065 at 80 m. For `cvd_binary`, threshold tuning was
counterproductive (average –0.005) because the validation set is too small to reliably generalize.

**Q9.3 Why bootstrap confidence intervals with subject-level resampling?**

Subjects are the independent statistical unit, not windows. Windows within a subject are correlated
(they come from the same person's physiology). Resampling at the window level would underestimate
variance because it ignores this within-subject correlation. Subject-level bootstrap (resample
subjects with replacement, re-aggregate each bootstrap replicate's windows) correctly reflects the
effective sample size. 1,000 bootstrap replicates were used; the 2.5th and 97.5th percentiles form
the 95% CI. CIs are particularly important for comparing adjacent context lengths on the saturation
curve where differences may not be statistically meaningful.

**Q9.4 Why is Cohen's κ used for sleep staging instead of AUROC?**

Cohen's κ (kappa) measures agreement between predicted and true stage labels while correcting for
chance agreement. It is the standard primary metric in the sleep staging literature (e.g., AASM
scoring inter-rater reliability is reported as κ). AUROC in the 5-class OvR setting is also
computed for reference, but it is a less interpretable summary of staging performance than κ. Early
stopping for sleep staging monitors val\_kappa specifically because val\_auroc was slow to plateau
and caused early stopping to fire prematurely at long context lengths.

---

## 10. Comparison with State of the Art

**Q10.1 Why is there no direct AUROC comparison with OSF, SleepMaMi, or SleepFounder?**

Three reasons make direct comparison infeasible or misleading:

1. **Training data contamination**: OSF (arXiv:2603.00190, ICML 2026) was pre-trained on SHHS and
   MrOS, which are our primary test cohorts. Comparing AUROC directly would not be a fair
   evaluation of generalization — both systems would have "seen" SHHS/MrOS data.

2. **Different evaluation protocols**: SleepFounder fine-tunes the full model on downstream tasks
   (all backbone weights updated); we train only a lightweight head on frozen SleepFM embeddings.
   AUROC differences would reflect both backbone quality and downstream fine-tuning depth, not a
   clean single-variable comparison.

3. **Different task definitions**: SleepFounder reports OSA detection AUROC = 0.917, which uses a
   different cohort, a different AHI threshold, and a different label derivation procedure. Their
   sex AUROC = 0.85 and age MAE = 6.9 years are useful reference points but not directly
   comparable because of protocol differences.

Instead of claiming numerical superiority, the paper frames its contribution as orthogonal:
none of the three SOTA systems systematically varies context length as a scientific variable.
SleepFounder, SleepMaMi, and OSF all use fixed context (epoch-level or full-night). Our primary
contribution is the context-length analysis, not encoder performance.

**Q10.2 What would a fair comparison require?**

A fair backbone comparison would require: (a) using OSF's publicly released weights as the frozen
encoder, (b) running the same context-length sweep on the same four NSRR cohorts, (c) using the
same task definitions and splits, and (d) reporting on held-out cohorts not included in OSF's
pre-training. The project documentation notes this as future work: "replacing SleepFM with OSF
embeddings is a one-line change." Such an experiment was not run in the current study because it
requires re-extracting embeddings for all ~15,000 subjects using OSF's model.

---

## 11. Versioning — v3 vs v2 vs v1

**Q11.1 What changed between phase0 (v1), phase0\_v2, and phase0\_v3?**

**phase0 (v1) — original protocol (stale; do not use):**
- Used non-overlapping N-aligned window positions for training, val, and inference.
- At 240 m context, training K effectively collapsed to K = 1 or K = 2 (only 1–2 non-overlapping
  windows fit a typical night). This was a hidden asymmetry.
- Val and test during training used the same non-overlapping pool, so the early-stopping signal
  was unreliable at long contexts.
- Batch sizes varied across context lengths (gradient accumulation not systematically applied).

**phase0\_v2 — interim fix (stale; do not use):**
- Partial fixes applied. Results in `phase0_v2/` directories.
- Non-overlapping window sampling was not yet fixed.
- Results are invalid for the paper.

**phase0\_v3 — current protocol (all results valid):**
- **Key fix 1 (overlapping training windows):** Training samples from the overlapping pool (any
  start in [0, T − N]), ensuring K = 5 at all context lengths. Val and test during training also
  use overlapping pool with K = 5 evenly-spaced positions — a deterministic, stable early-stopping
  signal at all context lengths including 240 m.
- **Key fix 2 (inference unchanged):** Inference uses non-overlapping stride-N windows (K\_max =
  99,999 effectively) — unchanged.
- **Batch fix:** Cohort filter (min\_recording\_patches = 2880) removes subjects shorter than
  240 m, guaranteeing all-False padding masks at every context length. With the Transformer mask
  fix (src\_key\_padding\_mask = None when all-False), Flash attention fires at all context
  lengths and batch = 32 fits in memory everywhere.
- **Context-specific LR:** 5e-5 at 120 m and 240 m (halved from 1e-4).
- **Results dir:** `phase0_v3/`, logs in `logs_v3/`.

**Q11.2 Why is v3 the definitive version?**

The v2 overlapping-window bug (Issue 1 in `docs/TRAINING_PROTOCOL_FIXES.md`) meant that any
AUROC difference between short and long contexts in v2 results could partly reflect training-data
asymmetry (the 30 s model receiving more gradient updates per epoch than the 240 m model) rather
than context quality. The v3 fix eliminates this confound: the only primary variable between
experiments is now the context length L. Paper claim: "All models were trained with K = 5
randomly-sampled overlapping windows per subject per epoch, keeping the number of gradient updates
per subject constant across context lengths."

---

## 12. Compute and FLOPs Analysis

**Q12.1 How are FLOPs computed for the compute-scaling analysis?**

FLOPs are computed analytically from quantities recorded in `metrics.json` for each training run,
using the following head-specific formulas (factor of 3 covers forward pass, backward pass, and
optimizer step):

| Head | FLOPs per gradient step |
|---|---|
| LSTM | 3 × seq\_len × 4 × hidden\_dim × (input\_dim + hidden\_dim) |
| Transformer | 3 × seq\_len × (seq\_len × hidden\_dim + 4 × hidden\_dim²) |
| MeanPool | 3 × seq\_len × input\_dim |

Where `seq_len = N` (patches in context window), `input_dim = 512` (SleepFM embedding size),
`hidden_dim = 128` (seq2label) or 256 (sleep staging). Total training FLOPs to epoch E:
`effective_batch_size × effective_steps_per_epoch × FLOPs_per_step × E`.

These values are recorded in `training_curves.csv` (per epoch) and `metrics.json` (at best epoch)
and collected into `results/collected/phase0_v3/training.csv`.

**Q12.2 What hardware was used?**

All training and inference were run on the **Compute Canada** HPC cluster using **NVIDIA H100 MIG
slices** (10 GB = 9.75 GiB GPU memory per MIG slice, typically on the `rorqual` partition).
Preprocessing (EDF → HDF5) and embedding extraction ran on the same cluster, with preprocessing
taking up to 26 h wall time for the largest datasets (SHHS, 8,444 subjects).

**Q12.3 What were typical training runtimes per experiment?**

For `sex_binary_lstm` (N ≈ 9,500 subjects, large n\_size):

| Context | Observed wall time | Stopping epoch |
|---|---|---|
| 30 s | 31 min | 18 |
| 10 min | 46 min | 17 |
| 80 min | ~111 min (estimated) | ~25 |
| 120 min | ~185 min (estimated) | ~25 |

The 50% safety margin baked into wall-time estimates (`gen_commands.py` `_TRAIN_HOURS` table) means
jobs rarely time out without resume; when they do, automatic requeue via SLURM `--requeue` and a
per-epoch `resume.pt` checkpoint enables seamless continuation.

---

## 13. Things That Did Not Work or Were Surprising

**Q13.1 What experimental design was flawed in v2 and had to be corrected?**

The most consequential flaw was the non-overlapping window sampling in v2 (`context_window_dataset.py`).
The code used integer division (`n_windows = T // N`) to determine the number of valid windows
and sampled from N-aligned positions only. For 240 m context (N = 2880) on a typical 8-hour night
(T ≈ 5760): `n_windows = 5760 // 2880 = 2` → K = min(5, 2) = 2. The 30 s model was effectively
getting 5 gradient updates per subject per epoch while the 240 m model was getting only 2.
This invalidated all v2 context-length comparisons.

**Q13.2 What caused CUDA OOM for the Transformer at 240 m before the fix?**

Before the cohort filter (`min_recording_patches = 2880`), real training batches at 240 m
occasionally contained subjects shorter than 2880 patches. Zero-padding those subjects produced a
non-zero `src_key_padding_mask` tensor. PyTorch's fused SDPA (Scaled Dot-Product Attention) kernel
falls back to O(N²) Math attention when any mask entry is non-zero, instead of the O(N) Flash
attention path. At N = 2881 patches (240 m + CLS token), a single padded sample in a batch of 168
attempted to allocate ~42 GB — immediately OOM on a 9.75 GB MIG slice.

The fix was two-fold: (a) exclude short recordings, and (b) explicitly pass
`src_key_padding_mask = None` when the mask is all-False (signaling to PyTorch that Flash attention
is safe). Both fixes together ensure that training logs confirm `[Attn] Flash (mask=None, O(N)
memory)` for every batch.

**Q13.3 Was it surprising that MeanPool was competitive at any context length?**

Yes. For several tasks (particularly `bmi_binary` and `sex_binary`), MeanPool — which discards all
temporal ordering and simply averages patch embeddings — achieved AUROC within a few percentage
points of LSTM and Transformer at long context lengths. This suggests that for these tasks, the
temporal structure within a context window adds only marginal information beyond the bag-of-patches
representation. The insight is that if the SleepFM embeddings are already rich enough at the
patch level, temporal integration within the head may not contribute much. This finding
strengthens the case for deploying lightweight aggregation in resource-constrained settings.

**Q13.4 What was the sleep staging cohort issue?**

Including STAGES in the sleep staging training set consistently hurt Cohen's κ relative to
excluding it. STAGES is a multi-site sleep clinic cohort with different demographic and clinical
characteristics from SHHS, MrOS, and APPLES. The primary sleep staging model (`sleep_staging_lstm`,
`sleep_staging_transformer`, `sleep_staging_mean_pool`) was therefore trained on SHHS + MrOS +
APPLES only. A STAGES-included comparison run (`sleep_staging_lstm_with_stages`,
`sleep_staging_transformer_with_stages`) is maintained for ablation reference.

**Q13.5 What was unexpected about post-hoc threshold tuning?**

Two surprises emerged from the threshold tuning analysis:

1. **`osa_binary_apples_postqc`** at t = 0.5 predicted class 1 (moderate/severe OSA) for ~98% of
   all test subjects — yielding a balanced accuracy of only ~0.58, despite AUROC = 0.74. With the
   val-optimized threshold (t\_opt ≈ 0.76), balanced accuracy jumped to ~0.80 (+0.22). This extreme
   miscalibration at t = 0.5 was not anticipated and would have made the table results look much
   worse than the model actually performs.

2. **`depression_extreme_binary`** at long contexts (80 m, 120 m) was highly miscalibrated at t =
   0.5 despite being approximately balanced at short contexts. The val-optimized threshold gained
   +0.065 BA at 80 m. Why miscalibration worsens with longer context for this task is unclear —
   possibly the extreme-group label imbalance (87% negative) interacts with longer-context
   representations differently.

3. **`cvd_binary`** showed negative or near-zero gains from threshold tuning (average –0.005), and
   the val split is too small (~1,300 subjects for MrOS+SHHS combined) to reliably generalize the
   optimal threshold to the test split. For this task, reporting at t = 0.5 is preferable.

**Q13.6 What was the modality ablation architecture bug?**

All 25 modality ablation experiments completed on 2026-06-17 used the wrong architecture:
`hidden_dim: 256, num_layers: 2` (the sleep staging head configuration) instead of
`hidden_dim: 128, num_layers: 1` (the seq2label architecture used in the v3 baseline). This was a
copy-paste error when creating `configs/phase0_v3_abl_config.yaml` from the staging config. All 25
runs were invalidated and are being rerun with the corrected 128/1 architecture. The ablation
results in the paper will reflect only the corrected runs.

**Q13.7 What deferred tasks showed near-chance performance?**

`rested_morning` (MrOS only) showed AUROC 0.52–0.54 in phase0 — effectively chance. This is
consistent with the weak relationship between subjective morning restedness and overnight
polysomnographic features: morning ratings reflect multifactorial factors including expectations,
prior sleep debt, and circadian phase that are only partially encoded in single-night PSG. This task
is likely to be excluded from the paper unless v3 results improve substantially.

`anxiety_binary` (STAGES only, GAD-7 ≥ 10) showed AUROC ~0.57 in phase0. Anxiety symptoms may be
less directly encoded in PSG architecture than OSA or depression extremes, and STAGES-only training
limits statistical power.

**Q13.8 What is the sleep staging test-set composition confound?**

For sleep staging with the `complete_only` padding policy, longer context windows exclude more
anchor epochs from evaluation (epochs within L/2 of the recording boundary are excluded). At 240 m
context, approximately 50% of epochs are excluded — particularly the sleep-onset N1 epochs at the
start of the night (the hardest class). This means longer-context models are evaluated on a
slightly easier subset (fewer N1-heavy boundary epochs), which could artificially inflate κ at
long contexts relative to short contexts. A common-set evaluation (all models evaluated on only
240 m-valid anchors) is planned as a supplementary cross-check via `scripts/analyze_common_eval_set.py`.

---

*Document generated from codebase analysis — primary sources:
`docs/EXPERIMENTS_GUIDE.md`, `docs/TRAINING_PROTOCOL_FIXES.md`, `docs/PAPER_PLAN.md`,
`docs/context_length_experiment_design.md`, `docs/SOTA_COMPARISON_AND_ABLATIONS.md`,
`docs/cohort_filter.md`, `docs/PREPROCESSING_PIPELINE.md`, `docs/sleep_staging_design.md`,
`docs/POSTHOC_THRESHOLD_TUNING.md`, `docs/v2_implementation_notes.md`,
`docs/ANALYSIS_IDEAS.md`, and `CLASSIFICATION_TARGETS_ANALYSIS.md`.*
