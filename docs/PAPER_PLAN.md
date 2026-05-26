# Paper Plan — JBHI Submission
*Written 2026-05-26. Do not start writing LaTeX until this plan is approved.*

---

## A. Paper Identity

### Working Title Candidates

1. **"Context Efficiency in Overnight PSG: Task-Specific Saturation Points for Foundation Model-Based Clinical Prediction"** *(current template title, conservative — good)*
2. **"How Much of the Night Do You Need? Systematic Context-Length Analysis for Clinical Prediction from Overnight Polysomnography"** *(more conversational, broader appeal)*
3. **"Long Context or Many Short Windows? An Iso-Compute Analysis of Overnight PSG for Clinical Prediction"** *(emphasises H2, the most novel analysis angle)*

**Recommendation:** Candidate 1 or 3. Candidate 1 is most JBHI-appropriate (methods + clinical framing). Candidate 3 is most distinctive — no other PSG paper asks this question explicitly. **Decision needed from you.**

### Target Venue
IEEE Journal of Biomedical and Health Informatics (JBHI).
IEEE Transactions double-column format. Target: **10 pages** main text (including references, figures, tables) + unlimited supplementary material. Abstract: 150–250 words.

The template in `JBHI_submission/main.tex` is the standard IEEE Transactions/Journal color template — correct format.

### Core Hypothesis / Main Claim (2–3 sentences)

The amount of PSG temporal context required for near-optimal clinical prediction is **task-specific** and **measurable**: physiologically complex tasks (e.g., OSA detection) benefit from contexts of 80–240 minutes, while others (e.g., sleep staging, BMI classification) saturate within 10–40 minutes. Critically, **aggregating many short-context predictions cannot fully substitute for a single long-context forward pass** at matched total signal budgets for the most context-sensitive tasks, indicating that long-range temporal integration inside the model is contributing beyond what post-hoc averaging can recover. These findings provide actionable guidance for deployment: the minimum context length L* per task defines the computational cost floor for acceptable performance.

### What Makes This Novel vs SOTA

| SOTA paper | What they do | What we do differently |
|---|---|---|
| OSF (ICML 2026) | New pre-training paradigm, channel masking, 85M params | We study context length as an independent variable in *downstream evaluation* using a frozen backbone. Orthogonal axis. |
| SleepMaMi (arXiv 2026) | Hierarchical Micro/Macro dual encoder; Mamba for full-night macro | They implicitly acknowledge temporal scale matters but use a fixed full-night macro context. We sweep 6 lengths and quantify diminishing returns. |
| SleepFounder (medRxiv 2025) | 800K hours of cardiorespiratory-only pre-training | Full model fine-tuning, 2 modality types, 1 context length. We test 15 tasks × 6 contexts with a frozen encoder. |
| All three | Fixed context (epoch-level or full-night) | None systematically vary context length as a scientific variable. This is our primary contribution. |

Our secondary contribution is **task breadth**: 15 clinical tasks (sleep staging, OSA, CVD, BMI, sex, age, depression, sleep quality, sleepiness, etc.) across 4 diverse cohorts (SHHS, MrOS, APPLES, STAGES), enabling a cross-task sensitivity analysis that no prior work provides.

---

## B. Section Outline

### Abstract
- State the problem (how much PSG context is needed for clinical prediction? unknown systematically)
- State the method (frozen SleepFM + 3 lightweight heads + 6 context lengths × up to 15 tasks × 4 cohorts)
- State the primary findings (H1: task-specific saturation at L*; H2: short-window aggregation partially but not fully substitutes; head comparison; practical guidance)
- State implications (deployment cost, model design)
- **Length**: 150–250 words (one paragraph, required by IEEE)
- **Status**: write last (requires final numbers)

---

### I. Introduction (~600–700 words, ~1 column)

**Content:**
- Clinical motivation: PSG is the gold standard for sleep disorders; overnight recordings span 6–8 hours but current models use 30-second epochs or full nights
- Problem: the optimal temporal context for downstream clinical tasks is unknown; using too little loses signal, using too much wastes computation
- Key insight: this is a two-dimensional question — training context length L (what the model integrates) × inference aggregation K (how many windows per patient)
- Why foundation models make this timely: frozen SleepFM provides per-patch embeddings; we can sweep context length cheaply without retraining the encoder
- Four hypotheses (H1–H4): (1) saturation at task-specific L*, (2) aggregation substitution (H2), (3) aggregation saturation (H3), (4) temporal head advantage (H4)
- Summary of contributions (bullet list)
- Paper structure sentence

**Figures/Tables anchoring this section:** None (Introduction typically has no figures)

**Status:** Can be written now (does not require results). Draft when Methods is done.

---

### II. Related Work (~400–500 words, ~1 column)

**Content (3 paragraphs):**

**P1 — Sleep Foundation Models:**
SleepFM (our encoder, frozen) → OSF (ICML 2026: channel masking, scaling laws) → SleepMaMi (Macro+Micro hierarchy, Mamba) → SleepFounder (800K hours, cardiorespiratory only). Framing: all four train or fine-tune a backbone at fixed context; none systematically vary context length in downstream evaluation.

**P2 — Context Length in Time-Series / Clinical AI:**
Brief survey of context-length effects in EHR sequences (e.g., Transformer-based EHR models showing diminishing returns after ~1 year of history), ECG and PPG (short-segment vs multi-hour aggregation), sleep scoring (epoch-level vs multi-epoch context for staging). Key point: PSG-specific context-length analysis is missing. The closest analogue is `sleep scoring with variable context windows` (cite relevant staging papers if any exist).

**P3 — Multi-Window Aggregation:**
Test-time aggregation (majority vote, mean-prob) as an inference-time strategy. Connection to multi-instance learning (MIL). SleepMaMi's finding that Macro > Micro supports our hypothesis. OSF's channel masking finding motivates our modality note (out of scope for this paper but acknowledged).

**Figures/Tables:** None

**Status:** Can be drafted now. Will need to update citations based on final literature search.

---

### III. Methods (~1,200–1,500 words, ~2.5 columns)

This is the most objective section and can be **written now**. Full detail below in Section D.

**Subsections:**
- III-A: Datasets
- III-B: SleepFM Encoder (frozen)
- III-C: Preprocessing and Embedding Extraction
- III-D: Context-Window Dataset
- III-E: Downstream Sequence Heads (LSTM, Transformer, MeanPool)
- III-F: Training Protocol
- III-G: Context-Length Sweep Design
- III-H: Evaluation Protocol

**Figures anchoring this section:**
- **Fig 1 (System Overview):** Flowchart showing: PSG recording → channel grouping → SleepFM (frozen) → [T × 512] patch embeddings → context window extraction (length L) → sequence head (LSTM/Transformer/MeanPool) → prediction → aggregation. Single-column or double-column depending on layout.

**Status:** ALL of this can be written now. No experimental results needed.

---

### IV. Experiments and Results (~1,500–2,000 words, ~3 columns)

**Subsections:**
- IV-A: Context Saturation (H1) — saturation curves, AUROC vs L at K=all
- IV-B: Aggregation Substitution (H2) — iso-compute heatmap, Pareto front
- IV-C: Aggregation Saturation (H3) — AUROC vs K curves per context
- IV-D: Head Comparison (H4) — LSTM vs Transformer vs MeanPool
- IV-E: Sleep Staging — Cohen's κ analysis (seq2seq, separate from IV-A)
- IV-F: Task Sensitivity Analysis — cross-task scatter (difficulty vs sensitivity)

**Figures anchoring this section:**
- **Fig 2:** Saturation curves (AUROC vs context length, log x-axis) for all Tier 1 tasks, LSTM head; error bands = 95% bootstrap CI. One line per task. [PENDING: Groups C+D]
- **Fig 3:** 2D iso-compute heatmap for one hero task (apnea_binary or bmi_binary), LSTM head. L on y-axis, K on x-axis, AUROC as heatmap color, iso-compute lines overlaid. [PENDING or PARTIAL]
- **Fig 4:** Head comparison: 3-panel saturation curves for one representative Tier 1 task (one line per head). [PENDING: Groups C+D]
- **Fig 5:** Sleep staging: Cohen's κ vs context length, 3 heads. [PENDING: Group D]
- **Table I:** Main results table — best test AUROC per task × head × context. Can have [PENDING] entries.
- **Table II:** L* per task (smallest context within 0.5% of best AUROC). [PENDING]

**Status:** Largely PENDING. Some partial results for bmi_binary_lstm (4/6 contexts), sleep_efficiency_binary_lstm (partial), osa_binary_apples_postqc_lstm (40m–120m: 0.738, 0.721, 0.742), depression_extreme_binary_lstm (80m: 0.742, 120m: 0.750). Phase 0 legacy results exist but use old protocol — DO NOT USE in v3 paper figures.

---

### V. Discussion (~400–500 words, ~0.8 column)

**Content:**
- Interpret H1: which tasks are context-sensitive, which are not, and why (physiological reasoning)
- Interpret H2: when aggregation substitutes and when it doesn't — practical implications
- Practical recommendation: the L* table as a deployment guide ("for OSA screening, use ≥80 minutes of context")
- Limitations: fixed K=5 training windows (note as limitation; token-budget ablation not yet run), frozen backbone (cannot claim these findings transfer to other encoders), cohort-specific confounds (MrOS all-male for sex_binary, STAGES single-site)
- Comparison to related work: SleepMaMi's Macro/Micro finding (their Mamba macro-encoder confirms our H4); SleepFounder's disease prediction AUROC (we cannot compare directly — note evaluation protocol differences)
- Future directions: Ideas 1–2 (AMTA+CSL, early-exit); Mamba head; OSF backbone

**Figures/Tables:** None

**Status:** Can be outlined now; finalize after results are in.

---

### VI. Conclusion (~150–200 words, ~0.3 column)

**Content:**
- 1-sentence restatement of the problem
- 1-sentence method summary
- 2-3 sentences of key findings (task-specific saturation at L*; aggregation substitution is partial for complex tasks; head comparison)
- 1-sentence practical contribution (L* table as deployment guide)
- 1-sentence future outlook

**Status:** Draft after results; write last.

---

## C. Figure and Table Plan

Target for main paper: **~8 figures/tables** at 12–14 pages. The exact count and which tasks are featured figures will be finalised once all results are in.

### Main Paper Figures (proposed: 6 figures + 2 tables = 8 items)

| ID | Name | What it shows | Location | Requires | Generator |
|---|---|---|---|---|---|
| **Fig 1** | System overview | PSG → SleepFM (frozen) → patches → head → prediction pipeline | Main | None (can draw now) | Hand-drawn / matplotlib |
| **Fig 2** | Saturation curves (2–3 tasks) | AUROC vs context length (log x), 3 heads per panel, 95% CI bands; 2–3 task panels chosen after results in (e.g., high-sensitivity apnea, low-sensitivity sleep_efficiency, mid-range bmi) | Main | Groups C+D ✗ PENDING | `plot_saturation.py --heads lstm transformer mean_pool --collected-dir` |
| **Fig 3** | 2D iso-compute heatmap | L×K grid, AUROC as colour, iso-compute lines; 1 hero task (finalise after results) | Main | Partial (bmi_binary) or PENDING | `plot_iso_compute.py` (heatmap) |
| **Fig 4** | Sleep staging saturation | Cohen's κ vs context length, 3 heads; per-stage F1 as inset or sub-panel | Main | Group D ✗ PENDING | `plot_saturation.py` (kappa metric) |
| **Fig 5** | Task sensitivity scatter | ΔAUROC (30s→best) vs baseline difficulty (1−AUROC@30s); each dot = one task; reference lines at medians | Main | All Groups ✗ PENDING | `plot_task_comparison.py` (§6A) |
| **Fig 6** | Modality ablation | Bar chart: AUROC for All / RESP+EKG / BAS only / No-BAS conditions at 2 contexts, 1–2 tasks | Main (if results available) | Ablation runs ✗ PENDING | Custom bar plot |
| **Table I** | Main results | Best test AUROC per task × best context (LSTM head); all tasks with AUROC ≥ ~0.62; 95% CI; balanced accuracy at t_opt; [PENDING] cells where experiments not yet done | Main | All Groups | `collect_results_v2.py` → manual |
| **Table II** | L* and context sensitivity | Per task: L* (min context within 0.5% of best), ΔAUROC (30s→best), N, datasets | Main | All Groups | `plot_task_comparison.py` (§6C) |

### Supplementary Figures

| ID | Name | What it shows | Requires |
|---|---|---|---|
| **S1** | Per-task saturation curves (all 3 heads) | One multi-panel figure: one panel per Tier 1 task, 3 lines per head | Groups C+D |
| **S2** | Iso-compute deep dive | Pareto front, min-cost frontier, marginal gain, double-tradeoff (4 plots) for hero task | See Fig 3 |
| **S3** | AUROC vs K (aggregation saturation, H3) | Per-context K-sweep curves for 2 representative tasks; shows where K-saturation occurs | PARTIAL available |
| **S4** | Task sensitivity scatter (§6A) | Scatter: baseline difficulty (1−AUROC@30s) vs context gain (ΔAUROC); each task is a dot | All Groups |
| **S5** | Per-dataset saturation | Cohort breakdown for multi-dataset tasks (SHHS, MrOS, APPLES separately) | Groups C+D |
| **S6** | Tier 2 tasks results | Saturation curves for psqi, depression_extreme, osa_postqc, osa_severity, cvd, sleepiness | Groups A+B |
| **S7** | Threshold tuning | Balanced accuracy at t=0.5 vs t_opt for imbalanced tasks (bmi_binary, osa_postqc) | Post-inference analysis |
| **S8** | Subject-level prediction stability | Within-subject variance distribution and K* histogram (min windows to correct classification) | Inference parquets |

---

## D. Methods Checklist

Every element that the Methods section must cover:

### III-A: Datasets

| Dataset | N (approx) | Population | Notes for paper |
|---|---|---|---|
| SHHS (Sleep Heart Health Study) | ~5,000 (v1+v2 combined) | Community adults, 2 visits | Large epidemiological cohort; v1 and v2 contribute independently where applicable |
| MrOS (Osteoporotic Fractures in Men Sleep Study) | ~3,000 (v1+v2 combined) | Older men (≥65), 2 visits | All-male; excluded from sex_binary |
| APPLES | ~1,103 usable | OSA-enriched; rich questionnaire data | 412 subjects have no PSG on NSRR (excluded automatically) |
| STAGES | ~1,500 | Multi-site; psychiatric questionnaires (PHQ-9, GAD-7, ISI) | 152 subjects with NaN embeddings excluded via blocklist |

After the cohort consistency filter (recordings < 240 min excluded from all context lengths): 20 subjects removed overall (≤0.21% of largest tasks; full list in `docs/excluded_subjects_T_lt_2880.csv`).

**Inclusion/exclusion:** subjects with usable PSG embeddings and at least one label for the task under study. Splits: 70% train / 15% val / 15% test, stratified by subject (not window), seed=42.

**Task label definitions** (document each task's threshold and source):
- `sex_binary`: Female=1 (self-reported)
- `sleep_efficiency_binary`: TST/TIB < 0.85 = poor (PSG-derived)
- `bmi_binary`: BMI ≥ 30 (WHO obesity threshold)
- `age_class`: <50 / 50–64 / ≥65 (3-class)
- `apnea_binary`: AHI ≥ 15 (moderate-severe OSA, standard clinical threshold)
- `sleep_staging`: 5-class (W/N1/N2/N3/REM), PSG-scored
- `psqi_binary`: PSQI > 5 (MrOS only)
- `depression_extreme_binary`: extreme-group design — BDI ≤9→0 or ≥20→1 (APPLES); PHQ-9 ≤4→0 or ≥15→1 (STAGES); middle group excluded
- `osa_binary_apples_postqc`: clinician-adjudicated severity: Non-rand+Mild→0, Mod+Severe→1 (APPLES only)
- `osa_severity_apples`: 4-class AHI severity (APPLES only)
- `cvd_binary`: SHHS = any_cvd composite; MrOS = cvchd only — **note definition mismatch in paper**
- `sleepiness_binary`: ESS ≥ 11

### III-B: SleepFM Encoder (Frozen)

- **Architecture**: SetTransformer; contrastive pre-training (leave-one-out strategy among 4 modality groups)
- **Input**: 5-second patches at 128 Hz (640 samples per patch) per channel, organized into 4 modality groups
- **Modality groups and channel priority:**
  - BAS (brain/EOG): up to 10 channels (C3-M2, C4-M1, LOC, ROC, O1-M2, O2-M1, F3-M2, F4-M1, A1, A2)
  - RESP (respiratory): up to 7 channels (Airflow, Thor, ABD, SpO2, HR, Snore, RespRate)
  - EKG: up to 2 channels
  - EMG: up to 4 channels (CHIN, LLEG, RLEG, EMG)
  - Missing channels → zero-padded with mask=True
- **Tokenizer**: Conv1D cascade (6 × [Conv1d → BN → ELU → LN], stride=2 each) → AdaptiveAvgPool1d → Linear; maps 640 samples → 128-dim patch embedding per channel
- **Spatial pooling across channels**: AttentionPooling (Transformer encoder layer) produces one 128-dim vector per modality group per patch
- **Output per subject**: `[T, 4, 128]` where T = total 5-second patches in the recording; flattened to `[T, 512]` before downstream use
- **Key property**: the encoder is **frozen throughout all experiments**. No gradients flow through it. This allows fair comparison across context lengths (the same representation is used at all L).

### III-C: Preprocessing and Embedding Extraction

- Raw PSG EDF files → HDF5 format at 128 Hz (`preprocess_signals.py`)
- Channel identification and priority-based selection (`extract_nsrr_channels.py`)
- SleepFM embedding extraction: chunked inference in 300-second (5-minute) segments, batch_size=16 (`extract_sleepfm_embeddings.py`)
- Output: per-subject `.npy` files (float16), ~5.6 MB each; ~22 GB total across ~4,000 subjects
- Embeddings are pre-extracted once and memory-mapped at training time — only the requested context window slice is loaded

### III-D: Context-Window Dataset (ContextWindowDataset)

**seq2label (all clinical prediction tasks):**
- Index: (subject, window_start) — K windows per subject per epoch
- Input: N consecutive patches starting at window_start, padded if recording ends early
- Label: night-level scalar (same for all windows of a subject)
- Training: K_train = 5 windows, randomly positioned within [0, T−N]; overlapping windows allowed
- Validation/test during training: K_val = 5 windows, evenly-spaced across [0, T−N], deterministic
- Inference: K_infer = all non-overlapping stride-N windows (T//N windows per subject)

**seq2seq (sleep staging):**
- Index: (subject, anchor_patch_idx) where anchor = a specific 30-sec epoch
- Input: N patches *ending at* the anchor (past-only window, causal)
- Label: scalar stage of the anchor epoch (0=W, 1=N1, 2=N2, 3=N3, 4=REM)
- Early anchors where past < N → zero-padded on the left with mask=True

**Cohort consistency filter:** subjects with total recording T < 2,880 patches (< 240 minutes) are excluded from all context lengths — not just from 240m. This ensures identical subject populations across the entire L sweep, preventing cohort shifts from confounding performance differences.

### III-E: Downstream Sequence Heads

All heads receive input `x ∈ ℝ^{B×N×512}` and a padding mask `mask ∈ {0,1}^{B×N}` (True = padded), producing `logits ∈ ℝ^{B×C}`.

| Head | Architecture | Parameters | Notes |
|---|---|---|---|
| **MeanPool** | Masked temporal mean → Dropout → Linear(512, C) | ~1K | No temporal order; strongest baseline for context-insensitive tasks |
| **LSTM** | 1-layer BiLSTM(512→128, bidirectional) → pack_padded_sequence → last hidden → Dropout → Linear(256, C) | ~655K | Handles variable-length via pack_padded_sequence; final hidden = cat(forward, backward) |
| **Transformer** | Linear(512→128) → CLS token prepend → sinusoidal PE → 1-layer TransformerEncoder(d_model=128, nhead=8, FF_dim=512, Pre-LN) → CLS output → Dropout → Linear(128, C) | ~264K | Flash attention when mask is all-False (O(N) memory); O(N²) Math attention otherwise |

All heads trained from scratch per (task, context length). The encoder is never updated.

### III-F: Training Protocol

- **Optimizer**: AdamW (weight_decay=1e-3)
- **Learning rate**: 1e-4 for contexts 30s–80m; 5e-5 for 120m and 240m (longer contexts have more correlated overlapping windows → reduced effective diversity per gradient step)
- **Schedule**: cosine annealing over training epochs
- **Epochs**: up to 40; early stopping on validation AUROC (patience=10)
- **Batch size**: 32 at all context lengths (no gradient accumulation needed after cohort filter + Flash attention fix)
- **Class imbalance**: inverse-frequency class weights in CrossEntropyLoss; no WeightedRandomSampler
- **Training K**: K_train = 5 windows per subject per epoch (fixed, context-length-independent). **Paper justification**: fixing K at 5 across all context lengths ensures that each model receives the same number of gradient updates per subject regardless of L — the only variable between experiments is the context window length itself. The alternative (token-budget K, where K ∝ 1/L) was explicitly rejected because it confounds training-exposure intensity with context length, making observed performance differences uninterpretable. Reference: `docs/context_length_experiment_design.md` §3 and §13.
- **Checkpointing**: best checkpoint saved by val_auroc; per-epoch resume.pt for SLURM timeout resilience
- **Paper claim**: "All models were trained with batch size 32, identical across all context lengths. The Transformer head uses Flash attention (O(N) memory) at all context lengths after a cohort consistency filter ensures padding-free batches. Training K was fixed at 5 windows per subject per epoch to ensure identical gradient exposure per subject across all context lengths."

### III-G: Context-Length Sweep Design

Six context lengths: 30s, 10m, 40m, 80m, 120m, 240m (= 6, 120, 480, 960, 1440, 2880 patches).

Rationale for these lengths: cover 3 decades (0.5 to 240 minutes) on a roughly log-spaced grid; 240m corresponds to the typical PSG recording duration; 30s corresponds to the standard sleep-staging epoch. The gap between 30s and 10m (where most tasks show the steepest ascent) is acknowledged as a limitation.

One model is trained per (task, head, context length). The context window length L is the **only variable** between experiments — the encoder, head architecture, optimizer, and subject splits are identical.

### III-H: Evaluation Protocol

**Primary metric**: AUROC (threshold-free; macro OvR for multi-class). Early stopping uses val_auroc.

**Secondary metric**: balanced accuracy at the optimal decision threshold selected on the val set (post-hoc threshold tuning for imbalanced binary tasks with significant recall imbalance at t=0.5). See footnote in paper tables.

**Sleep staging primary metric**: Cohen's κ + per-stage F1 (W, N1, N2, N3, REM). AUROC also logged for reference.

**Subject-level prediction (clinical prediction tasks)**: after inference over all non-overlapping windows, aggregate per-subject using mean-probability (soft) aggregation → compute AUROC. Also compare majority-vote aggregation.

**K-window sweep**: post-hoc, on CPU, from saved inference parquets: K ∈ {1, 5, 10, 20, 50, all} (sparse) and K ∈ {1, 2, ..., 500, all} (dense, for iso-compute analysis). No GPU needed.

**Iso-compute analysis**: for each (L, K) pair, compute AUROC from the K-window subsampled parquets; build 2D grid; draw iso-compute contour lines (K × L_min = constant) to compare configurations with the same total PSG budget per patient.

**Confidence intervals**: 95% bootstrap CIs (subject-level resampling, 1,000 bootstrap samples) on AUROC and balanced accuracy. CIs are needed for saturation curve error bands and for comparing L* vs L*±1 contexts.

---

## E. Main Paper vs Supplementary Split

**Principle**: main paper = the full story at 12–14 pages — saturation curves for 2–3 featured tasks, head comparison, iso-compute analysis, sleep staging, and a comprehensive results table covering all tasks with decent AUROC. Supplementary = per-task saturation figures for the remaining tasks, per-dataset breakdowns, ablation details, and implementation notes.

### Main Paper (target: 12–14 pages double-column)

| Content | Justification |
|---|---|
| System overview (Fig 1) | Required for reader to understand the pipeline |
| Saturation curves for 2–3 featured tasks across all heads (Fig 2) | Primary result for H1 and H4 — chosen after all results are in |
| 2D iso-compute heatmap for 1 hero task (Fig 3) | Primary result for H2+H3 |
| Sleep staging saturation: Cohen's κ + per-stage F1 vs context length (Fig 4) | Full Tier 1 seq2seq result |
| Task sensitivity scatter: ΔAUROC vs baseline difficulty (Fig 5) | Cross-task summary, shows task-specific context requirements |
| Modality ablation summary (Fig 6, if results available) | Addresses reviewer question about which modalities matter |
| Main results table — best AUROC + L* for ALL tasks with AUROC ≥ ~0.62 (Table I) | Full breadth claim supported; Tier 1 + Tier 2 in one table |
| L* and context sensitivity summary (Table II) | Actionable deployment guidance |

### Supplementary

| Content | Section |
|---|---|
| Per-task saturation curves for tasks not featured as main figures (all 3 heads) | S1 |
| Iso-compute deep dive: Pareto front, marginal gain, double-tradeoff (hero task) | S2 |
| Aggregation saturation curves (AUROC vs K, H3) for representative tasks | S3 |
| Per-dataset (cohort-level) saturation breakdown for multi-dataset tasks | S4 |
| Post-hoc threshold tuning: balanced accuracy tables at t_opt vs t=0.5 | S5 |
| Modality ablation details for all conditions (full table if main paper shows summary only) | S6 |
| Deferred task results (insomnia, anxiety, rested_morning) if AUROC > 0.60; else brief mention | S7 |
| Implementation details: embedding extraction, channel priority, cohort filter | S8 |
| Subject-level prediction stability: K* histogram, within-subject variance | S9 |
| Excluded subjects list (cohort filter) | S10 |

---

## F. Writing Order

| Priority | Section | Can write now? | Blocking on |
|---|---|---|---|
| 1 | **Methods (III)** | **YES** — fully writeable | Nothing |
| 2 | **Fig 1 (pipeline diagram)** | **YES** | Nothing |
| 3 | **Related Work (II)** | **YES** (structure fixed) | Final citations polish |
| 4 | **Introduction (I)** | **YES** (draft) | Should wait until Methods done for consistency |
| 5 | **Table I skeleton** | **YES** (mark [PENDING] cells) | All Groups A–D for full table |
| 6 | **Results IV-A (saturation, partial)** | **PARTIAL** — bmi_binary_lstm 4 contexts, osa_postqc 3 contexts available | Groups C+D for Tier 1 saturation curves |
| 7 | **Results IV-F (sensitivity matrix)** | **BLOCKED** | All Tier 1 results |
| 8 | **Results IV-B,C (iso-compute)** | **PARTIAL** — bmi_binary_lstm partial heatmap | Complete bmi_binary_lstm (Group A) + inference |
| 9 | **Results IV-D (head comparison)** | **BLOCKED** | Groups C+D (sex_binary mean_pool, bmi_binary mean_pool, age_class all heads) |
| 10 | **Results IV-E (sleep staging)** | **BLOCKED** | Group D (sleep_staging all 3 heads) |
| 11 | **Discussion (V)** | **PARTIAL** draft | Complete results |
| 12 | **Conclusion (VI)** | After results | Complete results |
| 13 | **Abstract** | **LAST** | All results and sections |

**Recommended starting point**: Write Methods (Section III) first — it is fully writeable today, it is the most objective section, and every other section references it. Then Related Work, then Introduction (can draft without numbers), then Results as experiments complete (Groups A→B→C→D in parallel with writing).

**Note on page budget (12–14 pages):** The extra page room vs a standard 10-page paper allows: (a) a fuller Methods section including the cohort table and all head architecture details, (b) 2–3 featured saturation figures instead of 1, (c) a fuller results table covering all tasks, and (d) the modality ablation subsection. Plan for ~2.5 pages Methods, ~3.5 pages Results, ~1 page Discussion, with the remaining budget split across Introduction, Related Work, and figures.

---

## G. Open Questions for Your Approval

These are framing and scope decisions that affect the structure of the paper. I need your input before writing begins.

### G1. Hero task for the iso-compute heatmap (Fig 3)
The 2D heatmap (L × K grid) is expensive to compute (requires dense K sweep) and is the most visually complex figure. We can only show one or two tasks in the main paper.

**Options:**
- `bmi_binary_lstm`: partially available now (4/6 contexts); large N (~15k), clean binary label, no label noise
- `apnea_binary_lstm`: primary clinical result; physiologically motivated for context-sensitivity; but PENDING (Group D)
- `sex_binary_lstm`: large N (~13k), strong signal expected; also PENDING (Group C)

**Recommendation**: bmi_binary_lstm for the initial heatmap (available soonest); apnea_binary for the paper if Group D finishes in time. Need your decision.

**Your answer**: Not finalised — want at least 2–3 task figures in the paper. Will decide once all results are in and we can see which tasks tell the most distinct stories (e.g., one high-sensitivity task like apnea, one low-sensitivity task like sleep efficiency, and possibly one mid-range task). The heatmap figure will cover whichever task(s) show the clearest iso-compute structure.

### G2. Sleep staging: equal weight or secondary?
Sleep staging is a different task type (seq2seq, anchor-based, per-epoch label) with different analysis (no K-aggregation, Cohen's κ not AUROC). Including it at equal weight with clinical prediction tasks may dilute the story; treating it as a secondary result simplifies the narrative.

**Options:**
- (A) Full Tier 1 treatment — Fig 5 in main paper, full discussion
- (B) Secondary result — Supplementary figure + one paragraph in Results discussing κ saturation
- (C) Drop from this paper — keep for a separate methods/staging paper

**Recommendation**: Option A, because staging is clinically important and the saturation result (κ improves 30s→40m then plateaus) is clean and interpretable. But if the paper is already long, move to supplementary. **Your call.**

**Your answer**: Option A — sleep staging is included as a main-paper task with full Tier 1 treatment. Even where specific analyses do not apply (e.g., the K-aggregation sweep and iso-compute heatmap are not relevant for seq2seq), the saturation curve (κ vs context length) and per-stage F1 are reported in the main text. Experiments will be run soon.

### G3. Is the training K=5 ablation required before submission?
Current protocol: K_train = 5 windows per subject at all context lengths (fixed). At 30s context there are ~960 possible windows; at 240m there are ~2. The model trained at 30s has seen a much smaller fraction of available windows per epoch.

**Risk**: a reviewer may argue the short-context model is data-hungry at train time and the comparison is unfair. The token-budget ablation (`sex_binary_lstm_kbudget`) exists in the registry but has not been run.

**Options:**
- (A) Run the ablation; include as supplementary if results are consistent
- (B) Acknowledge as a known limitation in Discussion; cite that AUROC at long contexts (240m) is similar to or better than at short contexts despite fewer training windows, suggesting K=5 is sufficient
- (C) Ignore it (not recommended — a strong reviewer will notice)

**Recommendation**: Option B for now; option A before camera-ready if a reviewer asks. **Your input needed.**

**Your answer**: No ablation run needed. The design choice is **K=5 fixed** and the justification is that this holds the *number of windows seen per subject per epoch constant* across all context lengths — ensuring that each model receives the same number of gradient signals per subject regardless of L. This is the fairest comparison from a training-exposure standpoint, and is the reason the token-budget approach (which changes K with L, confounding the two variables) was explicitly rejected. This justification already exists in `docs/context_length_experiment_design.md` §3 and §13, and in `configs/phase0_v3_config.yaml` (the `windows_strategy: "fixed"` comment). The paper will cite this rationale and acknowledge the fraction-of-available-windows asymmetry as a design trade-off, not a flaw. **No token-budget ablation runs required.**

### G4. Deferred tasks: include or exclude?
`insomnia_binary_lstm`, `rested_morning_lstm`, `anxiety_binary_lstm` all have AUROC ≤ 0.60 in phase0. We could:
- (A) Run them under v3 protocol, include in supplementary regardless of AUROC ("completeness")
- (B) Run them but include only if AUROC > 0.62; otherwise mention in Discussion as negative results
- (C) Skip entirely — they add cluster compute but not paper signal

**Recommendation**: Option B. Include in supplementary as "tasks where PSG provided no discriminative signal under any context length" — this is scientifically valid and shows the framework's limits. **Your call.**

**Your answer**: Probably exclude deferred tasks from the paper if v3 results remain poor (AUROC < 0.60). Decision deferred until experiments run. If any deferred task improves above ~0.62 with the v3 overlapping-window protocol, it will be included in supplementary as a near-chance negative result with brief discussion.

### G5. Tier 2 tasks: how prominently featured?
`depression_extreme_binary_lstm` (80m: 0.742, 120m: 0.750) and `osa_binary_apples_postqc_lstm` (40m: 0.738, 120m: 0.742) have decent AUROC but small N (~1.5–1.8k). Do they appear in Table I in the main paper, or only in supplementary?

**Recommendation**: Include in Table I with a footnote about small N and single-dataset limitation. The depression result (AUROC 0.75 from PSG alone, extreme-group design) is a compelling Tier 2 highlight. **Your input needed.**

**Your answer**: All tasks with decent results are included in the main text. The main results table (Table I) covers all tasks with AUROC ≥ ~0.62, regardless of tier. Per-task saturation figures for tasks beyond the 2–3 featured ones go in supplementary, but the summary numbers appear in the main paper. This keeps the breadth claim (15 tasks, 4 cohorts) fully supported by numbers in the main text without requiring a figure per task.

### G6. Page target: 8 or 10 pages?
JBHI regular papers are typically 8–10 double-column pages. 8 pages requires aggressive supplementary pushing; 10 pages allows slightly more results detail.

**Recommendation**: target 10 pages to accommodate the breadth (15 tasks, 6 contexts, 3 heads). Trim to 8 if reviewers request. **Confirm this is the right submission type.**

**Your answer**: **12–14 pages** (double-column). This is a full-length journal paper with substantial breadth; 10 pages would be too tight to cover all tasks, figures, and the iso-compute analysis properly. May reconsider if venue changes later.

### G7. Modality ablation experiment
SOTA_COMPARISON doc identifies the zero-out modality ablation (Experiment A) as Priority 1, moderate effort. The ablation (zero out 128-dim slices of the 512-dim embedding) answers the reviewer question "which modalities drive your results?" and directly addresses OSF's channel masking finding.

Running this adds ~12 training jobs (3 tasks × 2 context lengths × 4-5 modality conditions).

**Options:**
- (A) Run before submission — include as a main-paper ablation or supplementary
- (B) Mark as "future work" in Discussion
- (C) Run pilot on bmi_binary_lstm only to at least address the reviewer question

**Recommendation**: Option C (pilot on bmi_binary_lstm at 10m and 120m) before submission. This is low-risk and directly counters a predictable reviewer concern. **Your call.**

**Your answer**: Will run the modality zero-out ablation on **1–2 tasks** (e.g., apnea_binary and bmi_binary at representative context lengths). This directly addresses the expected OSF-related reviewer question. Results will go in supplementary if there is no space in main paper, or in main paper as a brief ablation subsection if the finding is notable. Running soon.

---

## H. Key Design Decisions Already Fixed (Do Not Change)

These are established by the existing codebase and docs — no open questions:

- **Metric hierarchy**: AUROC primary (threshold-free), balanced accuracy at t_opt secondary for imbalanced binary tasks, Cohen's κ for sleep staging
- **No raw accuracy** for any imbalanced task
- **Frozen encoder**: SleepFM `model_base` (SetTransformer, contrastive pre-training)
- **Patch size**: 5 seconds (640 samples at 128 Hz)
- **Embedding dim**: 512 (= 4 modalities × 128 dim per modality)
- **K_train = 5 windows** per subject (fixed, all context lengths)
- **Batch size = 32**, accum_steps = 1, at all context lengths
- **Cohort consistency filter**: min_recording_patches = 2880 (≥ 240 min required for inclusion)
- **Split seed = 42**, stratified by subject
- **Early stopping monitor**: val_auroc, patience = 10
- **Context lengths**: 30s, 10m, 40m, 80m, 120m, 240m (6 lengths)

---

## I. Summary of What Is Currently Available vs Pending

### Available now (v3 protocol, correct accum=1):

| Experiment | Contexts done | Best AUROC known |
|---|---|---|
| `bmi_binary_lstm` | 30s, 10m, 40m, 80m (4/6) | 0.767 @ 80m (K=all) |
| `sleep_efficiency_binary_lstm` | 30s, 10m, 40m, 80m (4/6) | ~0.770 @ 240m (old accum) |
| `sleep_efficiency_binary_transformer` | 30s, 10m, 40m (3/6) | ~0.797 @ 240m (old accum) |
| `psqi_binary_lstm` | 30s, 10m, 40m, 80m (4/6) | ~0.520 @ 120m |
| `depression_extreme_binary_lstm` | 80m, 120m (2/5) | 0.750 @ 120m |
| `osa_binary_apples_postqc_lstm` | 40m, 80m, 120m (3/5) | 0.742 @ 120m |

### Pending (Groups A–D from RERUN_CHECKLIST.md):

| Group | Key experiments | Context runs | Priority |
|---|---|---|---|
| A | bmi_lstm 120m; psqi 120m+240m; sleep_eff_lstm 120m+240m; sleep_eff_transformer 80m+120m+240m | 8 | Immediate |
| B | depression_extreme 30s+10m+40m; osa_postqc 30s+10m; osa_severity all 5 | 10 | Next |
| C | sex_binary (3 heads × 6), age_class (3 heads × 6), bmi_binary/sleep_eff mean_pool heads | 48 | High |
| D | apnea_binary (3 heads × 6), sleep_staging (3 heads × 6), cvd_binary, sleepiness_binary | 48 | High |
| E | insomnia, rested_morning, anxiety | 15 | After A–D |

**Paper is writable in full once Groups A–D are complete.** The primary paper figures (Fig 2, Fig 4, Fig 5) require Group C/D.

---

*Ready to proceed to LaTeX writing once you review this plan and answer the open questions in Section G.*
