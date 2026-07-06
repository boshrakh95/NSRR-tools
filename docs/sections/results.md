# Results Section — Planning Document

Reference for writing the Results section in the LaTeX paper. Records all available results,
pending items, figure/table placement decisions, and specific numbers. Update as experiments finish.

---

## Experiment completion status

| Experiment set | Status | Notes |
|---|---|---|
| Fast-channel seq2label (LSTM + Transformer) | **DONE** | All 6 contexts, 8 remaining tasks |
| Full-channel seq2label (LSTM + Transformer) | **DONE** | Same tasks and heads |
| Modality ablation (25 jobs) | **DONE** | 5 tasks × 5 conditions, as of 2026-06-17 |
| MeanPool head | **PENDING** | Not yet in analysis.csv; needed for H4 |
| Sleep staging (multi-dataset re-run) | **PENDING** | SHHS+MrOS+APPLES, val_kappa monitor |
| Bootstrap CIs (all tasks) | **PARTIAL** | Present for some rows; need full sweep |
| Iso-compute heatmap (dense K sweep) | **PENDING** | Requires `build_heatmap_df.py` run |
| K-grid table (all tasks) | **PARTIAL** | Only sex_binary_lstm done |
| Per-cohort breakdown (all tasks) | **PARTIAL** | Only sex_binary_lstm done |
| Token-budget sensitivity (Supplementary) | **PENDING** | Not yet run |

**Remaining tasks in paper (after dropping psqi, sleepiness, cvd, osa_severity):**

| Task | Heads | Note |
|---|---|---|
| sex_binary | LSTM, Transformer | MeanPool pending |
| age_class | LSTM, Transformer | MeanPool pending |
| apnea_binary | LSTM, Transformer | MeanPool pending |
| bmi_binary | LSTM, Transformer | MeanPool pending |
| sleep_efficiency_binary | LSTM, Transformer | MeanPool pending |
| sleep_staging | — | Full re-run pending |
| depression_extreme_binary | LSTM | LSTM only; small N (229 test) |
| osa_binary_apples_postqc | LSTM | LSTM only; small N (161 test) |

---

## Main paper / supplementary placement decisions

TBME limits: typically 8–10 display items in main text. Target: ≤6 figures, ≤5 tables for the
whole paper (Table I in Methods is the task definition table).

### Main paper figures

| # | Content | File (phase0_v3) | Status |
|---|---|---|---|
| Fig 1 | Pipeline overview | `fig1.png` (placeholder) | Placeholder |
| Fig 2 | Saturation curves — AUROC vs L, multi-panel for 6 primary tasks + sleep staging | `saturation/saturation_{task}_auroc_test.png` (one per task) | Exists for seq2label |
| Fig 3 | Cross-task context sensitivity — scatter or bar chart of ΔAUROC vs task | `task_comparison/task_comparison_6A_scatter.png` or `6B_bars.png` | **EXISTS** |
| Fig 4 | Modality ablation — grouped bar chart, AUROC drop per condition per task | Regenerate from Table 6 numbers | Generate from table |
| Fig 5 | Sleep staging κ vs L, per head + per-stage F1 at best L | PENDING | PENDING |

Note: Channel expansion (fast vs full) is a clear finding but space is tight. Present as a
supplementary figure unless sleep staging is still pending at submission — in that case promote
channel comparison to main Fig 5 and add sleep staging later.

> **[RENAME NOTE — do before final submission]**
> Inference K (post-hoc window count swept in results) and training K (=5, fixed in Methods)
> both use the letter K, causing reader confusion. Resolution: rename training K to **W**
> (or "w windows per subject per epoch" in prose). All figures/tables use K for inference —
> do NOT rename those. Only rename the training count in Methods §III-F and any table caption
> that says "training always used K=5 overlapping windows". Replace every such occurrence with
> "training used w=5 windows per subject per epoch" or "Training sampled w=5 overlapping
> windows per subject per epoch (Section III-F)."

### Main paper tables (Results section)

> **[TABLE REDESIGN — see TABLES_PLAN.md for full spec]**
> The table structure below reflects the NEW plan agreed 2026-07-04. See
> `docs/TABLES_PLAN.md` for the authoritative spec of each table's columns,
> numbers, placement, and supplementary counterparts.

| # | Content | File | Status |
|---|---|---|---|
| Table II | Saturation + aggregation (H1+H3 unified) | NEW — see TABLES_PLAN.md | IMPLEMENTED |
| Table III | L* per task + Δ from 30s baseline (H1 detail) | see TABLES_PLAN.md | IMPLEMENTED |
| Table IV | Head comparison at LSTM L* (H4) | see TABLES_PLAN.md | IMPLEMENTED |
| Table V | Modality ablation — AUROC per condition per task | `table6_modality.{md,tex}` | **EXISTS, complete** |

Channel comparison (fast vs full): fold into Table II as extra columns or present inline in text.
If space allows, add as Table VI; otherwise move to supplementary.

### Supplementary figures

| # | Content | File | Status |
|---|---|---|---|
| S-Fig 1 | Iso-compute heatmap (L × K grid) for sex_binary and apnea_binary | `{task}_lstm/auroc_test/heatmap_auroc.png` | Exists (phase0_v2); regenerate v3 |
| S-Fig 2 | K-sweep curves — AUROC vs K at each L, for sex_binary and sleep_efficiency | `{task}_{head}_test_window_sweep_auroc.png` | **EXISTS** |
| S-Fig 3 | Cohort saturation — per-dataset AUROC vs L for apnea_binary | `apnea_binary_lstm/apnea_binary_lstm_cohort_saturation_7A.png` | **EXISTS** |
| S-Fig 4 | Precision-recall curves vs context, for apnea_binary | `apnea_binary_lstm/apnea_binary_lstm_pr_8A_curves.png` | **EXISTS** |
| S-Fig 5 | Calibration (ECE vs context) for representative task | `{task}/calibration_2B_ece_vs_context.png` | **EXISTS** |
| S-Fig 6 | Channel comparison saturation overlays (fast vs full) for sex, apnea, sleep_eff | Generate from both analysis CSVs | Generate |
| S-Fig 7 | Sleep staging common evaluation set (240m-valid anchors only) | PENDING | PENDING |

Note: window_position (4A/4B), subject_consistency (5A/5B/5C), kstar (9A/9B) figures exist but
are lower priority; include only if page budget allows.

### Supplementary tables

| # | Content | File | Status |
|---|---|---|---|
| S-Table I | Excluded subjects (cohort filter) | Already in supplementary.tex | EXISTS |
| S-Table II | Per-task subject counts | Already in supplementary.tex | EXISTS |
| S-Table III | Bandpass filter details | Already in supplementary.tex | EXISTS |
| S-Table IV | K-grid: AUROC × K for sex_binary_lstm | `table3_kgrid_sex_binary_lstm_fast.{md,tex}` | **EXISTS** |
| S-Table V | Cross-task context sensitivity ranking | `table4_sensitivity_fast_lstm.{md,tex}` | PARTIAL (3 tasks only) |
| S-Table VI | Bootstrap CI summary (AUROC ± 95% CI per task at L*) | `table10_ci_fast.{md,tex}` | PARTIAL |
| S-Table VII | Cohort breakdown for sex_binary and apnea_binary at L* | `table9_cohort_sex_binary_lstm_fast.{md,tex}` | PARTIAL |
| S-Table VIII | Threshold tuning details (Δ balanced accuracy per task) | From POSTHOC_THRESHOLD_TUNING.md | Compute |
| S-Table IX | K=5 window sampling sensitivity (fixed K vs token-budget) | Pending experiment | PENDING |
| S-Table X | Sleep staging: common evaluation set comparison | Pending staging run | PENDING |

---

## IV-A. Context-Length Saturation (H1)

### Narrative
All six clinical prediction tasks show some improvement with longer context, but the magnitude
and the point of saturation vary substantially across tasks. Sleep efficiency prediction benefits
most from long context (L*=240m, ΔAUROC=+0.091 for LSTM, +0.124 for Transformer), consistent
with sleep efficiency encoding information distributed across the full night. Apnea prediction
requires up to 120 minutes (ΔAUROC=+0.074 LSTM, +0.104 Transformer). Sex and age classification
saturate earlier (LSTM L*=120m and 80m respectively). BMI shows minimal context benefit
(ΔAUROC=+0.006 LSTM). This heterogeneity supports task-specific context requirements rather
than a single universal temporal scale.

Depression (L*=10m) and OSA-APPLES (L*=40m) each have small test sets (229 and 161 subjects);
their non-monotonic long-context curves should be interpreted cautiously.

### Table II — Main performance table (fast-channel, test split)

Numbers: K=5 (clinical deployment, matching training) and K=all (full-night ceiling).

| Task | N_test | Head | Best L | AUROC@K=5 | AUROC@K=all |
|---|---|---|---|---|---|
| sex_binary | 1433 | LSTM | 120m | 0.872 | 0.872 |
| sex_binary | 1433 | Transformer | 120m | 0.905 | 0.910 |
| sex_binary | 1433 | MeanPool | — | PENDING | PENDING |
| age_class | ~1860 | LSTM | 120m | 0.893 | 0.893 |
| age_class | ~1860 | Transformer | 120m | 0.902 | 0.905 |
| age_class | ~1860 | MeanPool | — | PENDING | PENDING |
| apnea_binary | 2054 | LSTM | 120m | 0.831 | 0.832 |
| apnea_binary | 2054 | Transformer | 120m | 0.856 | 0.857 |
| apnea_binary | 2054 | MeanPool | — | PENDING | PENDING |
| bmi_binary | 1856 | LSTM | 80m | 0.763 | 0.767 |
| bmi_binary | 1856 | Transformer | 80m | 0.767 | 0.777 |
| bmi_binary | 1856 | MeanPool | — | PENDING | PENDING |
| sleep_efficiency_binary | 2023 | LSTM | 240m | 0.788 | 0.788 |
| sleep_efficiency_binary | 2023 | Transformer | 240m | 0.831 | 0.831 |
| sleep_efficiency_binary | 2023 | MeanPool | — | PENDING | PENDING |
| sleep_staging | ~15000 | LSTM | — | PENDING (κ) | PENDING |
| sleep_staging | ~15000 | Transformer | — | PENDING (κ) | PENDING |
| depression_extreme_binary | 229 | LSTM | 10m | 0.776 | 0.770 |
| osa_binary_apples_postqc | 161 | LSTM | 10m | 0.823 | 0.834 |

For channel comparison columns (fast vs full): add as separate column pair, or fold into a
footnote if space is tight. Full-channel best AUROCs for key tasks:
- apnea_binary/Transformer: 0.857 → 0.901 (+0.044)
- sex_binary/Transformer: 0.910 → 0.929 (+0.019)
- bmi_binary/Transformer: 0.777 → 0.816 (+0.039)

### Table III — Saturation threshold L* per task/head

| Task | LSTM L* | LSTM ΔAUROC | Transformer L* | Transformer ΔAUROC |
|---|---|---|---|---|
| sex_binary | 120m | +0.047 | 240m | +0.078 |
| age_class | 80m | +0.028 | 120m | +0.051 |
| apnea_binary | 120m | +0.074 | 120m | +0.104 |
| bmi_binary | 10m | +0.006 | 240m | +0.030 |
| sleep_efficiency_binary | 240m | +0.091 | 240m | +0.124 |
| depression_extreme_binary | 10m | +0.013 | — | — |
| osa_binary_apples_postqc | 40m | +0.065 | — | — |
| sleep_staging | PENDING | — | PENDING | — |

ΔAUROC = K=all AUROC at best L minus K=all AUROC at 30s. L* = smallest L within 0.005 of peak.

Key message: L* ranges from 10m to 240m. Physiological tasks (sleep efficiency, apnea) require
long context; demographic/metabolic tasks (sex, BMI) saturate earlier.

### Figure 2 — Saturation curves

Multi-panel: one subplot per task (2×3 grid for 6 primary tasks; + sleep staging when available).
X-axis: context length on log scale (0.5m to 240m). Y-axis: AUROC, starting at 0.5.
Lines: LSTM (solid), Transformer (dashed), MeanPool (dotted, pending). Error bands from CI.
Mark L* with a vertical dashed line per head.

Source files: `results/figures/phase0_v3/saturation/saturation_{task}_auroc_test.png`
(individual; may need to regenerate as combined multi-panel with consistent axes).

Tasks to include: sex_binary, age_class, apnea_binary, bmi_binary, sleep_efficiency_binary,
sleep_staging (placeholder panel until run completes).

### Figure 3 — Cross-task context sensitivity

Source: `results/figures/phase0_v3/task_comparison/task_comparison_6B_bars.png` (bar chart)
OR `task_comparison_6A_scatter.png` (scatter: ΔAUROC vs AUROC@30s as proxy for task difficulty).

The scatter is more informative for the paper narrative — shows no correlation between baseline
task difficulty and context sensitivity. Use `6A_scatter.png` as primary; `6B_bars.png` as
supplementary alternative.

Note: figures include dropped tasks (psqi, sleepiness, cvd). Need to regenerate after removing
those tasks, or annotate on the figure which ones are excluded.

---

## IV-B. Head Architecture Comparison (H4)

### Status
LSTM vs Transformer: available for all tasks. MeanPool: PENDING.
Write subsection now with LSTM/Transformer; add MeanPool placeholder.

### Table IV — Head comparison at L* (each task's best context)

| Task | Context | LSTM@K=5 | Transformer@K=5 | MeanPool@K=5 | Temporal advantage† |
|---|---|---|---|---|---|
| sex_binary | 120m | 0.872 | 0.905 | PENDING | +0.033 (T-L) |
| age_class | 120m | 0.893 | 0.902 | PENDING | +0.009 |
| apnea_binary | 120m | 0.831 | 0.856 | PENDING | +0.025 |
| bmi_binary | 80m | 0.763 | 0.767 | PENDING | +0.004 |
| sleep_efficiency | 240m | 0.788 | 0.831 | PENDING | +0.043 |
| sleep_staging | — | PENDING κ | PENDING κ | PENDING κ | — |

†Transformer minus LSTM at the same context. MeanPool comparison needed for full H4 test.

Key finding (partial): Transformer consistently outperforms LSTM. Advantage is largest for tasks
with strong signal and long optimal context (sex +0.033, sleep_efficiency +0.032, apnea +0.025).
Cannot fully test H4 until MeanPool results are available.

---

## IV-C. Aggregation Analysis (H2, H3)

### K-sweep findings (H3 — aggregation saturation)

Data available for sex_binary_lstm (S-Table IV):

| L | K=1 | K=5 | K=10 | K=20 | K=all |
|---|---|---|---|---|---|
| 30s | 0.687 | 0.801 | 0.817 | 0.822 | 0.825 |
| 10m | 0.724 | 0.831 | 0.842 | 0.850 | 0.850 |
| 40m | 0.750 | 0.838 | 0.843 | 0.845 | 0.845 |
| 80m | 0.785 | 0.859 | 0.861 | — | 0.861 |
| 120m | 0.807 | 0.872 | — | — | 0.872 |
| 240m | 0.844 | — | — | — | 0.857 |

At medium-to-long contexts (≥40m), K=5 captures ≥99% of K=all benefit. At short contexts (30s),
even K=all (0.825) does not reach K=5 at 80m (0.859), demonstrating that aggregation cannot
substitute for longer context beyond a ceiling.

### Iso-compute analysis (H2 — aggregation substitution)

Full dense K sweep PENDING. Sparse reading from analysis.csv for sex_binary/LSTM at 80-min
budget: L=80m,K=1 → AUROC=0.798; L=10m,K≈8 → AUROC≈0.833. Preliminary: aggregating short
windows marginally outperforms a single long window at matched budget, but the difference is
small. Full heatmap analysis needed for definitive claim.

Report this analysis in text with caveat that iso-compute figure is pending. S-Fig 1 (heatmap)
will be supplementary when available.

---

## IV-D. Channel Count Expansion

### Numbers (fast vs full, K=all, best context per task/head)

| Task | Head | Fast AUROC | Full AUROC | Δ |
|---|---|---|---|---|
| apnea_binary | Transformer | 0.857 | 0.901 | **+0.044** |
| apnea_binary | LSTM | 0.832 | 0.874 | **+0.042** |
| bmi_binary | Transformer | 0.777 | 0.816 | **+0.039** |
| bmi_binary | LSTM | 0.767 | 0.802 | **+0.036** |
| sex_binary | LSTM | 0.873 | 0.906 | **+0.033** |
| sex_binary | Transformer | 0.910 | 0.929 | +0.019 |
| sleep_efficiency | LSTM | 0.788 | 0.810 | +0.022 |
| sleep_efficiency | Transformer | 0.831 | 0.825 | −0.006 |
| age_class | LSTM | 0.893 | 0.901 | +0.008 |
| age_class | Transformer | 0.905 | 0.911 | +0.006 |
| depression_extreme | LSTM | 0.770 | 0.752 | −0.018 |
| osa_binary_apples_postqc | LSTM | 0.834 | 0.772 | −0.062 ‡ |

‡ OSA-APPLES full-channel drop: likely small-N artefact (161 test subjects). Include in table
with a comment mark and footnote: "Full-channel performance for this task (N=161) should be
interpreted cautiously; the result is sensitive to sampling variability." — per user decision to
include now but flag for possible exclusion later.

Key finding: Full channel (up to 23 channels) improves AUROC for tasks with rich physiological
signal (apnea +0.042–0.044, BMI +0.036–0.039, sex +0.019–0.033). Gains diminish for tasks with
low inherent predictability from PSG (age +0.006–0.008). Small-N tasks show inconsistent results.

Integrate into Table II as extra columns, or present as inline numbers in the text (more readable).

---

## IV-E. Modality Group Ablation

### Table V — Modality ablation (Table 6, complete, 25/25 conditions)

Fast-channel LSTM, at task-specific context (120m for sex/apnea/sleep_eff/age; 40m for BMI):

| Task | Full | No BAS (Δ) | No RESP (Δ) | No EKG (Δ) | Cardio only (Δ) | BAS only (Δ) |
|---|---|---|---|---|---|---|
| sex_binary | 0.872 | 0.799 (−0.073) | 0.843 (−0.029) | 0.782 (−0.090) | 0.796 (−0.076) | 0.777 (−0.096) |
| apnea_binary | 0.832 | 0.782 (−0.050) | 0.766 (−0.066) | 0.808 (−0.024) | 0.770 (−0.062) | 0.723 (−0.109) |
| sleep_efficiency | 0.780 | 0.682 (−0.099) | 0.760 (−0.021) | 0.750 (−0.030) | 0.670 (−0.110) | 0.786 (+0.005) |
| age_class | 0.893 | 0.835 (−0.059) | 0.882 (−0.012) | 0.868 (−0.025) | 0.821 (−0.072) | 0.860 (−0.034) |
| bmi_binary | 0.756 | 0.728 (−0.028) | 0.759 (+0.003) | 0.758 (+0.002) | 0.663 (−0.093) | 0.741 (−0.015) |

Note: sleep_efficiency full-channel AUROC at 120m was 0.780 (fast-channel, used as ablation
baseline); full-channel value at 120m is 0.810. Modality ablation uses fast-channel baseline
for consistency with the ablation conditions.

### Task-by-task key findings

**Apnea:** RESP most necessary (−0.066) > No BAS (−0.050) > No EKG (−0.024). BAS-only is worst
(−0.109). Replicates OSF finding: respiratory channels are necessary for OSA detection.
RESP alone is not sufficient (cardio-only = 0.770 with RESP+EKG). Neural activity contributes
via arousal-related EEG fragmentation.

**Sleep efficiency:** BAS dominates: No BAS largest drop (−0.099). BAS-only ≈ Full (−0.005,
noise level). Three-way convergence confirms EEG/EOG alone explains sleep efficiency, consistent
with its grounding in sleep staging. Cardio-only worst (−0.110).

**Sex (unexpected finding):** EKG is the most necessary single modality: No EKG drops −0.090,
larger than No BAS (−0.073). Cardiac signal (HRV, heart rate, sex differences in cardiac
electrophysiology) carries more sex-discriminative information than EEG in the SleepFM embedding
space. BAS-only is worst overall (−0.096). This was not anticipated and is a notable finding.

**Age:** BAS most necessary (−0.059) → EKG (−0.025) → RESP (−0.012, borderline noise).
Physiological aging concentrated in EEG changes and cardiac function; respiration adds less.

**BMI:** No single modality clearly necessary (No RESP +0.003, No EKG +0.002 are noise-level;
No BAS −0.028 borderline). Cardio-only (RESP+EKG) drops by −0.093 despite neither RESP nor EKG
alone being necessary — suggests BAS×EMG interaction (or EMG's anti-noise role). Unclear;
acknowledge in discussion.

Caveat: Single training run per condition; |Δ| < 0.02 is noise-level. |Δ| > 0.05 very likely
real at N_test 1430–2054.

---

## IV-F. Sleep Staging

**STATUS: PENDING.** No paper-ready results.

Primary run: SHHS+MrOS+APPLES, centered windows, complete_only, hidden=256, num_layers=2,
val_kappa. STAGES excluded.

What to report when results arrive:
1. Cohen's κ vs L for LSTM and Transformer (primary figure — Fig 5)
2. Per-stage F1 at best L: W, N1, N2, N3, REM. N1 expected lowest.
3. Note STAGES exclusion (footnote already in Table I).
4. Note that at 240m context, ~50% of anchors excluded by complete_only policy.
5. Supplementary: comparison run with STAGES (expected lower κ).
6. Supplementary: common evaluation set (240m-valid anchors only, S-Table X).

Placeholder text: "Sleep staging results will be reported upon completion of the multi-cohort
training run. Preliminary single-cohort analyses suggest performance saturates around 10–40 min
of context for κ, consistent with human scorers using ±1–2 min of surrounding context."

---

## Numbers NOT to report (dropped tasks — reference only)

| Task | Peak AUROC | Note |
|---|---|---|
| psqi_binary | 0.557 | Near chance; no context benefit |
| sleepiness_binary | 0.629 | Flat across all L; no PSG signal |
| cvd_binary | 0.689 | Weak and flat |

These may appear in Discussion as evidence that some endpoints are not predictable from overnight
PSG under any context length — label noise or PSG-label mismatch suspected.

---

## Threshold tuning notes (for table captions and S-Table VIII)

Tasks where t* (val-selected threshold) is used for balanced accuracy reporting:
- **osa_binary_apples_postqc**: CRITICAL. Default t=0.5 gives BA≈0.55; tuned BA improves by
  up to +0.220 at 10m context. AUROC unchanged. Must note in table caption.
- **depression_extreme_binary**: +0.065 improvement at 80m; use tuned for consistency.
- **bmi_binary**: modest gain (+0.013–0.027); include for consistency.
- **sex_binary/LSTM**: small (+0.009); include for consistency.

Tasks where default t=0.5 is retained (tuning does not help):
- apnea_binary (both heads): near zero or negative gain
- sleep_efficiency_binary (both heads): near zero gain
- age_class (both heads): multi-class task, threshold concept differs
- sex_binary/Transformer: marginal
