# Supplementary Material — Planning Document

This document records the full planned content of supplementary.tex. It covers:
1. Content already written in supplementary.tex (datasets, preprocessing, existing tables)
2. New results sections decided for supplementary placement
3. Notes on what still needs to be added when pending experiments finish

When asked to write the actual supplementary.tex, use this document as the reference.

---

## Current content in supplementary.tex (already written)

### Section S-I: PSG Datasets — Extended Description

Four subsections fully written:

**S-I.A SHHS** — SHHS1 (~6441 subjects, 1995–1998, age 40–98, 53% female) and SHHS2 (~3295
re-exams). In-home Compumedics devices. SHHS1 scored R&K; SHHS2 scored AASM 2007. Labels: AHI,
ESS, CVD (any_cvd composite), BMI, age, sex. Exclusions: 1 subject recording <240 min.

**S-I.B MrOS** — Visit 1 (~2911 subjects) and Visit 2 (~2560). All male. In-home VITAPORT-3.
AASM 2007. Labels: AHI, PSQI, CVD (cvchd, CHD only — narrower than SHHS), BMI, age. ISI at V2
only; excluded from insomnia task.

**S-I.C APPLES** — 1516 enrolled; 1103 usable (412 withdrawn or missing PSG, 1 NaN embedding).
In-laboratory full montage. AASM 2007. Labels: AHI, BDI, ESS, BMI, age, sex,
clinician-adjudicated OSA severity (osa_binary_apples_postqc).

**S-I.D STAGES** — ~1350 usable (152 STLK subjects excluded for NaN embeddings from HDF5
conversion issue; blocklisted). In-laboratory. AASM 2012. Labels: PHQ-9, GAD-7, ISI, ESS, AHI
(from separate clinical XLSX matched on s_code), sex, age. STAGES excluded from sleep staging
(recordings ~10× longer; contributes >54% of training items from 10% of subjects; preliminary
kappa: 0.62 without STAGES vs 0.55 with STAGES).

**S-I.E Cohort Consistency Filter Table** (tab:supp-excluded)
- APPLES: 19 excluded, 5–230 min
- SHHS: 1 excluded, 180 min
- MrOS: 0 excluded
- STAGES: 0 excluded
- Total: 20 subjects across primary tasks (≤0.21%)

**S-I.F Per-Task Subject Counts Table** (tab:supp-task-n)
Current table has Tier 1/Tier 2 labelling and includes dropped tasks (psqi, sleepiness, cvd,
osa_severity). **NEEDS UPDATE** to reflect final task list after removals:
- Keep: sex_binary, sleep_efficiency_binary, bmi_binary, age_class, apnea_binary, sleep_staging,
  depression_extreme_binary, osa_binary_apples_postqc
- Remove rows: psqi_binary, osa_severity_apples, cvd_binary, sleepiness_binary
- Remove Tier labels from the table

---

### Section S-II: Signal Preprocessing Pipeline — Extended Details

Four subsections fully written:

**S-II.A Channel Mapping** — case-insensitive matching against priority-ordered list;
e.g., C3M2/EEG C3-A2/C3-A2 → C3-M2. Zero-padding for absent channels with boolean mask.
NOTE: includes specific config filename (channel_definitions.yaml) and texttt formatting —
acceptable in supplementary (more technical context).

**S-II.B Bandpass Filtering Details** — FIR via MNE filter_data (firwin, FFT convolution);
Butterworth fallback if MNE fails. Table (tab:supp-filters) with cutoffs and rationale:
- EEG/EOG: 0.3–35 Hz
- EKG/ECG: 0.5–45 Hz
- EMG: 10–100 Hz
- Respiratory: 0.05–2 Hz

**S-II.C Resampling Strategy** — resample_poly for integer ratios, np.interp for non-integer.
Minor length mismatches corrected by truncation or single-sample edge-padding.

**S-II.D Z-Score Normalisation** — per-channel over full recording. Flat channels: mean-centred,
unit scale. NaN→0, Inf→±10.

**S-II.E HDF5 Storage Format** — float16, chunked at 38400 samples (5 min at 128 Hz), gzip level 4.
Attributes store: sampling rate, duration, channel names, original fs, per-channel mean/std/min/max.

**S-II.F Sleep Stage Annotation Processing** — NSRR XML (SHHS, MrOS), CSV (STAGES),
tab-separated .annot (APPLES). Discrepancy >2 epochs: truncate or pad with unknown-label.
Label mapping: stage 5 → 4 (REM); SHHS1 N4 → N3. Unknown/artefact (−1) excluded.

---

### Section S-III: K=5 Window Sampling — Sensitivity Analysis

**STATUS: PENDING** — token-budget ablation experiment not yet run.

Placeholder: [PENDING — token-budget sensitivity experiment results]

Content to add when done:
- Saturation curve comparison for sex_binary (or one other task): K=5-fixed vs token-budget
- Expected finding: curves diverge only at 30s and 10m contexts; converge at ≥40m
- Numbers from make_table4_sensitivity.py output after token-budget experiment
- Reference: TRAINING_PROTOCOL_FIXES.md Issue 3 for rationale

---

### Section S-IV: Additional Results

**STATUS: MOSTLY PENDING** — placeholder in supplementary.tex

---

## New sections to add to supplementary.tex

### Section S-IV: Aggregation Saturation and Iso-Compute Analysis

#### S-IV.A K-Window Sweep Table (S-Table IV)

Source: `results/tables/table3_kgrid_sex_binary_lstm_fast.{csv,md,tex}`

Content (sex_binary/LSTM, fast-channel, test split):

| Context L | K=1 | K=5 | K=10 | K=20 | K=50 | K=all |
|---|---|---|---|---|---|---|
| 30s | 0.701 | 0.805 | 0.815 | 0.821 | 0.823 | 0.824 |
| 10m | 0.720 | 0.823 | 0.836 | 0.840 | 0.842 | 0.842 |
| 40m | 0.760 | 0.853 | 0.862 | 0.866 | — | 0.866 |
| 80m | 0.798 | 0.866 | 0.868 | — | — | 0.868 |
| 120m | 0.822 | 0.855 | — | — | — | 0.855 |
| 240m | 0.791 | — | — | — | — | 0.812 |

Caption note: "—" indicates fewer than K non-overlapping windows available for typical subjects
at that context length. All values are AUROC (mean-probability aggregation, test split).

When K-grid tables for other tasks become available (after dense K sweep on cluster), add apnea
and sleep_efficiency tables as well.

#### S-IV.B Iso-Compute Analysis Figure (S-Fig 1)

Source: `results/figures/phase0_v3/{task}_{head}/auroc_test/heatmap_auroc.png`
(phase0_v2 versions exist; phase0_v3 version requires dense K sweep to be run)

Content: 2D AUROC heatmap (L × K grid), iso-compute contour lines.
Show for sex_binary_lstm as primary example. Caption should explain iso-compute diagonal.

STATUS: Use phase0_v2 heatmap now as illustration; regenerate v3 version after dense K sweep.

#### S-IV.C K-Sweep Curves (S-Fig 2)

Source: `results/figures/phase0_v3/{task}_{head}/{task}_{head}_test_window_sweep_auroc.png`

Show curves for sex_binary_lstm and apnea_binary_lstm: AUROC vs K for each context length.
Demonstrates H3 (aggregation saturation: diminishing returns beyond K≈5).

---

### Section S-V: Cross-Task Context Sensitivity

#### S-V.A Sensitivity Ranking Table (S-Table V)

Source: `results/tables/table4_sensitivity_fast_lstm.{csv,md,tex}` (currently 3 tasks; extend
to all 8 tasks after make_table4_sensitivity.py is run with the updated task list)

Full table (LSTM, fast-channel, K=all):

| Task | AUROC@30s | Best AUROC | ΔAUROC | L* | Category |
|---|---|---|---|---|---|
| sleep_efficiency_binary | 0.697 | 0.799 | +0.102 | 240m | Sleep quality |
| apnea_binary | 0.758 | 0.832 | +0.074 | 120m | Respiratory |
| osa_binary_apples_postqc | 0.769 | 0.834 | +0.065 | 40m | Respiratory |
| sex_binary | 0.825 | 0.872 | +0.047 | 80m | Demographics |
| age_class | 0.865 | 0.893 | +0.028 | 40m | Demographics |
| depression_extreme_binary | 0.757 | 0.770 | +0.013 | 10m | Mental health |
| bmi_binary | 0.760 | 0.767 | +0.006 | 10m | Metabolic |

Note: dropped tasks for reference (not reported in Results):
- cvd_binary: ΔAUROC=+0.009 (near zero), sleepiness_binary: 0.000, psqi_binary: 0.000

#### S-V.B Task Comparison Figure (S-Fig 3 or main Fig 3)

Source: `results/figures/phase0_v3/task_comparison/task_comparison_6A_scatter.png`
(scatter: AUROC@30s on x-axis, ΔAUROC on y-axis, one point per task)

NOTE: this figure includes dropped tasks. Needs to be regenerated with updated task list,
or simply annotate dropped tasks as excluded. Ask user whether to regenerate or annotate.

---

### Section S-VI: Statistical Confidence Intervals

#### S-VI.A Bootstrap CI Table (S-Table VI)

Source: `results/tables/table10_ci_fast.{csv,md,tex}` (partially populated)

Content when CIs are fully computed (run `analyze --bootstrap 1000` for all tasks):

| Task | Head | L* | AUROC@K=all [95% CI] | AUROC@K=5 [95% CI] |
|---|---|---|---|---|
| sex_binary | LSTM | 80m | 0.872 [CI pending] | 0.872 [CI pending] |
| sex_binary | Transformer | 240m | 0.910 [0.894–0.925] | 0.905 [CI pending] |
| apnea_binary | LSTM | 120m | 0.832 [CI pending] | 0.831 [CI pending] |
| … | | | | |

Only sex_binary/Transformer K=all is currently populated with a CI (0.910 [0.894–0.925]).
All other CIs are pending the bootstrap run.

Run command: `bash scripts/run_analysis.sh [all tasks/heads] --bootstrap 1000 --analyze-only`

---

### Section S-VII: Per-Cohort Breakdown

#### S-VII.A Cohort Saturation Table (S-Table VII)

Source: `results/tables/table9_cohort_sex_binary_lstm_fast.{csv,md,tex}` (sex_binary_lstm only)

Current data:
| Task | Head | Context | Dataset | N | AUROC |
|---|---|---|---|---|---|
| sex_binary | LSTM | 80m | Overall | 1430 | 0.861 |
| sex_binary | LSTM | 80m | APPLES | 166 | 0.782 |
| sex_binary | LSTM | 80m | SHHS | 1264 | 0.868 |

When additional per-cohort tables are generated: add sex_binary_transformer, apnea_binary_lstm,
apnea_binary_transformer (apnea is the most clinically important; cohort breakdown is most
relevant there since cohorts differ in OSA severity enrichment: APPLES is OSA-enriched, SHHS
is community-based, MrOS is older males).

#### S-VII.B Cohort Saturation Figure (S-Fig 4)

Source: `results/figures/phase0_v3/apnea_binary_lstm/apnea_binary_lstm_cohort_saturation_7A.png`

Shows AUROC vs L broken out by dataset for apnea_binary_lstm. Demonstrates that context benefit
is consistent across cohorts (robustness check), or reveals cohort-specific patterns.

---

### Section S-VIII: Post-Hoc Threshold Tuning

#### S-VIII.A Threshold Tuning Results (S-Table VIII)

Source: results from POSTHOC_THRESHOLD_TUNING.md and analysis.csv

Table to build showing, for each binary task and head, the val-selected threshold t* and the
balanced accuracy improvement at the best context:

| Task | Head | Default BA@t=0.5 | Val-selected t* | Tuned BA@t* | Δ BA | Note |
|---|---|---|---|---|---|---|
| osa_binary_apples_postqc | LSTM | ~0.55 (10m) | val-selected | 0.774 (10m) | **+0.220** | Critical; default non-informative |
| depression_extreme_binary | LSTM | varies | val-selected | — | +0.065 (80m) | Use tuned throughout |
| bmi_binary | Transformer | — | val-selected | — | +0.027 (30s) | Modest |
| bmi_binary | LSTM | — | val-selected | — | +0.015 | Modest |
| sex_binary | LSTM | — | val-selected | — | +0.009 | Small |
| apnea_binary | LSTM | — | t=0.5 retained | — | ~0 | No benefit |
| sleep_efficiency | LSTM | — | t=0.5 retained | — | ~0 | No benefit |

Note for paper: AUROC is always reported at t=0.5 (threshold-free). Balanced accuracy at t* is
the secondary metric only. The large t* improvement for osa_binary_apples_postqc reflects the
extreme class imbalance in that subset; AUROC is not affected.

---

### Section S-IX: Calibration Analysis

#### S-IX.A ECE vs Context Figure (S-Fig 5)

Source: `results/figures/phase0_v3/{task}_{head}/{task}_calibration_2B_ece_vs_context.png`

Show for apnea_binary_lstm and sex_binary_lstm. Expected calibration (ECE) to decrease with
longer context — more confident and better calibrated predictions when more context is available.

This section can be brief: one figure + 1–2 sentences noting that calibration improves with
longer context, and ECE flattens near L*. Useful for discussion of deployment reliability.

---

### Section S-X: Sleep Staging Extended Analysis (PENDING)

**STATUS: PENDING** — waiting for multi-dataset staging run to complete.

Content to add when staging results are available:

1. **Common evaluation set comparison**: All context lengths evaluated on the same epoch subset
   (anchors valid for the 240m window, ~50% of all epochs). This removes the confound that longer
   contexts have fewer valid anchors under complete_only policy.

2. **STAGES inclusion comparison**: κ with STAGES in training (expected ~0.55 at 10m) vs without
   STAGES (expected ~0.62 at 10m). Demonstrates the training data domination problem.

3. **Per-stage F1 table**: Full W/N1/N2/N3/REM F1 at best context for LSTM and Transformer.

4. **Confusion matrix**: Stage confusion at best context for LSTM. Expected: N1 most confused
   with Wake; N3 best separated.

---

### Section S-XI: Channel Count Comparison Extended

#### S-XI.A Fast vs Full Saturation Overlays (S-Fig 6)

Content: For sex_binary, apnea_binary, and bmi_binary — saturation curves overlaid, fast-channel
vs full-channel (dashed vs solid), LSTM. Shows that full-channel models saturate at shorter L
for some tasks (particularly sex_binary where L* shifts from 120m to ~40m).

Source: Generate from both `results/collected/phase0_v3/analysis.csv` and
`results/collected/phase0_v3_full/analysis.csv` using plot_saturation.py with both datasets.

---

## Updates needed in existing supplementary.tex content

1. **S-I.F Per-Task Subject Counts table**: Remove dropped tasks (psqi, sleepiness, cvd,
   osa_severity), remove Tier labels. Update to final 8-task list.

2. **S-III K=5 Sensitivity**: Fill in after token-budget experiment.

3. **S-IV Additional Results placeholder**: Replace with actual sections from this document.

4. **Table/figure reference cross-checks**: When main paper tables get final numbers,
   ensure any supplementary cross-references (e.g., "see Section IV for AUROC") are consistent.

---

## Full planned supplementary section outline

```
S-I    PSG Datasets — Extended Description
  S-I.A  SHHS
  S-I.B  MrOS
  S-I.C  APPLES
  S-I.D  STAGES
  S-I.E  Cohort Consistency Filter (S-Table I)
  S-I.F  Per-Task Subject Counts (S-Table II) [UPDATE: remove dropped tasks]

S-II   Signal Preprocessing — Extended Details
  S-II.A Channel Mapping
  S-II.B Bandpass Filtering (S-Table III)
  S-II.C Resampling
  S-II.D Z-Score Normalisation
  S-II.E HDF5 Storage
  S-II.F Sleep Stage Annotation Processing

S-III  K=5 Window Sampling Sensitivity [PENDING]

S-IV   Aggregation Saturation and Iso-Compute Analysis
  S-IV.A K-Window Sweep Table (S-Table IV) [EXISTS]
  S-IV.B Iso-Compute Heatmap (S-Fig 1) [PENDING dense K sweep]
  S-IV.C K-Sweep Curves (S-Fig 2) [EXISTS]

S-V    Cross-Task Context Sensitivity
  S-V.A  Sensitivity Ranking Table (S-Table V) [PARTIAL]
  S-V.B  Task Comparison Scatter (S-Fig 3) [EXISTS, needs task list update]

S-VI   Statistical Confidence Intervals
  S-VI.A Bootstrap CI Table (S-Table VI) [PARTIAL]

S-VII  Per-Cohort Breakdown
  S-VII.A Cohort Table (S-Table VII) [PARTIAL: sex_binary_lstm only]
  S-VII.B Cohort Saturation Figure (S-Fig 4) [EXISTS: apnea_binary_lstm]

S-VIII Post-Hoc Threshold Tuning
  S-VIII.A Threshold Tuning Table (S-Table VIII) [compute from POSTHOC doc]

S-IX   Calibration Analysis
  S-IX.A ECE vs Context Figure (S-Fig 5) [EXISTS]

S-X    Sleep Staging Extended Analysis [PENDING]
  S-X.A  Common Evaluation Set Comparison
  S-X.B  STAGES Inclusion Comparison
  S-X.C  Per-Stage F1 Table
  S-X.D  Confusion Matrix

S-XI   Channel Count Comparison Extended
  S-XI.A Fast vs Full Saturation Overlays (S-Fig 6) [generate]
```
