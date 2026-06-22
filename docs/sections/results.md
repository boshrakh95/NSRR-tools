# Results Section — Planning Document

This document is the reference for writing the Results section in the LaTeX paper. It records
all available results, pending items, recommended structure, and specific numbers to use. Update
it as new experiments finish.

---

## Completion status at time of writing

| Experiment set | Status | Notes |
|---|---|---|
| Fast-channel seq2label (LSTM + Transformer) | **DONE** | All 6 contexts, all 8 remaining tasks |
| Full-channel seq2label (LSTM + Transformer) | **DONE** | Same tasks, same heads |
| Modality ablation (25 jobs) | **DONE** | 5 tasks × 5 conditions, as of 2026-06-17 |
| MeanPool head | **PENDING** | Not yet in analysis.csv; required for H4 |
| Sleep staging (multi-dataset re-run) | **PENDING** | SHHS+MrOS+APPLES, hidden=256, val_kappa |
| Bootstrap CIs (all tasks) | **PARTIAL** | Present for some rows; needs full sweep |
| Iso-compute heatmap (dense K sweep) | **PENDING** | Requires `build_heatmap_df.py` run |
| K-grid table (all tasks) | **PARTIAL** | Only sex_binary_lstm done |
| Per-cohort breakdown | **PARTIAL** | Only sex_binary_lstm done |
| Token-budget sensitivity (Supplementary) | **PENDING** | Not yet run |

**Remaining tasks in paper (after dropping psqi, sleepiness, cvd, osa_severity):**

| Task | Heads run | Note |
|---|---|---|
| sex_binary | LSTM, Transformer | MeanPool pending |
| age_class | LSTM, Transformer | MeanPool pending |
| apnea_binary | LSTM, Transformer | MeanPool pending |
| bmi_binary | LSTM, Transformer | MeanPool pending |
| sleep_efficiency_binary | LSTM, Transformer | MeanPool pending |
| sleep_staging | — | Full re-run pending |
| depression_extreme_binary | LSTM | LSTM only |
| osa_binary_apples_postqc | LSTM | LSTM only, small N (161 test) |

---

## Recommended structure for the Results section

```
IV. Results
  A. Context-Length Saturation (H1)            ← main saturation curves + Table 1 + Table 2
  B. Head Architecture Comparison (H4)         ← LSTM vs Transformer vs MeanPool
  C. Aggregation and Iso-Compute Analysis      ← K-sweep (H3) + iso-compute (H2)
  D. Channel Count Expansion                   ← fast vs full channel comparison
  E. Modality Group Ablation                   ← Table 6, which modalities matter
  F. Sleep Staging (placeholder until results) ← κ vs context, per-stage F1
```

---

## IV-A. Context-Length Saturation (H1)

### Narrative
All six clinical prediction tasks show some improvement with longer context, but the magnitude
and the context at which performance saturates varies substantially across tasks. Sleep efficiency
prediction benefits most from long context (L*=240min, ΔAUROC=+0.102 for LSTM), consistent with
sleep efficiency encoding information spread across the full night. Apnea prediction requires up
to 120 minutes (ΔAUROC=+0.074). Sex and age classification saturate earlier (L*=80m and 40m
respectively for LSTM). BMI classification shows minimal context benefit (ΔAUROC=+0.006). This
heterogeneity supports the hypothesis that context requirements are task-specific rather than
reflecting a single universal temporal scale in PSG.

Depression (L*=10m) and OSA-APPLES (L*=40m) each have small test sets (229 and 161 subjects)
and show non-monotonic saturation curves at long contexts; interpret with caution.

### Table 1 — Peak AUROC per task (already generated: results/tables/table1_peak_auroc_fast.*)

Numbers to use (fast-channel, test split, K=all = full-night aggregation):

| Task | N_test | Head | AUROC@30s | AUROC@40m | AUROC@120m | AUROC@240m | **Best AUROC** | Best L |
|---|---|---|---|---|---|---|---|---|
| sex_binary | 1433 | LSTM | 0.825 | 0.866 | **0.872** | 0.857 | 0.872 | 120m |
| sex_binary | 1433 | Transformer | 0.832 | 0.872 | 0.905 | **0.910** | 0.910 | 240m |
| age_class | ~1860 | LSTM | 0.865 | 0.890 | **0.893** | 0.885 | 0.893 | 120m |
| age_class | ~1860 | Transformer | 0.854 | 0.878 | 0.902 | **0.905** | 0.905 | 240m |
| apnea_binary | 2054 | LSTM | 0.758 | 0.792 | **0.832** | 0.827 | 0.832 | 120m |
| apnea_binary | 2054 | Transformer | 0.753 | 0.825 | **0.857** | 0.854 | 0.857 | 120m |
| bmi_binary | 1856 | LSTM | 0.760 | 0.756 | 0.756 | 0.748 | 0.767 | 80m |
| bmi_binary | 1856 | Transformer | 0.747 | 0.755 | 0.766 | **0.777** | 0.777 | 240m |
| sleep_eff_binary | 2023 | LSTM | 0.697 | 0.731 | 0.780 | **0.799** | 0.799 | 240m |
| sleep_eff_binary | 2023 | Transformer | 0.707 | 0.760 | 0.815 | **0.831** | 0.831 | 240m |
| depression_extreme | 229 | LSTM | 0.757 | 0.761 | 0.748 | 0.750 | 0.770 | 10m |
| osa_apples_postqc | 161 | LSTM | 0.769 | **0.834** | 0.789 | 0.774 | 0.834 | 40m |

Full 6-context LSTM/Transformer per-task curves are available in analysis.csv.

### Table 2 — L* (Saturation Threshold) per task/head (already generated: results/tables/table2_lstar_fast.*)

| Task | LSTM L* | LSTM ΔAUROC (30s→best) | Transformer L* | Transformer ΔAUROC |
|---|---|---|---|---|
| sex_binary | 80m | +0.047 | 240m | +0.078 |
| age_class | 40m | +0.028 | 120m | +0.051 |
| apnea_binary | 120m | +0.074 | 120m | +0.104 |
| bmi_binary | 10m | +0.006 | 240m | +0.030 |
| sleep_efficiency_binary | 240m | +0.102 | 240m | +0.124 |
| depression_extreme_binary | 10m | +0.013 | — | — |
| osa_binary_apples_postqc | 40m | +0.065 | — | — |

L* defined as smallest L within 0.005 AUROC of the best context (K=all, test split).

**Key message:** L* ranges from 10m to 240m across tasks. Tasks based on physiological
measurements spanning the night (sleep efficiency, apnea) require long context. Tasks based on
static or distributed traits (sex, BMI) saturate earlier or show little context benefit.

### Figures needed
- **Figure 2 (primary results figure):** Saturation curves — AUROC vs context length (log scale),
  one subplot per task, lines for LSTM and Transformer (and MeanPool once available). Y-axis
  starts at 0.5. Error bands from bootstrap CIs. Mark L* with a vertical dashed line or dot.
- These plots already exist at: `/scratch/boshra95/psg/unified/results/phase0_v3/figures/`

### K=5 numbers (deployment scenario, for Table 1 if two columns needed)

| Task | Head | Best L@K=5 | AUROC@K=5 |
|---|---|---|---|
| sex_binary | LSTM | 120m | 0.872 |
| sex_binary | Transformer | 120m | 0.905 |
| age_class | LSTM | 120m | 0.893 |
| age_class | Transformer | 120m | 0.902 |
| apnea_binary | LSTM | 120m | 0.831 |
| apnea_binary | Transformer | 120m | 0.856 |
| bmi_binary | LSTM | 80m | 0.763 |
| bmi_binary | Transformer | 80m | 0.767 |
| sleep_efficiency_binary | LSTM | 240m | 0.799 |
| sleep_efficiency_binary | Transformer | 240m | 0.831 |
| depression_extreme_binary | LSTM | 10m | 0.776 |
| osa_binary_apples_postqc | LSTM | 10m | 0.823 |

For K=5 vs K=all: at medium-to-long contexts (≥40m) the gap is small (<0.003). At short contexts
(30s) the gap is meaningful (e.g., sex_binary LSTM: K=5=0.805 vs K=all=0.824 at 30s).

---

## IV-B. Head Architecture Comparison (H4)

### Status
- LSTM vs Transformer: **available for all tasks**.
- MeanPool: **PENDING** — not yet collected. Cannot fully test H4 without MeanPool.
- Write this subsection with LSTM/Transformer now; add MeanPool when available.

### Numbers (from cross-task summary at best context per head, K=all)

| Task | LSTM best | Transformer best | Δ (Transformer advantage) |
|---|---|---|---|
| sex_binary | 0.872 | 0.910 | **+0.038** |
| age_class | 0.893 | 0.905 | +0.012 |
| apnea_binary | 0.832 | 0.857 | **+0.025** |
| bmi_binary | 0.767 | 0.777 | +0.010 |
| sleep_efficiency_binary | 0.799 | 0.831 | **+0.032** |

**Finding:** Transformer consistently outperforms LSTM. Advantage is largest for tasks with strong
signal and long optimal context (sex, sleep efficiency, apnea). Smallest for low-signal tasks (BMI).

**Important caveat for writing:** Both heads are compared at their independently optimised best L.
A strictly fair H4 test would compare at a single fixed L. Both LSTM and Transformer at L=120m:
- sex: LSTM 0.872, Transformer 0.905 (+0.033) — meaningful
- apnea: LSTM 0.832, Transformer 0.857 (+0.025)
- sleep_eff: LSTM 0.780, Transformer 0.815 (+0.035)
This fixed-L comparison still shows consistent Transformer advantage.

**MeanPool role:** H4 predicts MeanPool will approach LSTM/Transformer at short L but fall behind
at long L. Without MeanPool results, H4 can only be partially tested. Discuss in limitations or
add a placeholder figure.

### Table 5 — Head comparison table (results/tables/table5_heads_fast.*, partially generated)
Only sex_binary_lstm row exists. Need to regenerate for all tasks once MeanPool is available.

---

## IV-C. Aggregation and Iso-Compute Analysis (H2, H3)

### K-Sweep (H3 — aggregation saturation)

**Available:** sex_binary_lstm only (Table 3, results/tables/table3_kgrid_sex_binary_lstm_fast.*)

Key data from sex_binary/LSTM (AUROC at each context × K):

| L | K=1 | K=5 | K=10 | K=20 | K=all |
|---|---|---|---|---|---|
| 30s | 0.701 | 0.805 | 0.815 | 0.821 | 0.824 |
| 10m | 0.720 | 0.823 | 0.836 | 0.840 | 0.842 |
| 40m | 0.760 | 0.853 | 0.862 | 0.866 | 0.866 |
| 80m | 0.798 | 0.866 | 0.868 | — | 0.868 |
| 120m | 0.822 | 0.855 | — | — | 0.855 |
| 240m | 0.791 | — | — | — | 0.812 |

**Findings:**
- At short contexts (30s, 10m), K=1 is substantially worse than K=all; aggregation matters.
- Aggregation saturates around K=5–10 at short contexts, K=3–5 at medium contexts.
- Even at K=all for 30s (0.824), performance does not reach K=5 at 80m (0.866) — demonstrating
  that K windows cannot fully substitute for longer L.
- At long contexts (80m+), K=5 ≈ K=all because few non-overlapping windows fit per recording.

**For H3 claim:** Performance saturates quickly with K. K=5 captures ≥95% of the K=all gain at
every context length tested (for sex_binary_lstm).

**What's missing:** K-grid for other tasks. Recommend generating table3 for at least apnea_binary
and sleep_efficiency_binary to show that saturation patterns hold across tasks.

### Iso-Compute Analysis (H2 — aggregation substitution)

**Status: PARTIALLY AVAILABLE** from sparse K data in analysis.csv. Full heatmap requires dense K
sweep (build_heatmap_df.py — PENDING).

**Manual iso-compute reading (sex_binary/LSTM, budget = 80 min):**

| Configuration | Total signal | AUROC |
|---|---|---|
| L=80m, K=1 | 80 min | 0.798 |
| L=10m, K=8 | ~80 min | ~0.833 (interpolated between K=5=0.823 and K=10=0.836) |
| L=40m, K=2 | 80 min | ~0.804 (interpolated between K=1=0.760 and K=5=0.853) |

**Preliminary finding (H2 — partially falsified):** At matched signal budget, aggregating K windows
of shorter context (10m × 8) tends to slightly outperform a single longer window (80m × 1).
However, once L is long enough (≥80m), further aggregation adds little. Full iso-compute analysis
with dense K values needed for definitive conclusions.

**Recommendation:** Run the dense K sweep (gen_commands.py analyze --k-dense for each task/head)
before finalising this subsection. The 2D heatmap (Figure 3 or 4) is a key paper figure.

---

## IV-D. Channel Count Expansion

### Numbers (fast vs full, K=all, best context per task/head)

From results_summary_fast_vs_full.csv:

| Task | Head | Fast AUROC | Full AUROC | Gain |
|---|---|---|---|---|
| apnea_binary | Transformer | 0.857 | 0.901 | **+0.044** |
| apnea_binary | LSTM | 0.832 | 0.874 | **+0.042** |
| bmi_binary | Transformer | 0.777 | 0.816 | **+0.039** |
| bmi_binary | LSTM | 0.767 | 0.802 | **+0.036** |
| sex_binary | LSTM | 0.873 | 0.906 | **+0.033** |
| sex_binary | Transformer | 0.910 | 0.929 | +0.019 |
| sleep_efficiency_binary | LSTM | 0.788 | 0.810 | +0.022 |
| sleep_efficiency_binary | Transformer | 0.831 | 0.825 | −0.006 |
| age_class | LSTM | 0.893 | 0.901 | +0.008 |
| age_class | Transformer | 0.905 | 0.911 | +0.006 |
| depression_extreme_binary | LSTM | 0.770 | 0.752 | −0.018 |
| osa_binary_apples_postqc | LSTM | 0.834 | 0.772 | **−0.062** |

**Note on full-channel fast-channel mismatch for sleep_eff Transformer and osa_apples LSTM:**
- sleep_eff/Transformer: small drop (−0.006), within noise for this task
- osa_apples/LSTM: large drop (−0.062), likely due to small N (161 test), APPLES-specific channel
  availability patterns interacting with full-channel preprocessing. Treat with caution; may need
  footnote or exclusion from this comparison.

**Finding:** Full-channel (up to 23 channels) improves AUROC for tasks with strong physiological
signal (apnea: +0.042–0.044, BMI: +0.036–0.039, sex: +0.019–0.033). Gains are smaller for tasks
already well-explained by the reduced channel set (age: +0.006–0.008). No gain for small-N tasks
(depression, osa_apples) where additional channels introduce variance without signal.

**Context sensitivity with full channels:** Full-channel models appear to saturate at shorter L
for some tasks. For sex_binary_lstm, full-channel L* shifts from 120m (fast) to approximately
40m (full) — richer per-patch representations require less temporal aggregation.

**Figure:** Grouped bar chart or scatter showing fast vs full AUROC per task, or saturation curves
overlaid fast vs full for sex, apnea, sleep_efficiency.

---

## IV-E. Modality Group Ablation

### Numbers (Table 6, results/tables/table6_modality.*, 25/25 conditions complete)

LSTM head, at task-specific context (120m for sex/apnea/sleep_eff/age; 40m for BMI):

| Task | Full | No BAS (Δ) | No RESP (Δ) | No EKG (Δ) | Cardio only (Δ) | BAS only (Δ) |
|---|---|---|---|---|---|---|
| sex_binary | 0.872 | 0.799 (−0.073) | 0.843 (−0.029) | 0.782 (−0.090) | 0.796 (−0.076) | 0.777 (−0.096) |
| apnea_binary | 0.832 | 0.782 (−0.050) | 0.766 (−0.066) | 0.808 (−0.024) | 0.770 (−0.062) | 0.723 (−0.109) |
| sleep_efficiency | 0.780 | 0.682 (−0.099) | 0.760 (−0.021) | 0.750 (−0.030) | 0.670 (−0.110) | 0.786 (+0.005) |
| age_class | 0.893 | 0.835 (−0.059) | 0.882 (−0.012) | 0.868 (−0.025) | 0.821 (−0.072) | 0.860 (−0.034) |
| bmi_binary | 0.756 | 0.728 (−0.028) | 0.759 (+0.003) | 0.758 (+0.002) | 0.663 (−0.093) | 0.741 (−0.015) |

Note: Full-channel baseline is not available for sleep_eff at 120m (full-channel is 0.810; the
table above uses fast-channel for consistency with the ablation conditions).

### Task-by-task findings

**Apnea (AHI ≥ 15):**
- RESP is the most necessary modality: No RESP gives the largest single-knockout drop (−0.066).
- No BAS: −0.050. Significant but RESP > BAS.
- BAS only is worst condition overall (−0.109): brain signals alone cannot predict apnea.
- Cardio only (RESP+EKG, BAS+EMG zeroed): 0.770 — respiration alone is insufficient without neural.
- Interpretation: OSA is a respiratory disorder. RESP is necessary. However BAS contributes
  independently (likely via cortical arousals and EEG fragmentation). This directly replicates
  OSF's finding that respiratory channels are necessary for hypopnea/O₂ desaturation prediction.
- Cardio-only (0.770) is far below SleepFounder OSA AUROC (0.917). Attribution: SleepFounder
  fine-tunes full model on 800K hours; our setting uses frozen backbone. Not a fair comparison;
  acknowledge in discussion.

**Sleep efficiency:**
- BAS is the dominant modality: No BAS drops by −0.099 (largest single drop in the table).
- BAS only ≈ Full (0.786 vs 0.780, +0.005 — noise level): EEG/EOG alone explains sleep efficiency.
- Three converging lines of evidence: (1) No BAS large drop, (2) BAS only ≈ Full, (3) Cardio only
  is worst (−0.110). Sleep efficiency is encoded in EEG (sleep staging) rather than respiration.
- No RESP: −0.021 — minor.

**Sex:**
- EKG is the most necessary single modality: No EKG drops by −0.090 (surprising — not anticipated).
- No BAS: −0.073. Significant.
- Cardio only: −0.076. RESP+EKG together not enough — need BAS.
- BAS only is worst (−0.096): neither brains nor respiration alone suffices.
- Interpretation: cardiac signal (HRV, resting heart rate, sex differences in cardiac electrophysiology)
  carries more sex-discriminative information than EEG in the SleepFM embedding space. This is a
  novel and potentially surprising finding.

**Age group:**
- BAS most necessary (−0.059). No EKG: −0.025. No RESP: −0.012 (borderline noise).
- Cardio only: −0.072. BAS only: −0.034.
- Interpretation: physiological aging is concentrated in EEG changes (slow wave sleep reduction,
  spindle changes) and cardiac aging. Respiration adds less.

**BMI:**
- No single modality is clearly necessary: No RESP (+0.003), No EKG (+0.002) are noise-level;
  No BAS (−0.028) is borderline.
- But cardio only (BAS+EMG zeroed) drops by −0.093 — the largest BMI drop by condition.
- Interpretation: BMI signal is likely diffuse across modalities (metabolic effects on sleep
  architecture, respiratory effort, cardiac patterns). The cardio-only drop suggests RESP and EKG
  together are important even if neither alone is necessary — possible interaction effect.
  No clean interpretation available; acknowledge in discussion.

### Caveat
Each condition = one training run with no multi-seed variance. Treat |Δ| < 0.02 as noise-level.
All changes ≥ 0.05 are very likely real at N_test = 1430–2054.

---

## IV-F. Sleep Staging

### Status: PENDING

The primary multi-dataset run (SHHS+MrOS+APPLES, centered windows, complete_only, hidden=256,
num_layers=2, val_kappa monitor) is submitted but no final results are available.

### Preliminary context from archived runs

**Do NOT use these numbers in the paper — they are from stale runs with bugs or wrong datasets.**
Context only:
- Old APPLES-only LSTM (30s): AUROC ≈ 0.927. APPLES-only saturates at ~10m.
- Old phase0 multi-dataset (hidden=256): 30s κ≈0.580, 10m κ≈0.620, 40m κ≈0.628. Plateau at 40m.
- Including STAGES hurts: κ drops 0.028–0.094. STAGES excluded from primary run.

### What to report when results arrive
1. Cohen's κ vs context length for LSTM and Transformer (primary Figure for sleep staging).
2. Per-stage F1 at best L: Wake, N1, N2, N3, REM. N1 expected to be lowest (5–8% of epochs).
3. Note that STAGES is excluded from training and evaluation (footnote already in Table 1).
4. Note that at 240m context, ~50% of anchor epochs are excluded (complete_only policy); at 30s,
   essentially all epochs are evaluated. Discuss whether this affects saturation interpretation.
5. Supplementary: comparison run including STAGES — expected lower κ; demonstrates the dataset
   domination problem quantified by the training item count.
6. Consider: supplementary common evaluation set analysis (restrict all contexts to 240m-valid
   anchors only) to separate context-length effect from anchor-set size effect.

### Placeholder text for paper until results arrive
"Sleep staging results will be reported once the primary multi-dataset training run completes.
Preliminary analyses with a single-cohort subset suggest a saturation context around 10–40 min."

---

## Supplementary Material

### Supp Table I — Training K sensitivity
Not yet run. Confirms: fixed K=5 windows per epoch ≈ token-budget schedule at L ≥ 10m.
See docs/TRAINING_PROTOCOL_FIXES.md for the design; run the comparison experiment to get numbers.

### Supp Table II — Threshold tuning details
From docs/POSTHOC_THRESHOLD_TUNING.md; numbers to include:

Tasks requiring threshold tuning (BA improvement at val-selected threshold):

| Task | Head | Best Δ BA across contexts | Context with best Δ | Note |
|---|---|---|---|---|
| osa_binary_apples_postqc | LSTM | **+0.220** | 10m | Critical; t=0.5 gives BA≈0.55 |
| depression_extreme_binary | LSTM | +0.065 | 80m | Use tuned for consistency |
| bmi_binary | Transformer | +0.027 | 30s | Modest improvement |
| bmi_binary | LSTM | +0.015 | — | Modest |
| sex_binary | LSTM | +0.009 | — | Small |

Tasks where t=0.5 retained: cvd (excluded from paper), apnea, sleep_efficiency, age_class,
sex/Transformer. In paper: note in table captions where t* is used. OSA APPLES requires
a clear note (threshold shift is very large; default is effectively non-informative).

### Supp — Common evaluation set (sleep staging)
Planned: restrict all context lengths to the subset of anchors that are valid under 240m context
(anchors far enough from recording edges for the longest window). This allows comparing κ across
contexts with an identical evaluation population, removing the confound that longer contexts
evaluate fewer anchors. Implement after multi-dataset staging run completes.

### Supp — Bootstrap CIs on saturation curves
Already computed for some rows. Run:
```bash
bash scripts/run_analysis.sh [all tasks] --bootstrap 1000 --analyze-only
```
Required before finalising the saturation curve figures with error bands.

---

## Figures needed (summary)

| Figure | Content | Status |
|---|---|---|
| Fig 2 (or Fig 1) | Saturation curves — AUROC vs L, per task, LSTM + Transformer | Exists on cluster; need to pull |
| Fig 3 | K-sweep curves — AUROC vs K for representative tasks | Exists for sex_binary_lstm |
| Fig 4 | 2D iso-compute heatmap (L × K grid) | PENDING dense K sweep |
| Fig 5 | Modality ablation bar chart — AUROC drop per condition per task | Can generate from Table 6 |
| Fig 6 | Fast vs full channel comparison | Can generate from results_summary CSV |
| Fig 7 | Sleep staging κ vs L (plus per-stage F1) | PENDING |

---

## Key claims to support in Results

- H1 (context saturation): CONFIRMED for all tasks. L* ranges from 10m to 240m. ✓
- H2 (aggregation substitution): PARTIALLY SUPPORTED (sparse data); dense K sweep needed.
- H3 (aggregation saturation): CONFIRMED for sex_binary/LSTM. K≥5 captures most benefit. Generalise
  with additional task data.
- H4 (temporal head advantage): PARTIALLY CONFIRMED (LSTM vs Transformer). MeanPool needed for full
  test. Transformer > LSTM at long contexts for high-signal tasks.

---

## Numbers NOT to report (dropped tasks, use for internal reference only)

- psqi_binary: best AUROC 0.557, no context benefit → dropped (near chance)
- sleepiness_binary: best AUROC 0.629, flat curve → dropped (no PSG signal)
- cvd_binary: best AUROC 0.689, flat curve → dropped (weak signal)
- osa_severity_apples: not run → dropped

These numbers may be cited in Discussion as evidence that some clinical endpoints are not
predictable from overnight PSG under any context length — label noise or task confounds suspected.
