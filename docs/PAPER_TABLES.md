# Paper Tables — Structure and Rationale

This document describes the tables planned for the paper, what each shows, how it is constructed, and why it is useful. Results are not filled in here — this is a structural reference for paper writing.

A separate table for **channel ablation** (zeroing out BAS/RESP/EKG/EMG modality groups) is planned after those experiments run; see `SOTA_COMPARISON_AND_ABLATIONS.md §4A`.

---

## Table 1 — Main Performance Table: Peak AUROC Across Tasks (seq2label)

### What it shows
The primary result table. For each clinical task and each head architecture (LSTM, Transformer, MeanPool), reports the best test AUROC achieved across all context lengths, at K=5 windows (the clinical deployment scenario matching training conditions) and K=all (full-night aggregation ceiling).

### Structure

| Task | N_test | Head | Best L | AUROC@K=5 | AUROC@K=all |
|------|--------|------|--------|-----------|-------------|
| sex_binary | … | LSTM | … | … ± CI | … ± CI |
| sex_binary | … | Transformer | … | … ± CI | … ± CI |
| sex_binary | … | MeanPool | … | … ± CI | … ± CI |
| bmi_binary | … | LSTM | … | … | … |
| … | | | | | |

**Columns:**
- `N_test`: number of test subjects
- `Best L`: context length at which AUROC is highest (task-specific saturation point)
- `AUROC@K=5` and `AUROC@K=all`: mean ± 95% bootstrap CI
- Group rows by task; use light separator lines between tasks

### How to generate
`scripts/summarize_results.py` produces the core numbers. CIs come from `analyze_windows.py --bootstrap` columns in `window_analysis_test.csv`. Use `--latex` flag for LaTeX formatting.

### Why it is useful
This is the first thing reviewers and readers look at. It establishes the empirical ceiling of the system on each task and motivates the rest of the analysis. The "Best L" column directly previews the saturation analysis (Table 2).

### Note on fast vs full channel
The best context length shown here is independently optimised per channel configuration. The fast-channel (7 PSG channels) and full-channel (~23 channels) best-L values may differ because each channel set saturates at a different context. This is the *peak capability* view, not a controlled ablation — for a channel-only comparison at fixed L, see Table 6 (channel ablation).

---

## Table 2 — Context Saturation Summary: L* per Task

### What it shows
For each task and head, the **saturation context length L***: the smallest L where AUROC is within 0.5% of the task's best AUROC. This directly answers H1 (context saturation) and tells practitioners "how much context do you actually need?"

### Structure

| Task | LSTM L* | Transformer L* | MeanPool L* | AUROC gain (30s → L*) |
|------|---------|----------------|-------------|----------------------|
| sex_binary | … | … | … | … |
| bmi_binary | … | … | … | … |
| apnea_binary | … | … | … | … |
| … | | | | |

**Columns:**
- `L*` expressed in human-readable units (e.g., 40m, 120m)
- `AUROC gain`: the absolute improvement in AUROC from the 30s baseline to L*; quantifies how much longer context helps
- Sort rows by LSTM L* to reveal which tasks need the most context

### How to generate
From `results/collected/phase0_v3/analysis.csv` (or full-channel equivalent). L* definition: `min L s.t. AUROC(L, K=all) ≥ max_AUROC − 0.005`. Already computed by `plot_task_comparison.py` (§6C of ANALYSIS_IDEAS.md).

### Why it is useful
The most actionable finding for clinical deployment. A clinical reader wants to know: "for OSA screening, do I need a full night or is 10 minutes enough?" L* per task answers this directly. Also answers reviewer questions about the shape of the saturation curve.

---

## Table 3 — Aggregation Saturation: AUROC vs K at Representative Contexts

### What it shows
For one representative task and head (e.g., `sex_binary_lstm`), the full AUROC × K table at selected context lengths. This is the numerical companion to the heatmap figure and directly answers H3 (aggregation saturation): how many windows are needed per patient in deployment?

### Structure

| Context L | K=1 | K=5 | K=10 | K=20 | K=50 | K=all |
|-----------|-----|-----|------|------|------|-------|
| 30s | … | … | … | … | … | … |
| 10m | … | … | … | … | … | … |
| 40m | … | … | … | … | … | … |
| 80m | … | … | … | … | … | … |
| 120m | … | … | … | … | … | … |
| 240m | … | … | … | … | … | … |

**Rows:** each trained context length; **Columns:** inference-time K values.

### How to generate
From `window_analysis_test.csv` per experiment, assembled by `build_heatmap_df.py`. The heatmap figure shows this visually; this table gives exact numbers for a supplementary.

### Why it is useful
The heatmap figure is the key visual but tables let readers extract exact numbers. Showing that AUROC@K=5 is within X% of K=all at long contexts supports the claim that 5 windows is sufficient for deployment. Iso-compute diagonal cells (where K×L ≈ constant) are the direct answer to H2.

---

## Table 4 — Cross-Task Sensitivity: AUROC Gain from Context (seq2label)

### What it shows
For each task, the AUROC improvement from the shortest context (30s) to the best context at K=all. Tasks are ranked by this "context sensitivity." This is the summary figure for the cross-task comparison.

### Structure

| Task | AUROC@30s | AUROC@best L | Context sensitivity (Δ) | Task category |
|------|-----------|--------------|------------------------|---------------|
| apnea_binary | … | … | … | respiratory |
| sleep_efficiency_binary | … | … | … | sleep quality |
| sex_binary | … | … | … | demographics |
| … | | | | |

**Sorted by** context sensitivity descending. A task near the top has high context sensitivity (long context needed); near the bottom means short context suffices.

### How to generate
`plot_task_comparison.py` (ANALYSIS_IDEAS.md §6) generates the scatter and bar plots from `analysis.csv`. The table is the numerical companion.

### Why it is useful
Answers the core clinical question in one table: which tasks benefit from long context and which don't? A reviewer can immediately see whether the paper's hypothesis is supported across task types. Also surfaces the biological insight: do tasks dependent on slow physiological rhythms (sleep efficiency, OSA severity) need more context than instantaneous markers (sex)?

---

## Table 5 — Head Architecture Comparison at Fixed Context

### What it shows
For each task at its saturation context L*, compare LSTM vs Transformer vs MeanPool. This answers H4: does temporal modeling add value over simple averaging?

### Structure

| Task | Context | LSTM AUROC | Transformer AUROC | MeanPool AUROC | Temporal advantage (LSTM − MeanPool) |
|------|---------|------------|-------------------|----------------|--------------------------------------|
| sex_binary | L* = … | … ± CI | … ± CI | … ± CI | … |
| bmi_binary | L* = … | … | … | … | … |
| … | | | | | |

**Evaluated at:** K=all (or K=5 for deployment column). Context fixed at each task's saturation point for the best head.

### How to generate
From `analysis.csv` filtered to `context_length == L*` per task. `summarize_results.py --compare` produces a simplified version; this table is more explicit about the head comparison.

### Why it is useful
Directly tests whether LSTM and Transformer justify their additional complexity. If MeanPool matches LSTM at L*, temporal ordering within the context window is not critical. If LSTM/Transformer consistently exceed MeanPool, there are genuine temporal dependencies in PSG that require sequence modeling. Expected to be a main result for the architecture contribution.

---

## Table 6 — Channel Ablation: Modality Group Contribution

**Status: Planned — runs not yet complete. See `SOTA_COMPARISON_AND_ABLATIONS.md §4A`.**

### What it shows
For each clinical task, the AUROC under different modality group combinations: all 4 groups (baseline), cardiorespiratory only (RESP+EKG), EEG only (BAS), and potentially single-modality baselines. This directly addresses the OSF and SleepFounder findings on channel importance.

### Structure

| Task | All (BAS+RESP+EKG+EMG) | Cardiorespiratory (RESP+EKG) | EEG only (BAS) | No EEG (RESP+EKG+EMG) |
|------|----------------------|------------------------------|----------------|----------------------|
| bmi_binary | … | … | … | … |
| apnea_binary | … | … | … | … |
| sleep_staging | … | … | … | … |
| … | | | | |

**Fixed at:** one context length (10m or L* for each task), K=all. Head: LSTM.

### How to generate
Requires re-training heads with zeroed modality slices of the 512-dim SleepFM embedding. Controlled by `--zero-modalities` flag in the dataset loader (to be implemented; see SOTA doc §4A).

### Why it is useful
Directly answers: "which modalities carry the clinical signal?" Critical for the paper's positioning relative to OSF (channel masking) and SleepFounder (zero-burden cardiorespiratory). Shows task-specificity: apnea detection might be driven by RESP, sleep staging by BAS (EEG), sex by any modality. This table lets practitioners know whether they need full PSG or could use wearable devices.

---

## Table 7 — SOTA Comparison

### What it shows
Our best results on overlapping tasks vs. SleepFounder, SleepMaMi, and OSF, with explicit notes on evaluation protocol differences. Not a direct head-to-head (impossible due to different training data and protocols) but a reference for positioning.

### Structure

| Task | Metric | Our result (frozen SleepFM) | SleepFounder | SleepMaMi | OSF |
|------|--------|-----------------------------|--------------|-----------|-----|
| sleep_staging | Cohen's κ | … | 0.65 | 81.9% acc | AUC=97.3 |
| sex_binary | AUROC | … | 0.85 | — | — |
| apnea detection | AUROC | … | 0.917 | Incl. SDB | — |
| CVD prediction | AUROC | … | 0.81 (CHD) | — | 0.681 |

### How to generate
Our numbers from `summarize_results.py`. SOTA numbers from the three papers documented in `SOTA_COMPARISON_AND_ABLATIONS.md §1`.

### Why it is useful
Provides context for the empirical results. The evaluation protocol differences (frozen vs fine-tuned, different training sets) must be noted as caveats. The positioning argument: our numbers are competitive despite using only a frozen backbone and lightweight head, demonstrating data efficiency relative to full fine-tuning.

---

## Table 8 — Sleep Staging Results (seq2seq)

**Status: Planned — staging runs not yet complete.**

### What it shows
For the sleep staging task, Cohen's κ and per-stage F1 vs context length, for LSTM and Transformer heads. Sleep staging is seq2seq (one prediction per 30-second epoch) so it has no K-aggregation; the only axis is context length L.

### Structure

| Context L | LSTM κ | LSTM F1 (W) | LSTM F1 (N1) | LSTM F1 (N2) | LSTM F1 (N3) | LSTM F1 (R) | Transformer κ | MeanPool κ |
|-----------|--------|------------|--------------|--------------|--------------|-------------|---------------|------------|
| 30s | … | … | … | … | … | … | … | … |
| 10m | … | … | … | … | … | … | … | … |
| 40m | … | … | … | … | … | … | … | … |
| 80m | … | … | … | … | … | … | … | … |
| 120m | … | … | … | … | … | … | … | … |
| 240m | … | … | … | … | … | … | … | … |

**Primary metric:** Cohen's κ (standard in sleep staging literature). Per-stage F1 for Wake, N1, N2, N3, REM shows where context helps most.

### How to generate
From `window_analysis_test.csv` for the sleep_staging experiments. Note: sleep staging uses the `complete_only` window policy (centred windows, not causal); results on a common evaluation set (240m-valid anchors only) will be in a separate supplementary section.

### Why it is useful
Sleep staging is the established benchmark for PSG models. κ vs context length directly answers whether longer context improves epoch-level staging — a different question from the subject-level clinical prediction (seq2label) tasks. N1 staging is expected to be the stage most sensitive to context length because N1 occurs at sleep onset and is surrounded by context that distinguishes it from wake.

---

## Table 9 — Cohort Breakdown (Supplementary)

### What it shows
For representative tasks (sex_binary, bmi_binary, apnea_binary), AUROC broken down by dataset (APPLES, SHHS, MrOS, STAGES) at the saturation context L*. Shows whether context-length benefits are cohort-specific or universal.

### Structure

| Task | Context L* | Overall AUROC | APPLES AUROC (N=…) | SHHS AUROC (N=…) | MrOS AUROC (N=…) | STAGES AUROC (N=…) |
|------|------------|---------------|-------------------|------------------|------------------|-------------------|
| sex_binary | … | … | … | … | … | … |
| bmi_binary | … | … | … | … | … | … |
| apnea_binary | … | … | … | … | … | … |

### How to generate
`plot_cohort_saturation.py` (ANALYSIS_IDEAS.md §7). Filters predictions parquets by the `dataset` column.

### Why it is useful
Robustness check. If one cohort drives all the benefit, the finding does not generalise. Also mirrors OSF's in-domain (SHHS) vs out-of-domain (MrOS) evaluation, allowing a rough comparison under our evaluation protocol.

---

## Table 10 — Bootstrap CI Summary (Supplementary)

### What it shows
For all seq2label tasks at L* and K=5 (primary deployment scenario), AUROC with 95% bootstrap confidence intervals. Confirms statistical significance of performance differences.

### Structure

| Task | Head | L* | AUROC (mean) | 95% CI | K=5 AUROC | 95% CI |
|------|------|----|--------------|--------|-----------|--------|
| sex_binary | LSTM | … | … | [lo, hi] | … | [lo, hi] |
| sex_binary | Transformer | … | … | [lo, hi] | … | [lo, hi] |
| … | | | | | | |

### How to generate
CI columns from `analyze_windows.py --bootstrap 1000` are in `window_analysis_test.csv` and collected into `analysis.csv`. `summarize_results.py --latex` includes these if CI columns are present.

### Why it is useful
Reviewers at medical AI venues expect CI on AUROC estimates. With 200–500 test subjects, point estimates can be noisy; CIs distinguish real improvements from sampling noise. Essential for claiming that context length L₁ is significantly better than L₀.

---

## Summary of Tables and Placement

| Table | Location | Fills in after |
|-------|----------|----------------|
| 1 — Peak AUROC (seq2label) | Main text | Phase 0 v3 runs complete |
| 2 — Saturation L* per task | Main text | Phase 0 v3 runs complete |
| 3 — AUROC vs K at representative task | Main or Supplementary | Phase 0 v3 + dense K analysis |
| 4 — Cross-task sensitivity ranking | Main text | Phase 0 v3 runs complete |
| 5 — Head architecture comparison | Main or Supplementary | Phase 0 v3 runs complete |
| 6 — Channel ablation | Main text | Channel ablation experiments (§4A SOTA doc) |
| 7 — SOTA comparison | Main text | All main runs + SOTA numbers |
| 8 — Sleep staging (seq2seq) | Main text | Staging runs + analysis |
| 9 — Cohort breakdown | Supplementary | Phase 0 v3 runs + dataset filter |
| 10 — Bootstrap CI summary | Supplementary | Bootstrap CI runs complete |

---

## Notes for Writing

- **Units:** All AUROC values to 3 decimal places (e.g., 0.873). CIs to 3 decimals. Cohen's κ to 3 decimals.
- **Context lengths:** Use human-readable labels (30s, 10m, 40m, 80m, 120m, 240m) not raw minutes.
- **Bold:** Bold the best entry per task row in Tables 1 and 5.
- **Protocol caveat for Table 7:** Always include a footnote: "Our results use a frozen SleepFM backbone and lightweight task head. Comparison models fine-tune the full pre-trained encoder. Direct AUROC comparison is approximate."
- **Channel note for Tables 1–5:** State whether results are fast-channel (7 channels) or full-channel (~23 channels). Fast-channel results are the primary reported numbers; full-channel results are reported as an extension.
- **N_test consistency:** The same test subjects must be used across all context lengths per task. Document any N differences due to the `complete_only` window policy in staging.
