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

---

## How to Generate Tables

### Prerequisites

All table scripts (except Table 9) read from the collected `analysis.csv`. Run these before generating tables:

```bash
# Step 1: window analysis with dense K and bootstrap CIs (if not done)
bash scripts/run_analysis.sh sex_binary_lstm bmi_binary_lstm ... --bootstrap 1000

# Step 2: collect results into analysis.csv
python scripts/gen_commands.py collect sex_binary_lstm bmi_binary_lstm ... | bash
```

Table 9 additionally reads parquet files from scratch — no extra setup needed beyond running inference.

### Individual table scripts

Each script writes `results/tables/<stem>.{csv,md,tex}` and prints the table to stdout.

**Table 1 — Peak AUROC:**
```bash
# Fast channel, all tasks
python scripts/make_table1_peak_auroc.py

# Specific tasks only
python scripts/make_table1_peak_auroc.py --tasks sex_binary bmi_binary apnea_binary

# Full channel
python scripts/make_table1_peak_auroc.py \
    --collected-dir results/collected/phase0_v3_full --channel full

# With LaTeX output
python scripts/make_table1_peak_auroc.py --latex
```

**Table 2 — Saturation L\*:**
```bash
python scripts/make_table2_lstar.py
python scripts/make_table2_lstar.py --tolerance 0.01   # looser L* criterion
python scripts/make_table2_lstar.py --tasks sex_binary apnea_binary --latex
```

**Table 3 — AUROC×K grid (one experiment at a time):**
```bash
python scripts/make_table3_kgrid.py sex_binary_lstm
python scripts/make_table3_kgrid.py sex_binary_lstm --k-values 1 5 10 20 all
python scripts/make_table3_kgrid.py bmi_binary_transformer --latex
```

**Table 4 — Cross-task sensitivity:**
```bash
python scripts/make_table4_sensitivity.py
python scripts/make_table4_sensitivity.py --head lstm
python scripts/make_table4_sensitivity.py --tasks sex_binary bmi_binary apnea_binary --latex
```

**Table 5 — Head comparison at L\*:**
```bash
python scripts/make_table5_heads.py
python scripts/make_table5_heads.py --heads lstm transformer   # skip mean_pool if not run
python scripts/make_table5_heads.py --tasks sex_binary apnea_binary --latex
```

**Table 9 — Cohort breakdown (one experiment at a time):**
```bash
python scripts/make_table9_cohort.py sex_binary_lstm
python scripts/make_table9_cohort.py sex_binary_lstm --context 40m   # force a specific context
python scripts/make_table9_cohort.py bmi_binary_lstm --datasets apples shhs mros
# For full channel (parquets on psg_full):
python scripts/make_table9_cohort.py sex_binary_lstm \
    --results-dir /scratch/boshra95/psg_full/unified/results/phase0_v3_full \
    --collected-dir results/collected/phase0_v3_full --channel full
```

**Table 10 — Bootstrap CI summary:**
```bash
python scripts/make_table10_ci.py
python scripts/make_table10_ci.py --tasks sex_binary bmi_binary apnea_binary
python scripts/make_table10_ci.py --latex
# Note: if CI columns are absent, AUROC is shown without CI — run analyze --bootstrap 1000 first
```

### Via gen_commands.py

All table scripts are wired into `gen_commands.py` and auto-derive paths from the registry:

```bash
# Print the command (then pipe to bash, or copy-paste)
python scripts/gen_commands.py table-1
python scripts/gen_commands.py table-2
python scripts/gen_commands.py table-3 sex_binary_lstm
python scripts/gen_commands.py table-4
python scripts/gen_commands.py table-5
python scripts/gen_commands.py table-9 sex_binary_lstm
python scripts/gen_commands.py table-10

# Full channel (auto-routes to phase0_v3_full collected dir)
python scripts/gen_commands.py --registry experiments/v2_full_registry.yaml table-1

# With task subset
python scripts/gen_commands.py table-1 --tasks sex_binary bmi_binary apnea_binary

# Execute directly
python scripts/gen_commands.py table-1 | bash
```

---

## gen_tables.sh — All Tables in One Script

`scripts/gen_tables.sh` runs all available table scripts sequentially. It mirrors the structure of `run_analysis.sh`.

### Single experiment (Tables 3 and 9 use the exp_id; Tables 1,2,4,5,10 use all tasks):
```bash
bash scripts/gen_tables.sh sex_binary_lstm
```

### Multiple experiments, restrict multi-task tables to specific tasks:
```bash
bash scripts/gen_tables.sh sex_binary_lstm sex_binary_transformer bmi_binary_lstm \
    --tasks sex_binary bmi_binary
```

### Select which tables to generate:
```bash
# Only K grid and cohort tables
bash scripts/gen_tables.sh sex_binary_lstm bmi_binary_lstm --tables 3 9

# Only multi-task tables (no exp-specific ones)
bash scripts/gen_tables.sh --tables 1 2 4 5 10
```

### Full channel:
```bash
bash scripts/gen_tables.sh sex_binary_lstm bmi_binary_lstm \
    --registry experiments/v2_full_registry.yaml
```

### Dry run (print commands without executing):
```bash
bash scripts/gen_tables.sh sex_binary_lstm bmi_binary_lstm apnea_binary_lstm --dry-run
```

---

## Loop Recipe — Run Tables for Multiple Tasks

To generate per-experiment tables (3 and 9) for every task, loop over tasks and call gen_tables.sh with the lstm experiment per task:

```bash
TASKS=(sex_binary bmi_binary apnea_binary age_class sleep_efficiency_binary
       cvd_binary sleepiness_binary depression_extreme_binary
       osa_binary_apples_postqc psqi_binary)

for TASK in "${TASKS[@]}"; do
    echo "=== $TASK ==="
    bash scripts/gen_tables.sh "${TASK}_lstm" \
        --tasks "$TASK" \
        --tables 3 9
done
```

To regenerate all multi-task tables once and then loop for per-task tables:

```bash
# Step 1: multi-task tables (run once, uses all tasks)
bash scripts/gen_tables.sh --tables 1 2 4 5 10

# Step 2: per-task tables (loop over each experiment)
for EXP in sex_binary_lstm sex_binary_transformer bmi_binary_lstm bmi_binary_transformer \
           apnea_binary_lstm apnea_binary_transformer age_class_lstm age_class_transformer \
           sleep_efficiency_binary_lstm sleep_efficiency_binary_transformer \
           cvd_binary_lstm sleepiness_binary_lstm depression_extreme_binary_lstm \
           osa_binary_apples_postqc_lstm psqi_binary_lstm; do
    bash scripts/gen_tables.sh "$EXP" --tables 3 9
done
```

To generate all tables for both fast and full channels:

```bash
for REGISTRY in experiments/v2_registry.yaml experiments/v2_full_registry.yaml; do
    bash scripts/gen_tables.sh sex_binary_lstm bmi_binary_lstm apnea_binary_lstm \
        --registry "$REGISTRY" \
        --tables 1 2 4 5 10
done
```

---

## Output Files

All table scripts write to `results/tables/` (committed to the repo):

```
results/tables/
  table1_peak_auroc_fast.csv     ← machine-readable
  table1_peak_auroc_fast.md      ← GitHub-renderable markdown
  table1_peak_auroc_fast.tex     ← LaTeX (booktabs)
  table1_peak_auroc_full.csv     ← full-channel version
  ...
  table2_lstar_fast.{csv,md,tex}
  table3_kgrid_sex_binary_lstm_fast.{csv,md,tex}
  table3_kgrid_bmi_binary_lstm_fast.{csv,md,tex}
  table4_sensitivity_fast_lstm.{csv,md,tex}
  table5_heads_fast.{csv,md,tex}
  table9_cohort_sex_binary_lstm_fast.{csv,md,tex}
  table10_ci_fast.{csv,md,tex}
```

**Quick view in VSCode:** Open any `.md` file and press `Ctrl+Shift+V` (or `Cmd+Shift+V` on Mac) to preview the rendered table.
