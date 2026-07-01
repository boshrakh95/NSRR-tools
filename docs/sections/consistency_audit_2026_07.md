# Consistency Audit — Figures, Tables, and TeX — July 2026

This document reports the results of a systematic consistency audit across all
figure-generation scripts, all table-generation scripts, and the manually written
TeX tables in `TBME_submission/generic-color.tex` and
`TBME_submission/supplementary.tex`.

---

## 1. Scope

For each figure and table, I audited:
- **Data source** (which file is read)
- **K value** (number of inference windows aggregated per subject)
- **Split** (always `test` unless noted)
- **Metric** (column name and what it represents)

I then compared values in the TeX manuscript against the current
`phase0_v3` `analysis.csv` to detect version mismatches, stale numbers, or
copy-paste errors.

---

## 2. Figure Scripts — Audit Results

| Figure | Script | Data Source | K | Metric | Notes |
|---|---|---|---|---|---|
| Fig 1 (saturation) | `plot_saturation.py` 1A | `analysis.csv` | `k='all'` | `mean_prob_auroc` | **Fixed in July session** (was `test_auroc` from `summary.csv`) |
| S-Fig 8 (compute scaling) | `plot_scaling_laws.py` 1B | `training.csv` | N/A (training eval) | `test_auroc` (segment-level) | ⚠️ See §4-B |
| Fig 2 (iso-compute heatmap) | `plot_iso_compute.py` | `heatmap_df_test.csv` | dense 1…K_max | `auroc` col | Confirmed = `mean_prob_auroc` (see §2a) |
| Fig 3 (task landscape 6A/6C) | `plot_task_comparison.py` | `analysis.csv` | `k='all'` | `mean_prob_auroc` | ✓ |
| Fig 4 (modality bar) | `plot_modality_bar.py` | `analysis.csv` | `k='all'` | `mean_prob_auroc` | ✓ |
| Fig 5 / S-Fig 12 (aggregate scaling) | `plot_aggregate_scaling.py` | `analysis.csv` | `k='all'` | `mean_prob_auroc` | ✓ New script |
| S-Fig 2 (channel compare) | `plot_channel_comparison.py` | `analysis.csv` | `k='all'` | `mean_prob_auroc` | ✓ |
| S-Fig (calibration 2A/2B) | `plot_calibration.py` | parquet windows | **K=10** | ECE / calibration | ⚠️ See §4-C |
| S-Fig (PR curves 8A/8B) | `plot_precision_recall.py` | parquet windows | K=all | AUC-PR | ✓ |
| S-Fig (subject consistency 5A/5C) | `plot_subject_consistency.py` | parquet windows | K=all | correct classification | ✓ |
| S-Fig (cohort saturation 7A) | `plot_cohort_saturation.py` | parquet windows | K=all | mean_prob_auroc (recomputed) | ✓ See §2b |
| S-Fig (window position 4A/4B) | `plot_window_position.py` | parquet windows | K=all | mean_prob_auroc | ✓ |
| S-Fig (K* histogram 9A) | `plot_subject_kstar.py` | parquet windows | 1…K_MAX | K* statistic | ✓ (by design, sweeps K) |

### 2a. `heatmap_df` `auroc` column = `mean_prob_auroc`

The iso-compute figures read `auroc` from `heatmap_df_{split}.csv`. I verified
numerically that this column equals the subject-level mean-probability AUROC:

```
sex_binary / transformer / 30s / K≈1062 (K=all):
  heatmap_df  auroc          = 0.831752
  analysis.csv mean_prob_auroc = 0.831752  ✓
```

The naming divergence (`auroc` vs `mean_prob_auroc`) is an internal codebase
inconsistency but does not affect correctness. The column `seg_auroc` in
`heatmap_df` is the segment-level AUROC and is distinct.

### 2b. Cohort saturation re-derives from parquets

`plot_cohort_saturation.py` (7A) reads raw parquet files and recomputes
per-dataset AUROC using mean-probability aggregation over all windows (K=all).
This is necessary because `analysis.csv` does not preserve per-dataset rows.
The AUROC computation is identical to the paper's primary metric, so values
should agree with `analysis.csv` to within rounding when summed across datasets.
**No inconsistency.**

---

## 3. Table Scripts — Audit Results

| Table | Script | Data Source | K | Metric | Notes |
|---|---|---|---|---|---|
| Table 1 (peak AUROC) | `make_table1_peak_auroc.py` | `analysis.csv` | K=5 AND K=all (separate best L per K) | `mean_prob_auroc` | ✓ |
| Table 2 (L*) | `make_table2_lstar.py` | `analysis.csv` | `k='all'` | `mean_prob_auroc` | ✓ |
| Table 3 (K grid) | `make_table3_kgrid.py` | `analysis.csv` | 1, 5, 10, 20, 50, all | `mean_prob_auroc` | ✓ |
| Table 4 (sensitivity) | `make_table4_sensitivity.py` | `analysis.csv` | `k='all'` | `mean_prob_auroc` | ✓ |
| Table 5 (heads) | `make_table5_heads.py` | `analysis.csv` | K=5 AND K=all at LSTM L* | `mean_prob_auroc` | ⚠️ See §4-D |
| Table 6 (modality) | `make_table6_modality.py` | `analysis.csv` | `k='all'` | `mean_prob_auroc` | ⚠️ See §4-E |
| Table 9 (cohort) | `make_table9_cohort.py` | parquet windows at L* | K=all | mean-prob AUROC (recomputed) | ✓ |
| Table 10 (CI) | `make_table10_ci.py` | `analysis.csv` | K=5 AND K=all at L* | `mean_prob_auroc` [95% CI] | ✓ |

All table scripts use `split='test'` by default. ✓

---

## 4. Known Issues

### 4-A. BUG (FIXED): `plot_saturation.py` was using wrong data source

**Status: FIXED in July 2026 session.**

Before the fix, `plot_saturation.py` read `test_auroc` from `summary.csv`, which
is the segment-level AUROC from the training evaluation loop (effectively K=1).
The paper's primary metric is `mean_prob_auroc` from `analysis.csv` at K=all.

Example discrepancy before fix:
```
sex_binary / transformer / 240m:
  summary.csv  test_auroc      = 0.892   (segment-level, K=1)
  analysis.csv mean_prob_auroc = 0.910   (subject-level, K=all)
```

The fix added `load_from_analysis()` and changed the default data path to
`{results_dir}/collected`. All saturation curves now use the correct metric.

---

### 4-B. DESIGN NOTE: `plot_scaling_laws.py` 1B uses segment-level AUROC

**Status: Intentional but must be captioned carefully.**

Figure 1B ("AUROC vs compute at best epoch") reads `test_auroc` from
`training.csv`. This is the segment-level AUROC recorded during the training
evaluation loop (evaluated at the end of each epoch on the validation set; here
reported on test set at best epoch). It is **not** the same as the paper's primary
metric `mean_prob_auroc` at K=all.

Why this is appropriate: 1B plots the compute scaling law — how training-time
AUROC scales with total FLOPs at the best checkpoint. This requires the metric
that was actually computed during training, not an inference-time post-aggregation.

**Risk**: If any number from 1B (e.g., "sex_binary transformer at best epoch = X")
is cited in the text alongside Table 1 or Table 2 numbers, it will differ because
different metrics are used. The figure caption must clearly state:
"AUROC at best training epoch (evaluated on non-overlapping windows, before
subject-level aggregation); not directly comparable to Table I/II K=all values."

---

### 4-C. DESIGN NOTE: Calibration figures use K=10, not K=5

**Status: Intentional choice, needs caption note.**

`plot_calibration.py` 2A (reliability diagrams) and 2B (ECE vs context) fix
`K=10` windows per subject. The "deployment scenario" used in Table 1 and
elsewhere is K=5. This creates a mild inconsistency: the calibration figures
do not correspond to the same inference scenario as the main performance tables.

K=10 is a defensible choice for calibration (more averaging → smoother
probabilities → better-characterised reliability), but should be stated
explicitly. Recommended caption addition:
"Probabilities are aggregated from K=10 randomly sampled windows per subject.
See Table I for K=5 (deployment) and K=all (ceiling) AUROC values."

---

### 4-D. DESIGN NOTE: Table 5 uses LSTM L* for all heads

**Status: Intentional (fair comparison), needs caption note.**

`make_table5_heads.py` reports AUROC at the **LSTM's** saturation context L*
for all three heads (LSTM, Transformer, MeanPool). This allows within-row
comparison at a fixed context. As a consequence, the Transformer AUROC in
Table 5 may be lower than the Transformer's own peak (which occurs at its own
best context).

Example (sex_binary):
```
LSTM L* = 120m
Transformer@120m (K=5) = 0.905   ← Table 5 value
Transformer@240m (K=5) = n/a (only 3 windows available)
Transformer best = 0.910 (K=all, 240m)  ← Table 1 value
```

Table 5 and Table 1 both show 0.905 for sex_binary Transformer because the
LSTM's L*=120m is also the Transformer's best K=5 context. For tasks where
LSTM L* < Transformer best context, Table 5 will show a lower value than Table 1.
The caption already notes "evaluated at LSTM's L*" but this should be called out
explicitly in the text to pre-empt reader confusion.

---

### 4-E. DESIGN NOTE: Table 6 / Fig 4 use ablation context lengths

**Status: Intentional, needs caption note.**

The modality ablation table (Table 6) and modality bar figure (Fig 4) report
AUROC at task-specific ablation context lengths:
- Sex, apnea, sleep_efficiency, age_group: 120m
- BMI: 40m

Table 1 reports best AUROC across all contexts. As a result, numbers in
Table 6 ("Full" column) will be lower than Table 1 for tasks that peak
above the ablation context. Example:
```
sleep_efficiency lstm:
  Table 1 peak (240m, K=all) = 0.788
  Table 6 Full (120m, K=all) = 0.778
```

This is by design (ablations were trained at these fixed contexts), but if
a reader compares Table 6 "Full" to Table 1 without reading captions they
will see different numbers for "LSTM on sleep_efficiency". Caption must state
the context used.

---

## 5. TeX Manuscript Number Discrepancies

**CRITICAL FINDING: TeX numbers are from multiple different analysis snapshots,
not all consistent with the current phase0_v3 `analysis.csv`.**

### 5-A. Sleep efficiency LSTM AUROC — WRONG IN ALL TEX TABLES

The current `phase0_v3` `analysis.csv` gives:

```
sleep_efficiency_binary / lstm / 240m / K=all  →  mean_prob_auroc = 0.788
sleep_efficiency_binary / lstm / 240m / K=5    →  mean_prob_auroc = 0.788
sleep_efficiency_binary / lstm / 30s  / K=all  →  mean_prob_auroc = 0.697
ΔAUROC = 0.788 − 0.697 = +0.091
```

The TeX manuscript shows **0.799** in every location this value appears:

| Location | TeX value | Correct value | Error |
|---|---|---|---|
| Main paper Results text | "0.699 at 30 s to 0.799 at 240 min" | 0.697 → 0.788 | +0.011 |
| Table I (tab:performance) AUROC@K=5 | 0.799 | 0.788 | +0.011 |
| Table I (tab:performance) AUROC@Kmax | 0.799 | 0.788 | +0.011 |
| Table II (tab:lstar) ΔAUROC | +0.102 | +0.091 | +0.011 |
| Supp Table S-sensitivity | Best=0.799, Δ=+0.102 | Best=0.788, Δ=+0.091 | +0.011 |
| Supp Table S-CI (tab:supp-ci) | 0.799 [PENDING] | 0.788 | +0.011 |

**Likely cause**: The sleep_efficiency_binary_lstm experiment was rerun between
when the TeX tables were written and the current state of analysis.csv. The
number 0.799 is consistent with an earlier checkpoint or run. The current best
value is 0.788.

**Action required**: Update all occurrences of 0.799 → 0.788, and +0.102 → +0.091,
for sleep efficiency LSTM in the TeX manuscript after verifying that analysis.csv
is final.

---

### 5-B. Sex LSTM L* — WRONG IN ALL TEX TABLES

The current `phase0_v3` `analysis.csv` gives for sex_binary / lstm / K=all:

```
80m  AUROC = 0.861
120m AUROC = 0.872  ← best
Best = 0.872, threshold = 0.872 − 0.005 = 0.867
80m: 0.861 < 0.867  → does NOT qualify
120m: 0.872 ≥ 0.867 → qualifies  →  L* = 120m
```

The TeX manuscript shows **L*=80m** in every location:

| Location | TeX value | Correct value |
|---|---|---|
| Table II (tab:lstar) | L*=80m | L*=120m |
| Main paper Results text | "(L*=80 min)" | "(L*=120 min)" |
| Supp Table S-CI | L*=80m | L*=120m |
| Supp Table S-sensitivity | L*=80m | L*=120m |
| Supp cohort table caption | "at L*=80min (LSTM)" | "at L*=120min (LSTM)" |

**Note**: The AUROC values (0.872 K=5, 0.872 Kmax) and ΔAUROC (+0.047) in all
TeX tables are **correct** — these values don't change with L*.

**Note on cohort table**: The per-cohort AUROC values reported (All=0.861,
SHHS=0.868, APPLES=0.782, N_test=1430) are from the 80m context, not 120m.
After correcting L* to 120m, these numbers will also need to be regenerated
from the 120m context parquet.

---

### 5-C. Age LSTM L* — WRONG IN TEX, AND INCONSISTENT WITHIN TEX

The current `phase0_v3` `analysis.csv` gives for age_class / lstm / K=all:

```
40m  AUROC = 0.887
80m  AUROC = 0.890
120m AUROC = 0.893  ← best
Best = 0.893, threshold = 0.893 − 0.005 = 0.888
40m: 0.887 < 0.888  → does NOT qualify (misses by 0.001!)
80m: 0.890 ≥ 0.888  → qualifies  →  L* = 80m
```

The TeX manuscript is **internally inconsistent**:

| Location | TeX value | Correct value |
|---|---|---|
| Table II (tab:lstar) | L*=40m | L*=80m |
| Main paper Results text | "(L*=40 min)" | "(L*=80 min)" |
| Supp Table S-sensitivity | L*=40m | L*=80m |
| Supp Table S-CI (tab:supp-ci) | **L*=120m** | L*=80m |

The supplementary CI table has yet another value (120m), making three different
L* values for age LSTM across the manuscript: 40m (main text / Table II), 120m
(supp CI table), and 80m (correct from current data).

**Note**: The AUROC (0.893 K=5, 0.893 Kmax) and ΔAUROC (+0.028) are **correct**.

---

### 5-D. Supplementary K-grid Table (tab:supp-kgrid) — FROM PHASE0_V2, NOT PHASE0_V3

The supplementary K-grid table (sex_binary lstm, all contexts and K values) was
verified to match `phase0_v2` results exactly, **not** the current `phase0_v3`.
Selected comparison:

| L | K | Supp TeX | phase0_v2 | phase0_v3 | Diff |
|---|---|---|---|---|---|
| 30s | 1 | 0.701 | 0.701 ✓ | 0.687 ✗ | +0.014 |
| 10m | 5 | 0.823 | 0.823 ✓ | 0.831 ✗ | −0.008 |
| 40m | 5 | 0.853 | 0.853 ✓ | 0.838 ✗ | +0.015 |
| 80m | 5 | 0.866 | 0.866 ✓ | 0.859 ✗ | +0.007 |
| 120m | K=all | 0.855 | 0.855 ✓ | 0.872 ✗ | −0.017 |
| 240m | K=all | 0.812 | 0.812 ✓ | 0.857 ✗ | −0.045 |

The differences at 240m are especially large (−0.045 for K=all, −0.053 for K=1).
Phase0_v3 is the current production experiment; this supplementary table must be
regenerated using `make_table3_kgrid.py sex_binary_lstm`.

---

### 5-E. TeX Numbers That Are Correct

The following TeX table entries were verified to match current `phase0_v3`
`analysis.csv` exactly (within ±0.001 rounding):

- **Table I**: All tasks except sleep_efficiency LSTM. ✓
- **Table II**: All ΔAUROC values. ✓ All L* values except Sex LSTM and Age LSTM. ✓
- **Main text numbers**: Apnea LSTM and Transformer, Sex Transformer, Age Transformer,
  BMI both heads, Depression LSTM, OSA LSTM. All correct. ✓
- **Supp CI table**: All AUROC values (CIs are all [PENDING], so no CI numbers to check). ✓

---

## 6. Summary Table — All Issues

**Legend:** ✅ = implemented in script · 📄 = TeX-only (for paper-writing agent) · ⬜ = no action needed

| # | Severity | Status | Location | Issue | Action taken / required |
|---|---|---|---|---|---|
| 1 | 🟢 Bug | ✅ Fixed July 2026 | `plot_saturation.py` | Was using `test_auroc` (segment K=1) instead of `mean_prob_auroc` (K=all) | Fixed: switched to `analysis.csv` `mean_prob_auroc` at K=all |
| 2 | 🔴 Critical | 📄 Pending TeX | TeX manuscript (multiple) | Sleep eff. LSTM AUROC = **0.799** everywhere; correct = **0.788**. ΔAUROC = +0.102; correct = +0.091 | **Paper agent**: replace 0.799→0.788 and +0.102→+0.091 in Table I, Table II, Results text, Supp S-sensitivity, Supp S-CI |
| 3 | 🟠 Important | 📄 Pending TeX | TeX Table II + supp tables | Sex LSTM L* = **80m**; correct = **120m** (80m AUROC=0.861 < threshold 0.867) | **Paper agent**: replace L*=80m→120m for sex LSTM in Table II, Results text, Supp S-CI, Supp S-sensitivity |
| 4 | 🟠 Important | 📄 Pending TeX | TeX Supp cohort table | Per-cohort AUROC for sex LSTM is at **80m** context (old L*): Overall=0.861, SHHS=0.868, APPLES=0.782 | **Paper agent**: run `make_table9_cohort.py sex_binary_lstm --context 120m`; replace table numbers |
| 5 | 🟠 Important | 📄 Pending TeX | TeX Table II + Supp S-CI | Age LSTM L* inconsistent within TeX: 40m (Table II / S-sensitivity), 120m (Supp S-CI); correct = **80m** | **Paper agent**: standardise to 80m in Table II, Supp S-CI, Supp S-sensitivity, Results text |
| 6 | 🟡 Moderate | 📄 Pending TeX | Supp K-grid table (tab:supp-kgrid) | All values from **phase0_v2**, not phase0_v3; differences up to 0.045 AUROC at 240m | **Paper agent**: run `make_table3_kgrid.py sex_binary_lstm --latex` and paste output into TeX |
| 7 | 🔵 Design | ✅ Fixed July 2026 | `plot_calibration.py` 2A/2B | Default K was **10**; paper deployment scenario is **K=5** | **Done**: `--k` default changed 10→5; docstring updated (`code_changes_2026_07.md` §P6) |
| 8 | 🔵 Design | ✅ Fixed July 2026 | `plot_scaling_laws.py` 1B | Y-axis "Test AUROC at best epoch" ambiguous vs subject-level `mean_prob_auroc` | **Done**: label changed to "Segment AUROC at best epoch (training eval)" |
| 9 | 🔵 Design | ✅ Fixed July 2026 | `make_table5_heads.py` generated caption | Caption did not warn Transformer@Table5 may < Transformer@Table1 | **Done**: caption now states all heads evaluated at LSTM L* and refers reader to Table I for peaks |
| 10 | 🔵 Design | ✅ Fixed July 2026 | `make_table6_modality.py` generated caption | Caption did not state per-task context lengths or warn values differ from Table I | **Done**: caption now states "120m/40m per task; values differ from Table I best-context" |
| 11 | 🔵 Internal | ⬜ No action needed | `heatmap_df_test.csv` | Column `auroc` = `mean_prob_auroc` but named differently from `analysis.csv` | Confirmed equivalent numerically; internal naming only, no reader impact |

---

## 7. Remaining Work for Paper-Writing Agent

Script-level fixes (issues 1, 7–10) are complete. The following require
**manual edits to the TeX source only** — the underlying data and code are correct.

### Step 1 — Sleep efficiency LSTM AUROC (issue 2)

Replace the stale value **0.799 → 0.788** and **+0.102 → +0.091** in:
- `generic-color.tex`: Results §IV-A text ("0.697 at 30 s to 0.799 at 240 min")
- `generic-color.tex`: Table I (`tab:performance`) sleep efficiency LSTM row, both AUROC columns
- `generic-color.tex`: Table II (`tab:lstar`) sleep efficiency LSTM ΔAUROC cell
- `supplementary.tex`: Table S-sensitivity (`tab:supp-sensitivity`) sleep eff. row
- `supplementary.tex`: Table S-CI (`tab:supp-ci`) sleep eff. LSTM AUROC cell

### Step 2 — Sex LSTM L* (issue 3)

Replace **L*=80m → L*=120m** for sex LSTM in:
- `generic-color.tex`: Results §IV-A text ("L*=80 min")
- `generic-color.tex`: Table II (`tab:lstar`) sex row, LSTM column
- `supplementary.tex`: Table S-CI (`tab:supp-ci`) sex LSTM L* cell
- `supplementary.tex`: Table S-sensitivity sex row L* cell

### Step 3 — Sex LSTM cohort table (issue 4)

The supplementary cohort table used **80m context** parquets.
Run: `python scripts/make_table9_cohort.py sex_binary_lstm --context 120m --latex`
Replace `tab:supp-cohort` numbers (Overall, SHHS, APPLES AUROC and N values) and
update the caption "at L*=80min" → "at L*=120min".

### Step 4 — Age LSTM L* (issue 5)

Replace the inconsistent L* values; correct value = **80m**:
- `generic-color.tex`: Results text ("L*=40 min" for age)
- `generic-color.tex`: Table II age LSTM L* cell (currently 40m)
- `supplementary.tex`: Table S-sensitivity age row L* (currently 40m)
- `supplementary.tex`: Table S-CI age LSTM L* cell (currently **120m** — different error)

### Step 5 — Regenerate K-grid table (issue 6)

The supplementary K-grid table uses phase0_v2 data. Run:
```
python scripts/make_table3_kgrid.py sex_binary_lstm \
    --collected-dir results/collected/phase0_v3 --latex
```
Copy the output into `supplementary.tex` table `tab:supp-kgrid`.
Also update the in-text references to specific K-grid numbers in §IV-C
(e.g. "0.701 at K=1 to 0.824 at K=Kmax" at 30s → "0.687 at K=1 to 0.825").

---

*Report generated: July 2026. Last updated: July 2026 (issues 1, 7–10 implemented).*
*Data source: `/scratch/boshra95/psg/unified/results/phase0_v3/collected/analysis.csv`.*
