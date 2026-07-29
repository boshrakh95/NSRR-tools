# Code Changes — July 2026

This file documents all figure-generation code changes made in the 2026-07 session.
Changes are grouped by priority as originally scoped.

---

## P1 — Bug Fix: `plot_saturation.py` data source

**Problem:** The script was reading `test_auroc` from `summary.csv` — the segment-level
AUROC from the training evaluation loop at K=1. All other paper figures use subject-level
mean-pool AUROC (`mean_prob_auroc`) from `analysis.csv` at `k='all'`. The difference was
material: e.g. sex_binary Transformer at 240m was 89.2% (summary) vs 91.0% (analysis).

**Fix — `scripts/plot_saturation.py`:**
- Added `METRIC_COL` dict mapping metric names to `analysis.csv` column names
  (`mean_prob_auroc`, `mean_prob_balanced_accuracy`)
- Added `load_from_analysis(collected_dir, task, head, split, metric, run_tag)` function
  that reads `analysis.csv`, filters to `k='all'` and `split='test'`, and returns
  `(context_length_min, value)` rows
- Falls back to `summary.csv` with a printed warning if `analysis.csv` not found
- `--collected-dir` now defaults to `{results_dir}/collected` (always present)
- Updated rcParams: `figure.dpi=300`, `font.family='serif'`, `font.size=9`
- Updated `HEAD_STYLE` to TBME palette (see §Style below)
- Removed `ax.set_title()`
- Updated axis labels: `"Context length (minutes)"`, `METRIC_LABEL.get(metric, metric)`

**Fix — `scripts/gen_commands.py` (`build_saturation_cmd`):**
- `--collected-dir` is now always injected pointing to `{results_dir}/collected`,
  matching the new default in `plot_saturation.py`

---

## P2 — New script: `scripts/plot_modality_bar.py`

Generates **Fig 4** (Modality Contribution) — a 1×5 grouped horizontal bar chart.

- **Layout:** 1 row × 5 cols, `figsize=(14, 3.5)`, `dpi=300`
- **Tasks (one panel each):** sex_binary (120m), apnea_binary (120m),
  sleep_efficiency_binary (120m), age_class (120m), bmi_binary (40m)
- **Five ablation conditions per panel:**
  - `abl_no_bas` — No brain/eye channels (RESP+EKG+EMG only) — orange `#E86A33`
  - `abl_no_resp` — No respiratory (BAS+EKG+EMG only) — blue `#3A7EBF`
  - `abl_no_ekg` — No cardiac (BAS+RESP+EMG only) — green `#44A15E`
  - `abl_cardio` — Cardio only (RESP+EKG) — red `#C94040`
  - `abl_bas_only` — Brain/eye only — purple `#7B5EA7`
- **Reference lines:** x=0 solid black (fast-ch baseline), full-ch delta as dashed gray
- **Δ values** annotated at bar ends in 7pt
- **Panel labels** (a)–(e) + task name below each panel
- **Outputs:** `modality_ablation_bar.{png,pdf}`
- **Default paths:** `--abl-dir results/collected/phase0_v3_abl`,
  `--fast-dir results/collected/phase0_v3`,
  `--full-dir results/collected/phase0_v3_full`

---

## P3 — New script: `scripts/plot_channel_comparison.py`

Generates **S-Fig 2** (Fast vs Full Channel) — a 2×3 overlay plot.

- **Layout:** 2 rows × 3 cols, `figsize=(13, 8)`, `dpi=300`
- **Tasks:** sex_binary, apnea_binary, sleep_efficiency_binary, bmi_binary,
  age_class, osa_binary_apples_postqc
- **Fast-channel:** dashed `--`, **Full-channel:** solid `-`, both Transformer color
  `#E86A33`
- Annotates Δ at 240m per panel
- OSA panel: "APPLES-only cohort" text annotation
- Shared legend via `fig.legend()` at top
- Log-scale x-axis with context-length tick labels
- **Outputs:** `channel_comparison.{png,pdf}`

---

## P4+P5 — TBME style applied to all existing `plot_*.py` scripts

### TBME style rules applied everywhere

```
figure.dpi      = 300
font.family     = 'serif'
font.size       = 9
axis labels     = fontsize=10
tick labels     = fontsize=9
panel letters   = fontsize=8, via ax.text(0.5, -0.18, f'({chr(97+i)})', ...)
```

**TBME HEAD_STYLE palette:**

| Head | Color | Line style |
|---|---|---|
| LSTM | `#3A7EBF` | solid `-` |
| Transformer | `#E86A33` | dashed `--` |
| MeanPool | `#44A15E` | dotted `:` |

All `ax.set_title()` and `fig.suptitle()` calls removed from every script.
Context/task labels moved to `ax.text()` below the x-axis.

---

### `scripts/plot_calibration.py`

- rcParams updated (DPI 150→300, serif, font.size 9)
- HEAD_STYLE colors updated to TBME palette
- `fig.suptitle()` removed from `plot_reliability_diagrams`
- Per-context `ax.set_title()` replaced with `ax.text()` below x-axis (includes ECE value)
- Axis labels: fontsize 11→10
- **2C removed from default `--plots`** (was `["2A","2B","2C"]`, now `["2A","2B"]`)

### `scripts/plot_precision_recall.py`

- rcParams updated (DPI 150→300, serif, font.size 9)
- HEAD_STYLE colors updated to TBME palette
- `ax.set_title()` removed from 8A, 8B, 8C plot functions
- Axis labels: fontsize 11→10
- **8C removed from default `--plots`** (was `["8A","8B","8C"]`, now `["8A","8B"]`)

### `scripts/plot_task_comparison.py`

- rcParams updated (DPI 150→300, serif, font.size 9)
- `ax.set_title()` removed from 6A, 6B, 6C
- Axis labels: fontsize 11→10
- **6B removed from default `--plots`** (was `["6A","6B","6C"]`, now `["6A","6C"]`)
- **Sleep efficiency arrow added to 6C:** annotates ">240m" with arrow when
  sleep_efficiency_binary has `lstar_min` at 240m (not saturated)

### `scripts/plot_iso_compute.py`

- rcParams block added at top (DPI 300, serif, font.size 9)
- `ax.set_title()` and `fig.suptitle()` removed from all 7 plot functions
  (heatmap, vs_k, vs_total, pareto, min_cost, marginal, double)
- Axis labels: "k" → "K", fontsize 12→10
- **`plot_double()` call commented out in the main loop** — function retained for
  reference but `double_tradeoff` is blacklisted

### `scripts/plot_subject_consistency.py`

- rcParams updated (DPI 150→300, serif, font.size 9)
- `fig.suptitle()` removed from 5A
- Per-context `ax.set_title()` in 5A replaced with `ax.text()` below x-axis
- 5B axis labels updated: "K (windows aggregated per subject)" → "K (windows per subject)"
- **5B removed from default `--plots`** (was `["5A","5B","5C"]`, now `["5A","5C"]`)
- **5C redesigned:**
  - x-tick labels changed from `f"{i}/{n_ctx}"` to `str(i)` (plain integer)
  - x-axis label changed to `f"Number of context lengths correctly predicted (out of {n_ctx})"`
  - Cumulative line added on a twin y-axis: fraction of subjects classified correctly
    at ≥i context lengths (black dashed, circle markers)

### `scripts/plot_cohort_saturation.py`

- rcParams updated (DPI 150→300, serif, font.size 9)
- `ax.set_title()` removed from 7A and 7B
- Axis labels: "Context Length (log scale)" → "Context length (minutes)", fontsize 11→10
- **7B removed from default `--plots`** (was `["7A","7B"]`, now `["7A"]`)

### `scripts/plot_window_position.py`

- rcParams updated (DPI 150→300, serif, font.size 9)
- `fig.suptitle()` and both `ax.set_title()` calls removed from 4A
- Panel labels added below x-axis: `(a) Positive subjects`, `(b) Negative subjects`
- `ax.set_title()` removed from 4B
- Axis labels: fontsize 11→10

### `scripts/plot_scaling_laws.py`

- rcParams updated (DPI 150→300, serif, font.size 9)
- HEAD_STYLE colors updated to TBME palette
- `fig.suptitle()` removed from 1A
- Per-context `ax_flat.set_title()` in 1A replaced with `ax_flat.text()` below x-axis
- `ax.set_title()` removed from 1B and 1C
- Axis labels: fontsize 11→10
- **1C removed from default `--plots`** (was `["1A","1B","1C"]`, now `["1A","1B"]`)

### `scripts/plot_subject_kstar.py`

- rcParams updated (DPI 150→300, serif, font.size 9)
- `fig.suptitle()` removed from 9A
- Per-context `ax.set_title()` in 9A replaced with `ax.text()` below x-axis
  (includes panel letter)
- Bar color updated: `#4C72B0` → `#3A7EBF`; annotation color: `#DD8452` → `#E86A33`
- `ax.set_title()` removed from 9B; axis labels: fontsize 11→10
- **9B removed from default `--plots`** (was `["9A","9B"]`, now `["9A"]`)

### `scripts/plot_context_comparison.py`

- rcParams block added at top (DPI 300, serif, font.size 9) — this is the older
  Phase 0 (pre-v3) comparison script; included for completeness
- `ax.set_title()` removed from `make_single_figure`
- `ax.set_title()` removed from `make_summary_figure`; replaced with `ax.text()` below x-axis
- `fig.suptitle()` removed from `make_summary_figure`
- `fig.savefig(dpi=150)` → `dpi=300` in both figure functions

---

## New script: `scripts/plot_aggregate_scaling.py`

Generates **S-Fig 12 / Fig 5 (TBD)** — aggregate context-length scaling analysis.
Placement (main vs supplementary) decided after viewing results.

**Figure layout:** 1 row × 3 panels, `figsize=(14, 4.5)`, `dpi=300`

**Panel (a) — ΔAUROC from 30s baseline:**
- X: context length (log scale, 30s–240m); Y: ΔAUROC in pp
- One bold line per head (TBME colors), ±1 std shading across tasks
- Individual per-task curves shown as thin faint lines (alpha=0.20) behind mean
- Reveals average gain per head and inter-task variability

**Panel (b) — Normalised gain:**
- Same structure; Y rescaled so 0% = 30s AUROC, 100% = 240m AUROC per task
- Tasks with total gain ≤ 0.005 are excluded from normalization (NaN)
- Shows the *shape* of the gain curve independent of how context-sensitive each task is
- Dashed gray reference line at 100% (ceiling)

**Panel (c) — Log-linear slope per head:**
- Slope b from OLS fit: ΔAUROC ~ a + b × log₂(context_min), fitted per (head, task)
- Bar = mean slope across tasks; error bar = ±1 std across tasks
- Individual task slope values shown as semi-transparent dots
- Y-axis: "ΔAUROC per log₂ doubling (pp)" — e.g. b=2.0 means each doubling of context
  gives ~2 pp on average

**Key CLI arguments:**
- `--collected-dir` — path to directory containing `analysis.csv` (default: `results/collected`)
- `--results-dir` — scratch results directory for output
- `--tasks` — explicit task list (default: all 7 retained tasks)
- `--exclude-tasks` — remove specific tasks from the average (e.g.
  `--exclude-tasks depression_extreme_binary` to test robustness without the
  non-monotonic outlier)
- `--heads` — heads to include (default: lstm transformer mean_pool)

**Output:** `{results_dir}/figures/aggregate/aggregate_scaling.{png,pdf}`

**Also prints** a slope summary table to stdout for quick reference.

---

---

## P6 — Consistency audit fixes (from `consistency_audit_2026_07.md`)

Four script-level fixes applied after a cross-audit of figures, tables, and the
TeX manuscript. No TeX files were changed here — those corrections are handled
separately by the paper-writing agent.

### `scripts/plot_calibration.py` — default K 10 → 5

**Problem (audit issue #7):** The calibration figures (2A reliability diagram,
2B ECE vs context) used K=10 windows per subject by default, while the paper's
deployment scenario is K=5 everywhere else (Table I, Table II, etc.).

**Fix:**
- `--k` argument default changed from `10` → `5`
- Docstring updated: "at a fixed K (default: K=5 windows per subject, matching
  the paper's deployment scenario)"
- K=10 remains available via `--k 10` for any downstream comparison.

---

### `scripts/plot_scaling_laws.py` — y-axis label for 1B

**Problem (audit issue #8):** The y-axis label on the compute scaling law plot
(1B) read `"Test AUROC at best epoch"`, which a reader could confuse with the
paper's primary metric (`mean_prob_auroc` at K=all from `analysis.csv`). The
1B metric is the segment-level AUROC from the training evaluation loop, not
subject-level aggregated AUROC.

**Fix:**
- Y-axis label changed to `"Segment AUROC at best epoch (training eval)"`
- This makes the metric distinction explicit in the figure itself.

---

### `scripts/make_table5_heads.py` — generated LaTeX caption

**Problem (audit issue #9):** The generated table caption did not explicitly
warn readers that Transformer and MeanPool AUROC values in Table 5 may be lower
than those in Table I because Table 5 evaluates all heads at the LSTM's L*, not
each head's own best context.

**Fix:** Caption updated to include:
> "All heads are evaluated at this same L\*. Consequently, Transformer and
> MeanPool AUROC values here may be lower than their own-best-context peak
> reported in Table I."

---

### `scripts/make_table6_modality.py` — generated LaTeX caption

**Problem (audit issue #10):** The generated table caption did not state the
per-task context lengths used for the ablation comparison, making it impossible
for a reader to understand why Table 6 "Full" values differ from Table I peaks.

**Fix:** Caption updated to include:
> "Context lengths are fixed at the ablation training context per task
> (Sex/Apnea/Sleep-eff/Age: 120 min; BMI: 40 min); values therefore differ
> from Table I which reports each head's best context."
>
> Condition descriptions also cleaned up; SleepFounder aside moved to prose.

---

---

## P7 — Bug fix: `plot_scaling_laws.py` 1A uses BA instead of loss

**Problem:** The 1A uShape plot showed CE loss vs epoch on the y-axis, but the training
early-stopping criterion is balanced accuracy (`val_bal_acc`). The plot was
therefore showing a different signal than what determined the best checkpoint.
All 21 uShape files (7 tasks × 3 heads) were flagged `[FLAG: CODE CHANGE]` in
`figure_interpretations_v3.md`.

**Fix — `scripts/plot_scaling_laws.py` `plot_uShape()`:**
- `train_loss` / `val_loss` → `train_bal_acc` / `val_bal_acc` (columns from `training.csv`)
- Y-axis label: `"Loss"` → `"Balanced Accuracy"`
- Legend labels: `"Train loss"` / `"Val loss"` → `"Train BA"` / `"Val BA"`
- `ax_flat.set_ylim(bottom=0.0)` added (BA is bounded 0–1)
- Generalisation-gap fill_between condition flipped:
  for loss `where=vl > tr` (val loss > train loss = overfit);
  for BA `where=tr > vl` (train BA > val BA = overfit)
- Docstring updated to describe BA curves

**After fix:** `run_figures.sh` Step 3 updated to `--plots 1A 1B` (was `--plots 1B`).
`paper_figures.md` S-Fig 11 status updated to "Ready".

---

## P8 — Bug fix: `gen_commands.py` defaults were overriding plot-script exclusions

**Problem:** In the July 2026 session, all plot scripts had their `--plots`/`--metric`
defaults updated to exclude blacklisted outputs (2C, 5B, 6B, 7B, 8C, 9B,
`balanced_accuracy` metric). However, `gen_commands.py` still had the **old stale**
defaults in three places: the `build_*_cmd()` functions, the `cmd_*()` handlers,
and the argparse `add_argument()` defaults. When `run_figures.sh` called
`gen_cmd subcommand ...` without an explicit `--plots`, `gen_commands.py` injected
the old list (e.g. `--plots 2A 2B 2C`) which overrode the plot script's correct
default. The result: blacklisted figures were being silently generated every run.

**Fix — `scripts/gen_commands.py`** (all three layers per subcommand):

| Subcommand | Old default | New default |
|---|---|---|
| `iso-plots` `--metric` | `auroc balanced_accuracy` | `auroc` |
| `saturation` `--metric` | `auroc balanced_accuracy` | `auroc` |
| `scaling-laws` `--plots` | `1A 1B 1C` | `1A 1B` |
| `calibration` `--plots` | `2A 2B 2C` | `2A 2B` |
| `subject-consistency` `--plots` | `5A 5B 5C` | `5A 5C` |
| `task-comparison` `--plots` | `6A 6B 6C` | `6A 6C` |
| `cohort-saturation` `--plots` | `7A 7B` | `7A` |
| `precision-recall` `--plots` | `8A 8B 8C` | `8A 8B` |
| `subject-kstar` `--plots` | `9A 9B` | `9A` |

**Fix — `scripts/run_figures.sh`:** All steps now also pass explicit `--plots`/`--metric`
flags so the pipeline cannot silently regress if gen_commands.py defaults drift again.

---

## Blacklisted outputs (never generated by default)

These plot functions still exist in the scripts but are excluded from default `--plots`
and should not appear in the paper:

| Output | Script | Reason |
|---|---|---|
| `*_calibration_2C_ece_vs_k` | `plot_calibration.py` | Blacklisted per instructions |
| `*_subject_consistency_5B_variance_vs_k` | `plot_subject_consistency.py` | Blacklisted |
| `task_comparison_6B_bars` | `plot_task_comparison.py` | Redundant with Fig 1 + Table II |
| `*_pr_8C_vote_sweep` | `plot_precision_recall.py` | Majority-vote removed from paper |
| `*_kstar_9B_coverage` | `plot_subject_kstar.py` | Blacklisted |
| `*_cohort_saturation_7B_n` | `plot_cohort_saturation.py` | N documented in methods only |
| `double_tradeoff` | `plot_iso_compute.py` | Redundant with heatmap + pareto |
| `*_1C_optimal_epoch` | `plot_scaling_laws.py` | Non-monotonic; not interpretable |
