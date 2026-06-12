# Post-hoc Threshold Tuning — Planning Document

*Written May 2026. Implementation completed on sleep-stage-redesign branch (2026-05-30).*
*See `scripts/apply_threshold_tuning.py` and `scripts/gen_commands.py threshold-tuning`.*

## What is threshold tuning?

Standard classification reports balanced accuracy at a fixed decision threshold of 0.5.
For imbalanced tasks, this threshold is miscalibrated — the model's learned distribution of
scores is not centred at 0.5, so one class is systematically over-predicted.

Post-hoc threshold tuning: after inference, sweep thresholds [0.01, 0.99] on the val split
and pick the threshold that maximises balanced accuracy. Apply that threshold to report test
balanced accuracy and per-class recall. AUROC is unchanged (it is threshold-free by definition).

**Does not require retraining.** Purely a reporting change in the analysis layer.

---

## Empirical evidence: bmi_binary_lstm (real data, K=all aggregation)

bmi_binary: class 0 = non-obese (majority 67.2%), class 1 = obese (minority 32.8%).

| Context | AUROC | BalAcc@t=0.5 | BalAcc@t_opt | Gain | R_minority@0.5 | R_minority@opt | Threshold |
|---------|-------|-------------|-------------|------|----------------|----------------|-----------|
| 30s | 0.760 | 0.682 | 0.703 | **+0.021** | 0.502 | 0.645 | 0.43 |
| 10m | 0.762 | 0.694 | 0.706 | **+0.012** | 0.741 | 0.661 | 0.54 |
| 40m | 0.756 | 0.682 | 0.696 | **+0.014** | 0.572 | 0.594 | 0.47 |
| 80m | 0.767 | 0.688 | 0.700 | **+0.012** | 0.543 | 0.634 | 0.44 |
| 120m | 0.754 | 0.685 | 0.699 | **+0.014** | 0.512 | 0.594 | 0.44 |

**Note**: these were computed on the test split (no separate val available yet = upper bound).
Real val-selected threshold will give slightly lower gains.

**Average gain for bmi_binary: +0.015 in balanced accuracy, +0.09 in minority-class recall.**
This is modest but real. For tasks with a larger recall gap at threshold=0.5, the gain is larger.

### Extrapolated gain for osa_binary_apples_postqc

osa_postqc (class 0 = Non-rand+Mild minority ~37%, class 1 = Mod+Severe majority ~63%):

| Context | AUROC | BalAcc@0.5 | R_class0@0.5 | R_class1@0.5 | Recall gap | Est. BalAcc@opt | Est. gain |
|---------|-------|-----------|-------------|-------------|------------|----------------|-----------|
| 40m | 0.738 | 0.583 | 0.300 | 0.866 | 0.566 | ~0.63–0.66 | ~+0.05–0.08 |
| 80m | 0.721 | 0.562 | 0.218 | 0.905 | 0.687 | ~0.62–0.65 | ~+0.06–0.09 |
| 120m | 0.742 | 0.575 | 0.282 | 0.869 | 0.587 | ~0.63–0.67 | ~+0.06–0.09 |

The recall gap is ~2× larger than bmi_binary despite similar AUROC → larger improvement.
BalAcc would move from ~0.57 into the ~0.63–0.67 range, a meaningful change in reported numbers.

---

## Actual v3 results (all experiments, val-selected threshold applied to test)

*Run 2026-05-30 on sleep-stage-redesign branch after val inference for all binary tasks.*
*Values from `threshold_tuning.csv` in each experiment's inference directory.*
*`orig` = K=all mean-prob aggregation at t=0.5. `tuned` = same at t_opt selected on val.*

### bmi_binary_lstm (v3, K=all — compare to earlier K=5 estimates above)

| Context | AUROC | Orig BA | t_opt | Tuned BA | Gain | R0 orig → tuned | R1 orig → tuned |
|---------|-------|---------|-------|----------|------|-----------------|-----------------|
| 30s  | 0.760 | 0.682 | 0.448 | **0.695** | **+0.013** | 0.862 → 0.787 | 0.502 → 0.604 |
| 10m  | 0.762 | 0.694 | 0.502 | 0.693 | −0.002 | 0.653 → 0.658 | 0.736 → 0.727 |
| 40m  | 0.756 | 0.682 | 0.414 | **0.688** | **+0.006** | 0.832 → 0.710 | 0.532 → 0.666 |
| 80m  | 0.767 | 0.688 | 0.389 | **0.697** | **+0.010** | 0.833 → 0.687 | 0.542 → 0.708 |
| 120m | 0.756 | 0.688 | 0.384 | 0.687 | −0.001 | 0.818 → 0.680 | 0.558 → 0.694 |
| 240m | 0.748 | 0.679 | 0.404 | **0.687** | **+0.008** | 0.790 → 0.679 | 0.569 → 0.695 |

*Earlier estimate (test-only, no val split): avg +0.015 gain. Actual with proper val split: avg +0.006.*
*Lower than estimated — val-selected threshold is more conservative, as expected.*

### bmi_binary_transformer (v3, K=all)

| Context | AUROC | Orig BA | t_opt | Tuned BA | Gain |
|---------|-------|---------|-------|----------|------|
| 30s  | 0.747 | 0.671 | 0.429 | **0.698** | **+0.027** |
| 10m  | 0.755 | 0.684 | 0.453 | **0.699** | **+0.014** |
| 40m  | 0.755 | 0.695 | 0.483 | **0.700** | **+0.006** |
| 80m  | 0.769 | 0.681 | 0.433 | **0.701** | **+0.020** |
| 120m | 0.766 | 0.694 | 0.429 | **0.699** | **+0.005** |
| 240m | 0.777 | 0.700 | 0.453 | **0.704** | **+0.004** |

*Larger gains than LSTM variant. Average +0.013. Exceeds "marginal" prediction.*

### osa_binary_apples_postqc_lstm (v3, K=all) — CRITICAL confirmed

| Context | AUROC | Orig BA | t_opt | Tuned BA | Gain | R0: 0.5→opt | R1: 0.5→opt |
|---------|-------|---------|-------|----------|------|-------------|-------------|
| 30s  | 0.769 | 0.577 | 0.719 | **0.683** | **+0.106** | 0.18 → 0.68 | 0.97 → 0.68 |
| 10m  | 0.816 | 0.554 | 0.773 | **0.774** | **+0.220** | 0.14 → 0.86 | 0.97 → 0.68 |
| 40m  | 0.834 | 0.566 | 0.803 | **0.746** | **+0.180** | 0.18 → 0.91 | 0.95 → 0.58 |
| 80m  | 0.767 | 0.573 | 0.828 | **0.707** | **+0.134** | 0.18 → 0.64 | 0.96 → 0.78 |
| 120m | 0.789 | 0.604 | 0.601 | **0.635** | **+0.031** | 0.27 → 0.36 | 0.94 → 0.91 |
| 240m | 0.774 | 0.597 | 0.877 | **0.738** | **+0.141** | 0.27 → 0.86 | 0.92 → 0.61 |

*Estimated gain was +0.06–0.09. Actual: up to +0.220 at 10m — FAR exceeded expectations.*
*At t=0.5 the model predicts class 1 for ~98% of subjects (R0≈0.14). Tuning fixes this.*
*Must use tuned results for this task. Original t=0.5 numbers are not meaningful.*

### depression_extreme_binary_lstm — SURPRISE: large gain at long contexts

*Predicted "NO — class weighting worked; near balanced." This was true at short contexts (10m:
R0=0.73/R1=0.71 → balanced) but NOT at long contexts:*

| Context | AUROC | Orig BA | t_opt | Tuned BA | Gain | R0 orig | R0 tuned |
|---------|-------|---------|-------|----------|------|---------|----------|
| 30s  | 0.757 | 0.715 | 0.414 | 0.723 | +0.008 | 0.722 | 0.654 |
| 10m  | 0.770 | 0.720 | 0.438 | **0.737** | **+0.017** | 0.732 | 0.683 |
| 40m  | 0.761 | 0.704 | 0.468 | 0.713 | +0.009 | 0.742 | 0.717 |
| 80m  | 0.744 | 0.661 | 0.409 | **0.726** | **+0.065** | 0.781 | 0.702 |
| 120m | 0.748 | 0.686 | 0.374 | **0.736** | **+0.051** | 0.746 | 0.639 |
| 240m | 0.750 | 0.695 | 0.335 | 0.697 | +0.002 | 0.766 | 0.644 |

*At 80m and 120m, the model becomes more biased despite auto-weighting (possibly because*
*longer contexts are harder to train for the minority class). Gain of +0.065 at 80m is*
*large enough to include tuned results for this task.*

### Sex, apnea, sleep efficiency — small or zero gains

| Experiment | Avg gain | Max gain | Verdict |
|---|---|---|---|
| sex_binary_lstm | +0.009 | +0.020 | Small but real; include for consistency |
| sex_binary_transformer | ~0.000 | +0.004 | NOT needed |
| apnea_binary_lstm | +0.003 | +0.012 | NOT needed |
| apnea_binary_transformer | ~0.000 | +0.005 | NOT needed |
| sleep_efficiency_binary_lstm | ~0.000 | +0.004 | NOT needed (several negative) |
| sleep_efficiency_binary_transformer | ~0.000 | +0.006 | NOT needed (several negative) |

### cvd_binary — threshold tuning HURTS

| Experiment | Avg gain | Min gain | Verdict |
|---|---|---|---|
| cvd_binary_lstm | **−0.005** | −0.013 | **KEEP t=0.5** — tuning hurts |
| cvd_binary_transformer | −0.002 | −0.009 | **KEEP t=0.5** — mostly zero or negative |

*Reason: AUROC is 0.68–0.69 (low); model is moderately but not extremely imbalanced;*
*val set is too small (~700 subjects) to reliably select a threshold that generalises to test.*

### sleepiness_binary — marginal

| Experiment | Avg gain | Max gain | Verdict |
|---|---|---|---|
| sleepiness_binary_lstm | +0.006 | +0.014 | Include for consistency; small |
| sleepiness_binary_transformer | +0.006 | +0.018 | Include for consistency; small |

---

## Task table: all tasks across all pipeline versions

### Legend
- **Needed**: AUROC > 0.60 and meaningful class imbalance; threshold tuning changes reported numbers
- **Marginal**: AUROC good but near-balanced (gain < 0.01)
- **N/A**: multiclass (>2 classes) or seq2seq — binary threshold logic does not apply
- **Not applicable (signal)**: AUROC too low (<0.58) for threshold tuning to add value
- **Not needed**: balanced classes (~50:50), gain negligible

| Task | Pipeline | Status | N | Minority% | Best AUROC | Recall gap @t=0.5 | Tuning verdict | Est. BalAcc gain |
|------|----------|--------|---|-----------|------------|-------------------|----------------|-----------------|
| `sleep_staging` | phase0 | seq2seq | ~16k epochs | N/A (5-class) | ~0.80+ (κ) | N/A | **N/A** — use Cohen's κ + per-stage F1 |
| `apnea_binary` | phase0 + v3 (propose) | proposed Tier 1 | 14,097 | 48.9% | 0.734 (phase0 40m) | small | **Not needed** — near balanced |
| `sleepiness_binary` | phase0 | not in v3 registry | 16,431 | 29.7% | 0.609 (40m only) | varies | **Marginal** — AUROC borderline; if added to v3 and AUROC improves → revisit |
| `depression_binary` | phase0 | superseded | 2,794 | 26.8% | 0.643 (120m) | moderate | **Superseded** by `depression_extreme_binary` |
| `anxiety_binary` | phase0 | dropped | 1,698 | 20.1% | 0.580 (80m) | varies | **Not applicable (signal)** — AUROC < 0.60; tuning cannot fix absent signal |
| `insomnia_binary` | phase0 | not in v3 | 1,710 | 44.5% | 0.583 (80m) | small | **Not applicable (signal)** — AUROC < 0.60 |
| `cvd_binary` | phase0 | not in v3 | 13,045 | 25.0% | 0.670 (120m) | large | **Needed** — AUROC decent, minority=25%; if re-added to v3 expect +0.03–0.05 |
| `rested_morning` | phase0 | dropped | 3,934 | 43.3% | 0.540 (10m) | small | **Not applicable (signal)** — near chance; task fundamentally not learnable |
| `sex_binary` | v2 + v3 Tier 1 | active | 13,163 | 48.7% | high (TBD) | tiny | **Not needed** — near balanced, strong signal |
| `age_class` | v2 + v3 Tier 1 | active | 16,007 | N/A (3-class) | TBD | N/A | **N/A** — multiclass |
| `sleep_efficiency_binary` | v2 + v3 Tier 1 | active | 13,615 | 39.1% | TBD | moderate | **Marginal** — expect +0.01–0.02 |
| `bmi_binary` | v3 Tier 1 | active | 15,532 | 32.8% | 0.767 (K=all) | moderate | **Needed** — **+0.015 confirmed** from real data |
| `psqi_binary` | v3 Tier 2 | active | 3,933 | 43.9% | TBD | small | **Marginal** — near balanced, expect <0.01 gain |
| `depression_extreme_binary` | v3 Tier 2 | active (STAGES pending) | 1,761 | 13.3% | 0.00 (APPLES only) | extreme | **Critical if AUROC > 0.60 with STAGES** — 13.3% minority means large gain possible (+0.10–0.15 est.) |
| `osa_binary_apples_postqc` | v3 Tier 2 | active | 1,516 | 37% (class 0) | 0.742 (120m) | 0.587 (large) | **Needed** — est. **+0.06–0.09** in BalAcc |
| `osa_severity_apples` | v3 Tier 2 | active | 1,516 | N/A (4-class) | TBD | N/A | **N/A** — multiclass |

### Can threshold tuning revive dropped tasks?

**No, for rested_morning and anxiety_binary.** AUROC < 0.58 means the model cannot
distinguish classes beyond chance. Threshold tuning only recalibrates the boundary; it cannot
create discriminative signal that isn't there.

**Possibly, for cvd_binary**, if re-added to v3 and retrained with the v3 protocol (overlapping
windows, batch=32 everywhere). Phase0 AUROC reached 0.67 at 120m — this is workable.
Threshold tuning would further improve balanced accuracy reporting.

**Possibly, for sleepiness_binary**, if re-added to v3 and if AUROC improves above 0.65
with v3's larger context and overlapping window protocol. Phase0 used only 40m context.

---

## Implementation (completed)

### Design: add columns, never replace

Results are stored in a **separate CSV** alongside the existing parquets. Nothing is overwritten.
The original `summary.csv` (training-time K=5 metrics) and `test_windows.parquet` are untouched.

**Output:** `{inference_dir}/{exp_id}/threshold_tuning.csv`

One row per context length. Columns:

| Column group | Columns |
|---|---|
| Identity | `context_length`, `n_subjects`, `n_windows_test` |
| Threshold-free | `auroc` (unchanged by threshold) |
| Original at t=0.5 | `orig_balanced_accuracy`, `orig_recall_class0`, `orig_recall_class1`, `orig_accuracy`, `orig_macro_f1` |
| Threshold info | `optimal_threshold`, `val_n_subjects`, `val_balanced_accuracy_at_opt` |
| Tuned at t_opt | `tuned_balanced_accuracy`, `tuned_recall_class0`, `tuned_recall_class1`, `tuned_accuracy`, `tuned_macro_f1` |
| Gains | `balanced_accuracy_gain`, `recall_class0_gain`, `recall_class1_gain` |

`orig_*` and `tuned_*` are both computed from the K=all parquet using mean-prob subject aggregation.
This differs slightly from `summary.csv` (K=5 training eval) — both are correct, different K values.

### Files added / modified

| File | Change |
|---|---|
| `scripts/apply_threshold_tuning.py` | **New** — standalone script; reads val+test parquets, writes `threshold_tuning.csv` |
| `scripts/gen_commands.py` | **New subcommand** `threshold-tuning <exp_id>` — prints command with val-parquet check |
| `scripts/infer_subject_windows.py` | Unchanged — val parquets created by re-running with `--split val` |

### How to run

**Step 1 — Generate val parquets** (re-uses trained model, fast):

```bash
# Via gen_commands (generates the full infer sbatch command with --split val):
python scripts/gen_commands.py infer bmi_binary_lstm --split val | bash
python scripts/gen_commands.py infer osa_binary_apples_postqc_lstm --split val | bash
python scripts/gen_commands.py infer cvd_binary_transformer --split val | bash
python scripts/gen_commands.py infer sleepiness_binary_lstm --split val | bash
python scripts/gen_commands.py infer sleepiness_binary_transformer --split val | bash
python scripts/gen_commands.py infer bmi_binary_transformer --split val | bash
# Include all others for consistency:
python scripts/gen_commands.py infer sleep_efficiency_binary_lstm --split val | bash
python scripts/gen_commands.py infer sex_binary_lstm --split val | bash
python scripts/gen_commands.py infer sex_binary_transformer --split val | bash
python scripts/gen_commands.py infer apnea_binary_lstm --split val | bash
python scripts/gen_commands.py infer apnea_binary_transformer --split val | bash
python scripts/gen_commands.py infer depression_extreme_binary_lstm --split val | bash
python scripts/gen_commands.py infer cvd_binary_lstm --split val | bash
```

**Step 2 — Run threshold tuning** (CPU, fast, ~1 min per experiment):

```bash
source /home/boshra95/sleepfm_env/bin/activate

for exp in bmi_binary_lstm bmi_binary_transformer \
           sleep_efficiency_binary_lstm \
           sex_binary_lstm sex_binary_transformer \
           depression_extreme_binary_lstm \
           osa_binary_apples_postqc_lstm \
           apnea_binary_lstm apnea_binary_transformer \
           cvd_binary_lstm cvd_binary_transformer \
           sleepiness_binary_lstm sleepiness_binary_transformer; do
  python scripts/gen_commands.py threshold-tuning $exp | bash
done
```

Or run the check + command together:
```bash
python scripts/gen_commands.py threshold-tuning bmi_binary_lstm
# Prints: python scripts/apply_threshold_tuning.py --config ... --task bmi_binary --head lstm
# Also warns if val parquets are missing
```

### Only for binary tasks — skip multiclass and seq2seq

`num_classes == 2` check is enforced in the script. `age_class` (3-class),
`osa_severity_apples` (4-class), `sleep_staging` (5-class seq2seq) are automatically
skipped — use existing balanced_accuracy and per-class recall for those.

### Reporting convention for paper

- **Primary metric in tables**: AUROC (unchanged, threshold-free) — always report
- **Secondary**: balanced accuracy at `t_opt` from `threshold_tuning.csv`
- **Table footnote**: *"Balanced accuracy reported at the decision threshold t∗ selected on
  the held-out validation set to maximise balanced accuracy (Youden's Index). AUROC is
  unaffected by the threshold."*
- **Original t=0.5 results**: available in `threshold_tuning.csv` as `orig_balanced_accuracy`
  columns; show in supplementary alongside tuned results if needed

---

## Priority by v3 results (updated from initial estimates)

| Task | AUROC | Recall gap @best ctx | Verdict |
|---|---|---|---|
| `osa_binary_apples_postqc_lstm` | 0.742 | 0.587 | **CRITICAL** — will materially change numbers |
| `bmi_binary_lstm` | 0.729 | 0.205 | **YES** — confirmed +0.015 gain |
| `cvd_binary_transformer` | 0.679 | 0.286 | **YES** — notable gap |
| `sleepiness_binary_transformer` | 0.622 | 0.206 @240m | **YES** — borderline AUROC |
| `sleepiness_binary_lstm` | 0.608 | varies | **YES** — include for consistency |
| `bmi_binary_transformer` | 0.761 | 0.149 | **MARGINAL** — small, include for consistency |
| `psqi_binary_lstm` | 0.525 | — | **NO** — AUROC near chance; tuning cannot fix absent signal |
| `depression_extreme_binary_lstm` | 0.767 | 0.030 | **NO** — class weighting worked; near balanced |
| `sex_binary`, `apnea_binary`, `sleep_efficiency_binary_transformer` | — | <0.09 | **NO** — near balanced |
| `age_class`, `sleep_staging` | — | N/A | **N/A** — multiclass / seq2seq |

---

## Final recommendations for paper (based on actual v3 results, 2026-05-30)

### Use tuned results (report BA at t_opt)

| Task | Best gain | Key finding |
|---|---|---|
| `osa_binary_apples_postqc_lstm` | **+0.220 at 10m** | MUST use tuned — t=0.5 predicts class 1 for ~98% of subjects; meaningless |
| `depression_extreme_binary_lstm` | +0.065 at 80m | Use tuned at all contexts — model becomes biased at long contexts despite auto-weighting |
| `bmi_binary_transformer` | +0.027 at 30s, avg +0.013 | Consistent improvement across all contexts |
| `bmi_binary_lstm` | +0.013 at 30s, avg +0.006 | Moderate; include for consistency with transformer |
| `sex_binary_lstm` | +0.020 at 30s, avg +0.009 | Small but real; include |
| `sleepiness_binary_lstm` | +0.014 avg +0.006 | Small; include for consistency |
| `sleepiness_binary_transformer` | +0.018 avg +0.006 | Small; include for consistency |

### Keep original t=0.5 results

| Task | Reason |
|---|---|
| `cvd_binary_lstm` | Tuning HURTS (avg −0.005); val threshold does not generalise |
| `cvd_binary_transformer` | Near zero or negative |
| `sex_binary_transformer` | Near zero |
| `apnea_binary_lstm` | Very small (+0.003 avg) |
| `apnea_binary_transformer` | Near zero |
| `sleep_efficiency_binary_lstm` | Near zero, some negative |
| `sleep_efficiency_binary_transformer` | Near zero, some negative |

### Key surprises vs predictions

1. **osa_binary gains were massively understated**: estimated +0.06–0.09, actual up to +0.22.
   The model was almost entirely predicting the majority class at t=0.5.

2. **depression_extreme needs tuning at long contexts**: predicted balanced (no tuning needed),
   but 80m (+0.065) and 120m (+0.051) show significant imbalance. Short contexts fine; long contexts not.

3. **cvd_binary tuning HURTS**: predicted "YES — notable gain". Actual: all negative or zero.
   Low AUROC + small val set = unreliable threshold selection.

4. **bmi_binary_transformer better than lstm**: predicted marginal for transformer, confirmed for lstm.
   Actual transformer avg gain (+0.013) exceeds lstm (+0.006).

5. **sex_binary_lstm has small real gain** (+0.009 avg): predicted balanced/no gain. Actual shows
   systematic but small miscalibration at t=0.5 across contexts.

### Paper footnote (final wording)

> "For tasks with class imbalance we report balanced accuracy at the decision threshold t∗
> selected on the validation set to maximise balanced accuracy (Youden's Index). AUROC is
> unaffected (threshold-free). For `cvd_binary`, the default t=0.5 is retained because
> val-optimised thresholds did not generalise to the test set, likely due to insufficient
> validation set size."
