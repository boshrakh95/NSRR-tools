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
