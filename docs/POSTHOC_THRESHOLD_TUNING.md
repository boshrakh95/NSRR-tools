# Post-hoc Threshold Tuning — Planning Document

*Written May 2026. Do not implement until all v3 training/inference runs are finished.*

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

## Implementation plan

### What needs to change

**Current state**: inference saves `test_windows.parquet` with columns
`[subject_id, dataset, true_label, pred_label, prob_class0, prob_class1, window_idx]`.
Val parquets are NOT saved. Threshold is implicitly 0.5 via `pred_label = argmax(logits)`.

**Required additions**:
1. **Save `val_windows.parquet`** during inference — same format as test, on the val split.
   This is the held-out data for threshold selection. Must be added to `infer_context_sweep.py`.
2. **Threshold selection script** — reads val parquet, finds optimal threshold, applies to test.
3. **Reporting** — replace `test_balanced_accuracy`/`test_recall_classX` in summary with
   threshold-tuned versions; add `optimal_threshold` column.

### Files to modify (when implementing)

| File | Change |
|------|--------|
| `scripts/infer_context_sweep.py` | Add val-split inference pass; save `val_windows.parquet` |
| `scripts/analyze_results.py` (or equivalent) | Add `apply_threshold_tuning()` function after loading parquets |
| `scripts/build_heatmap_df.py` | Use tuned balanced accuracy for heatmap colour column |

### Core logic (do not add to codebase yet — reference only)

```python
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

def tune_threshold(val_parquet_path: str, test_parquet_path: str,
                   agg: str = "mean") -> dict:
    """
    Select decision threshold on val set, apply to test set.
    agg: how to aggregate window-level scores to subject level ('mean' or 'max').
    Returns dict with threshold, tuned test metrics.
    """
    val_df  = pd.read_parquet(val_parquet_path)
    test_df = pd.read_parquet(test_parquet_path)

    def aggregate(df):
        return df.groupby("subject_id").agg(
            true_label=("true_label", "first"),
            score=("prob_class1", agg)
        ).reset_index()

    val_agg  = aggregate(val_df)
    test_agg = aggregate(test_df)

    # Find threshold maximising val balanced accuracy
    thresholds = np.linspace(0.01, 0.99, 200)
    best_t = max(
        thresholds,
        key=lambda t: balanced_accuracy_score(
            val_agg["true_label"], (val_agg["score"] > t).astype(int)
        )
    )

    # Apply to test
    test_preds = (test_agg["score"] > best_t).astype(int)
    labels = test_agg["true_label"].values
    n_classes = labels.max() + 1

    result = {
        "optimal_threshold":        round(float(best_t), 4),
        "test_balanced_accuracy":   float(balanced_accuracy_score(labels, test_preds)),
    }
    for c in range(n_classes):
        result[f"test_recall_class{c}"] = float(
            recall_score(labels, test_preds, pos_label=c, average="binary")
        )
    return result
```

### Only for binary tasks — skip multiclass and seq2seq

Apply only when `num_classes == 2`. For `age_class` (3-class), `osa_severity_apples` (4-class),
and `sleep_staging` (5-class seq2seq), skip entirely and use existing metrics.

### Reporting convention for paper

- **Primary metric**: AUROC (unchanged, threshold-free)
- **Secondary**: balanced accuracy at `t_opt` (selected on val, applied to test)
- **Footer note** in tables: "Balanced accuracy reported at decision threshold selected on the
  validation set to maximise balanced accuracy; AUROC is unaffected."

---

## Priority order when implementing

1. `bmi_binary` — confirmed +0.015 gain, already has inference parquets (missing val parquet)
2. `osa_binary_apples_postqc` — est. +0.06–0.09 gain, will materially change reported numbers
3. `depression_extreme_binary` — critical only if AUROC > 0.60 with STAGES
4. `sleep_efficiency_binary` — marginal but large N, worth including for consistency
5. `cvd_binary` — only if re-added to v3 registry
6. All others — either N/A or not worth implementing

---

## When to implement

After all v3 training and inference runs are complete. Do not implement piecemeal — apply to
all tasks in one pass so the paper tables are consistent.

Estimated effort: 1 day (modify infer script + add analysis function + re-run analysis).
