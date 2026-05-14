# Analysis Ideas and Extensions

This document proposes new analyses that extend or deepen the core H1–H4 findings, organized by what data they require and how hard they are to implement. Each section states the scientific question, the expected output, what data/code changes are needed, and the connection to the paper.

**Status legend:** ⬜ TODO · 🔄 In progress · ✅ Done

---

## Table of Contents

1. [Overfitting Curves and Compute Scaling Laws](#1-overfitting-curves-and-compute-scaling-laws)
2. [Model Calibration and Reliability](#2-model-calibration-and-reliability)
3. [Bootstrap Confidence Intervals](#3-bootstrap-confidence-intervals)
4. [Window Position and Temporal Structure](#4-window-position-and-temporal-structure)
5. [Per-Subject Consistency and Prediction Stability](#5-per-subject-consistency-and-prediction-stability)
6. [Task × Context Sensitivity Matrix](#6-task--context-sensitivity-matrix)
7. [Cohort-Stratified Saturation](#7-cohort-stratified-saturation)
8. [Clinical Operating Points on Precision-Recall Curves](#8-clinical-operating-points-on-precision-recall-curves)
9. [Subject-Level K-Saturation Heterogeneity](#9-subject-level-k-saturation-heterogeneity)
10. [Head Architecture at Equal Compute](#10-head-architecture-at-equal-compute)

---

## 1. Overfitting Curves and Compute Scaling Laws

### Scientific question

Does training longer (past early stopping) reveal a context-length-dependent overfitting regime? And does model performance follow a power-law relationship with training compute — the way large language models do?

Two sub-questions:
- **U-shape curves:** At what epoch does validation loss begin rising (i.e., overfitting start)? Does this threshold differ across context lengths and heads? Does a longer-context model have more regularization from the task difficulty, or less from fewer windows per epoch?
- **Scaling laws:** If we plot best achievable validation AUROC as a function of total training compute (FLOPs), do we see a predictable curve? Does the curve's slope differ across context lengths — implying different data efficiency per compute budget?

### Connection to paper

This is a novel empirical contribution that connects the PSG domain to the broader neural scaling laws literature (Hoffmann et al. 2022 / Chinchilla; see https://howtoscalenn.github.io/). The key claim would be: *for frozen-backbone fine-tuning on PSG data, performance scales predictably with training compute, and longer-context models are not necessarily less compute-efficient despite fewer training windows per epoch.*

This also turns a limitation (early stopping prevents full observation of overfitting) into a deliberate experiment design choice.

### Expected outputs

- **Figure A:** U-shape curves — train_loss and val_loss vs epoch for each context length on one plot (one subplot per head or one panel per context). The x-axis extends well past the early-stop epoch to reveal the overfitting valley and the rising right arm.
- **Figure B:** Compute scaling plot — best val AUROC (y) vs total training FLOPs (x), one point per (context, head). Fit a power law: `AUROC = AUROC_max - A × FLOPs^(-b)`. If points fall on a shared curve, context length and head type differ only in how they trade compute for performance; if they fall on separate curves, longer contexts are fundamentally different.
- **Figure C:** Optimal epoch vs context length — the epoch at which val loss is minimum (the "valley" of the U-shape) plotted against context length in minutes. Answers: does longer context require more or fewer gradient steps to converge?

### Data required

- **What already exists:** `training_curves.csv` contains per-epoch val_loss/val_bal_acc but only up to the early-stop epoch. The full history up to that epoch is recoverable. However, the post-overfitting region is missing because training halts.
- **What needs to be added:**

**Option A — Extend training after convergence (recommended):**
Add an `--overfit-epochs N` flag to `train_context_sweep.py`. After the normal training loop ends (early stopping or max epochs), run `N` additional epochs without saving to `best_model.pt`, recording only the loss/metrics. This adds no risk to the trained model but generates the right arm of the U-shape. Cost: one additional training job per context with `--overfit-epochs 50`.

**Option B — Run separate "no early stop" jobs:**
Add a `--no-early-stop` flag that disables early stopping and always runs for `max_epochs`. Use a different `run_tag` so results don't overwrite the base run. This is cleaner for the scaling law analysis but doubles training cost.

**FLOPs estimation:** Add a `compute_training_flops()` function to `train_context_sweep.py` that computes FLOPs per gradient step analytically:
```python
# LSTM: 4 × seq_len × (input_dim × hidden_dim + hidden_dim²) × 2 (fwd+bwd)
# Transformer: 4 × seq_len² × hidden_dim × num_heads × 2
# MeanPool: trivial — seq_len × hidden_dim × 2
# Multiply by steps_per_epoch × n_epochs to get total FLOPs
```
Save `flops_per_step` to `metrics.json` so the collector can read it. Then `total_train_flops = flops_per_step × n_epochs × steps_per_epoch`.

**Periodic snapshots:** Save a model snapshot every `--snapshot-interval` epochs (e.g., 5) to `snapshots/epoch_{N}.pt`. These allow running inference at intermediate checkpoints to generate learning curves in test-set metric space, not just val-set metric space. This is optional but needed for the most rigorous scaling law plot.

### Implementation changes needed

| File | Change |
|------|--------|
| `scripts/train_context_sweep.py` | Add `--overfit-epochs`, `--no-early-stop`, `--snapshot-interval` flags; add FLOPs computation; save `flops_per_step` to `metrics.json` |
| `scripts/collect_results_v2.py` | Add `flops_per_step` and `total_train_flops` columns to `training.csv` |
| `scripts/plot_scaling_laws.py` | New script: reads `training.csv`, fits power law, generates Figures A–C |

### Priority

**High.** This is the most novel and publishable addition. It requires the most code change but produces figures that no prior PSG ML paper has shown. Run on one task first (recommended: `sex_binary_lstm` across all 6 contexts) to validate, then extend to all tasks.

---

## 2. Model Calibration and Reliability

### Scientific question

When the model outputs probability 0.8, are 80% of those subjects truly positive? **Calibration** is the alignment between predicted confidence and actual outcome frequency. Poorly calibrated models mislead clinicians even when their AUROC is high: a model that outputs 0.51 for all positives has excellent AUROC but is useless for clinical risk stratification.

Sub-questions:
- Does longer context improve calibration?
- Does aggregating more windows (larger K) improve calibration?
- Does mean-probability aggregation produce better-calibrated predictions than majority-vote?
- Do multi-class tasks (age_class, osa_severity) show systematic per-class calibration failures?

### Connection to paper

All major medical AI venues (NeurIPS, MLHC, CHIL, NEJM AI) now routinely ask about calibration. Showing that longer context not only improves discrimination (AUROC) but also produces better-calibrated outputs is a strong result. It makes the system more clinically actionable.

### Expected outputs

- **Reliability diagrams (calibration plots):** Bin predicted probabilities into 10 equal-width bins (0–0.1, 0.1–0.2, …). For each bin, plot the mean predicted probability (x) vs the fraction of true positives (y). A perfectly calibrated model lies on the diagonal. One plot per (task, context_length, K) — or an array plot comparing context lengths side by side.
- **ECE vs context length:** Expected Calibration Error (ECE = Σ_bins |accuracy - confidence| × weight) as a line plot with one line per head, x = context length. Mirrors the saturation curve.
- **ECE vs K:** For a fixed context length, how does ECE change as you aggregate more windows? Does mean-prob calibration converge to perfect calibration at large K (central limit theorem argument)?

### Data required

Fully derivable from existing `predictions/*.parquet`. No new data needs to be saved. For K > 1, you need to average `prob_class1` over K windows per subject first.

```python
# ECE from existing parquet:
from sklearn.calibration import calibration_curve

df = pd.read_parquet("predictions/sex_binary_lstm_10m_test.parquet")
# Average K windows per subject (K=5 example)
subj = df.groupby(["subject_id","dataset"]).head(5).groupby(["subject_id","dataset"]).agg(
    prob=("prob_class1","mean"), true_label=("true_label","first")).reset_index()
fraction_pos, mean_pred = calibration_curve(subj["true_label"], subj["prob"], n_bins=10)
```

### Implementation changes needed

| File | Change |
|------|--------|
| `scripts/plot_calibration.py` | New script: reads predictions parquets, computes ECE and reliability diagrams per (task, head, context, K) |
| `scripts/collect_results_v2.py` | Optionally add `ece_k1`, `ece_k5`, `ece_kall` columns to `analysis.csv` |

### Priority

**High.** Zero changes to training/inference code. All data is already available. One new script.

---

## 3. Bootstrap Confidence Intervals

### Scientific question

Are the observed AUROC differences between context lengths statistically significant? With 200–500 test subjects, point estimates can be noisy. Reviewers will ask: is 0.758 at 10m meaningfully better than 0.741 at 30s?

### Connection to paper

Tables and saturation curves become much stronger with error bars. Bootstrap CIs are the standard non-parametric approach for AUROC confidence intervals in medical ML. This does not change any conclusions but substantially strengthens the evidence.

### Expected outputs

- All paper tables gain `±CI` columns: `0.758 ± 0.012 (95% CI: 0.735–0.782)`
- Saturation curves gain shaded confidence bands around each line
- Heatmap cells gain CI annotations (optional)
- Statistical significance markers on the saturation curve at pairs where CIs do not overlap

### Data required

Fully derivable from existing predictions parquets. Bootstrap resamples subjects (not windows — subject-level resampling preserves the correlation structure of windows from the same subject).

```python
import numpy as np
from sklearn.metrics import roc_auc_score

def bootstrap_auroc(subjects_df, n_boot=1000, ci=0.95):
    """subjects_df: one row per subject, columns: true_label, mean_prob"""
    aucs = []
    rng = np.random.default_rng(42)
    subjects = subjects_df.index.values
    for _ in range(n_boot):
        idx = rng.choice(subjects, size=len(subjects), replace=True)
        boot = subjects_df.loc[idx]
        if boot["true_label"].nunique() < 2:
            continue
        aucs.append(roc_auc_score(boot["true_label"], boot["mean_prob"]))
    lo = np.percentile(aucs, (1 - ci) / 2 * 100)
    hi = np.percentile(aucs, (1 + ci) / 2 * 100)
    return np.mean(aucs), lo, hi
```

### Implementation changes needed

| File | Change |
|------|--------|
| `scripts/analyze_windows.py` | Add `--bootstrap N` flag; compute CIs for AUROC and balanced_accuracy at each K; add `auroc_ci_lo`, `auroc_ci_hi` columns to `window_analysis_{split}.csv` |
| `scripts/collect_results_v2.py` | Collect CI columns into `analysis.csv` |
| `scripts/plot_saturation.py` | Add shaded CI bands when CI columns are present |

### Priority

**High.** Pure post-processing from existing data. Expected runtime: <5 minutes per experiment (1000 bootstrap resamples over ~500 subjects).

---

## 4. Window Position and Temporal Structure

### Scientific question

Does it matter *where in the night* a PSG window comes from? `window_idx=0` is the first window at the start of the recording; `window_idx=N-1` is the last. If the model predicts better from mid-night or late-night windows, that tells us *where in a PSG study the clinical signal is concentrated* — a biologically meaningful finding.

Sub-questions:
- Does mean predicted probability (aggregated over subjects) vary as a function of normalized window position?
- For positive vs negative subjects, do early-night and late-night windows produce different confidence scores?
- Does longer context reduce position bias (the model sees more of the night in one window, so position matters less)?
- Are there tasks where position matters more (e.g., sleep efficiency is more predictable from late-night windows, after sleep debt accumulates; sex is position-independent)?

### Connection to paper

This is a mechanistic insight, not just a performance metric. It explains *what* the model learned, not just *how well* it learned. A Figure 4 (supplementary) showing probability profiles across the night, stratified by task and context length, would be a memorable and interpretable finding for medical readers.

### Expected outputs

- **Position-probability profile:** For each task and context length, compute the mean `prob_class1` per normalized window position (0 = start of night, 1 = end), averaged over positive subjects and over negative subjects separately. Plot as a line with a confidence band.
- **Position variance vs context length:** Does the standard deviation of per-window predictions (within a subject) decrease as context length increases? (Longer context → more self-consistent, position-independent predictions.)

### Data required

Existing predictions parquets have `window_idx`. You need to normalize by total windows per subject:
```python
df["norm_pos"] = df.groupby(["subject_id","dataset"])["window_idx"].transform(
    lambda x: x / x.max() if x.max() > 0 else 0.0
)
```
No new data needs to be saved.

### Implementation changes needed

| File | Change |
|------|--------|
| `scripts/plot_window_position.py` | New script: reads predictions parquets, computes position profiles per (task, context_length) |

### Priority

**Medium.** All data is already available. Generates a biologically interpretable supplementary figure.

---

## 5. Per-Subject Consistency and Prediction Stability

### Scientific question

For a given subject, how variable are the model's predictions across different windows of the same night? High variance within a subject means the model is uncertain about them. Low variance means the model confidently and consistently predicts the same class across all windows.

Sub-questions:
- Does within-subject variance decrease as context length increases? (Longer windows → more stable predictions?)
- Are there systematic "hard subjects" where all models at all context lengths predict incorrectly? Conversely, are there "easy subjects" that every model gets right even at 30s?
- Does within-subject variance correlate with demographic variables (age, sex, dataset) or with signal quality?

### Connection to paper

This motivates the aggregation approach: if within-subject variance is high, averaging many windows (large K) will reduce uncertainty — this directly connects to H3. It also identifies the hardest cases, which could be turned into a clinical insight: "subjects where PSG-based prediction remains unreliable even with the full night."

### Expected outputs

- **Per-subject variance distribution:** Histogram of `std(prob_class1)` over windows, stratified by true class (positive vs negative) and by correct/incorrect prediction.
- **Variance vs K plot:** Show how aggregated prediction entropy decreases as a function of K — connecting variance reduction to AUROC improvement (linking this to the H3 result).
- **Hard-subject analysis table:** Count of subjects correctly classified by 0, 1, 2, 3, … context lengths. "Hard" = incorrectly classified at all context lengths.

### Data required

Existing predictions parquets. No new data needed. Per-subject statistics can be computed as:
```python
subj_stats = df.groupby(["subject_id","dataset"]).agg(
    mean_prob=("prob_class1","mean"),
    std_prob=("prob_class1","std"),
    n_windows=("prob_class1","count"),
    true_label=("true_label","first"),
).reset_index()
subj_stats["correct"] = (subj_stats["mean_prob"] > 0.5) == subj_stats["true_label"]
```

### Implementation changes needed

| File | Change |
|------|--------|
| `scripts/plot_subject_consistency.py` | New script: reads parquets, computes within-subject variance profiles |

### Priority

**Medium.** Biologically meaningful and requires no new data collection.

---

## 6. Task × Context Sensitivity Matrix

### Scientific question

Across all tasks, which ones benefit the most from longer context? Which tasks show strong performance even at 30 seconds? This cross-task comparison is the key finding for readers who want to know: *should I train a long-context PSG model for my specific clinical task?*

Define **context sensitivity** for a task as:
```
sensitivity = AUROC(best_context, K=all) − AUROC(30s, K=all)
```

And **baseline difficulty** as `1 - AUROC(30s, K=all)` (how much room for improvement exists).

Sub-questions:
- Is there a correlation between task difficulty and context sensitivity? (Hard tasks might benefit more from context.)
- Do tasks that depend on slow physiological changes (e.g., sleep efficiency, age) benefit more from long context than tasks with fast discriminative signal (e.g., sex)?
- Does the optimal context length L* vary by task? Is there a task-specific "saturation point"?

### Connection to paper

This is the primary cross-task summary figure. It tells practitioners: for OSA detection, you may need X minutes; for sex prediction, 30s is enough. This is actionable clinical guidance.

### Expected outputs

- **2D scatter plot:** Each task is a point. X-axis = baseline difficulty (1 − AUROC@30s). Y-axis = context sensitivity (ΔAUROC from 30s to best context). Label each point with the task name and color by task type (sleep quality, demographics, comorbidity). Tasks in the upper-right quadrant are both hard and context-dependent.
- **Sorted bar chart:** Tasks ranked by context sensitivity, with bars showing AUROC at each context length stacked or grouped. This is the clearest cross-task summary for a paper table.
- **L* per task:** For each task, the context length at which performance plateaus (defined as the smallest L where AUROC is within 0.5% of the maximum). Visualize as a dot plot with L* on the x-axis and task on the y-axis.

### Data required

Fully derivable from existing `analysis.csv` (k="all") and `training.csv` (best epoch test_auroc). No new data needed. Can be produced as soon as multiple tasks have been run.

### Implementation changes needed

| File | Change |
|------|--------|
| `scripts/plot_task_comparison.py` | New script: reads `analysis.csv`, computes sensitivity matrix and L* per task, generates scatter and bar charts |

### Priority

**High.** Critical cross-task summary for the paper. All data is available once multiple tasks have been run. No code changes to training/inference.

---

## 7. Cohort-Stratified Saturation

### Scientific question

Does the optimal context length differ by dataset (cohort)? APPLES, SHHS, and MrOS differ in demographics, recording protocols, PSG equipment, and clinical indication. A model trained on all three may mask cohort-level heterogeneity.

Sub-questions:
- Does AUROC vs context length have a different shape for APPLES vs SHHS vs MrOS?
- Are certain tasks whose population is concentrated in one cohort (e.g., psqi_binary is MrOS-only) more strongly context-dependent than cross-cohort tasks?
- Does the model generalize — i.e., if you train on APPLES+SHHS, does performance on MrOS subjects (as a hold-out group) saturate at the same context length?

### Connection to paper

Shows robustness of the context saturation result across different clinical populations. Also identifies if results are driven by one cohort.

### Expected outputs

- **Per-cohort saturation curves:** For tasks with multiple datasets (sex_binary, bmi_binary), show AUROC vs context length split by dataset (three lines per plot).
- **Dataset contribution table:** For each (task, context_length), show n_subjects per dataset and per-dataset AUROC side by side.

### Data required

Existing predictions parquets have a `dataset` column. Filter by dataset before computing metrics.

```python
for ds in ["apples", "shhs", "mros"]:
    sub = df[df["dataset"] == ds].groupby(["subject_id"]).agg(...)
    auroc = roc_auc_score(sub["true_label"], sub["mean_prob"])
```

### Implementation changes needed

| File | Change |
|------|--------|
| `scripts/plot_cohort_saturation.py` | New script: reads parquets filtered by dataset, generates per-cohort saturation curves |

### Priority

**Medium.** Important for robustness, straightforward to implement.

---

## 8. Clinical Operating Points on Precision-Recall Curves

### Scientific question

AUROC treats all operating points equally. In clinical practice, the cost of a false negative and a false positive are not symmetric:
- **OSA detection:** Missing a positive (low sensitivity/recall) is clinically dangerous. Operate at high recall, accept lower precision.
- **Depression screening:** High specificity is needed to avoid burdening referred patients unnecessarily.
- **Sleep efficiency:** Moderate tradeoff; depends on whether the tool is for triage or confirmation.

How does the precision-recall tradeoff change with context length and K?

### Connection to paper

Task-specific clinical operating point analysis is increasingly required by medical AI reviewers. It shows that the paper understands the clinical deployment context, not just the benchmark metric.

### Expected outputs

- **PR curves at key K values:** For each task, one PR curve per context length (all at K=all or at a fixed K). Show how the AUC-PR (average precision) changes with context length — this is a more informative metric than AUROC for imbalanced tasks.
- **Recall at fixed precision table:** For each task, the recall achieved at the precision threshold corresponding to the clinical requirement (e.g., 90% sensitivity for OSA, or 80% specificity for depression). Show how this recall changes with context length and K.
- **Precision-recall at K sweep:** For a fixed context, show how PR curves shift as K increases from 1 to "all" — K aggregation improves recall at all precision thresholds, but does it improve more at the high-recall or high-precision end?
- **Threshold-sweep majority vote (Plot B from context_length_experiment_design.md):** Instead of threshold = K/2, sweep the majority-vote threshold t from 1 to K. At t=1 ("any positive window = predict positive") the model has high recall; at t=K ("all windows must vote positive") it has high precision. This precision-recall tradeoff via threshold is a clinically meaningful way to present majority-vote aggregation.

### Data required

Existing predictions parquets contain `prob_class1` for computing PR curves. For the majority-vote threshold sweep:
```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# Majority vote at threshold t out of K
votes = (df.groupby(["subject_id","dataset"]).head(K)["pred_label"]
           .groupby(["subject_id","dataset"]).sum())  # number of positive votes
# At threshold t: predict positive if votes >= t
for t in range(1, K+1):
    y_pred = (votes >= t).astype(int)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
```

### Implementation changes needed

| File | Change |
|------|--------|
| `scripts/plot_precision_recall.py` | New script: PR curves per (task, context_length, K); majority-vote threshold sweep |
| `scripts/analyze_windows.py` | Optionally add `--pr-threshold` flag to compute recall at a fixed precision target per K |

### Priority

**Medium-high.** Especially important for OSA, depression, and PSQI tasks (Tier 2). Generates clinically interpretable figures that make the paper more accessible to medical readers.

---

## 9. Subject-Level K-Saturation Heterogeneity

### Scientific question

H3 (aggregation saturation) is currently studied at the population level: AUROC plateaus after K* windows for the average subject. But is K* the same for all subjects? Some subjects may be "easy" — 1 window is enough. Others may be "hard" — they require many windows before the aggregated prediction stabilizes.

Sub-questions:
- What is the distribution of per-subject K* (minimum windows to reach correct prediction)?
- Do "hard" subjects have longer K* at all context lengths, or does longer context "rescue" them at lower K?
- Can we predict which subjects will need more windows? (e.g., from recording length, dataset, or a proxy for signal quality derived from prediction variance at K=1)

### Connection to paper

This connects H3 (when does aggregation saturate?) to a deployability claim: "95% of subjects reach correct prediction within K=5 windows at L=40m" is a much more actionable statement for clinical deployment than "population AUROC saturates at K=5."

### Expected outputs

- **Distribution of per-subject K*:** Histogram of the minimum K required for each subject to be correctly predicted, stratified by context length.
- **Cumulative coverage curve:** For each context length, the fraction of subjects correctly predicted as a function of K (a CDF-style plot). Shows when you have "covered" 50%, 80%, 95% of subjects.
- **K* scatter per subject:** For a subset of subjects, show K* at L=30s vs K* at L=40m. Points below the diagonal are subjects that "benefited from longer context" in the sense that fewer windows were needed.

### Data required

Existing predictions parquets, iterated over K values per subject. No new data needed but requires a nested loop: for each subject, find the smallest K where the mean-prob prediction is correct.

### Implementation changes needed

| File | Change |
|------|--------|
| `scripts/plot_subject_kstar.py` | New script: reads parquets, computes per-subject K*, generates coverage curves |

### Priority

**Low-medium.** Interesting supplementary result. Requires careful definition of K* (what counts as "correctly predicted" — at the current threshold, or by probability exceeding 0.5?).

---

## 10. Head Architecture at Equal Compute

### Scientific question

The saturation curve comparison (LSTM vs Transformer vs MeanPool) is currently at equal epochs and equal data. But the three architectures have very different FLOPs per gradient step. At the same total compute budget (FLOPs), which head performs best?

This matters because: a MeanPool head runs a training epoch 5–10× faster than an LSTM. If a MeanPool model trained for 300 epochs beats an LSTM trained for 30 epochs at the same total FLOPs, then temporal modeling is not adding value beyond simply processing more gradient updates.

### Connection to paper

The "does temporal modeling help?" question (implicit in the H1/Figure 1 head comparison) has a cleaner answer when framed as equal-compute rather than equal-epochs. This guards against a confound where LSTM appears better simply because it's harder to train and implicitly regularizes.

### Expected outputs

- **Compute-normalized saturation curve:** X-axis = total training FLOPs (estimated), Y-axis = best val AUROC. One line per head per context length. If MeanPool and LSTM fall on the same curve, temporal modeling only helps when compute is held equal by architecture.
- **FLOPs table:** Estimated FLOPs per gradient step for each (head, context_length) combination, shown as a supplementary table.

### Data required

Requires adding `flops_per_step` and `n_steps` to `metrics.json` (see Section 1). Then `total_train_flops = flops_per_step × n_steps` and FLOPs is available in `training.csv` after collecting results.

### Implementation changes needed

Same as Section 1 (FLOPs estimation in `train_context_sweep.py`). Once FLOPs are in `training.csv`, this plot is a simple extension of `plot_saturation.py`.

### Priority

**Low.** Requires the same FLOPs addition from Section 1. Add once Section 1 is implemented.

---

## Implementation Roadmap

### Phase 1 — Zero new data needed (can start now)

| Analysis | Script to write | Est. effort |
|----------|----------------|-------------|
| Calibration + ECE (§2) | `plot_calibration.py` | 1–2 days |
| Bootstrap CIs (§3) | add to `analyze_windows.py` | 1 day |
| Task × context sensitivity matrix (§6) | `plot_task_comparison.py` | 1 day |
| Cohort-stratified saturation (§7) | `plot_cohort_saturation.py` | 1 day |
| Window position analysis (§4) | `plot_window_position.py` | 1–2 days |
| Per-subject consistency (§5) | `plot_subject_consistency.py` | 1 day |
| PR curves + threshold sweep (§8) | `plot_precision_recall.py` | 2 days |
| Subject-level K* (§9) | `plot_subject_kstar.py` | 2 days |

### Phase 2 — Requires training code changes + retraining

| Analysis | Training change | Est. GPU time |
|----------|----------------|---------------|
| Overfitting curves (§1, part A) | Add `--overfit-epochs 50` | +50 epochs per context = ~1–3h per context on H100 |
| Compute scaling laws (§1, part B) | Add FLOPs logging to `metrics.json` | No retraining needed (analytical) |
| Periodic snapshots (§1, part C) | Add `--snapshot-interval` | Disk cost; no GPU cost |

### Phase 3 — Requires Phase 2 + new scripts

| Analysis | Depends on |
|----------|-----------|
| FLOPs-normalized head comparison (§10) | FLOPs in metrics.json (§1) |
| Scaling law fit (§1, Figure B) | FLOPs + overfit training curves |

---

## Notes on Task-Specific Analysis

**`osa_severity_apples` (4-class):** Precision-recall at a clinical threshold is especially important here. The standard in sleep medicine is to treat AHI ≥ 15 (moderate) as clinically significant. A natural dichotomization for evaluation is "severity class ≥ 2 = clinically significant" even though the model is 4-class. Per-class recall (mild vs moderate vs severe) is also more informative than macro AUROC for this task.

**`age_class` (3-class):** Calibration per class matters: is the model overconfident about the ≥65 class because MrOS (all class 2) dominates? The middle class (50–64) is likely the hardest — per-class confusion matrix and recall analysis will reveal whether the middle class is simply absorbed by the adjacent classes.

**`depression_extreme_binary`:** The extreme-group design (middle BDI range dropped) inflates AUROC by design. When reporting results, show what AUROC would be if the middle group were included (this requires re-running inference on the dropped subjects, which were removed from `task_subjects/`). This is important for clinical validity.

**`sex_binary`:** This is the easiest task (AUROC typically 0.9+ in PSG studies) and likely saturates at short context. Its primary value in this paper is as a sanity check and as the "hero" heatmap task (where iso-compute comparisons are clearest and most statistically well-powered).

**`sleep_efficiency_binary`:** The clinical threshold (85%) is debated. A sensitivity analysis varying the threshold from 80% to 90% and showing whether the context saturation result is stable would strengthen this task's findings.
