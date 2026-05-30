# Sleep Staging — Design Decisions & Analysis Plan

> Created: 2026-05-28  
> Covers: window selection strategy, training fairness, and analysis/plotting plan for the
> seq2seq sleep-staging task. Update this file as decisions are made.

---

## 1. The Two Problems with the Current Setup

### 1A. K=5 Does Not Work for seq2seq

For seq2label tasks (apnea, BMI, …), `windows_per_subject=5` correctly limits each subject
to 5 non-overlapping context windows per epoch → ~47K training items.

For seq2seq (sleep staging), each "item" is a single-epoch prediction. K=5 windows of
length L would contribute only 5 × L/5sec prediction targets per subject — which is a tiny
number and ignores most of the night. The dataset code instead falls back to all anchor
epochs in valid windows, giving **K ≈ 966/subject** (measured across all contexts).
Result: **1.6M training items** vs ~47K for seq2label (35× more), and epochs take 30–70
min depending on context length.

### 1B. Edge Epochs Are Mostly Padding

For a centered context window of length L, an anchor epoch within L/2 of the recording
start/end has real signal on only one side. At 240m context, the first and last 2 hours of
every recording are dominated by padding. Observed effect:

| Context | N patches | Flash? | Cause |
|---------|-----------|--------|-------|
| 30s (N=6)   | ✅ Flash | No padding (6-patch window never hits edges for valid recordings) |
| 10m (N=120) | ❌ No Flash | Some edge epochs padded |
| 40m (N=480) | ✅ Flash | Apparently few/no padded windows in sampled batch |
| 80m (N=960) | ❌ No Flash | Padded edge epochs exist |
| 120m (N=1440)| ❌ No Flash | Many padded edge epochs |
| 240m (N=2880)| ❌ No Flash | Large fraction padded |

Padding disables Flash attention for any batch containing a padded window, making training
dramatically slower at long contexts.

---

## 2. Window Selection Options

### Option A — Complete-Context Only (Recommended)

Only include anchor epochs where the full L-length window fits entirely within the recording:
`anchor_idx ≥ L/2  AND  anchor_idx ≤ T − L/2`

**Effect on dataset:**

| Context | Excluded epochs per recording | Remaining fraction |
|---------|------------------------------|-------------------|
| 30s     | ~6 epochs (~30 sec)          | ~99.9%            |
| 10m     | ~120 epochs (~10 min total)  | ~97%              |
| 40m     | ~480 epochs (~40 min total)  | ~91%              |
| 80m     | ~960 epochs (~80 min total)  | ~83%              |
| 120m    | ~1440 epochs (~120 min total)| ~75%              |
| 240m    | ~2880 epochs (~240 min total)| ~50%              |

**Pros:**
- No padding → Flash fires for ALL contexts → 2–5× faster training
- Clean training signal: every gradient comes from real PSG data
- **Fairest comparison**: each model is evaluated on the epochs where it operates under
  its full promised context. The performance curve truly answers "does more context help?"
- Items/epoch drops substantially (fewer edge epochs) → faster epochs

**Cons:**
- Test set shrinks with context length. At 240m, ~50% of epochs are excluded from evaluation.
  This must be stated clearly in the Methods section.
- Cannot score the very beginning/end of the night with long-context models.

**Paper defence:**
> "We restrict both training and evaluation to anchor epochs where the full context window
> is available within the recording (anchor ≥ L/2 from both boundaries). This ensures no
> padding is introduced, enables efficient Flash attention, and allows a fair comparison
> across context lengths: each model is evaluated under identical conditions — given its full
> promised context. We show in the appendix that including edge epochs (with padding) does
> not change the relative ordering of context lengths."

---

### Option B — Causal / Past-Only Context

Context window = L minutes *before* the anchor epoch (no future signal).

**Pros:** No padding needed at start-of-night; clinically motivated (real-time scoring).  
**Cons:** Different task (measures "how much past helps" not "how much context helps");
not comparable with literature using bidirectional context; future signal is also informative
for staging (especially at night transitions).  
**Verdict:** Reserve for a sensitivity analysis or future work. Not the primary design.

---

### Option C — All Epochs with Padding (Current)

Keep all epochs; pad missing signal.

**Verdict:** Not recommended as primary. Keeps Flash disabled at most contexts, confounds
the context-length analysis (long-context models are penalized by serving mostly-padded
windows), and makes training prohibitively slow at 120m and 240m.

---

## 3. K Does Not Apply to seq2seq — Use All Complete-Context Epochs

The K concept is a **seq2label aggregation mechanism**: sample K windows per subject at
inference, aggregate predictions across them (majority vote / mean probability) to get one
subject-level score. K controls how much of the night is used to classify the subject.

For seq2seq there is no subject-level aggregation. Each epoch is predicted independently
and its prediction stands alone. There is no K to vary and no majority vote to compute.

**Training:** use ALL complete-context anchor epochs per subject. No K limit. Items/epoch
grows with context length (a 240m window contributes 2880 anchor epochs; a 30s window
contributes 6) — this is unavoidable and correct.

**Evaluation:** compute metrics (kappa, per-stage recall, AUROC) averaged over ALL
complete-context test epochs. The epoch count differs across context lengths (fewer at
longer contexts since edge epochs are excluded) but the metrics remain comparable — they
are epoch-averaged quantities. Report epoch counts per context in the Methods.

**Do epoch counts need to match across contexts?**

Strictly no — kappa and AUROC are averages. But the epoch *composition* differs:
short contexts (30s) include sleep-onset epochs (first ~0 min excluded) while long
contexts (240m) exclude the first/last ~2 hours. The epochs at recording boundaries
are predominantly Wake and N1 transitions:
- Wake (first and last few minutes): usually easy to stage (strong alpha activity)
- N1 transitions (sleep onset): hardest class (~5% prevalence, lowest recall)

Net effect: short contexts include slightly more hard N1 epochs in their test set
→ their kappa is slightly deflated relative to 240m → the context-length curve
**understates** the benefit of long context. This is a conservative bias, not a
liberal one, but it is still a confound.

**→ See §9 for the post-hoc Common Evaluation Set analysis that resolves this.**

**Paper defence (primary results):**
> "At each context length L, we evaluate on all anchor epochs where the full L-length
> context window is available within the recording. The number of evaluated epochs
> decreases as L increases (by design — a 240-min window cannot be centred on epochs
> within the first or last 120 min of the recording). Cohen's kappa and per-stage
> recall are epoch-averaged metrics; we additionally verify in the Supplementary
> that restricting all models to the common evaluation set (anchors valid at 240m)
> does not change the qualitative trend."

**Consequence for the analysis pipeline:** the K-sweep in `analyze_windows.py --k-dense`
is meaningless for seq2seq and should be skipped. The primary analysis tool is the
saturation curve (context length → kappa) which reads directly from `summary.csv`.

---

## 3b. Common Evaluation Set Problem and Post-hoc Analysis

### The problem

When `seq2seq_padding_policy` is `"complete_only"` or `"max_fraction"`, each context
length evaluates on a different subset of anchor epochs:

| Context | Excluded per end (centered) | Approximate % of 8h night evaluated |
|---------|----------------------------|--------------------------------------|
| 30s     | ~0 min                     | ~100%                                |
| 10m     | ~5 min                     | ~98%                                 |
| 40m     | ~20 min                    | ~92%                                 |
| 80m     | ~40 min                    | ~83%                                 |
| 120m    | ~60 min                    | ~75%                                 |
| 240m    | ~120 min                   | ~50%                                 |

Comparing kappa at 30s vs 240m is comparing on **different epoch populations**.
The bias direction: short contexts see harder sleep-onset N1 epochs → kappa
slightly deflated → context-length effect is *understated*, not overstated.

This applies to both `complete_only` and `max_fraction` (any policy that excludes
different fractions of the night at different context lengths).

It does NOT apply to `allow_all` (all contexts evaluate all epochs, equal populations).

### The solution: common evaluation set

Restrict ALL context lengths to the same anchor set: the epochs that are valid at
the **longest context** (240m by default). This is the intersection of all per-context
valid sets.

Implementation: the 240m inference parquet already contains only 240m-valid anchors.
For each other context, filter its parquet to rows whose
`(subject_id, dataset, anchor_patch_end)` triple appears in the 240m parquet.

The `anchor_patch_end` column is saved in the inference parquet by
`infer_subject_windows.py` for all seq2seq tasks.

### What goes where in the paper

| Analysis | Results section |
|---|---|
| Each model evaluated on its own complete-context set | **Main results** (Table 1, Figure 1) |
| All models evaluated on the 240m-valid common set | **Supplementary** (robustness check) |

The supplementary analysis defends against the reviewer comment
*"Your short-context and long-context models were evaluated on different epoch
subsets — the comparison may not be fair."*

If the qualitative trend is the same in both analyses (expected), the paper is robust.

### Code

```bash
# Run after inference for all contexts
python scripts/analyze_common_eval_set.py \
    --config configs/phase0_v3_config.yaml \
    --task sleep_staging --head lstm \
    --split test

# Output: results/phase0_v3/inference/sleep_staging_lstm/common_eval_test_summary.csv
# Columns: context_length, common_kappa, common_auroc, common_balanced_accuracy,
#          common_recall_class{0-4}, n_epochs_common, n_epochs_all, common_fraction
```

Use `plot_saturation.py` side-by-side on `summary.csv` (main) and
`common_eval_test_summary.csv` (supplementary) to produce the two versions of
the saturation curve.

---

## 4. Architecture Issue (Must Fix Before Final Runs)

V1 (SHHS+MrOS+APPLES, 3 datasets) used **2-layer bidirectional LSTM, hidden=256 → 3.16M params**.
V3 currently uses **1-layer bidirectional LSTM, hidden=128 → 658K params** (4.8× smaller).

V3 is significantly worse across all contexts and stages (Kappa: ~0.58 → ~0.54 at 10m;
N3 recall: 0.78 → 0.60). This is primarily the architecture downgrade, not the STAGES
addition. **Restore hidden=256, num_layers=2 before final runs.**

Config change needed in `configs/phase0_v3_config.yaml`:
```yaml
model:
  hidden_dim: 256   # was 128
  num_layers: 2     # was 1
```

---

## 5. Analysis Plan for Sleep Staging

### 5A. Analyses from run_analysis.sh That WORK for seq2seq

`analyze_windows.py` already handles seq2seq correctly: it reports **segment-level only**
(no majority-vote or mean-prob aggregation, which don't apply to per-epoch prediction).
Metrics: AUROC, BalAcc, MacroF1, Cohen's Kappa per context and K.

| Step | Script | Applicable? | Notes |
|------|--------|-------------|-------|
| 1. analyze --k-dense | analyze_windows.py | ❌ Skip | K sweep is a seq2label concept (aggregation over windows). For seq2seq every epoch is independent — there is nothing to aggregate. Metrics are already in summary.csv from training. |
| 2. collect | collect_results.py | ✅ Yes | Aggregates summary.csv; works as-is. |
| 3. build-heatmap | build_heatmap_df.py | ❌ Skip | Requires k-dense analysis output; K axis has no meaning for seq2seq. |
| 4. iso-compute | plot_iso_compute.py | ❌ Skip | Depends on heatmap df; K-based compute tradeoff irrelevant. |
| 5. saturation | plot_saturation.py | ✅ **Primary** | Core plot. Use Kappa as primary metric (add kappa to --metric). Reads summary.csv directly — no analyze step needed. |
| 6. scaling-laws | plot_scaling_laws.py | ✅ Yes | Context-length scaling law; relevant. |
| 7. calibration | plot_calibration.py | ✅ Yes | Per-epoch softmax calibration; meaningful for seq2seq. |
| 8. window-position | plot_window_position.py | ⚠️ Reinterpret | "Window position" = epoch's index within its context window → becomes time-of-night performance. Needs relabelling for interpretability. |
| 9. subject-consistency | plot_subject_consistency.py | ✅ Yes | Hard subjects = consistently misclassified regardless of context. Meaningful. |
| 10. cohort-saturation | plot_cohort_saturation.py | ✅ Yes | Performance by dataset (SHHS vs MrOS vs STAGES vs APPLES). |
| 11. precision-recall | plot_precision_recall.py | ✅ Yes | Per-class PR curves; critical for minority N1 class. |
| 12. subject-kstar | plot_subject_kstar.py | ❌ Skip | K* = minimum K for correct subject-level prediction. Not applicable to per-epoch seq2seq. |
| 13. task-comparison | plot_task_comparison.py | ✅ Yes | Include sleep_staging alongside binary tasks for cross-task comparison. |

### 5B. Sleep-Staging-Specific Analyses to Add

These don't exist yet and would strengthen the paper significantly:

#### Per-stage saturation curves
Plot each stage's recall/F1/AUROC vs context length separately.
Expected finding: N1 and N3 benefit most from longer context (harder stages); W and REM
plateau early. This is the most paper-worthy sleep-staging-specific plot.

```bash
# Conceptual command — needs a new script: plot_per_stage_saturation.py
python scripts/plot_per_stage_saturation.py \
  --task sleep_staging --head lstm \
  --results-dir ... --split test
```

#### Confusion matrix evolution
5×5 confusion matrix at each context length: W, N1, N2, N3, REM.
Shows which confusions are resolved by more context (e.g., N1/W confusion at 30s →
resolved at 40m; N3/N2 confusion persists even at 240m).

#### Time-of-night performance
Divide the night into 1-hour bins. Compute Cohen's kappa per bin vs context length.
Shows whether context helps more at night transitions (lights-out, REM cycles) than
in the middle of stable sleep.

#### Stage transition accuracy
Separate evaluation on epochs that are at a sleep stage transition vs mid-stage epochs.
Transitions (e.g., N2→N3) are the hardest cases and most sensitive to context length.

#### N1 deep-dive
N1 is ~5–8% of epochs (most imbalanced). Plot N1 recall and precision separately across
context lengths. N1 is where the context-length story is probably strongest.

---

## 6. Primary Metrics for the Paper

For seq2seq sleep staging, use in this order of importance:

1. **Cohen's Kappa** — standard in sleep staging literature; accounts for chance agreement
2. **Per-stage recall** (W, N1, N2, N3, REM) — reviewer expectation for staging papers
3. **Macro-F1** — summary of per-stage performance
4. **AUROC** (macro OvR) — for context-length curve; comparable across tasks
5. **Accuracy** — report but not primary (inflated by N2 majority class)

Do NOT report accuracy alone. N2 is ~50% of epochs so a trivial classifier gets ~50% accuracy.

---

## 7. Decision Checklist

- [ ] Implement Option A (complete-context filtering) in ContextWindowDataset for seq2seq
- [ ] Restore architecture: hidden_dim=256, num_layers=2 in phase0_v3_config.yaml
- [ ] Rerun sleep_staging_lstm all 6 contexts with new design
- [ ] Rerun sleep_staging_transformer all 6 contexts with NO_WANDB=1
- [ ] Add `sleep_staging` to SEQ2SEQ_TASKS set in run_analysis.sh / gen_commands.py
      (already in analyze_windows.py; check plot scripts handle 5-class correctly)
- [ ] Write plot_per_stage_saturation.py
- [ ] Write plot_confusion_evolution.py
- [ ] Run run_analysis.sh on sleep_staging after inference completes
- [ ] Run per-stage and confusion analyses

---

## 8. Implementation (completed on sleep-stage-redesign branch)

### Code changes

**`src/nsrr_tools/datasets/context_window_dataset.py`**

Three new config params read from `cfg["dataset"]`:

| Param | Type | Default | Meaning |
|---|---|---|---|
| `seq2seq_context_mode` | str | `"causal"` | `"causal"` or `"centered"` |
| `seq2seq_padding_policy` | str | `"allow_all"` | `"allow_all"`, `"max_fraction"`, or `"complete_only"` |
| `seq2seq_max_padding_fraction` | float | `0.5` | threshold for `"max_fraction"` policy |

**Context mode semantics (N = total patches, 6 patches = 1 epoch = 30 sec):**

- `causal`: window = `[anchor_end - N : anchor_end]` — past N patches ending at anchor.
- `centered`: window = `[anchor_start - half_past : anchor_end + half_future]`
  where `half_past = (N - 6) // 2`, `half_future = N - 6 - half_past`.

  | Context | half_past | half_future | Excluded from each end |
  |---------|-----------|-------------|------------------------|
  | 30s  (N=6)    | 0    | 0    | 0 epochs (~0 min)     |
  | 10m  (N=120)  | 57   | 57   | 9 epochs (~4.75 min)  |
  | 40m  (N=480)  | 237  | 237  | 39 epochs (~19.75 min)|
  | 80m  (N=960)  | 477  | 477  | 79 epochs (~39.75 min)|
  | 120m (N=1440) | 717  | 717  | 119 epochs (~59.75 min)|
  | 240m (N=2880) | 1437 | 1437 | 239 epochs (~119.75 min each end)|

**Padding policy:**

- `allow_all`: existing behaviour. Causal mode still applies legacy `min_past` filter
  (`min_past_denom=8`, `max_min_past_patches=240`). Centered includes all anchors.
- `max_fraction`: exclude anchors where `total_padding / N > max_padding_fraction`.
- `complete_only`: exclude anchors with any padding (zero-padding guarantee).
  For centered: requires `anchor_start >= half_past AND anchor_end + half_future <= T_eff`.
  For causal: requires `anchor_end >= N`.

**Default values preserve existing behaviour exactly** — old results can be reproduced
without changing any call site.

New methods added:
- `_get_causal_window(emb, T, anchor_patch_end)` — extracted from old `_get_seq2seq_window`
- `_get_centered_window(emb, T, anchor_patch_end)` — new symmetric window extraction
- `_get_seq2seq_window` now dispatches to the above based on `seq2seq_context_mode`

### Config (`configs/phase0_v3_config.yaml`)

New params added under `dataset:` with their defaults. Model size updated for sleep staging.

**Results directory → model size mapping:**

| Results directory | context_mode | padding_policy | hidden_dim | num_layers |
|---|---|---|---|---|
| `sleep_staging_lstm_old_arch128` | causal | allow_all | 128 | 1 |
| `sleep_staging_transformer_old_arch128` | causal | allow_all | 128 | 1 |
| `sleep_staging_lstm` (new) | centered | complete_only | 256 | 2 |
| `sleep_staging_transformer` (new) | centered | complete_only | 256 | 2 |

### analyze_windows.py

Already handles seq2seq correctly (segment-level only, Kappa). No change needed for
the standard pipeline.

### plot scripts

Most scripts operate on parquet files (agnostic to window design). They work for
sleep_staging as-is. Exception: `plot_window_position.py` — its position axis represents
epoch position within the context window; for seq2seq the meaningful axis would be
time-of-night, which needs a new script.
