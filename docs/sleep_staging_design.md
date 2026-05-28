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

## 3. K Strategy for seq2seq

Since K=5 doesn't apply directly to seq2seq, define a new K criterion:

**Recommended: K_windows = 5 (complete-context windows per subject)**

With Option A, each subject contributes all complete-context anchor epochs. To match the
seq2label fairness criterion (5 context windows per subject per training epoch), we can
**randomly sample 5 non-overlapping complete-context windows** and use all anchor epochs
within them. This gives:

| Context | Epochs per 5 windows | Items/subject |
|---------|---------------------|--------------|
| 30s     | 5 × 6 = 30          | 30           |
| 10m     | 5 × 120 = 600       | 600          |
| 40m     | 5 × 480 = 2400      | 2400         |
| 80m     | 5 × 960 = 4800      | 4800         |
| 120m    | 5 × 1440 = 7200     | 7200         |
| 240m    | 5 × 2880 = 14400    | 14400        |

Total training items: ~1700 subjects × K_windows epochs = 51K–24M depending on context.
This grows with context, which is unavoidable — longer contexts inherently contribute more
prediction targets per window. 

**Alternative: fixed epoch cap** (e.g., 2880 epochs/subject regardless of context):
Each subject contributes at most N_max = 2880 anchor epochs per training epoch. This keeps
items/epoch roughly constant across contexts but requires different sampling logic.

**Recommended for paper:** K_windows=5 (consistent with seq2label fairness argument; cite
both criteria in Methods). Run sensitivity check with epoch-cap strategy.

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
| 1. analyze --k-dense | analyze_windows.py | ✅ Yes | K = number of scored epochs; interpret as "how many epochs needed for stable metrics" |
| 2. collect | collect_results.py | ✅ Yes | |
| 3. build-heatmap | build_heatmap_df.py | ⚠️ Modified meaning | K axis = epoch count, not window count; still informative |
| 4. iso-compute | plot_iso_compute.py | ⚠️ Weaker | Compute-efficiency tradeoff less clear for seq2seq |
| 5. saturation | plot_saturation.py | ✅ Primary | Core context-length curve; use Kappa as primary metric |
| 6. scaling-laws | plot_scaling_laws.py | ✅ Yes | |
| 7. calibration | plot_calibration.py | ✅ Yes | Are per-epoch softmax probs calibrated? |
| 8. window-position | plot_window_position.py | ⚠️ Reinterpret | Position = epoch index in night → becomes "time-of-night performance" |
| 9. subject-consistency | plot_subject_consistency.py | ✅ Yes | Hard subjects = always misclassified across contexts |
| 10. cohort-saturation | plot_cohort_saturation.py | ✅ Yes | SHHS vs MrOS vs STAGES vs APPLES |
| 11. precision-recall | plot_precision_recall.py | ✅ Yes | Especially important for minority N1 class |
| 12. subject-kstar | plot_subject_kstar.py | ⚠️ Weak | K* doesn't map cleanly to seq2seq |
| 13. task-comparison | plot_task_comparison.py | ✅ Yes | Include sleep_staging alongside binary tasks |

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
- [ ] Implement K_windows=5 sampling for seq2seq (5 non-overlapping complete windows/subject)
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

## 8. What Changes in the Code

### ContextWindowDataset (seq2seq mode)

Current behaviour: `windows_per_subject=5` is ignored; all anchor epochs are included
including edge epochs → K≈966, padding present.

Needed changes:
1. In seq2seq mode, filter anchor epochs to complete-context only:
   `anchor_patch_idx >= half_N  AND  anchor_patch_idx <= T - half_N`
2. Implement window-based K sampling: group anchors into non-overlapping windows of size N,
   sample K_max such windows per subject, include all anchors within them.

### analyze_windows.py

Already handles seq2seq correctly (segment-level only, reports Kappa). No change needed
for the standard pipeline. Add `--plot-metric kappa cohen_kappa` support if not present.

### plot scripts

Most scripts operate on parquet files (subject_id, true_label, pred_label, prob_class*)
and are agnostic to seq2seq vs seq2label. They should work for sleep_staging as-is.
Exception: `plot_window_position.py` — its "position" axis needs relabelling as
"time of night (epoch index)" for seq2seq to be interpretable.
