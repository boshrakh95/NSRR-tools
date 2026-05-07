# Context Length Experiment Design
## Research Questions, Experimental Design, and Analysis Plan

---

## 1. The Central Research Question

> **Does training a model on longer PSG context windows teach it something that aggregating many short-context windows cannot recover — and if so, how much context is actually needed?**

This is one question, not two. It has two interacting components:

- **Training context length (L):** what the model learns to integrate within a single forward pass
- **Inference aggregation (K):** how many windows per patient you average at prediction time

These two dimensions define the space of possible deployment strategies. The central question is: for a fixed total signal budget per patient (K × L = constant), which configuration is best?

If the answer is "they're all about equal" → short models with majority voting are sufficient; no expensive long-context training needed.  
If the answer is "larger L is better, even at the same K×L budget" → there are long-range temporal dependencies in PSG that require end-to-end long-context learning; aggregation alone cannot substitute.

This is a meaningful and useful result for the field either way.

---

## 2. What the 2D Grid Actually Captures

### The experiment grid

You train one model per context length L. For each trained model, you run inference with all available windows, then post-hoc sweep K. This gives a grid:

|  | K=1 | K=5 | K=10 | K=20 | K=50 | K=all |
|--|-----|-----|------|------|------|-------|
| **L=30s** | model₃₀ₛ, K=1 | model₃₀ₛ, K=5 | … | | | |
| **L=10m** | model₁₀ₘ, K=1 | model₁₀ₘ, K=5 | … | | | |
| **L=40m** | model₄₀ₘ, K=1 | … | | | | |
| **L=80m** | model₈₀ₘ, K=1 | … | | | | |
| **L=120m** | model₁₂₀ₘ, K=1 | … | | | | |

**Moving along a row** (fixed L, varying K): how does inference-time aggregation help a given model? This is free — answered from the already-inferred parquets with no retraining.

**Moving along a column** (fixed K, varying L): does training on longer sequences help when you use the same number of windows at inference? This requires separately trained models at each L.

**Moving along an iso-compute diagonal** (K × L = constant, e.g., all cells where K × L_minutes ≈ 80 minutes): this is the central comparison. Each cell on the diagonal represents a different trade-off: few long windows (large L, small K) vs many short windows (small L, large K). All cells on the same diagonal give a patient the same total signal at inference. Which configuration works best?

### What the diagonals actually test

A 10m model with K=8 and an 80m model with K=1 both consume approximately 80 minutes of PSG per patient at inference. But they are fundamentally different:

- The 10m model was trained to extract features from a 10-minute window. It never learned to integrate signals across an 80-minute span. Its 8 predictions are averaged by you, externally.
- The 80m model was trained to integrate 80 minutes in one forward pass. Long-range temporal dependencies within that 80 minutes are captured inside the model.

If these two strategies give the same AUROC: averaging does the work, the model doesn't need to see long context.  
If the 80m model is better: there is something the network learns from long-context training that averaging destroys.

---

## 3. Separating Train-Time and Inference-Time Effects

This is the source of the conceptual confusion, so it is worth being explicit:

| | Controlled by | Can be explored without retraining? |
|--|--------------|--------------------------------------|
| Training context length L | registry `contexts` list | No — each L is a separate trained model |
| Inference K (windows/subject) | `analyze_windows.py --k-values` | **Yes** — post-hoc subsampling of existing parquets |
| Training K (windows/subject/epoch) | `windows_per_subject` in config | No — separate training run per K value |
| Aggregation method (mean-prob, majority-vote) | `analyze_windows.py` | **Yes** — same parquet, different aggregation |

**The K dimension of the heatmap is free.** After running inference with all windows (which you already do), you can sweep any K you want in `analyze_windows.py` without touching the GPU. The only thing that costs compute is adding new rows to the grid (new context lengths L, or training K ablations).

### Training K (K_train)

`windows_per_subject` in your config (currently 5) controls how many windows per subject are sampled each training epoch. This is separate from the inference K that you sweep post-hoc.

The tension:
- **K_train = 5:** Each epoch, the model sees 5 randomly-sampled windows per subject, regardless of context length. This makes training exposure roughly equal across L values. But the model learns to do well from just 5 windows — it may not be optimized for large-K inference.
- **K_train = all:** The model sees all available windows. For 30s context (~960 windows/subject), this means far more training signal per epoch than for 120m (~4 windows/subject). More training data, but an unfair comparison across L values.
- **K_train = token budget:** K_train × L_minutes = constant. For L=30s, K_train≈160; for L=80m, K_train≈1. Each subject contributes the same total signal per epoch regardless of L. This is the principled approach for comparing context lengths fairly.

**K_train = 5 is a defensible and common choice** — it is not wrong. But it means models trained at short contexts have seen their windows many fewer times relative to the available data. Document this and, optionally, run one ablation to verify it doesn't change conclusions.

---

## 4. Research Questions for the Paper

Framed as testable hypotheses:

**H1 — Context saturation:** Subject-level prediction performance increases with training context length L and saturates at some task-specific threshold L*. Beyond L*, additional context provides negligible benefit.

*Answered by:* the L-axis of the grid (K fixed at "all" or a large value). Look for the L value where AUROC stops improving.

**H2 — Aggregation substitution:** For a fixed inference budget (K × L = constant), performance is approximately equal regardless of how that budget is split between L and K. In other words, aggregating many short-context predictions is as good as one long-context prediction.

*Answered by:* the iso-compute diagonals of the heatmap. If cells along each diagonal are flat → aggregation substitutes. If cells slope toward large L → it doesn't.

**H3 — Aggregation saturation:** For a fixed trained model (fixed L), subject-level performance saturates after K windows. There is a minimum K at which performance is near-optimal.

*Answered by:* each row of the grid (L fixed, K varying). The saturation point tells you the minimum number of windows needed per patient in deployment.

**H4 — Aggregation method:** Mean-probability aggregation outperforms majority-vote at small K, converging at large K. (Already supported by the existing analyze_windows output.)

These four hypotheses are clear, answerable with your current pipeline, and directly useful for the field.

---

## 5. What Context Lengths You Actually Need

### Current plan: 5 lengths

`[30s, 10m, 40m, 80m, 120m]`

This covers about 2 decades of context (0.5 min to 120 min), but has two problems:
- **Gap at the short end:** There's nothing between 30s and 10m (a 20× jump). This is where many tasks likely show the steepest improvement, so the heatmap will have a large uninformative gap at the top.
- **No anchor at the long end:** 120m is roughly half a night for many subjects. You don't know whether performance would improve further with a full night.

### Recommended additions

| Context | Why |
|---------|-----|
| `2m` | Fills the short-context gap; captures the 30s→10m regime |
| `5m` | Another point in the ascending region |
| `240m` | Midpoint between 120m and full_night |
| `full_night` | Anchors the ceiling; answers "does the whole night help?" |

With `[30s, 2m, 5m, 10m, 40m, 80m, 120m, 240m, full_night]` the heatmap covers 3 decades of context with roughly even logarithmic spacing.

**Priority:** If compute is limited, add `5m` and `full_night` first. These anchor both ends of the regime where you expect the curve to move.

**Note on full_night:** Variable-length sequences require `collate_fn` padding. Transformer head must be skipped (O(N²) memory). Already supported in your codebase.

---

## 6. Experimental Plan

### Tier 1 experiments (full comparison, all three heads)

Run all four Tier 1 tasks × recommended 9 context lengths × 3 heads.  
This is the set that answers H1–H4 comprehensively.

For paper: probably show the full grid for one task (sex_binary or bmi_binary) and a summary comparison across tasks.

### Tier 2 experiments (lstm only)

Run the 4 Tier 2 tasks × 5 original context lengths × lstm only. These are lower priority — run after Tier 1 is done.

### Ablation: training K

Run one task (recommended: `sex_binary_lstm`) with two additional training K settings:

1. `run_tag: kall` — `windows_per_subject: 9999` (train on all available windows)
2. `run_tag: kbudget` — `windows_per_subject = budget_min / L_min` where budget = 80 min (token budget approach)

Use the same inference and analysis pipeline. If the heatmap shape is qualitatively similar to the K=5 baseline, your main conclusions are robust to this choice. If not, use the token budget approach for the main results.

This ablation requires 2 × 9 = 18 additional training jobs for one task.

### Seq2seq tasks (sleep staging)

For sleep staging, subject-level aggregation doesn't apply (each 30-second epoch has its own label). The research questions simplify:
- H1 (context saturation): Does segment-level accuracy increase with L?
- H4 does not apply (no aggregation)
- No heatmap needed; just the saturation curve (AUROC vs L at K=all segments)

This is a simpler analysis — run the sweep and report the saturation curve.

---

## 7. Outputs and Figures for the Paper

### Figure 1: Saturation curve

For each task, one line per head: test AUROC vs context length L (at K=all windows). This is the primary result for H1. Shows where performance saturates and whether heads diverge.

*Already constructable from `summary.csv` once all L values are trained.*

### Figure 2: The 2D heatmap

For one representative task (e.g., `sex_binary_lstm`): a grid with L on the Y-axis, K on the X-axis, cell color = test AUROC (mean-prob aggregation). Iso-compute contour lines overlaid (where K × L_minutes = constant). This is the primary result for H2 and H3.

*Requires a new script `plot_context_heatmap.py`.*

### Figure 3: Row curves (K-sweep per context)

For each context length L, a line showing how AUROC improves with K (also showing mean-prob vs majority-vote). This is Figure 2 in a different projection and is cleaner for showing H3.

*Already generated by `analyze_windows.py --plot`.*

### Figure 4: Head comparison (optional)

For the same task, one heatmap per head side-by-side (or the saturation curve with one line per head). Answers: does temporal modeling (lstm/transformer) outperform simple mean-pooling, and does the advantage grow with L?

### Supplementary: iso-compute curve

For each iso-compute diagonal: one line showing how AUROC varies as you move from small L (many short windows) to large L (one long window), at fixed total signal budget. This is the sharpest answer to H2.

---

## 8. What Existing Scripts Cover (and What Needs to Be Built)

| Figure / analysis | Script | Status |
|---|---|---|
| Saturation curve (AUROC vs L) | `collect_results.py` or manual from `summary.csv` | Needs minor script (read CSVs across context lengths) |
| Per-context K-sweep table + line plot | `analyze_windows.py` | **Already implemented** |
| 2D heatmap with iso-compute lines | `plot_context_heatmap.py` | **Needs to be written** |
| Head comparison plots | Run `analyze_windows.py` separately per head | Needs wrapper or manual |
| Iso-compute curve | Part of heatmap script | **Needs to be written** |

---

## 9. Confounds to Document

**9.1 Subjects available per context length differ.** At long contexts, some subjects have fewer available windows and may be dropped from training if they have zero non-overlapping windows. Your logs show slight N differences across contexts. This is an inherent confound of long-context training — document it and check that test set subjects are consistent.

**9.2 Training K creates a deployment distribution shift.** Models trained with K=5 were never trained to predict from a single window (K=1) or from 50 windows (K=50). The K-sweep tests them outside the training regime. This is expected and acceptable but should be noted.

**9.3 ISO-compute diagonals compare different models.** When you compare (L=10m, K=8) vs (L=80m, K=1) on the same diagonal, you are comparing two differently-trained models, not the same model in two deployment settings. The comparison is meaningful for the research question but is not a controlled experiment in the strict sense.

**9.4 Aggregation quality depends on window diversity.** For K=5 from 960 possible 30s windows, the 5 sampled windows may be redundant (e.g., all from the first hour of sleep). Evenly-spaced selection (your current strategy) mitigates this. Document the selection strategy.

**9.5 Multi-dataset subjects.** Your training set spans APPLES, SHHS, MrOS. Dataset-level effects may interact with context length (some cohorts have shorter recordings). If possible, check whether results hold within each dataset separately.

---

## 10. What to Ask Your Supervisor

Before adding new context lengths (which costs significant GPU time), align on:

1. **Is H2 (aggregation substitution) central to the paper, or supplementary?** If central, you need the full heatmap, which requires more context lengths. If supplementary, the saturation curve (H1) may be the primary figure and the heatmap a secondary one.

2. **Which task is the "hero" experiment?** The full heatmap with all 9 context lengths and the ablation is expensive to run for all tasks. Agree on one primary task (probably `sex_binary` — large N, clean binary label, already shows strong signal) for the detailed analysis, and others for replication.

3. **Is training K an ablation or the main method?** The token-budget approach is more principled but adds training cost. If the supervisor agrees K=5 is a reasonable design choice (comparable to what's done in the EHR/ECG literature), skip the ablation and note the limitation.

4. **Seq2seq (sleep staging):** Is this in scope for this paper, or a separate paper? It has different analysis (no aggregation; purely context saturation) and may dilute the message if combined.

---

## 11. Mock Plots vs. Current Pipeline: Gap Analysis

This section documents the comparison between the mock visualizations in `mock-compute-optimal-tradeoffs-plots-main/` (PLOTS.md, plot_binary.py, plot_heatmap.py) and the current experiment design and code. It records what needs to change before results can feed directly into those plots.

### 11.1 Where mock plots and current design agree

The conceptual framing is fully consistent:
- The heatmap axes (context length on Y, K on X, metric as color) match Section 2 of this document exactly.
- Iso-compute lines defined as `K × L_minutes = constant` — same definition used here.
- "Pareto front" plot (Plot 4) maps to H2 of Section 4: which (L, K) config is optimal at each total budget?
- "Min-cost frontier" (Plot 5) maps to H3: what is the minimum compute to reach a target performance?
- "Double tradeoff" (Plot 7) — "should you double K or double context?" — is the most direct operationalization of H2.
- `total_compute_min = context_length_min × k` is the right definition of total signal consumed per patient at inference.

### 11.2 Context lengths: mock uses 9, current plan uses 5

The mock scripts use: `[0.5, 5, 10, 20, 40, 80, 120, 180, 240]` minutes (= "30s, 5m, 10m, 20m, 40m, 80m, 120m, 180m, 240m").

The current registry has: `[30s, 10m, 40m, 80m, 120m]`.

Missing from current plan:
- **5m** (or 2m): fills the steep-ascent regime between 30s and 10m — likely where most tasks improve fastest
- **20m**: fills the 10m→40m gap
- **180m or 240m**: fills the gap before full_night and anchors the long end

Without these, the heatmap has only 5 rows and large uninformative gaps. The Pareto front and "double tradeoff" plots will have coarse transitions between context length regimes. For a paper figure, 5 rows is marginal; 8–9 rows is much more convincing. **Action:** add at minimum `5m` and `240m` to Tier 1 experiment contexts.

### 11.3 K sweep density: current pipeline has 6 discrete values, plots need ~20+

Current `analyze_windows.py` default: `K_VALUES = [1, 5, 10, 20, 50, "all"]` — only 6 points per context row.

Mock plots sweep K continuously up to `floor(budget / L)`, using approximately these values:
```
[1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 60, 80, 96, 120, 160, 192, 240, 320, 384, 480, 640, 768, 960]
```

With only 6 points per row, the heatmap cells will be very coarse, the Pareto front will have jagged transitions, and the marginal-gain and min-cost curves will be nearly unusable.

**What to change in `analyze_windows.py`:** Add a `--k-dense` flag that sweeps a much larger set of K values. A practical set for real data (where K is naturally capped by available windows):
```
[1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100, 120, 160, 200, 250, 320, 400, 500, "all"]
```
In practice, most subjects at 80m or 120m have fewer than 10 windows, so large K values will simply be skipped ("K > max available — skipping") — no wasted compute. The flag adds negligible runtime since the parquet is already loaded.

The default (sparse) K list stays for the quick per-context markdown table. The dense list is used only when building the heatmap DataFrame.

### 11.4 Output format mismatch: column names

The mock plot functions expect a single flat DataFrame with columns:

| Expected column | Current `analyze_windows.py` output | Fix |
|---|---|---|
| `context_length_min` | not present | compute from `context_length` string |
| `context_label` | `context_length` (string, e.g. "10m") | rename |
| `k` | `k` (int or "all") | replace "all" with numeric max |
| `accuracy` / `auroc` / `f1` | `mean_prob_auroc`, `seg_auroc`, etc. | choose aggregation method, rename |
| `total_compute_min` | not present | `context_length_min × k` |

The current pipeline also outputs one CSV per experiment split (`window_analysis_test.csv`) that **does include all context lengths** as rows, just with different column names. So the data structure is right; only the column names and a few derived columns are missing.

**What to build:** A new script `scripts/build_heatmap_df.py` that:
1. Reads `window_analysis_test.csv` for a given (task, head) — or re-runs the K sweep from parquets with the dense K list
2. Renames/derives the required columns (convert "10m"→10.0 for `context_length_min`, select `mean_prob_auroc` as the primary metric column named `auroc`, compute `total_compute_min`)
3. Saves a single combined DataFrame: `heatmap_df_{task}_{head}.csv` ready for the plot functions

**String-to-minutes conversion function needed:**
```python
def ctx_to_minutes(s: str) -> float:
    if s.endswith("s"):  return float(s[:-1]) / 60
    if s.endswith("m"):  return float(s[:-1])
    if s.endswith("h"):  return float(s[:-1]) * 60
    if s == "full_night": return 480.0  # approximate
    raise ValueError(f"Unknown context format: {s}")
```

### 11.5 Budget parameter: mock uses 16h, real data needs ~8h

The mock uses `budget_min = 960` (16 hours). Real PSG studies are ~7–8 hours. This matters for:
- Which cells are NaN in the heatmap (cells where `K × L > budget`)
- The x-axis range of the Pareto front and min-cost plots
- The upper end of the marginal-gain plot

**Recommended `budget_min` for real data:** Use the 90th percentile of actual recording lengths across your datasets, in minutes. Approximately 480 minutes (8 hours) for APPLES/SHHS/MrOS. This should be configurable in `build_heatmap_df.py` and the plot scripts.

For K caps in practice: a 30s model from an 8h study has at most ~960 windows; a 120m model has at most ~4. The heatmap will automatically show NaN (grey) for cells that exceed actual available data.

### 11.6 Additional plots that require pipeline additions

Beyond the 7 core plots in PLOTS.md, the mock `plot_binary.py` includes three additional plots:

**Plot A — ROC curves at iso-compute budgets:**
Requires: for each (iso-budget, L) pair, select K=floor(iso/L) windows per subject, take their `prob_class1` scores, and compute a ROC curve. 
Doable from existing parquets (they contain `prob_class0`, `prob_class1` per window). Needs a new script section, not a pipeline change.

**Plot B — Recall at fixed precision:**
Requires sweeping the voting threshold (predict positive only if ≥t votes out of K, for t=1,...,K). This goes beyond the standard majority-vote (threshold = K/2). 
Doable from existing parquets but requires a loop over threshold values within `evaluate_at_k()`. Can be added as a post-processing step without rerunning training or inference.

**Plot C — Which metric picks which optimal context:**
Requires running the Pareto-front analysis for each metric separately. 
Doable once the combined heatmap DataFrame is ready — no pipeline changes.

**Priority:** Plots A, B, C are supplementary. Build the 7 core plots first.

### 11.7 `analyze_windows.py` output to check/update

Current output columns that are correctly named and can be used as-is:
- `k`, `n_subjects`, `n_segments` ✓
- `mean_prob_auroc`, `mean_prob_balanced_accuracy`, `mean_prob_macro_f1` → rename to `auroc`, `balanced_accuracy`, `f1` in the heatmap DataFrame
- `majority_auroc`, `majority_balanced_accuracy` → include as alternative aggregation columns
- `seg_auroc` → include for the "segment-level" baseline

Current output not yet present:
- `context_length_min` (float) — derive from `context_length` string
- `total_compute_min` — compute as `context_length_min × k`
- Dense K values — add `--k-dense` flag

### 11.8 Summary: what to build before running the actual plots

| Item | Priority | Requires retraining? | Notes |
|------|----------|---------------------|-------|
| Add 5m and 240m context lengths to registry | High | **Yes** | 2 new training jobs per (task, head) |
| Add `--k-dense` flag to `analyze_windows.py` | High | No | ~20 K values, 1–2h CPU |
| Write `build_heatmap_df.py` | High | No | Reads existing CSVs, renames columns |
| Write `plot_context_heatmap.py` | High | No | Adapts mock functions to real data |
| ROC at iso-compute (Plot A) | Medium | No | Reads parquet directly |
| Recall at fixed precision (Plot B) | Low | No | Adds threshold loop to evaluate_at_k |
| Metric comparison (Plot C) | Low | No | Derived from heatmap DataFrame |

---

## 12. Implementing Configurable Training K Strategy

This section describes how to add a `windows_strategy` config option to support both the current fixed-K approach and the token-budget approach, without changing any code now. Implement this when ready to run the ablation.

### 12.1 What to add to the config file

In `configs/phase0_v2_config.yaml`, under the `training:` section, add:

```yaml
training:
  # ... existing fields (lr, epochs, etc.) ...

  # How many windows to sample per subject per training epoch.
  # "fixed": always K = windows_per_subject (current behavior)
  # "token_budget": K = floor(token_budget_minutes / context_length_minutes), min 1
  #   This keeps total signal seen per subject constant across context lengths.
  windows_strategy: "fixed"        # "fixed" | "token_budget"
  token_budget_minutes: 80         # used only when windows_strategy = token_budget
  # windows_per_subject: 5         # used only when windows_strategy = fixed (already in dataset section)
```

### 12.2 What to change in `train_context_sweep.py`

In `train_one_context()`, right before the `make_ds()` call, insert:

```python
# ── Training K strategy ────────────────────────────────────────────────────
windows_strategy = t_cfg.get("windows_strategy", "fixed")
if windows_strategy == "token_budget":
    budget_min_ctx = float(t_cfg.get("token_budget_minutes", 80))
    # parse_context_length returns number of 30s steps; × 0.5 converts to minutes
    ctx_minutes = parse_context_length(context_length) * 0.5
    if not is_full_night:
        k_train = max(1, int(budget_min_ctx / ctx_minutes))
        cfg["dataset"]["windows_per_subject"] = k_train
        print(f"  Token budget: {budget_min_ctx:.0f}min / {ctx_minutes:.1f}min = K_train={k_train}")
    # full_night: always K=1 (one window per subject = one full recording)
```

This change is isolated — it only affects training-time data loading, not inference or analysis.

### 12.3 How to run the ablation

Use `run_tag` in the registry so results go to separate folders and don't overwrite the K=5 baseline:

```yaml
# In v2_registry.yaml, add these two entries (example for sex_binary_lstm):

sex_binary_lstm_kbudget:
  task: sex_binary
  task_type: seq2label
  head: lstm
  datasets: [apples, shhs]
  contexts: [30s, 5m, 10m, 40m, 80m, 120m]
  batch_size: 32
  lr: 1.0e-4
  run_tag: "kbudget"
  n_size: large
  tier: 3           # run after Tier 1 and 2

# windows_strategy and token_budget_minutes go in the config yaml, not per experiment.
# Create a separate config file for this ablation:
# configs/phase0_v2_kbudget_config.yaml — identical to v2_config but with:
#   training.windows_strategy: "token_budget"
#   training.token_budget_minutes: 80
# Then add CONFIG=configs/phase0_v2_kbudget_config.yaml to the gen_commands call.
```

Token budget of 80 minutes gives these training K values across context lengths:

| Context | Length (min) | K_train (token budget, 80m) | K_train (current fixed) |
|---------|-------------|----------------------------|------------------------|
| 30s | 0.5 | 160 | 5 |
| 5m | 5 | 16 | 5 |
| 10m | 10 | 8 | 5 |
| 40m | 40 | 2 | 5 |
| 80m | 80 | 1 | 5 |
| 120m | 120 | 1 | 5 |

Note: at 80m and 120m, both strategies give effectively K=1 or K≈1, so they will agree. The biggest difference is at short contexts (30s), where the token-budget model sees 32× more windows per epoch.

### 12.4 What the ablation tells you

If the K=5 heatmap and the K=token-budget heatmap have the same shape (even if absolute values differ slightly):
→ K=5 is robust; use it as the main method and note it in the paper.

If the shapes diverge (e.g., token-budget shows stronger advantage for short-context models):
→ The training regime confounds the comparison; use token-budget as the main method and report K=5 as a sensitivity check.

Either result is informative and publishable.
