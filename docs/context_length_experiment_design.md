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

### The three K values and their windowing pools

| K | Used where | Windowing pool | Value |
|---|-----------|---------------|-------|
| **K_train** | Training dataloader, each epoch | **Overlapping** — any start in [0, T−N], random | `windows_per_subject` (default 5) |
| **K_val** | Validation eval during training (early stopping) | **Overlapping** — evenly spaced across [0, T−N], deterministic | K_max = 5 at all context lengths |
| **K_infer** | `infer_subject_windows.py` | **Non-overlapping** stride-N | T//N (all) |

Important implications:
- **K_val = K_max = 5 at all context lengths.** Val and test during training use the overlapping pool (triggered when K_max ≤ 100), so K=5 is achieved at 240m just as at shorter contexts. The 5 positions are evenly spaced and fixed, so the early-stopping signal is equally stable across the entire context-length sweep.
- **K_infer is always T//N** because `infer_subject_windows.py` sets `windows_per_subject = 99,999`, routing to the non-overlapping stride-N branch which returns all floor(T/N) positions deterministically. Inference is unaffected.
- The overlapping pool makes K=5 achievable at all context lengths in both training and evaluation.

### Training K (K_train)

`windows_per_subject` in your config (currently 5) controls how many windows per subject are sampled each training epoch. This is separate from the inference K that you sweep post-hoc.

The tension:
- **K_train = 5:** Each epoch, the model sees 5 randomly-sampled windows per subject, regardless of context length. This equalises gradient updates per subject per epoch across all L values — the key criterion for a fair context-length comparison.
- **K_train = all:** The model sees all available windows. For 30s context (~960 windows/subject), this means ~240× more gradient updates per epoch than for 120m (~4 windows/subject). More training data, but an unfair comparison: you measure training compute, not context utility.
- **K_train = token budget:** K_train × L_minutes = constant (e.g. budget=80m → K=160 at 30s, K=1 at 80m+). Each subject contributes the same total signal per epoch. Fair in the information sense but gives the 30s model 160× more gradient updates than the 120m model — a different confound.

**K_train = 5 is the correct default for comparing context lengths.** The right fairness criterion is equal gradient updates per subject per epoch (not equal information). K=all and token-budget both introduce asymmetric training dynamics that would confound the context-length comparison. Note that K=5 controls per-epoch exposure, not total data seen across training — with enough epochs and random sampling the model covers most of the window space regardless of context length.

**Two fairness criteria, neither is universally right:**

| Criterion | K=5 fixed | Token budget |
|---|---|---|
| Equal gradient updates/subject/epoch | ✅ | ✗ (160× more at 30s) |
| Equal information/subject/epoch | ✗ | ✅ |
| Converge at long contexts (80m+) | ✅ both give K≈1 | ✅ |

Because both strategies agree at long contexts, any difference in the saturation curve shape would appear only at short contexts (30s, 10m). Running a one-task token-budget ablation (§13) directly tests this and pre-empts reviewer objections.

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

Six tasks × 6 context lengths × 3 heads:
`sex_binary`, `sleep_efficiency_binary`, `bmi_binary`, `age_class`, `apnea_binary`, `sleep_staging`

For paper: show the full heatmap grid for one representative task (bmi_binary or sex_binary) and the saturation curve across all Tier 1 tasks. `apnea_binary` is the primary OSA result. `sleep_staging` is the primary seq2seq result — primary metric is Cohen’s κ (not AUROC).

### Tier 2 experiments (lstm only)

Six tasks × 5–6 context lengths × lstm only:
`psqi_binary`, `depression_extreme_binary`, `osa_binary_apples_postqc`, `osa_severity_apples` (5 contexts, small N), `cvd_binary`, `sleepiness_binary` (6 contexts, large N)

Run after Tier 1 is complete.

### Seq2seq tasks (sleep staging)

For sleep staging, subject-level aggregation doesn’t apply (each 30-second epoch has its own label). The research questions simplify:
- H1 (context saturation): Does per-epoch accuracy / Cohen’s κ increase with L?
- H4 does not apply (no aggregation)
- No heatmap needed; just the saturation curve (κ and per-stage F1 vs L)
- Primary metric: Cohen’s κ and per-stage F1. AUROC also logged for reference.

The mean_pool head loses position information and is expected to underperform on the anchor task; its saturation curve is included as a no-temporal-context baseline.

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
| Saturation curve (AUROC vs L) | `plot_saturation.py` | **Already implemented** |
| Per-context K-sweep table + line plot | `analyze_windows.py` | **Already implemented** |
| 2D heatmap with iso-compute lines | `build_heatmap_df.py` + `plot_iso_compute.py` | **Already implemented** |
| Head comparison plots | `plot_saturation.py --heads lstm transformer mean_pool` | **Already implemented** |
| Iso-compute curve (7 plots) | `plot_iso_compute.py` | **Already implemented** |
| Collect all results into flat CSVs | `collect_results_v2.py` | **Already implemented** |
| Bootstrap 95% CIs on AUROC / bal-acc | `analyze_windows.py --bootstrap N` | **Already implemented** (config-driven via `gen_commands.py`) |
| U-shape overfitting curves | `plot_scaling_laws.py` | ✅ **Done** — reads `training.csv`; gen_commands: `scaling-laws` |
| FLOPs vs AUROC scaling law | `plot_scaling_laws.py` | ✅ **Done** — reads `training.csv`; FLOPs computed analytically from `seq_len`, `hidden_dim`, `steps_per_epoch` |
| ECE calibration + reliability diagrams | `plot_calibration.py` | ✅ **Done** — reads `*_windows.parquet`; gen_commands: `calibration` |
| Window-position probability profiles | `plot_window_position.py` | ✅ **Done** — reads `*_windows.parquet`; gen_commands: `window-position` |
| Within-subject variance across windows | `plot_subject_consistency.py` | ✅ **Done** — reads `*_windows.parquet`; gen_commands: `subject-consistency` |
| Cross-task sensitivity matrix | `plot_task_comparison.py` | ✅ **Done** — reads `analysis.csv` (multi-task); gen_commands: `task-comparison` |
| Per-dataset saturation curves | `plot_cohort_saturation.py` | ✅ **Done** — reads `*_windows.parquet` filtered by `dataset` column; gen_commands: `cohort-saturation` |
| Precision-recall curves + vote sweep | `plot_precision_recall.py` | ✅ **Done** — reads `*_windows.parquet`; gen_commands: `precision-recall` |
| K* distribution + coverage curves | `plot_subject_kstar.py` | ✅ **Done** — reads `*_windows.parquet` (not CSV); gen_commands: `subject-kstar` |
| Saturation curves with CI bands | `plot_saturation.py --collected-dir` | ✅ **Done** — reads `analysis.csv` CI columns; gen_commands: `saturation --collected-dir` |

See `docs/ANALYSIS_IDEAS.md` for the scientific motivation, expected outputs, and full implementation details for all scripts. Use `gen_commands.py` subcommands to generate the exact commands.

**Results collection inputs and outputs:**

`scripts/collect_results_v2.py` reads the raw per-run output files from scratch and writes flat CSVs and parquets:

- **Reads:** `{task}_{head}/context_{L}/training_curves.csv`, `metrics.json` (for training); `inference/{task}_{head}/window_analysis_{split}.csv` (for analysis); `inference/{task}_{head}/context_{L}/{split}_windows.parquet` (for per-window predictions)
- **Writes:** `results/collected/training.csv`, `results/collected/analysis.csv` (repo + scratch); `collected/predictions/*.parquet` (scratch only)

These flat files replace the need to read individual per-run CSVs and JSONs when making plots and tables. See `docs/RESULTS_COLLECTION.md` for full column schemas and usage examples.

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

| Item | Priority | Requires retraining? | Status | Notes |
|------|----------|---------------------|--------|-------|
| Add 240m context length to registry | High | **Yes** | ✅ Done | Added to all Tier 1 experiments |
| Add `--k-dense` flag to `analyze_windows.py` | High | No | ✅ Done | See §14 Step 1 |
| Write `build_heatmap_df.py` | High | No | ✅ Done | See §14 Step 2 |
| Write `plot_iso_compute.py` (7 plots) | High | No | ✅ Done | See §14 Step 3; replaces `plot_context_heatmap.py` |
| Write `plot_saturation.py` | High | No | ✅ Done | See §14 Step 4 |
| Integrate into `gen_commands.py` (core) | High | No | ✅ Done | See §14 Step 5; 3 subcommands |
| Write §1–§9 extended plot scripts | High | Partial | ✅ Done | All 8 scripts written; see §8 table and ANALYSIS_IDEAS.md |
| Add 9 extended subcommands to `gen_commands.py` | High | No | ✅ Done | `collect`, `scaling-laws`, `calibration`, `window-position`, `subject-consistency`, `task-comparison`, `cohort-saturation`, `precision-recall`, `subject-kstar` |
| Saturation CI bands (`plot_saturation.py`) | Medium | No | ✅ Done | `--collected-dir` flag reads bootstrap CI columns from `analysis.csv` |
| ROC at iso-compute (Plot A) | Medium | No | ⬜ TODO | See §14 Step 6 |
| Recall at fixed precision (Plot B) | Low | No | ⬜ TODO | See §14 Step 6 |
| Metric comparison (Plot C) | Low | No | ⬜ TODO | See §14 Step 6 |

### 11.9 Required changes to `analyze_windows.py` for the heatmap

The mock heatmap uses ~28 K values on a log-spaced grid:
```
[1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 60, 80, 96, 120, 160, 192, 240, 320, 384, 480, 640, 768, 960]
```
Current default is only 6 values `[1, 5, 10, 20, 50, "all"]`. With 6 columns the heatmap cells are too coarse, iso-compute lines don't land on real cells, and the Pareto/marginal plots are unusable.

**Change needed:** Add a `DENSE_K_VALUES` list and a `--k-dense` flag:
```python
DENSE_K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50,
                  60, 80, 100, 120, 160, 200, 250, 320, 400, 500, "all"]
```
Use `--k-dense` when building the heatmap DataFrame; the default sparse list stays for the quick markdown table.

K values that exceed `max_windows_available` for a context length are silently skipped — no wasted compute. For 120m context (~4 windows per subject), only K=1,2,3,4 will actually run.

---

## 12. Batch Size Protocol for the Paper

**Single protocol — batch size 32, accum_steps 1, at every context length.**

This was established after resolving two root causes of CUDA OOM at 240m on the Transformer head:

1. **Cohort filter** (`dataset.min_recording_patches=2880`): ensures all subjects across all context lengths have recordings ≥ 240m, so padding masks are always all-False. See `docs/cohort_filter.md`.
2. **Mask fix** (`TransformerHead.forward()`): passes `src_key_padding_mask=None` when the mask is all-False → PyTorch selects Flash attention (O(N) memory). Without this, a float mask (even all-zeros) triggered O(N²) Math attention, trying to allocate ~42 GB at batch=168 on a 9.75 GiB GPU — always OOM.

With both fixes, batch=32 fits at every context length (confirmed: `[Attn] Flash (mask=None, O(N) memory) | N=2880`).

**Scientific justification:** Using the same batch size at all L ensures identical training dynamics across the sweep. Same subjects per gradient update, same number of gradient steps per epoch, same stochastic noise level. Context window length is the only variable between experiments, which is exactly what the paper needs to claim.

**Paper claim:** "All models were trained with batch size 32, identical across all context lengths. No gradient accumulation was required."

**Registry configuration (current):**
```yaml
gradient_accumulation:
  enabled: true
  effective_batch: 32
  context_micro_batch:
    "30s":  32   # accum=1
    "10m":  32   # accum=1
    "40m":  32   # accum=1
    "80m":  32   # accum=1
    "120m": 32   # accum=1
    "240m": 32   # accum=1
```

`gen_commands.py` computes `accum_steps = effective_batch / micro_batch = 1` for all contexts. The gradient accumulation code path is kept for flexibility (see below) but is a no-op at these settings.

### 12.1 If memory constraints arise on a different GPU or head

Lower the `context_micro_batch` for the affected context; the infrastructure handles accumulation automatically:

```yaml
# Example: 240m requires micro_batch=8 on a smaller GPU:
  context_micro_batch:
    "240m": 8   # gen_commands.py sets ACCUM_STEPS=4 automatically
```

`gen_commands.py` emits `BATCH_SIZE=8 ACCUM_STEPS=4`, and the training script accumulates gradients over 4 micro-batches before each optimizer step. The effective gradient is mathematically identical to batch=32 with accum=1.

**Paper wording for this case:** "All models were trained with effective batch size 32, achieved via gradient accumulation at context lengths where GPU memory required a smaller micro-batch."

---

## 13. Configurable Training K Strategy

**Status: Implemented.** The `windows_strategy` option is live in `configs/phase0_v3_config.yaml` and `scripts/train_context_sweep.py`.

### 12.1 Config options

In `configs/phase0_v3_config.yaml`, under `training:`:

```yaml
  windows_strategy: "fixed"        # "fixed" | "token_budget"
  token_budget_minutes: 80         # used only when windows_strategy = "token_budget"
```

- `"fixed"`: always K = `dataset.windows_per_subject` (default 5)
- `"token_budget"`: K = `max(1, floor(token_budget_minutes / ctx_minutes))` — keeps total signal per subject roughly constant across context lengths

### 12.2 How `train_context_sweep.py` applies this

Right before dataset creation in `train_one_context()`, the script reads `windows_strategy` from config. If `"token_budget"` and not `full_night`, it overwrites `cfg["dataset"]["windows_per_subject"]` in-memory. The printed log line shows which K is used and why.

### 12.3 How to run the token-budget ablation

Use `run_tag` in the registry so results go to separate folders and don't overwrite the K=5 baseline:

```yaml
sex_binary_lstm_kbudget:
  task: sex_binary
  head: lstm
  datasets: [apples, shhs]
  contexts: [30s, 10m, 40m, 80m, 120m, 240m]
  run_tag: "kbudget"
  tier: 3
```

Use a separate config with `windows_strategy: "token_budget"` and pass it via `--config`. Do not modify the main `phase0_v3_config.yaml` for the ablation.

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

### 12.5 Reviewer pre-answer and paper wording

**Anticipated reviewer comment:** *"With K=5 at 30s, the model only sees 5 of ~960 available windows per epoch — a tiny fraction of the available data. This may underfit the short-context models and bias your comparison toward longer contexts."*

**Rebuttal:** This conflates two distinct fairness criteria. K=5 fixed equalises *gradient updates per subject per epoch* across all context lengths — the relevant criterion for comparing models trained at different L. K=all gives the 30s model ~240× more gradient updates per epoch than the 120m model, which would confound the comparison. K=5 controls per-epoch exposure, not total data seen across training; with sufficient epochs and random sampling, models cover the window space regardless of context length. Furthermore, both K=5 and the token-budget schedule converge at L≥80m (both give K≈1), so any difference in the saturation curve is localised to short contexts and directly tested by the ablation.

**Recommended Methods wording:**
> "At each context length, K=5 windows were randomly sampled per subject per training epoch, keeping the number of gradient updates per subject constant across context lengths — the relevant criterion for an unconfounded comparison of context-length effects. As a sensitivity analysis, we repeated the experiment for [task] using a token-budget schedule (K × L = 80 min, yielding K=160 at 30s), and found no qualitative difference in the saturation curve (Supplementary Table X), confirming that the K=5 choice does not bias results toward longer contexts."

**Practical note:** Only run this ablation for one representative Tier 1 task (recommended: `sex_binary_lstm` or `bmi_binary_lstm`). A positive result (same saturation curve shape) is sufficient to close the reviewer concern for all tasks. Registry entry and config are in §12.3; use `run_tag: "kbudget"` to keep results in a separate directory.

---

## 14. Implementation Plan: Iso-Compute Plots

This section is the step-by-step workplan for implementing the 7 core iso-compute plots (from `mock-compute-optimal-tradeoffs-plots-main/`) on real experimental data.

**Design decisions (confirmed):**
- Primary metric: `mean_prob_auroc` (AUROC). Secondary: `balanced_accuracy` via `--metric` flag.
- One experiment per plot (single task × single head per figure).
- Build with current 6 context lengths; adding new contexts later just adds heatmap rows automatically.
- Budget default: 480 minutes (8h, 90th-percentile PSG recording length for APPLES/SHHS/MrOS).

**Status legend:** ⬜ TODO · 🔄 In progress · ✅ Done

---

### Step 1 — Add `--k-dense` flag to `analyze_windows.py` ✅

**File:** `scripts/analyze_windows.py`

**What:** Add a `DENSE_K_VALUES` constant and a `--k-dense` CLI flag. When set, the script uses the dense list instead of the default 6-point sparse list. The default sparse list stays unchanged (used for the quick markdown table).

```python
DENSE_K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50,
                  60, 80, 100, 120, 160, 200, 250, 320, 400, 500, "all"]
```

K values that exceed the maximum available windows for a context length are silently skipped — no wasted compute.

**Why:** With only 6 K values per context row, iso-compute lines don't land on real cells, the Pareto front is jagged, and the marginal-gain plot is unusable. ~25 K values gives smooth curves.

**Re-run command after change:**
```bash
python scripts/gen_commands.py analyze sex_binary_lstm --plot | bash
# and separately for the dense version:
/home/boshra95/sleepfm_env/bin/python scripts/analyze_windows.py \
    --task sex_binary --head lstm \
    --results-dir /scratch/boshra95/psg/unified/results/phase0_v2 \
    --k-dense --splits test
```

---

### Step 2 — Write `scripts/build_heatmap_df.py` ✅

**What:** Reads the per-split `window_analysis_{split}.csv` produced by `analyze_windows.py` and outputs a heatmap-ready DataFrame.

**Input:** `{results_dir}/inference/{task}_{head}/window_analysis_{split}.csv`
**Output:** `{results_dir}/inference/{task}_{head}/heatmap_df_{split}.csv`

**Transformations applied:**
1. Parse `context_length` strings → `context_length_min` float:
   ```python
   "30s" → 0.5,  "10m" → 10.0,  "240m" → 240.0,  "full_night" → 480.0
   ```
2. Replace `k == "all"` with the numeric max K for that context (= `n_segments / n_subjects`).
3. Rename columns to match mock plot API:
   - `mean_prob_auroc` → `auroc`
   - `mean_prob_balanced_accuracy` → `balanced_accuracy`
   - `mean_prob_macro_f1` → `f1`
   - Keep `seg_auroc`, `majority_auroc` as alternative columns.
4. Add `total_compute_min = context_length_min × k`.
5. Drop rows where any required column is NaN (contexts with no data).

**CLI:**
```bash
python scripts/build_heatmap_df.py \
    --task sex_binary --head lstm \
    --results-dir /scratch/.../results/phase0_v2 \
    --split test
```

**Output columns:** `context_length_min, context_label, k, auroc, balanced_accuracy, f1, seg_auroc, majority_auroc, total_compute_min, n_subjects, n_segments`

---

### Step 3 — Write `scripts/plot_iso_compute.py` ✅

**What:** Produces all 7 iso-compute plots from a heatmap DataFrame. Adapted directly from the mock functions in `mock-compute-optimal-tradeoffs-plots-main/PLOTS.md`, with changes for real data.

**Input:** `heatmap_df_{split}.csv` produced by Step 2.
**Output:** Saves to `{results_dir}/figures/{task}_{head}/` as both `.png` and `.pdf`.

**CLI:**
```bash
python scripts/plot_iso_compute.py \
    --task sex_binary --head lstm \
    --results-dir /scratch/.../results/phase0_v2 \
    --split test \
    --metric auroc \               # auroc | balanced_accuracy
    --budget 480                   # minutes; default 480 (8h)
```

**7 plots produced** (one file each, named by plot type and metric):

| # | File | What it shows |
|---|------|---------------|
| 1 | `heatmap_{metric}.png` | 2D grid: L (Y) × K (X), cell color = metric. Iso-compute lines overlaid. |
| 2 | `metric_vs_k_{metric}.png` | Per-context curves of metric vs K on log-x axis, with iso-compute lines crossing curves. |
| 3 | `metric_vs_total_context_{metric}.png` | Per-context curves vs total context minutes (= L × K) on log-x axis. |
| 4 | `pareto_front_{metric}.png` | Pareto-optimal (L, K) at each compute budget, colored by optimal L, annotated with K. |
| 5 | `min_cost_frontier_{metric}.png` | Minimum compute needed to reach each target metric value, per context length. |
| 6 | `marginal_gain_{metric}.png` | Per-additional-vote gain vs K (log–log), showing diminishing returns across context lengths. |
| 7 | `double_tradeoff_{metric}.png` | Grid of subplots: gain from doubling K vs switching to 2× longer context, per starting L. |

**Differences from mock:**
- Primary metric is `auroc` (not `accuracy`). `--metric balanced_accuracy` produces the same 7 plots for that metric.
- Budget default is 480 min (not 960).
- Annotation targets in min-cost plot adjusted to AUROC range (e.g., 0.55, 0.65, 0.70, 0.75, 0.80 instead of accuracy thresholds).
- Context lengths sorted and labeled from real data; no synthetic curve fitting.

**Dependencies:** `matplotlib`, `seaborn`, `numpy`, `pandas` — all in `sleepfm_env`.

---

### Step 4 — Write `scripts/plot_saturation.py` ✅

**What:** "Figure 1" for the paper — AUROC (and balanced_accuracy) vs context length at K=all, one line per head. Reads directly from `summary.csv` files (no parquet or heatmap DF needed).

**Input:** For each (task, head), reads `{results_dir}/{task}_{head}/summary.csv` and extracts `test_auroc` at the `context_length` for each row.

**CLI:**
```bash
python scripts/plot_saturation.py \
    --task sex_binary \
    --heads lstm transformer mean_pool \
    --results-dir /scratch/.../results/phase0_v2 \
    --metric auroc                # also: balanced_accuracy
```

**Output:** `{results_dir}/figures/saturation_{task}_{metric}.png`

**What it shows:** One line per head, x-axis = context length (log scale in minutes), y-axis = test metric. This answers H1 (context saturation) and the head comparison (H4). Constructable immediately from existing `summary.csv` files — no dense K sweep needed.

---

### Step 5 — Integrate into `gen_commands.py` ✅

Add two new subcommands:

```bash
# Build heatmap DataFrame (Step 2)
python scripts/gen_commands.py build-heatmap sex_binary_lstm [--split test]

# Produce all iso-compute plots (Step 3)
python scripts/gen_commands.py iso-plots sex_binary_lstm [--metric auroc] [--budget 480]

# Produce saturation curve (Step 4)
python scripts/gen_commands.py saturation sex_binary [--heads lstm transformer mean_pool]
```

Each subcommand emits a shell command using `sleepfm_env` Python, similar to how `analyze` works.
Also update `gen_commands.py list` and `status` to check for `heatmap_df_test.csv` as a pipeline completion indicator.

---

### Step 6 — Supplementary plots (lower priority) ⬜

Three additional plots from `plot_binary.py`, after Steps 1–5 are done:

**Plot A — ROC curves at iso-compute budgets:**
For each (iso-budget, L) pair, select K=floor(budget/L) windows per subject, aggregate `prob_class1` scores, compute ROC curve. Add as a function in `plot_iso_compute.py` or a separate `plot_roc_iso.py`. Reads directly from the per-window parquets.

**Plot B — Recall at fixed precision (threshold sweep):**
Instead of standard majority-vote (threshold = K/2), sweep threshold t=1,...,K ("predict positive if ≥t of K windows say positive"). Shows precision-recall tradeoff across aggregation strategies. Requires adding a threshold loop to `evaluate_at_k()` in `analyze_windows.py` or a post-processing step on the parquet.

**Plot C — Metric comparison (which metric picks which optimal L):**
Run the Pareto-front analysis (Plot 4) independently for AUROC, balanced_accuracy, and F1. Overlay on one figure to show whether the optimal context length is stable across metrics. Derived from the existing heatmap DataFrame — no new data needed.

---

### Dependency map

```
analyze_windows.py --k-dense    (Step 1)
        ↓
build_heatmap_df.py             (Step 2)
        ↓
plot_iso_compute.py             (Step 3)   ← 7 iso-compute plots

summary.csv (already exists)
        ↓
plot_saturation.py              (Step 4)   ← saturation curve

gen_commands.py                 (Step 5)   ← wires Steps 2-4 into pipeline
```

Steps 1–2 are prerequisites for Step 3. Steps 1–4 are independent of each other otherwise. Start with Step 1 (smallest change), then 2 and 4 in parallel, then 3, then 5.

### Hero experiment for initial plots

Use `sex_binary_lstm` (6 trained contexts: 30s, 10m, 40m, 80m, 120m, 240m; inference done on test split). Adding 5m context later will add a row to the heatmap automatically without any code changes.

---

## 15. Extended Analyses for Paper Depth

Beyond H1–H4 and the 7 iso-compute plots, a second set of analyses adds scientific depth. Full details and scientific motivation are in `docs/ANALYSIS_IDEAS.md`. This section summarises the design decisions that drove the pipeline changes made to support them.

### 14.1 Overfitting phase and U-shape curves (ANALYSIS_IDEAS §7)

**Scientific question:** Does training longer than early-stopping help or hurt? How do context lengths differ in their tendency to overfit?

**Why added to training code:** Early-stopped training only reveals the left and flat portion of the val-loss curve. To expose the right arm (rising val-loss, falling train-loss), training must continue past the stopping point without corrupting `best_model.pt`. The new `overfit_epochs` config option does exactly this: it runs extra epochs, flags them in `training_curves.csv` as `is_overfit_epoch: True`, and never updates the best checkpoint.

**Expected finding:** Short-context models (30s, 10m) with small effective sequence lengths may overfit faster to a relatively easier pattern. Long-context models may show a shallower overfitting slope because the task requires genuine temporal integration that the dataset supports. The difference in the overfitting slope across context lengths is itself a result about model capacity.

**Data pipeline:** `training_curves.csv` → collected into `training.csv` (all rows) → `plot_scaling_laws.py`. Filter `is_overfit_epoch == False` for all other uses.

### 14.2 Neural scaling laws (ANALYSIS_IDEAS §8)

**Scientific question:** Does test performance (AUROC) follow a power-law in training compute (FLOPs)? Is there a Chinchilla-style optimal context length for a given compute budget?

**Why added:** Scaling-law analysis requires (compute, performance) pairs at multiple training budgets, not just the best epoch. Two mechanisms provide these:

1. **Epoch-level points:** Each epoch in `training.csv` gives one (cumulative FLOPs, val_auroc) point. FLOPs per epoch are computed analytically post-hoc from `seq_len × steps_per_epoch × FLOPs_per_token(head_type, hidden_dim)`.
2. **Snapshot-level points (optional):** When `save_snapshots: true`, intermediate model states are saved every N epochs. Running inference on snapshots gives test AUROC at multiple compute budgets.

**FLOPs formulas (analytical, no profiling needed):**
- LSTM: FLOPs/step ≈ `4 × seq_len × hidden_dim × (input_dim + hidden_dim)` (4 gates, each a matmul)
- Transformer: FLOPs/step ≈ `2 × seq_len² × hidden_dim + 4 × seq_len × hidden_dim × input_dim` (attention + FF)
- MeanPool: FLOPs/step ≈ `seq_len × input_dim` (elementwise reduction only)

All inputs (`seq_len`, `hidden_dim`, `input_dim`, `steps_per_epoch`, `n_trainable_params`) are now recorded in `metrics.json` and collected into `training.csv` for this purpose.

### 14.3 Calibration analysis (ANALYSIS_IDEAS §3)

**Scientific question:** Are the predicted probabilities well-calibrated, especially at longer context lengths where AUROC improves? A highly discriminative model can still be overconfident or underconfident.

**Data needed:** Per-window `prob_class*` from `*_windows.parquet`. No new training or inference needed.

**Implementation note:** Expected Calibration Error (ECE) and reliability diagrams require binning predicted probabilities and comparing bin mean probability vs observed positive fraction. This is computed in a new `plot_calibration.py` script that reads parquets directly — no dependency on `collect_results_v2.py`.

### 14.4 Window position analysis (ANALYSIS_IDEAS §4)

**Scientific question:** Is there a "privileged" time-of-night? Do the first hours of sleep carry more diagnostic signal than late sleep for a given task?

**Data needed:** Each row of `*_windows.parquet` includes a `window_start` or `window_index` field. Group predictions by relative position in the night and compute mean predicted probability per bin.

**Why this matters for the paper:** If early-night windows are systematically more informative, the optimal K-window selection strategy is not uniform sampling but position-weighted sampling. This supports a practical recommendation.

### 14.5 Architecture: which analyses depend on which scripts

The pipeline architecture for new analyses follows the same pattern as the existing iso-compute pipeline:

| Analysis type | Data source | Depends on `collect_results_v2.py`? |
|---------------|------------|--------------------------------------|
| U-shape / overfitting curves | `training_curves.csv` (per run) or `training.csv` | Either (prefer raw for single experiment) |
| Scaling laws (FLOPs vs AUROC) | `training.csv` + analytical FLOPs | Yes — needs data from all context lengths in one place |
| Calibration, window position, PR curves | `*_windows.parquet` (per run) | No — reads parquets directly |
| Cross-task sensitivity, cohort breakdown | `analysis.csv` | Yes — multi-experiment data |
| K* distribution | `window_analysis_*.csv` (per run) | No — reads single experiment's CSV |
| Subject consistency | `*_windows.parquet` (per run) | No |

`collect_results_v2.py` is a convenience aggregator, not a required computation step. All the numbers it stores were already computed by earlier pipeline stages. Single-experiment plots bypass it; cross-experiment plots use it to avoid scanning many directories.
