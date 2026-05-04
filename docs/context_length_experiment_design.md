# Context Length Experiment Design
## Research Questions, Confounds, and Proposed Analysis

---

## 1. The Core Research Questions

This project has two distinct (but related) research questions that are easy to conflate:

**Q1 — Optimal training context:**
> "For a given task, how long a context window should a model be trained on to maximize subject-level prediction performance?"

**Q2 — Optimal inference-time aggregation:**
> "Given a model trained at a fixed context length, how many windows of inference per subject are needed before performance saturates? Is majority voting over many short windows better than a single long window?"

These are often stated together as *"what is the optimal total context?"* but they have different answers and require different experimental designs to answer. The key distinction is:

- **Q1** concerns the model's ability to extract information from a fixed window — a property of the training regime.
- **Q2** concerns how subject-level predictions are aggregated at inference — this requires no retraining and can be answered entirely from the already-inferred all-windows parquets.

---

## 2. The K Parameter: Training vs Analysis

### What K means in your current code

There are actually **two different K concepts** in the pipeline:

| Where | Variable | What it controls |
|-------|----------|-----------------|
| Training | `windows_per_subject` in config | Max windows sampled per subject per epoch (currently K=5) |
| Inference | Overridden to 99,999 in `infer_subject_windows.py` | **All windows** are dumped to parquet |
| Analysis | `--k-values` in `analyze_windows.py` | **Post-hoc subsampling** from the already-dumped all-windows parquet |

The important insight: **the K sweep in `analyze_windows.py` costs nothing extra** — it is entirely a post-processing step on the parquet. You can explore K=1, 5, 10, 20, 50, or "all" without rerunning training or inference. The heatmap you described can be built from your planned experiments without any additional GPU time.

### The training K problem your professor raised

Your original justification for K=5 was fair: if 30s context has 960 non-overlapping windows per 8-hour night and 120m context has only 4, using all windows would mean the 30s model "sees" 240× more data per subject per epoch. This is a legitimate confound for comparing context lengths in training.

Your professor's counterpoint is also legitimate: 5 randomly-sampled windows out of 960 could be the 5 noisiest ones. The law of large numbers benefits the "use all" approach.

**The real tension:** training with K=5 means the model learns a "few-shot aggregation" regime. At inference time when you sweep K, the model was never trained to aggregate information across 50 windows — you're extrapolating outside the training distribution.

### Options for training K

| Strategy | K formula | What it does |
|----------|-----------|-------------|
| **Fixed K** (current, K=5) | K is constant across context lengths | Fair comparison across context lengths. Model learns to perform well from K windows. Biases toward short context at inference. |
| **All windows** | K = all available | Maximum training signal. Short context models learn from far more windows per epoch. Unfair comparison but highest model quality. |
| **Token budget** | K × L = constant (e.g., 80 min of signal per subject per epoch) | Makes total context exposure equal across context lengths. K_30s≈160, K_10m≈8, K_40m≈2, K_80m≈1. Principled comparison. |
| **Epoch normalization** | Reduce epochs inversely with K, keeping K × epochs constant | Equates total gradient updates across context lengths. Separable from K choice. |

**Recommendation:** For the published result, run the **token budget** approach for at least one task as a validation experiment, then compare against current K=5. If results agree, K=5 is defensible. If they diverge, use token budget. This is one additional training run per task (cheap: just change config, same inference machinery).

For now, **K=5 is fine to proceed with** — it is standard in the sleep PSG literature and you can note it as a design choice with analysis supporting robustness.

---

## 3. The Heatmap Analysis

### What you described

> "A plot with x-axis of K, y-axis of context length, and color (heatmap) of a performance metric. I'll have iso-context curves showing for each total context what is the pattern of performance in different configs."

This is exactly right and it is a compelling visualization. Here is the precise definition:

**Grid axes:**
- Y-axis (rows): trained context length L ∈ {30s, 10m, 40m, 80m, 120m, ...}
- X-axis (columns): inference-time K ∈ {1, 2, 5, 10, 20, 50, all}
- Cell color: subject-level mean-prob AUROC (best aggregation method)

**What each cell represents:** a model trained at context length L, evaluated with K evenly-spaced windows per subject, using mean-prob aggregation.

**Iso-compute contours:** overlay lines where `K × L_minutes = constant` (e.g., 10 min, 40 min, 80 min, 300 min). These are the "diagonal" lines you described. Along one iso-compute line, K and L trade off such that the total signal consumed per subject at inference is constant.

**What this answers:**
- *Following a row leftward:* diminishing returns from adding more windows. Where does it plateau?
- *Following a column downward:* does longer context help at K=1? (most challenging for the model)
- *Following an iso-compute line:* for a fixed inference budget, which is better — one long window or many short ones?

### Is this already buildable from your planned experiments?

**Yes, almost entirely.** After training at 5 context lengths and running all-windows inference, the heatmap is a pure post-processing step. What you need to add is:

1. A new script `scripts/plot_context_heatmap.py` that reads the per-context parquets and constructs the 2D grid
2. Overlay of iso-compute contour lines

What you do **not** need:
- Additional training runs for the heatmap itself
- Changes to inference

### Gap: the heatmap will be sparse without more context lengths

With only 5 context lengths (30s, 10m, 40m, 80m, 120m), the Y-axis has only 5 rows. The shape of the plot will be hard to interpret because:
- There is a large gap between 30s (0.5 min) and 10m
- There is no point beyond 120m

**Recommended additions to fill the heatmap:**
- Add **2m** and **5m** to capture the short-context regime better (the 0.5–10m range is where many tasks likely saturate or start to improve)
- Add **4h (240m)** and **full_night** to anchor the long end

These 4 additional context lengths require 4 new training jobs per (task, head) but use exactly the same pipeline.

---

## 4. Proposed Experiment Structure

### 4.1 Core sweep (already planned)

Train all 16 experiments × 5 context lengths {30s, 10m, 40m, 80m, 120m} with K_train=5.

This gives the baseline table and partial heatmap.

### 4.2 Recommended additions for the heatmap

Add to `contexts` in `v2_registry.yaml` for Tier 1 tasks:

```
contexts: [30s, 2m, 5m, 10m, 40m, 80m, 120m, 240m, full_night]
```

Priority ordering if compute is limited:
1. **`full_night`** — anchors the right end of the heatmap; needed to know if more context ever helps
2. **`5m`** — fills the critical 1–10 min gap where tasks likely show the steepest improvement
3. **`240m`** — fills the 120m–full_night gap
4. **`2m`** — finishes the short-context resolution

For Tier 2 tasks (small N), skip 240m and full_night or run lstm only.

### 4.3 K training ablation (one task, low cost)

For `sex_binary_lstm` only, train two additional variants:
- `sex_binary_lstm_kall` with `windows_per_subject: 9999` (K=all)
- `sex_binary_lstm_kbudget` with `windows_per_subject` = `budget_minutes / context_length_minutes` (token budget, budget = 80m)

Use `run_tag: "kall"` and `run_tag: "kbudget"` in the registry so results land in separate folders without overwriting. Compare these heatmaps against the K=5 baseline to validate that K=5 is robust.

### 4.4 Head comparison

For Tier 1 tasks you already run 3 heads (lstm, transformer, mean_pool). This answers an implicit question: does the temporal modeling in lstm or transformer help at all, or is simple mean-pooling sufficient? If mean-pool matches lstm at long contexts, temporal structure may not matter much for these tasks.

### 4.5 Dataset generalization (deferred)

For tasks that span multiple datasets (sex_binary, bmi_binary), compare performance when training on subset vs all datasets. Useful for understanding whether the embedding is already dataset-invariant.

---

## 5. Analysis Outputs

### 5.1 Table: Context length vs test AUROC (already generated)

From `train_context_sweep.py` → `summary.csv`. One row per context, columns: train/val/test auroc, balanced_acc, macro_f1.

This answers Q1 directly: does performance increase monotonically with context? Where does it saturate?

### 5.2 Table: K sweep for each context (already generated)

From `analyze_windows.py` → `window_analysis.csv`. Shows how test AUROC changes with K for each trained context length. Aggregation methods compared: segment, mean-prob, majority-vote.

This answers Q2: how many windows do you need?

### 5.3 NEW: 2D heatmap — K × context length

Needs a new script `plot_context_heatmap.py`. Input: all `*_windows.parquet` files for one (task, head). Output: a single figure with:
- Heatmap cells colored by test AUROC (mean-prob aggregation)
- Iso-compute contour lines overlaid (e.g., 10 min, 40 min, 80 min, 320 min of total signal)
- Possibly a second panel: difference from the max row (how much do you lose by using a model trained at a shorter context?)

### 5.4 NEW: Cross-context comparison line plot

One figure per task: K-sweep curves for all context lengths on the same axes (already partially done in `plot_window_sweep` but only within one context). This directly answers: "for K=10, is 80m better than 30s?"

### 5.5 Window analysis summary: the "plateau plot"

For each context length, plot test AUROC vs K (already doable). The saturation point of each curve shows the minimum K needed for that context. This is a "cost-effectiveness" plot.

---

## 6. What Each Existing Script Already Covers

| Question | Script | Output |
|----------|--------|--------|
| Does performance improve with context length? | `train_context_sweep.py` | `metrics.json`, `summary.csv` |
| How many windows needed per subject (for one context)? | `analyze_windows.py` | `window_analysis.csv`, `window_analysis.md`, line plots |
| Is mean-prob or majority-vote better? | `analyze_windows.py` | Same output, all three methods side-by-side |
| Head comparison: lstm vs transformer vs mean_pool? | `analyze_windows.py` run separately per head | Compare CSVs manually or via new summary script |
| 2D heatmap: K × context length? | **Not yet implemented** | `plot_context_heatmap.py` needs to be written |
| Optimal training K (ablation)? | **Not yet implemented** | Needs K=all and K=budget training runs |

---

## 7. Confounds to Document Carefully

**7.1 Dataset coverage varies by context length.** At 80m and 120m, some subjects have too few non-overlapping windows even for K=5, so their N_train is lower than at 30s. Your logs already show this (`Items — train: 38126` at 10m vs `37826` at 80m). This slightly reduces comparability. Document it in your analysis.

**7.2 Training K creates a distribution shift at inference.** Models trained with K=5 have never seen subject-level averaging of 50+ windows during training. Their performance may be under-optimized for large K at inference. The K training ablation (§4.3) quantifies this.

**7.3 ISO-compute lines cross-context-length comparisons are not pure.** At iso-compute K×L=80min: a 30s model with K=160 saw 160 random 30s snippets. A 80m model with K=1 saw one contiguous 80-minute window. These are fundamentally different from the model's perspective — the first requires aggregation of many short windows, the second requires extracting information from one long sequence. The heatmap does not distinguish these two uses of "80 minutes of context."

**7.4 For seq2label tasks**, a "subject ground truth" exists (e.g., BMI category). Subject-level aggregation (mean-prob, majority-vote) is well-defined. For seq2seq tasks (sleep staging), there is no single subject-level ground truth. Your existing code already separates these — `analyze_windows.py` skips subject-level aggregation for sleep_staging.

**7.5 Window position bias.** Your `evenly-spaced` selection strategy in analysis is matched to training's sampling strategy. This is correct. If you add `random` or `first` comparisons, the results will differ and must be interpreted separately.

---

## 8. Checkpoint and Output Design for the Heatmap

**What you currently save:**

- Training: `best_model.pt`, `metrics.json`, `resume.pt` (during training), `summary.csv`
- Inference: `test_windows.parquet` (one per context)
- Analysis: `window_analysis.csv`, `window_analysis.md`

**What the heatmap needs:**

The heatmap script only needs the per-context parquets — it does the K sub-sampling itself. No changes to the checkpoint or inference pipeline are needed.

**One important addition:** save the **average number of windows per subject** for each context in `metrics.json`. This is needed to draw the iso-compute contour lines accurately (since actual avg windows/subject varies by context length and dataset). You can also get this from the parquet directly, but having it in `metrics.json` is convenient.

**Where to save heatmap output:**
```
results/phase0_v2/figures/
  {task}_{head}_heatmap_auroc.png
  {task}_{head}_heatmap_balanced_acc.png
  {task}_heatmap_head_comparison.png   # all three heads, same task, one fig
```

---

## 9. Summary: What to Implement Next

Priority order:

**Already done — no further action:**
- [x] Training at 30s, 10m, 40m, 80m, 120m with K=5
- [x] All-windows inference dumped to parquet
- [x] K-sweep analysis with mean-prob / majority-vote / segment metrics per context
- [x] Line plots per context: AUROC vs K

**Short term (implement now):**
- [ ] Add `2m`, `5m`, `240m`, `full_night` to context lists in registry (Tier 1 only)
- [ ] Write `scripts/plot_context_heatmap.py` to generate the 2D K × context heatmap with iso-compute lines

**Medium term (after initial results):**
- [ ] K training ablation: run `sex_binary_lstm_kall` and `sex_binary_lstm_kbudget` for one task to validate that K=5 is robust
- [ ] Cross-context comparison line plot (one script, reuses existing parquets)

**Long term / if results are promising:**
- [ ] Regression tasks (age, BMI) once regression head is implemented
- [ ] Dataset generalization ablations
- [ ] seq2seq (sleep staging) with adapted analysis

---

## 10. Open Questions to Resolve With Supervisor

1. **Is the heatmap the right primary figure?** Alternatively, the most impactful result might be the single "saturation curve" (AUROC vs context length at K=all) — cleaner for a paper figure, answers Q1 directly.

2. **K=5 vs K=token-budget for training:** Is it worth spending compute on the K ablation? Or is K=5 standard enough to not require justification?

3. **Which tasks to prioritize for the heatmap?** `sex_binary` is the cleanest (binary, large N, strong signal at 80m). Start there.

4. **Full-night context:** Is full_night realistic given the dataset structure? The `ContextWindowDataset` supports it (FULL_NIGHT_SENTINEL), but variable-length batches require the collate_fn and transformer head must be skipped.

5. **Inter-subject vs intra-subject performance:** All current analysis averages over all windows across all subjects. Would it be useful to compute per-subject AUROC (for subjects with ≥K windows) and show variance? Some subjects may be consistently easy or hard regardless of context.
