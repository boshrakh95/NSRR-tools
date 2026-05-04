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
