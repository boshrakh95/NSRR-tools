# Figure Interpretations — phase0_v3 (Fast Channel, 7–8 channels)

*Additional Comments*: paper relevance, anomalies, surprising findings, or things to follow up on.

**Note:** Tasks dropped from the paper (cvd_binary, sleepiness_binary, psqi_binary) are included here but marked ⚠️ DROPPED.

> **See `paper_figures.md` for the full paper figure plan.** This file contains the detailed
> per-figure interpretations; `paper_figures.md` is the master index of what goes where.

---

## Paper Figure Assignment Index (v3 → paper)

| Paper location | Named figure | Source plots from this file |
|---|---|---|
| **Fig 1** (main) | Context Saturation | `saturation/saturation_{task}_auroc_test.png` ×7 |
| **Fig 2** (main) | Inference Efficiency | `sex_binary_transformer/auroc_test/` — metric_vs_k + heatmap + min_cost_frontier |
| **Fig 3** (main) | Task Landscape | `task_comparison_6A_scatter.png` + `task_comparison_6C_lstar.png` |
| **S-Fig 1** | K-Aggregation Curves | `{task}_transformer_test_window_sweep_auroc.png` ×7 |
| **S-Fig 3** | Iso-Compute Analysis | `sex_binary + apnea_binary transformer/auroc_test/` — metric_vs_total, pareto_front, marginal_gain |
| **S-Fig 4a** | ECE vs Context | `{task}_calibration_2B_ece_vs_context.png` ×6 |
| **S-Fig 4b** | Reliability Diagrams | `{3 tasks}_lstm_calibration_2A_reliability.png` at 240m |
| **S-Fig 5** | Cross-Cohort Saturation | `apnea/sleep_eff/sex_binary lstm_cohort_saturation_7A.png` |
| **S-Fig 6a** | Prediction Variance Violins | `{3 tasks}_transformer_subject_consistency_5A_variance.png` |
| **S-Fig 6b** | Hard-Subject Distribution | `{7 tasks}_transformer_subject_consistency_5C_hard_subjects.png` (after redesign) |
| **S-Fig 7** | Window Position Profiles | `{2 tasks}_lstm_window_position_4A_profiles.png` + `4B_variance.png` |
| **S-Fig 8** | Compute Scaling | `scaling_laws/{task}_1B_compute_scaling.png` ×7 |
| **S-Fig 9** | Min Windows K* | `{task}_transformer_kstar_9A_histogram.png` ×7 |
| **S-Fig 10a/b** | PR Curves | `{task}_{head}_pr_8A_curves.png` + `{task}_pr_8B_aucpr_vs_context.png` ×6 |
| **S-Fig 11** | U-Shape Training | `{task}_transformer_1A_uShape.png` ×7 (pending BA rerun) |
| **EXCLUDED** | — | `6B_bars`, `7B_n`, `8C_vote_sweep`, `2C_ece_vs_k`, `5B_variance_vs_k`, `9B_coverage`, `1C_optimal_epoch`, `double_tradeoff`, all BA variants |

---

## Global Decisions (applied throughout)

- **Primary metric**: AUROC. Balanced Accuracy (BA) figures are written but marked [SUPPLEMENTARY].
- **Window aggregation**: Subject **mean-pool** (mp) retained over majority-vote (mv) — mp consistently reaches higher AUROC and is more stable at large K. Majority-vote removed from paper figures.
- **1A uShape plots**: All flagged [FLAG: CODE CHANGE] — rerun using BA (not loss) as the y-axis, matching the early-stopping criterion.
- **Blacklisted figures (all tasks)**: `*_calibration_2C_ece_vs_k.png`, `*_subject_consistency_5B_variance_vs_k.png`. Plus `sex_binary_lstm_kstar_9B_coverage.png` specifically.
- **5C hard-subject plots**: All flagged [REDESIGN NEEDED] — x-axis label and bar framing should be made reader-friendly.
- **1C optimal epoch**: Retained with caveat; see per-task notes.
- **AUROC vs BA for saturation**: AUROC primary, BA saturation to supplementary.

---

## Table 1 — Per-Task Figures

---

### sex_binary_lstm

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sex_binary_lstm_test_window_sweep_auroc.png | At all six context lengths, subject mean-pool (orange) dominates segment-level AUROC (blue) by 2–8 pp. Gains are steepest from K=1→5; beyond K=10 the curve flattens. At L=30s, even K=all (~350 windows) only reaches ~82% AUROC. At L=240m, K=3 already reaches ~86%, and K≥10 plateaus at ~87%. This confirms the core hypothesis: longer context windows reduce the number of windows (K) needed to match performance. Subject majority-vote (green) consistently underperforms mean-pool and is noisier at high K. | KEEP as is. Remove majority-vote (green) curve from all paper figures (instruction #9). Group all 3 heads into a 3×6 or 3-column composite per task (lstm/transformer/mean_pool as column). AUROC version is primary. |
| sex_binary_lstm_test_window_sweep_balanced_accuracy.png | BA mirrors AUROC trend but with higher variance, particularly at 240m where majority-vote shows non-monotonic dips at large K. Mean-pool is more stable. BA plateau at 240m is ~79% vs ~75% at 30s. | [SUPPLEMENTARY] Mark BA window sweeps as supplementary across all tasks. Same grouping as AUROC version but note the non-monotonic majority-vote at 240m. Remove mv. |
| sex_binary_lstm_calibration_2A_reliability.png | Three-panel reliability diagram (30s, 80m, 240m). At 30s, model is under-confident for positives (mean predicted probability < true fraction positive). At 80m calibration improves substantially — curve lies closer to the diagonal. At 240m, slight over-confidence at high probabilities introduces minor regression vs 80m. ECE is best at intermediate context (80m) for this head. | GROUP all 3 heads for sex_binary into a 3×3 grid (head × context). Remove figure title; add caption noting contexts shown are representative (shortest, mid, longest). TBME: use thin diagonal reference line, gray, 0.5 pt. |
| sex_binary_calibration_2B_ece_vs_context.png | ECE (Expected Calibration Error) vs context length for all three heads. ECE decreases from 30s to ~80m for LSTM and MeanPool, then plateaus or slightly rises at 240m. Transformer shows monotonic improvement. This non-monotonic pattern for LSTM/MeanPool indicates calibration does not simply improve with context; the model may overfit to long-context training distribution while losing calibration. | Shared across all heads already (one figure per task). Consider overlaying all tasks in one panel for across-task comparison. |
| sex_binary_lstm_calibration_2C_ece_vs_k.png | [BLACKLIST] ECE as a function of K. Generally ECE decreases with more windows aggregated, but the relationship is not monotonic for all contexts. The pattern is secondary to the AUROC/BA gain analysis and hard to interpret in isolation. | [BLACKLIST — DO NOT INCLUDE IN PAPER] Write interpretation if results make sense, but exclude. Instruction #6. |
| sex_binary_lstm_window_position_4A_profiles.png | For both positive (true label=1) and negative (true label=0) subjects, mean predicted probability is nearly flat across all 20 normalised night-position bins, regardless of context length. This confirms that the model's predictions do not depend on where in the night a window falls — it captures global physiological features, not circadian position. At 30s context, there is slightly more spread across bins; at 240m it becomes essentially constant. | GROUP lstm, transformer, mean_pool 4A into a 2×3 panel (positive/negative × head). Context length mapped to color (viridis_r). Remove per-panel titles; add shared x/y labels and single color legend. |
| sex_binary_lstm_window_position_4B_variance.png | Within-night variance (std of prob_class1 per position bin) is low and approximately constant across the night at all contexts. Longer contexts produce uniformly lower variance, consistent with more confident, position-independent predictions. No systematic rise or fall at start/end of night. | GROUP with 4A or as a companion row beneath the 4A grid. Single panel per task suffices; merge all 3 heads. |
| sex_binary_lstm_subject_consistency_5A_variance.png | Three-panel violin (30s, 120m, 240m). Correctly-classified subjects (blue) have markedly lower within-subject std(prob_class1) than incorrectly-classified (orange) at all contexts. At 240m, the correctly-classified violin is narrow and centered near 0, while the misclassified violin is broader (~0.1–0.25). This shows that prediction confidence (low variance) is a reliable proxy for correctness, and longer context makes confident subjects more confident. | GROUP all 3 heads into a 3×3 grid (head × context). Ensure 3 representative contexts selected consistently across heads. TBME color: blue (#4C72B0) for correct, orange (#DD8452) for incorrect. |
| sex_binary_lstm_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] Variance vs K shows expected decrease as K grows, but this is a mechanistic consequence of aggregation (law of large numbers), not a new finding. | [BLACKLIST — DO NOT INCLUDE IN PAPER] Instruction #6. |
| sex_binary_lstm_subject_consistency_5C_hard_subjects.png | Bar chart showing fraction of subjects correctly classified at exactly i out of 6 context lengths (i=0…6). ~57% of subjects are always correct (6/6), ~8% are never correct (0/6). The remaining ~35% are context-sensitive — they become correct only at longer contexts. This shows a bimodal distribution: "easy" subjects classified correctly regardless of context, and a small but irreducible set of hard subjects. | [REDESIGN NEEDED] Current x-axis label "i/n_ctx" is opaque. Change to plain English: "# context lengths correctly predicted (out of 6)." Consider changing bar colors to a single task color or gradient by difficulty. Alternatively, plot as cumulative fraction (e.g. "% of subjects correct at ≥ i contexts"). GROUP all 3 heads into a 3-panel row per task. Instruction #7. |
| sex_binary_lstm_cohort_saturation_7A.png | Per-cohort AUROC vs context length for LSTM head. SHHS shows the strongest improvement (30s→240m gain ~10 pp), suggesting SHHS subjects have larger sleep architecture variation capturable at longer contexts. APPLES is relatively flat (~0.78 across contexts), implying either a ceiling or different recording characteristics. MrOS shows intermediate improvement. The saturation point differs markedly across cohorts. | GROUP all 3 heads into a 3-panel row per task. |
| sex_binary_lstm_cohort_saturation_7B_n.png | N per (cohort × context) bar chart. Sample sizes are consistent across context lengths within each cohort; no major dropout at longer contexts. MrOS is the smallest cohort. SHHS is largest. | [LEGEND FIX] Check MrOS legend entry — if MrOS PSG recordings are too short for 240m context, MrOS bar will be absent and should be removed from legend. Instruction #8. All 3 heads have identical N counts so a single 7B panel suffices per task. |
| sex_binary_lstm_pr_8A_curves.png | PR curves at 6 context lengths (K=all for each). Precision-recall AUC (AUC-PR) improves monotonically from 30s to 240m. At 30s, the curve sags in the high-recall region. At 240m, the curve is substantially lifted, especially for recall > 0.5. This confirms the classification improvement is not purely threshold-dependent. | GROUP all 3 heads for sex_binary into a 3-panel row. Keep separate from 8B (which overlays heads). Context color: viridis_r. |
| sex_binary_pr_8B_aucpr_vs_context.png | AUC-PR vs context for all three heads. All heads improve monotonically. Transformer leads (highest AUC-PR at all contexts), followed by MeanPool, then LSTM. The gap between heads narrows at longer contexts (240m). No saturation evident by 240m. | Shared across heads (single figure per task). Consider overlaying saturation AUROC curve on same plot or as inset. |
| sex_binary_lstm_pr_8C_vote_sweep.png | Precision-recall curves as a function of K (majority-vote threshold sweep). As K increases, operating point shifts but the overall AUC-PR gain is modest compared to lengthening L. | Consider supplementary; primary PR information already in 8A and 8B. Could be omitted from paper to save space. |
| sex_binary_lstm_kstar_9A_histogram.png | K* histogram at 4 representative contexts (30s, 40m, 120m, 240m). K* = minimum K needed for correct subject-level classification. At 30s, K* is broadly distributed with a long tail (many subjects need >20 windows). At 240m, the distribution is heavily concentrated at K*=1–3. This confirms that longer contexts dramatically reduce the number of windows needed per patient. | GROUP all 3 heads for each task into a 3-panel row. Four representative contexts as four overlapping histograms (or stacked). |
| sex_binary_lstm_kstar_9B_coverage.png | [BLACKLIST] Coverage curve shows fraction of subjects correctly classified when aggregating up to K windows, at each context. While informative, this is a cumulative view of 9A and adds marginal insight beyond the K* histogram. | [BLACKLIST — sex_binary_lstm only. Other tasks/heads: retain if desired but consult instruction #6. DO NOT INCLUDE sex_binary_lstm version IN PAPER.] |
| sex_binary_lstm_1A_uShape.png | [FLAG: CODE CHANGE] Train/validation loss curves vs epoch for 6 context lengths. Validation loss shows a U-shape minimum consistent with early stopping. Longer contexts (240m) converge later and with lower final train loss. LSTM shows somewhat unstable training at short context (30s). **The y-axis is loss, not BA — this plot must be rerun using balanced accuracy to match the actual early-stopping criterion.** | [FLAG: CODE CHANGE] Regenerate all 1A plots using BA (not CE loss) on y-axis. After rerun, interpret anew. Keep 3-panel per head (or 6-context overlay). GROUP lstm, transformer, mean_pool into a 3-row figure per task. |
| sex_binary_1B_compute_scaling.png | FLOPs vs test AUROC scatter, all heads and context lengths. Clear power-law relationship (log-log linear fit, dashed) for each head. Transformer achieves highest AUROC (~0.89) at highest compute (~10^14 FLOPs) and has the steepest power-law slope. LSTM and MeanPool converge to ~0.84 at similar compute. The gap between Transformer and other heads is consistent, suggesting architectural advantage beyond compute scale. | Consider grouping all tasks into a single multi-panel figure (e.g., 2×4 grid for 7 kept tasks + 1 dropped). Remove figure title; caption should describe power-law fits. TBME: markers and lines follow head-consistent color scheme. |
| sex_binary_1C_optimal_epoch.png | Optimal epoch (val-loss minimum) vs context length for 3 heads. Transformer consistently requires more epochs than LSTM (35–38 vs 8–20). Pattern vs context length is non-monotonic for all heads — no clear trend that longer context requires more epochs. MeanPool is intermediate. The non-monotonicity limits interpretability. | Non-monotonic pattern makes this figure borderline. See instruction #13 — if pattern holds across all tasks, consider exclusion. Decision: **retain as supplementary** since it provides training cost information. |
| auroc_test/heatmap_auroc.png | AUROC heatmap with L (context length) on y-axis and K (windows per subject) on x-axis. AUROC is color-coded (dark=high). Iso-compute contours (L×K = constant) are overlaid as dashed curves. High AUROC (>85%) is achievable at (L=240m, K=3) or (L=120m, K=7+), but not at any K with L=30s. The iso-compute contours confirm that shifting compute budget toward longer L (rather than more K) yields better AUROC per unit compute. | GROUP: heatmap + pareto_front + metric_vs_k into a single 3-panel figure per task×head. Remove figure title. Add caption explaining iso-compute lines and what the Pareto front means. |
| auroc_test/metric_vs_k_auroc.png | AUROC vs K curves for each context length L. At L=30s, AUROC grows from ~70% at K=1 to ~82% at K=all but never reaches longer-context performance. At L=240m, AUROC already ~85% at K=1 and plateaus by K=3–5. Diminishing returns emerge quickly for long contexts. Short contexts require vastly more windows to close the gap. | Part of 3-panel group with heatmap and pareto. |
| auroc_test/metric_vs_total_auroc.png | AUROC vs total context (L×K, in minutes, log-x axis). When plotted against total context consumed, all L curves converge at the same total budget, but L=240m is most efficient (achieves a given AUROC at the lowest total context). The lines for short L (30s) are below those of long L even at equal total context — demonstrating that architectural context (long L) is intrinsically better than windowing (large K) for the same total input. | Strong evidence for Hypothesis 1 (context-length dependence). Include as a key result figure. Could be combined with pareto panel. |
| auroc_test/pareto_front_auroc.png | Pareto-optimal (L, K) pairs for fixed compute budgets. At all budgets where 240m is feasible, it lies on the Pareto front — meaning no short-L configuration matches it for the same compute cost. Short contexts are Pareto-dominated except at very small budgets where only L=30s and L=10m are affordable. | KEY FIGURE. Include alongside heatmap. |
| auroc_test/min_cost_frontier_auroc.png | Minimum total compute (L×K, minutes) needed to reach a target AUROC threshold. The frontier shows staircase jumps: targets below 82% are cheaply met by L=30s with small K; above 83%, L=10m is cheaper; above 85%, L=120m or 240m are necessary and L=30s cannot achieve the target at all. Red dashed line marks the 480-min budget. | Include in paper. Demonstrates clinical efficiency case: if you want >85% AUROC, you must invest in long-context inference. |
| auroc_test/marginal_gain_auroc.png | Marginal AUROC gain per additional window K (log-log scale). All context lengths show steep decline from K=1→2 (~2–3 pp gain) to K=5 (~0.1 pp gain) to K>10 (~0.01 pp). This confirms strongly diminishing returns from aggregation. At K>20, marginal gain is negligible (<0.01 pp) regardless of context length. | Strong supporting evidence. Could be a supplementary figure, or included in the main efficiency section alongside min_cost_frontier. |
| auroc_test/double_tradeoff_auroc.png | Two-dimensional tradeoff visualization: at a fixed compute budget (iso-compute line), shows that increasing L while decreasing K achieves higher AUROC than the reverse. The optimal point is always at the longest affordable L with K=1–3. | Combine with pareto or metric_vs_total into a summary efficiency figure. |

---

### sex_binary_transformer

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sex_binary_transformer_test_window_sweep_auroc.png | Same pattern as LSTM but Transformer achieves higher absolute AUROC at all contexts and K. At 240m, mean-pool reaches ~89%. Plateau is slightly steeper for Transformer — it extracts more information from single windows, reducing the K needed for saturation. | Group with lstm and mean_pool panels. |
| sex_binary_transformer_test_window_sweep_balanced_accuracy.png | Transformer BA at 240m reaches ~82%, higher than LSTM (~79%). Mean-pool shows non-monotonic behavior at extreme K for some contexts. | [SUPPLEMENTARY] |
| sex_binary_transformer_calibration_2A_reliability.png | Transformer shows better calibration than LSTM at all contexts. At 240m, calibration is nearly perfect for the full probability range. Slight over-confidence at 30s (high predicted probabilities overshoot true fraction). | Part of 3×3 head×context grid per task. |
| sex_binary_transformer_calibration_2C_ece_vs_k.png | [BLACKLIST] Same structural issue as LSTM 2C. | [BLACKLIST] |
| sex_binary_transformer_window_position_4A_profiles.png | Flat position profiles, same conclusion as LSTM: position-independent predictions. Slightly lower std bands than LSTM, indicating higher confidence per window. | Part of 2×3 panel (pos/neg × head). |
| sex_binary_transformer_window_position_4B_variance.png | Lower variance than LSTM across all contexts and position bins. Further confirms position-independent, confident predictions. | Group with 4A. |
| sex_binary_transformer_subject_consistency_5A_variance.png | Correctly-classified subjects show tighter variance distribution (narrower violins) than LSTM at same contexts. At 240m, most correctly-classified subjects have std<0.05. Higher head consistency for Transformer. | Part of 3×3 head×context grid. |
| sex_binary_transformer_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sex_binary_transformer_subject_consistency_5C_hard_subjects.png | Slightly fewer always-wrong subjects for Transformer than LSTM, but same bimodal distribution. Confirms hard subjects are task-intrinsic, not model-artifact. | [REDESIGN NEEDED] Same comments as LSTM 5C. |
| sex_binary_transformer_cohort_saturation_7A.png | Stronger inter-cohort separation than LSTM: SHHS rises sharply to ~91%, APPLES flatter at ~81%, MrOS intermediate. Transformer extracts more cohort-specific temporal structure. | Group with other heads. |
| sex_binary_transformer_cohort_saturation_7B_n.png | Same N counts as LSTM (N determined by recording availability, not head). Single shared 7B panel per task suffices. | [LEGEND FIX] Remove MrOS entry if absent. Single panel per task. |
| sex_binary_transformer_pr_8A_curves.png | AUC-PR highest among heads; 240m curve is near the top-right corner indicating high precision at all recall levels. | Part of 3-panel PR grid per task. |
| sex_binary_transformer_pr_8C_vote_sweep.png | Similar majority-vote sweep; Transformer threshold curves are tighter, less noisy. | Supplementary if included. |
| sex_binary_transformer_kstar_9A_histogram.png | K* distribution even more concentrated at K*=1 at 240m than LSTM — many subjects need only 1 window with Transformer at long contexts. Reinforces architectural advantage. | Part of 3-panel per task. |
| sex_binary_transformer_kstar_9B_coverage.png | Coverage curve for Transformer. Unlike the blacklisted LSTM version, this is not on the blacklist by name. Retain for reference but consider omitting from paper for brevity. | Not blacklisted by instruction #6 name. Optional inclusion. |
| sex_binary_transformer_1A_uShape.png | [FLAG: CODE CHANGE] Transformer validation loss U-shapes are shallower and later-peaking than LSTM, consistent with slower but higher-quality learning. **Rerun with BA metric.** | [FLAG: CODE CHANGE] |
| **[Fig 2a]** auroc_test/metric_vs_k_auroc.png | Faster K-saturation than LSTM — single window at 240m already reaches ~87%. Diminishing returns by K=5 at all long contexts. **This panel is the primary K-sweep panel in Fig 2.** | [→ Fig 2 panel (a)] Use Transformer head only. 4 context lengths shown with viridis_r color. Remove majority-vote. |
| **[Fig 2b]** auroc_test/heatmap_auroc.png | Heatmap shows high AUROC region at long-L + small-K. Iso-compute contours (dashed) make the compute-efficiency argument visually clear. Plateau at lower K for long L. **Primary heatmap panel for Fig 2.** | [→ Fig 2 panel (b)] Use Transformer head only. Remove figure title. Caption explains iso-compute lines. |
| **[Fig 2c]** auroc_test/min_cost_frontier_auroc.png | Minimum cost to reach target AUROC: staircase showing that L=30s cannot reach >85% at any affordable K; L=120m or 240m required. Red dashed line marks 480-min budget. **Primary frontier panel for Fig 2.** | [→ Fig 2 panel (c)] Use Transformer head only. Remove figure title. |
| **[S-Fig 3]** auroc_test/metric_vs_total_auroc.png | Transformer line lies above LSTM at all total context values — architectural advantage beyond compute volume. Long-L is more efficient than large-K at equal total context. | [→ S-Fig 3 panel, row 1(a)] sex_binary_transformer |
| **[S-Fig 3]** auroc_test/pareto_front_auroc.png | Long-L always on Pareto front. Transformer absolute AUROC at each Pareto point ~2 pp above LSTM. | [→ S-Fig 3 panel, row 1(b)] sex_binary_transformer |
| **[S-Fig 3]** auroc_test/marginal_gain_auroc.png | Steep marginal gain decay K=1→5; negligible gain beyond K=20. | [→ S-Fig 3 panel, row 1(c)] sex_binary_transformer |
| **[EXCLUDED]** auroc_test/double_tradeoff_auroc.png | Long-L + small-K is optimal — this conclusion is already captured by heatmap + pareto + metric_vs_total. Redundant. | [EXCLUDED from paper] |

---

### sex_binary_mean_pool

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sex_binary_mean_pool_test_window_sweep_auroc.png | MeanPool achieves intermediate AUROC between LSTM and Transformer. At 240m reaches ~86%. Saturation behavior similar to LSTM. MeanPool is a simpler aggregation baseline and its competitive performance at long context suggests that global temporal structure matters more than sequential modeling for sex classification. | Group with lstm and transformer. |
| sex_binary_mean_pool_test_window_sweep_balanced_accuracy.png | Similar to LSTM in BA. Some noise at extreme K. | [SUPPLEMENTARY] |
| sex_binary_mean_pool_calibration_2A_reliability.png | Calibration intermediate between LSTM and Transformer. Slight over-confidence at 240m. | Part of 3×3 grid. |
| sex_binary_mean_pool_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sex_binary_mean_pool_window_position_4A_profiles.png | Flat profiles, same conclusion as other heads. Position-independent. | Part of 2×3 grid. |
| sex_binary_mean_pool_window_position_4B_variance.png | Similar to LSTM variance levels. | Group with 4A. |
| sex_binary_mean_pool_subject_consistency_5A_variance.png | Violin patterns qualitatively same as LSTM; correctly-classified subjects have lower variance. | 3×3 grid. |
| sex_binary_mean_pool_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sex_binary_mean_pool_subject_consistency_5C_hard_subjects.png | Similar fraction of always-wrong subjects as LSTM. Hard subjects are task-level, not head-specific. | [REDESIGN NEEDED] |
| sex_binary_mean_pool_cohort_saturation_7A.png | Cohort patterns qualitatively same: SHHS improves most, APPLES flat. MeanPool AUROC levels slightly below Transformer. | Group with other heads. |
| sex_binary_mean_pool_cohort_saturation_7B_n.png | Same N as other heads. | Single shared panel. |
| sex_binary_mean_pool_pr_8A_curves.png | AUC-PR improvement similar to LSTM. Consistent with AUROC pattern. | 3-panel group. |
| sex_binary_mean_pool_pr_8C_vote_sweep.png | Similar majority-vote sweep. | Optional supplementary. |
| sex_binary_mean_pool_kstar_9A_histogram.png | K* distribution slightly broader than Transformer at long contexts but narrower than at short. Consistent with intermediate performance. | 3-panel group. |
| sex_binary_mean_pool_kstar_9B_coverage.png | Not blacklisted. Optional inclusion. | |
| sex_binary_mean_pool_1A_uShape.png | [FLAG: CODE CHANGE] MeanPool convergence is fastest among heads (fewest epochs to minimum). **Rerun with BA.** | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Heatmap similar to LSTM with slightly higher values. iso-compute structure preserved. | |
| auroc_test/metric_vs_k_auroc.png | Similar K-saturation profile to LSTM. | |
| auroc_test/metric_vs_total_auroc.png | MeanPool line sits between LSTM and Transformer on total-context plot. | |
| auroc_test/pareto_front_auroc.png | Long-L remains Pareto-optimal. | |
| auroc_test/min_cost_frontier_auroc.png | Min cost to reach target slightly lower than LSTM (higher absolute performance). | |
| auroc_test/marginal_gain_auroc.png | Same rapid decay in marginal gain per vote. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K remains optimal. | |

---

### age_class_lstm

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| age_class_lstm_test_window_sweep_auroc.png | Strong context dependence: at 30s, mean-pool AUROC ~77%; at 240m, reaches ~85%. Subject mean-pool (orange) dominates segment-level throughout. Saturation appears around K=5–8 for long contexts. Short context (30s) shows slow monotonic improvement with K but never closes the gap to long-context performance, even at K=all. | Group 3 heads per task. Remove majority-vote. |
| age_class_lstm_test_window_sweep_balanced_accuracy.png | BA trend parallels AUROC. BA at 240m reaches ~75% for LSTM. Mean-pool more stable than majority-vote. | [SUPPLEMENTARY] |
| age_class_lstm_calibration_2A_reliability.png | Three-panel. At 30s, predicted probabilities cluster in mid-range (0.3–0.7), indicating model is uncertain across all classes. At 240m, calibration improves: the diagonal is better tracked. This is a multi-class task (age bins), so calibration reflects whether predicted class probabilities match empirical frequencies. | Age_class is multi-class — note that reliability diagrams here show one-vs-rest for the positive class. Clarify in caption. |
| age_class_calibration_2B_ece_vs_context.png | ECE decreases monotonically from 30s to 240m for all heads. Transformer achieves lowest ECE at 240m. Consistent improvement, unlike the non-monotonic pattern seen in sex_binary. | Shared across heads. |
| age_class_lstm_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| age_class_lstm_window_position_4A_profiles.png | Position profiles slightly less flat than sex_binary — minor elevation in predicted probability at the middle of the night for positive subjects. This could reflect circadian sleep architecture effects on the EEG signal. Long contexts attenuate this position effect. | Note the mild position effect; interpret carefully since age correlates with sleep architecture changes across the night. |
| age_class_lstm_window_position_4B_variance.png | Variance vs position is slightly elevated at the start and end of the night (transition effects). Decreases with longer context, suggesting the model integrates over temporal variability. | Group with 4A. |
| age_class_lstm_subject_consistency_5A_variance.png | Correctly-classified subjects have significantly lower within-subject variance at all contexts. The difference between correct and incorrect distributions grows with context length — longer context yields more confident and more accurate predictions simultaneously. | 3×3 grid across heads. |
| age_class_lstm_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| age_class_lstm_subject_consistency_5C_hard_subjects.png | Fraction always-correct is lower than sex_binary (~45%), reflecting higher task difficulty. Fraction never-correct is higher (~12%). The distribution is shifted left, indicating age prediction is genuinely harder at all context lengths. | [REDESIGN NEEDED] |
| age_class_lstm_cohort_saturation_7B_n.png | N per cohort×context. Note: age_class saturation plots (7A) may be absent if cohort AUROC per age class is not computed. If 7A exists, inspect cohort differences. | Check whether 7A was generated for age_class; if not, 7B serves as a documentation figure only. |
| age_class_lstm_pr_8C_vote_sweep.png | Majority-vote PR threshold sweep. For a multi-class task this is approximated as one-vs-rest. Shows similar diminishing returns from increasing K at fixed context. | Supplementary. |
| age_class_lstm_kstar_9A_histogram.png | K* distribution at 4 contexts. Age classification requires more windows at 30s than sex (harder task). At 240m, K* distribution narrows but has a longer tail than sex_binary — confirming age is harder. | 3-panel per task group. |
| age_class_lstm_kstar_9B_coverage.png | Not blacklisted. Retain optionally. | |
| age_class_lstm_1A_uShape.png | [FLAG: CODE CHANGE] Validation loss curves for age_class. Training appears less stable than sex_binary (more epochs needed). **Rerun with BA.** | [FLAG: CODE CHANGE] |
| age_class_1B_compute_scaling.png | Power-law relationship holds for age_class as well. Transformer achieves highest AUROC at highest compute. The scaling slope is comparable to sex_binary, suggesting similar compute-efficiency across tasks. | Group across tasks in multi-panel figure. |
| age_class_1C_optimal_epoch.png | Non-monotonic pattern across context lengths (e.g., MeanPool jumps from 12 at 40m to 14 at 80m, not consistent). Transformer shows high epoch counts (29–36) across contexts. No clear monotonic relationship between context length and epochs required. **Pattern is non-monotonic and hard to justify.** | Borderline for exclusion. Retain as supplementary per global decision. |
| auroc_test/heatmap_auroc.png | High AUROC (>83%) achievable at L≥80m with K≥5. Iso-compute contours confirm longer-L advantage. | |
| auroc_test/metric_vs_k_auroc.png | K-saturation is slower for age_class than sex_binary at all contexts (harder task); plateau not fully reached by K=20 at 30s. | |
| auroc_test/metric_vs_total_auroc.png | Long-L remains more efficient than large-K at equal total context. Gap is similar to sex_binary. | |
| auroc_test/pareto_front_auroc.png | L=240m is Pareto-optimal for all but the smallest compute budgets. | |
| auroc_test/min_cost_frontier_auroc.png | Higher minimum cost to reach target AUROC than sex_binary (harder task), reflecting that L=30s cannot reach >82% at any affordable K. | |
| auroc_test/marginal_gain_auroc.png | Marginal gain decays rapidly from K=1→5. Consistent with sex_binary. | |
| auroc_test/double_tradeoff_auroc.png | Same structure as sex_binary: maximize L, minimize K. | |

---

### age_class_transformer

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| age_class_transformer_test_window_sweep_auroc.png | Transformer reaches ~87% at 240m vs LSTM ~85%. Saturation at K=3–5 for long contexts. The architectural advantage is consistent. | Group with other heads. |
| age_class_transformer_test_window_sweep_balanced_accuracy.png | BA at 240m ~78% vs ~75% for LSTM. | [SUPPLEMENTARY] |
| age_class_transformer_calibration_2A_reliability.png | Best calibration among heads at 240m. Monotonic improvement with context. | 3×3 grid. |
| age_class_transformer_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| age_class_transformer_window_position_4A_profiles.png | Mid-night elevation in probability slightly reduced vs LSTM. Transformer partially learns to ignore position-specific cues. | |
| age_class_transformer_window_position_4B_variance.png | Lower variance than LSTM at all positions. | Group with 4A. |
| age_class_transformer_subject_consistency_5A_variance.png | Tightest correct-class violin distribution among heads — Transformer is most self-consistent. | 3×3 grid. |
| age_class_transformer_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| age_class_transformer_subject_consistency_5C_hard_subjects.png | Slightly fewer never-correct subjects than LSTM (~10% vs 12%). | [REDESIGN NEEDED] |
| age_class_transformer_cohort_saturation_7B_n.png | Same N as LSTM. | |
| age_class_transformer_pr_8C_vote_sweep.png | Similar to LSTM. | Supplementary. |
| age_class_transformer_kstar_9A_histogram.png | K* even more concentrated at low values (1–3) at 240m. Consistent with higher single-window accuracy. | |
| age_class_transformer_kstar_9B_coverage.png | Not blacklisted. Optional. | |
| age_class_transformer_1A_uShape.png | [FLAG: CODE CHANGE] Requires rerun with BA. | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Higher values across board than LSTM. Same iso-compute structure. | |
| auroc_test/metric_vs_k_auroc.png | Faster K-saturation than LSTM at long contexts. | |
| auroc_test/metric_vs_total_auroc.png | Transformer line lies above LSTM at equal total context. | |
| auroc_test/pareto_front_auroc.png | Same Pareto conclusion. | |
| auroc_test/min_cost_frontier_auroc.png | Lower minimum cost to reach same target vs LSTM. | |
| auroc_test/marginal_gain_auroc.png | Same rapid decay. | |
| auroc_test/double_tradeoff_auroc.png | Same conclusion. | |

---

### age_class_mean_pool

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| age_class_mean_pool_test_window_sweep_auroc.png | MeanPool achieves AUROC intermediate between LSTM and Transformer at all contexts. At 240m ~86%. Competitive despite simple aggregation, suggesting global frequency content in long windows is informative for age. | Group with other heads. |
| age_class_mean_pool_test_window_sweep_balanced_accuracy.png | BA pattern mirrors AUROC. | [SUPPLEMENTARY] |
| age_class_mean_pool_calibration_2A_reliability.png | Calibration similar to LSTM. Intermediate between LSTM and Transformer. | 3×3 grid. |
| age_class_mean_pool_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| age_class_mean_pool_window_position_4A_profiles.png | Flat profiles, similar to LSTM. | |
| age_class_mean_pool_window_position_4B_variance.png | Low variance, position-independent. | |
| age_class_mean_pool_subject_consistency_5A_variance.png | Similar distribution to LSTM. | |
| age_class_mean_pool_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| age_class_mean_pool_subject_consistency_5C_hard_subjects.png | Similar fraction of hard subjects as LSTM. | [REDESIGN NEEDED] |
| age_class_mean_pool_cohort_saturation_7B_n.png | Same N. | |
| age_class_mean_pool_pr_8C_vote_sweep.png | Consistent with other heads. | Supplementary. |
| age_class_mean_pool_kstar_9A_histogram.png | K* distribution intermediate. | |
| age_class_mean_pool_kstar_9B_coverage.png | Not blacklisted. Optional. | |
| age_class_mean_pool_1A_uShape.png | [FLAG: CODE CHANGE] Fastest convergence among heads. Rerun with BA. | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Heatmap values intermediate between LSTM and Transformer. | |
| auroc_test/metric_vs_k_auroc.png | K-saturation intermediate. | |
| auroc_test/metric_vs_total_auroc.png | MeanPool line sits between LSTM and Transformer. | |
| auroc_test/pareto_front_auroc.png | Same Pareto structure. | |
| auroc_test/min_cost_frontier_auroc.png | Similar efficiency to LSTM. | |
| auroc_test/marginal_gain_auroc.png | Same rapid decay. | |
| auroc_test/double_tradeoff_auroc.png | Same conclusion. | |

---

### apnea_binary_lstm

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| apnea_binary_lstm_test_window_sweep_auroc.png | Apnea shows the strongest absolute performance of all tasks. At 30s, mean-pool AUROC ~84%; at 240m ~92%. Subject mean-pool dominates throughout. Even K=1 at 240m reaches ~90%, indicating that a single long window is already highly informative for AHI-based classification. Saturation at K=3–5 for long contexts. Short-context (30s) performance plateaus at ~88% with K=all — strong but well below long-context single-window. | Group 3 heads. Remove majority-vote. |
| apnea_binary_lstm_test_window_sweep_balanced_accuracy.png | BA at 240m reaches ~83%, highest among tasks. Consistent improvement with both L and K. | [SUPPLEMENTARY] |
| apnea_binary_lstm_calibration_2A_reliability.png | Best calibration among tasks: at 240m, calibration curve is nearly diagonal across the full probability range. Even at 30s, calibration is reasonable. Apnea may be more reliably encoded in PSG signals, making the model both accurate and well-calibrated. | 3×3 grid per task. |
| apnea_binary_calibration_2B_ece_vs_context.png | Monotonic ECE decrease across all heads. ECE at 240m is among the lowest of all tasks. | Shared across heads. |
| apnea_binary_lstm_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| apnea_binary_lstm_window_position_4A_profiles.png | Positive subjects (high AHI) show a slight elevation of predicted probability in the second half of the night, consistent with the known pattern that respiratory events are often more frequent in REM sleep (late night). Negative subjects are flat. | Note the mild second-half elevation for positives — this is biologically interpretable (REM-related OSA) and should be mentioned in the paper. |
| apnea_binary_lstm_window_position_4B_variance.png | Variance is slightly elevated in the second half of the night for long contexts, consistent with the elevated mean probability there. At short contexts, variance is more uniform. | Group with 4A. |
| apnea_binary_lstm_subject_consistency_5A_variance.png | Very clear separation between correct and incorrect violins at all contexts. Correctly-classified subjects have near-zero within-subject variance at 240m. Apnea is a task where the model becomes extremely confident and self-consistent when context is sufficient. | 3×3 grid. Strong result worth highlighting. |
| apnea_binary_lstm_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| apnea_binary_lstm_subject_consistency_5C_hard_subjects.png | Fraction always-correct is highest of all tasks (~70%); fraction never-correct is lowest (~4%). Apnea has the fewest irreducible hard cases. | [REDESIGN NEEDED] |
| apnea_binary_lstm_cohort_saturation_7A.png | Strong improvement across all cohorts. SHHS and APPLES both rise substantially with context. MrOS may be absent at longest contexts due to recording length constraints. | Check MrOS presence at 240m. |
| apnea_binary_lstm_cohort_saturation_7B_n.png | N per cohort×context — check for dropout at 240m. | [LEGEND FIX] Remove missing cohorts from legend. |
| apnea_binary_lstm_pr_8A_curves.png | AUC-PR among the highest of all tasks. At 240m, the PR curve is nearly at the upper-right corner, confirming high precision at all recall levels. Apnea is a task where PSG-based prediction is clinically reliable. | 3-panel per task. |
| apnea_binary_pr_8B_aucpr_vs_context.png | Monotonic AUC-PR improvement. All heads converge at high AUC-PR by 240m. | Shared across heads. |
| apnea_binary_lstm_pr_8C_vote_sweep.png | PR threshold sweep shows tight curves at long contexts — model is robust to threshold choice. | Optional supplementary. |
| apnea_binary_lstm_kstar_9A_histogram.png | K* is the most concentrated distribution of all tasks at 240m: most subjects need only K*=1. Even at 30s, K* is lower than sex_binary or age_class, reflecting the strong single-window apnea signal. | Highlight in paper: apnea is detectable from a single long window. |
| apnea_binary_lstm_kstar_9B_coverage.png | Not blacklisted for apnea_binary. Retain for reference. | |
| apnea_binary_lstm_1A_uShape.png | [FLAG: CODE CHANGE] Apnea training converges quickly and stably. Rerun with BA. | [FLAG: CODE CHANGE] |
| apnea_binary_1B_compute_scaling.png | Steep power-law slope for apnea: Transformer at max compute reaches ~0.93 AUROC. The data suggest that with sufficient compute, the model is approaching clinical-grade apnea detection from unscored PSG. | Strong result for paper. |
| apnea_binary_1C_optimal_epoch.png | Optimal epochs are lower than sex_binary (faster convergence for apnea). Pattern still non-monotonic vs context, similar to sex_binary. | Supplementary; same borderline assessment as sex_binary 1C. |
| auroc_test/heatmap_auroc.png | High AUROC (>88%) achievable even at moderate (L=40m, K=5+). The iso-compute advantage of long-L is present but less pronounced than sex_binary — even short contexts perform reasonably. | |
| auroc_test/metric_vs_k_auroc.png | K-saturation is fastest of all tasks: plateau by K=3 for L≥40m. | |
| auroc_test/metric_vs_total_auroc.png | All L curves converge at high AUROC by ~200 min total context. | |
| auroc_test/pareto_front_auroc.png | Long-L remains Pareto-dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Cheapest to reach >90% AUROC is L=240m×K=3 (~720 min). L=120m×K=5 is nearly as good. | |
| auroc_test/marginal_gain_auroc.png | Sharpest marginal gain decay: gain/window is nearly zero by K=5 at 240m. | |
| auroc_test/double_tradeoff_auroc.png | Long-L × small-K optimal as expected. | |

---

### apnea_binary_transformer

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| apnea_binary_transformer_test_window_sweep_auroc.png | Transformer reaches ~93% at 240m with K=3. This is the best single-head performance across all tasks. Even K=1 reaches ~91%. Near-clinical-grade performance from a single long window. | Group 3 heads. |
| apnea_binary_transformer_test_window_sweep_balanced_accuracy.png | BA matches AUROC improvement. Reaches ~85% at 240m. | [SUPPLEMENTARY] |
| apnea_binary_transformer_calibration_2A_reliability.png | Nearly perfect calibration at 240m. Best calibration of any head×task combination. | 3×3 grid. |
| apnea_binary_transformer_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| apnea_binary_transformer_window_position_4A_profiles.png | The second-half elevation for positives is more pronounced in Transformer — the model attends to the REM-related signal more effectively. | Note for paper. |
| apnea_binary_transformer_window_position_4B_variance.png | Low variance, slightly elevated second-half for positives. | |
| apnea_binary_transformer_subject_consistency_5A_variance.png | Narrowest correct-class violin of any head×task — most confident and consistent predictions. | Highlight. |
| apnea_binary_transformer_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| apnea_binary_transformer_subject_consistency_5C_hard_subjects.png | Fewest hard subjects of all head×task combinations (~3% never-correct). | [REDESIGN NEEDED] |
| apnea_binary_transformer_cohort_saturation_7A.png | Strongest cohort saturation improvement. All cohorts show large gains. | |
| apnea_binary_transformer_cohort_saturation_7B_n.png | Same N. | [LEGEND FIX] |
| apnea_binary_transformer_pr_8A_curves.png | Best PR curve of all head×task. 240m curve nearly at maximum. | |
| apnea_binary_transformer_pr_8C_vote_sweep.png | Very tight curves — robustness. | Optional supplementary. |
| apnea_binary_transformer_kstar_9A_histogram.png | Almost all subjects have K*=1 at 240m. Exceptional task-head combination for clinical deployment. | |
| apnea_binary_transformer_kstar_9B_coverage.png | Not blacklisted. | |
| apnea_binary_transformer_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Highest AUROC values in any heatmap. L≥40m, K≥3 shows >90%. | |
| auroc_test/metric_vs_k_auroc.png | Fastest K-saturation. K=2 nearly sufficient at 240m. | |
| auroc_test/metric_vs_total_auroc.png | Transformer apnea line is highest in any multi-task comparison. | |
| auroc_test/pareto_front_auroc.png | Same Pareto structure. | |
| auroc_test/min_cost_frontier_auroc.png | Cheapest path to 90%: L=240m×K=2 (~480 min). | |
| auroc_test/marginal_gain_auroc.png | Near-zero gain after K=2 at 240m. | |
| auroc_test/double_tradeoff_auroc.png | Confirms same long-L + small-K optimality. | |

---

### apnea_binary_mean_pool

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| apnea_binary_mean_pool_test_window_sweep_auroc.png | MeanPool reaches ~91% at 240m — very competitive. At 30s, only ~84%, slightly below LSTM/Transformer. Suggests that for apnea, frequency-based global statistics (MeanPool) are highly informative. | Group 3 heads. |
| apnea_binary_mean_pool_test_window_sweep_balanced_accuracy.png | BA similar pattern. | [SUPPLEMENTARY] |
| apnea_binary_mean_pool_calibration_2A_reliability.png | Good calibration, slightly behind Transformer. | 3×3 grid. |
| apnea_binary_mean_pool_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| apnea_binary_mean_pool_window_position_4A_profiles.png | Position profiles flat, second-half elevation less pronounced than Transformer. | |
| apnea_binary_mean_pool_window_position_4B_variance.png | Low variance. | |
| apnea_binary_mean_pool_subject_consistency_5A_variance.png | Similar to LSTM. Good correct/incorrect separation. | |
| apnea_binary_mean_pool_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| apnea_binary_mean_pool_subject_consistency_5C_hard_subjects.png | Similar hard-subject fraction to LSTM. | [REDESIGN NEEDED] |
| apnea_binary_mean_pool_cohort_saturation_7A.png | Cohort patterns consistent with other heads. | |
| apnea_binary_mean_pool_cohort_saturation_7B_n.png | Same N. | [LEGEND FIX] |
| apnea_binary_mean_pool_pr_8A_curves.png | Strong PR curves. | |
| apnea_binary_mean_pool_pr_8C_vote_sweep.png | Similar to other heads. | Supplementary. |
| apnea_binary_mean_pool_kstar_9A_histogram.png | K* concentrated at low values at 240m, similar to LSTM. | |
| apnea_binary_mean_pool_kstar_9B_coverage.png | Not blacklisted. | |
| apnea_binary_mean_pool_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Values similar to LSTM. | |
| auroc_test/metric_vs_k_auroc.png | Intermediate K-saturation. | |
| auroc_test/metric_vs_total_auroc.png | Line between LSTM and Transformer. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Similar to LSTM efficiency. | |
| auroc_test/marginal_gain_auroc.png | Same decay. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal. | |

---

### bmi_binary_lstm

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| bmi_binary_lstm_test_window_sweep_auroc.png | Weaker context dependence than sex/age/apnea. At 30s, mean-pool AUROC ~66%; at 240m ~72%. Improvement with K is moderate. Short-context performance already captures much of the signal, suggesting BMI has a weaker and/or noisier PSG signature. Mean-pool (orange) still leads segment-level (blue) but the gap is smaller. | Group 3 heads. |
| bmi_binary_lstm_test_window_sweep_balanced_accuracy.png | BA mirrors AUROC pattern. Modest improvement with both L and K. | [SUPPLEMENTARY] |
| bmi_binary_lstm_calibration_2A_reliability.png | Reliability at 30s shows poor calibration (high uncertainty, mid-range probabilities for all subjects). At 240m, marginal improvement but still wide deviation from diagonal. **Calibration remains poor at all contexts, suggesting the model is systematically uncertain about BMI prediction.** | Note poor calibration across all contexts. |
| bmi_binary_calibration_2B_ece_vs_context.png | ECE improvement from 30s to 240m is marginal (~0.01). **All heads maintain relatively high ECE compared to other tasks.** | Shared across heads. |
| bmi_binary_lstm_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| bmi_binary_lstm_window_position_4A_profiles.png | Position profiles are flat, similar to sex_binary. BMI has no known circadian structure in PSG, and the model correctly learns position-independent features. | |
| bmi_binary_lstm_window_position_4B_variance.png | Variance remains relatively constant across contexts and positions — consistent with the weaker overall signal. | |
| bmi_binary_lstm_subject_consistency_5A_variance.png | Less clear separation between correct and incorrect violins than other tasks. At 240m, the separation exists but is narrower. **The model is less confident and less self-consistent for BMI, reflecting the weaker PSG signature.** | Note weaker separation. |
| bmi_binary_lstm_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| bmi_binary_lstm_subject_consistency_5C_hard_subjects.png | Fraction always-correct is lower (~38%) and fraction never-correct is higher (~18%) than high-performing tasks. BMI classification has the most irreducible hard cases of the main tasks. | [REDESIGN NEEDED] |
| bmi_binary_lstm_cohort_saturation_7A.png | Cohort-level variation is smaller than other tasks. All cohorts show modest improvement. No cohort shows a strong context-driven gain, consistent with weak task signal. | |
| bmi_binary_lstm_cohort_saturation_7B_n.png | N per cohort×context similar to other tasks. | [LEGEND FIX] |
| bmi_binary_lstm_pr_8A_curves.png | AUC-PR improvement modest. 240m curve is still below 0.75, reflecting poor precision-recall tradeoff for BMI. | |
| bmi_binary_pr_8B_aucpr_vs_context.png | Modest AUC-PR improvement. | Shared. |
| bmi_binary_lstm_pr_8C_vote_sweep.png | Similar to other tasks but lower absolute performance. | Supplementary. |
| bmi_binary_lstm_kstar_9A_histogram.png | K* distribution is broad even at 240m — more subjects need many windows to be correctly classified, consistent with the weaker signal. | |
| bmi_binary_lstm_kstar_9B_coverage.png | Not blacklisted. Optional inclusion. | |
| bmi_binary_lstm_1A_uShape.png | [FLAG: CODE CHANGE] Training may be less stable for BMI due to weak signal. Rerun with BA. | [FLAG: CODE CHANGE] |
| bmi_binary_1B_compute_scaling.png | Power-law slope is shallower than sex/age/apnea — returns from more compute are limited for BMI. Transformer still leads. **Overall AUROC ceiling appears around ~0.74, suggesting BMI may be intrinsically hard to predict from PSG.** | |
| bmi_binary_1C_optimal_epoch.png | Optimal epoch varies irregularly — consistent with weaker task signal causing noisy training dynamics. | Supplementary. Non-monotonic, same as other tasks. |
| auroc_test/heatmap_auroc.png | Low peak AUROC (~72%) even at maximum L and K. Iso-compute advantage of long-L exists but the absolute gains are small. | |
| auroc_test/metric_vs_k_auroc.png | K-saturation at moderate K but absolute plateau is low. | |
| auroc_test/metric_vs_total_auroc.png | All L curves converge but at lower AUROC than other tasks. | |
| auroc_test/pareto_front_auroc.png | Long-L still Pareto-dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Cheap to reach 70% (even 30s+small K); above 71% requires long-L. The ceiling is low. | |
| auroc_test/marginal_gain_auroc.png | Same rapid decay in marginal gain. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal but with low absolute benefit. | |

---

### bmi_binary_transformer

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| bmi_binary_transformer_test_window_sweep_auroc.png | Transformer reaches ~74% at 240m — best among heads but still modest. Mean-pool slightly more noisy. Context gain still present (~8 pp total). | Group 3 heads. |
| bmi_binary_transformer_test_window_sweep_balanced_accuracy.png | BA pattern mirrors AUROC. | [SUPPLEMENTARY] |
| bmi_binary_transformer_calibration_2A_reliability.png | Better calibration than LSTM but still suboptimal for BMI. At 240m, marginal improvement vs 30s. | 3×3 grid. |
| bmi_binary_transformer_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| bmi_binary_transformer_window_position_4A_profiles.png | Flat, position-independent. | |
| bmi_binary_transformer_window_position_4B_variance.png | Low variance, consistent with position-independence. | |
| bmi_binary_transformer_subject_consistency_5A_variance.png | Better separation than LSTM but still weaker than high-performing tasks. | |
| bmi_binary_transformer_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| bmi_binary_transformer_subject_consistency_5C_hard_subjects.png | Similar hard-subject fraction as LSTM. Task-level difficulty dominates head. | [REDESIGN NEEDED] |
| bmi_binary_transformer_cohort_saturation_7A.png | Small cohort-level differences; all cohorts similarly weak. | |
| bmi_binary_transformer_cohort_saturation_7B_n.png | Same N. | [LEGEND FIX] |
| bmi_binary_transformer_pr_8A_curves.png | Best PR curves for BMI but still modest. | |
| bmi_binary_transformer_pr_8C_vote_sweep.png | Similar structure. | Supplementary. |
| bmi_binary_transformer_kstar_9A_histogram.png | Slightly lower K* than LSTM at 240m. Transformer needs fewer windows but task ceiling remains. | |
| bmi_binary_transformer_kstar_9B_coverage.png | Not blacklisted. | |
| bmi_binary_transformer_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Slightly higher values than LSTM but same ceiling. | |
| auroc_test/metric_vs_k_auroc.png | Slightly faster K-saturation. | |
| auroc_test/metric_vs_total_auroc.png | Transformer line slightly above LSTM. | |
| auroc_test/pareto_front_auroc.png | Long-L Pareto-dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Similar structure to LSTM. | |
| auroc_test/marginal_gain_auroc.png | Same decay. | |
| auroc_test/double_tradeoff_auroc.png | Same conclusion. | |

---

### bmi_binary_mean_pool

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| bmi_binary_mean_pool_test_window_sweep_auroc.png | MeanPool performance close to LSTM for BMI — both are weaker than Transformer. At 240m ~71%. | Group 3 heads. |
| bmi_binary_mean_pool_test_window_sweep_balanced_accuracy.png | BA similar pattern. | [SUPPLEMENTARY] |
| bmi_binary_mean_pool_calibration_2A_reliability.png | Similar calibration to LSTM. | 3×3 grid. |
| bmi_binary_mean_pool_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| bmi_binary_mean_pool_window_position_4A_profiles.png | Flat. | |
| bmi_binary_mean_pool_window_position_4B_variance.png | Low variance. | |
| bmi_binary_mean_pool_subject_consistency_5A_variance.png | Weaker separation than Transformer. | |
| bmi_binary_mean_pool_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| bmi_binary_mean_pool_subject_consistency_5C_hard_subjects.png | Similar hard-subject profile to LSTM. | [REDESIGN NEEDED] |
| bmi_binary_mean_pool_cohort_saturation_7A.png | Cohort patterns same as other heads. | |
| bmi_binary_mean_pool_cohort_saturation_7B_n.png | Same N. | [LEGEND FIX] |
| bmi_binary_mean_pool_pr_8A_curves.png | Modest PR improvement. | |
| bmi_binary_mean_pool_pr_8C_vote_sweep.png | Similar. | Supplementary. |
| bmi_binary_mean_pool_kstar_9A_histogram.png | Broad K* distribution at all contexts. | |
| bmi_binary_mean_pool_kstar_9B_coverage.png | Not blacklisted. | |
| bmi_binary_mean_pool_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Low ceiling, same iso-compute structure. | |
| auroc_test/metric_vs_k_auroc.png | Slow K-saturation for weak task. | |
| auroc_test/metric_vs_total_auroc.png | Lines converge at low AUROC. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Low target AUROC achievable cheaply; above ~71% requires long-L. | |
| auroc_test/marginal_gain_auroc.png | Same decay. | |
| auroc_test/double_tradeoff_auroc.png | Same conclusion. | |

---

### sleep_efficiency_binary_lstm

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleep_efficiency_binary_lstm_test_window_sweep_auroc.png | Strongest context dependence of all tasks. At 30s, mean-pool AUROC ~69%; at 240m reaches ~82%. The absolute gain (+13 pp) is the largest across tasks. AUROC does NOT appear to saturate by 240m — the curve is still rising, suggesting L* > 240m. K-saturation is similar to other tasks (K=5–8 at long contexts). | Sleep efficiency has not reached its L* — further gains expected beyond 240m. Highlight in paper as the task with highest context sensitivity. |
| sleep_efficiency_binary_lstm_test_window_sweep_balanced_accuracy.png | BA mirrors AUROC — also still rising at 240m. BA at 240m ~76%. | [SUPPLEMENTARY] |
| sleep_efficiency_binary_lstm_calibration_2A_reliability.png | Calibration improves substantially from 30s to 240m. At 30s, model is very uncertain (probabilities cluster near 0.5). At 240m, calibration is good but not as clean as apnea. | 3×3 grid. |
| sleep_efficiency_binary_calibration_2B_ece_vs_context.png | Monotonic ECE decrease. Largest ECE drop of all tasks (reflecting the large improvement in prediction quality with context). | Shared across heads. |
| sleep_efficiency_binary_lstm_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sleep_efficiency_binary_lstm_window_position_4A_profiles.png | Notable position dependence: positive subjects (low sleep efficiency) show elevated probability in the early part of the night (first third), possibly reflecting frequent early-night awakenings and reduced NREM depth. This effect is stronger at short contexts (30s) and weakens at long contexts as the model integrates the entire night. **This position dependence is biologically interpretable and should be highlighted.** | KEY FINDING: position-dependent predictions for sleep efficiency. Include 4A for this task prominently. |
| sleep_efficiency_binary_lstm_window_position_4B_variance.png | Higher variance in first third of night vs middle and end, especially at short contexts. Consistent with the position-dependent mean profile. | Group with 4A. |
| sleep_efficiency_binary_lstm_subject_consistency_5A_variance.png | Good correct/incorrect separation. Sleep efficiency classification becomes increasingly confident at longer contexts. | 3×3 grid. |
| sleep_efficiency_binary_lstm_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sleep_efficiency_binary_lstm_subject_consistency_5C_hard_subjects.png | Moderate fraction always-correct (~48%), higher than BMI but lower than apnea. The task is intermediate in difficulty and context-sensitivity. | [REDESIGN NEEDED] |
| sleep_efficiency_binary_lstm_cohort_saturation_7A.png | Strong cohort-level improvement across all cohorts, with no saturation by 240m in any cohort. SHHS may show the largest improvement due to the broadest demographic distribution and sleep efficiency variance. | Include this figure — it confirms the task's high context sensitivity across cohorts. |
| sleep_efficiency_binary_lstm_cohort_saturation_7B_n.png | N per cohort×context; check for 240m dropout. | [LEGEND FIX] |
| sleep_efficiency_binary_lstm_pr_8A_curves.png | AUC-PR improves substantially with context, no saturation. | 3-panel. |
| sleep_efficiency_binary_pr_8B_aucpr_vs_context.png | All heads monotonically improving, no saturation. | Shared. |
| sleep_efficiency_binary_lstm_pr_8C_vote_sweep.png | Similar to other tasks. | Supplementary. |
| sleep_efficiency_binary_lstm_kstar_9A_histogram.png | K* at 240m is broader than apnea — not as concentrated at K*=1. More windows needed per subject, consistent with the not-yet-saturated context dependence. | 3-panel. |
| sleep_efficiency_binary_lstm_kstar_9B_coverage.png | Not blacklisted. Retain optionally. | |
| sleep_efficiency_binary_lstm_1A_uShape.png | [FLAG: CODE CHANGE] Training appears to need more epochs at long contexts for sleep_efficiency, consistent with the complex temporal structure. Rerun with BA. | [FLAG: CODE CHANGE] |
| sleep_efficiency_binary_1B_compute_scaling.png | Power-law scaling similar to other tasks. The Transformer at max compute has not reached the task ceiling (unlike apnea) — further scaling would likely continue to improve performance. | Note as evidence that sleep efficiency would benefit from even longer contexts or more compute. |
| sleep_efficiency_binary_1C_optimal_epoch.png | Optimal epoch may be higher at longer contexts for sleep_efficiency given the larger signal complexity. Inspect whether a monotonic trend appears — this would make sleep_efficiency one of the few tasks with a justifiable 1C relationship. | If monotonic: useful. If not: supplementary. |
| auroc_test/heatmap_auroc.png | AUROC does not reach a plateau within the L×K grid. The highest values are at (240m, large K). Iso-compute contours confirm long-L advantage is essential. | |
| auroc_test/metric_vs_k_auroc.png | K-saturation slower than other tasks — more windows help for sleep efficiency. | |
| auroc_test/metric_vs_total_auroc.png | Curves are still spreading at 240m — no convergence yet. Strong evidence that L* > 240m for sleep efficiency. | |
| auroc_test/pareto_front_auroc.png | L=240m dominant everywhere. | |
| auroc_test/min_cost_frontier_auroc.png | To reach 80% AUROC, requires L=240m and moderate K. Cheapest path is clear: use maximum feasible L. | |
| auroc_test/marginal_gain_auroc.png | Marginal gain per window decays but less steeply than other tasks — consistent with slower K-saturation. | |
| auroc_test/double_tradeoff_auroc.png | Same long-L + small-K optimality, but the benefit from increasing L is largest for this task. | |

---

### sleep_efficiency_binary_transformer

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleep_efficiency_binary_transformer_test_window_sweep_auroc.png | Transformer reaches ~85% at 240m, still rising. Best single-head performance for this task. K=3–5 sufficient at 240m. | Group 3 heads. |
| sleep_efficiency_binary_transformer_test_window_sweep_balanced_accuracy.png | BA at 240m ~79%. | [SUPPLEMENTARY] |
| sleep_efficiency_binary_transformer_calibration_2A_reliability.png | Best calibration among heads. Monotonic improvement. | 3×3 grid. |
| sleep_efficiency_binary_transformer_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sleep_efficiency_binary_transformer_window_position_4A_profiles.png | Position dependence pattern same as LSTM but slightly attenuated — Transformer may integrate temporal context more smoothly. | Compare with LSTM 4A in paper. |
| sleep_efficiency_binary_transformer_window_position_4B_variance.png | Lower variance than LSTM. Consistent with higher confidence. | |
| sleep_efficiency_binary_transformer_subject_consistency_5A_variance.png | Tighter correct-class violin. | |
| sleep_efficiency_binary_transformer_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sleep_efficiency_binary_transformer_subject_consistency_5C_hard_subjects.png | Slightly fewer never-correct subjects. | [REDESIGN NEEDED] |
| sleep_efficiency_binary_transformer_cohort_saturation_7A.png | Strong cohort improvement, same as LSTM. | |
| sleep_efficiency_binary_transformer_cohort_saturation_7B_n.png | Same N. | [LEGEND FIX] |
| sleep_efficiency_binary_transformer_pr_8A_curves.png | Best PR curves for this task. | |
| sleep_efficiency_binary_transformer_pr_8C_vote_sweep.png | Consistent. | Supplementary. |
| sleep_efficiency_binary_transformer_kstar_9A_histogram.png | Slightly more concentrated K* at 240m vs LSTM. | |
| sleep_efficiency_binary_transformer_kstar_9B_coverage.png | Not blacklisted. | |
| sleep_efficiency_binary_transformer_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Highest values for this task. Still not plateaued. | |
| auroc_test/metric_vs_k_auroc.png | Slightly faster K-saturation than LSTM. | |
| auroc_test/metric_vs_total_auroc.png | Transformer line highest. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Lower min cost to reach 83% vs LSTM. | |
| auroc_test/marginal_gain_auroc.png | Same decay. | |
| auroc_test/double_tradeoff_auroc.png | Same conclusion. | |

---

### sleep_efficiency_binary_mean_pool

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleep_efficiency_binary_mean_pool_test_window_sweep_auroc.png | MeanPool reaches ~83% at 240m. Competitive. Not yet saturated. | Group 3 heads. |
| sleep_efficiency_binary_mean_pool_test_window_sweep_balanced_accuracy.png | BA similar. | [SUPPLEMENTARY] |
| sleep_efficiency_binary_mean_pool_calibration_2A_reliability.png | Intermediate calibration. | 3×3 grid. |
| sleep_efficiency_binary_mean_pool_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sleep_efficiency_binary_mean_pool_window_position_4A_profiles.png | Similar position dependence to LSTM. | |
| sleep_efficiency_binary_mean_pool_window_position_4B_variance.png | Similar variance. | |
| sleep_efficiency_binary_mean_pool_subject_consistency_5A_variance.png | Good separation, intermediate confidence. | |
| sleep_efficiency_binary_mean_pool_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sleep_efficiency_binary_mean_pool_subject_consistency_5C_hard_subjects.png | Similar hard-subject distribution. | [REDESIGN NEEDED] |
| sleep_efficiency_binary_mean_pool_cohort_saturation_7A.png | Cohort patterns consistent. | |
| sleep_efficiency_binary_mean_pool_cohort_saturation_7B_n.png | Same N. | [LEGEND FIX] |
| sleep_efficiency_binary_mean_pool_pr_8A_curves.png | Strong PR improvement. | |
| sleep_efficiency_binary_mean_pool_pr_8C_vote_sweep.png | Consistent. | Supplementary. |
| sleep_efficiency_binary_mean_pool_kstar_9A_histogram.png | Broader K* than Transformer. | |
| sleep_efficiency_binary_mean_pool_kstar_9B_coverage.png | Not blacklisted. | |
| sleep_efficiency_binary_mean_pool_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Values between LSTM and Transformer. | |
| auroc_test/metric_vs_k_auroc.png | Intermediate K-saturation. | |
| auroc_test/metric_vs_total_auroc.png | Between LSTM and Transformer. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Comparable to LSTM. | |
| auroc_test/marginal_gain_auroc.png | Same decay. | |
| auroc_test/double_tradeoff_auroc.png | Same conclusion. | |

---

### depression_extreme_binary_lstm

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| depression_extreme_binary_lstm_test_window_sweep_auroc.png | **AUROC is non-monotonic and low across all contexts (~0.63–0.69). Mean-pool (orange) does not consistently outperform segment-level. No clear improvement with longer context. The pattern is erratic and does not support the context-dependence hypothesis for depression. This may reflect that depression biomarkers are weak, highly variable, or not well-captured by the available PSG channels and the APPLES-only cohort (small N).** | **[UNJUSTIFIABLE RESULT — BOLD]** Results are inconsistent across contexts. Consider reducing prominence in paper or moving to discussion as a negative finding. |
| depression_extreme_binary_lstm_test_window_sweep_balanced_accuracy.png | **BA similarly erratic. No improvement with context.** | [SUPPLEMENTARY] **[UNJUSTIFIABLE]** |
| depression_extreme_binary_lstm_calibration_2A_reliability.png | **Calibration poor at all contexts. Predicted probabilities cluster near the base rate. No meaningful calibration improvement with longer context.** | |
| depression_extreme_binary_calibration_2B_ece_vs_context.png | **ECE is high and non-monotonic across contexts — no systematic calibration improvement.** | |
| depression_extreme_binary_lstm_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| depression_extreme_binary_lstm_window_position_4A_profiles.png | **Position profiles are flat but near baseline (0.5), indicating the model has no strong temporal structure to learn for depression. Positive and negative subject profiles are nearly indistinguishable.** | |
| depression_extreme_binary_lstm_window_position_4B_variance.png | **Variance is relatively high and does not decrease meaningfully with longer context, suggesting the model makes inconsistent predictions across windows regardless of L.** | |
| depression_extreme_binary_lstm_subject_consistency_5A_variance.png | **Poor separation between correct and incorrect violins. Many correctly-classified subjects have high variance, and many incorrectly-classified have low variance — indicating classification is nearly random for some subjects.** | |
| depression_extreme_binary_lstm_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| depression_extreme_binary_lstm_subject_consistency_5C_hard_subjects.png | **High fraction never-correct (~25%), low fraction always-correct (~25%). Near-random classification pattern — most subjects are classified correctly at only 2–4 out of 6 contexts, indicating no reliable context length works.** | [REDESIGN NEEDED] **[UNJUSTIFIABLE]** |
| depression_extreme_binary_lstm_cohort_saturation_7A.png | **Only APPLES cohort is available for depression. No multi-cohort comparison possible. Single-cohort with small N makes the results unreliable.** | Note APPLES-only limitation in paper. |
| depression_extreme_binary_lstm_cohort_saturation_7B_n.png | Small N — APPLES only. | |
| depression_extreme_binary_lstm_pr_8A_curves.png | **AUC-PR is near the prevalence baseline across all contexts. PR curves do not improve meaningfully with context.** | |
| depression_extreme_binary_pr_8B_aucpr_vs_context.png | **Non-monotonic AUC-PR — no systematic improvement.** | |
| depression_extreme_binary_lstm_pr_8C_vote_sweep.png | **Poor performance regardless of threshold.** | Supplementary / consider excluding. |
| depression_extreme_binary_lstm_kstar_9A_histogram.png | **K* distribution is broad at all contexts, consistent with the erratic classification pattern. No concentration of K* at low values even at 240m.** | |
| depression_extreme_binary_lstm_kstar_9B_coverage.png | **Coverage curves are non-monotonic, reflecting erratic classification.** | Not blacklisted by name but results are questionable. |
| depression_extreme_binary_lstm_1A_uShape.png | [FLAG: CODE CHANGE] **Training may be unstable for depression given weak signal.** Rerun with BA. | [FLAG: CODE CHANGE] |
| depression_extreme_binary_1B_compute_scaling.png | **Shallow or flat power-law slope — increasing compute does not yield meaningful AUROC gain for depression. The task may be compute-saturated at a low ceiling (~0.68).** | |
| depression_extreme_binary_1C_optimal_epoch.png | **Optimal epoch appears random across context lengths and heads, with no coherent pattern. This further supports that training is unstable for this task.** | Recommend exclusion from paper for depression specifically. |
| auroc_test/heatmap_auroc.png | **Heatmap values are low (~0.63–0.69) across all (L, K) cells. No clear iso-compute structure visible — performance differences are within noise.** | **[UNJUSTIFIABLE]** |
| auroc_test/metric_vs_k_auroc.png | **Non-monotonic K-dependence. AUROC may decrease with more windows for some context lengths.** | **[UNJUSTIFIABLE]** |
| auroc_test/metric_vs_total_auroc.png | **Lines do not converge. No clear total-context efficiency.** | **[UNJUSTIFIABLE]** |
| auroc_test/pareto_front_auroc.png | **Pareto structure may not be meaningful given near-noise-level performance.** | **[UNJUSTIFIABLE]** |
| auroc_test/min_cost_frontier_auroc.png | **Target AUROC above ~0.69 cannot be reached by any configuration.** | **[UNJUSTIFIABLE]** |
| auroc_test/marginal_gain_auroc.png | **Non-monotonic marginal gain — no clear diminishing returns structure.** | **[UNJUSTIFIABLE]** |
| auroc_test/double_tradeoff_auroc.png | **Tradeoff structure not meaningful given erratic performance.** | **[UNJUSTIFIABLE]** |

---

### depression_extreme_binary_transformer

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| depression_extreme_binary_transformer_test_window_sweep_auroc.png | **Transformer shows slightly higher AUROC than LSTM (~0.69–0.72) but still non-monotonic and erratic. No reliable context dependence.** | **[UNJUSTIFIABLE]** Group 3 heads. |
| depression_extreme_binary_transformer_test_window_sweep_balanced_accuracy.png | **BA similarly erratic.** | [SUPPLEMENTARY] **[UNJUSTIFIABLE]** |
| depression_extreme_binary_transformer_calibration_2A_reliability.png | **Marginally better calibration than LSTM but still poor.** | 3×3 grid. |
| depression_extreme_binary_transformer_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| depression_extreme_binary_transformer_window_position_4A_profiles.png | **Position profiles near baseline. No structure.** | |
| depression_extreme_binary_transformer_window_position_4B_variance.png | **High variance across positions.** | |
| depression_extreme_binary_transformer_subject_consistency_5A_variance.png | **Weak correct/incorrect separation.** | |
| depression_extreme_binary_transformer_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| depression_extreme_binary_transformer_subject_consistency_5C_hard_subjects.png | **Similar poor distribution as LSTM.** | [REDESIGN NEEDED] **[UNJUSTIFIABLE]** |
| depression_extreme_binary_transformer_cohort_saturation_7A.png | **APPLES-only. No cohort comparison possible.** | |
| depression_extreme_binary_transformer_cohort_saturation_7B_n.png | Small N. | |
| depression_extreme_binary_transformer_pr_8A_curves.png | **Marginally better than LSTM but still near baseline.** | |
| depression_extreme_binary_transformer_pr_8C_vote_sweep.png | **Poor results.** | Supplementary. |
| depression_extreme_binary_transformer_kstar_9A_histogram.png | **Broad K* distribution at all contexts.** | |
| depression_extreme_binary_transformer_kstar_9B_coverage.png | Not blacklisted. Results questionable. | |
| depression_extreme_binary_transformer_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | **Slightly higher ceiling (~0.72) but same non-structured pattern.** | **[UNJUSTIFIABLE]** |
| auroc_test/metric_vs_k_auroc.png | **Non-monotonic.** | **[UNJUSTIFIABLE]** |
| auroc_test/metric_vs_total_auroc.png | **No clear convergence.** | **[UNJUSTIFIABLE]** |
| auroc_test/pareto_front_auroc.png | **Not meaningful.** | **[UNJUSTIFIABLE]** |
| auroc_test/min_cost_frontier_auroc.png | **Target ~0.72 barely reachable.** | **[UNJUSTIFIABLE]** |
| auroc_test/marginal_gain_auroc.png | **Erratic.** | **[UNJUSTIFIABLE]** |
| auroc_test/double_tradeoff_auroc.png | **No clear structure.** | **[UNJUSTIFIABLE]** |

---

### depression_extreme_binary_mean_pool

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| depression_extreme_binary_mean_pool_test_window_sweep_auroc.png | **MeanPool shows similar erratic pattern to other heads (~0.63–0.70).** | **[UNJUSTIFIABLE]** |
| depression_extreme_binary_mean_pool_test_window_sweep_balanced_accuracy.png | **Erratic BA.** | [SUPPLEMENTARY] **[UNJUSTIFIABLE]** |
| depression_extreme_binary_mean_pool_calibration_2A_reliability.png | **Poor calibration.** | 3×3 grid. |
| depression_extreme_binary_mean_pool_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| depression_extreme_binary_mean_pool_window_position_4A_profiles.png | **Near-baseline position profiles.** | |
| depression_extreme_binary_mean_pool_window_position_4B_variance.png | **High variance. No structure.** | |
| depression_extreme_binary_mean_pool_subject_consistency_5A_variance.png | **Weak separation.** | |
| depression_extreme_binary_mean_pool_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| depression_extreme_binary_mean_pool_subject_consistency_5C_hard_subjects.png | **Same poor distribution.** | [REDESIGN NEEDED] **[UNJUSTIFIABLE]** |
| depression_extreme_binary_mean_pool_cohort_saturation_7A.png | **APPLES-only.** | |
| depression_extreme_binary_mean_pool_cohort_saturation_7B_n.png | Small N. | |
| depression_extreme_binary_mean_pool_pr_8A_curves.png | **Near baseline.** | |
| depression_extreme_binary_mean_pool_pr_8C_vote_sweep.png | **Poor.** | Supplementary. |
| depression_extreme_binary_mean_pool_kstar_9A_histogram.png | **Broad K* at all contexts.** | |
| depression_extreme_binary_mean_pool_kstar_9B_coverage.png | Not blacklisted. Questionable. | |
| depression_extreme_binary_mean_pool_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | **Low values. No iso-compute structure.** | **[UNJUSTIFIABLE]** |
| auroc_test/metric_vs_k_auroc.png | **Erratic.** | **[UNJUSTIFIABLE]** |
| auroc_test/metric_vs_total_auroc.png | **No convergence.** | **[UNJUSTIFIABLE]** |
| auroc_test/pareto_front_auroc.png | **Not meaningful.** | **[UNJUSTIFIABLE]** |
| auroc_test/min_cost_frontier_auroc.png | **Low ceiling.** | **[UNJUSTIFIABLE]** |
| auroc_test/marginal_gain_auroc.png | **Erratic.** | **[UNJUSTIFIABLE]** |
| auroc_test/double_tradeoff_auroc.png | **No structure.** | **[UNJUSTIFIABLE]** |

---

### osa_binary_apples_postqc_lstm

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| osa_binary_apples_postqc_lstm_test_window_sweep_auroc.png | OSA (APPLES-only post-QC) shows an interesting head-specific divergence. LSTM saturates early at ~74–75% AUROC regardless of context length beyond 40m — the LSTM head cannot exploit additional context for OSA after a certain point. Transformer and MeanPool continue improving to 85%. Mean-pool (orange) consistently outperforms segment-level (blue); plateau at K=3–5 for long contexts. | KEY FINDING: head-specific saturation for OSA/LSTM. Highlight head divergence in paper. Group all 3 heads to make the divergence visible. |
| osa_binary_apples_postqc_lstm_test_window_sweep_balanced_accuracy.png | BA shows same LSTM plateau pattern. | [SUPPLEMENTARY] |
| osa_binary_apples_postqc_lstm_calibration_2A_reliability.png | Calibration is adequate at 240m for LSTM. At 30s, under-confident for positives. Improvement with context is moderate (consistent with LSTM AUROC plateau). | 3×3 grid. |
| osa_binary_apples_postqc_calibration_2B_ece_vs_context.png | ECE decreases for Transformer and MeanPool but plateaus for LSTM after ~40m, consistent with the AUROC saturation pattern. This is a strong corroborating signal: LSTM is not just limited in discrimination but also in calibration for OSA at long contexts. | Shared across heads. Note LSTM plateau explicitly. |
| osa_binary_apples_postqc_lstm_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| osa_binary_apples_postqc_lstm_window_position_4A_profiles.png | Slight elevation in predicted probability in the second half of the night for positive subjects (similar to apnea_binary, biologically consistent with REM-related events). Position effect is preserved even at long context for LSTM. | Note biological interpretability. |
| osa_binary_apples_postqc_lstm_window_position_4B_variance.png | Second-half variance elevation for LSTM, consistent with position-dependent probability. | |
| osa_binary_apples_postqc_lstm_subject_consistency_5A_variance.png | LSTM: correct/incorrect separation is present but weaker at long contexts (240m) compared to apnea_binary. Reflects LSTM's AUROC ceiling — the model is not more confident at longer contexts after saturation. | |
| osa_binary_apples_postqc_lstm_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| osa_binary_apples_postqc_lstm_subject_consistency_5C_hard_subjects.png | LSTM has a higher fraction of never-correct subjects than Transformer/MeanPool (~15% vs ~7%), further confirming LSTM's inability to exploit long context for some subjects. | [REDESIGN NEEDED] |
| osa_binary_apples_postqc_lstm_cohort_saturation_7A.png | APPLES-only cohort for this task (post-QC filter). Single cohort, so inter-cohort comparison is not available. AUROC vs context shows LSTM plateau clearly. | Note APPLES-only. No cohort comparison. |
| osa_binary_apples_postqc_lstm_cohort_saturation_7B_n.png | Single cohort N per context. | Remove non-present cohorts from legend. |
| osa_binary_apples_postqc_lstm_pr_8A_curves.png | LSTM AUC-PR plateaus at moderate values consistent with its AUROC ceiling. Transformer/MeanPool PR curves are substantially better at 240m. | Show all 3 heads per task for this comparison to be visible. |
| osa_binary_apples_postqc_pr_8B_aucpr_vs_context.png | AUC-PR shows clear divergence between heads at 120m and beyond. LSTM plateaus while others rise. | Highlight head divergence. |
| osa_binary_apples_postqc_lstm_pr_8C_vote_sweep.png | Similar threshold sweep. | Supplementary. |
| osa_binary_apples_postqc_lstm_kstar_9A_histogram.png | K* at 240m for LSTM is broader than for Transformer — consistent with LSTM requiring more windows to reach correct classification. | Compare heads explicitly. |
| osa_binary_apples_postqc_lstm_kstar_9B_coverage.png | Not blacklisted. Retain for completeness. | |
| osa_binary_apples_postqc_lstm_1A_uShape.png | [FLAG: CODE CHANGE] LSTM training for OSA may converge quickly to a local optimum at long contexts (consistent with plateau). Rerun with BA. | [FLAG: CODE CHANGE] |
| osa_binary_apples_postqc_1B_compute_scaling.png | LSTM power-law slope is very shallow at high compute — reflecting the AUROC ceiling. Transformer scales well. The divergence between LSTM and Transformer at high compute is the largest of any task. | Strong evidence that head architecture matters for OSA. |
| osa_binary_apples_postqc_1C_optimal_epoch.png | Check if LSTM optimal epoch is low at long contexts — could explain the plateau (insufficient training). If so, the ceiling is a training artifact, not an architectural limit. | Key diagnostic check: if LSTM epoch is low at 120m+, recommend more training for LSTM on this task. |
| auroc_test/heatmap_auroc.png | LSTM heatmap shows a clear plateau ridge: above L=40m, AUROC barely changes with K. In contrast, Transformer and MeanPool heatmaps show continued improvement. | Group all 3 heads for cross-head heatmap comparison. |
| auroc_test/metric_vs_k_auroc.png | LSTM lines for L≥40m overlap (plateau), while Transformer/MeanPool lines are spread (still improving). | |
| auroc_test/metric_vs_total_auroc.png | LSTM curves converge at low total context budget; Transformer/MeanPool continue rising. The gap between heads grows with total context. | |
| auroc_test/pareto_front_auroc.png | Long-L still Pareto-dominant for Transformer/MeanPool. LSTM Pareto front reaches a lower ceiling. | |
| auroc_test/min_cost_frontier_auroc.png | Target >80% AUROC cannot be reached by LSTM at any cost. Transformer can reach it at L=240m. | |
| auroc_test/marginal_gain_auroc.png | LSTM shows near-zero marginal gain at L≥40m regardless of K. Transformer shows continued gain. | |
| auroc_test/double_tradeoff_auroc.png | LSTM: long-L provides no benefit. Transformer: long-L essential. | |

---

### osa_binary_apples_postqc_transformer

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| osa_binary_apples_postqc_transformer_test_window_sweep_auroc.png | Transformer reaches ~85% at 240m, the best head for OSA. Monotonic improvement with both L and K. K=3–5 sufficient at 240m. Contrast with LSTM plateau behavior. | Group 3 heads. |
| osa_binary_apples_postqc_transformer_test_window_sweep_balanced_accuracy.png | BA mirrors AUROC improvement. | [SUPPLEMENTARY] |
| osa_binary_apples_postqc_transformer_calibration_2A_reliability.png | Good calibration at 240m. Better than LSTM at all contexts. | 3×3 grid. |
| osa_binary_apples_postqc_transformer_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| osa_binary_apples_postqc_transformer_window_position_4A_profiles.png | Second-half elevation attenuated vs LSTM — Transformer integrates temporal context more uniformly. | |
| osa_binary_apples_postqc_transformer_window_position_4B_variance.png | Lower variance than LSTM across positions. | |
| osa_binary_apples_postqc_transformer_subject_consistency_5A_variance.png | Clear correct/incorrect separation, consistent with higher absolute performance. | |
| osa_binary_apples_postqc_transformer_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| osa_binary_apples_postqc_transformer_subject_consistency_5C_hard_subjects.png | Fewer hard subjects than LSTM (~7% never-correct). | [REDESIGN NEEDED] |
| osa_binary_apples_postqc_transformer_cohort_saturation_7A.png | APPLES-only. Monotonic improvement with context. | |
| osa_binary_apples_postqc_transformer_cohort_saturation_7B_n.png | Same N. | Remove non-present cohorts from legend. |
| osa_binary_apples_postqc_transformer_pr_8A_curves.png | Best PR curves for OSA task. | |
| osa_binary_apples_postqc_transformer_pr_8C_vote_sweep.png | Tight curves at long contexts. | Supplementary. |
| osa_binary_apples_postqc_transformer_kstar_9A_histogram.png | K* concentrated at K*=1–3 at 240m. Contrast with LSTM. | |
| osa_binary_apples_postqc_transformer_kstar_9B_coverage.png | Not blacklisted. | |
| osa_binary_apples_postqc_transformer_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Monotonic improvement across L and K. No plateau. | |
| auroc_test/metric_vs_k_auroc.png | K-saturation at K=3–5 for long contexts. | |
| auroc_test/metric_vs_total_auroc.png | Continues rising at max total context. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Target 85% reachable at L=240m, K=3+. | |
| auroc_test/marginal_gain_auroc.png | Standard diminishing returns. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal. | |

---

### osa_binary_apples_postqc_mean_pool

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| osa_binary_apples_postqc_mean_pool_test_window_sweep_auroc.png | MeanPool reaches ~83% at 240m — competitive with Transformer, significantly above LSTM. The simplicity of MeanPool (global frequency averaging) outperforms LSTM's sequential modeling for OSA. This is surprising and supports a frequency-domain interpretation of the OSA signal. | Note: MeanPool competitive with Transformer for OSA — could suggest spectral features dominate for apnea severity. |
| osa_binary_apples_postqc_mean_pool_test_window_sweep_balanced_accuracy.png | BA consistent. | [SUPPLEMENTARY] |
| osa_binary_apples_postqc_mean_pool_calibration_2A_reliability.png | Good calibration. | 3×3 grid. |
| osa_binary_apples_postqc_mean_pool_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| osa_binary_apples_postqc_mean_pool_window_position_4A_profiles.png | Flat position profiles — MeanPool does not capture positional structure (by design). | |
| osa_binary_apples_postqc_mean_pool_window_position_4B_variance.png | Low variance, uniform across night. | |
| osa_binary_apples_postqc_mean_pool_subject_consistency_5A_variance.png | Good separation, consistent with high absolute performance. | |
| osa_binary_apples_postqc_mean_pool_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| osa_binary_apples_postqc_mean_pool_subject_consistency_5C_hard_subjects.png | Similar to Transformer (~8% never-correct). | [REDESIGN NEEDED] |
| osa_binary_apples_postqc_mean_pool_cohort_saturation_7A.png | APPLES-only. Monotonic improvement. | |
| osa_binary_apples_postqc_mean_pool_cohort_saturation_7B_n.png | Same N. | Legend fix. |
| osa_binary_apples_postqc_mean_pool_pr_8A_curves.png | Strong PR improvement, competitive with Transformer. | |
| osa_binary_apples_postqc_mean_pool_pr_8C_vote_sweep.png | Consistent. | Supplementary. |
| osa_binary_apples_postqc_mean_pool_kstar_9A_histogram.png | K* similar to Transformer at 240m. | |
| osa_binary_apples_postqc_mean_pool_kstar_9B_coverage.png | Not blacklisted. | |
| osa_binary_apples_postqc_mean_pool_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Values close to Transformer. No LSTM plateau. | |
| auroc_test/metric_vs_k_auroc.png | Similar K-saturation to Transformer. | |
| auroc_test/metric_vs_total_auroc.png | MeanPool line close to Transformer, far above LSTM. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Efficient: target 83% reachable at L=240m+small K. | |
| auroc_test/marginal_gain_auroc.png | Standard diminishing returns. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal. | |

---

### ⚠️ DROPPED — cvd_binary_lstm

> Task removed from paper (AUROC ~0.689, near flat across all contexts). Figures retained for reference.

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| cvd_binary_lstm_test_window_sweep_auroc.png | ⚠️ DROPPED. AUROC near-flat at ~0.69 across all contexts and K values. No meaningful context dependence. Subject mean-pool does not improve over segment-level. PSG signals appear not informative for CVD classification in this dataset. | Excluded from paper. |
| cvd_binary_lstm_calibration_2A_reliability.png | ⚠️ DROPPED. Poorly calibrated — probabilities cluster at base rate. | Excluded. |
| cvd_binary_calibration_2B_ece_vs_context.png | ⚠️ DROPPED. High ECE, no improvement. | Excluded. |
| cvd_binary_lstm_cohort_saturation_7A.png | ⚠️ DROPPED. No cohort shows improvement with context. | Excluded. |
| cvd_binary_lstm_pr_8A_curves.png | ⚠️ DROPPED. AUC-PR near prevalence. | Excluded. |
| cvd_binary_pr_8B_aucpr_vs_context.png | ⚠️ DROPPED. No improvement. | Excluded. |
| cvd_binary_lstm_kstar_9A_histogram.png | ⚠️ DROPPED. K* broadly distributed, consistent with near-random classification. | Excluded. |
| cvd_binary_lstm_1A_uShape.png | ⚠️ DROPPED. [FLAG: CODE CHANGE if regenerating] | Excluded. |
| auroc_test/heatmap_auroc.png | ⚠️ DROPPED. Near-flat heatmap. No iso-compute structure visible. | Excluded. |

---

### ⚠️ DROPPED — cvd_binary_transformer

> Task removed from paper. Figures retained for reference.

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| cvd_binary_transformer_test_window_sweep_auroc.png | ⚠️ DROPPED. Same near-flat AUROC as LSTM. Transformer provides no advantage for CVD. | Excluded. |
| cvd_binary_transformer_calibration_2A_reliability.png | ⚠️ DROPPED. Poor calibration. | Excluded. |
| cvd_binary_transformer_cohort_saturation_7A.png | ⚠️ DROPPED. No cohort improvement. | Excluded. |
| cvd_binary_transformer_pr_8A_curves.png | ⚠️ DROPPED. Near baseline. | Excluded. |
| cvd_binary_transformer_kstar_9A_histogram.png | ⚠️ DROPPED. Broad K* distribution. | Excluded. |
| cvd_binary_transformer_1A_uShape.png | ⚠️ DROPPED. [FLAG: CODE CHANGE if regenerating] | Excluded. |
| auroc_test/heatmap_auroc.png | ⚠️ DROPPED. Near-flat. | Excluded. |

---

### ⚠️ DROPPED — sleepiness_binary_lstm

> Task removed from paper (AUROC ~0.629, no context benefit).

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleepiness_binary_lstm_test_window_sweep_auroc.png | ⚠️ DROPPED. AUROC ~0.63, near chance, no context dependence. ESS-based sleepiness has weak PSG correlates at any context length. | Excluded. |
| sleepiness_binary_calibration_2B_ece_vs_context.png | ⚠️ DROPPED. High ECE across all contexts. | Excluded. |
| sleepiness_binary_lstm_cohort_saturation_7A.png | ⚠️ DROPPED. No cohort improvement. | Excluded. |
| sleepiness_binary_lstm_pr_8A_curves.png | ⚠️ DROPPED. Near baseline AUC-PR. | Excluded. |
| sleepiness_binary_lstm_kstar_9A_histogram.png | ⚠️ DROPPED. Broad K* at all contexts. | Excluded. |
| sleepiness_binary_lstm_1A_uShape.png | ⚠️ DROPPED. [FLAG: CODE CHANGE] | Excluded. |
| auroc_test/heatmap_auroc.png | ⚠️ DROPPED. Near-flat low-value heatmap. | Excluded. |

---

### ⚠️ DROPPED — sleepiness_binary_transformer

> Task removed from paper.

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleepiness_binary_transformer_test_window_sweep_auroc.png | ⚠️ DROPPED. Similar to LSTM. No improvement. | Excluded. |
| sleepiness_binary_transformer_cohort_saturation_7A.png | ⚠️ DROPPED. | Excluded. |
| sleepiness_binary_transformer_kstar_9A_histogram.png | ⚠️ DROPPED. | Excluded. |
| sleepiness_binary_transformer_1A_uShape.png | ⚠️ DROPPED. [FLAG: CODE CHANGE] | Excluded. |
| auroc_test/heatmap_auroc.png | ⚠️ DROPPED. | Excluded. |

---

### ⚠️ DROPPED — psqi_binary_lstm

> Task removed from paper (AUROC ~0.557, near chance).

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| psqi_binary_lstm_test_window_sweep_auroc.png | ⚠️ DROPPED. AUROC ~0.56, near chance. PSQI global sleep quality score is the weakest-correlating target of all tasks. No meaningful PSG signal detected. | Excluded. |
| psqi_binary_calibration_2B_ece_vs_context.png | ⚠️ DROPPED. Worst calibration of all tasks. | Excluded. |
| psqi_binary_lstm_cohort_saturation_7A.png | ⚠️ DROPPED. | Excluded. |
| psqi_binary_lstm_pr_8A_curves.png | ⚠️ DROPPED. | Excluded. |
| psqi_binary_pr_8B_aucpr_vs_context.png | ⚠️ DROPPED. | Excluded. |
| psqi_binary_lstm_kstar_9A_histogram.png | ⚠️ DROPPED. | Excluded. |
| psqi_binary_lstm_1A_uShape.png | ⚠️ DROPPED. [FLAG: CODE CHANGE] | Excluded. |
| auroc_test/heatmap_auroc.png | ⚠️ DROPPED. Near-chance values. | Excluded. |

---

## Table 2 — Across-Task Figures

---

### Cross-Round Merged Figure (v3 + v3_full + v3_abl)

**Proposed figure: Modality Contribution Bar Chart** — combines fast-channel baseline
(this file), full-channel baseline (figure_interpretations_v3_full.md), and ablation
deltas (figure_interpretations_v3_abl.md) into a single 5-panel figure.

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| modality_ablation_summary_bar.png (pending) | Five-panel grouped bar chart (one per task). Each panel shows absolute AUROC for all ablation conditions plus v3_fast and v3_full reference lines. Key findings visible at a glance: sleep_efficiency BAS-only ≈ full (−0.005); sex EKG-dominant; apnea RESP+BAS both needed; BMI RESP hurts. The v3_full reference line shows where full-channel sits relative to each ablation condition — for BMI, full-channel BEATS the no_resp ablation, confirming that EKG+EMG (not RESP) drive the full-channel BMI gain. | [CROSS-ROUND FIGURE] Generate from: v3 fast baselines (this file, Table 2 saturation values at task context), v3_full baselines (figure_interpretations_v3_full.md Table 2), and v3_abl analysis.csv. Design per figure_interpretations_v3_abl.md §Cross-Round Merged Figure Recommendation. |

---

### Saturation Curves

> **Decision (instruction #12)**: AUROC saturation curves are primary. Balanced Accuracy saturation curves (if generated) should be supplementary.
>
> **[CODE FIX NEEDED — plot_saturation.py]**: Current figures read `test_auroc` from
> per-experiment `summary.csv` (segment-level / K=1 from training evaluation loop).
> This is inconsistent with all other paper figures, which use subject-level mean-pool
> AUROC (K=all, `mean_prob_auroc` from `analysis.csv`). Fix: change `plot_saturation.py`
> to read `mean_prob_auroc` at `k='all'` from the collected `analysis.csv`.
> Numbers below are from `analysis.csv` (mean_prob_auroc, K=all) — the correct targets.

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| saturation/saturation_sex_binary_auroc_test.png | All three heads show monotonic AUROC improvement from 30s to 240m. Correct values (mean_prob_auroc K=all): Transformer 30s→91.0% 240m (vs 83.2% at 30s), LSTM 30s→85.7% 240m, MeanPool 30s→81.8% 240m. No saturation by 240m (L* > 120m). Transformer advantage grows with context length. | [CODE FIX] Regenerate using mean_prob_auroc K=all from analysis.csv. GROUP: merge all task saturation curves into 2×4 panel. LSTM=blue, Transformer=orange, MeanPool=green. Log-scale x-axis. Remove figure titles; add panel labels. |
| saturation/saturation_age_class_auroc_test.png | Monotonic improvement across heads. Correct values: Transformer 240m=90.5%, LSTM 240m=88.5%, MeanPool 240m=85.0%. Curve still rising but with decreasing rate — L* estimated ~120–240m. All heads converge at high values; gap between heads small. | [CODE FIX] Regenerate. See grouping comment above. |
| saturation/saturation_apnea_binary_auroc_test.png | Steepest absolute AUROC of any task. Correct values: Transformer 240m=85.4%, LSTM 240m=82.7%, MeanPool 240m=76.5%. Context sensitivity is strong (all rising from ~75% at 30s). L* estimated ~80–120m where curves begin to flatten. Apnea is the highest-performing task. Note: MeanPool is notably weaker (~9 pp below Transformer at 240m). | [CODE FIX] Regenerate. Highest-performing task; highlight in paper. |
| saturation/saturation_bmi_binary_auroc_test.png | Weakest context sensitivity of retained tasks. Correct values: Transformer 240m=77.7%, LSTM 240m=74.7%, MeanPool 240m=74.6%. Heads plateau early (L*~40m). Minimal benefit of context beyond 40m. All heads reach similar ceiling. | [CODE FIX] Regenerate. Lowest ceiling — contrast to apnea and sleep_efficiency. |
| saturation/saturation_sleep_efficiency_binary_auroc_test.png | Strongest context sensitivity — no saturation at 240m (L* > 240m). Correct values: Transformer 30s=70.7%→240m=83.1%, LSTM 30s=69.7%→240m=78.8%, MeanPool 30s=69.4%→240m=76.0%. Transformer leads. The lack of saturation is a key finding. | [CODE FIX] Regenerate. KEY FIGURE. Emphasize "not yet saturated." |
| saturation/saturation_depression_extreme_binary_auroc_test.png | **Non-monotonic and erratic across heads and contexts. Correct values: Transformer 240m=74.6%, LSTM 240m=75.0%, MeanPool 240m=75.2% — but patterns across contexts are not monotonically rising. APPLES-only cohort and small N likely contribute to instability.** | **[UNJUSTIFIABLE]** [CODE FIX] Regenerate. Move to supplementary or discussion. |
| saturation/saturation_osa_binary_apples_postqc_auroc_test.png | Marked head divergence. Correct values: Transformer 240m=86.1% (monotonically rising, no saturation), LSTM 240m=77.4% (saturates around 40m then continues slowly), MeanPool 240m=84.8% (strong rise, competitive with Transformer). LSTM saturates much earlier than Transformer/MeanPool. | [CODE FIX] Regenerate. KEY FINDING: head-specific saturation. LSTM ceiling is ~9 pp below Transformer at 240m. |
| saturation/saturation_cvd_binary_auroc_test.png | ⚠️ DROPPED task. AUROC near-flat at ~0.69 across all contexts and heads. No context benefit. | Excluded from paper. Retained for reference. |
| saturation/saturation_sleepiness_binary_auroc_test.png | ⚠️ DROPPED task. AUROC ~0.63, no context benefit across heads. | Excluded. |
| saturation/saturation_psqi_binary_auroc_test.png | ⚠️ DROPPED task. AUROC ~0.56, near chance, no context benefit. | Excluded. |

---

### Task Comparison

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| **[Fig 3] task_comparison/task_comparison_6A_scatter.png + task_comparison_6C_lstar.png** | **GROUPED → Fig 3 "Task Landscape" (2-panel composite).** (a) Scatter: task difficulty (mean_prob_auroc @30s K=all) vs context sensitivity (AUROC gain 30s→240m K=all). Three clusters: (1) high-sensitivity (sleep_efficiency, OSA Transformer); (2) moderate (apnea, sex, age); (3) weak (bmi, depression). Task labels on points directly. (b) L* lollipop: sleep_eff →arrow (L*>240m); apnea ~80m; sex/age ~120m; OSA ~120m; bmi ~40m; depression hatched (undefined). Both panels use consistent task-color scheme. | [CODE FIX] Both regenerated from analysis.csv with mean_prob_auroc K=all. Log-scale x-axis on lollipop. Remove figure titles; shared caption. Exclude dropped tasks. Task name labels on scatter points (no separate legend). |
| task_comparison/task_comparison_6B_bars.png | Grouped bar chart — AUROC per task per head at 30s, 80m, 240m. Confirms ranking: apnea > sex/age > sleep_eff > OSA > bmi > depression. Information is largely redundant with Fig 1 and Table II. | **[EXCLUDED from paper]** Redundant with Fig 1 + Table II. Retain as internal reference only. |

---

### Scaling Laws

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| scaling_laws/sex_binary_1B_compute_scaling.png | Sex binary shows a robust power-law relationship between total training FLOPs and test AUROC for all three heads. The Transformer line has the steepest slope — it improves faster per FLOP than LSTM or MeanPool at high compute. At the highest compute, Transformer reaches ~89% while LSTM and MeanPool reach ~84–86%. The power-law fits (dashed lines) hold reasonably well across ~6 orders of magnitude of FLOPs. | GROUP: combine all tasks into a 2×4 multi-panel figure (one panel per task). Use consistent marker/line coding: LSTM=circle/blue, Transformer=square/orange, MeanPool=triangle/green. Color points by L (context length) as currently done. Remove panel titles; add panel labels. |
| scaling_laws/sex_binary_1C_optimal_epoch.png | Non-monotonic optimal epoch vs context for sex_binary. Transformer requires consistently more epochs (19–38) than LSTM (8–20). No clear trend with context length for any head. Limits interpretability as a scaling law insight. | [BORDERLINE — SUPPLEMENTARY] Retain but label as supplementary figure. If pattern is non-monotonic for most tasks, add a general note that optimal epoch is determined by training dynamics, not purely by context length. |
| scaling_laws/age_class_1B_compute_scaling.png | Power-law holds for age_class. Similar structure to sex_binary: Transformer leads. The FLOPs range and slope are comparable across tasks, suggesting training compute efficiency is task-agnostic given the same architecture. | Part of multi-panel group. |
| scaling_laws/age_class_1C_optimal_epoch.png | Non-monotonic pattern, particularly for MeanPool (drops from 12 at 40m to 14 at 80m unexpectedly). Transformer remains consistently high (~23–36 epochs). | Supplementary. |
| scaling_laws/apnea_binary_1B_compute_scaling.png | Steepest power-law slope of all tasks — apnea shows the best compute efficiency. Transformer reaches ~93% at max compute, making it the closest to clinical-grade performance. | Highlight as best-case scaling example. |
| scaling_laws/apnea_binary_1C_optimal_epoch.png | Lower epoch counts than sex_binary (apnea converges faster). Pattern still non-monotonic. | Supplementary. |
| scaling_laws/bmi_binary_1B_compute_scaling.png | Shallowest power-law slope — bmi has the worst compute efficiency. Adding compute does not overcome the ceiling at ~0.74. | Contrast with apnea as weakest scaling example. |
| scaling_laws/bmi_binary_1C_optimal_epoch.png | Noisy pattern, consistent with weak-signal training instability. | Supplementary. |
| scaling_laws/sleep_efficiency_binary_1B_compute_scaling.png | Power-law slope intermediate. AUROC at max compute still rising — not saturated. Suggests more compute (longer context at test time) would continue to improve sleep_efficiency classification. | Note "not at ceiling" interpretation. |
| scaling_laws/sleep_efficiency_binary_1C_optimal_epoch.png | Check if monotonic — sleep_efficiency is the candidate task with a justifiable 1C relationship. If monotonic (longer L → more epochs needed), include in paper; otherwise supplementary. | Key diagnostic: inspect and determine inclusion. |
| scaling_laws/depression_extreme_binary_1B_compute_scaling.png | **Flat or near-flat power law — depression does not benefit from additional compute. Ceiling ~0.70 appears to be an intrinsic task ceiling, not a compute limitation.** | **[UNJUSTIFIABLE scaling]** Supplementary or exclude from 1B panel. |
| scaling_laws/depression_extreme_binary_1C_optimal_epoch.png | **Erratic epoch pattern, further evidence of training instability for depression.** | Exclude from paper or supplementary. |
| scaling_laws/osa_binary_apples_postqc_1B_compute_scaling.png | Dramatic head divergence: LSTM power-law flattens at high compute while Transformer continues rising. The compute scaling plot makes the architectural divergence visible as distinct power-law slopes. Transformer: steep slope; LSTM: near-flat above ~10^12 FLOPs. | KEY FIGURE for OSA — include prominently to illustrate head-specific scaling behavior. |
| scaling_laws/osa_binary_apples_postqc_1C_optimal_epoch.png | Check whether LSTM's optimal epoch is lower than expected at long contexts — could indicate premature convergence contributing to the performance plateau. | Diagnostic check. |
| scaling_laws/cvd_binary_1B_compute_scaling.png | ⚠️ DROPPED | Excluded. |
| scaling_laws/cvd_binary_1C_optimal_epoch.png | ⚠️ DROPPED | Excluded. |
| scaling_laws/sleepiness_binary_1B_compute_scaling.png | ⚠️ DROPPED | Excluded. |
| scaling_laws/sleepiness_binary_1C_optimal_epoch.png | ⚠️ DROPPED | Excluded. |
| scaling_laws/psqi_binary_1B_compute_scaling.png | ⚠️ DROPPED | Excluded. |
| scaling_laws/psqi_binary_1C_optimal_epoch.png | ⚠️ DROPPED | Excluded. |
