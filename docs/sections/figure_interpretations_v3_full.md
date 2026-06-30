# Figure Interpretations — phase0_v3_full (Full Channel, up to 23 channels)

**How to use:** Fill in the *Interpretation* column with what you observe in the figure
for this specific task/head combination. Use *Additional Comments* for paper relevance,
anomalies, surprising findings, or things to follow up on.

**Note:** Tasks dropped from the paper (cvd_binary, sleepiness_binary, psqi_binary) are
included here but marked ⚠️ DROPPED.

---

## Global Decisions (same as v3 unless noted)

- **Primary metric**: AUROC. Balanced Accuracy figures marked [SUPPLEMENTARY].
- **Window aggregation**: Subject mean-pool (mp) retained; majority-vote removed.
- **1A uShape plots**: All [FLAG: CODE CHANGE] — rerun with BA metric.
- **Blacklisted figures (all tasks)**: `*_calibration_2C_ece_vs_k.png`, `*_subject_consistency_5B_variance_vs_k.png`.
- **5C hard-subject plots**: All [REDESIGN NEEDED].
- **1C optimal epoch**: Supplementary across all tasks.

## Cross-Condition Summary: Full-Channel vs Fast-Channel

A central finding from comparing phase0_v3 (fast/7–8 ch) and phase0_v3_full (full/up to 23 ch):

| Task | Fast-ch Transformer@240m | Full-ch Transformer@240m | Direction | Note |
|---|---|---|---|---|
| sex_binary | ~89% | 90.6% | **Full BETTER (+1.6 pp)** | Body-signature channels help |
| apnea_binary | ~93% | 88.2% | **Full WORSE (−4.8 pp)** | EEG-only signal; more channels hurt |
| sleep_efficiency | ~85% | 79.2% | **Full WORSE (−5.8 pp)** | EEG-dominant task |
| bmi_binary | ~74% | 80.0% | **Full BETTER (+6 pp)** | Body-composition channels help |
| depression | ~65–70% | 72.9% | Full slightly better | Still erratic; APPLES-only |
| osa_binary_apples_postqc | ~85% | 80.8% | **Full WORSE (−4.2 pp)** | LSTM still plateaus; Transformer lower ceiling |

**Interpretation of the pattern:** Full-channel is NOT universally better. Tasks whose
signal is primarily encoded in EEG rhythms/sleep staging (apnea via arousal EEG,
sleep efficiency via NREM/REM structure) are hurt by adding channels — possibly because
additional respiratory/cardiac channels introduce optimization challenges without adding
discriminative power. Tasks whose signal relates to body composition (BMI, sex) benefit
from the additional modalities (limb EMG, respiratory belt, cardiac metrics). This
task-modality interaction is a key finding for the discussion section.

**MeanPool degradation**: MeanPool is more harmed than LSTM/Transformer by full-channel,
since simple spectral averaging across many heterogeneous channels yields noisier global
representations. Transformer's cross-channel attention mechanism handles multi-modal
integration more gracefully.

---

## Table 1 — Per-Task Figures

---

### sex_binary_lstm

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sex_binary_lstm_test_window_sweep_auroc.png | Full-channel LSTM reaches ~87% at 240m (vs ~84% fast-channel) — a modest improvement. At 30s, full-channel LSTM is ~75%, slightly below fast-channel (~78%). The short-context penalty is smaller for LSTM than MeanPool, suggesting LSTM can integrate multi-channel information sequentially even at shorter windows. K-saturation pattern same as fast-channel (K=5–8 at long contexts). | Group 3 heads. Remove majority-vote. |
| sex_binary_lstm_test_window_sweep_balanced_accuracy.png | BA pattern mirrors AUROC, with same short-context penalty and long-context improvement vs fast-channel. | [SUPPLEMENTARY] |
| sex_binary_lstm_calibration_2A_reliability.png | Calibration at 240m is similar to or slightly better than fast-channel, consistent with higher absolute AUROC. Short-context calibration may be slightly worse due to the added channel complexity at low K. | 3×3 head×context grid per task (same format as v3). |
| sex_binary_calibration_2B_ece_vs_context.png | ECE improvement pattern similar to fast-channel but slightly lower ECE at 240m for Transformer, reflecting better calibration with full-channel for sex classification. | Shared across heads. |
| sex_binary_lstm_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sex_binary_lstm_window_position_4A_profiles.png | Position profiles flat, same conclusion as fast-channel. Additional channels do not introduce position-dependent artifacts. | Group 3 heads. |
| sex_binary_lstm_window_position_4B_variance.png | Low and flat variance, same as fast-channel. | Group with 4A. |
| sex_binary_lstm_subject_consistency_5A_variance.png | Clear correct/incorrect separation, similar to fast-channel. Slight improvement in the separation at 240m vs fast-channel, reflecting higher absolute accuracy. | 3×3 grid. |
| sex_binary_lstm_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sex_binary_lstm_subject_consistency_5C_hard_subjects.png | Hard-subject fraction slightly lower than fast-channel at 240m, consistent with full-channel improvement. | [REDESIGN NEEDED] |
| sex_binary_lstm_cohort_saturation_7A.png | Cohort-level pattern qualitatively same as fast-channel. SHHS likely benefits most from additional channels; APPLES may show less improvement. | Group 3 heads. |
| sex_binary_lstm_cohort_saturation_7B_n.png | Same N as fast-channel (channel availability does not change subject count if all channels are imputed/available). | [LEGEND FIX] Check MrOS. Single panel per task. |
| sex_binary_lstm_pr_8A_curves.png | AUC-PR improvement at long contexts slightly better than fast-channel. | 3-panel per task. |
| sex_binary_pr_8B_aucpr_vs_context.png | Monotonic AUC-PR improvement, best at 240m. | Shared across heads. |
| sex_binary_lstm_pr_8C_vote_sweep.png | Consistent with fast-channel; majority-vote threshold sweep. | Supplementary. |
| sex_binary_lstm_kstar_9A_histogram.png | K* distribution at 240m similar to fast-channel; slightly more concentrated at K*=1. | 3-panel per task. |
| sex_binary_lstm_kstar_9B_coverage.png | Not blacklisted. Optional. | |
| sex_binary_lstm_1A_uShape.png | [FLAG: CODE CHANGE] Rerun with BA metric. | [FLAG: CODE CHANGE] |
| sex_binary_1B_compute_scaling.png | Full-channel power-law: Transformer slope slightly steeper, reflecting improved efficiency with multi-modal data for sex classification. All heads' scaling curves shift upward vs fast-channel. | Group across tasks in multi-panel figure. |
| sex_binary_1C_optimal_epoch.png | Non-monotonic pattern likely preserved. Full-channel may require more epochs at long contexts due to added input complexity. | Supplementary. |
| auroc_test/heatmap_auroc.png | Full-channel heatmap values are slightly higher than fast-channel at long-L cells. The iso-compute advantage of long-L is preserved. | Group with metric_vs_k and pareto. |
| auroc_test/metric_vs_k_auroc.png | K-saturation pattern same as fast-channel. Slightly higher plateau at each L. | |
| auroc_test/metric_vs_total_auroc.png | Full-channel lines sit above fast-channel at same total context — confirming multi-modal benefit for sex. | Compare vs fast-channel overlay in paper. |
| auroc_test/pareto_front_auroc.png | Long-L remains Pareto-dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Slightly lower minimum cost to reach high AUROC targets vs fast-channel. | |
| auroc_test/marginal_gain_auroc.png | Same rapid decay. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal. | |

---

### sex_binary_transformer

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sex_binary_transformer_test_window_sweep_auroc.png | Transformer with full-channel reaches ~90.6% at 240m (K=all) vs ~89% fast-channel — the largest absolute improvement from full-channel among heads. At 30s, ~75% vs ~78% fast-channel (worse at short context). Short-context penalty is clear: the Transformer needs more context to benefit from additional channels. At 80m+, full-channel clearly outperforms fast. | Group 3 heads. Remove mv. |
| sex_binary_transformer_test_window_sweep_balanced_accuracy.png | BA follows same pattern. | [SUPPLEMENTARY] |
| sex_binary_transformer_calibration_2A_reliability.png | Best calibration of any head×condition for sex. Near-perfect diagonal at 240m. | 3×3 grid. |
| sex_binary_transformer_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sex_binary_transformer_window_position_4A_profiles.png | Flat position profiles. Full-channel adds no position-dependent artifacts. | |
| sex_binary_transformer_window_position_4B_variance.png | Low variance. | |
| sex_binary_transformer_subject_consistency_5A_variance.png | Tightest correct-class violin of any head×condition. Near-zero variance at 240m for correctly-classified subjects. | |
| sex_binary_transformer_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sex_binary_transformer_subject_consistency_5C_hard_subjects.png | Fewest hard subjects. | [REDESIGN NEEDED] |
| sex_binary_transformer_cohort_saturation_7A.png | SHHS shows strongest improvement; cross-cohort improvement confirmed with full-channel. | Group 3 heads. |
| sex_binary_transformer_cohort_saturation_7B_n.png | Same N. | [LEGEND FIX] |
| sex_binary_transformer_pr_8A_curves.png | Best PR curves for sex task in full-channel. | |
| sex_binary_transformer_pr_8C_vote_sweep.png | Consistent. | Supplementary. |
| sex_binary_transformer_kstar_9A_histogram.png | Most concentrated K* at 240m — single window sufficient for most subjects. | |
| sex_binary_transformer_kstar_9B_coverage.png | Not blacklisted. Optional. | |
| sex_binary_transformer_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Highest values in the sex_binary full-channel heatmaps. | |
| auroc_test/metric_vs_k_auroc.png | Fastest K-saturation; plateau by K=3 at 240m. | |
| auroc_test/metric_vs_total_auroc.png | Transformer full-channel line highest for sex. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Most efficient path to high AUROC target. | |
| auroc_test/marginal_gain_auroc.png | Rapid decay. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal. | |

---

### sex_binary_mean_pool

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sex_binary_mean_pool_test_window_sweep_auroc.png | Full-channel MeanPool reaches ~82% at 240m — notably lower than fast-channel MeanPool (~86%). This is the starkest degradation: MeanPool's simple frequency averaging does not benefit from additional channels and is actively harmed by the increased input dimensionality. The more channels, the less discriminative the average becomes. At 30s, full MeanPool ~72%, similar to fast-channel short-context. | MeanPool degradation in full-channel is the largest of any head for sex. Highlight this contrast. |
| sex_binary_mean_pool_test_window_sweep_balanced_accuracy.png | BA shows same MeanPool degradation pattern. | [SUPPLEMENTARY] |
| sex_binary_mean_pool_calibration_2A_reliability.png | Calibration similar to fast-channel or slightly worse due to lower absolute accuracy. | 3×3 grid. |
| sex_binary_mean_pool_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sex_binary_mean_pool_window_position_4A_profiles.png | Flat position profiles. | |
| sex_binary_mean_pool_window_position_4B_variance.png | Similar or slightly higher variance than fast-channel (less confident predictions). | |
| sex_binary_mean_pool_subject_consistency_5A_variance.png | Weaker correct/incorrect separation vs fast-channel MeanPool. | |
| sex_binary_mean_pool_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sex_binary_mean_pool_subject_consistency_5C_hard_subjects.png | More hard subjects than fast-channel MeanPool, reflecting degraded performance. | [REDESIGN NEEDED] |
| sex_binary_mean_pool_cohort_saturation_7A.png | Cohort patterns same structure but at lower AUROC ceiling. | |
| sex_binary_mean_pool_cohort_saturation_7B_n.png | Same N. | [LEGEND FIX] |
| sex_binary_mean_pool_pr_8A_curves.png | Lower AUC-PR than fast-channel MeanPool at same context. | |
| sex_binary_mean_pool_pr_8C_vote_sweep.png | Consistent. | Supplementary. |
| sex_binary_mean_pool_kstar_9A_histogram.png | Broader K* distribution vs fast-channel — more windows needed per subject. | |
| sex_binary_mean_pool_kstar_9B_coverage.png | Not blacklisted. | |
| sex_binary_mean_pool_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Lower ceiling than fast-channel MeanPool. Iso-compute structure preserved. | |
| auroc_test/metric_vs_k_auroc.png | Slower K-saturation vs fast-channel MeanPool. | |
| auroc_test/metric_vs_total_auroc.png | MeanPool full-channel line below fast-channel MeanPool. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant but lower ceiling. | |
| auroc_test/min_cost_frontier_auroc.png | Higher minimum cost to reach same target vs fast-channel MeanPool. | |
| auroc_test/marginal_gain_auroc.png | Same rapid decay. | |
| auroc_test/double_tradeoff_auroc.png | Same structure. | |

---

### age_class_lstm

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| age_class_lstm_test_window_sweep_auroc.png | Age classification with full-channel: pattern similar to fast-channel (strong context dependence, Transformer leads). Full-channel impact on age is uncertain without direct comparison data, but likely mixed — EEG-based sleep staging is important for age prediction, so more channels may not uniformly help. Monitor AUROC numbers vs fast-channel when available. | Group 3 heads. See cross-condition summary for general guidance. |
| age_class_lstm_test_window_sweep_balanced_accuracy.png | BA pattern expected to mirror AUROC. | [SUPPLEMENTARY] |
| age_class_lstm_calibration_2A_reliability.png | Similar calibration trends as fast-channel. Full-channel may improve calibration slightly at long contexts. | 3×3 grid. |
| age_class_calibration_2B_ece_vs_context.png | Monotonic ECE decrease expected. Compare to fast-channel for magnitude. | Shared across heads. |
| age_class_lstm_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| age_class_lstm_window_position_4A_profiles.png | Mid-night elevation for positives may change with full-channel — additional cardiac/respiratory channels could reinforce or attenuate the positional effect. | Compare 4A profiles to fast-channel. |
| age_class_lstm_window_position_4B_variance.png | Similar variance pattern expected. | Group with 4A. |
| age_class_lstm_subject_consistency_5A_variance.png | Correct/incorrect separation pattern expected to be maintained. | 3×3 grid. |
| age_class_lstm_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| age_class_lstm_subject_consistency_5C_hard_subjects.png | Hard-subject fraction expected to be similar to fast-channel unless full-channel meaningfully changes classification. | [REDESIGN NEEDED] |
| age_class_lstm_cohort_saturation_7B_n.png | Same N as fast-channel. | |
| age_class_lstm_pr_8C_vote_sweep.png | Same structure as fast-channel. | Supplementary. |
| age_class_lstm_kstar_9A_histogram.png | K* distribution expected to be similar to fast-channel. | |
| age_class_lstm_kstar_9B_coverage.png | Not blacklisted. Optional. | |
| age_class_lstm_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| age_class_1B_compute_scaling.png | Compare slope vs fast-channel to assess compute efficiency of multi-modal training for age. | Multi-panel group. |
| age_class_1C_optimal_epoch.png | Non-monotonic pattern expected; same supplementary status as fast-channel. | Supplementary. |
| auroc_test/heatmap_auroc.png | Compare values to fast-channel heatmap for age. | |
| auroc_test/metric_vs_k_auroc.png | Same K-saturation structure. | |
| auroc_test/metric_vs_total_auroc.png | Compare lines to fast-channel. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Compare cost frontier to fast-channel. | |
| auroc_test/marginal_gain_auroc.png | Rapid decay expected. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal. | |

---

### age_class_transformer

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| age_class_transformer_test_window_sweep_auroc.png | Transformer likely best head for age in full-channel. Compare to fast-channel Transformer to assess multi-modal benefit. | Group 3 heads. |
| age_class_transformer_test_window_sweep_balanced_accuracy.png | BA mirrors AUROC. | [SUPPLEMENTARY] |
| age_class_transformer_calibration_2A_reliability.png | Best calibration among heads. | 3×3 grid. |
| age_class_transformer_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| age_class_transformer_window_position_4A_profiles.png | Compare position dependence to fast-channel. | |
| age_class_transformer_window_position_4B_variance.png | Low variance expected. | |
| age_class_transformer_subject_consistency_5A_variance.png | Tightest violins among heads. | |
| age_class_transformer_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| age_class_transformer_subject_consistency_5C_hard_subjects.png | Hard-subject analysis. | [REDESIGN NEEDED] |
| age_class_transformer_cohort_saturation_7B_n.png | Same N. | |
| age_class_transformer_pr_8C_vote_sweep.png | Consistent. | Supplementary. |
| age_class_transformer_kstar_9A_histogram.png | Low K* at long contexts for best head. | |
| age_class_transformer_kstar_9B_coverage.png | Not blacklisted. | |
| age_class_transformer_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Higher values than LSTM. | |
| auroc_test/metric_vs_k_auroc.png | Faster K-saturation. | |
| auroc_test/metric_vs_total_auroc.png | Transformer line highest. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | More efficient than LSTM. | |
| auroc_test/marginal_gain_auroc.png | Rapid decay. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal. | |

---

### age_class_mean_pool

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| age_class_mean_pool_test_window_sweep_auroc.png | MeanPool with full-channel for age: expected degradation vs fast-channel MeanPool, based on the sex_binary pattern. Age classification relies heavily on EEG spectral features that do not scale with additional channel averaging. | Group 3 heads. |
| age_class_mean_pool_test_window_sweep_balanced_accuracy.png | BA mirrors AUROC. | [SUPPLEMENTARY] |
| age_class_mean_pool_calibration_2A_reliability.png | Similar to fast-channel or slightly worse. | 3×3 grid. |
| age_class_mean_pool_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| age_class_mean_pool_window_position_4A_profiles.png | Flat profiles. | |
| age_class_mean_pool_window_position_4B_variance.png | Low variance. | |
| age_class_mean_pool_subject_consistency_5A_variance.png | Similar or weaker separation vs fast-channel MeanPool. | |
| age_class_mean_pool_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| age_class_mean_pool_subject_consistency_5C_hard_subjects.png | Similar hard-subject distribution. | [REDESIGN NEEDED] |
| age_class_mean_pool_cohort_saturation_7B_n.png | Same N. | |
| age_class_mean_pool_pr_8C_vote_sweep.png | Consistent. | Supplementary. |
| age_class_mean_pool_kstar_9A_histogram.png | Similar K* distribution. | |
| age_class_mean_pool_kstar_9B_coverage.png | Not blacklisted. | |
| age_class_mean_pool_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Intermediate values. | |
| auroc_test/metric_vs_k_auroc.png | Same saturation structure. | |
| auroc_test/metric_vs_total_auroc.png | MeanPool line between LSTM and Transformer. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Similar to LSTM. | |
| auroc_test/marginal_gain_auroc.png | Rapid decay. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal. | |

---

### apnea_binary_lstm

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| apnea_binary_lstm_test_window_sweep_auroc.png | Full-channel LSTM for apnea starts at ~70.4% at 30s and rises to ~85.5% at 240m. This is dramatically lower than fast-channel LSTM (~84%→92%). **The full-channel model has significantly degraded apnea prediction despite having more physiological information available, including respiratory effort signals.** The context dependence pattern is preserved (monotonic improvement) but the ceiling is lower by ~6–7 pp. K-saturation pattern likely similar (K=3–5 at long contexts). | KEY FINDING: Full-channel HURTS apnea prediction. Discuss in paper. Possible explanation: the PSG respiratory channels (thoracic/abdominal effort, SpO2) are the AHI ground truth sources, and when included in input, the model faces a harder optimization problem since it must learn to integrate EEG + the very signals used to define the label. Alternatively, full-channel training with more input dimensions requires more data/training epochs. |
| apnea_binary_lstm_test_window_sweep_balanced_accuracy.png | BA shows same degradation pattern vs fast-channel. | [SUPPLEMENTARY] |
| apnea_binary_lstm_calibration_2A_reliability.png | **Calibration at all contexts is worse than fast-channel LSTM — predicted probabilities are more uncertain. The model is less confident about apnea in full-channel mode, consistent with the harder optimization landscape.** | 3×3 grid per task. Note calibration degradation. |
| apnea_binary_calibration_2B_ece_vs_context.png | **ECE improvement is preserved (monotonic) but ECE values are higher than fast-channel at all contexts, reflecting reduced calibration quality.** | Shared across heads. |
| apnea_binary_lstm_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| apnea_binary_lstm_window_position_4A_profiles.png | Second-half elevation for positives may be attenuated in full-channel — if respiratory channels are included, the position-specific REM signal from EEG is diluted. Compare to fast-channel 4A. | Inspect for attenuation of the second-half REM-related elevation. |
| apnea_binary_lstm_window_position_4B_variance.png | Variance may be higher in full-channel due to lower overall confidence. | Group with 4A. |
| apnea_binary_lstm_subject_consistency_5A_variance.png | **Weaker correct/incorrect separation than fast-channel LSTM. The model is less self-consistent for apnea prediction when trained on full channels.** | 3×3 grid. |
| apnea_binary_lstm_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| apnea_binary_lstm_subject_consistency_5C_hard_subjects.png | More never-correct subjects than fast-channel (~8–10% vs ~4%), consistent with degraded performance. | [REDESIGN NEEDED] |
| apnea_binary_lstm_cohort_saturation_7A.png | Cohort-level improvement pattern preserved; SHHS and APPLES both still improve with context. Lower absolute AUROC at all cohorts vs fast-channel. | Group 3 heads. |
| apnea_binary_lstm_cohort_saturation_7B_n.png | Same N. | [LEGEND FIX] |
| apnea_binary_lstm_pr_8A_curves.png | **AUC-PR substantially lower than fast-channel LSTM. The PR curve at 240m is below the fast-channel 80m curve.** | Note the degradation. |
| apnea_binary_pr_8B_aucpr_vs_context.png | Monotonic improvement preserved; ceiling lower. | Shared across heads. |
| apnea_binary_lstm_pr_8C_vote_sweep.png | Similar structure, lower absolute values. | Supplementary. |
| apnea_binary_lstm_kstar_9A_histogram.png | K* distribution is broader than fast-channel LSTM — more windows needed per subject, consistent with lower single-window accuracy. | 3-panel per task. |
| apnea_binary_lstm_kstar_9B_coverage.png | Not blacklisted. | |
| apnea_binary_lstm_1A_uShape.png | [FLAG: CODE CHANGE] Full-channel training may show longer convergence time due to higher input complexity. Rerun with BA. | [FLAG: CODE CHANGE] |
| apnea_binary_1B_compute_scaling.png | **Power-law slope is shallower than fast-channel — additional FLOPs in full-channel do not translate to AUROC gains for apnea as efficiently. The ceiling at max compute (~0.88 for Transformer) is below the fast-channel ceiling (~0.93).** | Strong evidence that channel selection (fast > full) matters for apnea. |
| apnea_binary_1C_optimal_epoch.png | May require more epochs to converge with full-channel input. Non-monotonic pattern expected. | Supplementary. |
| auroc_test/heatmap_auroc.png | **Full-channel heatmap values are substantially lower than fast-channel. The iso-compute advantage of long-L is preserved but at a lower absolute AUROC ceiling.** | |
| auroc_test/metric_vs_k_auroc.png | K-saturation pattern preserved; lower plateau at each L. | |
| auroc_test/metric_vs_total_auroc.png | Full-channel lines lie below fast-channel — direct evidence that more channels hurt apnea prediction at all compute budgets. | KEY COMPARISON FIGURE: overlay fast vs full for apnea. |
| auroc_test/pareto_front_auroc.png | Long-L remains Pareto-dominant. | |
| auroc_test/min_cost_frontier_auroc.png | **Target AUROC above ~0.88 cannot be reached in full-channel at any affordable K for apnea (vs ~0.93 in fast-channel).** | |
| auroc_test/marginal_gain_auroc.png | Rapid decay, same structure. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal but lower ceiling. | |

---

### apnea_binary_transformer

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| apnea_binary_transformer_test_window_sweep_auroc.png | Full-channel Transformer reaches ~88.2% at 240m (vs ~93% fast-channel). At 30s, ~70.1% (vs ~84% fast). **The degradation is most severe at short contexts (14 pp gap at 30s) and narrows somewhat at 240m (5 pp gap).** This suggests full-channel models need longer context to start utilizing multi-modal information effectively, but still cannot match the fast-channel performance. | Group 3 heads. Key finding for paper. |
| apnea_binary_transformer_test_window_sweep_balanced_accuracy.png | BA shows similar degradation. | [SUPPLEMENTARY] |
| apnea_binary_transformer_calibration_2A_reliability.png | Calibration improved vs LSTM but still below fast-channel Transformer calibration at same context. | 3×3 grid. |
| apnea_binary_transformer_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| apnea_binary_transformer_window_position_4A_profiles.png | Check if second-half elevation is attenuated. Transformer's attention may partially recover the positional signal. | Compare to fast-channel. |
| apnea_binary_transformer_window_position_4B_variance.png | Higher variance than fast-channel Transformer. | |
| apnea_binary_transformer_subject_consistency_5A_variance.png | Weaker separation than fast-channel Transformer. | |
| apnea_binary_transformer_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| apnea_binary_transformer_subject_consistency_5C_hard_subjects.png | More hard subjects than fast-channel Transformer. | [REDESIGN NEEDED] |
| apnea_binary_transformer_cohort_saturation_7A.png | Cohort improvement preserved but lower ceiling. | |
| apnea_binary_transformer_cohort_saturation_7B_n.png | Same N. | [LEGEND FIX] |
| apnea_binary_transformer_pr_8A_curves.png | Lower AUC-PR than fast-channel Transformer. | |
| apnea_binary_transformer_pr_8C_vote_sweep.png | Similar structure. | Supplementary. |
| apnea_binary_transformer_kstar_9A_histogram.png | Broader K* than fast-channel Transformer. | |
| apnea_binary_transformer_kstar_9B_coverage.png | Not blacklisted. | |
| apnea_binary_transformer_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Lower ceiling than fast-channel Transformer heatmap. | |
| auroc_test/metric_vs_k_auroc.png | Slower K-saturation than fast-channel Transformer. | |
| auroc_test/metric_vs_total_auroc.png | Full-channel Transformer line below fast-channel at all total context. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Higher cost to reach same target vs fast-channel Transformer. | |
| auroc_test/marginal_gain_auroc.png | Rapid decay. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal. | |

---

### apnea_binary_mean_pool

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| apnea_binary_mean_pool_test_window_sweep_auroc.png | **Full-channel MeanPool for apnea: ~81% at 240m, substantially below fast-channel MeanPool (~91%). The largest MeanPool degradation across tasks. Spectral averaging over 23 heterogeneous channels cannot capture the focused EEG arousal signal that makes fast-channel MeanPool competitive for apnea.** | Largest channel degradation for MeanPool. |
| apnea_binary_mean_pool_test_window_sweep_balanced_accuracy.png | BA shows same degradation. | [SUPPLEMENTARY] |
| apnea_binary_mean_pool_calibration_2A_reliability.png | Lowest calibration quality among heads for full-channel apnea. | 3×3 grid. |
| apnea_binary_mean_pool_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| apnea_binary_mean_pool_window_position_4A_profiles.png | Position profiles flat (by design for MeanPool). No recovery of second-half elevation seen in fast-channel. | |
| apnea_binary_mean_pool_window_position_4B_variance.png | Higher variance than fast-channel MeanPool. | |
| apnea_binary_mean_pool_subject_consistency_5A_variance.png | Weakest separation among heads in full-channel. | |
| apnea_binary_mean_pool_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| apnea_binary_mean_pool_subject_consistency_5C_hard_subjects.png | Most hard subjects among heads in full-channel for apnea. | [REDESIGN NEEDED] |
| apnea_binary_mean_pool_cohort_saturation_7A.png | Cohort patterns preserved at lower AUROC. | |
| apnea_binary_mean_pool_cohort_saturation_7B_n.png | Same N. | [LEGEND FIX] |
| apnea_binary_mean_pool_pr_8A_curves.png | Lowest AUC-PR for full-channel apnea. | |
| apnea_binary_mean_pool_pr_8C_vote_sweep.png | Consistent structure. | Supplementary. |
| apnea_binary_mean_pool_kstar_9A_histogram.png | Broadest K* among full-channel heads for apnea. | |
| apnea_binary_mean_pool_kstar_9B_coverage.png | Not blacklisted. | |
| apnea_binary_mean_pool_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Lowest values in apnea full-channel heatmaps. | |
| auroc_test/metric_vs_k_auroc.png | Slowest K-saturation. | |
| auroc_test/metric_vs_total_auroc.png | MeanPool full-channel line lowest for apnea. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant but low ceiling. | |
| auroc_test/min_cost_frontier_auroc.png | Highest cost to reach modest targets. | |
| auroc_test/marginal_gain_auroc.png | Rapid decay. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal. | |

---

### bmi_binary_lstm

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| bmi_binary_lstm_test_window_sweep_auroc.png | Full-channel LSTM reaches ~78.4% at 240m (vs ~72% fast-channel). A substantial improvement of ~6 pp. At 30s, ~71.7% vs ~66% fast-channel (also better). **Full-channel consistently outperforms fast-channel for BMI — the additional channels (limb EMG, cardiac metrics, respiratory signals) encode body-composition-relevant information that the fast EEG/EOG channels do not.** K-saturation pattern similar to fast-channel. | KEY FINDING: Full-channel benefits BMI. Body-morphology signal is distributed across PSG modalities. |
| bmi_binary_lstm_test_window_sweep_balanced_accuracy.png | BA shows same improvement pattern. | [SUPPLEMENTARY] |
| bmi_binary_lstm_calibration_2A_reliability.png | Calibration improved vs fast-channel, consistent with better absolute AUROC. The model is more confident and better-calibrated when all channels are available for BMI. | 3×3 grid. |
| bmi_binary_calibration_2B_ece_vs_context.png | ECE decreases more in full-channel than fast-channel for BMI — the additional channels enable more confident predictions. | Shared across heads. |
| bmi_binary_lstm_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| bmi_binary_lstm_window_position_4A_profiles.png | Position profiles flat, same as fast-channel. BMI has no circadian structure. | |
| bmi_binary_lstm_window_position_4B_variance.png | Lower variance vs fast-channel for BMI — model is more confident per window with additional channels. | |
| bmi_binary_lstm_subject_consistency_5A_variance.png | Better correct/incorrect separation vs fast-channel LSTM for BMI. Model is more self-consistent with full-channel for BMI prediction. | 3×3 grid. |
| bmi_binary_lstm_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| bmi_binary_lstm_subject_consistency_5C_hard_subjects.png | Fewer hard subjects vs fast-channel LSTM — full-channel improves BMI classification for some previously hard cases. | [REDESIGN NEEDED] |
| bmi_binary_lstm_cohort_saturation_7A.png | Cohort improvement patterns stronger than fast-channel. All cohorts benefit from additional channels for BMI. | Group 3 heads. |
| bmi_binary_lstm_cohort_saturation_7B_n.png | Same N. | [LEGEND FIX] |
| bmi_binary_lstm_pr_8A_curves.png | Better AUC-PR than fast-channel LSTM for BMI. | |
| bmi_binary_pr_8B_aucpr_vs_context.png | Monotonic improvement; better ceiling than fast-channel. | Shared across heads. |
| bmi_binary_lstm_pr_8C_vote_sweep.png | Tighter curves vs fast-channel (more confident). | Supplementary. |
| bmi_binary_lstm_kstar_9A_histogram.png | K* slightly more concentrated vs fast-channel, consistent with improved single-window accuracy. | 3-panel. |
| bmi_binary_lstm_kstar_9B_coverage.png | Not blacklisted. Optional. | |
| bmi_binary_lstm_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| bmi_binary_1B_compute_scaling.png | Steeper power-law slope than fast-channel for BMI — full-channel extracts more per FLOP for body-composition tasks. | |
| bmi_binary_1C_optimal_epoch.png | Pattern may be slightly more monotonic with full-channel (body-composition tasks may benefit from more training when additional channels are included). | Supplementary. |
| auroc_test/heatmap_auroc.png | Substantially higher values than fast-channel for BMI. | |
| auroc_test/metric_vs_k_auroc.png | Higher plateau at each L. | |
| auroc_test/metric_vs_total_auroc.png | Full-channel lines above fast-channel for BMI — direct evidence of multi-modal benefit. | KEY COMPARISON FIGURE for BMI. |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Lower minimum cost to reach targets vs fast-channel. | |
| auroc_test/marginal_gain_auroc.png | Rapid decay, same structure. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal. | |

---

### bmi_binary_transformer

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| bmi_binary_transformer_test_window_sweep_auroc.png | Full-channel Transformer reaches ~80.0% at 240m (vs ~74% fast-channel). Best head for full-channel BMI. +6 pp improvement. At 30s ~71.5% vs ~66% fast. Context dependence strongly preserved with larger context gain vs fast-channel. | Group 3 heads. Best performer for BMI in full-channel. |
| bmi_binary_transformer_test_window_sweep_balanced_accuracy.png | BA shows same improvement. | [SUPPLEMENTARY] |
| bmi_binary_transformer_calibration_2A_reliability.png | Best calibration for full-channel BMI. | 3×3 grid. |
| bmi_binary_transformer_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| bmi_binary_transformer_window_position_4A_profiles.png | Flat, position-independent. | |
| bmi_binary_transformer_window_position_4B_variance.png | Lower variance than fast-channel Transformer for BMI. | |
| bmi_binary_transformer_subject_consistency_5A_variance.png | Better separation vs fast-channel Transformer. | |
| bmi_binary_transformer_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| bmi_binary_transformer_subject_consistency_5C_hard_subjects.png | Fewer hard subjects vs fast-channel Transformer. | [REDESIGN NEEDED] |
| bmi_binary_transformer_cohort_saturation_7A.png | All cohorts benefit. | |
| bmi_binary_transformer_cohort_saturation_7B_n.png | Same N. | [LEGEND FIX] |
| bmi_binary_transformer_pr_8A_curves.png | Best PR curves for full-channel BMI. | |
| bmi_binary_transformer_pr_8C_vote_sweep.png | Consistent. | Supplementary. |
| bmi_binary_transformer_kstar_9A_histogram.png | Concentrated K* at 240m. | |
| bmi_binary_transformer_kstar_9B_coverage.png | Not blacklisted. | |
| bmi_binary_transformer_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Highest values in BMI full-channel. | |
| auroc_test/metric_vs_k_auroc.png | Faster K-saturation than LSTM. | |
| auroc_test/metric_vs_total_auroc.png | Transformer line highest for BMI. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Most efficient path for BMI. | |
| auroc_test/marginal_gain_auroc.png | Rapid decay. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal. | |

---

### bmi_binary_mean_pool

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| bmi_binary_mean_pool_test_window_sweep_auroc.png | Full-channel MeanPool for BMI reaches ~76.5% at 240m (vs ~71% fast-channel). Improves with full-channel, unlike the MeanPool degradation seen in apnea/sleep-efficiency. For BMI, spectral averaging across more modalities captures additional body-composition signal. | MeanPool for BMI improves with full-channel — contrast with apnea where MeanPool degrades. |
| bmi_binary_mean_pool_test_window_sweep_balanced_accuracy.png | BA same pattern. | [SUPPLEMENTARY] |
| bmi_binary_mean_pool_calibration_2A_reliability.png | Better calibration than fast-channel for BMI. | 3×3 grid. |
| bmi_binary_mean_pool_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| bmi_binary_mean_pool_window_position_4A_profiles.png | Flat. | |
| bmi_binary_mean_pool_window_position_4B_variance.png | Lower variance vs fast-channel for BMI. | |
| bmi_binary_mean_pool_subject_consistency_5A_variance.png | Better separation vs fast-channel MeanPool. | |
| bmi_binary_mean_pool_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| bmi_binary_mean_pool_subject_consistency_5C_hard_subjects.png | Fewer hard subjects than fast-channel MeanPool. | [REDESIGN NEEDED] |
| bmi_binary_mean_pool_cohort_saturation_7A.png | Cohort improvement consistent. | |
| bmi_binary_mean_pool_cohort_saturation_7B_n.png | Same N. | [LEGEND FIX] |
| bmi_binary_mean_pool_pr_8A_curves.png | Better PR than fast-channel for BMI. | |
| bmi_binary_mean_pool_pr_8C_vote_sweep.png | Consistent. | Supplementary. |
| bmi_binary_mean_pool_kstar_9A_histogram.png | Concentrated K* at 240m compared to fast-channel. | |
| bmi_binary_mean_pool_kstar_9B_coverage.png | Not blacklisted. | |
| bmi_binary_mean_pool_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Higher values than fast-channel BMI MeanPool. | |
| auroc_test/metric_vs_k_auroc.png | Same structure but higher plateau. | |
| auroc_test/metric_vs_total_auroc.png | Full-channel line above fast-channel for MeanPool BMI. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | More efficient than fast-channel for BMI. | |
| auroc_test/marginal_gain_auroc.png | Rapid decay. | |
| auroc_test/double_tradeoff_auroc.png | Same conclusion. | |

---

### sleep_efficiency_binary_lstm

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleep_efficiency_binary_lstm_test_window_sweep_auroc.png | Full-channel LSTM for sleep efficiency reaches ~77.5% at 240m (vs ~82% fast-channel). A meaningful drop of ~5 pp. At 30s, ~66.7% vs ~69% (similar at short context). **The short-context performance is similar, but the ceiling at long contexts is substantially lower.** Context dependence is preserved (still not saturated at 240m), but the absolute gain is smaller. K-saturation pattern preserved at K=5–8. | KEY FINDING: Full-channel hurts sleep efficiency, consistent with the EEG-dominant pattern seen in apnea. |
| sleep_efficiency_binary_lstm_test_window_sweep_balanced_accuracy.png | BA shows same degradation. | [SUPPLEMENTARY] |
| sleep_efficiency_binary_lstm_calibration_2A_reliability.png | Calibration at 240m is below fast-channel LSTM quality for sleep efficiency. The model is less confident. | 3×3 grid. |
| sleep_efficiency_binary_calibration_2B_ece_vs_context.png | Monotonic ECE decrease preserved but ECE at 240m is higher than fast-channel. | Shared. |
| sleep_efficiency_binary_lstm_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sleep_efficiency_binary_lstm_window_position_4A_profiles.png | **The early-night elevation for positive subjects (seen in fast-channel) may be attenuated in full-channel — additional channels may dilute the EEG-based early-night signal.** Compare to fast-channel 4A to assess attenuation. | Important comparison for mechanistic interpretation. |
| sleep_efficiency_binary_lstm_window_position_4B_variance.png | Higher variance or different pattern vs fast-channel. | Group with 4A. |
| sleep_efficiency_binary_lstm_subject_consistency_5A_variance.png | Weaker correct/incorrect separation than fast-channel LSTM for sleep efficiency. | 3×3 grid. |
| sleep_efficiency_binary_lstm_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sleep_efficiency_binary_lstm_subject_consistency_5C_hard_subjects.png | More hard subjects than fast-channel, consistent with degraded performance. | [REDESIGN NEEDED] |
| sleep_efficiency_binary_lstm_cohort_saturation_7A.png | Context sensitivity preserved across cohorts but lower ceiling. Still no saturation at 240m. | Group 3 heads. |
| sleep_efficiency_binary_lstm_cohort_saturation_7B_n.png | Same N. | [LEGEND FIX] |
| sleep_efficiency_binary_lstm_pr_8A_curves.png | AUC-PR improvement preserved; lower ceiling than fast-channel. | |
| sleep_efficiency_binary_pr_8B_aucpr_vs_context.png | Monotonic improvement; lower ceiling. | Shared. |
| sleep_efficiency_binary_lstm_pr_8C_vote_sweep.png | Consistent. | Supplementary. |
| sleep_efficiency_binary_lstm_kstar_9A_histogram.png | Broader K* than fast-channel; more windows needed per subject. | |
| sleep_efficiency_binary_lstm_kstar_9B_coverage.png | Not blacklisted. | |
| sleep_efficiency_binary_lstm_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| sleep_efficiency_binary_1B_compute_scaling.png | Power-law slope shallower than fast-channel for sleep efficiency. | |
| sleep_efficiency_binary_1C_optimal_epoch.png | Check if monotonic relationship is preserved or disrupted by full-channel input. | Supplementary. |
| auroc_test/heatmap_auroc.png | Lower values than fast-channel but unsaturated pattern preserved. | |
| auroc_test/metric_vs_k_auroc.png | K-saturation pattern same; lower plateau. | |
| auroc_test/metric_vs_total_auroc.png | Full-channel lines below fast-channel for sleep efficiency. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Higher cost to reach target AUROC due to lower ceiling. | |
| auroc_test/marginal_gain_auroc.png | Rapid decay but slightly slower (consistent with slower saturation). | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal. | |

---

### sleep_efficiency_binary_transformer

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleep_efficiency_binary_transformer_test_window_sweep_auroc.png | Full-channel Transformer reaches ~79.2% at 240m (vs ~85% fast-channel). The best full-channel head for sleep efficiency, but still 5.8 pp below fast-channel Transformer. Still no saturation at 240m — L* remains >240m in full-channel as well, consistent with the task's high context sensitivity. | Group 3 heads. KEY: no saturation persists in full-channel. |
| sleep_efficiency_binary_transformer_test_window_sweep_balanced_accuracy.png | BA same pattern. | [SUPPLEMENTARY] |
| sleep_efficiency_binary_transformer_calibration_2A_reliability.png | Best calibration among full-channel heads for sleep efficiency, but below fast-channel Transformer. | 3×3 grid. |
| sleep_efficiency_binary_transformer_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sleep_efficiency_binary_transformer_window_position_4A_profiles.png | Check attenuation of early-night elevation vs fast-channel Transformer. Transformer's attention may better recover the position-dependent signal. | Compare to fast-channel 4A. |
| sleep_efficiency_binary_transformer_window_position_4B_variance.png | Lower variance than LSTM. | |
| sleep_efficiency_binary_transformer_subject_consistency_5A_variance.png | Better separation than full-channel LSTM but below fast-channel Transformer. | |
| sleep_efficiency_binary_transformer_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sleep_efficiency_binary_transformer_subject_consistency_5C_hard_subjects.png | Fewer hard subjects than LSTM in full-channel. | [REDESIGN NEEDED] |
| sleep_efficiency_binary_transformer_cohort_saturation_7A.png | Context sensitivity preserved; lower ceiling but still unsaturated. | Group 3 heads. |
| sleep_efficiency_binary_transformer_cohort_saturation_7B_n.png | Same N. | [LEGEND FIX] |
| sleep_efficiency_binary_transformer_pr_8A_curves.png | Best PR curves in full-channel for sleep efficiency. | |
| sleep_efficiency_binary_transformer_pr_8C_vote_sweep.png | Consistent. | Supplementary. |
| sleep_efficiency_binary_transformer_kstar_9A_histogram.png | Slightly more concentrated K* vs LSTM in full-channel. | |
| sleep_efficiency_binary_transformer_kstar_9B_coverage.png | Not blacklisted. | |
| sleep_efficiency_binary_transformer_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Highest values in sleep efficiency full-channel; still unsaturated. | |
| auroc_test/metric_vs_k_auroc.png | Slower K-saturation than fast-channel Transformer. | |
| auroc_test/metric_vs_total_auroc.png | Full-channel Transformer line below fast-channel. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Higher cost to reach target. | |
| auroc_test/marginal_gain_auroc.png | Rapid decay. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal. | |

---

### sleep_efficiency_binary_mean_pool

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleep_efficiency_binary_mean_pool_test_window_sweep_auroc.png | Full-channel MeanPool reaches ~73.8% at 240m (vs ~83% fast-channel). Largest MeanPool degradation of all tasks (-9 pp). Sleep efficiency's EEG-based signal is completely diluted by multi-channel averaging. | Strongest channel-degradation case for MeanPool. |
| sleep_efficiency_binary_mean_pool_test_window_sweep_balanced_accuracy.png | BA same degradation. | [SUPPLEMENTARY] |
| sleep_efficiency_binary_mean_pool_calibration_2A_reliability.png | Poor calibration, worse than fast-channel. | 3×3 grid. |
| sleep_efficiency_binary_mean_pool_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sleep_efficiency_binary_mean_pool_window_position_4A_profiles.png | Early-night elevation likely absent in full-channel MeanPool. | |
| sleep_efficiency_binary_mean_pool_window_position_4B_variance.png | Higher variance. | |
| sleep_efficiency_binary_mean_pool_subject_consistency_5A_variance.png | Weakest separation among heads in full-channel for sleep efficiency. | |
| sleep_efficiency_binary_mean_pool_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| sleep_efficiency_binary_mean_pool_subject_consistency_5C_hard_subjects.png | Most hard subjects in full-channel. | [REDESIGN NEEDED] |
| sleep_efficiency_binary_mean_pool_cohort_saturation_7A.png | Lower ceiling, context sensitivity preserved. | |
| sleep_efficiency_binary_mean_pool_cohort_saturation_7B_n.png | Same N. | [LEGEND FIX] |
| sleep_efficiency_binary_mean_pool_pr_8A_curves.png | Lowest AUC-PR of full-channel heads for sleep efficiency. | |
| sleep_efficiency_binary_mean_pool_pr_8C_vote_sweep.png | Consistent. | Supplementary. |
| sleep_efficiency_binary_mean_pool_kstar_9A_histogram.png | Broadest K* among heads. | |
| sleep_efficiency_binary_mean_pool_kstar_9B_coverage.png | Not blacklisted. | |
| sleep_efficiency_binary_mean_pool_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Lowest values in sleep efficiency full-channel. | |
| auroc_test/metric_vs_k_auroc.png | Slowest K-saturation. | |
| auroc_test/metric_vs_total_auroc.png | Full-channel MeanPool line lowest for sleep efficiency. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Highest cost to reach targets. | |
| auroc_test/marginal_gain_auroc.png | Same decay. | |
| auroc_test/double_tradeoff_auroc.png | Same conclusion. | |

---

### depression_extreme_binary_lstm

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| depression_extreme_binary_lstm_test_window_sweep_auroc.png | **Full-channel LSTM for depression: similar erratic, non-monotonic pattern as fast-channel. AUROC ranges ~72–73% with no systematic context dependence. The performance is marginally higher than fast-channel (~63–69%) but the non-monotonic character is unchanged.** APPLES-only cohort limits N and generalizability. | **[UNJUSTIFIABLE]** Same classification as fast-channel. |
| depression_extreme_binary_lstm_test_window_sweep_balanced_accuracy.png | **Erratic BA pattern.** | [SUPPLEMENTARY] **[UNJUSTIFIABLE]** |
| depression_extreme_binary_lstm_calibration_2A_reliability.png | **Poorly calibrated. Marginally better than fast-channel but still far from diagonal.** | 3×3 grid. |
| depression_extreme_binary_calibration_2B_ece_vs_context.png | **High ECE, non-monotonic, same issues as fast-channel.** | Shared. |
| depression_extreme_binary_lstm_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| depression_extreme_binary_lstm_window_position_4A_profiles.png | **Near-baseline profiles. No interpretable position structure.** | |
| depression_extreme_binary_lstm_window_position_4B_variance.png | **High variance. No reduction with context.** | |
| depression_extreme_binary_lstm_subject_consistency_5A_variance.png | **Weak correct/incorrect separation.** | |
| depression_extreme_binary_lstm_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| depression_extreme_binary_lstm_subject_consistency_5C_hard_subjects.png | **Similar poor distribution to fast-channel.** | [REDESIGN NEEDED] **[UNJUSTIFIABLE]** |
| depression_extreme_binary_lstm_cohort_saturation_7A.png | **APPLES-only. No cohort comparison. Slight absolute improvement vs fast-channel.** | |
| depression_extreme_binary_lstm_cohort_saturation_7B_n.png | Small N. | |
| depression_extreme_binary_lstm_pr_8A_curves.png | **Slightly better AUC-PR than fast-channel LSTM but still near baseline.** | |
| depression_extreme_binary_pr_8B_aucpr_vs_context.png | **Non-monotonic improvement.** | |
| depression_extreme_binary_lstm_pr_8C_vote_sweep.png | **Poor.** | Supplementary. |
| depression_extreme_binary_lstm_kstar_9A_histogram.png | **Broad K* at all contexts.** | |
| depression_extreme_binary_lstm_kstar_9B_coverage.png | Not blacklisted. Questionable. | |
| depression_extreme_binary_lstm_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| depression_extreme_binary_1B_compute_scaling.png | **Flat or near-flat scaling. Ceiling higher than fast-channel (~0.73 vs ~0.68) but still very low.** | **[UNJUSTIFIABLE]** |
| depression_extreme_binary_1C_optimal_epoch.png | **Erratic. Full-channel does not stabilize training.** | Exclude from paper. |
| auroc_test/heatmap_auroc.png | **Slightly higher values than fast-channel but non-structured pattern.** | **[UNJUSTIFIABLE]** |
| auroc_test/metric_vs_k_auroc.png | **Non-monotonic.** | **[UNJUSTIFIABLE]** |
| auroc_test/metric_vs_total_auroc.png | **No clear convergence.** | **[UNJUSTIFIABLE]** |
| auroc_test/pareto_front_auroc.png | **Not meaningful.** | **[UNJUSTIFIABLE]** |
| auroc_test/min_cost_frontier_auroc.png | **Low ceiling.** | **[UNJUSTIFIABLE]** |
| auroc_test/marginal_gain_auroc.png | **Erratic.** | **[UNJUSTIFIABLE]** |
| auroc_test/double_tradeoff_auroc.png | **No structure.** | **[UNJUSTIFIABLE]** |

---

### depression_extreme_binary_transformer

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| depression_extreme_binary_transformer_test_window_sweep_auroc.png | Full-channel Transformer shows the most promising result for depression: AUROC rises from ~72.3% at 30s to ~74.5% at 80m before dropping to ~72.9% at 240m. **While higher than fast-channel Transformer (which peaked around ~70%), the non-monotonic collapse from 80m to 240m is unexpected and erratic.** This is the best head for full-channel depression but still unreliable. | **[UNJUSTIFIABLE at 240m]** Best available head for depression; include as the "best case" result while noting instability. |
| depression_extreme_binary_transformer_test_window_sweep_balanced_accuracy.png | **BA mirrors AUROC non-monotonic pattern.** | [SUPPLEMENTARY] **[UNJUSTIFIABLE]** |
| depression_extreme_binary_transformer_calibration_2A_reliability.png | **Better calibration than LSTM in full-channel but still erratic.** | 3×3 grid. |
| depression_extreme_binary_transformer_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| depression_extreme_binary_transformer_window_position_4A_profiles.png | **Near-baseline position profiles.** | |
| depression_extreme_binary_transformer_window_position_4B_variance.png | **Variable.** | |
| depression_extreme_binary_transformer_subject_consistency_5A_variance.png | **Slightly better separation than LSTM but still weak.** | |
| depression_extreme_binary_transformer_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| depression_extreme_binary_transformer_subject_consistency_5C_hard_subjects.png | **Moderate improvement over fast-channel Transformer but pattern non-systematic.** | [REDESIGN NEEDED] **[UNJUSTIFIABLE]** |
| depression_extreme_binary_transformer_cohort_saturation_7A.png | **APPLES-only.** | |
| depression_extreme_binary_transformer_cohort_saturation_7B_n.png | Small N. | |
| depression_extreme_binary_transformer_pr_8A_curves.png | **Best PR for depression full-channel at 80m; drops at 240m.** | |
| depression_extreme_binary_transformer_pr_8C_vote_sweep.png | **Inconsistent.** | Supplementary. |
| depression_extreme_binary_transformer_kstar_9A_histogram.png | **Broad K* at all contexts.** | |
| depression_extreme_binary_transformer_kstar_9B_coverage.png | Not blacklisted. Questionable. | |
| depression_extreme_binary_transformer_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | **Non-monotonic structure, peak around 80m context.** | **[UNJUSTIFIABLE]** |
| auroc_test/metric_vs_k_auroc.png | **Non-monotonic across contexts.** | **[UNJUSTIFIABLE]** |
| auroc_test/metric_vs_total_auroc.png | **Lines cross — no consistent winner.** | **[UNJUSTIFIABLE]** |
| auroc_test/pareto_front_auroc.png | **L=80m may appear Pareto-optimal due to the non-monotonic pattern — misleading.** | **[UNJUSTIFIABLE]** |
| auroc_test/min_cost_frontier_auroc.png | **Target above 0.74 unreachable.** | **[UNJUSTIFIABLE]** |
| auroc_test/marginal_gain_auroc.png | **Non-monotonic.** | **[UNJUSTIFIABLE]** |
| auroc_test/double_tradeoff_auroc.png | **No clear structure.** | **[UNJUSTIFIABLE]** |

---

### depression_extreme_binary_mean_pool

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| depression_extreme_binary_mean_pool_test_window_sweep_auroc.png | **Full-channel MeanPool for depression: AUROC ranges 71–74% with erratic context dependence. Not clearly better or worse than fast-channel MeanPool. The non-monotonic pattern persists.** | **[UNJUSTIFIABLE]** |
| depression_extreme_binary_mean_pool_test_window_sweep_balanced_accuracy.png | **Erratic BA.** | [SUPPLEMENTARY] **[UNJUSTIFIABLE]** |
| depression_extreme_binary_mean_pool_calibration_2A_reliability.png | **Poor calibration.** | 3×3 grid. |
| depression_extreme_binary_mean_pool_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| depression_extreme_binary_mean_pool_window_position_4A_profiles.png | **Near-baseline.** | |
| depression_extreme_binary_mean_pool_window_position_4B_variance.png | **High variance.** | |
| depression_extreme_binary_mean_pool_subject_consistency_5A_variance.png | **Weak separation.** | |
| depression_extreme_binary_mean_pool_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| depression_extreme_binary_mean_pool_subject_consistency_5C_hard_subjects.png | **Similar poor distribution.** | [REDESIGN NEEDED] **[UNJUSTIFIABLE]** |
| depression_extreme_binary_mean_pool_cohort_saturation_7A.png | **APPLES-only.** | |
| depression_extreme_binary_mean_pool_cohort_saturation_7B_n.png | Small N. | |
| depression_extreme_binary_mean_pool_pr_8A_curves.png | **Near baseline.** | |
| depression_extreme_binary_mean_pool_pr_8C_vote_sweep.png | **Poor.** | Supplementary. |
| depression_extreme_binary_mean_pool_kstar_9A_histogram.png | **Broad K*.** | |
| depression_extreme_binary_mean_pool_kstar_9B_coverage.png | Not blacklisted. Questionable. | |
| depression_extreme_binary_mean_pool_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | **Low values, non-structured.** | **[UNJUSTIFIABLE]** |
| auroc_test/metric_vs_k_auroc.png | **Non-monotonic.** | **[UNJUSTIFIABLE]** |
| auroc_test/metric_vs_total_auroc.png | **No convergence.** | **[UNJUSTIFIABLE]** |
| auroc_test/pareto_front_auroc.png | **Not meaningful.** | **[UNJUSTIFIABLE]** |
| auroc_test/min_cost_frontier_auroc.png | **Low ceiling.** | **[UNJUSTIFIABLE]** |
| auroc_test/marginal_gain_auroc.png | **Erratic.** | **[UNJUSTIFIABLE]** |
| auroc_test/double_tradeoff_auroc.png | **No structure.** | **[UNJUSTIFIABLE]** |

---

### osa_binary_apples_postqc_lstm

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| osa_binary_apples_postqc_lstm_test_window_sweep_auroc.png | **Full-channel LSTM for OSA shows a new failure mode: non-monotonic AUROC across context lengths.** At 40m the model reaches ~70.6%, then drops sharply to ~66.5% at 80m before recovering to ~71.0% at 240m. This oscillation suggests the LSTM cannot stably exploit long-context information in full-channel mode for OSA, making it unreliable. **This is an unjustifiable result — the drop-recover pattern at 80m is inconsistent with the hypothesis and likely reflects optimization instability.** | **[UNJUSTIFIABLE for LSTM full-channel]** Strong reason to not use LSTM for OSA in full-channel setting. |
| osa_binary_apples_postqc_lstm_test_window_sweep_balanced_accuracy.png | **Non-monotonic BA mirrors AUROC instability.** | [SUPPLEMENTARY] **[UNJUSTIFIABLE]** |
| osa_binary_apples_postqc_lstm_calibration_2A_reliability.png | **Poorly calibrated with oscillating quality across contexts, consistent with the unstable AUROC pattern.** | 3×3 grid. |
| osa_binary_apples_postqc_calibration_2B_ece_vs_context.png | **Non-monotonic ECE for LSTM; Transformer and MeanPool may show monotonic ECE improvement.** | Shared across heads. |
| osa_binary_apples_postqc_lstm_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| osa_binary_apples_postqc_lstm_window_position_4A_profiles.png | **Position profiles may show erratic structure due to LSTM instability in full-channel.** | |
| osa_binary_apples_postqc_lstm_window_position_4B_variance.png | **Likely higher variance than fast-channel LSTM.** | |
| osa_binary_apples_postqc_lstm_subject_consistency_5A_variance.png | **Weak correct/incorrect separation due to unstable predictions.** | |
| osa_binary_apples_postqc_lstm_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| osa_binary_apples_postqc_lstm_subject_consistency_5C_hard_subjects.png | **Many hard subjects due to LSTM instability.** | [REDESIGN NEEDED] **[UNJUSTIFIABLE]** |
| osa_binary_apples_postqc_lstm_cohort_saturation_7A.png | **APPLES-only. Oscillating AUROC visible at cohort level.** | |
| osa_binary_apples_postqc_lstm_cohort_saturation_7B_n.png | Single cohort. | Legend fix. |
| osa_binary_apples_postqc_lstm_pr_8A_curves.png | **Oscillating PR quality across contexts.** | |
| osa_binary_apples_postqc_pr_8B_aucpr_vs_context.png | **Non-monotonic for LSTM; monotonic for Transformer/MeanPool.** | |
| osa_binary_apples_postqc_lstm_pr_8C_vote_sweep.png | **Inconsistent.** | Supplementary. |
| osa_binary_apples_postqc_lstm_kstar_9A_histogram.png | **Broad, inconsistent K* distribution.** | |
| osa_binary_apples_postqc_lstm_kstar_9B_coverage.png | Not blacklisted. Questionable. | |
| osa_binary_apples_postqc_lstm_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| osa_binary_apples_postqc_1B_compute_scaling.png | **LSTM scaling curve non-monotonic in full-channel — further evidence of optimization instability.** | |
| osa_binary_apples_postqc_1C_optimal_epoch.png | **Check LSTM optimal epoch at 80m — early stopping at a bad point may explain the 80m dip.** | Key diagnostic. |
| auroc_test/heatmap_auroc.png | **Non-monotonic pattern visible; dip at 80m context.** | **[UNJUSTIFIABLE for LSTM]** |
| auroc_test/metric_vs_k_auroc.png | **Non-monotonic lines for LSTM.** | **[UNJUSTIFIABLE]** |
| auroc_test/metric_vs_total_auroc.png | **LSTM lines cross — no consistent ordering.** | **[UNJUSTIFIABLE]** |
| auroc_test/pareto_front_auroc.png | **LSTM Pareto front unreliable.** | **[UNJUSTIFIABLE]** |
| auroc_test/min_cost_frontier_auroc.png | **LSTM cost frontier meaningless given instability.** | **[UNJUSTIFIABLE]** |
| auroc_test/marginal_gain_auroc.png | **Non-monotonic marginal gain for LSTM.** | **[UNJUSTIFIABLE]** |
| auroc_test/double_tradeoff_auroc.png | **No structure for LSTM.** | **[UNJUSTIFIABLE]** |

---

### osa_binary_apples_postqc_transformer

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| osa_binary_apples_postqc_transformer_test_window_sweep_auroc.png | Full-channel Transformer for OSA reaches ~80.8% at 240m (vs ~85% fast-channel). Lower ceiling but monotonic improvement preserved. Context dependence persists. MeanPool at 240m (~78.9%) is very competitive with Transformer — smaller head gap than fast-channel. K-saturation at K=3–5 for long contexts. | Note that in full-channel, the Transformer-MeanPool gap narrows for OSA (vs fast-channel where Transformer clearly leads). |
| osa_binary_apples_postqc_transformer_test_window_sweep_balanced_accuracy.png | BA mirrors AUROC. | [SUPPLEMENTARY] |
| osa_binary_apples_postqc_transformer_calibration_2A_reliability.png | Calibration improves monotonically for Transformer. | 3×3 grid. |
| osa_binary_apples_postqc_transformer_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| osa_binary_apples_postqc_transformer_window_position_4A_profiles.png | Second-half elevation for positives may be attenuated vs fast-channel. | |
| osa_binary_apples_postqc_transformer_window_position_4B_variance.png | Lower variance than LSTM. Stable predictions. | |
| osa_binary_apples_postqc_transformer_subject_consistency_5A_variance.png | Good correct/incorrect separation for Transformer. | |
| osa_binary_apples_postqc_transformer_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| osa_binary_apples_postqc_transformer_subject_consistency_5C_hard_subjects.png | Fewer hard subjects than LSTM. | [REDESIGN NEEDED] |
| osa_binary_apples_postqc_transformer_cohort_saturation_7A.png | APPLES-only. Monotonic improvement. | |
| osa_binary_apples_postqc_transformer_cohort_saturation_7B_n.png | Single cohort N. | Legend fix. |
| osa_binary_apples_postqc_transformer_pr_8A_curves.png | Monotonic PR improvement; ceiling at 240m. | |
| osa_binary_apples_postqc_transformer_pr_8C_vote_sweep.png | Consistent. | Supplementary. |
| osa_binary_apples_postqc_transformer_kstar_9A_histogram.png | Concentrated K* at 240m. | |
| osa_binary_apples_postqc_transformer_kstar_9B_coverage.png | Not blacklisted. | |
| osa_binary_apples_postqc_transformer_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Monotonic heatmap; lower ceiling than fast-channel. | |
| auroc_test/metric_vs_k_auroc.png | K-saturation at K=3–5. | |
| auroc_test/metric_vs_total_auroc.png | Transformer line continues rising; below fast-channel. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Target 80% reachable at L=240m+K=3. | |
| auroc_test/marginal_gain_auroc.png | Standard diminishing returns. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal. | |

---

### osa_binary_apples_postqc_mean_pool

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| osa_binary_apples_postqc_mean_pool_test_window_sweep_auroc.png | Full-channel MeanPool for OSA reaches ~78.9% at 240m (vs ~83% fast-channel). Lower than fast-channel but competitive with Transformer in full-channel. **MeanPool in full-channel for OSA is more competitive with Transformer than in fast-channel — this aligns with the frequency-domain nature of respiratory signals (which are informative for OSA) being well-captured by spectral averaging across many channels.** | Interesting narrowing of Transformer-MeanPool gap in full-channel for OSA. |
| osa_binary_apples_postqc_mean_pool_test_window_sweep_balanced_accuracy.png | BA mirrors AUROC. | [SUPPLEMENTARY] |
| osa_binary_apples_postqc_mean_pool_calibration_2A_reliability.png | Good calibration for MeanPool in full-channel. | 3×3 grid. |
| osa_binary_apples_postqc_mean_pool_calibration_2C_ece_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| osa_binary_apples_postqc_mean_pool_window_position_4A_profiles.png | Flat profiles (by design for MeanPool). | |
| osa_binary_apples_postqc_mean_pool_window_position_4B_variance.png | Low variance, uniform. | |
| osa_binary_apples_postqc_mean_pool_subject_consistency_5A_variance.png | Good separation, competitive with Transformer for OSA full-channel. | |
| osa_binary_apples_postqc_mean_pool_subject_consistency_5B_variance_vs_k.png | [BLACKLIST] | [BLACKLIST] |
| osa_binary_apples_postqc_mean_pool_subject_consistency_5C_hard_subjects.png | Similar to Transformer. | [REDESIGN NEEDED] |
| osa_binary_apples_postqc_mean_pool_cohort_saturation_7A.png | APPLES-only. Monotonic improvement. | |
| osa_binary_apples_postqc_mean_pool_cohort_saturation_7B_n.png | Same N. | Legend fix. |
| osa_binary_apples_postqc_mean_pool_pr_8A_curves.png | Strong PR improvement. | |
| osa_binary_apples_postqc_mean_pool_pr_8C_vote_sweep.png | Consistent. | Supplementary. |
| osa_binary_apples_postqc_mean_pool_kstar_9A_histogram.png | Concentrated K* at 240m, similar to Transformer. | |
| osa_binary_apples_postqc_mean_pool_kstar_9B_coverage.png | Not blacklisted. | |
| osa_binary_apples_postqc_mean_pool_1A_uShape.png | [FLAG: CODE CHANGE] | [FLAG: CODE CHANGE] |
| auroc_test/heatmap_auroc.png | Values close to Transformer. Monotonic structure. | |
| auroc_test/metric_vs_k_auroc.png | Similar K-saturation to Transformer. | |
| auroc_test/metric_vs_total_auroc.png | MeanPool close to Transformer for OSA full-channel. | |
| auroc_test/pareto_front_auroc.png | Long-L dominant. | |
| auroc_test/min_cost_frontier_auroc.png | Efficient: target ~79% reachable at L=240m+small K. | |
| auroc_test/marginal_gain_auroc.png | Standard decay. | |
| auroc_test/double_tradeoff_auroc.png | Long-L + small-K optimal. | |

---

### ⚠️ DROPPED — cvd_binary_lstm

> Task removed from paper. Figures retained for reference.

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| cvd_binary_lstm_test_window_sweep_auroc.png | ⚠️ DROPPED. Same near-flat performance as fast-channel. Full-channel does not improve CVD prediction. | Excluded. |
| cvd_binary_lstm_calibration_2A_reliability.png | ⚠️ DROPPED. | Excluded. |

---

### ⚠️ DROPPED — cvd_binary_transformer

> Task removed from paper. Figures retained for reference.

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| cvd_binary_transformer_test_window_sweep_auroc.png | ⚠️ DROPPED. | Excluded. |

---

## Table 2 — Across-Task Figures (Full Channel)

### Saturation Curves

> **Decision**: AUROC primary, BA supplementary. Same as v3.

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| saturation/saturation_sex_binary_auroc_test.png | Full-channel sex_binary: LSTM 75%→87%, Transformer 75%→90.6%, MeanPool 72%→82%. Transformer leads, all heads show strong monotonic improvement. No saturation at 240m. **Transformer full-channel outperforms fast-channel at long contexts (+1.6 pp at 240m), while MeanPool lags behind fast-channel (-4 pp).** Body-composition channels (limb EMG, cardiac) benefit Transformer and LSTM but not MeanPool for sex classification. | GROUP all task saturation curves into 2×4 panel. Overlay fast-channel (dashed) and full-channel (solid) for head-by-head comparison in paper. |
| saturation/saturation_apnea_binary_auroc_test.png | **Full-channel apnea: LSTM 70.4%→85.5%, Transformer 70.1%→88.2%, MeanPool 67.8%→81.0%. All heads substantially WORSE than fast-channel (fast: ~84%→93%). The gap is 5–10 pp at all context lengths. Context dependence is preserved (monotonic improvement) but the ceiling is lower. MeanPool is most degraded (−10 pp at 240m vs fast-channel). This strongly implies that apnea is an EEG-dominant task; adding respiratory channels does not help and may hurt via optimization complexity.** | KEY FINDING: Full-channel hurts apnea. Discuss modality interaction in paper. |
| saturation/saturation_bmi_binary_auroc_test.png | Full-channel BMI: LSTM 71.7%→78.4%, Transformer 71.5%→80.0%, MeanPool 69.0%→76.5%. **All heads BETTER than fast-channel (+6 pp at 240m for Transformer). BMI classification benefits from multi-modal PSG — body composition is reflected in respiratory mechanics, cardiac rhythm, limb EMG, and sleep architecture simultaneously, and full-channel captures all these dimensions.** Context sensitivity is stronger in full-channel (curve rises more steeply). | KEY FINDING: Full-channel benefits BMI. Strongest reversal of the fast-channel result. |
| saturation/saturation_sleep_efficiency_binary_auroc_test.png | Full-channel sleep efficiency: LSTM 66.7%→77.5%, Transformer 66.5%→79.2%, MeanPool 66.0%→73.8%. **All heads WORSE than fast-channel (~70%→85% Transformer). The early-night position dependence may be attenuated. Still no saturation at 240m (consistent with fast-channel), but the ceiling is ~6 pp lower.** L* remains >240m. | Context sensitivity preserved but ceiling lowered. EEG-dominant task. |
| saturation/saturation_depression_extreme_binary_auroc_test.png | Full-channel depression: LSTM ~72.2–73.0% (non-monotonic), Transformer 72.3%→74.5% at 80m then drops to 72.9% at 240m, MeanPool oscillating 71–73.6%. **Slightly higher floor than fast-channel (~72% vs ~63%) but non-monotonic character preserved.** Transformer shows the best result at 80m (74.5%) before collapsing. APPLES-only. | **[UNJUSTIFIABLE]** Slight improvement vs fast-channel in absolute AUROC but pattern remains non-monotonic and unreliable. |
| saturation/saturation_osa_binary_apples_postqc_auroc_test.png | Full-channel OSA: LSTM non-monotonic (64.4%→70.6% at 40m, drops to 66.5% at 80m, recovers to 71.0% at 240m), Transformer 65.1%→80.8%, MeanPool 64.1%→78.9%. **LSTM is actively harmful in full-channel for OSA (non-monotonic/unstable). Transformer and MeanPool show monotonic improvement but at a lower ceiling than fast-channel (~80–81% vs ~85–83%).** The narrowing of the Transformer-MeanPool gap in full-channel is notable — respiratory channels captured by spectral averaging help MeanPool for OSA. | KEY FINDING: LSTM failure is exacerbated in full-channel for OSA. Discuss modality-architecture interaction. |
| saturation/saturation_cvd_binary_auroc_test.png | ⚠️ DROPPED. | Excluded. |
| saturation/saturation_sleepiness_binary_auroc_test.png | ⚠️ DROPPED. | Excluded. |
| saturation/saturation_psqi_binary_auroc_test.png | ⚠️ DROPPED. | Excluded. |

---

### Task Comparison

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| task_comparison/task_comparison_6A_scatter.png | Full-channel task scatter (difficulty vs context sensitivity). Pattern shifts: BMI moves upward (higher AUROC at short context in full-channel, higher gain), while apnea and sleep_efficiency shift downward (lower absolute AUROC). OSA shifts: Transformer still high gain, but LSTM is removed from consideration due to instability. Depression remains in the low-gain, low-difficulty quadrant. | [CODE FIX] Regenerate without dropped tasks. Overlay fast vs full-channel as two marker types (e.g., filled=full, open=fast). |
| task_comparison/task_comparison_6B_bars.png | Full-channel grouped bars show the same task ranking as fast-channel, but with BMI notably higher and apnea/sleep_efficiency lower. The channel-dependent performance differences are most visible in this cross-task view. | [CODE FIX] Regenerate. Consider side-by-side fast vs full panels for this figure. |
| task_comparison/task_comparison_6C_lstar.png | L* values in full-channel: sleep_efficiency L*>240m (same as fast-channel); apnea L* may increase slightly (harder to saturate when absolute AUROC is lower); BMI L* shortens (higher ceiling reached earlier). Depression L* undefined. | [CODE FIX] Regenerate. Show both fast-channel and full-channel L* side by side per task. |

---

### Scaling Laws (Full Channel)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| scaling_laws/sex_binary_1B_compute_scaling.png | Full-channel power-law for sex: Transformer slope is steeper vs fast-channel, reflecting better compute efficiency with multi-modal data for body-based tasks. LSTM and MeanPool also improve vs fast-channel but MeanPool less so. | Group all tasks in multi-panel figure. Overlay fast vs full-channel as two line styles. |
| scaling_laws/sex_binary_1C_optimal_epoch.png | Non-monotonic pattern expected; full-channel may require more epochs. Supplementary. | Supplementary. |
| scaling_laws/age_class_1B_compute_scaling.png | Full-channel impact on age scaling uncertain without direct comparison data. Monitor vs fast-channel. | Multi-panel. |
| scaling_laws/age_class_1C_optimal_epoch.png | Non-monotonic; supplementary. | Supplementary. |
| scaling_laws/apnea_binary_1B_compute_scaling.png | **Shallower power-law slope than fast-channel for apnea — full-channel is less compute-efficient. Adding compute does not recover the performance lost from including non-EEG channels.** | Note as evidence against full-channel for apnea. |
| scaling_laws/apnea_binary_1C_optimal_epoch.png | May require more epochs in full-channel due to added complexity. | Supplementary. |
| scaling_laws/bmi_binary_1B_compute_scaling.png | **Steeper power-law slope than fast-channel for BMI — full-channel is more compute-efficient. More FLOPs translate to larger AUROC gains when multi-modal body-composition signals are available.** | Key contrast: BMI and apnea show opposite channel × compute interactions. |
| scaling_laws/bmi_binary_1C_optimal_epoch.png | May show slightly more monotonic pattern if full-channel training is more stable for BMI. | Supplementary. |
| scaling_laws/sleep_efficiency_binary_1B_compute_scaling.png | **Shallower slope than fast-channel. Full-channel hurts compute efficiency for sleep efficiency.** | |
| scaling_laws/sleep_efficiency_binary_1C_optimal_epoch.png | Check for monotonicity — was more systematic in fast-channel. | Supplementary/key diagnostic. |
| scaling_laws/depression_extreme_binary_1B_compute_scaling.png | **Slightly steeper than fast-channel (ceiling at ~0.73 vs ~0.70) but still essentially flat — depression remains compute-inefficient regardless of channel set.** | **[UNJUSTIFIABLE]** |
| scaling_laws/depression_extreme_binary_1C_optimal_epoch.png | **Erratic. Exclude from paper.** | Exclude. |
| scaling_laws/osa_binary_apples_postqc_1B_compute_scaling.png | **LSTM full-channel scaling is non-monotonic or flat. Transformer full-channel slope is preserved but lower absolute AUROC ceiling vs fast-channel. Head divergence is still visible but LSTM is now unstable rather than just plateaued.** | Note LSTM instability vs fast-channel plateau for OSA. |
| scaling_laws/osa_binary_apples_postqc_1C_optimal_epoch.png | **Check LSTM optimal epoch at 80m context — a very early stopping point there would explain the AUROC dip.** | Critical diagnostic for LSTM OSA instability. |
| scaling_laws/cvd_binary_1B_compute_scaling.png | ⚠️ DROPPED | Excluded. |
| scaling_laws/cvd_binary_1C_optimal_epoch.png | ⚠️ DROPPED | Excluded. |
| scaling_laws/sleepiness_binary_1B_compute_scaling.png | ⚠️ DROPPED | Excluded. |
| scaling_laws/sleepiness_binary_1C_optimal_epoch.png | ⚠️ DROPPED | Excluded. |
| scaling_laws/psqi_binary_1B_compute_scaling.png | ⚠️ DROPPED | Excluded. |
| scaling_laws/psqi_binary_1C_optimal_epoch.png | ⚠️ DROPPED | Excluded. |
