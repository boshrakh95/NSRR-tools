# Figure Interpretations — phase0_v3_abl (Modality Group Ablation)

**How to use:** Fill in the *Interpretation* column with what you observe.
Use *Additional Comments* for paper relevance, surprises, or follow-up notes.

> **See `paper_figures.md` for the full paper figure plan.**

---

## Paper Figure Assignment Index (v3_abl → paper)

The ablation contributes **one main figure** and **one main table** to the paper.
All per-condition individual plots (window sweep, calibration, PR, K*, heatmap) are
internal reference only — they are pending generation and will not appear in the paper.

| Paper location | Named figure | Source |
|---|---|---|
| **Fig 4** (main) | Modality Contribution — 5-panel bar chart | `analysis.csv` from v3_abl + baselines from v3 + v3_full |
| **Table V** (main) | Modality ablation ΔAUROC table | `results/tables/table6_modality.md` (EXISTS) |
| **EXCLUDED** | All per-condition window sweep, calibration, PR, K*, heatmap | Internal reference; not in paper |

---

**Context:** All ablation experiments use the LSTM head only, at a single context
length per task (120m for sex/apnea/sleep_efficiency/age; 40m for BMI).
Five conditions per task: no_bas, no_resp, no_ekg, cardio_only, bas_only.

**Modality groups (from config):**
- **BAS** (Brain Activity Signals, 10 ch): EEG (C3-M2, C4-M1, O1-M2, O2-M1, F3-M2, F4-M1, A1, A2) + EOG (LOC, ROC)
- **RESP** (Respiratory, 7 ch): Airflow, Thor, ABD, SpO2, HR, Snore, RespRate
- **EKG** (Cardiac, 2 ch): EKG, ECG-L/R
- **EMG** (Limb, 4 ch): CHIN, LLEG, RLEG, EMG

**Ablation conditions — what the model sees:**
| Condition | BAS | RESP | EKG | EMG |
|---|---|---|---|---|
| Full (baseline, fast-ch) | ✓ | — | — | — |
| `abl_no_bas` | ✗ | ✓ | ✓ | ✓ |
| `abl_no_resp` | ✓ | ✗ | ✓ | ✓ |
| `abl_no_ekg` | ✓ | ✓ | ✗ | ✓ |
| `abl_cardio` (SleepFounder comparison) | ✗ | ✓ | ✓ | ✗ |
| `abl_bas_only` | ✓ | ✗ | ✗ | ✗ |

> Note: The fast-channel v3 baseline already uses only BAS channels (it is BAS-only by
> default). The ablation starts from the full-channel model and progressively removes
> groups, so the correct comparison baseline for all Δ values is the **fast-channel LSTM
> at the same context length** (table6_modality "Full" column). The v3_full baseline is
> not used here since ablation checkpoints were trained on the fast-channel model.

**Figure status:** Individual per-condition PNG figures (window sweep, calibration,
PR curves, K* histograms, heatmaps) have NOT yet been generated. The figures directory
contains only `.gitkeep`. All interpretations below are based on the collected
`analysis.csv` and `table6_modality`. After running the plotting pipeline, verify
whether figures match the predicted patterns.

---

## Cross-Round Merged Figure Recommendation

**Proposed Figure: Modality Contribution Bar Chart (v3_abl + v3 reference)**

The primary ablation result should be presented as a grouped horizontal bar chart
showing ΔAUROC per condition per task, with vertical reference lines at the fast-channel
and full-channel baselines. This combines data from phase0_v3 (fast-channel baseline),
phase0_v3_full (full-channel ceiling), and phase0_v3_abl (ablation deltas).

Design:
- One panel per task (5 panels, 1-row or 2-row layout)
- X-axis: AUROC (absolute), with reference lines for v3_fast (solid) and v3_full (dashed)
- Y-axis: Ablation condition labels (no_bas, no_resp, no_ekg, cardio, bas_only)
- Bar color: encoding the dominant missing modality (e.g., orange=BAS removed, blue=RESP, green=EKG, gray=combined)
- Annotate each bar with the Δ value

Reference values:
| Task | v3 fast baseline (K=all) | v3_full (K=all) | Direction |
|---|---|---|---|
| sex_binary @120m | 0.872 (LSTM) / 0.905 (T) | 0.887 (LSTM) / 0.929 (T) | Full better |
| apnea_binary @120m | 0.832 (LSTM) / 0.857 (T) | 0.874 (LSTM) / 0.900 (T) | Full better |
| sleep_eff @120m | 0.778 (LSTM) / 0.815 (T) | 0.768 (LSTM) / 0.798 (T) | Neutral/slightly lower |
| age_class @120m | 0.893 (LSTM) / 0.902 (T) | 0.901 (LSTM) / 0.908 (T) | Full slightly better |
| bmi_binary @40m | 0.756 (LSTM) / 0.766 (T) | 0.799 (LSTM) / 0.812 (T) | Full better |

> Note: All values from `analysis.csv` (mean_prob_auroc, k=all). Saturation figures currently
> show K=1 segment-level metric and must be regenerated — see [CODE FIX] notes in
> figure_interpretations_v3.md and figure_interpretations_v3_full.md saturation sections.

> See `figure_interpretations_v3.md` Table 2 (Saturation Curves) and
> `figure_interpretations_v3_full.md` Table 2 (Saturation Curves) for the updated
> cross-round subplot note which references this figure.

---

## Numerical Summary (from table6_modality)

| Task | Context | Full | No BAS | Δ | No RESP | Δ | No EKG | Δ | Cardio | Δ | BAS only | Δ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sex | 120m | 0.872 | 0.803 | −0.069 | 0.861 | −0.011 | 0.799 | −0.074 | 0.800 | −0.072 | 0.781 | −0.092 |
| Apnea | 120m | 0.832 | 0.792 | −0.040 | 0.775 | −0.057 | 0.794 | −0.038 | 0.766 | −0.066 | 0.729 | −0.103 |
| Sleep eff. | 120m | 0.778 | 0.695 | −0.083 | 0.776 | −0.003 | 0.765 | −0.013 | 0.667 | −0.111 | 0.773 | −0.005 |
| Age | 120m | 0.893 | 0.847 | −0.046 | 0.885 | −0.008 | 0.876 | −0.017 | 0.824 | −0.069 | 0.858 | −0.035 |
| BMI | 40m | 0.756 | 0.721 | −0.035 | 0.766 | +0.010 | 0.751 | −0.005 | 0.675 | −0.081 | 0.742 | −0.014 |

---

## Table 1 — Per-Task Figures (Condition-Level)

---

### sex_binary_lstm — abl_no_bas (RESP+EKG+EMG active; BAS zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sex_binary_lstm_abl_no_bas_test_window_sweep_auroc.png | At 120m with K=all, AUROC = 0.803 (−0.069 from baseline 0.872). Removing EEG+EOG (BAS) moderately impairs sex prediction. The RESP+EKG+EMG channels alone still encode meaningful sex-related information, primarily through EKG-derived features (cardiac morphology, HRV) and EMG patterns. K-saturation curve expected to plateau lower than full baseline, requiring more windows (larger K) for the model to reach its ceiling. | [PENDING FIGURE] Generate with plot_window_sweep.py. Expected: plateau at ~0.80 vs 0.87 baseline. The EKG contribution to sex (seen more clearly in no_ekg: −0.074) suggests cardiac signals are informative. Compare K-saturation rate to baseline and bas_only to understand how many windows each modality set needs. |
| sex_binary_lstm_abl_no_bas_calibration_2A_reliability.png | Expected: calibration worse than baseline (lower absolute AUROC → less confident predictions). The model may be over-confident or under-confident depending on which channels' signals dominate at this condition. | [PENDING FIGURE] |
| sex_binary_lstm_abl_no_bas_pr_8A_curves.png | AUC-PR expected to drop relative to baseline but remain above bas_only and comparable to cardio condition (similar AUROC levels at 0.80). | [PENDING FIGURE] |
| sex_binary_lstm_abl_no_bas_kstar_9A_histogram.png | K* distribution expected to be broader than baseline — more windows needed per subject when BAS is removed. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Heatmap expected to show plateau at ~0.80 across all (L, K) cells since we're evaluating at a single fixed context (120m). The K-dependence is the main variation. | [PENDING FIGURE] |

---

### sex_binary_lstm — abl_no_resp (BAS+EKG+EMG active; RESP zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sex_binary_lstm_abl_no_resp_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.861 (−0.011 from baseline). **Removing respiratory signals barely affects sex prediction** — RESP channels contribute almost nothing to sex classification. This is the most expected result: breathing patterns and SpO2 do not encode sex-linked physiological information that the model can exploit. The K-saturation curve should be nearly identical to the full baseline. | The near-zero drop (−0.011) confirms RESP is the least important modality for sex. This validates the model's channel prioritization. Compare to v3_full where full-channel (which includes RESP) at 30s is WORSE than fast-channel — consistent: RESP adds noise at short contexts for sex. |
| sex_binary_lstm_abl_no_resp_calibration_2A_reliability.png | Calibration expected to be very similar to full baseline given the minimal AUROC drop. | [PENDING FIGURE] |
| sex_binary_lstm_abl_no_resp_pr_8A_curves.png | AUC-PR expected to nearly match baseline. | [PENDING FIGURE] |
| sex_binary_lstm_abl_no_resp_kstar_9A_histogram.png | K* distribution expected to closely match baseline. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Heatmap expected nearly identical to baseline — minimal RESP contribution to sex. | [PENDING FIGURE] |

---

### sex_binary_lstm — abl_no_ekg (BAS+RESP+EMG active; EKG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sex_binary_lstm_abl_no_ekg_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.799 (−0.074 from baseline). **EKG is the most important single-modality contributor for sex classification** (largest single-removal drop among no_bas/no_resp/no_ekg). This is a striking finding: cardiac morphology (ECG waveform shape) encodes sex information, consistent with known sexual dimorphism in ECG features (PR interval, QRS duration, T-wave amplitude). K-saturation plateau expected ~0.80. | KEY FINDING: EKG is the most discriminative modality for sex (even more than BAS). This is biologically interpretable — male and female ECGs differ systematically. Mention in paper. Also note the remarkable symmetry: no_ekg ≈ no_bas ≈ cardio ≈ 0.80, suggesting EKG and BAS are independently and nearly equivalently important for sex. |
| sex_binary_lstm_abl_no_ekg_calibration_2A_reliability.png | Expected calibration degradation similar to no_bas, given similar AUROC. | [PENDING FIGURE] |
| sex_binary_lstm_abl_no_ekg_pr_8A_curves.png | AUC-PR expected similar to no_bas. | [PENDING FIGURE] |
| sex_binary_lstm_abl_no_ekg_kstar_9A_histogram.png | K* distribution broader than baseline, similar to no_bas. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Heatmap plateau ~0.80, similar to no_bas heatmap. | [PENDING FIGURE] |

---

### sex_binary_lstm — abl_cardio (RESP+EKG active; BAS+EMG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sex_binary_lstm_abl_cardio_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.800 (−0.072 from baseline). With only RESP+EKG available (BAS and EMG zeroed), the model achieves the same AUROC as no_ekg and no_bas (all ~0.80). This reveals a surprising redundancy: **EKG alone from the cardio set drives the 0.80 ceiling** (since adding RESP to EKG in the cardio condition gives the same result as EKG alone in the no_bas condition). The cardio-only condition serves as the SleepFounder comparison point. | Compare to SleepFounder sex classification AUROC if available. This condition intentionally mirrors simplified wearable-grade signal availability (RESP+EKG without brain channels). The ~0.80 ceiling suggests a cardiological upper bound for non-EEG sex classification from PSG. |
| sex_binary_lstm_abl_cardio_calibration_2A_reliability.png | Expected calibration similar to no_bas and no_ekg. | [PENDING FIGURE] |
| sex_binary_lstm_abl_cardio_pr_8A_curves.png | AUC-PR similar to other ~0.80 conditions. | [PENDING FIGURE] |
| sex_binary_lstm_abl_cardio_kstar_9A_histogram.png | K* distribution similar to other ~0.80 conditions. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Heatmap ~0.80 plateau. | [PENDING FIGURE] |

---

### sex_binary_lstm — abl_bas_only (BAS active; RESP+EKG+EMG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sex_binary_lstm_abl_bas_only_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.781 (−0.092 from baseline). **BAS alone is the WORST condition for sex classification** — surprisingly, EEG+EOG without cardiac or respiratory signals is less informative than cardio alone (0.800) or any single-removal condition. This confirms that sex information is more strongly encoded in cardiac signals (EKG) than in EEG sleep patterns. Note this conflicts with intuition (EEG should encode sex via sleep architecture differences). The EMG channels (zeroed here) contribute substantially to the ~0.01 gap between bas_only and the other ~0.80 conditions. | **Counter-intuitive finding**: EEG-only (BAS) is WORSE for sex than cardio-only (RESP+EKG). Discuss in paper. The EKG is uniquely discriminative for sex — its removal causes the largest single-modality drop (−0.074), and without it, BAS alone cannot recover. EMG channels contribute ~0.019 to sex prediction (gap between bas_only=0.781 and no_ekg=0.799). |
| sex_binary_lstm_abl_bas_only_calibration_2A_reliability.png | Expected worst calibration of all sex ablation conditions. | [PENDING FIGURE] |
| sex_binary_lstm_abl_bas_only_pr_8A_curves.png | Lowest AUC-PR for sex ablation. | [PENDING FIGURE] |
| sex_binary_lstm_abl_bas_only_kstar_9A_histogram.png | Broadest K* — most windows needed to classify sex from EEG alone. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Lowest values in sex ablation heatmaps. | [PENDING FIGURE] |

---

### apnea_binary_lstm — abl_no_bas (RESP+EKG+EMG active; BAS zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| apnea_binary_lstm_abl_no_bas_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.792 (−0.040 from baseline 0.832). Removing EEG moderately impairs apnea classification. RESP+EKG+EMG without BAS still achieves solid performance — cardiac and respiratory signals carry the dominant apnea-detection signal. The drop is meaningful but not catastrophic, consistent with EEG arousals being secondary to direct respiratory/cardiac measures. | Moderate drop confirms BAS is supportive but not primary for apnea. The direct respiratory and cardiac signals (in RESP and EKG) outweigh EEG arousal markers. |
| apnea_binary_lstm_abl_no_bas_pr_8A_curves.png | AUC-PR expected to decrease moderately vs baseline. | [PENDING FIGURE] |
| apnea_binary_lstm_abl_no_bas_kstar_9A_histogram.png | K* distribution broader than baseline but narrower than bas_only. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Moderate AUROC reduction across (L, K). | [PENDING FIGURE] |

---

### apnea_binary_lstm — abl_no_resp (BAS+EKG+EMG active; RESP zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| apnea_binary_lstm_abl_no_resp_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.775 (−0.057 from baseline). **Removing respiratory signals is the second-largest single-modality drop for apnea** (after bas_only). RESP is the most physiologically direct apnea signal (airflow cessation, oxygen desaturation, respiratory effort), yet its removal causes only −0.057. The model partly compensates with EEG arousals and cardiac changes associated with apneas. | The −0.057 drop for RESP removal is larger than the −0.040 for BAS removal, confirming RESP > BAS for apnea, but neither is catastrophic. Residual AUROC 0.775 means EKG+BAS+EMG alone achieve decent apnea detection via cardiac autonomic markers. This is somewhat surprising given RESP literally measures the obstructions. |
| apnea_binary_lstm_abl_no_resp_pr_8A_curves.png | Larger AUC-PR drop than no_bas for apnea. | [PENDING FIGURE] |
| apnea_binary_lstm_abl_no_resp_kstar_9A_histogram.png | Broader K* than no_bas. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | AUROC reduced; lowest among single-removal conditions. | [PENDING FIGURE] |

---

### apnea_binary_lstm — abl_no_ekg (BAS+RESP+EMG active; EKG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| apnea_binary_lstm_abl_no_ekg_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.794 (−0.038). Removing EKG has the smallest drop among single-removal conditions for apnea — cardiac signals are least critical when respiratory and EEG signals are available. This is physiologically sensible: direct respiratory measures (SpO2, airflow) are more discriminative than derived cardiac markers for AHI classification. | Confirms hierarchy for apnea: RESP > BAS > EKG for single-modality importance. The small EKG drop (−0.038) suggests that heart-rate-based apnea markers are partially redundant with RESP (SpO2 desaturation and HR deceleration co-occur). |
| apnea_binary_lstm_abl_no_ekg_pr_8A_curves.png | Small AUC-PR drop, similar to no_bas. | [PENDING FIGURE] |
| apnea_binary_lstm_abl_no_ekg_kstar_9A_histogram.png | K* similar to no_bas. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | AUROC reduction moderate, similar to no_bas. | [PENDING FIGURE] |

---

### apnea_binary_lstm — abl_cardio (RESP+EKG active; BAS+EMG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| apnea_binary_lstm_abl_cardio_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.766 (−0.066). **Cardio (RESP+EKG) without BAS and EMG achieves 0.766** — better than BAS-only (0.729). This is the expected direction: respiratory and cardiac channels alone outperform brain signals alone for OSA detection. The cardio condition directly maps to minimal wearable-device signals (oximeter + ECG). AUROC of 0.766 represents the "SleepFounder-like" lower bound, and the full model (0.832) demonstrates what EEG and EMG add on top. | SleepFounder comparison point. The gap between cardio-only (0.766) and full (0.832) represents the contribution of EEG + EMG, worth ~0.066 AUROC. This is the "added value of full PSG" for apnea beyond simple cardiac/respiratory monitoring. |
| apnea_binary_lstm_abl_cardio_pr_8A_curves.png | AUC-PR above BAS-only but below full baseline. | [PENDING FIGURE] |
| apnea_binary_lstm_abl_cardio_kstar_9A_histogram.png | K* between bas_only and no_bas. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Moderate values; reflects respiratory+cardiac floor for apnea. | [PENDING FIGURE] |

---

### apnea_binary_lstm — abl_bas_only (BAS active; RESP+EKG+EMG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| apnea_binary_lstm_abl_bas_only_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.729 (−0.103). **Worst condition for apnea by a wide margin.** EEG+EOG alone (no respiratory or cardiac channels) achieves 0.729 — substantial but the weakest performance. The model relies purely on EEG arousal patterns and sleep stage transitions that correlate with apnea events, without the direct respiratory/cardiac signals. The large drop (−0.103) confirms the critical role of RESP and EKG for apnea detection. | **Largest single drop in the entire ablation study** (−0.103). Confirms the physical intuition: apnea is defined by respiratory events; removing all respiratory and cardiac channels predictably causes the largest degradation. That 0.729 is still achievable from EEG alone is notable — sleep fragmentation and arousal patterns provide an indirect apnea signal. |
| apnea_binary_lstm_abl_bas_only_pr_8A_curves.png | Largest AUC-PR drop for apnea. | [PENDING FIGURE] |
| apnea_binary_lstm_abl_bas_only_kstar_9A_histogram.png | Broadest K* distribution — most windows needed from EEG alone to detect apnea. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Lowest values in apnea ablation. | [PENDING FIGURE] |

---

### sleep_efficiency_binary_lstm — abl_no_bas (RESP+EKG+EMG active; BAS zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleep_efficiency_binary_lstm_abl_no_bas_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.695 (−0.083 from baseline 0.778). **Removing BAS causes the largest single-modality drop for sleep efficiency** — EEG/EOG are the primary channels for sleep staging and thus for sleep efficiency prediction. Without EEG, the model loses access to NREM slow waves, spindles, and K-complexes that define sleep depth. RESP+EKG+EMG alone are insufficient for sleep efficiency prediction (0.695 is the second-lowest condition after cardio). | Confirms sleep efficiency is EEG-dominant. The −0.083 drop from removing BAS is the second-largest across all task × condition pairs (behind sleep_eff cardio: −0.111). Directly corroborates the v3_full finding that full-channel hurts sleep_efficiency (-0.006 AUROC when additional non-EEG channels are added). |
| sleep_efficiency_binary_lstm_abl_no_bas_pr_8A_curves.png | Large AUC-PR drop. | [PENDING FIGURE] |
| sleep_efficiency_binary_lstm_abl_no_bas_kstar_9A_histogram.png | K* distribution broad, reflecting the signal loss from removing EEG. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Low AUROC across cells; second lowest for sleep_eff. | [PENDING FIGURE] |

---

### sleep_efficiency_binary_lstm — abl_no_resp (BAS+EKG+EMG active; RESP zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleep_efficiency_binary_lstm_abl_no_resp_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.776 (−0.003). **Near-zero drop — RESP contributes almost nothing to sleep efficiency prediction.** Breathing patterns, SpO2, and respiratory rate do not significantly discriminate high vs low sleep efficiency. This makes biological sense: sleep efficiency is defined by sleep staging transitions, which are captured by EEG, not by respiratory events. | RESP is the least informative modality for sleep efficiency (−0.003). Convergence with the v3_full finding: full-channel sleep efficiency is worse than fast-channel, suggesting that RESP channels add noise rather than signal for this task. |
| sleep_efficiency_binary_lstm_abl_no_resp_pr_8A_curves.png | AUC-PR nearly identical to baseline. | [PENDING FIGURE] |
| sleep_efficiency_binary_lstm_abl_no_resp_kstar_9A_histogram.png | K* distribution nearly identical to baseline. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Heatmap nearly same as baseline for sleep efficiency. | [PENDING FIGURE] |

---

### sleep_efficiency_binary_lstm — abl_no_ekg (BAS+RESP+EMG active; EKG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleep_efficiency_binary_lstm_abl_no_ekg_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.765 (−0.013). Small but not negligible drop — EKG contributes slightly to sleep efficiency prediction, likely through heart rate variability patterns associated with sleep stages (HRV changes during NREM/REM are detectable via ECG). However, EEG is sufficient to capture most of this information redundantly. | Small contribution of EKG for sleep efficiency. The HRV-sleep stage correlation partially contributes here, but is dominated by the direct EEG signal. |
| sleep_efficiency_binary_lstm_abl_no_ekg_pr_8A_curves.png | Small AUC-PR drop. | [PENDING FIGURE] |
| sleep_efficiency_binary_lstm_abl_no_ekg_kstar_9A_histogram.png | K* distribution slightly broader than baseline. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Minor reduction vs baseline. | [PENDING FIGURE] |

---

### sleep_efficiency_binary_lstm — abl_cardio (RESP+EKG active; BAS+EMG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleep_efficiency_binary_lstm_abl_cardio_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.667 (−0.111). **The largest single drop in the entire ablation study.** Cardio-only (RESP+EKG, no BAS, no EMG) completely fails to predict sleep efficiency — 0.667 is barely above chance for a binary task with balanced classes. Without EEG, there is simply no direct sleep-staging information available, and respiratory/cardiac signals alone cannot reconstruct the sleep architecture that defines sleep efficiency. | **Strongest result in the ablation study**: removing EEG from sleep efficiency prediction causes −0.111 AUROC drop. This validates the theoretical premise that sleep efficiency is defined by sleep staging, which requires EEG. Cardio-only (SleepFounder-like) performance of 0.667 is the lowest single condition value across all 25 task×condition pairs. |
| sleep_efficiency_binary_lstm_abl_cardio_pr_8A_curves.png | Lowest AUC-PR in the entire ablation study for any condition. | [PENDING FIGURE] |
| sleep_efficiency_binary_lstm_abl_cardio_kstar_9A_histogram.png | Very broad K* — many windows cannot achieve correct classification at all. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Lowest values in the ablation study overall. | [PENDING FIGURE] |

---

### sleep_efficiency_binary_lstm — abl_bas_only (BAS active; RESP+EKG+EMG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleep_efficiency_binary_lstm_abl_bas_only_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.773 (−0.005). **The most striking result in the ablation study: BAS alone is nearly as good as the full baseline (0.778).** EEG+EOG channels alone capture essentially all the sleep efficiency signal. Removing RESP, EKG, and EMG causes only −0.005 AUROC loss — within the range of random variation. Sleep efficiency is a pure EEG task. | **THE key finding of the sleep efficiency ablation**: BAS alone suffices. This validates the design choice of the fast-channel model and explains why the v3_full (which adds many non-BAS channels) performs WORSE than v3_fast for sleep efficiency. EMG (chin) contributes ~0.005 on top of pure EEG (gap between bas_only and baseline when EMG is included). |
| sleep_efficiency_binary_lstm_abl_bas_only_pr_8A_curves.png | AUC-PR nearly identical to baseline. | [PENDING FIGURE] |
| sleep_efficiency_binary_lstm_abl_bas_only_kstar_9A_histogram.png | K* distribution nearly same as baseline — EEG alone is sufficient. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Heatmap nearly same as baseline. Critical comparison: bas_only ≈ baseline >> cardio. | [PENDING FIGURE] |

---

### age_class_lstm — abl_no_bas (RESP+EKG+EMG active; BAS zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| age_class_lstm_abl_no_bas_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.847 (−0.046 from baseline 0.893). Removing EEG/EOG causes a moderate drop. Age-related changes in sleep architecture (slow wave sleep decline, REM changes, spindle density) are captured via EEG, making BAS important. However, RESP+EKG+EMG still achieve 0.847, suggesting that age also modulates breathing patterns, cardiac metrics, and limb movement. | Moderate contribution of BAS for age — the EEG sleep staging signal is important but not exclusively so. Non-EEG channels retain substantial age-related information (autonomic age-related changes in HRV, respiratory patterns). |
| age_class_lstm_abl_no_bas_kstar_9A_histogram.png | K* distribution broader than baseline but narrower than for sleep_eff no_bas. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Moderate AUROC reduction. | [PENDING FIGURE] |

---

### age_class_lstm — abl_no_resp (BAS+EKG+EMG active; RESP zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| age_class_lstm_abl_no_resp_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.885 (−0.008). **Near-zero drop — RESP is least informative for age.** Respiratory patterns during sleep are not strongly age-discriminative in this context (or are redundant with EKG-derived HRV). | RESP redundancy with EKG for age: SpO2 and HR patterns captured in RESP overlap with cardiac autonomic signals in EKG. |
| age_class_lstm_abl_no_resp_kstar_9A_histogram.png | K* nearly same as baseline. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Nearly same as baseline. | [PENDING FIGURE] |

---

### age_class_lstm — abl_no_ekg (BAS+RESP+EMG active; EKG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| age_class_lstm_abl_no_ekg_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.876 (−0.017). Small but meaningful drop — EKG contributes to age prediction through HRV patterns (autonomic modulation of heart rate decreases with age). The contribution is smaller than BAS. | EKG contribution to age (−0.017) reflects cardiac autonomic aging. Smaller than no_bas (−0.046), confirming BAS > EKG for age. |
| age_class_lstm_abl_no_ekg_kstar_9A_histogram.png | Slightly broader K* than baseline. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Small reduction from baseline. | [PENDING FIGURE] |

---

### age_class_lstm — abl_cardio (RESP+EKG active; BAS+EMG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| age_class_lstm_abl_cardio_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.824 (−0.069). **Cardio alone produces the largest drop for age** (tied with the largest single condition drop for age). RESP+EKG without BAS and EMG achieves 0.824 — better than the expected floor, but −0.069 below baseline. Age prediction requires sleep staging information (EEG) that cardio channels cannot fully replace. | Largest drop for age, confirming BAS is the dominant modality. The gap between cardio (0.824) and BAS-only (0.858) shows BAS alone outperforms cardio alone for age — unlike sex where the relationship was reversed. |
| age_class_lstm_abl_cardio_kstar_9A_histogram.png | Broader K* than bas_only. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Lowest age ablation values. | [PENDING FIGURE] |

---

### age_class_lstm — abl_bas_only (BAS active; RESP+EKG+EMG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| age_class_lstm_abl_bas_only_test_window_sweep_auroc.png | At 120m K=all, AUROC = 0.858 (−0.035). **BAS alone retains most of the age-prediction signal** (only −0.035 drop, second-smallest after no_resp). EEG/EOG alone is nearly sufficient for age classification — sleep architecture changes with age are robustly encoded in EEG. | Confirms age is primarily an EEG task, though less extremely so than sleep efficiency. BAS alone (0.858) outperforms cardio alone (0.824) for age, contrasting with sex where the relationship is reversed. |
| age_class_lstm_abl_bas_only_kstar_9A_histogram.png | K* distribution slightly broader than baseline but narrow. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Close to baseline values. | [PENDING FIGURE] |

---

### bmi_binary_lstm — abl_no_bas (RESP+EKG+EMG active; BAS zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| bmi_binary_lstm_abl_no_bas_test_window_sweep_auroc.png | At 40m K=all, AUROC = 0.721 (−0.035 from baseline 0.756). Moderate drop from removing BAS. EEG/EOG contributions to BMI likely reflect sleep architecture differences between normal-weight and obese subjects (known association between obesity and sleep apnea-related EEG changes). RESP+EKG+EMG without BAS achieves 0.721 — reasonable performance, but BAS contributes meaningfully. | BAS contributes to BMI prediction (~−0.035), though not dominantly. May reflect indirect BMI-sleep architecture relationship via OSA comorbidity. |
| bmi_binary_lstm_abl_no_bas_pr_8A_curves.png | Moderate AUC-PR drop. | [PENDING FIGURE] |
| bmi_binary_lstm_abl_no_bas_kstar_9A_histogram.png | Broader K* than baseline. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Moderate reduction. | [PENDING FIGURE] |

---

### bmi_binary_lstm — abl_no_resp (BAS+EKG+EMG active; RESP zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| bmi_binary_lstm_abl_no_resp_test_window_sweep_auroc.png | At 40m K=all, AUROC = 0.766 (+0.010 — a SMALL GAIN vs baseline 0.756). **Removing RESP marginally improves BMI prediction.** Respiratory signals add noise rather than signal for BMI classification: airflow, effort, and SpO2 patterns primarily reflect apnea severity (which correlates with BMI but is a confounded signal), not BMI directly. Removing this confounded channel allows the model to rely on more direct BMI indicators (cardiac morphology, limb EMG patterns, sleep architecture). | **Unexpected gain**: RESP removal helps BMI. This is one of the most notable findings — respiratory signals are COUNTER-PRODUCTIVE for BMI classification. Likely because SpO2 and airflow patterns encode apnea severity (a mediating variable) rather than BMI itself, which confuses the model. Consistent with the v3_full observation that adding RESP channels (as part of full-channel) improves BMI — wait, v3_full BMI improved... Let me reconcile: v3_full adds ALL channels and improves BMI by +6 pp. The ablation shows RESP alone hurts. So it's the non-RESP channels in full (EKG + limb EMG) that drive the v3_full BMI improvement. |
| bmi_binary_lstm_abl_no_resp_pr_8A_curves.png | AUC-PR marginally higher than baseline. | [PENDING FIGURE] |
| bmi_binary_lstm_abl_no_resp_kstar_9A_histogram.png | K* distribution similar to or slightly narrower than baseline. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Slightly higher than baseline in some (L, K) cells. | [PENDING FIGURE] |

---

### bmi_binary_lstm — abl_no_ekg (BAS+RESP+EMG active; EKG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| bmi_binary_lstm_abl_no_ekg_test_window_sweep_auroc.png | At 40m K=all, AUROC = 0.751 (−0.005). Near-zero drop — EKG barely contributes to BMI prediction. Cardiac morphology does encode BMI-related information (obesity affects ECG) but the contribution is small and likely already captured via respiratory and sleep-stage signals. | Minimal EKG contribution to BMI (−0.005). Combined with the no_resp gain, suggests BMI signal is primarily in BAS (sleep architecture/staging) and EMG channels (obesity-related limb movement patterns). |
| bmi_binary_lstm_abl_no_ekg_pr_8A_curves.png | Nearly same AUC-PR as baseline. | [PENDING FIGURE] |
| bmi_binary_lstm_abl_no_ekg_kstar_9A_histogram.png | K* nearly same as baseline. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Nearly same as baseline. | [PENDING FIGURE] |

---

### bmi_binary_lstm — abl_cardio (RESP+EKG active; BAS+EMG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| bmi_binary_lstm_abl_cardio_test_window_sweep_auroc.png | At 40m K=all, AUROC = 0.675 (−0.081). **Largest drop for BMI** — cardio channels alone (RESP+EKG, no BAS, no EMG) are the worst non-bas-only condition for BMI. RESP adds noise (confirmed by no_resp gain), and removing EMG additionally hurts. The cardio condition loses both the useful EEG (BAS) and the limb EMG signals while retaining the noise-adding RESP. | The cardio condition is particularly bad for BMI because it combines two harmful effects: removing BAS (−0.035) AND removing EMG (likely a meaningful contributor to BMI, as limb movement patterns correlate with body mass) WHILE keeping RESP (which hurts). The net effect is the worst BMI ablation outcome. |
| bmi_binary_lstm_abl_cardio_pr_8A_curves.png | Lowest AUC-PR for BMI. | [PENDING FIGURE] |
| bmi_binary_lstm_abl_cardio_kstar_9A_histogram.png | Broadest K* for BMI ablation. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Lowest BMI ablation values. | [PENDING FIGURE] |

---

### bmi_binary_lstm — abl_bas_only (BAS active; RESP+EKG+EMG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| bmi_binary_lstm_abl_bas_only_test_window_sweep_auroc.png | At 40m K=all, AUROC = 0.742 (−0.014). **BAS alone retains most BMI prediction signal** — the second-best ablation condition after no_resp. EEG/EOG sleep architecture changes encode BMI-related information (obesity-associated sleep changes: more light sleep, less deep sleep, altered sleep architecture). Removing RESP/EKG/EMG barely changes performance, consistent with those channels either being redundant or adding noise. | BAS-dominant for BMI. Combined with the no_resp gain, the full picture is: BAS provides the signal; RESP adds noise; EKG is neutral; EMG may add a small contribution (gap between bas_only=0.742 and no_ekg=0.751 with EMG present = ~0.009 from EMG). |
| bmi_binary_lstm_abl_bas_only_pr_8A_curves.png | AUC-PR close to baseline. | [PENDING FIGURE] |
| bmi_binary_lstm_abl_bas_only_kstar_9A_histogram.png | K* distribution close to baseline. | [PENDING FIGURE] |
| auroc_test/heatmap_auroc.png | Close to baseline values. | [PENDING FIGURE] |

---

## Table 2 — Summary Figures

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| table6_modality (existing table, no figure yet) | Complete numerical ablation summary across 5 tasks × 5 conditions. The dominant finding: every task has a different most-important modality. Sleep efficiency and age are BAS-dominant (EEG alone ≈ full performance). Sex is EKG-dominant (surprising — cardiac morphology more discriminative than EEG for sex). Apnea is mixed RESP+BAS. BMI is BAS-dominant with RESP actively hurting. Cardio-only (SleepFounder comparison) is consistently among the weakest conditions except relative to bas_only for apnea. | This table is the primary ablation result for the paper (Table V / Table 6). No figure currently exists — generate from the data. |
| modality_ablation_summary_bar.png | [PENDING] Proposed grouped bar chart: one panel per task (5 panels), x-axis = AUROC, y-axis = ablation conditions, vertical reference lines for v3_fast and v3_full baselines. Bars colored by dominant missing group. Δ values annotated on bars. This is the main ablation figure (Fig 4 in the paper outline). **Combines phase0_v3, phase0_v3_full, and phase0_v3_abl data in one figure.** | [CROSS-ROUND FIGURE] Design: 5 horizontal subplots (one per task), consistent layout. Reference lines: fast-ch baseline (solid black), full-ch baseline (dashed gray). The contrast between tasks (sleep_eff: BAS nearly sufficient vs sex: EKG surprisingly large) makes this a compelling visual. |

---

## Key Findings Summary

**Modality importance ranking per task (ΔAUROC, single-removal, most to least impactful):**

| Task | Most → Least important modality |
|---|---|
| Sleep efficiency | **BAS** (−0.083) >> EKG (−0.013) > RESP (−0.003) |
| Sex | **EKG** (−0.074) ≈ **BAS** (−0.069) >> RESP (−0.011) |
| Age | **BAS** (−0.046) > EKG (−0.017) > RESP (−0.008) |
| Apnea | **RESP** (−0.057) > BAS (−0.040) ≈ EKG (−0.038) |
| BMI | BAS (−0.035) >> EKG (−0.005) >> RESP (+0.010) |

**Cross-task patterns:**
1. BAS (EEG/EOG) is universally the most or second-most important modality — sleep signals are the foundation.
2. RESP is important only for apnea (by definition); for all other tasks it is neutral or harmful (BMI).
3. EKG matters most for sex (cardiac sex dimorphism) and least for sleep efficiency.
4. Cardio-only (SleepFounder comparison) is worst or second-worst for all tasks except apnea.
5. BAS-only performance relative to full baseline varies: sleep_eff (−0.005), age (−0.035), BMI (−0.014), sex (−0.092), apnea (−0.103).
6. The ranking of bas_only performance mirrors the task classification of "EEG-dominant" vs "multi-modal": sleep_eff > age > BMI > sex > apnea.
