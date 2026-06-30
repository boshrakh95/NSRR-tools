# Figure Interpretations — phase0_v3_abl (Modality Group Ablation)

**How to use:** Fill in the *Interpretation* column with what you observe.
Use *Additional Comments* for paper relevance, surprises, or follow-up notes.

**Context:** All ablation experiments use the LSTM head only, at a single context
length per task (120m for sex/apnea/sleep_efficiency/age; 40m for BMI).
Five conditions per task: no_bas, no_resp, no_ekg, cardio_only, bas_only.

**Status:** Figures directory is currently empty — figures will be generated after
inference and analysis pipeline runs on the trained ablation checkpoints.

---

## Table 1 — Per-Task Figures

The ablation pipeline produces the same figure types as the main experiments
(window sweep, calibration, PR curves, etc.) but for each task × condition.
Rows are pre-populated; fill in once figures are generated.

---

### sex_binary_lstm — abl_no_bas (RESP+EKG+EMG active, BAS zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sex_binary_lstm_abl_no_bas_test_window_sweep_auroc.png | | Compare to full baseline 0.872 |
| sex_binary_lstm_abl_no_bas_calibration_2A_reliability.png | | |
| sex_binary_lstm_abl_no_bas_pr_8A_curves.png | | |
| sex_binary_lstm_abl_no_bas_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### sex_binary_lstm — abl_no_resp (BAS+EKG+EMG active, RESP zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sex_binary_lstm_abl_no_resp_test_window_sweep_auroc.png | | Compare to full baseline 0.872 |
| sex_binary_lstm_abl_no_resp_calibration_2A_reliability.png | | |
| sex_binary_lstm_abl_no_resp_pr_8A_curves.png | | |
| sex_binary_lstm_abl_no_resp_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### sex_binary_lstm — abl_no_ekg (BAS+RESP+EMG active, EKG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sex_binary_lstm_abl_no_ekg_test_window_sweep_auroc.png | | Compare to full baseline 0.872; expected large drop |
| sex_binary_lstm_abl_no_ekg_calibration_2A_reliability.png | | |
| sex_binary_lstm_abl_no_ekg_pr_8A_curves.png | | |
| sex_binary_lstm_abl_no_ekg_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### sex_binary_lstm — abl_cardio (RESP+EKG active, BAS+EMG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sex_binary_lstm_abl_cardio_test_window_sweep_auroc.png | | Compare to SleepFounder cardiorespiratory baseline |
| sex_binary_lstm_abl_cardio_calibration_2A_reliability.png | | |
| sex_binary_lstm_abl_cardio_pr_8A_curves.png | | |
| sex_binary_lstm_abl_cardio_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### sex_binary_lstm — abl_bas_only (BAS active, RESP+EKG+EMG zeroed)

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sex_binary_lstm_abl_bas_only_test_window_sweep_auroc.png | | Compare to full baseline 0.872; expected worst condition |
| sex_binary_lstm_abl_bas_only_calibration_2A_reliability.png | | |
| sex_binary_lstm_abl_bas_only_pr_8A_curves.png | | |
| sex_binary_lstm_abl_bas_only_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### apnea_binary_lstm — abl_no_bas

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| apnea_binary_lstm_abl_no_bas_test_window_sweep_auroc.png | | Full baseline 0.832; expect moderate drop |
| apnea_binary_lstm_abl_no_bas_pr_8A_curves.png | | |
| apnea_binary_lstm_abl_no_bas_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### apnea_binary_lstm — abl_no_resp

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| apnea_binary_lstm_abl_no_resp_test_window_sweep_auroc.png | | Full baseline 0.832; expect LARGEST drop (RESP most necessary) |
| apnea_binary_lstm_abl_no_resp_pr_8A_curves.png | | |
| apnea_binary_lstm_abl_no_resp_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### apnea_binary_lstm — abl_no_ekg

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| apnea_binary_lstm_abl_no_ekg_test_window_sweep_auroc.png | | Full baseline 0.832; expect small drop |
| apnea_binary_lstm_abl_no_ekg_pr_8A_curves.png | | |
| apnea_binary_lstm_abl_no_ekg_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### apnea_binary_lstm — abl_cardio

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| apnea_binary_lstm_abl_cardio_test_window_sweep_auroc.png | | Full baseline 0.832; compare to SleepFounder OSA 0.917 |
| apnea_binary_lstm_abl_cardio_pr_8A_curves.png | | |
| apnea_binary_lstm_abl_cardio_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### apnea_binary_lstm — abl_bas_only

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| apnea_binary_lstm_abl_bas_only_test_window_sweep_auroc.png | | Full baseline 0.832; expect worst (brain signals alone insufficient for OSA) |
| apnea_binary_lstm_abl_bas_only_pr_8A_curves.png | | |
| apnea_binary_lstm_abl_bas_only_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### sleep_efficiency_binary_lstm — abl_no_bas

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleep_efficiency_binary_lstm_abl_no_bas_test_window_sweep_auroc.png | | Full baseline 0.780; expect largest drop (EEG drives sleep efficiency) |
| sleep_efficiency_binary_lstm_abl_no_bas_pr_8A_curves.png | | |
| sleep_efficiency_binary_lstm_abl_no_bas_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### sleep_efficiency_binary_lstm — abl_no_resp

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleep_efficiency_binary_lstm_abl_no_resp_test_window_sweep_auroc.png | | Full baseline 0.780; expect small drop |
| sleep_efficiency_binary_lstm_abl_no_resp_pr_8A_curves.png | | |
| sleep_efficiency_binary_lstm_abl_no_resp_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### sleep_efficiency_binary_lstm — abl_no_ekg

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleep_efficiency_binary_lstm_abl_no_ekg_test_window_sweep_auroc.png | | Full baseline 0.780; expect small-moderate drop |
| sleep_efficiency_binary_lstm_abl_no_ekg_pr_8A_curves.png | | |
| sleep_efficiency_binary_lstm_abl_no_ekg_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### sleep_efficiency_binary_lstm — abl_cardio

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleep_efficiency_binary_lstm_abl_cardio_test_window_sweep_auroc.png | | Full baseline 0.780; expect largest drop (BAS+EMG zeroed) |
| sleep_efficiency_binary_lstm_abl_cardio_pr_8A_curves.png | | |
| sleep_efficiency_binary_lstm_abl_cardio_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### sleep_efficiency_binary_lstm — abl_bas_only

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| sleep_efficiency_binary_lstm_abl_bas_only_test_window_sweep_auroc.png | | Full baseline 0.780; expect ~same as full (BAS is sufficient) |
| sleep_efficiency_binary_lstm_abl_bas_only_pr_8A_curves.png | | |
| sleep_efficiency_binary_lstm_abl_bas_only_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### age_class_lstm — abl_no_bas

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| age_class_lstm_abl_no_bas_test_window_sweep_auroc.png | | Full baseline 0.893; expect moderate drop |
| age_class_lstm_abl_no_bas_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### age_class_lstm — abl_no_resp

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| age_class_lstm_abl_no_resp_test_window_sweep_auroc.png | | Full baseline 0.893; expect small drop |
| age_class_lstm_abl_no_resp_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### age_class_lstm — abl_no_ekg

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| age_class_lstm_abl_no_ekg_test_window_sweep_auroc.png | | Full baseline 0.893; expect moderate drop |
| age_class_lstm_abl_no_ekg_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### age_class_lstm — abl_cardio

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| age_class_lstm_abl_cardio_test_window_sweep_auroc.png | | Full baseline 0.893 |
| age_class_lstm_abl_cardio_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### age_class_lstm — abl_bas_only

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| age_class_lstm_abl_bas_only_test_window_sweep_auroc.png | | Full baseline 0.893; expect moderate drop |
| age_class_lstm_abl_bas_only_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### bmi_binary_lstm — abl_no_bas

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| bmi_binary_lstm_abl_no_bas_test_window_sweep_auroc.png | | Full baseline 0.756; expect borderline drop |
| bmi_binary_lstm_abl_no_bas_pr_8A_curves.png | | |
| bmi_binary_lstm_abl_no_bas_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### bmi_binary_lstm — abl_no_resp

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| bmi_binary_lstm_abl_no_resp_test_window_sweep_auroc.png | | Full baseline 0.756; expect near-zero drop |
| bmi_binary_lstm_abl_no_resp_pr_8A_curves.png | | |
| bmi_binary_lstm_abl_no_resp_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### bmi_binary_lstm — abl_no_ekg

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| bmi_binary_lstm_abl_no_ekg_test_window_sweep_auroc.png | | Full baseline 0.756; expect near-zero drop |
| bmi_binary_lstm_abl_no_ekg_pr_8A_curves.png | | |
| bmi_binary_lstm_abl_no_ekg_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### bmi_binary_lstm — abl_cardio

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| bmi_binary_lstm_abl_cardio_test_window_sweep_auroc.png | | Full baseline 0.756; expect largest drop for BMI |
| bmi_binary_lstm_abl_cardio_pr_8A_curves.png | | |
| bmi_binary_lstm_abl_cardio_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

### bmi_binary_lstm — abl_bas_only

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| bmi_binary_lstm_abl_bas_only_test_window_sweep_auroc.png | | Full baseline 0.756; expect moderate drop |
| bmi_binary_lstm_abl_bas_only_pr_8A_curves.png | | |
| bmi_binary_lstm_abl_bas_only_kstar_9A_histogram.png | | |
| auroc_test/heatmap_auroc.png | | |

---

## Table 2 — Across-Task Figures (Ablation)

No across-task ablation figures currently defined in the pipeline.
The primary ablation summary is Table 6 (tab:modality) in the main paper.
If a bar chart summary figure is generated, add it here.

| Figure Name | Interpretation | Additional Comments |
|---|---|---|
| modality_ablation_summary_bar.png | *(to be generated)* | Main paper Fig 4 placeholder |
