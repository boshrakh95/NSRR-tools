# Paper Figure Plan — Phase 0 (TBME)

**Last updated:** 2026-07-01  
**Status:** Ready for regeneration. Run `bash scripts/run_figures.sh` to produce all figures.
Saturation code fix (P1) applied. 1A uShape BA-metric fix applied 2026-07-01 (S-Fig 11 now ready).

---

## Guiding Principles

- **v3 (fast-channel, 7–8 ch)** is the primary experiment → all main figures come from here.
- **v3_full (full-channel)** and **v3_abl (ablation)** each contribute **one main figure** (or are in supplementary + tables only).
- Channel comparison (fast vs full) is more efficiently shown as extra columns in Table II or as S-Fig 2; no dedicated main figure needed.
- **TBME target:** 4–5 main figures + 5 main tables. No figure should duplicate information already in a table.
- Dropped tasks (cvd_binary, sleepiness_binary, psqi_binary) are excluded from all paper figures.
- **Primary metric throughout:** subject-level mean-pool AUROC at K=all (`mean_prob_auroc` from `analysis.csv`).  
  Saturation code fix (P1) applied 2026-07 — `plot_saturation.py` now reads `mean_prob_auroc` K=all from `analysis.csv`.

---

## Main Paper Tables (for reference)

| Table | Name | Source file | Status |
|---|---|---|---|
| Table I | Task definitions | manual | Exists |
| Table II | Peak AUROC: fast-ch + full-ch columns, K=5 & K=all at best L | `table1_peak_auroc_fast/full` | Exists (needs full-ch columns merged) |
| Table III | Saturation: L* per task/head + ΔAUROC | `table2_lstar_fast` | Exists (regenerate after code fix) |
| Table IV | Head comparison at L* | `table5_heads_fast` | Exists |
| Table V | Modality ablation ΔAUROC | `table6_modality` | **EXISTS, complete** |

---

## Main Paper Figures

---

### Fig 1 — Context-Length Saturation `[v3]`

**Named:** *"Context Saturation"*  
**Layout:** 2 rows × 4 cols (7 task panels + 1 col for shared legend / blank)  
**Content:** Subject-level AUROC (mean_prob_auroc, K=all) vs context length (log-x scale, 30s–240m) for each of the 7 retained tasks. Three lines per panel: LSTM (blue, solid), Transformer (orange, dashed), MeanPool (green, dotted).  
**Caption elements:** annotate each point with value; L* arrow for sleep_efficiency (>240m); show dominant finding per panel in panel label.  

| Panel | Task | Key message |
|---|---|---|
| (a) | sex_binary | Monotonic; no saturation; Transformer leads (+5 pp over LSTM at 240m) |
| (b) | age_class | Monotonic; near-saturation at 240m |
| (c) | apnea_binary | Steepest rise; highest absolute AUROC; L*≈80–120m |
| (d) | bmi_binary | Weakest sensitivity; earliest saturation (L*≈40m) |
| (e) | sleep_efficiency | Strongest sensitivity; NOT saturated at 240m (L*>240m) → arrow |
| (f) | depression_extreme | **Erratic / non-monotonic** → bold/hatched panel; negative result |
| (g) | osa_binary_apples_postqc | Head divergence: LSTM saturates ~40m; Transformer/MeanPool rise to 240m |

**Source plots:** `saturation/saturation_{task}_auroc_test.png` ×7 (phase0_v3)  
**Command:** `bash scripts/run_figures.sh --skip-iso --skip-cross-round --skip-tables`

**Code fix required:** `plot_saturation.py` must read `mean_prob_auroc` K=all from `analysis.csv` instead of `test_auroc` from `summary.csv`.

---

### Fig 2 — Aggregation Efficiency `[v3, sex_binary_transformer as lead example]`

**Named:** *"Inference Efficiency"*  
**Layout:** 1 row × 3 panels  
**Content:** Shows that longer context + few windows is always better than short context + many windows at equal compute, using sex_binary Transformer as the primary illustration.

| Panel | Plot type | Source file | Key message |
|---|---|---|---|
| (a) AUROC vs K | `metric_vs_k_auroc.png` | `sex_binary_transformer/auroc_test/` | Each context curve saturates; long-L saturates at lower K |
| (b) Iso-compute heatmap | `heatmap_auroc.png` | `sex_binary_transformer/auroc_test/` | High AUROC region is top-left (long L, small K); iso-compute contours visible |
| (c) Min-cost frontier | `min_cost_frontier_auroc.png` | `sex_binary_transformer/auroc_test/` | Cheapest path to target AUROC: always use max-L; 30s cannot reach >85% |

**Caption elements:** Annotate iso-compute lines on heatmap; annotate K* on metric_vs_k; label 480-min budget line on frontier.

**Note:** Use Transformer head for all 3 panels (best head, most illustrative). Include apnea_binary Transformer as a parallel set in S-Fig 3 for second example.

---

### Fig 3 — Task Landscape `[v3]`

**Named:** *"Task Landscape"*  
**Layout:** 1 row × 2 panels  
**Content:** Cross-task summary showing which tasks benefit most from context and how long they need.

| Panel | Plot type | Source file | Key message |
|---|---|---|---|
| (a) Scatter | `task_comparison_6A_scatter.png` | `task_comparison/` | 3 clusters: EEG-dominant+high-sensitivity (sleep_eff), strong+moderate (apnea/sex/age), weak (bmi/depression) |
| (b) L* lollipop | `task_comparison_6C_lstar.png` | `task_comparison/` | sleep_eff L*>240m (arrow); apnea L*≈80m; bmi L*≈40m; depression undefined (hatched) |

**Note:** 6B bars are **excluded** — information is redundant with Fig 1 and Table II.  
**Code fix:** Regenerate both without dropped tasks. Task labels on scatter points directly (no legend).  
Log-scale x-axis on lollipop to match Fig 1.

---

### Fig 4 — Modality Contribution `[v3_abl + v3 + v3_full references — CROSS-ROUND]`

**Named:** *"Modality Contribution"*  
**Layout:** 1 row × 5 panels (one per task), OR 2 rows × 3 (with one empty)  
**Content:** For each task: horizontal grouped bar chart of 5 ablation conditions (ΔAUROC from fast-ch baseline). Vertical reference lines at fast-ch baseline (solid) and full-ch ceiling (dashed).

| Ablation condition | Bar color |
|---|---|
| No BAS (RESP+EKG+EMG only) | Orange |
| No RESP (BAS+EKG+EMG only) | Blue |
| No EKG (BAS+RESP+EMG only) | Green |
| Cardio only (RESP+EKG) | Red |
| BAS only (EEG/EOG only) | Purple |

**Data source:** `analysis.csv` from phase0_v3_abl + baselines from phase0_v3 and phase0_v3_full.  
**Key messages per panel:**

| Panel | Task | Dominant finding |
|---|---|---|
| (a) | sleep_efficiency | BAS-only ≈ full (−0.005); cardio-only is worst (−0.111) |
| (b) | sex_binary | EKG most important (−0.074); BAS-only WORST (−0.092) |
| (c) | apnea_binary | RESP > BAS > EKG; BAS-only worst (−0.103) |
| (d) | age_class | BAS dominant (BAS-only −0.035); cardio-only worst (−0.069) |
| (e) | bmi_binary | RESP removal gives +0.010 (bars LEFT of zero); cardio-only worst (−0.081) |

**Status:** Awaiting v3_abl rerun completion (correct 128/1 architecture). Generate with:
```bash
python scripts/plot_modality_bar.py
```

---

## Supplementary Figures

> **No space limit in supplementary.** Figures are grouped only when they are naturally
> the same plot type compared across tasks/conditions, or when two panels genuinely
> answer one question together. Do not force-merge plots that serve different purposes
> just to save space.

---

### S-Fig 1 — Window Sweep per Task `[v3, Transformer, mean-pool line only]`

**Named:** *"K-Aggregation Curves"*  
**Layout:** 7 separate figures (one per task), OR one 7-panel composite if comparing tasks directly is the goal. Prefer separate — each task stands alone.  
**Content:** AUROC (mean_prob_auroc) vs K for 4 representative context lengths (30s, 40m, 120m, 240m). Subject mean-pool line only (remove segment-level and majority-vote for clarity).  
**Source:** `{task}_transformer_test_window_sweep_auroc.png` ×7 (phase0_v3).  
**Grouping rationale:** Same plot type across tasks; readers compare tasks by scanning panels. Natural composite if combined, or separate for ease of reading.

---

### S-Fig 2 — Fast vs Full Channel `[v3 vs v3_full, Transformer]`

**Named:** *"Channel Count Effect"*  
**Layout:** 2 rows × 3 cols (6 task panels; OSA gets a note label)  
**Content:** Saturation curves overlaid per task: fast-channel (dashed) and full-channel (solid), Transformer head. Annotate Δ at 240m on each panel.  
**Source:** Both `analysis.csv` files, after saturation code fix.  
**Key messages per panel:** apnea +4.7 pp; BMI +3.9 pp; sex +1.0 pp; sleep_eff −0.6 pp (neutral); OSA −4.4 pp (highlight); age +0.6 pp.  
**Grouping rationale:** Two conditions (fast/full) compared across tasks — naturally one panel per task with overlay lines.

---

### S-Fig 3 — Iso-Compute Analysis `[v3, sex + apnea, Transformer]`

**Named:** *"Iso-Compute Efficiency (Full)"*  
**Layout:** 2 rows × 3 cols (sex_binary row 1, apnea_binary row 2)  
**Content per row:** (a) metric_vs_total (AUROC vs L×K), (b) pareto_front, (c) marginal_gain  
**Source:** `{task}_transformer/auroc_test/` (phase0_v3).  
**Grouping rationale:** Same three-panel analysis for two tasks — showing the result is general. The two-row format lets readers compare tasks directly. Keep together.

---

### S-Fig 4a — ECE vs Context `[v3]`

**Named:** *"Calibration: ECE vs Context Length"*  
**Layout:** 2 rows × 3 cols (6 task panels, all 3 heads per panel)  
**Content:** ECE vs context (2B) for all 6 retained tasks.  
**Source:** `{task}_calibration_2B_ece_vs_context.png` ×6 (phase0_v3).  
**Grouping rationale:** Same plot type across tasks. Separate from 2A because it answers a different question (trend) vs 2A (absolute calibration).

---

### S-Fig 4b — Reliability Diagrams `[v3]`

**Named:** *"Calibration: Reliability Diagrams at 240m"*  
**Layout:** 1 row × 3 cols (apnea, sleep_efficiency, sex_binary)  
**Content:** 2A reliability diagrams at 240m (3 context panels each, all heads), showing whether predicted probabilities match true fractions.  
**Source:** `{task}_lstm_calibration_2A_reliability.png` ×3 (phase0_v3).  
**Grouping rationale:** Same plot type for the 3 most interesting calibration cases. Kept separate from 4a because 2A and 2B answer different questions.

---

### S-Fig 5 — Cohort Saturation `[v3, 3 tasks, LSTM]`

**Named:** *"Cross-Cohort Context Sensitivity"*  
**Layout:** 1 row × 3 cols (apnea, sleep_efficiency, sex_binary)  
**Content:** Per-cohort AUROC vs context (7A, LSTM head) for the 3 tasks with multi-cohort data.  
**Source:** `{task}_lstm_cohort_saturation_7A.png` ×3 (phase0_v3).  
**Grouping rationale:** Same plot type, comparing cohort differences across tasks. OSA (APPLES-only) omitted since single-cohort adds no comparison.

---

### S-Fig 6a — Prediction Variance Violins `[v3, 3 tasks, Transformer]`

**Named:** *"Within-Subject Prediction Variance"*  
**Layout:** 3 cols × 3 rows (3 tasks × 3 contexts: 30s, 120m, 240m)  
**Content:** 5A violin plots (within-subject std(prob), correct vs incorrect) for sex_binary, apnea_binary, sleep_efficiency.  
**Source:** `{task}_transformer_subject_consistency_5A_variance.png` ×3 (phase0_v3).

---

### S-Fig 6b — Hard-Subject Analysis `[v3, all 7 tasks, Transformer]`

**Named:** *"Hard-Subject Distribution"*  
**Layout:** 2 rows × 4 cols (7 task panels)  
**Content:** 5C bar charts — fraction of subjects correctly classified at 0, 1, 2, …, 6 of 6 context lengths. After redesign: plain English x-axis ("# context lengths correctly predicted"), cumulative or sorted bars.  
**Source:** `{task}_transformer_subject_consistency_5C_hard_subjects.png` ×7 (phase0_v3).  
**Note:** 5C redesigned (2026-07): x-axis now plain integer count, cumulative fraction line added on twin y-axis.

---

### S-Fig 7 — Window Position Profiles `[v3, 2 tasks, LSTM]`

**Named:** *"Prediction vs Night Position"*  
**Layout:** 2 rows × 3 cols (task × {positive subjects | negative subjects | variance 4B})  
**Content:** 4A position-probability profiles + 4B position variance for:
- Row 1: sleep_efficiency_lstm (early-night elevation — biologically meaningful)
- Row 2: sex_binary_lstm (flat — null control; confirms position-independence)

**Source:** `{task}_lstm_window_position_4A_profiles.png` + `{task}_lstm_window_position_4B_variance.png` (phase0_v3).  
**Grouping rationale:** 4A and 4B together answer "does position matter?" — 4A shows the mean, 4B shows the variance. Natural pair within each task row.

---

### S-Fig 8 — Compute Scaling Laws `[v3, 7 tasks]`

**Named:** *"Compute Scaling"*  
**Layout:** 2 rows × 4 cols (7 task panels)  
**Content:** 1B FLOPs vs test AUROC scatter, all heads and context lengths, power-law fits.  
**Source:** `scaling_laws/{task}_1B_compute_scaling.png` ×7 (phase0_v3).  
**Note:** 1C optimal epoch is **excluded** — non-monotonic. 1A uShape pending rerun with BA.

---

### S-Fig 9 — K* Minimum Windows `[v3, Transformer, 7 tasks]`

**Named:** *"Minimum Windows for Correct Classification"*  
**Layout:** 2 rows × 4 cols (7 task panels)  
**Content:** 9A K* histograms at 4 context lengths (30s, 40m, 120m, 240m), Transformer head.  
**Source:** `{task}_transformer_kstar_9A_histogram.png` ×7 (phase0_v3).

---

### S-Fig 10 — PR Curves `[v3, 6 tasks]`

**Named:** *"Precision-Recall Curves"*  
**Layout:** Two separate figures:  
- S-Fig 10a: 8A PR curves per context (all heads), 6 tasks (2×3 panel)  
- S-Fig 10b: 8B AUC-PR vs context (multi-head line), 6 tasks (2×3 panel)  
**Source:** `{task}_{head}_pr_8A_curves.png` + `{task}_pr_8B_aucpr_vs_context.png` (phase0_v3).  
**Grouping rationale:** 8A and 8B answer different questions (curve shape vs scalar trend), so separate figures. Within each, same plot type across tasks.

---

### S-Fig 11 — U-Shape Training Curves `[v3, Transformer, pending]`

**Named:** *"Training Convergence"*  
**Layout:** 2 rows × 4 cols (7 task panels)  
**Content:** 1A BA vs epoch (after rerun with BA metric replacing CE loss). One panel per task, Transformer head.  
**Status:** Ready — BA-metric fix applied 2026-07-01. Run Step 3 of `run_figures.sh` (includes `--plots 1A 1B`).

---

### S-Fig 12 (or Fig 5) — Aggregate Context-Length Scaling `[v3, all tasks × all heads]`

**Named:** *"General Context Scaling Law"*  
**Layout:** 1 row × 3 panels, `figsize=(14, 4.5)`  
**Script:** `scripts/plot_aggregate_scaling.py`  
**Status:** Ready to generate (saturation code fix applied). Placement decided after viewing std bands.
**Command:** `python scripts/plot_aggregate_scaling.py --collected-dir results/collected/phase0_v3 --results-dir /scratch/boshra95/psg/unified/results/phase0_v3`

**Content:**

| Panel | Type | Key question |
|---|---|---|
| (a) | ΔAUROC from 30s baseline vs context (log-x), one line per head ±1 std across tasks, faint individual task lines | Is there a consistent average gain curve? How variable is it across tasks? |
| (b) | Normalised gain (0%=30s, 100%=240m) vs context, same structure | What fraction of total achievable gain is captured at each context, regardless of task sensitivity? |
| (c) | Bar chart: log-linear slope b per head (pp per log₂ doubling), ±1 std across tasks, individual task dots overlaid | Is Transformer steeper than LSTM? Is the scaling rate consistent across tasks? |

**Placement rationale:**
- **Promote to main (Fig 5)** if: std bands in (a)/(b) are tight (< ±2 pp at 240m), meaning
  the scaling law is universal; and panel (c) shows a clear head ranking.
- **Keep as S-Fig 12** if: std bands are wide, showing the result is highly task-dependent
  (interesting, but not a "law"); the per-task story in Fig 1 is then the stronger contribution.
- Regardless of placement: panel (c) slope values are reportable in the main text as a
  one-sentence quantitative summary.

**Key parameters:**
- Tasks: all 7 retained (use `--exclude-tasks depression_extreme_binary` to check robustness
  without the non-monotonic outlier)
- Heads: lstm, transformer, mean_pool
- Metric: mean_prob_auroc (K=all, split=test)
- x-axis: log scale, standard context labels (30s → 240m)

**Data source:** `phase0_v3/collected/analysis.csv`

---

## Excluded from Paper Entirely

| Figure type | Reason |
|---|---|
| `*_calibration_2C_ece_vs_k.png` | Blacklisted |
| `*_subject_consistency_5B_variance_vs_k.png` | Blacklisted |
| `sex_binary_lstm_kstar_9B_coverage.png` | Blacklisted |
| `*_cohort_saturation_7B_n.png` | N documented in methods; not a result figure |
| `task_comparison_6B_bars.png` | Redundant with Fig 1 + Table II |
| `*_pr_8C_vote_sweep.png` | Majority-vote removed from paper |
| `auroc_test/double_tradeoff_auroc.png` | Redundant with heatmap + pareto |
| `*_1C_optimal_epoch.png` | Non-monotonic across all tasks; not interpretable |
| `balanced_accuracy_test/` all figures | BA versions marked supplementary in interpretation files; omit unless space permits |
| All v3_full per-condition detail figures | Channel story told through S-Fig 2 + Table II |
| All v3_abl per-condition individual figures | Ablation story told through Fig 4 + Table V |

---

## Figure-to-Interpretation-File Cross-Reference

| Paper location | Named figure | Source experiment | Interpretation file section |
|---|---|---|---|
| Fig 1 | Context Saturation | v3 | `figure_interpretations_v3.md` §Saturation Curves |
| Fig 2 | Inference Efficiency | v3 | `figure_interpretations_v3.md` §sex_binary_transformer §iso-compute |
| Fig 3 | Task Landscape | v3 | `figure_interpretations_v3.md` §Task Comparison |
| Fig 4 | Modality Contribution | v3_abl + refs | `figure_interpretations_v3_abl.md` §Summary Figures |
| S-Fig 1 | K-Aggregation Curves | v3 | `figure_interpretations_v3.md` §test_window_sweep rows |
| S-Fig 2 | Fast vs Full Channel | v3 + v3_full | `figure_interpretations_v3_full.md` §Saturation Curves |
| S-Fig 3 | Iso-Compute Analysis | v3 | `figure_interpretations_v3.md` §iso-compute (metric_vs_total, pareto, marginal_gain) |
| S-Fig 4 | Prediction Calibration | v3 | `figure_interpretations_v3.md` §calibration rows |
| S-Fig 5 | Cross-Cohort Saturation | v3 | `figure_interpretations_v3.md` §cohort_saturation_7A rows |
| S-Fig 6 | Per-Subject Stability | v3 | `figure_interpretations_v3.md` §subject_consistency 5A + 5C rows |
| S-Fig 7 | Temporal Position Profiles | v3 | `figure_interpretations_v3.md` §window_position_4A rows |
| S-Fig 8 | Compute Scaling | v3 | `figure_interpretations_v3.md` §Scaling Laws 1B rows |
| S-Fig 9 | Min Windows K* | v3 | `figure_interpretations_v3.md` §kstar_9A rows |
| S-Fig 12 / Fig 5 (TBD) | Aggregate Context Scaling | v3 | — (generate with `plot_aggregate_scaling.py`) |
