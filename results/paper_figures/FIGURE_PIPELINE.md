# Paper Figure Pipeline

This document describes the full pipeline for generating all paper figures for the TBME submission.

---

## Directory Layout

```
NSRR-tools/results/paper_figures/
├── FIGURE_PIPELINE.md          ← this file
├── final/                      ← paper figure outputs (main + supplementary)
│   ├── main_fig2_kvsk.pdf
│   ├── main_fig3_heatmap.pdf
│   ├── main_fig4_iso_main.pdf
│   ├── main_fig5_mincost.pdf
│   ├── main_fig6_pr_curves.pdf
│   ├── sfig1_k_aggregation.pdf
│   ├── sfig2_saturation.pdf
│   ├── sfig3_compute_scaling.pdf
│   ├── sfig4_task_landscape.pdf
│   ├── sfig5_channel_comparison.pdf
│   ├── sfig6_modality_ablation.pdf
│   ├── sfig7_aggregate_scaling.pdf
│   ├── sfig8_variance_violins.pdf
│   ├── sfig9_aucpr.pdf
│   ├── sfig10_reliability.pdf
│   ├── sfig11_hard_subjects.pdf
│   └── sfig12_position_variance.pdf
├── explore/                    ← exploratory figures (xfig_* series, not yet in paper)
│   ├── final/                  ← saved xfig PDFs/PNGs (keep separate from above)
│   │   ├── xfig_02_threshold_unlock.pdf
│   │   ├── xfig_04_deployment_grid.pdf
│   │   ├── xfig_06_modality_radar.pdf
│   │   ├── xfig_08_night_fingerprint.pdf
│   │   ├── xfig_12_subject_stability.pdf
│   │   ├── xfig_14_task_clustermap.pdf
│   │   ├── xfig_19_ablation_clustermap.pdf
│   │   ├── xfig_25_sota_bubble.pdf
│   │   ├── xfig_28_significance_markers.pdf
│   │   └── xfig_30_waterfall.pdf
│   └── notebooks/
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── data_explore.py  ← data loaders for xfig series
│       │   └── panels_explore.py ← panel functions for xfig series
│       ├── run_all_figures.py   ← ✅ USE THIS to regenerate all xfig PDFs
│       ├── _create_notebooks_explore.py ← regenerates xfig *.ipynb files
│       ├── xfig_02_threshold_unlock.ipynb
│       ├── xfig_04_deployment_grid.ipynb
│       ├── xfig_06_modality_radar.ipynb
│       ├── xfig_08_night_fingerprint.ipynb
│       ├── xfig_12_subject_stability.ipynb
│       ├── xfig_14_task_clustermap.ipynb
│       ├── xfig_19_ablation_clustermap.ipynb
│       ├── xfig_25_sota_bubble.ipynb
│       ├── xfig_28_significance_markers.ipynb
│       └── xfig_30_waterfall.ipynb
└── notebooks/                  ← main/supp figure notebooks
    ├── utils/
    │   ├── __init__.py
    │   ├── style.py            ← TBME constants, task lists, style
    │   ├── data.py             ← data loading functions
    │   └── panels.py           ← all panel plotting functions
    ├── _create_notebooks.py    ← re-generates all .ipynb files from source
    ├── main_fig2_kvsk.ipynb
    ├── main_fig3_heatmap.ipynb
    ├── main_fig4_iso_main.ipynb
    ├── main_fig5_mincost.ipynb
    ├── main_fig6_pr_curves.ipynb
    ├── sfig1_k_aggregation.ipynb
    ├── sfig2_saturation.ipynb
    ├── sfig3_compute_scaling.ipynb
    ├── sfig4_task_landscape.ipynb
    ├── sfig5_channel_comparison.ipynb
    ├── sfig6_modality_ablation.ipynb
    ├── sfig7_aggregate_scaling.ipynb
    ├── sfig8_variance_violins.ipynb
    ├── sfig9_aucpr.ipynb
    ├── sfig10_reliability.ipynb
    ├── sfig11_hard_subjects.ipynb
    └── sfig12_position_variance.ipynb
```

---

## Design Principles

**One notebook per figure.** Each notebook:
1. Loads data from `final_results/` (CSV, parquets)
2. Creates a `matplotlib` figure at the exact TBME paper size
3. Calls panel functions from `utils/panels.py` to fill each subplot
4. Saves PDF + PNG to `results/paper_figures/final/`

**Panel functions take `ax` as first argument.** Functions in `panels.py` are self-contained:
```python
panels.kvsk_panel(ax, heatmap_df)
panels.saturation_panel(ax, analysis_df, task, heads=["lstm", "transformer"])
panels.mincost_panel(ax, heatmap_df)
```
This allows individual panels to be rearranged, resized, or reused across figures without rewriting any plotting code.

**TBME sizes:** `FULL_W = 7.0 in` (double column), `HALF_W = 3.5 in` (single column), 300 DPI.

---

## Data Sources

| Data | Location | Used by |
|---|---|---|
| `analysis.csv` (v3) | `final_results/phase0_v3/collected/analysis.csv` | S-Fig 2, 3, 4, 7 |
| `analysis.csv` (v3_full) | `final_results/phase0_v3_full/collected/analysis.csv` | S-Fig 5 |
| `analysis.csv` (v3_abl) | `final_results/phase0_v3_abl/collected/analysis.csv` | S-Fig 6 |
| `training.csv` (v3) | `final_results/phase0_v3/collected/training.csv` | S-Fig 3 |
| `heatmap_df_test.csv` | `final_results/phase0_v3/inference/{task}_{head}/heatmap_df_test.csv` | Main Fig 2–5 |
| Prediction parquets | `final_results/phase0_v3/collected/predictions/{task}_{head}_{ctx}_{split}.parquet` | Main Fig 6, S-Fig 8–12 |

---

## How to Run Figures

### Main + Supplementary figures (main paper, sfig*)

**Interactive (recommended for editing):**
```bash
# Open Jupyter using the project venv — use the full path, not just 'jupyter'
/Users/boshra/NSRR-workspace/NSRR-tools/.venv/bin/jupyter lab \
    results/paper_figures/notebooks/
```
Open the desired notebook (e.g., `main_fig2_kvsk.ipynb`), run all cells.
PDF is saved to `results/paper_figures/final/`.

> **Do NOT use `jupyter lab` or `jupyter nbconvert` from the system PATH.**
> The system Jupyter (Miniconda) uses a different Python that does not have
> matplotlib or the project packages. Always use the full venv path above.

---

### Exploratory figures (xfig_* series)

The exploratory figures use `run_all_figures.py` — a standalone script that does
not require `nbconvert`. It works reliably with the project venv:

**Regenerate all 10 xfig figures:**
```bash
cd /Users/boshra/NSRR-workspace/NSRR-tools/results/paper_figures/explore/notebooks
/Users/boshra/NSRR-workspace/NSRR-tools/.venv/bin/python run_all_figures.py
```

**Regenerate specific figures by number:**
```bash
# Single figure
/Users/boshra/NSRR-workspace/NSRR-tools/.venv/bin/python run_all_figures.py 30

# Multiple figures
/Users/boshra/NSRR-workspace/NSRR-tools/.venv/bin/python run_all_figures.py 08 12 25
```

Valid numbers: `02 04 06 08 12 14 19 25 28 30`

Output goes to `explore/final/xfig_NN_name.{pdf,png}`.

**Interactive editing of xfig notebooks:**
Open any `xfig_*.ipynb` in Jupyter using the same venv path above, but point to the
explore notebooks directory:
```bash
/Users/boshra/NSRR-workspace/NSRR-tools/.venv/bin/jupyter lab \
    results/paper_figures/explore/notebooks/
```

> `nbconvert --execute` does NOT work for either figure set because the venv does
> not have `nbconvert` installed. Use `run_all_figures.py` for batch regeneration
> of xfig figures, and Jupyter Lab (venv path) for interactive editing of all notebooks.

---

## How to Modify a Figure

### Change which tasks appear
In the notebook's configuration cell, edit `TASKS`:
```python
TASKS = MAIN_TASKS           # sex, bmi, age, sleep_eff, apnea
TASKS = ALL_TASKS            # all including depression, OSA, CVD
TASKS = ["sex_binary", "apnea_binary"]  # custom subset
```

### Change layout (rows × columns)
```python
N_COLS = 2   # change this; N_ROWS is computed automatically
N_ROWS = (len(TASKS) + N_COLS - 1) // N_COLS
```

### Change head (LSTM / Transformer / MeanPool)
```python
HEAD = "lstm"         # change this
HEAD = "transformer"
HEAD = "mean_pool"
```

### Change metric
```python
METRIC = "auroc"              # default
METRIC = "balanced_accuracy"  # also available in heatmap_df
```

### Change figure size
```python
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FULL_W, N_ROWS * 2.5))
#                                                            ^^^^^
#  Increase row height for more breathing room; decrease for compact layout.
#  FULL_W (7.0 in) should stay fixed for TBME double-column figures.
```

### Change output filename
```python
save_figure(fig, FINAL_OUT, "main_fig2_kvsk_v2")   # change stem
```

---

## Figure Catalogue

### Main Paper Figures

| # | Notebook | What it shows | Data |
|---|---|---|---|
| 1 | `main_fig1_placeholder.ipynb` | Pipeline schematic (manual) | — |
| 2 | `main_fig2_kvsk.ipynb` | AUROC vs K (iso-compute), 5 tasks | heatmap_df |
| 3 | `main_fig3_heatmap.ipynb` | 2D context × K heatmap, 5 tasks | heatmap_df |
| 4 | `main_fig4_iso_main.ipynb` | AUROC vs total compute + Pareto, 5 tasks | heatmap_df |
| 5 | `main_fig5_mincost.ipynb` | Min-cost frontier, 5 tasks | heatmap_df |
| 6 | `main_fig6_pr_curves.ipynb` | Precision-Recall curves, binary tasks | parquets |

### Supplementary Figures

| # | Notebook | What it shows | Data |
|---|---|---|---|
| 1 | `sfig1_k_aggregation.ipynb` | AUROC vs K at fixed context lengths, all tasks | analysis.csv |
| 2 | `sfig2_saturation.ipynb` | AUROC vs context length, all heads overlaid, main tasks | analysis.csv |
| 3 | `sfig3_compute_scaling.ipynb` | Test AUROC vs training compute proxy | training.csv |
| 4 | `sfig4_task_landscape.ipynb` | (a) difficulty vs sensitivity scatter; (b) L* lollipop | analysis.csv |
| 5 | `sfig5_channel_comparison.ipynb` | Fast-ch vs full-ch saturation overlay, all tasks | analysis.csv (v3 + v3_full) |
| 6 | `sfig6_modality_ablation.ipynb` | ΔAUROC per ablation condition, main tasks | analysis.csv (v3 + v3_abl) |
| 7 | `sfig7_aggregate_scaling.ipynb` | (a) ΔAUROC mean±std; (b) normalised gain; (c) slope bar | analysis.csv |
| 8 | `sfig8_variance_violins.ipynb` | Within-subject pred std, correct vs incorrect | parquets |
| 9 | `sfig9_aucpr.ipynb` | PR curves per task (maybe) | parquets |
| 10 | `sfig10_reliability.ipynb` | Reliability diagrams at 240m (maybe) | parquets |
| 11 | `sfig11_hard_subjects.ipynb` | Fraction correct at N contexts (maybe) | parquets |
| 12 | `sfig12_position_variance.ipynb` | Mean prob vs window position (maybe) | parquets |

### Exploratory Figures (xfig_* series)

Candidate figures for the paper — not yet assigned main/supp slots. See
`docs/NEW_PLOT_IDEAS.md` for full analysis, interpretation, and paper-fit verdict.

| # | Script key | What it shows | Status | Data |
|---|---|---|---|---|
| xfig_02 | `02` | Clinical threshold unlock map | ✅ safe to include | analysis.csv |
| xfig_04 | `04` | Deployment scenario grid (budget × required AUROC) | ✅ safe | heatmap_df |
| xfig_06 | `06` | Modality importance radar chart | ✅ safe | table6_modality.csv |
| xfig_08 | `08` | Night fingerprint heatmap (4 subject case studies) | ⚠️ frame carefully | parquets |
| xfig_12 | `12` | Subject prediction stability grid (all subjects) | ✅ safe | parquets |
| xfig_14 | `14` | Task similarity clustermap | ✅ safe | analysis.csv |
| xfig_19 | `19` | Modality ablation clustermap | ✅ safe | table6_modality.csv |
| xfig_25 | `25` | SOTA comparison bubble chart | ⚠️ caveats needed | hardcoded + analysis.csv |
| xfig_28 | `28` | Saturation curves with bootstrap significance markers | ⚠️ frame carefully | analysis.csv |
| xfig_30 | `30` | AUROC gain waterfall (aggregation + context + arch) | ✅ safe | analysis.csv |

---

## Task Definitions

```python
MAIN_TASKS = ["sex_binary", "bmi_binary", "age_class",
              "sleep_efficiency_binary", "apnea_binary"]

SUPP_TASKS = ["depression_extreme_binary", "osa_binary_apples_postqc", "cvd_binary"]

ALL_TASKS  = MAIN_TASKS + SUPP_TASKS
```

Task labels for display are in `utils/style.py: TASK_LABEL` (short) and `TASK_LABEL_LONG`.

---

## Experiment Key

| Experiment | Description | Results dir |
|---|---|---|
| `phase0_v3` | Fast-channel (7-8 ch), primary results | `final_results/phase0_v3/` |
| `phase0_v3_full` | Full-channel (up to 23 ch), channel expansion | `final_results/phase0_v3_full/` |
| `phase0_v3_abl` | Modality ablation (groups zeroed), reuses v3 embeddings | `final_results/phase0_v3_abl/` |

---

## LaTeX Integration

All figures go into `final/`. Reference them in LaTeX as:
```latex
\includegraphics[width=\columnwidth]{../../results/paper_figures/final/main_fig2_kvsk.pdf}
```
Or for double-column:
```latex
\includegraphics[width=\textwidth]{../../results/paper_figures/final/main_fig2_kvsk.pdf}
```

---

## Regenerating Notebooks from Scratch

If you edit `_create_notebooks.py` (e.g., to add a new figure or change cell content), regenerate all notebooks:
```bash
cd results/paper_figures/notebooks
python3 _create_notebooks.py
```
This overwrites all `.ipynb` files.  Any manual edits to notebooks will be lost — edit `_create_notebooks.py` instead.

---

## Adding a New Figure

1. Add a `cells_sfigN = [md(...), code(SETUP), code(...), ...]` block in `_create_notebooks.py`
2. Add the panel function to `utils/panels.py` with signature `panel_name(ax, data, ...)`
3. Add the new notebook name to the `NOTEBOOKS` dict in `_create_notebooks.py`
4. Run `python3 _create_notebooks.py` to generate the notebook
5. Test: open notebook in Jupyter and run all cells

---

## Requirements

```
matplotlib >= 3.7
pandas >= 2.0
numpy >= 1.24
seaborn >= 0.12
scikit-learn >= 1.2   # for PR curves and calibration
scipy >= 1.10         # for hierarchical clustering in xfig_14, xfig_19
```

All are installed in the project venv at `NSRR-tools/.venv/`.

**What is NOT in the venv:** `nbconvert`, `nbclient`, `jupyterlab`.
Do not use `jupyter nbconvert --execute` — it will fail with missing module errors
because it picks up the system Python. Use `run_all_figures.py` for batch runs
and the venv `jupyter lab` for interactive editing (see "How to Run" above).
