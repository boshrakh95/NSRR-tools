# Paper Figure Pipeline

This document describes the full pipeline for generating all paper figures for the TBME submission.

---

## Directory Layout

```
NSRR-tools/results/paper_figures/
├── FIGURE_PIPELINE.md          ← this file
├── final/                      ← all output PDFs and PNGs (LaTeX reads from here)
│   ├── main_fig1_pipeline_placeholder.pdf
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
│   ├── sfig9_aucpr.pdf         ← maybe
│   ├── sfig10_reliability.pdf  ← maybe
│   ├── sfig11_hard_subjects.pdf ← maybe
│   └── sfig12_position_variance.pdf ← maybe
└── notebooks/
    ├── utils/
    │   ├── __init__.py
    │   ├── style.py            ← TBME constants, task lists, style
    │   ├── data.py             ← data loading functions
    │   └── panels.py           ← all panel plotting functions
    ├── main_fig1_placeholder.ipynb
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
    ├── sfig12_position_variance.ipynb
    └── _create_notebooks.py    ← re-generates all .ipynb files from source
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

## How to Run a Single Figure

1. Open a terminal in `NSRR-tools/results/paper_figures/notebooks/`
2. Launch Jupyter:
   ```bash
   # activate your project venv first
   source /Users/boshra/NSRR-workspace/NSRR-tools/.venv/bin/activate
   jupyter lab
   ```
3. Open the desired notebook (e.g., `main_fig2_kvsk.ipynb`)
4. Run all cells (`Shift+Enter` or "Run All")
5. PDF is saved to `../final/main_fig2_kvsk.pdf`

**To run all figures non-interactively** (e.g., in a script):
```bash
cd results/paper_figures/notebooks
for nb in *.ipynb; do
    jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```

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
```

All are installed in the project venv at `NSRR-tools/.venv/`.
