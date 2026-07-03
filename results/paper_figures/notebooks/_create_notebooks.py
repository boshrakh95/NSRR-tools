#!/usr/bin/env python3
"""Helper script to (re-)create all paper-figure notebooks.

Run from the notebooks/ directory:
  python _create_notebooks.py

This is NOT meant to be run from inside Jupyter — it's a generator script
that creates the .ipynb files.  Run it once, then open the notebooks.
"""
import json, uuid, sys
from pathlib import Path

HERE = Path(__file__).parent


def _id():
    return uuid.uuid4().hex[:16]


def nb(cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }


def md(text):
    return {
        "cell_type": "markdown",
        "id": _id(),
        "metadata": {},
        "source": text,
    }


def code(src):
    return {
        "cell_type": "code",
        "id": _id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src,
    }


SETUP = """\
import sys
from pathlib import Path

# ── Workspace root (parent of NSRR-tools/) ────────────────────────────────────
WORKSPACE_ROOT = Path("../../../../..").resolve()   # adjust if notebook depth differs
NSRR_TOOLS     = WORKSPACE_ROOT / "NSRR-tools"
FINAL_RESULTS  = WORKSPACE_ROOT / "final_results"
PAPER_FIGURES  = NSRR_TOOLS / "results" / "paper_figures"
FINAL_OUT      = PAPER_FIGURES / "final"
FINAL_OUT.mkdir(parents=True, exist_ok=True)

# Add utils to path
sys.path.insert(0, str(Path(".").resolve()))

from utils.style import (
    apply_tbme_style, save_figure, FULL_W, HALF_W,
    MAIN_TASKS, SUPP_TASKS, ALL_TASKS, BINARY_MAIN,
    HEAD_STYLE, TASK_LABEL,
)
from utils.data import set_root, load_analysis, load_heatmap, load_parquets
from utils import panels

import matplotlib
matplotlib.use("Agg")   # comment out in Jupyter to get inline plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

set_root(WORKSPACE_ROOT)
apply_tbme_style()
print("Setup OK — workspace root:", WORKSPACE_ROOT)\
"""


# ── SHARED LABEL CELL ────────────────────────────────────────────────────────

def _panel_label_cell():
    return code("""\
# Panel labeling helper
def add_panel_label(ax, label, x=0.02, y=0.97):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", fontfamily="serif")\
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main Fig 1  — placeholder
# ─────────────────────────────────────────────────────────────────────────────

cells_fig1 = [
    md("# Main Fig 1 — Pipeline Diagram (placeholder)\n\n"
       "This figure will be created manually (schematic / diagram).\n"
       "No results data needed."),
    code(SETUP),
    code("""\
# Create a placeholder PDF so LaTeX compilation does not break.
fig, ax = plt.subplots(figsize=(FULL_W, 3.5))
ax.text(0.5, 0.5,
        "Fig 1: Processing + Training + Testing Pipeline\\n(to be created manually)",
        ha="center", va="center", fontsize=12, color="gray",
        transform=ax.transAxes)
ax.set_axis_off()
save_figure(fig, FINAL_OUT, "main_fig1_pipeline_placeholder")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main Fig 2 — AUROC vs K  (kvsk, main tasks, Transformer)
# ─────────────────────────────────────────────────────────────────────────────

cells_fig2 = [
    md("# Main Fig 2 — AUROC vs K (iso-compute)\n\n"
       "**Data**: `heatmap_df_test.csv` per task/head from `final_results/phase0_v3/inference/`  \n"
       "**Tasks**: main 5  \n"
       "**Head**: Transformer  \n"
       "**Layout**: 2 rows × 3 cols (5 panels, last cell hidden)\n\n"
       "To change: `HEAD`, `TASKS`, `N_COLS`, or `METRIC`."
       ),
    code(SETUP),
    _panel_label_cell(),
    code("""\
HEAD   = "transformer"
TASKS  = MAIN_TASKS          # change to ALL_TASKS for supplementary variant
METRIC = "auroc"
N_COLS = 3
N_ROWS = (len(TASKS) + N_COLS - 1) // N_COLS

# Load heatmap DataFrames
hmaps = {t: load_heatmap("phase0_v3", t, HEAD) for t in TASKS}
print({t: len(v) for t, v in hmaps.items()})\
"""),
    code("""\
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FULL_W, N_ROWS * 2.1))
axes_flat = axes.flatten()

for i, (ax, task) in enumerate(zip(axes_flat, TASKS)):
    panels.kvsk_panel(ax, hmaps[task], col=METRIC)
    ax.set_title(TASK_LABEL[task], fontsize=8)
    add_panel_label(ax, f"({chr(97+i)})")

# Hide unused axes
for ax in axes_flat[len(TASKS):]:
    ax.set_visible(False)

fig.tight_layout(h_pad=1.5, w_pad=1.0)
save_figure(fig, FINAL_OUT, "main_fig2_kvsk")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main Fig 3 — Heatmap (main tasks, Transformer)
# ─────────────────────────────────────────────────────────────────────────────

cells_fig3 = [
    md("# Main Fig 3 — Iso-compute Heatmap\n\n"
       "**Data**: `heatmap_df_test.csv`  \n"
       "**Tasks**: main 5  \n"
       "**Head**: Transformer  \n"
       "**Layout**: 2×3  \n\n"
       "Heatmap cell = AUROC (%) at that (context, K) combination.  "
       "Dashed lines = iso-compute budgets."),
    code(SETUP),
    _panel_label_cell(),
    code("""\
HEAD   = "transformer"
TASKS  = MAIN_TASKS
METRIC = "auroc"
N_COLS = 3
N_ROWS = (len(TASKS) + N_COLS - 1) // N_COLS

hmaps = {t: load_heatmap("phase0_v3", t, HEAD) for t in TASKS}\
"""),
    code("""\
# Each heatmap panel needs a bit more height than kvsk
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FULL_W, N_ROWS * 2.4))
axes_flat = axes.flatten()

for i, (ax, task) in enumerate(zip(axes_flat, TASKS)):
    panels.heatmap_panel(ax, hmaps[task], col=METRIC)
    ax.set_title(TASK_LABEL[task], fontsize=8)
    add_panel_label(ax, f"({chr(97+i)})")

for ax in axes_flat[len(TASKS):]:
    ax.set_visible(False)

fig.tight_layout(h_pad=1.8, w_pad=0.8)
save_figure(fig, FINAL_OUT, "main_fig3_heatmap")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main Fig 4 — Iso-main (vs_total + pareto, main tasks, Transformer)
# ─────────────────────────────────────────────────────────────────────────────

cells_fig4 = [
    md("# Main Fig 4 — Iso-compute: vs-Total + Pareto\n\n"
       "**Layout**: 2 rows × N_TASKS cols  \n"
       "Row 1 = AUROC vs total compute (L×K); Row 2 = Pareto-optimal frontier."),
    code(SETUP),
    _panel_label_cell(),
    code("""\
HEAD   = "transformer"
TASKS  = MAIN_TASKS
METRIC = "auroc"
N_TASKS = len(TASKS)

hmaps = {t: load_heatmap("phase0_v3", t, HEAD) for t in TASKS}\
"""),
    code("""\
fig, axes = plt.subplots(2, N_TASKS, figsize=(FULL_W, 4.0))

panel_idx = 0
for col, task in enumerate(TASKS):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]

    panels.vs_total_panel(ax_top, hmaps[task], col=METRIC)
    ax_top.set_title(TASK_LABEL[task], fontsize=8)
    add_panel_label(ax_top, f"({chr(97 + col)})")

    panels.pareto_panel(ax_bot, hmaps[task], col=METRIC)
    add_panel_label(ax_bot, f"({chr(97 + N_TASKS + col)})")

    # Only leftmost column gets y-labels
    if col > 0:
        ax_top.set_ylabel("")
        ax_bot.set_ylabel("")
    if col < N_TASKS - 1:
        ax_top.get_legend().remove() if ax_top.get_legend() else None
        ax_bot.get_legend().remove() if ax_bot.get_legend() else None

fig.tight_layout(h_pad=1.5, w_pad=0.8)
save_figure(fig, FINAL_OUT, "main_fig4_iso_main")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main Fig 5 — Min-cost frontier (main tasks)
# ─────────────────────────────────────────────────────────────────────────────

cells_fig5 = [
    md("# Main Fig 5 — Min-cost frontier\n\n"
       "**Question**: what is the cheapest way to reach a given AUROC level?  \n"
       "**Layout**: 1 row × 5 tasks (or 2×3 if preferred)"),
    code(SETUP),
    _panel_label_cell(),
    code("""\
HEAD   = "transformer"
TASKS  = MAIN_TASKS
METRIC = "auroc"
N_COLS = 3
N_ROWS = (len(TASKS) + N_COLS - 1) // N_COLS

hmaps = {t: load_heatmap("phase0_v3", t, HEAD) for t in TASKS}\
"""),
    code("""\
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FULL_W, N_ROWS * 2.1))
axes_flat = axes.flatten()

for i, (ax, task) in enumerate(zip(axes_flat, TASKS)):
    panels.mincost_panel(ax, hmaps[task], col=METRIC)
    ax.set_title(TASK_LABEL[task], fontsize=8)
    add_panel_label(ax, f"({chr(97+i)})")

for ax in axes_flat[len(TASKS):]:
    ax.set_visible(False)

fig.tight_layout(h_pad=1.5, w_pad=1.0)
save_figure(fig, FINAL_OUT, "main_fig5_mincost")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main Fig 6 — PR curves (binary main tasks)
# ─────────────────────────────────────────────────────────────────────────────

cells_fig6 = [
    md("# Main Fig 6 — Precision-Recall Curves\n\n"
       "**Data**: collected prediction parquets (phase0_v3)  \n"
       "**Tasks**: binary main tasks (sex, bmi, sleep_eff, apnea)  \n"
       "**Head**: Transformer  \n"
       "Requires sklearn."),
    code(SETUP),
    _panel_label_cell(),
    code("""\
HEAD   = "transformer"
TASKS  = BINARY_MAIN
SPLIT  = "test"
SHOW_CONTEXTS = ["30s", "40m", "120m", "240m"]
N_COLS = 2
N_ROWS = (len(TASKS) + N_COLS - 1) // N_COLS

# Load parquets for each task
pqs = {t: load_parquets("phase0_v3", t, HEAD, SPLIT) for t in TASKS}
print({t: list(v.keys()) for t, v in pqs.items()})\
"""),
    code("""\
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FULL_W, N_ROWS * 2.2))
axes_flat = axes.flatten()

for i, (ax, task) in enumerate(zip(axes_flat, TASKS)):
    panels.pr_curves_panel(ax, pqs[task], contexts=SHOW_CONTEXTS)
    ax.set_title(TASK_LABEL[task], fontsize=8)
    add_panel_label(ax, f"({chr(97+i)})")

for ax in axes_flat[len(TASKS):]:
    ax.set_visible(False)

fig.tight_layout(h_pad=1.5, w_pad=1.0)
save_figure(fig, FINAL_OUT, "main_fig6_pr_curves")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# S-Fig 1 — K-aggregation (all tasks)
# ─────────────────────────────────────────────────────────────────────────────

cells_sfig1 = [
    md("# S-Fig 1 — K-Aggregation\n\n"
       "AUROC vs K (number of windows aggregated), at representative context lengths.  \n"
       "**Source**: analysis.csv (all k values)  \n"
       "**Tasks**: all tasks"),
    code(SETUP),
    _panel_label_cell(),
    code("""\
HEAD     = "transformer"
TASKS    = ALL_TASKS
CONTEXTS = ["40m", "120m", "240m"]   # show these context lengths per panel
HEADS    = ["lstm", "transformer"]
N_COLS   = 4
N_ROWS   = (len(TASKS) + N_COLS - 1) // N_COLS

# Load full analysis.csv (all k values, not just k='all')
import pandas as pd
from pathlib import Path
_ana_path = WORKSPACE_ROOT / "final_results" / "phase0_v3" / "collected" / "analysis.csv"
df_all_k = pd.read_csv(_ana_path)
df_all_k["context_length_min"] = df_all_k["context_length"].map(
    lambda s: {"30s": 0.5, "10m": 10.0, "40m": 40.0,
               "80m": 80.0, "120m": 120.0, "240m": 240.0}.get(str(s).strip())
)
print("Loaded:", df_all_k.shape, "rows")\
"""),
    code("""\
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FULL_W, N_ROWS * 2.0))
axes_flat = axes.flatten()

for i, (ax, task) in enumerate(zip(axes_flat, TASKS)):
    panels.k_agg_panel(ax, df_all_k, task, heads=HEADS, contexts=CONTEXTS)
    ax.set_title(TASK_LABEL.get(task, task), fontsize=8)
    add_panel_label(ax, f"({chr(97+i)})")

for ax in axes_flat[len(TASKS):]:
    ax.set_visible(False)

fig.tight_layout(h_pad=1.5, w_pad=1.0)
save_figure(fig, FINAL_OUT, "sfig1_k_aggregation")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# S-Fig 2 — Saturation curves (main tasks, all heads overlaid)
# ─────────────────────────────────────────────────────────────────────────────

cells_sfig2 = [
    md("# S-Fig 2 — Saturation Curves (all heads overlaid)\n\n"
       "AUROC vs context length, LSTM / Transformer / MeanPool on the same axes.  \n"
       "**Source**: analysis.csv (k=all)  \n"
       "**Tasks**: main tasks"),
    code(SETUP),
    _panel_label_cell(),
    code("""\
TASKS  = MAIN_TASKS
HEADS  = ["lstm", "transformer", "mean_pool"]
SPLIT  = "test"
N_COLS = 3
N_ROWS = (len(TASKS) + N_COLS - 1) // N_COLS

df = load_analysis("phase0_v3", split=SPLIT, k="all")\
"""),
    code("""\
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FULL_W, N_ROWS * 2.1))
axes_flat = axes.flatten()

for i, (ax, task) in enumerate(zip(axes_flat, TASKS)):
    panels.saturation_panel(ax, df, task, heads=HEADS, show_values=True)
    ax.set_title(TASK_LABEL[task], fontsize=8)
    add_panel_label(ax, f"({chr(97+i)})")
    # Only first panel gets legend
    if i > 0 and ax.get_legend():
        ax.get_legend().remove()

# Shared legend
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=7,
           bbox_to_anchor=(0.5, 1.02), frameon=False)

for ax in axes_flat[len(TASKS):]:
    ax.set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.98], h_pad=1.5, w_pad=1.0)
save_figure(fig, FINAL_OUT, "sfig2_saturation")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# S-Fig 3 — Compute scaling (all tasks, LSTM)
# ─────────────────────────────────────────────────────────────────────────────

cells_sfig3 = [
    md("# S-Fig 3 — Compute Scaling\n\n"
       "Test AUROC at best epoch vs cumulative training compute, per context length.  \n"
       "**Source**: training.csv  \n"
       "**Tasks**: all tasks"),
    code(SETUP),
    _panel_label_cell(),
    code("""\
from utils.data import load_training
TASKS  = ALL_TASKS
HEADS  = ["lstm", "transformer"]
N_COLS = 4
N_ROWS = (len(TASKS) + N_COLS - 1) // N_COLS

df_train = load_training("phase0_v3")
print("training.csv columns:", list(df_train.columns)[:15])
print("tasks in training.csv:", sorted(df_train["task"].unique()) if "task" in df_train.columns else "no task col")\
"""),
    code("""\
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FULL_W, N_ROWS * 2.0))
axes_flat = axes.flatten()

for i, (ax, task) in enumerate(zip(axes_flat, TASKS)):
    panels.compute_scaling_panel(ax, df_train, task, heads=HEADS)
    ax.set_title(TASK_LABEL.get(task, task), fontsize=8)
    add_panel_label(ax, f"({chr(97+i)})")

for ax in axes_flat[len(TASKS):]:
    ax.set_visible(False)

fig.tight_layout(h_pad=1.5, w_pad=1.0)
save_figure(fig, FINAL_OUT, "sfig3_compute_scaling")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# S-Fig 4 — Task landscape (scatter + L*)
# ─────────────────────────────────────────────────────────────────────────────

cells_sfig4 = [
    md("# S-Fig 4 — Task Landscape\n\n"
       "Panel (a): scatter — task difficulty vs context sensitivity.  \n"
       "Panel (b): lollipop — saturation context L* per task.  \n"
       "**Source**: analysis.csv (k=all)  \n"
       "**Tasks**: all tasks"),
    code(SETUP),
    _panel_label_cell(),
    code("""\
TASKS  = ALL_TASKS
HEAD   = "lstm"
SPLIT  = "test"

df = load_analysis("phase0_v3", split=SPLIT, k="all")\
"""),
    code("""\
fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.0))

panels.task_scatter_panel(axes[0], df, tasks=TASKS, head=HEAD)
add_panel_label(axes[0], "(a)")
axes[0].set_title("Task difficulty vs context sensitivity", fontsize=8)

panels.lstar_panel(axes[1], df, tasks=TASKS, head=HEAD, tol=0.005)
add_panel_label(axes[1], "(b)")
axes[1].set_title("Saturation context L*", fontsize=8)

fig.tight_layout(w_pad=2.0)
save_figure(fig, FINAL_OUT, "sfig4_task_landscape")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# S-Fig 5 — Channel comparison (v3 vs v3_full)
# ─────────────────────────────────────────────────────────────────────────────

cells_sfig5 = [
    md("# S-Fig 5 — Channel Comparison (Fast vs Full)\n\n"
       "Fast-channel (7-8 channels) vs full-channel (up to 23 channels).  \n"
       "**Source**: analysis.csv from phase0_v3 and phase0_v3_full  \n"
       "**Tasks**: all tasks with full-channel results  \n"
       "**Head**: Transformer"),
    code(SETUP),
    _panel_label_cell(),
    code("""\
HEAD   = "transformer"
SPLIT  = "test"
TASKS  = ALL_TASKS   # only those with full-ch data will plot

df_fast = load_analysis("phase0_v3",      split=SPLIT, k="all")
df_full = load_analysis("phase0_v3_full", split=SPLIT, k="all")

# Check which tasks exist in full channel
full_tasks = df_full["task"].unique().tolist()
print("Tasks in full-ch:", sorted(full_tasks))
TASKS_PLOT = [t for t in TASKS if t in full_tasks]\
"""),
    code("""\
N_COLS = 3
N_ROWS = (len(TASKS_PLOT) + N_COLS - 1) // N_COLS

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FULL_W, N_ROWS * 2.1))
axes_flat = axes.flatten()

for i, (ax, task) in enumerate(zip(axes_flat, TASKS_PLOT)):
    panels.channel_comparison_panel(ax, df_fast, df_full, task, head=HEAD)
    ax.set_title(TASK_LABEL.get(task, task), fontsize=8)
    add_panel_label(ax, f"({chr(97+i)})")
    if i > 0 and ax.get_legend():
        ax.get_legend().remove()

# Shared legend
handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7,
           bbox_to_anchor=(0.5, 1.02), frameon=False)

for ax in axes_flat[len(TASKS_PLOT):]:
    ax.set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.5, w_pad=1.0)
save_figure(fig, FINAL_OUT, "sfig5_channel_comparison")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# S-Fig 6 — Modality ablation bar (v3 vs v3_abl)
# ─────────────────────────────────────────────────────────────────────────────

cells_sfig6 = [
    md("# S-Fig 6 — Modality Ablation Bar Chart\n\n"
       "ΔAUROC per ablation condition (No BAS, No RESP, etc.) relative to fast-ch baseline.  \n"
       "**Source**: phase0_v3_abl, phase0_v3, phase0_v3_full  \n"
       "**Tasks**: main tasks (ablation only covers these)  \n"
       "**Head**: LSTM"),
    code(SETUP),
    _panel_label_cell(),
    code("""\
HEAD   = "lstm"
SPLIT  = "test"
TASKS  = MAIN_TASKS    # ablation only covers main tasks

# For ablation, do NOT filter by k — load raw then pass to function
import pandas as pd
from pathlib import Path

def _load_raw(exp):
    p = WORKSPACE_ROOT / "final_results" / exp / "collected" / "analysis.csv"
    df = pd.read_csv(p)
    if "context_length_min" not in df.columns:
        df["context_length_min"] = df["context_length"].map(
            {"30s": 0.5, "10m": 10.0, "40m": 40.0,
             "80m": 80.0, "120m": 120.0, "240m": 240.0}.get)
    return df

df_abl  = _load_raw("phase0_v3_abl")
df_fast = _load_raw("phase0_v3")
df_full = _load_raw("phase0_v3_full")

print("Ablation run_tags:", sorted(df_abl["run_tag"].unique()))\
"""),
    code("""\
N_COLS = 5   # one column per task side-by-side (classic modality bar layout)
fig, axes = plt.subplots(1, N_COLS, figsize=(FULL_W, 2.8), sharey=False)

for col, (ax, task) in enumerate(zip(axes, TASKS)):
    panels.modality_bar_panel(ax, df_abl, df_fast, df_full, task, head=HEAD, split=SPLIT)
    ax.set_title(TASK_LABEL[task], fontsize=8)
    add_panel_label(ax, f"({chr(97+col)})")
    if col > 0:
        ax.set_yticklabels([])

fig.tight_layout(w_pad=0.5)
save_figure(fig, FINAL_OUT, "sfig6_modality_ablation")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# S-Fig 7 — Aggregate scaling (delta + norm + slope bar)
# ─────────────────────────────────────────────────────────────────────────────

cells_sfig7 = [
    md("# S-Fig 7 — Aggregate Context-Length Scaling\n\n"
       "(a) ΔAUROC from 30s baseline, mean ± std across tasks.  \n"
       "(b) Normalised gain (0 = 30s, 100 = 240m).  \n"
       "(c) Log-linear slope per head.  \n"
       "**Source**: analysis.csv (k=all)"),
    code(SETUP),
    _panel_label_cell(),
    code("""\
TASKS  = ALL_TASKS
HEADS  = ["lstm", "transformer", "mean_pool"]
SPLIT  = "test"

df = load_analysis("phase0_v3", split=SPLIT, k="all")\
"""),
    code("""\
fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 3.0))

panels.delta_panel(axes[0], df, tasks=TASKS, heads=HEADS)
add_panel_label(axes[0], "(a)")

panels.norm_panel(axes[1], df, tasks=TASKS, heads=HEADS)
add_panel_label(axes[1], "(b)")
if axes[1].get_legend():
    axes[1].get_legend().remove()

panels.slope_bar_panel(axes[2], df, tasks=TASKS, heads=HEADS)
add_panel_label(axes[2], "(c)")

fig.tight_layout(w_pad=1.5)
save_figure(fig, FINAL_OUT, "sfig7_aggregate_scaling")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# S-Fig 8 — Variance violins (all tasks)
# ─────────────────────────────────────────────────────────────────────────────

cells_sfig8 = [
    md("# S-Fig 8 — Within-Subject Variance Violins\n\n"
       "Distribution of within-subject prediction std(prob) for correct vs incorrect subjects.  \n"
       "**Source**: collected prediction parquets  \n"
       "**Head**: Transformer  \n"
       "**Tasks**: all tasks with parquets available"),
    code(SETUP),
    _panel_label_cell(),
    code("""\
HEAD   = "transformer"
SPLIT  = "test"
TASKS  = ALL_TASKS
SHOW_CONTEXTS = ["30s", "40m", "120m", "240m"]
N_COLS = 4
N_ROWS = (len(TASKS) + N_COLS - 1) // N_COLS

# Load parquets (only contexts we care about)
pqs = {}
for task in TASKS:
    p = load_parquets("phase0_v3", task, HEAD, SPLIT)
    if p:
        pqs[task] = p
print("Tasks with parquets:", sorted(pqs.keys()))\
"""),
    code("""\
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FULL_W, N_ROWS * 2.2))
axes_flat = axes.flatten()

plot_tasks = [t for t in TASKS if t in pqs]
for i, (ax, task) in enumerate(zip(axes_flat, plot_tasks)):
    panels.variance_violin_panel(ax, pqs[task], contexts=SHOW_CONTEXTS)
    ax.set_title(TASK_LABEL.get(task, task), fontsize=8)
    add_panel_label(ax, f"({chr(97+i)})")
    if i > 0 and ax.get_legend():
        ax.get_legend().remove()

handles, labels = axes_flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7,
           bbox_to_anchor=(0.5, 1.02), frameon=False)

for ax in axes_flat[len(plot_tasks):]:
    ax.set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.5, w_pad=1.0)
save_figure(fig, FINAL_OUT, "sfig8_variance_violins")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# S-Fig 9 — AUC-PR (maybe)
# ─────────────────────────────────────────────────────────────────────────────

cells_sfig9 = [
    md("# S-Fig 9 — AUC-PR vs Context (maybe)\n\n"
       "Precision-Recall curves at each context, one panel per task.  \n"
       "**Source**: collected prediction parquets  \n"
       "**Tasks**: binary main + binary supp"),
    code(SETUP),
    _panel_label_cell(),
    code("""\
HEAD   = "transformer"
SPLIT  = "test"
TASKS  = BINARY_MAIN
SHOW_CONTEXTS = ["30s", "10m", "40m", "80m", "120m", "240m"]
N_COLS = 2
N_ROWS = (len(TASKS) + N_COLS - 1) // N_COLS

pqs = {t: load_parquets("phase0_v3", t, HEAD, SPLIT) for t in TASKS}\
"""),
    code("""\
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FULL_W, N_ROWS * 2.2))
axes_flat = axes.flatten()

for i, (ax, task) in enumerate(zip(axes_flat, TASKS)):
    panels.pr_curves_panel(ax, pqs[task], contexts=SHOW_CONTEXTS)
    ax.set_title(TASK_LABEL.get(task, task), fontsize=8)
    add_panel_label(ax, f"({chr(97+i)})")

for ax in axes_flat[len(TASKS):]:
    ax.set_visible(False)

fig.tight_layout(h_pad=1.5, w_pad=1.0)
save_figure(fig, FINAL_OUT, "sfig9_aucpr")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# S-Fig 10 — Reliability diagrams (maybe)
# ─────────────────────────────────────────────────────────────────────────────

cells_sfig10 = [
    md("# S-Fig 10 — Reliability Diagrams (maybe)\n\n"
       "Calibration check: predicted probability vs actual fraction positive.  \n"
       "**Source**: collected parquets at context=240m  \n"
       "**Tasks**: main tasks"),
    code(SETUP),
    _panel_label_cell(),
    code("""\
HEAD    = "lstm"
SPLIT   = "test"
CONTEXT = "240m"   # show calibration at longest context
TASKS   = MAIN_TASKS
N_COLS  = 3
N_ROWS  = (len(TASKS) + N_COLS - 1) // N_COLS

pqs = {t: load_parquets("phase0_v3", t, HEAD, SPLIT) for t in TASKS}\
"""),
    code("""\
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FULL_W, N_ROWS * 2.2))
axes_flat = axes.flatten()

for i, (ax, task) in enumerate(zip(axes_flat, TASKS)):
    panels.reliability_panel(ax, pqs[task], context=CONTEXT)
    ax.set_title(TASK_LABEL.get(task, task), fontsize=8)
    add_panel_label(ax, f"({chr(97+i)})")

for ax in axes_flat[len(TASKS):]:
    ax.set_visible(False)

fig.tight_layout(h_pad=1.5, w_pad=1.0)
save_figure(fig, FINAL_OUT, "sfig10_reliability")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# S-Fig 11 — Hard subjects (maybe)
# ─────────────────────────────────────────────────────────────────────────────

cells_sfig11 = [
    md("# S-Fig 11 — Hard Subjects (maybe)\n\n"
       "Fraction of subjects correctly predicted at 0, 1, …, N context lengths.  \n"
       "**Source**: collected parquets  \n"
       "**Tasks**: main tasks"),
    code(SETUP),
    _panel_label_cell(),
    code("""\
HEAD   = "transformer"
SPLIT  = "test"
TASKS  = MAIN_TASKS
N_COLS = 3
N_ROWS = (len(TASKS) + N_COLS - 1) // N_COLS

pqs = {t: load_parquets("phase0_v3", t, HEAD, SPLIT) for t in TASKS}\
"""),
    code("""\
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FULL_W, N_ROWS * 2.2))
axes_flat = axes.flatten()

for i, (ax, task) in enumerate(zip(axes_flat, TASKS)):
    panels.hard_subjects_panel(ax, pqs[task])
    ax.set_title(TASK_LABEL.get(task, task), fontsize=8)
    add_panel_label(ax, f"({chr(97+i)})")

for ax in axes_flat[len(TASKS):]:
    ax.set_visible(False)

fig.tight_layout(h_pad=1.5, w_pad=1.0)
save_figure(fig, FINAL_OUT, "sfig11_hard_subjects")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# S-Fig 12 — Window position (maybe)
# ─────────────────────────────────────────────────────────────────────────────

cells_sfig12 = [
    md("# S-Fig 12 — Window Position Profiles (maybe)\n\n"
       "Mean predicted probability vs normalised position in the night recording.  \n"
       "**Source**: collected parquets (need window_idx column)  \n"
       "**Tasks**: main tasks"),
    code(SETUP),
    _panel_label_cell(),
    code("""\
HEAD   = "lstm"
SPLIT  = "test"
TASKS  = MAIN_TASKS
SHOW_CONTEXTS = ["30s", "120m", "240m"]
N_COLS = 3
N_ROWS = (len(TASKS) + N_COLS - 1) // N_COLS

pqs = {t: load_parquets("phase0_v3", t, HEAD, SPLIT) for t in TASKS}\
"""),
    code("""\
fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FULL_W, N_ROWS * 2.1))
axes_flat = axes.flatten()

for i, (ax, task) in enumerate(zip(axes_flat, TASKS)):
    panels.position_panel(ax, pqs[task], contexts=SHOW_CONTEXTS)
    ax.set_title(TASK_LABEL.get(task, task), fontsize=8)
    add_panel_label(ax, f"({chr(97+i)})")
    if i > 0 and ax.get_legend():
        ax.get_legend().remove()

for ax in axes_flat[len(TASKS):]:
    ax.set_visible(False)

fig.tight_layout(h_pad=1.5, w_pad=1.0)
save_figure(fig, FINAL_OUT, "sfig12_position_variance")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# Write all notebooks
# ─────────────────────────────────────────────────────────────────────────────

NOTEBOOKS = {
    "main_fig1_placeholder.ipynb": cells_fig1,
    "main_fig2_kvsk.ipynb":        cells_fig2,
    "main_fig3_heatmap.ipynb":     cells_fig3,
    "main_fig4_iso_main.ipynb":    cells_fig4,
    "main_fig5_mincost.ipynb":     cells_fig5,
    "main_fig6_pr_curves.ipynb":   cells_fig6,
    "sfig1_k_aggregation.ipynb":   cells_sfig1,
    "sfig2_saturation.ipynb":      cells_sfig2,
    "sfig3_compute_scaling.ipynb": cells_sfig3,
    "sfig4_task_landscape.ipynb":  cells_sfig4,
    "sfig5_channel_comparison.ipynb": cells_sfig5,
    "sfig6_modality_ablation.ipynb":  cells_sfig6,
    "sfig7_aggregate_scaling.ipynb":  cells_sfig7,
    "sfig8_variance_violins.ipynb":   cells_sfig8,
    "sfig9_aucpr.ipynb":              cells_sfig9,
    "sfig10_reliability.ipynb":       cells_sfig10,
    "sfig11_hard_subjects.ipynb":     cells_sfig11,
    "sfig12_position_variance.ipynb": cells_sfig12,
}

if __name__ == "__main__":
    for name, cells in NOTEBOOKS.items():
        out = HERE / name
        out.write_text(json.dumps(nb(cells), indent=1, ensure_ascii=False))
        print(f"  wrote → {out.name}")
    print(f"\nDone.  {len(NOTEBOOKS)} notebooks written to {HERE}")
