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


def save_cell(stem):
    """Separate cell: run this only when you're happy with the figure."""
    return code(f"""\
# ── Run this cell when the figure looks good ─────────────────────────────────
save_figure(fig, FINAL_OUT, "{stem}")
print("Saved →", FINAL_OUT / "{stem}.pdf")\
""")


INLINE_MAGIC = """\
%matplotlib inline
%load_ext autoreload
%autoreload 2\
"""

SETUP = """\
import sys
from pathlib import Path

# ── Workspace root: auto-detect by looking for final_results/ ─────────────────
def _find_workspace():
    \"\"\"Walk up from CWD until we find a directory containing final_results/.\"\"\"
    candidate = Path.cwd().resolve()
    for _ in range(10):
        if (candidate / "final_results").exists():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    # Explicit fallback (edit this if auto-detect fails)
    return Path("/Users/boshra/NSRR-workspace").resolve()

WORKSPACE_ROOT = _find_workspace()
NSRR_TOOLS     = WORKSPACE_ROOT / "NSRR-tools"
FINAL_RESULTS  = WORKSPACE_ROOT / "final_results"
PAPER_FIGURES  = NSRR_TOOLS / "results" / "paper_figures"
FINAL_OUT      = PAPER_FIGURES / "final"
FINAL_OUT.mkdir(parents=True, exist_ok=True)

# Add utils to path (notebooks/utils/)
_nb_dir = PAPER_FIGURES / "notebooks"
sys.path.insert(0, str(_nb_dir))

from utils.style import (
    apply_tbme_style, save_figure, FULL_W, HALF_W,
    MAIN_TASKS, SUPP_TASKS, ALL_TASKS, BINARY_MAIN,
    HEAD_STYLE, TASK_LABEL, FONT_ANNOT, FONT_BASE, FONT_LABEL,
)
from utils.data import set_root, load_analysis, load_heatmap, load_parquets
from utils import panels

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

set_root(WORKSPACE_ROOT)
apply_tbme_style()

# ── Confirm the workspace root is correct ─────────────────────────────────────
_ok = (WORKSPACE_ROOT / "final_results").exists()
print(f"WORKSPACE_ROOT : {WORKSPACE_ROOT}")
print(f"final_results/ : {'✓ found' if _ok else '✗ NOT FOUND — edit _find_workspace() fallback'}")\
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
# ROW_H controls how tall each row is (inches).
# Increase it to make panels bigger — try 2.5 or 3.0.
ROW_H = 2.1

# Build a subplot_mosaic layout so the last (incomplete) row is centred.
# Each panel label is repeated across 2 virtual columns; "." = ignored cell.
#
# Example for 5 tasks, N_COLS=3 (6 virtual columns):
#   a a b b c c
#   . d d e e .
labels  = [chr(97 + i) for i in range(len(TASKS))]
n_last  = len(TASKS) % N_COLS or N_COLS   # panels in the last row
n_full  = len(TASKS) // N_COLS

mosaic = []
for row in range(n_full):
    row_labels = labels[row * N_COLS : (row + 1) * N_COLS]
    mosaic.append([lbl for lbl in row_labels for _ in range(2)])

if n_last < N_COLS:   # incomplete → centre it
    last_labels = labels[n_full * N_COLS:]
    pad = N_COLS - n_last          # ignored cells on each side
    mosaic.append(
        ["."] * pad +
        [lbl for lbl in last_labels for _ in range(2)] +
        ["."] * pad
    )

fig, axd = plt.subplot_mosaic(mosaic, figsize=(FULL_W, N_ROWS * ROW_H))

for i, (lbl, task) in enumerate(zip(labels, TASKS)):
    ax = axd[lbl]
    panels.kvsk_panel(ax, hmaps[task], col=METRIC)
    ax.set_title(TASK_LABEL[task], fontsize=8)
    add_panel_label(ax, f"({chr(97+i)})")

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
       "**Layout**: 5 rows × 1 column (full 7-in width per heatmap)  \n\n"
       "Heatmap cell = AUROC (%) at that (context, K) combination.  "
       "Dashed lines = iso-compute budgets.\n\n"
       "**Tip**: `ROW_H` controls row height. Increase for more vertical "
       "breathing room, decrease for a more compact figure."),
    code(SETUP),
    _panel_label_cell(),
    code("""\
HEAD   = "transformer"
TASKS  = MAIN_TASKS   # 5 tasks → 2-2-1 layout
METRIC = "auroc"
ROW_H  = 2.0   # inches per row — increase for more vertical space

hmaps = {t: load_heatmap("phase0_v3", t, HEAD) for t in TASKS}
print("Loaded:", {t: len(v) for t, v in hmaps.items()})\
"""),
    code("""\
import matplotlib.gridspec as gridspec

# 2-2-1 layout: rows 0-1 have 2 panels each, row 2 has 1 centred panel.
fig = plt.figure(figsize=(FULL_W, 3 * ROW_H))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.8, wspace=0.4)

axes_flat = [
    fig.add_subplot(gs[0, 0:2]),   # (a) left
    fig.add_subplot(gs[0, 2:4]),   # (b) right
    fig.add_subplot(gs[1, 0:2]),   # (c) left
    fig.add_subplot(gs[1, 2:4]),   # (d) right
    fig.add_subplot(gs[2, 1:3]),   # (e) centred
]

# Per-panel flags: (show_ylabels, show_cbar_label)
#   Left panels  → y-labels ✓, cbar visible but no text label
#   Right panels → y-labels ✗, cbar visible with "AUROC (%)" label
#   Centred lone → y-labels ✓, cbar with label
panel_flags = [
    (True,  False),   # (a) left
    (False, True),    # (b) right
    (True,  False),   # (c) left
    (False, True),    # (d) right
    (True,  True),    # (e) centred
]

for i, (ax, task) in enumerate(zip(axes_flat, TASKS)):
    show_y, show_cbar_lbl = panel_flags[i]
    panels.heatmap_panel(ax, hmaps[task], col=METRIC,
                         show_ylabels=show_y, show_cbar_label=show_cbar_lbl)
    ax.set_title(TASK_LABEL[task], fontsize=8, pad=3)
    add_panel_label(ax, f"({chr(97+i)})")

save_figure(fig, FINAL_OUT, "main_fig3_heatmap")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main Fig 4 — Iso-main (vs_total + pareto, main tasks, Transformer)
# ─────────────────────────────────────────────────────────────────────────────

cells_fig4 = [
    md("# Main Fig 4 — Iso-compute: vs-Total + Pareto\n\n"
       "**Layout**: N_TASKS rows × 2 cols  \n"
       "Left col = AUROC vs total compute (L×K) with legend  \n"
       "Right col = Pareto-optimal frontier (text annotations, no legend)\n\n"
       "Each panel is FULL_W/2 = 3.5 in wide. Increase ROW_H for taller panels."),
    code(SETUP),
    _panel_label_cell(),
    code("""\
HEAD    = "transformer"
TASKS   = MAIN_TASKS
METRIC  = "auroc"
ROW_H   = 1.2   # inches per task row (5 rows × 1.2 = 6 in total → fits on a page)

hmaps = {t: load_heatmap("phase0_v3", t, HEAD) for t in TASKS}
print("Loaded:", {t: len(v) for t, v in hmaps.items()})\
"""),
    code("""\
fig, axes = plt.subplots(len(TASKS), 2, figsize=(FULL_W, len(TASKS) * ROW_H))

for row, task in enumerate(TASKS):
    ax_left  = axes[row, 0]   # vs_total
    ax_right = axes[row, 1]   # pareto

    panels.vs_total_panel(ax_left,  hmaps[task], col=METRIC)
    panels.pareto_panel  (ax_right, hmaps[task], col=METRIC)

    # Title only on the left panel
    ax_left.set_title(TASK_LABEL[task], fontsize=8)

    # Panel labels: a/b for row 0, c/d for row 1, …
    add_panel_label(ax_left,  f"({chr(97 + row * 2)})")
    add_panel_label(ax_right, f"({chr(97 + row * 2 + 1)})")

    # Legend: keep on left (vs_total) for first row only; suppress on pareto always
    if row > 0 and ax_left.get_legend():
        ax_left.get_legend().remove()
    if ax_right.get_legend():
        ax_right.get_legend().remove()

fig.tight_layout(h_pad=1.2, w_pad=1.0)
save_figure(fig, FINAL_OUT, "main_fig4_iso_main")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# Main Fig 5 — Min-cost frontier (main tasks)
# ─────────────────────────────────────────────────────────────────────────────

cells_fig5 = [
    md("# Main Fig 5 — Min-cost frontier\n\n"
       "**Layout**: 2 rows × 3 cols (mosaic), lower row centred  \n"
       "Legend appears once, outside panel (e) on the right.  \n"
       "Increase ROW_H for larger panels."),
    code(SETUP),
    _panel_label_cell(),
    code("""\
HEAD   = "transformer"
TASKS  = MAIN_TASKS
METRIC = "auroc"
N_COLS = 3
N_ROWS = (len(TASKS) + N_COLS - 1) // N_COLS
ROW_H  = 2.0   # inches per row

hmaps = {t: load_heatmap("phase0_v3", t, HEAD) for t in TASKS}\
"""),
    code("""\
# ── Mosaic: last row centred ──────────────────────────────────────────────────
labels = [chr(97 + i) for i in range(len(TASKS))]
n_last = len(TASKS) % N_COLS or N_COLS
n_full = len(TASKS) // N_COLS

mosaic = []
for row in range(n_full):
    row_labels = labels[row * N_COLS : (row + 1) * N_COLS]
    mosaic.append([lbl for lbl in row_labels for _ in range(2)])
if n_last < N_COLS:
    pad = N_COLS - n_last
    last_labels = labels[n_full * N_COLS:]
    mosaic.append(["."] * pad +
                  [lbl for lbl in last_labels for _ in range(2)] +
                  ["."] * pad)

fig, axd = plt.subplot_mosaic(mosaic, figsize=(FULL_W, N_ROWS * ROW_H))

# ── Draw panels ───────────────────────────────────────────────────────────────
for i, (lbl, task) in enumerate(zip(labels, TASKS)):
    panels.mincost_panel(axd[lbl], hmaps[task], col=METRIC)
    axd[lbl].set_title(TASK_LABEL[task], fontsize=8)
    add_panel_label(axd[lbl], f"({chr(97 + i)})")

# ── Collect legend handles BEFORE removing legends ────────────────────────────
handles, leg_labels = axd[labels[0]].get_legend_handles_labels()
for lbl in labels:
    if axd[lbl].get_legend():
        axd[lbl].get_legend().remove()

# ── Strip x-labels from top row (shared by bottom row) ───────────────────────
for lbl in labels[:N_COLS]:
    axd[lbl].set_xlabel("")
    axd[lbl].tick_params(labelbottom=False)

# ── Strip y-labels from non-leftmost panels of each row ──────────────────────
for i, lbl in enumerate(labels):
    if i % N_COLS != 0:
        axd[lbl].set_ylabel("")
        axd[lbl].tick_params(labelleft=False)

# ── Single legend outside bottom-right panel (e) ─────────────────────────────
axd[labels[-1]].legend(
    handles, leg_labels,
    title="Context L", title_fontsize=FONT_ANNOT,
    fontsize=FONT_ANNOT, frameon=False,
    loc="center left", bbox_to_anchor=(1.02, 0.5),
)

fig.tight_layout(h_pad=1.2, w_pad=1.0)
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
       "AUROC vs K at representative context lengths.  \n"
       "**Source**: analysis.csv (all k values)  \n"
       "**Tasks**: main + OSA (depression and CVD excluded)  \n"
       "**Layout**: 3 rows × 2 cols. Increase ROW_H for larger panels."),
    code(SETUP),
    _panel_label_cell(),
    code("""\
# Tasks: main 5 + OSA (exclude depression and CVD)
TASKS    = MAIN_TASKS + ["osa_binary_apples_postqc"]
CONTEXTS = ["40m", "120m", "240m"]   # context lengths shown per panel
HEADS    = ["lstm", "transformer"]
N_COLS   = 2
N_ROWS   = 3   # 6 tasks → 3 × 2, all rows full
ROW_H    = 2.4   # inches per row — increase for larger panels

import pandas as pd
_ana_path = WORKSPACE_ROOT / "final_results" / "phase0_v3" / "collected" / "analysis.csv"
df_all_k = pd.read_csv(_ana_path)
df_all_k["context_length_min"] = df_all_k["context_length"].map(
    lambda s: {"30s": 0.5, "10m": 10.0, "40m": 40.0,
               "80m": 80.0, "120m": 120.0, "240m": 240.0}.get(str(s).strip())
)
print("Tasks:", TASKS)
print("Loaded:", df_all_k.shape, "rows")\
"""),
    code("""\
labels = [chr(97 + i) for i in range(len(TASKS))]

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FULL_W, N_ROWS * ROW_H))
axes_flat = axes.flatten()

for i, (ax, task) in enumerate(zip(axes_flat, TASKS)):
    panels.k_agg_panel(ax, df_all_k, task, heads=HEADS, contexts=CONTEXTS)
    ax.set_title(TASK_LABEL.get(task, task), fontsize=8)
    add_panel_label(ax, f"({labels[i]})")

fig.tight_layout(h_pad=0.8, w_pad=1.0)
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
TASKS  = MAIN_TASKS   # 5 tasks → 2-2-1 layout
HEADS  = ["lstm", "transformer", "mean_pool"]
SPLIT  = "test"
FIG_W  = 10.0   # wider than TBME column — ok for supplementary/preview
ROW_H  = 3.0    # inches per row — increase for even bigger panels

df = load_analysis("phase0_v3", split=SPLIT, k="all")\
"""),
    code("""\
# Mosaic 2-2-1: each label spans 2 virtual columns; "." = empty cell
# a a b b
# c c d d
# . e e .
mosaic = [
    ["a", "a", "b", "b"],
    ["c", "c", "d", "d"],
    [".",  "e", "e", "."],
]
labels = ["a", "b", "c", "d", "e"]

fig, axd = plt.subplot_mosaic(mosaic, figsize=(FIG_W, 3 * ROW_H))

for i, (lbl, task) in enumerate(zip(labels, TASKS)):
    panels.saturation_panel(axd[lbl], df, task, heads=HEADS, show_values=True)
    axd[lbl].set_title(TASK_LABEL[task], fontsize=8)
    add_panel_label(axd[lbl], f"({lbl})")

# Shared legend above the figure
handles, leg_labels = axd["a"].get_legend_handles_labels()
for lbl in labels:
    if axd[lbl].get_legend():
        axd[lbl].get_legend().remove()
fig.legend(handles, leg_labels, loc="upper center", ncol=3,
           fontsize=FONT_ANNOT, bbox_to_anchor=(0.5, 1.02), frameon=False,
           handlelength=3.5)

fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.2, w_pad=1.0)
save_figure(fig, FINAL_OUT, "sfig2_saturation")
plt.show()\
"""),
]


# ─────────────────────────────────────────────────────────────────────────────
# S-Fig 3 — Compute scaling (all tasks, LSTM)
# ─────────────────────────────────────────────────────────────────────────────

cells_sfig3 = [
    md("# S-Fig 3 — Compute Scaling\n\n"
       "Test AUROC at best epoch vs total training FLOPs.  \n"
       "Point colour = context length (viridis), marker shape = head architecture.  \n"
       "**Source**: training.csv  \n"
       "**Tasks**: main + OSA (depression and CVD excluded)  \n"
       "Increase ROW_H for bigger panels."),
    code(SETUP),
    _panel_label_cell(),
    code("""\
from utils.data import load_training
TASKS  = MAIN_TASKS + ["osa_binary_apples_postqc"]   # exclude depression, cvd
HEADS  = ["lstm", "transformer", "mean_pool"]
N_COLS = 3
N_ROWS = (len(TASKS) + N_COLS - 1) // N_COLS
FIG_W  = 11.0   # wider than TBME column — ok for supplementary
ROW_H  = 3.0    # inches per row

df_train = load_training("phase0_v3")
print("Tasks:", TASKS)
print("training.csv shape:", df_train.shape)\
"""),
    code("""\
labels  = [chr(97 + i) for i in range(len(TASKS))]
n_last  = len(TASKS) % N_COLS or N_COLS
n_full  = len(TASKS) // N_COLS

mosaic = []
for row in range(n_full):
    rl = labels[row * N_COLS : (row + 1) * N_COLS]
    mosaic.append([l for l in rl for _ in range(2)])
if n_last < N_COLS:
    pad = N_COLS - n_last
    ll  = labels[n_full * N_COLS:]
    mosaic.append(["."] * pad + [l for l in ll for _ in range(2)] + ["."] * pad)

fig, axd = plt.subplot_mosaic(mosaic, figsize=(FIG_W, N_ROWS * ROW_H))

for i, (lbl, task) in enumerate(zip(labels, TASKS)):
    panels.compute_scaling_panel(axd[lbl], df_train, task, heads=HEADS)
    axd[lbl].set_title(TASK_LABEL.get(task, task), fontsize=8)
    add_panel_label(axd[lbl], f"({lbl})")

fig.tight_layout(h_pad=1.2, w_pad=1.0)
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
# NOTE: phase0_v3_abl was trained with LSTM only.
# Changing HEAD to "transformer" or "mean_pool" will produce empty plots
# because no ablation rows exist for those heads.
HEAD   = "lstm"   # ← only valid option for this figure
SPLIT  = "test"
TASKS  = MAIN_TASKS    # ablation only covers main tasks

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

abl_heads = sorted(df_abl["head"].unique())
print(f"Ablation heads available: {abl_heads}")
if HEAD not in abl_heads:
    print(f"WARNING: HEAD='{HEAD}' not in ablation data → figure will be empty!")\
"""),
    code("""\
N_COLS = 2
N_ROWS = (len(TASKS) + N_COLS - 1) // N_COLS
ROW_H  = 2.4   # inches per row

# Mosaic: 2-2-1 centred layout
labels = [chr(97 + i) for i in range(len(TASKS))]
n_last = len(TASKS) % N_COLS or N_COLS
n_full = len(TASKS) // N_COLS

mosaic = []
for row in range(n_full):
    rl = labels[row * N_COLS : (row + 1) * N_COLS]
    mosaic.append([l for l in rl for _ in range(2)])
if n_last < N_COLS:
    pad = N_COLS - n_last
    ll  = labels[n_full * N_COLS:]
    mosaic.append(["."] * pad + [l for l in ll for _ in range(2)] + ["."] * pad)

fig, axd = plt.subplot_mosaic(mosaic, figsize=(FULL_W, N_ROWS * ROW_H))

for i, (lbl, task) in enumerate(zip(labels, TASKS)):
    ax = axd[lbl]
    col_idx = i % N_COLS   # 0 = left column, 1 = right column
    panels.modality_bar_panel(ax, df_abl, df_fast, df_full, task, head=HEAD, split=SPLIT)
    ax.set_title(TASK_LABEL[task], fontsize=8)
    add_panel_label(ax, f"({lbl})")
    # Hide y-tick labels on right-column panels (left column carries them)
    if col_idx != 0:
        ax.set_yticklabels([])

fig.tight_layout(h_pad=1.0, w_pad=0.8)
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
fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 4.0))

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
HEAD     = "lstm"
SPLIT    = "test"
# 3 representative contexts overlaid per panel (30s shortest, 120m mid, 240m longest)
CONTEXTS = ["30s", "120m", "240m"]
K        = 5   # windows aggregated per subject (matches paper deployment)
TASKS    = [t for t in MAIN_TASKS if t != "age_class"]   # age is 3-class, skip
N_COLS   = 2
N_ROWS   = (len(TASKS) + N_COLS - 1) // N_COLS
ROW_H    = 2.6

pqs = {t: load_parquets("phase0_v3", t, HEAD, SPLIT) for t in TASKS}
print("Loaded tasks:", list(pqs.keys()))\
"""),
    code("""\
labels = [chr(97 + i) for i in range(len(TASKS))]
n_last = len(TASKS) % N_COLS or N_COLS
n_full = len(TASKS) // N_COLS

mosaic = []
for row in range(n_full):
    rl = labels[row * N_COLS : (row + 1) * N_COLS]
    mosaic.append([l for l in rl for _ in range(2)])
if n_last < N_COLS:
    pad = N_COLS - n_last; ll = labels[n_full * N_COLS:]
    mosaic.append(["."] * pad + [l for l in ll for _ in range(2)] + ["."] * pad)

fig, axd = plt.subplot_mosaic(mosaic, figsize=(FULL_W, N_ROWS * ROW_H))

for i, (lbl, task) in enumerate(zip(labels, TASKS)):
    panels.reliability_panel(axd[lbl], pqs[task], contexts=CONTEXTS, k=K)
    axd[lbl].set_title(TASK_LABEL.get(task, task), fontsize=8)
    add_panel_label(axd[lbl], f"({lbl})")

fig.tight_layout(h_pad=1.2, w_pad=1.0)
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

import re as _re

def postprocess(cells):
    """Transform cells for interactive Jupyter use:

    1. Inject ``%matplotlib inline`` as the first code cell (right after the
       title markdown), so plt.show() displays figures inline.

    2. For any code cell that calls save_figure(fig, FINAL_OUT, "STEM"):
       - Remove that line from the draw cell (plt.show() stays → figure shown).
       - Append a *separate* save cell that the user runs only when satisfied.
    """
    # Step 1: inject inline magic after the first markdown cell
    result = []
    injected = False
    for cell in cells:
        result.append(cell)
        if not injected and cell["cell_type"] == "markdown":
            result.append(code("%matplotlib inline"))
            injected = True

    # Step 2: split save_figure out of draw cells
    pat = _re.compile(r'^save_figure\(fig, FINAL_OUT, "(.+?)"\)\s*$', _re.MULTILINE)
    final = []
    for cell in result:
        if cell["cell_type"] != "code":
            final.append(cell)
            continue
        src = cell["source"]
        m = pat.search(src)
        if not m:
            final.append(cell)
            continue
        stem = m.group(1)
        # Draw cell: drop the save_figure line
        draw_src = "\n".join(
            ln for ln in src.splitlines() if not ln.startswith("save_figure(")
        ).rstrip()
        final.append({**cell, "id": _id(), "source": draw_src})
        # Save cell: separate, so user runs it deliberately
        final.append(code(
            f"# ── Run when figure looks good ──────────────────────────────────\n"
            f'save_figure(fig, FINAL_OUT, "{stem}")\n'
            f'print("Saved →", FINAL_OUT / "{stem}.pdf")'
        ))

    return final


if __name__ == "__main__":
    for name, cells in NOTEBOOKS.items():
        out = HERE / name
        out.write_text(json.dumps(nb(postprocess(cells)), indent=1, ensure_ascii=False))
        print(f"  wrote → {out.name}")
    print(f"\nDone.  {len(NOTEBOOKS)} notebooks written to {HERE}")
