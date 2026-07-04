#!/usr/bin/env python3
"""Generate all exploratory figure notebooks (xfig_* series).

Run from this directory:
    python _create_notebooks_explore.py

Figures correspond to idea numbers in docs/NEW_PLOT_IDEAS.md.
Output notebooks go into the same directory; figures go to ../final/.
Do NOT rename outputs to main_fig* or sfig* until you decide to keep them.
"""

import json
import uuid
from pathlib import Path

HERE = Path(__file__).parent
FINAL_OUT_DIR = HERE.parent / "final"


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
    return {"cell_type": "markdown", "id": _id(), "metadata": {}, "source": text}


def code(src):
    return {
        "cell_type": "code",
        "id": _id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src,
    }


# ── Shared setup cell injected at the top of every notebook ──────────────────

SETUP = """\
import sys
from pathlib import Path

# ── Workspace root: auto-detect by looking for final_results/ ─────────────────
def _find_workspace():
    candidate = Path.cwd().resolve()
    for _ in range(10):
        if (candidate / "final_results").exists():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return Path("/Users/boshra/NSRR-workspace").resolve()

WORKSPACE_ROOT = _find_workspace()
NSRR_TOOLS     = WORKSPACE_ROOT / "NSRR-tools"
EXPLORE_DIR    = NSRR_TOOLS / "results" / "paper_figures" / "explore"
FINAL_OUT      = EXPLORE_DIR / "final"
FINAL_OUT.mkdir(parents=True, exist_ok=True)
TABLES_DIR     = NSRR_TOOLS / "results" / "tables"

# Add explore utils to path
_nb_dir = EXPLORE_DIR / "notebooks"
sys.path.insert(0, str(_nb_dir))

from utils.data_explore import (
    set_root, load_analysis, load_analysis_all_k,
    load_heatmap, load_parquets, load_modality_table,
    subject_predictions, subject_correctness_matrix, CONTEXT_TO_MIN, CTX_ORDER,
)
from utils import panels_explore as xp

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

set_root(WORKSPACE_ROOT)
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 7,
})

# ── Constants ──────────────────────────────────────────────────────────────────
MAIN_TASKS = ["sex_binary", "bmi_binary", "age_class",
              "sleep_efficiency_binary", "apnea_binary"]
TASK_LABEL = xp.TASK_LABEL

_ok = (WORKSPACE_ROOT / "final_results").exists()
print(f"WORKSPACE_ROOT : {WORKSPACE_ROOT}")
print(f"final_results/ : {'✓ found' if _ok else '✗ NOT FOUND'}")
"""


def _save_cell(stem):
    return code(
        f"# ── Run when figure looks good ──────────────────────────────────\n"
        f"fig.savefig(str(FINAL_OUT / '{stem}.pdf'), bbox_inches='tight')\n"
        f"fig.savefig(str(FINAL_OUT / '{stem}.png'), dpi=150, bbox_inches='tight')\n"
        f"print('Saved →', FINAL_OUT / '{stem}.pdf')"
    )


def _panel_label(ax_expr, lbl):
    return (f"{ax_expr}.text(0.02, 0.97, '{lbl}', transform={ax_expr}.transAxes,\n"
            f"    fontsize=8, fontweight='bold', va='top', fontfamily='serif')")


def postprocess(cells):
    """Inject %matplotlib inline after first markdown cell."""
    result = []
    injected = False
    for cell in cells:
        result.append(cell)
        if not injected and cell["cell_type"] == "markdown":
            result.append(code("%matplotlib inline\n%load_ext autoreload\n%autoreload 2"))
            injected = True
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_02 — Clinical Threshold Unlock Map
# ═══════════════════════════════════════════════════════════════════════════════

cells_xfig02 = [
    md("# xfig_02 — Clinical Threshold Unlock Map\n\n"
       "For each task (column) and required AUROC threshold (row): which is the\n"
       "**first context length** where performance first meets that threshold?\n\n"
       "Gray = never reached. Color (viridis_r) = shorter context = easier to achieve.\n\n"
       "Idea #2 from `docs/NEW_PLOT_IDEAS.md`.\n\n"
       "**Data**: `analysis.csv`, k=all, split=test."),
    code(SETUP),
    code("""\
# ── Config ────────────────────────────────────────────────────────────────────
TASKS      = MAIN_TASKS   # change to include supp tasks if desired
HEAD       = "transformer"
THRESHOLDS = [0.70, 0.75, 0.80, 0.85, 0.90]

df = load_analysis("phase0_v3", split="test", k="all")
print("Loaded analysis.csv:", df.shape, "rows")
print("Heads:", sorted(df["head"].unique()))
print("Tasks:", sorted(df["task"].unique()))\
"""),
    code("""\
fig, ax = plt.subplots(figsize=(7.0, 3.5))

xp.threshold_unlock_heatmap(
    ax, df,
    tasks=TASKS,
    head=HEAD,
    thresholds=THRESHOLDS,
)
ax.set_title(
    f"First context length to reach target AUROC ({HEAD})",
    fontsize=8, pad=6,
)
fig.tight_layout()
plt.show()\
"""),
    _save_cell("xfig_02_threshold_unlock"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_04 — Deployment Scenario Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

cells_xfig04 = [
    md("# xfig_04 — Deployment Scenario Heatmap\n\n"
       "For a given **recording budget** (columns) and **required AUROC** (rows): "
       "what is the best achievable AUROC and which (L, K) configuration achieves it?\n\n"
       "Each panel = one task. Cells show the best AUROC achievable with that budget\n"
       "and the (L, K) configuration. Green = target met; red = target not met.\n\n"
       "Idea #4 from `docs/NEW_PLOT_IDEAS.md`.\n\n"
       "**Data**: `heatmap_df_test.csv` (dense K sweep) per task."),
    code(SETUP),
    code("""\
# ── Config ────────────────────────────────────────────────────────────────────
TASKS   = MAIN_TASKS
HEAD    = "transformer"
BUDGETS = [30, 60, 120, 240, 480]     # total signal minutes
TARGETS = [0.70, 0.75, 0.80, 0.85]   # required AUROC

hmaps = {t: load_heatmap("phase0_v3", t, HEAD) for t in TASKS}
print("Heatmap sizes:", {t: len(v) for t, v in hmaps.items()})\
"""),
    code("""\
N_COLS = 3
N_ROWS = (len(TASKS) + N_COLS - 1) // N_COLS
ROW_H  = 2.4

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(7.0, N_ROWS * ROW_H))
axes_flat = axes.flatten()

for i, (ax, task) in enumerate(zip(axes_flat, TASKS)):
    xp.deployment_scenario_panel(ax, hmaps[task], task,
                                  budgets_min=BUDGETS, targets=TARGETS)
    ax.set_title(TASK_LABEL.get(task, task), fontsize=8)
    ax.text(0.02, 0.97, f"({chr(97+i)})", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top")

for ax in axes_flat[len(TASKS):]:
    ax.set_visible(False)

fig.suptitle("Optimal (L, K) strategy per budget and required AUROC (Transformer)",
             fontsize=8, y=1.01)
fig.tight_layout(h_pad=1.5, w_pad=1.2)
plt.show()\
"""),
    _save_cell("xfig_04_deployment_grid"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_06 — Modality Radar Chart
# ═══════════════════════════════════════════════════════════════════════════════

cells_xfig06 = [
    md("# xfig_06 — Modality Radar Chart\n\n"
       "Polar/radar chart where each axis = modality importance for a task.\n"
       "Importance = |ΔAUROC| when that modality is removed (from ablation table).\n"
       "Longer spoke = that modality contributes more to this task.\n\n"
       "One polygon per task. 5 spokes: No BAS, No RESP, No EKG, Cardio only, BAS only.\n\n"
       "Idea #6 from `docs/NEW_PLOT_IDEAS.md`.\n\n"
       "**Data**: `results/tables/table6_modality.csv`."),
    code(SETUP),
    code("""\
# ── Load modality table ───────────────────────────────────────────────────────
mod_df = load_modality_table(NSRR_TOOLS)
print(mod_df.to_string())
print("\\nColumns:", mod_df.columns.tolist())\
"""),
    code("""\
# Task names in the table vs internal keys — map them
TASKS = MAIN_TASKS

# Map internal task keys to the 'Task' column values in table6_modality.csv
TASK_TABLE_MAP = {
    "sex_binary":                "Sex",
    "apnea_binary":              "Sleep apnea",
    "sleep_efficiency_binary":   "Sleep efficiency",
    "age_class":                 "Age",
    "bmi_binary":                "BMI",
}

fig = plt.figure(figsize=(5.5, 5.5))
ax = fig.add_subplot(111, projection="polar")

handles, labels = xp.modality_radar_panel(ax, mod_df, tasks=TASKS)
ax.legend(handles, [TASK_LABEL.get(t, t) for t in TASKS],
          loc="upper left", bbox_to_anchor=(1.15, 1.1),
          fontsize=7, frameon=False)
ax.set_title("Modality importance: |ΔAUROC| when modality removed",
             fontsize=8, pad=15)
fig.tight_layout()
plt.show()\
"""),
    _save_cell("xfig_06_modality_radar"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_08 — Night Fingerprint Heatmap
# ═══════════════════════════════════════════════════════════════════════════════

cells_xfig08 = [
    md("# xfig_08 — Night Fingerprint Heatmap\n\n"
       "For a few representative subjects, show a 2D heatmap:\n"
       "- **rows** = context lengths (30s → 240m)\n"
       "- **columns** = normalised window position in the night (0% → 100%)\n"
       "- **color** = predicted probability of positive class\n\n"
       "Shows how the model's certainty changes as a function of BOTH where in\n"
       "the night a window falls AND how much context the model has.\n\n"
       "Idea #8 from `docs/NEW_PLOT_IDEAS.md`.\n\n"
       "**Data**: `collected/predictions/*.parquet` (needs `window_idx` column)."),
    code(SETUP),
    code("""\
# ── Config ────────────────────────────────────────────────────────────────────
TASK  = "sex_binary"         # change to any binary task
HEAD  = "transformer"
SPLIT = "test"
CONTEXTS = ["30s", "10m", "40m", "80m", "120m", "240m"]
N_BINS = 25   # night position bins

pqs = load_parquets("phase0_v3", TASK, HEAD, SPLIT)
print("Contexts loaded:", list(pqs.keys()))
if pqs:
    sample = list(pqs.values())[0]
    print("Columns:", sample.columns.tolist())
    print("window_idx present:", "window_idx" in sample.columns)\
"""),
    code("""\
# ── Pick 4 representative subjects ────────────────────────────────────────────
reps = xp.pick_representative_subjects(pqs, n_per_type=1)
print("Representative subjects:", reps)

# Flatten to a list: [always_correct, always_wrong, context_sensitive_pos, context_sensitive_neg]
SUBJECT_TYPES = {
    "Always correct":          "always_correct",
    "Always wrong":            "always_wrong",
    "Improves with context":   "context_sensitive_pos",
    "Worsens with context":    "context_sensitive_neg",
}
subjects_to_plot = []
titles = []
for title, key in SUBJECT_TYPES.items():
    ids = reps.get(key, [])
    if ids:
        subjects_to_plot.append(ids[0])
        titles.append(title)\
"""),
    code("""\
if not subjects_to_plot:
    print("No subjects found — check that parquets have window_idx column")
else:
    n_sub = len(subjects_to_plot)
    fig, axes = plt.subplots(1, n_sub, figsize=(3.5 * n_sub, 3.5))
    if n_sub == 1:
        axes = [axes]

    for ax, subj, title in zip(axes, subjects_to_plot, titles):
        xp.night_fingerprint_panel(ax, pqs, subj,
                                    contexts=CONTEXTS, n_bins=N_BINS)
        ax.set_title(f"{title}\\n(ID: {subj[:8]}…)", fontsize=7)

    fig.suptitle(
        f"Night fingerprint — {TASK_LABEL.get(TASK, TASK)} / {HEAD.upper()}",
        fontsize=8, y=1.02,
    )
    fig.tight_layout()
    plt.show()\
"""),
    _save_cell("xfig_08_night_fingerprint"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_12 — Subject Prediction Stability Grid
# ═══════════════════════════════════════════════════════════════════════════════

cells_xfig12 = [
    md("# xfig_12 — Subject Prediction Stability Grid\n\n"
       "Heatmap where:\n"
       "- **rows** = test subjects (sorted by true_label then by prediction entropy)\n"
       "- **columns** = context lengths\n"
       "- **color** = mean predicted probability at K=all\n\n"
       "A dashed line separates true-negatives (bottom) from true-positives (top).\n"
       "Subjects with uniform color across columns = stable predictions.\n"
       "Subjects with variable color = context-sensitive.\n\n"
       "Idea #12 from `docs/NEW_PLOT_IDEAS.md`.\n\n"
       "**Data**: `collected/predictions/*.parquet`."),
    code(SETUP),
    code("""\
# ── Config ────────────────────────────────────────────────────────────────────
TASK         = "apnea_binary"   # most interpretable for this figure
HEAD         = "transformer"
SPLIT        = "test"
MAX_SUBJECTS = 300    # subsample if larger
CONTEXTS     = ["30s", "10m", "40m", "80m", "120m", "240m"]

pqs = load_parquets("phase0_v3", TASK, HEAD, SPLIT)
print("Contexts available:", list(pqs.keys()))
if pqs:
    n = sum(len(df["subject_id"].unique()) for df in pqs.values())
    print(f"~{n // len(pqs)} unique subjects per context")\
"""),
    code("""\
fig, ax = plt.subplots(figsize=(7.0, 5.0))

xp.subject_stability_heatmap(
    ax, pqs,
    contexts=CONTEXTS,
    max_subjects=MAX_SUBJECTS,
)
ax.set_title(
    f"Per-subject prediction stability — {TASK_LABEL.get(TASK, TASK)} / {HEAD.upper()}",
    fontsize=8,
)
fig.tight_layout()
plt.show()\
"""),
    _save_cell("xfig_12_subject_stability"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_14 — Task Similarity Clustermap
# ═══════════════════════════════════════════════════════════════════════════════

cells_xfig14 = [
    md("# xfig_14 — Task Similarity Clustermap\n\n"
       "Clusters tasks by the **shape of their saturation curve** (AUROC at each\n"
       "context length as a 7-D feature vector).\n\n"
       "Rows = tasks (hierarchically clustered); columns = context lengths (fixed order).\n"
       "Color = AUROC. Tasks with similar curves are placed near each other.\n\n"
       "Idea #14 from `docs/NEW_PLOT_IDEAS.md`.\n\n"
       "**Data**: `analysis.csv`, k=all."),
    code(SETUP),
    code("""\
# ── Config ────────────────────────────────────────────────────────────────────
TASKS = MAIN_TASKS + ["depression_extreme_binary", "osa_binary_apples_postqc"]
HEAD  = "lstm"

df = load_analysis("phase0_v3", split="test", k="all")
print("Tasks in analysis.csv:", sorted(df["task"].unique()))\
"""),
    code("""\
g = xp.task_clustermap(
    df,
    tasks=TASKS,
    head=HEAD,
    figsize=(7.0, 4.0),
)
plt.show()\
"""),
    code("""\
# ── Save ──────────────────────────────────────────────────────────────────────
g.fig.savefig(str(FINAL_OUT / "xfig_14_task_clustermap.pdf"), bbox_inches="tight")
g.fig.savefig(str(FINAL_OUT / "xfig_14_task_clustermap.png"), dpi=150, bbox_inches="tight")
print("Saved →", FINAL_OUT / "xfig_14_task_clustermap.pdf")\
"""),
]


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_19 — Modality Ablation Clustermap
# ═══════════════════════════════════════════════════════════════════════════════

cells_xfig19 = [
    md("# xfig_19 — Modality Ablation Clustermap\n\n"
       "Diverging clustermap of ΔAUROC from `table6_modality.csv`.\n\n"
       "- **Rows** = tasks (hierarchically clustered by ablation profile)\n"
       "- **Columns** = ablation conditions (hierarchically clustered)\n"
       "- **Color** = ΔAUROC: red = harmful removal, blue = slight improvement, white = neutral\n\n"
       "Reveals natural groupings: sleep_efficiency + age cluster (BAS-dominant);\n"
       "apnea stands alone (RESP-dominant). This is Table V as a visual.\n\n"
       "Idea #19 from `docs/NEW_PLOT_IDEAS.md`.\n\n"
       "**Data**: `results/tables/table6_modality.csv`."),
    code(SETUP),
    code("""\
mod_df = load_modality_table(NSRR_TOOLS)
print(mod_df)\
"""),
    code("""\
g = xp.ablation_clustermap(mod_df, figsize=(6.0, 3.5))
plt.show()\
"""),
    code("""\
g.fig.savefig(str(FINAL_OUT / "xfig_19_ablation_clustermap.pdf"), bbox_inches="tight")
g.fig.savefig(str(FINAL_OUT / "xfig_19_ablation_clustermap.png"), dpi=150, bbox_inches="tight")
print("Saved →", FINAL_OUT / "xfig_19_ablation_clustermap.pdf")\
"""),
]


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_25 — SOTA Comparison Bubble Chart
# ═══════════════════════════════════════════════════════════════════════════════

cells_xfig25 = [
    md("# xfig_25 — SOTA Comparison Bubble Chart\n\n"
       "Positions our results alongside SleepFounder, OSF, and SleepMaMi.\n\n"
       "- **x-axis** = pre-training data volume (hours, log scale)\n"
       "- **y-axis** = AUROC on matching tasks\n"
       "- **Filled markers** = method uses EEG; **open markers** = cardio-only (no EEG)\n\n"
       "⚠️ Different evaluation protocols — comparison is approximate.\n\n"
       "Idea #25 from `docs/NEW_PLOT_IDEAS.md`.\n\n"
       "**Data**: hardcoded SOTA numbers + analysis.csv for our results."),
    code(SETUP),
    code("""\
# Load our results for reference (numbers in panels_explore.SOTA_DATA are hardcoded)
df = load_analysis("phase0_v3", split="test", k="all")

# Optional: verify our hardcoded numbers against the actual CSV
for task, head, ctx in [
    ("apnea_binary", "lstm", "120m"),
    ("apnea_binary", "transformer", "120m"),
    ("sex_binary",   "lstm", "120m"),
    ("sex_binary",   "transformer", "240m"),
]:
    row = df[(df.task == task) & (df.head == head) & (df.context_length == ctx)]
    if not row.empty:
        print(f"{task}/{head}/{ctx}: {row['mean_prob_auroc'].values[0]:.3f}")\
"""),
    code("""\
fig, ax = plt.subplots(figsize=(6.0, 4.0))
xp.sota_bubble_panel(ax)
ax.set_title("SOTA comparison (⚠ different eval protocols — approximate)",
             fontsize=8)
fig.tight_layout()
plt.show()\
"""),
    _save_cell("xfig_25_sota_bubble"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_28 — Saturation Curves with Significance Markers
# ═══════════════════════════════════════════════════════════════════════════════

cells_xfig28 = [
    md("# xfig_28 — Saturation Curves with Bootstrap Significance Markers\n\n"
       "Saturation curves with 95% CI bands. Between adjacent context pairs,\n"
       "a `**` marker appears where CIs do NOT overlap (significant improvement);\n"
       "`ns` where they do overlap.\n\n"
       "If bootstrap CIs are not in analysis.csv yet (columns `mean_prob_auroc_ci_lo/hi`),\n"
       "the notebook falls back to annotating the point values instead.\n\n"
       "Idea #28 from `docs/NEW_PLOT_IDEAS.md`.\n\n"
       "**Data**: `analysis.csv`, k=all, with optional CI columns."),
    code(SETUP),
    code("""\
# ── Config ────────────────────────────────────────────────────────────────────
TASKS  = MAIN_TASKS
HEAD   = "transformer"
N_COLS = 3
N_ROWS = (len(TASKS) + N_COLS - 1) // N_COLS

df = load_analysis("phase0_v3", split="test", k="all")

has_ci = "mean_prob_auroc_ci_lo" in df.columns and df["mean_prob_auroc_ci_lo"].notna().any()
print(f"Bootstrap CIs available: {has_ci}")
print(f"CI coverage: {df['mean_prob_auroc_ci_lo'].notna().sum()} / {len(df)} rows")\
"""),
    code("""\
labels = [chr(97 + i) for i in range(len(TASKS))]
n_last = len(TASKS) % N_COLS or N_COLS
n_full = len(TASKS) // N_COLS

mosaic = []
for row in range(n_full):
    rl = labels[row * N_COLS:(row + 1) * N_COLS]
    mosaic.append([l for l in rl for _ in range(2)])
if n_last < N_COLS:
    pad = N_COLS - n_last
    ll  = labels[n_full * N_COLS:]
    mosaic.append(["."] * pad + [l for l in ll for _ in range(2)] + ["."] * pad)

fig, axd = plt.subplot_mosaic(mosaic, figsize=(7.0, N_ROWS * 2.3))

for i, (lbl, task) in enumerate(zip(labels, TASKS)):
    ax = axd[lbl]
    xp.saturation_significance_panel(ax, df, task, head=HEAD)
    ax.set_title(TASK_LABEL.get(task, task), fontsize=8)
    ax.text(0.02, 0.97, f"({lbl})", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top")

fig.suptitle(
    f"Context-length saturation ({HEAD}) — ** non-overlapping 95% CI; ns overlapping",
    fontsize=8, y=1.02,
)
fig.tight_layout(h_pad=1.5, w_pad=1.0)
plt.show()\
"""),
    _save_cell("xfig_28_significance_markers"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_30 — Waterfall Decomposition
# ═══════════════════════════════════════════════════════════════════════════════

cells_xfig30 = [
    md("# xfig_30 — Performance Gain Waterfall\n\n"
       "Decomposes AUROC at 240m K=5 Transformer into additive contributions:\n\n"
       "1. **Base**: MeanPool @ 30s, K=1 (minimal effort baseline)\n"
       "2. **+Aggregation**: K=1 → K=5 at 30s (inference-time gain, free)\n"
       "3. **+Context**: 30s → 240m at K=5 MeanPool (training context gain)\n"
       "4. **+Architecture**: MeanPool → Transformer at 240m, K=5\n"
       "5. **=Final**: Transformer @ 240m, K=5\n\n"
       "One panel per task. Shows which factor contributes most.\n\n"
       "Idea #30 from `docs/NEW_PLOT_IDEAS.md`.\n\n"
       "**Data**: specific cells from `analysis.csv`."),
    code(SETUP),
    code("""\
# ── Config ────────────────────────────────────────────────────────────────────
TASKS  = MAIN_TASKS
N_COLS = 3
N_ROWS = (len(TASKS) + N_COLS - 1) // N_COLS

# Load analysis.csv with ALL k values (not just k='all')
df_allk = load_analysis_all_k("phase0_v3", split="test")

# Quick sanity check: print K=1 and K=5 for sex_binary/mean_pool/30s
check = df_allk[
    (df_allk.task == "sex_binary") &
    (df_allk.head == "mean_pool") &
    (df_allk.context_length == "30s") &
    (df_allk.k.isin(["1", "5"]))
][["task", "head", "context_length", "k", "mean_prob_auroc"]]
print(check.to_string())\
"""),
    code("""\
labels = [chr(97 + i) for i in range(len(TASKS))]
n_last = len(TASKS) % N_COLS or N_COLS
n_full = len(TASKS) // N_COLS

mosaic = []
for row in range(n_full):
    rl = labels[row * N_COLS:(row + 1) * N_COLS]
    mosaic.append([l for l in rl for _ in range(2)])
if n_last < N_COLS:
    pad = N_COLS - n_last
    ll  = labels[n_full * N_COLS:]
    mosaic.append(["."] * pad + [l for l in ll for _ in range(2)] + ["."] * pad)

fig, axd = plt.subplot_mosaic(mosaic, figsize=(7.0, N_ROWS * 2.8))

for i, (lbl, task) in enumerate(zip(labels, TASKS)):
    ax = axd[lbl]
    xp.waterfall_panel(ax, df_allk, task)
    ax.set_title(TASK_LABEL.get(task, task), fontsize=8)
    ax.text(0.02, 0.97, f"({lbl})", transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top")
    # Only leftmost panels of each row get y-label
    if i % N_COLS != 0:
        ax.set_ylabel("")

fig.suptitle("AUROC gain decomposition: aggregation + context + architecture",
             fontsize=8, y=1.01)
fig.tight_layout(h_pad=1.8, w_pad=0.8)
plt.show()\
"""),
    _save_cell("xfig_30_waterfall"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Write all notebooks
# ═══════════════════════════════════════════════════════════════════════════════

NOTEBOOKS = {
    "xfig_02_threshold_unlock.ipynb":    cells_xfig02,
    "xfig_04_deployment_grid.ipynb":     cells_xfig04,
    "xfig_06_modality_radar.ipynb":      cells_xfig06,
    "xfig_08_night_fingerprint.ipynb":   cells_xfig08,
    "xfig_12_subject_stability.ipynb":   cells_xfig12,
    "xfig_14_task_clustermap.ipynb":     cells_xfig14,
    "xfig_19_ablation_clustermap.ipynb": cells_xfig19,
    "xfig_25_sota_bubble.ipynb":         cells_xfig25,
    "xfig_28_significance_markers.ipynb":cells_xfig28,
    "xfig_30_waterfall.ipynb":           cells_xfig30,
}

if __name__ == "__main__":
    for name, cells in NOTEBOOKS.items():
        out = HERE / name
        out.write_text(
            json.dumps(nb(postprocess(cells)), indent=1, ensure_ascii=False)
        )
        print(f"  wrote → {out.name}")
    print(f"\nDone. {len(NOTEBOOKS)} notebooks written to {HERE}")
