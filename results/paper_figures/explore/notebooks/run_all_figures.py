#!/usr/bin/env python3
"""Run all exploratory figures without nbconvert.

Usage (from the notebooks/ directory):
    /Users/boshra/NSRR-workspace/NSRR-tools/.venv/bin/python run_all_figures.py

Figures are saved to ../final/ with full xfig_NN_name filenames.
Run individual figures by passing their number(s):
    python run_all_figures.py 02 06 30
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

def _find_workspace():
    candidate = HERE
    for _ in range(10):
        if (candidate / "final_results").exists():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return Path("/Users/boshra/NSRR-workspace").resolve()

WORKSPACE_ROOT = _find_workspace()
NSRR_TOOLS     = WORKSPACE_ROOT / "NSRR-tools"
FINAL_OUT      = HERE.parent / "final"
TABLES_DIR     = NSRR_TOOLS / "results" / "tables"
FINAL_OUT.mkdir(parents=True, exist_ok=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from utils.data_explore import (
    set_root, load_analysis, load_analysis_all_k,
    load_heatmap, load_parquets, load_modality_table,
    subject_correctness_matrix, CONTEXT_TO_MIN, CTX_ORDER,
)
from utils import panels_explore as xp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

set_root(WORKSPACE_ROOT)
matplotlib.rcParams.update({
    "figure.dpi":      150,
    "savefig.bbox":    "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family":     "serif",
    "font.size":       8,
    "axes.labelsize":  7,
})

MAIN_TASKS = ["sex_binary", "bmi_binary", "age_class",
              "sleep_efficiency_binary", "apnea_binary"]
ALL_TASKS  = MAIN_TASKS + ["depression_extreme_binary", "osa_binary_apples_postqc"]
TASK_LABEL = xp.TASK_LABEL

print(f"WORKSPACE_ROOT : {WORKSPACE_ROOT}")
print(f"FINAL_OUT      : {FINAL_OUT}")


def _save(fig, stem):
    fig.savefig(str(FINAL_OUT / f"{stem}.pdf"), bbox_inches="tight")
    fig.savefig(str(FINAL_OUT / f"{stem}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {stem}.pdf")


def _panel_label(ax, lbl):
    ax.text(0.02, 0.97, lbl, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top", fontfamily="serif")


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_02 — Clinical Threshold Unlock Map
# ═══════════════════════════════════════════════════════════════════════════════
def run_02():
    print("Running xfig_02_threshold_unlock ...")
    df = load_analysis("phase0_v3", split="test", k="all")
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    xp.threshold_unlock_heatmap(ax, df, tasks=MAIN_TASKS, head="transformer",
                                 thresholds=[0.70, 0.75, 0.80, 0.85, 0.90])
    ax.set_title("First context length to reach target AUROC (Transformer, K=all)",
                 fontsize=8, pad=6)
    fig.tight_layout()
    _save(fig, "xfig_02_threshold_unlock")


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_04 — Deployment Scenario Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def run_04():
    print("Running xfig_04_deployment_grid ...")
    hmaps = {t: load_heatmap("phase0_v3", t, "transformer") for t in MAIN_TASKS}
    budgets = [30, 60, 120, 240, 480]
    targets = [0.70, 0.75, 0.80, 0.85]

    N_COLS, N_ROWS, ROW_H = 3, 2, 2.4
    labels  = [chr(97 + i) for i in range(len(MAIN_TASKS))]
    n_last  = len(MAIN_TASKS) % N_COLS or N_COLS
    n_full  = len(MAIN_TASKS) // N_COLS

    mosaic = []
    for row in range(n_full):
        rl = labels[row * N_COLS:(row + 1) * N_COLS]
        mosaic.append([l for l in rl for _ in range(2)])
    if n_last < N_COLS:
        pad = N_COLS - n_last
        ll  = labels[n_full * N_COLS:]
        mosaic.append(["."] * pad + [l for l in ll for _ in range(2)] + ["."] * pad)

    fig, axd = plt.subplot_mosaic(mosaic, figsize=(7.0, N_ROWS * ROW_H))
    for i, (lbl, task) in enumerate(zip(labels, MAIN_TASKS)):
        xp.deployment_scenario_panel(axd[lbl], hmaps[task], task,
                                      budgets_min=budgets, targets=targets)
        axd[lbl].set_title(TASK_LABEL.get(task, task), fontsize=8)
        _panel_label(axd[lbl], f"({lbl})")

    fig.suptitle("Optimal (L, K) strategy per budget and required AUROC (Transformer)",
                 fontsize=8, y=1.01)
    fig.tight_layout(h_pad=1.5, w_pad=1.2)
    _save(fig, "xfig_04_deployment_grid")


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_06 — Modality Radar Chart
# ═══════════════════════════════════════════════════════════════════════════════
def run_06():
    print("Running xfig_06_modality_radar ...")
    mod_df = load_modality_table(NSRR_TOOLS)
    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(111, projection="polar")
    handles, labels = xp.modality_radar_panel(ax, mod_df, tasks=MAIN_TASKS)
    ax.legend(handles, [TASK_LABEL.get(t, t) for t in MAIN_TASKS],
              loc="upper left", bbox_to_anchor=(1.15, 1.1),
              fontsize=7, frameon=False)
    ax.set_title("Modality importance: |ΔAUROC| when modality removed",
                 fontsize=8, pad=15)
    fig.tight_layout()
    _save(fig, "xfig_06_modality_radar")


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_08 — Night Fingerprint Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def run_08():
    print("Running xfig_08_night_fingerprint ...")
    pqs = load_parquets("phase0_v3", "sex_binary", "transformer", "test")
    reps = xp.pick_representative_subjects(pqs, n_per_type=1)

    SUBJECT_TYPES = {
        "Always correct":         "always_correct",
        "Always wrong":           "always_wrong",
        "Improves with context":  "context_sensitive_pos",
        "Worsens with context":   "context_sensitive_neg",
    }
    subjects_to_plot, titles = [], []
    for title, key in SUBJECT_TYPES.items():
        ids = reps.get(key, [])
        if ids:
            subjects_to_plot.append(ids[0])
            titles.append(title)

    if not subjects_to_plot:
        print("  WARNING: no subjects found (window_idx missing?)")
        return

    n_sub = len(subjects_to_plot)
    fig, axes = plt.subplots(1, n_sub, figsize=(3.5 * n_sub, 3.5))
    if n_sub == 1:
        axes = [axes]
    for ax, subj, title in zip(axes, subjects_to_plot, titles):
        xp.night_fingerprint_panel(ax, pqs, subj, n_bins=25)
        ax.set_title(f"{title}\n(ID: {subj[:8]}…)", fontsize=7)

    fig.suptitle("Night fingerprint — Sex / Transformer", fontsize=8, y=1.02)
    fig.tight_layout()
    _save(fig, "xfig_08_night_fingerprint")


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_12 — Subject Prediction Stability Grid
# ═══════════════════════════════════════════════════════════════════════════════
def run_12():
    print("Running xfig_12_subject_stability ...")
    pqs = load_parquets("phase0_v3", "apnea_binary", "transformer", "test")
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    xp.subject_stability_heatmap(ax, pqs, max_subjects=300)
    ax.set_title("Per-subject prediction stability — Apnea / Transformer", fontsize=8)
    fig.tight_layout()
    _save(fig, "xfig_12_subject_stability")


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_14 — Task Similarity Clustermap
# ═══════════════════════════════════════════════════════════════════════════════
def run_14():
    print("Running xfig_14_task_clustermap ...")
    df = load_analysis("phase0_v3", split="test", k="all")
    g = xp.task_clustermap(df, tasks=ALL_TASKS, head="lstm", figsize=(7.0, 4.0))
    g.fig.savefig(str(FINAL_OUT / "xfig_14_task_clustermap.pdf"), bbox_inches="tight")
    g.fig.savefig(str(FINAL_OUT / "xfig_14_task_clustermap.png"),
                  dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    print("  saved → xfig_14_task_clustermap.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_19 — Modality Ablation Clustermap
# ═══════════════════════════════════════════════════════════════════════════════
def run_19():
    print("Running xfig_19_ablation_clustermap ...")
    mod_df = load_modality_table(NSRR_TOOLS)
    g = xp.ablation_clustermap(mod_df, figsize=(6.0, 3.5))
    g.fig.savefig(str(FINAL_OUT / "xfig_19_ablation_clustermap.pdf"), bbox_inches="tight")
    g.fig.savefig(str(FINAL_OUT / "xfig_19_ablation_clustermap.png"),
                  dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    print("  saved → xfig_19_ablation_clustermap.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_25 — SOTA Comparison Bubble Chart
# ═══════════════════════════════════════════════════════════════════════════════
def run_25():
    print("Running xfig_25_sota_bubble ...")
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    xp.sota_bubble_panel(ax)
    ax.set_title("SOTA comparison (⚠ different eval protocols — approximate)",
                 fontsize=8)
    fig.tight_layout()
    _save(fig, "xfig_25_sota_bubble")


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_28 — Saturation Curves with Significance Markers
# ═══════════════════════════════════════════════════════════════════════════════
def run_28():
    print("Running xfig_28_significance_markers ...")
    df = load_analysis("phase0_v3", split="test", k="all")
    labels = [chr(97 + i) for i in range(len(MAIN_TASKS))]
    N_COLS, N_ROWS = 3, 2

    n_last = len(MAIN_TASKS) % N_COLS or N_COLS
    n_full = len(MAIN_TASKS) // N_COLS
    mosaic = []
    for row in range(n_full):
        rl = labels[row * N_COLS:(row + 1) * N_COLS]
        mosaic.append([l for l in rl for _ in range(2)])
    if n_last < N_COLS:
        pad = N_COLS - n_last
        ll  = labels[n_full * N_COLS:]
        mosaic.append(["."] * pad + [l for l in ll for _ in range(2)] + ["."] * pad)

    fig, axd = plt.subplot_mosaic(mosaic, figsize=(7.0, N_ROWS * 2.3))
    for i, (lbl, task) in enumerate(zip(labels, MAIN_TASKS)):
        xp.saturation_significance_panel(axd[lbl], df, task, head="transformer")
        axd[lbl].set_title(TASK_LABEL.get(task, task), fontsize=8)
        _panel_label(axd[lbl], f"({lbl})")

    fig.suptitle(
        "Context-length saturation (Transformer) — ** non-overlapping 95% CI; ns overlapping",
        fontsize=8, y=1.02,
    )
    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    _save(fig, "xfig_28_significance_markers")


# ═══════════════════════════════════════════════════════════════════════════════
# xfig_30 — Waterfall Decomposition (K=all)
# ═══════════════════════════════════════════════════════════════════════════════
def run_30():
    print("Running xfig_30_waterfall ...")
    df_allk = load_analysis_all_k("phase0_v3", split="test")
    labels  = [chr(97 + i) for i in range(len(MAIN_TASKS))]
    N_COLS  = 3
    n_last  = len(MAIN_TASKS) % N_COLS or N_COLS
    n_full  = len(MAIN_TASKS) // N_COLS
    mosaic  = []
    for row in range(n_full):
        rl = labels[row * N_COLS:(row + 1) * N_COLS]
        mosaic.append([l for l in rl for _ in range(2)])
    if n_last < N_COLS:
        pad = N_COLS - n_last
        ll  = labels[n_full * N_COLS:]
        mosaic.append(["."] * pad + [l for l in ll for _ in range(2)] + ["."] * pad)

    fig, axd = plt.subplot_mosaic(mosaic, figsize=(7.0, N_COLS * 1.9))
    for i, (lbl, task) in enumerate(zip(labels, MAIN_TASKS)):
        xp.waterfall_panel(axd[lbl], df_allk, task)
        axd[lbl].set_title(TASK_LABEL.get(task, task), fontsize=8)
        _panel_label(axd[lbl], f"({lbl})")
        if i % N_COLS != 0:
            axd[lbl].set_ylabel("")

    fig.suptitle("AUROC gain decomposition: aggregation + context + architecture (K=all)",
                 fontsize=8, y=1.01)
    fig.tight_layout(h_pad=1.8, w_pad=0.8)
    _save(fig, "xfig_30_waterfall")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

RUNNERS = {
    "02": run_02,
    "04": run_04,
    "06": run_06,
    "08": run_08,
    "12": run_12,
    "14": run_14,
    "19": run_19,
    "25": run_25,
    "28": run_28,
    "30": run_30,
}

if __name__ == "__main__":
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(RUNNERS.keys())
    unknown   = [r for r in requested if r not in RUNNERS]
    if unknown:
        print(f"Unknown figure numbers: {unknown}")
        print(f"Valid: {list(RUNNERS.keys())}")
        sys.exit(1)

    failed = []
    for key in requested:
        try:
            RUNNERS[key]()
        except Exception as e:
            print(f"  ✗ xfig_{key} FAILED: {e}")
            import traceback; traceback.print_exc()
            failed.append(key)

    print(f"\nDone. {len(requested) - len(failed)}/{len(requested)} succeeded.")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)
