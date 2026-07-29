#!/usr/bin/env python3
"""
plot_task_comparison.py — §6 Task × Context Sensitivity Matrix

Reads results/collected/analysis.csv (output of collect_results_v2.py) and
produces three cross-task comparison figures:

  6A: Sensitivity scatter — each task is a point.
      X = baseline difficulty (1 − AUROC at shortest context, K=all)
      Y = context sensitivity (ΔAUROC from shortest → best context, K=all)
      Upper-right quadrant = hard tasks that benefit most from longer context.

  6B: AUROC bars per task — tasks sorted by context sensitivity (ascending),
      grouped bars showing AUROC at each context length.

  6C: L* per task — dot chart showing the saturation context length per task
      (smallest L where AUROC ≥ best_auroc − 0.005).

Requires: multiple tasks to have been run and collected into analysis.csv.

Usage:
  python scripts/plot_task_comparison.py \\
      --head lstm \\
      --split test \\
      --collected-dir results/collected \\
      --results-dir /scratch/boshra95/psg/unified/results/phase0_v2

  # Filter to specific tasks:
  python scripts/plot_task_comparison.py \\
      --tasks sex_binary bmi_binary sleep_efficiency_binary age_class \\
      --head lstm ...

Output:
  {results_dir}/figures/task_comparison/task_comparison_6A_scatter.{png,pdf}
  {results_dir}/figures/task_comparison/task_comparison_6B_bars.{png,pdf}
  {results_dir}/figures/task_comparison/task_comparison_6C_lstar.{png,pdf}
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from repo_sync import configure_repo_figures, save_figure
import pandas as pd

# ── Style ─────────────────────────────────────────────────────────────────────

CONTEXT_TO_MIN = {
    "30s": 0.5, "10m": 10.0, "40m": 40.0,
    "80m": 80.0, "120m": 120.0, "240m": 240.0,
}
CTX_ORDER = {c: i for i, c in enumerate(CONTEXT_TO_MIN)}

# Friendly task labels (override for display)
TASK_LABELS = {
    "sex_binary":                    "Sex (binary)",
    "bmi_binary":                    "BMI (binary)",
    "sleep_efficiency_binary":       "Sleep Efficiency",
    "age_class":                     "Age (3-class)",
    "psqi_binary":                   "PSQI (binary)",
    "depression_extreme_binary":     "Depression",
    "osa_binary_apples_postqc":      "OSA (binary)",
    "osa_severity_apples":           "OSA Severity",
}

TASK_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
]

plt.rcParams.update({
    "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "serif", "font.size": 9,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_analysis_csv(collected_dir: Path) -> pd.DataFrame:
    p = collected_dir / "analysis.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"analysis.csv not found at {p}. "
            "Run collect_results_v2.py first."
        )
    df = pd.read_csv(p)
    if "context_min" not in df.columns and "context_length" in df.columns:
        df["context_min"] = df["context_length"].map(
            lambda s: CONTEXT_TO_MIN.get(str(s).strip()))
    return df


def build_task_summary(df: pd.DataFrame, head: str, split: str) -> pd.DataFrame:
    """
    For each task: compute baseline AUROC (min context), best AUROC (max context),
    sensitivity, difficulty, and L* (saturation context length).

    Returns one row per task with: task, baseline_auroc, best_auroc, sensitivity,
    difficulty, lstar_ctx, lstar_min.
    """
    sub = df[
        (df["head"] == head) &
        (df["split"] == split) &
        (df["k"].astype(str) == "all") &
        df["mean_prob_auroc"].notna()
    ].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["context_min"] = sub["context_length"].map(
        lambda s: CONTEXT_TO_MIN.get(str(s).strip()))

    rows = []
    for task, grp in sub.groupby("task"):
        grp = grp.sort_values("context_min")
        if grp.empty or grp["mean_prob_auroc"].isna().all():
            continue
        baseline_auc = float(grp.iloc[0]["mean_prob_auroc"])
        best_auc     = float(grp["mean_prob_auroc"].max())
        sensitivity  = best_auc - baseline_auc
        difficulty   = 1.0 - baseline_auc
        threshold    = best_auc - 0.005
        lstar_rows   = grp[grp["mean_prob_auroc"] >= threshold]
        if lstar_rows.empty:
            lstar_ctx = str(grp.iloc[-1]["context_length"])
            lstar_min = float(grp.iloc[-1]["context_min"])
        else:
            lstar_ctx = str(lstar_rows.iloc[0]["context_length"])
            lstar_min = float(lstar_rows.iloc[0]["context_min"])
        rows.append({
            "task": task, "baseline_auroc": baseline_auc,
            "best_auroc": best_auc, "sensitivity": sensitivity,
            "difficulty": difficulty, "lstar_ctx": lstar_ctx,
            "lstar_min": lstar_min,
            "_grp": grp,   # keep for plot 6B
        })
    return pd.DataFrame(rows)


# ── Plot 6A: Sensitivity scatter ──────────────────────────────────────────────

def plot_sensitivity_scatter(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        print("  [6A] No data")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    task_list = summary["task"].tolist()
    col_cycle = [TASK_COLORS[i % len(TASK_COLORS)] for i in range(len(task_list))]

    for (_, row), color in zip(summary.iterrows(), col_cycle):
        ax.scatter(row["difficulty"], row["sensitivity"],
                   color=color, s=180, zorder=5,
                   edgecolors="white", lw=1.5)
        label = TASK_LABELS.get(row["task"], row["task"])
        ax.annotate(label, (row["difficulty"], row["sensitivity"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=8.5)

    # Reference lines
    median_sens = float(summary["sensitivity"].median())
    ax.axhline(median_sens, color="gray", ls="--", lw=1, alpha=0.5,
               label=f"Median sensitivity ({median_sens:.3f})")
    median_diff = float(summary["difficulty"].median())
    ax.axvline(median_diff, color="gray", ls=":", lw=1, alpha=0.5)

    ax.set_xlabel("Baseline difficulty  (1 − AUROC at shortest context, K=all)", fontsize=10)
    ax.set_ylabel("Context sensitivity  (ΔAUROC from shortest → best context)", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()

    stem = "task_comparison_6A_scatter"
    save_figure(fig, out_dir, stem)
    plt.close(fig)


# ── Plot 6B: AUROC bars by task ───────────────────────────────────────────────

def plot_sensitivity_bars(summary: pd.DataFrame, df_all: pd.DataFrame,
                          head: str, split: str, out_dir: Path) -> None:
    if summary.empty:
        print("  [6B] No data")
        return

    tasks_sorted = summary.sort_values("sensitivity")["task"].tolist()

    # Get all context lengths present
    sub = df_all[
        (df_all["head"] == head) &
        (df_all["split"] == split) &
        (df_all["k"].astype(str) == "all") &
        df_all["mean_prob_auroc"].notna()
    ].copy()
    contexts = sorted(sub["context_length"].unique(),
                      key=lambda c: CTX_ORDER.get(c, 99))

    n_ctx = len(contexts)
    n_ctx = cm.get_cmap("viridis_r", max(n_ctx, 2))
    ctx_colors = {c: n_ctx(i / max(len(contexts) - 1, 1))
                  for i, c in enumerate(contexts)}

    x     = np.arange(len(tasks_sorted))
    width = 0.8 / max(len(contexts), 1)
    fig, ax = plt.subplots(figsize=(max(10, len(tasks_sorted) * 1.6), 6))

    for j, ctx in enumerate(contexts):
        aucs = []
        for task in tasks_sorted:
            row = sub[(sub["task"] == task) & (sub["context_length"] == ctx)]
            aucs.append(float(row["mean_prob_auroc"].iloc[0])
                        if not row.empty else np.nan)
        offset = (j - len(contexts) / 2 + 0.5) * width
        ax.bar(x + offset, aucs, width * 0.92,
               color=ctx_colors[ctx], label=ctx,
               edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [TASK_LABELS.get(t, t) for t in tasks_sorted],
        rotation=20, ha="right", fontsize=9,
    )
    ax.set_ylabel("AUROC (%)  K=all", fontsize=10)
    ax.legend(title="Context", fontsize=8, title_fontsize=9, ncol=3,
              loc="upper left")
    ax.set_ylim(bottom=max(0.45, float(sub["mean_prob_auroc"].min()) - 0.05))
    fig.tight_layout()

    stem = "task_comparison_6B_bars"
    save_figure(fig, out_dir, stem)
    plt.close(fig)


# ── Plot 6C: L* per task ──────────────────────────────────────────────────────

def plot_lstar(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        print("  [6C] No data")
        return

    tasks_sorted = summary.sort_values("lstar_min")["task"].tolist()
    col_cycle    = TASK_COLORS[:len(tasks_sorted)]
    y            = np.arange(len(tasks_sorted))
    lstar_mins   = [float(summary[summary["task"] == t]["lstar_min"].iloc[0])
                    for t in tasks_sorted]

    fig, ax = plt.subplots(figsize=(8, max(4, len(tasks_sorted) * 0.6 + 1)))
    ax.scatter(lstar_mins, y, c=col_cycle, s=160, zorder=5,
               edgecolors="white", lw=1.5)
    ax.hlines(y, 0.3, lstar_mins, colors=col_cycle, lw=2, alpha=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [TASK_LABELS.get(t, t) for t in tasks_sorted], fontsize=10
    )
    ax.set_xscale("log")
    # Tick labels for known context mins
    tick_vals = sorted(CONTEXT_TO_MIN.values())
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(
        [k for k, v in sorted(CONTEXT_TO_MIN.items(), key=lambda x: x[1])],
        fontsize=9,
    )
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("L* — saturation context length (minutes)", fontsize=10)

    # Arrow annotation for sleep_efficiency (L* > 240m — not saturated)
    se_row = summary[summary["task"] == "sleep_efficiency_binary"]
    if not se_row.empty:
        se_y = tasks_sorted.index("sleep_efficiency_binary")
        se_x = float(se_row["lstar_min"].iloc[0])
        ax.annotate(">240m", (se_x, se_y),
                    xytext=(se_x * 1.15, se_y),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1),
                    fontsize=8, va="center")
    fig.tight_layout()

    stem = "task_comparison_6C_lstar"
    save_figure(fig, out_dir, stem)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-task context sensitivity matrix (§6)."
    )
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Tasks to include (default: all found in analysis.csv)")
    parser.add_argument("--head", default="lstm",
                        help="Head to use for comparison (default: lstm)")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--collected-dir", type=Path,
                        default=Path("results/collected"),
                        dest="collected_dir")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/scratch/boshra95/psg/unified/results/phase0_v2"),
                        dest="results_dir")
    parser.add_argument("--plots", nargs="+", default=["6A", "6C"])
    parser.add_argument("--repo-figures-dir", type=Path, default=None,
                        dest="repo_figures_dir",
                        help="Also mirror PNGs into this repo dir (e.g. "
                             "results/figures/phase0_v3). Default: no repo mirror.")
    args = parser.parse_args()

    configure_repo_figures(args.results_dir, args.repo_figures_dir)
    out_dir = args.results_dir / "figures" / "task_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading analysis.csv from {args.collected_dir} ...")
    try:
        df = load_analysis_csv(args.collected_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    if args.tasks:
        df = df[df["task"].isin(args.tasks)]

    summary = build_task_summary(df, args.head, args.split)
    if summary.empty:
        print(f"ERROR: No data for head='{args.head}' split='{args.split}' k='all' "
              "in analysis.csv. Check that multiple tasks have been collected.")
        return

    tasks_found = summary["task"].tolist()
    print(f"Tasks found: {tasks_found}")
    print(f"Head: {args.head}  Split: {args.split}")

    if "6A" in args.plots:
        plot_sensitivity_scatter(summary, out_dir)
    if "6B" in args.plots:
        plot_sensitivity_bars(summary, df, args.head, args.split, out_dir)
    if "6C" in args.plots:
        plot_lstar(summary, out_dir)

    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
