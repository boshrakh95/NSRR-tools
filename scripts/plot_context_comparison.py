#!/usr/bin/env python3
"""
plot_context_comparison.py — Phase 0, Context-length comparison plots

Produces per-task figures comparing:
  - Segment-level AUROC (K=5 windows, current training eval)
  - Subject-level AUROC via mean-prob aggregation (all windows)
  - Subject-level balanced accuracy via mean-prob

Figures are saved as PNG to:
    {results_dir}/figures/{task}_{head}_{metric}.png

Also produces a multi-task summary figure with one subplot per task.

Usage
─────
  # All tasks, all heads, save figures:
  python scripts/plot_context_comparison.py

  # Specific tasks / heads:
  python scripts/plot_context_comparison.py --tasks apnea_binary cvd_binary --heads lstm

  # Choose metric (auroc | balanced_accuracy):
  python scripts/plot_context_comparison.py --metric balanced_accuracy

  # Preview without saving:
  python scripts/plot_context_comparison.py --show
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; override with --show
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent

# ── Config ────────────────────────────────────────────────────────────────────

CONTEXT_ORDER = ["30s", "10m", "40m", "80m"]
CONTEXT_LABELS = {"30s": "30s", "10m": "10m", "40m": "40m", "80m": "80m"}

TASK_ORDER = [
    "sleep_staging",
    "apnea_binary",
    "apnea_class",
    "sleepiness_binary",
    "sleepiness_class",
    "cvd_binary",
    "rested_morning",
    "depression_binary",
    "depression_class",
    "insomnia_binary",
    "anxiety_binary",
]

HEAD_ORDER = ["lstm", "transformer", "mean_pool"]

# Line style per evaluation method
LINE_STYLES = {
    "seg_k5":       {"color": "#4C72B0", "marker": "o", "linestyle": "--",
                     "label": "Segment (K=5)"},
    "subj_mean":    {"color": "#DD8452", "marker": "s", "linestyle": "-",
                     "label": "Subject (mean-prob, all wins)"},
    "subj_majority":{"color": "#55A868", "marker": "^", "linestyle": ":",
                     "label": "Subject (maj-vote, all wins)"},
}

METRIC_YLABELS = {
    "auroc":             "AUROC (%)",
    "balanced_accuracy": "Balanced Accuracy (%)",
    "macro_f1":          "Macro F1 (%)",
}

METRIC_TITLE = {
    "auroc":             "AUROC",
    "balanced_accuracy": "Balanced Accuracy",
    "macro_f1":          "Macro F1",
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_segment_results(results_dir: Path) -> pd.DataFrame:
    """Load segment-level results from summary.csv files."""
    frames = []
    for csv in results_dir.glob("*/summary.csv"):
        frames.append(pd.read_csv(csv))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_subject_results(results_dir: Path) -> pd.DataFrame:
    """Load subject-level results from subject_metrics.json files."""
    import json
    inf_dir = results_dir / "inference"
    rows = []
    for jf in sorted(inf_dir.glob("**/subject_metrics.json")):
        try:
            with open(jf) as f:
                m = json.load(f)
        except Exception:
            continue

        exp_name = jf.parent.parent.name
        ctx      = jf.parent.name.replace("context_", "")

        for head in HEAD_ORDER:
            if exp_name.endswith(f"_{head}"):
                task = exp_name[: -(len(head) + 1)]
                break
        else:
            task = exp_name
            head = "unknown"

        for method in ("mean_prob", "majority_vote"):
            sub = m.get(method, {})
            if not sub:
                continue
            rows.append({
                "task":           task,
                "head_type":      head,
                "context_length": ctx,
                "aggregation":    method,
                **{f"subj_{k}": v for k, v in sub.items()},
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Plotting ──────────────────────────────────────────────────────────────────

def _ctx_positions(contexts: list) -> list:
    """Map context labels to numeric x positions in CONTEXT_ORDER."""
    return [CONTEXT_ORDER.index(c) if c in CONTEXT_ORDER else i
            for i, c in enumerate(contexts)]


def plot_task_head(ax, task: str, head: str, metric: str,
                   seg_df: pd.DataFrame, subj_df: pd.DataFrame):
    """Draw all lines for one (task, head) panel onto ax."""
    drawn_any = False

    # ── Segment-level (K=5) ──────────────────────────────────────────────────
    if not seg_df.empty:
        t = seg_df[(seg_df["task"] == task) & (seg_df["head_type"] == head)].copy()
        t["_ord"] = t["context_length"].map(
            lambda c: CONTEXT_ORDER.index(c) if c in CONTEXT_ORDER else 99
        )
        t = t.sort_values("_ord")
        col = f"test_{metric}"
        if col in t.columns and not t[col].isna().all():
            xs = _ctx_positions(t["context_length"].tolist())
            ys = t[col].values * 100
            s = LINE_STYLES["seg_k5"]
            ax.plot(xs, ys, color=s["color"], marker=s["marker"],
                    linestyle=s["linestyle"], label=s["label"], linewidth=2,
                    markersize=7)
            drawn_any = True

    # ── Subject-level ────────────────────────────────────────────────────────
    if not subj_df.empty:
        for method, style_key in [("mean_prob", "subj_mean"),
                                   ("majority_vote", "subj_majority")]:
            t = subj_df[
                (subj_df["task"] == task) &
                (subj_df["head_type"] == head) &
                (subj_df["aggregation"] == method)
            ].copy()
            if t.empty:
                continue
            t["_ord"] = t["context_length"].map(
                lambda c: CONTEXT_ORDER.index(c) if c in CONTEXT_ORDER else 99
            )
            t = t.sort_values("_ord")
            col = f"subj_{metric}"
            if col not in t.columns or t[col].isna().all():
                continue
            xs = _ctx_positions(t["context_length"].tolist())
            ys = t[col].values * 100
            s = LINE_STYLES[style_key]
            ax.plot(xs, ys, color=s["color"], marker=s["marker"],
                    linestyle=s["linestyle"], label=s["label"], linewidth=2,
                    markersize=7)
            drawn_any = True

    return drawn_any


def make_single_figure(task: str, head: str, metric: str,
                        seg_df: pd.DataFrame, subj_df: pd.DataFrame,
                        out_path: Path, show: bool = False):
    """One figure for a single (task, head, metric)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    drawn = plot_task_head(ax, task, head, metric, seg_df, subj_df)
    if not drawn:
        plt.close(fig)
        return

    ax.set_title(f"{task}  ·  {head}  —  {METRIC_TITLE.get(metric, metric)}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Context length", fontsize=11)
    ax.set_ylabel(METRIC_YLABELS.get(metric, metric), fontsize=11)
    ax.set_xticks(range(len(CONTEXT_ORDER)))
    ax.set_xticklabels(CONTEXT_ORDER)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if show:
        matplotlib.use("TkAgg")
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    plt.close(fig)


def make_summary_figure(tasks: list, head: str, metric: str,
                         seg_df: pd.DataFrame, subj_df: pd.DataFrame,
                         out_path: Path, show: bool = False):
    """Grid figure with one subplot per task."""
    n = len(tasks)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows),
                              squeeze=False)
    axes_flat = axes.flatten()

    for i, task in enumerate(tasks):
        ax = axes_flat[i]
        drawn = plot_task_head(ax, task, head, metric, seg_df, subj_df)
        ax.set_title(task.replace("_", " "), fontsize=10, fontweight="bold")
        ax.set_xlabel("Context", fontsize=9)
        ax.set_ylabel(METRIC_YLABELS.get(metric, metric), fontsize=9)
        ax.set_xticks(range(len(CONTEXT_ORDER)))
        ax.set_xticklabels(CONTEXT_ORDER, fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    # Hide unused axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Phase 0 context-length comparison  ·  head={head}  ·  {METRIC_TITLE.get(metric, metric)}",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.tight_layout()

    if show:
        matplotlib.use("TkAgg")
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot context-length comparison: segment vs subject-level."
    )
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/scratch/boshra95/psg/unified/results/phase0"))
    parser.add_argument("--tasks",  nargs="+", default=None,
                        help="Tasks to plot (default: all with results)")
    parser.add_argument("--heads",  nargs="+", default=None,
                        help="Heads to plot (default: all with results)")
    parser.add_argument("--metric", default="auroc",
                        choices=["auroc", "balanced_accuracy", "macro_f1"],
                        help="Metric to plot (default: auroc)")
    parser.add_argument("--show",   action="store_true",
                        help="Display interactively instead of saving")
    parser.add_argument("--summary-only", action="store_true", dest="summary_only",
                        help="Only produce the multi-task summary figure, skip per-task figures")
    args = parser.parse_args()

    seg_df  = load_segment_results(args.results_dir)
    subj_df = load_subject_results(args.results_dir)

    if seg_df.empty and subj_df.empty:
        print("No results found. Run training and/or inference first.")
        return

    fig_dir = args.results_dir / "figures"

    # Determine which tasks / heads to plot
    tasks_with_data = set()
    if not seg_df.empty:
        tasks_with_data.update(seg_df["task"].unique())
    if not subj_df.empty:
        tasks_with_data.update(subj_df["task"].unique())
    tasks = args.tasks or [t for t in TASK_ORDER if t in tasks_with_data]

    heads_with_data = set()
    if not seg_df.empty:
        heads_with_data.update(seg_df["head_type"].unique())
    if not subj_df.empty:
        heads_with_data.update(subj_df["head_type"].unique())
    heads = args.heads or [h for h in HEAD_ORDER if h in heads_with_data]

    metric = args.metric
    print(f"Plotting metric: {metric}")
    print(f"Tasks:  {tasks}")
    print(f"Heads:  {heads}")
    print()

    for head in heads:
        # Per-task individual figures
        if not args.summary_only:
            for task in tasks:
                out = fig_dir / f"{task}_{head}_{metric}.png"
                make_single_figure(task, head, metric, seg_df, subj_df, out, args.show)

        # Summary grid figure
        out = fig_dir / f"summary_{head}_{metric}.png"
        make_summary_figure(tasks, head, metric, seg_df, subj_df, out, args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()
