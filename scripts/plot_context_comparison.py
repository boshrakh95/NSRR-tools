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

# Canonical sort order — extended to cover all context lengths used in Phase 0.
# Any context not listed here is appended at the end, sorted by duration.
CONTEXT_ORDER = ["30s", "2m", "5m", "10m", "20m", "40m", "80m", "120m", "200m", "full_night"]

SEQ2SEQ_TASKS = {"sleep_staging"}  # subject-level aggregation is not meaningful for these


def _ctx_to_seconds(ctx: str) -> float:
    """Convert a context length string to seconds for sorting."""
    if ctx == "full_night":
        return float("inf")
    if ctx.endswith("s"):
        return float(ctx[:-1])
    if ctx.endswith("m"):
        return float(ctx[:-1]) * 60
    return float("inf")


def _active_ctx_order(seg_df: pd.DataFrame, subj_df: pd.DataFrame,
                      task: str, head: str) -> list:
    """Return the sorted context lengths that actually appear in the data
    for this (task, head) pair."""
    ctxs: set = set()
    if not seg_df.empty:
        mask = (seg_df["task"] == task) & (seg_df["head_type"] == head)
        ctxs.update(seg_df.loc[mask, "context_length"].dropna().tolist())
    if not subj_df.empty:
        mask = (subj_df["task"] == task) & (subj_df["head_type"] == head)
        ctxs.update(subj_df.loc[mask, "context_length"].dropna().tolist())
    known   = [c for c in CONTEXT_ORDER if c in ctxs]
    unknown = sorted(ctxs - set(CONTEXT_ORDER), key=_ctx_to_seconds)
    return known + unknown

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

def _parse_exp_name(exp_name: str) -> tuple[str, str]:
    """Extract (task, head) from an experiment folder name, handling run tags
    and parenthetical suffixes.

    Examples:
      apnea_binary_lstm                   → ("apnea_binary",    "lstm")
      sleepiness_binary_lstm_lr1e4        → ("sleepiness_binary","lstm")
      apnea_binary_transformer_lr1e4      → ("apnea_binary",    "transformer")
      depression_binary_lstm(only_apples) → ("depression_binary","lstm")
      anxiety_binary_mean_pool            → ("anxiety_binary",  "mean_pool")
    """
    for head in HEAD_ORDER:
        marker = f"_{head}"
        idx = exp_name.find(marker)
        if idx == -1:
            continue
        after = exp_name[idx + len(marker):]
        # Valid boundary: end of string, start of run-tag (_), or parenthetical
        if after == "" or after.startswith("_") or after.startswith("("):
            return exp_name[:idx], head
    return exp_name, "unknown"


def _folder_run_tag(folder_name: str) -> str:
    """Return the run tag embedded in an experiment folder name.

    Examples:
      apnea_binary_lstm                      → ""
      apnea_binary_lstm_lr1e4                → "lr1e4"
      sleepiness_binary_transformer_lr1e4    → "lr1e4"
      sleepiness_binary_transformer(old)     → "(old)"   [parenthetical = special tag]
      depression_binary_lstm(only_apples)    → "(only_apples)"
    """
    # Folders with parenthetical markers are non-standard; give them a unique
    # non-empty tag so they are excluded from both "" and "lr1e4" filters.
    paren_idx = folder_name.find("(")
    if paren_idx != -1:
        return folder_name[paren_idx:]

    for head in HEAD_ORDER:
        if folder_name.endswith(f"_{head}"):
            return ""
        marker = f"_{head}_"
        idx = folder_name.find(marker)
        if idx != -1:
            return folder_name[idx + len(marker):]
    return ""


def _read_summary_csv(path: Path) -> pd.DataFrame:
    """Read a summary.csv robustly, handling a mixed-format issue where newer
    training runs insert an extra 'best_val_monitor' field before 'n_epochs_run'
    that the original header doesn't declare."""
    import io
    lines = path.read_text().splitlines()
    if not lines:
        return pd.DataFrame()
    header = lines[0].split(",")
    n_header = len(header)
    has_extra = any(len(l.split(",")) > n_header for l in lines[1:] if l.strip())
    if has_extra and "best_val_loss" in header:
        idx = header.index("best_val_loss")
        header.insert(idx, "best_val_monitor")
        # Old-format rows (one field short) get NaN in best_val_monitor automatically
        text = ",".join(header) + "\n" + "\n".join(lines[1:])
        return pd.read_csv(io.StringIO(text), on_bad_lines="skip")
    return pd.read_csv(io.StringIO("\n".join(lines)))


def load_segment_results(results_dir: Path, run_tag: str | None = None) -> pd.DataFrame:
    """Load segment-level results from summary.csv files.

    run_tag: if not None, only load folders whose run tag matches exactly.
             Pass "" to load only untagged experiments (e.g. apnea_binary_lstm).
    """
    frames = []
    for csv in results_dir.glob("*/summary.csv"):
        if run_tag is not None and _folder_run_tag(csv.parent.name) != run_tag:
            continue
        try:
            frames.append(_read_summary_csv(csv))
        except Exception as e:
            print(f"  Warning: skipping {csv} ({e})")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # summary.csv accumulates a new row on every rerun of the same context;
    # keep the first occurrence (earliest training run) per (task, head, context).
    df = df.drop_duplicates(subset=["task", "head_type", "context_length"],
                            keep="first")
    return df


def load_subject_results(results_dir: Path, run_tag: str | None = None) -> pd.DataFrame:
    """Load subject-level results from subject_metrics.json files.

    run_tag: same semantics as load_segment_results.
    """
    import json
    inf_dir = results_dir / "inference"
    rows = []
    for jf in sorted(inf_dir.glob("**/subject_metrics.json")):
        exp_name = jf.parent.parent.name
        if run_tag is not None and _folder_run_tag(exp_name) != run_tag:
            continue
        try:
            with open(jf) as f:
                m = json.load(f)
        except Exception:
            continue

        ctx = jf.parent.name.replace("context_", "")
        task, head = _parse_exp_name(exp_name)

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

def plot_task_head(ax, task: str, head: str, metric: str,
                   seg_df: pd.DataFrame, subj_df: pd.DataFrame,
                   ctx_order: list) -> bool:
    """Draw all lines for one (task, head) panel onto ax."""
    pos = {c: i for i, c in enumerate(ctx_order)}
    drawn_any = False

    # ── Segment-level (K=5) ──────────────────────────────────────────────────
    if not seg_df.empty:
        t = seg_df[(seg_df["task"] == task) & (seg_df["head_type"] == head)].copy()
        t = t[t["context_length"].isin(ctx_order)].copy()
        t["_ord"] = t["context_length"].map(pos)
        t = t.sort_values("_ord")
        col = f"test_{metric}"
        if col in t.columns and not t[col].isna().all():
            xs = t["_ord"].tolist()
            ys = t[col].values * 100
            s = LINE_STYLES["seg_k5"]
            ax.plot(xs, ys, color=s["color"], marker=s["marker"],
                    linestyle=s["linestyle"], label=s["label"], linewidth=2,
                    markersize=7)
            drawn_any = True

    # ── Subject-level ────────────────────────────────────────────────────────
    if not subj_df.empty and task not in SEQ2SEQ_TASKS:
        for method, style_key in [("mean_prob", "subj_mean"),
                                   ("majority_vote", "subj_majority")]:
            t = subj_df[
                (subj_df["task"] == task) &
                (subj_df["head_type"] == head) &
                (subj_df["aggregation"] == method)
            ].copy()
            if t.empty:
                continue
            t = t[t["context_length"].isin(ctx_order)].copy()
            t["_ord"] = t["context_length"].map(pos)
            t = t.sort_values("_ord")
            col = f"subj_{metric}"
            if col not in t.columns or t[col].isna().all():
                continue
            xs = t["_ord"].tolist()
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
    ctx_order = _active_ctx_order(seg_df, subj_df, task, head)
    if not ctx_order:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    drawn = plot_task_head(ax, task, head, metric, seg_df, subj_df, ctx_order)
    if not drawn:
        plt.close(fig)
        return

    ax.set_title(f"{task}  ·  {head}  —  {METRIC_TITLE.get(metric, metric)}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Context length", fontsize=11)
    ax.set_ylabel(METRIC_YLABELS.get(metric, metric), fontsize=11)
    ax.set_xticks(range(len(ctx_order)))
    ax.set_xticklabels(ctx_order)
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
        ctx_order = _active_ctx_order(seg_df, subj_df, task, head)
        drawn = plot_task_head(ax, task, head, metric, seg_df, subj_df, ctx_order)
        ax.set_title(task.replace("_", " "), fontsize=10, fontweight="bold")
        ax.set_xlabel("Context", fontsize=9)
        ax.set_ylabel(METRIC_YLABELS.get(metric, metric), fontsize=9)
        ax.set_xticks(range(len(ctx_order)))
        ax.set_xticklabels(ctx_order, fontsize=8)
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
    parser.add_argument("--run-tag", default=None, dest="run_tag",
                        help="Only include experiments with this run tag "
                             "(e.g. 'lr1e4').  Pass '' to include only untagged runs. "
                             "Default: include all.")
    args = parser.parse_args()

    seg_df  = load_segment_results(args.results_dir, run_tag=args.run_tag)
    subj_df = load_subject_results(args.results_dir, run_tag=args.run_tag)

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

    tag_suffix = f"_{args.run_tag}" if args.run_tag else ""

    for head in heads:
        # Per-task individual figures
        if not args.summary_only:
            for task in tasks:
                out = fig_dir / f"{task}_{head}{tag_suffix}_{metric}.png"
                make_single_figure(task, head, metric, seg_df, subj_df, out, args.show)

        # Summary grid figure
        out = fig_dir / f"summary_{head}{tag_suffix}_{metric}.png"
        make_summary_figure(tasks, head, metric, seg_df, subj_df, out, args.show)

    print("\nDone.")


if __name__ == "__main__":
    main()
