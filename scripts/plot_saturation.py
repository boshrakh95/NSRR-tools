#!/usr/bin/env python3
"""
plot_saturation.py — Context-length saturation curves (Fig 1 in paper).

Data source: analysis.csv (from collect_results_v2.py), filtered to k='all'.
Primary column: mean_prob_auroc (subject-level mean-pool AUROC at K=all).

Falls back to summary.csv test_auroc only if analysis.csv is not found — this
is the segment-level K=1 metric and should NOT be used for paper figures.

Usage:
  python scripts/plot_saturation.py \\
      --task sex_binary \\
      --heads lstm transformer mean_pool \\
      --results-dir /scratch/boshra95/psg/unified/results/phase0_v3 \\
      --collected-dir results/collected/phase0_v3 \\
      --metric auroc

Output:
  {results_dir}/figures/saturation/saturation_{task}_{metric}_{split}.{png,pdf}
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from repo_sync import configure_repo_figures, default_repo_figures_dir, save_figure


# ── Global style ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "serif",
    "font.size": 9,
})

# ── Config ────────────────────────────────────────────────────────────────────

CONTEXT_TO_MIN = {
    "30s": 0.5, "10m": 10.0, "40m": 40.0,
    "80m": 80.0, "120m": 120.0, "240m": 240.0,
}

HEAD_STYLE = {
    "lstm":        {"color": "#3A7EBF", "marker": "o", "ls": "-",  "label": "LSTM"},
    "transformer": {"color": "#E86A33", "marker": "s", "ls": "--", "label": "Transformer"},
    "mean_pool":   {"color": "#44A15E", "marker": "^", "ls": ":",  "label": "Mean Pool"},
}

METRIC_LABEL = {
    "auroc":             "AUROC (%)",
    "balanced_accuracy": "Balanced Accuracy (%)",
    "accuracy":          "Accuracy (%)",
    "macro_f1":          "Macro F1 (%)",
}

# analysis.csv column to use for each metric
METRIC_COL = {
    "auroc":             "mean_prob_auroc",
    "balanced_accuracy": "mean_prob_balanced_accuracy",
}

# CI column pairs in analysis.csv
CI_COLS = {
    "auroc":             ("mean_prob_auroc_ci_lo",    "mean_prob_auroc_ci_hi"),
    "balanced_accuracy": ("mean_prob_bal_acc_ci_lo",  "mean_prob_bal_acc_ci_hi"),
}


def parse_context_min(s: str) -> float:
    if s in CONTEXT_TO_MIN:
        return CONTEXT_TO_MIN[s]
    s = s.strip()
    if s.endswith("m"):
        return float(s[:-1])
    if s.endswith("s"):
        return float(s[:-1]) / 60.0
    return float(s)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_from_analysis(collected_dir: Path, task: str, head: str,
                       split: str, metric: str,
                       run_tag: str = "") -> pd.DataFrame:
    """Load saturation data from analysis.csv at k='all'.

    Returns DataFrame with columns: context_length, context_length_min, value
    (already on [0,1] scale). Empty if data unavailable.
    """
    col = METRIC_COL.get(metric)
    if col is None or collected_dir is None:
        return pd.DataFrame()
    p = collected_dir / "analysis.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "run_tag" not in df.columns:
        df["run_tag"] = ""
    mask = (
        (df["task"] == task) &
        (df["head"] == head) &
        (df["split"] == split) &
        (df["k"].astype(str) == "all") &
        (df["run_tag"].fillna("") == (run_tag or "")) &
        df[col].notna()
    )
    sub = df.loc[mask, ["context_length", "context_length_min", col]].copy()
    sub = sub.sort_values("context_length_min").reset_index(drop=True)
    sub = sub.rename(columns={col: "value"})
    return sub


def load_ci_data(collected_dir: Path, task: str, head: str,
                 split: str, metric: str) -> dict:
    """Load bootstrap CI bounds from analysis.csv at k='all'.
    Returns dict: context_label -> (ci_lo, ci_hi) or empty dict if unavailable.
    """
    if collected_dir is None:
        return {}
    p = collected_dir / "analysis.csv"
    if not p.exists():
        return {}
    ci_lo_col, ci_hi_col = CI_COLS.get(metric, (None, None))
    if ci_lo_col is None:
        return {}
    df = pd.read_csv(p)
    sub = df[
        (df["task"] == task) &
        (df["head"] == head) &
        (df["split"] == split) &
        (df["k"].astype(str) == "all") &
        df[ci_lo_col].notna() &
        df[ci_hi_col].notna()
    ]
    return {
        str(row["context_length"]): (float(row[ci_lo_col]), float(row[ci_hi_col]))
        for _, row in sub.iterrows()
    }


def load_summary_fallback(results_dir: Path, task: str, head: str,
                          run_tag: str, split: str, metric: str):
    """Fallback: read test_auroc from summary.csv. Returns (xs, ys) or (None, None)."""
    exp_id   = f"{task}_{head}" + (f"_{run_tag}" if run_tag else "")
    csv_path = results_dir / exp_id / "summary.csv"
    if not csv_path.exists():
        return None, None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="warn")
    df["context_length_min"] = df["context_length"].map(parse_context_min)
    df = df.sort_values("context_length_min").reset_index(drop=True)
    col = f"{split}_{metric}" if f"{split}_{metric}" in df.columns else f"val_{metric}"
    if col not in df.columns:
        return None, None
    return df["context_length_min"].values, df[col].values


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_saturation(task: str, heads: list, results_dir: Path,
                    metrics: list, run_tag: str, out_dir: Path,
                    split: str = "test",
                    collected_dir: Path | None = None):
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(9, 6))
        any_data = False
        has_ci   = False
        all_xs_seen: list[float] = []

        for head in heads:
            # ── Primary: analysis.csv ─────────────────────────────────────────
            adf = load_from_analysis(collected_dir, task, head, split, metric, run_tag)
            if not adf.empty:
                xs = adf["context_length_min"].values
                ys = adf["value"].values * 100
                all_xs_seen.extend(xs.tolist())
            else:
                # Fallback to summary.csv (warns user; not the primary metric)
                xs, raw_ys = load_summary_fallback(
                    results_dir, task, head, run_tag, split, metric)
                if xs is None:
                    print(f"  [skip] No data for {task}_{head} — "
                          f"run collect_results_v2.py or check --collected-dir")
                    continue
                ys = raw_ys * 100
                all_xs_seen.extend(xs.tolist())
                print(f"  [warning] {task}_{head}: using summary.csv fallback "
                      f"(segment-level metric, not mean-pool); regenerate after collect.")

            style = HEAD_STYLE.get(head, {
                "color": "grey", "marker": "o", "ls": "-",
                "label": head.replace("_", " ").title(),
            })

            ax.plot(xs, ys,
                    color=style["color"], marker=style["marker"],
                    linestyle=style["ls"], linewidth=2.0, markersize=7,
                    label=style["label"])

            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}", (x, y),
                            textcoords="offset points", xytext=(0, 7),
                            fontsize=7, ha="center", color=style["color"])

            # Optional CI bands
            ci_map = load_ci_data(collected_dir, task, head, split, metric)
            if ci_map:
                ctx_str_map = {v: k for k, v in CONTEXT_TO_MIN.items()}
                ci_xs, ci_lo_vals, ci_hi_vals = [], [], []
                for x in xs:
                    ctx_lbl = ctx_str_map.get(float(x))
                    if ctx_lbl and ctx_lbl in ci_map:
                        lo, hi = ci_map[ctx_lbl]
                        ci_xs.append(x)
                        ci_lo_vals.append(lo * 100)
                        ci_hi_vals.append(hi * 100)
                if ci_xs:
                    ax.fill_between(ci_xs, ci_lo_vals, ci_hi_vals,
                                    color=style["color"], alpha=0.15)
                    has_ci = True

            any_data = True

        if not any_data:
            plt.close(fig)
            continue

        ax.set_xscale("log")
        all_xs_sorted = sorted(set(all_xs_seen))
        ctx_str = {v: k for k, v in CONTEXT_TO_MIN.items()}
        ax.set_xticks(all_xs_sorted)
        ax.set_xticklabels(
            [ctx_str.get(x, f"{x:.0f}m") for x in all_xs_sorted], fontsize=9)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        ax.set_xlabel("Context length (minutes)", fontsize=10)
        ylabel = METRIC_LABEL.get(metric, metric)
        if has_ci:
            ylabel += "  [shading = 95% CI]"
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, which="major", alpha=0.3)
        ax.grid(True, which="minor", alpha=0.08)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        stem = f"saturation_{task}_{metric}_{split}"
        save_figure(fig, out_dir, stem)
        plt.close(fig)
        if has_ci:
            print("  (with CI bands)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot metric vs context length (saturation curve) per head."
    )
    parser.add_argument("--task",    required=True)
    parser.add_argument("--heads",   nargs="+",
                        default=["lstm", "transformer", "mean_pool"])
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/scratch/boshra95/psg/unified/results/phase0_v3"),
                        dest="results_dir")
    parser.add_argument("--metric",  nargs="+",
                        default=["auroc", "balanced_accuracy"])
    parser.add_argument("--split",   default="test", choices=["val", "test"])
    parser.add_argument("--run-tag", default="", dest="run_tag")
    parser.add_argument("--collected-dir", type=Path, default=None,
                        dest="collected_dir",
                        help="Directory containing analysis.csv. "
                             "Defaults to results_dir/collected.")
    parser.add_argument("--repo-figures-dir", type=Path, default=None,
                        dest="repo_figures_dir")
    args = parser.parse_args()

    # Default collected_dir to results_dir/collected
    collected_dir = args.collected_dir or (args.results_dir / "collected")

    configure_repo_figures(args.results_dir, args.repo_figures_dir)
    out_dir = args.results_dir / "figures" / "saturation"

    print(f"Task: {args.task}  Heads: {args.heads}  Metrics: {args.metric}")
    print(f"  Data source: {collected_dir}/analysis.csv  (k=all, {args.split})")
    plot_saturation(
        task=args.task,
        heads=args.heads,
        results_dir=args.results_dir,
        metrics=args.metric,
        run_tag=args.run_tag,
        out_dir=out_dir,
        split=args.split,
        collected_dir=collected_dir,
    )


if __name__ == "__main__":
    main()
