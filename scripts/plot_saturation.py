#!/usr/bin/env python3
"""
plot_saturation.py — Step 4 of iso-compute analysis pipeline.

"Figure 1" for the paper: metric vs context length at K=all, one line per head.
Reads summary.csv files (no parquet or heatmap DF needed).

Answers H1 (does longer context improve performance, and where does it saturate?)
and H4 (which head benefits most from longer context?).

Usage:
  python scripts/plot_saturation.py \\
      --task sex_binary \\
      --heads lstm transformer mean_pool \\
      --results-dir /scratch/boshra95/psg/unified/results/phase0_v2 \\
      --metric auroc balanced_accuracy

Output:
  {results_dir}/figures/saturation_{task}_{metric}.{png,pdf}
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# ── Config ────────────────────────────────────────────────────────────────────

CONTEXT_TO_MIN = {
    "30s": 0.5, "10m": 10.0, "40m": 40.0,
    "80m": 80.0, "120m": 120.0, "240m": 240.0,
}

HEAD_STYLE = {
    "lstm":       {"color": "#4C72B0", "marker": "o", "ls": "-",  "label": "LSTM"},
    "transformer":{"color": "#DD8452", "marker": "s", "ls": "--", "label": "Transformer"},
    "mean_pool":  {"color": "#55A868", "marker": "^", "ls": ":",  "label": "Mean Pool"},
}

METRIC_LABEL = {
    "auroc":             "AUROC",
    "balanced_accuracy": "Balanced Accuracy",
    "accuracy":          "Accuracy",
    "macro_f1":          "Macro F1",
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


def load_summary(results_dir: Path, task: str, head: str, run_tag: str) -> pd.DataFrame:
    exp_id   = f"{task}_{head}" + (f"_{run_tag}" if run_tag else "")
    csv_path = results_dir / exp_id / "summary.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["context_length_min"] = df["context_length"].map(parse_context_min)
    return df.sort_values("context_length_min").reset_index(drop=True)


def plot_saturation(task: str, heads: list, results_dir: Path,
                    metrics: list, run_tag: str, out_dir: Path,
                    split: str = "test"):
    for metric in metrics:
        val_col  = f"val_{metric}"
        test_col = f"{split}_{metric}"

        fig, ax = plt.subplots(figsize=(9, 6))
        any_data = False

        for head in heads:
            df = load_summary(results_dir, task, head, run_tag)
            if df.empty:
                print(f"  [skip] No summary.csv for {task}_{head}")
                continue

            col = test_col if test_col in df.columns else val_col
            if col not in df.columns:
                print(f"  [skip] Column '{col}' not found in {task}_{head}/summary.csv")
                continue

            style = HEAD_STYLE.get(head, {
                "color": "grey", "marker": "o", "ls": "-",
                "label": head.replace("_", " ").title(),
            })
            xs = df["context_length_min"].values
            ys = df[col].values * 100

            ax.plot(xs, ys,
                    color=style["color"], marker=style["marker"],
                    linestyle=style["ls"], linewidth=2.5, markersize=8,
                    label=style["label"])

            # Annotate each point with its value
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}", (x, y),
                            textcoords="offset points", xytext=(0, 7),
                            fontsize=8, ha="center", color=style["color"])
            any_data = True

        if not any_data:
            plt.close(fig)
            continue

        ax.set_xscale("log")
        # Custom x-tick labels matching context strings
        all_xs = sorted(set(
            v for h in heads
            for v in load_summary(results_dir, task, h, run_tag)["context_length_min"].tolist()
        ))
        ctx_str = {v: k for k, v in CONTEXT_TO_MIN.items()}
        ax.set_xticks(all_xs)
        ax.set_xticklabels([ctx_str.get(x, f"{x:.0f}m") for x in all_xs], fontsize=10)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        ax.set_xlabel("Context length (log scale)", fontsize=12)
        ax.set_ylabel(f"{METRIC_LABEL.get(metric, metric)} ({split}) (%)", fontsize=12)
        ax.set_title(f"{task}  —  {METRIC_LABEL.get(metric, metric)} vs Context Length",
                     fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, which="major", alpha=0.35)
        ax.grid(True, which="minor", alpha=0.1)

        plt.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"saturation_{task}_{metric}_{split}"
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"{stem}.{ext}",
                        dpi=150 if ext == "png" else None,
                        bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_dir.name}/{stem}.{{png,pdf}}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot metric vs context length (saturation curve) per head."
    )
    parser.add_argument("--task",    required=True,
                        help="Task name, e.g. sex_binary")
    parser.add_argument("--heads",   nargs="+",
                        default=["lstm", "transformer", "mean_pool"],
                        help="Heads to compare (default: lstm transformer mean_pool)")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/scratch/boshra95/psg/unified/results/phase0_v2"),
                        dest="results_dir")
    parser.add_argument("--metric",  nargs="+",
                        default=["auroc", "balanced_accuracy"],
                        help="Metrics to plot (default: auroc balanced_accuracy)")
    parser.add_argument("--split",   default="test",
                        choices=["val", "test"],
                        help="Which split to use (default: test)")
    parser.add_argument("--run-tag", default="", dest="run_tag")
    args = parser.parse_args()

    out_dir = args.results_dir / "figures"

    print(f"Task: {args.task}  Heads: {args.heads}  Metrics: {args.metric}")
    plot_saturation(
        task=args.task,
        heads=args.heads,
        results_dir=args.results_dir,
        metrics=args.metric,
        run_tag=args.run_tag,
        out_dir=out_dir,
        split=args.split,
    )


if __name__ == "__main__":
    main()
