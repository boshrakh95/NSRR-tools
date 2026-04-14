#!/usr/bin/env python3
"""
analyze_windows.py — Phase 0, Window-count analysis

Reads already-dumped all-windows parquets (from infer_subject_windows.py) and
computes metrics at different values of K (windows per subject), for both:

  segment-level   : average metrics over all K × N_subjects segments
  subject mean-prob: average softmax probs across K windows → one score/subject
  subject majority : mode of K hard predictions → one label/subject

This lets you answer:
  - Does majority voting help vs plain segment averaging at the same K?
  - How many windows do you need before performance plateaus?
  - Is 10m×5 windows better than 30s×20 windows?

Works on whatever context lengths have parquets available — no need for all
lengths to be present.

Output
──────
  {inference_dir}/{task}_{head}/window_analysis.csv   — full K-sweep table
  {inference_dir}/{task}_{head}/window_analysis.md    — markdown table
  {figures_dir}/{task}_{head}_window_sweep_{metric}.png (optional, --plot)

Usage
─────
  # Analyse all available context lengths for a task/head:
  python scripts/analyze_windows.py --task apnea_binary --head lstm

  # Specific context lengths only:
  python scripts/analyze_windows.py --task cvd_binary --head lstm --contexts 10m 40m

  # Custom K values:
  python scripts/analyze_windows.py --task apnea_binary --head lstm --k-values 1 5 10 20 all

  # Also produce plots:
  python scripts/analyze_windows.py --task apnea_binary --head lstm --plot

  # Choose window selection strategy (default: evenly-spaced):
  python scripts/analyze_windows.py --task apnea_binary --head lstm --window-strategy first
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import mode as scipy_mode

try:
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
        cohen_kappa_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_K_VALUES = [1, 5, 10, 20, 50, "all"]
CONTEXT_ORDER    = ["30s", "10m", "40m", "80m"]

# ── Window selection ──────────────────────────────────────────────────────────

def select_windows(grp: pd.DataFrame, k: int, strategy: str) -> pd.DataFrame:
    """Select K windows from a subject group."""
    n = len(grp)
    if k >= n:
        return grp
    if strategy == "first":
        return grp.iloc[:k]
    elif strategy == "last":
        return grp.iloc[-k:]
    elif strategy == "random":
        return grp.sample(k, random_state=42)
    else:  # evenly-spaced (matches val/test eval in training)
        positions = np.linspace(0, n - 1, k, dtype=int)
        return grp.iloc[positions]


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics_from_arrays(targets: np.ndarray, preds: np.ndarray,
                                 probs: np.ndarray, num_classes: int,
                                 task: str) -> dict:
    """Compute classification metrics. probs shape: (N, C)."""
    m = {"accuracy": float((preds == targets).mean())}
    if not HAS_SKLEARN:
        return m
    m["balanced_accuracy"] = float(balanced_accuracy_score(targets, preds))
    m["macro_f1"] = float(f1_score(targets, preds, average="macro", zero_division=0))
    try:
        if num_classes == 2:
            m["auroc"] = float(roc_auc_score(targets, probs[:, 1]))
        else:
            m["auroc"] = float(
                roc_auc_score(targets, probs, multi_class="ovr", average="macro")
            )
    except ValueError:
        m["auroc"] = float("nan")
    if task == "sleep_staging" and HAS_SKLEARN:
        m["cohen_kappa"] = float(cohen_kappa_score(targets, preds))
    return m


def evaluate_at_k(df: pd.DataFrame, k_val: int | str,
                  num_classes: int, task: str, strategy: str) -> dict:
    """
    Given the full all-windows parquet for one (context, task, head),
    evaluate at K windows per subject.

    Returns dict with keys:
        k, n_subjects, n_segments,
        seg_{metric},          ← segment-level (avg over K*N segments)
        mean_prob_{metric},    ← subject mean-prob aggregation
        majority_{metric},     ← subject majority-vote aggregation
    """
    prob_cols = [f"prob_class{c}" for c in range(num_classes)]
    k_int = None if k_val == "all" else int(k_val)

    all_seg_targets, all_seg_preds, all_seg_probs = [], [], []
    subj_targets, subj_mean_preds, subj_majority_preds, subj_mean_probs = [], [], [], []

    for (sid, dset), grp in df.groupby(["subject_id", "dataset"], sort=False):
        grp = grp.sort_values("window_idx").reset_index(drop=True)
        sub = select_windows(grp, k_int if k_int is not None else len(grp), strategy)

        true_label = int(sub["true_label"].iloc[0])
        seg_preds  = sub["pred_label"].values
        seg_probs  = sub[prob_cols].values.astype(np.float32)

        # Segment-level accumulation
        all_seg_targets.append(np.full(len(sub), true_label, dtype=np.int32))
        all_seg_preds.append(seg_preds.astype(np.int32))
        all_seg_probs.append(seg_probs)

        # Subject-level accumulation
        mean_prob  = seg_probs.mean(axis=0)
        mean_pred  = int(mean_prob.argmax())
        majority   = int(scipy_mode(seg_preds, keepdims=True).mode[0])

        subj_targets.append(true_label)
        subj_mean_preds.append(mean_pred)
        subj_majority_preds.append(majority)
        subj_mean_probs.append(mean_prob)

    # Concatenate segment arrays
    seg_targets = np.concatenate(all_seg_targets)
    seg_preds   = np.concatenate(all_seg_preds)
    seg_probs   = np.vstack(all_seg_probs)

    # Numpy arrays for subject level
    subj_targets        = np.array(subj_targets)
    subj_mean_preds     = np.array(subj_mean_preds)
    subj_majority_preds = np.array(subj_majority_preds)
    subj_mean_probs     = np.vstack(subj_mean_probs)

    seg_metrics      = compute_metrics_from_arrays(seg_targets, seg_preds, seg_probs, num_classes, task)
    mean_prob_metrics = compute_metrics_from_arrays(subj_targets, subj_mean_preds, subj_mean_probs, num_classes, task)
    majority_metrics  = compute_metrics_from_arrays(subj_targets, subj_majority_preds, subj_mean_probs, num_classes, task)

    row = {
        "k":          k_val,
        "n_subjects": len(subj_targets),
        "n_segments": len(seg_targets),
    }
    for k, v in seg_metrics.items():
        row[f"seg_{k}"] = v
    for k, v in mean_prob_metrics.items():
        row[f"mean_prob_{k}"] = v
    for k, v in majority_metrics.items():
        row[f"majority_{k}"] = v

    return row


# ── Markdown table ────────────────────────────────────────────────────────────

def fmt(val, pct=True) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return f"{val * 100:.1f}" if pct else str(val)


HEADER_MAP = {
    "seg_auroc":                  "Seg-AUROC",
    "seg_balanced_accuracy":      "Seg-BalAcc",
    "seg_macro_f1":               "Seg-F1",
    "seg_cohen_kappa":            "Seg-Kappa",
    "mean_prob_auroc":            "MP-AUROC",
    "mean_prob_balanced_accuracy":"MP-BalAcc",
    "mean_prob_macro_f1":         "MP-F1",
    "mean_prob_cohen_kappa":      "MP-Kappa",
    "majority_auroc":             "MV-AUROC",
    "majority_balanced_accuracy": "MV-BalAcc",
    "majority_macro_f1":          "MV-F1",
    "majority_cohen_kappa":       "MV-Kappa",
}


def _split_to_markdown(results_df: pd.DataFrame, strategy: str) -> str:
    """Render per-context tables for one split's results_df."""
    lines = [
        f"_Window selection: **{strategy}**. "
        "Metrics in %. MP = mean-prob aggregation. MV = majority-vote._\n",
    ]

    ctx_order = [c for c in CONTEXT_ORDER if c in results_df["context_length"].unique()]
    other     = [c for c in results_df["context_length"].unique() if c not in ctx_order]

    for ctx in ctx_order + other:
        cdf = results_df[results_df["context_length"] == ctx]
        if cdf.empty:
            continue

        lines.append(f"## Context: `{ctx}`\n")

        metric_cols = [
            f"{p}_{m}"
            for p in ["seg", "mean_prob", "majority"]
            for m in ["auroc", "balanced_accuracy", "macro_f1", "cohen_kappa"]
            if f"{p}_{m}" in cdf.columns and not cdf[f"{p}_{m}"].isna().all()
        ]

        headers = ["K", "N-subj", "N-seg"] + [HEADER_MAP.get(c, c) for c in metric_cols]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for _, row in cdf.iterrows():
            cells = [str(row["k"]), str(int(row["n_subjects"])), str(int(row["n_segments"]))]
            for col in metric_cols:
                cells.append(fmt(row.get(col, float("nan"))))
            lines.append("| " + " | ".join(cells) + " |")

        lines.append("")

    return "\n".join(lines)


def to_markdown(results_df: pd.DataFrame, task: str, head: str, strategy: str) -> str:
    """Legacy wrapper kept for compatibility."""
    header = [
        f"# Window-count analysis: `{task}` · `{head}`\n",
        "---\n",
    ]
    return "\n".join(header) + _split_to_markdown(results_df, strategy)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_window_sweep(results_df: pd.DataFrame, task: str, head: str,
                      metric: str, out_path: Path):
    """One figure per context length, showing all three methods vs K."""
    contexts = [c for c in CONTEXT_ORDER if c in results_df["context_length"].unique()]
    contexts += [c for c in results_df["context_length"].unique() if c not in contexts]
    if not contexts:
        return

    ncols = min(len(contexts), 3)
    nrows = (len(contexts) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    methods = [
        (f"seg_{metric}",       "Segment-level",    "#4C72B0", "o", "--"),
        (f"mean_prob_{metric}", "Subject mean-prob", "#DD8452", "s", "-"),
        (f"majority_{metric}",  "Subject maj-vote",  "#55A868", "^", ":"),
    ]

    for i, ctx in enumerate(contexts):
        ax   = axes_flat[i]
        cdf  = results_df[results_df["context_length"] == ctx].copy()
        # Build numeric x axis: K value or max windows for "all"
        xs, xlabels = [], []
        for _, row in cdf.iterrows():
            k = row["k"]
            xs.append(int(row["n_segments"] / max(row["n_subjects"], 1)) if k == "all" else int(k))
            xlabels.append(str(k))

        for col, label, color, marker, ls in methods:
            if col not in cdf.columns:
                continue
            ys = cdf[col].values * 100
            ax.plot(range(len(xs)), ys, color=color, marker=marker,
                    linestyle=ls, label=label, linewidth=2, markersize=7)

        ax.set_title(f"ctx={ctx}", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_xlabel("K (windows/subject)", fontsize=9)
        ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(len(contexts), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"{task}  ·  {head}  —  {metric.replace('_', ' ').title()} vs windows/subject",
        fontsize=12, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse how metrics change with number of windows per subject."
    )
    parser.add_argument("--task",   required=True, help="Task name, e.g. apnea_binary")
    parser.add_argument("--head",   required=True, help="Head, e.g. lstm")
    parser.add_argument("--contexts", nargs="+", default=None,
                        help="Context lengths to include (default: all available)")
    parser.add_argument("--k-values", nargs="+", default=None, dest="k_values",
                        help='K values to evaluate, e.g. --k-values 1 5 10 20 all '
                             '(default: 1 5 10 20 50 all)')
    parser.add_argument("--window-strategy", default="evenly-spaced",
                        choices=["evenly-spaced", "first", "last", "random"],
                        dest="window_strategy",
                        help="How to select K windows per subject (default: evenly-spaced)")
    parser.add_argument("--plot", action="store_true",
                        help="Also save comparison plots")
    parser.add_argument("--plot-metric", default="auroc", dest="plot_metric",
                        choices=["auroc", "balanced_accuracy", "macro_f1"],
                        help="Metric to plot (default: auroc)")
    parser.add_argument("--splits", nargs="+", default=["test"],
                        choices=["train", "val", "test"],
                        help="Splits to analyse (default: test). Use --splits val test for both.")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/scratch/boshra95/psg/unified/results/phase0"),
                        dest="results_dir")
    parser.add_argument("--run-tag",    default="", dest="run_tag",
                        help="Must match the --run-tag used during training/inference (default: no suffix).")
    args = parser.parse_args()

    exp_id  = f"{args.task}_{args.head}" + (f"_{args.run_tag}" if args.run_tag else "")
    inf_dir = args.results_dir / "inference" / exp_id
    if not inf_dir.exists():
        print(f"No inference directory found: {inf_dir}")
        print("Run infer_subject_windows.py first.")
        return

    # Parse K values once
    raw_k = args.k_values or [str(k) for k in DEFAULT_K_VALUES]
    k_values = ["all" if v == "all" else int(v) for v in raw_k]

    print(f"Task: {args.task}  Head: {args.head}  Splits: {args.splits}")
    print()

    # ── Process each split independently ─────────────────────────────────────
    # split_contents: {split_name -> markdown string for that split's tables}
    split_contents: dict = {}

    for split in args.splits:
        parquets = sorted(inf_dir.glob(f"context_*/{split}_windows.parquet"))
        if args.contexts:
            parquets = [p for p in parquets if any(
                p.parent.name == f"context_{c}" for c in args.contexts
            )]

        if not parquets:
            print(f"[{split}] No {split}_windows.parquet found — skipping.")
            print()
            continue

        print(f"{'='*60}")
        print(f"  SPLIT: {split.upper()}")
        print(f"  Contexts: {[p.parent.name.replace('context_','') for p in parquets]}")
        print(f"{'='*60}")

        split_rows = []

        for parquet_path in parquets:
            ctx = parquet_path.parent.name.replace("context_", "")
            print(f"\n── Context: {ctx} ──")

            df = pd.read_parquet(parquet_path)
            prob_cols   = [c for c in df.columns if c.startswith("prob_class")]
            num_classes = len(prob_cols)

            seg_metrics_json = (args.results_dir / exp_id
                                / f"context_{ctx}" / "metrics.json")
            if seg_metrics_json.exists():
                with open(seg_metrics_json) as f:
                    num_classes = json.load(f)["num_classes"]

            n_subjects = df.groupby(["subject_id", "dataset"]).ngroups
            max_k      = df.groupby(["subject_id", "dataset"]).size().max()
            print(f"  Subjects: {n_subjects:,} | Max windows/subject: {max_k}")

            for k_val in k_values:
                k_int = max_k if k_val == "all" else int(k_val)
                if k_int > max_k:
                    print(f"  [skip] K={k_val} > max available ({max_k})")
                    continue

                row = evaluate_at_k(df, k_val, num_classes, args.task,
                                    args.window_strategy)
                row["context_length"] = ctx
                row["split"]          = split
                split_rows.append(row)

                # Print full metrics for all methods
                METRICS = ["auroc", "balanced_accuracy", "macro_f1", "cohen_kappa"]
                MNAMES  = {"auroc": "AUROC", "balanced_accuracy": "BalAcc",
                           "macro_f1": "MacroF1", "cohen_kappa": "Kappa"}
                prefixes = [("seg",       "Segment   "),
                            ("mean_prob", "Mean-prob "),
                            ("majority",  "Maj-vote  ")]

                print(f"\n  K={k_val}  "
                      f"(n_subjects={row['n_subjects']:,}  "
                      f"n_segments={row['n_segments']:,})")
                for prefix, label in prefixes:
                    parts = []
                    for m in METRICS:
                        col = f"{prefix}_{m}"
                        val = row.get(col)
                        if val is not None and not (
                            isinstance(val, float) and np.isnan(val)
                        ):
                            parts.append(f"{MNAMES[m]}={val*100:.1f}%")
                    if parts:
                        print(f"    {label}: " + "  ".join(parts))

        print()

        if not split_rows:
            continue

        split_df = pd.DataFrame(split_rows)

        # Save per-split CSV
        out_csv = inf_dir / f"window_analysis_{split}.csv"
        split_df.to_csv(out_csv, index=False)
        print(f"CSV saved: {out_csv}")

        # Collect markdown section for this split
        split_contents[split] = _split_to_markdown(split_df, args.window_strategy)

        # Plot per split
        if args.plot:
            out_fig = (args.results_dir / "figures" /
                       f"{args.task}_{args.head}_{split}_window_sweep_{args.plot_metric}.png")
            plot_window_sweep(split_df, args.task, args.head, args.plot_metric, out_fig)

    # ── Save combined markdown (all splits, separate sections) ────────────────
    out_md = inf_dir / "window_analysis.md"

    # Load existing split sections from disk so we don't lose splits not in this run
    import re
    existing_contents: dict = {}
    if out_md.exists():
        existing = out_md.read_text()
        for m in re.finditer(r'\n---\n# Split: (\w+)\n(.*?)(?=\n---\n# Split: |\Z)',
                             existing, re.DOTALL):
            existing_contents[m.group(1).lower()] = m.group(2)

    # Merge: new run overrides, old splits are preserved
    existing_contents.update(split_contents)

    # Write header + splits in a fixed order: val then test
    lines = [
        f"# Window-count analysis: `{args.task}` · `{args.head}`\n",
        f"_Window selection: **{args.window_strategy}**. Metrics in %._\n",
    ]
    for split in ["val", "test"] + [s for s in existing_contents if s not in ("val", "test")]:
        if split in existing_contents:
            lines.append(f"\n---\n# Split: {split.upper()}\n")
            lines.append(existing_contents[split])

    out_md.write_text("\n".join(lines))
    print(f"\nMarkdown saved: {out_md}")


if __name__ == "__main__":
    main()
