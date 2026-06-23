#!/usr/bin/env python3
"""
plot_calibration.py — §2 Model Calibration and Reliability

Reads per-window prediction parquets (from infer_subject_windows.py) and produces:

  2A: Reliability diagrams — fraction of positives vs mean predicted probability
      for three representative context lengths (30s / mid / longest).
      Shows whether predicted confidence matches actual outcome frequency.

  2B: ECE vs context length — Expected Calibration Error for each head,
      at a fixed K (default: K=10 windows per subject).

  2C: ECE vs K — how calibration improves as more windows are aggregated,
      for each context length (fixed head).

Expected Calibration Error (ECE):
  ECE = Σ_bins (n_bin / N) × |mean_confidence_bin − fraction_positive_bin|

All computations use mean-probability aggregation (average prob_class1 across K
evenly-spaced windows per subject), following the same selection strategy as
analyze_windows.py.

Usage:
  python scripts/plot_calibration.py \\
      --task sex_binary \\
      --head lstm \\
      --results-dir /scratch/boshra95/psg/unified/results/phase0_v2 \\
      --split test

  # Multiple heads for plot 2B:
  python scripts/plot_calibration.py \\
      --task sex_binary --heads lstm transformer mean_pool \\
      --results-dir ...

Output:
  {results_dir}/figures/{task}_{head}/calibration_2A_reliability.{png,pdf}
  {results_dir}/figures/{task}_{head}/calibration_2B_ece_vs_context.{png,pdf}
  {results_dir}/figures/{task}_{head}/calibration_2C_ece_vs_k.{png,pdf}
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from repo_sync import configure_repo_figures, save_figure

# ── Style ─────────────────────────────────────────────────────────────────────

CONTEXT_TO_MIN = {
    "30s": 0.5, "10m": 10.0, "40m": 40.0,
    "80m": 80.0, "120m": 120.0, "240m": 240.0,
}
CTX_ORDER = {c: i for i, c in enumerate(CONTEXT_TO_MIN)}

HEAD_STYLE = {
    "lstm":       {"color": "#4C72B0", "marker": "o", "ls": "-",  "label": "LSTM"},
    "transformer":{"color": "#DD8452", "marker": "s", "ls": "--", "label": "Transformer"},
    "mean_pool":  {"color": "#55A868", "marker": "^", "ls": ":",  "label": "Mean Pool"},
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


# ── I/O helpers ───────────────────────────────────────────────────────────────

def inference_dir(results_dir: Path, task: str, head: str, run_tag: str) -> Path:
    exp_id = f"{task}_{head}" + (f"_{run_tag}" if run_tag else "")
    return results_dir / "inference" / exp_id


def load_parquets(results_dir: Path, task: str, head: str,
                  split: str, run_tag: str = "",
                  contexts: list = None) -> dict:
    """Return {context_str: DataFrame} for all available contexts."""
    idir  = inference_dir(results_dir, task, head, run_tag)
    data  = {}
    for ctx_dir in sorted(idir.glob("context_*")):
        ctx = ctx_dir.name.removeprefix("context_")
        if contexts and ctx not in contexts:
            continue
        pq = ctx_dir / f"{split}_windows.parquet"
        if pq.exists():
            data[ctx] = pd.read_parquet(pq)
    return data


# ── Window aggregation ────────────────────────────────────────────────────────

def _select_k(group: pd.DataFrame, k: int) -> pd.DataFrame:
    n = len(group)
    if n <= k:
        return group
    idx = np.linspace(0, n - 1, k, dtype=int)
    return group.iloc[idx]


def aggregate_subjects(df: pd.DataFrame, k="all", prob_col: str = "prob_class1") -> pd.DataFrame:
    """Aggregate K evenly-spaced windows per subject → one row per subject."""
    if prob_col not in df.columns:
        # Fall back to prob_class0 complement if binary, else skip
        if "prob_class0" in df.columns and "prob_class1" not in df.columns:
            df = df.copy()
            df["prob_class1"] = 1.0 - df["prob_class0"]
        else:
            return pd.DataFrame()

    sorted_df = df.sort_values(["subject_id", "dataset", "window_idx"])
    if k != "all":
        k = int(k)
        sorted_df = (
            sorted_df
            .groupby(["subject_id", "dataset"], group_keys=False)
            .apply(lambda g: _select_k(g, k))
            .reset_index(drop=True)
        )
    return (
        sorted_df
        .groupby(["subject_id", "dataset"])
        .agg(
            mean_prob=(prob_col, "mean"),
            true_label=("true_label", "first"),
            n_windows=(prob_col, "count"),
        )
        .reset_index()
    )


# ── ECE computation ───────────────────────────────────────────────────────────

def compute_ece(y_true: np.ndarray, prob_pred: np.ndarray,
                n_bins: int = 10) -> tuple:
    """
    Returns (ece, bin_mids, frac_pos, bin_weights).
    bin_mids and frac_pos are the calibration curve points.
    """
    bins  = np.linspace(0.0, 1.0, n_bins + 1)
    ece   = 0.0
    mids, fracs, weights = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob_pred >= lo) & (prob_pred < hi)
        n    = mask.sum()
        if n < 2:
            continue
        conf = prob_pred[mask].mean()
        acc  = y_true[mask].mean()
        w    = n / len(y_true)
        ece += w * abs(conf - acc)
        mids.append(conf)
        fracs.append(acc)
        weights.append(w)
    return ece, np.array(mids), np.array(fracs), np.array(weights)


# ── Plot 2A: Reliability diagrams ─────────────────────────────────────────────

def plot_reliability_diagrams(parquets: dict, task: str, head: str,
                              k: int, out_dir: Path) -> None:
    contexts = sorted(parquets.keys(), key=lambda c: CTX_ORDER.get(c, 99))
    if not contexts:
        print("  [2A] No parquets found")
        return

    # Show at most 3 representative contexts
    show = [contexts[0]]
    if len(contexts) > 2:
        show.append(contexts[len(contexts) // 2])
    show.append(contexts[-1])
    show = list(dict.fromkeys(show))  # deduplicate

    fig, axes = plt.subplots(1, len(show), figsize=(5.5 * len(show), 5), sharey=True)
    if len(show) == 1:
        axes = [axes]
    fig.suptitle(
        f"Reliability Diagrams — {task}  ({head.upper()},  K={k})\n"
        "Diagonal = perfect calibration",
        fontsize=11,
    )

    for ax, ctx in zip(axes, show):
        df  = parquets[ctx]
        sub = aggregate_subjects(df, k=k)
        if sub.empty or sub["true_label"].nunique() < 2:
            ax.set_title(f"L={ctx}\n(insufficient data)")
            continue

        ece, mids, fracs, wts = compute_ece(
            sub["true_label"].values, sub["mean_prob"].values)

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
        if len(mids):
            ax.bar(mids, fracs, width=0.08, alpha=0.35, color="#4C72B0",
                   label="Fraction positive")
            ax.plot(mids, fracs, "o-", color="#4C72B0", lw=2, ms=6)
        ax.fill_between([0, 1], [0, 1], [0, 1], alpha=0.04, color="gray")
        ax.set_title(
            f"L = {ctx}" +
            (f"  ({CONTEXT_TO_MIN[ctx]:.0f} min)" if ctx in CONTEXT_TO_MIN else "") +
            f"\nECE = {ece:.4f}",
            fontsize=10,
        )
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    fig.tight_layout()
    stem = f"{task}_{head}_calibration_2A_reliability"
    save_figure(fig, out_dir, stem)
    plt.close(fig)


# ── Plot 2B: ECE vs context length ────────────────────────────────────────────

def plot_ece_vs_context(all_parquets: dict, task: str, heads: list,
                        k: int, results_dir: Path, split: str,
                        run_tag: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False

    for head in heads:
        pqs = all_parquets.get(head, {})
        if not pqs:
            continue
        hs = HEAD_STYLE.get(head, {"color": "gray", "marker": "o", "ls": "-", "label": head})
        contexts = sorted(pqs.keys(), key=lambda c: CTX_ORDER.get(c, 99))
        xs, ys   = [], []
        for ctx in contexts:
            sub = aggregate_subjects(pqs[ctx], k=k)
            if sub.empty or sub["true_label"].nunique() < 2:
                continue
            ece, *_ = compute_ece(sub["true_label"].values, sub["mean_prob"].values)
            ctx_min  = CONTEXT_TO_MIN.get(ctx)
            if ctx_min is None:
                continue
            xs.append(ctx_min); ys.append(ece)
        if xs:
            ax.plot(xs, ys, color=hs["color"], marker=hs["marker"],
                    ls=hs["ls"], lw=2, ms=8, label=hs["label"])
            any_data = True

    if not any_data:
        plt.close(fig)
        print("  [2B] No data to plot")
        return

    ctx_min_vals = sorted(CONTEXT_TO_MIN.values())
    ax.set_xscale("log")
    ax.set_xticks(ctx_min_vals)
    ax.set_xticklabels(
        [k for k, v in sorted(CONTEXT_TO_MIN.items(), key=lambda x: x[1])],
        fontsize=9,
    )
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Context Length", fontsize=11)
    ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=11)
    ax.set_title(
        f"ECE vs Context Length — {task}  (K={k})\n"
        "Lower is better; longer context should improve calibration",
        fontsize=11,
    )
    ax.legend()
    fig.tight_layout()

    stem = f"{task}_calibration_2B_ece_vs_context"
    save_figure(fig, out_dir, stem)
    plt.close(fig)


# ── Plot 2C: ECE vs K ─────────────────────────────────────────────────────────

def plot_ece_vs_k(parquets: dict, task: str, head: str,
                  k_values: list, out_dir: Path) -> None:
    contexts = sorted(parquets.keys(), key=lambda c: CTX_ORDER.get(c, 99))
    if not contexts:
        return

    n = max(len(contexts), 2)
    cmap = cm.get_cmap("viridis_r", n)
    colors = {c: cmap(i / (n - 1)) for i, c in enumerate(contexts)}

    fig, ax = plt.subplots(figsize=(8, 5))
    any_data = False

    for ctx in contexts:
        df     = parquets[ctx]
        max_k  = df.groupby(["subject_id", "dataset"]).size().min()
        valid_k = [k for k in k_values if k == "all" or int(k) <= max_k]
        xs, ys = [], []
        for k in valid_k:
            sub = aggregate_subjects(df, k=k)
            if sub.empty or sub["true_label"].nunique() < 2:
                continue
            ece, *_ = compute_ece(sub["true_label"].values, sub["mean_prob"].values)
            # x-axis: numeric K (use actual count for "all")
            if k == "all":
                xval = int(df.groupby(["subject_id", "dataset"]).size().median())
            else:
                xval = int(k)
            xs.append(xval); ys.append(ece)
        if xs:
            ax.plot(xs, ys, color=colors[ctx], lw=2, marker="o",
                    ms=5, label=ctx)
            any_data = True

    if not any_data:
        plt.close(fig)
        print("  [2C] No data")
        return

    ax.set_xlabel("K (windows per subject at inference)", fontsize=11)
    ax.set_ylabel("Expected Calibration Error (ECE)", fontsize=11)
    ax.set_title(
        f"ECE vs K — {task}  ({head.upper()})\n"
        "Aggregating more windows reduces calibration error",
        fontsize=11,
    )
    ax.legend(title="Context", fontsize=8, title_fontsize=9)
    fig.tight_layout()

    stem = f"{task}_{head}_calibration_2C_ece_vs_k"
    save_figure(fig, out_dir, stem)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

import matplotlib.ticker  # noqa: E402 (needed in plot_ece_vs_context)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot model calibration: reliability diagrams, ECE vs context, ECE vs K."
    )
    parser.add_argument("--task",    required=True)
    parser.add_argument("--head",    default="lstm",
                        help="Primary head for plots 2A and 2C (default: lstm)")
    parser.add_argument("--heads",   nargs="+",
                        default=None,
                        help="Heads for plot 2B ECE vs context (default: uses --head only). "
                             "Set to 'lstm transformer mean_pool' for multi-head comparison.")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/scratch/boshra95/psg/unified/results/phase0_v2"),
                        dest="results_dir")
    parser.add_argument("--split",   default="test", choices=["train", "val", "test"])
    parser.add_argument("--run-tag", default="", dest="run_tag")
    parser.add_argument("--k",       type=int, default=10,
                        help="K for reliability diagram and ECE vs context (default: 10)")
    parser.add_argument("--k-values", nargs="+",
                        default=["1", "2", "3", "5", "8", "10", "15", "20", "30", "50", "all"],
                        dest="k_values",
                        help="K values for ECE vs K plot (default: 1 2 3 5 8 10 15 20 30 50 all)")
    parser.add_argument("--plots",   nargs="+", default=["2A", "2B", "2C"])
    parser.add_argument("--repo-figures-dir", type=Path, default=None,
                        dest="repo_figures_dir",
                        help="Also mirror PNGs into this repo dir (e.g. "
                             "results/figures/phase0_v3). Default: no repo mirror.")
    args = parser.parse_args()

    configure_repo_figures(args.results_dir, args.repo_figures_dir)
    heads_2b = args.heads or [args.head]
    k_values = [
        "all" if str(k).lower() == "all" else int(k)
        for k in args.k_values
    ]

    out_dir = args.results_dir / "figures" / f"{args.task}_{args.head}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load parquets for primary head
    print(f"Loading parquets for {args.task}/{args.head} [{args.split}] ...")
    primary_pqs = load_parquets(args.results_dir, args.task, args.head,
                                args.split, args.run_tag)
    if not primary_pqs:
        print(f"ERROR: No parquets found in "
              f"{inference_dir(args.results_dir, args.task, args.head, args.run_tag)}")
        return
    print(f"  Found contexts: {sorted(primary_pqs.keys())}")

    if "2A" in args.plots:
        plot_reliability_diagrams(primary_pqs, args.task, args.head, args.k, out_dir)

    if "2B" in args.plots:
        all_pqs = {}
        for h in heads_2b:
            all_pqs[h] = load_parquets(args.results_dir, args.task, h,
                                       args.split, args.run_tag)
        plot_ece_vs_context(all_pqs, args.task, heads_2b, args.k,
                            args.results_dir, args.split, args.run_tag, out_dir)

    if "2C" in args.plots:
        plot_ece_vs_k(primary_pqs, args.task, args.head, k_values, out_dir)

    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
