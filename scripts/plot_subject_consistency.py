#!/usr/bin/env python3
"""
plot_subject_consistency.py — §5 Per-Subject Consistency and Prediction Stability

Reads per-window prediction parquets and characterises how variable predictions
are within a single subject across their night's windows.

  5A: Variance distribution — violin plots of within-subject std(prob_class1)
      for correctly- vs incorrectly-classified subjects, per context length.
      High variance = uncertain model; low variance = confident and consistent.

  5B: Prediction variance vs K — shows how aggregating K windows reduces
      within-subject prediction variance (mean std of K-window mean predictions,
      across subjects), confirming that aggregation stabilises predictions.

  5C: Hard-subject analysis — fraction of subjects correctly predicted at
      0, 1, 2, … all context lengths. Subjects at "0" are never correctly
      classified regardless of context and represent irreducible hard cases.

Usage:
  python scripts/plot_subject_consistency.py \\
      --task sex_binary \\
      --head lstm \\
      --results-dir /scratch/boshra95/psg/unified/results/phase0_v2 \\
      --split test

Output:
  {results_dir}/figures/{task}_{head}/subject_consistency_5A_variance.{png,pdf}
  {results_dir}/figures/{task}_{head}/subject_consistency_5B_variance_vs_k.{png,pdf}
  {results_dir}/figures/{task}_{head}/subject_consistency_5C_hard_subjects.{png,pdf}
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Style ─────────────────────────────────────────────────────────────────────

CONTEXT_TO_MIN = {
    "30s": 0.5, "10m": 10.0, "40m": 40.0,
    "80m": 80.0, "120m": 120.0, "240m": 240.0,
}
CTX_ORDER = {c: i for i, c in enumerate(CONTEXT_TO_MIN)}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_parquets(results_dir: Path, task: str, head: str,
                  split: str, run_tag: str = "") -> dict:
    exp_id = f"{task}_{head}" + (f"_{run_tag}" if run_tag else "")
    idir   = results_dir / "inference" / exp_id
    data   = {}
    for ctx_dir in sorted(idir.glob("context_*")):
        ctx = ctx_dir.name.removeprefix("context_")
        pq  = ctx_dir / f"{split}_windows.parquet"
        if pq.exists():
            data[ctx] = pd.read_parquet(pq)
    return data


def ctx_color_map(contexts: list) -> dict:
    n = max(len(contexts), 2)
    cmap = cm.get_cmap("viridis_r", n)
    return {c: cmap(i / (n - 1)) for i, c in enumerate(contexts)}


# ── Subject-level stats ────────────────────────────────────────────────────────

def subject_stats(df: pd.DataFrame, prob_col: str = "prob_class1") -> pd.DataFrame:
    """
    Compute per-subject statistics.
    Returns df with: subject_id, dataset, true_label, mean_prob, std_prob,
    n_windows, correct (mean_prob > 0.5 matches true_label).
    """
    if prob_col not in df.columns:
        return pd.DataFrame()
    stats = (
        df.groupby(["subject_id", "dataset"])
        .agg(
            mean_prob=(prob_col, "mean"),
            std_prob=(prob_col, "std"),
            n_windows=(prob_col, "count"),
            true_label=("true_label", "first"),
        )
        .reset_index()
    )
    stats["std_prob"]  = stats["std_prob"].fillna(0.0)
    stats["correct"]   = (stats["mean_prob"] > 0.5) == stats["true_label"].astype(bool)
    return stats


def _select_k(group: pd.DataFrame, k: int) -> pd.DataFrame:
    n = len(group)
    if n <= k:
        return group
    idx = np.linspace(0, n - 1, k, dtype=int)
    return group.iloc[idx]


# ── Plot 5A: Variance distribution ────────────────────────────────────────────

def plot_variance_distribution(parquets: dict, task: str, head: str,
                               out_dir: Path) -> None:
    contexts = sorted(parquets.keys(), key=lambda c: CTX_ORDER.get(c, 99))
    # Show at most 3 representative contexts
    show = [contexts[0]]
    if len(contexts) > 2:
        show.append(contexts[len(contexts) // 2])
    show.append(contexts[-1])
    show = list(dict.fromkeys(show))

    n = len(show)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle(
        f"Within-Subject Prediction Variance — {task}  ({head.upper()})\n"
        "std(prob_class1) across all windows for each subject",
        fontsize=11,
    )

    any_data = False
    for ax, ctx in zip(axes, show):
        df    = parquets[ctx]
        stats = subject_stats(df)
        if stats.empty:
            ax.set_title(f"L={ctx}\n(no data)"); continue
        correct_std   = stats.loc[stats["correct"],  "std_prob"].dropna().values
        incorrect_std = stats.loc[~stats["correct"], "std_prob"].dropna().values
        if len(correct_std) < 2 or len(incorrect_std) < 2:
            ax.set_title(f"L={ctx}\n(insufficient)"); continue
        vp = ax.violinplot([correct_std, incorrect_std],
                           positions=[0, 1], showmedians=True, showextrema=False)
        vp["cmedians"].set_color("red")
        vp["bodies"][0].set_facecolor("#4C72B0"); vp["bodies"][0].set_alpha(0.7)
        vp["bodies"][1].set_facecolor("#DD8452"); vp["bodies"][1].set_alpha(0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Correct", "Incorrect"])
        ax.set_title(
            f"L = {ctx}" + (f"  ({CONTEXT_TO_MIN[ctx]:.0f} min)"
                            if ctx in CONTEXT_TO_MIN else ""),
            fontsize=10,
        )
        ax.set_ylabel("std(prob_class1)")
        any_data = True

    if not any_data:
        plt.close(fig)
        print("  [5A] No data")
        return

    fig.tight_layout()
    stem = f"{task}_{head}_subject_consistency_5A_variance"
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}",
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"  [5A] Saved: {stem}.{{png,pdf}}")


# ── Plot 5B: Prediction variance vs K ─────────────────────────────────────────

def plot_variance_vs_k(parquets: dict, task: str, head: str, out_dir: Path) -> None:
    contexts = sorted(parquets.keys(), key=lambda c: CTX_ORDER.get(c, 99))
    colors   = ctx_color_map(contexts)
    k_values = [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]

    fig, ax = plt.subplots(figsize=(9, 5))
    any_data = False

    for ctx in contexts:
        df      = parquets[ctx]
        if "prob_class1" not in df.columns:
            continue
        max_k   = df.groupby(["subject_id", "dataset"]).size().min()
        xs, ys  = [], []
        for k in k_values:
            if k > max_k:
                continue
            sampled = (
                df.sort_values(["subject_id", "dataset", "window_idx"])
                .groupby(["subject_id", "dataset"], group_keys=False)
                .apply(lambda g: _select_k(g, k))
                .reset_index(drop=True)
            )
            subj_means = (
                sampled
                .groupby(["subject_id", "dataset"])["prob_class1"]
                .mean()
            )
            xs.append(k)
            ys.append(float(subj_means.std()))

        if xs:
            ax.plot(xs, ys, color=colors[ctx], lw=2, marker="o", ms=5, label=ctx)
            any_data = True

    if not any_data:
        plt.close(fig)
        print("  [5B] No data")
        return

    ax.set_xlabel("K (windows aggregated per subject)", fontsize=11)
    ax.set_ylabel("Std of per-subject mean(prob_class1)", fontsize=11)
    ax.set_title(
        f"Prediction Variance vs K — {task}  ({head.upper()})\n"
        "Aggregation stabilises predictions (lower std = more consistent)",
        fontsize=11,
    )
    ax.legend(title="Context", fontsize=8, title_fontsize=9, ncol=2)
    fig.tight_layout()

    stem = f"{task}_{head}_subject_consistency_5B_variance_vs_k"
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}",
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"  [5B] Saved: {stem}.{{png,pdf}}")


# ── Plot 5C: Hard-subject analysis ────────────────────────────────────────────

def plot_hard_subject_analysis(parquets: dict, task: str, head: str,
                               out_dir: Path) -> None:
    contexts = sorted(parquets.keys(), key=lambda c: CTX_ORDER.get(c, 99))
    if not contexts:
        return

    # Build a set of all subjects (subject_id, dataset) seen across all contexts
    all_ids: dict = {}  # (sid, ds) -> {ctx: correct}
    for ctx in contexts:
        df    = parquets[ctx]
        stats = subject_stats(df)
        if stats.empty:
            continue
        for _, row in stats.iterrows():
            key = (row["subject_id"], row["dataset"])
            if key not in all_ids:
                all_ids[key] = {}
            all_ids[key][ctx] = bool(row["correct"])

    if not all_ids:
        print("  [5C] No data")
        return

    # Count how many context lengths each subject is correctly predicted at
    n_ctx = len(contexts)
    correct_counts = np.array([
        sum(1 for ctx in contexts if vals.get(ctx, False))
        for vals in all_ids.values()
    ])

    n_subj = len(correct_counts)
    n = cm.get_cmap("viridis_r", n_ctx + 1)
    colors = [n(i / n_ctx) for i in range(n_ctx + 1)]

    fig, ax = plt.subplots(figsize=(max(8, n_ctx * 1.5), 5))
    for i in range(n_ctx + 1):
        frac = (correct_counts == i).sum() / n_subj * 100
        bar  = ax.bar(i, frac, color=colors[i], edgecolor="white", width=0.7)
        if frac > 1:
            ax.text(i, frac + 0.3, f"{frac:.1f}%",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(n_ctx + 1))
    ax.set_xticklabels([f"{i}/{n_ctx}" for i in range(n_ctx + 1)], fontsize=10)
    ax.set_xlabel("Number of context lengths at which subject is correctly predicted", fontsize=11)
    ax.set_ylabel("Fraction of subjects (%)", fontsize=11)
    ax.set_title(
        f"Hard-Subject Analysis — {task}  ({head.upper()})\n"
        f"N = {n_subj} subjects · 0/{n_ctx} = never correct (irreducible hard cases)",
        fontsize=11,
    )
    fig.tight_layout()

    stem = f"{task}_{head}_subject_consistency_5C_hard_subjects"
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}",
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"  [5C] Saved: {stem}.{{png,pdf}}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-subject consistency (§5)."
    )
    parser.add_argument("--task",    required=True)
    parser.add_argument("--head",    default="lstm")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/scratch/boshra95/psg/unified/results/phase0_v2"),
                        dest="results_dir")
    parser.add_argument("--split",   default="test", choices=["train", "val", "test"])
    parser.add_argument("--run-tag", default="", dest="run_tag")
    parser.add_argument("--plots",   nargs="+", default=["5A", "5B", "5C"])
    args = parser.parse_args()

    out_dir = args.results_dir / "figures" / f"{args.task}_{args.head}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading parquets for {args.task}/{args.head} [{args.split}] ...")
    parquets = load_parquets(args.results_dir, args.task, args.head,
                              args.split, args.run_tag)
    if not parquets:
        print("ERROR: No parquets found. Run inference first.")
        return
    print(f"  Found contexts: {sorted(parquets.keys())}")

    if "5A" in args.plots:
        plot_variance_distribution(parquets, args.task, args.head, out_dir)
    if "5B" in args.plots:
        plot_variance_vs_k(parquets, args.task, args.head, out_dir)
    if "5C" in args.plots:
        plot_hard_subject_analysis(parquets, args.task, args.head, out_dir)

    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
