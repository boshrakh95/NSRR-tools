#!/usr/bin/env python3
"""
plot_precision_recall.py — §8 Precision-Recall Analysis

Reads per-window prediction parquets and produces three precision-recall figures:

  8A: PR curves at K=all per context length — one curve per context on the
      same axes, with AUC-PR annotated. Shows whether longer context shifts
      the PR frontier.

  8B: AUC-PR (average precision) vs context length — one line per head,
      analogous to the AUROC saturation plot. Requires multiple heads to
      have been inferred; with a single head still shows a meaningful curve.

  8C: Majority-vote threshold sweep — instead of varying the probability
      threshold, vary the "vote threshold" t = 1..K (number of windows that
      must predict positive for the subject to be called positive). For each
      t, compute subject-level precision and recall, yielding a discrete PR
      tradeoff curve purely from the vote count.

Usage:
  python scripts/plot_precision_recall.py \\
      --task sex_binary \\
      --head lstm \\
      --results-dir /scratch/boshra95/psg/unified/results/phase0_v2 \\
      --split test

  # Multi-head comparison for 8B:
  python scripts/plot_precision_recall.py \\
      --task sex_binary \\
      --heads lstm transformer mean_pool \\
      --results-dir /scratch/boshra95/psg/unified/results/phase0_v2 \\
      --split test \\
      --plots 8B

Output:
  {results_dir}/figures/{task}_{head}/pr_8A_curves.{png,pdf}
  {results_dir}/figures/{task}_{head}/pr_8B_aucpr_vs_context.{png,pdf}
  {results_dir}/figures/{task}_{head}/pr_8C_vote_sweep.{png,pdf}
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

try:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── Style ─────────────────────────────────────────────────────────────────────

CONTEXT_TO_MIN = {
    "30s": 0.5, "10m": 10.0, "40m": 40.0,
    "80m": 80.0, "120m": 120.0, "240m": 240.0,
}
CTX_ORDER = {c: i for i, c in enumerate(CONTEXT_TO_MIN)}

HEAD_STYLE = {
    "lstm":        {"color": "#3A7EBF", "marker": "o", "ls": "-",  "label": "LSTM"},
    "transformer": {"color": "#E86A33", "marker": "s", "ls": "--", "label": "Transformer"},
    "mean_pool":   {"color": "#44A15E", "marker": "^", "ls": ":",  "label": "Mean Pool"},
}

plt.rcParams.update({
    "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "serif", "font.size": 9,
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


# ── Aggregation helpers ────────────────────────────────────────────────────────

def aggregate_subjects(df: pd.DataFrame, k: int | None,
                       prob_col: str = "prob_class1") -> pd.DataFrame:
    """
    Aggregate all windows per subject (k=None → K=all) or evenly-sampled K windows.
    Returns DataFrame with: subject_id, dataset, mean_prob, true_label.
    """
    if prob_col not in df.columns:
        return pd.DataFrame()

    if k is not None:
        def _select_k(g):
            n = len(g)
            if n <= k:
                return g
            idx = np.linspace(0, n - 1, k, dtype=int)
            return g.iloc[idx]
        df = (
            df.sort_values(["subject_id", "dataset", "window_idx"])
            .groupby(["subject_id", "dataset"], group_keys=False)
            .apply(_select_k)
            .reset_index(drop=True)
        )

    return (
        df.groupby(["subject_id", "dataset"])
        .agg(mean_prob=(prob_col, "mean"), true_label=("true_label", "first"))
        .reset_index()
    )


# ── Plot 8A: PR curves per context length ─────────────────────────────────────

def plot_pr_curves(parquets: dict, task: str, head: str, out_dir: Path) -> None:
    if not HAS_SKLEARN:
        print("  [8A] sklearn not available"); return

    contexts = sorted(parquets.keys(), key=lambda c: CTX_ORDER.get(c, 99))
    if not contexts:
        print("  [8A] No parquets found"); return

    colors = ctx_color_map(contexts)
    fig, ax = plt.subplots(figsize=(8, 7))
    any_data = False

    # Chance line (prevalence)
    all_labels = []
    for ctx in contexts:
        df = parquets[ctx]
        if "true_label" in df.columns:
            all_labels.extend(df["true_label"].tolist())
    if all_labels:
        prevalence = np.mean(all_labels)
        ax.axhline(prevalence, color="gray", ls=":", lw=1, alpha=0.6,
                   label=f"Chance (prev={prevalence:.2f})")

    for ctx in contexts:
        df   = parquets[ctx]
        subj = aggregate_subjects(df, k=None)
        if subj.empty or subj["true_label"].nunique() < 2 or len(subj) < 5:
            continue
        try:
            prec, rec, _ = precision_recall_curve(
                subj["true_label"], subj["mean_prob"])
            ap = average_precision_score(subj["true_label"], subj["mean_prob"])
        except Exception:
            continue
        lbl = f"L={ctx}  AP={ap:.3f}"
        ax.plot(rec, prec, color=colors[ctx], lw=2, label=lbl)
        any_data = True

    if not any_data:
        plt.close(fig)
        print("  [8A] No valid data for PR curves")
        return

    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.legend(title="Context", fontsize=8, title_fontsize=9, ncol=2,
              loc="upper right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()

    stem = f"{task}_{head}_pr_8A_curves"
    save_figure(fig, out_dir, stem)
    plt.close(fig)


# ── Plot 8B: AUC-PR vs context length (multi-head) ────────────────────────────

def plot_aucpr_vs_context(results_dir: Path, task: str, heads: list,
                          split: str, run_tag: str, out_dir: Path) -> None:
    if not HAS_SKLEARN:
        print("  [8B] sklearn not available"); return

    fig, ax = plt.subplots(figsize=(9, 6))
    any_data = False

    for head in heads:
        parquets = load_parquets(results_dir, task, head, split, run_tag)
        if not parquets:
            continue
        contexts = sorted(parquets.keys(), key=lambda c: CTX_ORDER.get(c, 99))
        style = HEAD_STYLE.get(
            head.lower(),
            {"color": "gray", "marker": "o", "ls": "-", "label": head.upper()},
        )
        xs, ys = [], []
        for ctx in contexts:
            df   = parquets[ctx]
            subj = aggregate_subjects(df, k=None)
            if subj.empty or subj["true_label"].nunique() < 2 or len(subj) < 5:
                continue
            ctx_min = CONTEXT_TO_MIN.get(ctx)
            if ctx_min is None:
                continue
            try:
                ap = average_precision_score(subj["true_label"], subj["mean_prob"])
            except Exception:
                continue
            xs.append(ctx_min); ys.append(ap)

        if not xs:
            continue
        ax.plot(xs, ys, color=style["color"], marker=style["marker"],
                ls=style["ls"], lw=2, ms=8, label=style["label"])
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", (x, y),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=7, ha="center", color=style["color"])
        any_data = True

    if not any_data:
        plt.close(fig)
        print("  [8B] No valid data")
        return

    # X-axis: log scale with context labels
    ctx_min_vals = sorted(CONTEXT_TO_MIN.values())
    ax.set_xscale("log")
    ax.set_xticks(ctx_min_vals)
    ax.set_xticklabels(
        [k for k, v in sorted(CONTEXT_TO_MIN.items(), key=lambda x: x[1])],
        fontsize=9,
    )
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Context length (minutes)", fontsize=10)
    ax.set_ylabel("AUC-PR (Average Precision)  K=all", fontsize=10)
    ax.legend(title="Head", fontsize=10)
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()

    stem = f"{task}_pr_8B_aucpr_vs_context"
    save_figure(fig, out_dir, stem)
    plt.close(fig)


# ── Plot 8C: Majority-vote threshold sweep ────────────────────────────────────

def plot_vote_sweep(parquets: dict, task: str, head: str,
                    contexts_show: list | None, out_dir: Path) -> None:
    """
    For each context, vary vote threshold t = 1..K.
    A subject is predicted positive if at least t of its K windows have
    prob_class1 > 0.5.  Compute precision and recall at each t.
    """
    contexts = sorted(parquets.keys(), key=lambda c: CTX_ORDER.get(c, 99))
    if not contexts:
        print("  [8C] No parquets found"); return

    if contexts_show:
        contexts = [c for c in contexts if c in contexts_show]
    else:
        # Show at most 4 representative contexts
        step = max(1, len(contexts) // 4)
        contexts = contexts[::step][:4]

    colors = ctx_color_map(contexts)
    fig, ax = plt.subplots(figsize=(8, 7))
    any_data = False

    for ctx in contexts:
        df = parquets[ctx]
        if "prob_class1" not in df.columns:
            continue

        # Subject-level: list of (n_pos_votes, total_windows, true_label)
        subj = (
            df.assign(pos_vote=lambda d: (d["prob_class1"] > 0.5).astype(int))
            .groupby(["subject_id", "dataset"])
            .agg(
                n_pos_votes=("pos_vote", "sum"),
                n_windows=("pos_vote", "count"),
                true_label=("true_label", "first"),
            )
            .reset_index()
        )
        if subj.empty or subj["true_label"].nunique() < 2:
            continue

        max_k = int(subj["n_windows"].max())
        if max_k < 1:
            continue

        precs, recs, ts = [], [], []
        for t in range(1, max_k + 1):
            pred_pos = (subj["n_pos_votes"] >= t).astype(int)
            tp = ((pred_pos == 1) & (subj["true_label"] == 1)).sum()
            fp = ((pred_pos == 1) & (subj["true_label"] == 0)).sum()
            fn = ((pred_pos == 0) & (subj["true_label"] == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precs.append(prec); recs.append(rec); ts.append(t)

        if not precs:
            continue

        # Plot as scatter + line, annotate t at key points
        ax.plot(recs, precs, color=colors[ctx], lw=2, marker="o", ms=5,
                label=f"L={ctx}")
        # Annotate first, mid, last t values
        annot_idx = {0, len(ts) // 2, len(ts) - 1}
        for i in annot_idx:
            if i < len(ts):
                ax.annotate(f"t={ts[i]}", (recs[i], precs[i]),
                            textcoords="offset points", xytext=(4, 3),
                            fontsize=7, color=colors[ctx])
        any_data = True

    if not any_data:
        plt.close(fig)
        print("  [8C] No valid data for vote sweep")
        return

    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.legend(title="Context", fontsize=9, title_fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()

    stem = f"{task}_{head}_pr_8C_vote_sweep"
    save_figure(fig, out_dir, stem)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precision-recall analysis (§8)."
    )
    parser.add_argument("--task",    required=True)
    parser.add_argument("--head",    default="lstm",
                        help="Primary head for 8A and 8C (default: lstm)")
    parser.add_argument("--heads",   nargs="+", default=None,
                        help="Heads to compare for 8B (default: same as --head)")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/scratch/boshra95/psg/unified/results/phase0_v2"),
                        dest="results_dir")
    parser.add_argument("--split",   default="test", choices=["train", "val", "test"])
    parser.add_argument("--run-tag", default="", dest="run_tag")
    parser.add_argument("--contexts-show", nargs="+", default=None,
                        dest="contexts_show",
                        help="Restrict 8C vote sweep to these context labels")
    parser.add_argument("--plots",   nargs="+", default=["8A", "8B"])
    parser.add_argument("--repo-figures-dir", type=Path, default=None,
                        dest="repo_figures_dir",
                        help="Also mirror PNGs into this repo dir (e.g. "
                             "results/figures/phase0_v3). Default: no repo mirror.")
    args = parser.parse_args()

    if not HAS_SKLEARN:
        print("ERROR: scikit-learn not installed. "
              "Activate sleepfm_env: source ~/sleepfm_env/bin/activate")
        return

    configure_repo_figures(args.results_dir, args.repo_figures_dir)
    heads_8b = args.heads if args.heads else [args.head]
    out_dir  = args.results_dir / "figures" / f"{args.task}_{args.head}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if "8A" in args.plots or "8C" in args.plots:
        print(f"Loading parquets for {args.task}/{args.head} [{args.split}] ...")
        parquets = load_parquets(args.results_dir, args.task, args.head,
                                  args.split, args.run_tag)
        if not parquets:
            print("ERROR: No parquets found. Run inference first.")
            return
        print(f"  Found contexts: {sorted(parquets.keys())}")

        if "8A" in args.plots:
            plot_pr_curves(parquets, args.task, args.head, out_dir)
        if "8C" in args.plots:
            plot_vote_sweep(parquets, args.task, args.head,
                            args.contexts_show, out_dir)

    if "8B" in args.plots:
        print(f"Computing AUC-PR vs context for heads: {heads_8b} ...")
        plot_aucpr_vs_context(args.results_dir, args.task, heads_8b,
                              args.split, args.run_tag, out_dir)

    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
