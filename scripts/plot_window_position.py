#!/usr/bin/env python3
"""
plot_window_position.py — §4 Window Position and Temporal Structure

Reads per-window prediction parquets (from infer_subject_windows.py) and shows
how the model's predicted probability varies with normalised window position within
the night recording.

  4A: Position-probability profiles — mean prob_class1 vs normalised window
      position (0 = start, 1 = end of recording), plotted separately for
      truly-positive and truly-negative subjects, per context length.

  4B: Prediction variance vs position — mean within-subject std(prob) at each
      position bin, showing whether predictions become less position-dependent
      as context length grows.

Normalised window position:
  norm_pos = window_idx / max(window_idx per subject)

Window bins: 20 equal bins from 0 to 1.

Usage:
  python scripts/plot_window_position.py \\
      --task sex_binary \\
      --head lstm \\
      --results-dir /scratch/boshra95/psg/unified/results/phase0_v2 \\
      --split test

Output:
  {results_dir}/figures/{task}_{head}/window_position_4A_profiles.{png,pdf}
  {results_dir}/figures/{task}_{head}/window_position_4B_variance.{png,pdf}
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

N_BINS = 20  # position bins

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


# ── Normalise positions and compute profile ────────────────────────────────────

def compute_position_profile(df: pd.DataFrame,
                              prob_col: str = "prob_class1",
                              n_bins: int = N_BINS) -> dict:
    """
    Compute mean prob and mean std per position bin, split by true_label.
    Returns dict with keys:
      bin_centers, pos_mean, pos_std, neg_mean, neg_std, all_mean, all_std
    """
    if prob_col not in df.columns:
        return {}

    # Normalise window position within each subject
    df = df.copy()
    max_idx = df.groupby(["subject_id", "dataset"])["window_idx"].transform("max")
    df["norm_pos"] = np.where(max_idx > 0, df["window_idx"] / max_idx, 0.0)

    bins       = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    pos_mean = np.full(n_bins, np.nan)
    pos_std  = np.full(n_bins, np.nan)
    neg_mean = np.full(n_bins, np.nan)
    neg_std  = np.full(n_bins, np.nan)
    all_mean = np.full(n_bins, np.nan)
    all_std  = np.full(n_bins, np.nan)

    pos_df = df[df["true_label"] == 1]
    neg_df = df[df["true_label"] == 0]

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask_all = (df["norm_pos"] >= lo) & (df["norm_pos"] < hi)
        mask_pos = (pos_df["norm_pos"] >= lo) & (pos_df["norm_pos"] < hi)
        mask_neg = (neg_df["norm_pos"] >= lo) & (neg_df["norm_pos"] < hi)

        if mask_all.sum() >= 5:
            vals = df.loc[mask_all, prob_col].values
            all_mean[i] = vals.mean()
            all_std[i]  = vals.std()
        if mask_pos.sum() >= 5:
            vals = pos_df.loc[mask_pos, prob_col].values
            pos_mean[i] = vals.mean()
            pos_std[i]  = vals.std()
        if mask_neg.sum() >= 5:
            vals = neg_df.loc[mask_neg, prob_col].values
            neg_mean[i] = vals.mean()
            neg_std[i]  = vals.std()

    return {
        "bin_centers": bin_centers,
        "pos_mean": pos_mean, "pos_std": pos_std,
        "neg_mean": neg_mean, "neg_std": neg_std,
        "all_mean": all_mean, "all_std": all_std,
    }


# ── Plot 4A: Position-probability profiles ─────────────────────────────────────

def plot_position_profiles(parquets: dict, task: str, head: str, out_dir: Path) -> None:
    contexts = sorted(parquets.keys(), key=lambda c: CTX_ORDER.get(c, 99))
    if not contexts:
        print("  [4A] No parquets found")
        return

    colors = ctx_color_map(contexts)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_pos, ax_neg = axes

    any_data = False
    for ctx in contexts:
        df  = parquets[ctx]
        if "prob_class1" not in df.columns:
            continue
        pr  = compute_position_profile(df)
        if not pr:
            continue
        c   = colors[ctx]
        bin_centers = pr["bin_centers"]

        # Positive subjects
        valid_p = ~np.isnan(pr["pos_mean"])
        if valid_p.any():
            ax_pos.plot(bin_centers[valid_p], pr["pos_mean"][valid_p],
                        color=c, lw=2, label=f"L={ctx}")
            ax_pos.fill_between(
                bin_centers[valid_p],
                (pr["pos_mean"] - pr["pos_std"])[valid_p],
                (pr["pos_mean"] + pr["pos_std"])[valid_p],
                color=c, alpha=0.12,
            )

        # Negative subjects
        valid_n = ~np.isnan(pr["neg_mean"])
        if valid_n.any():
            ax_neg.plot(bin_centers[valid_n], pr["neg_mean"][valid_n],
                        color=c, lw=2, label=f"L={ctx}")
            ax_neg.fill_between(
                bin_centers[valid_n],
                (pr["neg_mean"] - pr["neg_std"])[valid_n],
                (pr["neg_mean"] + pr["neg_std"])[valid_n],
                color=c, alpha=0.12,
            )
        any_data = True

    if not any_data:
        plt.close(fig)
        print("  [4A] No valid data")
        return

    for ax, title in [
        (ax_pos, "Positive subjects (true label = 1)"),
        (ax_neg, "Negative subjects (true label = 0)"),
    ]:
        ax.axvline(0.5, color="gray", ls=":", lw=1, alpha=0.5)
        ax.set_xlabel("Normalised window position\n(0 = start, 1 = end of night)", fontsize=10)
        ax.set_ylabel("Mean prob_class1 (±1 SD)", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.legend(title="Context", fontsize=8, title_fontsize=9, ncol=2)

    fig.suptitle(
        f"Window Position Profiles — {task}  ({head.upper()})\n"
        "Does predicted probability depend on where in the night the window falls?",
        fontsize=11,
    )
    fig.tight_layout()

    stem = f"{task}_{head}_window_position_4A_profiles"
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}",
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4A] Saved: {stem}.{{png,pdf}}")


# ── Plot 4B: Prediction variance vs position ──────────────────────────────────

def plot_variance_vs_position(parquets: dict, task: str, head: str, out_dir: Path) -> None:
    contexts = sorted(parquets.keys(), key=lambda c: CTX_ORDER.get(c, 99))
    if not contexts:
        return

    colors = ctx_color_map(contexts)
    fig, ax = plt.subplots(figsize=(9, 5))
    any_data = False

    for ctx in contexts:
        df = parquets[ctx]
        if "prob_class1" not in df.columns:
            continue
        pr = compute_position_profile(df)
        if not pr:
            continue
        valid = ~np.isnan(pr["all_std"])
        if valid.any():
            ax.plot(pr["bin_centers"][valid], pr["all_std"][valid],
                    color=colors[ctx], lw=2, label=f"L={ctx}")
            any_data = True

    if not any_data:
        plt.close(fig)
        print("  [4B] No valid data")
        return

    ax.axvline(0.5, color="gray", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("Normalised window position (0 = start, 1 = end)", fontsize=11)
    ax.set_ylabel("Mean std(prob_class1) per position bin", fontsize=11)
    ax.set_title(
        f"Prediction Variance vs Night Position — {task}  ({head.upper()})\n"
        "Lower variance = position-independent predictions (longer context may flatten this)",
        fontsize=11,
    )
    ax.legend(title="Context", fontsize=8, title_fontsize=9, ncol=2)
    ax.set_xlim(0, 1)
    fig.tight_layout()

    stem = f"{task}_{head}_window_position_4B_variance"
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}",
                    dpi=150 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4B] Saved: {stem}.{{png,pdf}}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot window position profiles (§4 Window Position Analysis)."
    )
    parser.add_argument("--task",    required=True)
    parser.add_argument("--head",    default="lstm")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/scratch/boshra95/psg/unified/results/phase0_v2"),
                        dest="results_dir")
    parser.add_argument("--split",   default="test", choices=["train", "val", "test"])
    parser.add_argument("--run-tag", default="", dest="run_tag")
    parser.add_argument("--contexts", nargs="+", default=None,
                        help="Restrict to these context lengths (default: all available)")
    parser.add_argument("--plots",   nargs="+", default=["4A", "4B"])
    args = parser.parse_args()

    out_dir = args.results_dir / "figures" / f"{args.task}_{args.head}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading parquets for {args.task}/{args.head} [{args.split}] ...")
    parquets = load_parquets(args.results_dir, args.task, args.head,
                              args.split, args.run_tag)
    if args.contexts:
        parquets = {c: df for c, df in parquets.items() if c in args.contexts}
    if not parquets:
        print("ERROR: No parquets found. Run inference first.")
        return
    print(f"  Found contexts: {sorted(parquets.keys())}")

    if "4A" in args.plots:
        plot_position_profiles(parquets, args.task, args.head, out_dir)
    if "4B" in args.plots:
        plot_variance_vs_position(parquets, args.task, args.head, out_dir)

    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
