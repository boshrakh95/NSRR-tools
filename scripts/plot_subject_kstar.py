#!/usr/bin/env python3
"""
plot_subject_kstar.py — §9 Per-Subject K* (Minimum Windows for Correct Classification)

For each subject, finds K* = the minimum number of randomly-sampled windows
needed for the model to correctly classify them (majority-vote or mean-prob
criterion), averaged over several random draws.

  9A: K* distribution histogram — distribution of K* across all subjects
      in the test set, per context length.  Shows which context length has
      the most "easy" subjects (low K*) and how many subjects are always
      hard (K* = ∞ / never correct).

  9B: Coverage curves — fraction of subjects correctly classified using at
      most K windows, as a function of K.  One line per context length.
      Analogous to a CDF: at K=1 we see how many subjects a single window
      gets right; by K=K_max we reach the ceiling.

K* estimation procedure (per subject per context):
  For k in 1..min(K_MAX, n_windows):
    sample REPS random subsets of size k from the subject's windows
    if mean(prob_class1 of subset) > 0.5 matches true_label:
      K* = k; break
  If never correct across all k: K* = inf (marked separately)

Usage:
  python scripts/plot_subject_kstar.py \\
      --task sex_binary \\
      --head lstm \\
      --results-dir /scratch/boshra95/psg/unified/results/phase0_v2 \\
      --split test

  # Restrict to specific contexts:
  python scripts/plot_subject_kstar.py \\
      --task sex_binary \\
      --head lstm \\
      --contexts 30s 10m 80m 240m \\
      --kmax 20 --reps 20

Output:
  {results_dir}/figures/{task}_{head}/kstar_9A_histogram.{png,pdf}
  {results_dir}/figures/{task}_{head}/kstar_9B_coverage.{png,pdf}
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

K_MAX_DEFAULT = 30   # maximum K to evaluate
REPS_DEFAULT  = 20   # random draws per (subject, k)

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


# ── K* estimation ─────────────────────────────────────────────────────────────

def estimate_kstar(df: pd.DataFrame, k_max: int, reps: int,
                   prob_col: str = "prob_class1",
                   rng: np.random.Generator | None = None) -> pd.Series:
    """
    For each subject, estimate K* = minimum k where a random subset of k
    windows produces a correct classification (mean prob_class1 > 0.5
    matching true_label).

    Returns a Series indexed by (subject_id, dataset) with float values:
    K* value or np.inf if never correctly classified at any k ≤ k_max.
    """
    if prob_col not in df.columns:
        return pd.Series(dtype=float)

    if rng is None:
        rng = np.random.default_rng(42)

    kstar_vals = {}
    for (sid, ds), grp in df.groupby(["subject_id", "dataset"]):
        probs      = grp[prob_col].values
        true_label = int(grp["true_label"].iloc[0])
        n_windows  = len(probs)
        this_kmax  = min(k_max, n_windows)

        found_k = np.inf
        for k in range(1, this_kmax + 1):
            correct_any = False
            for _ in range(reps):
                idx   = rng.choice(n_windows, size=k, replace=False)
                pred  = int(probs[idx].mean() > 0.5)
                if pred == true_label:
                    correct_any = True
                    break
            if correct_any:
                found_k = k
                break
        kstar_vals[(sid, ds)] = found_k

    return pd.Series(kstar_vals, dtype=float)


# ── Plot 9A: K* distribution histogram ────────────────────────────────────────

def plot_kstar_histogram(parquets: dict, task: str, head: str,
                         k_max: int, reps: int, out_dir: Path) -> None:
    contexts = sorted(parquets.keys(), key=lambda c: CTX_ORDER.get(c, 99))
    if not contexts:
        print("  [9A] No parquets found"); return

    # Show up to 4 representative contexts side-by-side
    step = max(1, len(contexts) // 4)
    show = contexts[::step][:4]

    n = len(show)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle(
        f"K* Distribution — {task}  ({head.upper()})\n"
        f"K* = minimum windows for correct classification (reps={reps})",
        fontsize=11,
    )

    rng = np.random.default_rng(42)
    any_data = False

    for ax, ctx in zip(axes, show):
        df    = parquets[ctx]
        kstar = estimate_kstar(df, k_max, reps, rng=rng)
        if kstar.empty:
            ax.set_title(f"L={ctx}\n(no data)"); continue

        finite_mask = np.isfinite(kstar.values)
        finite_vals = kstar.values[finite_mask]
        n_inf       = (~finite_mask).sum()
        n_total     = len(kstar)

        bins = np.arange(0.5, k_max + 1.5, 1)
        ax.hist(finite_vals, bins=bins, color="#4C72B0", edgecolor="white",
                alpha=0.85, rwidth=0.8)

        # Annotate never-correct fraction
        if n_inf > 0:
            ax.text(0.97, 0.97, f"Never correct:\n{n_inf}/{n_total} ({100*n_inf/n_total:.1f}%)",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="#DD8452",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DD8452", alpha=0.8))

        ctx_label = f"L={ctx}"
        if ctx in CONTEXT_TO_MIN:
            ctx_label += f" ({CONTEXT_TO_MIN[ctx]:.0f} min)"
        ax.set_title(ctx_label, fontsize=10)
        ax.set_xlabel("K*")
        ax.set_ylabel("Number of subjects")
        ax.set_xlim(0.5, k_max + 0.5)
        any_data = True

    if not any_data:
        plt.close(fig); print("  [9A] No data"); return

    fig.tight_layout()
    stem = f"{task}_{head}_kstar_9A_histogram"
    save_figure(fig, out_dir, stem)
    plt.close(fig)


# ── Plot 9B: Coverage curves ──────────────────────────────────────────────────

def plot_coverage_curves(parquets: dict, task: str, head: str,
                         k_max: int, reps: int, out_dir: Path) -> None:
    contexts = sorted(parquets.keys(), key=lambda c: CTX_ORDER.get(c, 99))
    if not contexts:
        print("  [9B] No parquets found"); return

    colors  = ctx_color_map(contexts)
    fig, ax = plt.subplots(figsize=(9, 6))
    any_data = False
    rng = np.random.default_rng(42)

    for ctx in contexts:
        df    = parquets[ctx]
        kstar = estimate_kstar(df, k_max, reps, rng=rng)
        if kstar.empty:
            continue

        n_total = len(kstar)
        ks      = np.arange(1, k_max + 1)
        # Fraction correctly classified using AT MOST k windows
        coverage = np.array([
            (np.isfinite(kstar.values) & (kstar.values <= k)).sum() / n_total
            for k in ks
        ])

        ctx_min = CONTEXT_TO_MIN.get(ctx)
        lbl     = f"L={ctx}" + (f" ({ctx_min:.0f} min)" if ctx_min else "")
        ax.plot(ks, coverage * 100, color=colors[ctx], lw=2, label=lbl)
        any_data = True

    if not any_data:
        plt.close(fig); print("  [9B] No data"); return

    ax.set_xlabel("K  (maximum windows used per subject)", fontsize=11)
    ax.set_ylabel("Subjects correctly classified (%)", fontsize=11)
    ax.set_title(
        f"Coverage Curves — {task}  ({head.upper()})\n"
        f"Fraction of subjects correctly classified using ≤K windows (reps={reps})",
        fontsize=11,
    )
    ax.legend(title="Context", fontsize=8, title_fontsize=9, ncol=2,
              loc="lower right")
    ax.set_xlim(1, k_max)
    ax.set_ylim(0, 100)
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()

    stem = f"{task}_{head}_kstar_9B_coverage"
    save_figure(fig, out_dir, stem)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-subject K* analysis (§9)."
    )
    parser.add_argument("--task",    required=True)
    parser.add_argument("--head",    default="lstm")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/scratch/boshra95/psg/unified/results/phase0_v2"),
                        dest="results_dir")
    parser.add_argument("--split",   default="test", choices=["train", "val", "test"])
    parser.add_argument("--run-tag", default="", dest="run_tag")
    parser.add_argument("--contexts", nargs="+", default=None,
                        help="Restrict to these context lengths (default: all)")
    parser.add_argument("--kmax",    type=int, default=K_MAX_DEFAULT,
                        dest="k_max",
                        help=f"Maximum K to evaluate (default: {K_MAX_DEFAULT})")
    parser.add_argument("--reps",    type=int, default=REPS_DEFAULT,
                        help=f"Random draws per (subject, k) (default: {REPS_DEFAULT})")
    parser.add_argument("--plots",   nargs="+", default=["9A", "9B"])
    parser.add_argument("--repo-figures-dir", type=Path, default=None,
                        dest="repo_figures_dir",
                        help="Also mirror PNGs into this repo dir (e.g. "
                             "results/figures/phase0_v3). Default: no repo mirror.")
    args = parser.parse_args()

    configure_repo_figures(args.results_dir, args.repo_figures_dir)
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
    print(f"  K_max={args.k_max}  reps={args.reps}")

    if "9A" in args.plots:
        plot_kstar_histogram(parquets, args.task, args.head,
                             args.k_max, args.reps, out_dir)
    if "9B" in args.plots:
        plot_coverage_curves(parquets, args.task, args.head,
                             args.k_max, args.reps, out_dir)

    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
