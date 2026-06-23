#!/usr/bin/env python3
"""
plot_cohort_saturation.py — §7 Cohort-Stratified Saturation

Reads per-window prediction parquets and computes AUROC separately for each
dataset (cohort), showing whether the context saturation result holds across
APPLES, SHHS, and MrOS independently.

  7A: Per-cohort saturation curves — AUROC vs context length at K=all windows,
      one line per dataset (per panel if multiple tasks).

  7B: Per-cohort N table plot — n_subjects per (dataset, context_length) shown
      as a bar chart; documents how many subjects each context uses per cohort.

Usage:
  python scripts/plot_cohort_saturation.py \\
      --task sex_binary \\
      --head lstm \\
      --results-dir /scratch/boshra95/psg/unified/results/phase0_v2 \\
      --split test \\
      --datasets apples shhs mros

Output:
  {results_dir}/figures/{task}_{head}/cohort_saturation_7A.{png,pdf}
  {results_dir}/figures/{task}_{head}/cohort_saturation_7B_n.{png,pdf}
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
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── Style ─────────────────────────────────────────────────────────────────────

CONTEXT_TO_MIN = {
    "30s": 0.5, "10m": 10.0, "40m": 40.0,
    "80m": 80.0, "120m": 120.0, "240m": 240.0,
}
CTX_ORDER = {c: i for i, c in enumerate(CONTEXT_TO_MIN)}

DATASET_STYLE = {
    "apples": {"color": "#1f77b4", "marker": "o", "ls": "-",  "label": "APPLES"},
    "shhs":   {"color": "#ff7f0e", "marker": "s", "ls": "--", "label": "SHHS"},
    "mros":   {"color": "#2ca02c", "marker": "^", "ls": ":",  "label": "MrOS"},
    "stages": {"color": "#9467bd", "marker": "D", "ls": "-.", "label": "STAGES"},
}

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


# ── Per-cohort AUROC ──────────────────────────────────────────────────────────

def compute_cohort_auroc(df: pd.DataFrame, dataset: str,
                         prob_col: str = "prob_class1") -> tuple:
    """
    Filter to one dataset, aggregate all windows per subject (K=all),
    compute AUROC and N_subjects.
    Returns (auroc, n_subjects) or (nan, 0).
    """
    if not HAS_SKLEARN or prob_col not in df.columns:
        return float("nan"), 0

    sub = df[df["dataset"].str.lower() == dataset.lower()].copy()
    if sub.empty:
        return float("nan"), 0

    subj = (
        sub.groupby(["subject_id"])
        .agg(mean_prob=(prob_col, "mean"), true_label=("true_label", "first"))
        .reset_index()
    )
    if subj["true_label"].nunique() < 2 or len(subj) < 5:
        return float("nan"), len(subj)

    try:
        auc = float(roc_auc_score(subj["true_label"], subj["mean_prob"]))
    except Exception:
        auc = float("nan")
    return auc, len(subj)


# ── Plot 7A: Per-cohort saturation curves ─────────────────────────────────────

def plot_cohort_saturation(parquets: dict, task: str, head: str,
                           datasets: list, metric: str,
                           out_dir: Path) -> None:
    contexts = sorted(parquets.keys(), key=lambda c: CTX_ORDER.get(c, 99))
    if not contexts:
        print("  [7A] No parquets found")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    any_data = False

    for ds in datasets:
        ds_style = DATASET_STYLE.get(
            ds.lower(),
            {"color": "gray", "marker": "o", "ls": "-", "label": ds.upper()},
        )
        xs, ys, ns = [], [], []
        for ctx in contexts:
            df  = parquets[ctx]
            auc, n = compute_cohort_auroc(df, ds)
            ctx_min = CONTEXT_TO_MIN.get(ctx)
            if ctx_min is None or np.isnan(auc) or n < 5:
                continue
            xs.append(ctx_min); ys.append(auc); ns.append(n)
        if not xs:
            continue

        ax.plot(xs, ys, color=ds_style["color"], marker=ds_style["marker"],
                ls=ds_style["ls"], lw=2, ms=8, label=ds_style["label"])
        # Annotate N on each point
        for x, y, n in zip(xs, ys, ns):
            ax.annotate(f"N={n}", (x, y),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=7, ha="center", color=ds_style["color"])
        any_data = True

    if not any_data:
        plt.close(fig)
        print("  [7A] No data with ≥5 subjects per dataset")
        return

    ctx_min_vals = sorted(set(CONTEXT_TO_MIN[c] for c in contexts
                              if c in CONTEXT_TO_MIN))
    ax.set_xscale("log")
    ax.set_xticks(ctx_min_vals)
    ax.set_xticklabels(
        [k for k, v in sorted(CONTEXT_TO_MIN.items(), key=lambda x: x[1])
         if v in ctx_min_vals],
        fontsize=9,
    )
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Context Length (log scale)", fontsize=11)
    ax.set_ylabel(f"Test AUROC  (K=all)", fontsize=11)
    ax.set_title(
        f"Cohort-Stratified Saturation — {task}  ({head.upper()})\n"
        "Does the context saturation result hold across each dataset independently?",
        fontsize=11,
    )
    ax.legend(title="Dataset", fontsize=10)
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()

    stem = f"{task}_{head}_cohort_saturation_7A"
    save_figure(fig, out_dir, stem)
    plt.close(fig)


# ── Plot 7B: Per-cohort N (subject count) ─────────────────────────────────────

def plot_cohort_n(parquets: dict, task: str, head: str,
                  datasets: list, out_dir: Path) -> None:
    contexts = sorted(parquets.keys(), key=lambda c: CTX_ORDER.get(c, 99))
    if not contexts:
        return

    x      = np.arange(len(contexts))
    width  = 0.8 / max(len(datasets), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(contexts) * 1.4), 5))

    for i, ds in enumerate(datasets):
        ds_style = DATASET_STYLE.get(
            ds.lower(), {"color": "gray", "label": ds.upper()})
        ns = []
        for ctx in contexts:
            df  = parquets[ctx]
            sub = df[df["dataset"].str.lower() == ds.lower()]
            n   = sub.groupby(["subject_id"]).ngroups
            ns.append(n)
        offset = (i - len(datasets) / 2 + 0.5) * width
        ax.bar(x + offset, ns, width * 0.9,
               color=ds_style["color"], label=ds_style["label"],
               edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(contexts, fontsize=10)
    ax.set_xlabel("Context Length", fontsize=11)
    ax.set_ylabel("Number of subjects in test set", fontsize=11)
    ax.set_title(
        f"Test-Set Subject Count per Dataset — {task}  ({head.upper()})\n"
        "Subject count may drop at longer contexts (fewer qualifying windows)",
        fontsize=11,
    )
    ax.legend(title="Dataset")
    fig.tight_layout()

    stem = f"{task}_{head}_cohort_saturation_7B_n"
    save_figure(fig, out_dir, stem)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot cohort-stratified saturation curves (§7)."
    )
    parser.add_argument("--task",    required=True)
    parser.add_argument("--head",    default="lstm")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/scratch/boshra95/psg/unified/results/phase0_v2"),
                        dest="results_dir")
    parser.add_argument("--split",   default="test", choices=["train", "val", "test"])
    parser.add_argument("--run-tag", default="", dest="run_tag")
    parser.add_argument("--datasets", nargs="+",
                        default=["apples", "shhs", "mros"],
                        help="Datasets to include (default: apples shhs mros)")
    parser.add_argument("--metric",  default="auroc",
                        choices=["auroc"],
                        help="Metric (currently only auroc supported, default: auroc)")
    parser.add_argument("--plots",   nargs="+", default=["7A", "7B"])
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
    out_dir = args.results_dir / "figures" / f"{args.task}_{args.head}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading parquets for {args.task}/{args.head} [{args.split}] ...")
    parquets = load_parquets(args.results_dir, args.task, args.head,
                              args.split, args.run_tag)
    if not parquets:
        print("ERROR: No parquets found. Run inference first.")
        return
    print(f"  Found contexts: {sorted(parquets.keys())}")

    if "7A" in args.plots:
        plot_cohort_saturation(parquets, args.task, args.head,
                               args.datasets, args.metric, out_dir)
    if "7B" in args.plots:
        plot_cohort_n(parquets, args.task, args.head, args.datasets, out_dir)

    print(f"\nAll outputs → {out_dir}")


if __name__ == "__main__":
    main()
