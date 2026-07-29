#!/usr/bin/env python3
"""
plot_modality_bar.py — Fig 4: Modality Contribution bar chart.

5-panel horizontal grouped bar chart showing ΔAUROC for each ablation
condition relative to the fast-channel baseline, one panel per task.
Vertical reference lines at fast-ch baseline (solid) and full-ch (dashed).

Data sources:
  - results/collected/phase0_v3_abl/analysis.csv  (ablation conditions)
  - results/collected/phase0_v3/analysis.csv       (fast-ch baseline)
  - results/collected/phase0_v3_full/analysis.csv  (full-ch reference)

Usage:
  python scripts/plot_modality_bar.py
  python scripts/plot_modality_bar.py \\
      --abl-dir  results/collected/phase0_v3_abl \\
      --fast-dir results/collected/phase0_v3 \\
      --full-dir results/collected/phase0_v3_full \\
      --out-dir  /scratch/boshra95/psg/unified/results/phase0_v3_abl/figures \\
      --repo-figures-dir results/figures/phase0_v3_abl

Output:
  {out_dir}/modality_ablation_bar.{png,pdf}
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from repo_sync import configure_repo_figures, save_figure


# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "serif",
    "font.size": 9,
})

# Five ablation conditions: run_tag → (label, color)
CONDITIONS = [
    ("abl_no_bas",   "No BAS",    "#E86A33"),
    ("abl_no_resp",  "No RESP",   "#3A7EBF"),
    ("abl_no_ekg",   "No EKG",    "#44A15E"),
    ("abl_cardio",   "Cardio only","#C94040"),
    ("abl_bas_only", "BAS only",  "#7B5EA7"),
]

# Five tasks in left-to-right order, with their ablation context length
TASKS = [
    ("sex_binary",              "Sex",            "120m"),
    ("apnea_binary",            "Apnea",          "120m"),
    ("sleep_efficiency_binary", "Sleep Eff.",     "120m"),
    ("age_class",               "Age",            "120m"),
    ("bmi_binary",              "BMI",            "40m"),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_auroc(csv_path: Path, task: str, head: str, split: str,
               context: str, run_tag: str = "") -> float | None:
    """Read mean_prob_auroc from an analysis.csv for a specific row."""
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "run_tag" not in df.columns:
        df["run_tag"] = ""
    mask = (
        (df["task"] == task) &
        (df["head"] == head) &
        (df["split"] == split) &
        (df["k"].astype(str) == "all") &
        (df["context_length"].astype(str) == context) &
        (df["run_tag"].fillna("") == run_tag)
    )
    sub = df.loc[mask, "mean_prob_auroc"].dropna()
    return float(sub.iloc[0]) if not sub.empty else None


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_bar(abl_dir: Path, fast_dir: Path, full_dir: Path,
             out_dir: Path, split: str = "test"):
    n_tasks = len(TASKS)
    n_conds = len(CONDITIONS)

    fig, axes = plt.subplots(1, n_tasks, figsize=(14, 3.5), sharey=False)

    for col_idx, (task, task_label, ctx) in enumerate(TASKS):
        ax = axes[col_idx]

        # Fast-ch baseline (LSTM, run_tag="")
        fast_val = load_auroc(fast_dir / "analysis.csv", task, "lstm", split, ctx, "")
        # Full-ch reference (LSTM, run_tag="")
        full_val = load_auroc(full_dir / "analysis.csv", task, "lstm", split, ctx, "")

        if fast_val is None:
            ax.text(0.5, 0.5, "no baseline", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")
            ax.set_title(task_label, fontsize=9)
            continue

        # Collect deltas for each condition
        y_labels = []
        deltas    = []
        colors    = []
        for run_tag, cond_label, color in CONDITIONS:
            abl_val = load_auroc(
                abl_dir / "analysis.csv", task, "lstm", split, ctx, run_tag)
            if abl_val is None:
                delta = float("nan")
            else:
                delta = abl_val - fast_val
            y_labels.append(cond_label)
            deltas.append(delta)
            colors.append(color)

        y_pos = np.arange(n_conds)

        # Horizontal bars
        for i, (delta, color) in enumerate(zip(deltas, colors)):
            if not np.isnan(delta):
                ax.barh(i, delta, color=color, height=0.55, edgecolor="white",
                        linewidth=0.5)
                sign = "+" if delta >= 0 else ""
                ax.text(delta + (0.001 if delta >= 0 else -0.001),
                        i, f"{sign}{delta:.3f}",
                        va="center", ha="left" if delta >= 0 else "right",
                        fontsize=7, color=color)

        # Baseline reference line (x=0 = fast-ch baseline)
        ax.axvline(0, color="black", lw=1.0, zorder=3)

        # Full-channel reference line
        if full_val is not None:
            full_delta = full_val - fast_val
            ax.axvline(full_delta, color="gray", lw=1.0, ls="--", zorder=2)
            ax.text(full_delta, n_conds - 0.5, "full-ch",
                    fontsize=6, color="gray", ha="center", va="bottom",
                    rotation=90)

        ax.set_yticks(y_pos)
        if col_idx == 0:
            ax.set_yticklabels(y_labels, fontsize=9)
        else:
            ax.set_yticklabels([""] * n_conds)

        ax.set_xlabel("ΔAUROC from baseline", fontsize=9)
        ax.set_ylim(-0.6, n_conds - 0.4)

        # Symmetric x-axis with some padding
        all_deltas = [d for d in deltas if not np.isnan(d)]
        if full_val is not None:
            all_deltas.append(full_val - fast_val)
        if all_deltas:
            xmax = max(0.04, max(abs(d) for d in all_deltas) + 0.015)
            ax.set_xlim(-xmax, xmax)

        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)

        # Panel label below x-axis
        ax.text(0.5, -0.28, f"({chr(97 + col_idx)})",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8, fontfamily="serif")
        ax.text(0.5, -0.18, task_label, transform=ax.transAxes,
                ha="center", va="top", fontsize=8, fontfamily="serif")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir, "modality_ablation_bar")
    plt.close(fig)
    print(f"  Saved → {out_dir}/modality_ablation_bar.{{png,pdf}}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot Fig 4: modality ablation ΔAUROC bar chart."
    )
    parser.add_argument("--abl-dir",  type=Path,
                        default=Path("results/collected/phase0_v3_abl"),
                        dest="abl_dir",
                        help="Collected dir for ablation experiments")
    parser.add_argument("--fast-dir", type=Path,
                        default=Path("results/collected/phase0_v3"),
                        dest="fast_dir",
                        help="Collected dir for fast-channel baseline")
    parser.add_argument("--full-dir", type=Path,
                        default=Path("results/collected/phase0_v3_full"),
                        dest="full_dir",
                        help="Collected dir for full-channel reference")
    parser.add_argument("--out-dir",  type=Path,
                        default=Path(
                            "/scratch/boshra95/psg/unified/results/"
                            "phase0_v3_abl/figures/phase0_v3_abl"),
                        dest="out_dir")
    parser.add_argument("--split",    default="test", choices=["val", "test"])
    parser.add_argument("--repo-figures-dir", type=Path, default=None,
                        dest="repo_figures_dir")
    args = parser.parse_args()

    results_dir = Path(
        "/scratch/boshra95/psg/unified/results/phase0_v3_abl")
    configure_repo_figures(results_dir, args.repo_figures_dir)

    print(f"Ablation data: {args.abl_dir}")
    print(f"Fast baseline: {args.fast_dir}")
    print(f"Full reference: {args.full_dir}")
    plot_bar(args.abl_dir, args.fast_dir, args.full_dir, args.out_dir, args.split)


if __name__ == "__main__":
    main()
