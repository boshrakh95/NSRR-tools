#!/usr/bin/env python3
"""
plot_channel_comparison.py — S-Fig 2: Fast vs Full Channel saturation overlay.

6-panel figure (2 rows × 3 cols): one panel per task, each showing
fast-channel (dashed) and full-channel (solid) Transformer saturation curves.
Annotates Δ at 240m on each panel. OSA panel notes APPLES-only cohort.

Data source: analysis.csv from phase0_v3 and phase0_v3_full,
Transformer head, k='all', split='test', column mean_prob_auroc.

Usage:
  python scripts/plot_channel_comparison.py
  python scripts/plot_channel_comparison.py \\
      --fast-dir results/collected/phase0_v3 \\
      --full-dir results/collected/phase0_v3_full \\
      --out-dir  /scratch/boshra95/psg/unified/results/phase0_v3_full/figures \\
      --repo-figures-dir results/figures/phase0_v3_full

Output:
  {out_dir}/channel_comparison.{png,pdf}
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

CONTEXT_TO_MIN = {
    "30s": 0.5, "10m": 10.0, "40m": 40.0,
    "80m": 80.0, "120m": 120.0, "240m": 240.0,
}

TRANSFORMER_COLOR = "#E86A33"

TASKS = [
    ("sex_binary",              "Sex (binary)"),
    ("apnea_binary",            "Apnea (binary)"),
    ("sleep_efficiency_binary", "Sleep Efficiency"),
    ("bmi_binary",              "BMI (binary)"),
    ("age_class",               "Age (3-class)"),
    ("osa_binary_apples_postqc","OSA (APPLES only)"),
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_saturation(csv_path: Path, task: str, head: str = "transformer",
                    split: str = "test") -> pd.DataFrame:
    """Load mean_prob_auroc vs context_length_min from analysis.csv at k=all."""
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if "run_tag" not in df.columns:
        df["run_tag"] = ""
    mask = (
        (df["task"] == task) &
        (df["head"] == head) &
        (df["split"] == split) &
        (df["k"].astype(str) == "all") &
        (df["run_tag"].fillna("") == "") &
        df["mean_prob_auroc"].notna()
    )
    sub = df.loc[mask, ["context_length", "context_length_min",
                        "mean_prob_auroc"]].copy()
    return sub.sort_values("context_length_min").reset_index(drop=True)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_comparison(fast_dir: Path, full_dir: Path, out_dir: Path,
                    split: str = "test", head: str = "transformer"):
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 8))
    axes_flat  = axes.flatten()

    ctx_str = {v: k for k, v in CONTEXT_TO_MIN.items()}

    for idx, (task, task_label) in enumerate(TASKS):
        ax = axes_flat[idx]

        fast_df = load_saturation(fast_dir / "analysis.csv", task, head, split)
        full_df = load_saturation(full_dir / "analysis.csv", task, head, split)

        any_data = False

        if not fast_df.empty:
            xs = fast_df["context_length_min"].values
            ys = fast_df["mean_prob_auroc"].values * 100
            ax.plot(xs, ys, color=TRANSFORMER_COLOR, ls="--", lw=2.0,
                    marker="o", markersize=6, label="Fast-channel")
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}", (x, y),
                            textcoords="offset points", xytext=(0, 6),
                            fontsize=6.5, ha="center", color=TRANSFORMER_COLOR,
                            alpha=0.7)
            any_data = True

        if not full_df.empty:
            xs2 = full_df["context_length_min"].values
            ys2 = full_df["mean_prob_auroc"].values * 100
            ax.plot(xs2, ys2, color=TRANSFORMER_COLOR, ls="-", lw=2.0,
                    marker="s", markersize=6, label="Full-channel")
            any_data = True

        # Annotate Δ at 240m
        if not fast_df.empty and not full_df.empty:
            fast_240 = fast_df[fast_df["context_length"] == "240m"]["mean_prob_auroc"]
            full_240 = full_df[full_df["context_length"] == "240m"]["mean_prob_auroc"]
            if not fast_240.empty and not full_240.empty:
                delta_pp = (float(full_240.iloc[0]) - float(fast_240.iloc[0])) * 100
                sign = "+" if delta_pp >= 0 else ""
                ax.text(0.97, 0.05, f"Δ240m: {sign}{delta_pp:.1f} pp",
                        transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=7.5, color="gray",
                        bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

        if not any_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")

        ax.set_xscale("log")
        all_xs = sorted(CONTEXT_TO_MIN.values())
        ax.set_xticks(all_xs)
        ax.set_xticklabels(
            [ctx_str.get(x, f"{x:.0f}") for x in all_xs], fontsize=8)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        ax.set_xlabel("Context length (minutes)", fontsize=9)
        ax.set_ylabel("AUROC (%)", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, which="major", alpha=0.25)

        # APPLES-only note for OSA
        if "osa" in task:
            ax.text(0.03, 0.97, "APPLES-only cohort",
                    transform=ax.transAxes, ha="left", va="top",
                    fontsize=7, color="gray", style="italic")

        # Panel label below x-axis label
        ax.text(0.5, -0.22, f"({chr(97 + idx)})",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8, fontfamily="serif")
        ax.text(0.5, -0.13, task_label, transform=ax.transAxes,
                ha="center", va="top", fontsize=8, fontfamily="serif")

    # Shared legend at top
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels,
                   loc="upper center", ncol=2, fontsize=9,
                   bbox_to_anchor=(0.5, 1.01),
                   frameon=False)

    plt.tight_layout(rect=[0, 0.04, 1, 0.98])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir, "channel_comparison")
    plt.close(fig)
    print(f"  Saved → {out_dir}/channel_comparison.{{png,pdf}}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="S-Fig 2: fast vs full channel saturation overlay."
    )
    parser.add_argument("--fast-dir", type=Path,
                        default=Path("results/collected/phase0_v3"),
                        dest="fast_dir")
    parser.add_argument("--full-dir", type=Path,
                        default=Path("results/collected/phase0_v3_full"),
                        dest="full_dir")
    parser.add_argument("--out-dir",  type=Path,
                        default=Path(
                            "/scratch/boshra95/psg/unified/results/"
                            "phase0_v3_full/figures/phase0_v3_full"),
                        dest="out_dir")
    parser.add_argument("--split",    default="test", choices=["val", "test"])
    parser.add_argument("--head",     default="transformer")
    parser.add_argument("--repo-figures-dir", type=Path, default=None,
                        dest="repo_figures_dir")
    args = parser.parse_args()

    results_dir = Path(
        "/scratch/boshra95/psg/unified/results/phase0_v3_full")
    configure_repo_figures(results_dir, args.repo_figures_dir)

    print(f"Fast-ch: {args.fast_dir}")
    print(f"Full-ch: {args.full_dir}")
    plot_comparison(args.fast_dir, args.full_dir, args.out_dir,
                    args.split, args.head)


if __name__ == "__main__":
    main()
