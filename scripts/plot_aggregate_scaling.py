#!/usr/bin/env python3
"""
plot_aggregate_scaling.py — Aggregate Context-Length Scaling Analysis

Answers: is there a general law for how AUROC improves with context length,
and does it hold across tasks? Shows mean ± std across tasks, per head.

  (a) ΔAUROC from 30s baseline — one line per head, ±1 std shading across tasks.
      Individual task curves shown as faint lines. Reveals average gain and
      inter-task variability.

  (b) Normalised gain (0 = 30s performance, 1 = 240m performance) — same
      structure. Shows the *shape* of the gain curve independent of how
      context-sensitive each task is.

  (c) Log-linear slope b per head — bar chart with ±1 std across tasks.
      b = AUROC pp gained per doubling of context length (from OLS fit
      ΔAUROC ~ a + b × log₂(context_min) per task).

Data source: analysis.csv (mean_prob_auroc, k='all', split='test').

Usage:
  python scripts/plot_aggregate_scaling.py \\
      --collected-dir results/collected/phase0_v3 \\
      --results-dir /scratch/boshra95/psg/unified/results/phase0_v3

  # Exclude non-monotonic task from the average:
  python scripts/plot_aggregate_scaling.py \\
      --exclude-tasks depression_extreme_binary

Output:
  {results_dir}/figures/aggregate/aggregate_scaling.{png,pdf}
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

CONTEXT_TO_MIN = {
    "30s": 0.5, "10m": 10.0, "40m": 40.0,
    "80m": 80.0, "120m": 120.0, "240m": 240.0,
}
STD_CTX = [0.5, 10.0, 40.0, 80.0, 120.0, 240.0]
CTX_LABELS = ["30s", "10m", "40m", "80m", "120m", "240m"]

TASKS_DEFAULT = [
    "sex_binary", "apnea_binary", "sleep_efficiency_binary",
    "bmi_binary", "age_class", "depression_extreme_binary",
    "osa_binary_apples_postqc",
]

TASK_LABELS = {
    "sex_binary":                "Sex",
    "apnea_binary":              "Apnea",
    "sleep_efficiency_binary":   "Sleep Eff.",
    "bmi_binary":                "BMI",
    "age_class":                 "Age",
    "depression_extreme_binary": "Depression",
    "osa_binary_apples_postqc":  "OSA",
}

HEAD_STYLE = {
    "lstm":        {"color": "#3A7EBF", "marker": "o", "ls": "-",  "label": "LSTM"},
    "transformer": {"color": "#E86A33", "marker": "s", "ls": "--", "label": "Transformer"},
    "mean_pool":   {"color": "#44A15E", "marker": "^", "ls": ":",  "label": "MeanPool"},
}

plt.rcParams.update({
    "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "serif", "font.size": 9,
})


# ── Data loading ──────────────────────────────────────────────────────────────

def load_analysis(collected_dir: Path, tasks: list, heads: list,
                  split: str = "test") -> pd.DataFrame:
    p = collected_dir / "analysis.csv"
    if not p.exists():
        raise FileNotFoundError(f"analysis.csv not found at {p}")

    df = pd.read_csv(p)

    if "context_length_min" not in df.columns and "context_length" in df.columns:
        df["context_length_min"] = df["context_length"].map(
            lambda s: CONTEXT_TO_MIN.get(str(s).strip(), None)
        )

    df = df[
        (df["split"] == split) &
        (df["k"].astype(str) == "all") &
        df["mean_prob_auroc"].notna() &
        df["context_length_min"].notna()
    ]
    if tasks:
        df = df[df["task"].isin(tasks)]
    if heads:
        df = df[df["head"].isin(heads)]

    return df


# ── Per-task curve computation ────────────────────────────────────────────────

def build_curves(df: pd.DataFrame, tasks: list, heads: list) -> dict:
    """
    Returns curve_data[head][task] = {
        'ctx': array of context_length_min values (sorted),
        'auroc': array of mean_prob_auroc,
        'delta': AUROC - AUROC[0],
        'norm': delta / (AUROC[-1] - AUROC[0]),  NaN if total gain ≤ 0
        'slope': b from OLS fit of delta ~ b*log2(ctx) + a,
    }
    """
    result = {h: {} for h in heads}

    for head in heads:
        for task in tasks:
            sub = df[(df["head"] == head) & (df["task"] == task)].copy()
            sub = sub.sort_values("context_length_min").reset_index(drop=True)
            if len(sub) < 2:
                continue

            ctx   = sub["context_length_min"].values
            auroc = sub["mean_prob_auroc"].values
            delta = auroc - auroc[0]

            total_gain = auroc[-1] - auroc[0]
            norm = delta / total_gain if total_gain > 0.005 else np.full_like(delta, np.nan)

            # OLS slope: delta ~ a + b*log2(ctx)
            valid = (ctx > 0) & ~np.isnan(delta)
            if valid.sum() >= 3:
                coeffs = np.polyfit(np.log2(ctx[valid]), delta[valid], deg=1)
                slope = float(coeffs[0])
            else:
                slope = np.nan

            result[head][task] = {
                "ctx": ctx, "auroc": auroc, "delta": delta,
                "norm": norm, "slope": slope,
            }

    return result


def aggregate_curves(curve_data: dict, heads: list) -> dict:
    """
    Per head: for each standard context point, average delta and norm
    across all tasks that have data at that point.
    Returns agg[head] = {ctx, mean_delta, std_delta, mean_norm, std_norm, n_tasks}.
    """
    agg = {}
    for head in heads:
        task_data = curve_data[head]
        delta_at = {c: [] for c in STD_CTX}
        norm_at  = {c: [] for c in STD_CTX}

        for d in task_data.values():
            for c in STD_CTX:
                idx = np.where(np.abs(d["ctx"] - c) < 0.05)[0]
                if idx.size:
                    delta_at[c].append(d["delta"][idx[0]])
                    v = d["norm"][idx[0]]
                    if not np.isnan(v):
                        norm_at[c].append(v)

        valid_ctx = [c for c in STD_CTX if len(delta_at[c]) >= 1]
        if not valid_ctx:
            continue

        agg[head] = {
            "ctx":        np.array(valid_ctx),
            "mean_delta": np.array([np.mean(delta_at[c]) for c in valid_ctx]),
            "std_delta":  np.array([np.std(delta_at[c], ddof=1) if len(delta_at[c]) > 1
                                    else 0.0 for c in valid_ctx]),
            "mean_norm":  np.array([np.mean(norm_at[c]) if norm_at[c] else np.nan
                                    for c in valid_ctx]),
            "std_norm":   np.array([np.std(norm_at[c], ddof=1) if len(norm_at[c]) > 1
                                    else np.nan for c in valid_ctx]),
            "n_tasks":    [len(delta_at[c]) for c in valid_ctx],
        }

    return agg


def compute_slopes(curve_data: dict, heads: list) -> dict:
    """Return slopes[head] = {'per_task': {task: slope}, 'mean': float, 'std': float}."""
    slopes = {}
    for head in heads:
        per_task = {
            task: d["slope"]
            for task, d in curve_data[head].items()
            if not np.isnan(d["slope"])
        }
        if not per_task:
            continue
        vals = list(per_task.values())
        slopes[head] = {
            "per_task": per_task,
            "mean": np.mean(vals),
            "std":  np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
        }
    return slopes


# ── Plotting ──────────────────────────────────────────────────────────────────

def _log_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(STD_CTX)
    ax.set_xticklabels(CTX_LABELS, fontsize=9)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.axhline(0, color="gray", ls=":", lw=0.8, alpha=0.5)


def plot_aggregate(agg: dict, curve_data: dict, slopes: dict,
                   heads: list, tasks: list, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax_a, ax_b, ax_c = axes

    for head in heads:
        if head not in agg:
            continue
        style = HEAD_STYLE.get(head, {"color": "gray", "marker": "o",
                                      "ls": "-", "label": head})
        d = agg[head]
        ctx = d["ctx"]

        # ── Individual task lines (faint) ────────────────────────────────────
        for task, td in curve_data[head].items():
            t_label = TASK_LABELS.get(task, task)
            ax_a.plot(td["ctx"], td["delta"] * 100,
                      color=style["color"], lw=0.7, alpha=0.20, ls="-")
            if not np.all(np.isnan(td["norm"])):
                ax_b.plot(td["ctx"], td["norm"] * 100,
                          color=style["color"], lw=0.7, alpha=0.20, ls="-")

        # ── Mean ± 1 std ─────────────────────────────────────────────────────
        ax_a.fill_between(
            ctx,
            (d["mean_delta"] - d["std_delta"]) * 100,
            (d["mean_delta"] + d["std_delta"]) * 100,
            color=style["color"], alpha=0.15,
        )
        ax_a.plot(ctx, d["mean_delta"] * 100,
                  color=style["color"], ls=style["ls"],
                  marker=style["marker"], lw=2, ms=6, label=style["label"])

        valid_n = ~np.isnan(d["mean_norm"])
        if valid_n.any():
            mn = d["mean_norm"][valid_n]
            sn = np.where(np.isnan(d["std_norm"][valid_n]), 0.0,
                          d["std_norm"][valid_n])
            ax_b.fill_between(ctx[valid_n], (mn - sn) * 100, (mn + sn) * 100,
                               color=style["color"], alpha=0.15)
            ax_b.plot(ctx[valid_n], mn * 100,
                      color=style["color"], ls=style["ls"],
                      marker=style["marker"], lw=2, ms=6, label=style["label"])

    # ── Panel (a) formatting ─────────────────────────────────────────────────
    _log_axis(ax_a)
    ax_a.set_xlabel("Context length", fontsize=10)
    ax_a.set_ylabel("ΔAUROC from 30s baseline (pp)", fontsize=10)
    ax_a.legend(fontsize=9)

    # ── Panel (b) formatting ─────────────────────────────────────────────────
    _log_axis(ax_b)
    ax_b.axhline(100, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax_b.set_xlabel("Context length", fontsize=10)
    ax_b.set_ylabel("Normalised gain (%)\n(0 = 30s, 100 = 240m)", fontsize=10)
    ax_b.set_ylim(-20, 130)
    ax_b.legend(fontsize=9)

    # ── Panel (c): slope bar chart ───────────────────────────────────────────
    ordered_heads = [h for h in heads if h in slopes]
    x_pos  = np.arange(len(ordered_heads))
    colors = [HEAD_STYLE.get(h, {"color": "gray"})["color"] for h in ordered_heads]
    labels = [HEAD_STYLE.get(h, {"label": h})["label"] for h in ordered_heads]
    means  = [slopes[h]["mean"] * 100 for h in ordered_heads]
    stds   = [slopes[h]["std"]  * 100 for h in ordered_heads]

    # Individual task slopes as dots
    for xi, head in zip(x_pos, ordered_heads):
        for slope_val in slopes[head]["per_task"].values():
            ax_c.scatter(xi, slope_val * 100, color=colors[xi],
                         alpha=0.35, s=20, zorder=3)

    bars = ax_c.bar(x_pos, means, color=colors, edgecolor="white",
                    width=0.45, alpha=0.85, zorder=2)
    ax_c.errorbar(x_pos, means, yerr=stds,
                  fmt="none", color="black", capsize=5, lw=1.5, zorder=4)

    for bar, val in zip(bars, means):
        ax_c.text(bar.get_x() + bar.get_width() / 2,
                  val + max(stds) * 0.05 + 0.05,
                  f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(labels, fontsize=9)
    ax_c.set_ylabel("ΔAUROC per log₂ doubling (pp)", fontsize=10)
    ax_c.axhline(0, color="gray", ls=":", lw=0.8)

    # ── Panel labels ─────────────────────────────────────────────────────────
    for i, ax in enumerate(axes):
        ax.text(0.5, -0.18, f"({chr(97+i)})", transform=ax.transAxes,
                ha="center", va="top", fontsize=8, fontfamily="serif")

    n_tasks_used = len([t for t in tasks
                        if any(t in curve_data[h] for h in heads)])
    fig.suptitle(
        f"Aggregate context-length scaling  ·  N={n_tasks_used} tasks  "
        f"·  faint lines = individual tasks",
        fontsize=8, color="gray", y=1.01,
    )

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir, "aggregate_scaling")
    plt.close(fig)

    # Print slope summary
    print("\nLog-linear slopes (ΔAUROC pp per log₂ doubling of context):")
    for head in ordered_heads:
        s = slopes[head]
        per_task_str = ", ".join(
            f"{TASK_LABELS.get(t, t)}={v*100:.2f}"
            for t, v in s["per_task"].items()
        )
        print(f"  {HEAD_STYLE.get(head,{}).get('label', head):12s} "
              f"mean={s['mean']*100:.2f} pp  std={s['std']*100:.2f}  "
              f"[{per_task_str}]")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate context-length scaling analysis."
    )
    parser.add_argument("--collected-dir", type=Path,
                        default=Path("results/collected"),
                        dest="collected_dir",
                        help="Directory containing analysis.csv")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/scratch/boshra95/psg/unified/results/phase0_v3"),
                        dest="results_dir")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Tasks to include (default: all 7 retained tasks)")
    parser.add_argument("--exclude-tasks", nargs="+", default=None,
                        dest="exclude_tasks",
                        help="Tasks to exclude from the average (e.g. depression_extreme_binary)")
    parser.add_argument("--heads", nargs="+",
                        default=["lstm", "transformer", "mean_pool"],
                        help="Heads to include")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--repo-figures-dir", type=Path, default=None,
                        dest="repo_figures_dir")
    args = parser.parse_args()

    tasks = args.tasks or TASKS_DEFAULT
    if args.exclude_tasks:
        tasks = [t for t in tasks if t not in args.exclude_tasks]

    configure_repo_figures(args.results_dir, args.repo_figures_dir)
    out_dir = args.results_dir / "figures" / "aggregate"

    print(f"Loading analysis.csv from {args.collected_dir} ...")
    try:
        df = load_analysis(args.collected_dir, tasks, args.heads, args.split)
    except FileNotFoundError as e:
        print(f"ERROR: {e}"); return

    print(f"  Tasks: {tasks}")
    print(f"  Heads: {args.heads}")
    print(f"  Rows:  {len(df)}")

    curve_data = build_curves(df, tasks, args.heads)
    agg        = aggregate_curves(curve_data, args.heads)
    slopes     = compute_slopes(curve_data, args.heads)

    if not agg:
        print("ERROR: No data to plot."); return

    plot_aggregate(agg, curve_data, slopes, args.heads, tasks, out_dir)
    print(f"\nOutput → {out_dir}/aggregate_scaling.{{png,pdf}}")


if __name__ == "__main__":
    main()
