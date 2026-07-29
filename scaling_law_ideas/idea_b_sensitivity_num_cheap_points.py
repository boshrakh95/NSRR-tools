#!/usr/bin/env python3
"""
Idea B sensitivity check — how does held-out prediction error change as the
number of cheap training points shrinks from 4 down to 2?

idea_b_context_extrapolation.py fit on the 4 cheapest contexts
(30s, 10m, 40m, 80m) and predicted the 2 held-out expensive ones
(120m, 240m). This script asks the natural follow-up: what happens if we
go cheaper still — fit on only 3 (30s, 10m, 40m) or only 2 (30s, 10m) — and
predict *all* remaining, longer contexts?

Answered without implementing until asked (see prior turn): fitting N=3
free parameters (a, b, c) to N=3 points leaves zero residual degrees of
freedom (an exact interpolation, not a validated fit), and N=2 points is
under-determined for a 3-parameter model entirely, so that case is
expected to fail outright — included specifically to show *where* the
approach breaks, not because it's expected to work.

Does not modify idea_b_context_extrapolation.py; imports its fitting
function directly (read-only reuse). Reads only analysis.csv (existing,
read-only). Writes only to scaling_law_ideas/output/ (new files only).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from idea_b_context_extrapolation import (
    WORKSPACE_ROOT, ANALYSIS_CSV, TASK_LABEL, ALL_CONTEXTS, fit_power_law, power_law,
)

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Contexts ordered cheapest-first; the first N are used for fitting.
CONTEXTS_ORDERED = ["30s", "10m", "40m", "80m", "120m", "240m"]
TRAIN_SIZES = [4, 3, 2]  # 4 = idea_b_context_extrapolation.py's original case


def main():
    df = pd.read_csv(ANALYSIS_CSV)
    sub = df[(df["head"] == "transformer") & (df["split"] == "test") & (df["k"] == "all")].copy()

    records = []
    for task in TASK_LABEL:
        g = sub[sub.task == task].set_index("context_length").loc[ALL_CONTEXTS]
        L_all = g["context_length_min"].values
        y_all = g["mean_prob_auroc"].values

        for n_train in TRAIN_SIZES:
            train_ctx = CONTEXTS_ORDERED[:n_train]
            held_out_ctx = CONTEXTS_ORDERED[n_train:]
            L_train = g.loc[train_ctx, "context_length_min"].values
            y_train = g.loc[train_ctx, "mean_prob_auroc"].values

            popt = fit_power_law(L_train, y_train)
            for ctx in held_out_ctx:
                actual = g.loc[ctx, "mean_prob_auroc"]
                Lc = g.loc[ctx, "context_length_min"]
                pred = power_law(Lc, *popt) if popt is not None else float("nan")
                err = pred - actual if popt is not None else float("nan")
                records.append({
                    "task": task, "n_train_points": n_train,
                    "train_contexts": ",".join(train_ctx),
                    "held_out_context": ctx, "context_min": Lc,
                    "actual_auroc": actual, "predicted_auroc": pred,
                    "error_auroc_points": err, "abs_error_auroc_points": abs(err) if popt is not None else float("nan"),
                    "fit_succeeded": popt is not None,
                })

    results = pd.DataFrame(records)
    results.to_csv(OUT_DIR / "idea_b_sensitivity_errors.csv", index=False)
    print(f"Saved table -> {OUT_DIR / 'idea_b_sensitivity_errors.csv'}")

    print("\n=== Mean/max abs error by n_train_points, across all tasks+held-out contexts ===")
    print(results.groupby("n_train_points")["abs_error_auroc_points"].agg(["mean", "max", "count"]).round(4))

    print("\n=== Fit failures (popt is None) ===")
    fails = results[~results["fit_succeeded"]]
    if len(fails):
        print(fails[["task", "n_train_points"]].drop_duplicates().to_string(index=False))
    else:
        print("none — curve_fit converged in every case")

    print("\n=== Per-task mean abs error by n_train_points ===")
    pivot = results.pivot_table(index="task", columns="n_train_points", values="abs_error_auroc_points", aggfunc="mean")
    print(pivot.round(4).to_string())

    # ── Plot: mean abs error vs n_train_points, one line per task + overall mean ──
    fig, ax = plt.subplots(figsize=(7, 5))
    for task in TASK_LABEL:
        t = results[results.task == task].groupby("n_train_points")["abs_error_auroc_points"].mean()
        ax.plot(t.index, t.values * 100, marker="o", ms=4, alpha=0.6, label=TASK_LABEL[task])
    overall = results.groupby("n_train_points")["abs_error_auroc_points"].mean()
    ax.plot(overall.index, overall.values * 100, marker="s", ms=8, color="black", lw=2.5, label="mean across tasks")
    ax.invert_xaxis()  # so "cheaper" (fewer points) reads left-to-right as "more extreme"
    ax.set_xticks([4, 3, 2])
    ax.set_xlabel("Number of cheap context lengths used to fit (30s,10m,40m,80m → fewer)")
    ax.set_ylabel("Mean |predicted - actual| AUROC (points)")
    ax.set_title("Idea B sensitivity: prediction error vs. how few cheap points are used")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_png = OUT_DIR / "idea_b_sensitivity_num_cheap_points.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot -> {out_png}")


if __name__ == "__main__":
    main()
