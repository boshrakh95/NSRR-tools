#!/usr/bin/env python3
"""
Idea B — Held-out context-length extrapolation test.

Question: if we had only ever run the four CHEAP context lengths
(30s, 10m, 40m, 80m), could we have predicted AUROC at the two EXPENSIVE
ones (120m, 240m) well enough to be useful? This is the direct analog of
the LLM scaling-law practice (fit cheap/small runs, forecast expensive/
large ones) applied to context length instead of model/compute size.

Method: fit a saturating power law AUROC(L) = c - a * L^(-b) (same
functional form already used in utils/panels.py's FLOPs power-law fit)
using only the 4 cheap points, per (task, head). Evaluate the fit at
L=120 and L=240 (held out, never seen by the fit) and compare to the real
analysis.csv values there. Also fits the same form on all 6 points as a
reference ("how good could an in-sample fit ever be for this curve
shape") to separate "extrapolation is hard" from "this functional form is
just a bad match for this task's curve".

Reads only final_results/phase0_v3/collected/analysis.csv (existing file,
read-only). Writes only to scaling_law_ideas/output/ (new directory).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def find_workspace():
    candidate = Path.cwd().resolve()
    for _ in range(10):
        if (candidate / "final_results").exists():
            return candidate
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return Path("/Users/boshra/NSRR-workspace").resolve()


WORKSPACE_ROOT = find_workspace()
ANALYSIS_CSV = WORKSPACE_ROOT / "final_results" / "phase0_v3" / "collected" / "analysis.csv"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHEAP_CONTEXTS = ["30s", "10m", "40m", "80m"]
HELD_OUT_CONTEXTS = ["120m", "240m"]
ALL_CONTEXTS = CHEAP_CONTEXTS + HELD_OUT_CONTEXTS

TASK_LABEL = {
    "sex_binary": "Sex", "age_class": "Age", "apnea_binary": "Apnea (AHI≥15)",
    "bmi_binary": "BMI", "sleep_efficiency_binary": "Sleep Efficiency",
    "depression_extreme_binary": "Depression", "osa_binary_apples_postqc": "OSA (APPLES)",
}


def power_law(L, a, b, c):
    return c - a * np.asarray(L, dtype=float) ** (-b)


def fit_power_law(L, y):
    """Same bounds/init as utils/panels.py's _power_law fit."""
    L = np.asarray(L, dtype=float)
    y = np.asarray(y, dtype=float)
    try:
        popt, _ = curve_fit(power_law, L, y, p0=[1.0, 0.3, y.max()],
                             bounds=([0, 0, 0], [np.inf, 5, 1.5]), maxfev=20000)
        return popt
    except Exception:
        return None


def main():
    print(f"WORKSPACE_ROOT: {WORKSPACE_ROOT}")
    df = pd.read_csv(ANALYSIS_CSV)
    sub = df[(df["head"] == "transformer") & (df["split"] == "test") & (df["k"] == "all")].copy()

    records = []
    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    axes_flat = axes.flatten()

    tasks = list(TASK_LABEL.keys())
    for i, task in enumerate(tasks):
        g = sub[sub.task == task].set_index("context_length").loc[ALL_CONTEXTS]
        L_all = g["context_length_min"].values
        y_all = g["mean_prob_auroc"].values
        L_cheap = g.loc[CHEAP_CONTEXTS, "context_length_min"].values
        y_cheap = g.loc[CHEAP_CONTEXTS, "mean_prob_auroc"].values

        popt_cheap = fit_power_law(L_cheap, y_cheap)
        popt_full = fit_power_law(L_all, y_all)

        ax = axes_flat[i]
        ax.scatter(L_cheap, y_cheap * 100, color="#3A7EBF", zorder=5, label="used for fit (≤80m)")
        ax.scatter(g.loc[HELD_OUT_CONTEXTS, "context_length_min"],
                   g.loc[HELD_OUT_CONTEXTS, "mean_prob_auroc"] * 100,
                   color="#E86A33", marker="s", zorder=5, label="held out (120m/240m)")

        Lgrid = np.logspace(np.log10(0.4), np.log10(300), 200)
        if popt_cheap is not None:
            ax.plot(Lgrid, power_law(Lgrid, *popt_cheap) * 100, color="#3A7EBF", ls="--",
                    lw=1.2, label="fit on ≤80m only")
        if popt_full is not None:
            ax.plot(Lgrid, power_law(Lgrid, *popt_full) * 100, color="gray", ls=":",
                    lw=1.0, label="fit on all 6 (reference)")

        for ctx in HELD_OUT_CONTEXTS:
            actual = g.loc[ctx, "mean_prob_auroc"]
            Lc = g.loc[ctx, "context_length_min"]
            pred = power_law(Lc, *popt_cheap) if popt_cheap is not None else float("nan")
            err = pred - actual
            records.append({
                "task": task, "held_out_context": ctx, "context_min": Lc,
                "actual_auroc": actual, "predicted_auroc": pred,
                "error_auroc_points": err, "abs_error_auroc_points": abs(err),
            })

        ax.set_xscale("log")
        ax.set_title(TASK_LABEL[task], fontsize=10)
        ax.set_xlabel("Context length (min)")
        ax.set_ylabel("AUROC (%)")
        ax.legend(fontsize=6.5, frameon=False, loc="lower right")
        ax.grid(alpha=0.25)

    axes_flat[-1].set_visible(False)
    fig.suptitle("Idea B: power-law fit on cheap contexts (≤80m) only, extrapolated to 120m/240m (held out)")
    fig.tight_layout()
    out_png = OUT_DIR / "idea_b_context_extrapolation.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {out_png}")

    results = pd.DataFrame(records)
    results.to_csv(OUT_DIR / "idea_b_extrapolation_errors.csv", index=False)
    print(f"Saved table -> {OUT_DIR / 'idea_b_extrapolation_errors.csv'}")

    print("\n=== Extrapolation error (predicted - actual, AUROC points), Transformer, test split ===")
    print(results[["task", "held_out_context", "actual_auroc", "predicted_auroc", "error_auroc_points"]]
          .round(4).to_string(index=False))

    print("\n=== Mean absolute error by held-out context, across all 7 tasks ===")
    print(results.groupby("held_out_context")["abs_error_auroc_points"].agg(["mean", "max"]).round(4))


if __name__ == "__main__":
    main()
