#!/usr/bin/env python3
"""
Idea E — Joint context x compute scaling surface.

Question: instead of treating context length (L) and inference-time
aggregation (K) as two separate axes (the existing iso-compute heatmap is
a lookup table over both), can a single smooth joint functional form
AUROC(L, K) fit the whole (L, K) -> AUROC surface at once? Directly
analogous to Montgomery et al. 2025 ("Predicting Task Performance with
Context-aware Scaling Laws", arXiv:2510.14919), who fit one joint function
of training compute and context length for LLMs.

Functional form (Chinchilla-style additive two-term deficit, mirroring
the paper's own established power-law convention rather than inventing a
new one):

    AUROC(L, K) = c - a * L^(-p) - b * K^(-q)

c = asymptotic ceiling; the two power-law terms are the "loss" from
having finite context and finite aggregation, assumed additively
separable. If this simple separable form fits well, it's a single
interpretable equation replacing the heatmap. If it fits poorly with a
systematic residual pattern, that is itself informative: it would say
context and aggregation are NOT simply separable/substitutable for that
task -- directly quantifying the paper's own "context-irreplaceable"
(H2 exception) story rather than just asserting it qualitatively.

Validated via leave-one-context-length-out: fit on 5 of the 6 context
lengths (all their K values), predict the 6th (held out) context length's
entire K-curve, compare to actual. This is a genuine held-out test (the
held-out context length's data never enters that fold's fit), unlike
Idea A.

Prototyped on the 2 tasks recommended in the original proposal (sex,
sleep efficiency -- highest existing signal) plus BMI as a third
(context-insensitive) reference case, since a flat task is a useful
contrast: if the joint law fits BMI trivially well (small p) but fits
sleep efficiency poorly, that itself is a clean illustration of the
substitutability story.

Reads only final_results/phase0_v3/collected/analysis.csv (existing file,
read-only). Writes only to scaling_law_ideas/output/ (new files only).
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

TASKS = ["sex_binary", "sleep_efficiency_binary", "bmi_binary"]
TASK_LABEL = {"sex_binary": "Sex", "sleep_efficiency_binary": "Sleep Efficiency", "bmi_binary": "BMI"}
CONTEXTS = ["30s", "10m", "40m", "80m", "120m", "240m"]

BOUNDS = ([0, 0, 0, 0, 0], [np.inf, np.inf, 1.0, 5, 5])
P0 = [0.3, 0.3, 0.95, 0.3, 0.3]


def joint_form(LK, a, b, c, p, q):
    L, K = LK
    return c - a * np.asarray(L, dtype=float) ** (-p) - b * np.asarray(K, dtype=float) ** (-q)


def fit_joint(L, K, y):
    try:
        popt, _ = curve_fit(joint_form, (L, K), y, p0=P0, bounds=BOUNDS, maxfev=40000)
        return popt
    except Exception as e:
        print(f"    fit failed: {e}")
        return None


def load_task_grid(df, task):
    sub = df[(df["head"] == "transformer") & (df["split"] == "test")
             & (df["task"] == task) & (df["k"] != "all")].copy()
    sub["k_num"] = sub["k"].astype(int)
    return sub[["context_length", "context_length_min", "k_num", "mean_prob_auroc"]].rename(
        columns={"context_length_min": "L", "k_num": "K", "mean_prob_auroc": "y"})


def main():
    print(f"WORKSPACE_ROOT: {WORKSPACE_ROOT}")
    df = pd.read_csv(ANALYSIS_CSV)

    all_records = []
    fig, axes = plt.subplots(len(TASKS), 2, figsize=(12, 4 * len(TASKS)))

    for ti, task in enumerate(TASKS):
        print(f"\n=== {task} ===")
        grid = load_task_grid(df, task)
        L, K, y = grid["L"].values, grid["K"].values, grid["y"].values
        print(f"  n points: {len(grid)}  (context lengths: {sorted(grid['context_length'].unique(), key=lambda c: CONTEXTS.index(c))})")

        # Full in-sample fit (reference: best this functional form can ever do)
        popt_full = fit_joint(L, K, y)
        if popt_full is not None:
            pred_full = joint_form((L, K), *popt_full)
            resid_full = pred_full - y
            print(f"  FULL fit params: a={popt_full[0]:.4f} b={popt_full[1]:.4f} c={popt_full[2]:.4f} "
                  f"p={popt_full[3]:.4f} q={popt_full[4]:.4f}")
            print(f"  FULL in-sample: mean|resid|={np.abs(resid_full).mean():.4f}  max|resid|={np.abs(resid_full).max():.4f}")

        # Leave-one-context-length-out
        loo_records = []
        for held_out_ctx in grid["context_length"].unique():
            train = grid[grid["context_length"] != held_out_ctx]
            test = grid[grid["context_length"] == held_out_ctx]
            popt = fit_joint(train["L"].values, train["K"].values, train["y"].values)
            if popt is None:
                continue
            pred = joint_form((test["L"].values, test["K"].values), *popt)
            err = pred - test["y"].values
            for k, actual, p, e in zip(test["K"].values, test["y"].values, pred, err):
                loo_records.append({"task": task, "held_out_context": held_out_ctx,
                                     "K": k, "actual": actual, "predicted": p, "error": e})
        loo_df = pd.DataFrame(loo_records)
        all_records.append(loo_df)

        mae = loo_df["error"].abs().mean()
        print(f"  Leave-one-context-out: mean|error|={mae:.4f}  max|error|={loo_df['error'].abs().max():.4f}")
        print(loo_df.groupby("held_out_context")["error"].apply(lambda s: s.abs().mean()).round(4)
              .reindex([c for c in CONTEXTS if c in loo_df["held_out_context"].unique()]))

        # ── Plot: (left) actual vs fitted surface as AUROC-vs-K curves per L;
        #    (right) leave-one-context-out predicted vs actual per held-out L ──
        ax_l, ax_r = axes[ti]
        cmap = plt.cm.viridis(np.linspace(0, 1, len(CONTEXTS)))
        ctx_color = dict(zip(CONTEXTS, cmap))
        for ctx in grid["context_length"].unique():
            g = grid[grid["context_length"] == ctx].sort_values("K")
            ax_l.plot(g["K"], g["y"] * 100, "o", color=ctx_color[ctx], ms=4, label=ctx)
            if popt_full is not None:
                Kgrid = np.logspace(0, np.log10(max(g["K"].max(), 2)), 50)
                Lval = g["L"].iloc[0]
                ax_l.plot(Kgrid, joint_form((np.full_like(Kgrid, Lval), Kgrid), *popt_full) * 100,
                          "-", color=ctx_color[ctx], lw=1, alpha=0.6)
        ax_l.set_xscale("log")
        ax_l.set_xlabel("K")
        ax_l.set_ylabel("AUROC (%)")
        ax_l.set_title(f"{TASK_LABEL[task]}: data (dots) vs full joint fit (lines)")
        ax_l.legend(fontsize=6, frameon=False, ncol=2)
        ax_l.grid(alpha=0.25)

        ax_r.scatter(loo_df["actual"] * 100, loo_df["predicted"] * 100, c="#3A7EBF", s=20)
        lims = [loo_df[["actual", "predicted"]].min().min() * 100 - 1,
                loo_df[["actual", "predicted"]].max().max() * 100 + 1]
        ax_r.plot(lims, lims, "k--", lw=1)
        ax_r.set_xlim(lims); ax_r.set_ylim(lims)
        ax_r.set_xlabel("Actual AUROC (%)")
        ax_r.set_ylabel("Predicted AUROC (%), leave-one-context-out")
        ax_r.set_title(f"{TASK_LABEL[task]}: held-out prediction (MAE={mae*100:.2f} pts)")
        ax_r.grid(alpha=0.25)

    fig.tight_layout()
    out_png = OUT_DIR / "idea_e_joint_scaling_surface.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot -> {out_png}")

    full_results = pd.concat(all_records, ignore_index=True)
    full_results.to_csv(OUT_DIR / "idea_e_loo_errors.csv", index=False)
    print(f"Saved table -> {OUT_DIR / 'idea_e_loo_errors.csv'}")

    print("\n=== Summary: mean |leave-one-context-out error| by task ===")
    print(full_results.groupby("task")["error"].apply(lambda s: s.abs().mean()).round(4))


if __name__ == "__main__":
    main()
