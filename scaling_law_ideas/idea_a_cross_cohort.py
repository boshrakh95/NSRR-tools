#!/usr/bin/env python3
"""
Idea A — Cross-cohort replication check.

Question: does the AUROC-vs-context-length "saturation" pattern seen in our
SMALLEST cohorts (APPLES, STAGES) match the pattern seen in our LARGEST
cohort (SHHS)? This is a direct, no-retraining test of whether the paper's
context-efficiency findings are a property of the task/physiology (and
therefore likely to replicate at larger scale) or an artefact of one
particular cohort's demographics, recording protocol, or size.

Method: exactly reproduces the paper's own mean-prob AUROC convention
(scripts/analyze_windows.py: evaluate_at_k, K="all"), the only difference
being an added group-by on the `dataset` column before computing AUROC, so
results are directly comparable to analysis.csv's pooled numbers (which are
recomputed here too, as a sanity check).

Reads only existing files under final_results/phase0_v3/collected/. Writes
only to scaling_law_ideas/output/ (new directory, nothing existing is
touched or overwritten).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


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
PRED_DIR = WORKSPACE_ROOT / "final_results" / "phase0_v3" / "collected" / "predictions"
OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXTS = ["30s", "10m", "40m", "80m", "120m", "240m"]
CONTEXT_TO_MIN = {"30s": 0.5, "10m": 10.0, "40m": 40.0, "80m": 80.0, "120m": 120.0, "240m": 240.0}

# Tasks with >=3 cohorts represented at test time (checked directly against
# the parquets beforehand) — sex_binary only has 2 (apples, shhs), included
# separately as a secondary/weaker check.
TASKS_MULTI_COHORT = ["apnea_binary", "sleep_efficiency_binary", "bmi_binary", "age_class"]
TASKS_TWO_COHORT = ["sex_binary"]

NUM_CLASSES = {
    "age_class": 3,
    "apnea_binary": 2,
    "sleep_efficiency_binary": 2,
    "bmi_binary": 2,
    "sex_binary": 2,
}

TASK_LABEL = {
    "sex_binary": "Sex",
    "age_class": "Age",
    "apnea_binary": "Apnea (AHI≥15)",
    "bmi_binary": "BMI",
    "sleep_efficiency_binary": "Sleep Efficiency",
}

COHORT_COLOR = {
    "apples": "#3A7EBF", "stages": "#E86A33", "mros": "#44A15E", "shhs": "#7B5EA7",
}
MIN_SUBJECTS_FOR_AUROC = 20  # skip (cohort, context) cells with too few subjects / one-class-only


def subject_mean_probs(df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    """Reproduce analyze_windows.py's K='all' mean-prob aggregation, per subject."""
    prob_cols = [f"prob_class{c}" for c in range(num_classes)]
    rows = []
    for (sid, dset), grp in df.groupby(["subject_id", "dataset"], sort=False):
        mean_prob = grp[prob_cols].values.astype(np.float64).mean(axis=0)
        rows.append({
            "subject_id": sid, "dataset": dset,
            "true_label": int(grp["true_label"].iloc[0]),
            **{f"mean_prob_class{c}": mean_prob[c] for c in range(num_classes)},
        })
    return pd.DataFrame(rows)


def auroc_from_subject_table(sub: pd.DataFrame, num_classes: int):
    if sub["true_label"].nunique() < 2 or len(sub) < MIN_SUBJECTS_FOR_AUROC:
        return float("nan")
    y = sub["true_label"].values
    if num_classes == 2:
        p = sub["mean_prob_class1"].values
        return float(roc_auc_score(y, p))
    P = sub[[f"mean_prob_class{c}" for c in range(num_classes)]].values
    try:
        return float(roc_auc_score(y, P, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def run_task(task: str):
    num_classes = NUM_CLASSES[task]
    records = []
    for ctx in CONTEXTS:
        f = PRED_DIR / f"{task}_transformer_{ctx}_test.parquet"
        if not f.exists():
            print(f"  [skip] missing {f.name}")
            continue
        df = pd.read_parquet(f)
        subj = subject_mean_probs(df, num_classes)

        # Pooled (sanity check vs analysis.csv)
        pooled_auroc = auroc_from_subject_table(subj, num_classes)
        records.append({"task": task, "context": ctx, "context_min": CONTEXT_TO_MIN[ctx],
                         "dataset": "POOLED", "n_subjects": len(subj), "auroc": pooled_auroc})

        # Per-cohort
        for dset, sub in subj.groupby("dataset"):
            auroc = auroc_from_subject_table(sub, num_classes)
            records.append({"task": task, "context": ctx, "context_min": CONTEXT_TO_MIN[ctx],
                             "dataset": dset, "n_subjects": len(sub), "auroc": auroc})
    return pd.DataFrame(records)


def main():
    print(f"WORKSPACE_ROOT: {WORKSPACE_ROOT}")
    print(f"PRED_DIR exists: {PRED_DIR.exists()}")

    all_results = []
    for task in TASKS_MULTI_COHORT + TASKS_TWO_COHORT:
        print(f"\n=== {task} ===")
        res = run_task(task)
        all_results.append(res)
        print(res.pivot_table(index="context_min", columns="dataset", values="auroc").to_string())

    full = pd.concat(all_results, ignore_index=True)
    full.to_csv(OUT_DIR / "idea_a_per_cohort_auroc.csv", index=False)
    print(f"\nSaved table -> {OUT_DIR / 'idea_a_per_cohort_auroc.csv'}")

    # ── Sanity check: pooled recomputed AUROC vs analysis.csv ──────────────
    analysis = pd.read_csv(WORKSPACE_ROOT / "final_results" / "phase0_v3" / "collected" / "analysis.csv")
    analysis = analysis[(analysis["head"] == "transformer") & (analysis["split"] == "test") & (analysis["k"] == "all")]
    print("\n=== Sanity check: recomputed POOLED AUROC vs analysis.csv mean_prob_auroc ===")
    max_diff = 0.0
    for task in TASKS_MULTI_COHORT + TASKS_TWO_COHORT:
        for ctx in CONTEXTS:
            mine = full[(full.task == task) & (full.context == ctx) & (full.dataset == "POOLED")]["auroc"]
            theirs = analysis[(analysis.task == task) & (analysis.context_length == ctx)]["mean_prob_auroc"]
            if len(mine) and len(theirs):
                diff = abs(mine.values[0] - theirs.values[0])
                max_diff = max(max_diff, diff)
                if diff > 1e-6:
                    print(f"  {task:28s} {ctx:5s}  mine={mine.values[0]:.4f}  analysis.csv={theirs.values[0]:.4f}  diff={diff:.5f}")
    print(f"Max abs diff across all (task, context): {max_diff:.6f}  "
          f"({'MATCH' if max_diff < 1e-4 else 'MISMATCH - investigate'})")

    # ── Plot: small multiples, one panel per task, one line per cohort ──────
    tasks_to_plot = TASKS_MULTI_COHORT + TASKS_TWO_COHORT
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()
    for i, task in enumerate(tasks_to_plot):
        ax = axes_flat[i]
        sub = full[(full.task == task) & (full.dataset != "POOLED")]
        pooled = full[(full.task == task) & (full.dataset == "POOLED")]
        for dset, g in sub.groupby("dataset"):
            g = g.sort_values("context_min")
            n_med = int(g["n_subjects"].median())
            ax.plot(g["context_min"], g["auroc"] * 100, marker="o", ms=4,
                    color=COHORT_COLOR.get(dset, "gray"), label=f"{dset} (n≈{n_med})")
        pooled = pooled.sort_values("context_min")
        ax.plot(pooled["context_min"], pooled["auroc"] * 100, color="black", ls="--",
                lw=1.2, label="pooled (all cohorts)")
        ax.set_xscale("log")
        ax.set_title(TASK_LABEL.get(task, task))
        ax.set_xlabel("Context length (min)")
        ax.set_ylabel("AUROC (%)")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(alpha=0.25)
    for j in range(len(tasks_to_plot), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Idea A: AUROC vs context length, stratified by cohort (Transformer, test split)")
    fig.tight_layout()
    out_png = OUT_DIR / "idea_a_cross_cohort.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {out_png}")


if __name__ == "__main__":
    main()
