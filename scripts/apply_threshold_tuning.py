#!/usr/bin/env python3
"""
apply_threshold_tuning.py — Post-hoc decision threshold optimisation for binary tasks.

PURPOSE
───────
For class-imbalanced binary tasks (e.g. osa_binary, bmi_binary, cvd_binary), the default
decision threshold of 0.5 is often miscalibrated: the model's score distribution is not
centred at 0.5, so one class is systematically over-predicted.

This script:
  1. Aggregates window-level probabilities to subject-level scores (mean-prob, K=all).
  2. Sweeps thresholds on the VAL split to find t_opt = argmax balanced_accuracy.
  3. Applies t_opt to the TEST split and records both original (t=0.5) and tuned metrics.
  4. Writes results to {inference_dir}/{exp_id}/threshold_tuning.csv, adding NEW columns
     alongside the originals — existing results are NEVER overwritten.

WHY: AUROC is unaffected (threshold-free). Balanced accuracy at t=0.5 understates
performance on imbalanced tasks. Selecting t_opt on held-out VAL and reporting on TEST
is standard practice (equivalent to Youden's Index on the ROC curve).

PREREQUISITES
─────────────
  - test_windows.parquet already exists (produced by infer_subject_windows.py --split test)
  - val_windows.parquet must also exist:
      python scripts/gen_commands.py infer <exp_id> --split val | bash
    (re-uses trained model; no GPU needed if --cpu flag added)
  - Only for binary tasks (num_classes == 2). Skips multiclass and seq2seq.

OUTPUT
──────
  {inference_dir}/{exp_id}/threshold_tuning.csv

  Columns (one row per context length):
    context_length, n_subjects, n_windows_test,
    auroc,                                        ← threshold-free, unchanged
    orig_balanced_accuracy,                       ← K=all aggregation at t=0.5
    orig_recall_class0, orig_recall_class1,
    orig_accuracy, orig_macro_f1,
    optimal_threshold,                            ← selected on val
    val_n_subjects, val_balanced_accuracy_at_opt, ← transparency: val performance
    tuned_balanced_accuracy,                      ← K=all at t_opt on test
    tuned_recall_class0, tuned_recall_class1,
    tuned_accuracy, tuned_macro_f1,
    balanced_accuracy_gain,                       ← tuned - orig
    recall_class0_gain, recall_class1_gain

  NOTE: orig_* is recomputed from the K=all parquet using mean-prob aggregation.
  This differs slightly from summary.csv which records K=5 training-eval metrics.
  Both are correct; they measure performance at different K values.

Usage:
    python scripts/apply_threshold_tuning.py \\
        --config configs/phase0_v3_config.yaml \\
        --task bmi_binary --head lstm

    # All contexts in one pass (default); restrict with --contexts:
    python scripts/apply_threshold_tuning.py ... --contexts 10m 40m 80m

    # Run on a specific experiment with run_tag:
    python scripts/apply_threshold_tuning.py ... --task bmi_binary --head lstm --run-tag mf25

    # Dry-run: print threshold table without writing CSV:
    python scripts/apply_threshold_tuning.py ... --dry-run
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        recall_score,
        roc_auc_score,
    )
    HAS_SKLEARN = True
except ImportError:
    print("ERROR: sklearn not found. Install: pip install scikit-learn", file=sys.stderr)
    sys.exit(1)

CONTEXT_ORDER = ["30s", "10m", "40m", "80m", "120m", "240m"]
N_THRESHOLD_STEPS = 200


# ── Helpers ───────────────────────────────────────────────────────────────────

def aggregate_subjects(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate window-level parquet to subject-level mean-prob score (K=all)."""
    prob1 = df.groupby(["subject_id", "dataset"]).agg(
        true_label=("true_label", "first"),
        score=("prob_class1", "mean"),
    ).reset_index()
    return prob1


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, t: float) -> dict:
    preds = (scores > t).astype(int)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "accuracy":          float((preds == labels).mean()),
        "recall_class0":     float(recall_score(labels, preds, pos_label=0,
                                                average="binary", zero_division=0)),
        "recall_class1":     float(recall_score(labels, preds, pos_label=1,
                                                average="binary", zero_division=0)),
        "macro_f1":          float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


def find_optimal_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return (t_opt, val_balanced_accuracy_at_opt)."""
    thresholds = np.linspace(0.01, 0.99, N_THRESHOLD_STEPS)
    bal_accs   = [
        balanced_accuracy_score(labels, (scores > t).astype(int))
        for t in thresholds
    ]
    best_idx = int(np.argmax(bal_accs))
    return float(thresholds[best_idx]), float(bal_accs[best_idx])


def compute_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return float("nan")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc decision threshold optimisation for binary tasks."
    )
    parser.add_argument("--config",    required=True)
    parser.add_argument("--task",      required=True,
                        help="e.g. bmi_binary, osa_binary_apples_postqc")
    parser.add_argument("--head",      required=True,
                        help="e.g. lstm | transformer | mean_pool")
    parser.add_argument("--run-tag",   default="", dest="run_tag")
    parser.add_argument("--contexts",  default=None, nargs="+",
                        help="Restrict to these context lengths (default: all found).")
    parser.add_argument("--dry-run",   action="store_true", dest="dry_run",
                        help="Print table but do not write CSV.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["logging"]["results_dir"])
    exp_id      = f"{args.task}_{args.head}" + (f"_{args.run_tag}" if args.run_tag else "")
    inf_dir     = results_dir / "inference" / exp_id

    if not inf_dir.exists():
        print(f"ERROR: Inference directory not found: {inf_dir}")
        print("  Run inference first: python scripts/gen_commands.py infer ... | bash")
        sys.exit(1)

    # ── Detect num_classes from a test parquet ────────────────────────────────
    sample_pq = next(inf_dir.glob("context_*/test_windows.parquet"), None)
    if sample_pq is None:
        print(f"ERROR: No test_windows.parquet found under {inf_dir}.")
        sys.exit(1)
    sample_df  = pd.read_parquet(sample_pq, columns=["prob_class0", "prob_class1"])
    num_classes = len([c for c in sample_df.columns if c.startswith("prob_class")])
    if num_classes != 2:
        print(f"ERROR: {exp_id} has {num_classes} classes — threshold tuning only "
              "applies to binary (num_classes=2) tasks.")
        sys.exit(1)

    # ── Find available contexts ───────────────────────────────────────────────
    available = sorted(
        [p.parent.name.replace("context_", "")
         for p in inf_dir.glob("context_*/test_windows.parquet")],
        key=lambda c: CONTEXT_ORDER.index(c) if c in CONTEXT_ORDER else 99,
    )
    if args.contexts:
        available = [c for c in available if c in args.contexts]
    if not available:
        print(f"ERROR: No test parquets found under {inf_dir}.")
        sys.exit(1)

    print(f"\nThreshold optimisation: {exp_id}")
    print(f"Contexts: {available}")
    print()
    print(f"{'Ctx':<6}  {'Orig BalAcc':>11}  {'t_opt':>6}  {'Tuned BalAcc':>12}  "
          f"{'Gain':>6}  {'R0 t0.5':>7}  {'R0 opt':>6}  {'R1 t0.5':>7}  {'R1 opt':>6}")
    print("-" * 85)

    rows = []
    skipped = []

    for ctx in available:
        test_pq = inf_dir / f"context_{ctx}" / "test_windows.parquet"
        val_pq  = inf_dir / f"context_{ctx}" / "val_windows.parquet"

        if not val_pq.exists():
            print(f"  {ctx:<4}  [SKIP] val_windows.parquet missing — "
                  f"run: gen_commands.py infer {exp_id} --split val | bash")
            skipped.append(ctx)
            continue

        test_df = pd.read_parquet(test_pq)
        val_df  = pd.read_parquet(val_pq)

        test_agg = aggregate_subjects(test_df)
        val_agg  = aggregate_subjects(val_df)

        test_labels  = test_agg["true_label"].values
        test_scores  = test_agg["score"].values
        val_labels   = val_agg["true_label"].values
        val_scores   = val_agg["score"].values

        # Threshold selection on val
        t_opt, val_bal_acc = find_optimal_threshold(val_labels, val_scores)

        # Metrics on test at t=0.5 (original) and t_opt (tuned)
        orig  = metrics_at_threshold(test_labels, test_scores, 0.5)
        tuned = metrics_at_threshold(test_labels, test_scores, t_opt)
        auroc = compute_auroc(test_labels, test_scores)

        gain_bal  = tuned["balanced_accuracy"] - orig["balanced_accuracy"]
        gain_r0   = tuned["recall_class0"]     - orig["recall_class0"]
        gain_r1   = tuned["recall_class1"]     - orig["recall_class1"]

        print(f"  {ctx:<4}  {orig['balanced_accuracy']:>11.4f}  {t_opt:>6.3f}  "
              f"{tuned['balanced_accuracy']:>12.4f}  {gain_bal:>+6.4f}  "
              f"{orig['recall_class0']:>7.4f}  {tuned['recall_class0']:>6.4f}  "
              f"{orig['recall_class1']:>7.4f}  {tuned['recall_class1']:>6.4f}")

        rows.append({
            "context_length":             ctx,
            "n_subjects":                 int(test_agg.shape[0]),
            "n_windows_test":             int(len(test_df)),
            "auroc":                      round(auroc, 6),
            # Original at t=0.5
            "orig_balanced_accuracy":     round(orig["balanced_accuracy"], 6),
            "orig_recall_class0":         round(orig["recall_class0"], 6),
            "orig_recall_class1":         round(orig["recall_class1"], 6),
            "orig_accuracy":              round(orig["accuracy"], 6),
            "orig_macro_f1":              round(orig["macro_f1"], 6),
            # Threshold selection info
            "optimal_threshold":          round(t_opt, 4),
            "val_n_subjects":             int(val_agg.shape[0]),
            "val_balanced_accuracy_at_opt": round(val_bal_acc, 6),
            # Tuned at t_opt
            "tuned_balanced_accuracy":    round(tuned["balanced_accuracy"], 6),
            "tuned_recall_class0":        round(tuned["recall_class0"], 6),
            "tuned_recall_class1":        round(tuned["recall_class1"], 6),
            "tuned_accuracy":             round(tuned["accuracy"], 6),
            "tuned_macro_f1":             round(tuned["macro_f1"], 6),
            # Gains
            "balanced_accuracy_gain":     round(gain_bal, 6),
            "recall_class0_gain":         round(gain_r0, 6),
            "recall_class1_gain":         round(gain_r1, 6),
        })

    if not rows:
        print("\nNo results to save (all contexts skipped — val parquets missing).")
        print("Run val inference first, then re-run this script.")
        sys.exit(0)

    if skipped:
        print(f"\n[WARNING] Skipped {len(skipped)} context(s) — val parquet missing: {skipped}")

    out_csv = inf_dir / "threshold_tuning.csv"

    if args.dry_run:
        print(f"\n[DRY-RUN] Would write {len(rows)} rows to: {out_csv}")
    else:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\nSaved → {out_csv}")
        print(f"  {len(rows)} context(s) processed.")
        if skipped:
            print(f"  {len(skipped)} context(s) skipped (re-run after generating val parquets).")

    # Summary
    if rows:
        df_out = pd.DataFrame(rows)
        avg_gain = df_out["balanced_accuracy_gain"].mean()
        max_gain = df_out["balanced_accuracy_gain"].max()
        print(f"\nAverage BalAcc gain: {avg_gain:+.4f}  |  Max gain: {max_gain:+.4f}")
        print("AUROC unchanged (threshold-free metric).")


if __name__ == "__main__":
    main()
