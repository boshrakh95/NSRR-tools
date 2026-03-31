#!/usr/bin/env python3
"""
compute_subject_metrics.py — Phase 0, Subject-level aggregation

Reads a per-window parquet produced by infer_subject_windows.py, aggregates
predictions per subject using two strategies, then computes and saves metrics.

Aggregation strategies
──────────────────────
  mean_prob   : average softmax probabilities across all windows per subject
                → subject-level AUROC and all other metrics (best for ranking)
  majority_vote: mode of per-window hard predictions
                → subject-level accuracy / balanced_accuracy / macro_f1

Output
──────
  {parquet_dir}/subject_metrics.json   — metrics dict (same schema as metrics.json)

Usage
─────
  # From a specific parquet file:
  python scripts/compute_subject_metrics.py \\
      --parquet results/phase0/inference/apnea_binary_lstm/context_10m/test_windows.parquet

  # Batch: process all parquets under the inference dir:
  python scripts/compute_subject_metrics.py --all \\
      --inference-dir /scratch/boshra95/psg/unified/results/phase0/inference
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from scipy.stats import mode as scipy_mode

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
        cohen_kappa_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("scikit-learn not found — some metrics will be skipped.")

import torch


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_subjects(df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    """
    Group by (subject_id, dataset), aggregate probabilities and predictions.

    Returns one row per subject with columns:
        subject_id, dataset, true_label,
        mean_prob_class0 … mean_prob_classN,   ← mean-prob aggregation
        majority_pred                            ← majority vote hard prediction
    """
    prob_cols = [f"prob_class{c}" for c in range(num_classes)]
    rows = []

    for (sid, dset), grp in df.groupby(["subject_id", "dataset"], sort=False):
        true_label = int(grp["true_label"].iloc[0])   # same for all windows

        # Mean probabilities
        mean_probs = grp[prob_cols].mean().values.astype(np.float32)

        # Majority vote (most common hard prediction)
        preds = grp["pred_label"].values
        majority = int(scipy_mode(preds, keepdims=True).mode[0])

        row = {
            "subject_id": sid,
            "dataset":    dset,
            "true_label": true_label,
            "majority_pred": majority,
            "n_windows": len(grp),
        }
        for c in range(num_classes):
            row[f"mean_prob_class{c}"] = float(mean_probs[c])

        rows.append(row)

    return pd.DataFrame(rows)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(subj_df: pd.DataFrame, num_classes: int, task: str) -> dict:
    """Compute subject-level metrics from the aggregated subject DataFrame."""
    targets = subj_df["true_label"].values
    prob_cols = [f"mean_prob_class{c}" for c in range(num_classes)]
    probs = subj_df[prob_cols].values            # (N_subjects, num_classes)

    # Hard predictions from mean-prob (argmax of averaged probabilities)
    mean_prob_preds = probs.argmax(axis=1)
    majority_preds  = subj_df["majority_pred"].values

    def _base_metrics(preds):
        m = {"accuracy": float((preds == targets).mean())}
        if not HAS_SKLEARN:
            return m
        m["balanced_accuracy"] = float(balanced_accuracy_score(targets, preds))
        m["macro_f1"] = float(f1_score(targets, preds, average="macro", zero_division=0))
        for c in range(num_classes):
            mask = targets == c
            m[f"recall_class{c}"] = float((preds[mask] == c).mean()) if mask.any() else float("nan")
        return m

    # AUROC — always from mean probabilities (not hard vote)
    def _auroc():
        if not HAS_SKLEARN:
            return {}
        try:
            if num_classes == 2:
                auc = float(roc_auc_score(targets, probs[:, 1]))
            else:
                auc = float(roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
            return {"auroc": auc}
        except ValueError:
            return {"auroc": float("nan")}

    auroc_dict = _auroc()

    # Kappa — sleep staging only
    kappa_dict = {}
    if task == "sleep_staging" and HAS_SKLEARN:
        kappa_dict["cohen_kappa"] = float(cohen_kappa_score(targets, mean_prob_preds))

    mean_prob_metrics = {**_base_metrics(mean_prob_preds), **auroc_dict, **kappa_dict}
    majority_metrics  = {**_base_metrics(majority_preds),  **auroc_dict, **kappa_dict}

    return {
        "mean_prob":    mean_prob_metrics,
        "majority_vote": majority_metrics,
    }


# ── Process one parquet ───────────────────────────────────────────────────────

def process_parquet(parquet_path: Path) -> dict:
    """Read parquet, aggregate, compute metrics, save JSON. Returns metrics dict."""
    df = pd.read_parquet(parquet_path)

    # Derive num_classes from probability columns
    prob_cols = [c for c in df.columns if c.startswith("prob_class")]
    num_classes = len(prob_cols)

    # Infer task from directory name: .../apnea_binary_lstm/context_10m/test_windows.parquet
    task = parquet_path.parent.parent.name.rsplit("_", 1)[0]   # strip _headtype
    split = parquet_path.stem.replace("_windows", "")           # e.g. "test"

    print(f"  {parquet_path.parent.parent.name} / {parquet_path.parent.name} / {parquet_path.name}")
    print(f"    Windows: {len(df):,} | Subjects: {df.groupby(['subject_id','dataset']).ngroups:,} | Classes: {num_classes}")

    subj_df = aggregate_subjects(df, num_classes)

    metrics = compute_metrics(subj_df, num_classes, task)
    metrics["n_subjects"] = len(subj_df)
    metrics["n_windows_total"] = len(df)
    metrics["avg_windows_per_subject"] = float(len(df) / max(len(subj_df), 1))
    metrics["split"] = split
    metrics["task"] = task
    metrics["num_classes"] = num_classes

    # Save subject-level aggregated DataFrame alongside the parquet
    subj_parquet = parquet_path.parent / f"{split}_subjects.parquet"
    subj_df.to_parquet(subj_parquet, index=False)

    # Save metrics JSON
    out_json = parquet_path.parent / "subject_metrics.json"
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    mp = metrics["mean_prob"]
    mv = metrics["majority_vote"]
    print(f"    mean-prob  → AUROC={mp.get('auroc', float('nan'))*100:.1f}%  "
          f"BalAcc={mp.get('balanced_accuracy', float('nan'))*100:.1f}%  "
          f"MacroF1={mp.get('macro_f1', float('nan'))*100:.1f}%")
    print(f"    maj-vote   → AUROC={mv.get('auroc', float('nan'))*100:.1f}%  "
          f"BalAcc={mv.get('balanced_accuracy', float('nan'))*100:.1f}%  "
          f"MacroF1={mv.get('macro_f1', float('nan'))*100:.1f}%")
    print(f"    Saved → {out_json}")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-window parquet to subject-level metrics."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--parquet", type=Path,
                       help="Path to a specific *_windows.parquet file")
    group.add_argument("--all", action="store_true",
                       help="Process all *_windows.parquet files under --inference-dir")
    parser.add_argument("--inference-dir", type=Path,
                        default=Path("/scratch/boshra95/psg/unified/results/phase0/inference"),
                        help="Root inference directory (used with --all)")
    args = parser.parse_args()

    if args.all:
        parquets = sorted(args.inference_dir.glob("**/test_windows.parquet"))
        if not parquets:
            print(f"No test_windows.parquet files found under {args.inference_dir}")
            return
        print(f"Found {len(parquets)} parquet file(s):\n")
        for p in parquets:
            process_parquet(p)
            print()
    else:
        process_parquet(args.parquet)


if __name__ == "__main__":
    main()
