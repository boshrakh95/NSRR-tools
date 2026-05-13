#!/usr/bin/env python3
"""
Collect all phase0_v2 training, analysis, and prediction results into flat files.

Run from the NSRR-tools root on either cluster:
    python scripts/collect_results_v2.py

Each run scans the local results directory and appends only new rows —
rows whose key is already present are skipped. Run on both clusters
independently; the CSVs accumulate results from whichever cluster ran the jobs.

Outputs written to BOTH the repo and scratch
--------------------------------------------
  results/collected/training.csv  — one row per (task, head, context, epoch)
  results/collected/analysis.csv  — one row per (task, head, context, k, split)

  Commit results/collected/ to git to sync across clusters:
    git add results/collected/ && git commit -m 'collect results' && git push

Output written to scratch only (too large for git)
---------------------------------------------------
  <scratch>/collected/predictions/{task}_{head}_{context}_{split}.parquet
    — one row per (task, head, context, split, subject, window)
    — load all at once with: pd.read_parquet('collected/predictions/')

Column reference
----------------
training.csv
  key:        task, head, context_length, epoch
  every row:  is_best_epoch, train_loss, val_loss, train_bal_acc,
              val_bal_acc, val_auroc, num_classes, n_train, n_val, n_test,
              n_epochs_run, training_time_min
  best only:  {train,val,test}_{accuracy,balanced_accuracy,macro_f1,auroc}
              {train,val,test}_recall_class{0..4}
  → filter is_best_epoch=True for paper tables; all rows for learning curves

analysis.csv
  key:        task, head, context_length, k, split
  every row:  context_length_min, total_compute_min (= ctx_min × k; NaN for k='all'),
              n_subjects, n_segments,
              {seg,mean_prob,majority}_{accuracy,balanced_accuracy,macro_f1,auroc}
  → use for every post-training plot (K-saturation, iso-compute, Pareto fronts)
  → k='all' rows = inference over every available window (max coverage)
  → total_compute_min is the iso-compute axis
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

RESULTS_DIR   = Path("/scratch/boshra95/psg/unified/results/phase0_v2")
INFERENCE_DIR = RESULTS_DIR / "inference"
SCRATCH_OUT   = RESULTS_DIR / "collected"
REPO_OUT      = Path(__file__).parent.parent / "results" / "collected"

# ── Constants ─────────────────────────────────────────────────────────────────

KNOWN_HEADS = ["mean_pool", "transformer", "lstm"]  # longest suffix checked first

CTX_MINUTES = {
    "30s": 0.5, "10m": 10.0, "40m": 40.0,
    "80m": 80.0, "120m": 120.0, "240m": 240.0,
}
CTX_ORDER = {c: i for i, c in enumerate(CTX_MINUTES)}

MAX_CLASSES = 5   # handles binary(2) through sleep_staging(5)

SKIP_DIRS = {"inference", "figures", "collected"}

TRAIN_KEY    = ("task", "head", "context_length", "epoch")
ANALYSIS_KEY = ("task", "head", "context_length", "k", "split")

# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_exp_dir(name: str) -> tuple[str | None, str | None]:
    """'sleep_efficiency_binary_mean_pool' → ('sleep_efficiency_binary', 'mean_pool')."""
    for head in KNOWN_HEADS:
        if name.endswith(f"_{head}"):
            return name[: -(len(head) + 1)], head
    return None, None


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def done_keys(df: pd.DataFrame, key_cols: tuple) -> set:
    if df.empty or not all(c in df.columns for c in key_cols):
        return set()
    return set(zip(*[df[c].astype(str) for c in key_cols]))


def write_csv(df: pd.DataFrame, paths: list[Path]) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)


def ctx_sort_key(df: pd.DataFrame) -> pd.Series:
    return df["context_length"].map(CTX_ORDER).fillna(99)

# ── Training collector ────────────────────────────────────────────────────────

def collect_training(results_dir: Path, out_paths: list[Path]) -> int:
    existing = load_csv(out_paths[0]) if out_paths[0].exists() else load_csv(out_paths[1])
    done = done_keys(existing, TRAIN_KEY)
    new_rows: list[dict] = []

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name in SKIP_DIRS:
            continue
        task, head = parse_exp_dir(exp_dir.name)
        if task is None:
            continue

        for ctx_dir in sorted(exp_dir.glob("context_*")):
            ctx          = ctx_dir.name.removeprefix("context_")
            curves_path  = ctx_dir / "training_curves.csv"
            metrics_path = ctx_dir / "metrics.json"

            if not curves_path.exists():
                continue

            curves  = pd.read_csv(curves_path)
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

            # Identify best epoch from the early-stopping monitor column
            monitor_col_map = {
                "val_auroc":             "val_auroc",
                "val_balanced_accuracy": "val_bal_acc",
            }
            monitor_col = monitor_col_map.get(
                metrics.get("early_stopping_monitor", "val_auroc"), "val_auroc"
            )
            if monitor_col in curves.columns:
                best_epoch = int(curves.loc[curves[monitor_col].idxmax(), "epoch"])
            else:
                best_epoch = int(curves["epoch"].iloc[-1])

            for _, row in curves.iterrows():
                epoch = int(row["epoch"])
                if (task, head, ctx, str(epoch)) in done:
                    continue

                is_best = epoch == best_epoch
                r: dict = {
                    "task":              task,
                    "head":              head,
                    "context_length":    ctx,
                    "epoch":             epoch,
                    # per-epoch curves
                    "is_best_epoch":     is_best,
                    "train_loss":        row.get("train_loss"),
                    "val_loss":          row.get("val_loss"),
                    "train_bal_acc":     row.get("train_bal_acc"),
                    "val_bal_acc":       row.get("val_bal_acc"),
                    "val_auroc":         row.get("val_auroc"),
                    # metadata repeated on every row so any subset is self-contained
                    "num_classes":       metrics.get("num_classes"),
                    "n_train":           metrics.get("n_train"),
                    "n_val":             metrics.get("n_val"),
                    "n_test":            metrics.get("n_test"),
                    "n_epochs_run":      metrics.get("n_epochs_run"),
                    "training_time_min": metrics.get("training_time_min"),
                }

                # Detailed split metrics only exist in metrics.json (best epoch)
                if is_best:
                    for split in ("train", "val", "test"):
                        sd = metrics.get(split, {})
                        r[f"{split}_accuracy"]          = sd.get("accuracy")
                        r[f"{split}_balanced_accuracy"] = sd.get("balanced_accuracy")
                        r[f"{split}_macro_f1"]          = sd.get("macro_f1")
                        r[f"{split}_auroc"]             = sd.get("auroc")
                        for c in range(MAX_CLASSES):
                            r[f"{split}_recall_class{c}"] = sd.get(f"recall_class{c}")

                new_rows.append(r)

    if not new_rows:
        return 0

    combined = pd.concat(
        [existing, pd.DataFrame(new_rows)], ignore_index=True
    ) if not existing.empty else pd.DataFrame(new_rows)

    combined["_s"] = ctx_sort_key(combined)
    combined = (combined
                .sort_values(["task", "head", "_s", "epoch"])
                .drop(columns=["_s"])
                .reset_index(drop=True))
    write_csv(combined, out_paths)
    return len(new_rows)


# ── Analysis collector ────────────────────────────────────────────────────────

def collect_analysis(inference_dir: Path, out_paths: list[Path]) -> int:
    existing = load_csv(out_paths[0]) if out_paths[0].exists() else load_csv(out_paths[1])
    done = done_keys(existing, ANALYSIS_KEY)
    new_rows: list[dict] = []

    for exp_dir in sorted(inference_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        task, head = parse_exp_dir(exp_dir.name)
        if task is None:
            continue

        for wa_path in sorted(exp_dir.glob("window_analysis_*.csv")):
            split = wa_path.stem.removeprefix("window_analysis_")
            wa    = pd.read_csv(wa_path)

            for _, row in wa.iterrows():
                k_val = str(row["k"])
                ctx   = str(row["context_length"])
                if (task, head, ctx, k_val, split) in done:
                    continue

                ctx_min = CTX_MINUTES.get(ctx)
                try:
                    total_compute: float | None = ctx_min * int(k_val) if ctx_min else None
                except (ValueError, TypeError):
                    total_compute = None  # k = 'all'

                new_rows.append({
                    "task":               task,
                    "head":               head,
                    "context_length":     ctx,
                    "k":                  k_val,
                    "split":              split,
                    "context_length_min": ctx_min,
                    "total_compute_min":  total_compute,
                    "n_subjects":         row.get("n_subjects"),
                    "n_segments":         row.get("n_segments"),
                    # segment-level
                    "seg_accuracy":              row.get("seg_accuracy"),
                    "seg_balanced_accuracy":     row.get("seg_balanced_accuracy"),
                    "seg_macro_f1":              row.get("seg_macro_f1"),
                    "seg_auroc":                 row.get("seg_auroc"),
                    # mean-prob aggregation
                    "mean_prob_accuracy":         row.get("mean_prob_accuracy"),
                    "mean_prob_balanced_accuracy":row.get("mean_prob_balanced_accuracy"),
                    "mean_prob_macro_f1":         row.get("mean_prob_macro_f1"),
                    "mean_prob_auroc":            row.get("mean_prob_auroc"),
                    # majority-vote aggregation
                    "majority_accuracy":          row.get("majority_accuracy"),
                    "majority_balanced_accuracy": row.get("majority_balanced_accuracy"),
                    "majority_macro_f1":          row.get("majority_macro_f1"),
                    "majority_auroc":             row.get("majority_auroc"),
                })

    if not new_rows:
        return 0

    combined = pd.concat(
        [existing, pd.DataFrame(new_rows)], ignore_index=True
    ) if not existing.empty else pd.DataFrame(new_rows)

    combined["_s"] = ctx_sort_key(combined)
    combined = (combined
                .sort_values(["task", "head", "_s", "k", "split"])
                .drop(columns=["_s"])
                .reset_index(drop=True))
    write_csv(combined, out_paths)
    return len(new_rows)


# ── Predictions collector ─────────────────────────────────────────────────────

def collect_predictions(inference_dir: Path, scratch_out: Path) -> int:
    pred_dir = scratch_out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for exp_dir in sorted(inference_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        task, head = parse_exp_dir(exp_dir.name)
        if task is None:
            continue

        for ctx_dir in sorted(exp_dir.glob("context_*")):
            ctx = ctx_dir.name.removeprefix("context_")

            for pq_path in sorted(ctx_dir.glob("*_windows.parquet")):
                split    = pq_path.stem.removesuffix("_windows")
                out_path = pred_dir / f"{task}_{head}_{ctx}_{split}.parquet"
                if out_path.exists():
                    continue

                df = pd.read_parquet(pq_path)
                df.insert(0, "task",           task)
                df.insert(1, "head",           head)
                df.insert(2, "context_length", ctx)
                df.insert(3, "split",          split)

                # Pad to MAX_CLASSES so all tasks share one schema
                for c in range(MAX_CLASSES):
                    col = f"prob_class{c}"
                    if col not in df.columns:
                        df[col] = np.nan

                df.to_parquet(out_path, index=False)
                print(f"    predictions/{out_path.name}: {len(df):,} rows")
                count += 1

    return count


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Scanning:    {RESULTS_DIR}")
    print(f"Repo out:    {REPO_OUT}")
    print(f"Scratch out: {SCRATCH_OUT}\n")

    train_out    = [REPO_OUT / "training.csv",  SCRATCH_OUT / "training.csv"]
    analysis_out = [REPO_OUT / "analysis.csv",  SCRATCH_OUT / "analysis.csv"]

    print("Collecting training results...")
    n = collect_training(RESULTS_DIR, train_out)
    print(f"  → {n} new rows" if n else "  → nothing new")

    print("Collecting window analysis results...")
    n = collect_analysis(INFERENCE_DIR, analysis_out)
    print(f"  → {n} new rows" if n else "  → nothing new")

    print("Collecting per-window predictions (scratch only)...")
    n = collect_predictions(INFERENCE_DIR, SCRATCH_OUT)
    print(f"  → {n} new parquet files" if n else "  → nothing new")

    print("\nTo sync across clusters:")
    print("  git add results/collected/ && git commit -m 'collect results' && git push")


if __name__ == "__main__":
    main()
