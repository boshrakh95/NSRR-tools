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
  every row:  is_best_epoch, is_overfit_epoch, train_loss, val_loss,
              train_bal_acc, val_bal_acc, val_auroc,
              num_classes, n_train, n_val, n_test,
              n_epochs_run, n_overfit_epochs, training_time_min,
              batch_size, seq_len, steps_per_epoch,
              windows_per_subject_train, n_trainable_params,
              input_dim, hidden_dim, save_snapshots, snapshot_interval
  best only:  {train,val,test}_{accuracy,balanced_accuracy,macro_f1,auroc}
              {train,val,test}_recall_class{0..4}
  → filter is_best_epoch=True for paper tables
  → all rows (including is_overfit_epoch=True) for U-shape / scaling-law plots
  → seq_len × steps_per_epoch × FLOPs_per_token gives total compute per epoch

analysis.csv
  key:        task, head, context_length, k, split
  every row:  context_length_min, total_compute_min (= ctx_min × k; NaN for k='all'),
              n_subjects, n_segments,
              {seg,mean_prob,majority}_{accuracy,balanced_accuracy,macro_f1,auroc}
              mean_prob_{auroc,bal_acc}_ci_{lo,hi}  (NaN if bootstrap disabled)
  → use for every post-training plot (K-saturation, iso-compute, Pareto fronts)
  → k='all' rows = inference over every available window (max coverage)
  → total_compute_min is the iso-compute axis
  → CI columns populated only when analyze_windows was run with --bootstrap N > 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Default paths (overridden by CLI --results-dir / --out-dir / --repo-out) ──

_DEFAULT_RESULTS = Path("/scratch/boshra95/psg/unified/results/phase0_v3")
_REPO_COLLECTED  = Path(__file__).parent.parent / "results" / "collected"

# These module-level names are set by main() after arg parsing.
RESULTS_DIR: Path
INFERENCE_DIR: Path
SCRATCH_OUT: Path
REPO_OUT: Path
FORCE: bool

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

def collect_training(results_dir: Path, out_paths: list[Path],
                     exp_ids: set[str] | None = None) -> int:
    existing = load_csv(out_paths[0]) if out_paths[0].exists() else load_csv(out_paths[1])
    done = set() if FORCE else done_keys(existing, TRAIN_KEY)
    new_rows: list[dict] = []

    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name in SKIP_DIRS:
            continue
        if exp_ids and exp_dir.name not in exp_ids:
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

            # Identify best epoch — exclude overfit-phase rows from idxmax()
            monitor_col_map = {
                "val_auroc":             "val_auroc",
                "val_balanced_accuracy": "val_bal_acc",
            }
            monitor_col = monitor_col_map.get(
                metrics.get("early_stopping_monitor", "val_auroc"), "val_auroc"
            )
            if "is_overfit_epoch" in curves.columns:
                normal_curves = curves[~curves["is_overfit_epoch"].fillna(False)]
            else:
                normal_curves = curves
            if monitor_col in normal_curves.columns and not normal_curves.empty:
                best_epoch = int(normal_curves.loc[normal_curves[monitor_col].idxmax(), "epoch"])
            else:
                best_epoch = int(normal_curves["epoch"].iloc[-1]) if not normal_curves.empty else int(curves["epoch"].iloc[-1])

            for _, row in curves.iterrows():
                epoch = int(row["epoch"])
                if (task, head, ctx, str(epoch)) in done:
                    continue

                is_best      = epoch == best_epoch
                is_overfit   = bool(row.get("is_overfit_epoch", False))
                r: dict = {
                    "task":              task,
                    "head":              head,
                    "context_length":    ctx,
                    "epoch":             epoch,
                    # per-epoch curves
                    "is_best_epoch":     is_best,
                    "is_overfit_epoch":  is_overfit,
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
                    "n_overfit_epochs":  metrics.get("n_overfit_epochs"),
                    "training_time_min": metrics.get("training_time_min"),
                    # compute / model-size metadata (for scaling-law analysis)
                    "batch_size":                metrics.get("batch_size"),
                    "seq_len":                   metrics.get("seq_len"),
                    "steps_per_epoch":           metrics.get("steps_per_epoch"),
                    "windows_per_subject_train": metrics.get("windows_per_subject_train"),
                    "n_trainable_params":        metrics.get("n_trainable_params"),
                    "input_dim":                 metrics.get("input_dim"),
                    "hidden_dim":                metrics.get("hidden_dim"),
                    "save_snapshots":            metrics.get("save_snapshots"),
                    "snapshot_interval":         metrics.get("snapshot_interval"),
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

def collect_analysis(inference_dir: Path, out_paths: list[Path],
                     exp_ids: set[str] | None = None) -> int:
    existing = load_csv(out_paths[0]) if out_paths[0].exists() else load_csv(out_paths[1])
    done = set() if FORCE else done_keys(existing, ANALYSIS_KEY)
    new_rows: list[dict] = []

    for exp_dir in sorted(inference_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_ids and exp_dir.name not in exp_ids:
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
                    # bootstrap 95% CIs (NaN when bootstrap disabled or N<2 subjects)
                    "mean_prob_auroc_ci_lo":     row.get("mean_prob_auroc_ci_lo"),
                    "mean_prob_auroc_ci_hi":     row.get("mean_prob_auroc_ci_hi"),
                    "mean_prob_bal_acc_ci_lo":   row.get("mean_prob_bal_acc_ci_lo"),
                    "mean_prob_bal_acc_ci_hi":   row.get("mean_prob_bal_acc_ci_hi"),
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

def collect_predictions(inference_dir: Path, scratch_out: Path,
                        exp_ids: set[str] | None = None) -> int:
    pred_dir = scratch_out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for exp_dir in sorted(inference_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_ids and exp_dir.name not in exp_ids:
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
    global RESULTS_DIR, INFERENCE_DIR, SCRATCH_OUT, REPO_OUT, FORCE

    parser = argparse.ArgumentParser(
        description="Collect phase0 training, analysis, and prediction results into flat files."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=_DEFAULT_RESULTS,
        help=f"Root results directory to scan (default: {_DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help=(
            "Scratch output directory for collected/ files. "
            "Defaults to <results-dir>/collected."
        ),
    )
    parser.add_argument(
        "--repo-out", type=Path, default=None,
        help=(
            "Repo output directory for training.csv and analysis.csv. "
            "Defaults to results/collected/<results-dir-name>/ so that "
            "fast-channel (phase0_v3) and full-channel (phase0_v3_full) "
            "never overwrite each other."
        ),
    )
    parser.add_argument(
        "--force", action="store_true",
        help=(
            "Re-collect all rows even if the key already exists. "
            "Use after re-running analyze (e.g. with --bootstrap) to pick up "
            "updated CI columns and AUROC values."
        ),
    )
    parser.add_argument(
        "--exp-ids", nargs="+", default=None,
        help=(
            "Optional filter: only collect these experiment folder names "
            "(e.g. sleep_efficiency_binary_transformer bmi_binary_lstm). "
            "If omitted, all experiments are collected."
        ),
    )
    parser.add_argument(
        "--bootstrap", type=int, default=None,
        help=(
            "Ignored — bootstrap CIs are generated by analyze_windows.py "
            "(--bootstrap N), not by this script. Pass --bootstrap to "
            "gen_commands.py analyze ... to regenerate them."
        ),
    )
    args = parser.parse_args()

    if args.bootstrap is not None:
        print(
            "NOTE: --bootstrap is not used by collect_results_v2.py. "
            "Bootstrap CIs are generated by analyze_windows.py.\n"
            "To regenerate: python scripts/gen_commands.py analyze <exp_id> "
            "--bootstrap <N> | bash\n"
        )

    RESULTS_DIR   = args.results_dir
    INFERENCE_DIR = RESULTS_DIR / "inference"
    SCRATCH_OUT   = args.out_dir if args.out_dir else RESULTS_DIR / "collected"
    # Default repo-out: results/collected/<results-dir-name>/ so channels stay separate.
    REPO_OUT      = args.repo_out if args.repo_out else _REPO_COLLECTED / RESULTS_DIR.name
    FORCE         = args.force

    exp_ids = set(args.exp_ids) if args.exp_ids else None

    REPO_OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH_OUT.mkdir(parents=True, exist_ok=True)

    print(f"Scanning:    {RESULTS_DIR}")
    print(f"Repo out:    {REPO_OUT}")
    print(f"Scratch out: {SCRATCH_OUT}")
    if exp_ids:
        print(f"Filtering:   {sorted(exp_ids)}")
    print()

    train_out    = [REPO_OUT / "training.csv",  SCRATCH_OUT / "training.csv"]
    analysis_out = [REPO_OUT / "analysis.csv",  SCRATCH_OUT / "analysis.csv"]

    print("Collecting training results...")
    n = collect_training(RESULTS_DIR, train_out, exp_ids)
    print(f"  → {n} new rows" if n else "  → nothing new")

    print("Collecting window analysis results...")
    n = collect_analysis(INFERENCE_DIR, analysis_out, exp_ids)
    print(f"  → {n} new rows" if n else "  → nothing new")

    print("Collecting per-window predictions (scratch only)...")
    n = collect_predictions(INFERENCE_DIR, SCRATCH_OUT, exp_ids)
    print(f"  → {n} new parquet files" if n else "  → nothing new")

    print("\nTo sync across clusters:")
    print("  git add results/collected/ && git commit -m 'collect results' && git push")


if __name__ == "__main__":
    main()
