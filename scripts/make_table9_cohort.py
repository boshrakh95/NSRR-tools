#!/usr/bin/env python3
"""
make_table9_cohort.py — Table 9: Per-cohort AUROC breakdown.

For a specific experiment (task × head), reads the test_windows.parquet files
and computes per-dataset AUROC at the saturation context L*.

Reads from: <results-dir>/inference/<task>_<head>/context_<L>/test_windows.parquet
Writes to:  results/tables/table9_cohort_<exp_id>.{csv,md,tex}

Usage:
  python scripts/make_table9_cohort.py sex_binary_lstm
  python scripts/make_table9_cohort.py sex_binary_lstm --context 40m
  python scripts/make_table9_cohort.py bmi_binary_lstm --datasets apples shhs mros
  python scripts/make_table9_cohort.py sex_binary_lstm --latex
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent))
from table_utils import (
    TASK_DISPLAY, load_analysis, compute_lstar,
    fmt_auroc, df_to_markdown, latex_table, save_outputs,
    sort_contexts,
)

_DEFAULT_RESULTS   = Path("/scratch/boshra95/psg/unified/results/phase0_v3")
_DEFAULT_COLLECTED = Path(__file__).parent.parent / "results" / "collected" / "phase0_v3"
_DEFAULT_OUT       = Path(__file__).parent.parent / "results" / "tables"


def load_parquet(results_dir: Path, task: str, head: str, context: str,
                 split: str = "test") -> pd.DataFrame | None:
    p = results_dir / "inference" / f"{task}_{head}" / f"context_{context}" / f"{split}_windows.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def auroc_for_dataset(df: pd.DataFrame, dataset: str | None = None) -> tuple[float | None, int]:
    if dataset is not None:
        df = df[df["dataset"] == dataset]
    if df.empty:
        return None, 0
    # aggregate windows per subject (mean prob)
    agg = (df.groupby(["subject_id", "dataset"])
             .agg(prob=("prob_class1", "mean"), label=("true_label", "first"))
             .reset_index())
    if agg["label"].nunique() < 2:
        return None, len(agg)
    return roc_auc_score(agg["label"], agg["prob"]), len(agg)


def build(results_dir: Path, collected_dir: Path, task: str, head: str,
          context: str | None, datasets: list, split: str) -> pd.DataFrame:
    # Determine context to use
    if context is None:
        df_analysis = load_analysis(collected_dir, split=split)
        sub_a = df_analysis[(df_analysis["task"] == task) & (df_analysis["head"] == head)]
        context = compute_lstar(sub_a) or "40m"
        print(f"  Using L* = {context} for {task}/{head}")

    df_parquet = load_parquet(results_dir, task, head, context, split)
    if df_parquet is None:
        raise FileNotFoundError(
            f"Parquet not found: {results_dir}/inference/{task}_{head}/context_{context}/{split}_windows.parquet"
        )

    available_datasets = sorted(df_parquet["dataset"].unique())
    if not datasets:
        datasets = available_datasets

    display = TASK_DISPLAY.get(task, task)
    overall_auroc, overall_n = auroc_for_dataset(df_parquet)
    rows = [{"Task": display, "Head": head, "Context": context,
             "Dataset": "Overall", "N": overall_n,
             "AUROC": fmt_auroc(overall_auroc)}]

    for ds in datasets:
        if ds not in available_datasets:
            continue
        auroc, n = auroc_for_dataset(df_parquet, ds)
        rows.append({"Task": display, "Head": head, "Context": context,
                     "Dataset": ds, "N": n, "AUROC": fmt_auroc(auroc)})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_id", help="Experiment ID, e.g. sex_binary_lstm")
    parser.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS)
    parser.add_argument("--collected-dir", type=Path, default=_DEFAULT_COLLECTED)
    parser.add_argument("--channel", default="fast")
    parser.add_argument("--context", default=None,
                        help="Context length to evaluate (default: L* from analysis.csv)")
    parser.add_argument("--datasets", nargs="+", default=[],
                        help="Datasets to include (default: all present in parquet)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    # parse exp_id
    parts = args.exp_id.rsplit("_", 1)
    head  = parts[-1]
    task  = "_".join(parts[:-1])
    if head == "pool":
        parts = args.exp_id.rsplit("_", 2)
        head  = "_".join(parts[-2:])
        task  = parts[0]

    table = build(args.results_dir, args.collected_dir, task, head,
                  args.context, args.datasets, args.split)

    print(f"\nTable 9 — Cohort breakdown: {args.exp_id} (channel: {args.channel})")
    print(df_to_markdown(table))

    ctx_used = table["Context"].iloc[0] if len(table) > 0 else "?"
    cap = (f"Per-cohort AUROC breakdown for {TASK_DISPLAY.get(task, task)} ({head}) "
           f"at context {ctx_used} (channel: {args.channel}). "
           "AUROC computed from mean-probability aggregation over all inference windows per subject.")
    tex = latex_table(table, cap, f"cohort_{args.exp_id}_{args.channel}")

    if args.latex:
        print("\n" + tex)

    save_outputs(table, args.out, f"table9_cohort_{args.exp_id}_{args.channel}", latex_str=tex)


if __name__ == "__main__":
    main()
