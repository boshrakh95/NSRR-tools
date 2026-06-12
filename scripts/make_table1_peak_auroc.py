#!/usr/bin/env python3
"""
make_table1_peak_auroc.py — Table 1: Peak AUROC across seq2label tasks.

For each task × head, finds the best AUROC across all context lengths at
K=5 (deployment scenario) and K=all (ceiling). Reports the context length
that achieved the best AUROC so readers know which L saturates each task.

Reads from: results/collected/<channel>/analysis.csv
Writes to:  results/tables/table1_peak_auroc.{csv,md,tex}

Usage:
  python scripts/make_table1_peak_auroc.py
  python scripts/make_table1_peak_auroc.py --collected-dir results/collected/phase0_v3_full --channel full
  python scripts/make_table1_peak_auroc.py --tasks sex_binary bmi_binary apnea_binary
  python scripts/make_table1_peak_auroc.py --latex
"""

import argparse
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from table_utils import (
    SEQ2LABEL_TASKS, TASK_DISPLAY, TASK_ORDER,
    load_analysis, filter_tasks, best_row, n_subjects,
    fmt_auroc, df_to_markdown, latex_table, save_outputs,
)

_DEFAULT_COLLECTED = Path(__file__).parent.parent / "results" / "collected" / "phase0_v3"
_DEFAULT_OUT       = Path(__file__).parent.parent / "results" / "tables"


def build(df_all: pd.DataFrame, tasks: list, heads: list, k_deploy: int = 5) -> pd.DataFrame:
    df = filter_tasks(df_all, tasks, heads)
    rows = []
    for task_id in (tasks or TASK_ORDER):
        if task_id not in df["task"].unique():
            continue
        display = TASK_DISPLAY.get(task_id, task_id)
        task_heads = [h for h in (heads or ["lstm", "transformer"]) if
                      not df[(df["task"] == task_id) & (df["head"] == h)].empty]
        for head in task_heads:
            sub = df[(df["task"] == task_id) & (df["head"] == head)]
            n   = n_subjects(sub)
            r5  = best_row(sub, k_deploy)
            rall = best_row(sub, "all")
            rows.append({
                "Task":               display,
                "Head":               head,
                "N_test":             n if n is not None else "—",
                f"Best_L@K={k_deploy}": r5["context_length"]  if r5  is not None else "—",
                f"AUROC@K={k_deploy}":  fmt_auroc(r5["mean_prob_auroc"]  if r5  is not None else None),
                "Best_L@K=all":       rall["context_length"] if rall is not None else "—",
                "AUROC@K=all":        fmt_auroc(rall["mean_prob_auroc"] if rall is not None else None),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected-dir", type=Path, default=_DEFAULT_COLLECTED,
                        help="Path to collected directory containing analysis.csv")
    parser.add_argument("--channel", default="fast",
                        help="Label for this channel configuration (default: fast)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Task IDs to include (default: all in analysis.csv)")
    parser.add_argument("--heads", nargs="+", default=None,
                        help="Heads to include (default: all present)")
    parser.add_argument("--k-deploy", type=int, default=5,
                        help="K for the deployment column (default: 5)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                        help="Output directory (default: results/tables/)")
    parser.add_argument("--results-dir", type=Path, default=None, dest="results_dir",
                        help="Scratch results directory; if given also saves to <results-dir>/tables/")
    parser.add_argument("--latex", action="store_true", help="Print LaTeX table to stdout")
    args = parser.parse_args()

    df_all = load_analysis(args.collected_dir, split=args.split)
    tasks  = args.tasks or sorted(df_all["task"].unique(),
                                  key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 999)
    heads  = args.heads

    table = build(df_all, tasks, heads, k_deploy=args.k_deploy)

    print(f"\nTable 1 — Peak AUROC (channel: {args.channel}, split: {args.split})")
    print(df_to_markdown(table))

    cap = (f"Peak test AUROC for each seq2label task and head (channel: {args.channel}). "
           f"Best\\_L: context length at which AUROC is highest. "
           f"K={args.k_deploy}: deployment scenario (K windows per subject). "
           f"K=all: full-night ceiling.")
    lbl = f"peak_auroc_{args.channel}"
    tex = latex_table(table, cap, lbl)

    if args.latex:
        print("\n" + tex)

    stem = f"table1_peak_auroc_{args.channel}"
    scratch = Path(args.results_dir) / "tables" if args.results_dir else None
    save_outputs(table, args.out, stem, latex_str=tex, scratch_dir=scratch)


if __name__ == "__main__":
    main()
