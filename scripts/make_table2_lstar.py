#!/usr/bin/env python3
"""
make_table2_lstar.py — Table 2: Saturation context L* per task.

For each task × head, L* is the smallest context length where
AUROC(K=all) is within `tolerance` of the task's maximum AUROC.
Also reports the absolute AUROC gain from 30s to L*.

Reads from: results/collected/<channel>/analysis.csv
Writes to:  results/tables/table2_lstar_<channel>.{csv,md,tex}

Usage:
  python scripts/make_table2_lstar.py
  python scripts/make_table2_lstar.py --tolerance 0.01
  python scripts/make_table2_lstar.py --tasks sex_binary apnea_binary --latex
"""

import argparse
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from table_utils import (
    SEQ2LABEL_TASKS, TASK_DISPLAY, TASK_ORDER,
    load_analysis, filter_tasks, best_row, compute_lstar, n_subjects,
    fmt_auroc, df_to_markdown, latex_table, save_outputs,
)

_DEFAULT_COLLECTED = Path(__file__).parent.parent / "results" / "collected" / "phase0_v3"
_DEFAULT_OUT       = Path(__file__).parent.parent / "results" / "tables"


def build(df_all: pd.DataFrame, tasks: list, heads: list,
          tolerance: float = 0.005) -> pd.DataFrame:
    df = filter_tasks(df_all, tasks, heads)
    rows = []
    for task_id in (tasks or TASK_ORDER):
        if task_id not in df["task"].unique():
            continue
        display = TASK_DISPLAY.get(task_id, task_id)
        task_heads = [h for h in (heads or ["lstm", "transformer"]) if
                      not df[(df["task"] == task_id) & (df["head"] == h)].empty]
        for head in task_heads:
            sub   = df[(df["task"] == task_id) & (df["head"] == head)]
            lstar = compute_lstar(sub, tolerance=tolerance)

            # AUROC at 30s (baseline) and at best context, both at K=all
            r30  = best_row(sub[sub["context_length"] == "30s"], "all")
            rmax = best_row(sub, "all")

            auroc_30s  = r30["mean_prob_auroc"]  if r30  is not None else None
            auroc_best = rmax["mean_prob_auroc"] if rmax is not None else None

            if auroc_30s is not None and auroc_best is not None:
                delta = auroc_best - auroc_30s
                delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            else:
                delta_str = "—"

            rows.append({
                "Task":            display,
                "Head":            head,
                "L*":              lstar if lstar else "—",
                "AUROC@30s":       fmt_auroc(auroc_30s),
                "AUROC@L*":        fmt_auroc(auroc_best),
                "Δ(30s→L*)":       delta_str,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected-dir", type=Path, default=_DEFAULT_COLLECTED)
    parser.add_argument("--channel", default="fast")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--heads", nargs="+", default=None)
    parser.add_argument("--tolerance", type=float, default=0.005,
                        help="AUROC tolerance for L* (default: 0.005 = 0.5%%)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--results-dir", type=Path, default=None, dest="results_dir",
                        help="Scratch results directory; also saves to <results-dir>/tables/")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    df_all = load_analysis(args.collected_dir, split=args.split)
    tasks  = args.tasks or sorted(df_all["task"].unique(),
                                  key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 999)
    table = build(df_all, tasks, args.heads, tolerance=args.tolerance)

    print(f"\nTable 2 — Saturation L* (channel: {args.channel}, tolerance: {args.tolerance})")
    print(df_to_markdown(table))

    cap = (f"Saturation context length L* per task and head (channel: {args.channel}). "
           f"L* = smallest context where AUROC(K=all) is within "
           f"{args.tolerance:.3f} of the task maximum. "
           r"$\Delta$ = absolute AUROC gain from 30\,s to L*.")
    tex = latex_table(table, cap, f"lstar_{args.channel}")

    if args.latex:
        print("\n" + tex)

    scratch = Path(args.results_dir) / "tables" if args.results_dir else None
    save_outputs(table, args.out, f"table2_lstar_{args.channel}", latex_str=tex, scratch_dir=scratch)


if __name__ == "__main__":
    main()
