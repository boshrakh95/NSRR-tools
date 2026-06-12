#!/usr/bin/env python3
"""
make_table4_sensitivity.py — Table 4: Cross-task context sensitivity.

Context sensitivity = AUROC(best_L, K=all) − AUROC(30s, K=all).
Tasks are ranked from most to least context-sensitive.
Shows which clinical tasks benefit most from longer PSG context.

Reads from: results/collected/<channel>/analysis.csv
Writes to:  results/tables/table4_sensitivity_<channel>.{csv,md,tex}

Usage:
  python scripts/make_table4_sensitivity.py
  python scripts/make_table4_sensitivity.py --head lstm --channel fast
  python scripts/make_table4_sensitivity.py --tasks sex_binary bmi_binary apnea_binary
"""

import argparse
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from table_utils import (
    TASK_DISPLAY, TASK_ORDER,
    load_analysis, filter_tasks, best_row, compute_lstar,
    fmt_auroc, df_to_markdown, latex_table, save_outputs,
)

_DEFAULT_COLLECTED = Path(__file__).parent.parent / "results" / "collected" / "phase0_v3"
_DEFAULT_OUT       = Path(__file__).parent.parent / "results" / "tables"


def build(df_all: pd.DataFrame, tasks: list, head: str) -> pd.DataFrame:
    df = filter_tasks(df_all, tasks, [head])
    rows = []
    for task_id in (tasks or TASK_ORDER):
        if task_id not in df["task"].unique():
            continue
        sub = df[(df["task"] == task_id) & (df["head"] == head)]
        if sub.empty:
            continue

        r30  = best_row(sub[sub["context_length"] == "30s"], "all")
        rmax = best_row(sub, "all")
        lstar = compute_lstar(sub)

        auroc_30  = r30["mean_prob_auroc"]  if r30  is not None else None
        auroc_max = rmax["mean_prob_auroc"] if rmax is not None else None

        if auroc_30 is not None and auroc_max is not None:
            sensitivity = auroc_max - auroc_30
            delta_str   = f"+{sensitivity:.3f}"
            difficulty  = f"{1 - auroc_30:.3f}"
        else:
            sensitivity, delta_str, difficulty = None, "—", "—"

        rows.append({
            "Task":             TASK_DISPLAY.get(task_id, task_id),
            "AUROC@30s":        fmt_auroc(auroc_30),
            "AUROC@best_L":     fmt_auroc(auroc_max),
            "Best_L":           rmax["context_length"] if rmax is not None else "—",
            "L*":               lstar if lstar else "—",
            "Sensitivity (Δ)":  delta_str,
            "Difficulty":       difficulty,
        })

    # Sort by sensitivity descending (most context-sensitive first)
    result = pd.DataFrame(rows)
    if "Sensitivity (Δ)" in result.columns:
        def _key(s):
            try: return -float(s.lstrip("+"))
            except: return 0.0
        result = result.sort_values("Sensitivity (Δ)", key=lambda c: c.map(_key))
    return result.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected-dir", type=Path, default=_DEFAULT_COLLECTED)
    parser.add_argument("--channel", default="fast")
    parser.add_argument("--head", default="lstm",
                        help="Head to use for sensitivity ranking (default: lstm)")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--results-dir", type=Path, default=None, dest="results_dir",
                        help="Scratch results directory; also saves to <results-dir>/tables/")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    df_all = load_analysis(args.collected_dir, split=args.split)
    tasks  = args.tasks or sorted(df_all["task"].unique(),
                                  key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 999)
    table = build(df_all, tasks, args.head)

    print(f"\nTable 4 — Context sensitivity (head: {args.head}, channel: {args.channel})")
    print(df_to_markdown(table))

    cap = (f"Cross-task context sensitivity (head: {args.head}, channel: {args.channel}). "
           "Sensitivity = AUROC(best\\_L, K=all) − AUROC(30s, K=all). "
           "Tasks sorted by sensitivity descending. "
           "Difficulty = 1 − AUROC@30s (room for improvement from baseline).")
    tex = latex_table(table, cap, f"sensitivity_{args.channel}_{args.head}")

    if args.latex:
        print("\n" + tex)

    scratch = Path(args.results_dir) / "tables" if args.results_dir else None
    save_outputs(table, args.out, f"table4_sensitivity_{args.channel}_{args.head}",
                 latex_str=tex, scratch_dir=scratch)


if __name__ == "__main__":
    main()
