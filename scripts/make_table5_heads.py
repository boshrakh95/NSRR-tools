#!/usr/bin/env python3
"""
make_table5_heads.py — Table 5: Head architecture comparison at L*.

For each task, compares LSTM, Transformer, and MeanPool at the task's
saturation context L* (computed from the LSTM head). Reports AUROC at
K=5 and K=all, and the temporal advantage (LSTM − MeanPool).

Reads from: results/collected/<channel>/analysis.csv
Writes to:  results/tables/table5_heads_<channel>.{csv,md,tex}

Usage:
  python scripts/make_table5_heads.py
  python scripts/make_table5_heads.py --k-deploy 5
  python scripts/make_table5_heads.py --lstar-head lstm --latex
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


def build(df_all: pd.DataFrame, tasks: list, heads: list,
          k_deploy: int, lstar_head: str, tolerance: float) -> pd.DataFrame:
    df = filter_tasks(df_all, tasks, None)
    rows = []
    for task_id in (tasks or TASK_ORDER):
        if task_id not in df["task"].unique():
            continue
        display = TASK_DISPLAY.get(task_id, task_id)

        # Compute L* from the reference head (lstm by default)
        ref_sub = df[(df["task"] == task_id) & (df["head"] == lstar_head)]
        lstar   = compute_lstar(ref_sub, tolerance=tolerance) if not ref_sub.empty else None

        row = {"Task": display, "L*": lstar if lstar else "—"}

        auroc_kd  = {}
        auroc_all = {}
        for head in heads:
            head_sub = df[(df["task"] == task_id) & (df["head"] == head)]
            if head_sub.empty or lstar is None:
                auroc_kd[head]  = None
                auroc_all[head] = None
                continue
            at_lstar = head_sub[head_sub["context_length"] == lstar]
            r_kd  = best_row(at_lstar, k_deploy)
            r_all = best_row(at_lstar, "all")
            auroc_kd[head]  = r_kd["mean_prob_auroc"]  if r_kd  is not None else None
            auroc_all[head] = r_all["mean_prob_auroc"] if r_all is not None else None

        for head in heads:
            row[f"{head}@K={k_deploy}"]  = fmt_auroc(auroc_kd.get(head))
            row[f"{head}@K=all"]         = fmt_auroc(auroc_all.get(head))

        # Temporal advantage: LSTM AUROC − MeanPool AUROC at K=all
        if "lstm" in heads and "mean_pool" in heads:
            a_lstm = auroc_all.get("lstm")
            a_mp   = auroc_all.get("mean_pool")
            if a_lstm is not None and a_mp is not None:
                adv = a_lstm - a_mp
                row["Temporal_adv_(LSTM-MP)"] = f"{adv:+.3f}"
            else:
                row["Temporal_adv_(LSTM-MP)"] = "—"

        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected-dir", type=Path, default=_DEFAULT_COLLECTED)
    parser.add_argument("--channel", default="fast")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--heads", nargs="+", default=["lstm", "transformer", "mean_pool"],
                        help="Heads to compare (default: lstm transformer mean_pool)")
    parser.add_argument("--k-deploy", type=int, default=5)
    parser.add_argument("--lstar-head", default="lstm",
                        help="Head used to compute L* for each task (default: lstm)")
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--results-dir", type=Path, default=None, dest="results_dir",
                        help="Scratch results directory; also saves to <results-dir>/tables/")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    df_all = load_analysis(args.collected_dir, split=args.split)
    tasks  = args.tasks or sorted(df_all["task"].unique(),
                                  key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 999)
    table = build(df_all, tasks, args.heads, args.k_deploy, args.lstar_head, args.tolerance)

    print(f"\nTable 5 — Head comparison at L* (channel: {args.channel})")
    print(df_to_markdown(table))

    cap = (f"Head architecture comparison at task-specific saturation context L* "
           f"(channel: {args.channel}, split: test, K={args.k_deploy} and K=all). "
           f"L* is the saturation context of the {args.lstar_head.upper()} head "
           f"(smallest L within {args.tolerance:.3f} AUROC of its peak); all heads are "
           f"evaluated at this same L*. Consequently, Transformer and MeanPool AUROC values "
           f"here may be lower than their own-best-context peak reported in Table~I. "
           f"Temporal\\_adv = AUROC(lstm) $-$ AUROC(mean\\_pool) at K=all.")
    tex = latex_table(table, cap, f"heads_{args.channel}")

    if args.latex:
        print("\n" + tex)

    scratch = Path(args.results_dir) / "tables" if args.results_dir else None
    save_outputs(table, args.out, f"table5_heads_{args.channel}", latex_str=tex, scratch_dir=scratch)


if __name__ == "__main__":
    main()
