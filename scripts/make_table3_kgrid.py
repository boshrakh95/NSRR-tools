#!/usr/bin/env python3
"""
make_table3_kgrid.py — Table 3: AUROC vs K grid for one task × head.

Produces a pivot table with context lengths as rows and K values as columns.
This is the numerical companion to the 2D heatmap figure.

Reads from: results/collected/<channel>/analysis.csv
Writes to:  results/tables/table3_kgrid_<exp_id>.{csv,md,tex}

Usage:
  python scripts/make_table3_kgrid.py sex_binary_lstm
  python scripts/make_table3_kgrid.py sex_binary_lstm --k-values 1 5 10 20 all
  python scripts/make_table3_kgrid.py sex_binary_transformer --latex
"""

import argparse
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from table_utils import (
    TASK_DISPLAY, load_analysis, fmt_auroc,
    df_to_markdown, latex_table, save_outputs, sort_contexts,
)

_DEFAULT_COLLECTED = Path(__file__).parent.parent / "results" / "collected" / "phase0_v3"
_DEFAULT_OUT       = Path(__file__).parent.parent / "results" / "tables"
_DEFAULT_K         = [1, 5, 10, 20, 50, "all"]


def build(df_all: pd.DataFrame, task: str, head: str,
          k_values: list) -> pd.DataFrame:
    sub = df_all[(df_all["task"] == task) & (df_all["head"] == head)].copy()
    if sub.empty:
        raise ValueError(f"No data found for task={task} head={head}")

    contexts = sort_contexts(sub["context_length"].unique())
    rows = []
    for ctx in contexts:
        ctx_sub = sub[sub["context_length"] == ctx]
        row = {"Context_L": ctx}
        for k in k_values:
            if k == "all":
                r = ctx_sub[ctx_sub["k"] == "all"]
            else:
                r = ctx_sub[ctx_sub["k_num"] == float(k)]
            if r.empty:
                row[f"K={k}"] = "—"
            else:
                row[f"K={k}"] = fmt_auroc(r.iloc[0]["mean_prob_auroc"])
        rows.append(row)
    return pd.DataFrame(rows)


def parse_k_values(raw: list) -> list:
    result = []
    for v in raw:
        if v.lower() == "all":
            result.append("all")
        else:
            result.append(int(v))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_id", help="Experiment ID, e.g. sex_binary_lstm")
    parser.add_argument("--collected-dir", type=Path, default=_DEFAULT_COLLECTED)
    parser.add_argument("--channel", default="fast")
    parser.add_argument("--k-values", nargs="+", default=None,
                        help="K values to show as columns (default: 1 5 10 20 50 all)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--results-dir", type=Path, default=None, dest="results_dir",
                        help="Scratch results directory; also saves to <results-dir>/tables/")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    # parse exp_id → task + head
    parts = args.exp_id.rsplit("_", 1)
    head  = parts[-1]
    task  = "_".join(parts[:-1])
    if head not in ("lstm", "transformer", "mean_pool"):
        # handle mean_pool (two-word suffix)
        parts = args.exp_id.rsplit("_", 2)
        head  = "_".join(parts[-2:])
        task  = parts[0]

    k_values = parse_k_values(args.k_values) if args.k_values else _DEFAULT_K

    df_all = load_analysis(args.collected_dir, split=args.split)
    table  = build(df_all, task, head, k_values)

    display = TASK_DISPLAY.get(task, task)
    print(f"\nTable 3 — AUROC×K grid: {display} ({head}, channel: {args.channel})")
    print(df_to_markdown(table))

    cap = (f"AUROC vs K grid for {display} ({head}, channel: {args.channel}). "
           "Rows: training context length L. Columns: inference-time K (windows per subject). "
           "Cells with identical K×L are on the same iso-compute diagonal.")
    tex = latex_table(table, cap, f"kgrid_{args.exp_id}_{args.channel}")

    if args.latex:
        print("\n" + tex)

    stem = f"table3_kgrid_{args.exp_id}_{args.channel}"
    scratch = Path(args.results_dir) / "tables" if args.results_dir else None
    save_outputs(table, args.out, stem, latex_str=tex, scratch_dir=scratch)


if __name__ == "__main__":
    main()
