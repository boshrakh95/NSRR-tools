#!/usr/bin/env python3
"""
make_table10_ci.py — Table 10: Bootstrap CI summary (supplementary).

For each task × head at L* and K=5, reports AUROC with 95% bootstrap CIs.
Requires analysis.csv to have CI columns (from `analyze_windows.py --bootstrap N`).

Reads from: results/collected/<channel>/analysis.csv
Writes to:  results/tables/table10_ci_<channel>.{csv,md,tex}

Usage:
  python scripts/make_table10_ci.py
  python scripts/make_table10_ci.py --k-deploy 5
  python scripts/make_table10_ci.py --tasks sex_binary bmi_binary --latex
"""

import argparse
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from table_utils import (
    TASK_DISPLAY, TASK_ORDER,
    load_analysis, filter_tasks, best_row, compute_lstar, n_subjects,
    fmt_auroc, df_to_markdown, latex_table, save_outputs,
)

_DEFAULT_COLLECTED = Path(__file__).parent.parent / "results" / "collected" / "phase0_v3"
_DEFAULT_OUT       = Path(__file__).parent.parent / "results" / "tables"


def fmt_with_ci(auroc, ci_lo, ci_hi) -> str:
    if auroc is None or pd.isna(auroc):
        return "—"
    if ci_lo is None or pd.isna(ci_lo) or ci_hi is None or pd.isna(ci_hi):
        return fmt_auroc(auroc)
    return f"{auroc:.3f} [{ci_lo:.3f}–{ci_hi:.3f}]"


def build(df_all: pd.DataFrame, tasks: list, heads: list, k_deploy: int) -> pd.DataFrame:
    df = filter_tasks(df_all, tasks, heads)
    has_ci = "mean_prob_auroc_ci_lo" in df.columns

    rows = []
    for task_id in (tasks or TASK_ORDER):
        if task_id not in df["task"].unique():
            continue
        display = TASK_DISPLAY.get(task_id, task_id)
        task_heads = [h for h in (heads or ["lstm", "transformer"]) if
                      not df[(df["task"] == task_id) & (df["head"] == h)].empty]
        for head in task_heads:
            sub   = df[(df["task"] == task_id) & (df["head"] == head)]
            lstar = compute_lstar(sub)
            n     = n_subjects(sub)

            # Best row at L* and k_deploy
            at_lstar = sub[sub["context_length"] == lstar] if lstar else sub
            r_kd  = best_row(at_lstar, k_deploy)
            r_all = best_row(at_lstar, "all")

            def _auroc_ci(r):
                if r is None:
                    return "—", "—"
                auroc = r.get("mean_prob_auroc")
                if has_ci:
                    ci_lo = r.get("mean_prob_auroc_ci_lo")
                    ci_hi = r.get("mean_prob_auroc_ci_hi")
                else:
                    ci_lo = ci_hi = float("nan")
                return fmt_with_ci(auroc, ci_lo, ci_hi), r.get("context_length", "—")

            ci_kd,  ctx_kd  = _auroc_ci(r_kd)
            ci_all, ctx_all = _auroc_ci(r_all)

            rows.append({
                "Task":                  display,
                "Head":                  head,
                "N_test":                n if n else "—",
                "L*":                    lstar if lstar else "—",
                f"AUROC@K={k_deploy} [95% CI]": ci_kd,
                "AUROC@K=all [95% CI]":  ci_all,
            })

    note = "" if has_ci else "\n[CI columns absent — run analyze_windows.py --bootstrap N first]"
    return pd.DataFrame(rows), note


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collected-dir", type=Path, default=_DEFAULT_COLLECTED)
    parser.add_argument("--channel", default="fast")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--heads", nargs="+", default=None)
    parser.add_argument("--k-deploy", type=int, default=5)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--results-dir", type=Path, default=None, dest="results_dir",
                        help="Scratch results directory; also saves to <results-dir>/tables/")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    df_all = load_analysis(args.collected_dir, split=args.split)
    tasks  = args.tasks or sorted(df_all["task"].unique(),
                                  key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 999)
    table, note = build(df_all, tasks, args.heads, args.k_deploy)

    print(f"\nTable 10 — Bootstrap CI summary (channel: {args.channel}){note}")
    print(df_to_markdown(table))

    cap = (f"AUROC with 95\\%% bootstrap confidence intervals at saturation context L* "
           f"(channel: {args.channel}). "
           f"K={args.k_deploy}: deployment scenario. K=all: full-night ceiling. "
           f"CI computed by subject-level bootstrap resampling (N=1000).")
    tex = latex_table(table, cap, f"ci_{args.channel}")

    if args.latex:
        print("\n" + tex)

    scratch = Path(args.results_dir) / "tables" if args.results_dir else None
    save_outputs(table, args.out, f"table10_ci_{args.channel}", latex_str=tex, scratch_dir=scratch)


if __name__ == "__main__":
    main()
