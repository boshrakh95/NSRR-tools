#!/usr/bin/env python3
"""
summarize_results.py — Build a summary comparison table for seq2label results.

Reads window_analysis_test.csv from each experiment's inference directory and
produces a clean console table and an optional CSV/LaTeX export.

Usage:
  # Fast-channel only
  python scripts/summarize_results.py

  # Fast vs full comparison
  python scripts/summarize_results.py --compare

  # Save to CSV
  python scripts/summarize_results.py --compare --out results_summary.csv

  # LaTeX table
  python scripts/summarize_results.py --compare --latex
"""

import argparse
import os
import pandas as pd

FAST_INFER = "/scratch/boshra95/psg/unified/results/phase0_v3/inference"
FULL_INFER = "/scratch/boshra95/psg_full/unified/results/phase0_v3_full/inference"

# Ordered task list with display names for the paper
SEQ2LABEL_TASKS = [
    ("sex_binary",                 "Sex",                      ["lstm", "transformer"]),
    ("age_class",                  "Age group",                ["lstm", "transformer"]),
    ("bmi_binary",                 "BMI (obese)",              ["lstm", "transformer"]),
    ("apnea_binary",               "Sleep apnea (AHI≥15)",    ["lstm", "transformer"]),
    ("sleep_efficiency_binary",    "Sleep efficiency",         ["lstm", "transformer"]),
    ("cvd_binary",                 "CVD history",              ["lstm", "transformer"]),
    ("sleepiness_binary",          "Daytime sleepiness",       ["lstm", "transformer"]),
    ("depression_extreme_binary",  "Depression (extreme)",     ["lstm"]),
    ("osa_binary_apples_postqc",   "OSA (APPLES)",             ["lstm"]),
    ("psqi_binary",                "Sleep quality (PSQI)",     ["lstm"]),
]


def load_window_analysis(infer_base, task, head, split="test"):
    path = os.path.join(infer_base, f"{task}_{head}", f"window_analysis_{split}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["k_num"] = pd.to_numeric(df["k"], errors="coerce")
    return df


def best_auroc(df, k_val):
    if df is None:
        return None, None
    if k_val == "all":
        sub = df[df["k"] == "all"]
    else:
        sub = df[df["k_num"] == k_val]
    if sub.empty:
        return None, None
    idx = sub["mean_prob_auroc"].idxmax()
    r = sub.loc[idx]
    return round(r["mean_prob_auroc"], 4), r["context_length"]


def n_subjects(df):
    if df is None:
        return None
    # take from the k=1 row (any context) — n_subjects is constant across k
    sub = df[df["k_num"] == 1]
    if sub.empty:
        return None
    return int(sub["n_subjects"].iloc[0])


def build_table(infer_dirs: dict, k_values=(5, "all")):
    rows = []
    for task_id, task_label, heads in SEQ2LABEL_TASKS:
        for head in heads:
            row = {"task_id": task_id, "task": task_label, "head": head}
            for channel, infer_base in infer_dirs.items():
                df = load_window_analysis(infer_base, task_id, head)
                row[f"{channel}_n"] = n_subjects(df)
                for k in k_values:
                    auroc, ctx = best_auroc(df, k)
                    row[f"{channel}_auroc_k{k}"] = auroc
                    row[f"{channel}_ctx_k{k}"] = ctx
            rows.append(row)
    return pd.DataFrame(rows)


def print_table(df, channels, k=5):
    header = f"{'Task':<32} {'Head':<12} {'N':>6}"
    for ch in channels:
        header += f"  {ch + ' AUROC':>12}  {'ctx':>6}"
    if len(channels) == 2:
        header += f"  {'Δ':>7}"
    print(header)
    print("-" * len(header))

    for _, r in df.iterrows():
        n = r.get(f"{channels[0]}_n") or "-"
        line = f"{r['task']:<32} {r['head']:<12} {str(n):>6}"
        aurocs = []
        for ch in channels:
            a = r.get(f"{ch}_auroc_k{k}")
            c = r.get(f"{ch}_ctx_k{k}", "")
            a_str = f"{a:.4f}" if a is not None else "  -   "
            line += f"  {a_str:>12}  {str(c):>6}"
            aurocs.append(a)
        if len(channels) == 2 and all(a is not None for a in aurocs):
            delta = aurocs[1] - aurocs[0]
            sign = "+" if delta >= 0 else ""
            line += f"  {sign}{delta:.4f}"
        print(line)


def to_latex(df, channels, k=5):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Seq2label task performance (AUROC, k=" + str(k) + r", best context length)}",
        r"\label{tab:seq2label_results}",
        r"\begin{tabular}{llr" + "cc" * len(channels) + ("c" if len(channels) == 2 else "") + r"}",
        r"\toprule",
    ]
    ch_header = " & ".join(f"{ch.replace('_','-')} AUROC & ctx" for ch in channels)
    delta_col = r" & $\Delta$" if len(channels) == 2 else ""
    lines.append(r"Task & Head & $N$" + " & " + ch_header + delta_col + r" \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        n = r.get(f"{channels[0]}_n") or "-"
        row_parts = [r["task"].replace("_", r"\_"), r["head"], str(n)]
        aurocs = []
        for ch in channels:
            a = r.get(f"{ch}_auroc_k{k}")
            c = r.get(f"{ch}_ctx_k{k}", "")
            row_parts.append(f"{a:.4f}" if a is not None else "—")
            row_parts.append(str(c) if c else "—")
            aurocs.append(a)
        if len(channels) == 2 and all(a is not None for a in aurocs):
            delta = aurocs[1] - aurocs[0]
            sign = "+" if delta >= 0 else ""
            row_parts.append(f"{sign}{delta:.4f}")
        lines.append(" & ".join(row_parts) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true",
                        help="Compare fast-channel vs full-channel (default: fast only)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of windows for primary metric (default: 5)")
    parser.add_argument("--split", default="test",
                        help="Evaluation split (default: test)")
    parser.add_argument("--out", default="",
                        help="Save summary to this CSV path")
    parser.add_argument("--results-dir", default="", dest="results_dir",
                        help="Scratch results directory; if given saves CSV+MD to "
                             "<results-dir>/tables/summary_<channel>.{csv,md}")
    parser.add_argument("--latex", action="store_true",
                        help="Print LaTeX table instead of plain text")
    args = parser.parse_args()

    if args.compare:
        infer_dirs = {"fast": FAST_INFER, "full": FULL_INFER}
    else:
        infer_dirs = {"fast": FAST_INFER}

    channels = list(infer_dirs.keys())
    df = build_table(infer_dirs)

    if args.latex:
        print(to_latex(df, channels, k=args.k))
    else:
        print(f"\nSeq2label results — AUROC at k={args.k} windows, best context length")
        print(f"Channels: {', '.join(channels)}\n")
        print_table(df, channels, k=args.k)
        print()
        if args.compare:
            print(f"\n--- k=all ceiling ---\n")
            print_table(df, channels, k="all")

    save_cols = ["task", "head"] + [c for c in df.columns if c not in ("task_id",)]

    if args.out:
        df[save_cols].to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")

    if args.results_dir:
        import os
        channel_label = "compare" if args.compare else "fast"
        tables_dir = os.path.join(args.results_dir, "tables")
        os.makedirs(tables_dir, exist_ok=True)
        csv_path = os.path.join(tables_dir, f"summary_{channel_label}.csv")
        df[save_cols].to_csv(csv_path, index=False)
        print(f"Saved to {csv_path} (scratch)")


if __name__ == "__main__":
    main()
