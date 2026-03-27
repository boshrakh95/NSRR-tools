"""
Collect Phase 0 sweep results from all summary.csv files and produce:
  1. A master CSV  (results_dir/master_results.csv)
  2. A markdown report (results_dir/RESULTS.md)

Usage:
    python scripts/collect_results.py [--results-dir /scratch/.../results/phase0]
"""

import argparse
import textwrap
from pathlib import Path

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────

CONTEXT_ORDER = ["30s", "10m", "40m", "80m"]

TASK_ORDER = [
    "sleep_staging",
    "apnea_binary",
    "apnea_class",
    "sleepiness_binary",
    "sleepiness_class",
    "cvd_binary",
    "rested_morning",
    "depression_binary",
    "depression_class",
    "insomnia_binary",
    "anxiety_binary",
]

HEAD_ORDER = ["lstm", "transformer", "mean_pool"]

# Metrics shown per task type
BINARY_METRICS   = ["test_auroc", "test_balanced_accuracy", "test_macro_f1",
                    "test_recall_class0", "test_recall_class1"]
MULTICLASS_METRICS = ["test_auroc", "test_balanced_accuracy", "test_macro_f1",
                      "test_cohen_kappa"]
STAGING_METRICS  = ["test_auroc", "test_balanced_accuracy", "test_macro_f1",
                    "test_cohen_kappa",
                    "test_recall_class0", "test_recall_class1",
                    "test_recall_class2", "test_recall_class3", "test_recall_class4"]

STAGING_CLASS_NAMES = {
    "test_recall_class0": "Wake",
    "test_recall_class1": "N1",
    "test_recall_class2": "N2",
    "test_recall_class3": "N3",
    "test_recall_class4": "REM",
}

METRIC_DISPLAY = {
    "test_auroc":            "AUROC",
    "test_balanced_accuracy":"Bal-Acc",
    "test_macro_f1":         "Macro-F1",
    "test_cohen_kappa":      "Kappa",
    "test_recall_class0":    "Rec-0",
    "test_recall_class1":    "Rec-1",
    "test_recall_class2":    "Rec-2",
    "test_recall_class3":    "Rec-3",
    "test_recall_class4":    "Rec-4",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_all_summaries(results_dir: Path) -> pd.DataFrame:
    frames = []
    for csv in results_dir.glob("*/summary.csv"):
        df = pd.read_csv(csv)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No summary.csv files found under {results_dir}")
    master = pd.concat(frames, ignore_index=True)

    # Sort by task / head / context
    master["_ctx_ord"] = master["context_length"].map(
        lambda c: CONTEXT_ORDER.index(c) if c in CONTEXT_ORDER else 99
    )
    master["_task_ord"] = master["task"].map(
        lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 99
    )
    master["_head_ord"] = master["head_type"].map(
        lambda h: HEAD_ORDER.index(h) if h in HEAD_ORDER else 99
    )
    master = master.sort_values(["_task_ord", "_head_ord", "_ctx_ord"]).drop(
        columns=["_ctx_ord", "_task_ord", "_head_ord"]
    )
    return master


def fmt(val, pct=True) -> str:
    """Format a metric value for display."""
    if pd.isna(val):
        return "—"
    if pct:
        return f"{val * 100:.1f}"
    return f"{val:.3f}"


def metrics_for_task(task: str, num_classes: int) -> list:
    if task == "sleep_staging":
        return STAGING_METRICS
    elif num_classes == 2:
        return BINARY_METRICS
    else:
        return MULTICLASS_METRICS


def display_name(col: str, task: str) -> str:
    """Return human-readable column header."""
    if task == "sleep_staging" and col in STAGING_CLASS_NAMES:
        return STAGING_CLASS_NAMES[col]
    return METRIC_DISPLAY.get(col, col)


# ── Markdown generation ───────────────────────────────────────────────────────

def md_table(rows: list[dict], columns: list[str]) -> str:
    """Render a list of dicts as a markdown table."""
    header = "| " + " | ".join(columns) + " |"
    sep    = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines  = [header, sep]
    for row in rows:
        line = "| " + " | ".join(str(row.get(c, "—")) for c in columns) + " |"
        lines.append(line)
    return "\n".join(lines)


def build_markdown(master: pd.DataFrame) -> str:
    sections = []
    sections.append("# Phase 0 Results\n")
    sections.append(
        "_All metrics are on the **test** split (%). "
        "Best context per task highlighted with ← if AUROC is highest._\n"
    )

    for task in master["task"].unique():
        task_df = master[master["task"] == task]
        task_type = task_df["task_type"].iloc[0]
        num_classes = int(task_df["num_classes"].iloc[0])
        metrics = metrics_for_task(task, num_classes)

        sections.append(f"---\n## {task}  `{task_type}` · {num_classes} classes\n")

        for head in HEAD_ORDER:
            head_df = task_df[task_df["head_type"] == head]
            if head_df.empty:
                continue

            # Sort by context
            head_df = head_df.copy()
            head_df["_ord"] = head_df["context_length"].map(
                lambda c: CONTEXT_ORDER.index(c) if c in CONTEXT_ORDER else 99
            )
            head_df = head_df.sort_values("_ord")

            col_headers = ["Context", "N-test"] + [display_name(m, task) for m in metrics]
            rows = []

            best_auroc_idx = head_df["test_auroc"].idxmax() if "test_auroc" in head_df else None

            for _, row in head_df.iterrows():
                r = {
                    "Context": row["context_length"],
                    "N-test":  f"{int(row['n_test']):,}",
                }
                for m in metrics:
                    val = row.get(m, float("nan"))
                    r[display_name(m, task)] = fmt(val)
                # Mark best AUROC row
                if row.name == best_auroc_idx:
                    r["Context"] = row["context_length"] + " ←"
                rows.append(r)

            sections.append(f"### Head: `{head}`\n")
            sections.append(md_table(rows, col_headers))
            sections.append("")

    return "\n".join(sections)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect Phase 0 sweep results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/scratch/boshra95/psg/unified/results/phase0"),
    )
    parser.add_argument("--no-save", action="store_true",
                        help="Print only, don't write files")
    args = parser.parse_args()

    master = load_all_summaries(args.results_dir)

    # ── Master CSV ────────────────────────────────────────────────────────────
    master_csv = args.results_dir / "master_results.csv"
    if not args.no_save:
        master.to_csv(master_csv, index=False)
        print(f"Master CSV saved: {master_csv}")

    # ── Markdown ──────────────────────────────────────────────────────────────
    md = build_markdown(master)
    results_md = args.results_dir / "RESULTS.md"
    if not args.no_save:
        results_md.write_text(md)
        print(f"Markdown report: {results_md}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 0 — QUICK SUMMARY  (test AUROC %)")
    print("=" * 72)

    pivot_cols = ["task", "head_type", "context_length", "test_auroc",
                  "test_balanced_accuracy", "n_test"]
    available  = [c for c in pivot_cols if c in master.columns]
    summary    = master[available].copy()
    summary["test_auroc"]            = (summary["test_auroc"] * 100).round(1)
    summary["test_balanced_accuracy"]= (summary["test_balanced_accuracy"] * 100).round(1)

    print(summary.to_string(index=False))
    print()

    # Per-task best context
    print("── Best context per task (by test AUROC) ──")
    for task in TASK_ORDER:
        t = summary[summary["task"] == task]
        if t.empty:
            continue
        best = t.loc[t["test_auroc"].idxmax()]
        print(
            f"  {task:<22}  {best['head_type']:<12}  "
            f"ctx={best['context_length']:<4}  "
            f"AUROC={best['test_auroc']:.1f}%  "
            f"BalAcc={best['test_balanced_accuracy']:.1f}%"
        )


if __name__ == "__main__":
    main()
