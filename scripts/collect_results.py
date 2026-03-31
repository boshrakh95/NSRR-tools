"""
Collect Phase 0 sweep results from all summary.csv files and produce:
  1. A master CSV  (results_dir/master_results.csv)
  2. A markdown report (results_dir/RESULTS.md)

Usage:
    python scripts/collect_results.py [--results-dir /scratch/.../results/phase0]
"""

import argparse
import json
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

# Core metrics shown per task type (applied to each split: train / val / test)
BINARY_CORE     = ["auroc", "balanced_accuracy", "macro_f1",
                   "recall_class0", "recall_class1"]
MULTICLASS_CORE = ["auroc", "balanced_accuracy", "macro_f1", "cohen_kappa"]
STAGING_CORE    = ["auroc", "balanced_accuracy", "macro_f1", "cohen_kappa",
                   "recall_class0", "recall_class1",
                   "recall_class2", "recall_class3", "recall_class4"]

SPLITS = ["train", "val", "test"]

STAGING_CLASS_NAMES = {
    "recall_class0": "Wake",
    "recall_class1": "N1",
    "recall_class2": "N2",
    "recall_class3": "N3",
    "recall_class4": "REM",
}

METRIC_DISPLAY = {
    "auroc":            "AUROC",
    "balanced_accuracy":"Bal-Acc",
    "macro_f1":         "Macro-F1",
    "cohen_kappa":      "Kappa",
    "recall_class0":    "Rec-0",
    "recall_class1":    "Rec-1",
    "recall_class2":    "Rec-2",
    "recall_class3":    "Rec-3",
    "recall_class4":    "Rec-4",
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


def core_metrics_for_task(task: str, num_classes: int) -> list:
    if task == "sleep_staging":
        return STAGING_CORE
    elif num_classes == 2:
        return BINARY_CORE
    else:
        return MULTICLASS_CORE


def display_name(metric: str, task: str) -> str:
    """Return human-readable metric name (without split prefix)."""
    if task == "sleep_staging" and metric in STAGING_CLASS_NAMES:
        return STAGING_CLASS_NAMES[metric]
    return METRIC_DISPLAY.get(metric, metric)


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
        "_All metrics in %. "
        "Columns grouped as Train / Val / Test. "
        "Best context per head highlighted with ← (by test AUROC)._\n"
    )

    for task in TASK_ORDER:
        task_df = master[master["task"] == task]
        if task_df.empty:
            continue
        task_type   = task_df["task_type"].iloc[0]
        num_classes = int(task_df["num_classes"].iloc[0])
        core        = core_metrics_for_task(task, num_classes)

        sections.append(f"---\n## {task}  `{task_type}` · {num_classes} classes\n")

        for head in HEAD_ORDER:
            head_df = task_df[task_df["head_type"] == head].copy()
            if head_df.empty:
                continue

            head_df["_ord"] = head_df["context_length"].map(
                lambda c: CONTEXT_ORDER.index(c) if c in CONTEXT_ORDER else 99
            )
            head_df = head_df.sort_values("_ord")

            # Build column headers: Context | N-train | metric×split...
            # Only include splits that are present in the data
            available_splits = [
                s for s in SPLITS
                if any(f"{s}_{m}" in head_df.columns for m in core)
            ]

            col_headers = ["Context", "N-train", "N-val", "N-test"]
            for split in available_splits:
                for m in core:
                    col = f"{split}_{m}"
                    if col in head_df.columns:
                        label = f"{split[0].upper()}-{display_name(m, task)}"
                        col_headers.append(label)

            best_auroc_idx = (
                head_df["test_auroc"].idxmax()
                if "test_auroc" in head_df.columns else None
            )

            rows = []
            for _, row in head_df.iterrows():
                r = {
                    "Context": row["context_length"],
                    "N-train": f"{int(row['n_train']):,}",
                    "N-val":   f"{int(row['n_val']):,}",
                    "N-test":  f"{int(row['n_test']):,}",
                }
                for split in available_splits:
                    for m in core:
                        col = f"{split}_{m}"
                        if col in head_df.columns:
                            label = f"{split[0].upper()}-{display_name(m, task)}"
                            r[label] = fmt(row.get(col, float("nan")))
                if row.name == best_auroc_idx:
                    r["Context"] = row["context_length"] + " ←"
                rows.append(r)

            sections.append(f"### Head: `{head}`\n")
            sections.append(md_table(rows, col_headers))
            sections.append("")

    return "\n".join(sections)


# ── Subject-level results ─────────────────────────────────────────────────────

def load_subject_metrics(results_dir: Path) -> pd.DataFrame:
    """
    Scan results_dir/inference/**/subject_metrics.json and return a flat
    DataFrame with one row per (task, head, context, aggregation_method).
    Returns empty DataFrame if no inference has been run yet.
    """
    inf_dir = results_dir / "inference"
    rows = []
    for jf in sorted(inf_dir.glob("**/subject_metrics.json")):
        try:
            with open(jf) as f:
                m = json.load(f)
        except Exception:
            continue

        # Path: inference/{task}_{head}/context_{ctx}/subject_metrics.json
        exp_name = jf.parent.parent.name          # e.g. apnea_binary_lstm
        ctx_name = jf.parent.name.replace("context_", "")  # e.g. 10m

        # Split exp_name into task + head (head has no underscores after task)
        for head in HEAD_ORDER:
            if exp_name.endswith(f"_{head}"):
                task = exp_name[: -(len(head) + 1)]
                break
        else:
            task = exp_name
            head = "unknown"

        for method in ("mean_prob", "majority_vote"):
            sub = m.get(method, {})
            if not sub:
                continue
            row = {
                "task":            task,
                "head_type":       head,
                "context_length":  ctx_name,
                "aggregation":     method,
                "n_subjects":      m.get("n_subjects"),
                "avg_windows":     m.get("avg_windows_per_subject"),
            }
            for k, v in sub.items():
                row[f"subj_{k}"] = v
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["_ctx_ord"]  = df["context_length"].map(lambda c: CONTEXT_ORDER.index(c) if c in CONTEXT_ORDER else 99)
    df["_task_ord"] = df["task"].map(lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 99)
    df["_head_ord"] = df["head_type"].map(lambda h: HEAD_ORDER.index(h) if h in HEAD_ORDER else 99)
    df = df.sort_values(["_task_ord", "_head_ord", "_ctx_ord", "aggregation"]).drop(
        columns=["_ctx_ord", "_task_ord", "_head_ord"]
    )
    return df


def build_subject_markdown(subj_df: pd.DataFrame) -> str:
    """Build a markdown section for subject-level results."""
    sections = ["---", "# Subject-level Results (all-window aggregation)\n"]
    sections.append(
        "_Segment-level metrics use K=5 windows (training eval). "
        "Subject-level metrics run inference on ALL available windows, "
        "then aggregate per subject._\n"
        "_mean-prob: argmax of averaged softmax probabilities. "
        "maj-vote: mode of per-window hard predictions. "
        "AUROC always from mean-prob._\n"
    )

    for task in TASK_ORDER:
        task_df = subj_df[subj_df["task"] == task]
        if task_df.empty:
            continue

        sections.append(f"---\n## {task}\n")

        for head in HEAD_ORDER:
            head_df = task_df[task_df["head_type"] == head].copy()
            if head_df.empty:
                continue

            head_df["_ord"] = head_df["context_length"].map(
                lambda c: CONTEXT_ORDER.index(c) if c in CONTEXT_ORDER else 99
            )
            head_df = head_df.sort_values(["_ord", "aggregation"])

            col_headers = ["Context", "Method", "N-subj", "Avg-wins",
                           "AUROC", "Bal-Acc", "Macro-F1"]
            if "subj_cohen_kappa" in head_df.columns:
                col_headers.append("Kappa")

            rows = []
            for _, row in head_df.iterrows():
                r = {
                    "Context":  row["context_length"],
                    "Method":   row["aggregation"].replace("_", "-"),
                    "N-subj":   f"{int(row['n_subjects']):,}" if pd.notna(row.get("n_subjects")) else "—",
                    "Avg-wins": f"{row['avg_windows']:.0f}" if pd.notna(row.get("avg_windows")) else "—",
                    "AUROC":    fmt(row.get("subj_auroc", float("nan"))),
                    "Bal-Acc":  fmt(row.get("subj_balanced_accuracy", float("nan"))),
                    "Macro-F1": fmt(row.get("subj_macro_f1", float("nan"))),
                }
                if "subj_cohen_kappa" in head_df.columns:
                    r["Kappa"] = fmt(row.get("subj_cohen_kappa", float("nan")))
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

    # ── Subject-level results ─────────────────────────────────────────────────
    subj_df = load_subject_metrics(args.results_dir)

    # ── Markdown ──────────────────────────────────────────────────────────────
    md = build_markdown(master)
    if not subj_df.empty:
        md += "\n\n" + build_subject_markdown(subj_df)
    results_md = args.results_dir / "RESULTS.md"
    if not args.no_save:
        results_md.write_text(md)
        print(f"Markdown report: {results_md}")

    # ── Console summary — segment-level ──────────────────────────────────────
    print("\n" + "=" * 72)
    print("PHASE 0 — SEGMENT-LEVEL SUMMARY  (test AUROC %)")
    print("=" * 72)

    pivot_cols = ["task", "head_type", "context_length",
                  "train_auroc", "val_auroc", "test_auroc",
                  "test_balanced_accuracy", "n_test"]
    available = [c for c in pivot_cols if c in master.columns]
    summary   = master[available].copy()
    for col in ["train_auroc", "val_auroc", "test_auroc", "test_balanced_accuracy"]:
        if col in summary.columns:
            summary[col] = (summary[col] * 100).round(1)
    print(summary.to_string(index=False))
    print()

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

    # ── Console summary — subject-level ──────────────────────────────────────
    if not subj_df.empty:
        print("\n" + "=" * 72)
        print("PHASE 0 — SUBJECT-LEVEL SUMMARY  (mean-prob AUROC %)")
        print("=" * 72)
        mp = subj_df[subj_df["aggregation"] == "mean_prob"].copy()
        show_cols = ["task", "head_type", "context_length",
                     "subj_auroc", "subj_balanced_accuracy", "n_subjects", "avg_windows"]
        show_avail = [c for c in show_cols if c in mp.columns]
        mp_show = mp[show_avail].copy()
        for col in ["subj_auroc", "subj_balanced_accuracy"]:
            if col in mp_show.columns:
                mp_show[col] = (mp_show[col] * 100).round(1)
        if "avg_windows" in mp_show.columns:
            mp_show["avg_windows"] = mp_show["avg_windows"].round(0).astype(int)
        print(mp_show.to_string(index=False))


if __name__ == "__main__":
    main()
