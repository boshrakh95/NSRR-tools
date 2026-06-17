#!/usr/bin/env python3
"""
make_table6_modality.py — Table 6: Modality group ablation.

For each ablation task, compares AUROC across four conditions: the full-channel
baseline (all four SleepFM modality groups active) and three training-time
ablation conditions (no_bas, cardio, bas_only) where the named groups were
zeroed in the embedding at both training and inference time. See
docs/SOTA_COMPARISON_AND_ABLATIONS.md §4A and docs/sections/methods.md §III-I
for the full design rationale.

Reads from:
  results/collected/phase0_v3/analysis.csv      (baseline / "full" condition)
  results/collected/phase0_v3_abl/analysis.csv  (ablation conditions, keyed by run_tag)
Writes to:
  results/tables/table6_modality.{csv,md,tex}

Usage:
  python scripts/make_table6_modality.py
  python scripts/make_table6_modality.py --latex
  python scripts/make_table6_modality.py --k 5
  python scripts/make_table6_modality.py --tasks sex_binary apnea_binary
"""

import argparse
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from table_utils import df_to_markdown, latex_table, save_outputs, fmt_auroc

_REPO_ROOT        = Path(__file__).parent.parent
_DEFAULT_BASELINE = _REPO_ROOT / "results" / "collected" / "phase0_v3"
_DEFAULT_ABLATION = _REPO_ROOT / "results" / "collected" / "phase0_v3_abl"
_DEFAULT_OUT      = _REPO_ROOT / "results" / "tables"

# (task_id, display name, context used for ablation) — matches
# experiments/v2_ablation_registry.yaml and SOTA_COMPARISON_AND_ABLATIONS.md §A.3
ABLATION_TASKS = [
    ("sex_binary",              "Sex",                   "120m"),
    ("apnea_binary",            "Sleep apnea (AHI≥15)", "120m"),
    ("sleep_efficiency_binary", "Sleep efficiency",       "120m"),
    ("age_class",               "Age group",              "120m"),
    ("bmi_binary",              "BMI (obese)",            "40m"),
]

# (run_tag, display name) — order matches SOTA_COMPARISON_AND_ABLATIONS.md §A.1/A.6
CONDITIONS = [
    ("abl_no_bas",   "No BAS"),
    ("abl_cardio",   "Cardio only"),
    ("abl_bas_only", "BAS only"),
]


def _load(collected_dir: Path, split: str) -> pd.DataFrame:
    path = Path(collected_dir) / "analysis.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'gen_commands.py collect' for this registry first "
            f"(see docs/EXPERIMENTS_GUIDE.md → Modality ablation → Step 4)."
        )
    df = pd.read_csv(path)
    df = df[df["split"] == split].copy()
    # Older collected CSVs (predating the ablation run_tag fix) have no run_tag
    # column at all; treat every row in them as untagged ("full" baseline).
    if "run_tag" not in df.columns:
        df["run_tag"] = ""
    df["run_tag"] = df["run_tag"].fillna("")
    return df


def _lookup(df: pd.DataFrame, task: str, run_tag: str, context: str, head: str, k: str,
            metric: str = "mean_prob_auroc"):
    sub = df[(df["task"] == task) & (df["run_tag"] == run_tag) &
             (df["context_length"] == context) & (df["head"] == head) & (df["k"] == k)]
    if sub.empty:
        return None, None
    row = sub.loc[sub[metric].idxmax()]
    return row[metric], row.get("n_subjects")


def build(df_base: pd.DataFrame, df_abl: pd.DataFrame, tasks: list,
          head: str, k: str) -> pd.DataFrame:
    rows = []
    for task_id, display, context in tasks:
        full_auroc, n_full = _lookup(df_base, task_id, "", context, head, k)
        row = {
            "Task":    display,
            "Context": context,
            "N_test":  int(n_full) if n_full is not None and pd.notna(n_full) else "—",
            "Full":    fmt_auroc(full_auroc),
        }
        for run_tag, label in CONDITIONS:
            auroc, _ = _lookup(df_abl, task_id, run_tag, context, head, k)
            row[label] = fmt_auroc(auroc)
            if auroc is not None and full_auroc is not None:
                row[f"Δ({label})"] = f"{auroc - full_auroc:+.3f}"
            else:
                row[f"Δ({label})"] = "—"
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-dir", type=Path, default=_DEFAULT_BASELINE,
                        help="Collected dir for the full-channel baseline (default: phase0_v3)")
    parser.add_argument("--ablation-dir", type=Path, default=_DEFAULT_ABLATION,
                        help="Collected dir for ablation results (default: phase0_v3_abl)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Subset of task IDs to include (default: all 5 ablation tasks)")
    parser.add_argument("--head", default="lstm",
                        help="Head to report (default: lstm — the only head run for ablation)")
    parser.add_argument("--k", default="all",
                        help="K value to report (default: all — full-night ceiling)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--results-dir", type=Path, default=None, dest="results_dir",
                        help="Ablation scratch results dir; also saves to <results-dir>/tables/")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    df_base = _load(args.baseline_dir, args.split)
    df_abl  = _load(args.ablation_dir, args.split)

    tasks = ABLATION_TASKS
    if args.tasks:
        tasks = [t for t in ABLATION_TASKS if t[0] in args.tasks]

    table = build(df_base, df_abl, tasks, args.head, args.k)

    print(f"\nTable 6 — Modality group ablation (head: {args.head}, K={args.k}, split: {args.split})")
    print(df_to_markdown(table))

    if (table.drop(columns=["Task", "Context", "N_test"]) == "—").any(axis=None):
        print("\n— means that (task, condition) hasn't finished training/inference/analysis yet.")
        print("Check status with:")
        print("  python scripts/gen_commands.py --registry experiments/v2_ablation_registry.yaml status")

    cap = (f"Modality group ablation (training-time zeroing). Full = all four SleepFM "
           f"modality groups active (baseline, fast-channel). No BAS = EEG/EOG zeroed "
           f"(RESP+EKG+EMG active). Cardio only = BAS+EMG zeroed (RESP+EKG active; "
           f"SleepFounder comparison). BAS only = RESP+EKG+EMG zeroed (BAS active). "
           f"$\\Delta$ = condition AUROC minus Full AUROC. Head: {args.head}. K={args.k}.")
    tex = latex_table(table, cap, "modality_ablation")

    if args.latex:
        print("\n" + tex)

    scratch = Path(args.results_dir) / "tables" if args.results_dir else None
    save_outputs(table, args.out, "table6_modality", latex_str=tex, scratch_dir=scratch)


if __name__ == "__main__":
    main()
