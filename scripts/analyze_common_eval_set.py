#!/usr/bin/env python3
"""
analyze_common_eval_set.py — Post-hoc common evaluation set analysis for seq2seq tasks.

PURPOSE
───────
When seq2seq_padding_policy is "complete_only" or "max_fraction", the set of anchor
epochs evaluated at each context length is different: short contexts (30s) include
nearly all epochs, while long contexts (240m) include only the middle portion of the
recording (the first/last ~2 hours are excluded because the full symmetric window does
not fit there).

This means comparing kappa at 30s vs 240m is comparing models on DIFFERENT epoch
subsets.  The bias direction is unfavourable to short contexts: they include hard
sleep-onset N1 epochs that long contexts never see, so their kappa may be slightly
lower for a compositional reason unrelated to context length.

This script produces a SUPPLEMENTARY analysis: all context lengths are evaluated on
the SAME "common evaluation set" — the anchor epochs that are valid at the longest
(reference) context length (default: 240m).  This set is the INTERSECTION of all
per-context evaluation sets and is automatically available as the 240m parquet, since
that parquet already contains only 240m-valid anchors.

ALGORITHM
─────────
1. Load the reference parquet (default context_240m/test_windows.parquet).
2. Build the common set: {(subject_id, dataset, anchor_patch_end)} from that parquet.
3. For every context, filter its parquet to rows whose (subject_id, dataset,
   anchor_patch_end) triple is in the common set.
4. Compute metrics on filtered rows.
5. Write a summary CSV with the same column layout as the training summary.csv so
   the same saturation-plot scripts can be used.

PREREQUISITES
─────────────
- Inference parquets must have been produced with the seq2seq task type so that the
  anchor_patch_end column is present (added by infer_subject_windows.py for seq2seq).
- Use "complete_only" or "max_fraction" padding policy during inference so that each
  parquet already contains only clean (non-degenerate) anchors.

OUTPUT
──────
  {inference_dir}/{task}_{head}/common_eval_{split}_summary.csv

Usage:
    python scripts/analyze_common_eval_set.py \\
        --config configs/phase0_v3_config.yaml \\
        --task sleep_staging --head lstm \\
        --split test

    # Use a different reference context (e.g. 120m instead of 240m):
    python scripts/analyze_common_eval_set.py ... --ref-context 120m
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

try:
    from sklearn.metrics import (
        balanced_accuracy_score,
        cohen_kappa_score,
        f1_score,
        roc_auc_score,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("WARNING: sklearn not found — only accuracy will be computed.")

CONTEXT_ORDER = ["30s", "10m", "40m", "80m", "120m", "240m"]


def compute_metrics(df: pd.DataFrame, num_classes: int) -> dict:
    """Compute classification metrics from a filtered parquet DataFrame."""
    targets = df["true_label"].values
    preds   = df["pred_label"].values
    prob_cols = [f"prob_class{c}" for c in range(num_classes)]
    probs   = df[prob_cols].values.astype(np.float32)

    m = {
        "n_epochs":         len(targets),
        "n_subjects":       df.groupby(["subject_id", "dataset"]).ngroups,
        "accuracy":         float((preds == targets).mean()),
    }
    if not HAS_SKLEARN:
        return m

    m["balanced_accuracy"] = float(balanced_accuracy_score(targets, preds))
    m["macro_f1"]          = float(f1_score(targets, preds, average="macro", zero_division=0))
    m["cohen_kappa"]       = float(cohen_kappa_score(targets, preds))

    try:
        if num_classes == 2:
            m["auroc"] = float(roc_auc_score(targets, probs[:, 1]))
        else:
            m["auroc"] = float(
                roc_auc_score(targets, probs, multi_class="ovr", average="macro")
            )
    except ValueError:
        m["auroc"] = float("nan")

    # Per-class recall
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() > 0:
            m[f"recall_class{c}"] = float((preds[mask] == c).mean())
        else:
            m[f"recall_class{c}"] = float("nan")

    return m


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc common evaluation set analysis for seq2seq tasks."
    )
    parser.add_argument("--config",       required=True)
    parser.add_argument("--task",         required=True,
                        help="e.g. sleep_staging")
    parser.add_argument("--head",         required=True,
                        help="e.g. lstm | transformer")
    parser.add_argument("--split",        default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--ref-context",  default="240m", dest="ref_context",
                        help="Reference context whose valid anchors define the "
                             "common set (default: 240m).")
    parser.add_argument("--run-tag",      default="", dest="run_tag")
    parser.add_argument("--contexts",     default=None, nargs="+",
                        help="Restrict to these context lengths (default: all found).")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["logging"]["results_dir"])
    exp_id      = f"{args.task}_{args.head}" + (f"_{args.run_tag}" if args.run_tag else "")
    inf_dir     = results_dir / "inference" / exp_id

    # ── Load reference parquet (defines common set) ──────────────────────────
    ref_parquet = inf_dir / f"context_{args.ref_context}" / f"{args.split}_windows.parquet"
    if not ref_parquet.exists():
        print(f"ERROR: Reference parquet not found: {ref_parquet}")
        print(f"  Run inference for {args.ref_context} first.")
        sys.exit(1)

    df_ref = pd.read_parquet(ref_parquet)
    if "anchor_patch_end" not in df_ref.columns:
        print("ERROR: anchor_patch_end column missing from reference parquet.")
        print("  Re-run inference with the updated infer_subject_windows.py "
              "(seq2seq tasks now save anchor_patch_end).")
        sys.exit(1)

    # Common set: set of (subject_id, dataset, anchor_patch_end) triples
    common_set = set(
        zip(df_ref["subject_id"], df_ref["dataset"], df_ref["anchor_patch_end"])
    )
    print(f"Common evaluation set ({args.ref_context}): "
          f"{len(common_set):,} anchors across "
          f"{df_ref.groupby(['subject_id','dataset']).ngroups:,} subjects")

    # Infer num_classes from reference parquet
    prob_cols   = [c for c in df_ref.columns if c.startswith("prob_class")]
    num_classes = len(prob_cols)

    # ── Find all available context parquets ──────────────────────────────────
    available = sorted(
        [p.parent.name.replace("context_", "") for p in
         inf_dir.glob(f"context_*/{args.split}_windows.parquet")],
        key=lambda c: CONTEXT_ORDER.index(c) if c in CONTEXT_ORDER else 99,
    )
    if args.contexts:
        available = [c for c in available if c in args.contexts]
    if not available:
        print(f"ERROR: No {args.split}_windows.parquet found under {inf_dir}.")
        sys.exit(1)

    print(f"\nProcessing contexts: {available}")
    print(f"{'Context':<8}  {'All epochs':>12}  {'Common-set':>12}  "
          f"{'Fraction':>9}  {'Kappa':>7}  {'AUROC':>7}  {'Bal-Acc':>8}")
    print("-" * 72)

    rows = []
    for ctx in available:
        pq_path = inf_dir / f"context_{ctx}" / f"{args.split}_windows.parquet"
        df = pd.read_parquet(pq_path)

        if "anchor_patch_end" not in df.columns:
            print(f"  {ctx}: SKIP — anchor_patch_end column missing (re-run inference).")
            continue

        # Filter to common set
        mask = pd.Series(
            list(zip(df["subject_id"], df["dataset"], df["anchor_patch_end"]))
        ).isin(common_set).values
        df_filtered = df[mask].reset_index(drop=True)

        n_all    = len(df)
        n_common = len(df_filtered)
        frac     = n_common / n_all if n_all > 0 else 0.0

        if n_common == 0:
            print(f"  {ctx}: 0 anchors in common set — skipping metrics.")
            continue

        m = compute_metrics(df_filtered, num_classes)
        kappa    = m.get("cohen_kappa",       float("nan"))
        auroc    = m.get("auroc",             float("nan"))
        bal_acc  = m.get("balanced_accuracy", float("nan"))

        print(f"  {ctx:<6}  {n_all:>12,}  {n_common:>12,}  "
              f"{frac:>9.1%}  {kappa:>7.4f}  {auroc:>7.4f}  {bal_acc:>8.4f}")

        row = {
            "context_length": ctx,
            "task":           args.task,
            "head_type":      args.head,
            "ref_context":    args.ref_context,
            "split":          args.split,
            "n_epochs_all":   n_all,
            "n_epochs_common": n_common,
            "common_fraction": frac,
        }
        row.update({f"common_{k}": v for k, v in m.items()})
        rows.append(row)

    if not rows:
        print("\nNo results to save.")
        sys.exit(0)

    out_csv = inf_dir / f"common_eval_{args.split}_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")
    print(f"Use plot_saturation.py with this CSV to compare against the")
    print(f"per-context summary.csv (primary results).")


if __name__ == "__main__":
    main()
