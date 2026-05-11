#!/usr/bin/env python3
"""
build_heatmap_df.py — Step 2 of iso-compute analysis pipeline.

Reads window_analysis_{split}.csv (produced by analyze_windows.py --k-dense)
and outputs a heatmap-ready DataFrame for plot_iso_compute.py.

Transformations applied:
  1. Parse context_length strings → context_length_min (float, minutes)
  2. Replace k=="all" with numeric max K (= n_segments / n_subjects)
  3. Rename mean_prob_{metric} → {metric}  (auroc, balanced_accuracy, f1)
  4. Add total_compute_min = context_length_min × k
  5. Drop rows where all requested metric columns are NaN

Usage:
  python scripts/build_heatmap_df.py \\
      --task sex_binary --head lstm \\
      --results-dir /scratch/boshra95/psg/unified/results/phase0_v2 \\
      --split test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CONTEXT_TO_MIN = {
    "30s": 0.5,
    "1m": 1.0,
    "2m": 2.0,
    "5m": 5.0,
    "10m": 10.0,
    "20m": 20.0,
    "40m": 40.0,
    "80m": 80.0,
    "120m": 120.0,
    "240m": 240.0,
    "full_night": 480.0,
}


def parse_context_min(s: str) -> float:
    if s in CONTEXT_TO_MIN:
        return CONTEXT_TO_MIN[s]
    s = s.strip()
    if s.endswith("m"):
        return float(s[:-1])
    if s.endswith("s"):
        return float(s[:-1]) / 60.0
    return float(s)


def main():
    parser = argparse.ArgumentParser(
        description="Build heatmap DataFrame for iso-compute plot pipeline."
    )
    parser.add_argument("--task",    required=True)
    parser.add_argument("--head",    required=True)
    parser.add_argument("--results-dir", type=Path,
                        default=Path("/scratch/boshra95/psg/unified/results/phase0_v2"),
                        dest="results_dir")
    parser.add_argument("--split",   default="test", choices=["train", "val", "test"])
    parser.add_argument("--run-tag", default="", dest="run_tag")
    parser.add_argument("--metrics", nargs="+",
                        default=["auroc", "balanced_accuracy"],
                        help="Metrics to expose as primary columns (default: auroc balanced_accuracy)")
    args = parser.parse_args()

    exp_id  = f"{args.task}_{args.head}" + (f"_{args.run_tag}" if args.run_tag else "")
    inf_dir = args.results_dir / "inference" / exp_id
    csv_in  = inf_dir / f"window_analysis_{args.split}.csv"

    if not csv_in.exists():
        print(f"Not found: {csv_in}")
        print("Run analyze_windows.py (ideally with --k-dense) first.")
        return

    df = pd.read_csv(csv_in, dtype={"k": str})

    # 1. Parse context_length → context_length_min
    df["context_length_min"] = df["context_length"].map(parse_context_min)
    df["context_label"]      = df["context_length"]

    # 2. Replace k=="all" with numeric max K for that context
    def resolve_k(row):
        if str(row["k"]).strip().lower() == "all":
            return float(row["n_segments"]) / max(float(row["n_subjects"]), 1)
        return float(row["k"])
    df["k"] = df.apply(resolve_k, axis=1).astype(float)

    # 3. Rename mean_prob_{metric} → {metric}
    rename = {f"mean_prob_{m}": m for m in args.metrics if f"mean_prob_{m}" in df.columns}
    df = df.rename(columns=rename)

    # 4. Add total_compute_min
    df["total_compute_min"] = df["context_length_min"] * df["k"]

    # 5. Drop rows where ALL requested metric columns are NaN
    present_metrics = [m for m in args.metrics if m in df.columns]
    if present_metrics:
        df = df.dropna(subset=present_metrics, how="all")

    # Sort for clean output
    df = df.sort_values(["context_length_min", "k"]).reset_index(drop=True)

    # Reorder columns: primary first
    primary = ["context_length_min", "context_label", "k", "total_compute_min",
               "n_subjects", "n_segments"]
    primary += present_metrics
    extras  = [c for c in df.columns if c.startswith(("seg_", "majority_"))]
    others  = [c for c in df.columns
               if c not in primary + extras + ["context_length", "split"]]
    final   = [c for c in primary + extras + others if c in df.columns]
    df      = df[final]

    out_path = inf_dir / f"heatmap_df_{args.split}.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved:     {out_path}")
    print(f"Rows:      {len(df)}")
    print(f"Contexts:  {sorted(df['context_label'].unique())}")
    print(f"K range:   {df['k'].min():.0f} – {df['k'].max():.0f}")
    print(f"Compute:   {df['total_compute_min'].min():.1f} – "
          f"{df['total_compute_min'].max():.1f} min")
    for m in present_metrics:
        valid = df[m].dropna()
        if not valid.empty:
            print(f"  {m}: {valid.min():.3f} – {valid.max():.3f}")


if __name__ == "__main__":
    main()
