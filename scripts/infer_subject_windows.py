#!/usr/bin/env python3
"""
infer_subject_windows.py — Phase 0, Subject-level inference

Loads a trained checkpoint and runs inference on ALL available windows per
subject (not capped at K=5), saving per-window probabilities and predictions
for downstream subject-level aggregation (majority voting, mean-prob AUROC).

Output (per run):
    {inference_dir}/{task}_{head}/context_{ctx}/{split}_windows.parquet

Parquet columns:
    subject_id, dataset, window_idx,   — subject identity and window position
    true_label,                         — ground truth label
    pred_label,                         — argmax prediction
    prob_class0 … prob_classN           — softmax probabilities per class

Usage:
    python scripts/infer_subject_windows.py \\
        --config configs/phase0_config.yaml \\
        --task apnea_binary --task-type seq2label --head lstm --context 10m

    # Restrict to specific datasets:
    python scripts/infer_subject_windows.py \\
        --config configs/phase0_config.yaml \\
        --task cvd_binary --task-type seq2label --head lstm --context 30s \\
        --datasets shhs mros apples

    # Run on val split instead of test:
    python scripts/infer_subject_windows.py ... --split val

    # Use only K=5 windows (reproduces training eval exactly):
    python scripts/infer_subject_windows.py ... --no-all-windows
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
from nsrr_tools.datasets.context_window_dataset import (
    ContextWindowDataset,
    FULL_NIGHT_SENTINEL,
)
from nsrr_tools.models.sequence_head import build_head


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_dataset(cfg: dict, split: str, context_length: str,
                  task: str, task_type: str,
                  datasets_filter: list, all_windows: bool) -> ContextWindowDataset:
    """Build a ContextWindowDataset, optionally overriding K_max to use all windows."""
    if all_windows:
        # Temporarily override windows_per_subject to a very large number so
        # every non-overlapping window is included.  This does not modify
        # phase0_config.yaml — only the in-memory cfg copy.
        cfg["dataset"]["windows_per_subject"] = 99_999

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = ContextWindowDataset(
            cfg=cfg,
            split=split,
            context_length=context_length,
            task=task,
            task_type=task_type,
            datasets=datasets_filter,
        )
    return ds


def get_subject_ids(ds: ContextWindowDataset) -> list:
    """Return a list of (subject_id, dataset_name) aligned to ds._index."""
    ids = []
    for row_idx, _, _ in ds._index:
        row = ds.df.iloc[row_idx]
        ids.append((str(row["subject_id"]), str(row["dataset"])))
    return ids


def run_inference(model: torch.nn.Module, loader: DataLoader,
                  device: torch.device, num_classes: int):
    """Return (logits_np, targets_np) over the full loader."""
    model.eval()
    all_logits  = []
    all_targets = []
    with torch.no_grad():
        for x, mask, y in loader:
            x    = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            logits = model(x, mask)
            all_logits.append(logits.cpu().float().numpy())
            all_targets.append(y.numpy())
    logits_np  = np.concatenate(all_logits,  axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    return logits_np, targets_np


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all-window inference for subject-level aggregation."
    )
    parser.add_argument("--config",     required=True, help="Path to phase0_config.yaml")
    parser.add_argument("--task",       required=True, help="Task name, e.g. apnea_binary")
    parser.add_argument("--task-type",  required=True, dest="task_type",
                        help="seq2label | seq2seq")
    parser.add_argument("--head",       required=True, dest="head_type",
                        help="lstm | transformer | mean_pool")
    parser.add_argument("--context",    required=True,
                        help="Context length, e.g. 30s, 10m, 40m")
    parser.add_argument("--split",      default="test",
                        choices=["train", "val", "test"],
                        help="Which split to run inference on (default: test)")
    parser.add_argument("--datasets",   default=None, nargs="+",
                        help="Restrict to these datasets, e.g. shhs mros apples")
    parser.add_argument("--no-all-windows", action="store_true", dest="no_all_windows",
                        help="Use K=5 windows (reproduces training eval) instead of all windows")
    parser.add_argument("--batch-size", default=512, type=int, dest="batch_size")
    parser.add_argument("--num-workers", default=4, type=int, dest="num_workers")
    parser.add_argument("--cpu",        action="store_true")
    parser.add_argument("--out-dir",    default=None, dest="out_dir",
                        help="Override output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    all_windows = not args.no_all_windows
    ctx         = args.context

    # ── Locate checkpoint ─────────────────────────────────────────────────────
    results_dir = Path(cfg["logging"]["results_dir"])
    ckpt_path   = results_dir / f"{args.task}_{args.head_type}" / f"context_{ctx}" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Make sure the model has been trained for task={args.task}, "
            f"head={args.head_type}, context={ctx}."
        )

    # ── Output path ───────────────────────────────────────────────────────────
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = results_dir / "inference" / f"{args.task}_{args.head_type}" / f"context_{ctx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = out_dir / f"{args.split}_windows.parquet"

    print("=" * 68)
    print("Phase 0 — Subject-level inference")
    print("=" * 68)
    print(f"  Task:        {args.task}  ({args.task_type})")
    print(f"  Head:        {args.head_type}")
    print(f"  Context:     {ctx}")
    print(f"  Split:       {args.split}")
    print(f"  All windows: {all_windows}")
    print(f"  Datasets:    {args.datasets or '(all)'}")
    print(f"  Checkpoint:  {ckpt_path}")
    print(f"  Output:      {out_parquet}")
    print(f"  Device:      {device}")
    print()

    # ── Build dataset ─────────────────────────────────────────────────────────
    ds = build_dataset(
        cfg=cfg,
        split=args.split,
        context_length=ctx,
        task=args.task,
        task_type=args.task_type,
        datasets_filter=args.datasets,
        all_windows=all_windows,
    )
    print(f"  Dataset items: {len(ds):,}  (subjects: {len(ds.df):,})")

    # Build subject ID list aligned to the flat index
    subject_ids = get_subject_ids(ds)   # list of (subject_id, dataset_name)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=ds.collate_fn,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    # Read num_classes from the saved metrics.json so we don't need to re-derive it
    metrics_path = ckpt_path.parent / "metrics.json"
    with open(metrics_path) as f:
        saved_metrics = json.load(f)
    num_classes = saved_metrics["num_classes"]

    cfg["model"]["num_classes"] = num_classes
    model = build_head(cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"  num_classes: {num_classes}")
    print(f"  Params:      {sum(p.numel() for p in model.parameters()):,}")
    print()

    # ── Inference ─────────────────────────────────────────────────────────────
    print("  Running inference...")
    logits_np, targets_np = run_inference(model, loader, device, num_classes)

    # Softmax probabilities
    probs = torch.softmax(torch.from_numpy(logits_np), dim=-1).numpy()
    preds = logits_np.argmax(axis=1)

    # ── Build output DataFrame ────────────────────────────────────────────────
    rows = {
        "subject_id":  [sid for sid, _ in subject_ids],
        "dataset":     [dname for _, dname in subject_ids],
        "true_label":  targets_np.astype(np.int16),
        "pred_label":  preds.astype(np.int16),
    }
    for c in range(num_classes):
        rows[f"prob_class{c}"] = probs[:, c].astype(np.float32)

    # Add window index within each subject (0, 1, 2, …)
    window_idx = np.zeros(len(subject_ids), dtype=np.int32)
    seen: dict = {}
    for i, (sid, dname) in enumerate(subject_ids):
        key = (sid, dname)
        window_idx[i] = seen.get(key, 0)
        seen[key] = seen.get(key, 0) + 1
    rows["window_idx"] = window_idx

    df_out = pd.DataFrame(rows)

    # ── Save ──────────────────────────────────────────────────────────────────
    df_out.to_parquet(out_parquet, index=False)
    print(f"  Saved {len(df_out):,} rows → {out_parquet}")

    # Quick sanity: segment-level accuracy
    seg_acc = (df_out["pred_label"] == df_out["true_label"]).mean()
    n_subjects = df_out.groupby(["subject_id", "dataset"]).ngroups
    print(f"  Segment-level accuracy: {seg_acc * 100:.2f}%")
    print(f"  Unique subjects:        {n_subjects:,}")
    windows_per_subj = len(df_out) / max(n_subjects, 1)
    print(f"  Avg windows/subject:    {windows_per_subj:.1f}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
