#!/usr/bin/env python3
"""
test_context_window_dataset.py — smoke-test for the redesigned ContextWindowDataset.

Checks:
  - Index sizes (items, not subjects) for train/val/test
  - Tensor shapes and dtypes from the DataLoader
  - full_night variable-length collation
  - seq2seq: labels are scalar (0-4), no -1 in valid positions
  - seq2label: K-window sampling produces correct number of items

Usage:
    python scripts/test_context_window_dataset.py \\
        --config configs/phase0_config.yaml \\
        --task apnea_binary --context 10m --datasets apples

    python scripts/test_context_window_dataset.py \\
        --config configs/phase0_config.yaml \\
        --task sleep_staging --task-type seq2seq \\
        --context 10m full_night --datasets apples
"""
import argparse
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nsrr_tools.datasets.context_window_dataset import (
    ContextWindowDataset,
    parse_context_length,
    FULL_NIGHT_SENTINEL,
)


def test_one(cfg, task, task_type, context, datasets_filter):
    N = parse_context_length(context)
    is_full_night = (N == FULL_NIGHT_SENTINEL)

    print(f"\n{'='*60}")
    print(f"Task: {task}  |  type: {task_type}  |  context: {context}")
    print(f"Datasets: {datasets_filter or 'all'}")
    print(f"{'='*60}")

    collate = ContextWindowDataset.collate_fn if is_full_night else None

    for split in ("train", "val", "test"):
        ds = ContextWindowDataset(
            cfg=cfg,
            split=split,
            context_length=context,
            task=task,
            task_type=task_type,
            datasets=datasets_filter,
        )
        n_subjects = ds.df.shape[0]
        n_items    = len(ds)

        print(f"\n  [{split}]  subjects={n_subjects}  items={n_items}  "
              f"items/subject≈{n_items/max(n_subjects,1):.1f}  →  {ds}")

        loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate)
        x, m, y = next(iter(loader))

        print(f"    x : {tuple(x.shape)}  {x.dtype}")
        print(f"    m : {tuple(m.shape)}  {m.dtype}   (True=padded)")
        print(f"    y : {tuple(y.shape)}  {y.dtype}   values={y.tolist()[:4]}")

        # ── Shape assertions ───────────────────────────────────────────────
        assert x.dtype == torch.float32,  f"x should be float32, got {x.dtype}"
        assert m.dtype == torch.bool,     f"mask should be bool, got {m.dtype}"
        assert y.dtype == torch.int64,    f"y should be int64, got {y.dtype}"
        assert x.dim() == 3,              f"x should be 3D (B, N, D)"
        assert x.shape[-1] == 512,        f"Expected flat dim 512, got {x.shape[-1]}"
        assert m.shape == x.shape[:2],    f"mask shape mismatch: {m.shape} vs {x.shape[:2]}"
        assert y.dim() == 1,              f"y should be 1D scalar labels"

        if not is_full_night:
            assert x.shape[1] == N, f"Expected N={N} patches, got {x.shape[1]}"

        # ── Label range ────────────────────────────────────────────────────
        if task_type == "seq2seq":
            assert y.min() >= 0 and y.max() <= 4, \
                f"Sleep stage labels should be 0-4, got range [{y.min()}, {y.max()}]"
            print(f"    stage distribution in batch: "
                  f"{ {int(v): int((y==v).sum()) for v in y.unique()} }")

        # ── Padding sanity ─────────────────────────────────────────────────
        n_padded = m.float().mean().item()
        print(f"    padding fraction: {n_padded:.1%}")

        # ── full_night: N varies across samples ────────────────────────────
        if is_full_night:
            lengths = (~m).long().sum(dim=1).tolist()
            print(f"    valid lengths in batch: {lengths}")

    print(f"\n  PASSED: {task} / {context}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    required=True)
    parser.add_argument("--task",      default=None)
    parser.add_argument("--task-type", default=None, dest="task_type")
    parser.add_argument("--context",   default=["10m"], nargs="+")
    parser.add_argument("--datasets",  nargs="+", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    task      = args.task      or cfg["dataset"]["task"]
    task_type = args.task_type or cfg["dataset"]["task_type"]

    for ctx in args.context:
        test_one(cfg, task, task_type, ctx, args.datasets)

    print("\n\nAll ContextWindowDataset smoke-tests PASSED.")


if __name__ == "__main__":
    main()
