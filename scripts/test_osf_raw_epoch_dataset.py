#!/usr/bin/env python3
"""
test_osf_raw_epoch_dataset.py — smoke-test for OSFRawEpochWindowDataset.

Forked from test_osf_context_window_dataset.py per
docs/TSFM_OSF_IMPLEMENTATION_PLAN.md checklist item 2.2. Simplified for
OSFRawEpochWindowDataset's narrower scope (seq2label only, no full_night)
and checks raw-signal shapes ([N, 12, 1920]) instead of embedding shapes
([N, 1536]).

Checks:
  - Index sizes (items, not subjects) for train/val/test
  - Tensor shapes and dtypes from the DataLoader
  - K-window sampling produces correct number of items
  - Padding sanity for short recordings

Usage:
    python scripts/test_osf_raw_epoch_dataset.py \\
        --config configs/phase0_osf_lora_config.yaml \\
        --task apnea_binary --context 10m --datasets apples --limit 2
"""
import argparse
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nsrr_tools.datasets.osf_raw_epoch_dataset import (
    OSFRawEpochWindowDataset,
    parse_context_length,
    N_CHANNELS,
    EPOCH_SAMPLES,
)


def test_one(cfg, task, context, datasets_filter, limit):
    N = parse_context_length(context)

    print(f"\n{'='*60}")
    print(f"Task: {task}  |  context: {context}  ({N} epochs)")
    print(f"Datasets: {datasets_filter or 'all'}")
    print(f"{'='*60}")

    for split in ("train", "val", "test"):
        ds = OSFRawEpochWindowDataset(
            cfg=cfg,
            split=split,
            context_length=context,
            task=task,
            datasets=datasets_filter,
            limit=limit,
        )
        n_subjects = ds.df.shape[0]
        n_items = len(ds)

        print(f"\n  [{split}]  subjects={n_subjects}  items={n_items}  "
              f"items/subject≈{n_items/max(n_subjects,1):.1f}  →  {ds}")

        if n_items == 0:
            print("    (no items — skipping DataLoader check for this split)")
            continue

        loader = DataLoader(ds, batch_size=2, shuffle=False)
        x, m, y = next(iter(loader))

        print(f"    x : {tuple(x.shape)}  {x.dtype}")
        print(f"    m : {tuple(m.shape)}  {m.dtype}   (True=padded)")
        print(f"    y : {tuple(y.shape)}  {y.dtype}   values={y.tolist()}")

        # ── Shape assertions ───────────────────────────────────────────────
        assert x.dtype == torch.float32, f"x should be float32, got {x.dtype}"
        assert m.dtype == torch.bool,    f"mask should be bool, got {m.dtype}"
        assert y.dtype == torch.int64,   f"y should be int64, got {y.dtype}"
        assert x.dim() == 4,             f"x should be 4D (B, N, 12, 1920), got {x.dim()}D"
        assert x.shape[1] == N,          f"Expected N={N} epochs, got {x.shape[1]}"
        assert x.shape[2] == N_CHANNELS, f"Expected {N_CHANNELS} channels, got {x.shape[2]}"
        assert x.shape[3] == EPOCH_SAMPLES, f"Expected {EPOCH_SAMPLES} samples/epoch, got {x.shape[3]}"
        assert m.shape == x.shape[:2],   f"mask shape mismatch: {m.shape} vs {x.shape[:2]}"
        assert y.dim() == 1,             f"y should be 1D scalar labels"
        assert not torch.isnan(x).any(), "x contains NaNs"

        n_padded = m.float().mean().item()
        print(f"    padding fraction: {n_padded:.1%}")
        print(f"    x stats: mean={x.mean().item():.4f} std={x.std().item():.4f} "
              f"min={x.min().item():.4f} max={x.max().item():.4f}")

    print(f"\n  PASSED: {task} / {context}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   required=True)
    parser.add_argument("--task",     default=None)
    parser.add_argument("--context",  default=["10m"], nargs="+")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--limit",    type=int, default=None,
                         help="Debug: only load the first N subjects from the CSV")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    task = args.task or cfg["dataset"]["task"]

    for ctx in args.context:
        test_one(cfg, task, ctx, args.datasets, args.limit)

    print("\n\nAll OSFRawEpochWindowDataset smoke-tests PASSED.")


if __name__ == "__main__":
    main()
