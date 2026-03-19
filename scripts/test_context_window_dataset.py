#!/usr/bin/env python3
"""
test_context_window_dataset.py — quick smoke-test for ContextWindowDataset.

Usage:
    python scripts/test_context_window_dataset.py \
        --config configs/phase0_config.yaml \
        --task apnea_binary \
        --context 10m \
        --datasets apples
"""
import argparse
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nsrr_tools.datasets.context_window_dataset import ContextWindowDataset, parse_context_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    required=True)
    parser.add_argument("--task",      default=None)
    parser.add_argument("--task-type", default=None, dest="task_type")
    parser.add_argument("--context",   default="10m")
    parser.add_argument("--datasets",  nargs="+", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    task      = args.task      or cfg["dataset"]["task"]
    task_type = args.task_type or cfg["dataset"]["task_type"]
    N = parse_context_length(args.context)

    print(f"Task: {task}  |  task_type: {task_type}  |  context: {args.context} ({N} patches)")
    print(f"Datasets filter: {args.datasets or 'all'}")
    print()

    for split in ("train", "val", "test"):
        ds = ContextWindowDataset(
            cfg=cfg,
            split=split,
            context_length=args.context,
            task=task,
            task_type=task_type,
            datasets=args.datasets,
        )
        print(f"  {split:5s}: {len(ds)} subjects  →  {ds}")

        loader = DataLoader(ds, batch_size=4, shuffle=False)
        x, m, y = next(iter(loader))
        print(f"         x: {tuple(x.shape)} {x.dtype}")
        print(f"         m: {tuple(m.shape)} {m.dtype}  (True=padded)")
        print(f"         y: {tuple(y.shape)} {y.dtype}  values={y.tolist()[:4]}")
        assert x.shape[-1] == 512, "Expected flat dim 512"
        assert x.shape[1]  == N,   f"Expected N={N} patches"
        assert m.shape      == x.shape[:2]
        print()

    print("ContextWindowDataset smoke-test PASSED.")


if __name__ == "__main__":
    main()
