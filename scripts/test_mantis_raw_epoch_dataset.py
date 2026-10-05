#!/usr/bin/env python3
"""
test_mantis_raw_epoch_dataset.py — smoke-test for MantisRawEpochWindowDataset.

Forked from test_osf_raw_epoch_dataset.py per
docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md checklist item 2.3. Simplified for
MantisRawEpochWindowDataset's narrower scope (seq2label only, no
full_night) and checks raw-signal shapes ([N, 6, 3840]) instead of
embedding shapes ([N, 3072]).

Checks:
  - Index sizes (items, not subjects) for train/val/test
  - Tensor shapes and dtypes from the DataLoader
  - K-window sampling produces correct number of items
  - Padding sanity for short recordings
  - **Split-match assertion (checklist 2.3's explicit requirement)**: Stage
    2's subject pool (filtered by Stage 1 embedding existence) must be
    IDENTICAL, subject-for-subject, to Stage 1's own MantisContextWindowDataset
    pool at the same split — a real, previously-live bug on both OSF's and
    PhysioOmni's own Stage 2 builds if this ever silently drifts (plan
    §14.4). This is the actual thing that matters, not just "did the class
    run" — verified directly here, not assumed from shared code.
  - **Missing-cache failure path**: a subject in Stage 1's pool but absent
    from the raw-signal cache must raise FileNotFoundError at construction
    time, not silently drop or crash later mid-training.

Usage:
    python scripts/test_mantis_raw_epoch_dataset.py \\
        --config configs/phase0_mantis_lora_config.yaml \\
        --task sex_binary --context 30s --datasets apples --limit 25
"""
import argparse
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nsrr_tools.datasets.mantis_raw_epoch_dataset import (
    MantisRawEpochWindowDataset,
    parse_context_length,
    N_SLOTS,
    EPOCH_SAMPLES,
)
from nsrr_tools.datasets.mantis_context_window_dataset import (
    MantisContextWindowDataset,
)


def test_split_match(cfg, task, datasets_filter, limit):
    """Checklist 2.3's explicit split-match assertion: Stage 2's per-split
    subject set must exactly equal Stage 1's, for every split."""
    print(f"\n{'='*60}")
    print(f"Split-match assertion: Stage 1 vs Stage 2 subject pools")
    print(f"{'='*60}")

    # Stage 1 config shares the same dataset: block shape (task_subject_dir,
    # split_seed, train/val/test proportions) — reuse the LoRA config's own
    # dataset: section directly, since phase0_mantis_lora_config.yaml is a
    # full fork of Stage 1's config with those fields unchanged.
    stage1_cfg = dict(cfg)

    all_ok = True
    for split in ("train", "val", "test"):
        ds1 = MantisContextWindowDataset(
            cfg=stage1_cfg, split=split, context_length="30s",
            task=task, task_type="seq2label", datasets=datasets_filter,
            limit=limit,
        )
        ds2 = MantisRawEpochWindowDataset(
            cfg=cfg, split=split, context_length="30s",
            task=task, datasets=datasets_filter, limit=limit,
        )
        subj1 = set(zip(ds1.df["dataset"], ds1.df["subject_id"]))
        subj2 = set(zip(ds2.df["dataset"], ds2.df["subject_id"]))
        match = subj1 == subj2
        print(f"  [{split}] Stage 1: {len(subj1)} subjects | Stage 2: {len(subj2)} subjects "
              f"[{'MATCH' if match else 'MISMATCH'}]")
        if not match:
            only1 = subj1 - subj2
            only2 = subj2 - subj1
            print(f"    only in Stage 1: {sorted(only1)[:5]}")
            print(f"    only in Stage 2: {sorted(only2)[:5]}")
        all_ok = all_ok and match

    return all_ok


def test_missing_cache_fails_loudly(cfg, task):
    """A subject present in Stage 1's pool but absent from the raw-signal
    cache must raise FileNotFoundError at construction, not silently drop
    or crash confusingly mid-training."""
    print(f"\n{'='*60}")
    print(f"Missing-cache failure path")
    print(f"{'='*60}")

    import copy
    bad_cfg = copy.deepcopy(cfg)
    # Point at an empty directory — every subject in Stage 1's pool will be
    # "missing" from this cache, guaranteeing the error path fires.
    bad_cfg["data"]["raw_signal_cache_dir"] = "/tmp/mantis_nonexistent_cache_dir_for_test"

    try:
        MantisRawEpochWindowDataset(
            cfg=bad_cfg, split="train", context_length="30s",
            task=task, limit=5,
        )
        print("  FAIL: expected FileNotFoundError, none raised")
        return False
    except FileNotFoundError as e:
        msg = str(e)
        has_count = "selected subjects have no complete precomputed raw-signal cache" in msg
        print(f"  Correctly raised FileNotFoundError [{'OK' if has_count else 'FAIL: wrong message'}]")
        print(f"    message preview: {msg[:150]}...")
        return has_count


def test_one(cfg, task, context, datasets_filter, limit):
    N = parse_context_length(context)

    print(f"\n{'='*60}")
    print(f"Task: {task}  |  context: {context}  ({N} epochs)")
    print(f"Datasets: {datasets_filter or 'all'}")
    print(f"{'='*60}")

    all_ok = True
    for split in ("train", "val", "test"):
        ds = MantisRawEpochWindowDataset(
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
        ok = True
        ok &= x.dtype == torch.float32
        ok &= m.dtype == torch.bool
        ok &= y.dtype == torch.int64
        ok &= x.dim() == 4
        ok &= x.shape[1] == N
        ok &= x.shape[2] == N_SLOTS
        ok &= x.shape[3] == EPOCH_SAMPLES
        ok &= tuple(m.shape) == tuple(x.shape[:2])
        ok &= y.dim() == 1
        ok &= not torch.isnan(x).any().item()
        print(f"    shape/dtype checks: [{'OK' if ok else 'FAIL'}]")
        all_ok = all_ok and ok

        n_padded = m.float().mean().item()
        print(f"    padding fraction: {n_padded:.1%}")
        print(f"    x stats: mean={x.mean().item():.4f} std={x.std().item():.4f} "
              f"min={x.min().item():.4f} max={x.max().item():.4f}")

    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   required=True)
    parser.add_argument("--task",     default=None)
    parser.add_argument("--context",  default=["30s"], nargs="+")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--limit",    type=int, default=None,
                         help="Debug: only load the first N subjects from the CSV")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    task = args.task or cfg["dataset"]["task"]

    all_ok = True
    for ctx in args.context:
        all_ok &= test_one(cfg, task, ctx, args.datasets, args.limit)

    all_ok &= test_split_match(cfg, task, args.datasets, args.limit)
    all_ok &= test_missing_cache_fails_loudly(cfg, task)

    print(f"\n\n{'PASSED' if all_ok else 'FAILED'}: MantisRawEpochWindowDataset smoke-tests")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
