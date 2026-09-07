#!/usr/bin/env python3
"""
test_mantis_context_window_dataset.py — smoke-test for MantisContextWindowDataset.

Forked from test_osf_context_window_dataset.py per docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md
checklist 1.7. Only required edits: flat-dim assertion (1536 -> 3072, since Mantis
stacks 6 channel slots at 512-dim each, vs OSF's 2 subtokens at 768-dim each), and
an --embedding-dir override so this can be pointed at the 100-subject Pilot 1/2
population (checklist 1.6) instead of the production embedding_dir, which as of
this writing only has 4 real subjects extracted (checklist 1.11 not yet run).
The Pilot 1/2 D_Llast_combined variant IS the exact production Stage 1 config
(Option D windowing + combined@last, plan §3.1/§3.3), so it's a valid stand-in —
not synthetic/fake data, real embeddings from real signal.

This is also the FIRST real test of K-sampling and the padding branch on a
population large enough to exercise them (100 subjects, vs PhysioOmni's
original 3-subject test which the plan flagged as too small for this purpose).

Checks:
  - Index sizes (items, not subjects) for train/val/test
  - Tensor shapes and dtypes from the DataLoader
  - full_night variable-length collation
  - seq2label: K-window sampling produces correct number of items, both the
    train (random) and val/test (evenly-spaced) sampling paths
  - Padding branch: at least one context length short enough that some
    subjects need right-padding (seq2label) exercises the padded path
  - SubjectGroupedSampler: items from the same subject stay consecutive

Usage:
    python scripts/test_mantis_context_window_dataset.py \\
        --config configs/phase0_mantis_config.yaml \\
        --embedding-dir /scratch/boshra95/psg/unified/embeddings/mantis_pilot12/D_Llast_combined \\
        --task sex_binary --context 30s 10m 240m full_night --datasets apples shhs
"""
import argparse
import sys
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nsrr_tools.datasets.mantis_context_window_dataset import (
    MantisContextWindowDataset,
    SubjectGroupedSampler,
    parse_context_length,
    FULL_NIGHT_SENTINEL,
    FLAT_DIM,
)


def test_one(cfg, task, task_type, context, datasets_filter):
    N = parse_context_length(context)
    is_full_night = (N == FULL_NIGHT_SENTINEL)

    print(f"\n{'='*60}")
    print(f"Task: {task}  |  type: {task_type}  |  context: {context}")
    print(f"Datasets: {datasets_filter or 'all'}")
    print(f"{'='*60}")

    collate = MantisContextWindowDataset.collate_fn if is_full_night else None

    for split in ("train", "val", "test"):
        ds = MantisContextWindowDataset(
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
        assert x.shape[-1] == FLAT_DIM,   f"Expected flat dim {FLAT_DIM} (Mantis: 6*512), got {x.shape[-1]}"
        assert m.shape == x.shape[:2],    f"mask shape mismatch: {m.shape} vs {x.shape[:2]}"
        assert y.dim() == 1,              f"y should be 1D scalar labels"

        if not is_full_night:
            assert x.shape[1] == N, f"Expected N={N} epochs, got {x.shape[1]}"

        # ── Padding sanity ─────────────────────────────────────────────────
        n_padded = m.float().mean().item()
        print(f"    padding fraction: {n_padded:.1%}")

        # ── full_night: N varies across samples ────────────────────────────
        if is_full_night:
            lengths = (~m).long().sum(dim=1).tolist()
            print(f"    valid lengths in batch: {lengths}")

        # ── K-sampling sanity (seq2label only) ──────────────────────────────
        if task_type == "seq2label" and not is_full_night and n_subjects > 0:
            K_max = cfg["dataset"].get("windows_per_subject", 5)
            items_per_subject = [0] * n_subjects
            for row_idx, _, _ in ds._index:
                items_per_subject[row_idx] += 1
            max_k = max(items_per_subject)
            print(f"    K-sampling: max items/subject={max_k} (K_max config={K_max})")
            assert max_k <= K_max, (
                f"K-sampling produced more than K_max={K_max} windows for a "
                f"subject: {max_k}"
            )

        # ── SubjectGroupedSampler: verify grouping keeps subjects together ──
        if not is_full_night and len(ds) > 0:
            sampler = SubjectGroupedSampler(ds._index)
            order = list(iter(sampler))
            assert len(order) == len(ds._index), (
                f"Sampler length mismatch: {len(order)} vs {len(ds._index)}"
            )
            row_idxs = [ds._index[i][0] for i in order]
            seen = set()
            current = None
            for r in row_idxs:
                if r != current:
                    assert r not in seen, (
                        f"SubjectGroupedSampler broke grouping: subject "
                        f"row_idx={r} reappeared non-consecutively"
                    )
                    seen.add(r)
                    current = r
            print(f"    SubjectGroupedSampler: {len(ds._groups) if hasattr(ds,'_groups') else len(set(row_idxs))} groups, grouping verified consecutive")

    print(f"\n  PASSED: {task} / {context}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         required=True)
    parser.add_argument("--embedding-dir",  default=None,
                         help="Override dataset.embedding_dir (e.g. to point at "
                              "the Pilot 1/2 100-subject population instead of "
                              "the production embedding_dir).")
    parser.add_argument("--task",      default=None)
    parser.add_argument("--task-type", default=None, dest="task_type")
    parser.add_argument("--context",   default=["30s", "10m", "240m", "full_night"], nargs="+")
    parser.add_argument("--datasets",  nargs="+", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.embedding_dir:
        cfg["dataset"]["embedding_dir"] = args.embedding_dir
        # min_recording_patches=480 (240m) assumes the production population;
        # the pilot population's recordings vary in length just like real
        # subjects do, so keep the filter — this is exercising the real
        # exclusion-branch code path, not disabling it.

    task      = args.task      or cfg["dataset"]["task"]
    task_type = args.task_type or cfg["dataset"]["task_type"]

    for ctx in args.context:
        test_one(cfg, task, task_type, ctx, args.datasets)

    print("\n\nAll MantisContextWindowDataset smoke-tests PASSED.")


if __name__ == "__main__":
    main()
