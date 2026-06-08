#!/usr/bin/env python3
"""
eval_checkpoint.py — Re-run test evaluation on a saved best_model.pt.

Reconstructs the EXACT original test split by temporarily hiding the
nan_blocklist.txt so the subject pool is identical to what was used during
training.  Blocked subjects produce NaN logits; those rows are filtered before
computing metrics.  This avoids train/test leakage that would result from
reshuffling the pool with blocked subjects removed.

Usage:
    python scripts/eval_checkpoint.py \
        --config configs/phase0_v3_full_config.yaml \
        --task sex_binary \
        --head lstm \
        --contexts 30s 10m 40m \
        --datasets apples shhs
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nsrr_tools.datasets.context_window_dataset import ContextWindowDataset
from nsrr_tools.models.sequence_head import build_head

try:
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def compute_metrics(logits: np.ndarray, targets: np.ndarray, num_classes: int) -> dict:
    preds = logits.argmax(axis=1)
    m = {"accuracy": float((preds == targets).mean())}
    if not HAS_SKLEARN:
        return m
    m["balanced_accuracy"] = float(balanced_accuracy_score(targets, preds))
    m["macro_f1"] = float(f1_score(targets, preds, average="macro", zero_division=0))
    for c in range(num_classes):
        mask = targets == c
        m[f"recall_class{c}"] = float((preds[mask] == c).mean()) if mask.any() else float("nan")
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    try:
        if num_classes == 2:
            m["auroc"] = float(roc_auc_score(targets, probs[:, 1]))
        else:
            m["auroc"] = float(roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
    except ValueError:
        m["auroc"] = float("nan")
    return m


def run_inference(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for x, mask, y in loader:
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            logits = model(x, mask)
            all_logits.append(logits.cpu().float().numpy())
            all_targets.append(y.cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    required=True)
    parser.add_argument("--task",      required=True)
    parser.add_argument("--head",      required=True)
    parser.add_argument("--contexts",  nargs="+", required=True)
    parser.add_argument("--task-type", default="seq2label")
    parser.add_argument("--datasets",  nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task: {args.task}  Head: {args.head}  Datasets: {args.datasets or '(all)'}")
    print()

    # ── Temporarily hide blocklist to reconstruct original split ──────────────
    # The blocklist is applied BEFORE the shuffle in ContextWindowDataset, so its
    # presence changes the subject pool and therefore the shuffled split
    # assignments.  Training ran with a specific pool; hiding the blocklist here
    # reproduces that exact pool so train/val/test assignments are identical to
    # what was used during training.  Blocked subjects produce NaN logits which
    # we filter out after inference.
    emb_dir = Path(cfg["dataset"]["embedding_dir"])
    blocklist_path = emb_dir / "nan_blocklist.txt"
    blocklist_bak  = emb_dir / f"nan_blocklist.txt.eval_bak_{os.getpid()}"
    blocklist_existed = blocklist_path.exists()

    results_dir = Path(cfg["logging"]["results_dir"])
    exp_dir = results_dir / f"{args.task}_{args.head}"

    if blocklist_existed:
        blocklist_path.rename(blocklist_bak)
        print(f"Blocklist hidden → original split will be reconstructed.")
        print(f"Blocked subjects will be identified by NaN logits and filtered post-inference.")
        print()

    try:
        for ctx in args.contexts:
            ckpt_path = exp_dir / f"context_{ctx}" / "best_model.pt"
            if not ckpt_path.exists():
                print(f"[{ctx}] MISSING checkpoint: {ckpt_path}")
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                test_ds = ContextWindowDataset(
                    cfg=cfg,
                    split="test",
                    context_length=ctx,
                    task=args.task,
                    task_type=args.task_type,
                    datasets=args.datasets,
                )

            num_classes = test_ds.num_classes
            loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=4, pin_memory=(device.type == "cuda"),
            )

            m_cfg = dict(cfg["model"])
            m_cfg["num_classes"] = num_classes
            m_cfg["head_type"]   = args.head
            model = build_head({**cfg, "model": m_cfg}).to(device)
            state = torch.load(ckpt_path, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)

            logits, targets = run_inference(model, loader, device)

            # Filter rows where model produced NaN logits (blocked subjects
            # whose embeddings are all-NaN — in the original test set but must
            # not contribute to metrics)
            nan_rows = np.isnan(logits).any(axis=1)
            n_nan_rows = int(nan_rows.sum())
            n_original = len(logits)
            if n_nan_rows > 0:
                logits  = logits[~nan_rows]
                targets = targets[~nan_rows]

            metrics = compute_metrics(logits, targets, num_classes)

            print(f"[{ctx}]  n_test={n_original}  nan_rows_removed={n_nan_rows}  n_clean={len(logits)}")
            print(f"  Test:  {metrics}")
            print()

    finally:
        if blocklist_existed and blocklist_bak.exists():
            blocklist_bak.rename(blocklist_path)
            print("Blocklist restored.")


if __name__ == "__main__":
    main()
