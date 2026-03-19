#!/usr/bin/env python3
"""
train_context_sweep.py — Phase 0, Step 4

Sweeps over all context lengths defined in phase0_config.yaml, training a
lightweight sequence head on frozen SleepFM embeddings for a single task.

For each context length:
  1. Build DataLoaders from ContextWindowDataset
  2. Instantiate the configured head (MeanPool / LSTM / Transformer)
  3. Train with early stopping; save best checkpoint
  4. Evaluate on test set; save metrics
  5. Append one row to summary.csv

USAGE
─────
  python scripts/train_context_sweep.py --config configs/phase0_config.yaml

  # Override task / head from command line:
  python scripts/train_context_sweep.py \
      --config configs/phase0_config.yaml \
      --task apnea_binary --task-type seq2label --head lstm

  # Resume a partial sweep (already-finished context lengths are skipped):
  python scripts/train_context_sweep.py --config configs/phase0_config.yaml

OUTPUT
──────
  {results_dir}/{task}_{head_type}/
    context_{len}/
      best_model.pt   — state dict of best val-loss checkpoint
      metrics.json    — train/val/test metrics for this context length
    summary.csv       — one row per context length (appended across runs)

METRICS
───────
  seq2label : AUROC, balanced_accuracy, macro_f1, per-class recall
  seq2seq   : per-class accuracy, macro_f1, cohen_kappa (ignores label=-1)
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# ── local imports ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
from nsrr_tools.datasets.context_window_dataset import (
    ContextWindowDataset,
    parse_context_length,
)
from nsrr_tools.models.sequence_head import build_head

# ── optional metric deps (sklearn) ─────────────────────────────────────────
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
    warnings.warn("scikit-learn not found; AUROC and some metrics will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics_seq2label(
    logits: np.ndarray,   # (N, C)
    targets: np.ndarray,  # (N,)
    num_classes: int,
) -> dict:
    preds = logits.argmax(axis=1)
    metrics = {}

    if HAS_SKLEARN:
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(targets, preds))
        metrics["macro_f1"] = float(f1_score(targets, preds, average="macro", zero_division=0))

        # Per-class recall
        for c in range(num_classes):
            mask = targets == c
            if mask.any():
                metrics[f"recall_class{c}"] = float((preds[mask] == c).mean())
            else:
                metrics[f"recall_class{c}"] = float("nan")

        # AUROC
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        try:
            if num_classes == 2:
                metrics["auroc"] = float(roc_auc_score(targets, probs[:, 1]))
            else:
                metrics["auroc"] = float(
                    roc_auc_score(targets, probs, multi_class="ovr", average="macro")
                )
        except ValueError:
            metrics["auroc"] = float("nan")

    metrics["accuracy"] = float((preds == targets).mean())
    return metrics


def compute_metrics_seq2seq(
    logits: np.ndarray,   # (N_total_patches, C)
    targets: np.ndarray,  # (N_total_patches,)  -1 = padded (ignored)
) -> dict:
    valid   = targets != -1
    preds   = logits[valid].argmax(axis=1)
    targets = targets[valid]

    metrics = {"n_valid_patches": int(valid.sum())}
    metrics["accuracy"] = float((preds == targets).mean())

    if HAS_SKLEARN:
        metrics["macro_f1"]    = float(f1_score(targets, preds, average="macro", zero_division=0))
        metrics["cohen_kappa"] = float(cohen_kappa_score(targets, preds))

        for c in np.unique(targets):
            m = targets == c
            metrics[f"recall_class{c}"] = float((preds[m] == c).mean())

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler,
    task_type: str,
    train: bool,
):
    """Run one epoch. Returns (avg_loss, all_logits_np, all_targets_np)."""
    model.train(train)
    total_loss = 0.0
    all_logits  = []
    all_targets = []

    with torch.set_grad_enabled(train):
        for x, mask, y in loader:
            x    = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y    = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                logits = model(x, mask)   # (B, C) or (B, N, C)

                if task_type == "seq2label":
                    loss = criterion(logits, y)
                else:
                    B, N, C = logits.shape
                    loss = criterion(logits.reshape(B * N, C), y.reshape(B * N))

            if train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * x.size(0)

            if task_type == "seq2label":
                all_logits.append(logits.detach().cpu().float().numpy())
                all_targets.append(y.detach().cpu().numpy())
            else:
                B, N, C = logits.shape
                all_logits.append(logits.detach().cpu().float().numpy().reshape(B * N, C))
                all_targets.append(y.detach().cpu().numpy().reshape(B * N))

    avg_loss   = total_loss / max(len(loader.dataset), 1)
    logits_np  = np.concatenate(all_logits,  axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    return avg_loss, logits_np, targets_np


# ─────────────────────────────────────────────────────────────────────────────
# Single context-length experiment
# ─────────────────────────────────────────────────────────────────────────────

def train_one_context(
    cfg: dict,
    context_length: str,
    task: str,
    task_type: str,
    head_type: str,
    out_dir: Path,
    device: torch.device,
    datasets_filter: list,
):
    t_cfg = cfg["training"]
    m_cfg = dict(cfg["model"])
    N     = parse_context_length(context_length)

    print(f"\n{'='*60}")
    print(f"Context: {context_length} ({N} patches = {N*5}s)")
    print(f"{'='*60}")

    # ── Datasets & loaders ────────────────────────────────────────────────
    def make_ds(split):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ContextWindowDataset(
                cfg=cfg, split=split, context_length=context_length,
                task=task, task_type=task_type, datasets=datasets_filter,
            )

    train_ds = make_ds("train")
    val_ds   = make_ds("val")
    test_ds  = make_ds("test")

    num_classes = train_ds.num_classes
    m_cfg["num_classes"] = num_classes
    m_cfg["head_type"]   = head_type
    m_cfg["task_type"]   = task_type
    cfg_patched = {**cfg, "model": m_cfg}

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  num_classes: {num_classes}")

    num_workers = min(4, max(1, len(train_ds) // 64))
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=num_workers, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False,
                              num_workers=num_workers, pin_memory=device.type == "cuda")
    test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False,
                              num_workers=num_workers, pin_memory=device.type == "cuda")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_head(cfg_patched).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Head params: {n_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(ignore_index=-1 if task_type == "seq2seq" else -100)

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    lr  = float(t_cfg["lr"])
    wd  = float(t_cfg["weight_decay"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    epochs = t_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = t_cfg.get("mixed_precision", True) and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Training loop ─────────────────────────────────────────────────────
    patience      = t_cfg.get("early_stopping_patience", 5)
    best_val_loss = float("inf")
    no_improve    = 0
    history       = []

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pt"

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(
            model, train_loader, optimizer, criterion, device, scaler, task_type, train=True
        )
        val_loss, _, _ = run_epoch(
            model, val_loader, None, criterion, device, None, task_type, train=False
        )
        scheduler.step()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve    = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f} | "
                f"best={best_val_loss:.4f} | patience={no_improve}/{patience}"
            )

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch}.")
            break

    elapsed = time.time() - t0
    print(f"  Training time: {elapsed/60:.1f} min")

    # ── Test evaluation ───────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    _, val_logits,  val_targets  = run_epoch(
        model, val_loader,  None, criterion, device, None, task_type, train=False
    )
    _, test_logits, test_targets = run_epoch(
        model, test_loader, None, criterion, device, None, task_type, train=False
    )

    if task_type == "seq2label":
        val_metrics  = compute_metrics_seq2label(val_logits,  val_targets,  num_classes)
        test_metrics = compute_metrics_seq2label(test_logits, test_targets, num_classes)
    else:
        val_metrics  = compute_metrics_seq2seq(val_logits,  val_targets)
        test_metrics = compute_metrics_seq2seq(test_logits, test_targets)

    metrics = {
        "context_length":    context_length,
        "context_patches":   N,
        "task":              task,
        "task_type":         task_type,
        "head_type":         head_type,
        "num_classes":       num_classes,
        "best_val_loss":     best_val_loss,
        "training_time_min": elapsed / 60,
        "n_epochs_run":      len(history),
        "val":               val_metrics,
        "test":              test_metrics,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Test: {test_metrics}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def append_to_summary(summary_path: Path, metrics: dict):
    import csv
    row = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    for split in ("val", "test"):
        for k, v in metrics.get(split, {}).items():
            row[f"{split}_{k}"] = v
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 0 context-length sweep")
    parser.add_argument("--config",    required=True)
    parser.add_argument("--task",      default=None)
    parser.add_argument("--task-type", default=None, dest="task_type")
    parser.add_argument("--head",      default=None, dest="head_type",
                        help="mean_pool | lstm | transformer")
    parser.add_argument("--context",   default=None, nargs="+",
                        help="Run only specific context lengths e.g. --context 5m 10m")
    parser.add_argument("--datasets",  default=None, nargs="+")
    parser.add_argument("--cpu",       action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    task      = args.task      or cfg["dataset"]["task"]
    task_type = args.task_type or cfg["dataset"]["task_type"]
    head_type = args.head_type or cfg["model"]["head_type"]

    context_lengths = args.context or cfg["dataset"]["context_lengths"]

    device = torch.device(
        "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device:          {device}")
    print(f"Task:            {task}  ({task_type})")
    print(f"Head:            {head_type}")
    print(f"Context lengths: {context_lengths}")

    results_dir  = Path(cfg["logging"]["results_dir"])
    exp_dir      = results_dir / f"{task}_{head_type}"
    summary_path = exp_dir / "summary.csv"
    exp_dir.mkdir(parents=True, exist_ok=True)

    for ctx in context_lengths:
        ctx_dir = exp_dir / f"context_{ctx}"
        if (ctx_dir / "metrics.json").exists():
            print(f"\n[SKIP] {ctx} — already done.")
            continue

        try:
            metrics = train_one_context(
                cfg=cfg,
                context_length=ctx,
                task=task,
                task_type=task_type,
                head_type=head_type,
                out_dir=ctx_dir,
                device=device,
                datasets_filter=args.datasets,
            )
            append_to_summary(summary_path, metrics)
        except Exception as exc:
            print(f"\n[ERROR] context={ctx}: {exc}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Sweep complete. Results: {exp_dir}")
    print(f"Summary CSV:     {summary_path}")


if __name__ == "__main__":
    main()
