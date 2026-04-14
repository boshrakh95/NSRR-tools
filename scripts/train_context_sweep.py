#!/usr/bin/env python3
"""
train_context_sweep.py — Phase 0, Step 4

Sweeps over context lengths defined in phase0_config.yaml, training a
lightweight sequence head on frozen SleepFM embeddings.

For each context length L:
  1. Build DataLoaders from ContextWindowDataset
  2. Instantiate the configured head (MeanPool / LSTM / Transformer)
  3. Train with early stopping on a configurable val metric; save best checkpoint
  4. Evaluate train/val/test on the best checkpoint; save metrics.json
  5. Append one row (val + test metrics) to summary.csv

BOTH task types (seq2label and anchor-based seq2seq) now produce scalar labels
so the training loop is identical for all tasks.

CLASS IMBALANCE HANDLING
────────────────────────
  class_weights (config: training.class_weights):
    "auto"  = inverse-frequency weights on training labels (recommended default)
    null    = uniform CE loss (use only for balanced tasks)
    [w0,w1] = manual per-class weights

  weighted_sampler (config: training.weighted_sampler):
    false (default) = random shuffle; true = WeightedRandomSampler so each
    training batch is approximately class-balanced. Skipped for full_night.
    Can be combined with class_weights for very severe imbalance.

  early_stopping_monitor (config: training.early_stopping_monitor):
    "val_auroc"             — recommended; threshold-independent, robust to imbalance
    "val_balanced_accuracy" — direct average recall across classes
    "val_macro_f1"          — useful when equal weight across classes matters
    "val_loss"              — original behaviour (lower is better)

  Per-epoch log shows balanced_accuracy (not raw accuracy) and the monitor
  metric; a '*' marks epochs where the checkpoint was updated.

CONTEXT LENGTH NOTES
────────────────────
  Fixed lengths (30s–80m): standard DataLoader, fixed tensor size.

  full_night: produces variable-length tensors (T varies per subject/anchor).
    DataLoader uses ContextWindowDataset.collate_fn to pad within each batch.
    Transformer head is automatically skipped for full_night (O(N²) memory).
    full_night may have fewer training examples (K=1 window per subject for
    seq2label) — consider --full-night-epochs to run for more epochs.

USAGE
─────
  python scripts/train_context_sweep.py --config configs/phase0_config.yaml

  # Override task / head:
  python scripts/train_context_sweep.py \\
      --config configs/phase0_config.yaml \\
      --task sleep_staging --task-type seq2seq --head lstm

  # Run only specific context lengths:
  python scripts/train_context_sweep.py \\
      --config configs/phase0_config.yaml --context 5m 10m full_night

  # Already-done context lengths are skipped automatically (safe to resubmit).

OUTPUT
──────
  {results_dir}/{task}_{head_type}/
    context_{L}/
      best_model.pt        — state dict of checkpoint with best val monitor metric
      metrics.json         — train/val/test metrics + early_stopping_monitor used
    summary.csv            — one row per completed context length (val + test metrics)

METRICS (all tasks)
───────────────────
  balanced_accuracy, macro_f1, per-class recall, accuracy
  binary tasks:      auroc
  multi-class tasks: macro OvR auroc
  sleep staging:     additionally cohen_kappa
"""

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    warnings.warn("wandb not installed — run tracking disabled.")

# ── local imports ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
from nsrr_tools.datasets.context_window_dataset import (
    ContextWindowDataset,
    parse_context_length,
    FULL_NIGHT_SENTINEL,
)
from nsrr_tools.models.sequence_head import build_head

# ── optional sklearn metrics ───────────────────────────────────────────────
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
    warnings.warn("scikit-learn not found — AUROC and some metrics will be skipped.")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    logits:      np.ndarray,   # (N, C)
    targets:     np.ndarray,   # (N,)
    num_classes: int,
    task:        str,
) -> dict:
    """Compute classification metrics from raw logits and integer targets."""
    preds = logits.argmax(axis=1)
    m = {"accuracy": float((preds == targets).mean())}

    if not HAS_SKLEARN:
        return m

    m["balanced_accuracy"] = float(balanced_accuracy_score(targets, preds))
    m["macro_f1"] = float(f1_score(targets, preds, average="macro", zero_division=0))

    # Per-class recall
    for c in range(num_classes):
        mask = targets == c
        if mask.any():
            m[f"recall_class{c}"] = float((preds[mask] == c).mean())
        else:
            m[f"recall_class{c}"] = float("nan")

    # AUROC
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    try:
        if num_classes == 2:
            m["auroc"] = float(roc_auc_score(targets, probs[:, 1]))
        else:
            m["auroc"] = float(
                roc_auc_score(targets, probs, multi_class="ovr", average="macro")
            )
    except ValueError:
        m["auroc"] = float("nan")

    # Cohen's kappa — extra diagnostic for sleep staging
    if task == "sleep_staging":
        m["cohen_kappa"] = float(cohen_kappa_score(targets, preds))

    return m


def compute_monitor_metric(
    monitor:     str,
    logits:      np.ndarray,
    targets:     np.ndarray,
    loss:        float,
    num_classes: int,
) -> float:
    """Return the scalar tracked for early stopping / checkpoint selection.

    monitor values:
      "val_loss"              — lower is better
      "val_auroc"             — higher is better
      "val_balanced_accuracy" — higher is better
      "val_macro_f1"          — higher is better
    """
    if monitor == "val_loss":
        return loss
    if not HAS_SKLEARN:
        return float("nan")
    preds = logits.argmax(axis=1)
    if monitor == "val_balanced_accuracy":
        return float(balanced_accuracy_score(targets, preds))
    if monitor == "val_macro_f1":
        return float(f1_score(targets, preds, average="macro", zero_division=0))
    if monitor == "val_auroc":
        probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
        try:
            if num_classes == 2:
                return float(roc_auc_score(targets, probs[:, 1]))
            return float(roc_auc_score(targets, probs, multi_class="ovr", average="macro"))
        except ValueError:
            return float("nan")
    raise ValueError(
        f"Unknown early_stopping_monitor: {monitor!r}. "
        "Choose from: val_loss, val_auroc, val_balanced_accuracy, val_macro_f1"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer,
    criterion: nn.Module,
    device:    torch.device,
    scaler,
    train:     bool,
    max_grad_norm: float = 1.0,
):
    """One epoch.  Returns (avg_loss, logits_np, targets_np)."""
    model.train(train)
    total_loss  = 0.0
    all_logits  = []
    all_targets = []

    with torch.set_grad_enabled(train):
        for x, mask, y in loader:
            x    = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            y    = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                logits = model(x, mask)          # (B, C)
                loss   = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

            total_loss  += loss.item() * x.size(0)
            all_logits.append(logits.detach().cpu().float().numpy())
            all_targets.append(y.detach().cpu().numpy())

    avg_loss   = total_loss / max(len(loader.dataset), 1)
    logits_np  = np.concatenate(all_logits,  axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    return avg_loss, logits_np, targets_np


# ─────────────────────────────────────────────────────────────────────────────
# Single context-length experiment
# ─────────────────────────────────────────────────────────────────────────────

def train_one_context(
    cfg:             dict,
    context_length:  str,
    task:            str,
    task_type:       str,
    head_type:       str,
    out_dir:         Path,
    device:          torch.device,
    datasets_filter: list,
    extra_epochs:    int,
    limit:           int,
    max_items:       int,
    use_wandb:       bool = False,
    wandb_project:   str  = "nsrr-phase0",
    wandb_entity:    str  = None,
    batch_size:      int  = 32,
):
    train_batch_size = batch_size
    eval_batch_size  = batch_size * 2
    t_cfg = cfg["training"]
    N     = parse_context_length(context_length)
    is_full_night = (N == FULL_NIGHT_SENTINEL)

    # Transformer is O(N²) — skip for full_night
    if is_full_night and head_type == "transformer":
        print(f"  [SKIP] Transformer head not supported for full_night context.")
        return None

    print(f"\n{'='*60}")
    print(f"Context: {context_length}")
    print(f"{'='*60}")

    # ── Datasets ──────────────────────────────────────────────────────────
    def make_ds(split):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ContextWindowDataset(
                cfg=cfg,
                split=split,
                context_length=context_length,
                task=task,
                task_type=task_type,
                datasets=datasets_filter,
                limit=limit,
                max_items=max_items,
            )

    train_ds = make_ds("train")
    val_ds   = make_ds("val")
    test_ds  = make_ds("test")

    num_classes = train_ds.num_classes
    print(f"  Items — train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")
    print(f"  num_classes: {num_classes}")

    # ── W&B run ────────────────────────────────────────────────────────────
    wb_run = None
    if use_wandb and HAS_WANDB:
        wb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=f"{exp_id}_{context_length}",
            group=exp_id,
            tags=[task, head_type, context_length, task_type],
            config={
                "task":           task,
                "task_type":      task_type,
                "head_type":      head_type,
                "context_length": context_length,
                "n_train":        len(train_ds),
                "n_val":          len(val_ds),
                "n_test":         len(test_ds),
                **{k: v for k, v in cfg["training"].items()
                   if not isinstance(v, (list, dict))},
                **{k: v for k, v in cfg["model"].items()
                   if not isinstance(v, (list, dict))},
            },
            dir=os.environ.get("WANDB_DIR", "/tmp"),
            reinit=True,
        )

    # ── Class weights (computed before DataLoaders so sampler can reuse) ───
    train_labels      = np.array([entry[2] for entry in train_ds._index])
    class_weights_cfg = t_cfg.get("class_weights")
    w_auto            = None   # kept for WeightedRandomSampler if configured

    if class_weights_cfg == "auto":
        counts = np.bincount(train_labels, minlength=num_classes).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        w_auto = len(train_labels) / (num_classes * counts)
        w_auto = w_auto / w_auto.sum() * num_classes   # normalize so mean=1
        print(f"  Auto class weights: {np.round(w_auto, 3).tolist()}")
        w = torch.tensor(w_auto, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=w)
    elif class_weights_cfg is not None:
        w_auto = np.array(class_weights_cfg, dtype=float)
        w = torch.tensor(w_auto, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── DataLoaders ────────────────────────────────────────────────────────
    collate     = ContextWindowDataset.collate_fn if is_full_night else None
    num_workers = min(4, max(1, len(train_ds) // 64))

    use_sampler = (
        t_cfg.get("weighted_sampler", False)
        and w_auto is not None
        and not is_full_night   # full_night has 1 window/subject; resampling adds no value
    )
    if use_sampler:
        sample_weights = torch.tensor(w_auto[train_labels], dtype=torch.float32)
        sampler        = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader   = DataLoader(
            train_ds, batch_size=train_batch_size, shuffle=False, sampler=sampler,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
            collate_fn=collate,
        )
        print(f"  WeightedRandomSampler: enabled")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=train_batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
            collate_fn=collate,
        )

    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=collate,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    m_cfg = dict(cfg["model"])
    m_cfg["num_classes"] = num_classes
    m_cfg["head_type"]   = head_type
    model = build_head({**cfg, "model": m_cfg}).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    # ── Optimizer & scheduler ──────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(t_cfg["lr"]),
        weight_decay=float(t_cfg["weight_decay"]),
    )
    epochs = t_cfg["epochs"] + extra_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = t_cfg.get("mixed_precision", True) and device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Training loop ──────────────────────────────────────────────────────
    patience  = t_cfg.get("early_stopping_patience", 5)
    monitor   = t_cfg.get("early_stopping_monitor", "val_loss")
    monitor_higher_is_better = (monitor != "val_loss")
    best_monitor = float("-inf") if monitor_higher_is_better else float("inf")
    monitor_label = monitor.replace("val_", "")   # e.g. "auroc", "balanced_accuracy"

    no_improve = 0
    history    = []

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pt"

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss, train_logits, train_targets = run_epoch(
            model, train_loader, optimizer, criterion, device, scaler, train=True
        )
        val_loss, val_logits, val_targets = run_epoch(
            model, val_loader, None, criterion, device, None, train=False
        )
        scheduler.step()

        # Balanced accuracy per epoch (meaningful for imbalanced tasks)
        if HAS_SKLEARN:
            train_bal_acc = float(balanced_accuracy_score(train_targets, train_logits.argmax(1)))
            val_bal_acc   = float(balanced_accuracy_score(val_targets,   val_logits.argmax(1)))
        else:
            train_bal_acc = float((train_logits.argmax(1) == train_targets).mean())
            val_bal_acc   = float((val_logits.argmax(1)   == val_targets).mean())

        val_monitor = compute_monitor_metric(monitor, val_logits, val_targets, val_loss, num_classes)

        history.append({
            "epoch":         epoch,
            "train_loss":    train_loss,    "val_loss":    val_loss,
            "train_bal_acc": train_bal_acc, "val_bal_acc": val_bal_acc,
            f"val_{monitor_label}": val_monitor,
        })

        improved = (
            val_monitor > best_monitor if monitor_higher_is_better
            else val_monitor < best_monitor
        )
        if improved:
            best_monitor = val_monitor
            no_improve   = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        # if epoch % 5 == 0 or epoch == 1: # to print every n epochs and the first epoch
        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"loss: train={train_loss:.4f}  val={val_loss:.4f} | "
            f"bal_acc: train={train_bal_acc:.3f}  val={val_bal_acc:.3f} | "
            f"{monitor_label}: val={val_monitor:.4f}  best={best_monitor:.4f}{'*' if improved else ''} | "
            f"patience={no_improve}/{patience}"
        )

        if wb_run is not None:
            wb_run.log({
                "train/loss":     train_loss,
                "val/loss":       val_loss,
                "train/bal_acc":  train_bal_acc,
                "val/bal_acc":    val_bal_acc,
                f"val/{monitor_label}": val_monitor,
                "lr":             optimizer.param_groups[0]["lr"],
            }, step=epoch)

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch}.")
            break

    elapsed = time.time() - t0
    print(f"  Training time: {elapsed/60:.1f} min")

    # ── Evaluation ─────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    _, train_logits, train_targets = run_epoch(
        model, train_loader, None, criterion, device, None, train=False
    )
    _, val_logits,  val_targets  = run_epoch(
        model, val_loader,  None, criterion, device, None, train=False
    )
    _, test_logits, test_targets = run_epoch(
        model, test_loader, None, criterion, device, None, train=False
    )

    train_metrics = compute_metrics(train_logits, train_targets, num_classes, task)
    val_metrics   = compute_metrics(val_logits,   val_targets,   num_classes, task)
    test_metrics  = compute_metrics(test_logits,  test_targets,  num_classes, task)

    metrics = {
        "context_length":    context_length,
        "task":              task,
        "task_type":         task_type,
        "head_type":         head_type,
        "num_classes":       num_classes,
        "n_train":           len(train_ds),
        "n_val":             len(val_ds),
        "n_test":            len(test_ds),
        "early_stopping_monitor": monitor,
        "best_val_monitor":       best_monitor,
        "n_epochs_run":      len(history),
        "training_time_min": elapsed / 60,
        "train":             train_metrics,
        "val":               val_metrics,
        "test":              test_metrics,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Train: {train_metrics}")
    print(f"  Val:   {val_metrics}")
    print(f"  Test:  {test_metrics}")

    if wb_run is not None:
        wb_run.summary.update({f"train/{k}": v for k, v in train_metrics.items()})
        wb_run.summary.update({f"val/{k}":   v for k, v in val_metrics.items()})
        wb_run.summary.update({f"test/{k}":  v for k, v in test_metrics.items()})
        wb_run.summary["training_time_min"] = elapsed / 60
        wb_run.summary["n_epochs_run"]      = len(history)
        wb_run.finish()

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Summary CSV
# ─────────────────────────────────────────────────────────────────────────────

def append_to_summary(summary_path: Path, metrics: dict):
    """Append one row (flattened val/test dicts) to summary.csv."""
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
    parser.add_argument("--config",    required=True, help="Path to phase0_config.yaml")
    parser.add_argument("--task",      default=None,  help="Override dataset.task")
    parser.add_argument("--task-type", default=None,  dest="task_type",
                        help="seq2label | seq2seq")
    parser.add_argument("--head",      default=None,  dest="head_type",
                        help="mean_pool | lstm | transformer")
    parser.add_argument("--context",   default=None,  nargs="+",
                        help="Run only these context lengths e.g. --context 5m 10m full_night")
    parser.add_argument("--datasets",  default=None,  nargs="+",
                        help="Restrict to these datasets (for debugging)")
    parser.add_argument("--limit",     default=None,  type=int,
                        help="Max subjects per split (for debugging, e.g. --limit 20)")
    parser.add_argument("--max-items", default=None,  type=int, dest="max_items",
                        help="Max total items per split after index build (for debugging, e.g. --max-items 200)")
    parser.add_argument("--full-night-epochs", default=0, type=int, dest="full_night_epochs",
                        help="Extra epochs to add for full_night (compensates fewer samples)")
    parser.add_argument("--cpu",           action="store_true")
    parser.add_argument("--wandb-project", default="nsrr-phase0", dest="wandb_project",
                        help="W&B project name (default: nsrr-phase0)")
    parser.add_argument("--wandb-entity",  default=None, dest="wandb_entity",
                        help="W&B entity/team (default: your personal account)")
    parser.add_argument("--no-wandb",      action="store_true", dest="no_wandb",
                        help="Disable W&B logging")
    parser.add_argument("--batch-size",    default=None, type=int, dest="batch_size",
                        help="Training batch size (default: 32). Val/test use 2× this value.")
    parser.add_argument("--lr",            default=None, type=float,
                        help="Override training.lr from config (e.g. --lr 1e-4)")
    parser.add_argument("--run-tag",       default="", dest="run_tag",
                        help="Suffix appended to experiment folder name, e.g. 'lr1e4'. "
                             "Allows multiple runs without overwriting. Default: no suffix.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    task      = args.task      or cfg["dataset"]["task"]
    task_type = args.task_type or cfg["dataset"]["task_type"]
    head_type = args.head_type or cfg["model"]["head_type"]
    train_batch_size = args.batch_size or 32

    # Apply LR override before passing cfg into training
    if args.lr is not None:
        cfg["training"]["lr"] = args.lr

    context_lengths = args.context or cfg["dataset"]["context_lengths"]

    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        # Non-interactive login — reads WANDB_API_KEY from environment
        wandb.login(relogin=False)

    device = torch.device(
        "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device:          {device}")
    print(f"Task:            {task}  ({task_type})")
    print(f"Head:            {head_type}")
    print(f"Context lengths: {context_lengths}")

    results_dir  = Path(cfg["logging"]["results_dir"])
    exp_id       = f"{task}_{head_type}" + (f"_{args.run_tag}" if args.run_tag else "")
    exp_dir      = results_dir / exp_id
    summary_path = exp_dir / "summary.csv"
    exp_dir.mkdir(parents=True, exist_ok=True)

    for ctx in context_lengths:
        ctx_dir = exp_dir / f"context_{ctx}"

        # Skip already-done (safe to resubmit)
        if (ctx_dir / "metrics.json").exists():
            print(f"\n[SKIP] {ctx} — metrics.json already exists.")
            continue

        # Transformer can't handle full_night — reported inside train_one_context
        extra = args.full_night_epochs if ctx == "full_night" else 0

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
                extra_epochs=extra,
                limit=args.limit,
                max_items=args.max_items,
                use_wandb=use_wandb,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                batch_size=train_batch_size,
            )
            if metrics is not None:
                append_to_summary(summary_path, metrics)
        except Exception as exc:
            print(f"\n[ERROR] context={ctx}: {exc}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Sweep complete. Results: {exp_dir}")
    print(f"Summary:         {summary_path}")


if __name__ == "__main__":
    main()
