#!/usr/bin/env python3
"""
train_osf_lora.py — OSF baseline, Stage 2 Step 1 (LoRA fine-tuning)

Genuinely new end-to-end training script — the OSF encoder is inside the
trainable graph (LoRA-adapted), so raw signal is loaded and encoded live
every training step, not precomputed once like Stage 1. See
docs/TSFM_OSF_IMPLEMENTATION_PLAN.md Appendix §6 for the full derivation.

NOT a fork of train_osf_context_sweep.py's file — but it deliberately
IMPORTS that file's low-level, already-tested building blocks
(run_epoch, compute_metrics, compute_monitor_metric, append_to_summary,
_classify_failure) rather than re-implementing them, for the same reason
checklist 2.2 factored channel-loading into a shared module: these
functions are backbone-agnostic (run_epoch just calls `model(x, mask)`
and doesn't care what's inside `model`), and duplicating them would risk
silent drift between Stage 1 and Stage 2's metric computation. Only the
model construction, checkpoint format, and warm-start logic are new.

STAGED (LP-FT), NOT JOINT: warm-starts the sequence head from the
matching Stage 1 checkpoint before LoRA fine-tuning begins — per
CLAUDE.md's "Frozen vs. LoRA-fine-tuned conditions" (Kumar et al. 2022
LP-FT justification). This is an already-made decision, not re-litigated
here.

MODEL ARCHITECTURE
───────────────────
  CombinedOSFLoRAModel(backbone, sequence_head) is built as ONE nn.Module
  BEFORE peft wrapping (per Appendix §6.1), so peft.get_peft_model() is
  called on the whole combined module with modules_to_save=["sequence_head"]
  — this makes peft's own state-dict save/load cover both the LoRA deltas
  AND the head in one call, rather than managing two separate checkpoint
  formats.

  forward(x, mask): x is raw [B, N, 12, 1920] signal (from
  OSFRawEpochWindowDataset). Each of the N epochs is run through the
  LoRA-adapted ViT's forward_encoding(..., return_sequence=False) in
  chunks (chunk_batch_size, same batching pattern as the extraction
  script), mean-pooled/stacked into [B, N, 1536], then fed through the
  sequence head — identical embedding shape/layout to Stage 1's precomputed
  embeddings, so the sequence head architecture is unchanged.

  Live-verified 2026-08-13 (checklist 2.1/this step): peft's __getattr__
  delegation correctly forwards custom methods like forward_encoding()
  through the wrapped model, and gradients do flow to LoRA parameters
  from a loss computed this way — not just a theoretical claim.

USAGE
─────
  python scripts/train_osf_lora.py --config configs/phase0_osf_lora_config.yaml \\
      --task apnea_binary --head lstm --context 30s \\
      --stage1-checkpoint /scratch/.../phase0_osf/apnea_binary_lstm/context_30s/best_model.pt

OUTPUT — same shape as Stage 1's, under phase0_osf_lora's results_dir:
  {results_dir}/{task}_{head_type}/
    context_{L}/
      best_model.pt   — peft state dict (LoRA deltas + sequence_head), NOT
                         the full 85M-param base model (kept frozen, never
                         needs saving — always reloadable from
                         embedding.checkpoint_dir)
      metrics.json
    summary.csv
"""

import argparse
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
from torch.utils.data import DataLoader

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

# ── local imports ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
from nsrr_tools.datasets.osf_raw_epoch_dataset import (
    OSFRawEpochWindowDataset,
    SubjectGroupedSampler,
    parse_context_length,
)
from nsrr_tools.models.sequence_head import build_head

# Reused, not duplicated — see module docstring.
# Side effect worth knowing: importing train_osf_context_sweep.py also
# registers ITS module-level SIGTERM handler (sys.exit(0) on SIGTERM, so
# bash's trap gets time to log TIMEOUT_REQUEUED before SLURM's SIGKILL) —
# this is exactly the auto-resume behavior jobs/train_osf_lora_gpu.sh needs
# too, so it's reused deliberately, not accidentally inherited.
sys.path.insert(0, str(_ROOT / "scripts"))
from train_osf_context_sweep import (  # noqa: E402
    run_epoch,
    compute_metrics,
    compute_monitor_metric,
    append_to_summary,
    _classify_failure,
)

# ── OSF backbone import ──────────────────────────────────────────────────────
_OSF_REPO = _ROOT.parent / "OSF-Open-Sleep-FM"
sys.path.insert(0, str(_OSF_REPO))
try:
    from osf.backbone.vit1d_cls import vit_base
except ImportError as e:
    print(f"Cannot import OSF: {e}\nExpected repo at: {_OSF_REPO}\nRun with osf_env.")
    sys.exit(1)

try:
    from sklearn.metrics import balanced_accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ─────────────────────────────────────────────────────────────────────────────
# Combined model
# ─────────────────────────────────────────────────────────────────────────────

class CombinedOSFLoRAModel(nn.Module):
    """OSF ViT backbone + sequence head, wrapped as one module BEFORE peft
    injection (Appendix §6.1) so peft's save/load covers both pieces.

    forward(x, mask) matches train_osf_context_sweep.py's run_epoch()
    expected signature exactly — x: [B, N, 12, 1920] raw signal, mask:
    [B, N] bool (True=padded) -> logits: [B, num_classes]. This is why
    run_epoch works completely unmodified for Stage 2: it has no idea
    what's inside `model`, only that model(x, mask) returns logits.
    """

    def __init__(self, backbone: nn.Module, sequence_head: nn.Module, chunk_batch_size: int = 16):
        super().__init__()
        self.backbone = backbone
        self.sequence_head = sequence_head
        self.chunk_batch_size = chunk_batch_size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, C, T = x.shape
        x_flat = x.reshape(B * N, C, T)

        cls_chunks, patch_chunks = [], []
        for i in range(0, x_flat.shape[0], self.chunk_batch_size):
            chunk = x_flat[i : i + self.chunk_batch_size]
            cls, patches = self.backbone.forward_encoding(chunk, return_sequence=False)
            cls_chunks.append(cls)
            patch_chunks.append(patches.mean(dim=1))

        cls_all = torch.cat(cls_chunks, dim=0)             # [B*N, 768]
        mean_patches_all = torch.cat(patch_chunks, dim=0)  # [B*N, 768]
        emb = torch.stack([cls_all, mean_patches_all], dim=1)  # [B*N, 2, 768]
        emb = emb.reshape(B, N, 2 * 768)                       # [B, N, 1536]

        return self.sequence_head(emb, mask)


def load_osf_backbone(checkpoint_path: str, device: torch.device):
    """Load the base OSF ViT (not yet LoRA-wrapped) — same loader logic as
    extract_osf_embeddings.py's load_model(), duplicated rather than
    imported since that script isn't meant to be used as a library and
    this is a single ~15-line function, unlike the channel-loading logic
    (checklist 2.2) which was genuinely at risk of drifting out of sync."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    meta = ckpt["metadata"]
    model = vit_base(
        num_leads=meta["num_leads"],
        seq_len=meta["seq_len"],
        patch_size=meta["patch_size_time"],
        patch_size_ch=meta["patch_size_ch"],
        lead_wise=meta["lead_wise"],
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    return model, meta


def build_combined_lora_model(cfg: dict, num_classes: int, head_type: str, device: torch.device):
    backbone, _meta = load_osf_backbone(cfg["embedding"]["checkpoint_dir"], device)

    m_cfg = dict(cfg["model"])
    m_cfg["num_classes"] = num_classes
    m_cfg["head_type"] = head_type
    sequence_head = build_head({**cfg, "model": m_cfg})

    chunk_bs = cfg["embedding"].get("chunk_batch_size", 16)
    combined = CombinedOSFLoRAModel(backbone, sequence_head, chunk_batch_size=chunk_bs)

    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        target_modules=lora_cfg["target_modules"],
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.0),
        modules_to_save=lora_cfg.get("modules_to_save", ["sequence_head"]),
    )
    peft_model = get_peft_model(combined, lora_config)
    return peft_model.to(device)


def warm_start_head_from_stage1(peft_model, stage1_checkpoint_path: str):
    """Load Stage 1's trained sequence_head weights into the combined
    module's head submodule before LoRA training starts (LP-FT staging —
    see module docstring). Stage 1 checkpoints are plain
    sequence-head-only state dicts (train_osf_context_sweep.py).

    Found live (checklist 2.3 pilot) that peft's `modules_to_save`
    doesn't leave sequence_head as a plain module — it wraps it in a
    `ModulesToSaveWrapper` holding TWO copies: `.original_module` (a
    frozen reference, restored if the adapter is ever disabled) and
    `.modules_to_save["default"]` (the actual trainable copy used during
    forward while the adapter is active). A plain `load_state_dict()`
    call fails with a key-prefix mismatch against the wrapper itself —
    load into both inner copies explicitly instead, so the warm start is
    correct regardless of adapter state.
    """
    stage1_state = torch.load(stage1_checkpoint_path, map_location="cpu", weights_only=False)
    wrapped_head = peft_model.base_model.model.sequence_head
    if hasattr(wrapped_head, "original_module"):
        wrapped_head.original_module.load_state_dict(stage1_state)
        for adapter_module in wrapped_head.modules_to_save.values():
            adapter_module.load_state_dict(stage1_state)
    else:
        wrapped_head.load_state_dict(stage1_state)
    print(f"  Warm-started sequence_head from: {stage1_checkpoint_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Single context-length experiment
# ─────────────────────────────────────────────────────────────────────────────

def train_one_context(
    cfg: dict,
    context_length: str,
    task: str,
    head_type: str,
    out_dir: Path,
    device: torch.device,
    datasets_filter: list,
    stage1_checkpoint: str,
    limit: int,
    max_items: int,
    batch_size: int = 32,
    accum_steps: int = 1,
    exp_id: str = None,
    cli_lr_set: bool = False,
):
    if exp_id is None:
        exp_id = f"{task}_{head_type}"
    t_cfg = cfg["training"]
    N = parse_context_length(context_length)

    # ── Per-context LR override (only when no CLI --lr was given) ──────────
    # Identical logic to train_osf_context_sweep.py's train_one_context —
    # found missing here during checklist 2.5's Stage-1-parity check.
    # phase0_osf_lora_config.yaml sets context_lr_overrides for 120m/240m
    # (5e-5, lower than the 1e-4 default) but nothing applied it before
    # this fix, silently leaving long-context LoRA runs at the wrong LR.
    if not cli_lr_set:
        ctx_lr_overrides = t_cfg.get("context_lr_overrides", {})
        if str(context_length) in ctx_lr_overrides:
            override_lr = float(ctx_lr_overrides[str(context_length)])
            t_cfg["lr"] = override_lr
            print(f"  LR override for {context_length}: {override_lr} (from context_lr_overrides)")

    print(f"\n{'='*60}")
    print(f"Context: {context_length}  ({N} epochs)")
    print(f"  batch_size: {batch_size}  accum_steps: {accum_steps}  "
          f"(effective batch: {batch_size * accum_steps})")
    print(f"{'='*60}")

    out_dir.mkdir(parents=True, exist_ok=True)
    resume_path = out_dir / "resume.pt"
    _resuming = resume_path.exists()

    def make_ds(split):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return OSFRawEpochWindowDataset(
                cfg=cfg, split=split, context_length=context_length, task=task,
                datasets=datasets_filter, limit=limit, max_items=max_items,
            )

    train_ds = make_ds("train")
    val_ds = make_ds("val")
    test_ds = make_ds("test")

    num_classes = train_ds.num_classes
    print(f"  Items — train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")
    print(f"  num_classes: {num_classes}")

    # ── Class weights — identical logic to train_osf_context_sweep.py ──────
    train_labels = np.array([entry[2] for entry in train_ds._index])
    class_weights_cfg = t_cfg.get("class_weights")
    if class_weights_cfg == "auto":
        counts = np.bincount(train_labels, minlength=num_classes).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        w_auto = len(train_labels) / (num_classes * counts)
        w_auto = w_auto / w_auto.sum() * num_classes
        print(f"  Auto class weights: {np.round(w_auto, 3).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(w_auto, dtype=torch.float32, device=device))
    elif class_weights_cfg is not None:
        w = np.array(class_weights_cfg, dtype=float)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    else:
        criterion = nn.CrossEntropyLoss()

    # ── DataLoaders — small num_workers default: raw-signal windows are far
    # larger tensors per item than Stage 1's precomputed embeddings ────────
    num_workers = min(2, max(0, len(train_ds) // 64))
    # SubjectGroupedSampler (checklist 2.5b): keeps each subject's items
    # consecutive so the per-worker single-subject cache in
    # OSFRawEpochWindowDataset actually hits, instead of missing on nearly
    # every item under plain shuffle=True — stacks with the raw-signal
    # cache fix rather than replacing it. sampler and shuffle are mutually
    # exclusive in DataLoader; only the train split needs this (val/test
    # already use shuffle=False, i.e. sequential index order, which is
    # naturally subject-grouped since _build_seq2label_index emits all of
    # one subject's windows together).
    train_sampler = SubjectGroupedSampler(train_ds._index)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                               num_workers=num_workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"))

    # ── Model — build fresh, warm-start head, resume LoRA state if present ─
    model = build_combined_lora_model(cfg, num_classes, head_type, device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    ckpt_path = out_dir / "best_model.pt"

    if _resuming:
        _rckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, _rckpt["peft_state_dict"])
        print(f"  [RESUME] Found checkpoint — continuing from epoch {_rckpt['epoch'] + 1}")
    elif stage1_checkpoint:
        warm_start_head_from_stage1(model, stage1_checkpoint)
    else:
        warnings.warn(
            "No stage1_checkpoint provided and no resume state found — "
            "sequence_head starts from random init, NOT the staged LP-FT "
            "procedure the plan calls for. Only intended for quick "
            "architecture-correctness pilots, not real runs.",
            stacklevel=2,
        )

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(t_cfg["lr"]), weight_decay=float(t_cfg["weight_decay"]),
    )
    epochs = t_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    patience = t_cfg.get("early_stopping_patience", 5)
    monitor = t_cfg.get("early_stopping_monitor", "val_loss")
    monitor_higher_is_better = (monitor != "val_loss")
    monitor_label = monitor.replace("val_", "")

    if _resuming:
        optimizer.load_state_dict(_rckpt["optimizer_state_dict"])
        scheduler.load_state_dict(_rckpt["scheduler_state_dict"])
        best_monitor = _rckpt["best_monitor"]
        no_improve = _rckpt["no_improve"]
        history = _rckpt["history"]
        start_epoch = _rckpt["epoch"] + 1
        del _rckpt
    else:
        best_monitor = float("-inf") if monitor_higher_is_better else float("inf")
        no_improve = 0
        history = []
        start_epoch = 1

    t0 = time.time()
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_logits, train_targets = run_epoch(
            model, train_loader, optimizer, criterion, device, None, train=True,
            accum_steps=accum_steps,
        )
        val_loss, val_logits, val_targets = run_epoch(
            model, val_loader, None, criterion, device, None, train=False,
        )
        scheduler.step()

        if HAS_SKLEARN:
            train_bal_acc = float(balanced_accuracy_score(train_targets, train_logits.argmax(1)))
            val_bal_acc = float(balanced_accuracy_score(val_targets, val_logits.argmax(1)))
        else:
            train_bal_acc = float((train_logits.argmax(1) == train_targets).mean())
            val_bal_acc = float((val_logits.argmax(1) == val_targets).mean())

        val_monitor = compute_monitor_metric(monitor, val_logits, val_targets, val_loss, num_classes)

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "train_bal_acc": train_bal_acc, "val_bal_acc": val_bal_acc,
            f"val_{monitor_label}": val_monitor,
        })

        improved = (val_monitor > best_monitor if monitor_higher_is_better else val_monitor < best_monitor)
        if improved:
            best_monitor = val_monitor
            no_improve = 0
            torch.save(get_peft_model_state_dict(model), ckpt_path)
        else:
            no_improve += 1

        print(
            f"  Epoch {epoch:3d}/{epochs} | loss: train={train_loss:.4f} val={val_loss:.4f} | "
            f"bal_acc: train={train_bal_acc:.3f} val={val_bal_acc:.3f} | "
            f"{monitor_label}: val={val_monitor:.4f} best={best_monitor:.4f}{'*' if improved else ''} | "
            f"patience={no_improve}/{patience}"
        )

        torch.save({
            "epoch": epoch,
            "peft_state_dict": get_peft_model_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_monitor": best_monitor,
            "no_improve": no_improve,
            "history": history,
            "accumulated_time_min": (time.time() - t0) / 60,
        }, resume_path)

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch}.")
            break

    elapsed = time.time() - t0
    print(f"  Training time: {elapsed/60:.1f} min")

    # ── Evaluation on best checkpoint ────────────────────────────────────────
    set_peft_model_state_dict(model, torch.load(ckpt_path, map_location="cpu", weights_only=False))

    _, train_logits, train_targets = run_epoch(model, train_loader, None, criterion, device, None, train=False)
    _, val_logits, val_targets = run_epoch(model, val_loader, None, criterion, device, None, train=False)
    _, test_logits, test_targets = run_epoch(model, test_loader, None, criterion, device, None, train=False)

    train_metrics = compute_metrics(train_logits, train_targets, num_classes, task)
    val_metrics = compute_metrics(val_logits, val_targets, num_classes, task)
    test_metrics = compute_metrics(test_logits, test_targets, num_classes, task)

    metrics = {
        "context_length": context_length, "task": task, "task_type": "seq2label",
        "head_type": head_type, "num_classes": num_classes,
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "early_stopping_monitor": monitor, "best_val_monitor": best_monitor,
        "n_epochs_run": len(history), "training_time_min": elapsed / 60,
        "batch_size": batch_size, "accum_steps": accum_steps,
        "effective_batch_size": batch_size * accum_steps,
        "n_trainable_params": n_trainable,
        "n_total_params": n_total,
        "stage1_checkpoint": stage1_checkpoint,
        "train": train_metrics, "val": val_metrics, "test": test_metrics,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if history:
        import csv
        with open(out_dir / "training_curves.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    resume_path.unlink(missing_ok=True)

    print(f"  Train: {train_metrics}")
    print(f"  Val:   {val_metrics}")
    print(f"  Test:  {test_metrics}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OSF Stage 2 LoRA fine-tuning sweep")
    parser.add_argument("--config", required=True, help="Path to phase0_osf_lora_config.yaml")
    parser.add_argument("--task", default=None, help="Override dataset.task")
    parser.add_argument("--head", default=None, dest="head_type", help="mean_pool | lstm | transformer")
    parser.add_argument("--context", default=None, nargs="+")
    parser.add_argument("--datasets", default=None, nargs="+")
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--max-items", default=None, type=int, dest="max_items")
    parser.add_argument("--stage1-checkpoint", default=None, dest="stage1_checkpoint",
                         help="Path to the matching Stage 1 best_model.pt to warm-start "
                              "the sequence_head from (LP-FT staging). If omitted and no "
                              "resume state exists, head starts from random init — only "
                              "intended for quick pilots, not real runs.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", default=None, type=int, dest="batch_size",
                         help="Micro-batch size. Defaults to 32 if omitted (same convention "
                              "as train_context_sweep.py/train_osf_context_sweep.py: "
                              "'args.batch_size or 32') — NOT assumed to fit at every "
                              "context; if it OOMs, lower --batch-size and raise "
                              "--accum-steps proportionally to keep effective_batch=32.")
    parser.add_argument("--accum-steps", default=1, type=int, dest="accum_steps",
                         help="Gradient accumulation steps (default: 1). effective_batch = "
                              "--batch-size * --accum-steps — set this so effective_batch "
                              "matches the 32 used by Stage 1/SleepFM for a fair comparison "
                              "(e.g. --batch-size 4 --accum-steps 8). Uses run_epoch()'s "
                              "existing accum_steps support (imported from "
                              "train_osf_context_sweep.py), same mechanism as Stage 1.")
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--run-tag", default="", dest="run_tag")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    task = args.task or cfg["dataset"]["task"]
    head_type = args.head_type or cfg["model"]["head_type"]
    _cli_lr_set = args.lr is not None
    if _cli_lr_set:
        cfg["training"]["lr"] = args.lr

    # Same fallback convention as train_context_sweep.py/train_osf_context_sweep.py.
    train_batch_size = args.batch_size or 32

    context_lengths = args.context or cfg["dataset"]["context_lengths"]

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device:          {device}")
    print(f"Task:            {task}")
    print(f"Head:            {head_type}")
    print(f"Context lengths: {context_lengths}")

    results_dir = Path(cfg["logging"]["results_dir"])
    exp_id = f"{task}_{head_type}" + (f"_{args.run_tag}" if args.run_tag else "")
    exp_dir = results_dir / exp_id
    summary_path = exp_dir / "summary.csv"
    exp_dir.mkdir(parents=True, exist_ok=True)

    any_failed = False
    failure_reasons = []

    for ctx in context_lengths:
        ctx_dir = exp_dir / f"context_{ctx}"

        if (ctx_dir / "metrics.json").exists():
            print(f"\n[SKIP] {ctx} — metrics.json already exists.")
            continue

        # Default stage1 checkpoint path if not explicitly given: mirror
        # the Stage 1 experiment folder naming under phase0_osf's results_dir.
        stage1_ckpt = args.stage1_checkpoint
        if stage1_ckpt is None:
            _guess = Path("/scratch/boshra95/psg_full/unified/results/phase0_osf") / f"{task}_{head_type}" / f"context_{ctx}" / "best_model.pt"
            if _guess.exists():
                stage1_ckpt = str(_guess)
                print(f"  Auto-detected Stage 1 checkpoint: {stage1_ckpt}")

        try:
            metrics = train_one_context(
                cfg=cfg, context_length=ctx, task=task, head_type=head_type,
                out_dir=ctx_dir, device=device, datasets_filter=args.datasets,
                stage1_checkpoint=stage1_ckpt, limit=args.limit, max_items=args.max_items,
                batch_size=train_batch_size, accum_steps=args.accum_steps,
                exp_id=exp_id, cli_lr_set=_cli_lr_set,
            )
            if metrics is not None:
                append_to_summary(summary_path, metrics)
        except Exception as exc:
            print(f"\n[ERROR] context={ctx}: {exc}")
            import traceback; traceback.print_exc()
            any_failed = True
            failure_reasons.append(f"{ctx}: {_classify_failure(exc)}")

    print(f"\n{'='*60}")
    print(f"Sweep complete. Results: {exp_dir}")
    if any_failed:
        print("Status: FAILED")
        reason_str = "; ".join(failure_reasons)
        reason_file = exp_dir / f"_failure_reason_{os.environ.get('SLURM_JOB_ID', 'local')}.txt"
        reason_file.write_text(reason_str)
        sys.exit(1)
    else:
        print("Status: SUCCESS")


if __name__ == "__main__":
    main()
