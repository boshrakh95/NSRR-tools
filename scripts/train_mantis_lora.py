#!/usr/bin/env python3
"""
train_mantis_lora.py — Mantis baseline, Stage 2 Step 1 (LoRA fine-tuning)
(checklist 2.4)

Genuinely new end-to-end training script — the Mantis encoder is inside
the trainable graph (LoRA-adapted), so raw signal is loaded and encoded
live every training step, not precomputed once like Stage 1. See
docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md §14 for the full derivation.

NOT a fork of train_mantis_context_sweep.py's file — but it deliberately
IMPORTS that file's low-level, already-tested building blocks (run_epoch,
compute_metrics, compute_monitor_metric, append_to_summary,
_classify_failure) rather than re-implementing them, same reason OSF's/
PhysioOmni's own Stage 2 scripts do: these functions are backbone-agnostic
(run_epoch just calls `model(x, mask)` and doesn't care what's inside
`model`), and duplicating them would risk silent drift between Stage 1 and
Stage 2's metric computation. Only the model construction, checkpoint
format, and warm-start logic are new. Importing train_mantis_context_sweep.py
also registers ITS module-level SIGTERM handler (sys.exit(0) on SIGTERM,
letting bash's trap log TIMEOUT_REQUEUED before SLURM's SIGKILL) — reused
deliberately, exactly the auto-resume behavior
jobs/train_mantis_lora_gpu.sh (not yet written) will need too.

STAGED (LP-FT), NOT JOINT: warm-starts the sequence head from the matching
Stage 1 checkpoint before LoRA fine-tuning begins (Kumar et al. 2022 LP-FT
justification) — an already-made decision, not re-litigated here.

WARM-START SOURCE DIFFERS BY CONTEXT LENGTH (plan §14.5), same design as
OSF's/PhysioOmni's own Stage 2: only **30s** warm-starts from Stage 1's
frozen-backbone head. **Every other context length warm-starts LoRA+head
TOGETHER from this task/head's OWN converged 30s Stage 2 checkpoint** — a
branch, not a 30s→10m→…→240m chain (compute scales ~linearly with context
length; independently LoRA-fine-tuning every length from Stage 1 isn't
achievable in project timeline). Readiness gates on metrics.json existing,
not best_model.pt (the latter is written from epoch 1 — a real
PhysioOmni bug came from trusting it too early, plan §4.10).

MODEL ARCHITECTURE
───────────────────
  CombinedMantisLoRAModel(backbone, sequence_head) is built as ONE
  nn.Module BEFORE peft wrapping (plan §14.2), so peft.get_peft_model() is
  called on the whole combined module with modules_to_save=["sequence_head"]
  — this makes peft's own state-dict save/load cover both the LoRA deltas
  AND the head in one call.

  forward(x, mask): x is raw [B, N, 6, 3840] signal (from
  MantisRawEpochWindowDataset). Every channel of every epoch goes through
  the SAME LoRA-adapted encoder in one batched call per chunk
  (chunk_batch_size counts CHANNEL-EPOCHS, plan §4.4/§4.6 — Mantis's own
  structural advantage over OSF/PhysioOmni: at 30s, micro_batch=32 gives
  OSF 32 items but Mantis 192, entering the compute-bound regime at a much
  shorter context). Absent channel slots (plan §2.2/§14.2's zero-fill
  contract, matching Stage 1 bit-for-bit) are detected directly from the
  cached raw signal rather than threaded through as a separate tensor:
  `load_subject_channels`/`save_signal_cache` already leave an absent
  slot's raw signal at EXACT zero for the whole recording (module
  docstring, mantis_channel_loader.py) — real physiological signal is
  never exactly all-zero across an entire window, so
  `(x[:, :, c] == 0).all(dim=(1,2))` is a safe, reliable per-batch-item
  presence check computed straight from the input already in hand. This
  keeps `forward(x, mask) -> logits` matching `run_epoch()`'s expected
  2-argument signature EXACTLY (no new tensor to plumb through the
  DataLoader/collate/run_epoch chain), which is why run_epoch is reused
  completely unmodified.

  Gradient-checkpointing (plan §4.3/§14.8's memory-mitigation ladder,
  opt-in, default OFF): `checkpoint_tokgen=True` checkpoints just the
  tokenizer conv (the single largest per-channel-epoch activation term at
  low depth, ~1.3% of the FLOPs for ~39% of the memory) —the FIRST rung to
  reach for, not the last. `checkpoint_chunks=True` checkpoints each whole
  chunk's backbone call (coarser, more memory saved, more recompute cost)
  — PhysioOmni's own rung, kept available here too. Both verified
  bit-identical against a non-checkpointed forward on the same input
  (scripts/test_mantis_lora_model.py) before being trusted for a real run,
  matching PhysioOmni's own "max abs diff 0.0" precedent.

USAGE
─────
  python scripts/train_mantis_lora.py --config configs/phase0_mantis_lora_config.yaml \\
      --task apnea_binary --head lstm --context 30s \\
      --stage1-checkpoint /scratch/.../phase0_mantis/apnea_binary_lstm/context_30s/best_model.pt

OUTPUT — same shape as Stage 1's, under phase0_mantis_lora's results_dir:
  {results_dir}/{task}_{head_type}/
    context_{L}/
      best_model.pt   — peft state dict (LoRA deltas + sequence_head), NOT
                         the full 8.11M-param base model (kept frozen,
                         never needs saving — always reloadable from
                         embedding.repo_id/local_dir)
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

# ── local imports ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
from nsrr_tools.datasets.mantis_raw_epoch_dataset import (
    MantisRawEpochWindowDataset,
    SubjectGroupedSampler,
    parse_context_length,
)
from nsrr_tools.datasets.mantis_channel_loader import load_mantis_backbone
from nsrr_tools.models.sequence_head import build_head

# Reused, not duplicated — see module docstring.
sys.path.insert(0, str(_ROOT / "scripts"))
from train_mantis_context_sweep import (  # noqa: E402
    run_epoch,
    compute_metrics,
    compute_monitor_metric,
    append_to_summary,
    _classify_failure,
)

try:
    from sklearn.metrics import balanced_accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Same hand-derived value as extract_mantis_embeddings.py's
# GFLOP_PER_CHANNEL_EPOCH (241 tokens, full depth, forward-only) — kept as
# its own copy rather than imported, matching the project's established
# convention that scripts aren't meant to be used as libraries (see OSF's
# own load_osf_backbone docstring for the same reasoning). Used for
# achieved-TFLOP/s instrumentation (plan §4.1) — the single most expensive
# mistake so far on this project was tuning for weeks around a slowness
# whose cause (TF32 off) was never checked; this makes that impossible to
# skip here too.
GFLOP_PER_CHANNEL_EPOCH = 5.29
H100_TF32_PEAK_TFLOPS = 495.0


# ─────────────────────────────────────────────────────────────────────────────
# Combined model
# ─────────────────────────────────────────────────────────────────────────────

class CombinedMantisLoRAModel(nn.Module):
    """Mantis backbone + sequence head, wrapped as one module BEFORE peft
    injection (plan §14.2) so peft's save/load covers both pieces.

    forward(x, mask) matches train_mantis_context_sweep.py's run_epoch()
    expected signature exactly — x: [B, N, 6, 3840] raw signal, mask:
    [B, N] bool (True=padded) -> logits: [B, num_classes]. This is why
    run_epoch works completely unmodified for Stage 2.
    """

    def __init__(
        self,
        backbone: nn.Module,
        sequence_head: nn.Module,
        chunk_batch_size: int = 192,
        checkpoint_tokgen: bool = False,
        checkpoint_chunks: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.sequence_head = sequence_head
        self.chunk_batch_size = chunk_batch_size
        self.checkpoint_tokgen = checkpoint_tokgen
        self.checkpoint_chunks = checkpoint_chunks

    def _backbone_chunk_forward(self, chunk: torch.Tensor) -> torch.Tensor:
        """One chunk through the backbone, with the two independent,
        opt-in gradient-checkpointing rungs (plan §4.3/§14.8)."""
        if self.checkpoint_tokgen:
            # Checkpoint ONLY the tokenizer conv — the largest single
            # activation term at low depth (~40% of per-channel-epoch
            # memory) for ~1.3% of the FLOPs (plan §4.3) — the cheapest
            # possible memory win, so it's the first rung, not the last.
            tokens = torch_checkpoint(
                self.backbone.tokgen_unit, chunk, use_reentrant=False
            )
            return self.backbone.transf_unit(tokens)
        if self.checkpoint_chunks:
            # Coarser: checkpoint the WHOLE chunk's backbone call
            # (PhysioOmni's own rung) — more memory saved, more recompute.
            return torch_checkpoint(self.backbone, chunk, use_reentrant=False)
        return self.backbone(chunk)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, C, L = x.shape

        # Detect absent channel slots directly from the raw input (see
        # module docstring) — an absent slot is EXACT zero for the whole
        # window by construction (mantis_channel_loader's zero-fill
        # contract), so this needs no extra tensor threaded through
        # run_epoch. Computed once per forward call, cheap relative to the
        # backbone pass itself.
        present = ~(x == 0).all(dim=(1, 3))  # [B, C] bool, True = present

        flat = x.reshape(B * N * C, 1, L)
        outs = []
        for i in range(0, flat.shape[0], self.chunk_batch_size):
            chunk = flat[i : i + self.chunk_batch_size]
            outs.append(self._backbone_chunk_forward(chunk))
        emb = torch.cat(outs, dim=0)          # [B*N*C, D]
        D = emb.shape[-1]
        emb = emb.reshape(B, N, C, D)

        # Zero the absent slots' embeddings AFTER the backbone — matches
        # Stage 1's exact-zero contract bit-for-bit (plan §2.2/§14.2).
        # Feeding zero SIGNAL through a nonlinear encoder does NOT itself
        # produce a zero EMBEDDING, which is why this explicit zeroing
        # step exists rather than relying on the input alone. Gradients
        # through a zeroed slice are exactly zero (multiplying by 0 in the
        # forward graph), so nothing leaks into the backbone's LoRA
        # params from an absent channel — verified directly in
        # scripts/test_mantis_lora_model.py, not just asserted.
        emb = emb * present.to(emb.dtype).view(B, 1, C, 1)
        emb = emb.reshape(B, N, C * D)

        return self.sequence_head(emb, mask)


def build_combined_lora_model(cfg: dict, num_classes: int, head_type: str, device: torch.device):
    emb_cfg = cfg["embedding"]
    checkpoint_source = emb_cfg["local_dir"] if Path(emb_cfg["local_dir"]).exists() else emb_cfg["repo_id"]
    backbone = load_mantis_backbone(
        checkpoint_source,
        seq_len=emb_cfg["seq_len"],
        num_patches=emb_cfg["num_patches"],
        return_transf_layer=emb_cfg["return_transf_layer"],
        output_token=emb_cfg["output_token"],
        device=device,
        pe_mode=emb_cfg.get("pe_mode", "extrapolate"),
    )

    m_cfg = dict(cfg["model"])
    m_cfg["num_classes"] = num_classes
    m_cfg["head_type"] = head_type
    sequence_head = build_head({**cfg, "model": m_cfg})

    chunk_bs = emb_cfg.get("chunk_batch_size", 192)
    checkpoint_tokgen = bool(cfg.get("training", {}).get("checkpoint_tokgen", False))
    checkpoint_chunks = bool(cfg.get("training", {}).get("checkpoint_chunks", False))
    combined = CombinedMantisLoRAModel(
        backbone, sequence_head, chunk_batch_size=chunk_bs,
        checkpoint_tokgen=checkpoint_tokgen, checkpoint_chunks=checkpoint_chunks,
    )

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
    sequence-head-only state dicts (train_mantis_context_sweep.py).

    peft's `modules_to_save` wraps sequence_head in a `ModulesToSaveWrapper`
    holding TWO copies: `.original_module` (frozen reference) and
    `.modules_to_save["default"]` (the trainable copy used during forward
    while the adapter is active). A plain `load_state_dict()` fails on the
    key-prefix mismatch against the wrapper itself — load into both inner
    copies explicitly, exactly as train_osf_lora.py/train_physioomni_lora.py
    already do (plan §14.5).
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


def warm_start_from_stage2_30s(peft_model, stage2_30s_checkpoint_path: str):
    """Load a previously-fine-tuned Stage 2 (LoRA) 30s checkpoint's FULL
    peft state dict (LoRA deltas + sequence_head together) as the starting
    point for fine-tuning at a DIFFERENT context length (plan §14.5).
    Already in peft's own format (produced by get_peft_model_state_dict
    during the 30s run), so set_peft_model_state_dict handles it directly.
    """
    state = torch.load(stage2_30s_checkpoint_path, map_location="cpu", weights_only=False)
    set_peft_model_state_dict(peft_model, state)
    print(f"  Warm-started LoRA+head from Stage 2 30s checkpoint: {stage2_30s_checkpoint_path}")


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
    stage2_30s_checkpoint: str = None,
):
    if exp_id is None:
        exp_id = f"{task}_{head_type}"
    t_cfg = cfg["training"]
    N = parse_context_length(context_length)

    # ── Per-context LR override (only when no CLI --lr was given) ──────────
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
            return MantisRawEpochWindowDataset(
                cfg=cfg, split=split, context_length=context_length, task=task,
                datasets=datasets_filter, limit=limit, max_items=max_items,
            )

    train_ds = make_ds("train")
    val_ds = make_ds("val")
    test_ds = make_ds("test")

    num_classes = train_ds.num_classes
    print(f"  Items — train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")
    print(f"  num_classes: {num_classes}")

    # ── Class weights — identical logic to train_mantis_context_sweep.py ───
    train_labels = np.array([entry[2] for entry in train_ds._index])
    class_weights_cfg = t_cfg.get("class_weights")
    w_auto = None
    if class_weights_cfg == "auto":
        counts = np.bincount(train_labels, minlength=num_classes).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        w_auto = len(train_labels) / (num_classes * counts)
        w_auto = w_auto / w_auto.sum() * num_classes
        print(f"  Auto class weights: {np.round(w_auto, 3).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(w_auto, dtype=torch.float32, device=device))
    elif class_weights_cfg is not None:
        w_auto = np.array(class_weights_cfg, dtype=float)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(w_auto, dtype=torch.float32, device=device))
    else:
        criterion = nn.CrossEntropyLoss()

    # ── DataLoaders — small num_workers default: raw-signal windows are far
    # larger tensors per item than Stage 1's precomputed embeddings ────────
    num_workers = min(2, max(0, len(train_ds) // 64))

    use_weighted_sampler = t_cfg.get("weighted_sampler", False) and w_auto is not None
    if use_weighted_sampler:
        sample_weights = torch.tensor(w_auto[train_labels], dtype=torch.float32)
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        print(f"  WeightedRandomSampler: enabled")
    else:
        # SubjectGroupedSampler: keeps each subject's items consecutive.
        # See mantis_raw_epoch_dataset.py's module docstring for why this
        # doesn't buy a materialized-array cache hit here (unlike OSF's
        # dataset) — kept anyway as the established per-project convention
        # and for the shape-cache dict-lookup locality it still provides.
        train_sampler = SubjectGroupedSampler(train_ds._index)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                               num_workers=num_workers, pin_memory=(device.type == "cuda"),
                               persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=(device.type == "cuda"),
                             persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"),
                              persistent_workers=(num_workers > 0))

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
    elif stage2_30s_checkpoint:
        warm_start_from_stage2_30s(model, stage2_30s_checkpoint)
    elif stage1_checkpoint:
        warm_start_head_from_stage1(model, stage1_checkpoint)
    else:
        warnings.warn(
            "No stage1_checkpoint or stage2_30s_checkpoint provided and no "
            "resume state found — sequence_head starts from random init, "
            "NOT the staged LP-FT procedure the plan calls for. Only "
            "intended for quick architecture-correctness pilots, not real "
            "runs.",
            stacklevel=2,
        )

    # ── Optimizer & scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(t_cfg["lr"]), weight_decay=float(t_cfg["weight_decay"]),
    )
    epochs = t_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # TF32, not autocast/fp16 (plan §4.2) — scaler always None. Matches
    # train_mantis_context_sweep.py's Stage 1 invariant exactly.
    scaler = None

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

    # Achieved-TFLOP/s (plan §4.1): fwd+bwd over every channel-epoch the
    # backbone actually processes per training epoch (unlike Stage 1,
    # Stage 2's cost is dominated by backbone compute, not the head).
    flops_per_channel_epoch = GFLOP_PER_CHANNEL_EPOCH * 1e9

    t0 = time.time()
    for epoch in range(start_epoch, epochs + 1):
        _t_epoch_start = time.time()
        train_loss, train_logits, train_targets = run_epoch(
            model, train_loader, optimizer, criterion, device, scaler, train=True,
            accum_steps=accum_steps,
        )
        _t_epoch_train = time.time() - _t_epoch_start
        val_loss, val_logits, val_targets = run_epoch(
            model, val_loader, None, criterion, device, None, train=False,
        )
        scheduler.step()

        if device.type == "cuda" and _t_epoch_train > 0:
            _channel_epochs = len(train_ds) * N * 6  # items x epochs/window x channels
            _achieved_flops = _channel_epochs * flops_per_channel_epoch * 3  # fwd+bwd
            _achieved_tflops = _achieved_flops / _t_epoch_train / 1e12
            _pct_of_peak = _achieved_tflops / H100_TF32_PEAK_TFLOPS * 100
        else:
            _achieved_tflops = float("nan")
            _pct_of_peak = float("nan")

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
            "achieved_TFLOPs": _achieved_tflops,
            "pct_of_h100_tf32_peak": _pct_of_peak,
        })

        improved = (val_monitor > best_monitor if monitor_higher_is_better else val_monitor < best_monitor)
        if improved:
            best_monitor = val_monitor
            no_improve = 0
            torch.save(get_peft_model_state_dict(model), ckpt_path)
        elif not ckpt_path.exists():
            # Safety net: same fix as train_mantis_context_sweep.py's own
            # (checklist 1.9) — if the monitor is NaN for every epoch so
            # far (degenerate val split), never leave best_model.pt
            # unwritten, which would otherwise crash the final evaluation
            # below with FileNotFoundError. Does not reset patience.
            torch.save(get_peft_model_state_dict(model), ckpt_path)
        else:
            no_improve += 1

        _tflop_str = (
            f"achieved={_achieved_tflops:.4f} TFLOP/s ({_pct_of_peak:.2f}% of H100 TF32 peak)"
            if device.type == "cuda" else "achieved=N/A (not CUDA)"
        )
        print(
            f"  Epoch {epoch:3d}/{epochs} | loss: train={train_loss:.4f} val={val_loss:.4f} | "
            f"bal_acc: train={train_bal_acc:.3f} val={val_bal_acc:.3f} | "
            f"{monitor_label}: val={val_monitor:.4f} best={best_monitor:.4f}{'*' if improved else ''} | "
            f"patience={no_improve}/{patience} | {_tflop_str}"
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

    _tflop_values = [h["achieved_TFLOPs"] for h in history
                     if not (isinstance(h["achieved_TFLOPs"], float) and np.isnan(h["achieved_TFLOPs"]))]
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
        "stage2_30s_checkpoint": stage2_30s_checkpoint,
        "achieved_TFLOPs_mean": float(np.mean(_tflop_values)) if _tflop_values else None,
        "pct_of_h100_tf32_peak_mean": (float(np.mean(_tflop_values)) / H100_TF32_PEAK_TFLOPS * 100) if _tflop_values else None,
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
    if metrics["achieved_TFLOPs_mean"] is not None:
        print(f"  Achieved TFLOP/s (mean): {metrics['achieved_TFLOPs_mean']:.4f} "
              f"({metrics['pct_of_h100_tf32_peak_mean']:.2f}% of H100 TF32 peak)")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mantis Stage 2 LoRA fine-tuning sweep")
    parser.add_argument("--config", required=True, help="Path to phase0_mantis_lora_config.yaml")
    parser.add_argument("--task", default=None, help="Override dataset.task")
    parser.add_argument("--head", default=None, dest="head_type", help="mean_pool | lstm | transformer")
    parser.add_argument("--context", default=None, nargs="+")
    parser.add_argument("--datasets", default=None, nargs="+")
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--max-items", default=None, type=int, dest="max_items")
    parser.add_argument("--stage1-checkpoint", default=None, dest="stage1_checkpoint",
                         help="Path to the matching Stage 1 best_model.pt to warm-start "
                              "the sequence_head from (LP-FT staging). Only used for the "
                              "30s context by default (plan §14.5) — for other contexts, "
                              "passing this explicitly OVERRIDES the default "
                              "warm-start-from-30s-LoRA-checkpoint behavior. If omitted "
                              "and no resume/stage2-30s state exists, head starts from "
                              "random init — only intended for quick pilots, not real runs.")
    parser.add_argument("--stage2-30s-checkpoint", default=None, dest="stage2_30s_checkpoint",
                         help="Path to this task/head's own Stage 2 (LoRA) 30s best_model.pt "
                              "to warm-start OTHER context lengths from (plan §14.5). "
                              "Auto-detected by default at "
                              "{results_dir}/{task}_{head}/context_30s/best_model.pt — "
                              "always the plain, untagged path, regardless of --run-tag. "
                              "Never used for the 30s context itself.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", default=None, type=int, dest="batch_size",
                         help="Micro-batch size. Defaults to 32 if omitted, same convention "
                              "as train_mantis_context_sweep.py — NOT assumed to fit at every "
                              "context; if it OOMs, lower --batch-size and raise "
                              "--accum-steps proportionally to keep effective_batch=32.")
    parser.add_argument("--accum-steps", default=1, type=int, dest="accum_steps",
                         help="Gradient accumulation steps. effective_batch = "
                              "--batch-size * --accum-steps.")
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--run-tag", default="", dest="run_tag")
    args = parser.parse_args()

    # TF32, not autocast/fp16 (plan §4.2) — must happen before any CUDA matmul.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    task = args.task or cfg["dataset"]["task"]
    head_type = args.head_type or cfg["model"]["head_type"]
    _cli_lr_set = args.lr is not None
    if _cli_lr_set:
        cfg["training"]["lr"] = args.lr

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

    # Stage 1's results_dir — needed to auto-detect the 30s warm-start
    # checkpoint. Derived from the LoRA config's stage1_embedding_dir
    # sibling convention: phase0_mantis_lora's own config points at
    # phase0_mantis's embeddings; its results live in the sibling
    # phase0_mantis results_dir (not read directly from this config, since
    # Stage 1's config isn't loaded here — matches the same pattern OSF's
    # own script uses, a hardcoded sibling path rather than cross-loading
    # two config files).
    stage1_results_dir = Path(str(results_dir).replace("phase0_mantis_lora", "phase0_mantis"))

    any_failed = False
    failure_reasons = []

    for ctx in context_lengths:
        ctx_dir = exp_dir / f"context_{ctx}"

        if (ctx_dir / "metrics.json").exists():
            print(f"\n[SKIP] {ctx} — metrics.json already exists.")
            continue

        # Warm-start source selection (plan §14.5) — 30s always warm-starts
        # from Stage 1; every other context length prefers this task/head's
        # OWN 30s Stage 2 LoRA checkpoint. Readiness gates on metrics.json,
        # not best_model.pt (the latter is written from epoch 1 — a real
        # PhysioOmni bug came from trusting it too early, plan §4.10).
        stage1_ckpt = args.stage1_checkpoint
        stage2_30s_ckpt = None

        if str(ctx) == "30s":
            if stage1_ckpt is None:
                _guess = stage1_results_dir / f"{task}_{head_type}" / "context_30s" / "best_model.pt"
                if _guess.exists():
                    stage1_ckpt = str(_guess)
                    print(f"  Auto-detected Stage 1 checkpoint: {stage1_ckpt}")
        else:
            stage2_30s_ckpt = args.stage2_30s_checkpoint
            if stage2_30s_ckpt is None:
                _guess_dir = results_dir / f"{task}_{head_type}" / "context_30s"
                if (_guess_dir / "metrics.json").exists():
                    stage2_30s_ckpt = str(_guess_dir / "best_model.pt")
                    print(f"  Auto-detected Stage 2 30s checkpoint: {stage2_30s_ckpt}")

            if stage2_30s_ckpt is None and stage1_ckpt is None:
                print(
                    f"\n[ERROR] context={ctx}: no converged Stage 2 30s checkpoint found at "
                    f"{results_dir / f'{task}_{head_type}' / 'context_30s'} "
                    f"(checked for metrics.json, not just best_model.pt) and no "
                    f"--stage1-checkpoint override given. Run the 30s context for "
                    f"this (task, head) first, or pass --stage2-30s-checkpoint / "
                    f"--stage1-checkpoint explicitly to override."
                )
                any_failed = True
                failure_reasons.append(f"{ctx}: no_stage2_30s_checkpoint_and_no_override")
                continue

        try:
            metrics = train_one_context(
                cfg=cfg, context_length=ctx, task=task, head_type=head_type,
                out_dir=ctx_dir, device=device, datasets_filter=args.datasets,
                stage1_checkpoint=stage1_ckpt, limit=args.limit, max_items=args.max_items,
                batch_size=train_batch_size, accum_steps=args.accum_steps,
                exp_id=exp_id, cli_lr_set=_cli_lr_set,
                stage2_30s_checkpoint=stage2_30s_ckpt,
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
