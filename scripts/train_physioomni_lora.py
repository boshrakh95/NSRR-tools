#!/usr/bin/env python3
"""
train_physioomni_lora.py — PhysioOmni baseline, Stage 2 Step 1 (LoRA fine-tuning)

Genuinely new end-to-end training script — PhysioOmni's 4 encoders are
inside the trainable graph (LoRA-adapted), so raw signal is loaded and
encoded live every training step, not precomputed once like Stage 1. See
docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md §15 for the full design
derivation (live-verified against the real checkpoint before being
written down, not just reasoned about).

NOT a fork of train_physioomni_context_sweep.py's file — but it
deliberately IMPORTS that file's low-level, already-tested building blocks
(run_epoch, compute_metrics, compute_monitor_metric, append_to_summary,
_classify_failure) rather than re-implementing them, same reuse pattern as
OSF's train_osf_lora.py: these functions are backbone-agnostic (run_epoch
just calls `model(x, mask)` and doesn't care what's inside `model`), and
duplicating them would risk silent drift between Stage 1 and Stage 2's
metric computation.

**Important call-signature adaptation**: run_epoch() does
`x = x.to(device, non_blocking=True)` and
`total_loss += loss.item() * x.size(0)` on whatever the DataLoader yields
as `x` — it has no idea `x` isn't a plain tensor. Stage 2's raw-signal
batches are NOT a single tensor (§15.2/15.4 — different subjects have
different modality-presence patterns) — so
physioomni_raw_epoch_dataset.py's collate_fn wraps the per-modality-grouped
structure in `PhysioOmniLoRABatch`, a tiny class implementing exactly
`.to(device)` and `.size(0)` and nothing else, and yields the standard
`(x, mask, y)` 3-tuple `run_epoch()` already unpacks. `model(x, mask)`
then calls `CombinedPhysioOmniLoRAModel.forward(batch, mask)` below with
`batch` = the `PhysioOmniLoRABatch` and `mask` = a plain BoolTensor[B,N] —
no change needed to run_epoch() itself.

STAGED (LP-FT), NOT JOINT: warm-starts the sequence head from the
matching Stage 1 checkpoint before LoRA fine-tuning begins — same
Kumar et al. 2022 LP-FT justification already established for OSF/SleepFM
(CLAUDE.md's "Frozen vs. LoRA-fine-tuned conditions").

WARM-START SOURCE DIFFERS BY CONTEXT LENGTH (plan §15.6, direct reuse of
OSF's own resolved design): only **30s** warm-starts from Stage 1
(unchanged). **Every other context length warm-starts from this task/
head's OWN 30s LoRA fine-tune** (`results_dir/{task}_{head}/context_30s/
best_model.pt` — always the plain, untagged path, regardless of the
current run's --run-tag).

MODEL ARCHITECTURE (plan §15.1/15.2/15.5)
───────────────────────────────────────────
  CombinedPhysioOmniLoRAModel holds all 4 encoders (self.eeg_encoder,
  self.eog_encoder, self.ecg_encoder, self.emg_encoder) + self.sequence_head
  as submodules, built as ONE nn.Module BEFORE peft wrapping, exactly like
  OSF's CombinedOSFLoRAModel pattern extended to 4 backbones. A SINGLE
  peft.get_peft_model() call over this combined module correctly finds and
  wraps all 4x12=48 attention blocks (96 c_attn/c_proj Linear layers) —
  live-verified 2026-08-19 against the real checkpoint (plan §15.1), not
  assumed.

  forward(batch): batch is physioomni_lora_collate_fn's output — a dict
  grouping each modality's present subjects by (modality, channel-count)
  (EEG only; EOG/ECG/EMG are always exactly 1 channel when present). For
  each group, runs that modality's LoRA-adapted encoder over the group's
  raw signal (chunk-batched exactly like Stage 1's chunk_batch_size
  pattern — epochs as the batch dimension, matching PhysioOmni's
  positional-embedding tables, which are sized for one epoch's own token
  layout, never a multi-epoch sequence — plan §15.5), scatters the
  resulting CLS outputs into the correct rows/columns of a [B,N,500]
  embedding tensor (present-mask semantics — a genuinely absent modality's
  slice stays exactly zero, no encoder forward pass run for it, identical
  contract to Stage 1's embeddings — plan §15.2), then feeds the sequence
  head.

  The per-modality patch-reshape + position-ID construction logic below is
  a DELIBERATE DUPLICATE of extract_physioomni_embeddings.py's
  _modality_forward() (not imported) — that function hardcodes
  `with torch.no_grad():`, wrong for Stage 2 where LoRA gradients must
  flow through it. Small (~20 line), single-purpose function, duplicated
  rather than factored into the shared channel-loader module — same
  precedent OSF's train_osf_lora.py already established
  (load_osf_backbone()'s own docstring) — and deliberately avoids touching
  extract_physioomni_embeddings.py at all, an already-verified,
  already-in-production script actively feeding real completed Stage 1
  training runs.

USAGE
─────
  python scripts/train_physioomni_lora.py --config configs/phase0_physioomni_lora_config.yaml \\
      --task sex_binary --head lstm --context 30s \\
      --stage1-checkpoint /scratch/.../phase0_physioomni/sex_binary_lstm/context_30s/best_model.pt

OUTPUT — same shape as Stage 1's, under phase0_physioomni_lora's results_dir:
  {results_dir}/{task}_{head_type}/
    context_{L}/
      best_model.pt   — peft state dict (LoRA deltas + sequence_head), NOT
                         the full ~13.9M-param base encoders (kept frozen,
                         never needs saving — always reloadable from
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
from torch.utils.checkpoint import checkpoint
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

# ── local imports ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
from nsrr_tools.datasets.physioomni_raw_epoch_dataset import (
    PhysioOmniRawEpochWindowDataset,
    SubjectGroupedSampler,
    physioomni_lora_collate_fn,
    parse_context_length,
)
from nsrr_tools.datasets.physioomni_channel_loader import NATIVE_HZ, PATCH_SAMPLES
from nsrr_tools.models.sequence_head import build_head

# Reused, not duplicated — see module docstring. Side effect worth knowing:
# importing train_physioomni_context_sweep.py also registers ITS module-
# level SIGTERM handler, exactly the auto-resume behavior
# jobs/train_physioomni_lora_gpu.sh needs too — reused deliberately, not
# accidentally inherited (same note OSF's train_osf_lora.py makes).
sys.path.insert(0, str(_ROOT / "scripts"))
from train_physioomni_context_sweep import (  # noqa: E402
    run_epoch,
    compute_metrics,
    compute_monitor_metric,
    append_to_summary,
    _classify_failure,
)

# ── PhysioOmni repo import ──────────────────────────────────────────────────
_PHYSIOOMNI_REPO = _ROOT.parent / "PhysioOmni"
sys.path.insert(0, str(_PHYSIOOMNI_REPO))
try:
    from model.neural_transformer import NeuralTransformer, NTConfig
    from dataset import standard_1020
except ImportError as e:
    print(f"Cannot import PhysioOmni: {e}\nExpected repo at: {_PHYSIOOMNI_REPO}\nRun with physioomni_env.")
    sys.exit(1)

try:
    from sklearn.metrics import balanced_accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ── TF32 tensor cores (2026-08-22) ──────────────────────────────────────────
# PyTorch 2.5 defaults BOTH of these OFF (allow_tf32=False,
# float32_matmul_precision="highest"), so every matmul in the 4 encoders +
# sequence head was running at true FP32 on an H100 — verified live in
# physioomni_env, and visible in the training logs as `dtype=float32`.
# H100 throughput: FP32 non-tensor ~67 TFLOPS vs TF32 tensor core ~495
# TFLOPS. TF32 keeps FP32's exponent range and reduces only mantissa bits
# (10 vs 23) on the matmul inputs, accumulating in FP32 — the standard
# A100/H100 default for deep learning, and unlike autocast it needs no
# GradScaler, changes no stored dtype, and leaves every checkpoint
# byte-compatible. Deliberately NOT torch.autocast here: autocast on CUDA
# defaults to float16 (not bfloat16), which this 12-layer transformer has
# not been validated under — TF32 is the free, low-risk rung.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

MODALITIES = ["EEG", "EOG", "ECG", "EMG"]
MODALITY_DIM = {"EEG": 200, "EOG": 100, "ECG": 100, "EMG": 100}
FLAT_DIM = sum(MODALITY_DIM.values())  # 500
SLOT_RANGE = {}
_off = 0
for _m in MODALITIES:
    SLOT_RANGE[_m] = (_off, _off + MODALITY_DIM[_m])
    _off += MODALITY_DIM[_m]


def _checkpointed_encoder_forward(encoder, chunk, chan_ids, time_ids):
    """Module-level (not a closure) so torch.utils.checkpoint.checkpoint()
    re-runs exactly this call during backward — mask/return_all_tokens are
    fixed, non-tensor arguments, kept out of checkpoint's own arg list
    (plan §15.8's gradient-checkpointing mitigation, see
    CombinedPhysioOmniLoRAModel._run_group)."""
    return encoder.forward_features(chunk, chan_ids, time_ids, mask=None, return_all_tokens=False)


# ─────────────────────────────────────────────────────────────────────────────
# Combined model (plan §15.1/15.2/15.5)
# ─────────────────────────────────────────────────────────────────────────────

class CombinedPhysioOmniLoRAModel(nn.Module):
    """4 PhysioOmni encoders + sequence head, wrapped as one module BEFORE
    peft injection (plan §15.1) so peft's save/load covers all of it.

    forward(batch, mask) matches train_physioomni_context_sweep.py's
    run_epoch() `model(x, mask)` call exactly — `batch` is a
    PhysioOmniLoRABatch (already .to(device)'d by run_epoch before this is
    called), `mask` is a plain BoolTensor[B,N]. See module docstring for
    the full reasoning.
    """

    def __init__(self, encoders: dict, sequence_head: nn.Module, chunk_batch_size: int = 16,
                 use_gradient_checkpointing: bool = False):
        super().__init__()
        self.eeg_encoder = encoders["EEG"]
        self.eog_encoder = encoders["EOG"]
        self.ecg_encoder = encoders["ECG"]
        self.emg_encoder = encoders["EMG"]
        self.sequence_head = sequence_head
        self.chunk_batch_size = chunk_batch_size
        # OPT-IN, default OFF (plan §15.8) — trades compute for memory (an
        # extra forward pass per chunk during backward), so it's only
        # worth paying for contexts that actually need the memory
        # headroom. Prefer a larger GPU allocation first (zero speed
        # cost, MIG partitions scale compute with memory too) — this is
        # the fallback for whichever context still doesn't fit.
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def _encoder_for(self, modality: str) -> nn.Module:
        return {
            "EEG": self.eeg_encoder, "EOG": self.eog_encoder,
            "ECG": self.ecg_encoder, "EMG": self.emg_encoder,
        }[modality]

    def _run_group(self, modality: str, x: torch.Tensor, labels: list, device) -> torch.Tensor:
        """x: [k, n_chans, N, epoch_samples] raw continuous per-epoch signal
        for the k subjects that have this modality (this exact channel
        count, for EEG). Returns CLS output [k, N, MODALITY_DIM[modality]].

        Reshape/position-ID logic deliberately mirrors
        extract_physioomni_embeddings.py's _modality_forward() exactly (see
        module docstring for why it's duplicated, not imported) — channel-
        major token order (all of channel 0's patches, then channel 1's,
        ...), matching how chan_ids/time_ids are built below.
        """
        k, n_chans, N, epoch_samples = x.shape
        patch_samples = PATCH_SAMPLES[modality]
        patches_per_epoch = epoch_samples // patch_samples
        encoder = self._encoder_for(modality)

        # x is already device-resident (PhysioOmniLoRABatch.to() moved it
        # before forward() was called) — no .to(device) needed here.
        x = x.permute(0, 2, 1, 3)  # [k, N, n_chans, epoch_samples]
        x = x.reshape(k * N, n_chans, patches_per_epoch, patch_samples)
        x = x.reshape(k * N, n_chans * patches_per_epoch, patch_samples)

        chan_ids, time_ids = [], []
        for label in labels:
            pos_id = standard_1020.index(label)
            chan_ids.extend([pos_id] * patches_per_epoch)
            time_ids.extend(range(patches_per_epoch))
        chan_ids_t = torch.tensor(chan_ids, dtype=torch.long, device=device)
        time_ids_t = torch.tensor(time_ids, dtype=torch.long, device=device)

        # Gradient checkpointing (plan §15.8) — OPT-IN, default OFF (see
        # __init__). Only engages when explicitly requested AND there's
        # more than one chunk to begin with AND we're training (eval has
        # no backward pass to recompute for). See __init__'s docstring for
        # why this isn't automatic: it trades compute for memory, so a
        # larger GPU allocation (zero speed cost) is the preferred fix
        # whenever it's sufficient — this is the fallback for whichever
        # context still doesn't fit even at the largest GPU size available.
        n_items = x.shape[0]
        use_checkpoint = (
            self.use_gradient_checkpointing
            and self.training
            and n_items > self.chunk_batch_size
        )

        cls_chunks = []
        for i in range(0, n_items, self.chunk_batch_size):
            chunk = x[i : i + self.chunk_batch_size]
            n = chunk.shape[0]
            ic = chan_ids_t.unsqueeze(0).expand(n, -1)
            it = time_ids_t.unsqueeze(0).expand(n, -1)
            if use_checkpoint:
                cls = checkpoint(_checkpointed_encoder_forward, encoder, chunk, ic, it,
                                  use_reentrant=False)
            else:
                cls = encoder.forward_features(chunk, ic, it, mask=None, return_all_tokens=False)
            cls_chunks.append(cls)
        cls_all = torch.cat(cls_chunks, dim=0)  # [k*N, modality_dim]
        return cls_all.reshape(k, N, -1)

    def forward(self, batch, mask: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        B = batch.size(0)
        N = mask.shape[1]
        emb = torch.zeros(B, N, FLAT_DIM, device=device, dtype=torch.float32)

        for n_chans, group in batch.eeg.items():
            cls = self._run_group("EEG", group["x"], group["labels"], device)
            s, e = SLOT_RANGE["EEG"]
            emb[group["batch_idx"], :, s:e] = cls

        for modality, group in (("EOG", batch.eog), ("ECG", batch.ecg), ("EMG", batch.emg)):
            if group is None:
                continue
            cls = self._run_group(modality, group["x"], group["labels"], device)
            s, e = SLOT_RANGE[modality]
            emb[group["batch_idx"], :, s:e] = cls

        return self.sequence_head(emb, mask)


def load_physioomni_encoders(checkpoint_path: str, device: torch.device) -> dict:
    """Load all 4 base (not yet LoRA-wrapped) PhysioOmni encoders — same
    loading logic as extract_physioomni_embeddings.py's load_models(),
    duplicated rather than imported (small, single-purpose function — same
    precedent as OSF's load_osf_backbone())."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_sd = ckpt["model"]
    encoders = {}
    for mod in MODALITIES:
        conf = NTConfig(**ckpt[f"{mod}_encoder_args"])
        enc = NeuralTransformer(conf)
        prefix = f"{mod}_encoder."
        filtered = {k[len(prefix):]: v for k, v in model_sd.items() if k.startswith(prefix)}
        missing, _unexpected = enc.load_state_dict(filtered, strict=False)
        if missing:
            raise RuntimeError(f"{mod} encoder has missing keys: {missing}")
        enc.to(device)
        encoders[mod] = enc
    return encoders


def build_combined_lora_model(cfg: dict, num_classes: int, head_type: str, device: torch.device,
                               gradient_checkpointing: bool = False):
    encoders = load_physioomni_encoders(cfg["embedding"]["checkpoint_dir"], device)

    m_cfg = dict(cfg["model"])
    m_cfg["num_classes"] = num_classes
    m_cfg["head_type"] = head_type
    sequence_head = build_head({**cfg, "model": m_cfg})

    chunk_bs = cfg["embedding"].get("chunk_batch_size", 16)
    combined = CombinedPhysioOmniLoRAModel(encoders, sequence_head, chunk_batch_size=chunk_bs,
                                            use_gradient_checkpointing=gradient_checkpointing)

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
    module's head submodule before LoRA training starts (LP-FT staging).
    Same ModulesToSaveWrapper two-copy handling OSF's checklist 2.3
    already found and fixed — peft's `modules_to_save` wraps sequence_head
    in a wrapper holding `.original_module` (frozen reference) and
    `.modules_to_save["default"]` (the trainable copy actually used during
    the forward pass while the adapter is active); a plain
    `load_state_dict()` fails with a key-prefix mismatch against the
    wrapper itself."""
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
    point for fine-tuning at a DIFFERENT context length (plan §15.6 —
    compute scales ~linearly with context length, likely worse here than
    OSF given up to 4 encoder forward passes per epoch instead of 1)."""
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
    gradient_checkpointing: bool = False,
    num_workers: int = None,
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
            return PhysioOmniRawEpochWindowDataset(
                cfg=cfg, split=split, context_length=context_length, task=task,
                datasets=datasets_filter, limit=limit, max_items=max_items,
            )

    train_ds = make_ds("train")
    val_ds = make_ds("val")
    test_ds = make_ds("test")

    num_classes = train_ds.num_classes
    print(f"  Items — train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")
    print(f"  num_classes: {num_classes}")

    # ── Class weights — identical logic to train_physioomni_context_sweep.py ──
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

    # ── DataLoaders ─────────────────────────────────────────────────────────
    # Raised from the old hard cap of 2 (2026-08-22). Two things changed:
    #
    #  1. Per-subject RAM dropped from ~174 MB (the old full-night float32
    #     materialization) to just the live window, because
    #     load_signal_cache now returns lazy _NpySliceReader handles. The
    #     old cap of 2 existed because of that ~174 MB/worker footprint.
    #  2. Measured on Lustre, this I/O is PER-OPERATION LATENCY bound
    #     (~20 ms/open, ~12 ms/read-op — see load_signal_cache's docstring
    #     for the full table), not bandwidth bound. Latency-bound I/O
    #     parallelizes close to linearly, so workers are the real lever.
    #
    # Honest scope: this matters at SHORT contexts (I/O is ~50% of a 30s
    # epoch) and is nearly irrelevant at long ones (~3.5% of an 80m epoch,
    # which is ~96% compute). Capped by the job's actual CPU allocation —
    # SLURM_CPUS_PER_TASK, minus one for the main process — so it can never
    # oversubscribe a job that asked for fewer cores. Override with
    # --num-workers.
    if num_workers is None:
        _cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0) or os.cpu_count() or 2)
        num_workers = max(0, min(_cpus - 1, len(train_ds) // 64, 8))
    print(f"  DataLoader workers: {num_workers}")

    use_weighted_sampler = t_cfg.get("weighted_sampler", False) and w_auto is not None
    if use_weighted_sampler:
        sample_weights = torch.tensor(w_auto[train_labels], dtype=torch.float32)
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        print(f"  WeightedRandomSampler: enabled")
    else:
        # SubjectGroupedSampler: keeps each subject's items consecutive so
        # the per-worker single-subject cache in
        # PhysioOmniRawEpochWindowDataset actually hits.
        train_sampler = SubjectGroupedSampler(train_ds._index)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler,
                               num_workers=num_workers, pin_memory=(device.type == "cuda"),
                               persistent_workers=(num_workers > 0),
                               collate_fn=physioomni_lora_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=(device.type == "cuda"),
                             persistent_workers=(num_workers > 0),
                             collate_fn=physioomni_lora_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type == "cuda"),
                              persistent_workers=(num_workers > 0),
                              collate_fn=physioomni_lora_collate_fn)

    # ── Model — build fresh, warm-start head, resume LoRA state if present ─
    model = build_combined_lora_model(cfg, num_classes, head_type, device,
                                       gradient_checkpointing=gradient_checkpointing)
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

    use_amp = t_cfg.get("mixed_precision", False) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

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
            model, train_loader, optimizer, criterion, device, scaler, train=True,
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
        "stage2_30s_checkpoint": stage2_30s_checkpoint,
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
    parser = argparse.ArgumentParser(description="PhysioOmni Stage 2 LoRA fine-tuning sweep")
    parser.add_argument("--config", required=True, help="Path to phase0_physioomni_lora_config.yaml")
    parser.add_argument("--task", default=None, help="Override dataset.task")
    parser.add_argument("--head", default=None, dest="head_type", help="lstm | transformer (mean_pool deferred, plan §15.10)")
    parser.add_argument("--context", default=None, nargs="+")
    parser.add_argument("--datasets", default=None, nargs="+")
    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--max-items", default=None, type=int, dest="max_items")
    parser.add_argument("--stage1-checkpoint", default=None, dest="stage1_checkpoint",
                         help="Path to the matching Stage 1 best_model.pt to warm-start "
                              "the sequence_head from (LP-FT staging). Only used for the "
                              "30s context by default — for other contexts, passing this "
                              "explicitly OVERRIDES the default warm-start-from-30s-LoRA "
                              "behavior. If omitted and no resume/stage2-30s state exists, "
                              "head starts from random init — only for quick pilots.")
    parser.add_argument("--stage2-30s-checkpoint", default=None, dest="stage2_30s_checkpoint",
                         help="Path to this task/head's own Stage 2 (LoRA) 30s best_model.pt "
                              "to warm-start OTHER context lengths from. Auto-detected by "
                              "default at {results_dir}/{task}_{head}/context_30s/best_model.pt "
                              "— always the plain, untagged path, regardless of --run-tag.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", default=None, type=int, dest="batch_size",
                         help="Micro-batch size. Defaults to 32 if omitted — NOT assumed to "
                              "fit at every context; if it OOMs, lower --batch-size and raise "
                              "--accum-steps proportionally to keep effective_batch=32.")
    parser.add_argument("--accum-steps", default=1, type=int, dest="accum_steps")
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--run-tag", default="", dest="run_tag")
    parser.add_argument("--gradient-checkpointing", action="store_true", dest="gradient_checkpointing",
                         help="Trade compute for memory (plan §15.8) — recompute each encoder "
                              "chunk's forward pass during backward instead of storing it. OPT-IN, "
                              "default OFF: only worth it for a context that still OOMs at "
                              "micro_batch=1 on the largest available GPU — a larger GPU allocation "
                              "is the preferred fix first (zero speed cost, unlike this flag).")
    parser.add_argument("--num-workers", default=None, type=int, dest="num_workers",
                         help="DataLoader worker processes. Default: SLURM_CPUS_PER_TASK-1, "
                              "capped at 8. Raise --cpus-per-task alongside this — the cache "
                              "read is Lustre-latency-bound and parallelizes near-linearly, "
                              "but only matters much at short contexts (see "
                              "load_signal_cache's docstring for the measured breakdown).")
    args = parser.parse_args()

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

    any_failed = False
    failure_reasons = []

    for ctx in context_lengths:
        ctx_dir = exp_dir / f"context_{ctx}"

        if (ctx_dir / "metrics.json").exists():
            print(f"\n[SKIP] {ctx} — metrics.json already exists.")
            continue

        stage1_ckpt = args.stage1_checkpoint
        stage2_30s_ckpt = None

        if str(ctx) == "30s":
            if stage1_ckpt is None:
                _guess = Path("/scratch/boshra95/psg/unified/results/phase0_physioomni") / f"{task}_{head_type}" / f"context_{ctx}" / "best_model.pt"
                if _guess.exists():
                    stage1_ckpt = str(_guess)
                    print(f"  Auto-detected Stage 1 checkpoint: {stage1_ckpt}")
        else:
            stage2_30s_ckpt = args.stage2_30s_checkpoint
            if stage2_30s_ckpt is None:
                _guess2 = results_dir / f"{task}_{head_type}" / "context_30s" / "best_model.pt"
                if _guess2.exists():
                    stage2_30s_ckpt = str(_guess2)
                    print(f"  Auto-detected Stage 2 30s checkpoint: {stage2_30s_ckpt}")

            if stage2_30s_ckpt is None and stage1_ckpt is None:
                print(
                    f"\n[ERROR] context={ctx}: no Stage 2 30s checkpoint found at "
                    f"{results_dir / f'{task}_{head_type}' / 'context_30s' / 'best_model.pt'} "
                    f"and no --stage1-checkpoint override given. Run the 30s context for "
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
                gradient_checkpointing=args.gradient_checkpointing,
                num_workers=args.num_workers,
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
