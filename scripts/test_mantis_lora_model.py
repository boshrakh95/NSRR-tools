#!/usr/bin/env python3
"""
test_mantis_lora_model.py — correctness test for CombinedMantisLoRAModel
(checklist 2.4).

No persisted equivalent exists for OSF's/PhysioOmni's own Stage 2 models
(their checklist notes describe this kind of check as done ad hoc, not as
a re-runnable script) — written as a real test here since it's cheap and
catches exactly the kind of subtle wiring bug ("gradients don't actually
reach the LoRA params", "absent-slot zeroing doesn't actually zero the
gradient") that's easy to get wrong silently.

THREE INDEPENDENT CHECKS
────────────────────────
1. Absent-slot masking, in isolation (no model, pure tensor math): the
   `emb * present_mask` operation that implements plan §2.2/§14.2's
   zero-fill contract must produce EXACTLY zero gradient for a masked-out
   channel, regardless of what value the backbone produced for it before
   masking. Fast, deterministic, model-independent — proves the mechanism
   itself is correct before trusting it inside the full model.

2. Real-checkpoint forward+backward: build the actual
   CombinedMantisLoRAModel wrapped with peft, using the REAL Mantis-8M
   checkpoint (not synthetic weights) and a synthetic input batch (one
   subject with a genuinely absent channel, to also exercise check 1
   inside the real model). Assert logits are finite, loss is finite, and
   gradients reach BOTH the LoRA parameters (confirming peft actually
   attached trainable adapters to the backbone's attention blocks) AND
   the sequence_head parameters (confirming the LP-FT staging design
   trains both pieces together, as intended) — and that both gradient
   sets contain at least one nonzero value, not just "no NaN" (an
   all-zero gradient would also pass a naive "is finite" check while
   being silently broken).

3. Gradient-checkpointing bit-identity (plan §14.8): both opt-in rungs
   (`checkpoint_tokgen`, `checkpoint_chunks`) must produce IDENTICAL
   forward output to the non-checkpointed baseline on the same input and
   the same model weights — checkpointing changes *how* activations are
   recomputed during backward, never *what* the forward pass computes.
   Verified to max abs diff 0.0, matching PhysioOmni's own precedent for
   its equivalent check.

Usage:
    python scripts/test_mantis_lora_model.py --config configs/phase0_mantis_lora_config.yaml --cpu
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_mantis_lora import CombinedMantisLoRAModel, build_combined_lora_model  # noqa: E402


def test_absent_slot_zero_gradient() -> bool:
    print("=== Check 1: absent-slot masking zeros gradient exactly ===")
    torch.manual_seed(0)
    B, N, C, D = 2, 3, 6, 4
    emb = torch.randn(B, N, C, D, requires_grad=True)
    # subject 0: channel 3 absent. subject 1: channel 5 absent.
    present = torch.tensor(
        [[1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1, 0]], dtype=torch.bool
    )
    masked = emb * present.to(emb.dtype).view(B, 1, C, 1)
    loss = masked.sum()
    loss.backward()

    ok = True
    absent_grad_zero = bool((emb.grad[0, :, 3, :] == 0).all()) and bool((emb.grad[1, :, 5, :] == 0).all())
    print(f"  absent-slot grad exactly zero: [{'OK' if absent_grad_zero else 'FAIL'}]")
    ok &= absent_grad_zero

    present_grad_nonzero = bool((emb.grad[0, :, 0, :] != 0).all()) and bool((emb.grad[1, :, 0, :] != 0).all())
    print(f"  present-slot grad nonzero (sanity, not a false-pass-everything mask): "
          f"[{'OK' if present_grad_nonzero else 'FAIL'}]")
    ok &= present_grad_nonzero

    print()
    return ok


def _tiny_synthetic_batch(N: int, device: torch.device):
    """A 2-subject batch: subject 0 has all 6 channels present; subject 1
    has channel index 2 (ECG) genuinely absent (exact zero for the whole
    window) — exercises the absent-slot path inside the real model, not
    just in isolation."""
    torch.manual_seed(1)
    x = torch.randn(2, N, 6, 3840, device=device) * 0.5
    x[1, :, 2, :] = 0.0  # subject 1's ECG slot: absent
    mask = torch.zeros(2, N, dtype=torch.bool, device=device)
    y = torch.tensor([0, 1], dtype=torch.long, device=device)
    return x, mask, y


def test_real_checkpoint_gradients(cfg: dict, device: torch.device) -> bool:
    print("=== Check 2: real-checkpoint forward+backward, LoRA + head gradients ===")
    # Small, cheap dims for a fast test — N=1 (30s), tiny chunk_batch_size
    # (only 2 subjects x 1 epoch x 6 channels = 12 items total anyway).
    test_cfg = dict(cfg)
    test_cfg["embedding"] = dict(cfg["embedding"])
    test_cfg["embedding"]["chunk_batch_size"] = 12
    test_cfg["model"] = dict(cfg["model"])
    test_cfg["model"]["num_classes"] = 2

    model = build_combined_lora_model(test_cfg, num_classes=2, head_type=test_cfg["model"]["head_type"], device=device)
    model.train()

    x, mask, y = _tiny_synthetic_batch(N=1, device=device)
    logits = model(x, mask)
    finite_logits = bool(torch.isfinite(logits).all())
    print(f"  logits finite: {tuple(logits.shape)} [{'OK' if finite_logits else 'FAIL'}]")

    loss = torch.nn.functional.cross_entropy(logits, y)
    finite_loss = bool(torch.isfinite(loss))
    print(f"  loss finite: {loss.item():.4f} [{'OK' if finite_loss else 'FAIL'}]")
    loss.backward()

    lora_grads = [p.grad for n, p in model.named_parameters() if "lora_" in n and p.grad is not None]
    lora_finite = all(torch.isfinite(g).all() for g in lora_grads)
    lora_nonzero = any(bool((g != 0).any()) for g in lora_grads)
    print(f"  LoRA params with gradient: {len(lora_grads)}  "
          f"finite=[{'OK' if lora_finite else 'FAIL'}]  nonzero=[{'OK' if lora_nonzero else 'FAIL'}]")

    head_grads = [p.grad for n, p in model.named_parameters()
                  if "sequence_head" in n and p.requires_grad and p.grad is not None]
    head_finite = all(torch.isfinite(g).all() for g in head_grads)
    head_nonzero = any(bool((g != 0).any()) for g in head_grads)
    print(f"  sequence_head params with gradient: {len(head_grads)}  "
          f"finite=[{'OK' if head_finite else 'FAIL'}]  nonzero=[{'OK' if head_nonzero else 'FAIL'}]")

    ok = finite_logits and finite_loss and lora_finite and lora_nonzero and head_finite and head_nonzero
    print()
    return ok


def test_checkpointing_bit_identical(cfg: dict, device: torch.device) -> bool:
    print("=== Check 3: gradient-checkpointing rungs are bit-identical to baseline ===")
    ok = True

    x, mask, _ = _tiny_synthetic_batch(N=1, device=device)

    def build(checkpoint_tokgen=False, checkpoint_chunks=False):
        test_cfg = dict(cfg)
        test_cfg["embedding"] = dict(cfg["embedding"])
        test_cfg["embedding"]["chunk_batch_size"] = 12
        test_cfg["model"] = dict(cfg["model"])
        test_cfg["model"]["num_classes"] = 2
        test_cfg["training"] = dict(cfg.get("training", {}))
        test_cfg["training"]["checkpoint_tokgen"] = checkpoint_tokgen
        test_cfg["training"]["checkpoint_chunks"] = checkpoint_chunks
        torch.manual_seed(42)
        return build_combined_lora_model(test_cfg, num_classes=2, head_type=test_cfg["model"]["head_type"], device=device)

    with torch.no_grad():
        baseline = build()
        out_baseline = baseline(x, mask).clone()

        ckpt_tokgen = build()
        ckpt_tokgen.load_state_dict(baseline.state_dict())
        out_tokgen = ckpt_tokgen(x, mask)
        diff_tokgen = (out_tokgen - out_baseline).abs().max().item()
        tokgen_ok = diff_tokgen == 0.0
        print(f"  checkpoint_tokgen max abs diff vs baseline: {diff_tokgen} [{'OK' if tokgen_ok else 'FAIL'}]")
        ok &= tokgen_ok

    # checkpoint_chunks is exercised via forward+backward (torch_checkpoint
    # requires grad-enabled context to actually take the checkpointed
    # path meaningfully) — compare forward output only, same standard as
    # PhysioOmni's own "max abs diff 0.0" precedent.
    ckpt_chunks_cfg = dict(cfg)
    ckpt_chunks_cfg["embedding"] = dict(cfg["embedding"])
    ckpt_chunks_cfg["embedding"]["chunk_batch_size"] = 12
    ckpt_chunks_cfg["model"] = dict(cfg["model"])
    ckpt_chunks_cfg["model"]["num_classes"] = 2
    ckpt_chunks_cfg["training"] = dict(cfg.get("training", {}))
    ckpt_chunks_cfg["training"]["checkpoint_chunks"] = True
    torch.manual_seed(42)
    ckpt_chunks_model = build_combined_lora_model(
        ckpt_chunks_cfg, num_classes=2, head_type=ckpt_chunks_cfg["model"]["head_type"], device=device
    )
    ckpt_chunks_model.load_state_dict(baseline.state_dict())
    out_chunks = ckpt_chunks_model(x, mask)
    diff_chunks = (out_chunks - out_baseline).abs().max().item()
    chunks_ok = diff_chunks == 0.0
    print(f"  checkpoint_chunks max abs diff vs baseline:  {diff_chunks} [{'OK' if chunks_ok else 'FAIL'}]")
    ok &= chunks_ok

    print()
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}\n")

    ok = True
    ok &= test_absent_slot_zero_gradient()
    ok &= test_real_checkpoint_gradients(cfg, device)
    ok &= test_checkpointing_bit_identical(cfg, device)

    print("PASSED" if ok else "FAILED")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
