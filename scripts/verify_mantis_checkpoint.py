#!/usr/bin/env python3
"""
verify_mantis_checkpoint.py — Mantis baseline, Phase 0 Step 3

Strict-verifies the downloaded Mantis-8M / MantisPlus checkpoints against
the real Mantis code, the way the checkpoint MUST be loaded for the
240-patch (Option D) model — never via `.from_pretrained()`. See
docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md §3.4 for the full derivation.

WHY THIS SCRIPT EXISTS (not just a read of the source)
────────────────────────────────────────────────────────
`MantisV1.from_pretrained()` rebuilds the model from the repo's own
`config.json` (`seq_len=512, num_patches=32`) and hard-raises on a shape
mismatch even with `strict=False` — verified empirically on 2026-09-06, not
assumed. The correct path is: build `MantisV1` directly at our own
`num_patches=240`, load the checkpoint's `model.safetensors` via
`safetensors.torch.load_file`, drop the keys that cannot and should not be
loaded, then `load_state_dict(strict=False)` (which is also what fires the
model's own `vit_unit.` -> `transf_unit.` rename pre-hook).

TWO KEYS MUST BE DROPPED, NOT ONE — a real second finding this script exists
to keep from being silently rediscovered:
  1. `vit_unit.pos_encoder.pe` — the sinusoidal positional buffer, sized for
     33 positions (num_patches=32 pretraining). Deterministic, regenerated
     by the constructor at whatever `num_patches` we ask for. Loading it
     raises `RuntimeError: size mismatch` even with `strict=False`.
  2. `prj.{0,1}.{weight,bias}` — the pretraining-only contrastive-loss
     projector. Its own shape depends on `output_token`: `combined` mode
     DOUBLES `self.hidden_dim` (see `MantisV1.__init__`), so the checkpoint's
     256-dim `prj` collides with our combined-mode model's 512-dim `prj`.
     `prj` is only ever read when `pre_training=True` (verified by reading
     `MantisV1.forward`) — we always run with `pre_training=False`, so it is
     dead weight regardless of what shape it ends up. Also raises a hard
     `RuntimeError` under `strict=False` if left in the state dict.

Live-verified missing-key sets (2026-09-06, this script's own output):
  Mantis-8M : {prj.0.weight, prj.0.bias, prj.1.weight, prj.1.bias,
               tokgen_unit.scalar_encoders.0.scales,
               tokgen_unit.scalar_encoders.1.scales,
               transf_unit.pos_encoder.pe}
  MantisPlus: same, MINUS the two scalar_encoders.scales entries (MantisPlus's
              checkpoint carries those two constant buffers; Mantis-8M's does
              not — the only real difference between the two checkpoints,
              docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md §1.1).
  Both: ZERO unexpected keys.

PARAMETER COUNT — a real correction to the widely-quoted "8.11M"
──────────────────────────────────────────────────────────────────
`8,112,384` (the number in every checkpoint's safetensors header, and
docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md §1.1's table) is the sum of ALL
tensors in the checkpoint file — including the `pos_encoder.pe` BUFFER
(not a trainable Parameter) and the dead-at-inference `prj` head (256-dim,
as originally pretrained). The number that actually matters for describing
model size, and for computing what fraction LoRA adapts, is the count of
`nn.Parameter`s actually exercised at frozen inference — i.e. everything
except `prj`: **8,037,632**, for BOTH checkpoints (they are architecturally
identical). This script asserts that number directly rather than restating
the checkpoint-file total.

USAGE
─────
  python scripts/verify_mantis_checkpoint.py
  python scripts/verify_mantis_checkpoint.py --checkpoint-dir /home/boshra95/mantis_checkpoints
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

from mantis.architecture import MantisV1

# ── Constants (docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md §1.1, §3.4) ─────────────
CHECKPOINTS = {
    "Mantis-8M": {
        "params_expected": 8_112_384,          # checkpoint-file total (incl. pe buffer + 256-dim prj)
        "extra_missing": {                      # Mantis-8M's checkpoint lacks these entirely
            "tokgen_unit.scalar_encoders.0.scales",
            "tokgen_unit.scalar_encoders.1.scales",
        },
    },
    "MantisPlus": {
        "params_expected": 8_112_402,
        "extra_missing": set(),                 # MantisPlus's checkpoint DOES carry the scales
    },
}
LIVE_PARAMS_EXPECTED = 8_037_632   # both checkpoints — excludes the dead `prj` head (see docstring)

ALWAYS_DROP = ["vit_unit.pos_encoder.pe", "prj.0.weight", "prj.0.bias", "prj.1.weight", "prj.1.bias"]
ALWAYS_MISSING_AFTER_DROP = {"transf_unit.pos_encoder.pe", "prj.0.weight", "prj.0.bias", "prj.1.weight", "prj.1.bias"}

LORA_TARGET_MODULES = ["to_qkv", "to_out.0"]
LORA_R = 8
LORA_EXPECTED_WRAPPED_MODULES = 12   # 6 transformer blocks x 2 target modules
LORA_EXPECTED_TRAINABLE_PARAMS = 221_184   # live-verified below; hand-derived in plan §1.3


def load_mantis_backbone(safetensors_path: Path, num_patches: int, seq_len: int,
                          return_transf_layer: int, output_token: str) -> MantisV1:
    """The one correct way to load a Mantis checkpoint at a non-native
    num_patches. Never use `.from_pretrained()` — see module docstring."""
    net = MantisV1(seq_len=seq_len, num_patches=num_patches,
                    return_transf_layer=return_transf_layer,
                    output_token=output_token, pre_training=False, device="cpu")
    sd = load_file(str(safetensors_path))
    for key in ALWAYS_DROP:
        sd.pop(key, None)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    return net, set(missing), set(unexpected)


def verify_one_checkpoint(name: str, checkpoint_dir: Path) -> bool:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    safetensors_path = checkpoint_dir / name / "model.safetensors"
    if not safetensors_path.exists():
        print(f"  FAIL: not found at {safetensors_path}")
        return False

    ok = True
    net, missing, unexpected = load_mantis_backbone(
        safetensors_path, num_patches=240, seq_len=3840,
        return_transf_layer=-1, output_token="combined",
    )

    expected_missing = ALWAYS_MISSING_AFTER_DROP | CHECKPOINTS[name]["extra_missing"]
    print(f"  missing keys   : {sorted(missing)}")
    print(f"  unexpected keys: {sorted(unexpected)}")
    if missing != expected_missing:
        print(f"  FAIL: missing set does not match expected {sorted(expected_missing)}")
        ok = False
    if unexpected:
        print(f"  FAIL: expected zero unexpected keys, got {sorted(unexpected)}")
        ok = False

    total_incl_dead_prj = sum(p.numel() for p in net.parameters())
    live_params = sum(p.numel() for n, p in net.named_parameters() if not n.startswith("prj."))
    print(f"  live params (excl. dead prj head): {live_params:,}"
          f"  (expected {LIVE_PARAMS_EXPECTED:,})")
    print(f"  [for reference] total incl. randomly-init combined-mode prj: {total_incl_dead_prj:,}")
    if live_params != LIVE_PARAMS_EXPECTED:
        print(f"  FAIL: live param count mismatch")
        ok = False

    # Zero BatchNorm — machine-checks §4.4's chunk_batch_size safety claim.
    has_bn = any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in net.modules())
    print(f"  has BatchNorm: {has_bn} (expected False)")
    if has_bn:
        print("  FAIL: found BatchNorm — chunk_batch_size is NOT a safe throughput knob, re-check plan §4.4")
        ok = False

    # 6-channel batching sanity (plan §4.6): (B,C,L) -> (B*C,1,L), one forward, reshape back.
    net.eval()
    B, C, L = 3, 6, 3840
    x = torch.randn(B, C, L)
    with torch.no_grad():
        out = net(x.reshape(B * C, 1, L))
    emb = out.reshape(B, C, -1)
    n_nan = torch.isnan(emb).sum().item()
    n_inf = torch.isinf(emb).sum().item()
    std = emb.std(dim=(0, 1)).mean().item()
    print(f"  batched-channel forward: shape={tuple(emb.shape)} (expect (3, 6, 512)), "
          f"NaN={n_nan}, Inf={n_inf}, mean-per-dim-std={std:.4f}")
    if emb.shape != (B, C, 512):
        print("  FAIL: unexpected output shape for combined @ last")
        ok = False
    if n_nan or n_inf:
        print("  FAIL: NaN/Inf in a random-input forward pass")
        ok = False
    if std < 1e-4:
        print("  FAIL: degenerate (near-zero-variance) output")
        ok = False

    # LoRA injection (plan §1.3) — live-verified against the REAL checkpoint, not reasoned about.
    from peft import LoraConfig, get_peft_model
    peft_net = get_peft_model(net, LoraConfig(
        target_modules=LORA_TARGET_MODULES, r=LORA_R, lora_alpha=16, lora_dropout=0.05,
    ))
    wrapped = [n for n, m in peft_net.named_modules() if hasattr(m, "lora_A")]
    trainable = sum(p.numel() for p in peft_net.parameters() if p.requires_grad)
    print(f"  LoRA-wrapped Linears: {len(wrapped)} (expected {LORA_EXPECTED_WRAPPED_MODULES})")
    print(f"  LoRA trainable params: {trainable:,} (expected {LORA_EXPECTED_TRAINABLE_PARAMS:,}, "
          f"= {100 * trainable / LIVE_PARAMS_EXPECTED:.2f}% of live params)")
    if len(wrapped) != LORA_EXPECTED_WRAPPED_MODULES:
        print("  FAIL: unexpected LoRA injection count")
        ok = False
    if trainable != LORA_EXPECTED_TRAINABLE_PARAMS:
        print("  FAIL: unexpected LoRA trainable-param count")
        ok = False

    # modules_to_save (plan §14.5): confirms peft keeps a full-rank trainable
    # copy of a named submodule alongside the LoRA-adapted backbone. Uses a
    # stand-in nn.Linear here, not the real sequence_head.py (Phase 1 work) —
    # the mechanism being tested is peft's, not our head's.
    class _CombinedStub(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.sequence_head = nn.Linear(512, 2)

    combined = _CombinedStub(net)
    peft_combined = get_peft_model(combined, LoraConfig(
        target_modules=LORA_TARGET_MODULES, r=LORA_R, lora_alpha=16, lora_dropout=0.05,
        modules_to_save=["sequence_head"],
    ))
    head_trainable = any(
        p.requires_grad for n, p in peft_combined.named_parameters() if "sequence_head" in n
    )
    backbone_non_lora_frozen = all(
        not p.requires_grad for n, p in peft_combined.named_parameters()
        if "backbone" in n and "lora_" not in n
    )
    print(f"  modules_to_save leaves sequence_head trainable: {head_trainable} (expected True)")
    print(f"  non-LoRA backbone params stay frozen: {backbone_non_lora_frozen} (expected True)")
    if not head_trainable or not backbone_non_lora_frozen:
        print("  FAIL: modules_to_save is not isolating the head as expected")
        ok = False

    print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Verify Mantis-8M / MantisPlus checkpoints load correctly")
    parser.add_argument("--checkpoint-dir", default="/home/boshra95/mantis_checkpoints",
                         help="Directory containing Mantis-8M/ and MantisPlus/ subdirs")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    results = {name: verify_one_checkpoint(name, checkpoint_dir) for name in CHECKPOINTS}

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    if not all(results.values()):
        sys.exit(1)
    print("\nAll checkpoints verified.")


if __name__ == "__main__":
    main()
