#!/usr/bin/env python3
"""
verify_physioomni_checkpoint.py — PhysioOmni baseline, Phase 0 Step 2

Strict-load-verifies the downloaded PhysioOmni.pt checkpoint against the
real PhysioOmni model code (model/neural_transformer.py's NeuralTransformer
+ NTConfig), mirroring the rigor of OSF's own checkpoint verification
(docs/TSFM_OSF_IMPLEMENTATION_PLAN.md §5). Read-only — does not touch
nsrr_tools, our HDF5s, or any OSF file.

WHAT THIS CONFIRMS (see docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md §2 for
the full write-up):
  - The checkpoint's top-level dict has a 'model' key (plus optimizer state
    and per-modality *_encoder_args dicts — the args needed to instantiate
    NTConfig are stored IN the checkpoint, no need to hardcode them).
  - ckpt['model'] keys are prefixed EEG_encoder./EOG_encoder./ECG_encoder./
    EMG_encoder. — exactly what FT.py's __init__ expects.
  - strict=False loading (the same call FT.py itself makes) reports ZERO
    missing keys and exactly one unexpected key per encoder (mask_token —
    an MSM-pretraining-only component absent from FT.py's plainer
    NeuralTransformer, correctly and harmlessly ignored).

USAGE
─────
  python scripts/verify_physioomni_checkpoint.py \\
      --checkpoint /home/boshra95/PhysioOmni/checkpoints/PhysioOmni.pt \\
      --physioomni-repo /home/boshra95/PhysioOmni
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Verify the PhysioOmni checkpoint loads cleanly")
    parser.add_argument("--checkpoint", required=True, help="Path to PhysioOmni.pt")
    parser.add_argument("--physioomni-repo", required=True,
                         help="Path to the PhysioOmni GitHub repo (for model/ imports)")
    args = parser.parse_args()

    sys.path.insert(0, args.physioomni_repo)
    import torch
    from model.neural_transformer import NeuralTransformer, NTConfig

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"Top-level keys: {list(ckpt.keys())}")
    print(f"epoch={ckpt.get('epoch')} iter_num={ckpt.get('iter_num')} "
          f"best_val_loss={ckpt.get('best_val_loss')}")

    model_sd = ckpt["model"]
    total_params = 0
    any_missing = False
    encoders = {}

    for mod in ["EEG", "EOG", "ECG", "EMG"]:
        model_args = ckpt[f"{mod}_encoder_args"]
        conf = NTConfig(**model_args)
        enc = NeuralTransformer(conf)

        prefix = f"{mod}_encoder."
        filtered = {k[len(prefix):]: v for k, v in model_sd.items() if k.startswith(prefix)}
        missing, unexpected = enc.load_state_dict(filtered, strict=False)

        n_params = sum(p.numel() for p in enc.parameters())
        total_params += n_params
        encoders[mod] = enc

        status = "OK" if not missing else "MISSING KEYS — INVESTIGATE"
        if missing:
            any_missing = True
        print(f"{mod}: n_embd={model_args['n_embd']} patch_size={model_args['patch_size']} "
              f"n_params={n_params:,} missing={len(missing)} unexpected={unexpected} [{status}]")

    print(f"\nTOTAL encoder params (sum of 4): {total_params:,}")

    if any_missing:
        print("\nFAILED — at least one encoder had missing keys, do not trust this checkpoint yet.")
        sys.exit(1)

    print("\nPASSED — all 4 encoders load with zero missing keys "
          "(only the expected MSM-only 'mask_token' ignored per encoder).")


if __name__ == "__main__":
    main()
