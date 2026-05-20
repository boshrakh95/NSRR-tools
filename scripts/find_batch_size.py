#!/usr/bin/env python3
"""
find_batch_size.py — Probe the maximum GPU batch size for memory-bounded training.

For each context length, runs a realistic forward+backward pass (same model
architecture and AMP setting as training) with find_executable_batch_size from
HuggingFace accelerate.  It starts at --starting-batch-size and reduces on OOM
until a batch fits.

A safety factor (default 0.85) is then applied and the result is rounded DOWN to
the nearest multiple of 8 (for tidy warp alignment).  Both the raw probed value
and the conservative safe value are written to batch_sizes.json.

gen_commands.py train (memory_bounded mode) reads the "safe" value automatically.

Usage — generate the sbatch command via gen_commands.py:

    python scripts/gen_commands.py probe-batch <exp_id>

Or run directly on a GPU node:

    python scripts/find_batch_size.py \\
        --config configs/phase0_v3_config.yaml \\
        --head transformer \\
        --exp-dir /scratch/.../sleep_efficiency_binary_transformer_membnd \\
        [--contexts 30s 10m 40m 80m 120m 240m] \\
        [--num-classes 2] \\
        [--starting-batch-size 256] \\
        [--safety-factor 0.85] \\
        [--no-amp]

Output:
    {exp_dir}/batch_sizes.json  — {context: {"probed": N, "safe": M}} mapping
    stdout                      — summary table + optional registry YAML snippet
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nsrr_tools.datasets.context_window_dataset import (
    FULL_NIGHT_SENTINEL,
    parse_context_length,
)
from nsrr_tools.models.sequence_head import build_head


def safe_batch_size(probed: int, safety_factor: float) -> int:
    """Apply safety_factor and round DOWN to the nearest multiple of 8.

    Multiple-of-8 keeps things aligned with CUDA warp sizes (32 threads,
    but matrix ops tile in multiples of 8).  Safety factor gives headroom
    for DataLoader buffers and framework overhead not captured by the probe.

    Examples (safety_factor=0.85):
        256 → int(256*0.85)=217 → floor(217/8)*8 = 216
        207 → int(207*0.85)=175 → floor(175/8)*8 = 168
          1 → max(1, ...) = 1
    """
    reduced = int(probed * safety_factor)
    aligned = max(8, (reduced // 8) * 8)
    return aligned


def probe_context(
    cfg: dict,
    head_type: str,
    seq_len: int,
    num_classes: int,
    device: torch.device,
    use_amp: bool,
    starting_batch_size: int = 256,
) -> int:
    """Return the raw largest batch size that fits in GPU memory (before safety rounding).

    Probes BOTH training and validation memory peaks, because the Transformer head
    selects different PyTorch attention backends depending on model.training:

    - model.train() (training phase): PyTorch slow path, SDPA backend chosen by
      dtype + mask. With mask=None and fp32 on H100, Flash attention is used → O(N).
    - model.eval() (validation phase): PyTorch fast path (_transformer_encoder_layer_fwd
      C++ kernel), which on older versions forced Math attention (O(N²)) for any
      non-None src_key_padding_mask — even an all-zeros float tensor.

    The mask fix in TransformerHead.forward() (pass None when key_mask is all-False)
    makes both modes use Flash attention. This probe tests eval mode specifically
    because validation is where OOM was previously observed (train completed, val
    crashed). Training forward+backward uses more peak memory than eval forward-only,
    so the probe tests the higher-memory path (train) for the batch limit and also
    confirms eval forward fits at the same batch.

    Mask assumption: all-False (no padded positions). This is valid because
    dataset.min_recording_patches=2880 in the config ensures every subject's
    recording is at least as long as the longest context window (240m). With no
    subjects shorter than the context, training and validation data never produce
    padded masks, so the probe's all-False mask accurately represents real data.

    WARNING: if min_recording_patches is disabled or set below the longest context
    length in the sweep, padded samples will appear in real batches. The Transformer
    head then passes a non-None mask, forcing Math attention (O(N²)) with fp32,
    which uses drastically more memory. See docs/cohort_filter.md.

    Args:
        cfg: Full phase0 config dict.
        head_type: "lstm", "transformer", or "mean_pool".
        seq_len: Number of 5-second embedding patches per context window.
        num_classes: Number of output classes.
        device: Target torch device (must be CUDA for meaningful results).
        use_amp: Whether to use torch.cuda.amp.autocast (matches training).
        starting_batch_size: Largest batch to try first; reduced on OOM.

    Returns:
        The raw probed batch size (before safety factor). Pass to safe_batch_size()
        to get the value suitable for training.
    """
    from accelerate.utils import find_executable_batch_size

    input_dim = cfg["model"]["input_dim"]
    m_cfg = {**cfg, "model": {**cfg["model"], "num_classes": num_classes, "head_type": head_type}}
    criterion = torch.nn.CrossEntropyLoss()

    result = {"batch_size": None}

    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def _probe(batch_size):
        # Fresh allocations every attempt — avoids leftover tensors from OOM calls.
        gc.collect()
        torch.cuda.empty_cache()

        model     = build_head(m_cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        x         = torch.randn(batch_size, seq_len, input_dim, device=device)
        # All-False mask: valid because min_recording_patches filter ensures no
        # padded samples exist in real data. See docstring for the full reasoning.
        mask      = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        y         = torch.randint(0, num_classes, (batch_size,), device=device)

        # ── Training step (forward + backward) ────────────────────────────────
        # Train mode must be tested first — backward pass consumes more memory
        # than eval forward-only, so this is the binding memory constraint.
        model.train()
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(x, mask)
            loss   = criterion(logits, y)

        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        del logits, loss
        gc.collect()
        torch.cuda.empty_cache()

        # ── Validation step (eval forward-only) ───────────────────────────────
        # PyTorch selects a different attention kernel in eval mode
        # (_transformer_encoder_layer_fwd C++ fast path). With mask=None (after
        # the cohort filter) this also uses Flash attention, but we test it
        # explicitly to catch any future regression.
        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            logits_val = model(x, mask)

        result["batch_size"] = batch_size

        del model, optimizer, x, mask, y, logits_val
        gc.collect()
        torch.cuda.empty_cache()

    _probe()
    return result["batch_size"]


def main():
    parser = argparse.ArgumentParser(
        description="Probe max GPU batch size per context for memory-bounded training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config",   required=True,
                        help="Path to phase0_config.yaml")
    parser.add_argument("--head",     required=True,
                        choices=["lstm", "transformer", "mean_pool"],
                        help="Head architecture to probe")
    parser.add_argument("--exp-dir",  required=True, dest="exp_dir",
                        help="Experiment output directory — batch_sizes.json written here")
    parser.add_argument("--contexts", nargs="+",
                        default=["30s", "10m", "40m", "80m", "120m", "240m"],
                        help="Context lengths to probe (default: 30s 10m 40m 80m 120m 240m)")
    parser.add_argument("--num-classes", type=int, default=2, dest="num_classes",
                        help="Number of output classes (default: 2)")
    parser.add_argument("--starting-batch-size", type=int, default=256,
                        dest="starting_batch_size",
                        help="Largest batch to try first; halved on OOM (default: 256)")
    parser.add_argument("--safety-factor", type=float, default=0.85, dest="safety_factor",
                        help="Fraction of probed batch to actually use (default: 0.85). "
                             "Provides headroom for DataLoader buffers not captured by the probe. "
                             "Result is rounded down to nearest multiple of 8.")
    parser.add_argument("--no-amp", action="store_true", dest="no_amp",
                        help="Disable AMP — probe in fp32 (use if training disables AMP)")
    parser.add_argument("--cpu", action="store_true",
                        help="Run on CPU for debugging; batch sizes won't reflect GPU capacity")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device  = torch.device(
        "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    use_amp = (
        cfg["training"].get("mixed_precision", True)
        and device.type == "cuda"
        and not args.no_amp
    )

    if device.type != "cuda":
        print(
            "[WARNING] No CUDA device found. Batch sizes probed on CPU will not "
            "reflect GPU memory capacity.",
            file=sys.stderr,
        )

    exp_dir = Path(args.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device:              {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        total_gb = props.total_memory / 1024**3
        print(f"GPU:                 {props.name}  ({total_gb:.1f} GB)")
    print(f"Head:                {args.head}")
    print(f"AMP:                 {use_amp}")
    print(f"Starting batch size: {args.starting_batch_size}")
    print(f"Safety factor:       {args.safety_factor}  (safe = floor(probed × {args.safety_factor} / 8) × 8)")
    print(f"Output dir:          {exp_dir}")
    print()

    # results[ctx] = {"probed": int, "safe": int}
    results: dict[str, dict] = {}

    for ctx in args.contexts:
        N = parse_context_length(ctx)

        if N == FULL_NIGHT_SENTINEL:
            print(f"[SKIP] {ctx:6s}: full_night — variable length, cannot probe statically")
            continue
        if args.head == "transformer" and N > 2880:
            print(f"[SKIP] {ctx:6s}: Transformer not used above 240m (O(N²) memory)")
            continue

        print(f"Probing {ctx:6s} (seq_len={N:5d}) ...", end=" ", flush=True)
        try:
            probed = probe_context(
                cfg=cfg,
                head_type=args.head,
                seq_len=N,
                num_classes=args.num_classes,
                device=device,
                use_amp=use_amp,
                starting_batch_size=args.starting_batch_size,
            )
            safe = safe_batch_size(probed, args.safety_factor)
            results[ctx] = {"probed": probed, "safe": safe}
            print(f"probed={probed:4d}  →  safe={safe:4d}")
        except RuntimeError as exc:
            print(f"FAILED — {exc}")
            results[ctx] = {"probed": 1, "safe": 1}

    # Write batch_sizes.json
    out_path = exp_dir / "batch_sizes.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {out_path}")

    # Summary table
    print()
    print(f"{'Context':<8}  {'probed':>7}  {'safe (used)':>11}")
    print("-" * 32)
    for ctx, v in results.items():
        print(f"{ctx:<8}  {v['probed']:>7}  {v['safe']:>11}")

    # Optional registry YAML snippet (safe values only)
    print()
    print("# ── Safe batch sizes (gen_commands.py reads batch_sizes.json automatically) ──")
    print("# To bake these into v2_registry.yaml memory_bounded defaults instead:")
    print("#   memory_bounded:")
    print("#     context_batch_size:")
    for ctx, v in results.items():
        print(f'#       "{ctx}": {v["safe"]}')


if __name__ == "__main__":
    main()
