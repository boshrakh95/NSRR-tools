#!/usr/bin/env python3
"""
find_batch_size.py — Probe the maximum GPU batch size for memory-bounded training.

For each context length, runs a realistic forward+backward pass (same model
architecture and AMP setting as training) with find_executable_batch_size from
HuggingFace accelerate.  It starts at --starting-batch-size and halves on OOM
until a batch fits, then writes the result to {exp_dir}/batch_sizes.json.

gen_commands.py train (memory_bounded mode) reads batch_sizes.json automatically
so no registry edit is needed after probing.

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
        [--no-amp]

Output:
    {exp_dir}/batch_sizes.json  — {context: batch_size} mapping
    stdout                      — YAML snippet to optionally paste into registry
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


def probe_context(
    cfg: dict,
    head_type: str,
    seq_len: int,
    num_classes: int,
    device: torch.device,
    use_amp: bool,
    starting_batch_size: int = 256,
) -> int:
    """Return the largest batch size (power of 2) that fits in GPU memory.

    Simulates a full training step: forward pass + loss + backward.
    Creates a fresh model for each attempt so no stale gradient state leaks
    across OOM retries.

    Args:
        cfg: Full phase0 config dict.
        head_type: "lstm", "transformer", or "mean_pool".
        seq_len: Number of 5-second embedding patches per context window.
        num_classes: Number of output classes.
        device: Target torch device (must be CUDA for meaningful results).
        use_amp: Whether to use torch.cuda.amp.autocast (matches training).
        starting_batch_size: Largest batch to try first; halved on OOM.

    Returns:
        The largest batch size that completed without OOM.
    """
    from accelerate.utils import find_executable_batch_size

    input_dim = cfg["model"]["input_dim"]
    m_cfg = {**cfg, "model": {**cfg["model"], "num_classes": num_classes, "head_type": head_type}}

    result = {"batch_size": None}

    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def _probe(batch_size):
        # Fresh allocations every attempt — avoids leftover tensors from OOM calls.
        gc.collect()
        torch.cuda.empty_cache()

        model     = build_head(m_cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        x         = torch.randn(batch_size, seq_len, input_dim, device=device)
        mask      = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = model(x, mask).mean()

        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        result["batch_size"] = batch_size

        del model, optimizer, x, mask, loss
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
    print(f"Output dir:          {exp_dir}")
    print()

    results: dict[str, int] = {}

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
            bs = probe_context(
                cfg=cfg,
                head_type=args.head,
                seq_len=N,
                num_classes=args.num_classes,
                device=device,
                use_amp=use_amp,
                starting_batch_size=args.starting_batch_size,
            )
            results[ctx] = bs
            print(f"max batch = {bs}")
        except RuntimeError as exc:
            print(f"FAILED — {exc}")
            results[ctx] = 1

    # Write batch_sizes.json
    out_path = exp_dir / "batch_sizes.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {out_path}")

    # Print optional YAML snippet
    print()
    print("# ── Probed batch sizes (gen_commands.py reads batch_sizes.json automatically) ──")
    print("# To override the global memory_bounded defaults in v2_registry.yaml instead:")
    print("#   memory_bounded:")
    print("#     context_batch_size:")
    for ctx, bs in results.items():
        print(f'#       "{ctx}": {bs}')


if __name__ == "__main__":
    main()
