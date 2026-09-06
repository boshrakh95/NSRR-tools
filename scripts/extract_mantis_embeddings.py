#!/usr/bin/env python3
"""
extract_mantis_embeddings.py — Mantis baseline, Stage 1 Step 1

Extracts per-30-second-epoch Mantis embeddings from the fast-channel HDF5
PSG files and saves one numpy array per subject. See
docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md §9/§10 for the full derivation of
every choice below.

WHY THIS LOOKS DIFFERENT FROM extract_osf_embeddings.py /
extract_physioomni_embeddings.py
─────────────────────────────────────────────────────────
Mantis is ONE encoder, channel-independent by construction — every one of
the 6 canonical channel slots (plan §2.2) goes through the SAME frozen
weights, batched together as one forward call (plan §4.6:
`(B, C, L) -> (B*C, 1, L)`, never Mantis's own `transform()`, which loops
channels one at a time). There is no per-modality architecture to reason
about, unlike PhysioOmni's four independent encoders or OSF's fixed
12-channel input.

ABSENT-SLOT CONTRACT (plan §2.2, DECIDED — different from Stage 2's §14.2)
──────────────────────────────────────────────────────────────────────────
If a slot has no resolvable channel for a subject, its embedding slice is
left at exact zero AND the backbone is never run on it — one subject at a
time, so unlike Stage 2 (which must keep a fixed-shape tensor across a
BATCH of subjects with differing absent-slot patterns and therefore runs
all 6 uniformly then zeros the output), Stage 1 can simply batch only the
slots that are actually present for this subject. Typically all 6; STAGES
has real per-subject gaps (plan §2.1) — this is exercised, not just
theoretical (see scripts/test_mantis_channel_loader.py's STLK00151 case).

OUTPUT FORMAT
─────────────
  {output_dir}/{dataset}/{subject_id}.npy
  dtype  : float16
  shape  : [T, 6, D]
    T = floor(n_samples_128hz / 3840) — total complete 30s epochs
    6 = SLOT_ORDER (EEG, EOG_L, EOG_R, ECG, EMG, RESP)
    D = embedding.embed_dim (512 for 'combined', the decided setting — §3.3)

USAGE
─────
  python extract_mantis_embeddings.py --config configs/phase0_mantis_config.yaml
  python extract_mantis_embeddings.py --config configs/phase0_mantis_config.yaml \\
      --datasets apples --limit 5 --cpu
  python extract_mantis_embeddings.py --config configs/phase0_mantis_config.yaml \\
      --no-skip          # re-extract even if .npy already exists
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nsrr_tools.datasets.mantis_channel_loader import (  # noqa: E402
    EPOCH_SAMPLES,
    N_SLOTS,
    SLOT_ORDER,
    epochs_to_model_input,
    load_mantis_backbone,
    load_subject_channels,
)

# ── TF32 (plan §4.2) — line one of every Mantis script that touches a GPU ────
# PyTorch 2.5 ships matmul TF32 OFF by default (~67 TFLOP/s instead of ~495 on
# H100, a ~7x penalty for nothing). cudnn's own flag already defaults True
# (verified, see plan §4.2's correction to an earlier over-broad claim) — set
# it anyway, harmless and self-documenting.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

# ── Achieved-FLOP/s instrumentation (plan §4.1, §13.3 Pilot 3 item 1) ────────
# "Compute achieved FLOP/s and compare to peak on the first real run" — the
# single check that would have caught PhysioOmni's 0.14%-of-peak Stage 2 run
# weeks earlier. GFLOP_PER_CHANNEL_EPOCH is plan §4.8's hand-derived count for
# OUR EXACT config (241 tokens = Option D, full 6-layer depth = return_transf_
# layer=-1, forward-only since Stage 1 has no backward pass) — not a generic
# estimate. If seq_len/num_patches/return_transf_layer ever change, this
# constant must be re-derived, not reused.
GFLOP_PER_CHANNEL_EPOCH = 5.29
H100_TF32_PEAK_TFLOPS = 495.0


def _timed_backbone_forward(backbone, x: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, float]:
    """Runs backbone(x) and returns (output, wall_seconds), timed correctly
    for GPU: CUDA kernel launches are ASYNCHRONOUS, so a naive
    time.time()-around-the-call would stop the clock before the GPU actually
    finishes and silently under-report elapsed time — exactly the kind of
    measurement mistake that let TF32-off go unnoticed for weeks on OSF/
    PhysioOmni. `torch.cuda.synchronize()` blocks until all queued GPU work
    is actually done before the second timestamp is taken."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = backbone(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


# ── Graceful-stop flag (set by SIGTERM handler) ───────────────────────────────
# Mirrors extract_osf_embeddings.py's / extract_physioomni_embeddings.py's
# pattern (finish current subject, then stop) — no per-subject resume
# checkpoint here either, only the out_path.exists() skip-logic below.
_stop_requested = False


def _handle_sigterm(signum, frame):
    global _stop_requested
    logger.warning("[SIGTERM] Stop requested — will exit after current subject completes.")
    _stop_requested = True


signal.signal(signal.SIGTERM, _handle_sigterm)


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_subject_embeddings(
    h5_path: Path,
    backbone,
    device: torch.device,
    chunk_batch_size: int,
    windowing: str,
    embed_dim: int,
    channel_candidates: dict,
) -> tuple[np.ndarray, dict, float, int]:
    """Extract [T, 6, embed_dim] float16 embeddings for one subject.

    Batches channels into ONE forward per chunk (plan §4.6): reshapes
    `(n_epochs, n_present_slots, 1, L)` to `(n_epochs * n_present_slots, 1, L)`
    before the backbone call, then scatters the result back by
    `present_idxs` — absent slots are never forwarded (plan §2.2's Stage 1
    contract), not run-then-zeroed (that is Stage 2's different, batch-driven
    design, plan §14.2).

    `chunk_batch_size` counts CHANNEL-epochs of PRESENT slots (plan §4.4) —
    derived per-subject from `chunk_batch_size // n_present`, since most
    subjects have all 6 slots present but some (chiefly STAGES) have fewer.

    Returns (embeddings, fill_info, forward_seconds, channel_epochs_processed).
    The last two are ONLY meaningful for `windowing == "full_epoch"` — see
    `GFLOP_PER_CHANNEL_EPOCH`'s docstring for why `subwindow` isn't counted
    the same way (it's 0 for subwindow so callers don't silently mis-account
    it into a TFLOP/s figure the constant was never derived for).
    """
    x, fill_info = load_subject_channels(h5_path, channel_candidates)
    t_epochs = x.shape[1] // EPOCH_SAMPLES
    if t_epochs == 0:
        raise ValueError(f"Recording too short (< 1 epoch) in {h5_path}")

    present_idxs = [i for i, slot in enumerate(SLOT_ORDER) if slot not in fill_info["slots_missing"]]
    if not present_idxs:
        raise ValueError(f"No Mantis-relevant channels found at all in {h5_path}")
    n_present = len(present_idxs)

    out = np.zeros((t_epochs, N_SLOTS, embed_dim), dtype=np.float32)
    epochs_per_chunk = max(1, chunk_batch_size // n_present)

    forward_seconds = 0.0
    channel_epochs_processed = 0

    for start in range(0, t_epochs, epochs_per_chunk):
        n = min(epochs_per_chunk, t_epochs - start)
        model_in = epochs_to_model_input(x, windowing, start, n)

        if windowing == "full_epoch":
            model_in = model_in.reshape(n, N_SLOTS, 1, EPOCH_SAMPLES)
            sel = model_in[:, present_idxs].reshape(n * n_present, 1, EPOCH_SAMPLES)
        elif windowing == "subwindow":
            model_in = model_in.reshape(n, N_SLOTS, 8, 1, 512)
            sel = model_in[:, present_idxs].reshape(n * n_present * 8, 1, 512)
        else:
            raise ValueError(f"Unknown windowing: {windowing!r}")

        sel = sel.to(device)  # OUTSIDE the timed call — this is data transfer,
                               # not compute; including it would understate the
                               # achieved-TFLOP/s figure the timer exists to give.
        emb_out, fwd_s = _timed_backbone_forward(backbone, sel, device)
        if windowing == "full_epoch":
            forward_seconds += fwd_s
            channel_epochs_processed += n * n_present  # GFLOP_PER_CHANNEL_EPOCH's own unit

        if windowing == "subwindow":
            emb_out = emb_out.reshape(n, n_present, 8, embed_dim).mean(dim=2)
        else:
            emb_out = emb_out.reshape(n, n_present, embed_dim)

        out[start:start + n][:, present_idxs, :] = emb_out.cpu().float().numpy()

    return out.astype(np.float16), fill_info, forward_seconds, channel_epochs_processed


# ─────────────────────────────────────────────────────────────────────────────
# Subject discovery — identical pattern to extract_osf/physioomni_embeddings.py
# ─────────────────────────────────────────────────────────────────────────────

def find_hdf5_files(hdf5_dir: str, datasets: list, limit: int = None) -> list:
    root = Path(hdf5_dir)
    subjects = []
    for dataset in datasets:
        h5_dir = root / dataset / "derived" / "hdf5_signals"
        if not h5_dir.exists():
            logger.warning(f"HDF5 dir not found, skipping: {h5_dir}")
            continue
        files = sorted(h5_dir.glob("*.h5"))
        for fp in files:
            subjects.append((dataset, fp.stem, fp))
        logger.info(f"  {dataset}: {len(files)} HDF5 files found")
    if limit:
        subjects = subjects[:limit]
    return subjects


def slice_subjects(subjects: list, start_idx: int, end_idx: int | None, limit: int | None) -> list:
    subjects = subjects[start_idx:end_idx]
    if limit:
        subjects = subjects[:limit]
    return subjects


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract Mantis embeddings (Stage 1 Step 1)")
    parser.add_argument("--config",      required=True, help="Path to phase0_mantis_config.yaml")
    parser.add_argument("--datasets",    nargs="+",     help="Override datasets list from config")
    parser.add_argument("--limit",       type=int,      help="Process only first N subjects (debug)")
    parser.add_argument("--start-idx",   type=int,      default=0,    help="First subject index (for parallel jobs)")
    parser.add_argument("--end-idx",     type=int,      default=None, help="Last subject index exclusive (for parallel jobs)")
    parser.add_argument("--no-skip",     action="store_true", help="Re-extract even if .npy exists")
    parser.add_argument("--cpu",         action="store_true", help="Force CPU (debugging only)")
    parser.add_argument("--gpu-fraction", type=float, default=1 / 7,
                         help="Fraction of a full H100 actually allocated (plan §4.1's peak-comparison "
                              "denominator = H100_TF32_PEAK_TFLOPS * this). Default 1/7 matches this "
                              "project's usual 1g.10gb MIG extraction slice (OSF's/PhysioOmni's own "
                              "extraction jobs use the same size). Pass 1.0 for a whole card "
                              "(--gpus=h100:1), 2/7 for 2g.20gb, 3/7 for 3g.40gb. NOT auto-detected: "
                              "a real Pilot 3 run (2026-09-06) found nvidia-smi reporting the FULL "
                              "card's 80GB even inside a 10GB MIG slice job, so device queries can't "
                              "be trusted here — the operator (whoever wrote the --gpus= spec) is the "
                              "only reliable source of truth.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    emb_cfg = cfg["embedding"]
    data_cfg = cfg["data"]

    datasets = args.datasets or emb_cfg["datasets"]
    output_dir = Path(emb_cfg["output_dir"])
    chunk_batch_size = emb_cfg.get("chunk_batch_size", 192)
    windowing = emb_cfg.get("windowing", "full_epoch")
    embed_dim = emb_cfg["embed_dim"]
    hdf5_dir = data_cfg["hdf5_dir"]
    channel_candidates = data_cfg["channel_candidates"]

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    for ds in datasets:
        (output_dir / ds).mkdir(parents=True, exist_ok=True)

    checkpoint_source = emb_cfg["local_dir"] if Path(emb_cfg["local_dir"]).exists() else emb_cfg["repo_id"]
    logger.info(f"Loading Mantis backbone from: {checkpoint_source} "
                f"(seq_len={emb_cfg['seq_len']}, num_patches={emb_cfg['num_patches']}, "
                f"return_transf_layer={emb_cfg['return_transf_layer']}, "
                f"output_token={emb_cfg['output_token']}, pe_mode={emb_cfg.get('pe_mode', 'extrapolate')})")
    backbone = load_mantis_backbone(
        checkpoint_source,
        seq_len=emb_cfg["seq_len"],
        num_patches=emb_cfg["num_patches"],
        return_transf_layer=emb_cfg["return_transf_layer"],
        output_token=emb_cfg["output_token"],
        device=device,
        pe_mode=emb_cfg.get("pe_mode", "extrapolate"),
    )

    # Assert embed_dim matches what the backbone actually outputs, on a tiny
    # dummy forward — cheap, and catches a config/checkpoint mismatch before
    # any real subject is processed rather than mid-extraction.
    with torch.no_grad():
        probe_out = backbone(torch.zeros(1, 1, emb_cfg["seq_len"], device=device))
    if probe_out.shape[-1] != embed_dim:
        raise ValueError(
            f"config embed_dim={embed_dim} but backbone actually outputs "
            f"{probe_out.shape[-1]}-dim — fix embedding.embed_dim in {args.config}"
        )

    logger.info(f"Scanning HDF5 files in: {hdf5_dir}")
    all_subjects = find_hdf5_files(hdf5_dir, datasets, limit=None)
    subjects = slice_subjects(all_subjects, args.start_idx, args.end_idx, args.limit)
    logger.info(
        f"Total available: {len(all_subjects)} | "
        f"This job: [{args.start_idx}:{args.end_idx}] = {len(subjects)} subjects"
    )

    n_ok = n_skip = n_err = 0
    total_forward_seconds = 0.0
    total_channel_epochs = 0
    t0 = time.time()
    fill_log_handles: dict[str, "object"] = {}

    for i, (dataset, subject_id, h5_path) in enumerate(subjects):
        out_path = output_dir / dataset / f"{subject_id}.npy"

        if out_path.exists() and not args.no_skip:
            n_skip += 1
            continue

        try:
            t_sub = time.time()
            emb, fill_info, fwd_s, n_chan_epochs = extract_subject_embeddings(
                h5_path=h5_path,
                backbone=backbone,
                device=device,
                chunk_batch_size=chunk_batch_size,
                windowing=windowing,
                embed_dim=embed_dim,
                channel_candidates=channel_candidates,
            )
            total_forward_seconds += fwd_s
            total_channel_epochs += n_chan_epochs
            np.save(out_path, emb)

            if dataset not in fill_log_handles:
                log_path = output_dir / dataset / "_channel_fill_log.jsonl"
                fill_log_handles[dataset] = open(log_path, "a")
            fill_log_handles[dataset].write(
                json.dumps({"subject_id": subject_id, "t_epochs": emb.shape[0], **fill_info}) + "\n"
            )
            fill_log_handles[dataset].flush()

            elapsed = time.time() - t_sub
            n_ok += 1

            if (i + 1) % 50 == 0 or args.limit:
                logger.info(
                    f"[{i+1}/{len(subjects)}] {dataset}/{subject_id} "
                    f"→ shape {emb.shape} in {elapsed:.1f}s "
                    f"(slots_missing: {fill_info['slots_missing']}, "
                    f"resp_source: {fill_info['resp_source']})"
                )

        except Exception as exc:
            logger.error(f"  FAILED {dataset}/{subject_id}: {exc}")
            n_err += 1

        if _stop_requested:
            logger.warning(
                f"[SIGTERM] Stopping after {n_ok + n_err} processed subjects. "
                f"Resubmit to continue (existing .npy files will be skipped)."
            )
            break

    for handle in fill_log_handles.values():
        handle.close()

    total = time.time() - t0
    logger.info(
        f"\nDone in {total/60:.1f} min — "
        f"extracted: {n_ok}, skipped: {n_skip}, errors: {n_err}"
        + (f" ({n_ok/total:.2f} subjects/s)" if n_ok and total > 0 else "")
    )

    # Achieved FLOP/s vs H100 TF32 peak (plan §4.1, §13.3 Pilot 3 item 1) —
    # ONLY meaningful for windowing='full_epoch' (GFLOP_PER_CHANNEL_EPOCH's
    # own scope, see extract_subject_embeddings' docstring).
    #
    # % of peak is computed against args.gpu_fraction * H100_TF32_PEAK_TFLOPS,
    # NOT a flat full-card denominator — a real Pilot 3 run (2026-09-06) on a
    # 1g.10gb slice initially reported "0.88% of H100 peak", which looked like
    # a red flag, when the true utilization of what was actually allocated was
    # ~6.2%. Device auto-detection was considered and rejected: nvidia-smi
    # reported the FULL card's 80GB from inside that same 10GB MIG job, so a
    # device query can't be trusted to self-report the slice size either —
    # --gpu-fraction is operator-set, matching whatever --gpus= was requested.
    if windowing == "full_epoch" and total_forward_seconds > 0:
        achieved_tflops = (total_channel_epochs * GFLOP_PER_CHANNEL_EPOCH) / total_forward_seconds / 1000
        allocated_peak_tflops = H100_TF32_PEAK_TFLOPS * args.gpu_fraction
        pct_of_allocated = 100 * achieved_tflops / allocated_peak_tflops
        pct_of_full_card = 100 * achieved_tflops / H100_TF32_PEAK_TFLOPS
        logger.info(
            f"Achieved: {achieved_tflops:.3f} TFLOP/s — "
            f"{pct_of_allocated:.2f}% of the {args.gpu_fraction:.3f}x-H100 allocation's own "
            f"~{allocated_peak_tflops:.1f} TFLOP/s ceiling "
            f"({pct_of_full_card:.2f}% of a full H100's {H100_TF32_PEAK_TFLOPS:.0f} TFLOP/s, for reference), "
            f"forward-only over {total_channel_epochs} channel-epochs in {total_forward_seconds:.1f}s "
            f"(device={device})"
        )
        if device.type == "cuda" and pct_of_allocated < 5.0:
            logger.warning(
                f"⚠️  Under 5% of the ALLOCATED GPU's own peak (not the full card's) — "
                f"STOP and diagnose before a real sweep (plan §4.1/§13.3): check TF32 is "
                f"actually active, chunk_batch_size, and whether this is overhead-bound "
                f"(see LORA_GPU_THROUGHPUT_INVESTIGATION.md). Also double-check --gpu-fraction "
                f"matches the actual --gpus= this job requested."
            )


if __name__ == "__main__":
    main()
