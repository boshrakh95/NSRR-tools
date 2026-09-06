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
) -> tuple[np.ndarray, dict]:
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

        with torch.no_grad():
            cls = backbone(sel.to(device))  # (n*n_present[*8], embed_dim)

        if windowing == "subwindow":
            cls = cls.reshape(n, n_present, 8, embed_dim).mean(dim=2)
        else:
            cls = cls.reshape(n, n_present, embed_dim)

        out[start:start + n][:, present_idxs, :] = cls.cpu().float().numpy()

    return out.astype(np.float16), fill_info


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
    t0 = time.time()
    fill_log_handles: dict[str, "object"] = {}

    for i, (dataset, subject_id, h5_path) in enumerate(subjects):
        out_path = output_dir / dataset / f"{subject_id}.npy"

        if out_path.exists() and not args.no_skip:
            n_skip += 1
            continue

        try:
            t_sub = time.time()
            emb, fill_info = extract_subject_embeddings(
                h5_path=h5_path,
                backbone=backbone,
                device=device,
                chunk_batch_size=chunk_batch_size,
                windowing=windowing,
                embed_dim=embed_dim,
                channel_candidates=channel_candidates,
            )
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


if __name__ == "__main__":
    main()
