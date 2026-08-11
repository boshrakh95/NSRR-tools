#!/usr/bin/env python3
"""
extract_osf_embeddings.py — OSF baseline, Stage 1 Step 1

Extracts per-30-second-epoch OSF embeddings from preprocessed full-channel
HDF5 PSG files and saves one numpy array per subject for all downstream
context-length experiments. Mirrors scripts/extract_sleepfm_embeddings.py's
structure — see docs/TSFM_OSF_IMPLEMENTATION_PLAN.md §3.1 for the full
derivation of every choice below (channel mapping, resampling, forward-pass
call signature, output shape).

OUTPUT FORMAT
─────────────
  {output_dir}/{dataset}/{subject_id}.npy
  dtype  : float16
  shape  : [T, 2, 768]
    T   = floor(n_samples_at_64hz / 1920) — total complete 30s epochs
    2   = [CLS token, mean-pooled patch tokens], both 768-dim
    768 = OSF vit_base width

  Incomplete trailing signal (< 30s epoch) is dropped, same convention as
  SleepFM's incomplete-chunk handling.

CHANNEL MAPPING
───────────────
OSF expects exactly 12 channels in a fixed order (ECG, EMG_Chin, EMG_LLeg,
EMG_RLeg, ABD, THX, NP, SN, EOG_E1_A2, EOG_E2_A1, EEG_C3_A2, EEG_C4_A1) —
see configs/phase0_osf_config.yaml's data.channel_order/channel_candidates
and docs/TSFM_OSF_IMPLEMENTATION_PLAN.md §1 for the full per-cohort
availability audit. Any channel absent from a subject's HDF5 is zero-filled
and logged (per-subject, to {output_dir}/{dataset}/_channel_fill_log.jsonl)
rather than skipping the subject. SHHS is a special case (see
build_channel_candidates() below): it has no distinguishable C3/C4 channel,
so its single generic "EEG" channel is duplicated into both EEG_C3_A2 and
EEG_C4_A1 — a decision confirmed with the user 2026-08-10, flagged as
provisional (see plan doc checklist item 0.5).

RESAMPLING
──────────
Our HDF5s are 128Hz; OSF expects 64Hz. Since 128/64=2 exactly, resampling
via linear interpolation at t=0, 1/64, 2/64, ... lands exactly on original
sample indices 0, 2, 4, ... — i.e. simple decimation (x[::2]) is exactly
equivalent to OSF's own linear-interpolation resampling method for this
specific rate pair, with zero interpolation error. See plan doc §2.

USAGE
─────
  python extract_osf_embeddings.py --config configs/phase0_osf_config.yaml
  python extract_osf_embeddings.py --config configs/phase0_osf_config.yaml \\
      --datasets apples --limit 5 --cpu
  python extract_osf_embeddings.py --config configs/phase0_osf_config.yaml \\
      --no-skip          # re-extract even if .npy already exists
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from loguru import logger

# ── Graceful-stop flag (set by SIGTERM handler) ───────────────────────────────
# Mirrors extract_sleepfm_embeddings.py's pattern (finish current subject,
# then stop), NOT train_context_sweep.py's (immediate sys.exit) — this script
# has no per-subject resume checkpoint, only the out_path.exists() skip-logic
# below, so an abrupt exit mid-subject would leave no trace of partial work.
_stop_requested = False

def _handle_sigterm(signum, frame):
    global _stop_requested
    logger.warning("[SIGTERM] Stop requested — will exit after current subject completes.")
    _stop_requested = True

signal.signal(signal.SIGTERM, _handle_sigterm)

# ── OSF imports ────────────────────────────────────────────────────────────────
# OSF-Open-Sleep-FM is not installed as a package; add the sibling repo to path.
_REPO = Path(__file__).resolve().parent.parent.parent / "OSF-Open-Sleep-FM"
sys.path.insert(0, str(_REPO))

try:
    from osf.backbone.vit1d_cls import vit_base
except ImportError as e:
    logger.error(
        f"Cannot import OSF: {e}\n"
        f"Expected repo at: {_REPO}\n"
        "Run with osf_env (not sleepfm_env)."
    )
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
# Fixed order OSF expects — do not reorder, fed positionally into a Conv2d
# patch-embedding layer indexed by channel position, not by name.
OSF_CHANNEL_ORDER = [
    "ECG", "EMG_Chin", "EMG_LLeg", "EMG_RLeg",
    "ABD", "THX", "NP", "SN",
    "EOG_E1_A2", "EOG_E2_A1", "EEG_C3_A2", "EEG_C4_A1",
]

EPOCH_SECONDS = 30
TARGET_HZ = 64
EPOCH_SAMPLES = EPOCH_SECONDS * TARGET_HZ  # 1920


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    """Load the frozen OSF ViT (vit_base) from the downloaded checkpoint.

    Returns:
        model : ViT in eval mode, on device
        meta  : metadata dict from the checkpoint (num_leads, seq_len,
                patch_size_time, patch_size_ch, lead_wise, width, depth, ...)
    """
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
    model.to(device).eval()

    logger.info(
        f"Loaded OSF vit_base: num_leads={meta['num_leads']}, seq_len={meta['seq_len']}, "
        f"patch_size_time={meta['patch_size_time']}, patch_size_ch={meta['patch_size_ch']}, "
        f"width={meta['width']}, depth={meta['depth']}"
    )
    return model, meta


# ─────────────────────────────────────────────────────────────────────────────
# Channel mapping
# ─────────────────────────────────────────────────────────────────────────────

def build_channel_candidates(dataset: str, cfg_candidates: dict) -> dict:
    """OSF channel -> ordered list of candidate our-HDF5 channel names.

    SHHS special case: no distinguishable C3/C4 channel exists in our
    full-channel HDF5s (confirmed structural, 0/50 subjects in a
    50-subject audit — docs/TSFM_OSF_IMPLEMENTATION_PLAN.md §1). Duplicate
    the single generic "EEG" channel into both EEG slots rather than
    looking for C3-M2/C4-M1 (which will never be found for SHHS).
    """
    candidates = {k: list(v) for k, v in cfg_candidates.items()}
    if dataset == "shhs":
        candidates["EEG_C3_A2"] = ["EEG"]
        candidates["EEG_C4_A1"] = ["EEG"]
    return candidates


def resample_128_to_64(x: np.ndarray) -> np.ndarray:
    """Resample 128Hz -> 64Hz. Exact decimation, not an approximation:
    since 128/64=2, linear interpolation at t=0,1/64,2/64,... lands exactly
    on original sample indices 0,2,4,... — equivalent to OSF's own
    pandas-reindex+linear-interpolate resampling with zero error for this
    specific rate pair. See docs/TSFM_OSF_IMPLEMENTATION_PLAN.md §2.
    """
    return x[::2]


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_subject_embeddings(
    h5_path: Path,
    dataset: str,
    model: "torch.nn.Module",
    device: torch.device,
    chunk_batch_size: int,
    channel_candidates: dict,
) -> tuple[np.ndarray, dict]:
    """Extract [T, 2, 768] float16 embeddings for one subject.

    Processing pipeline (docs/TSFM_OSF_IMPLEMENTATION_PLAN.md §2/§3.1):
      1. Load the 12 mapped channels from the full-channel HDF5 at 128Hz,
         zero-filling any missing channel (fallback candidates tried first).
      2. Resample each channel to 64Hz (exact decimation, see above).
      3. Chunk into contiguous, non-overlapping 30s (1920-sample) epochs;
         drop any incomplete trailing epoch.
      4. Batch epochs through ViT.forward_encoding(x, return_sequence=False)
         -> (cls: [B,768], patches: [B,90,768]).
      5. Mean-pool patches over the 90-token axis -> [B,768].
      6. Stack [cls, mean_pooled_patches] -> [B,2,768] per epoch.

    Returns:
        embeddings : np.ndarray [T, 2, 768], dtype float16
        fill_info  : dict — which OSF channels were zero-filled / which
                     fallback candidate was used, for the per-subject log
    """
    with h5py.File(h5_path, "r") as hf:
        available = set(hf.keys())

        channel_arrays = []
        fill_info = {"zero_filled": [], "fallback_used": {}}
        for osf_ch in OSF_CHANNEL_ORDER:
            candidates = channel_candidates[osf_ch]
            found_key = next((c for c in candidates if c in available), None)

            if found_key is None:
                fill_info["zero_filled"].append(osf_ch)
                channel_arrays.append(None)  # filled with zeros once n_samples_64 is known
                continue

            if found_key != candidates[0]:
                fill_info["fallback_used"][osf_ch] = found_key

            raw = hf[found_key][:].astype(np.float32)
            channel_arrays.append(resample_128_to_64(raw))

    # All present channels share the same original sample count (uniform
    # per-file 128Hz signals), so any non-None array gives us n_samples_64.
    present_arrays = [a for a in channel_arrays if a is not None]
    if not present_arrays:
        raise ValueError(f"No mapped OSF channels found at all in {h5_path}")
    n_samples_64 = present_arrays[0].shape[0]

    for i, arr in enumerate(channel_arrays):
        if arr is None:
            channel_arrays[i] = np.zeros(n_samples_64, dtype=np.float32)

    signal_matrix = np.stack(channel_arrays, axis=0)  # [12, n_samples_64]

    n_full_epochs = n_samples_64 // EPOCH_SAMPLES
    if n_full_epochs == 0:
        raise ValueError(
            f"Recording too short ({n_samples_64} samples at 64Hz < 1 epoch) in {h5_path}"
        )

    out = np.empty((n_full_epochs, 2, 768), dtype=np.float32)

    for batch_start in range(0, n_full_epochs, chunk_batch_size):
        batch_end = min(batch_start + chunk_batch_size, n_full_epochs)
        B = batch_end - batch_start

        x = np.empty((B, 12, EPOCH_SAMPLES), dtype=np.float32)
        for bi, ei in enumerate(range(batch_start, batch_end)):
            s = ei * EPOCH_SAMPLES
            e = s + EPOCH_SAMPLES
            x[bi] = signal_matrix[:, s:e]

        x_t = torch.from_numpy(x).to(device)

        with torch.no_grad():
            cls, patches = model.forward_encoding(x_t, return_sequence=False)
            mean_patches = patches.mean(dim=1)  # [B, 768]

        out[batch_start:batch_end, 0, :] = cls.cpu().float().numpy()
        out[batch_start:batch_end, 1, :] = mean_patches.cpu().float().numpy()

    return out.astype(np.float16), fill_info


# ─────────────────────────────────────────────────────────────────────────────
# Subject discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_hdf5_files(hdf5_dir: str, datasets: list, limit: int = None) -> list:
    """Scan {hdf5_dir}/{dataset}/derived/hdf5_signals/*.h5.

    Returns:
        list of (dataset, subject_id, h5_path) tuples
    """
    root = Path(hdf5_dir)
    subjects = []
    for dataset in datasets:
        h5_dir = root / dataset / "derived" / "hdf5_signals"
        if not h5_dir.exists():
            logger.warning(f"HDF5 dir not found, skipping: {h5_dir}")
            continue
        files = sorted(h5_dir.glob("*.h5"))
        for fp in files:
            subject_id = fp.stem
            subjects.append((dataset, subject_id, fp))
        logger.info(f"  {dataset}: {len(files)} HDF5 files found")

    if limit:
        subjects = subjects[:limit]
    return subjects


def slice_subjects(subjects: list, start_idx: int, end_idx: int | None, limit: int | None) -> list:
    """Apply --start-idx / --end-idx / --limit slicing."""
    subjects = subjects[start_idx:end_idx]
    if limit:
        subjects = subjects[:limit]
    return subjects


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract OSF embeddings (Stage 1 Step 1)")
    parser.add_argument("--config",      required=True, help="Path to phase0_osf_config.yaml")
    parser.add_argument("--datasets",    nargs="+",     help="Override datasets list from config")
    parser.add_argument("--limit",       type=int,      help="Process only first N subjects (debug)")
    parser.add_argument("--start-idx",   type=int,      default=0,    help="First subject index (for parallel jobs)")
    parser.add_argument("--end-idx",     type=int,      default=None, help="Last subject index exclusive (for parallel jobs)")
    parser.add_argument("--no-skip",     action="store_true", help="Re-extract even if .npy exists")
    parser.add_argument("--cpu",         action="store_true", help="Force CPU (debugging only)")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    emb_cfg  = cfg["embedding"]
    data_cfg = cfg["data"]

    datasets         = args.datasets or emb_cfg["datasets"]
    output_dir       = Path(emb_cfg["output_dir"])
    chunk_batch_size = emb_cfg.get("chunk_batch_size", 16)
    hdf5_dir         = data_cfg["hdf5_dir"]
    cfg_candidates   = data_cfg["channel_candidates"]

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    # ── Create output dirs ────────────────────────────────────────────────────
    for ds in datasets:
        (output_dir / ds).mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model, meta = load_model(emb_cfg["checkpoint_dir"], device)

    # ── Discover subjects ─────────────────────────────────────────────────────
    logger.info(f"Scanning HDF5 files in: {hdf5_dir}")
    all_subjects = find_hdf5_files(hdf5_dir, datasets, limit=None)
    subjects = slice_subjects(all_subjects, args.start_idx, args.end_idx, args.limit)
    logger.info(
        f"Total available: {len(all_subjects)} | "
        f"This job: [{args.start_idx}:{args.end_idx}] = {len(subjects)} subjects"
    )

    # ── Extraction loop ───────────────────────────────────────────────────────
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
            channel_candidates = build_channel_candidates(dataset, cfg_candidates)
            emb, fill_info = extract_subject_embeddings(
                h5_path=h5_path,
                dataset=dataset,
                model=model,
                device=device,
                chunk_batch_size=chunk_batch_size,
                channel_candidates=channel_candidates,
            )
            np.save(out_path, emb)

            if dataset not in fill_log_handles:
                log_path = output_dir / dataset / "_channel_fill_log.jsonl"
                fill_log_handles[dataset] = open(log_path, "a")
            fill_log_handles[dataset].write(
                json.dumps({"subject_id": subject_id, **fill_info}) + "\n"
            )
            fill_log_handles[dataset].flush()

            elapsed = time.time() - t_sub
            n_ok += 1

            if (i + 1) % 50 == 0 or args.limit:
                logger.info(
                    f"[{i+1}/{len(subjects)}] {dataset}/{subject_id} "
                    f"→ shape {emb.shape} in {elapsed:.1f}s "
                    f"(zero-filled: {fill_info['zero_filled']}, "
                    f"fallback: {fill_info['fallback_used']})"
                )

        except Exception as exc:
            logger.error(f"  FAILED {dataset}/{subject_id}: {exc}")
            n_err += 1

        # Graceful stop: SIGTERM was received — exit after this subject
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
    )


if __name__ == "__main__":
    main()
