#!/usr/bin/env python3
"""
extract_physioomni_embeddings.py — PhysioOmni baseline, Stage 1 Step 2

Extracts per-30-second-epoch PhysioOmni embeddings from the fast-channel
HDF5 PSG files and saves one numpy array per subject. Mirrors
scripts/extract_osf_embeddings.py's structure — see
docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md §6.3/§7 for the full
derivation of every choice below.

WHY THIS LOOKS DIFFERENT FROM extract_osf_embeddings.py
─────────────────────────────────────────────────────────
OSF is one ViT with a fixed 12-channel input tensor and one CLS output.
PhysioOmni is four INDEPENDENT encoders (EEG/EOG/ECG/EMG), each taking a
variable-length token sequence, each producing its own CLS output — there
is no unified fusion model in the released checkpoint (plan §3). This
script runs each modality's own encoder separately per subject and
concatenates the four CLS vectors into one flat embedding.

OUTPUT FORMAT
─────────────
  {output_dir}/{dataset}/{subject_id}.npy
  dtype  : float16
  shape  : [T, 500]
    T   = number of complete 30s epochs, computed as the MINIMUM epoch
          count across whichever modalities are actually present for this
          subject (all real channels in one HDF5 share the same raw
          128Hz sample count, so this is normally exact, not approximate)
    500 = 200 (EEG CLS) + 100 (EOG CLS) + 100 (ECG CLS) + 100 (EMG CLS),
          concatenated in that fixed order. A modality with zero real
          channels for this subject has its slice zero-filled (logged).

CHANNEL MAPPING / RESAMPLING / NORMALIZATION
─────────────────────────────────────────────
All delegated to nsrr_tools.datasets.physioomni_channel_loader (plan §7),
shared with any future Stage 2 raw-signal loader. SHHS's single generic
'EEG' channel is fed as ONE real EEG channel (not duplicated into two,
not zero-filled) — plan §4.5's final decision, verified working in
scripts/test_physioomni_channel_loader.py.

USAGE
─────
  python extract_physioomni_embeddings.py --config configs/phase0_physioomni_config.yaml
  python extract_physioomni_embeddings.py --config configs/phase0_physioomni_config.yaml \\
      --datasets apples --limit 5 --cpu
  python extract_physioomni_embeddings.py --config configs/phase0_physioomni_config.yaml \\
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
from nsrr_tools.datasets.physioomni_channel_loader import (  # noqa: E402
    EPOCH_SECONDS,
    NATIVE_HZ,
    PATCH_SAMPLES,
    build_channel_candidates,
    load_subject_signals,
)

# ── Graceful-stop flag (set by SIGTERM handler) ───────────────────────────────
# Mirrors extract_osf_embeddings.py's pattern (finish current subject, then
# stop) — no per-subject resume checkpoint here either, only the
# out_path.exists() skip-logic below.
_stop_requested = False

def _handle_sigterm(signum, frame):
    global _stop_requested
    logger.warning("[SIGTERM] Stop requested — will exit after current subject completes.")
    _stop_requested = True

signal.signal(signal.SIGTERM, _handle_sigterm)

# ── PhysioOmni imports ────────────────────────────────────────────────────────
# PhysioOmni is not installed as a package; add the sibling repo to path.
# Mirrors extract_osf_embeddings.py's exact _REPO computation (3 .parent
# calls from this script -> the directory CONTAINING NSRR-tools-omni,
# i.e. /home/boshra95/ regardless of this worktree's own folder name).
_REPO = Path(__file__).resolve().parent.parent.parent / "PhysioOmni"
sys.path.insert(0, str(_REPO))

try:
    from model.neural_transformer import NeuralTransformer, NTConfig
    from dataset import standard_1020
except ImportError as e:
    logger.error(
        f"Cannot import PhysioOmni: {e}\n"
        f"Expected repo at: {_REPO}\n"
        "Run with physioomni_env (not osf_env/sleepfm_env)."
    )
    sys.exit(1)

MODALITIES = ["EEG", "EOG", "ECG", "EMG"]
MODALITY_DIM = {"EEG": 200, "EOG": 100, "ECG": 100, "EMG": 100}
FLAT_DIM = sum(MODALITY_DIM.values())  # 500
# Fixed concatenation order + offset into the flat 500-dim output vector.
SLOT_RANGE = {}
_off = 0
for _m in MODALITIES:
    SLOT_RANGE[_m] = (_off, _off + MODALITY_DIM[_m])
    _off += MODALITY_DIM[_m]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading — mirrors verify_physioomni_checkpoint.py's already-verified
# loading logic exactly (checklist 0.2), not re-derived here.
# ─────────────────────────────────────────────────────────────────────────────

def load_models(checkpoint_path: str, device: torch.device) -> dict:
    """Load all 4 frozen NeuralTransformer encoders from PhysioOmni.pt.

    Returns:
        {"EEG": encoder, "EOG": encoder, "ECG": encoder, "EMG": encoder},
        each in eval mode, on device.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_sd = ckpt["model"]

    encoders = {}
    for mod in MODALITIES:
        args = ckpt[f"{mod}_encoder_args"]
        conf = NTConfig(**args)
        enc = NeuralTransformer(conf)

        prefix = f"{mod}_encoder."
        filtered = {k[len(prefix):]: v for k, v in model_sd.items() if k.startswith(prefix)}
        missing, unexpected = enc.load_state_dict(filtered, strict=False)
        if missing:
            raise RuntimeError(f"{mod} encoder has missing keys: {missing}")

        enc.to(device).eval()
        encoders[mod] = enc

    logger.info(
        f"Loaded 4 PhysioOmni encoders: "
        + ", ".join(f"{m}(n_embd={MODALITY_DIM[m]})" for m in MODALITIES)
    )
    return encoders


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction
# ─────────────────────────────────────────────────────────────────────────────

def _modality_forward(
    encoder,
    channel_list: list,
    epoch_start: int,
    n_epochs: int,
    modality: str,
    device: torch.device,
) -> np.ndarray:
    """Run one modality's encoder over `n_epochs` consecutive 30s epochs,
    for all real channels in `channel_list`.

    channel_list: [(label, full_resampled_array), ...] — 1 or 2 entries
        for EEG, exactly 1 for EOG/ECG/EMG (never called with an empty
        list — caller checks that first).

    Returns:
        CLS output, [n_epochs, MODALITY_DIM[modality]], float32.
    """
    native_hz = NATIVE_HZ[modality]
    patch_samples = PATCH_SAMPLES[modality]
    epoch_samples = EPOCH_SECONDS * native_hz
    patches_per_epoch = epoch_samples // patch_samples

    per_channel_patches = []
    chan_ids, time_ids = [], []
    for label, arr in channel_list:
        s = epoch_start * epoch_samples
        e = s + n_epochs * epoch_samples
        seg = arr[s:e].reshape(n_epochs, patches_per_epoch, patch_samples)
        per_channel_patches.append(seg)

        pos_id = standard_1020.index(label)
        chan_ids.extend([pos_id] * patches_per_epoch)
        time_ids.extend(range(patches_per_epoch))  # resets per channel, matches dataset.py's own construction

    x = np.concatenate(per_channel_patches, axis=1)  # [B, n_channels*patches_per_epoch, patch_samples]
    x_t = torch.from_numpy(x).float().to(device)

    input_chans = torch.tensor(chan_ids, dtype=torch.long, device=device).unsqueeze(0).expand(n_epochs, -1)
    input_times = torch.tensor(time_ids, dtype=torch.long, device=device).unsqueeze(0).expand(n_epochs, -1)

    with torch.no_grad():
        cls = encoder.forward_features(x_t, input_chans, input_times, mask=None, return_all_tokens=False)

    return cls.cpu().float().numpy()


def extract_subject_embeddings(
    h5_path: Path,
    dataset: str,
    encoders: dict,
    device: torch.device,
    chunk_batch_size: int,
    channel_candidates: dict,
) -> tuple[np.ndarray, dict]:
    """Extract [T, 500] float16 embeddings for one subject.

    Processing pipeline (docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md §6.3):
      1. Load + denormalize + resample every PhysioOmni-relevant channel
         (nsrr_tools.datasets.physioomni_channel_loader.load_subject_signals).
      2. T = min complete-30s-epoch count across whichever modalities are
         present (all channels in one HDF5 share the same raw 128Hz
         sample count, so this is normally exact).
      3. For each modality with >=1 real channel: batch chunk_batch_size
         epochs at a time through that modality's own frozen encoder,
         write its CLS output into the corresponding 100/200-dim slice.
         A modality with zero real channels is left zero-filled.

    Returns:
        embeddings : np.ndarray [T, 500], dtype float16
        fill_info  : dict — from load_subject_signals, plus which
                     modalities were entirely zero-filled, for the
                     per-subject fill log.
    """
    signals, fill_info = load_subject_signals(h5_path, channel_candidates)

    epoch_counts = [
        len(arr) // (EPOCH_SECONDS * NATIVE_HZ[modality])
        for modality, chans in signals.items()
        for _, arr in chans
    ]
    if not epoch_counts:
        raise ValueError(f"No PhysioOmni-relevant channels found at all in {h5_path}")
    t_epochs = min(epoch_counts)
    if t_epochs == 0:
        raise ValueError(f"Recording too short (< 1 epoch) in {h5_path}")

    out = np.zeros((t_epochs, FLAT_DIM), dtype=np.float32)
    fill_info["modalities_zero_filled"] = []

    for modality in MODALITIES:
        channel_list = signals[modality]
        start, end = SLOT_RANGE[modality]
        if not channel_list:
            fill_info["modalities_zero_filled"].append(modality)
            continue

        encoder = encoders[modality]
        for batch_start in range(0, t_epochs, chunk_batch_size):
            batch_end = min(batch_start + chunk_batch_size, t_epochs)
            n = batch_end - batch_start
            cls = _modality_forward(encoder, channel_list, batch_start, n, modality, device)
            out[batch_start:batch_end, start:end] = cls

    return out.astype(np.float16), fill_info


# ─────────────────────────────────────────────────────────────────────────────
# Subject discovery — identical to extract_osf_embeddings.py's
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
    parser = argparse.ArgumentParser(description="Extract PhysioOmni embeddings (Stage 1 Step 2)")
    parser.add_argument("--config",      required=True, help="Path to phase0_physioomni_config.yaml")
    parser.add_argument("--datasets",    nargs="+",     help="Override datasets list from config")
    parser.add_argument("--limit",       type=int,      help="Process only first N subjects (debug)")
    parser.add_argument("--start-idx",   type=int,      default=0,    help="First subject index (for parallel jobs)")
    parser.add_argument("--end-idx",     type=int,      default=None, help="Last subject index exclusive (for parallel jobs)")
    parser.add_argument("--no-skip",     action="store_true", help="Re-extract even if .npy exists")
    parser.add_argument("--cpu",         action="store_true", help="Force CPU (debugging only)")
    args = parser.parse_args()

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

    for ds in datasets:
        (output_dir / ds).mkdir(parents=True, exist_ok=True)

    encoders = load_models(emb_cfg["checkpoint_dir"], device)

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
            channel_candidates = build_channel_candidates(dataset, cfg_candidates)
            emb, fill_info = extract_subject_embeddings(
                h5_path=h5_path,
                dataset=dataset,
                encoders=encoders,
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
                    f"(eeg_channels: {fill_info['eeg_channel_count']}, "
                    f"zero-filled: {fill_info['modalities_zero_filled']})"
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
    )


if __name__ == "__main__":
    main()
