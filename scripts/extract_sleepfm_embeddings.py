#!/usr/bin/env python3
"""
extract_sleepfm_embeddings.py — Phase 0, Step 1

Extracts per-5-second SleepFM embeddings from preprocessed HDF5 PSG files
and saves one numpy array per subject for all downstream context-length experiments.

OUTPUT FORMAT
─────────────
  {output_dir}/{dataset}/{subject_id}.npy
  dtype  : float16
  shape  : [T, 4, 128]
    T   = floor(n_samples / patch_size)  — total 5-sec patches in the recording
    4   = SleepFM modalities in order: BAS, RESP, EKG, EMG
    128 = SleepFM embed_dim

  Incomplete trailing signal (< 5 min chunk) is dropped because the
  model requires full 5-min chunks (38 400 samples at 128 Hz).

DESIGN CHOICES
──────────────
• One file per subject (full night) — context windows are sliced at training
  time, so we never re-run the (expensive) encoder for different window sizes.
• GPU batching over chunks — for each modality we stack B=chunk_batch_size
  chunks into one forward pass, giving high GPU utilisation even for small
  recordings.
• Modality channel groups come from our modality_groups.yaml (not from
  SleepFM's channel_groups.json) because our HDF5 files already use
  standardised channel names (C3-M2, LOC, EKG, CHIN …).

USAGE
─────
  python extract_sleepfm_embeddings.py --config configs/phase0_config.yaml
  python extract_sleepfm_embeddings.py --config configs/phase0_config.yaml \\
      --datasets mros --limit 5
  python extract_sleepfm_embeddings.py --config configs/phase0_config.yaml \\
      --no-skip          # re-extract even if .npy already exists
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from loguru import logger

# ── SleepFM imports ───────────────────────────────────────────────────────────
# The sleepfm-clinical repository is not installed as a package; add to path.
_REPO = Path(__file__).resolve().parent.parent.parent / "sleepfm-clinical"
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "sleepfm"))

try:
    from models.models import SetTransformer
except ImportError as e:
    logger.error(
        f"Cannot import SleepFM: {e}\n"
        f"Expected repo at: {_REPO}\n"
        "Run with sleepfm_env or set SLEEPFM_REPO env var."
    )
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
MODALITY_ORDER = ["BAS", "RESP", "EKG", "EMG"]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_dir: str, device: torch.device):
    """Load frozen model_base SetTransformer.

    Returns:
        model   : SetTransformer in eval mode, on device
        cfg     : dict from config.json (patch_size, embed_dim, max channels …)
    """
    ckpt_path = Path(checkpoint_dir)
    with open(ckpt_path / "config.json") as f:
        cfg = json.load(f)

    model = SetTransformer(
        in_channels=cfg["in_channels"],
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        pooling_head=cfg["pooling_head"],
        dropout=0.0,          # always 0 for inference
    )

    state = torch.load(ckpt_path / "best.pt", map_location=device)["state_dict"]
    # Strip DataParallel 'module.' prefix if present
    if next(iter(state)).startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()

    logger.info(
        f"Loaded model_base: patch_size={cfg['patch_size']}, "
        f"embed_dim={cfg['embed_dim']}, "
        f"num_layers={cfg['num_layers']}"
    )
    return model, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Channel grouping helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_modality_channels(phase0_cfg: dict) -> dict:
    """Return ordered channel lists per modality from the config.

    Example output:
      {
        "BAS":  ["C3-M2", "C4-M1", "LOC", "ROC", ...],   # up to 10
        "RESP": ["Airflow", "Thor", ...],                  # up to 7
        "EKG":  ["EKG", "ECG-L"],                         # up to 2
        "EMG":  ["CHIN", "LLEG", "RLEG", "EMG"],          # up to 4
      }
    """
    priority = phase0_cfg["data"]["channel_priority"]
    max_ch   = phase0_cfg["data"]["max_channels"]
    return {
        mod: priority[mod][: max_ch[mod]]
        for mod in MODALITY_ORDER
    }


def select_channels_from_hdf5(hdf5_keys: list, modality_channels: dict) -> dict:
    """For each modality, pick which of our priority channels exist in this file.

    Returns:
        { modality: [channel_name, ...] }   — only channels present in the file
    """
    available = set(hdf5_keys)
    return {
        mod: [ch for ch in chs if ch in available]
        for mod, chs in modality_channels.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_subject_embeddings(
    h5_path: Path,
    model: "SetTransformer",
    model_cfg: dict,
    modality_channels: dict,
    device: torch.device,
    chunk_batch_size: int,
) -> np.ndarray:
    """Extract [T, 4, 128] float16 embeddings for one subject.

    Processing pipeline:
      1. Load full signal arrays from HDF5 (all at once; signals are float16,
         cast to float32 for the model).
      2. Group channels by SleepFM modality; zero-pad to max_channels.
      3. Batch B consecutive 5-min chunks per GPU forward pass.
      4. Collect e[1] (per-patch embeddings, shape [B, 60, 128]) for each
         modality, store into the output array.

    Args:
        h5_path          : Path to the subject's HDF5 file.
        model            : Frozen SetTransformer (eval, on device).
        model_cfg        : dict from model_base/config.json.
        modality_channels: priority channel lists per modality (from config).
        device           : torch.device.
        chunk_batch_size : Number of 5-min chunks per GPU forward pass.

    Returns:
        np.ndarray of shape [T, 4, 128], dtype float16.
        T = (n_signal_samples // chunk_size) * patches_per_chunk
    """
    chunk_size       = model_cfg["sampling_freq"] * 300  # 128 * 300 = 38 400
    patch_size       = model_cfg["patch_size"]           # 640
    patches_per_chunk = chunk_size // patch_size          # 60
    max_ch           = {
        "BAS":  model_cfg["BAS_CHANNELS"],
        "RESP": model_cfg["RESP_CHANNELS"],
        "EKG":  model_cfg["EKG_CHANNELS"],
        "EMG":  model_cfg["EMG_CHANNELS"],
    }

    with h5py.File(h5_path, "r") as hf:
        keys = list(hf.keys())
        n_samples = hf[keys[0]].shape[0]

        # Which of our priority channels are actually in this file?
        present = select_channels_from_hdf5(keys, modality_channels)

        # Load all channels into RAM as float32 (HDF5 data is float16)
        raw: dict[str, np.ndarray] = {}  # { channel_name: (n_samples,) }
        for mod_chs in present.values():
            for ch in mod_chs:
                if ch not in raw:
                    raw[ch] = hf[ch][:].astype(np.float32)

    n_full_chunks = n_samples // chunk_size
    if n_full_chunks == 0:
        raise ValueError(f"Recording too short ({n_samples} samples < 1 chunk)")

    T_total = n_full_chunks * patches_per_chunk
    out = np.empty((T_total, 4, 128), dtype=np.float32)

    # Pre-build constant channel masks per modality (same for every chunk)
    masks: dict[str, torch.Tensor] = {}
    for mod in MODALITY_ORDER:
        C_actual = len(present[mod])
        C_max    = max_ch[mod]
        mask = torch.zeros(1, C_max, dtype=torch.bool, device=device)
        mask[0, C_actual:] = True   # True = padded slot
        masks[mod] = mask

    # Process B chunks at a time, modality by modality
    for batch_start in range(0, n_full_chunks, chunk_batch_size):
        batch_end = min(batch_start + chunk_batch_size, n_full_chunks)
        B = batch_end - batch_start

        for mi, mod in enumerate(MODALITY_ORDER):
            ch_list = present[mod]
            C_actual = len(ch_list)
            C_max    = max_ch[mod]

            # Build (B, C_max, chunk_size) tensor
            x = torch.zeros(B, C_max, chunk_size, dtype=torch.float32)
            for ci_in_batch, ci in enumerate(range(batch_start, batch_end)):
                s = ci * chunk_size
                e = s + chunk_size
                for c_idx, ch in enumerate(ch_list):
                    x[ci_in_batch, c_idx, :] = torch.from_numpy(raw[ch][s:e])

            x = x.to(device)
            mask = masks[mod].expand(B, -1)   # (B, C_max)

            with torch.no_grad():
                _, patch_emb = model(x, mask)  # (B, patches_per_chunk, 128)

            # patch_emb: (B, 60, 128) → store into out[patch_start:patch_end, mi, :]
            patch_start = batch_start * patches_per_chunk
            patch_end   = batch_end   * patches_per_chunk
            pe = patch_emb.cpu().float().numpy()     # (B, 60, 128)
            out[patch_start:patch_end, mi, :] = pe.reshape(-1, 128)

    return out.astype(np.float16)


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
    parser = argparse.ArgumentParser(description="Extract SleepFM embeddings (Phase 0 Step 1)")
    parser.add_argument("--config",      required=True, help="Path to phase0_config.yaml")
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

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    # ── Create output dirs ────────────────────────────────────────────────────
    for ds in datasets:
        (output_dir / ds).mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model, model_cfg = load_model(emb_cfg["checkpoint_dir"], device)

    # ── Build channel priority lists ──────────────────────────────────────────
    modality_channels = build_modality_channels(cfg)
    logger.info("Modality channel priority:")
    for mod, chs in modality_channels.items():
        logger.info(f"  {mod} (max {model_cfg[f'{mod}_CHANNELS']}): {chs}")

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

    for i, (dataset, subject_id, h5_path) in enumerate(subjects):
        out_path = output_dir / dataset / f"{subject_id}.npy"

        if out_path.exists() and not args.no_skip:
            n_skip += 1
            continue

        try:
            t_sub = time.time()
            emb = extract_subject_embeddings(
                h5_path=h5_path,
                model=model,
                model_cfg=model_cfg,
                modality_channels=modality_channels,
                device=device,
                chunk_batch_size=chunk_batch_size,
            )
            np.save(out_path, emb)
            elapsed = time.time() - t_sub
            n_ok += 1

            if (i + 1) % 50 == 0 or args.limit:
                logger.info(
                    f"[{i+1}/{len(subjects)}] {dataset}/{subject_id} "
                    f"→ shape {emb.shape} in {elapsed:.1f}s"
                )

        except Exception as exc:
            logger.error(f"  FAILED {dataset}/{subject_id}: {exc}")
            n_err += 1

    total = time.time() - t0
    logger.info(
        f"\nDone in {total/60:.1f} min — "
        f"extracted: {n_ok}, skipped: {n_skip}, errors: {n_err}"
    )


if __name__ == "__main__":
    main()
