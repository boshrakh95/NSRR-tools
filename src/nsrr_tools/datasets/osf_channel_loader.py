"""osf_channel_loader.py — shared OSF 12-channel loading/resampling utility.

Factored out of scripts/extract_osf_embeddings.py (checklist item 2.2,
docs/TSFM_OSF_IMPLEMENTATION_PLAN.md Appendix §6.0) so Stage 1 (precomputed
embeddings) and Stage 2 (LoRA fine-tuning, raw signal loaded live every
training step) share exactly one implementation of channel mapping and
resampling, rather than risking the two drifting out of sync if this logic
is ever revisited (e.g. docs/OSF_CHANNEL_REPROCESSING_PLAN.md).

Used by:
  - scripts/extract_osf_embeddings.py       (Stage 1, precompute)
  - scripts/precompute_osf_raw_signal_cache.py  (Stage 2, offline precompute)
  - src/nsrr_tools/datasets/osf_raw_epoch_dataset.py  (Stage 2, reads the cache)
  - scripts/infer_osf_lora_subject_windows.py         (Stage 2, reads the cache)

RAW SIGNAL CACHE (added 2026-08-14, docs/TSFM_OSF_IMPLEMENTATION_PLAN.md
checklist 2.5b): load_and_resample_channels() against the ORIGINAL raw
HDF5 is expensive (per-channel HDF5 read + resample) and was found to be
the dominant cost of every Stage 2 training job — repeated from scratch on
every __getitem__, for every task/head/context/job, since the resampled
signal doesn't depend on any of those. save_signal_cache/load_signal_cache/
get_cached_epoch_count below read/write a precomputed per-subject cache
(built once, offline, by precompute_osf_raw_signal_cache.py) instead of
touching the raw HDF5 at all during training/inference.
"""

import h5py
import numpy as np

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


def load_and_resample_channels(
    h5_path,
    channel_candidates: dict,
    channel_order: list = OSF_CHANNEL_ORDER,
) -> tuple[np.ndarray, dict]:
    """Load OSF's 12 mapped channels from one subject's full-channel HDF5,
    resampled to 64Hz, zero-filling any missing channel.

    Args:
        h5_path: path to the subject's full-channel HDF5 file.
        channel_candidates: dict from build_channel_candidates() — OSF
            channel name -> ordered list of our-HDF5 candidate names.
        channel_order: fixed OSF channel order (default OSF_CHANNEL_ORDER).

    Returns:
        signal_matrix : np.ndarray [12, n_samples_64] float32
        fill_info     : dict — {"zero_filled": [...], "fallback_used": {...}}
                        for the per-subject channel-fill log.
    """
    with h5py.File(h5_path, "r") as hf:
        available = set(hf.keys())

        channel_arrays = []
        fill_info = {"zero_filled": [], "fallback_used": {}}
        for osf_ch in channel_order:
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
    return signal_matrix, fill_info


def get_epoch_count(h5_path) -> int:
    """Fast metadata-only read of how many complete 30s/64Hz epochs a
    subject's HDF5 contains, without loading any channel data. Used to
    build Stage 2's raw-signal shape cache (analogous to Stage 1's
    shape_cache.json over precomputed .npy files, but self-contained —
    doesn't depend on Stage 1 extraction having run for a given subject).
    Assumes all channels in the file share the same sample count at 128Hz
    (same assumption load_and_resample_channels relies on).

    NOTE: superseded by get_cached_epoch_count() for any subject whose
    raw-signal cache has been precomputed (checklist 2.5b) — this
    HDF5-scanning version is only used by precompute_osf_raw_signal_cache.py
    itself (building the cache in the first place) and as documentation of
    what the cache's shape means. OSFRawEpochWindowDataset no longer calls
    this directly.
    """
    with h5py.File(h5_path, "r") as hf:
        keys = list(hf.keys())
        if not keys:
            return 0
        n_samples_128 = hf[keys[0]].shape[0]
    n_samples_64 = n_samples_128 // 2
    return n_samples_64 // EPOCH_SAMPLES


# ─────────────────────────────────────────────────────────────────────────────
# Raw signal cache (checklist 2.5b) — precomputed once, offline, reused by
# every Stage 2 training/inference job instead of re-reading the raw HDF5.
# ─────────────────────────────────────────────────────────────────────────────

def cache_path_for(cache_dir, dataset: str, subject_id: str):
    """Path convention for one subject's cached signal: {cache_dir}/{dataset}/{subject_id}.npy"""
    from pathlib import Path
    return Path(cache_dir) / dataset / f"{subject_id}.npy"


def save_signal_cache(cache_path, signal_matrix: np.ndarray) -> None:
    """Save a resampled [12, n_samples_64] float32 signal matrix as float16
    (halves storage; the underlying HDF5 signal is already z-scored/
    normalized — confirmed no raw-amplitude scaling happens anywhere in
    load_and_resample_channels — so float16's ~3-decimal-digit precision is
    not a meaningful loss, same precedent as Stage 1's own float16
    embedding storage)."""
    from pathlib import Path
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, signal_matrix.astype(np.float16))


def load_signal_cache(cache_path) -> np.ndarray:
    """Load a cached signal matrix, memory-mapped (cheap repeated slicing,
    no full-array copy until actually sliced — same benefit the per-worker
    single-subject cache in OSFRawEpochWindowDataset relies on), cast back
    up to float32 for downstream math (matches load_and_resample_channels'
    return dtype, so callers don't need to know which source they read
    from)."""
    arr = np.load(cache_path, mmap_mode="r")
    return arr.astype(np.float32)


def get_cached_epoch_count(cache_path) -> int:
    """Fast shape-only read of a cached signal file's epoch count — no
    raw HDF5 scan needed once the cache exists. Reads via mmap (lazy;
    does not load the array's data into memory just to check its shape)."""
    arr = np.load(cache_path, mmap_mode="r")
    n_samples_64 = arr.shape[1]
    return n_samples_64 // EPOCH_SAMPLES
