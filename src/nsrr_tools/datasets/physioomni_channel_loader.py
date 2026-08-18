"""physioomni_channel_loader.py — shared PhysioOmni channel-loading/
resampling/normalization utility.

Mirrors the design of `osf_channel_loader.py` (built for OSF, checklist
item 2.2 of that implementation) — factored out as a shared module *from
the start* (not after-the-fact like OSF's own refactor) so Stage 1
(precomputed embeddings) and any future Stage 2 (LoRA fine-tuning, raw
signal loaded live) never risk drifting out of sync. See
docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md §7 for the design this
implements.

Used by:
  - scripts/extract_physioomni_embeddings.py   (Stage 1, precompute)
  - (future) any Stage 2 raw-signal loader, per the plan's Phase 2 outline

CHANNEL MAPPING (plan §4.2, §4.5)
──────────────────────────────────
PhysioOmni branch -> our fast-channel HDF5 source:
  EEG  : up to 2 channels, C3 (<- C3-M2) and C4 (<- C4-M1), each fed
         independently. SHHS is a special case (plan §4.5, FINAL DECISION,
         not duplication): SHHS has no C3-M2/C4-M1 at all, only a single
         generic 'EEG' channel — mapped to the C3 slot only, with the C4
         slot entirely OMITTED (not zero-filled, not duplicated) so SHHS's
         EEG branch naturally gets 1 real channel of tokens instead of 2.
         This is architecturally legitimate for PhysioOmni specifically
         (unlike OSF's fixed-tensor ViT): NeuralTransformer takes a
         variable-length token sequence per modality, so a 1-channel EEG
         branch is a real, supported input, not a workaround.
  EOG  : 1 derived channel, HEO = LOC - ROC (matches
         prepare_HMC_downstream.py's own derivation exactly). Needs BOTH
         LOC and ROC present; if either is missing, the whole EOG branch
         is absent for that subject.
  ECG  : 1 channel, EKG -> fallback ECG-L.
  EMG  : 1 channel, CHIN -> fallback generic EMG.

SAMPLE RATE (plan §5.1)
────────────────────────
Our HDF5s are uniformly 128Hz. PhysioOmni's own reference prep scripts
resample EEG/EOG to 200Hz and ECG/EMG to 500Hz before patchifying. No
exact-decimation shortcut exists for these ratios (unlike OSF's clean
128->64Hz 2:1 case) — FFT-based resampling (scipy.signal.resample) is used.

NORMALIZATION (plan §5.2, revised 2026-08-18)
───────────────────────────────────────────────
PhysioOmni expects raw amplitude scaled by /100 (µV/100), not z-scored
input. Every HDF5 stores per-channel `normalization_stats` (mean/std/min/
max of the signal immediately before z-scoring) — inverting the z-score
recovers that original scale. Traced signal_processor.py directly: there
is NO explicit unit conversion anywhere in it, so the unit MNE's raw EDF
reader returns is whatever the source EDF file's own header declares —
confirmed empirically to be file/cohort-dependent, NOT a fixed per-
channel-name rule (APPLES's ECG is µV-scale; SHHS's ECG is volts-scale,
same canonical channel name, different unit). invert_normalization() below
self-calibrates per channel using that channel's own stored std, rather
than a hardcoded per-channel-name table.
"""

from pathlib import Path

import h5py
import numpy as np
from scipy.signal import resample as scipy_resample

# ── Constants ─────────────────────────────────────────────────────────────────
EPOCH_SECONDS = 30

# PhysioOmni's own native per-modality resample target (plan §5.1).
NATIVE_HZ = {"EEG": 200, "EOG": 200, "ECG": 500, "EMG": 500}

# PhysioOmni's own per-modality patch length in samples (plan §3's table).
PATCH_SAMPLES = {"EEG": 200, "EOG": 100, "ECG": 100, "EMG": 100}

# PhysioOmni branch -> our fast-channel HDF5 candidate key(s), priority order.
# EEG has two independent slots (C3, C4); EOG/ECG/EMG have exactly one each.
# SHHS overrides this via build_channel_candidates() below (plan §4.5).
DEFAULT_CHANNEL_CANDIDATES = {
    "EEG_C3": ["C3-M2"],
    "EEG_C4": ["C4-M1"],
    "EOG_LOC": ["LOC"],
    "EOG_ROC": ["ROC"],
    "ECG": ["EKG", "ECG-L"],
    "EMG": ["CHIN", "EMG"],
}

# Which PhysioOmni slots belong to which modality branch, and the
# position-embedding label PhysioOmni's own standard_1020 vocabulary
# expects for each (plan §4.1 — position-embedding lookup key, not a
# literal referencing claim; matches prepare_HMC_downstream.py's own
# reference-suffix-stripping pattern).
SLOT_MODALITY = {
    "EEG_C3": "EEG", "EEG_C4": "EEG",
    "EOG_LOC": "EOG", "EOG_ROC": "EOG",   # consumed together, see derive_eog_heo()
    "ECG": "ECG",
    "EMG": "EMG",
}
SLOT_LABEL = {"EEG_C3": "C3", "EEG_C4": "C4", "ECG": "ECG", "EMG": "EMG"}


# ─────────────────────────────────────────────────────────────────────────────
# Channel candidates (per-dataset, handles SHHS's single-EEG-channel case)
# ─────────────────────────────────────────────────────────────────────────────

def build_channel_candidates(dataset: str, cfg_candidates: dict = None) -> dict:
    """PhysioOmni slot -> ordered list of our-HDF5 candidate names, for one
    dataset.

    SHHS special case (plan §4.5, FINAL DECISION — not duplication): no
    C3-M2/C4-M1 exists in SHHS's HDF5s, only a single generic 'EEG'
    channel. Map EEG_C3's candidates to ['EEG'] and DROP EEG_C4 entirely
    (no key at all) — the caller (load_subject_signals) naturally skips
    absent slots, so SHHS ends up with exactly one real EEG channel of
    tokens, not two duplicated ones.
    """
    base = cfg_candidates if cfg_candidates is not None else DEFAULT_CHANNEL_CANDIDATES
    candidates = {k: list(v) for k, v in base.items()}
    if dataset == "shhs":
        candidates["EEG_C3"] = ["EEG"]
        candidates.pop("EEG_C4", None)
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# Normalization inversion (plan §5.2)
# ─────────────────────────────────────────────────────────────────────────────

def invert_normalization(x_zscored: np.ndarray, stats: dict) -> np.ndarray:
    """Undo signal_processor.py's z-score normalization, self-calibrate to
    microvolts, then apply PhysioOmni's own /100 convention.

    Args:
        x_zscored: the raw HDF5 array for one channel (already z-scored).
        stats: that channel's entry from the HDF5's normalization_stats
            JSON attribute — {"mean": ..., "std": ..., "min": ..., "max": ...}.

    Returns:
        float32 array, PhysioOmni-ready scale (uV / 100).
    """
    x = x_zscored.astype(np.float32) * float(stats["std"]) + float(stats["mean"])
    if abs(stats["std"]) < 1.0:
        # This channel's pre-normalization scale was volts (MNE's default
        # for this source file), not already microvolts — convert.
        # Confirmed empirically: this is file/cohort-dependent, not a
        # fixed per-channel-name rule (plan §5.2).
        x = x * 1e6
    return x / 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Resampling (plan §5.1)
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_native_hz(x: np.ndarray, source_hz: float, target_hz: int) -> np.ndarray:
    """FFT-based resample (no exact-decimation shortcut exists for
    128->200/128->500, unlike OSF's clean 128->64 2:1 case)."""
    n_target = int(round(len(x) * target_hz / source_hz))
    return scipy_resample(x, n_target).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject loading
# ─────────────────────────────────────────────────────────────────────────────

def load_subject_signals(h5_path, channel_candidates: dict) -> tuple[dict, dict]:
    """Load, denormalize, and resample every PhysioOmni-relevant channel
    for one subject.

    Args:
        h5_path: path to the subject's fast-channel HDF5.
        channel_candidates: from build_channel_candidates() — PhysioOmni
            slot -> ordered list of our-HDF5 candidate names.

    Returns:
        signals: {"EEG": [("C3", arr_200hz), ("C4", arr_200hz)],  # 1 or 2 entries
                  "EOG": [("HEO", arr_200hz)],                     # 0 or 1 entry
                  "ECG": [("ECG", arr_500hz)],                     # 0 or 1 entry
                  "EMG": [("EMG", arr_500hz)]}                     # 0 or 1 entry
            Each array is float32, PhysioOmni-ready scale, resampled to
            that modality's native rate. A modality with an empty list is
            entirely absent for this subject (caller zero-fills, plan §6.3).
        fill_info: {"slots_found": {...}, "slots_missing": [...],
                    "fallback_used": {...}, "eeg_channel_count": 1 or 2}
            for the per-subject channel-fill log (plan §6.3 item 3).
    """
    fill_info = {"slots_found": {}, "slots_missing": [], "fallback_used": {}}

    with h5py.File(h5_path, "r") as hf:
        available = set(hf.keys())
        import json
        norm_stats = json.loads(hf.attrs["normalization_stats"])
        source_hz = float(hf.attrs.get("sampling_rate", 128))

        def _load_slot(slot: str) -> np.ndarray | None:
            for i, key in enumerate(channel_candidates.get(slot, [])):
                if key in available:
                    if i > 0:
                        fill_info["fallback_used"][slot] = key
                    fill_info["slots_found"][slot] = key
                    x = invert_normalization(hf[key][:], norm_stats[key])
                    modality = SLOT_MODALITY[slot]
                    return resample_to_native_hz(x, source_hz, NATIVE_HZ[modality])
            fill_info["slots_missing"].append(slot)
            return None

        eeg_c3 = _load_slot("EEG_C3")
        eeg_c4 = _load_slot("EEG_C4") if "EEG_C4" in channel_candidates else None
        loc = _load_slot("EOG_LOC")
        roc = _load_slot("EOG_ROC")
        ecg = _load_slot("ECG")
        emg = _load_slot("EMG")

    eeg = []
    if eeg_c3 is not None:
        eeg.append(("C3", eeg_c3))
    if eeg_c4 is not None:
        eeg.append(("C4", eeg_c4))
    fill_info["eeg_channel_count"] = len(eeg)

    eog = []
    if loc is not None and roc is not None:
        n = min(len(loc), len(roc))
        eog.append(("HEO", loc[:n] - roc[:n]))
    elif (loc is not None) != (roc is not None):
        # One present, one absent -- derivation needs both (plan §4.2).
        # Not a silently-degraded case: log it distinctly from "both missing".
        fill_info["slots_missing"].append("EOG_derivation_incomplete")

    signals = {
        "EEG": eeg,
        "EOG": eog,
        "ECG": [("ECG", ecg)] if ecg is not None else [],
        "EMG": [("EMG", emg)] if emg is not None else [],
    }
    return signals, fill_info


# ─────────────────────────────────────────────────────────────────────────────
# Patchification
# ─────────────────────────────────────────────────────────────────────────────

def chunk_into_patches(x: np.ndarray, patch_samples: int) -> np.ndarray:
    """[n_samples] -> [n_patches, patch_samples], dropping any incomplete
    trailing patch (same convention as OSF's epoch-chunking)."""
    n_patches = len(x) // patch_samples
    return x[: n_patches * patch_samples].reshape(n_patches, patch_samples)


def get_epoch_count(h5_path, source_hz_attr: str = "sampling_rate") -> int:
    """Fast metadata-only read of how many complete 30s epochs a subject's
    HDF5 contains, without loading channel data. Mirrors OSF's
    get_epoch_count() — used to build a dataset-class shape cache.
    Assumes all channels in the file share the same sample count at the
    file's native rate (same assumption load_subject_signals relies on).
    """
    with h5py.File(h5_path, "r") as hf:
        keys = [k for k in hf.keys()]
        if not keys:
            return 0
        n_samples = hf[keys[0]].shape[0]
        source_hz = float(hf.attrs.get(source_hz_attr, 128))
    return int(n_samples / source_hz / EPOCH_SECONDS)
