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

import json
import os
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


# ─────────────────────────────────────────────────────────────────────────────
# Raw signal cache (Phase 2/Stage 2 — plan §15.3) — precomputed once, offline,
# reused by every Stage 2 training/inference job instead of re-reading the
# raw HDF5 + re-resampling on every __getitem__.
#
# Unlike OSF's raw signal cache (osf_channel_loader.save_signal_cache/
# load_signal_cache — one fixed-shape [12, n_samples_64] matrix per subject,
# since every subject has the same 12 channels at the same rate, zero-filled
# if actually missing), PhysioOmni's channels are genuinely PRESENT-OR-ABSENT
# per subject (never zero-filled at this stage — see load_subject_signals'
# docstring), channel COUNT varies (EEG: 1 or 2), and different modalities
# run at different native rates (EEG/EOG 200Hz, ECG/EMG 500Hz) — no single
# fixed-shape array can hold all of that. Instead: one small .npy per PRESENT
# channel-slot, plus a meta.json — directly mirrors load_subject_signals()'s
# own {modality: [(label, arr), ...]} return structure, persisted, so
# building/reading the cache needs no new data-shape design.
# ─────────────────────────────────────────────────────────────────────────────

# (modality, label) -> cache filename. Fixed order matters: iterating this
# dict in insertion order reproduces load_subject_signals()' own EEG
# ordering (C3 before C4) when reconstructing the signals dict on load.
_CACHE_SLOT_FILENAMES = {
    ("EEG", "C3"):  "EEG_C3.npy",
    ("EEG", "C4"):  "EEG_C4.npy",
    ("EOG", "HEO"): "EOG_HEO.npy",
    ("ECG", "ECG"): "ECG.npy",
    ("EMG", "EMG"): "EMG.npy",
}


def cache_subject_dir(cache_dir, dataset: str, subject_id: str) -> Path:
    """Path convention for one subject's cached signals:
    {cache_dir}/{dataset}/{subject_id}/ (a directory, not a single file —
    holds one .npy per present channel-slot plus meta.json)."""
    return Path(cache_dir) / dataset / subject_id


def save_signal_cache(cache_dir, dataset: str, subject_id: str,
                       signals: dict, fill_info: dict) -> int:
    """Persist one subject's already-loaded/resampled/denormalized signals
    (load_subject_signals()' own return value) to the cache, as float16
    (halves storage; PhysioOmni-ready signal is not raw-amplitude-precision-
    sensitive at the ~3-decimal-digit level, same precedent as Stage 1's
    own float16 embedding storage and OSF's float16 raw-signal cache).

    Returns t_epochs (min epoch count across present channels — the same
    `min(...)` computation extract_physioomni_embeddings.py's
    extract_subject_embeddings() already does), also written into meta.json
    so shape lookups never need to open an array (cheaper than even OSF's
    own get_cached_epoch_count, which still does a shape-only mmap read).
    """
    subj_dir = cache_subject_dir(cache_dir, dataset, subject_id)
    subj_dir.mkdir(parents=True, exist_ok=True)

    epoch_counts = []
    for modality, chans in signals.items():
        for label, arr in chans:
            fname = _CACHE_SLOT_FILENAMES.get((modality, label))
            if fname is None:
                raise ValueError(
                    f"Unexpected (modality, label) from load_subject_signals: "
                    f"({modality!r}, {label!r}) — not in _CACHE_SLOT_FILENAMES."
                )
            np.save(subj_dir / fname, arr.astype(np.float16))
            epoch_counts.append(len(arr) // (EPOCH_SECONDS * NATIVE_HZ[modality]))

    t_epochs = min(epoch_counts) if epoch_counts else 0
    meta = {"t_epochs": t_epochs, **fill_info}
    # Atomic write (temp file + rename) — a worker killed mid-write (SIGTERM,
    # OOM) must never leave a truncated/empty meta.json sitting where
    # cache_exists() would treat it as "done": found live 2026-08-20, several
    # precompute jobs were OOM-killed / SIGTERM'd, leaving 81 zero-byte
    # meta.json files (all with fully-written, valid .npy siblings — only
    # the plain `open(...,"w")` + json.dump was non-atomic) that then broke
    # training with a JSONDecodeError. Same fix pattern already used
    # elsewhere in this project for exactly this risk (e.g.
    # infer_osf_lora_subject_windows.py's _save_resume_checkpoint).
    meta_path = subj_dir / "meta.json"
    tmp_path = meta_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(meta, f)
    os.replace(tmp_path, meta_path)
    return t_epochs


def load_meta(cache_dir, dataset: str, subject_id: str) -> dict:
    """Read one subject's meta.json (t_epochs + fill_info) — no array
    touched at all, cheapest possible shape/fill-info lookup."""
    subj_dir = cache_subject_dir(cache_dir, dataset, subject_id)
    with open(subj_dir / "meta.json") as f:
        return json.load(f)


def get_cached_t_epochs(cache_dir, dataset: str, subject_id: str) -> int:
    """Fast epoch-count lookup for a cached subject — reads only meta.json."""
    return load_meta(cache_dir, dataset, subject_id)["t_epochs"]


def load_signal_cache(cache_dir, dataset: str, subject_id: str) -> dict:
    """Reconstruct one subject's signals dict — SAME shape
    load_subject_signals() returns ({"EEG": [(label, arr), ...], "EOG": [...],
    "ECG": [...], "EMG": [...]}) — from the cache, instead of re-reading the
    raw HDF5. Arrays are read fully into memory as float32 (matching
    load_subject_signals' own return dtype) — same design as OSF's
    load_signal_cache: the win is avoiding repeated HDF5-read + channel-
    mapping + FFT-resample work across every window of the same subject,
    not out-of-core streaming, so a full per-subject read (not a lazy mmap
    slice) is the right tradeoff here, same as OSF's own precedent.
    """
    subj_dir = cache_subject_dir(cache_dir, dataset, subject_id)
    signals = {"EEG": [], "EOG": [], "ECG": [], "EMG": []}
    for (modality, label), fname in _CACHE_SLOT_FILENAMES.items():
        p = subj_dir / fname
        if p.exists():
            arr = np.load(p).astype(np.float32)
            signals[modality].append((label, arr))
    return signals


def cache_exists(cache_dir, dataset: str, subject_id: str) -> bool:
    """Whether a subject's cache is complete enough to use — meta.json is
    written last by save_signal_cache, so its presence implies every
    channel .npy for that subject was already written successfully."""
    return (cache_subject_dir(cache_dir, dataset, subject_id) / "meta.json").exists()
