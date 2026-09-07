"""mantis_channel_loader.py — shared Mantis channel-loading + backbone
loading utility.

Built as the shared module from day one (docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md
§7) — the same lesson OSF's and PhysioOmni's own channel loaders already
paid for: OSF originally had this logic inline in its extraction script,
then had to factor it out (and regression-test the refactor) once Stage 2
needed the identical loading path. Building it here from the start means
Stage 1 (`extract_mantis_embeddings.py`) and Stage 2 (a future raw-signal
dataset) never risk drifting out of sync.

Used by:
  - scripts/test_mantis_channel_loader.py     (this file's own smoke test)
  - scripts/extract_mantis_embeddings.py      (Stage 1, not yet written)
  - scripts/verify_mantis_checkpoint.py       (does NOT import from here —
    it predates this module and duplicates its own small loading snippet,
    the same precedent verify_physioomni_checkpoint.py set relative to
    physioomni_channel_loader.py)

CHANNEL MAPPING (plan §2.2) — measured, not assumed
────────────────────────────────────────────────────
Six fixed canonical slots, each with an ordered candidate list. Measured
across 250-subject random samples per cohort (plan §2.1): APPLES/MrOS/STAGES
carry 8 channels with cohort-specific names (`C3-M2`, `C4-M1`, `CHIN`,
`LLEG`…); SHHS carries 6, with a single generic `EEG` channel and a RESP
slot that resolves to `Airflow` for ~75% of subjects and `Thor` for ~25%.
A 7th (second-EEG) slot was deliberately rejected — it would be permanently
absent for SHHS's 8,444 subjects (56% of the population), see plan §2.2.

Unlike OSF/PhysioOmni, NO resampling and NO denormalization are needed:
  - Every HDF5 in the fast-channel tree is already 128 Hz (verified,
    plan §2.1) — Mantis has no native sample rate of its own, it just sees
    an array of floats, so 128 Hz is not a constraint to work around.
  - We feed the stored z-scored values AS-IS (plan §3.2, decided): Mantis's
    own tokenizer path (`TokenGeneratorUnit.ts_scaler`) re-z-scores every
    input series internally, so it is scale-invariant; the only
    scale-sensitive path (the per-patch mean/std `MultiScaledScalarEncoder`)
    has a scale grid centred on the O(1) values night-level z-scoring
    already produces. Restoring raw µV would *introduce* the
    cohort-dependent volts-vs-µV inconsistency PhysioOmni had to fight
    (plan §3.2), not remove one.

An absent slot is filled with exact zeros (not skipped) and logged per
subject — see `load_subject_channels`'s docstring.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

# ── Constants (plan §2.2, §7) ───────────────────────────────────────────────
EPOCH_SECONDS = 30
SOURCE_HZ = 128
EPOCH_SAMPLES = EPOCH_SECONDS * SOURCE_HZ  # 3840

SLOT_ORDER = ["EEG", "EOG_L", "EOG_R", "ECG", "EMG", "RESP"]
N_SLOTS = len(SLOT_ORDER)

# Canonical slot -> our-HDF5 candidate name(s), priority order. Measured
# from real per-cohort channel-key samples (plan §2.1) — no per-dataset
# override is needed (unlike PhysioOmni's SHHS special case): SHHS's
# generic 'EEG' key and MrOS/STAGES's 'CHIN'/'LLEG'/'RLEG' fallbacks are
# already just further entries in the same priority list.
DEFAULT_CHANNEL_CANDIDATES = {
    "EEG": ["C3-M2", "EEG", "C4-M1", "O1-M2"],
    "EOG_L": ["LOC"],
    "EOG_R": ["ROC"],
    "ECG": ["EKG", "ECG-L"],
    "EMG": ["CHIN", "EMG", "LLEG", "RLEG"],
    "RESP": ["Airflow", "Thor", "ABD"],
}

# Mantis architecture constants (plan §1.2, §3.4) — needed to build the
# 240-patch model and to know exactly which checkpoint keys are safe to drop.
_SAFE_TO_DROP = [
    "vit_unit.pos_encoder.pe",  # deterministic sinusoidal buffer, regenerated (§3.4)
    "prj.0.weight", "prj.0.bias",  # pretraining-only projector, dead at inference
    "prj.1.weight", "prj.1.bias",  # AND shape-mismatched in 'combined' mode (§1.0 #6)
]
_ALLOWED_MISSING_AFTER_DROP = {
    "transf_unit.pos_encoder.pe",
    "prj.0.weight", "prj.0.bias", "prj.1.weight", "prj.1.bias",
}
# Mantis-8M's checkpoint lacks these entirely; MantisPlus's carries them
# (plan §1.1) — allowed as EITHER present or missing, never unexpected.
_OPTIONAL_MISSING = {
    "tokgen_unit.scalar_encoders.0.scales",
    "tokgen_unit.scalar_encoders.1.scales",
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject channel loading (plan §2.2, §7)
# ─────────────────────────────────────────────────────────────────────────────

def load_subject_channels(h5_path, candidates: dict | None = None) -> tuple[np.ndarray, dict]:
    """Load all 6 canonical Mantis channel slots for one subject.

    NO resampling (every fast-channel HDF5 is already 128 Hz) and NO
    denormalization (module docstring, plan §3.2) — the stored z-scored
    values are used exactly as they are.

    Args:
        h5_path: path to the subject's fast-channel HDF5
            (.../psg/{dataset}/derived/hdf5_signals/{subject_id}.h5).
        candidates: slot -> ordered candidate list. Defaults to
            DEFAULT_CHANNEL_CANDIDATES.

    Returns:
        x: [6, n_samples] float32 array, SLOT_ORDER order. A slot with no
           resolvable candidate is left as exact zeros (not skipped) — the
           caller decides whether/how to act on that (Stage 1: skip the
           backbone forward for that slot and write zeros to its embedding
           slice, matching plan §2.2's absent-slot contract).
        fill_info: {"slots_found": {slot: key}, "slots_missing": [...],
                    "fallback_used": {slot: key}, "resp_source": str|None}
            `resp_source` is the exact key that filled the RESP slot
            (`Airflow`/`Thor`/`ABD`/None) — turns plan §5.5's sampled
            estimate of SHHS's Airflow-vs-Thor split into an exact
            per-subject, and eventually population-wide, number.
    """
    candidates = candidates if candidates is not None else DEFAULT_CHANNEL_CANDIDATES
    fill_info = {"slots_found": {}, "slots_missing": [], "fallback_used": {}}

    with h5py.File(h5_path, "r") as hf:
        available = set(hf.keys())
        # All channels in one HDF5 share the same raw sample count —
        # verified across all four cohorts (plan §2.1). Read one to get n.
        n_samples = hf[next(iter(hf.keys()))].shape[0]
        x = np.zeros((N_SLOTS, n_samples), dtype=np.float32)

        for i, slot in enumerate(SLOT_ORDER):
            for j, key in enumerate(candidates.get(slot, [])):
                if key in available:
                    if j > 0:
                        fill_info["fallback_used"][slot] = key
                    fill_info["slots_found"][slot] = key
                    x[i, :] = hf[key][:].astype(np.float32)
                    break
            else:
                fill_info["slots_missing"].append(slot)

    fill_info["resp_source"] = fill_info["slots_found"].get("RESP")
    return x, fill_info


def get_epoch_count(h5_path) -> int:
    """Fast metadata-only read of how many complete 30s epochs a subject's
    HDF5 contains, without loading channel data. Mirrors OSF's and
    PhysioOmni's get_epoch_count() — used to build the dataset-class shape
    cache without opening every channel."""
    with h5py.File(h5_path, "r") as hf:
        keys = list(hf.keys())
        if not keys:
            return 0
        n_samples = hf[keys[0]].shape[0]
    return n_samples // EPOCH_SAMPLES


# ─────────────────────────────────────────────────────────────────────────────
# Epoch -> model input (plan §3.1, §13.1)
# ─────────────────────────────────────────────────────────────────────────────

def epochs_to_model_input(x: np.ndarray, windowing: str, epoch_start: int, n_epochs: int) -> torch.Tensor:
    """[6, n_samples] (from load_subject_channels) -> model input tensor for
    `n_epochs` consecutive 30s epochs starting at `epoch_start`.

    `windowing` controls only the TENSOR SHAPE, not the backbone's
    positional buffer (that is `load_mantis_backbone`'s `pe_mode`, a
    separate, orthogonal knob — plan §3.1's two-key split). Option D and
    Option D-interp produce IDENTICAL output here ('full_epoch'); they
    differ only in the backbone's positional buffer.

    Args:
        windowing: 'full_epoch' (Option D / D-interp) or 'subwindow' (Option B).

    Returns:
        'full_epoch' -> float32 tensor [n_epochs * 6, 1, 3840]
        'subwindow'  -> float32 tensor [n_epochs * 6 * 8, 1, 512], after a
                        3840->4096 linear interpolation split into 8 clean
                        512-sample windows per channel-epoch.
        Both are epoch-major, channel-minor: reshape the backbone's output
        to [n_epochs, 6, D] (windowing='full_epoch') or, for 'subwindow',
        mean over the size-8 sub-window axis first, then to [n_epochs, 6, D].
    """
    C = N_SLOTS
    seg = x[:, epoch_start * EPOCH_SAMPLES: (epoch_start + n_epochs) * EPOCH_SAMPLES]
    seg = seg.reshape(C, n_epochs, EPOCH_SAMPLES).transpose(1, 0, 2)  # [n_epochs, 6, 3840]

    if windowing == "full_epoch":
        flat = seg.reshape(n_epochs * C, 1, EPOCH_SAMPLES)
        return torch.from_numpy(np.ascontiguousarray(flat)).float()

    if windowing == "subwindow":
        t = torch.from_numpy(np.ascontiguousarray(seg)).float().reshape(n_epochs * C, 1, EPOCH_SAMPLES)
        t = F.interpolate(t, size=4096, mode="linear", align_corners=False)  # [n_epochs*C, 1, 4096]
        t = t.reshape(n_epochs * C * 8, 1, 512)
        return t

    raise ValueError(f"Unknown windowing: {windowing!r} (expected 'full_epoch' or 'subwindow')")


# ─────────────────────────────────────────────────────────────────────────────
# Backbone loading (plan §3.4) — the one correct way to load a Mantis
# checkpoint at a non-native num_patches. NEVER use `.from_pretrained()` —
# it rebuilds the model from the repo's own config.json (seq_len=512,
# num_patches=32) and hard-raises on the resulting shape mismatch even
# with strict=False (verified empirically, plan §1.0 #2).
# ─────────────────────────────────────────────────────────────────────────────

def sinusoidal_pe(max_len: int, d_model: int, stride: float = 1.0) -> torch.Tensor:
    """Regenerates PositionalEncoding's own sinusoidal buffer exactly
    (stride=1.0 reproduces what the constructor itself builds — this
    function exists so 'interpolate' pe_mode can rescale it, not so
    'extrapolate' needs to call it explicitly).

    stride=1.0                -> Option D: positions 0..max_len-1
    stride=32/num_patches      -> Option D-interp: positions rescaled so the
                                  241 tokens occupy the same arc that 33
                                  tokens occupied in pretraining (plan §3.1).

    Shape matches transformer_v1_utils/positional_encoding.py's own buffer:
    (max_len, 1, d_model).
    """
    import math
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1) * stride
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    return pe


def load_mantis_backbone(
    repo_id_or_dir: str,
    seq_len: int,
    num_patches: int,
    return_transf_layer: int,
    output_token: str,
    device: str = "cpu",
    pe_mode: str = "extrapolate",
):
    """Manually load a Mantis-8M/MantisPlus checkpoint at our own
    `num_patches`, bypassing `.from_pretrained()` entirely (plan §3.4,
    §1.0 #2, #6 — live-verified against both real checkpoints in
    scripts/verify_mantis_checkpoint.py, though that script keeps its own
    inline copy of this logic rather than importing it, matching
    PhysioOmni's own checkpoint-verifier precedent).

    Args:
        repo_id_or_dir: either a HuggingFace repo id (downloads via
            hf_hub_download) or a local directory already containing
            model.safetensors (e.g. /home/boshra95/mantis_checkpoints/Mantis-8M).
        pe_mode: 'extrapolate' (Option D — plain sinusoid at the new length)
            or 'interpolate' (Option D-interp — rescaled sinusoid, plan §3.1).
            Irrelevant when the caller's windowing is 'subwindow' (num_patches
            stays at the pretrained 32 there, so the checkpoint's own buffer
            would already fit — but callers still go through this function
            for the missing-key handling either way).

    Returns:
        MantisV1, eval mode, on `device`.
    """
    from mantis.architecture import MantisV1
    from safetensors.torch import load_file

    net = MantisV1(
        seq_len=seq_len, num_patches=num_patches,
        return_transf_layer=return_transf_layer, output_token=output_token,
        pre_training=False, device=device,
    )

    safetensors_path = Path(repo_id_or_dir) / "model.safetensors"
    if safetensors_path.exists():
        path = str(safetensors_path)
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id_or_dir, "model.safetensors")

    sd = load_file(path, device="cpu")
    for key in _SAFE_TO_DROP:
        sd.pop(key, None)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    missing = set(missing)
    unexpected = set(unexpected)

    if unexpected:
        raise RuntimeError(f"Unexpected keys loading {repo_id_or_dir}: {sorted(unexpected)}")
    bad_missing = missing - _ALLOWED_MISSING_AFTER_DROP - _OPTIONAL_MISSING
    if bad_missing:
        raise RuntimeError(f"Unexpected missing keys loading {repo_id_or_dir}: {sorted(bad_missing)}")

    if pe_mode == "interpolate":
        stride = 32 / num_patches
        net.transf_unit.pos_encoder.pe.copy_(sinusoidal_pe(num_patches + 1, 256, stride=stride))
    elif pe_mode != "extrapolate":
        raise ValueError(f"Unknown pe_mode: {pe_mode!r} (expected 'extrapolate' or 'interpolate')")

    return net.eval().to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 raw-signal cache (plan §7, §14.3)
# ─────────────────────────────────────────────────────────────────────────────
# One fixed-shape [T_epochs, 6, 3840] float16 array per subject, EPOCH-MAJOR
# (T outermost, so one epoch's full 6-channel block — 6*3840 float16 =
# 46,080 bytes — is contiguous). A window of N consecutive epochs is
# therefore one contiguous byte range: 1 open, 1 seek, 1 read (plan §4.9) —
# unlike OSF's channel-major [12, n_samples] cache, which needs 12 strided
# reads for the same window.
#
# meta.json is a SEPARATE small sibling file ({subject_id}.meta.json, not
# embedded in the array), written LAST and only via temp-file + os.replace.
# This is the same real-failure-mode fix PhysioOmni's own cache needed
# (found 2026-08-20 there: SIGTERM/OOM-killed precompute jobs left
# truncated zero-byte meta.json files that a plain `open(...,"w")` would
# have produced, which then broke training with JSONDecodeError) — applied
# here from the start rather than discovered after a real job hits it.
# Because meta.json is written only after the .npy is already fully in
# place, cache_exists() checking meta.json's presence AND that it parses
# is sufficient to guarantee the array write also completed — no need to
# separately verify the (large) array file itself.

def cache_path_for(cache_dir, dataset: str, subject_id: str) -> Path:
    """{cache_dir}/{dataset}/{subject_id}.npy — the raw-signal array path."""
    return Path(cache_dir) / dataset / f"{subject_id}.npy"


def _meta_path_for(cache_dir, dataset: str, subject_id: str) -> Path:
    return Path(cache_dir) / dataset / f"{subject_id}.meta.json"


def save_signal_cache(cache_dir, dataset: str, subject_id: str,
                       x: np.ndarray, meta: dict) -> None:
    """Persist one subject's [T_epochs, 6, 3840] raw-signal array (cast to
    float16) plus its meta.json.

    `meta` is expected to carry at least `t_epochs` (int) — callers
    typically pass `{"t_epochs": x.shape[0], **fill_info}`, mirroring
    `load_subject_channels`'s own fill_info shape (`slots_found`,
    `slots_missing`, `fallback_used`, `resp_source`) plus a `present`
    0/1-per-slot list for Stage 2's zero-fill contract (plan §14.2).

    The array write uses the same atomic temp-file + os.replace() pattern
    already fixed in extract_mantis_embeddings.py (a killed job must never
    leave a truncated .npy sitting where a later read would trust it), and
    meta.json is written only afterward, also atomically — so meta.json's
    presence implies BOTH files are complete, not just itself.
    """
    out_path = cache_path_for(cache_dir, dataset, subject_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = out_path.parent / f"{subject_id}.tmp{os.getpid()}.npy"
    np.save(tmp_path, np.ascontiguousarray(x.astype(np.float16)))
    os.replace(tmp_path, out_path)

    meta_path = _meta_path_for(cache_dir, dataset, subject_id)
    meta_tmp = meta_path.with_suffix(".json.tmp")
    with open(meta_tmp, "w") as f:
        json.dump(meta, f)
    os.replace(meta_tmp, meta_path)


def load_signal_cache_window(cache_dir, dataset: str, subject_id: str,
                              e0: int, n: int) -> np.ndarray:
    """Read exactly `n` consecutive epochs starting at epoch `e0` from a
    cached subject's [T_epochs, 6, 3840] array via ONE seek + ONE
    contiguous read.

    Deliberately NOT `mmap_mode="r"` — measured slower than a full
    sequential read for this exact per-subject-per-window access pattern
    on this cluster's Lustre filesystem (PhysioOmni's own
    `_NpySliceReader` docstring has the numbers: mmap turns each page
    fault into a separate small RPC with no effective readahead, 0.68x-0.32x
    slower than a plain streamed read). Same physical filesystem, same
    lesson — do not "fix" this back to mmap without re-measuring.

    Returns: [n, 6, 3840] float16 array (a view into a freshly-read,
    read-only buffer — callers that need a mutable/float32 array should
    cast with `.astype(np.float32)`, which already copies).
    """
    path = cache_path_for(cache_dir, dataset, subject_id)
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)
        if fortran:
            raise ValueError(f"{path}: expected C-contiguous .npy, got Fortran-ordered")
        if len(shape) != 3 or tuple(shape[1:]) != (N_SLOTS, EPOCH_SAMPLES):
            raise ValueError(
                f"{path}: expected shape [T, {N_SLOTS}, {EPOCH_SAMPLES}], got {shape}"
            )
        header_end = f.tell()
        bytes_per_epoch = N_SLOTS * EPOCH_SAMPLES * dtype.itemsize
        f.seek(header_end + e0 * bytes_per_epoch)
        buf = f.read(n * bytes_per_epoch)
    if len(buf) != n * bytes_per_epoch:
        raise ValueError(
            f"{path}: requested {n} epochs from e0={e0} but only read "
            f"{len(buf) // bytes_per_epoch} — window out of range for a "
            f"cache with T={shape[0]} epochs"
        )
    return np.frombuffer(buf, dtype=dtype).reshape(n, N_SLOTS, EPOCH_SAMPLES)


def get_cached_t_epochs(cache_dir, dataset: str, subject_id: str) -> int:
    """Fast epoch-count lookup for a cached subject — reads only
    meta.json, never opens the (large) array file."""
    with open(_meta_path_for(cache_dir, dataset, subject_id)) as f:
        return json.load(f)["t_epochs"]


def cache_exists(cache_dir, dataset: str, subject_id: str) -> bool:
    """Whether a subject's raw-signal cache is complete enough to use.

    Parses meta.json rather than just checking existence — a zero-byte or
    truncated meta.json (the exact PhysioOmni failure mode this design
    avoids at the write side, kept as a read-side safety net too) must
    read as NOT cached, not crash the caller with JSONDecodeError.
    """
    meta_path = _meta_path_for(cache_dir, dataset, subject_id)
    if not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, OSError):
        return False
