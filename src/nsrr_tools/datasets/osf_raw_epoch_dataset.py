#!/usr/bin/env python3
"""
osf_raw_epoch_dataset.py — OSF baseline, Stage 2 (LoRA) raw-signal dataset

PyTorch Dataset that serves fixed-length context windows of RAW OSF input
signal (not precomputed embeddings) for LoRA fine-tuning — checklist item
2.2, docs/TSFM_OSF_IMPLEMENTATION_PLAN.md Appendix §6.0/§6.1.

Forked from osf_context_window_dataset.py (OSFContextWindowDataset): the
subject-list/split/K-sampling/window-position index-building logic is
IDENTICAL (pure integer arithmetic over T epochs and N context length, no
reference to what's actually stored per epoch) and copied near-verbatim.
The only real difference is *materialization*: OSFContextWindowDataset
slices a precomputed [T, 2, 768] embedding array; this class loads raw
signal live via nsrr_tools.datasets.osf_channel_loader and slices a
[12, n_samples_64] array by epoch-aligned sample ranges instead.

SCOPE (deliberately narrower than OSFContextWindowDataset for this first
pass): seq2label only. Stage 1 itself has not yet run sleep_staging
(seq2seq) either — all 5 Tier-1 tasks trained so far are seq2label — so
this is not a capability regression, just matching Stage 1's own current
scope. Add seq2seq support later if/when Stage 1 does.

INPUT
─────
  {hdf5_dir}/{dataset}/derived/hdf5_signals/{subject_id}.h5  (same source
  HDF5s Stage 1 reads — read-only shared input, not copied or modified)

  seq2label subject CSV: [unified_id, dataset, subject_id, visit, label]
  (same task_subjects/*.csv Stage 1 uses — same task_subject_dir/split_seed,
  required for a fair Stage 1 vs Stage 2 vs SleepFM comparison)

OUTPUT PER __getitem__
───────────────────────
  x    : float32 tensor  [N, 12, 1920]  N = context_epochs
  mask : bool tensor     [N]            True = padded position (no real signal)
  y    : int64 tensor    []             scalar class label
"""

import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from nsrr_tools.datasets.osf_channel_loader import (
    EPOCH_SAMPLES,
    build_channel_candidates,
    get_epoch_count,
    load_and_resample_channels,
)

# ── Constants — identical to osf_context_window_dataset.py ─────────────────────
PATCH_SECONDS = 30
PATCHES_PER_EPOCH = 1
N_CHANNELS = 12
FULL_NIGHT_SENTINEL = -1


def parse_context_length(s) -> int:
    """Convert context-length string to number of 30-sec epochs.
    Identical to osf_context_window_dataset.py's parse_context_length —
    copied as-is (no import dependency between the two dataset modules)."""
    if isinstance(s, int):
        return s
    s = s.strip().lower()
    if s == "full_night":
        return FULL_NIGHT_SENTINEL
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(s|m)", s)
    if m is None:
        raise ValueError(
            f"Cannot parse context length: {s!r}. "
            "Expected e.g. '30s', '10m', or 'full_night'."
        )
    value, unit = float(m.group(1)), m.group(2)
    seconds = value if unit == "s" else value * 60
    patches = seconds / PATCH_SECONDS
    if not patches.is_integer():
        raise ValueError(
            f"Context length {s!r} → {seconds}s is not divisible by "
            f"{PATCH_SECONDS}s patch size."
        )
    return int(patches)


def _build_raw_shape_cache(hdf5_dir: Path, df: pd.DataFrame) -> dict:
    """{dataset}/{subject_id} -> epoch count, via fast HDF5 metadata reads
    (nsrr_tools.datasets.osf_channel_loader.get_epoch_count — no full
    channel load). Self-contained: does NOT depend on Stage 1's
    shape_cache.json existing, unlike an earlier design option that would
    have coupled Stage 2 to Stage 1's precomputed-embeddings output."""
    cache = {}
    for _, row in df.iterrows():
        key = f"{row['dataset']}/{row['subject_id']}"
        h5_path = hdf5_dir / row["dataset"] / "derived" / "hdf5_signals" / f"{row['subject_id']}.h5"
        if h5_path.exists():
            cache[key] = get_epoch_count(h5_path)
    return cache


class OSFRawEpochWindowDataset(Dataset):
    """Fixed-length context-window dataset over RAW OSF input signal.

    Args:
        cfg            : Phase 0 OSF LoRA config dict (phase0_osf_lora_config.yaml).
        split          : "train", "val", or "test".
        context_length : Duration string e.g. "10m", or int (epochs). No
                         full_night support yet (deferred — see module docstring).
        task           : Task name matching the subject CSV filename stem.
        datasets       : Optional list of dataset names to restrict to.
        seed           : RNG seed for window sampling.
    """

    def __init__(
        self,
        cfg: dict,
        split: str,
        context_length,
        task: str = None,
        datasets: Optional[List[str]] = None,
        seed: int = 42,
        limit: Optional[int] = None,
        max_items: Optional[int] = None,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"

        ds_cfg = cfg["dataset"]
        data_cfg = cfg["data"]
        task = task or ds_cfg["task"]
        assert ds_cfg.get("task_type", "seq2label") == "seq2label", (
            "OSFRawEpochWindowDataset only supports seq2label so far "
            "(matches Stage 1's current scope) — seq2seq not implemented yet."
        )

        self.split = split
        self.task = task
        self.seed = seed
        self.hdf5_dir = Path(data_cfg["hdf5_dir"])

        self.N = parse_context_length(context_length)
        if self.N == FULL_NIGHT_SENTINEL:
            raise NotImplementedError(
                "full_night not supported by OSFRawEpochWindowDataset yet."
            )

        self._K_max = ds_cfg.get("windows_per_subject", 5)

        # ── Channel candidates per dataset (SHHS EEG special-case) ─────────
        self._cfg_candidates = data_cfg["channel_candidates"]
        self._channel_candidates_cache: dict = {}

        # ── Load subject list — identical logic to OSFContextWindowDataset ──
        task_subject_dir = Path(ds_cfg["task_subject_dir"])
        csv_path = task_subject_dir / f"{task}_subjects.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Subject CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if datasets:
            df = df[df["dataset"].isin(datasets)].reset_index(drop=True)

        # Keep only subjects with a raw HDF5 file (read-only source shared
        # with Stage 1 — same files, no separate copy).
        has_h5 = df.apply(
            lambda r: (
                self.hdf5_dir / r["dataset"] / "derived" / "hdf5_signals" / f"{r['subject_id']}.h5"
            ).exists(),
            axis=1,
        )
        n_before = len(df)
        df = df[has_h5].reset_index(drop=True)
        n_missing = n_before - len(df)
        if n_missing > 0:
            warnings.warn(
                f"{n_missing}/{n_before} subjects have no HDF5 file — skipped.",
                stacklevel=2,
            )

        # ── Train / val / test split — identical to OSFContextWindowDataset ─
        rng = np.random.default_rng(ds_cfg["split_seed"])
        idx = np.arange(len(df))
        rng.shuffle(idx)

        n = len(idx)
        n_train = int(n * ds_cfg["train_split"])
        n_val = int(n * ds_cfg["val_split"])

        if split == "train":
            idx = idx[:n_train]
        elif split == "val":
            idx = idx[n_train : n_train + n_val]
        else:
            idx = idx[n_train + n_val :]

        self.df = df.iloc[idx].reset_index(drop=True)

        if limit is not None:
            self.df = self.df.iloc[:limit].reset_index(drop=True)

        # ── Shape cache (epoch counts via fast HDF5 metadata reads) ────────
        self._shape_cache = _build_raw_shape_cache(self.hdf5_dir, self.df)

        # ── Minimum recording length filter — identical logic/units to
        # OSFContextWindowDataset (480 epochs = 240m at 30s/epoch) ─────────
        self._min_recording_patches = ds_cfg.get("min_recording_patches", 0)
        if self._min_recording_patches > 0:
            T_series = self.df.apply(
                lambda r: self._shape_cache.get(f"{r['dataset']}/{r['subject_id']}", 0),
                axis=1,
            )
            keep = T_series >= self._min_recording_patches
            n_excluded = (~keep).sum()
            if n_excluded > 0:
                min_min = self._min_recording_patches * PATCH_SECONDS // 60
                warnings.warn(
                    f"[{split}] Cohort filter: {n_excluded} subject(s) excluded "
                    f"(T < {self._min_recording_patches} epochs / {min_min}m). "
                    f"Set dataset.min_recording_patches=0 to disable.",
                    stacklevel=2,
                )
            self.df = self.df[keep].reset_index(drop=True)

        if "label" not in self.df.columns:
            raise ValueError(
                f"seq2label requires a 'label' column in {csv_path}. "
                f"Found: {list(self.df.columns)}"
            )
        self._max_items = max_items
        self._index = self._build_seq2label_index()
        if max_items is not None and len(self._index) > max_items:
            self._index = self._index[:max_items]

    # ── Channel candidates (per dataset, cached) ────────────────────────────

    def _get_channel_candidates(self, dataset: str) -> dict:
        if dataset not in self._channel_candidates_cache:
            self._channel_candidates_cache[dataset] = build_channel_candidates(
                dataset, self._cfg_candidates
            )
        return self._channel_candidates_cache[dataset]

    # ── Index builder — identical arithmetic to OSFContextWindowDataset's
    # _build_seq2label_index, copied as-is (pure integer arithmetic on T/N,
    # no reference to what's stored per epoch) ──────────────────────────────

    def _build_seq2label_index(self) -> List[Tuple[int, int, int]]:
        index = []
        rng = np.random.default_rng(self.seed)

        for row_idx, row in self.df.iterrows():
            cache_key = f"{row['dataset']}/{row['subject_id']}"
            T = self._shape_cache[cache_key]
            label = int(row["label"])
            N = self.N

            if T < N:
                index.append((row_idx, 0, label))
                continue

            if self.split == "train":
                n_valid = T - N + 1
                K = min(self._K_max, n_valid)
                starts = sorted(rng.choice(n_valid, size=K, replace=False).tolist())
            elif self._K_max <= 100:
                n_valid = T - N + 1
                K = min(self._K_max, n_valid)
                starts = np.linspace(0, n_valid - 1, K, dtype=int).tolist()
            else:
                n_windows = T // N
                K = min(self._K_max, n_windows)
                positions = np.linspace(0, n_windows - 1, K, dtype=int)
                starts = [int(p) * N for p in positions]

            for s in starts:
                index.append((row_idx, int(s), label))

        return index

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        row_idx, window_start, label = self._index[idx]
        row = self.df.iloc[row_idx]
        dataset = row["dataset"]

        h5_path = self.hdf5_dir / dataset / "derived" / "hdf5_signals" / f"{row['subject_id']}.h5"
        path_str = str(h5_path)

        # Per-worker single-subject cache (same pattern as
        # OSFContextWindowDataset — SubjectGroupedSampler keeps items from
        # one subject consecutive, so this avoids reloading raw signal
        # for every window of the same subject).
        if getattr(self, "_cached_path", None) != path_str:
            self._cached_path = path_str
            candidates = self._get_channel_candidates(dataset)
            signal_matrix, _fill_info = load_and_resample_channels(h5_path, candidates)
            self._cached_signal = signal_matrix  # [12, n_samples_64] float32
        signal = self._cached_signal
        T = signal.shape[1] // EPOCH_SAMPLES

        x, mask = self._get_seq2label_window(signal, T, window_start)

        x_t = torch.from_numpy(x)
        m_t = torch.from_numpy(mask)
        y_t = torch.tensor(label, dtype=torch.long)
        return x_t, m_t, y_t

    def _get_seq2label_window(
        self, signal: np.ndarray, T: int, window_start: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract N raw epochs starting at window_start, right-padded with
        zeros when the recording ends early.

        Returns:
            x    : (N, 12, 1920) float32
            mask : (N,)          bool — True = right-padded position
        """
        N = self.N
        end = window_start + N

        if end <= T:
            s0, s1 = window_start * EPOCH_SAMPLES, end * EPOCH_SAMPLES
            chunk = signal[:, s0:s1]                       # [12, N*1920]
            window = chunk.reshape(N_CHANNELS, N, EPOCH_SAMPLES).transpose(1, 0, 2)  # [N,12,1920]
            mask = np.zeros(N, dtype=bool)
        else:
            real_len = max(0, T - window_start)
            pad_len = N - real_len
            if real_len > 0:
                s0, s1 = window_start * EPOCH_SAMPLES, T * EPOCH_SAMPLES
                chunk = signal[:, s0:s1]
                real_window = chunk.reshape(N_CHANNELS, real_len, EPOCH_SAMPLES).transpose(1, 0, 2)
            else:
                real_window = np.empty((0, N_CHANNELS, EPOCH_SAMPLES), dtype=np.float32)
            pad = np.zeros((pad_len, N_CHANNELS, EPOCH_SAMPLES), dtype=np.float32)
            window = np.concatenate([real_window, pad], axis=0)
            mask = np.array([False] * real_len + [True] * pad_len, dtype=bool)

        return window.astype(np.float32), mask

    # ── Convenience ───────────────────────────────────────────────────────

    _TASK_NUM_CLASSES = {
        "sex_binary": 2,
        "sleep_efficiency_binary": 2,
        "bmi_binary": 2,
        "age_class": 3,
        "apnea_binary": 2,
    }

    @property
    def num_classes(self) -> int:
        if self.task in self._TASK_NUM_CLASSES:
            return self._TASK_NUM_CLASSES[self.task]
        return int(self.df["label"].max()) + 1

    def __repr__(self) -> str:
        return (
            f"OSFRawEpochWindowDataset("
            f"split={self.split}, context={self.N} epochs ({self.N * PATCH_SECONDS}s), "
            f"n_items={len(self._index)})"
        )
