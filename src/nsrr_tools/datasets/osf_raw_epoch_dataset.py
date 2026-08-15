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
slices a precomputed [T, 2, 768] embedding array; this class slices a
[12, n_samples_64] resampled-signal array instead.

READS FROM THE PRECOMPUTED RAW SIGNAL CACHE (checklist 2.5b), NOT the raw
HDF5, as of 2026-08-14. Originally called
nsrr_tools.datasets.osf_channel_loader.load_and_resample_channels() live
on every __getitem__ — found (via a real stalled GPU job, 54716906) to be
the dominant cost of every Stage 2 training run: repeated from scratch on
every item, for every task/head/context/job, even though the resampled
signal doesn't depend on any of those. scripts/precompute_osf_raw_signal_cache.py
now builds this once, offline, CPU-only; this class just reads it. See
that script's module docstring and docs/TSFM_OSF_IMPLEMENTATION_PLAN.md
checklist 2.5b for the full diagnosis/fix.

SCOPE (deliberately narrower than OSFContextWindowDataset for this first
pass): seq2label only. Stage 1 itself has not yet run sleep_staging
(seq2seq) either — all 5 Tier-1 tasks trained so far are seq2label — so
this is not a capability regression, just matching Stage 1's own current
scope. Add seq2seq support later if/when Stage 1 does.

SUBJECT/SPLIT SELECTION IS UNCHANGED BY THE CACHE — task_subject_dir,
split_seed, train/val/test proportions all work exactly as before. The
cache only changes *how* the signal for an already-selected subject is
read, never *which* subjects/splits are selected — required so Stage 1
and Stage 2 always train/evaluate on identical subjects.

INPUT
─────
  {raw_signal_cache_dir}/{dataset}/{subject_id}.npy  (precomputed by
  scripts/precompute_osf_raw_signal_cache.py — run that first. A subject
  present in the task CSV/split but missing from the cache raises a clear
  error at dataset-construction time, not a silent fallback to the raw
  HDF5 and not a silent subject drop — see __init__ below.)

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
    cache_path_for,
    get_cached_epoch_count,
    load_signal_cache,
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


def _build_raw_shape_cache(cache_dir: Path, df: pd.DataFrame) -> Tuple[dict, List[str]]:
    """{dataset}/{subject_id} -> epoch count, via a fast shape-only read of
    the precomputed raw-signal cache (nsrr_tools.datasets.osf_channel_loader.
    get_cached_epoch_count — mmap, no raw HDF5 touched at all). Returns
    (shape_dict, missing_keys) — missing_keys lists any subject in df whose
    cache file doesn't exist yet, so __init__ can fail loudly (see its
    docstring) instead of silently dropping subjects or falling back to a
    slow raw-HDF5 read."""
    cache = {}
    missing = []
    for _, row in df.iterrows():
        key = f"{row['dataset']}/{row['subject_id']}"
        cache_path = cache_path_for(cache_dir, row["dataset"], row["subject_id"])
        if cache_path.exists():
            cache[key] = get_cached_epoch_count(cache_path)
        else:
            missing.append(key)
    return cache, missing


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
        self.cache_dir = Path(data_cfg["raw_signal_cache_dir"])

        self.N = parse_context_length(context_length)
        if self.N == FULL_NIGHT_SENTINEL:
            raise NotImplementedError(
                "full_night not supported by OSFRawEpochWindowDataset yet."
            )

        self._K_max = ds_cfg.get("windows_per_subject", 5)

        # ── Load subject list — identical logic to OSFContextWindowDataset ──
        task_subject_dir = Path(ds_cfg["task_subject_dir"])
        csv_path = task_subject_dir / f"{task}_subjects.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Subject CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if datasets:
            df = df[df["dataset"].isin(datasets)].reset_index(drop=True)

        # Filter by Stage 1 embedding-file existence — NOT raw HDF5 or
        # Stage 2's own cache — so len(df) (and subject identity/order)
        # exactly matches what OSFContextWindowDataset (Stage 1) used at
        # split-computation time. Existence check ONLY; no embedding
        # contents are ever read here. Any other filter criterion risks a
        # completely different np.random.default_rng(split_seed).shuffle()
        # permutation the instant len(df) or subject order differs — live-
        # confirmed real, not hypothetical: MrOS has exactly 1 subject
        # (AA2557_v2) and STAGES exactly 1 (STLK00096) with a raw HDF5 but
        # no Stage 1 OSF embedding.
        stage1_embedding_dir = Path(ds_cfg["stage1_embedding_dir"])
        has_emb = df.apply(
            lambda r: (
                stage1_embedding_dir / r["dataset"] / f"{r['subject_id']}.npy"
            ).exists(),
            axis=1,
        )
        n_before = len(df)
        df = df[has_emb].reset_index(drop=True)
        n_missing = n_before - len(df)
        if n_missing > 0:
            warnings.warn(
                f"{n_missing}/{n_before} subjects have no Stage 1 embedding file "
                f"(under {stage1_embedding_dir}) — skipped, to match Stage 1's "
                f"exact subject pool at split-computation time.",
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

        # ── Shape cache (epoch counts, fast mmap shape read) + hard
        # completeness check — a subject selected above (same pool as
        # Stage 1) MUST have its raw-signal cache precomputed, or Stage 2
        # would either crash confusingly mid-training or (worse) silently
        # read stale/wrong data. Fail loudly and immediately instead. ────
        self._shape_cache, _missing_cache = _build_raw_shape_cache(self.cache_dir, self.df)
        if _missing_cache:
            preview = ", ".join(_missing_cache[:10])
            more = f" (+{len(_missing_cache) - 10} more)" if len(_missing_cache) > 10 else ""
            raise FileNotFoundError(
                f"[{split}] {len(_missing_cache)}/{len(self.df)} selected subjects have no "
                f"precomputed raw-signal cache under {self.cache_dir}: {preview}{more}\n"
                f"Run scripts/precompute_osf_raw_signal_cache.py first (see "
                f"docs/OSF_EXPERIMENTS_GUIDE.md Step 8.2b) — these subjects are part of "
                f"Stage 1's subject pool and cannot be silently dropped without breaking "
                f"the Stage 1/Stage 2 split match."
            )

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

        cache_path = cache_path_for(self.cache_dir, dataset, row["subject_id"])
        path_str = str(cache_path)

        # Per-worker single-subject cache — cheap even on a cache miss now
        # (mmap read, not a raw HDF5 channel-search + resample), but still
        # avoids re-reading the same subject's array for every window when
        # a SubjectGroupedSampler (see below) keeps items from one subject
        # consecutive.
        if getattr(self, "_cached_path", None) != path_str:
            self._cached_path = path_str
            self._cached_signal = load_signal_cache(cache_path)  # [12, n_samples_64] float32
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


# ─────────────────────────────────────────────────────────────────────────────
# SubjectGroupedSampler (checklist 2.5b)
# ─────────────────────────────────────────────────────────────────────────────

class SubjectGroupedSampler(torch.utils.data.Sampler):
    """Yield item indices grouped by subject, with per-epoch subject-order shuffle.

    Identical to osf_context_window_dataset.py's SubjectGroupedSampler
    (itself identical to context_window_dataset.py's) — purely index-
    arithmetic, no reference to embedding/signal shape. Copied as-is
    (not imported) so this module has no import dependency on the
    SleepFM-side or Stage-1-OSF-side dataset files.

    Added to Stage 2's train loader after the raw-signal cache fix
    (load_signal_cache is now cheap, but grouping still avoids redundant
    per-item mmap-open overhead within an epoch, stacking with the cache
    fix rather than replacing it — see docs/TSFM_OSF_IMPLEMENTATION_PLAN.md
    checklist 2.5b).

    Usage::

        sampler = SubjectGroupedSampler(train_ds._index)
        loader  = DataLoader(train_ds, batch_size=32, sampler=sampler,
                             shuffle=False, persistent_workers=True, ...)
    """

    def __init__(self, index: list, generator=None):
        from collections import defaultdict
        groups: dict = defaultdict(list)
        for item_idx, (row_idx, _, _) in enumerate(index):
            groups[row_idx].append(item_idx)
        self._groups = list(groups.values())
        self._generator = generator

    def __iter__(self):
        perm = torch.randperm(len(self._groups), generator=self._generator)
        for g_idx in perm.tolist():
            yield from self._groups[g_idx]

    def __len__(self) -> int:
        return sum(len(g) for g in self._groups)
