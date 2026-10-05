#!/usr/bin/env python3
"""
mantis_raw_epoch_dataset.py — Mantis baseline, Stage 2 (LoRA) raw-signal
dataset (checklist 2.3)

PyTorch Dataset that serves fixed-length context windows of RAW Mantis
input signal (not precomputed embeddings) for LoRA fine-tuning.

Near-verbatim fork of osf_raw_epoch_dataset.py (plan §14.4): the subject-
list/split/K-sampling/window-position index-building logic is IDENTICAL
(pure integer arithmetic over T epochs and N context length, no reference
to what's actually stored per epoch). The only real difference is
*materialization* — this class reads from the raw-signal cache built by
scripts/precompute_mantis_raw_signal_cache.py (checklist 2.2) instead of a
precomputed embedding array.

ONE GENUINE SIMPLIFICATION vs OSF's/PhysioOmni's own raw-epoch datasets:
Mantis's cache (mantis_channel_loader.save_signal_cache) is already
EPOCH-MAJOR `[T, 6, 3840]` on disk (plan §14.3's whole design point), so
`load_signal_cache_window()` returns a window already shaped `[n, 6, 3840]`
— no channel-major-to-epoch-major reshape/transpose is needed here, unlike
OSF's `_get_seq2label_window` (which stores channel-major `[12, n_samples]`
and must reshape+transpose on every read).

DELIBERATE NON-OPTIMIZATION, evidence-based (efficiency mandate,
2026-09-07 — read before "fixing" this): unlike OSF's/PhysioOmni's own
raw-epoch datasets, this class does NOT cache a per-worker materialized
full-subject array and slice windows from RAM — each `__getitem__` calls
`load_signal_cache_window()` directly, which opens+seeks+reads+closes the
subject's file fresh every time, even for consecutive windows from the
same subject under `SubjectGroupedSampler`. This is intentional, not an
oversight: PhysioOmni's own real, measured finding
(`physioomni_channel_loader._NpySliceReader`'s docstring) is that on this
cluster's Lustre filesystem, per-window I/O is ~50% of a 30s-context
epoch's total cost but only ~3.5% of an 80m-context epoch's — **long
contexts are ~96% compute-bound**, and long contexts are exactly where
LoRA fine-tuning's GPU-hour cost actually lives. Adding a persistent-open-
file-handle cache here would meaningfully help only the already-cheap 30s
case while adding real code complexity — not where the efficiency mandate
should spend its budget. If a future profiling pass finds otherwise for
Mantis's own specific access pattern, revisit with real numbers, not by
assumption (same "measure, don't guess" standard as everywhere else in
this project).

SCOPE (deliberately narrower, matching Stage 1's and OSF's/PhysioOmni's
own Stage 2 scope): seq2label only. sleep_staging (seq2seq) is out of
scope for all three TSFM baselines at every stage (plan §5.8).

SUBJECT/SPLIT SELECTION IS UNCHANGED BY THE CACHE — task_subject_dir,
split_seed, train/val/test proportions all work exactly as Stage 1's
MantisContextWindowDataset. The cache only changes *how* the signal for an
already-selected subject is read, never *which* subjects/splits are
selected — required so Stage 1 and Stage 2 always train/evaluate on
identical subjects (plan §14.4's split-matching discipline — a real,
previously-live bug on both OSF's and PhysioOmni's own Stage 2 builds, not
a theoretical concern).

INPUT
─────
  {raw_signal_cache_dir}/{dataset}/{subject_id}.npy + .meta.json
  (precomputed by scripts/precompute_mantis_raw_signal_cache.py — run that
  first. A subject present in the task CSV/split but missing from the
  cache raises a clear error at dataset-construction time, not a silent
  fallback to the raw HDF5 and not a silent subject drop — see __init__.)

  seq2label subject CSV: [unified_id, dataset, subject_id, visit, label]
  (same task_subjects/*.csv Stage 1 uses)

OUTPUT PER __getitem__
───────────────────────
  x    : float32 tensor  [N, 6, 3840]  N = context_epochs
  mask : bool tensor     [N]           True = padded position (no real signal)
  y    : int64 tensor    []            scalar class label
"""

import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from nsrr_tools.datasets.mantis_channel_loader import (
    EPOCH_SAMPLES,
    N_SLOTS,
    cache_exists,
    get_cached_t_epochs,
    load_signal_cache_window,
)

# ── Constants — identical to mantis_context_window_dataset.py ──────────────────
PATCH_SECONDS = 30
PATCHES_PER_EPOCH = 1
FULL_NIGHT_SENTINEL = -1


def parse_context_length(s) -> int:
    """Convert context-length string to number of 30-sec epochs.
    Identical to mantis_context_window_dataset.py's parse_context_length —
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


def _build_raw_shape_cache(cache_dir, df: pd.DataFrame) -> Tuple[dict, List[str]]:
    """{dataset}/{subject_id} -> epoch count, via cache_exists()/
    get_cached_t_epochs() (meta.json only, no array touched). Returns
    (shape_dict, missing_keys) — missing_keys lists any subject in df
    whose cache isn't complete yet, so __init__ can fail loudly (see its
    docstring) instead of silently dropping subjects or falling back to a
    slow raw-HDF5 read."""
    cache = {}
    missing = []
    for _, row in df.iterrows():
        key = f"{row['dataset']}/{row['subject_id']}"
        if cache_exists(cache_dir, row["dataset"], row["subject_id"]):
            cache[key] = get_cached_t_epochs(cache_dir, row["dataset"], row["subject_id"])
        else:
            missing.append(key)
    return cache, missing


class MantisRawEpochWindowDataset(Dataset):
    """Fixed-length context-window dataset over RAW Mantis input signal.

    Args:
        cfg            : Phase 0 Mantis LoRA config dict (phase0_mantis_lora_config.yaml).
        split          : "train", "val", or "test".
        context_length : Duration string e.g. "10m", or int (epochs). No
                         full_night support yet (deferred — matches OSF's
                         own Stage 2 scope, plan §14.4).
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
            "MantisRawEpochWindowDataset only supports seq2label so far "
            "(matches Stage 1's current scope) — seq2seq not implemented yet."
        )

        self.split = split
        self.task = task
        self.seed = seed
        self.cache_dir = Path(data_cfg["raw_signal_cache_dir"])

        self.N = parse_context_length(context_length)
        if self.N == FULL_NIGHT_SENTINEL:
            raise NotImplementedError(
                "full_night not supported by MantisRawEpochWindowDataset yet."
            )

        self._K_max = ds_cfg.get("windows_per_subject", 5)

        # ── Load subject list — identical logic to MantisContextWindowDataset ──
        task_subject_dir = Path(ds_cfg["task_subject_dir"])
        csv_path = task_subject_dir / f"{task}_subjects.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Subject CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if datasets:
            df = df[df["dataset"].isin(datasets)].reset_index(drop=True)

        # Filter by Stage 1 embedding-file existence — NOT the raw HDF5 or
        # Stage 2's own cache — so len(df) (and subject identity/order)
        # exactly matches what MantisContextWindowDataset (Stage 1) used at
        # split-computation time. Existence check ONLY; no embedding
        # contents are ever read here. Any other filter criterion risks a
        # completely different np.random.default_rng(split_seed).shuffle()
        # permutation the instant len(df) or subject order differs — a
        # real, previously-live bug on both OSF's and PhysioOmni's own
        # Stage 2 builds (plan §14.4), not a theoretical concern.
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

        # ── Train / val / test split — identical to MantisContextWindowDataset ─
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

        # ── Shape cache (epoch counts, meta.json only) + hard completeness
        # check — a subject selected above (same pool as Stage 1) MUST have
        # its raw-signal cache precomputed, or Stage 2 would either crash
        # confusingly mid-training or (worse) silently read stale/wrong
        # data. Fail loudly and immediately instead. ─────────────────────
        self._shape_cache, _missing_cache = _build_raw_shape_cache(self.cache_dir, self.df)
        if _missing_cache:
            preview = ", ".join(_missing_cache[:10])
            more = f" (+{len(_missing_cache) - 10} more)" if len(_missing_cache) > 10 else ""
            raise FileNotFoundError(
                f"[{split}] {len(_missing_cache)}/{len(self.df)} selected subjects have no "
                f"complete precomputed raw-signal cache under {self.cache_dir}: {preview}{more}\n"
                f"Run scripts/precompute_mantis_raw_signal_cache.py first (see "
                f"docs/MANTIS_EXPERIMENTS_GUIDE.md) — these subjects are part of "
                f"Stage 1's subject pool and cannot be silently dropped without breaking "
                f"the Stage 1/Stage 2 split match."
            )

        # ── Minimum recording length filter — identical logic/units to
        # MantisContextWindowDataset (480 epochs = 240m at 30s/epoch) ──────
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

    # ── Index builder — identical arithmetic to MantisContextWindowDataset's
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
        subject_id = row["subject_id"]

        cache_key = f"{dataset}/{subject_id}"
        T = self._shape_cache[cache_key]

        x, mask = self._get_seq2label_window(dataset, subject_id, T, window_start)

        x_t = torch.from_numpy(x)
        m_t = torch.from_numpy(mask)
        y_t = torch.tensor(label, dtype=torch.long)
        return x_t, m_t, y_t

    def _get_seq2label_window(
        self, dataset: str, subject_id: str, T: int, window_start: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract N raw epochs starting at window_start, right-padded with
        zeros when the recording ends early.

        Reads directly from the epoch-major cache via
        load_signal_cache_window() — already shaped [n, 6, 3840], no
        reshape/transpose needed (unlike OSF's channel-major cache). Only
        requests the REAL portion that exists (real_len = min(N, T -
        window_start)) — load_signal_cache_window() raises on any
        out-of-range request, so the padded portion is built separately,
        never requested from the cache.

        Returns:
            x    : (N, 6, 3840) float32
            mask : (N,)         bool — True = right-padded position
        """
        N = self.N
        end = window_start + N

        if end <= T:
            window16 = load_signal_cache_window(self.cache_dir, dataset, subject_id, window_start, N)
            mask = np.zeros(N, dtype=bool)
        else:
            real_len = max(0, T - window_start)
            pad_len = N - real_len
            if real_len > 0:
                real_window16 = load_signal_cache_window(
                    self.cache_dir, dataset, subject_id, window_start, real_len
                )
            else:
                real_window16 = np.empty((0, N_SLOTS, EPOCH_SAMPLES), dtype=np.float16)
            pad16 = np.zeros((pad_len, N_SLOTS, EPOCH_SAMPLES), dtype=np.float16)
            window16 = np.concatenate([real_window16, pad16], axis=0)
            mask = np.array([False] * real_len + [True] * pad_len, dtype=bool)

        return window16.astype(np.float32), mask

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
            f"MantisRawEpochWindowDataset("
            f"split={self.split}, context={self.N} epochs ({self.N * PATCH_SECONDS}s), "
            f"n_items={len(self._index)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SubjectGroupedSampler (checklist 2.3)
# ─────────────────────────────────────────────────────────────────────────────

class SubjectGroupedSampler(torch.utils.data.Sampler):
    """Yield item indices grouped by subject, with per-epoch subject-order shuffle.

    Identical to osf_raw_epoch_dataset.py's/mantis_context_window_dataset.py's
    SubjectGroupedSampler — purely index-arithmetic, no reference to signal
    shape. Copied as-is (not imported) so this module has no import
    dependency on Stage 1's or OSF's/PhysioOmni's dataset files.

    Note (see module docstring's "DELIBERATE NON-OPTIMIZATION" section):
    unlike OSF's Stage 2 dataset, grouping items by subject here does NOT
    save a materialized-array cache hit, since this class never
    materializes a full-subject array in the first place — kept anyway
    because it's still a real, free win for the shape cache's dict lookups
    and is the established convention every dataset class in this project
    uses for its train loader.

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
