#!/usr/bin/env python3
"""
context_window_dataset.py — Phase 0, Step 2 (redesigned)

PyTorch Dataset that serves fixed-length context windows of SleepFM embeddings
for the context-length sweep experiments.

CORE DESIGN PRINCIPLE
─────────────────────
The sweep compares performance at different context lengths L.  For the comparison
to be valid, every point on the curve must answer the SAME set of prediction
questions — only the amount of context given as input differs.

  seq2seq  (sleep staging):
    Index unit = (subject, anchor_epoch_t).
    For every 30-sec epoch t the model is asked: "given the past L seconds
    ending at t, what is the sleep stage of epoch t?"
    __len__ is FIXED regardless of L.  Only the window width changes.

  seq2label (night-level tasks):
    Index unit = (subject, window_k).
    K = min(K_max, n_available_windows) non-overlapping windows per subject.
    K_max is the same at every context length, so __len__ is approximately
    equal across the sweep (exact equality only breaks at full_night where K=1).

INPUT FILES
───────────
  {embedding_dir}/{dataset}/{subject_id}.npy
    dtype  : float16
    shape  : [T, 4, 128]   T = total 5-sec patches for the full night

  seq2label : subject CSV  [unified_id, dataset, subject_id, visit, label]
  seq2seq   : subject CSV  [unified_id, dataset, subject_id, visit,
                             annotation_path, n_epochs]
              annotation   : .npy  (n_epochs,) int8  — 30-sec epoch stages

OUTPUT PER __getitem__
──────────────────────
  x    : float32 tensor  [N, 512]   N = context_patches, 512 = 4×128
  mask : bool tensor     [N]        True = padded position (no real signal)
  y    : int64 tensor    []         scalar class label

  For seq2seq (anchor-based):
    N patches ending at anchor_t (past-only causal window).
    LEFT-padded when the recording start is less than N patches before anchor_t.
    y = stage of anchor_t epoch.

  For seq2label (K-window):
    N patches starting at window_start.
    RIGHT-padded when the recording is shorter than window_start + N.
    y = night-level label from subject CSV.

CONTEXT LENGTH STRINGS
──────────────────────
  "30s"        →    6 patches   (6 × 5s = 30s)
  "2m"         →   24 patches
  "5m"         →   60 patches
  "10m"        → 120 patches
  "20m"        → 240 patches
  "40m"        → 480 patches
  "80m"        → 960 patches
  "full_night" → all patches up to anchor_t (seq2seq) or whole night (seq2label)

  "full_night" produces variable-length tensors across subjects / anchors.
  Use dataset.collate_fn as the DataLoader collate_fn when context="full_night"
  to zero-pad within each batch to the longest sample in that batch.

MINIMUM PAST CONTEXT (seq2seq only)
────────────────────────────────────
  Anchors very close to the recording start are excluded when they would provide
  too little past context to be informative.

    min_past = max(PATCHES_PER_EPOCH, min(N // min_past_denom, max_min_past))

  Defaults: min_past_denom=8, max_min_past=240 patches (20 min).
  Results:
    30s  →  6 patches (30s)    80m  → 120 patches (10m)
    10m  → 15 patches (1.25m)  full_night → 240 patches (20m, capped)
"""

import json
import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ── Constants ─────────────────────────────────────────────────────────────────
PATCH_SECONDS    = 5           # each embedding patch = 5 seconds
PATCHES_PER_EPOCH = 6          # 30-sec sleep epoch / 5-sec patch
REM_ORIGINAL     = 5
REM_REMAPPED     = 4
EMBED_DIM        = 128
N_MODALITIES     = 4
FLAT_DIM         = N_MODALITIES * EMBED_DIM   # 512
FULL_NIGHT_SENTINEL = -1       # internal sentinel for full_night context length


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_context_length(s) -> int:
    """Convert context-length string to number of 5-sec patches.

    Returns FULL_NIGHT_SENTINEL (-1) for "full_night".

    Examples:
        "30s"        → 6
        "2m"         → 24
        "10m"        → 120
        "full_night" → -1
    """
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


def _compute_min_past(N: int, denom: int = 8, max_patches: int = 240) -> int:
    """Minimum past patches required before an anchor is included.

    Scales with context length to avoid wasting data at the night start for
    long contexts while still excluding degenerate zero-context anchors for
    short contexts.

    Formula: max(PATCHES_PER_EPOCH, min(N // denom, max_patches))
    For full_night (N=-1): use max_patches directly.
    """
    if N == FULL_NIGHT_SENTINEL:
        return max_patches
    return max(PATCHES_PER_EPOCH, min(N // denom, max_patches))


def _remap_stages(arr: np.ndarray) -> np.ndarray:
    """Remap REM stage 5 → 4 in-place (copy). Returns int8 array."""
    out = arr.copy().astype(np.int8)
    out[out == REM_ORIGINAL] = REM_REMAPPED
    return out


def _load_emb_shape(path: Path) -> int:
    """Return T (number of patches) without loading the full array."""
    return np.load(path, mmap_mode="r").shape[0]


def _build_shape_cache(embedding_dir: Path) -> dict:
    """
    Scan all .npy files under embedding_dir and return {rel_key: T} where
    rel_key = "{dataset}/{subject_id}".

    The result is written to {embedding_dir}/shape_cache.json so subsequent
    calls return immediately.  The cache is invalidated (rebuilt) only if
    new files appear that are not in the stored cache.
    """


    cache_path = embedding_dir / "shape_cache.json"

    # If cache exists, load and return immediately — skip filesystem scan.
    # Delete shape_cache.json manually if you add new embedding files.
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    # Cache missing: scan all .npy files and build it from scratch.
    print(f"  [shape_cache] Building cache (first run) …", flush=True)
    all_files = {
        f"{p.parent.name}/{p.stem}": p
        for p in embedding_dir.rglob("*.npy")
    }
    cached = {
        key: int(np.load(path, mmap_mode="r").shape[0])
        for key, path in sorted(all_files.items())
    }
    with open(cache_path, "w") as f:
        json.dump(cached, f)
    print(f"  [shape_cache] Cache saved → {cache_path} ({len(cached)} entries)", flush=True)

    return cached


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ContextWindowDataset(Dataset):
    """Fixed-length (or full-night) context-window dataset over SleepFM embeddings.

    Args:
        cfg            : Phase 0 config dict (from phase0_config.yaml).
        split          : "train", "val", or "test".
        context_length : Duration string e.g. "10m", "full_night", or int (patches).
        task           : Task name matching the subject CSV filename stem
                         (e.g. "apnea_binary") or "sleep_staging".
        task_type      : "seq2label" or "seq2seq".
        datasets       : Optional list of dataset names to restrict to.
        seed           : RNG seed for window/split sampling.
    """

    def __init__(
        self,
        cfg: dict,
        split: str,
        context_length,
        task: str = None,
        task_type: str = None,
        datasets: Optional[List[str]] = None,
        seed: int = 42,
        limit: Optional[int] = None,
        max_items: Optional[int] = None,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split!r}"

        ds_cfg    = cfg["dataset"]
        task      = task      or ds_cfg["task"]
        task_type = task_type or ds_cfg["task_type"]
        assert task_type in ("seq2label", "seq2seq"), f"Unknown task_type: {task_type!r}"

        self.split         = split
        self.task          = task
        self.task_type     = task_type
        self.seed          = seed
        self.embedding_dir = Path(cfg["dataset"]["embedding_dir"])

        # Parse context length (may be sentinel for full_night)
        self.N = parse_context_length(context_length)
        self.is_full_night = (self.N == FULL_NIGHT_SENTINEL)

        # min_past config (seq2seq only)
        min_past_denom   = ds_cfg.get("min_past_denom",   8)
        max_min_past     = ds_cfg.get("max_min_past_patches", 240)
        self._min_past   = _compute_min_past(self.N, min_past_denom, max_min_past)

        # K_max for seq2label
        self._K_max = ds_cfg.get("windows_per_subject", 5)

        # ── Shape cache (avoids per-subject .npy header reads on slow FS) ──
        self._shape_cache = _build_shape_cache(self.embedding_dir)

        # ── NaN blocklist (subjects with corrupt embedding files) ──────────
        blocklist_path = self.embedding_dir / "nan_blocklist.txt"
        if blocklist_path.exists():
            with open(blocklist_path) as f:
                self._nan_blocklist = {
                    line.split("\t")[0].strip()
                    for line in f if line.strip()
                }
            if self._nan_blocklist:
                warnings.warn(
                    f"NaN blocklist loaded: {len(self._nan_blocklist)} subjects will be "
                    f"excluded (see {blocklist_path}).",
                    stacklevel=2,
                )
        else:
            self._nan_blocklist = set()

        # ── Load subject list ──────────────────────────────────────────────
        task_subject_dir = Path(ds_cfg["task_subject_dir"])
        csv_path = task_subject_dir / f"{task}_subjects.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Subject CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if datasets:
            df = df[df["dataset"].isin(datasets)].reset_index(drop=True)

        # Keep only subjects with an embedding file
        has_emb = df.apply(
            lambda r: (
                self.embedding_dir / r["dataset"] / f"{r['subject_id']}.npy"
            ).exists(),
            axis=1,
        )
        n_before = len(df)
        df = df[has_emb].reset_index(drop=True)
        n_missing = n_before - len(df)
        if n_missing > 0:
            warnings.warn(
                f"{n_missing}/{n_before} subjects have no embedding file — skipped.",
                stacklevel=2,
            )

        # Remove subjects with known NaN embeddings
        if self._nan_blocklist:
            is_blocked = df.apply(
                lambda r: f"{r['dataset']}/{r['subject_id']}" in self._nan_blocklist,
                axis=1,
            )
            n_blocked = is_blocked.sum()
            df = df[~is_blocked].reset_index(drop=True)
            if n_blocked > 0:
                warnings.warn(
                    f"{n_blocked} subjects removed (NaN blocklist).",
                    stacklevel=2,
                )

        # ── Train / val / test split (subject-level) ───────────────────────
        rng = np.random.default_rng(ds_cfg["split_seed"])
        idx = np.arange(len(df))
        rng.shuffle(idx)

        n      = len(idx)
        n_train = int(n * ds_cfg["train_split"])
        n_val   = int(n * ds_cfg["val_split"])

        if split == "train":
            idx = idx[:n_train]
        elif split == "val":
            idx = idx[n_train : n_train + n_val]
        else:
            idx = idx[n_train + n_val :]

        self.df = df.iloc[idx].reset_index(drop=True)

        # Debug limit: keep only first N subjects
        if limit is not None:
            self.df = self.df.iloc[:limit].reset_index(drop=True)

        # ── Build flat index ───────────────────────────────────────────────
        # Each entry: (subject_row_idx, aux_int, label_int)
        #   seq2seq   aux_int = anchor_patch_end  (exclusive, i.e., last patch+1)
        #   seq2label aux_int = window_start
        self._max_items = max_items

        if task_type == "seq2seq":
            self._index = self._build_seq2seq_index()
        else:
            if "label" not in self.df.columns:
                raise ValueError(
                    f"seq2label requires a 'label' column in {csv_path}. "
                    f"Found: {list(self.df.columns)}"
                )
            self._index = self._build_seq2label_index()

        # Cap total items (for quick training-loop debug)
        if max_items is not None and len(self._index) > max_items:
            self._index = self._index[:max_items]

    # ── Index builders ─────────────────────────────────────────────────────

    def _build_seq2seq_index(self) -> List[Tuple[int, int, int]]:
        """Build (row_idx, anchor_patch_end, stage_label) for every valid anchor."""
        index = []
        for row_idx, row in self.df.iterrows():
            cache_key = f"{row['dataset']}/{row['subject_id']}"
            T = self._shape_cache[cache_key]

            # Load stage annotations (small array — OK to load here)
            ann_path = Path(row["annotation_path"])
            if not ann_path.exists():
                warnings.warn(f"Annotation not found: {ann_path} — subject skipped.")
                continue
            raw_stages = np.load(ann_path)        # (n_epochs,) int8
            stages     = _remap_stages(raw_stages)

            # Align: embedding patches vs annotation epochs
            # Each epoch = PATCHES_PER_EPOCH patches; total aligned patches:
            T_ann = len(stages) * PATCHES_PER_EPOCH
            T_eff = min(T, T_ann)                 # usable length

            n_epochs = T_eff // PATCHES_PER_EPOCH

            for epoch_idx in range(n_epochs):
                anchor_patch_end = (epoch_idx + 1) * PATCHES_PER_EPOCH
                # anchor_patch_end is the index AFTER the last patch of this epoch
                # past context available = anchor_patch_end patches

                if anchor_patch_end < self._min_past:
                    continue                       # too close to recording start

                label = int(stages[epoch_idx])
                if label < 0 or label > 4:
                    continue                       # unknown / artefact stage

                index.append((row_idx, anchor_patch_end, label))

        return index

    def _build_seq2label_index(self) -> List[Tuple[int, int, int]]:
        """Build (row_idx, window_start, label) for K windows per subject."""
        index = []
        rng   = np.random.default_rng(self.seed)

        for row_idx, row in self.df.iterrows():
            cache_key = f"{row['dataset']}/{row['subject_id']}"
            T     = self._shape_cache[cache_key]
            label = int(row["label"])

            if self.is_full_night:
                # One window: start = 0, covers the whole night
                index.append((row_idx, 0, label))
                continue

            N = self.N
            if T < N:
                # Recording shorter than one window: one zero-padded window
                index.append((row_idx, 0, label))
                continue

            n_windows = T // N                    # non-overlapping windows
            K = min(self._K_max, n_windows)

            if self.split == "train":
                # K random start positions (without replacement)
                starts = sorted(
                    rng.choice(n_windows, size=K, replace=False).tolist()
                )
                starts = [s * N for s in starts]
            else:
                # K evenly spaced windows (deterministic)
                positions = np.linspace(0, n_windows - 1, K, dtype=int)
                starts = [int(p) * N for p in positions]

            for s in starts:
                index.append((row_idx, s, label))

        return index

    # ── Dataset interface ──────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        row_idx, aux, label = self._index[idx]
        row = self.df.iloc[row_idx]

        npy_path = self.embedding_dir / row["dataset"] / f"{row['subject_id']}.npy"
        # mmap_mode='r': OS pages in only the slices we access (~6KB per item
        # for 30s context) instead of loading the full ~5MB file every call.
        emb = np.load(npy_path, mmap_mode="r")   # (T, 4, 128) float16
        T   = emb.shape[0]

        if self.task_type == "seq2seq":
            x, mask = self._get_seq2seq_window(emb, T, anchor_patch_end=aux)
        else:
            x, mask = self._get_seq2label_window(emb, T, window_start=aux)

        x_t = torch.from_numpy(x)
        m_t = torch.from_numpy(mask)
        y_t = torch.tensor(label, dtype=torch.long)
        return x_t, m_t, y_t

    # ── Window extraction ──────────────────────────────────────────────────

    def _get_seq2seq_window(
        self, emb: np.ndarray, T: int, anchor_patch_end: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract past-only window ending at anchor_patch_end.

        Window covers [anchor_patch_end - N : anchor_patch_end].
        Left-padded when anchor_patch_end < N (early-night anchors).

        Returns:
            x    : (N, 512) float32
            mask : (N,)     bool — True = left-padded position
        """
        if self.is_full_night:
            N = anchor_patch_end            # use all available past context
        else:
            N = self.N

        win_start = anchor_patch_end - N    # may be negative

        if win_start >= 0:
            # Normal case: slice directly from embedding
            window = emb[win_start : anchor_patch_end]   # (N, 4, 128)
            mask   = np.zeros(N, dtype=bool)
        else:
            # Recording starts after the window start: left-pad
            pad_len = -win_start
            real_len = anchor_patch_end        # = N - pad_len
            window = np.concatenate([
                np.zeros((pad_len, N_MODALITIES, EMBED_DIM), dtype=np.float16),
                emb[:real_len],
            ], axis=0)                         # (N, 4, 128)
            mask = np.array([True] * pad_len + [False] * real_len, dtype=bool)

        x = window.astype(np.float32).reshape(N, FLAT_DIM)
        return x, mask

    def _get_seq2label_window(
        self, emb: np.ndarray, T: int, window_start: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract window of length N starting at window_start.

        Right-padded when the recording ends before window_start + N.

        Returns:
            x    : (N, 512) float32
            mask : (N,)     bool — True = right-padded position
        """
        if self.is_full_night:
            N = T                           # whole recording, no padding needed
        else:
            N = self.N

        end = window_start + N
        if end <= T:
            window = emb[window_start : end]   # (N, 4, 128)
            mask   = np.zeros(N, dtype=bool)
        else:
            real_len = T - window_start
            pad_len  = N - real_len
            window = np.concatenate([
                emb[window_start : T],
                np.zeros((pad_len, N_MODALITIES, EMBED_DIM), dtype=np.float16),
            ], axis=0)
            mask = np.array([False] * real_len + [True] * pad_len, dtype=bool)

        x = window.astype(np.float32).reshape(N, FLAT_DIM)
        return x, mask

    # ── full_night collate ─────────────────────────────────────────────────

    @staticmethod
    def collate_fn(batch):
        """Pad variable-length full_night samples to the longest in the batch.

        Use as: DataLoader(..., collate_fn=ContextWindowDataset.collate_fn)
        Only needed when context_length="full_night".
        """
        xs, masks, ys = zip(*batch)
        max_N = max(x.shape[0] for x in xs)

        padded_x    = []
        padded_mask = []
        for x, mask in zip(xs, masks):
            n = x.shape[0]
            if n < max_N:
                pad = max_N - n
                x    = F.pad(x,    (0, 0, 0, pad))          # right-pad time dim
                mask = F.pad(mask, (0, pad), value=True)
            padded_x.append(x)
            padded_mask.append(mask)

        return (
            torch.stack(padded_x),
            torch.stack(padded_mask),
            torch.stack(ys),
        )

    # ── Convenience ───────────────────────────────────────────────────────

    # Known fixed class counts per task — prevents under-counting with small
    # subsets (e.g. --limit 2 may only contain one class).
    _TASK_NUM_CLASSES = {
        "sleep_staging":    5,   # Wake, N1, N2, N3, REM (seq2seq)
        "apnea_binary":     2,
        "apnea_class":      4,
        "anxiety_binary":   2,
        "cvd_binary":       2,
        "depression_binary":2,
        "depression_class": 4,
        "insomnia_binary":  2,
        "rested_morning":   2,
        "sleepiness_binary":2,
        "sleepiness_class": 3,
    }

    @property
    def num_classes(self) -> int:
        if self.task in self._TASK_NUM_CLASSES:
            return self._TASK_NUM_CLASSES[self.task]
        # Fallback: infer from data (may under-count with small subsets)
        if self.task_type == "seq2seq":
            return 5
        return int(self.df["label"].max()) + 1

    def __repr__(self) -> str:
        ctx = "full_night" if self.is_full_night else f"{self.N} patches ({self.N * PATCH_SECONDS}s)"
        return (
            f"ContextWindowDataset("
            f"split={self.split}, context={ctx}, "
            f"task_type={self.task_type}, n_items={len(self._index)})"
        )
