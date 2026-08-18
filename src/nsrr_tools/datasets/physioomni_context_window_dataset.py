#!/usr/bin/env python3
"""
physioomni_context_window_dataset.py — PhysioOmni baseline, Stage 1 Step 3

PyTorch Dataset that serves fixed-length context windows of PhysioOmni
embeddings for the context-length sweep experiments. Forked from
osf_context_window_dataset.py (OSFContextWindowDataset) per
docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md §8 — the K-sampling logic,
SubjectGroupedSampler, and window-building index math are shape-agnostic
and copied as-is; only the embedding-shape constants change, and the
zero_modality_indices ablation feature is dropped entirely (PhysioOmni has
no 4-modality-*group* structure to ablate — its "4 modalities" are already
EEG/EOG/ECG/EMG, the finest granularity, same reasoning OSF's fork used).

WHY THIS IS SIMPLER THAN OSF's FORK, NOT JUST A RENAME
─────────────────────────────────────────────────────────
OSF's embeddings are [T, 2, 768] — two distinct 768-dim sub-tokens (CLS +
mean-pooled patches) that OSF's fork keeps as an explicit 3D shape,
reshaping to (N, FLAT_DIM) only at window-extraction time. PhysioOmni's
embeddings (docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md §6.3) are already
one flat 500-dim vector per epoch (200 EEG + 100 EOG + 100 ECG + 100 EMG,
concatenated at extraction time by extract_physioomni_embeddings.py) — a
genuinely 2D [T, 500] array on disk, not [T, 1, 500]. There is no
meaningful "N_SUBTOKENS" dimension to carry through this fork at all, so
(unlike OSF's fork) every 3D pad-block shape and reshape call in the
original is simplified to plain 2D — this is a real simplification, not
cosmetic.

INPUT FILES
───────────
  {embedding_dir}/{dataset}/{subject_id}.npy
    dtype  : float16
    shape  : [T, 500]   T = total 30-sec epochs for the full night
                        500 = 200 (EEG CLS) + 100 (EOG CLS) + 100 (ECG CLS)
                              + 100 (EMG CLS), fixed concatenation order

  seq2label : subject CSV  [unified_id, dataset, subject_id, visit, label]
  seq2seq   : subject CSV  [unified_id, dataset, subject_id, visit,
                             annotation_path, n_epochs]
              annotation   : .npy  (n_epochs,) int8  — 30-sec epoch stages
              (carried for config-shape parity with OSF — PhysioOmni's
              Tier-1 task scope, sex/sleep_efficiency/bmi/age, is entirely
              seq2label; not exercised yet, see plan §12)

OUTPUT PER __getitem__
──────────────────────
  x    : float32 tensor  [N, 500]   N = context_epochs
  mask : bool tensor     [N]        True = padded position (no real signal)
  y    : int64 tensor    []         scalar class label

CONTEXT LENGTH STRINGS — identical to OSF's table (both use 30-second
epochs, plan §6.3):
  "30s"        →    1 epoch
  "10m"        →   20 epochs
  "40m"        →   80 epochs
  "80m"        →  160 epochs
  "120m"       →  240 epochs
  "240m"       →  480 epochs
  "full_night" → all epochs up to anchor_t (seq2seq) or whole night (seq2label)

PATCHES_PER_EPOCH=1, same reasoning as OSF: PhysioOmni's embedding
granularity is already one row per 30-second AASM epoch, so an "epoch" and
a "patch" are the same unit here too.

MINIMUM PAST CONTEXT (seq2seq only) — same formula and same 30s-epoch
units as OSF's fork (both use 30s epochs, so no unit recomputation is
needed the way OSF's fork needed relative to ContextWindowDataset's
5s-patch units):
    min_past = max(PATCHES_PER_EPOCH, min(N // min_past_denom, max_min_past))
  max_min_past_patches=40 represents the same 20-minute wall-clock cap as
  OSF's config. Set in configs/phase0_physioomni_config.yaml.
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
PATCH_SECONDS     = 30          # each PhysioOmni embedding row = one 30-second epoch
PATCHES_PER_EPOCH = 1           # 30-sec sleep epoch / 30-sec PhysioOmni embedding — same unit
REM_ORIGINAL      = 5
REM_REMAPPED      = 4
EMBED_DIM         = 500         # 200 (EEG) + 100 (EOG) + 100 (ECG) + 100 (EMG), already flat —
                                 # NO separate N_SUBTOKENS/FLAT_DIM distinction needed here (see
                                 # module docstring); EMBED_DIM *is* the per-epoch vector length.
FULL_NIGHT_SENTINEL = -1        # internal sentinel for full_night context length


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — identical to OSF's fork, pure integer arithmetic, copied as-is
# ─────────────────────────────────────────────────────────────────────────────

def parse_context_length(s) -> int:
    """Convert context-length string to number of 30-sec epochs.

    Returns FULL_NIGHT_SENTINEL (-1) for "full_night".

    Examples:
        "30s"        → 1
        "10m"        → 20
        "240m"       → 480
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


def _compute_min_past(N: int, denom: int = 8, max_patches: int = 40) -> int:
    """Minimum past patches required before an anchor is included.

    Formula: max(PATCHES_PER_EPOCH, min(N // denom, max_patches))
    For full_night (N=-1): use max_patches directly.
    Default max_patches=40 represents 20 minutes at 30s/epoch, same units
    as OSF's fork.
    """
    if N == FULL_NIGHT_SENTINEL:
        return max_patches
    return max(PATCHES_PER_EPOCH, min(N // denom, max_patches))


def _remap_stages(arr: np.ndarray) -> np.ndarray:
    """Remap REM stage 5 → 4 in-place (copy). Returns int8 array."""
    out = arr.copy().astype(np.int8)
    out[out == REM_ORIGINAL] = REM_REMAPPED
    return out


def _build_shape_cache(embedding_dir: Path) -> dict:
    """
    Scan all .npy files under embedding_dir and return {rel_key: T} where
    rel_key = "{dataset}/{subject_id}".

    The result is written to {embedding_dir}/shape_cache.json so subsequent
    calls return immediately. The cache is invalidated (rebuilt) only if
    new files appear that are not in the stored cache.
    """
    cache_path = embedding_dir / "shape_cache.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

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

class PhysioOmniContextWindowDataset(Dataset):
    """Fixed-length (or full-night) context-window dataset over PhysioOmni
    embeddings.

    Args:
        cfg            : Phase 0 PhysioOmni config dict (from
                         phase0_physioomni_config.yaml).
        split          : "train", "val", or "test".
        context_length : Duration string e.g. "10m", "full_night", or int (epochs).
        task           : Task name matching the subject CSV filename stem
                         (e.g. "sex_binary") or "sleep_staging".
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

        # ── seq2seq window design ─────────────────────────────────────────────
        self._seq2seq_context_mode     = ds_cfg.get("seq2seq_context_mode",        "causal")
        self._seq2seq_padding_policy   = ds_cfg.get("seq2seq_padding_policy",      "allow_all")
        self._seq2seq_max_padding_frac = float(ds_cfg.get("seq2seq_max_padding_fraction", 0.5))

        assert self._seq2seq_context_mode in ("causal", "centered"), (
            f"seq2seq_context_mode must be 'causal' or 'centered', "
            f"got {self._seq2seq_context_mode!r}"
        )
        assert self._seq2seq_padding_policy in ("allow_all", "max_fraction", "complete_only"), (
            f"seq2seq_padding_policy must be 'allow_all', 'max_fraction', or "
            f"'complete_only', got {self._seq2seq_padding_policy!r}"
        )

        if task_type == "seq2seq" and not self.is_full_night:
            self._half_past   = (self.N - PATCHES_PER_EPOCH) // 2
            self._half_future = self.N - PATCHES_PER_EPOCH - self._half_past
        else:
            self._half_past = self._half_future = 0

        min_past_denom   = ds_cfg.get("min_past_denom",   8)
        max_min_past     = ds_cfg.get("max_min_past_patches", 40)
        self._min_past   = _compute_min_past(self.N, min_past_denom, max_min_past)

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

        # ── Minimum recording length filter ───────────────────────────────
        # Applied BEFORE index building and at ALL context lengths so that the
        # same subject pool is used at every point on the context-length curve.
        # min_recording_patches is in 30s-epoch units (480 for 240m) — same
        # units as OSF's config, no recomputation needed since both use
        # 30-second epochs.
        self._min_recording_patches = ds_cfg.get("min_recording_patches", 0)
        if self._min_recording_patches > 0:
            T_series = self.df.apply(
                lambda r: self._shape_cache.get(
                    f"{r['dataset']}/{r['subject_id']}", 0
                ),
                axis=1,
            )
            keep = T_series >= self._min_recording_patches
            n_excluded = (~keep).sum()
            if n_excluded > 0:
                min_min = self._min_recording_patches * PATCH_SECONDS // 60
                warnings.warn(
                    f"[{split}] Cohort filter: {n_excluded} subject(s) excluded "
                    f"(T < {self._min_recording_patches} patches / {min_min}m). "
                    f"Set dataset.min_recording_patches=0 to disable.",
                    stacklevel=2,
                )
            self.df = self.df[keep].reset_index(drop=True)

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
    # Identical logic to OSF's fork — purely integer index-arithmetic on T
    # (from shape cache) and N (from parse_context_length), no reference to
    # embedding shape constants. Copied as-is.

    def _build_seq2seq_index(self) -> List[Tuple[int, int, int]]:
        """Build (row_idx, anchor_patch_end, stage_label) for every valid anchor."""
        policy    = self._seq2seq_padding_policy
        mode      = self._seq2seq_context_mode
        N         = self.N
        half_past = self._half_past
        half_fut  = self._half_future

        index = []
        for row_idx, row in self.df.iterrows():
            cache_key = f"{row['dataset']}/{row['subject_id']}"
            T = self._shape_cache[cache_key]

            ann_path = Path(row["annotation_path"])
            if not ann_path.exists():
                warnings.warn(f"Annotation not found: {ann_path} — subject skipped.")
                continue
            raw_stages = np.load(ann_path)
            stages     = _remap_stages(raw_stages)

            T_ann = len(stages) * PATCHES_PER_EPOCH
            T_eff = min(T, T_ann)
            n_epochs = T_eff // PATCHES_PER_EPOCH

            for epoch_idx in range(n_epochs):
                anchor_patch_start = epoch_idx * PATCHES_PER_EPOCH
                anchor_patch_end   = anchor_patch_start + PATCHES_PER_EPOCH

                label = int(stages[epoch_idx])
                if label < 0 or label > 4:
                    continue

                if self.is_full_night:
                    pass
                elif policy == "allow_all":
                    if mode == "causal" and anchor_patch_end < self._min_past:
                        continue
                else:
                    if mode == "causal":
                        pad = max(0, N - anchor_patch_end)
                    else:  # centered
                        left_pad  = max(0, half_past - anchor_patch_start)
                        right_pad = max(0, (anchor_patch_end + half_fut) - T_eff)
                        pad       = left_pad + right_pad

                    if policy == "complete_only" and pad > 0:
                        continue
                    elif policy == "max_fraction" and pad / N > self._seq2seq_max_padding_frac:
                        continue

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
                index.append((row_idx, 0, label))
                continue

            N = self.N
            if T < N:
                index.append((row_idx, 0, label))
                continue

            if self.split == "train":
                n_valid = T - N + 1
                K = min(self._K_max, n_valid)
                starts = sorted(
                    rng.choice(n_valid, size=K, replace=False).tolist()
                )
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
        row_idx, aux, label = self._index[idx]
        row = self.df.iloc[row_idx]

        npy_path = self.embedding_dir / row["dataset"] / f"{row['subject_id']}.npy"
        path_str = str(npy_path)

        # Per-worker mmap cache: reuse the open mmap when consecutive items share
        # the same subject file. Same SubjectGroupedSampler-dependent optimization
        # as OSF's fork.
        if getattr(self, "_cached_path", None) != path_str:
            self._cached_path = path_str
            self._cached_emb  = np.load(npy_path, mmap_mode="r")  # (T, 500) float16
        emb = self._cached_emb
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
    # Simpler than OSF's fork: embeddings are already 2D [T, 500] (no
    # N_SUBTOKENS middle dim, see module docstring), so pad blocks are plain
    # (n, EMBED_DIM) and no post-hoc reshape is needed anywhere below.

    def _get_seq2seq_window(
        self, emb: np.ndarray, T: int, anchor_patch_end: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract context window for seq2seq anchor.

        Dispatches to causal or centered based on seq2seq_context_mode.
        full_night is always causal (window = all past patches up to anchor).

        Returns:
            x    : (N, 500) float32
            mask : (N,)     bool — True = padded position (no real signal)
        """
        if self.is_full_night or self._seq2seq_context_mode == "causal":
            return self._get_causal_window(emb, T, anchor_patch_end)
        else:
            return self._get_centered_window(emb, T, anchor_patch_end)

    def _get_causal_window(
        self, emb: np.ndarray, T: int, anchor_patch_end: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Past-only window: [anchor_patch_end - N : anchor_patch_end].
        Left-padded when anchor_patch_end < N.
        """
        N = anchor_patch_end if self.is_full_night else self.N
        win_start = anchor_patch_end - N    # may be negative

        if win_start >= 0:
            window = emb[win_start : anchor_patch_end]
            mask   = np.zeros(N, dtype=bool)
        else:
            pad_len  = -win_start
            real_len = anchor_patch_end
            window = np.concatenate([
                np.zeros((pad_len, EMBED_DIM), dtype=np.float16),
                emb[:real_len],
            ], axis=0)
            mask = np.array([True] * pad_len + [False] * real_len, dtype=bool)

        x = window.astype(np.float32)      # (N, 500) — always a copy, safe to mutate
        return x, mask

    def _get_centered_window(
        self, emb: np.ndarray, T: int, anchor_patch_end: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Symmetric window centred on anchor epoch.

        Window layout (total = N patches):
            [half_past patches before anchor] [PATCHES_PER_EPOCH anchor patches] [half_future patches after]
        Left- and/or right-padded when near recording boundaries.
        """
        N                  = self.N
        anchor_patch_start = anchor_patch_end - PATCHES_PER_EPOCH
        win_start          = anchor_patch_start - self._half_past   # may be negative
        win_end            = anchor_patch_end   + self._half_future  # may exceed T

        if win_start >= 0 and win_end <= T:
            window = emb[win_start : win_end]
            mask   = np.zeros(N, dtype=bool)
        else:
            pieces_w, pieces_m = [], []

            if win_start < 0:
                lp = -win_start
                pieces_w.append(np.zeros((lp, EMBED_DIM), dtype=np.float16))
                pieces_m.append(np.ones(lp, dtype=bool))

            real_s = max(0, win_start)
            real_e = min(T, win_end)
            pieces_w.append(emb[real_s : real_e])
            pieces_m.append(np.zeros(real_e - real_s, dtype=bool))

            if win_end > T:
                rp = win_end - T
                pieces_w.append(np.zeros((rp, EMBED_DIM), dtype=np.float16))
                pieces_m.append(np.ones(rp, dtype=bool))

            window = np.concatenate(pieces_w, axis=0)
            mask   = np.concatenate(pieces_m)

        x = window.astype(np.float32)      # (N, 500) — always a copy, safe to mutate
        return x, mask

    def _get_seq2label_window(
        self, emb: np.ndarray, T: int, window_start: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract window of length N starting at window_start.

        Right-padded when the recording ends before window_start + N.

        Returns:
            x    : (N, 500) float32
            mask : (N,)     bool — True = right-padded position
        """
        if self.is_full_night:
            N = T                           # whole recording, no padding needed
        else:
            N = self.N

        end = window_start + N
        if end <= T:
            window = emb[window_start : end]   # (N, 500)
            mask   = np.zeros(N, dtype=bool)
        else:
            real_len = T - window_start
            pad_len  = N - real_len
            window = np.concatenate([
                emb[window_start : T],
                np.zeros((pad_len, EMBED_DIM), dtype=np.float16),
            ], axis=0)
            mask = np.array([False] * real_len + [True] * pad_len, dtype=bool)

        x = window.astype(np.float32)      # (N, 500) — always a copy, safe to mutate
        return x, mask

    # ── full_night collate ─────────────────────────────────────────────────

    @staticmethod
    def collate_fn(batch):
        """Pad variable-length full_night samples to the longest in the batch.

        Use as: DataLoader(..., collate_fn=PhysioOmniContextWindowDataset.collate_fn)
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

    # Same task->num_classes table as OSF's fork, for parity — only the
    # Tier-1 tasks (minus apnea_binary, excluded for PhysioOmni per plan
    # §0.4) are actually exercised yet, the rest kept for future parity.
    _TASK_NUM_CLASSES = {
        "sex_binary":               2,
        "sleep_efficiency_binary":  2,
        "bmi_binary":               2,
        "age_class":                3,
        "sleep_staging":    5,   # Wake, N1, N2, N3, REM (seq2seq) — deferred, plan §12
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
        if self.task_type == "seq2seq":
            return 5
        return int(self.df["label"].max()) + 1

    def __repr__(self) -> str:
        ctx = "full_night" if self.is_full_night else f"{self.N} epochs ({self.N * PATCH_SECONDS}s)"
        return (
            f"PhysioOmniContextWindowDataset("
            f"split={self.split}, context={ctx}, "
            f"task_type={self.task_type}, n_items={len(self._index)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# SubjectGroupedSampler — identical to OSF's fork, copied as-is (kept as its
# own class here rather than imported, so this module has no import
# dependency on the OSF-side file — see plan §1's worktree isolation rule).
# ─────────────────────────────────────────────────────────────────────────────

class SubjectGroupedSampler(torch.utils.data.Sampler):
    """Yield item indices grouped by subject, with per-epoch subject-order shuffle.

    Items that share a subject (same ``row_idx`` in the dataset's ``_index``) are
    emitted consecutively. The ORDER of subjects is reshuffled each time
    ``__iter__`` is called so each epoch sees subjects in a different order.

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
