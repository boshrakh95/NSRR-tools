#!/usr/bin/env python3
"""
context_window_dataset.py — Phase 0, Step 2

PyTorch Dataset that serves fixed-length context windows of SleepFM embeddings
for the context-length sweep experiments.

INPUT
─────
  {embedding_dir}/{dataset}/{subject_id}.npy
    dtype  : float16
    shape  : [T, 4, 128]   T = total 5-sec patches for the full night

  For seq2label: subject CSV with columns [unified_id, dataset, subject_id, visit, label]
  For seq2seq:   sleep_staging_subjects.csv with [unified_id, ..., annotation_path, n_epochs]

OUTPUT (per __getitem__)
────────────────────────
  embeddings : float32 tensor  [N, 512]   N = context_patches, 512 = 4×128
  mask       : bool tensor     [N]        True = padded position (recording shorter than N)
  label      : int64 tensor    []         scalar for seq2label
               int64 tensor    [N]        per-patch stage for seq2seq

CONTEXT LENGTH STRINGS
──────────────────────
  "30s" →   6 patches   (6 × 5s = 30s)
  "2m"  →  24 patches
  "5m"  →  60 patches
  "10m" → 120 patches
  "20m" → 240 patches
  "40m" → 480 patches
  "80m" → 960 patches

SLEEP STAGING LABEL ALIGNMENT
──────────────────────────────
  Annotations are stored as 30-sec epochs (one value per epoch).
  We expand to 5-sec resolution by repeating each label 6×.
  Then we truncate to min(T_embedding, T_annotation) to avoid off-by-one
  discrepancies at recording edges.

  Stage remapping: 5 (REM in NSRR/AASM convention) → 4
  Final classes: 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM

WINDOW SAMPLING
───────────────
  Train split : random start position (data augmentation)
  Val/Test split:
    seq2label → center window (deterministic, no randomness)
    seq2seq   → center window (same)
  Recordings shorter than N patches → zero-padded, mask marks padding.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── Constants ─────────────────────────────────────────────────────────────────
PATCH_SECONDS = 5          # each embedding patch = 5 seconds
PATCHES_PER_EPOCH = 6      # 30-sec sleep-staging epoch / 5-sec patch
REM_ORIGINAL = 5
REM_REMAPPED = 4
EMBED_DIM = 128
N_MODALITIES = 4
FLAT_DIM = N_MODALITIES * EMBED_DIM  # 512


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_context_length(s: str) -> int:
    """Convert context-length string to number of 5-sec patches.

    Examples:
        "30s" → 6
        "2m"  → 24
        "10m" → 120
    """
    s = s.strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)(s|m)", s)
    if m is None:
        raise ValueError(f"Cannot parse context length: {s!r}. Expected e.g. '30s', '10m'.")
    value, unit = float(m.group(1)), m.group(2)
    seconds = value if unit == "s" else value * 60
    patches = seconds / PATCH_SECONDS
    if not patches.is_integer():
        raise ValueError(f"Context length {s!r} → {seconds}s is not divisible by {PATCH_SECONDS}s patch size.")
    return int(patches)


def _expand_stage_labels(epoch_labels: np.ndarray, n_patches: int) -> np.ndarray:
    """Expand 30-sec epoch labels to 5-sec patch resolution.

    Args:
        epoch_labels : (n_epochs,) int array of sleep stages
        n_patches    : number of embedding patches in the recording

    Returns:
        (min(n_patches, n_epochs * PATCHES_PER_EPOCH),) int8 array
        with 5→4 remapping applied.
    """
    # Remap REM: 5 → 4
    labels = epoch_labels.copy()
    labels[labels == REM_ORIGINAL] = REM_REMAPPED

    # Repeat each epoch label PATCHES_PER_EPOCH times (30s → 5s resolution)
    expanded = np.repeat(labels, PATCHES_PER_EPOCH)  # (n_epochs * 6,)

    # Align with embedding length
    T = min(len(expanded), n_patches)
    return expanded[:T].astype(np.int8)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ContextWindowDataset(Dataset):
    """Fixed-length context-window dataset over SleepFM embeddings.

    Args:
        cfg            : Phase 0 config dict (from phase0_config.yaml).
        split          : "train", "val", or "test".
        context_length : Duration string e.g. "10m", or int (number of patches).
        task           : Task name matching the subject CSV filename stem
                         (e.g. "apnea_binary") or "sleep_staging".
        task_type      : "seq2label" or "seq2seq".
        datasets       : Optional list of datasets to restrict to. Defaults to
                         all datasets in the subject CSV.
        seed           : RNG seed for train-split random window sampling.
    """

    def __init__(
        self,
        cfg: dict,
        split: str,
        context_length,
        task: str = None,
        task_type: str = None,
        datasets: list = None,
        seed: int = 42,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"

        ds_cfg   = cfg["dataset"]
        task     = task     or ds_cfg["task"]
        task_type= task_type or ds_cfg["task_type"]

        assert task_type in ("seq2label", "seq2seq"), f"Unknown task_type: {task_type}"

        self.embedding_dir = Path(ds_cfg["embedding_dir"])
        self.split         = split
        self.task_type     = task_type
        self.seed          = seed

        # Parse context length
        if isinstance(context_length, int):
            self.N = context_length
        else:
            self.N = parse_context_length(context_length)

        # ── Load subject list ──────────────────────────────────────────────
        task_subject_dir = Path(ds_cfg["task_subject_dir"])
        csv_path = task_subject_dir / f"{task}_subjects.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Subject CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Filter by dataset
        if datasets:
            df = df[df["dataset"].isin(datasets)].reset_index(drop=True)

        # Filter to subjects that have an embedding file
        mask = df.apply(
            lambda r: (self.embedding_dir / r["dataset"] / f"{r['subject_id']}.npy").exists(),
            axis=1,
        )
        n_total = len(df)
        df = df[mask].reset_index(drop=True)
        n_missing = n_total - len(df)
        if n_missing > 0:
            import warnings
            warnings.warn(
                f"{n_missing}/{n_total} subjects have no embedding file and will be skipped.",
                stacklevel=2,
            )

        # ── Train/val/test split ───────────────────────────────────────────
        rng = np.random.default_rng(cfg["dataset"]["split_seed"])
        idx = np.arange(len(df))
        rng.shuffle(idx)

        n      = len(idx)
        n_train = int(n * cfg["dataset"]["train_split"])
        n_val   = int(n * cfg["dataset"]["val_split"])

        if split == "train":
            idx = idx[:n_train]
        elif split == "val":
            idx = idx[n_train : n_train + n_val]
        else:  # test
            idx = idx[n_train + n_val :]

        self.df = df.iloc[idx].reset_index(drop=True)

        # For seq2label: pre-read scalar labels from CSV
        # For seq2seq:   annotation_path is used per __getitem__
        if task_type == "seq2label":
            if "label" not in self.df.columns:
                raise ValueError(
                    f"seq2label requires a 'label' column in {csv_path}. "
                    f"Found columns: {list(self.df.columns)}"
                )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # ── Load embedding ─────────────────────────────────────────────────
        npy_path = self.embedding_dir / row["dataset"] / f"{row['subject_id']}.npy"
        emb = np.load(npy_path)          # (T, 4, 128) float16
        T = emb.shape[0]

        # ── Get label / stage sequence ─────────────────────────────────────
        if self.task_type == "seq2label":
            label_val = int(row["label"])
            stage_seq = None
        else:  # seq2seq
            raw_stages = np.load(row["annotation_path"])  # (n_epochs,) int8
            stage_seq  = _expand_stage_labels(raw_stages, T)  # (T_aligned,) int8
            T_aligned  = len(stage_seq)
            # Trim embedding to aligned length so indices always match
            emb = emb[:T_aligned]
            T   = T_aligned
            label_val = None

        # ── Sample window ──────────────────────────────────────────────────
        N = self.N
        if T >= N:
            # Enough data: pick a window
            if self.split == "train":
                rng = np.random.default_rng(self.seed + idx)
                start = int(rng.integers(0, T - N + 1))
            else:
                start = max(0, (T - N) // 2)  # center window
            window_emb    = emb[start : start + N]           # (N, 4, 128)
            window_mask   = np.zeros(N, dtype=bool)
            if stage_seq is not None:
                window_stages = stage_seq[start : start + N]  # (N,)
        else:
            # Shorter than N: zero-pad on the right
            pad = N - T
            window_emb  = np.concatenate(
                [emb, np.zeros((pad, N_MODALITIES, EMBED_DIM), dtype=np.float16)], axis=0
            )
            window_mask = np.array([False] * T + [True] * pad, dtype=bool)
            if stage_seq is not None:
                window_stages = np.concatenate(
                    [stage_seq, np.full(pad, -1, dtype=np.int8)], axis=0
                )

        # ── Build output tensors ───────────────────────────────────────────
        # Flatten modality dim: (N, 4, 128) → (N, 512)
        x = torch.from_numpy(window_emb.astype(np.float32).reshape(N, FLAT_DIM))
        m = torch.from_numpy(window_mask)

        if self.task_type == "seq2label":
            y = torch.tensor(label_val, dtype=torch.long)
        else:
            y = torch.from_numpy(window_stages.astype(np.int64))

        return x, m, y

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def num_classes(self) -> int:
        """Infer number of classes from the label column or task type."""
        if self.task_type == "seq2seq":
            return 5  # Wake, N1, N2, N3, REM
        return int(self.df["label"].max()) + 1

    def __repr__(self) -> str:
        return (
            f"ContextWindowDataset("
            f"split={self.split}, N={self.N} patches ({self.N * PATCH_SECONDS}s), "
            f"task_type={self.task_type}, n_subjects={len(self.df)})"
        )
