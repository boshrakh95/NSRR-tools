#!/usr/bin/env python3
"""
physioomni_raw_epoch_dataset.py — PhysioOmni baseline, Stage 2 (LoRA)
raw-signal dataset (plan §15.4)

PyTorch Dataset that serves fixed-length context windows of RAW PhysioOmni
input signal (not precomputed embeddings) for LoRA fine-tuning. Forked
structurally from osf_raw_epoch_dataset.py (OSFRawEpochWindowDataset): the
subject-list/split/K-sampling/window-position index-building logic is
IDENTICAL (pure integer arithmetic over T epochs and N context length, no
reference to what's actually stored per epoch) and copied near-verbatim.

WHAT'S GENUINELY DIFFERENT FROM OSF'S VERSION (plan §15.2/15.3/15.4) — not
just a renamed copy:
  - OSF's raw signal is always exactly [12, n_samples_64] per subject
    (missing channels zero-filled at the raw level) — every subject has an
    identical tensor shape. PhysioOmni's channels are genuinely
    PRESENT-OR-ABSENT per subject (never zero-filled at this stage — same
    contract as Stage 1's embeddings), so __getitem__ returns a dict keyed
    by modality (value None if that modality is entirely absent for this
    subject), not one fixed-shape tensor.
  - EEG additionally varies in CHANNEL COUNT (1 for SHHS, 2 elsewhere,
    §4.5) even among subjects that DO have EEG — so the custom collate_fn
    below groups EEG items by channel count (1 vs 2) as well as by
    presence, since a single batched tensor can't hold different sequence
    lengths.
  - Reads from the per-subject-per-slot cache (plan §15.3,
    physioomni_channel_loader.load_signal_cache), not one unified matrix.

READS FROM THE PRECOMPUTED RAW SIGNAL CACHE (plan §15.3), NOT the raw
HDF5 — scripts/precompute_physioomni_raw_signal_cache.py builds this once,
offline, CPU-only; this class just reads it.

SUBJECT/SPLIT SELECTION mirrors PhysioOmniContextWindowDataset (Stage 1)
exactly: task_subject_dir, split_seed, train/val/test proportions.
Subjects are filtered by "has a Stage 1 PhysioOmni embedding file" FIRST
(existence check only, contents never read) — required so Stage 1 and
Stage 2 always train/evaluate on identical subjects/splits (same
np.random.default_rng(split_seed).shuffle() population-size-sensitivity
issue OSF's own Stage 2 hit and fixed — see osf_raw_epoch_dataset.py's own
module docstring for the full original diagnosis).

SCOPE: seq2label only, matching Stage 1's own current scope. No stages
dataset (plan §15.3 — no PhysioOmni Tier-1 task needs it).

INPUT
─────
  {raw_signal_cache_dir}/{dataset}/{subject_id}/  (precomputed by
  scripts/precompute_physioomni_raw_signal_cache.py — run that first. A
  subject present in the task CSV/split but missing from the cache raises
  a clear error at dataset-construction time — see __init__ below.)

OUTPUT PER __getitem__ (BEFORE collate — see collate_fn for the batched shape)
───────────────────────────────────────────────────────────────────────────
  {
    "EEG": (x, ["C3"] or ["C3","C4"]) or None,   x: float32 [n_chans, N, 6000]
    "EOG": (x, ["HEO"]) or None,                 x: float32 [1, N, 6000]
    "ECG": (x, ["ECG"]) or None,                 x: float32 [1, N, 15000]
    "EMG": (x, ["EMG"]) or None,                 x: float32 [1, N, 15000]
    "mask": bool [N],   True = right-padded position (recording too short)
    "y":    int
  }
  6000 = 30s * 200Hz (EEG/EOG native rate); 15000 = 30s * 500Hz (ECG/EMG).
  Each channel's [N, epoch_samples] slab is still CONTINUOUS per-epoch raw
  signal, not yet patchified into PhysioOmni's patch_size chunks — the
  combined model's forward() does that (plan §15.5), matching how
  extract_physioomni_embeddings.py's _modality_forward() divides the same
  responsibility.
"""

import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from nsrr_tools.datasets.physioomni_channel_loader import (
    EPOCH_SECONDS,
    NATIVE_HZ,
    cache_exists,
    get_cached_t_epochs,
    load_signal_cache,
)

# ── Constants — identical semantics to physioomni_context_window_dataset.py ──
PATCH_SECONDS = 30
PATCHES_PER_EPOCH = 1
FULL_NIGHT_SENTINEL = -1
MODALITIES = ["EEG", "EOG", "ECG", "EMG"]


def parse_context_length(s) -> int:
    """Convert context-length string to number of 30-sec epochs. Identical
    to physioomni_context_window_dataset.py's — copied as-is (no import
    dependency between the two dataset modules, same convention OSF uses)."""
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
    """{dataset}/{subject_id} -> t_epochs, via meta.json only (no array
    touched at all — even cheaper than OSF's own get_cached_epoch_count,
    which still opens an npy header). Returns (shape_dict, missing_keys)."""
    cache = {}
    missing = []
    for _, row in df.iterrows():
        key = f"{row['dataset']}/{row['subject_id']}"
        if cache_exists(cache_dir, row["dataset"], row["subject_id"]):
            cache[key] = get_cached_t_epochs(cache_dir, row["dataset"], row["subject_id"])
        else:
            missing.append(key)
    return cache, missing


class PhysioOmniRawEpochWindowDataset(Dataset):
    """Fixed-length context-window dataset over RAW PhysioOmni input signal.

    Args:
        cfg            : Phase 0 PhysioOmni LoRA config dict
                         (phase0_physioomni_lora_config.yaml).
        split          : "train", "val", or "test".
        context_length : Duration string e.g. "10m", or int (epochs). No
                         full_night support yet (deferred, matches OSF's
                         own Stage 2 scope).
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
            "PhysioOmniRawEpochWindowDataset only supports seq2label so far "
            "(matches Stage 1's current scope) — seq2seq not implemented yet."
        )

        self.split = split
        self.task = task
        self.seed = seed
        self.cache_dir = Path(data_cfg["raw_signal_cache_dir"])

        self.N = parse_context_length(context_length)
        if self.N == FULL_NIGHT_SENTINEL:
            raise NotImplementedError(
                "full_night not supported by PhysioOmniRawEpochWindowDataset yet."
            )

        self._K_max = ds_cfg.get("windows_per_subject", 5)

        # ── Load subject list — identical logic to PhysioOmniContextWindowDataset ──
        task_subject_dir = Path(ds_cfg["task_subject_dir"])
        csv_path = task_subject_dir / f"{task}_subjects.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Subject CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if datasets:
            df = df[df["dataset"].isin(datasets)].reset_index(drop=True)

        # Filter by Stage 1 embedding-file existence — existence check
        # ONLY, no embedding contents ever read — so len(df) and subject
        # order exactly match what PhysioOmniContextWindowDataset (Stage
        # 1) used at split-computation time (plan §15.4 — same
        # split-matching discipline as OSF's own real, previously-live
        # bug fix, applied here from the start).
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

        # ── Train / val / test split — identical to PhysioOmniContextWindowDataset ─
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

        # ── Shape cache (t_epochs, meta.json-only read) + hard completeness
        # check — a subject selected above (same pool as Stage 1) MUST have
        # its raw-signal cache precomputed, or Stage 2 would either crash
        # confusingly mid-training or silently read stale/wrong data. Fail
        # loudly and immediately instead. ─────────────────────────────────
        self._shape_cache, _missing_cache = _build_raw_shape_cache(self.cache_dir, self.df)
        if _missing_cache:
            preview = ", ".join(_missing_cache[:10])
            more = f" (+{len(_missing_cache) - 10} more)" if len(_missing_cache) > 10 else ""
            raise FileNotFoundError(
                f"[{split}] {len(_missing_cache)}/{len(self.df)} selected subjects have no "
                f"precomputed raw-signal cache under {self.cache_dir}: {preview}{more}\n"
                f"Run scripts/precompute_physioomni_raw_signal_cache.py first (see "
                f"docs/PHYSIOOMNI_EXPERIMENTS_GUIDE.md Step 8) — these subjects are part of "
                f"Stage 1's subject pool and cannot be silently dropped without breaking "
                f"the Stage 1/Stage 2 split match."
            )

        # ── Minimum recording length filter — identical logic/units to
        # PhysioOmniContextWindowDataset (480 epochs = 240m at 30s/epoch) ──
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

    # ── Index builder — identical arithmetic to PhysioOmniContextWindowDataset's
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
        dataset, subject_id = row["dataset"], row["subject_id"]

        # Per-worker single-subject cache — avoids re-reading the same
        # subject's cache files for every window when SubjectGroupedSampler
        # keeps items from one subject consecutive (same pattern as OSF's
        # raw dataset).
        cache_key = f"{dataset}/{subject_id}"
        if getattr(self, "_cached_key", None) != cache_key:
            self._cached_key = cache_key
            self._cached_signals = load_signal_cache(self.cache_dir, dataset, subject_id)
        signals = self._cached_signals

        T = self._shape_cache[cache_key]
        N = self.N
        end = window_start + N
        real_len = min(N, max(0, T - window_start))
        pad_len = N - real_len

        item = {"y": int(label)}
        for modality in MODALITIES:
            chans = signals[modality]
            if not chans:
                item[modality] = None
                continue
            epoch_samples = EPOCH_SECONDS * NATIVE_HZ[modality]
            per_chan = []
            for _, arr in chans:
                if real_len > 0:
                    s0 = window_start * epoch_samples
                    s1 = (window_start + real_len) * epoch_samples
                    real = arr[s0:s1].reshape(real_len, epoch_samples)
                else:
                    real = np.empty((0, epoch_samples), dtype=np.float32)
                if pad_len > 0:
                    pad = np.zeros((pad_len, epoch_samples), dtype=np.float32)
                    chunk = np.concatenate([real, pad], axis=0)
                else:
                    chunk = real
                per_chan.append(chunk)
            x = np.stack(per_chan, axis=0).astype(np.float32)  # [n_chans, N, epoch_samples]
            item[modality] = (x, [lab for lab, _ in chans])

        item["mask"] = np.array([False] * real_len + [True] * pad_len, dtype=bool)
        return item

    # ── Convenience ───────────────────────────────────────────────────────

    _TASK_NUM_CLASSES = {
        "sex_binary": 2,
        "sleep_efficiency_binary": 2,
        "bmi_binary": 2,
        "age_class": 3,
    }

    @property
    def num_classes(self) -> int:
        if self.task in self._TASK_NUM_CLASSES:
            return self._TASK_NUM_CLASSES[self.task]
        return int(self.df["label"].max()) + 1

    def __repr__(self) -> str:
        return (
            f"PhysioOmniRawEpochWindowDataset("
            f"split={self.split}, context={self.N} epochs ({self.N * PATCH_SECONDS}s), "
            f"n_items={len(self._index)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Custom collate — groups by (modality, channel-count) so variable-length
# per-modality sequences (EEG: 1 or 2 channels) can be batched, plan §15.2/15.4
# ─────────────────────────────────────────────────────────────────────────────

class PhysioOmniLoRABatch:
    """Wraps the per-(modality, channel-count)-grouped raw signal batch
    (plan §15.2/15.4) so it can flow through
    train_physioomni_context_sweep.py's run_epoch() UNMODIFIED —
    run_epoch does `x = x.to(device, non_blocking=True)` and
    `total_loss += loss.item() * x.size(0)` on whatever `x` is (it has no
    idea x isn't a plain tensor for Stage 2), so this class implements
    exactly those two methods, nothing more. `model(x, mask)` then
    receives this object as `x` and a plain BoolTensor as `mask`.

    Structure:
      self.eeg  : {1: {"batch_idx": LongTensor[k], "x": FloatTensor[k,1,N,6000], "labels": ["C3"]},
                   2: {"batch_idx": LongTensor[k], "x": FloatTensor[k,2,N,6000], "labels": ["C3","C4"]}}
                  (only channel-counts actually present in this batch appear as keys —
                   EEG is grouped by channel count since different subjects have 1 or
                   2 real EEG channels, §4.5, which can't share one batched tensor)
      self.eog/ecg/emg : {"batch_idx": LongTensor[k], "x": FloatTensor[k,1,N,epoch_samples],
                           "labels": [...]} or None (always exactly 1 channel when present)
      self.batch_size  : B (int) — for .size(0)
    """

    def __init__(self, eeg: dict, eog, ecg, emg, batch_size: int):
        self.eeg = eeg
        self.eog = eog
        self.ecg = ecg
        self.emg = emg
        self.batch_size = batch_size

    @staticmethod
    def _move_group(g, device, non_blocking):
        if g is None:
            return None
        return {
            "batch_idx": g["batch_idx"].to(device, non_blocking=non_blocking),
            "x": g["x"].to(device, non_blocking=non_blocking),
            "labels": g["labels"],
        }

    def to(self, device, non_blocking=False):
        eeg_moved = {n: self._move_group(g, device, non_blocking) for n, g in self.eeg.items()}
        return PhysioOmniLoRABatch(
            eeg_moved,
            self._move_group(self.eog, device, non_blocking),
            self._move_group(self.ecg, device, non_blocking),
            self._move_group(self.emg, device, non_blocking),
            self.batch_size,
        )

    def size(self, dim: int) -> int:
        assert dim == 0, "PhysioOmniLoRABatch only supports size(0) (batch dimension)."
        return self.batch_size


def physioomni_lora_collate_fn(batch: list):
    """Batch a list of __getitem__ dicts into the (x, mask, y) 3-tuple
    train_physioomni_context_sweep.py's run_epoch() expects (plan
    §15.2/15.5) — x is a PhysioOmniLoRABatch (see above), mask is a plain
    BoolTensor[B,N], y is a plain LongTensor[B]. Matches exactly what a
    normal Dataset+DataLoader (x, mask, y) tuple looks like from
    run_epoch's point of view, even though x internally holds a
    per-modality-grouped structure instead of one tensor.
    """
    B = len(batch)
    y = torch.tensor([b["y"] for b in batch], dtype=torch.long)
    mask = torch.from_numpy(np.stack([b["mask"] for b in batch], axis=0))

    # EEG — grouped by channel count.
    eeg_groups: dict = {}
    for b_idx, b in enumerate(batch):
        entry = b["EEG"]
        if entry is None:
            continue
        x, labels = entry
        n_chans = x.shape[0]
        eeg_groups.setdefault(n_chans, {"idx": [], "x": [], "labels": labels})
        eeg_groups[n_chans]["idx"].append(b_idx)
        eeg_groups[n_chans]["x"].append(x)
    eeg = {
        n_chans: {
            "batch_idx": torch.tensor(g["idx"], dtype=torch.long),
            "x": torch.from_numpy(np.stack(g["x"], axis=0)),  # [k, n_chans, N, epoch_samples]
            "labels": g["labels"],
        }
        for n_chans, g in eeg_groups.items()
    }

    # EOG/ECG/EMG — simple present/absent grouping (always 1 channel when present).
    modality_groups = {}
    for modality in ("EOG", "ECG", "EMG"):
        idx, xs, labels = [], [], None
        for b_idx, b in enumerate(batch):
            entry = b[modality]
            if entry is None:
                continue
            x, labs = entry
            idx.append(b_idx)
            xs.append(x)
            labels = labs
        if idx:
            modality_groups[modality] = {
                "batch_idx": torch.tensor(idx, dtype=torch.long),
                "x": torch.from_numpy(np.stack(xs, axis=0)),  # [k, 1, N, epoch_samples]
                "labels": labels,
            }
        else:
            modality_groups[modality] = None

    x_batch = PhysioOmniLoRABatch(
        eeg, modality_groups["EOG"], modality_groups["ECG"], modality_groups["EMG"], B
    )
    return x_batch, mask, y


# ─────────────────────────────────────────────────────────────────────────────
# SubjectGroupedSampler — identical to osf_raw_epoch_dataset.py's (itself
# identical to every other dataset file's own copy) — copied as-is, not
# imported, so this module has no import dependency on the SleepFM/OSF/
# Stage-1-PhysioOmni dataset files.
# ─────────────────────────────────────────────────────────────────────────────

class SubjectGroupedSampler(torch.utils.data.Sampler):
    """Yield item indices grouped by subject, with per-epoch subject-order shuffle.

    Usage::

        sampler = SubjectGroupedSampler(train_ds._index)
        loader  = DataLoader(train_ds, batch_size=32, sampler=sampler,
                             shuffle=False, persistent_workers=True,
                             collate_fn=physioomni_lora_collate_fn, ...)
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
