#!/usr/bin/env python3
"""
precompute_osf_raw_signal_cache.py — OSF baseline, Stage 2 (LoRA) offline
raw-signal cache precompute (checklist 2.5b)

WHY THIS EXISTS
────────────────
train_osf_lora.py's OSFRawEpochWindowDataset was reading raw signal live
via load_and_resample_channels() on every __getitem__ (channel search
across the raw HDF5 + 128→64Hz resample), plus a ~47-minute HDF5-metadata
scan to build the epoch-count shape cache — BOTH redone from scratch on
every single training job, for every (task, head, context) combination,
even though the resampled signal is completely independent of task/head/
context/the LoRA-adapted backbone. Diagnosed live on a real GPU job
(54716906): 2+ hours with the GPU essentially idle (22GB just holding the
model) while 2 DataLoader workers burned CPU on redundant HDF5 reads,
because the default `shuffle=True` train loader defeats the per-worker
single-subject cache (consecutive shuffled items rarely share a subject).

This script precomputes the resampled 12-channel, 64Hz signal ONCE per
subject, CPU-only (no GPU, no model — just I/O + decimation), so every
subsequent Stage 2 job across every task/head/context reads a cheap
memory-mapped .npy instead of touching the raw HDF5 at all. Does NOT
touch subject/split selection — task_subject_dir/split_seed are untouched,
so Stage 1 and Stage 2 keep using identical subjects/splits.

OUTPUT FORMAT
─────────────
  {cache_dir}/{dataset}/{subject_id}.npy
  dtype  : float16 (halves storage; underlying signal is already z-scored,
           see osf_channel_loader.save_signal_cache's docstring)
  shape  : [12, n_samples_64]  — resampled, NOT epoch-chunked; downstream
           code (OSFRawEpochWindowDataset) slices epoch-aligned windows
           directly from this array.

Same channel mapping as extract_osf_embeddings.py (Stage 1) and
train_osf_lora.py (Stage 2 live path) — all three call the same
build_channel_candidates()/load_and_resample_channels() from
nsrr_tools.datasets.osf_channel_loader, so there is exactly one
implementation of channel mapping/resampling logic anywhere in the repo.

USAGE
─────
  # Small test
  python scripts/precompute_osf_raw_signal_cache.py \\
      --config configs/phase0_osf_lora_config.yaml --datasets apples --limit 5

  # Sharded full run (CPU only — see jobs/precompute_osf_raw_signal_cache.sh)
  python scripts/precompute_osf_raw_signal_cache.py \\
      --config configs/phase0_osf_lora_config.yaml \\
      --start-idx 0 --end-idx 5000 --num-workers 8

  # Re-run to fill in gaps (already-cached subjects are skipped automatically)
  python scripts/precompute_osf_raw_signal_cache.py --config configs/phase0_osf_lora_config.yaml
"""

import argparse
import json
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nsrr_tools.datasets.osf_channel_loader import (  # noqa: E402
    build_channel_candidates,
    load_and_resample_channels,
    save_signal_cache,
    cache_path_for,
)

# ── Graceful-stop flag ─────────────────────────────────────────────────────
# Same philosophy as extract_osf_embeddings.py: finish in-flight work, then
# stop, rather than an abrupt exit. With a process pool, "in-flight" means
# the currently-submitted batch, not a single subject — see main() below.
_stop_requested = False

def _handle_sigterm(signum, frame):
    global _stop_requested
    logger.warning("[SIGTERM] Stop requested — will exit after the current batch completes.")
    _stop_requested = True

signal.signal(signal.SIGTERM, _handle_sigterm)


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject worker (module-level function — required for ProcessPoolExecutor
# pickling)
# ─────────────────────────────────────────────────────────────────────────────

def _process_one_subject(args_tuple):
    dataset, subject_id, h5_path_str, cache_dir_str, cfg_candidates = args_tuple
    h5_path = Path(h5_path_str)
    out_path = cache_path_for(cache_dir_str, dataset, subject_id)
    try:
        channel_candidates = build_channel_candidates(dataset, cfg_candidates)
        signal_matrix, fill_info = load_and_resample_channels(h5_path, channel_candidates)
        save_signal_cache(out_path, signal_matrix)
        return (dataset, subject_id, "ok", signal_matrix.shape, fill_info, None)
    except Exception as exc:
        return (dataset, subject_id, "error", None, None, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Subject discovery — identical convention to extract_osf_embeddings.py
# ─────────────────────────────────────────────────────────────────────────────

def find_hdf5_files(hdf5_dir: str, datasets: list) -> list:
    root = Path(hdf5_dir)
    subjects = []
    for dataset in datasets:
        h5_dir = root / dataset / "derived" / "hdf5_signals"
        if not h5_dir.exists():
            logger.warning(f"HDF5 dir not found, skipping: {h5_dir}")
            continue
        files = sorted(h5_dir.glob("*.h5"))
        for fp in files:
            subjects.append((dataset, fp.stem, fp))
        logger.info(f"  {dataset}: {len(files)} HDF5 files found")
    return subjects


def slice_subjects(subjects: list, start_idx: int, end_idx, limit) -> list:
    subjects = subjects[start_idx:end_idx]
    if limit:
        subjects = subjects[:limit]
    return subjects


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Precompute OSF's 12-channel, 64Hz resampled signal cache (Stage 2, offline, CPU-only)"
    )
    parser.add_argument("--config",      required=True, help="Path to phase0_osf_lora_config.yaml")
    parser.add_argument("--datasets",    nargs="+",     help="Override datasets list from config")
    parser.add_argument("--limit",       type=int,      help="Process only first N subjects (debug)")
    parser.add_argument("--start-idx",   type=int,      default=0,    help="First subject index (for sharded jobs)")
    parser.add_argument("--end-idx",     type=int,      default=None, help="Last subject index exclusive (for sharded jobs)")
    parser.add_argument("--no-skip",     action="store_true", help="Re-cache even if .npy already exists")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel worker processes (CPU-bound, no GPU used)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    datasets = args.datasets or data_cfg.get("datasets") or cfg["dataset"]["datasets"]
    hdf5_dir = data_cfg["hdf5_dir"]
    cfg_candidates = data_cfg["channel_candidates"]
    cache_dir = data_cfg.get("raw_signal_cache_dir")
    if not cache_dir:
        logger.error("data.raw_signal_cache_dir not set in config — nothing to write to.")
        sys.exit(1)

    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Workers:   {args.num_workers} (CPU-only, no GPU)")

    logger.info(f"Scanning HDF5 files in: {hdf5_dir}")
    all_subjects = find_hdf5_files(hdf5_dir, datasets)
    subjects = slice_subjects(all_subjects, args.start_idx, args.end_idx, args.limit)
    logger.info(
        f"Total available: {len(all_subjects)} | "
        f"This job: [{args.start_idx}:{args.end_idx}] = {len(subjects)} subjects"
    )

    # ── Skip already-cached subjects ────────────────────────────────────────
    todo = []
    n_skip = 0
    for dataset, subject_id, h5_path in subjects:
        out_path = cache_path_for(cache_dir, dataset, subject_id)
        if out_path.exists() and not args.no_skip:
            n_skip += 1
            continue
        todo.append((dataset, subject_id, str(h5_path), cache_dir, cfg_candidates))
    logger.info(f"To process: {len(todo)}  |  Already cached (skipped): {n_skip}")

    # ── Parallel extraction ──────────────────────────────────────────────────
    n_ok = n_err = 0
    t0 = time.time()
    fill_log_handles: dict = {}
    BATCH = max(args.num_workers * 4, args.num_workers)

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for batch_start in range(0, len(todo), BATCH):
            batch = todo[batch_start : batch_start + BATCH]
            futures = [executor.submit(_process_one_subject, item) for item in batch]

            for fut in as_completed(futures):
                dataset, subject_id, status, shape, fill_info, err = fut.result()
                if status == "ok":
                    n_ok += 1
                    if dataset not in fill_log_handles:
                        log_path = Path(cache_dir) / dataset / "_channel_fill_log.jsonl"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        fill_log_handles[dataset] = open(log_path, "a")
                    fill_log_handles[dataset].write(
                        json.dumps({"subject_id": subject_id, **fill_info}) + "\n"
                    )
                    fill_log_handles[dataset].flush()
                else:
                    n_err += 1
                    logger.error(f"  FAILED {dataset}/{subject_id}: {err}")

            done = batch_start + len(batch)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            logger.info(
                f"[{done}/{len(todo)}] ok={n_ok} err={n_err} "
                f"({rate:.2f} subjects/s, {elapsed/60:.1f} min elapsed)"
            )

            if _stop_requested:
                logger.warning(
                    f"[SIGTERM] Stopping after {done}/{len(todo)} processed. "
                    f"Resubmit to continue (already-cached subjects are skipped)."
                )
                break

    for handle in fill_log_handles.values():
        handle.close()

    total = time.time() - t0
    logger.info(
        f"\nDone in {total/60:.1f} min — cached: {n_ok}, skipped: {n_skip}, errors: {n_err}"
    )


if __name__ == "__main__":
    main()
