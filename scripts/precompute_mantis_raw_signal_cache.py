#!/usr/bin/env python3
"""
precompute_mantis_raw_signal_cache.py — Mantis baseline, Stage 2 (LoRA)
offline raw-signal cache precompute (checklist 2.2)

WHY THIS EXISTS
────────────────
`train_mantis_lora.py`'s future raw-epoch dataset needs raw signal on every
training step (the backbone itself is being fine-tuned, so pre-extracted
Stage 1 embeddings can't be reused). Reading it live from the fast-channel
HDF5 tree on every `__getitem__` would put HDF5-chunk decompression on the
critical path of every training step, for every (task, head, context)
combination — the exact mistake OSF's Stage 2 first hit live on a real GPU
job (2+ hours with the GPU essentially idle, diagnosed in
`docs/TSFM_OSF_IMPLEMENTATION_PLAN.md`'s checklist 2.5b). This script
precomputes the per-subject signal ONCE, CPU-only (no GPU, no model — just
I/O + reshape), so every subsequent Stage 2 job reads a cheap local file
instead of touching the raw HDF5 at all.

WHY THIS LOOKS DIFFERENT FROM precompute_osf_raw_signal_cache.py /
precompute_physioomni_raw_signal_cache.py
─────────────────────────────────────────────────────────────────
Mantis needs **no resampling at all** — every fast-channel HDF5 is already
128 Hz (plan §3.2/§7), so this script is pure read + reshape, no `scipy`
FFT/decimation anywhere. It should be substantially faster than either
sibling's precompute; **time the first real shard before budgeting the
rest** (plan §14.3) — do not assume OSF's/PhysioOmni's per-subject numbers
transfer.

The cached array's LAYOUT also differs deliberately: **epoch-major
`[T_epochs, 6, 3840]`**, not channel-major `[6, n_samples]` like OSF's own
cache. This is the whole point of the design (plan §14.3/§4.9): a training
window of N consecutive epochs across all 6 channels is then ONE
contiguous byte range (`load_signal_cache_window`'s single seek+read),
where OSF's channel-major layout needs 12 strided reads for the same
window. `load_subject_channels` returns channel-major `[6, n_samples]`
(matching Stage 1's own loader, so there is exactly one channel-mapping
implementation in the repo); this script truncates to whole epochs and
transposes to epoch-major before caching — the identical reshape
`epochs_to_model_input()` already performs and
`test_mantis_channel_loader.py` already verifies against real data, reused
here rather than re-derived.

OUTPUT FORMAT
─────────────
  {cache_dir}/{dataset}/{subject_id}.npy         [T_epochs, 6, 3840] float16
  {cache_dir}/{dataset}/{subject_id}.meta.json   {"t_epochs", "slots_found",
                                                   "slots_missing", "present",
                                                   "resp_source"}

Same channel mapping as `extract_mantis_embeddings.py` (Stage 1) — both
call `load_subject_channels()` from `nsrr_tools.datasets.mantis_channel_loader`,
so there is exactly one implementation of channel-mapping logic in the repo.

USAGE
─────
  # Small test
  python scripts/precompute_mantis_raw_signal_cache.py \\
      --config configs/phase0_mantis_lora_config.yaml --datasets apples --limit 5

  # Sharded full run (CPU only — see jobs/precompute_mantis_raw_signal_cache.sh)
  python scripts/precompute_mantis_raw_signal_cache.py \\
      --config configs/phase0_mantis_lora_config.yaml \\
      --start-idx 0 --end-idx 5000 --num-workers 8

  # Re-run to fill in gaps (already-cached subjects are skipped automatically)
  python scripts/precompute_mantis_raw_signal_cache.py --config configs/phase0_mantis_lora_config.yaml
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
from nsrr_tools.datasets.mantis_channel_loader import (  # noqa: E402
    DEFAULT_CHANNEL_CANDIDATES,
    EPOCH_SAMPLES,
    N_SLOTS,
    SLOT_ORDER,
    cache_exists,
    load_subject_channels,
    save_signal_cache,
)

# ── Graceful-stop flag ─────────────────────────────────────────────────────
# Same philosophy as extract_mantis_embeddings.py: finish in-flight work,
# then stop, rather than an abrupt exit. With a process pool, "in-flight"
# means the currently-submitted batch, not a single subject — see main().
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
    dataset, subject_id, h5_path_str, cache_dir_str, candidates = args_tuple
    h5_path = Path(h5_path_str)
    try:
        x, fill_info = load_subject_channels(h5_path, candidates)  # [6, n_samples] float32

        t_epochs = x.shape[1] // EPOCH_SAMPLES
        if t_epochs == 0:
            return (dataset, subject_id, "error", None, None,
                    f"recording too short: {x.shape[1]} samples < 1 epoch ({EPOCH_SAMPLES})")

        # Truncate to whole epochs, then channel-major [6, T*3840] ->
        # epoch-major [T, 6, 3840] — identical transpose to
        # epochs_to_model_input()'s own reshape (mantis_channel_loader.py),
        # reused here rather than re-derived.
        x_trunc = x[:, : t_epochs * EPOCH_SAMPLES]
        x_epoch_major = x_trunc.reshape(N_SLOTS, t_epochs, EPOCH_SAMPLES).transpose(1, 0, 2)

        present = [0 if slot in fill_info["slots_missing"] else 1 for slot in SLOT_ORDER]
        meta = {
            "t_epochs": t_epochs,
            "slots_found": fill_info["slots_found"],
            "slots_missing": fill_info["slots_missing"],
            "fallback_used": fill_info["fallback_used"],
            "present": present,
            "resp_source": fill_info["resp_source"],
        }
        save_signal_cache(cache_dir_str, dataset, subject_id, x_epoch_major, meta)
        return (dataset, subject_id, "ok", x_epoch_major.shape, fill_info, None)
    except Exception as exc:
        return (dataset, subject_id, "error", None, None, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Subject discovery — identical convention to extract_mantis_embeddings.py
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
        description="Precompute Mantis's epoch-major raw-signal cache (Stage 2, offline, CPU-only)"
    )
    parser.add_argument("--config",      required=True, help="Path to phase0_mantis_lora_config.yaml")
    parser.add_argument("--datasets",    nargs="+",     help="Override datasets list from config")
    parser.add_argument("--limit",       type=int,      help="Process only first N subjects (debug)")
    parser.add_argument("--start-idx",   type=int,      default=0,    help="First subject index (for sharded jobs)")
    parser.add_argument("--end-idx",     type=int,      default=None, help="Last subject index exclusive (for sharded jobs)")
    parser.add_argument("--no-skip",     action="store_true", help="Re-cache even if already cached")
    parser.add_argument("--num-workers", type=int, default=8, help="Parallel worker processes (CPU-bound, no GPU used)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    datasets = args.datasets or cfg["embedding"].get("datasets") or cfg["dataset"]["datasets"]
    hdf5_dir = data_cfg["hdf5_dir"]
    candidates = data_cfg.get("channel_candidates", DEFAULT_CHANNEL_CANDIDATES)
    cache_dir = data_cfg.get("raw_signal_cache_dir")
    if not cache_dir:
        logger.error("data.raw_signal_cache_dir not set in config — nothing to write to.")
        sys.exit(1)

    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Workers:   {args.num_workers} (CPU-only, no GPU, no resampling)")

    logger.info(f"Scanning HDF5 files in: {hdf5_dir}")
    all_subjects = find_hdf5_files(hdf5_dir, datasets)
    subjects = slice_subjects(all_subjects, args.start_idx, args.end_idx, args.limit)
    logger.info(
        f"Total available: {len(all_subjects)} | "
        f"This job: [{args.start_idx}:{args.end_idx}] = {len(subjects)} subjects"
    )

    # ── Skip already-cached subjects ────────────────────────────────────────
    # cache_exists() parses meta.json rather than checking file existence —
    # a subject killed mid-write reads as NOT cached and gets retried, not
    # silently treated as done (plan §14.3/§4.10).
    todo = []
    n_skip = 0
    for dataset, subject_id, h5_path in subjects:
        if cache_exists(cache_dir, dataset, subject_id) and not args.no_skip:
            n_skip += 1
            continue
        todo.append((dataset, subject_id, str(h5_path), cache_dir, candidates))
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
                        json.dumps({"subject_id": subject_id, "shape": list(shape), **fill_info}) + "\n"
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
