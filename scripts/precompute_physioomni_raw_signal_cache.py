#!/usr/bin/env python3
"""
precompute_physioomni_raw_signal_cache.py — PhysioOmni baseline, Stage 2
(LoRA) offline raw-signal cache precompute (plan §15.3, checklist 2.2)

WHY THIS EXISTS
────────────────
Forked from precompute_osf_raw_signal_cache.py's own reasoning: OSF's real
Stage 2 build lost 2+ hours to a GPU job stalling on repeated live
HDF5-read + resample work inside the training dataset's __getitem__, redone
from scratch for every task/head/context combination even though the
resampled signal doesn't depend on any of those. This script precomputes
PhysioOmni's per-modality resampled/denormalized signal ONCE per subject,
CPU-only (no GPU, no model — just I/O + FFT resample), so every subsequent
Stage 2 job reads a cheap cached array instead of touching the raw HDF5.
Does NOT touch subject/split selection — task_subject_dir/split_seed are
untouched, so Stage 1 and Stage 2 use identical subjects/splits.

OUTPUT FORMAT — see nsrr_tools.datasets.physioomni_channel_loader's
"Raw signal cache" section (plan §15.3) for the full design rationale
(why this is per-subject-per-slot files, not a single fixed-shape matrix
like OSF's — PhysioOmni's channels are genuinely present-or-absent per
subject, at 2 different native rates, not a uniform 12-channel case):
  {cache_dir}/{dataset}/{subject_id}/
      EEG_C3.npy, EEG_C4.npy, EOG_HEO.npy, ECG.npy, EMG.npy   (only present ones)
      meta.json   {"t_epochs": ..., "slots_found": {...}, "slots_missing": [...]}

Same channel mapping/resampling as extract_physioomni_embeddings.py
(Stage 1) — both call load_subject_signals()/build_channel_candidates()
from nsrr_tools.datasets.physioomni_channel_loader, so there is exactly
one implementation of channel mapping/resampling logic anywhere in the repo.

SCOPE (plan §15.3): only datasets actually used by PhysioOmni's Tier-1
tasks need caching — apples/shhs/mros. STAGES is never used by any of the
4 tasks (apnea, the only task that would need it, is excluded entirely —
no respiratory pathway), so it defaults OUT of this script's --datasets
unless explicitly requested.

USAGE
─────
  # Small test
  python scripts/precompute_physioomni_raw_signal_cache.py \\
      --config configs/phase0_physioomni_lora_config.yaml --datasets apples --limit 5

  # Sharded full run (CPU only — see jobs/precompute_physioomni_raw_signal_cache.sh)
  python scripts/precompute_physioomni_raw_signal_cache.py \\
      --config configs/phase0_physioomni_lora_config.yaml \\
      --start-idx 0 --end-idx 5000 --num-workers 8

  # Re-run to fill in gaps (already-cached subjects are skipped automatically)
  python scripts/precompute_physioomni_raw_signal_cache.py --config configs/phase0_physioomni_lora_config.yaml
"""

import argparse
import json
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from nsrr_tools.datasets.physioomni_channel_loader import (  # noqa: E402
    build_channel_candidates,
    load_subject_signals,
    save_signal_cache,
    cache_exists,
)

# Datasets PhysioOmni's Tier-1 registry actually uses (plan §15.3) — stages
# is deliberately excluded by default (no task needs it; apnea is excluded
# entirely for PhysioOmni). Pass --datasets explicitly to override.
_DEFAULT_DATASETS = ["apples", "shhs", "mros"]

# ── Graceful-stop flag ─────────────────────────────────────────────────────
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
    try:
        channel_candidates = build_channel_candidates(dataset, cfg_candidates)
        signals, fill_info = load_subject_signals(h5_path, channel_candidates)
        t_epochs = save_signal_cache(cache_dir_str, dataset, subject_id, signals, fill_info)
        if t_epochs == 0:
            raise ValueError("No PhysioOmni-relevant channels found at all (t_epochs=0)")
        return (dataset, subject_id, "ok", t_epochs, fill_info, None)
    except Exception as exc:
        return (dataset, subject_id, "error", None, None, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Subject discovery — identical convention to extract_physioomni_embeddings.py
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
        description="Precompute PhysioOmni's per-modality resampled signal cache (Stage 2, offline, CPU-only)"
    )
    parser.add_argument("--config",      required=True, help="Path to phase0_physioomni_lora_config.yaml")
    parser.add_argument("--datasets",    nargs="+",     help=f"Override datasets list (default: {_DEFAULT_DATASETS})")
    parser.add_argument("--limit",       type=int,      help="Process only first N subjects (debug)")
    parser.add_argument("--start-idx",   type=int,      default=0,    help="First subject index (for sharded jobs)")
    parser.add_argument("--end-idx",     type=int,      default=None, help="Last subject index exclusive (for sharded jobs)")
    parser.add_argument("--no-skip",     action="store_true", help="Re-cache even if already cached")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel worker processes (CPU-bound, no GPU used)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    datasets = args.datasets or data_cfg.get("datasets") or _DEFAULT_DATASETS
    hdf5_dir = data_cfg["hdf5_dir"]
    cfg_candidates = data_cfg["channel_candidates"]
    cache_dir = data_cfg.get("raw_signal_cache_dir")
    if not cache_dir:
        logger.error("data.raw_signal_cache_dir not set in config — nothing to write to.")
        sys.exit(1)

    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Datasets:  {datasets}")
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
        if cache_exists(cache_dir, dataset, subject_id) and not args.no_skip:
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
                dataset, subject_id, status, t_epochs, fill_info, err = fut.result()
                if status == "ok":
                    n_ok += 1
                    if dataset not in fill_log_handles:
                        log_path = Path(cache_dir) / dataset / "_channel_fill_log.jsonl"
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        fill_log_handles[dataset] = open(log_path, "a")
                    fill_log_handles[dataset].write(
                        json.dumps({"subject_id": subject_id, "t_epochs": t_epochs, **fill_info}) + "\n"
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
