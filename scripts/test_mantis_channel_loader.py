#!/usr/bin/env python3
"""
test_mantis_channel_loader.py — Mantis baseline, Phase 1 Step 1

Smoke-tests src/nsrr_tools/datasets/mantis_channel_loader.py against real
fast-channel HDF5s across all four cohorts. See
docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md §2/§7 for the design this verifies.

WHAT THIS CHECKS (checklist 1.1)
─────────────────────────────────
  - The 6-slot canonical map resolves for real subjects in all 4 cohorts.
  - SHHS specifically uses its two documented fallbacks: the generic `EEG`
    channel (no C3-M2/C4-M1 in SHHS) and, for the RESP slot, EITHER
    `Airflow` OR `Thor` depending on the subject (plan §2.1's measured
    ~75%/~25% split) — this test reports the actual key used, not just that
    one resolved.
  - Zero NaN in every loaded slot.
  - Shape is exactly `[6, n]` with `n` equal to the raw HDF5 sample count
    (NO resampling — every fast-channel file is already 128 Hz, unlike
    OSF's 128->64Hz or PhysioOmni's 128->200/500Hz).
  - `epochs_to_model_input` produces the expected tensor shapes for both
    windowing modes, on real (not synthetic) signal.

  --test-cache (checklist 2.1) additionally round-trip-tests the Stage 2
  raw-signal cache functions (save/load/exists/get_cached_t_epochs) against
  a synthetic array — cheap, deterministic, and independent of real HDF5
  data, so it runs in under a second and needs no cluster data access.

USAGE
─────
  python scripts/test_mantis_channel_loader.py --datasets apples shhs mros stages
  python scripts/test_mantis_channel_loader.py --test-cache
"""

import argparse
import glob
import shutil
from pathlib import Path

import h5py
import numpy as np

from nsrr_tools.datasets.mantis_channel_loader import (
    EPOCH_SAMPLES,
    N_SLOTS,
    SLOT_ORDER,
    cache_exists,
    cache_path_for,
    epochs_to_model_input,
    get_cached_t_epochs,
    get_epoch_count,
    load_signal_cache_window,
    load_subject_channels,
    save_signal_cache,
)

FAST_HDF5_ROOT = Path("/scratch/boshra95/psg")


def check_subject(dataset: str, h5_path: Path) -> bool:
    print(f"=== {dataset}: {h5_path.name} ===")
    x, fill_info = load_subject_channels(h5_path)
    epoch_count = get_epoch_count(h5_path)

    with h5py.File(h5_path, "r") as hf:
        n_samples_real = hf[next(iter(hf.keys()))].shape[0]

    ok = True

    shape_ok = x.shape == (N_SLOTS, n_samples_real)
    print(f"  shape: {x.shape} (expected ({N_SLOTS}, {n_samples_real})) "
          f"[{'OK' if shape_ok else 'FAIL'}]")
    ok = ok and shape_ok

    for i, slot in enumerate(SLOT_ORDER):
        if slot in fill_info["slots_missing"]:
            print(f"  {slot}: ABSENT (zero-filled)")
            if not np.all(x[i] == 0.0):
                print(f"    FAIL: absent slot is not exactly zero")
                ok = False
            continue
        key = fill_info["slots_found"][slot]
        arr = x[i]
        has_nan = bool(np.isnan(arr).any())
        # Fast-channel HDF5s are already z-scored float16 -> real signal std
        # should be an O(1) number, not degenerate (near-zero) or absurd.
        sane_scale = 1e-4 < arr.std() < 1e4
        status = "OK" if (not has_nan and sane_scale) else "FAIL"
        if status == "FAIL":
            ok = False
        fb = " (FALLBACK)" if slot in fill_info["fallback_used"] else ""
        print(f"  {slot} <- '{key}'{fb}: std={arr.std():.4g} NaN={has_nan} [{status}]")

    print(f"  epoch_count (whole 30s epochs): {epoch_count}")
    print(f"  resp_source: {fill_info['resp_source']}")

    if dataset == "shhs":
        eeg_key = fill_info["slots_found"].get("EEG")
        shhs_eeg_ok = eeg_key == "EEG"
        print(f"  SHHS EEG-fallback check: resolved via '{eeg_key}' "
              f"[{'OK' if shhs_eeg_ok else 'FAIL — expected the generic EEG key'}]")
        ok = ok and shhs_eeg_ok

    # epochs_to_model_input sanity, on real signal, both windowing modes.
    if epoch_count >= 1:
        n_ep = min(2, epoch_count)
        full = epochs_to_model_input(x, "full_epoch", 0, n_ep)
        sub = epochs_to_model_input(x, "subwindow", 0, n_ep)
        full_shape_ok = tuple(full.shape) == (n_ep * N_SLOTS, 1, EPOCH_SAMPLES)
        sub_shape_ok = tuple(sub.shape) == (n_ep * N_SLOTS * 8, 1, 512)
        full_nan = bool(np.isnan(full.numpy()).any())
        sub_nan = bool(np.isnan(sub.numpy()).any())
        print(f"  epochs_to_model_input('full_epoch'): shape={tuple(full.shape)} "
              f"(expected {(n_ep * N_SLOTS, 1, EPOCH_SAMPLES)}) NaN={full_nan} "
              f"[{'OK' if full_shape_ok and not full_nan else 'FAIL'}]")
        print(f"  epochs_to_model_input('subwindow'):   shape={tuple(sub.shape)} "
              f"(expected {(n_ep * N_SLOTS * 8, 1, 512)}) NaN={sub_nan} "
              f"[{'OK' if sub_shape_ok and not sub_nan else 'FAIL'}]")
        ok = ok and full_shape_ok and sub_shape_ok and not full_nan and not sub_nan

    print()
    return ok


def test_signal_cache_roundtrip() -> bool:
    """Round-trip test for the Stage 2 raw-signal cache functions
    (checklist 2.1), against a synthetic array — no real HDF5 needed.
    Uses an intentionally non-round epoch count (137) to catch off-by-ones,
    and checks the failure paths (out-of-range window, incomplete cache)
    as carefully as the success path.
    """
    print("=== Cache round-trip test (synthetic data) ===")
    cache_dir = "/tmp/mantis_cache_roundtrip_test"
    shutil.rmtree(cache_dir, ignore_errors=True)
    ok = True

    dataset, subject_id = "apples", "TEST0001"
    T = 137
    rng = np.random.default_rng(42)
    x = rng.standard_normal((T, N_SLOTS, EPOCH_SAMPLES)).astype(np.float32)

    not_yet = cache_exists(cache_dir, dataset, subject_id) is False
    print(f"  cache_exists before write: False [{'OK' if not_yet else 'FAIL'}]")
    ok = ok and not_yet

    meta = {"t_epochs": T, "slots_found": {"EEG": "C3-M2"}, "slots_missing": ["ECG"],
            "fallback_used": {}, "present": [1, 1, 1, 0, 1, 1], "resp_source": "Airflow"}
    save_signal_cache(cache_dir, dataset, subject_id, x, meta)

    exists_after = cache_exists(cache_dir, dataset, subject_id) is True
    print(f"  cache_exists after write: True [{'OK' if exists_after else 'FAIL'}]")
    ok = ok and exists_after

    t = get_cached_t_epochs(cache_dir, dataset, subject_id)
    t_ok = t == T
    print(f"  get_cached_t_epochs: {t} (expected {T}) [{'OK' if t_ok else 'FAIL'}]")
    ok = ok and t_ok

    x16 = x.astype(np.float16)
    for e0, n in [(0, 1), (0, T), (T - 1, 1), (50, 30), (0, 5)]:
        w = load_signal_cache_window(cache_dir, dataset, subject_id, e0, n)
        expected = x16[e0:e0 + n]
        match = w.shape == expected.shape and w.dtype == np.float16 and np.array_equal(w, expected)
        print(f"  window e0={e0} n={n}: shape={w.shape} [{'OK' if match else 'FAIL'}]")
        ok = ok and match

    try:
        load_signal_cache_window(cache_dir, dataset, subject_id, T - 1, 5)
        print("  out-of-range window: did NOT raise [FAIL]")
        ok = False
    except ValueError:
        print("  out-of-range window: correctly raised ValueError [OK]")

    cache_path_for(cache_dir, dataset, "TEST0002").parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path_for(cache_dir, dataset, "TEST0002"), x16)
    incomplete_ok = cache_exists(cache_dir, dataset, "TEST0002") is False
    print(f"  incomplete cache (.npy only, no meta.json) reads as NOT cached "
          f"[{'OK' if incomplete_ok else 'FAIL'}]")
    ok = ok and incomplete_ok

    leftovers = glob.glob(f"{cache_dir}/**/*.tmp*", recursive=True)
    no_leftovers = leftovers == []
    print(f"  no leftover temp files [{'OK' if no_leftovers else 'FAIL: ' + str(leftovers)}]")
    ok = ok and no_leftovers

    shutil.rmtree(cache_dir, ignore_errors=True)
    print()
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["apples", "shhs", "mros", "stages"])
    parser.add_argument("--limit", type=int, default=2, help="Subjects per dataset")
    parser.add_argument("--test-cache", action="store_true", dest="test_cache",
                        help="Run only the Stage 2 raw-signal cache round-trip test "
                             "(synthetic data, no cluster access needed)")
    args = parser.parse_args()

    if args.test_cache:
        ok = test_signal_cache_roundtrip()
        print("PASSED" if ok else "FAILED")
        if not ok:
            raise SystemExit(1)
        return

    all_ok = True
    resp_sources = {}
    for dataset in args.datasets:
        h5_dir = FAST_HDF5_ROOT / dataset / "derived" / "hdf5_signals"
        files = sorted(h5_dir.glob("*.h5"))[: args.limit]
        if not files:
            print(f"WARNING: no HDF5 files found for {dataset} at {h5_dir}")
            all_ok = False
            continue
        for f in files:
            ok = check_subject(dataset, f)
            all_ok &= ok

    print("PASSED" if all_ok else "FAILED", "— see [OK]/[FAIL] tags above")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
