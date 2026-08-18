#!/usr/bin/env python3
"""
test_physioomni_channel_loader.py — PhysioOmni baseline, Phase 1 Step 1

Smoke-tests src/nsrr_tools/datasets/physioomni_channel_loader.py against
real fast-channel HDF5s. See docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md
§7/§4.5 for the design this verifies.

WHAT THIS CHECKS
────────────────
  - Channel loading + normalization inversion + resampling runs cleanly,
    no NaNs, sane (not e.g. 1e-8 or 1e8) recovered magnitudes.
  - SHHS gets exactly 1 EEG channel (from the generic 'EEG' key), NOT 2
    duplicated ones (plan §4.5's final decision) — the one thing this
    script checks explicitly, not just implicitly.
  - Non-SHHS cohorts get 2 EEG channels (C3, C4) when both are present.
  - Resampled shapes match the full-night resample length exactly
    (round(n_samples_128hz * native_hz / source_hz)) — NOT
    get_epoch_count()*30*native_hz, which truncates to whole epochs and
    undercounts whenever a subject's raw duration isn't an exact 30s-epoch
    multiple (a real gap this script's own first draft had, caught on
    STAGES). Epoch truncation is a later, separate step (extraction time),
    not something load_subject_signals() does itself.

USAGE
─────
  python scripts/test_physioomni_channel_loader.py --datasets apples shhs
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from nsrr_tools.datasets.physioomni_channel_loader import (
    NATIVE_HZ,
    build_channel_candidates,
    get_epoch_count,
    load_subject_signals,
)

FAST_HDF5_ROOT = Path("/scratch/boshra95/psg")


def check_subject(dataset: str, h5_path: Path) -> bool:
    print(f"=== {dataset}: {h5_path.name} ===")
    candidates = build_channel_candidates(dataset)
    signals, fill_info = load_subject_signals(h5_path, candidates)
    epoch_count = get_epoch_count(h5_path)  # informational only, printed below

    # NOTE: load_subject_signals() returns the FULL resampled night, not
    # truncated to whole 30s epochs (that truncation happens later, at
    # extraction time — matching OSF's own load_and_resample_channels()
    # design). So the correct expected length is the full-signal resample
    # length, NOT epoch_count*30*native_hz (which undercounts by the
    # trailing partial-epoch's worth whenever n_samples isn't an exact
    # multiple of source_hz*30 — this was a real test-script bug caught by
    # STAGES's first run, not a channel-loader bug: STAGES's raw sample
    # counts aren't always exact 30s-epoch multiples at 128Hz).
    with h5py.File(h5_path, "r") as hf:
        n_samples_128 = hf[next(iter(hf.keys()))].shape[0]
        source_hz = float(hf.attrs.get("sampling_rate", 128))

    ok = True
    for modality, chans in signals.items():
        if not chans:
            print(f"  {modality}: ABSENT")
            continue
        for label, arr in chans:
            has_nan = bool(np.isnan(arr).any())
            expected_len = round(n_samples_128 * NATIVE_HZ[modality] / source_hz)
            len_ok = abs(len(arr) - expected_len) < 2
            sane_scale = 1e-6 < arr.std() < 1e6
            status = "OK" if (not has_nan and len_ok and sane_scale) else "FAIL"
            if status == "FAIL":
                ok = False
            print(f"  {modality}/{label}: len={len(arr)} (expected={expected_len}) "
                  f"std={arr.std():.4g} NaN={has_nan} [{status}]")
    print(f"  epoch_count (whole 30s epochs, for reference): {epoch_count}")

    eeg_count = fill_info["eeg_channel_count"]
    if dataset == "shhs":
        shhs_ok = eeg_count == 1
        print(f"  SHHS EEG-channel-count check: {eeg_count} (expected 1) "
              f"[{'OK' if shhs_ok else 'FAIL — duplication or zero-fill bug!'}]")
        ok = ok and shhs_ok
    else:
        print(f"  EEG channel count: {eeg_count}")

    print(f"  fill_info: {fill_info}")
    print()
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["apples", "shhs", "mros", "stages"])
    parser.add_argument("--limit", type=int, default=2, help="Subjects per dataset")
    args = parser.parse_args()

    all_ok = True
    for dataset in args.datasets:
        h5_dir = FAST_HDF5_ROOT / dataset / "derived" / "hdf5_signals"
        files = sorted(h5_dir.glob("*.h5"))[: args.limit]
        for f in files:
            all_ok &= check_subject(dataset, f)

    print("PASSED" if all_ok else "FAILED", "— see [OK]/[FAIL] tags above")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
