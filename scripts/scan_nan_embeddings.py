#!/usr/bin/env python3
"""
Scan all embedding files for NaN values.

Outputs a list of subjects with NaN embeddings, then writes a blocklist
to <embedding_dir>/nan_blocklist.txt that ContextWindowDataset can use.

Usage:
    python scripts/scan_nan_embeddings.py
    python scripts/scan_nan_embeddings.py --workers 8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

cfg_path = _ROOT / "configs" / "phase0_config.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

EMB_DIR = Path(cfg["dataset"]["embedding_dir"])


def check_file(key: str) -> tuple[str, bool, int]:
    """Return (key, has_nan, n_nan_patches). Uses mmap — only pages in data as needed."""
    p = EMB_DIR / f"{key}.npy"
    try:
        x = np.load(p, mmap_mode="r").astype(np.float32)
        nan_mask = np.isnan(x)
        has_nan = bool(nan_mask.any())
        n_nan = int(nan_mask.reshape(x.shape[0], -1).any(axis=1).sum()) if has_nan else 0
        return key, has_nan, n_nan
    except Exception as e:
        print(f"  ERROR reading {key}: {e}", flush=True)
        return key, True, -1   # treat unreadable as bad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker processes (default: 4)")
    args = parser.parse_args()

    with open(EMB_DIR / "shape_cache.json") as f:
        cache = json.load(f)
    all_keys = list(cache.keys())
    total = len(all_keys)
    print(f"Scanning {total} embedding files for NaN  (workers={args.workers})", flush=True)
    print(f"Embedding dir: {EMB_DIR}", flush=True)

    bad = []
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(check_file, k): k for k in all_keys}
        for fut in as_completed(futs):
            key, has_nan, n_nan = fut.result()
            done += 1
            if done % 500 == 0 or done == total:
                print(f"  {done}/{total} checked  bad_so_far={len(bad)}", flush=True)
            if has_nan:
                bad.append((key, n_nan))
                print(f"  NaN found: {key}  nan_patches={n_nan}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Scan complete.  Files with NaN: {len(bad)} / {total}", flush=True)

    if bad:
        blocklist_path = EMB_DIR / "nan_blocklist.txt"
        with open(blocklist_path, "w") as f:
            for key, n in sorted(bad):
                f.write(f"{key}\t{n}\n")
        print(f"Blocklist written to: {blocklist_path}", flush=True)
        print("\nBad files:", flush=True)
        for key, n in sorted(bad, key=lambda t: -t[1]):
            print(f"  {key}  nan_patches={n}", flush=True)
    else:
        print("No NaN values found in any embedding file.", flush=True)


if __name__ == "__main__":
    main()
