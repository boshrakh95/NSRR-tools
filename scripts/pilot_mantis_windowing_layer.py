#!/usr/bin/env python3
"""
pilot_mantis_windowing_layer.py — Mantis Pilots 1+2, checklist 1.6

ONE-OFF PILOT SCRIPT (plan §13.1/§13.2/§13.4), not part of the permanent
Stage 1 pipeline (that is `extract_mantis_embeddings.py`). Runs 3
windowing variants (D, D-interp, B) and, in ONE forward pass per variant,
captures BOTH layer-2 (index 2, 0-based — 3rd transformer block) and
layer-6/last (full depth) hidden states, in BOTH cls-token and combined
(cls+mean) form → 12 embedding variants total, scored by
`probe_mantis_staging.run_probe()` (checklist 1.5).

SLEEP STAGING IS NOT A PAPER TASK — see probe_mantis_staging.py's own
docstring. This script's output feeds ONLY the windowing/layer decision
below, never a results table.

WHY MANUAL LAYER CAPTURE, NOT 4 SEPARATE MODEL FORWARD CALLS
──────────────────────────────────────────────────────────────
`MantisV1.forward()` only returns ONE (return_transf_layer, output_token)
combination per call. Getting all 4 from one pass means replicating
`TransformerUnit.forward()`'s own cls-prepend / positional-encoding /
layer-loop logic manually, capturing intermediate states as we go.
**Verified bit-identical (max abs diff 0.0) against calling the model's
own forward() for each of the 4 combinations individually, on the real
Mantis-8M checkpoint, 2026-09-07** — see `dual_layer_forward`'s docstring.

DECISION RULES (plan §13.1/§13.2 escape hatches — already decided, this
pilot either confirms them or triggers the stated exception)
──────────────────────────────────────────────────────────────
- Windowing: default is D (`full_epoch`+`extrapolate`). Switch to B
  (`subwindow`) ONLY if D's `combined@last` scores far below B's — roughly
  weighted F1 below ~0.60 where B reaches ~0.75+, a gap far larger than
  ordinary run-to-run noise.
- Output layer: `combined@last` is already decided on cross-model fairness
  grounds (§3.3) regardless of layer-2's score here — the layer-2 numbers
  are the paper's SUPPLEMENTARY "what we gave up" statistic, not something
  that overrides the fairness argument. The one thing that WOULD override
  it: `combined@last` scoring more than ~0.08 F1 worse than `combined@2` —
  that would mean the last layer is genuinely broken at 241 tokens, not
  merely suboptimal.

USAGE
─────
  python scripts/pilot_mantis_windowing_layer.py \\
      --datasets apples shhs --limit-per-dataset 50
  python scripts/pilot_mantis_windowing_layer.py --score-only   # re-score without re-extracting
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR.parent / "src"))
sys.path.insert(0, str(_SCRIPTS_DIR))

from nsrr_tools.datasets.mantis_channel_loader import (  # noqa: E402
    EPOCH_SAMPLES,
    N_SLOTS,
    SLOT_ORDER,
    epochs_to_model_input,
    load_mantis_backbone,
    load_subject_channels,
)
from probe_mantis_staging import run_probe  # noqa: E402

# ── TF32 (plan §4.2) ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

CHANNEL_CANDIDATES = {
    "EEG":   ["C3-M2", "EEG", "C4-M1", "O1-M2"],
    "EOG_L": ["LOC"],
    "EOG_R": ["ROC"],
    "ECG":   ["EKG", "ECG-L"],
    "EMG":   ["CHIN", "EMG", "LLEG", "RLEG"],
    "RESP":  ["Airflow", "Thor", "ABD"],
}
CHECKPOINT_LOCAL_DIR = "/home/boshra95/mantis_checkpoints/Mantis-8M"

# name -> (windowing, seq_len, num_patches, pe_mode) — plan §3.1's two-key split.
# D and D-interp share windowing/seq_len/num_patches, differing ONLY in
# pe_mode (the backbone's positional buffer). B uses the pretrained-native
# 512/32 shape, where pe_mode is moot (extrapolate with stride=1.0 reproduces
# the checkpoint's own buffer exactly, since num_patches==32 already).
VARIANTS = {
    "D":       ("full_epoch", 3840, 240, "extrapolate"),
    "Dinterp": ("full_epoch", 3840, 240, "interpolate"),
    "B":       ("subwindow",   512,  32, "extrapolate"),
}
LAYER_TOKEN_KEYS = [(2, "cls"), (2, "combined"), (-1, "cls"), (-1, "combined")]


def dual_layer_forward(backbone, x: torch.Tensor) -> dict:
    """One forward pass through Mantis's tokenizer + full transformer,
    capturing BOTH layer-2 and layer-last hidden states in BOTH cls and
    combined form — 4 embeddings from 1 pass. Manually replicates
    `TransformerUnit.forward()`'s cls-prepend/positional-encoding/unpack
    logic exactly. **Verified bit-identical (max abs diff 0.0) against
    calling `MantisV1`'s own forward() for each of the 4 combinations
    individually, on the real Mantis-8M checkpoint (2026-09-07).**

    x: [n, 1, seq_len]
    Returns: {(2,'cls'): [n,256], (2,'combined'): [n,512],
              (-1,'cls'): [n,256], (-1,'combined'): [n,512]}
    """
    tu = backbone.transf_unit
    x_embeddings = backbone.tokgen_unit(x)
    b = x_embeddings.shape[0]
    cls_tokens = tu.cls_token.unsqueeze(0).expand(b, -1)
    seq = torch.cat([cls_tokens.unsqueeze(1), x_embeddings], dim=1)
    seq = tu.pos_encoder(seq.transpose(0, 1)).transpose(0, 1)

    out = {}
    n_layers = len(tu.transformer.layers)
    for i, (attn, ff) in enumerate(tu.transformer.layers):
        seq = attn(seq) + seq
        seq = ff(seq) + seq
        if i == 2 or i == n_layers - 1:
            cls_tok = seq[:, 0, :]
            mean_tok = seq[:, 1:, :].mean(dim=1)
            key = 2 if i == 2 else -1
            out[(key, "cls")] = cls_tok
            out[(key, "combined")] = torch.cat([cls_tok, mean_tok], dim=1)
    return out


def extract_subject_all_variants(h5_path, backbone, device, chunk_batch_size, windowing):
    """Extract all 4 (layer,token) combinations for one subject, one
    windowing setting, in ONE pass over the subject's epochs. Mirrors
    extract_mantis_embeddings.py's present-slot-only batching (plan §2.2's
    Stage 1 absent-slot contract — absent slots are never forwarded, left
    exact zero) and chunking design exactly.

    Returns: ({(layer,token): np.ndarray[T, 6, D]}, fill_info)
    """
    x, fill_info = load_subject_channels(h5_path, CHANNEL_CANDIDATES)
    t_epochs = x.shape[1] // EPOCH_SAMPLES
    if t_epochs == 0:
        raise ValueError(f"Recording too short (< 1 epoch) in {h5_path}")

    present_idxs = [i for i, slot in enumerate(SLOT_ORDER) if slot not in fill_info["slots_missing"]]
    if not present_idxs:
        raise ValueError(f"No Mantis-relevant channels found at all in {h5_path}")
    n_present = len(present_idxs)

    items_per_epoch = 8 if windowing == "subwindow" else 1
    epochs_per_chunk = max(1, chunk_batch_size // (n_present * items_per_epoch))

    out = {
        key: np.zeros((t_epochs, N_SLOTS, 512 if key[1] == "combined" else 256), dtype=np.float32)
        for key in LAYER_TOKEN_KEYS
    }

    for start in range(0, t_epochs, epochs_per_chunk):
        n = min(epochs_per_chunk, t_epochs - start)
        model_in = epochs_to_model_input(x, windowing, start, n)

        if windowing == "full_epoch":
            model_in = model_in.reshape(n, N_SLOTS, 1, EPOCH_SAMPLES)
            sel = model_in[:, present_idxs].reshape(n * n_present, 1, EPOCH_SAMPLES)
        else:  # subwindow
            model_in = model_in.reshape(n, N_SLOTS, 8, 1, 512)
            sel = model_in[:, present_idxs].reshape(n * n_present * 8, 1, 512)

        sel = sel.to(device)
        with torch.no_grad():
            variants = dual_layer_forward(backbone, sel)

        for key, emb_out in variants.items():
            d = emb_out.shape[-1]
            if windowing == "subwindow":
                emb_out = emb_out.reshape(n, n_present, 8, d).mean(dim=2)
            else:
                emb_out = emb_out.reshape(n, n_present, d)
            out[key][start:start + n][:, present_idxs, :] = emb_out.cpu().float().numpy()

    return {key: arr.astype(np.float16) for key, arr in out.items()}, fill_info


def find_hdf5_files(hdf5_dir, datasets, limit_per_dataset=None):
    root = Path(hdf5_dir)
    subjects = []
    for dataset in datasets:
        h5_dir = root / dataset / "derived" / "hdf5_signals"
        files = sorted(h5_dir.glob("*.h5"))
        if limit_per_dataset:
            files = files[:limit_per_dataset]
        for fp in files:
            subjects.append((dataset, fp.stem, fp))
    return subjects


def variant_dir_key(variant_name: str, layer: int, token: str) -> str:
    layer_label = "L2" if layer == 2 else "Llast"
    return f"{variant_name}_{layer_label}_{token}"


def main():
    parser = argparse.ArgumentParser(description="Mantis Pilots 1+2 (checklist 1.6)")
    parser.add_argument("--output-root", default="/scratch/boshra95/psg/unified/embeddings/mantis_pilot12")
    parser.add_argument("--hdf5-dir", default="/scratch/boshra95/psg")
    parser.add_argument("--datasets", nargs="+", default=["apples", "shhs"])
    parser.add_argument("--limit-per-dataset", type=int, default=50)
    parser.add_argument("--chunk-batch-size", type=int, default=192)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-skip", action="store_true")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip extraction, just score existing --output-root dirs")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    output_root = Path(args.output_root)
    subjects = find_hdf5_files(args.hdf5_dir, args.datasets, args.limit_per_dataset)
    logger.info(f"{len(subjects)} subjects across {args.datasets}")

    if not args.score_only:
        for variant_name, (windowing, seq_len, num_patches, pe_mode) in VARIANTS.items():
            logger.info(
                f"=== Variant {variant_name}: windowing={windowing}, seq_len={seq_len}, "
                f"num_patches={num_patches}, pe_mode={pe_mode} ==="
            )
            backbone = load_mantis_backbone(
                CHECKPOINT_LOCAL_DIR, seq_len=seq_len, num_patches=num_patches,
                return_transf_layer=-1, output_token="combined", device=device, pe_mode=pe_mode,
            )
            t0 = time.time()
            n_ok = n_skip = n_err = 0
            for dataset, subject_id, h5_path in subjects:
                out_paths = {
                    key: output_root / variant_dir_key(variant_name, *key) / dataset / f"{subject_id}.npy"
                    for key in LAYER_TOKEN_KEYS
                }
                if not args.no_skip and all(p.exists() for p in out_paths.values()):
                    n_skip += 1
                    continue
                try:
                    variants_out, fill_info = extract_subject_all_variants(
                        h5_path, backbone, device, args.chunk_batch_size, windowing,
                    )
                    for key, emb in variants_out.items():
                        out_paths[key].parent.mkdir(parents=True, exist_ok=True)
                        np.save(out_paths[key], emb)
                    n_ok += 1
                except Exception as exc:
                    logger.error(f"  FAILED {dataset}/{subject_id} [{variant_name}]: {exc}")
                    n_err += 1
            logger.info(f"  {variant_name}: {n_ok} ok, {n_skip} skipped, {n_err} errors, "
                        f"{time.time() - t0:.1f}s")
            del backbone
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── Score all 12 variants with the probe (checklist 1.5's run_probe) ────
    logger.info("\n=== Scoring all 12 variants ===")
    results = {}
    for variant_name in VARIANTS:
        for layer, token in LAYER_TOKEN_KEYS:
            key = variant_dir_key(variant_name, layer, token)
            emb_dir = output_root / key
            try:
                res = run_probe(str(emb_dir), datasets=args.datasets, quiet=True)
                results[key] = res
                logger.info(
                    f"  {key:24s} F1={res['weighted_f1']:.4f} kappa={res['kappa']:.4f} "
                    f"(n_subj={res['n_subjects']}, test_subj={res['n_test_subj']})"
                )
            except SystemExit as e:
                logger.error(f"  {key:24s} SCORING FAILED: {e}")

    logger.info("\n=== SUMMARY TABLE (plan §13.1/§13.2) ===")
    logger.info(f"{'variant':10s} {'layer':6s} {'token':10s} {'F1':>8s} {'kappa':>8s}")
    for variant_name in VARIANTS:
        for layer, token in LAYER_TOKEN_KEYS:
            key = variant_dir_key(variant_name, layer, token)
            layer_label = "L2" if layer == 2 else "Llast"
            if key in results:
                r = results[key]
                logger.info(f"{variant_name:10s} {layer_label:6s} {token:10s} "
                            f"{r['weighted_f1']:8.4f} {r['kappa']:8.4f}")

    # ── Decision rules (plan §13.1/§13.2 escape hatches) ─────────────────────
    d_last = results.get(variant_dir_key("D", -1, "combined"))
    b_last = results.get(variant_dir_key("B", -1, "combined"))
    if d_last and b_last:
        gap = b_last["weighted_f1"] - d_last["weighted_f1"]
        logger.info(f"\nD vs B (combined@last): D={d_last['weighted_f1']:.4f} "
                    f"B={b_last['weighted_f1']:.4f} gap={gap:.4f}")
        if gap > 0.15:
            logger.warning("⚠️  ESCAPE HATCH TRIGGERED: B beats D by >0.15 F1 — "
                            "reconsider windowing default (plan §13.1).")
        else:
            logger.info("Windowing decision CONFIRMED: Option D (no escape hatch triggered).")

    d_l2 = results.get(variant_dir_key("D", 2, "combined"))
    if d_last and d_l2:
        gap2 = d_l2["weighted_f1"] - d_last["weighted_f1"]
        logger.info(f"\ncombined@2 vs combined@last (Option D): "
                    f"@2={d_l2['weighted_f1']:.4f} @last={d_last['weighted_f1']:.4f} gap={gap2:.4f}")
        if gap2 > 0.08:
            logger.warning("⚠️  ESCAPE HATCH TRIGGERED: layer-2 beats last layer by >0.08 F1 — "
                            "reconsider output-layer default (plan §13.2).")
        else:
            logger.info("Output-layer decision CONFIRMED: combined@last (no escape hatch triggered).")


if __name__ == "__main__":
    main()
