#!/usr/bin/env python3
"""
infer_physioomni_lora_subject_windows.py — PhysioOmni baseline, Stage 2
Step 2 (LoRA inference)

Loads a train_physioomni_lora.py checkpoint (LoRA deltas + sequence_head,
via peft's get_peft_model_state_dict/set_peft_model_state_dict) and runs
inference on ALL available windows per subject (not capped at K=5), saving
per-window probabilities/predictions for downstream subject-level
aggregation — same purpose and output schema as Stage 1's
infer_physioomni_subject_windows.py, but the LoRA-adapted encoders must run
live on raw signal per window instead of reading a precomputed .npy
embedding, since the fine-tuned weights change what they produce.

Reuses (not duplicates) train_physioomni_lora.py's model-construction code
(CombinedPhysioOmniLoRAModel via build_combined_lora_model) — same
reasoning as train_physioomni_lora.py importing run_epoch/compute_metrics
from train_physioomni_context_sweep.py.

SEQ2LABEL ONLY — matches PhysioOmniRawEpochWindowDataset's current scope.

BATCH SIZE — deliberately does NOT reuse Stage 1's context-length-based
auto-scaling formula (that formula assumes a cheap precomputed-embedding
lookup; here every item still runs live encoder forward passes, chunked
inside CombinedPhysioOmniLoRAModel.forward via chunk_batch_size). Fixed
--batch-size, default 32 — same starting point as OSF's own Stage 2
inference script, not independently calibrated for PhysioOmni.

RESUMABILITY — same periodic, time-based checkpointing as OSF's Stage 2
inference script: a `{split}_windows.resume.pt` file next to the output
parquet, refreshed every _CHECKPOINT_INTERVAL_SEC regardless of signal,
consumed automatically on the next invocation, deleted once the final
parquet is written.

Output (per context) — same schema as Stage 1's inference:
    {inference_dir}/{task}_{head}/context_{ctx}/{split}_windows.parquet

Parquet columns:
    subject_id, dataset, window_idx, true_label, pred_label, prob_class0…N

Usage:
    python scripts/infer_physioomni_lora_subject_windows.py \\
        --config configs/phase0_physioomni_lora_config.yaml \\
        --task sex_binary --head lstm --context 10m

    # Multiple contexts, already-done ones skipped automatically:
    python scripts/infer_physioomni_lora_subject_windows.py \\
        --config configs/phase0_physioomni_lora_config.yaml \\
        --task sex_binary --head lstm --context 30s 10m 40m 80m
"""

import argparse
import json
import os
import signal
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from peft import set_peft_model_state_dict
from torch.utils.data import DataLoader, Subset

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
from nsrr_tools.datasets.physioomni_raw_epoch_dataset import (
    PhysioOmniRawEpochWindowDataset,
    physioomni_lora_collate_fn,
)

sys.path.insert(0, str(_ROOT / "scripts"))
from train_physioomni_lora import build_combined_lora_model  # noqa: E402

_CHECKPOINT_INTERVAL_SEC = 300

_STOP_REQUESTED = False


def _handle_sigterm(signum, frame):
    global _STOP_REQUESTED
    print("\n[SIGTERM] Timeout — will save progress and stop after the current batch "
          "(periodic checkpoints already cover most of the loss)", flush=True)
    _STOP_REQUESTED = True

signal.signal(signal.SIGTERM, _handle_sigterm)


def _classify_failure(exc: BaseException) -> str:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return "oom"
    if isinstance(exc, (OSError, IOError)):
        return f"io_error: {str(exc)[:120]}"
    return f"error: {type(exc).__name__}: {str(exc)[:120]}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_dataset(cfg: dict, split: str, context_length: str,
                  task: str, datasets_filter: list, all_windows: bool,
                  limit: int = None) -> PhysioOmniRawEpochWindowDataset:
    """Build a PhysioOmniRawEpochWindowDataset, optionally overriding K_max
    to use all windows — same override technique as Stage 1's
    build_dataset (in-memory cfg copy only, never touches the yaml file)."""
    if all_windows:
        cfg["dataset"]["windows_per_subject"] = 99_999

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = PhysioOmniRawEpochWindowDataset(
            cfg=cfg,
            split=split,
            context_length=context_length,
            task=task,
            datasets=datasets_filter,
            limit=limit,
        )
    return ds


def get_subject_ids(ds: PhysioOmniRawEpochWindowDataset) -> list:
    """Return a list of (subject_id, dataset_name) aligned to ds._index."""
    ids = []
    for row_idx, _, _ in ds._index:
        row = ds.df.iloc[row_idx]
        ids.append((str(row["subject_id"]), str(row["dataset"])))
    return ids


def _save_resume_checkpoint(resume_path: Path, n_done: int,
                            logits_np: np.ndarray, targets_np: np.ndarray):
    """Atomic write (temp file + rename) so a kill mid-save can't leave a
    corrupt resume file."""
    tmp_path = resume_path.with_suffix(resume_path.suffix + ".tmp")
    torch.save({"n_done": n_done, "logits": logits_np, "targets": targets_np}, tmp_path)
    os.replace(tmp_path, resume_path)


def run_inference(model: torch.nn.Module, ds, device: torch.device,
                  batch_size: int, num_workers: int, resume_path: Path):
    """Return (logits_np, targets_np, completed) over the full dataset,
    resuming from `resume_path` if a previous attempt was interrupted
    partway through. Same periodic-checkpoint pattern as OSF's Stage 2
    inference script — see that script's run_inference() docstring for the
    full reasoning (resumability was added after a real, previously-live
    "restart from item 0" problem)."""
    n_total = len(ds)
    saved_logits  = None
    saved_targets = None
    n_done = 0

    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        n_done        = state["n_done"]
        saved_logits  = state["logits"]
        saved_targets = state["targets"]
        print(f"  [RESUME] {n_done:,}/{n_total:,} items already done "
              f"({100 * n_done / max(n_total, 1):.1f}%) — continuing", flush=True)

    if n_done >= n_total:
        return saved_logits, saved_targets, True

    remaining = Subset(ds, range(n_done, n_total))
    loader = DataLoader(
        remaining, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        collate_fn=physioomni_lora_collate_fn,
    )

    model.eval()
    new_logits  = [saved_logits]  if saved_logits  is not None else []
    new_targets = [saved_targets] if saved_targets is not None else []
    pending_logits, pending_targets = [], []
    last_checkpoint = time.monotonic()
    completed = True

    with torch.no_grad():
        for x, mask, y in loader:
            x    = x.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            logits = model(x, mask)
            pending_logits.append(logits.cpu().float().numpy())
            pending_targets.append(y.numpy())
            n_done += x.size(0)

            due = (time.monotonic() - last_checkpoint) >= _CHECKPOINT_INTERVAL_SEC
            if due or _STOP_REQUESTED:
                new_logits.append(np.concatenate(pending_logits, axis=0))
                new_targets.append(np.concatenate(pending_targets, axis=0))
                pending_logits, pending_targets = [], []
                _save_resume_checkpoint(
                    resume_path, n_done,
                    np.concatenate(new_logits, axis=0), np.concatenate(new_targets, axis=0),
                )
                last_checkpoint = time.monotonic()
                print(f"  [CHECKPOINT] {n_done:,}/{n_total:,} items "
                      f"({100 * n_done / max(n_total, 1):.1f}%)", flush=True)
                if _STOP_REQUESTED:
                    completed = False
                    break

    if pending_logits:
        new_logits.append(np.concatenate(pending_logits, axis=0))
        new_targets.append(np.concatenate(pending_targets, axis=0))

    logits_np  = np.concatenate(new_logits,  axis=0)
    targets_np = np.concatenate(new_targets, axis=0)
    return logits_np, targets_np, completed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all-window PhysioOmni-LoRA inference for subject-level aggregation."
    )
    parser.add_argument("--config",     required=True, help="Path to phase0_physioomni_lora_config.yaml")
    parser.add_argument("--task",       required=True, help="Task name, e.g. sex_binary")
    parser.add_argument("--head",       required=True, dest="head_type",
                        help="lstm | transformer (mean_pool deferred, plan §15.10)")
    parser.add_argument("--context",    default=None, nargs="+",
                        help="One or more context lengths. If omitted, auto-discovers "
                             "all available checkpoints.")
    parser.add_argument("--split",      default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--datasets",   default=None, nargs="+")
    parser.add_argument("--no-all-windows", action="store_true", dest="no_all_windows",
                        help="Use K=5 windows (reproduces training eval) instead of all windows")
    parser.add_argument("--batch-size", default=32, type=int, dest="batch_size",
                        help="Fixed micro-batch size (default: 32, NOT auto-scaled by "
                             "context length — lower it if a longer context OOMs)")
    parser.add_argument("--num-workers", default=2, type=int, dest="num_workers")
    parser.add_argument("--cpu",        action="store_true")
    parser.add_argument("--out-dir",    default=None, dest="out_dir")
    parser.add_argument("--run-tag",    default="", dest="run_tag")
    parser.add_argument("--limit",      default=None, type=int,
                        help="DEBUG ONLY: cap the subject pool to the first N subjects "
                             "of the split. Omit for real runs.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    all_windows = not args.no_all_windows

    results_dir = Path(cfg["logging"]["results_dir"])
    exp_id      = f"{args.task}_{args.head_type}" + (f"_{args.run_tag}" if args.run_tag else "")

    if args.context:
        contexts = args.context
    else:
        found = sorted(p.parent.name for p in
                       (results_dir / exp_id).glob("context_*/best_model.pt"))
        if not found:
            print(f"ERROR: No checkpoints found under {results_dir / exp_id}/context_*/")
            sys.exit(1)
        contexts = [d.replace("context_", "") for d in found]
        print(f"  Auto-discovered contexts: {contexts}")

    print("=" * 68)
    print("PhysioOmni baseline — Stage 2 (LoRA) — Subject-level inference")
    print("=" * 68)
    print(f"  Task:        {args.task}  (seq2label)")
    print(f"  Head:        {args.head_type}")
    print(f"  Contexts:    {contexts}")
    print(f"  Split:       {args.split}")
    print(f"  All windows: {all_windows}")
    print(f"  Datasets:    {args.datasets or '(all)'}")
    print(f"  Device:      {device}")
    if args.limit is not None:
        print(f"  Limit:       {args.limit} subjects  ⚠️  DEBUG ONLY — omit for real runs")
    print()

    any_failed = False
    interrupted = False
    failure_reasons: list[str] = []

    for ctx in contexts:
        print(f"\n{'='*60}")
        print(f"  Context: {ctx}")
        print(f"{'='*60}")

        if args.out_dir:
            out_dir = Path(args.out_dir) / f"context_{ctx}"
        else:
            out_dir = results_dir / "inference" / exp_id / f"context_{ctx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_parquet = out_dir / f"{args.split}_windows.parquet"
        resume_path = out_dir / f"{args.split}_windows.resume.pt"

        if out_parquet.exists():
            print(f"  [SKIP] {out_parquet.name} already exists — delete to rerun.")
            continue

        ckpt_path = results_dir / exp_id / f"context_{ctx}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        print(f"  Checkpoint:  {ckpt_path}")
        print(f"  Output:      {out_parquet}")

        model = None
        try:
            ds = build_dataset(
                cfg=cfg,
                split=args.split,
                context_length=ctx,
                task=args.task,
                datasets_filter=args.datasets,
                all_windows=all_windows,
                limit=args.limit,
            )
            print(f"  Dataset items: {len(ds):,}  (subjects: {len(ds.df):,})")

            subject_ids = get_subject_ids(ds)

            metrics_path = ckpt_path.parent / "metrics.json"
            with open(metrics_path) as f:
                saved_metrics = json.load(f)
            num_classes = saved_metrics["num_classes"]

            # Architecture (hidden_dim, num_layers, LoRA r/alpha) is read
            # straight from cfg, not auto-detected from the checkpoint's
            # tensor shapes — same convention as OSF's Stage 2 inference.
            # If cfg ever drifts from what a checkpoint was trained with,
            # set_peft_model_state_dict below fails loudly on a shape
            # mismatch rather than silently loading wrong weights.
            model = build_combined_lora_model(cfg, num_classes, args.head_type, device)
            peft_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            set_peft_model_state_dict(model, peft_state)
            model.eval()
            print(f"  num_classes: {num_classes}")

            print("  Running inference...")
            logits_np, targets_np, completed = run_inference(
                model, ds, device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                resume_path=resume_path,
            )

            if not completed:
                print(f"  [INTERRUPTED] Timeout signal received — progress saved to "
                      f"{resume_path.name}, will resume automatically on the next run.")
                interrupted = True
                break

            probs = torch.softmax(torch.from_numpy(logits_np), dim=-1).numpy()
            preds = logits_np.argmax(axis=1)

            rows = {
                "subject_id": [sid   for sid, _     in subject_ids],
                "dataset":    [dname for _,   dname in subject_ids],
                "true_label": targets_np.astype(np.int16),
                "pred_label": preds.astype(np.int16),
            }
            for c in range(num_classes):
                rows[f"prob_class{c}"] = probs[:, c].astype(np.float32)

            window_idx = np.zeros(len(subject_ids), dtype=np.int32)
            seen: dict = {}
            for i, (sid, dname) in enumerate(subject_ids):
                key = (sid, dname)
                window_idx[i] = seen.get(key, 0)
                seen[key] = seen.get(key, 0) + 1
            rows["window_idx"] = window_idx

            df_out = pd.DataFrame(rows)
            df_out.to_parquet(out_parquet, index=False)
            resume_path.unlink(missing_ok=True)

            seg_acc          = (df_out["pred_label"] == df_out["true_label"]).mean()
            n_subjects       = df_out.groupby(["subject_id", "dataset"]).ngroups
            windows_per_subj = len(df_out) / max(n_subjects, 1)
            print(f"  Saved {len(df_out):,} rows → {out_parquet}")
            print(f"  Segment accuracy: {seg_acc*100:.2f}%  |  "
                  f"Subjects: {n_subjects:,}  |  Avg windows: {windows_per_subj:.1f}")

        except Exception as exc:
            import traceback
            print(f"\n[ERROR] context={ctx}: {exc}")
            traceback.print_exc()
            any_failed = True
            failure_reasons.append(f"{ctx}: {_classify_failure(exc)}")
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    if interrupted:
        print("Stopped early — timeout signal received, progress saved for auto-resume.")
    elif any_failed:
        print("Contexts finished (with errors — see [ERROR] lines above).")
    else:
        print("All contexts processed successfully.")

    if any_failed:
        reason_str  = "; ".join(failure_reasons)
        infer_dir   = results_dir / "inference" / exp_id
        infer_dir.mkdir(parents=True, exist_ok=True)
        reason_file = infer_dir / f"_failure_reason_{os.environ.get('SLURM_JOB_ID', 'local')}.txt"
        reason_file.write_text(reason_str)
        print(f"\n[WARNING] One or more contexts failed: {reason_str}")
        sys.exit(1)


if __name__ == "__main__":
    main()
