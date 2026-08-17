#!/usr/bin/env python3
"""
infer_osf_lora_subject_windows.py — OSF baseline, Stage 2 Step 2 (LoRA inference)

Loads a train_osf_lora.py checkpoint (LoRA deltas + sequence_head, via
peft's get_peft_model_state_dict/set_peft_model_state_dict) and runs
inference on ALL available windows per subject (not capped at K=5), saving
per-window probabilities/predictions for downstream subject-level
aggregation — same purpose and output schema as Stage 1's
infer_osf_subject_windows.py, but the LoRA-adapted backbone must run live
on raw signal per window instead of reading a precomputed .npy embedding,
since the fine-tuned weights change what the backbone produces. See
docs/TSFM_OSF_IMPLEMENTATION_PLAN.md checklist 2.4.

Reuses (not duplicates) train_osf_lora.py's model-construction code
(CombinedOSFLoRAModel via build_combined_lora_model) — same reasoning as
train_osf_lora.py importing run_epoch/compute_metrics from
train_osf_context_sweep.py: this logic is already tested, and duplicating
it here would risk architecture drift between train and inference.

SEQ2LABEL ONLY — matches OSFRawEpochWindowDataset's current scope (Stage 1
hasn't trained sleep_staging/seq2seq either), so no anchor_patch_end
handling like Stage 1's inference script has for seq2seq.

BATCH SIZE — deliberately does NOT reuse Stage 1's context-length-based
auto-scaling formula (_ref_bs/_ref_N). That formula was calibrated for
looking up cheap precomputed embeddings; here every item still runs a full
LoRA-adapted ViT forward pass per raw epoch (chunked inside
CombinedOSFLoRAModel.forward via chunk_batch_size), so the compute/memory
profile is fundamentally different. Default here is a fixed --batch-size,
32 (raised from an original default of 4 on 2026-08-17: with no
gradient/effective-batch concept, batch size is a pure throughput knob for
inference, and 4 turned out to be badly under-batched — the same
per-call-overhead problem chunk_batch_size fixed for training. Real GPU
numbers, not a formula: at 30s context, batch=4 made a 2.3M-item "all
windows" inference pass take longer than its 3h wall-time estimate
without finishing; no direct A/B was re-run at batch=32 before this
default changed, so treat 32 as a reasonable starting point, not a
calibrated optimum — lower it if a longer context OOMs). RESUMABILITY —
added 2026-08-17: a timed-out/killed run previously had to restart the
*entire* multi-hour "all windows" pass from item 0 (no partial-progress
resume, unlike training's per-epoch resume.pt), which combined with the
wall-time-inheritance behavior in the job script's auto-resume trap could
loop forever without ever finishing. Fixed via periodic, time-based
in-loop checkpointing (see run_inference()) — a `{split}_windows.resume.pt`
file next to the output parquet, refreshed every _CHECKPOINT_INTERVAL_SEC
regardless of any signal (so the safety margin doesn't depend on racing a
SIGTERM against the job's hard kill), consumed automatically on the next
invocation, and deleted once the final parquet is written.

Output (per context) — same schema as Stage 1's inference:
    {inference_dir}/{task}_{head}/context_{ctx}/{split}_windows.parquet

Parquet columns:
    subject_id, dataset, window_idx,   — subject identity and window position
    true_label,                         — ground truth label
    pred_label,                         — argmax prediction
    prob_class0 … prob_classN           — softmax probabilities per class

Usage:
    python scripts/infer_osf_lora_subject_windows.py \\
        --config configs/phase0_osf_lora_config.yaml \\
        --task apnea_binary --head lstm --context 10m

    # Multiple contexts, already-done ones skipped automatically:
    python scripts/infer_osf_lora_subject_windows.py \\
        --config configs/phase0_osf_lora_config.yaml \\
        --task apnea_binary --head lstm --context 30s 10m 40m 80m

    # Use only K=5 windows (reproduces training eval exactly):
    python scripts/infer_osf_lora_subject_windows.py ... --no-all-windows
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
from nsrr_tools.datasets.osf_raw_epoch_dataset import OSFRawEpochWindowDataset

sys.path.insert(0, str(_ROOT / "scripts"))
from train_osf_lora import build_combined_lora_model  # noqa: E402

# How often run_inference() flushes partial progress to disk, independent
# of any signal — this (not the SIGTERM handler) is what bounds lost work
# on a timeout to ~this many seconds, since it doesn't race a hurried save
# against SLURM's hard kill. See module docstring's RESUMABILITY note.
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
                  limit: int = None) -> OSFRawEpochWindowDataset:
    """Build an OSFRawEpochWindowDataset, optionally overriding K_max to use
    all windows. Same override technique as Stage 1's build_dataset — the
    in-memory cfg copy's dataset.windows_per_subject is read as _K_max by
    the dataset class; this does not modify phase0_osf_lora_config.yaml.

    limit (debug only, NOT present in Stage 1's inference script): caps the
    subject pool to the first N subjects of the split. Stage 1's inference
    never needed this — it reads cheap precomputed embeddings, so full-scope
    inference is fast even without a cap. Stage 2 runs a live LoRA-adapted
    backbone forward pass per window, so full-scope CPU debugging is
    impractically slow without restricting subject count first."""
    if all_windows:
        cfg["dataset"]["windows_per_subject"] = 99_999

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = OSFRawEpochWindowDataset(
            cfg=cfg,
            split=split,
            context_length=context_length,
            task=task,
            datasets=datasets_filter,
            limit=limit,
        )
    return ds


def get_subject_ids(ds: OSFRawEpochWindowDataset) -> list:
    """Return a list of (subject_id, dataset_name) aligned to ds._index."""
    ids = []
    for row_idx, _, _ in ds._index:
        row = ds.df.iloc[row_idx]
        ids.append((str(row["subject_id"]), str(row["dataset"])))
    return ids


def _save_resume_checkpoint(resume_path: Path, n_done: int,
                            logits_np: np.ndarray, targets_np: np.ndarray):
    """Atomic write (temp file + rename) so a kill mid-save can't leave a
    corrupt resume file — this save itself is unhurried (runs on
    run_inference()'s own timer, not under signal pressure), but the
    process could still be killed at any instant by SLURM's hard wall-time
    limit."""
    tmp_path = resume_path.with_suffix(resume_path.suffix + ".tmp")
    torch.save({"n_done": n_done, "logits": logits_np, "targets": targets_np}, tmp_path)
    os.replace(tmp_path, resume_path)


def run_inference(model: torch.nn.Module, ds, device: torch.device,
                  batch_size: int, num_workers: int, resume_path: Path):
    """Return (logits_np, targets_np, completed) over the full dataset,
    resuming from `resume_path` if a previous attempt was interrupted
    partway through.

    `ds` is iterated in a fixed order (shuffle=False everywhere in this
    script), so "already done" is just "the first n_done items of ds" —
    resuming means building a DataLoader over Subset(ds, range(n_done,
    len(ds))) and continuing to append. Progress is flushed to
    `resume_path` (accumulated logits/targets so far, atomically) every
    _CHECKPOINT_INTERVAL_SEC, independent of the SIGTERM handler, so a
    timeout loses at most that interval of work rather than the entire
    multi-hour pass. `completed=False` means the caller should stop (the
    job is about to be killed) without treating this as an error."""
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
            n_done += x.shape[0]

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
        description="Run all-window OSF-LoRA inference for subject-level aggregation."
    )
    parser.add_argument("--config",     required=True, help="Path to phase0_osf_lora_config.yaml")
    parser.add_argument("--task",       required=True, help="Task name, e.g. apnea_binary")
    parser.add_argument("--head",       required=True, dest="head_type",
                        help="lstm | transformer | mean_pool")
    parser.add_argument("--context",    default=None, nargs="+",
                        help="One or more context lengths, e.g. --context 30s 10m 40m 80m. "
                             "If omitted, auto-discovers all available checkpoints.")
    parser.add_argument("--split",      default="test",
                        choices=["train", "val", "test"],
                        help="Which split to run inference on (default: test)")
    parser.add_argument("--datasets",   default=None, nargs="+",
                        help="Restrict to these datasets, e.g. apples shhs")
    parser.add_argument("--no-all-windows", action="store_true", dest="no_all_windows",
                        help="Use K=5 windows (reproduces training eval) instead of all windows")
    parser.add_argument("--batch-size", default=32, type=int, dest="batch_size",
                        help="Fixed micro-batch size (default: 32, raised 2026-08-17 from an "
                             "under-batched default of 4 — see module docstring's BATCH SIZE "
                             "note; NOT auto-scaled by context length, lower it if a longer "
                             "context OOMs)")
    parser.add_argument("--num-workers", default=2, type=int, dest="num_workers")
    parser.add_argument("--cpu",        action="store_true")
    parser.add_argument("--out-dir",    default=None, dest="out_dir",
                        help="Override output directory")
    parser.add_argument("--run-tag",    default="", dest="run_tag",
                        help="Must match the --run-tag used during training (default: no suffix).")
    parser.add_argument("--limit",      default=None, type=int,
                        help="DEBUG ONLY, not in Stage 1's inference script: cap the subject "
                             "pool to the first N subjects of the split. Real/final inference "
                             "runs should omit this — see build_dataset()'s docstring for why "
                             "Stage 2 needs it and Stage 1 doesn't.")
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
    print("OSF baseline — Stage 2 (LoRA) — Subject-level inference")
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

        # ── Output path ───────────────────────────────────────────────────────
        if args.out_dir:
            out_dir = Path(args.out_dir) / f"context_{ctx}"
        else:
            out_dir = results_dir / "inference" / exp_id / f"context_{ctx}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_parquet = out_dir / f"{args.split}_windows.parquet"
        resume_path = out_dir / f"{args.split}_windows.resume.pt"

        # ── Skip if already done ──────────────────────────────────────────────
        if out_parquet.exists():
            print(f"  [SKIP] {out_parquet.name} already exists — delete to rerun.")
            continue

        # ── Locate checkpoint ─────────────────────────────────────────────────
        ckpt_path = results_dir / exp_id / f"context_{ctx}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        print(f"  Checkpoint:  {ckpt_path}")
        print(f"  Output:      {out_parquet}")

        model = None
        try:
            # ── Build dataset ─────────────────────────────────────────────────
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

            # ── Load model ────────────────────────────────────────────────────
            metrics_path = ckpt_path.parent / "metrics.json"
            with open(metrics_path) as f:
                saved_metrics = json.load(f)
            num_classes = saved_metrics["num_classes"]

            # Unlike Stage 1's inference script, architecture (hidden_dim,
            # num_layers, LoRA r/alpha) is read straight from cfg, not
            # auto-detected from the checkpoint's tensor shapes — Stage 2's
            # config isn't varied per checkpoint the way Stage 1's was.
            # If cfg ever drifts from what a checkpoint was trained with,
            # set_peft_model_state_dict below fails loudly on a shape
            # mismatch rather than silently loading wrong weights.
            model = build_combined_lora_model(cfg, num_classes, args.head_type, device)
            peft_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            set_peft_model_state_dict(model, peft_state)
            model.eval()
            print(f"  num_classes: {num_classes}")

            # ── Inference (resumable — see run_inference()'s docstring) ────────
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

            # ── Build output DataFrame ────────────────────────────────────────
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
            resume_path.unlink(missing_ok=True)  # done — no longer needed

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
            # free GPU memory before the next context so OOM in one context
            # doesn't cascade into the next
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
    # interrupted (not any_failed): exit 0 — not an error, and bash's USR1
    # trap has already (or will imminently) fire the TIMEOUT_REQUEUED
    # resubmit independently of this process's own exit code.


if __name__ == "__main__":
    main()
