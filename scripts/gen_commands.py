#!/usr/bin/env python3
"""
Experiment command generator for v2 task-definition experiments.

Usage:
  python scripts/gen_commands.py list [--tier 1|2]
      List all experiments with status (pending/trained/inferred/analyzed).

  python scripts/gen_commands.py train <exp_id> [--context 30s 10m ...]
      Print sbatch command(s) for training. One job per context.
      Omit --context to print commands for all contexts in the registry.

  python scripts/gen_commands.py infer <exp_id> [--split test|val]
      Print the sbatch command for inference (auto-discovers trained contexts).

  python scripts/gen_commands.py analyze <exp_id> [--plot]
      Print the python command for window analysis.

  python scripts/gen_commands.py status [<exp_id>]
      Show detailed file-level status for one or all experiments.

  python scripts/gen_commands.py runs [<exp_id>]
      Show job run history (from logs_v2/status/*.jsonl tracking files).

Examples:
  python scripts/gen_commands.py list --tier 1
  python scripts/gen_commands.py train sex_binary_lstm
  python scripts/gen_commands.py train sex_binary_lstm --context 30s
  python scripts/gen_commands.py infer sex_binary_lstm
  python scripts/gen_commands.py infer sex_binary_lstm --split val
  python scripts/gen_commands.py analyze sex_binary_lstm --plot
  python scripts/gen_commands.py status
  python scripts/gen_commands.py runs
  python scripts/gen_commands.py runs sex_binary_lstm
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).parent.parent / "experiments" / "v2_registry.yaml"
JOBS_DIR = Path(__file__).parent.parent / "jobs"

# ── Wall-time lookup tables ────────────────────────────────────────────────────
# Conservative estimates with ~50-100% margin over observed runtimes.
# (n_size, head) → {context: hours}

# Calibrated from sex_binary_lstm (large, N~13k, K=5, batch=32) observed on H100:
#   30s=31min/18ep, 10m=46min/17ep, 40m=80min/24ep, 80m=111min/22ep, 120m=185min/25ep
# Estimates = max_epochs(30) × per_epoch × 1.5 safety, rounded to nearest 0.5h.
# Checkpoint resume means underestimates just cause one requeue — no data loss.
_TRAIN_HOURS = {
    ("large",  "lstm"):        {"30s": 2,   "10m": 3,   "40m": 3,   "80m": 4,   "120m": 6,   "240m": 12,  "full_night": 8 },
    ("large",  "transformer"): {"30s": 2,   "10m": 3,   "40m": 4,   "80m": 8,   "120m": 12,  "240m": 24,  "full_night": 24},
    ("large",  "mean_pool"):   {"30s": 1,   "10m": 1,   "40m": 1,   "80m": 1,   "120m": 1,   "240m": 2,   "full_night": 2 },
    ("medium", "lstm"):        {"30s": 1,   "10m": 2,   "40m": 2,   "80m": 3,   "120m": 4,   "240m": 8,   "full_night": 4 },
    ("medium", "transformer"): {"30s": 1,   "10m": 2,   "40m": 3,   "80m": 6,   "120m": 8,   "240m": 16,  "full_night": 24},
    ("medium", "mean_pool"):   {"30s": 1,   "10m": 1,   "40m": 1,   "80m": 1,   "120m": 1,   "240m": 1,   "full_night": 1 },
    ("small",  "lstm"):        {"30s": 1,   "10m": 1,   "40m": 1,   "80m": 2,   "120m": 2,   "240m": 4,   "full_night": 2 },
    ("small",  "transformer"): {"30s": 1,   "10m": 1,   "40m": 2,   "80m": 3,   "120m": 4,   "240m": 8,   "full_night": 24},
    ("small",  "mean_pool"):   {"30s": 1,   "10m": 1,   "40m": 1,   "80m": 1,   "120m": 1,   "240m": 1,   "full_night": 1 },
}

# Per-context inference hours (one job runs all contexts sequentially).
# Calibrated from sex_binary_lstm (large, apples+shhs, batch=512, H100 10GB MIG):
#   30s=~20 min (1.5M items), 10m=~3 min, 40m=~1 min, 80m=~1 min; all 4 in 26 min.
# Inference cost is dominated by 30s (many short windows); longer contexts are trivially fast.
# With auto-requeue an underestimate just causes one extra job — use tight values.
_INFER_HOURS_PER_CTX = {
    ("large",  "lstm"):        {"30s": 0.5, "10m": 0.1, "40m": 0.1, "80m": 0.1, "120m": 0.1, "240m": 0.1, "full_night": 0.5},
    ("large",  "transformer"): {"30s": 0.5, "10m": 0.1, "40m": 0.1, "80m": 0.1, "120m": 0.1, "240m": 0.1, "full_night": 0.5},
    ("large",  "mean_pool"):   {"30s": 0.25,"10m": 0.1, "40m": 0.1, "80m": 0.1, "120m": 0.1, "240m": 0.1, "full_night": 0.25},
    ("medium", "lstm"):        {"30s": 0.5, "10m": 0.1, "40m": 0.1, "80m": 0.1, "120m": 0.1, "240m": 0.1, "full_night": 0.5},
    ("medium", "transformer"): {"30s": 0.5, "10m": 0.1, "40m": 0.1, "80m": 0.1, "120m": 0.1, "240m": 0.1, "full_night": 0.5},
    ("medium", "mean_pool"):   {"30s": 0.25,"10m": 0.1, "40m": 0.1, "80m": 0.1, "120m": 0.1, "240m": 0.1, "full_night": 0.25},
    ("small",  "lstm"):        {"30s": 0.25,"10m": 0.1, "40m": 0.1, "80m": 0.1, "120m": 0.1, "240m": 0.1, "full_night": 0.25},
    ("small",  "transformer"): {"30s": 0.25,"10m": 0.1, "40m": 0.1, "80m": 0.1, "120m": 0.1, "240m": 0.1, "full_night": 0.25},
    ("small",  "mean_pool"):   {"30s": 0.25,"10m": 0.1, "40m": 0.1, "80m": 0.1, "120m": 0.1, "240m": 0.1, "full_night": 0.25},
}


def estimate_train_time(n_size: str, head: str, context: str) -> str:
    hours = _TRAIN_HOURS.get((n_size, head), {}).get(context, 24)
    return f"{int(hours):02d}:00:00"


def estimate_infer_time(n_size: str, head: str, contexts: list) -> str:
    per_ctx = _INFER_HOURS_PER_CTX.get((n_size, head), {})
    total = sum(per_ctx.get(ctx, 1.0) for ctx in contexts)
    total = max(total, 1.0)
    h = int(total)
    m = int((total - h) * 60)
    return f"{h:02d}:{m:02d}:00"


def format_lr(lr) -> str:
    """Format learning rate for filenames: 1.0e-4 → 1e-4"""
    s = f"{float(lr):.0e}"
    return re.sub(r"e(-?)0*(\d+)", r"e\1\2", s)


def load_registry() -> dict:
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def context_dir_name(ctx: str) -> str:
    return f"context_{ctx}"


def exp_folder(exp: dict, registry: dict) -> Path:
    results_dir = Path(registry["results_dir"])
    tag = exp.get("run_tag", "")
    suffix = f"_{tag}" if tag else ""
    return results_dir / f"{exp['task']}_{exp['head']}{suffix}"


def infer_folder(exp: dict, registry: dict) -> Path:
    infer_dir = Path(registry["inference_dir"])
    tag = exp.get("run_tag", "")
    suffix = f"_{tag}" if tag else ""
    return infer_dir / f"{exp['task']}_{exp['head']}{suffix}"


def _log_stem(exp: dict, step: str, context: str = "") -> str:
    """Build a descriptive log filename stem (no extension, no %j)."""
    tag = exp.get("run_tag", "")
    tag_part = f"_{tag}" if tag else ""
    lr_part = f"_lr{format_lr(exp['lr'])}"
    ctx_part = f"_{context}" if context else ""
    return f"{step}_{exp['task']}_{exp['head']}{tag_part}{ctx_part}{lr_part}"


# ── Status checks ─────────────────────────────────────────────────────────────

def trained_contexts(exp: dict, registry: dict) -> list:
    folder = exp_folder(exp, registry)
    return [ctx for ctx in exp["contexts"]
            if (folder / context_dir_name(ctx) / "best_model.pt").exists()]


def inferred_contexts(exp: dict, registry: dict, split: str = "test") -> list:
    folder = infer_folder(exp, registry)
    return [ctx for ctx in exp["contexts"]
            if (folder / context_dir_name(ctx) / f"{split}_windows.parquet").exists()]


def is_analyzed(exp: dict, registry: dict) -> bool:
    return (infer_folder(exp, registry) / "window_analysis.md").exists()


def exp_status(exp: dict, registry: dict) -> str:
    tr = trained_contexts(exp, registry)
    inf = inferred_contexts(exp, registry)
    ana = is_analyzed(exp, registry)
    n = len(exp["contexts"])
    if ana:
        return f"analyzed ({len(tr)}/{n} trained, {len(inf)}/{n} inferred)"
    if inf:
        return f"inferred ({len(inf)}/{n}), not analyzed"
    if tr:
        return f"trained ({len(tr)}/{n} contexts), not inferred"
    return "pending"


# ── Command builders ──────────────────────────────────────────────────────────

def build_train_cmd(exp: dict, registry: dict, context: str,
                    override_time: str = None, override_batch_size: int = None) -> str:
    cfg = registry["config"]
    logs_dir = registry.get("logs_dir", str(Path(__file__).parent.parent / "logs"))
    n_size = exp.get("n_size", "large")
    wall_time = override_time if override_time else estimate_train_time(n_size, exp["head"], context)
    stem = _log_stem(exp, "train", context)
    batch_size = override_batch_size if override_batch_size is not None else exp["batch_size"]

    env_vars = [
        f"TASK={exp['task']}",
        f"TASK_TYPE={exp['task_type']}",
        f"HEAD={exp['head']}",
        f"CONTEXT={context}",
        f"DATASETS=\"{' '.join(exp['datasets'])}\"",
        f"BATCH_SIZE={batch_size}",
        f"LR={exp['lr']}",
    ]
    if exp.get("run_tag"):
        env_vars.append(f"RUN_TAG={exp['run_tag']}")
    env_vars.append(f"CONFIG={cfg}")
    env_str = " ".join(env_vars)

    sbatch_opts = (
        f"--requeue "
        f"--time={wall_time} "
        f"--output={logs_dir}/{stem}_%j.out "
        f"--error={logs_dir}/{stem}_%j.err"
    )
    return f"{env_str} sbatch {sbatch_opts} {JOBS_DIR}/train_context_sweep_gpu.sh"


def build_infer_cmd(exp: dict, registry: dict, split: str = "test",
                    override_time: str = None, override_batch_size: int = None) -> str:
    cfg = registry["config"]
    logs_dir = registry.get("logs_dir", str(Path(__file__).parent.parent / "logs"))
    n_size = exp.get("n_size", "large")
    contexts_trained = trained_contexts(exp, registry)
    ctx_list = contexts_trained if contexts_trained else exp["contexts"]
    wall_time = override_time if override_time else estimate_infer_time(n_size, exp["head"], ctx_list)
    stem = _log_stem(exp, "infer")

    env_vars = [
        f"TASK={exp['task']}",
        f"TASK_TYPE={exp['task_type']}",
        f"HEAD={exp['head']}",
        f"CONTEXTS=\"{' '.join(ctx_list)}\"",
        f"SPLIT={split}",
        f"DATASETS=\"{' '.join(exp['datasets'])}\"",
    ]
    if override_batch_size is not None:
        env_vars.append(f"BATCH_SIZE={override_batch_size}")
    if exp.get("run_tag"):
        env_vars.append(f"RUN_TAG={exp['run_tag']}")
    env_vars.append(f"CONFIG={cfg}")
    env_str = " ".join(env_vars)

    sbatch_opts = (
        f"--requeue "
        f"--time={wall_time} "
        f"--output={logs_dir}/{stem}_%j.out "
        f"--error={logs_dir}/{stem}_%j.err"
    )
    return f"{env_str} sbatch {sbatch_opts} {JOBS_DIR}/infer_subject_windows_gpu.sh"


def build_analyze_cmd(exp: dict, registry: dict, plot: bool = False) -> str:
    infer_dir = Path(registry["inference_dir"])
    tag = exp.get("run_tag", "")
    cmd_parts = [
        "python scripts/analyze_windows.py",
        f"--task {exp['task']}",
        f"--head {exp['head']}",
        f"--results-dir {infer_dir}",
    ]
    if tag:
        cmd_parts.append(f"--run-tag {tag}")
    if plot:
        cmd_parts.append("--plot")
    return " ".join(cmd_parts)


# ── Subcommand handlers ───────────────────────────────────────────────────────

def cmd_list(args, registry):
    experiments = registry["experiments"]
    tier_filter = getattr(args, "tier", None)
    print(f"{'ID':<45} {'Tier':<6} {'N-ctx':<6} {'Datasets':<30} {'Status'}")
    print("-" * 115)
    for exp_id, exp in experiments.items():
        tier = exp.get("tier", "?")
        if tier_filter and str(tier) != str(tier_filter):
            continue
        datasets_str = ",".join(exp["datasets"])
        status = exp_status(exp, registry)
        print(f"{exp_id:<45} {str(tier):<6} {len(exp['contexts']):<6} {datasets_str:<30} {status}")


def cmd_train(args, registry):
    experiments = registry["experiments"]
    if args.exp_id not in experiments:
        print(f"ERROR: experiment '{args.exp_id}' not found in registry.", file=sys.stderr)
        sys.exit(1)
    exp = experiments[args.exp_id]
    contexts = args.context if args.context else exp["contexts"]
    n_size = exp.get("n_size", "large")
    print(f"# Training commands for: {args.exp_id}")
    print(f"# Task: {exp['task']}  Head: {exp['head']}  LR: {exp['lr']}  N-size: {n_size}")
    print(f"# Datasets: {exp['datasets']}")
    print(f"# Logs → {registry.get('logs_dir', 'logs/')}")
    print()
    for ctx in contexts:
        if ctx not in exp["contexts"]:
            print(f"# WARNING: context '{ctx}' not in registry for this experiment — skipping")
            continue
        trained = ctx in trained_contexts(exp, registry)
        wall = estimate_train_time(n_size, exp["head"], ctx)
        status_tag = "  # already trained" if trained else f"  # est. {wall}"
        print(build_train_cmd(exp, registry, ctx,
                              override_time=getattr(args, "override_time", None),
                              override_batch_size=getattr(args, "override_batch_size", None)) + status_tag)


def cmd_infer(args, registry):
    experiments = registry["experiments"]
    if args.exp_id not in experiments:
        print(f"ERROR: experiment '{args.exp_id}' not found.", file=sys.stderr)
        sys.exit(1)
    exp = experiments[args.exp_id]
    split = getattr(args, "split", "test")
    n_size = exp.get("n_size", "large")
    tr = trained_contexts(exp, registry)
    ctx_list = tr if tr else exp["contexts"]
    wall = estimate_infer_time(n_size, exp["head"], ctx_list)
    if not tr:
        print(f"# WARNING: no trained contexts found for '{args.exp_id}'. Command uses all contexts from registry.")
    print(f"# Inference command for: {args.exp_id}  split={split}")
    print(f"# Trained contexts: {tr or 'none found — check results dir'}")
    print(f"# Est. wall time: {wall}  Logs → {registry.get('logs_dir', 'logs/')}")
    print()
    print(build_infer_cmd(exp, registry, split,
                          override_time=getattr(args, "override_time", None),
                          override_batch_size=getattr(args, "override_batch_size", None)))


def cmd_analyze(args, registry):
    experiments = registry["experiments"]
    if args.exp_id not in experiments:
        print(f"ERROR: experiment '{args.exp_id}' not found.", file=sys.stderr)
        sys.exit(1)
    exp = experiments[args.exp_id]
    plot = getattr(args, "plot", False)
    print(f"# Window analysis command for: {args.exp_id}")
    print()
    print(build_analyze_cmd(exp, registry, plot))


def cmd_status(args, registry):
    experiments = registry["experiments"]
    target_id = getattr(args, "exp_id", None)
    for exp_id, exp in experiments.items():
        if target_id and exp_id != target_id:
            continue
        tr = trained_contexts(exp, registry)
        inf = inferred_contexts(exp, registry)
        ana = is_analyzed(exp, registry)
        print(f"\n{'='*60}")
        print(f"  {exp_id}  [tier {exp.get('tier','?')}]")
        print(f"  task={exp['task']}  head={exp['head']}  contexts={exp['contexts']}")
        print(f"  Trained:   {tr or 'none'}")
        print(f"  Inferred:  {inf or 'none'}")
        print(f"  Analyzed:  {'yes' if ana else 'no'}")


def cmd_runs(args, registry):
    logs_dir = Path(registry.get("logs_dir", str(Path(__file__).parent.parent / "logs_v2")))
    status_dir = logs_dir / "status"

    if not status_dir.exists():
        print("No status directory found yet (logs_v2/status/). No jobs tracked.")
        return

    target_id = getattr(args, "exp_id", None)
    target_task_head = None
    if target_id:
        if target_id not in registry["experiments"]:
            print(f"ERROR: '{target_id}' not in registry.", file=sys.stderr)
            sys.exit(1)
        exp = registry["experiments"][target_id]
        target_task_head = f"{exp['task']}_{exp['head']}"
        tag = exp.get("run_tag", "")
        if tag:
            target_task_head += f"_{tag}"

    files = sorted(status_dir.glob("*.jsonl"))
    if not files:
        print("No job history found in logs_v2/status/.")
        return

    shown = 0
    for jsonl_path in files:
        stem = jsonl_path.stem  # e.g. train_sex_binary_lstm_30s_lr1e-4

        if target_task_head and target_task_head not in stem:
            continue

        records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not records:
            continue

        shown += 1
        latest = records[-1]
        n_attempts = sum(1 for r in records if r["status"] in ("STARTED", "REQUEUED"))

        print(f"\n{'─'*72}")
        print(f"  {stem}")
        print(f"  Latest status : {latest['status']}")
        print(f"  Attempts      : {n_attempts}  |  Events: {len(records)}")
        print(f"  Latest job    : {latest['job_id']}  node={latest.get('node', '?')}  ts={latest['ts']}")
        if args.verbose or len(records) > 1:
            print(f"  History:")
            for r in records:
                print(f"    [{r['ts']}] status={r['status']:<20} job={r.get('job_id','?')}  node={r.get('node','?')}")

    if shown == 0:
        if target_id:
            print(f"No tracking records found for '{target_id}'.")
        else:
            print("No tracking records found.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate sbatch/python commands for v2 experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--registry", default=str(REGISTRY_PATH),
                        help="Path to v2_registry.yaml")
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List all experiments and their status")
    p_list.add_argument("--tier", default=None, help="Filter by tier (1 or 2)")

    p_train = sub.add_parser("train", help="Print train sbatch command(s)")
    p_train.add_argument("exp_id", help="Experiment ID from registry")
    p_train.add_argument("--context", nargs="+", default=None,
                         help="Specific context(s) to train (default: all in registry)")
    p_train.add_argument("--time", default=None, dest="override_time",
                         help="Override wall-time for all generated commands, e.g. --time 02:00:00")
    p_train.add_argument("--batch-size", type=int, default=None, dest="override_batch_size",
                         help="Override training batch size, e.g. --batch-size 16 (use for OOM)")

    p_infer = sub.add_parser("infer", help="Print inference sbatch command")
    p_infer.add_argument("exp_id", help="Experiment ID from registry")
    p_infer.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_infer.add_argument("--time", default=None, dest="override_time",
                         help="Override estimated wall-time, e.g. --time 01:30:00")
    p_infer.add_argument("--batch-size", type=int, default=None, dest="override_batch_size",
                         help="Override inference batch size (default: 512), e.g. --batch-size 128")

    p_analyze = sub.add_parser("analyze", help="Print window analysis command")
    p_analyze.add_argument("exp_id", help="Experiment ID from registry")
    p_analyze.add_argument("--plot", action="store_true", help="Include --plot flag")

    p_status = sub.add_parser("status", help="Show file-level status for experiment(s)")
    p_status.add_argument("exp_id", nargs="?", default=None,
                          help="Specific experiment ID (default: all)")

    p_runs = sub.add_parser("runs", help="Show job run history from tracking files")
    p_runs.add_argument("exp_id", nargs="?", default=None,
                        help="Filter by experiment ID (default: all)")
    p_runs.add_argument("-v", "--verbose", action="store_true",
                        help="Always show full history even for single-attempt jobs")

    args = parser.parse_args()
    registry = load_registry()

    dispatch = {
        "list":    cmd_list,
        "train":   cmd_train,
        "infer":   cmd_infer,
        "analyze": cmd_analyze,
        "status":  cmd_status,
        "runs":    cmd_runs,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.command](args, registry)


if __name__ == "__main__":
    main()
