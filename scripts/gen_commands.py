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

Examples:
  python scripts/gen_commands.py list --tier 1
  python scripts/gen_commands.py train sex_binary_lstm
  python scripts/gen_commands.py train sex_binary_lstm --context 30s
  python scripts/gen_commands.py infer sex_binary_lstm
  python scripts/gen_commands.py infer sex_binary_lstm --split val
  python scripts/gen_commands.py analyze sex_binary_lstm --plot
  python scripts/gen_commands.py status
"""

import argparse
import sys
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).parent.parent / "experiments" / "v2_registry.yaml"
JOBS_DIR = Path(__file__).parent.parent / "jobs"


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


# ── Status checks ─────────────────────────────────────────────────────────────

def trained_contexts(exp: dict, registry: dict) -> list[str]:
    folder = exp_folder(exp, registry)
    done = []
    for ctx in exp["contexts"]:
        ckpt = folder / context_dir_name(ctx) / "best_model.pt"
        if ckpt.exists():
            done.append(ctx)
    return done


def inferred_contexts(exp: dict, registry: dict, split: str = "test") -> list[str]:
    folder = infer_folder(exp, registry)
    done = []
    for ctx in exp["contexts"]:
        parquet = folder / context_dir_name(ctx) / f"{split}_windows.parquet"
        if parquet.exists():
            done.append(ctx)
    return done


def is_analyzed(exp: dict, registry: dict) -> bool:
    folder = infer_folder(exp, registry)
    return (folder / "window_analysis.md").exists()


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

def build_train_cmd(exp: dict, registry: dict, context: str) -> str:
    cfg = registry["config"]
    env_vars = [
        f"TASK={exp['task']}",
        f"TASK_TYPE={exp['task_type']}",
        f"HEAD={exp['head']}",
        f"CONTEXT={context}",
        f"DATASETS=\"{' '.join(exp['datasets'])}\"",
        f"BATCH_SIZE={exp['batch_size']}",
        f"LR={exp['lr']}",
    ]
    if exp.get("run_tag"):
        env_vars.append(f"RUN_TAG={exp['run_tag']}")
    env_vars.append(f"CONFIG={cfg}")
    env_str = " ".join(env_vars)
    return f"{env_str} sbatch {JOBS_DIR}/train_context_sweep_gpu.sh"


def build_infer_cmd(exp: dict, registry: dict, split: str = "test") -> str:
    cfg = registry["config"]
    contexts_trained = trained_contexts(exp, registry)
    ctx_list = contexts_trained if contexts_trained else exp["contexts"]

    env_vars = [
        f"TASK={exp['task']}",
        f"TASK_TYPE={exp['task_type']}",
        f"HEAD={exp['head']}",
        f"CONTEXTS=\"{' '.join(ctx_list)}\"",
        f"SPLIT={split}",
        f"DATASETS=\"{' '.join(exp['datasets'])}\"",
    ]
    if exp.get("run_tag"):
        env_vars.append(f"RUN_TAG={exp['run_tag']}")
    env_vars.append(f"CONFIG={cfg}")
    env_str = " ".join(env_vars)
    return f"{env_str} sbatch {JOBS_DIR}/infer_subject_windows_gpu.sh"


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
    print(f"{'ID':<40} {'Tier':<6} {'N-ctx':<6} {'Datasets':<30} {'Status'}")
    print("-" * 110)
    for exp_id, exp in experiments.items():
        tier = exp.get("tier", "?")
        if tier_filter and str(tier) != str(tier_filter):
            continue
        datasets_str = ",".join(exp["datasets"])
        status = exp_status(exp, registry)
        print(f"{exp_id:<40} {str(tier):<6} {len(exp['contexts']):<6} {datasets_str:<30} {status}")


def cmd_train(args, registry):
    experiments = registry["experiments"]
    if args.exp_id not in experiments:
        print(f"ERROR: experiment '{args.exp_id}' not found in registry.", file=sys.stderr)
        sys.exit(1)
    exp = experiments[args.exp_id]
    contexts = args.context if args.context else exp["contexts"]
    print(f"# Training commands for: {args.exp_id}")
    print(f"# Task: {exp['task']}  Head: {exp['head']}  LR: {exp['lr']}")
    print(f"# Datasets: {exp['datasets']}")
    print()
    for ctx in contexts:
        if ctx not in exp["contexts"]:
            print(f"# WARNING: context '{ctx}' not in registry for this experiment — skipping")
            continue
        trained = ctx in trained_contexts(exp, registry)
        status_tag = "  # already trained" if trained else ""
        print(build_train_cmd(exp, registry, ctx) + status_tag)


def cmd_infer(args, registry):
    experiments = registry["experiments"]
    if args.exp_id not in experiments:
        print(f"ERROR: experiment '{args.exp_id}' not found.", file=sys.stderr)
        sys.exit(1)
    exp = experiments[args.exp_id]
    split = getattr(args, "split", "test")
    tr = trained_contexts(exp, registry)
    if not tr:
        print(f"# WARNING: no trained contexts found for '{args.exp_id}'. Command uses all contexts from registry.")
    print(f"# Inference command for: {args.exp_id}  split={split}")
    print(f"# Trained contexts: {tr or 'none found — check results dir'}")
    print()
    print(build_infer_cmd(exp, registry, split))


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

    p_infer = sub.add_parser("infer", help="Print inference sbatch command")
    p_infer.add_argument("exp_id", help="Experiment ID from registry")
    p_infer.add_argument("--split", default="test", choices=["train", "val", "test"])

    p_analyze = sub.add_parser("analyze", help="Print window analysis command")
    p_analyze.add_argument("exp_id", help="Experiment ID from registry")
    p_analyze.add_argument("--plot", action="store_true", help="Include --plot flag")

    p_status = sub.add_parser("status", help="Show file-level status for experiment(s)")
    p_status.add_argument("exp_id", nargs="?", default=None,
                          help="Specific experiment ID (default: all)")

    args = parser.parse_args()
    registry = load_registry()

    dispatch = {
        "list":    cmd_list,
        "train":   cmd_train,
        "infer":   cmd_infer,
        "analyze": cmd_analyze,
        "status":  cmd_status,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.command](args, registry)


if __name__ == "__main__":
    main()
