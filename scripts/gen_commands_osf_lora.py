#!/usr/bin/env python3
"""
Experiment command generator for OSF Stage 2 (LoRA) experiments
(v2_osf_lora_registry.yaml).

Forked from gen_commands_osf.py — see
docs/TSFM_OSF_IMPLEMENTATION_PLAN.md checklist item 2.5. Same core
pipeline, status tracking, and command-generation approach as the Stage 1
OSF generator; a separate script rather than a gen_commands_osf.py
retrofit, same reasoning as gen_commands_osf.py itself being separate from
gen_commands.py (no backbone/stage hook in the registry schema).
Differences from gen_commands_osf.py, all deliberate:
  - Targets experiments/v2_osf_lora_registry.yaml, not v2_osf_registry.yaml.
  - Targets jobs/{train,infer}_osf_lora*_gpu.sh (Fir only).
  - HAS gradient-accumulation machinery (`resolve_batch_accum`,
    `ACCUM_STEPS`), same as gen_commands_osf.py — train_osf_lora.py
    gained --accum-steps 2026-08-14 (it already had run_epoch()'s
    accum_steps support imported from train_osf_context_sweep.py; it
    just wasn't wired through). effective_batch=32, same as Stage 1/
    SleepFM, required for comparable training dynamics, not just
    comparable results. No `memory_bounded` batch_mode branch, though —
    dropped along with probe-batch below (no probing infra exists yet).
  - NO `probe-batch` subcommand — no OSF-LoRA experiment uses a
    memory_bounded batch mode and no find_batch_size_osf_lora_gpu.sh
    exists; unlike gen_commands_osf.py this isn't even kept for schema
    parity, since Stage 2 has no batch_mode concept in its registry at all.
  - NO TASK_TYPE env var in generated train/infer commands —
    train_osf_lora.py / infer_osf_lora_subject_windows.py have no
    --task-type flag (seq2label-only for now).
  - Adds an optional --stage1-checkpoint override for `train` (normally
    omitted — train_osf_lora.py auto-detects the matching Stage 1
    checkpoint by task/head/context when none is given).
  - Wall-time lookup tables are placeholder copies of Stage 1's OWN
    (already-placeholder) OSF tables, scaled up by a flat, clearly-
    unverified multiplier — Stage 2 trains a full backbone forward+
    backward pass per raw epoch every step (not a cheap cached-embedding
    lookup like Stage 1), so it is qualitatively far more expensive per
    step, but no real GPU number exists yet to calibrate against (that's
    checklist 2.6). Auto-requeue means an underestimate just costs one
    resubmission, not lost work — same philosophy as Stage 1's own
    placeholder tables.
  - `analyze`/`build-heatmap`/`collect`/`threshold-tuning` subcommands are
    KEPT — they call the same underlying scripts as Stage 1's pipeline
    (analyze_windows.py, build_heatmap_df.py, collect_results_v2.py,
    apply_threshold_tuning.py), confirmed backbone/stage-agnostic already
    by gen_commands_osf.py, just pointed at Stage 2's own results_dir.

── Core pipeline ─────────────────────────────────────────────────────────────
  python scripts/gen_commands_osf_lora.py list [--tier 1|2]
      List all OSF-LoRA experiments with status (pending/trained/inferred/analyzed).

  python scripts/gen_commands_osf_lora.py train <exp_id> [--context 30s 10m ...]
      Print sbatch command(s) for LoRA training. One job per context.

  python scripts/gen_commands_osf_lora.py infer <exp_id> [--split test|val]
      Print the sbatch command for LoRA inference (auto-discovers trained contexts).

  python scripts/gen_commands_osf_lora.py analyze <exp_id> [--plot] [--k-dense] [--bootstrap N]
      Print the python command for window analysis (analyze_windows.py —
      reused unmodified; reads the OSF-LoRA results dir).

  python scripts/gen_commands_osf_lora.py build-heatmap <exp_id> [--split test]
      Print the build_heatmap_df.py command (iso-compute heatmap DataFrame;
      reused unmodified). Run analyze --k-dense first.

  python scripts/gen_commands_osf_lora.py collect [<exp_id> ...]
      Print collect_results_v2.py command to gather all OSF-LoRA results into
      training.csv and analysis.csv (reused unmodified).

  python scripts/gen_commands_osf_lora.py threshold-tuning <exp_id>
      Print apply_threshold_tuning.py command for a binary experiment
      (reused unmodified).

  python scripts/gen_commands_osf_lora.py status [<exp_id>]
      Show detailed file-level status for one or all experiments.

  python scripts/gen_commands_osf_lora.py runs [<exp_id>]
      Show job run history (from logs_osf_lora/status/*.jsonl tracking files).

Examples:
  python scripts/gen_commands_osf_lora.py list --tier 1
  python scripts/gen_commands_osf_lora.py train apnea_binary_lstm
  python scripts/gen_commands_osf_lora.py train apnea_binary_lstm --context 30s
  python scripts/gen_commands_osf_lora.py infer apnea_binary_lstm
  python scripts/gen_commands_osf_lora.py analyze apnea_binary_lstm --plot
  python scripts/gen_commands_osf_lora.py collect apnea_binary_lstm apnea_binary_transformer
  python scripts/gen_commands_osf_lora.py status
  python scripts/gen_commands_osf_lora.py runs apnea_binary_lstm
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).parent.parent / "experiments" / "v2_osf_lora_registry.yaml"
JOBS_DIR = Path(__file__).parent.parent / "jobs"

_TRAIN_SCRIPT = "train_osf_lora_gpu.sh"
_INFER_SCRIPT = "infer_osf_lora_subject_windows_gpu.sh"

# ── Wall-time lookup tables ────────────────────────────────────────────────────
# NOT YET CALIBRATED FOR OSF-LORA — placeholder: Stage 1's own (already-
# placeholder, see gen_commands_osf.py) OSF tables, scaled by a flat 6x
# multiplier. The 6x is an unverified, qualitative guess (full backbone
# forward+backward per raw epoch every step vs. Stage 1's cached-embedding
# lookup), not a measurement — revisit after checklist 2.6's real GPU pilot.
_WALLTIME_LORA_MULTIPLIER = 6

_TRAIN_HOURS_BASE = {
    ("large",  "lstm"):        {"30s": 1.5, "10m": 2,   "40m": 2.5, "80m": 3.5, "120m": 5,   "240m": 5,   "full_night": 6 },
    ("large",  "transformer"): {"30s": 1, "10m": 1.5, "40m": 2,   "80m": 3,   "120m": 5,   "240m": 6,  "full_night": 18},
    ("large",  "mean_pool"):   {"30s": 1,   "10m": 1,   "40m": 1,   "80m": 1,   "120m": 1,   "240m": 1.5, "full_night": 2 },
    ("medium", "lstm"):        {"30s": 1,   "10m": 1.5, "40m": 1.5, "80m": 2,   "120m": 3,   "240m": 4,   "full_night": 3 },
    ("medium", "transformer"): {"30s": 1,   "10m": 1.5, "40m": 2,   "80m": 4,   "120m": 6,   "240m": 5,  "full_night": 18},
    ("medium", "mean_pool"):   {"30s": 1,   "10m": 1,   "40m": 1,   "80m": 1,   "120m": 1,   "240m": 1,   "full_night": 1 },
    ("small",  "lstm"):        {"30s": 1,   "10m": 1,   "40m": 1,   "80m": 1.5, "120m": 2,   "240m": 3,   "full_night": 2 },
    ("small",  "transformer"): {"30s": 1,   "10m": 1,   "40m": 1.5, "80m": 2,   "120m": 3,   "240m": 4,   "full_night": 18},
    ("small",  "mean_pool"):   {"30s": 1,   "10m": 1,   "40m": 1,   "80m": 1,   "120m": 1,   "240m": 1,   "full_night": 1 },
}
_TRAIN_HOURS = {
    key: {ctx: round(h * _WALLTIME_LORA_MULTIPLIER, 2) for ctx, h in table.items()}
    for key, table in _TRAIN_HOURS_BASE.items()
}

# Per-context inference hours (one job runs all contexts sequentially).
# Also NOT YET CALIBRATED — same placeholder-multiplier approach as above.
_INFER_HOURS_PER_CTX_BASE = {
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
_INFER_HOURS_PER_CTX = {
    key: {ctx: round(h * _WALLTIME_LORA_MULTIPLIER, 2) for ctx, h in table.items()}
    for key, table in _INFER_HOURS_PER_CTX_BASE.items()
}


def estimate_train_time(n_size: str, head: str, context: str) -> str:
    hours = _TRAIN_HOURS.get((n_size, head), {}).get(context, 24)
    h = int(hours)
    m = int(round((hours - h) * 60))
    return f"{h:02d}:{m:02d}:00"


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


def load_registry(path=None) -> dict:
    with open(path or REGISTRY_PATH) as f:
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


def _log_stem(exp: dict, step: str, context: str = "", split: str = "") -> str:
    """Build a descriptive log filename stem (no extension, no %j).
    No arch_tag here (unlike gen_commands_osf.py) — no OSF-LoRA experiment
    is seq2seq, and train_osf_lora.py has no seq2seq_padding_policy concept
    at all yet, so there's nothing to disambiguate."""
    tag = exp.get("run_tag", "")
    tag_part = f"_{tag}" if tag else ""
    ctx_part = f"_{context}" if context else ""
    split_part = f"_{split}" if split else ""
    lr_part = f"_lr{format_lr(exp['lr'])}" if step == "train" else ""
    return f"{step}_{exp['task']}_{exp['head']}{tag_part}{ctx_part}{split_part}{lr_part}"


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

def resolve_batch_accum(exp: dict, registry: dict, context: str,
                        override_batch_size: int = None):
    """Return (micro_batch, accum_steps) for this context. Same grad_accum
    logic as gen_commands_osf.py's resolve_batch_accum — the memory_bounded
    branch is dropped (no OSF-LoRA experiment uses it, no probing script
    exists yet, see module docstring)."""
    if override_batch_size is not None:
        return override_batch_size, 1

    ga = registry.get("gradient_accumulation", {})
    if ga.get("enabled", False):
        effective_batch = int(ga.get("effective_batch", 32))
        ctx_map = ga.get("context_micro_batch", {})
        micro_batch = int(ctx_map.get(context, effective_batch))
        accum_steps = max(1, effective_batch // micro_batch)
        return micro_batch, accum_steps
    else:
        return exp["batch_size"], 1


def build_train_cmd(exp: dict, registry: dict, context: str,
                    override_time: str = None, override_batch_size: int = None,
                    stage1_checkpoint: str = None) -> str:
    cfg = exp.get("config") or registry["config"]
    logs_dir = registry.get("logs_dir", str(Path(__file__).parent.parent / "logs_osf_lora"))
    n_size = exp.get("n_size", "large")
    wall_time = override_time if override_time else estimate_train_time(n_size, exp["head"], context)
    stem = _log_stem(exp, "train", context)
    micro_batch, accum_steps = resolve_batch_accum(exp, registry, context, override_batch_size)

    env_vars = [
        f"TASK={exp['task']}",
        f"HEAD={exp['head']}",
        f"CONTEXT={context}",
        f"DATASETS=\"{' '.join(exp['datasets'])}\"",
        f"BATCH_SIZE={micro_batch}",
        f"ACCUM_STEPS={accum_steps}",
        f"LR={format_lr(exp['lr'])}",
    ]
    if exp.get("run_tag"):
        env_vars.append(f"RUN_TAG={exp['run_tag']}")
    if stage1_checkpoint:
        env_vars.append(f"STAGE1_CHECKPOINT={stage1_checkpoint}")
    env_vars.append(f"CONFIG={cfg}")
    env_vars.append(f"LOGS_DIR={logs_dir}")
    env_str = " ".join(env_vars)

    sbatch_opts = (
        f"--requeue "
        f"--time={wall_time} "
        f"--output={logs_dir}/{stem}_%j.out "
        f"--error={logs_dir}/{stem}_%j.err"
    )
    return f"{env_str} sbatch {sbatch_opts} {JOBS_DIR}/{_TRAIN_SCRIPT}"


def build_infer_cmd(exp: dict, registry: dict, split: str = "test",
                    override_time: str = None, override_batch_size: int = None) -> str:
    cfg = exp.get("config") or registry["config"]
    logs_dir = registry.get("logs_dir", str(Path(__file__).parent.parent / "logs_osf_lora"))
    n_size = exp.get("n_size", "large")
    contexts_trained = trained_contexts(exp, registry)
    ctx_list = contexts_trained if contexts_trained else exp["contexts"]
    wall_time = override_time if override_time else estimate_infer_time(n_size, exp["head"], ctx_list)
    stem = _log_stem(exp, "infer", split=split)

    env_vars = [
        f"TASK={exp['task']}",
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
    env_vars.append(f"LOGS_DIR={logs_dir}")
    env_str = " ".join(env_vars)

    sbatch_opts = (
        f"--requeue "
        f"--time={wall_time} "
        f"--output={logs_dir}/{stem}_%j.out "
        f"--error={logs_dir}/{stem}_%j.err"
    )
    return f"{env_str} sbatch {sbatch_opts} {JOBS_DIR}/{_INFER_SCRIPT}"


def build_analyze_cmd(exp: dict, registry: dict, plot: bool = False,
                      k_dense: bool = False,
                      bootstrap_override: int = None) -> str:
    results_dir = Path(registry["results_dir"])
    tag = exp.get("run_tag", "")
    # analyze_windows.py is reused unmodified — confirmed backbone/stage-
    # agnostic already by gen_commands_osf.py.
    python = registry.get("python_bin", "/home/boshra95/osf_env/bin/python")

    if bootstrap_override is not None:
        bootstrap_n = bootstrap_override
    else:
        bootstrap_n = 0
        cfg_path = Path(registry.get("config", ""))
        if cfg_path.exists():
            with open(cfg_path) as _f:
                _cfg = yaml.safe_load(_f)
            bootstrap_n = int(_cfg.get("analysis", {}).get("bootstrap_samples", 0))

    exp_id = exp['task'] + '_' + exp['head'] + (f"_{tag}" if tag else "")
    cmd_parts = [
        f"{python} scripts/analyze_windows.py",
        f"--task {exp['task']}",
        f"--head {exp['head']}",
        f"--results-dir {results_dir}",
        f"--repo-out {repo_inference_dir(results_dir)}/{exp_id}",
    ]
    if tag:
        cmd_parts.append(f"--run-tag {tag}")
    if k_dense:
        cmd_parts.append("--k-dense")
    if bootstrap_n > 0:
        cmd_parts.append(f"--bootstrap {bootstrap_n}")
    if plot:
        cmd_parts.append("--plot")
        cmd_parts.append("--plot-metric auroc balanced_accuracy")
        cmd_parts.append(f"--repo-figures-dir {repo_figures_dir(results_dir)}")
    return " ".join(cmd_parts)


def build_heatmap_cmd(exp: dict, registry: dict, split: str = "test") -> str:
    results_dir = Path(registry["results_dir"])
    tag = exp.get("run_tag", "")
    python = registry.get("python_bin", "/home/boshra95/osf_env/bin/python")
    cmd_parts = [
        f"{python} scripts/build_heatmap_df.py",
        f"--task {exp['task']}",
        f"--head {exp['head']}",
        f"--results-dir {results_dir}",
        f"--split {split}",
    ]
    if tag:
        cmd_parts.append(f"--run-tag {tag}")
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
    stage1_ckpt = getattr(args, "stage1_checkpoint", None)
    ga = registry.get("gradient_accumulation", {})
    ga_enabled = ga.get("enabled", False)
    if ga_enabled:
        batch_label = f"GRAD-ACCUM (effective_batch={ga.get('effective_batch', 32)}, accum varies by context)"
    else:
        batch_label = f"FLAT batch={exp['batch_size']}, accum_steps=1"
    print(f"# LoRA training commands for: {args.exp_id}")
    print(f"# Task: {exp['task']}  Head: {exp['head']}  LR: {exp['lr']}  N-size: {n_size}")
    print(f"# Datasets: {exp['datasets']}")
    print(f"# Batch mode: {batch_label}")
    print(f"# Stage 1 checkpoint: "
          f"{stage1_ckpt or '(auto-detected per context by train_osf_lora.py)'}")
    print(f"# Logs → {registry.get('logs_dir', 'logs_osf_lora/')}")
    print(f"# NOTE: wall-time estimates are placeholder, NOT GPU-calibrated for OSF-LoRA yet")
    print(f"# NOTE: micro-batch=32 is NOT verified to fit on GPU yet — if a run OOMs, "
          f"lower context_micro_batch in the registry and re-generate (accum_steps "
          f"auto-adjusts to keep effective_batch=32)")
    print()
    for ctx in contexts:
        if ctx not in exp["contexts"]:
            print(f"# WARNING: context '{ctx}' not in registry for this experiment — skipping")
            continue
        trained = ctx in trained_contexts(exp, registry)
        wall = estimate_train_time(n_size, exp["head"], ctx)
        micro_batch, accum_steps = resolve_batch_accum(
            exp, registry, ctx,
            override_batch_size=getattr(args, "override_batch_size", None),
        )
        eff_batch = micro_batch * accum_steps
        batch_tag = f"micro={micro_batch} accum={accum_steps} eff={eff_batch}"
        status_tag = "  # already trained" if trained else f"  # est. {wall}  [{batch_tag}]"
        print(build_train_cmd(exp, registry, ctx,
                              override_time=getattr(args, "override_time", None),
                              override_batch_size=getattr(args, "override_batch_size", None),
                              stage1_checkpoint=stage1_ckpt) + status_tag)


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
    print(f"# LoRA inference command for: {args.exp_id}  split={split}")
    print(f"# Trained contexts: {tr or 'none found — check results dir'}")
    print(f"# Est. wall time: {wall}  Logs → {registry.get('logs_dir', 'logs_osf_lora/')}")
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
    plot             = getattr(args, "plot", False)
    k_dense          = getattr(args, "k_dense", False)
    bootstrap_override = getattr(args, "bootstrap", None)
    print(f"# Window analysis command for: {args.exp_id}")
    print()
    print(build_analyze_cmd(exp, registry, plot, k_dense, bootstrap_override))


def cmd_build_heatmap(args, registry):
    experiments = registry["experiments"]
    if args.exp_id not in experiments:
        print(f"ERROR: experiment '{args.exp_id}' not found.", file=sys.stderr)
        sys.exit(1)
    exp   = experiments[args.exp_id]
    split = getattr(args, "split", "test")
    print(f"# Build heatmap DataFrame for iso-compute analysis: {args.exp_id}")
    print(f"# Prerequisite: run 'analyze {args.exp_id} --k-dense' first")
    print()
    print(build_heatmap_cmd(exp, registry, split))


def cmd_threshold_tuning(args, registry):
    """Print the apply_threshold_tuning.py command for a binary experiment."""
    experiments = registry["experiments"]
    if args.exp_id not in experiments:
        print(f"ERROR: experiment '{args.exp_id}' not found.", file=sys.stderr)
        sys.exit(1)
    exp = experiments[args.exp_id]
    if exp.get("num_classes", 2) != 2:
        print(f"# NOTE: {args.exp_id} has {exp.get('num_classes')} classes — "
              "threshold tuning only applies to binary tasks.", file=sys.stderr)
        sys.exit(1)

    python  = registry.get("python_bin", "/home/boshra95/osf_env/bin/python")
    cfg     = registry["config"]
    tag     = exp.get("run_tag", "")
    inf_dir = Path(registry["inference_dir"]) / (
        f"{exp['task']}_{exp['head']}" + (f"_{tag}" if tag else "")
    )
    val_missing = [
        ctx for ctx in trained_contexts(exp, registry)
        if not (inf_dir / f"context_{ctx}" / "val_windows.parquet").exists()
    ]

    print(f"# Threshold tuning for: {args.exp_id}")
    if val_missing:
        print(f"# ⚠ val parquets missing for: {val_missing}")
        print(f"#   Run val inference first (see gen_commands_osf_lora.py infer {args.exp_id} --split val)")
    print()
    results_dir = Path(registry["results_dir"])
    exp_id_full = f"{exp['task']}_{exp['head']}" + (f"_{tag}" if tag else "")
    cmd_parts = [
        f"{python} scripts/apply_threshold_tuning.py",
        f"--config {cfg}",
        f"--task {exp['task']}",
        f"--head {exp['head']}",
        f"--repo-out {repo_inference_dir(results_dir)}/{exp_id_full}",
    ]
    if tag:
        cmd_parts.append(f"--run-tag {tag}")
    print(" ".join(cmd_parts))


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
    logs_dir = Path(registry.get("logs_dir", str(Path(__file__).parent.parent / "logs_osf_lora")))
    status_dir = logs_dir / "status"

    if not status_dir.exists():
        print(f"No status directory found yet ({logs_dir}/status/). No jobs tracked.")
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
        print(f"No job history found in {logs_dir}/status/.")
        return

    shown = 0
    for jsonl_path in files:
        stem = jsonl_path.stem

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


# ── Extended analysis command builders ────────────────────────────────────────
# repo mirrors are round-tagged (results_dir.name, e.g. phase0_osf_lora) so
# different backbones/rounds/stages never overwrite each other in results/.
_REPO_ROOT = Path(__file__).parent.parent / "results"


def repo_figures_dir(results_dir: Path) -> str:
    return str(_REPO_ROOT / "figures" / Path(results_dir).name)


def repo_inference_dir(results_dir: Path) -> str:
    return str(_REPO_ROOT / "inference" / Path(results_dir).name)


def build_collect_cmd(exp_ids: list, registry: dict,
                      collected_dir: str = "") -> str:
    python = registry.get("python_bin", "/home/boshra95/osf_env/bin/python")
    results_dir = Path(registry["results_dir"])
    cdir = collected_dir or str(results_dir / "collected")
    repo_out  = str(_REPO_ROOT / "collected" / results_dir.name)
    cmd_parts = [
        f"{python} scripts/collect_results_v2.py",
        f"--results-dir {results_dir}",
        f"--out-dir {cdir}",
        f"--repo-out {repo_out}",
    ]
    if exp_ids:
        cmd_parts.append(f"--exp-ids {' '.join(exp_ids)}")
    return " ".join(cmd_parts)


def cmd_collect(args, registry):
    exp_ids = getattr(args, "exp_ids", []) or []
    collected_dir = getattr(args, "collected_dir", "")
    results_dir = Path(registry["results_dir"])
    cdir = collected_dir or str(results_dir / "collected")
    print("# Collect all OSF-LoRA experiment results into training.csv and analysis.csv")
    print(f"# Output → {cdir}/")
    print("# Note: bootstrap CIs are computed by analyze_windows.py (run 'analyze --bootstrap N' first)")
    print("# Prerequisite: inference parquets must exist (run infer first)")
    print()
    print(build_collect_cmd(exp_ids, registry, collected_dir))


def main():
    parser = argparse.ArgumentParser(
        description="Generate sbatch/python commands for OSF Stage 2 (LoRA) experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--registry", default=str(REGISTRY_PATH),
                        help="Path to v2_osf_lora_registry.yaml")
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List all experiments and their status")
    p_list.add_argument("--tier", default=None, help="Filter by tier (1 or 2)")

    p_train = sub.add_parser("train", help="Print LoRA train sbatch command(s)")
    p_train.add_argument("exp_id", help="Experiment ID from registry")
    p_train.add_argument("--context", nargs="+", default=None,
                         help="Specific context(s) to train (default: all in registry)")
    p_train.add_argument("--time", default=None, dest="override_time",
                         help="Override wall-time for all generated commands, e.g. --time 02:00:00")
    p_train.add_argument("--batch-size", type=int, default=None, dest="override_batch_size",
                         help="Override training batch size, e.g. --batch-size 8 (use for OOM/underutilization)")
    p_train.add_argument("--stage1-checkpoint", default=None, dest="stage1_checkpoint",
                         help="Override the Stage 1 checkpoint path for ALL generated contexts "
                              "(normally omitted — train_osf_lora.py auto-detects the matching "
                              "checkpoint per context)")

    p_infer = sub.add_parser("infer", help="Print LoRA inference sbatch command")
    p_infer.add_argument("exp_id", help="Experiment ID from registry")
    p_infer.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_infer.add_argument("--time", default=None, dest="override_time",
                         help="Override estimated wall-time, e.g. --time 01:30:00")
    p_infer.add_argument("--batch-size", type=int, default=None, dest="override_batch_size",
                         help="Override inference batch size, e.g. --batch-size 8")

    p_analyze = sub.add_parser("analyze", help="Print window analysis command")
    p_analyze.add_argument("exp_id", help="Experiment ID from registry")
    p_analyze.add_argument("--plot", action="store_true", help="Include --plot flag")
    p_analyze.add_argument("--k-dense", action="store_true", dest="k_dense",
                           help="Include --k-dense flag (~25 K values for iso-compute pipeline)")
    p_analyze.add_argument("--bootstrap", type=int, default=None,
                           help="Override bootstrap_samples from config (0 = off, e.g. --bootstrap 1000)")

    p_bh = sub.add_parser("build-heatmap",
                           help="Print build_heatmap_df.py command (iso-compute heatmap DataFrame)")
    p_bh.add_argument("exp_id", help="Experiment ID from registry")
    p_bh.add_argument("--split", default="test", choices=["train", "val", "test"])

    p_col = sub.add_parser("collect",
                            help="Print collect_results_v2.py command (gather all results)")
    p_col.add_argument("exp_ids", nargs="*", default=[],
                       help="Experiment IDs to collect (default: all in registry)")
    p_col.add_argument("--collected-dir", default="", dest="collected_dir",
                       help="Output directory (default: {results_dir}/collected)")

    p_tt = sub.add_parser("threshold-tuning",
                          help="Print apply_threshold_tuning.py command for a binary experiment")
    p_tt.add_argument("exp_id", help="Experiment ID, e.g. apnea_binary_lstm")

    p_status = sub.add_parser("status", help="Show file-level status for experiment(s)")
    p_status.add_argument("exp_id", nargs="?", default=None,
                          help="Specific experiment ID (default: all)")

    p_runs = sub.add_parser("runs", help="Show job run history from tracking files")
    p_runs.add_argument("exp_id", nargs="?", default=None,
                        help="Filter by experiment ID (default: all)")
    p_runs.add_argument("-v", "--verbose", action="store_true",
                        help="Always show full history even for single-attempt jobs")

    args = parser.parse_args()
    registry = load_registry(args.registry)

    dispatch = {
        "list":                 cmd_list,
        "train":                cmd_train,
        "infer":                cmd_infer,
        "analyze":              cmd_analyze,
        "build-heatmap":        cmd_build_heatmap,
        "collect":              cmd_collect,
        "threshold-tuning":     cmd_threshold_tuning,
        "status":               cmd_status,
        "runs":                 cmd_runs,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.command](args, registry)


if __name__ == "__main__":
    main()
