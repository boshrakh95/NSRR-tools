#!/usr/bin/env python3
"""
Experiment command generator for v2 task-definition experiments.

── Core pipeline ─────────────────────────────────────────────────────────────
  python scripts/gen_commands.py list [--tier 1|2]
      List all experiments with status (pending/trained/inferred/analyzed).

  python scripts/gen_commands.py train <exp_id> [--context 30s 10m ...]
      Print sbatch command(s) for training. One job per context.

  python scripts/gen_commands.py infer <exp_id> [--split test|val]
      Print the sbatch command for inference (auto-discovers trained contexts).

  python scripts/gen_commands.py analyze <exp_id> [--plot] [--k-dense] [--bootstrap N]
      Print the python command for window analysis. Use --k-dense for the
      iso-compute pipeline (~25 K values). --bootstrap N overrides the
      bootstrap_samples value from the config yaml (0 = off).

  python scripts/gen_commands.py status [<exp_id>]
      Show detailed file-level status for one or all experiments.

  python scripts/gen_commands.py runs [<exp_id>]
      Show job run history (from logs_v3/status/*.jsonl tracking files).

── Iso-compute pipeline ───────────────────────────────────────────────────────
  python scripts/gen_commands.py build-heatmap <exp_id> [--split test]
      Print the build_heatmap_df.py command (Step 2). Run analyze --k-dense first.

  python scripts/gen_commands.py iso-plots <exp_id> [--split test] [--metric auroc ...]
      Print the plot_iso_compute.py command (Step 3 — 7 iso-compute plots).

  python scripts/gen_commands.py saturation <task> [--heads lstm ...] [--metric auroc ...]
      Print the plot_saturation.py command (Step 4 — saturation curve across heads).
      Use --collected-dir to add bootstrap CI bands.

── Extended analyses (§1–§9) ─────────────────────────────────────────────────
  python scripts/gen_commands.py collect [<exp_id> ...]
      Print collect_results_v2.py command to gather all results into
      training.csv and analysis.csv. Required before task-comparison and
      scaling-laws subcommands. Note: bootstrap CIs are NOT produced here —
      run 'analyze --bootstrap N' first; collect picks up the CI columns.

  python scripts/gen_commands.py scaling-laws <task> [--heads lstm ...] [--plots 1A 1B 1C]
      Print plot_scaling_laws.py command (§1 — U-shape + FLOPs scaling law).
      Requires: collect first.

  python scripts/gen_commands.py calibration <exp_id> [--split test] [--plots 2A 2B 2C]
      Print plot_calibration.py command (§2 — reliability diagrams + ECE).

  python scripts/gen_commands.py window-position <exp_id> [--split test] [--plots 4A 4B]
      Print plot_window_position.py command (§4 — position-probability profiles).

  python scripts/gen_commands.py subject-consistency <exp_id> [--split test] [--plots 5A 5B 5C]
      Print plot_subject_consistency.py command (§5 — variance + hard subjects).

  python scripts/gen_commands.py task-comparison [--tasks t1 t2 ...] [--head lstm]
      Print plot_task_comparison.py command (§6 — cross-task sensitivity matrix).
      Requires: collect first.

  python scripts/gen_commands.py cohort-saturation <exp_id> [--split test] [--datasets apples shhs mros]
      Print plot_cohort_saturation.py command (§7 — per-cohort saturation curves).

  python scripts/gen_commands.py precision-recall <exp_id> [--split test] [--plots 8A 8B 8C]
      Print plot_precision_recall.py command (§8 — PR curves + vote sweep).

  python scripts/gen_commands.py subject-kstar <exp_id> [--split test] [--kmax 30] [--reps 20]
      Print plot_subject_kstar.py command (§9 — K* distribution + coverage curves).

Examples:
  python scripts/gen_commands.py list --tier 1
  python scripts/gen_commands.py train sex_binary_lstm
  python scripts/gen_commands.py train sex_binary_lstm --context 30s
  python scripts/gen_commands.py infer sex_binary_lstm
  python scripts/gen_commands.py infer sex_binary_lstm --batch-size 64
  python scripts/gen_commands.py analyze sex_binary_lstm --plot
  python scripts/gen_commands.py analyze sex_binary_lstm --k-dense
  python scripts/gen_commands.py analyze sex_binary_lstm --k-dense --bootstrap 1000
  python scripts/gen_commands.py build-heatmap sex_binary_lstm
  python scripts/gen_commands.py iso-plots sex_binary_lstm
  python scripts/gen_commands.py saturation sex_binary --heads lstm transformer mean_pool
  python scripts/gen_commands.py collect sex_binary_lstm sex_binary_transformer
  python scripts/gen_commands.py scaling-laws sex_binary --heads lstm transformer mean_pool
  python scripts/gen_commands.py calibration sex_binary_lstm
  python scripts/gen_commands.py window-position sex_binary_lstm
  python scripts/gen_commands.py subject-consistency sex_binary_lstm
  python scripts/gen_commands.py task-comparison --head lstm
  python scripts/gen_commands.py cohort-saturation sex_binary_lstm
  python scripts/gen_commands.py precision-recall sex_binary_lstm
  python scripts/gen_commands.py subject-kstar sex_binary_lstm --kmax 20
  python scripts/gen_commands.py status
  python scripts/gen_commands.py runs sex_binary_lstm
"""

import argparse
import json
import re
import socket
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).parent.parent / "experiments" / "v2_registry.yaml"
JOBS_DIR = Path(__file__).parent.parent / "jobs"

_ON_RORQUAL = "rorqual" in socket.gethostname()
_TRAIN_SCRIPT = "train_context_sweep_gpu_rorqual.sh" if _ON_RORQUAL else "train_context_sweep_gpu.sh"
_INFER_SCRIPT = "infer_subject_windows_gpu_rorqual.sh" if _ON_RORQUAL else "infer_subject_windows_gpu.sh"

# ── Wall-time lookup tables ────────────────────────────────────────────────────
# Calibrated from sex_binary_lstm (large, N~13k, K=5, batch=32, overlapping-window
# protocol v3) observed on H100:
#   30s=31min/18ep, 10m=46min/17ep, 40m=80min/24ep, 80m=111min/22ep, 120m=185min/25ep
# Estimates = max_epochs(40) × per_epoch × 1.3 safety, rounded to nearest 0.5h.
# Auto-requeue on timeout means an underestimate just triggers one extra job — no
# training state is lost (resume.pt is saved after every epoch).
# Values are fractional hours; estimate_train_time() converts to HH:MM:SS.
_TRAIN_HOURS = {
    ("large",  "lstm"):        {"30s": 1.5, "10m": 2,   "40m": 2.5, "80m": 3.5, "120m": 5,   "240m": 9,   "full_night": 6 },
    ("large",  "transformer"): {"30s": 1.5, "10m": 2.5, "40m": 3,   "80m": 6,   "120m": 9,   "240m": 18,  "full_night": 18},
    ("large",  "mean_pool"):   {"30s": 1,   "10m": 1,   "40m": 1,   "80m": 1,   "120m": 1,   "240m": 1.5, "full_night": 2 },
    ("medium", "lstm"):        {"30s": 1,   "10m": 1.5, "40m": 1.5, "80m": 2,   "120m": 3,   "240m": 6,   "full_night": 3 },
    ("medium", "transformer"): {"30s": 1,   "10m": 1.5, "40m": 2,   "80m": 4,   "120m": 6,   "240m": 12,  "full_night": 18},
    ("medium", "mean_pool"):   {"30s": 1,   "10m": 1,   "40m": 1,   "80m": 1,   "120m": 1,   "240m": 1,   "full_night": 1 },
    ("small",  "lstm"):        {"30s": 1,   "10m": 1,   "40m": 1,   "80m": 1.5, "120m": 2,   "240m": 3,   "full_night": 2 },
    ("small",  "transformer"): {"30s": 1,   "10m": 1,   "40m": 1.5, "80m": 2,   "120m": 3,   "240m": 6,   "full_night": 18},
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


def _log_stem(exp: dict, step: str, context: str = "", split: str = "") -> str:
    """Build a descriptive log filename stem (no extension, no %j).

    Training logs include context and LR (both vary per job).
    Inference logs include split but not LR (LR is irrelevant post-training),
    matching the pattern the SLURM script uses for its persistent .log file so
    that original-submission and timeout-resubmission filenames share the same
    stem prefix.
    """
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
    """Return (micro_batch, accum_steps) for this context.

    Two modes:
      "grad_accum"     (default): reads gradient_accumulation.context_micro_batch[context]
                                  and sets accum_steps so micro × accum = effective_batch.
      "memory_bounded"          : reads memory_bounded.context_batch_size[context] with
                                  accum_steps=1 always (no accumulation).

    Experiment entries opt into mode 2 with `batch_mode: memory_bounded`.
    A per-call override_batch_size disables both modes and uses the provided value.
    """
    if override_batch_size is not None:
        return override_batch_size, 1

    batch_mode = exp.get("batch_mode", "grad_accum")

    if batch_mode == "memory_bounded":
        mb = registry.get("memory_bounded", {})
        ctx_map = mb.get("context_batch_size", {})
        batch_size = int(ctx_map.get(context, exp["batch_size"]))
        return batch_size, 1

    # Default: grad_accum mode (uses gradient_accumulation section when enabled)
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
                    override_time: str = None, override_batch_size: int = None) -> str:
    cfg = registry["config"]
    logs_dir = registry.get("logs_dir", str(Path(__file__).parent.parent / "logs"))
    n_size = exp.get("n_size", "large")
    wall_time = override_time if override_time else estimate_train_time(n_size, exp["head"], context)
    stem = _log_stem(exp, "train", context)
    micro_batch, accum_steps = resolve_batch_accum(exp, registry, context, override_batch_size)

    env_vars = [
        f"TASK={exp['task']}",
        f"TASK_TYPE={exp['task_type']}",
        f"HEAD={exp['head']}",
        f"CONTEXT={context}",
        f"DATASETS=\"{' '.join(exp['datasets'])}\"",
        f"BATCH_SIZE={micro_batch}",
        f"ACCUM_STEPS={accum_steps}",
        f"LR={format_lr(exp['lr'])}",
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
    return f"{env_str} sbatch {sbatch_opts} {JOBS_DIR}/{_TRAIN_SCRIPT}"


def build_infer_cmd(exp: dict, registry: dict, split: str = "test",
                    override_time: str = None, override_batch_size: int = None) -> str:
    cfg = registry["config"]
    logs_dir = registry.get("logs_dir", str(Path(__file__).parent.parent / "logs"))
    n_size = exp.get("n_size", "large")
    contexts_trained = trained_contexts(exp, registry)
    ctx_list = contexts_trained if contexts_trained else exp["contexts"]
    wall_time = override_time if override_time else estimate_infer_time(n_size, exp["head"], ctx_list)
    stem = _log_stem(exp, "infer", split=split)

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
    return f"{env_str} sbatch {sbatch_opts} {JOBS_DIR}/{_INFER_SCRIPT}"


def build_analyze_cmd(exp: dict, registry: dict, plot: bool = False,
                      k_dense: bool = False,
                      bootstrap_override: int = None) -> str:
    results_dir = Path(registry["results_dir"])
    tag = exp.get("run_tag", "")
    # Use the virtualenv Python directly — analyze_windows.py requires sklearn
    # (balanced_accuracy, AUROC, F1) which lives in sleepfm_env, not the system Python.
    python = registry.get("python_bin", "/home/boshra95/sleepfm_env/bin/python")

    if bootstrap_override is not None:
        bootstrap_n = bootstrap_override
    else:
        # Read bootstrap_samples from the experiment config yaml (analysis.bootstrap_samples)
        bootstrap_n = 0
        cfg_path = Path(registry.get("config", ""))
        if cfg_path.exists():
            with open(cfg_path) as _f:
                _cfg = yaml.safe_load(_f)
            bootstrap_n = int(_cfg.get("analysis", {}).get("bootstrap_samples", 0))

    cmd_parts = [
        f"{python} scripts/analyze_windows.py",
        f"--task {exp['task']}",
        f"--head {exp['head']}",
        f"--results-dir {results_dir}",
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
    return " ".join(cmd_parts)


def build_heatmap_cmd(exp: dict, registry: dict, split: str = "test") -> str:
    results_dir = Path(registry["results_dir"])
    tag = exp.get("run_tag", "")
    python = registry.get("python_bin", "/home/boshra95/sleepfm_env/bin/python")
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


def build_iso_plots_cmd(exp: dict, registry: dict, split: str = "test",
                        metrics: list = None, budget: float = 480.0) -> str:
    results_dir = Path(registry["results_dir"])
    tag = exp.get("run_tag", "")
    python = registry.get("python_bin", "/home/boshra95/sleepfm_env/bin/python")
    metrics = metrics or ["auroc", "balanced_accuracy"]
    cmd_parts = [
        f"{python} scripts/plot_iso_compute.py",
        f"--task {exp['task']}",
        f"--head {exp['head']}",
        f"--results-dir {results_dir}",
        f"--split {split}",
        f"--metric {' '.join(metrics)}",
        f"--budget {budget:.0f}",
    ]
    if tag:
        cmd_parts.append(f"--run-tag {tag}")
    return " ".join(cmd_parts)


def build_saturation_cmd(task: str, heads: list, registry: dict,
                         metrics: list = None, split: str = "test",
                         run_tag: str = "") -> str:
    results_dir = Path(registry["results_dir"])
    python = registry.get("python_bin", "/home/boshra95/sleepfm_env/bin/python")
    metrics = metrics or ["auroc", "balanced_accuracy"]
    cmd_parts = [
        f"{python} scripts/plot_saturation.py",
        f"--task {task}",
        f"--heads {' '.join(heads)}",
        f"--results-dir {results_dir}",
        f"--metric {' '.join(metrics)}",
        f"--split {split}",
    ]
    if run_tag:
        cmd_parts.append(f"--run-tag {run_tag}")
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
    ga = registry.get("gradient_accumulation", {})
    ga_enabled = ga.get("enabled", False)
    batch_mode = exp.get("batch_mode", "grad_accum")
    if batch_mode == "memory_bounded":
        batch_label = "MEMORY-BOUNDED (accum_steps=1, batch varies by context)"
    elif ga_enabled:
        batch_label = f"GRAD-ACCUM (effective_batch={ga.get('effective_batch', 32)}, accum varies by context)"
    else:
        batch_label = f"FLAT batch={exp['batch_size']}, accum_steps=1"
    print(f"# Training commands for: {args.exp_id}")
    print(f"# Task: {exp['task']}  Head: {exp['head']}  LR: {exp['lr']}  N-size: {n_size}")
    print(f"# Datasets: {exp['datasets']}")
    print(f"# Batch mode: {batch_label}")
    print(f"# Logs → {registry.get('logs_dir', 'logs/')}")
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


def cmd_iso_plots(args, registry):
    experiments = registry["experiments"]
    if args.exp_id not in experiments:
        print(f"ERROR: experiment '{args.exp_id}' not found.", file=sys.stderr)
        sys.exit(1)
    exp     = experiments[args.exp_id]
    split   = getattr(args, "split", "test")
    metrics = getattr(args, "metric", None) or ["auroc", "balanced_accuracy"]
    budget  = getattr(args, "budget", 480.0)
    print(f"# Iso-compute plots (7 plots per metric) for: {args.exp_id}")
    print(f"# Prerequisite: run 'build-heatmap {args.exp_id}' first")
    print()
    print(build_iso_plots_cmd(exp, registry, split, metrics, budget))


def cmd_saturation(args, registry):
    task         = args.task
    heads        = getattr(args, "heads", ["lstm", "transformer", "mean_pool"])
    metrics      = getattr(args, "metric", None) or ["auroc", "balanced_accuracy"]
    split        = getattr(args, "split", "test")
    run_tag      = getattr(args, "run_tag", "")
    collected_dir = getattr(args, "collected_dir", "")
    print(f"# Saturation curve for task: {task}  heads: {heads}")
    print(f"# Reads {task}_{{head}}/summary.csv — no dense K sweep needed")
    if collected_dir:
        print(f"# CI bands: loading from {collected_dir}/analysis.csv")
    print()
    cmd = build_saturation_cmd(task, heads, registry, metrics, split, run_tag)
    if collected_dir:
        cmd += f" --collected-dir {collected_dir}"
    print(cmd)


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
    logs_dir = Path(registry.get("logs_dir", str(Path(__file__).parent.parent / "logs_v3")))
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


# ── Extended analysis command builders ────────────────────────────────────────

def build_collect_cmd(exp_ids: list, registry: dict,
                      collected_dir: str = "") -> str:
    python = registry.get("python_bin", "/home/boshra95/sleepfm_env/bin/python")
    results_dir = Path(registry["results_dir"])
    cdir = collected_dir or str(results_dir / "collected")
    cmd_parts = [
        f"{python} scripts/collect_results_v2.py",
        f"--results-dir {results_dir}",
        f"--out-dir {cdir}",
    ]
    if exp_ids:
        cmd_parts.append(f"--exp-ids {' '.join(exp_ids)}")
    return " ".join(cmd_parts)


def build_scaling_laws_cmd(task: str, heads: list, registry: dict,
                            collected_dir: str = "",
                            plots: list = None) -> str:
    python = registry.get("python_bin", "/home/boshra95/sleepfm_env/bin/python")
    results_dir = Path(registry["results_dir"])
    cdir = collected_dir or str(results_dir / "collected")
    plots = plots or ["1A", "1B", "1C"]
    cmd_parts = [
        f"{python} scripts/plot_scaling_laws.py",
        f"--task {task}",
        f"--heads {' '.join(heads)}",
        f"--collected-dir {cdir}",
        f"--results-dir {results_dir}",
        f"--plots {' '.join(plots)}",
    ]
    return " ".join(cmd_parts)


def build_calibration_cmd(exp: dict, registry: dict, split: str = "test",
                           heads_extra: list = None, plots: list = None) -> str:
    python = registry.get("python_bin", "/home/boshra95/sleepfm_env/bin/python")
    results_dir = Path(registry["results_dir"])
    tag = exp.get("run_tag", "")
    plots = plots or ["2A", "2B", "2C"]
    cmd_parts = [
        f"{python} scripts/plot_calibration.py",
        f"--task {exp['task']}",
        f"--head {exp['head']}",
        f"--results-dir {results_dir}",
        f"--split {split}",
        f"--plots {' '.join(plots)}",
    ]
    if tag:
        cmd_parts.append(f"--run-tag {tag}")
    if heads_extra:
        cmd_parts.append(f"--heads {' '.join(heads_extra)}")
    return " ".join(cmd_parts)


def build_window_position_cmd(exp: dict, registry: dict, split: str = "test",
                               plots: list = None) -> str:
    python = registry.get("python_bin", "/home/boshra95/sleepfm_env/bin/python")
    results_dir = Path(registry["results_dir"])
    tag = exp.get("run_tag", "")
    plots = plots or ["4A", "4B"]
    cmd_parts = [
        f"{python} scripts/plot_window_position.py",
        f"--task {exp['task']}",
        f"--head {exp['head']}",
        f"--results-dir {results_dir}",
        f"--split {split}",
        f"--plots {' '.join(plots)}",
    ]
    if tag:
        cmd_parts.append(f"--run-tag {tag}")
    return " ".join(cmd_parts)


def build_subject_consistency_cmd(exp: dict, registry: dict, split: str = "test",
                                   plots: list = None) -> str:
    python = registry.get("python_bin", "/home/boshra95/sleepfm_env/bin/python")
    results_dir = Path(registry["results_dir"])
    tag = exp.get("run_tag", "")
    plots = plots or ["5A", "5B", "5C"]
    cmd_parts = [
        f"{python} scripts/plot_subject_consistency.py",
        f"--task {exp['task']}",
        f"--head {exp['head']}",
        f"--results-dir {results_dir}",
        f"--split {split}",
        f"--plots {' '.join(plots)}",
    ]
    if tag:
        cmd_parts.append(f"--run-tag {tag}")
    return " ".join(cmd_parts)


def build_task_comparison_cmd(tasks: list, head: str, split: str,
                               registry: dict, collected_dir: str = "",
                               plots: list = None) -> str:
    python = registry.get("python_bin", "/home/boshra95/sleepfm_env/bin/python")
    results_dir = Path(registry["results_dir"])
    cdir = collected_dir or str(results_dir / "collected")
    plots = plots or ["6A", "6B", "6C"]
    cmd_parts = [
        f"{python} scripts/plot_task_comparison.py",
        f"--head {head}",
        f"--split {split}",
        f"--collected-dir {cdir}",
        f"--results-dir {results_dir}",
        f"--plots {' '.join(plots)}",
    ]
    if tasks:
        cmd_parts.append(f"--tasks {' '.join(tasks)}")
    return " ".join(cmd_parts)


def build_cohort_saturation_cmd(exp: dict, registry: dict, split: str = "test",
                                 datasets: list = None, plots: list = None) -> str:
    python = registry.get("python_bin", "/home/boshra95/sleepfm_env/bin/python")
    results_dir = Path(registry["results_dir"])
    tag = exp.get("run_tag", "")
    datasets = datasets or ["apples", "shhs", "mros"]
    plots = plots or ["7A", "7B"]
    cmd_parts = [
        f"{python} scripts/plot_cohort_saturation.py",
        f"--task {exp['task']}",
        f"--head {exp['head']}",
        f"--results-dir {results_dir}",
        f"--split {split}",
        f"--datasets {' '.join(datasets)}",
        f"--plots {' '.join(plots)}",
    ]
    if tag:
        cmd_parts.append(f"--run-tag {tag}")
    return " ".join(cmd_parts)


def build_precision_recall_cmd(exp: dict, registry: dict, split: str = "test",
                                heads_extra: list = None,
                                plots: list = None) -> str:
    python = registry.get("python_bin", "/home/boshra95/sleepfm_env/bin/python")
    results_dir = Path(registry["results_dir"])
    tag = exp.get("run_tag", "")
    plots = plots or ["8A", "8B", "8C"]
    cmd_parts = [
        f"{python} scripts/plot_precision_recall.py",
        f"--task {exp['task']}",
        f"--head {exp['head']}",
        f"--results-dir {results_dir}",
        f"--split {split}",
        f"--plots {' '.join(plots)}",
    ]
    if tag:
        cmd_parts.append(f"--run-tag {tag}")
    if heads_extra:
        cmd_parts.append(f"--heads {' '.join(heads_extra)}")
    return " ".join(cmd_parts)


def build_subject_kstar_cmd(exp: dict, registry: dict, split: str = "test",
                             k_max: int = 30, reps: int = 20,
                             plots: list = None) -> str:
    python = registry.get("python_bin", "/home/boshra95/sleepfm_env/bin/python")
    results_dir = Path(registry["results_dir"])
    tag = exp.get("run_tag", "")
    plots = plots or ["9A", "9B"]
    cmd_parts = [
        f"{python} scripts/plot_subject_kstar.py",
        f"--task {exp['task']}",
        f"--head {exp['head']}",
        f"--results-dir {results_dir}",
        f"--split {split}",
        f"--kmax {k_max}",
        f"--reps {reps}",
        f"--plots {' '.join(plots)}",
    ]
    if tag:
        cmd_parts.append(f"--run-tag {tag}")
    return " ".join(cmd_parts)


# ── Extended analysis handlers ─────────────────────────────────────────────────

def cmd_collect(args, registry):
    exp_ids = getattr(args, "exp_ids", []) or []
    collected_dir = getattr(args, "collected_dir", "")
    results_dir = Path(registry["results_dir"])
    cdir = collected_dir or str(results_dir / "collected")
    print("# Collect all experiment results into training.csv and analysis.csv")
    print(f"# Output → {cdir}/")
    print("# Note: bootstrap CIs are computed by analyze_windows.py (run 'analyze --bootstrap N' first)")
    print("# Prerequisite: inference parquets must exist (run infer first)")
    print()
    print(build_collect_cmd(exp_ids, registry, collected_dir))


def cmd_scaling_laws(args, registry):
    task    = args.task
    heads   = getattr(args, "heads", ["lstm", "transformer", "mean_pool"])
    plots   = getattr(args, "plots", ["1A", "1B", "1C"])
    cdir    = getattr(args, "collected_dir", "")
    results_dir = Path(registry["results_dir"])
    cdir_display = cdir or str(results_dir / "collected")
    print(f"# Scaling law plots (§1) for task: {task}  heads: {heads}")
    print(f"# Reads training.csv from: {cdir_display}/")
    print("# Prerequisite: run 'collect' first")
    print()
    print(build_scaling_laws_cmd(task, heads, registry, cdir, plots))


def cmd_calibration(args, registry):
    experiments = registry["experiments"]
    if args.exp_id not in experiments:
        print(f"ERROR: experiment '{args.exp_id}' not found.", file=sys.stderr)
        sys.exit(1)
    exp   = experiments[args.exp_id]
    split = getattr(args, "split", "test")
    plots = getattr(args, "plots", ["2A", "2B", "2C"])
    heads_extra = getattr(args, "heads", None)
    print(f"# Calibration plots (§2) for: {args.exp_id}  split={split}")
    print("# Prerequisite: inference parquets must exist")
    print()
    print(build_calibration_cmd(exp, registry, split, heads_extra, plots))


def cmd_window_position(args, registry):
    experiments = registry["experiments"]
    if args.exp_id not in experiments:
        print(f"ERROR: experiment '{args.exp_id}' not found.", file=sys.stderr)
        sys.exit(1)
    exp   = experiments[args.exp_id]
    split = getattr(args, "split", "test")
    plots = getattr(args, "plots", ["4A", "4B"])
    print(f"# Window position plots (§4) for: {args.exp_id}  split={split}")
    print("# Prerequisite: inference parquets must exist")
    print()
    print(build_window_position_cmd(exp, registry, split, plots))


def cmd_subject_consistency(args, registry):
    experiments = registry["experiments"]
    if args.exp_id not in experiments:
        print(f"ERROR: experiment '{args.exp_id}' not found.", file=sys.stderr)
        sys.exit(1)
    exp   = experiments[args.exp_id]
    split = getattr(args, "split", "test")
    plots = getattr(args, "plots", ["5A", "5B", "5C"])
    print(f"# Subject consistency plots (§5) for: {args.exp_id}  split={split}")
    print("# Prerequisite: inference parquets must exist")
    print()
    print(build_subject_consistency_cmd(exp, registry, split, plots))


def cmd_task_comparison(args, registry):
    tasks   = getattr(args, "tasks", None) or []
    head    = getattr(args, "head", "lstm")
    split   = getattr(args, "split", "test")
    plots   = getattr(args, "plots", ["6A", "6B", "6C"])
    cdir    = getattr(args, "collected_dir", "")
    results_dir = Path(registry["results_dir"])
    cdir_display = cdir or str(results_dir / "collected")
    print(f"# Task comparison plots (§6)  head={head}  split={split}")
    print(f"# Reads analysis.csv from: {cdir_display}/")
    print("# Prerequisite: run 'collect' for all tasks first")
    print()
    print(build_task_comparison_cmd(tasks, head, split, registry, cdir, plots))


def cmd_cohort_saturation(args, registry):
    experiments = registry["experiments"]
    if args.exp_id not in experiments:
        print(f"ERROR: experiment '{args.exp_id}' not found.", file=sys.stderr)
        sys.exit(1)
    exp      = experiments[args.exp_id]
    split    = getattr(args, "split", "test")
    datasets = getattr(args, "datasets", ["apples", "shhs", "mros"])
    plots    = getattr(args, "plots", ["7A", "7B"])
    print(f"# Cohort-stratified saturation plots (§7) for: {args.exp_id}  split={split}")
    print(f"# Datasets: {datasets}")
    print("# Prerequisite: inference parquets must exist and include 'dataset' column")
    print()
    print(build_cohort_saturation_cmd(exp, registry, split, datasets, plots))


def cmd_precision_recall(args, registry):
    experiments = registry["experiments"]
    if args.exp_id not in experiments:
        print(f"ERROR: experiment '{args.exp_id}' not found.", file=sys.stderr)
        sys.exit(1)
    exp   = experiments[args.exp_id]
    split = getattr(args, "split", "test")
    plots = getattr(args, "plots", ["8A", "8B", "8C"])
    heads_extra = getattr(args, "heads", None)
    print(f"# Precision-recall plots (§8) for: {args.exp_id}  split={split}")
    print("# Prerequisite: inference parquets must exist")
    print()
    print(build_precision_recall_cmd(exp, registry, split, heads_extra, plots))


def cmd_subject_kstar(args, registry):
    experiments = registry["experiments"]
    if args.exp_id not in experiments:
        print(f"ERROR: experiment '{args.exp_id}' not found.", file=sys.stderr)
        sys.exit(1)
    exp   = experiments[args.exp_id]
    split = getattr(args, "split", "test")
    k_max = getattr(args, "kmax", 30)
    reps  = getattr(args, "reps", 20)
    plots = getattr(args, "plots", ["9A", "9B"])
    print(f"# Subject K* plots (§9) for: {args.exp_id}  split={split}")
    print(f"# kmax={k_max}  reps={reps}  (note: may be slow for large datasets)")
    print("# Prerequisite: inference parquets must exist")
    print()
    print(build_subject_kstar_cmd(exp, registry, split, k_max, reps, plots))


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
    p_analyze.add_argument("--k-dense", action="store_true", dest="k_dense",
                           help="Include --k-dense flag (~25 K values for iso-compute pipeline)")
    p_analyze.add_argument("--bootstrap", type=int, default=None,
                           help="Override bootstrap_samples from config (0 = off, e.g. --bootstrap 1000)")

    p_bh = sub.add_parser("build-heatmap",
                           help="Print build_heatmap_df.py command (Step 2)")
    p_bh.add_argument("exp_id", help="Experiment ID from registry")
    p_bh.add_argument("--split", default="test", choices=["train", "val", "test"])

    p_iso = sub.add_parser("iso-plots",
                            help="Print plot_iso_compute.py command (Step 3 — 7 plots)")
    p_iso.add_argument("exp_id", help="Experiment ID from registry")
    p_iso.add_argument("--split",  default="test", choices=["train", "val", "test"])
    p_iso.add_argument("--metric", nargs="+", default=["auroc", "balanced_accuracy"],
                       help="Metric columns to plot (default: auroc balanced_accuracy)")
    p_iso.add_argument("--budget", type=float, default=480.0,
                       help="Max compute budget in minutes (default: 480)")

    p_sat = sub.add_parser("saturation",
                            help="Print plot_saturation.py command (Step 4 — head comparison)")
    p_sat.add_argument("task", help="Task name, e.g. sex_binary")
    p_sat.add_argument("--heads", nargs="+",
                       default=["lstm", "transformer", "mean_pool"],
                       help="Heads to compare (default: lstm transformer mean_pool)")
    p_sat.add_argument("--metric", nargs="+", default=["auroc", "balanced_accuracy"],
                       help="Metrics to plot (default: auroc balanced_accuracy)")
    p_sat.add_argument("--split", default="test", choices=["val", "test"])
    p_sat.add_argument("--run-tag", default="", dest="run_tag")
    p_sat.add_argument("--collected-dir", default="", dest="collected_dir",
                       help="Dir with analysis.csv (adds bootstrap CI bands if present)")

    # ── Extended analysis subcommands ──────────────────────────────────────────

    p_col = sub.add_parser("collect",
                            help="Print collect_results_v2.py command (gather all results)")
    p_col.add_argument("exp_ids", nargs="*", default=[],
                       help="Experiment IDs to collect (default: all in registry)")
    p_col.add_argument("--collected-dir", default="", dest="collected_dir",
                       help="Output directory (default: {results_dir}/collected)")

    p_sl = sub.add_parser("scaling-laws",
                           help="Print plot_scaling_laws.py command (§1 U-shape + scaling law)")
    p_sl.add_argument("task", help="Task name, e.g. sex_binary")
    p_sl.add_argument("--heads", nargs="+",
                      default=["lstm", "transformer", "mean_pool"])
    p_sl.add_argument("--plots", nargs="+", default=["1A", "1B", "1C"])
    p_sl.add_argument("--collected-dir", default="", dest="collected_dir")

    p_cal = sub.add_parser("calibration",
                            help="Print plot_calibration.py command (§2 reliability + ECE)")
    p_cal.add_argument("exp_id", help="Experiment ID from registry")
    p_cal.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_cal.add_argument("--heads", nargs="+", default=None,
                       help="Extra heads for 2B multi-head ECE comparison")
    p_cal.add_argument("--plots", nargs="+", default=["2A", "2B", "2C"])

    p_wp = sub.add_parser("window-position",
                           help="Print plot_window_position.py command (§4 position profiles)")
    p_wp.add_argument("exp_id", help="Experiment ID from registry")
    p_wp.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_wp.add_argument("--plots", nargs="+", default=["4A", "4B"])

    p_sc = sub.add_parser("subject-consistency",
                           help="Print plot_subject_consistency.py command (§5 variance + hard subjects)")
    p_sc.add_argument("exp_id", help="Experiment ID from registry")
    p_sc.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_sc.add_argument("--plots", nargs="+", default=["5A", "5B", "5C"])

    p_tc = sub.add_parser("task-comparison",
                           help="Print plot_task_comparison.py command (§6 sensitivity matrix)")
    p_tc.add_argument("--tasks", nargs="+", default=None,
                      help="Task names to include (default: all in analysis.csv)")
    p_tc.add_argument("--head", default="lstm")
    p_tc.add_argument("--split", default="test", choices=["val", "test"])
    p_tc.add_argument("--plots", nargs="+", default=["6A", "6B", "6C"])
    p_tc.add_argument("--collected-dir", default="", dest="collected_dir")

    p_cs = sub.add_parser("cohort-saturation",
                           help="Print plot_cohort_saturation.py command (§7 per-dataset saturation)")
    p_cs.add_argument("exp_id", help="Experiment ID from registry")
    p_cs.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_cs.add_argument("--datasets", nargs="+", default=["apples", "shhs", "mros"])
    p_cs.add_argument("--plots", nargs="+", default=["7A", "7B"])

    p_pr = sub.add_parser("precision-recall",
                           help="Print plot_precision_recall.py command (§8 PR curves + vote sweep)")
    p_pr.add_argument("exp_id", help="Experiment ID from registry")
    p_pr.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_pr.add_argument("--heads", nargs="+", default=None,
                      help="Extra heads for 8B AUC-PR vs context comparison")
    p_pr.add_argument("--plots", nargs="+", default=["8A", "8B", "8C"])

    p_ks = sub.add_parser("subject-kstar",
                           help="Print plot_subject_kstar.py command (§9 K* distribution + coverage)")
    p_ks.add_argument("exp_id", help="Experiment ID from registry")
    p_ks.add_argument("--split", default="test", choices=["train", "val", "test"])
    p_ks.add_argument("--kmax", type=int, default=30,
                      help="Maximum K to evaluate (default: 30)")
    p_ks.add_argument("--reps", type=int, default=20,
                      help="Random draws per (subject, k) (default: 20)")
    p_ks.add_argument("--plots", nargs="+", default=["9A", "9B"])

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
        "list":                 cmd_list,
        "train":                cmd_train,
        "infer":                cmd_infer,
        "analyze":              cmd_analyze,
        "build-heatmap":        cmd_build_heatmap,
        "iso-plots":            cmd_iso_plots,
        "saturation":           cmd_saturation,
        "collect":              cmd_collect,
        "scaling-laws":         cmd_scaling_laws,
        "calibration":          cmd_calibration,
        "window-position":      cmd_window_position,
        "subject-consistency":  cmd_subject_consistency,
        "task-comparison":      cmd_task_comparison,
        "cohort-saturation":    cmd_cohort_saturation,
        "precision-recall":     cmd_precision_recall,
        "subject-kstar":        cmd_subject_kstar,
        "status":               cmd_status,
        "runs":                 cmd_runs,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.command](args, registry)


if __name__ == "__main__":
    main()
