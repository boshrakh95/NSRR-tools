#!/usr/bin/env python3
"""
Experiment command generator for OSF baseline experiments (v2_osf_registry.yaml).

Forked from gen_commands.py — see docs/TSFM_OSF_IMPLEMENTATION_PLAN.md
checklist item 1.7. Same core pipeline, status tracking, and command-
generation approach as the SleepFM generator; a separate script rather than
a gen_commands.py retrofit per the "Code reuse assessment" decision in
CLAUDE.md (no backbone hook in the original registry schema/wall-time
tables). Differences from gen_commands.py, all deliberate:
  - Targets experiments/v2_osf_registry.yaml, not v2_registry.yaml.
  - Targets jobs/{train,infer}_osf_*_gpu.sh (Fir only — no rorqual variant
    exists yet for OSF; unlike gen_commands.py there is no rorqual branch).
  - No --zero-modalities / ZERO_MODALITIES support (OSF has no 4-modality-
    group structure to ablate — dropped from train_osf_context_sweep.py /
    infer_osf_subject_windows.py themselves, see checklist 1.5/1.6).
  - python_bin fallback defaults to osf_env, not sleepfm_env.
  - Figure/table subcommands (iso-plots, saturation, scaling-laws,
    calibration, window-position, subject-consistency, task-comparison,
    cohort-saturation, precision-recall, subject-kstar, table-1..table-10)
    are NOT ported — per CLAUDE.md, SleepFM's own plot_*.py/make_table*.py
    scripts are already superseded by notebooks (results/paper_figures/
    notebooks_npj/) for the current paper, so there's no reason to build a
    second, parallel figure-generation code path for OSF. Once OSF results
    exist, feed them into notebooks the same way SleepFM's are fed in.
  - Wall-time lookup tables are placeholder copies of SleepFM's (NOT YET
    CALIBRATED for OSF — no real GPU sweep has run yet). Revisit after the
    first real OSF sweep (checklist item 1.10); until then, generated
    --time values are rough starting points, and auto-requeue means an
    underestimate just costs one extra resubmission, not lost work.

── Core pipeline ─────────────────────────────────────────────────────────────
  python scripts/gen_commands_osf.py list [--tier 1|2]
      List all OSF experiments with status (pending/trained/inferred/analyzed).

  python scripts/gen_commands_osf.py probe-batch <exp_id> [--starting-batch-size 256]
      Print sbatch command to probe the max GPU batch size (memory_bounded
      experiments only). No OSF experiment currently uses batch_mode:
      memory_bounded, and jobs/find_batch_size_osf_gpu.sh does not exist yet
      — this subcommand is kept for schema parity / future use only.

  python scripts/gen_commands_osf.py train <exp_id> [--context 30s 10m ...]
      Print sbatch command(s) for training. One job per context.

  python scripts/gen_commands_osf.py infer <exp_id> [--split test|val]
      Print the sbatch command for inference (auto-discovers trained contexts).

  python scripts/gen_commands_osf.py analyze <exp_id> [--plot] [--k-dense] [--bootstrap N]
      Print the python command for window analysis (analyze_windows.py —
      backbone-agnostic, reused unmodified; reads the OSF results dir).

  python scripts/gen_commands_osf.py build-heatmap <exp_id> [--split test]
      Print the build_heatmap_df.py command (iso-compute heatmap DataFrame;
      reused unmodified). Run analyze --k-dense first.

  python scripts/gen_commands_osf.py collect [<exp_id> ...]
      Print collect_results_v2.py command to gather all OSF results into
      training.csv and analysis.csv (reused unmodified).

  python scripts/gen_commands_osf.py threshold-tuning <exp_id>
      Print apply_threshold_tuning.py command for a binary experiment
      (reused unmodified).

  python scripts/gen_commands_osf.py status [<exp_id>]
      Show detailed file-level status for one or all experiments.

  python scripts/gen_commands_osf.py runs [<exp_id>]
      Show job run history (from logs_osf/status/*.jsonl tracking files).

Examples:
  python scripts/gen_commands_osf.py list --tier 1
  python scripts/gen_commands_osf.py train apnea_binary_lstm
  python scripts/gen_commands_osf.py train apnea_binary_lstm --context 30s
  python scripts/gen_commands_osf.py infer apnea_binary_lstm
  python scripts/gen_commands_osf.py analyze apnea_binary_lstm --plot
  python scripts/gen_commands_osf.py collect apnea_binary_lstm apnea_binary_transformer
  python scripts/gen_commands_osf.py status
  python scripts/gen_commands_osf.py runs apnea_binary_lstm
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

REGISTRY_PATH = Path(__file__).parent.parent / "experiments" / "v2_osf_registry.yaml"
JOBS_DIR = Path(__file__).parent.parent / "jobs"

_TRAIN_SCRIPT = "train_osf_context_sweep_gpu.sh"
_INFER_SCRIPT = "infer_osf_subject_windows_gpu.sh"
_PROBE_BATCH_SCRIPT = "find_batch_size_osf_gpu.sh"  # NOT YET IMPLEMENTED — see probe-batch docstring above

# ── Wall-time lookup tables ────────────────────────────────────────────────────
# NOT YET CALIBRATED FOR OSF — placeholder copy of SleepFM's tables
# (scripts/gen_commands.py), which were calibrated from sex_binary_lstm on
# H100. OSF's per-epoch cost profile (12-channel ViT forward pass over
# 1536-dim flattened embeddings vs SleepFM's 512-dim) has not been measured
# on GPU as of 2026-08-11 — only CPU-debugged with --batch-size 2. Revisit
# these numbers after the first real GPU sweep (checklist item 1.10).
# Auto-requeue on timeout means an underestimate just triggers one extra
# job — no training state is lost (resume.pt is saved after every epoch).
_TRAIN_HOURS = {
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

# Per-context inference hours (one job runs all contexts sequentially).
# Also NOT YET CALIBRATED — see note above.
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


_cfg_arch_cache: dict = {}


def _cfg_arch_tag(exp: dict, registry: dict) -> str:
    """Return a short arch+padding tag for seq2seq (sleep staging) experiments only.

    No OSF registry entry is seq2seq yet (sleep_staging is deferred, see
    module docstring), but this is kept identical to gen_commands.py so
    _log_stem() behaves the same way if/when it's added.
    """
    if exp.get("task_type") != "seq2seq":
        return ""
    cfg_path = str(exp.get("config") or registry.get("config", ""))
    if not cfg_path:
        return ""
    if cfg_path not in _cfg_arch_cache:
        try:
            with open(cfg_path) as _f:
                cfg = yaml.safe_load(_f)
            hidden = cfg.get("model", {}).get("hidden_dim", "")
            layers = cfg.get("model", {}).get("num_layers", "")
            policy = cfg.get("dataset", {}).get("seq2seq_padding_policy", "")
            policy_short = {"complete_only": "conly", "allow_all": "allw"}.get(policy, "")
            arch = f"h{hidden}l{layers}" if (hidden and layers) else ""
            _cfg_arch_cache[cfg_path] = f"{arch}_{policy_short}" if (arch and policy_short) else arch
        except Exception:
            _cfg_arch_cache[cfg_path] = ""
    return _cfg_arch_cache[cfg_path]


def _log_stem(exp: dict, step: str, context: str = "", split: str = "",
              registry: dict = None) -> str:
    """Build a descriptive log filename stem (no extension, no %j)."""
    tag = exp.get("run_tag", "")
    tag_part = f"_{tag}" if tag else ""
    arch_tag  = _cfg_arch_tag(exp, registry) if registry else ""
    arch_part = f"_{arch_tag}" if arch_tag else ""
    ctx_part = f"_{context}" if context else ""
    split_part = f"_{split}" if split else ""
    lr_part = f"_lr{format_lr(exp['lr'])}" if step == "train" else ""
    return f"{step}_{exp['task']}_{exp['head']}{tag_part}{arch_part}{ctx_part}{split_part}{lr_part}"


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
    """Return (micro_batch, accum_steps) for this context. Identical logic
    to gen_commands.py's resolve_batch_accum — see that file for the two
    batch_mode semantics (grad_accum vs memory_bounded)."""
    if override_batch_size is not None:
        return override_batch_size, 1

    batch_mode = exp.get("batch_mode", "grad_accum")

    if batch_mode == "memory_bounded":
        bsz_file = exp_folder(exp, registry) / "batch_sizes.json"
        if bsz_file.exists():
            data = json.loads(bsz_file.read_text())
            if context in data:
                entry = data[context]
                batch_size = int(entry["safe"] if isinstance(entry, dict) else entry)
                return batch_size, 1
        mb = registry.get("memory_bounded", {})
        ctx_map = mb.get("context_batch_size", {})
        batch_size = int(ctx_map.get(context, exp["batch_size"]))
        return batch_size, 1

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
    cfg = exp.get("config") or registry["config"]
    logs_dir = registry.get("logs_dir", str(Path(__file__).parent.parent / "logs_osf"))
    n_size = exp.get("n_size", "large")
    wall_time = override_time if override_time else estimate_train_time(n_size, exp["head"], context)
    stem = _log_stem(exp, "train", context, registry=registry)
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
    logs_dir = registry.get("logs_dir", str(Path(__file__).parent.parent / "logs_osf"))
    n_size = exp.get("n_size", "large")
    contexts_trained = trained_contexts(exp, registry)
    ctx_list = contexts_trained if contexts_trained else exp["contexts"]
    wall_time = override_time if override_time else estimate_infer_time(n_size, exp["head"], ctx_list)
    stem = _log_stem(exp, "infer", split=split, registry=registry)

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
    # analyze_windows.py is reused unmodified — requires sklearn, which
    # lives in osf_env for this registry (osf_env has its own sklearn install,
    # same as sleepfm_env does for the SleepFM pipeline).
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


def build_probe_batch_cmd(exp: dict, registry: dict,
                          starting_batch_size: int = 256) -> str:
    """Generate the sbatch command to run find_batch_size.py for a
    memory-bounded exp. NOTE: jobs/find_batch_size_osf_gpu.sh does not
    exist yet — no current OSF experiment uses batch_mode: memory_bounded.
    Kept for schema parity; create that job script before this is used."""
    cfg      = registry["config"]
    logs_dir = registry.get("logs_dir", str(Path(__file__).parent.parent / "logs_osf"))
    out_dir  = exp_folder(exp, registry)
    head     = exp["head"]
    ctxs     = " ".join(exp["contexts"])
    num_cls  = exp.get("num_classes", 2)
    tag      = exp.get("run_tag", "")
    tag_part = f"_{tag}" if tag else ""
    stem     = f"probe_batch_{exp['task']}_{head}{tag_part}"

    env_vars = [
        f"HEAD={head}",
        f"EXP_DIR={out_dir}",
        f"CONFIG={cfg}",
        f"CONTEXTS=\"{ctxs}\"",
        f"NUM_CLASSES={num_cls}",
        f"STARTING_BATCH_SIZE={starting_batch_size}",
    ]
    env_str = " ".join(env_vars)
    sbatch_opts = (
        f"--time=00:20:00 "
        f"--output={logs_dir}/{stem}_%j.out "
        f"--error={logs_dir}/{stem}_%j.err"
    )
    return f"{env_str} sbatch {sbatch_opts} {JOBS_DIR}/{_PROBE_BATCH_SCRIPT}"


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


def cmd_probe_batch(args, registry):
    experiments = registry["experiments"]
    if args.exp_id not in experiments:
        print(f"ERROR: experiment '{args.exp_id}' not found in registry.", file=sys.stderr)
        sys.exit(1)
    exp = experiments[args.exp_id]
    if exp.get("batch_mode", "grad_accum") != "memory_bounded":
        print(
            f"WARNING: '{args.exp_id}' is not memory_bounded "
            f"(batch_mode={exp.get('batch_mode', 'grad_accum')}). "
            "probe-batch is only meaningful for memory_bounded experiments.",
            file=sys.stderr,
        )
    bsz_file = exp_folder(exp, registry) / "batch_sizes.json"
    probed_note = f"  # already probed: {bsz_file}" if bsz_file.exists() else ""
    starting = getattr(args, "starting_batch_size", 256)
    print(f"# Batch size probe for: {args.exp_id}  head={exp['head']}")
    print(f"# Writes: {exp_folder(exp, registry)}/batch_sizes.json")
    print(f"# NOTE: jobs/{_PROBE_BATCH_SCRIPT} does not exist yet — create it before running this.")
    print(f"# Run BEFORE 'gen_commands_osf.py train {args.exp_id}'{probed_note}")
    print()
    print(build_probe_batch_cmd(exp, registry, starting_batch_size=starting))


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
        bsz_file = exp_folder(exp, registry) / "batch_sizes.json"
        src = f"probed ({bsz_file})" if bsz_file.exists() else "registry defaults (run probe-batch first)"
        batch_label = f"MEMORY-BOUNDED (accum_steps=1, batch sizes from {src})"
    elif ga_enabled:
        batch_label = f"GRAD-ACCUM (effective_batch={ga.get('effective_batch', 32)}, accum varies by context)"
    else:
        batch_label = f"FLAT batch={exp['batch_size']}, accum_steps=1"
    print(f"# Training commands for: {args.exp_id}")
    print(f"# Task: {exp['task']}  Head: {exp['head']}  LR: {exp['lr']}  N-size: {n_size}")
    print(f"# Datasets: {exp['datasets']}")
    print(f"# Batch mode: {batch_label}")
    print(f"# Logs → {registry.get('logs_dir', 'logs_osf/')}")
    print(f"# NOTE: wall-time estimates are placeholder (not GPU-calibrated for OSF yet)")
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
    print(f"# Est. wall time: {wall}  Logs → {registry.get('logs_dir', 'logs_osf/')}")
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
        print(f"#   Run val inference first (see gen_commands_osf.py infer {args.exp_id} --split val)")
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
    logs_dir = Path(registry.get("logs_dir", str(Path(__file__).parent.parent / "logs_osf")))
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
# repo mirrors are round-tagged (results_dir.name, e.g. phase0_osf) so
# different backbones/rounds never overwrite each other in results/.
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
    print("# Collect all OSF experiment results into training.csv and analysis.csv")
    print(f"# Output → {cdir}/")
    print("# Note: bootstrap CIs are computed by analyze_windows.py (run 'analyze --bootstrap N' first)")
    print("# Prerequisite: inference parquets must exist (run infer first)")
    print()
    print(build_collect_cmd(exp_ids, registry, collected_dir))


def main():
    parser = argparse.ArgumentParser(
        description="Generate sbatch/python commands for OSF baseline experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--registry", default=str(REGISTRY_PATH),
                        help="Path to v2_osf_registry.yaml")
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="List all experiments and their status")
    p_list.add_argument("--tier", default=None, help="Filter by tier (1 or 2)")

    p_pb = sub.add_parser("probe-batch",
                           help="Print sbatch command to probe max batch size (memory_bounded only; script not yet created)")
    p_pb.add_argument("exp_id", help="Experiment ID from registry (must have batch_mode: memory_bounded)")
    p_pb.add_argument("--starting-batch-size", type=int, default=256, dest="starting_batch_size",
                      help="Largest batch to try first; halved on OOM (default: 256)")

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
                         help="Override inference batch size, e.g. --batch-size 128")

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
        "probe-batch":          cmd_probe_batch,
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
