#!/bin/bash
#SBATCH --job-name=mantis_lora_sweep
#SBATCH --account=def-egranger_gpu
#SBATCH --time=04:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000M
#SBATCH --exclude=fc11006,fc11013,fc11010,fc10713
#SBATCH --signal=B:USR1@120            # send SIGUSR1 to bash 120s before wall time
#SBATCH --output=/home/boshra95/NSRR-tools-mantis/logs_mantis_lora/%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools-mantis/logs_mantis_lora/%x_%j.err

# Mantis baseline — Stage 2 Step 1 — LoRA Fine-Tuning Sweep (checklist 2.5)
#
# Forked from jobs/train_osf_lora_gpu.sh — see
# docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md §14/§4.5. Same structure, same
# auto-resume mechanism, same status/log conventions as Stage 1's job
# script; only the venv, target script, config, log dir, and job name
# differ. No W&B here — train_mantis_lora.py doesn't wire it in (same
# known gap as Stage 1: check whether wandb is installed in mantis_env
# before assuming this works).
#
# GPU size: **whole H100 (`--gpus=h100:1`), NOT a MIG slice** — a
# deliberate, plan-mandated decision (§4.5), NOT the "get minimal, escalate
# only on real OOM" rule Stage 1 extraction follows. The reason is memory,
# not throughput: Stage 2's backward pass needs ~240 MB of activation
# memory per epoch-unit (micro_batch × N raw epochs per window, §4.3) — at
# 240m context (480 epochs/window) even a micro_batch=1 floor needs ~480
# epoch-units, which does not fit in anything smaller than the full 80 GB
# card. This is not optional at long contexts, unlike OSF's own 1g.10gb→
# 3g.40gb upgrade (which measured ZERO throughput speedup, since that was
# a throughput argument, not a memory one, and was tested only in the
# overhead-bound 30s regime). Treat any speed gain here as a bonus;
# measure real per-epoch cost at 40m+, never 30s (§13.3/§4.1).
#
# --time=04:00:00 default (not 24h) is ALSO a deliberate plan decision
# (§4.5): --time dominates queue position far more than GPU type on this
# cluster — a short wall-time request queues same-day vs. days-out for a
# long one — and per-epoch checkpointing + auto-resume make a short
# request nearly free. Let auto-resume handle anything that needs longer.
#
# Trains one (task, head_type) combination across all context lengths,
# warm-starting each context's weights per plan §14.5 (30s from Stage 1;
# every other context from this task/head's OWN converged 30s Stage 2
# checkpoint, auto-detected by train_mantis_lora.py if
# --stage1-checkpoint/--stage2-30s-checkpoint aren't passed). Already-
# finished context lengths are skipped automatically (safe to resubmit).
#
# Auto-resume on timeout: same mechanism as train_mantis_context_sweep_gpu.sh
# (--signal=B:USR1@120 + bash trap + resume.pt, saved every epoch).
#
# Memory-mitigation ladder (§4.3/§14.8), if the whole-card allocation
# still OOMs at the longest contexts: lower CONTEXT_MICRO_BATCH (raise
# ACCUM_STEPS proportionally to hold effective_batch=32), then set
# CHECKPOINT_TOKGEN=1 (first rung — ~1.3% extra compute for ~39% less
# activation memory, cheaper than whole-model checkpointing), then
# CHECKPOINT_CHUNKS=1 (coarser, more memory saved, more recompute cost).
#
# Usage examples:
#   sbatch --export=ALL,TASK=apnea_binary,HEAD=lstm jobs/train_mantis_lora_gpu.sh
#
#   # Single context, explicit Stage 1 checkpoint (skip auto-detection):
#   sbatch --export=ALL,TASK=apnea_binary,HEAD=lstm,CONTEXT=30s,\
#STAGE1_CHECKPOINT=/scratch/boshra95/psg/unified/results/phase0_mantis/apnea_binary_lstm/context_30s/best_model.pt \
#       jobs/train_mantis_lora_gpu.sh
#
#   # Filter to specific datasets:
#   sbatch --export=ALL,TASK=apnea_binary,HEAD=lstm,DATASETS="apples shhs" jobs/train_mantis_lora_gpu.sh
#
#   # Enable tokgen gradient-checkpointing (first memory-mitigation rung):
#   sbatch --export=ALL,TASK=apnea_binary,HEAD=lstm,CONTEXT=240m,CHECKPOINT_TOKGEN=1 jobs/train_mantis_lora_gpu.sh
#
# Or single default run (uses task/head from phase0_mantis_lora_config.yaml):
#   sbatch jobs/train_mantis_lora_gpu.sh

set -e

# Store absolute path early — needed for resubmission from within the job
_SCRIPT_PATH="$(realpath "$0")"
_PYTHON_PID=""

cd /home/boshra95/NSRR-tools-mantis
LOGS_DIR=${LOGS_DIR:-logs_mantis_lora}
mkdir -p "$LOGS_DIR"
mkdir -p "$LOGS_DIR/status"

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.10.13 2>/dev/null || true

source /home/boshra95/mantis_env/bin/activate

# Unbuffered stdout — without this, Python block-buffers when stdout isn't a
# TTY, so epoch progress can sit invisible in an internal buffer for a long
# time before appearing in the log file (same real gap found on OSF's own
# Stage 2 job, plan §4.10).
export PYTHONUNBUFFERED=1

# Fail fast if CUDA is not available — avoids silent CPU fallback
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available on node $SLURM_NODELIST'" || {
    echo "ERROR: CUDA not available. Cancel and resubmit with --exclude=$SLURM_NODELIST"
    exit 1
}

# ── Job parameters ────────────────────────────────────────────────────────────
CONFIG=${CONFIG:-"configs/phase0_mantis_lora_config.yaml"}
TASK=${TASK:-""}
HEAD=${HEAD:-""}
CONTEXT=${CONTEXT:-""}
DATASETS=${DATASETS:-""}
BATCH_SIZE=${BATCH_SIZE:-""}   # empty = train_mantis_lora.py falls back to 32
ACCUM_STEPS=${ACCUM_STEPS:-1}  # effective_batch = BATCH_SIZE × ACCUM_STEPS
LR=${LR:-""}
RUN_TAG=${RUN_TAG:-""}
STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-""}          # empty = auto-detect matching Stage 1 checkpoint
STAGE2_30S_CHECKPOINT=${STAGE2_30S_CHECKPOINT:-""}  # empty = auto-detect this task/head's own 30s Stage 2 checkpoint
CHECKPOINT_TOKGEN=${CHECKPOINT_TOKGEN:-""}          # set to 1 to enable tokgen gradient-checkpointing (memory rung 1)
CHECKPOINT_CHUNKS=${CHECKPOINT_CHUNKS:-""}          # set to 1 to enable whole-chunk gradient-checkpointing (memory rung 2)

# checkpoint_tokgen/checkpoint_chunks are config-level flags (read by
# build_combined_lora_model from cfg["training"]), not CLI args — patch a
# throwaway copy of the config in $SLURM_TMPDIR if either is requested,
# rather than editing the real config file (never on the compute node's
# node-local /tmp — see plan §4's node-local-tmp lesson).
if [ -n "$CHECKPOINT_TOKGEN" ] || [ -n "$CHECKPOINT_CHUNKS" ]; then
    _PATCHED_CONFIG="${SLURM_TMPDIR:-/scratch/boshra95/tmp}/mantis_lora_config_${SLURM_JOB_ID:-local}.yaml"
    mkdir -p "$(dirname "$_PATCHED_CONFIG")"
    python -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('training', {})
if '$CHECKPOINT_TOKGEN':
    cfg['training']['checkpoint_tokgen'] = True
if '$CHECKPOINT_CHUNKS':
    cfg['training']['checkpoint_chunks'] = True
with open('$_PATCHED_CONFIG', 'w') as f:
    yaml.safe_dump(cfg, f)
"
    CONFIG="$_PATCHED_CONFIG"
    echo "Gradient-checkpointing requested — patched config: $CONFIG"
fi

# ── Job run tracking ──────────────────────────────────────────────────────────
_EXP_TAG="${TASK}_${HEAD}"
[ -n "$RUN_TAG" ] && _EXP_TAG="${_EXP_TAG}_${RUN_TAG}"
_STATUS_FILE="$LOGS_DIR/status/train_${_EXP_TAG}_${CONTEXT:-nocontext}_lr${LR:-default}.jsonl"

_TRAIN_LOG="$LOGS_DIR/train_${_EXP_TAG}_${CONTEXT:-nocontext}_lr${LR:-default}.log"
exec > >(tee -a "$_TRAIN_LOG") 2>&1

_write_status() {
    local _status="$1"
    local _reason="${2:-}"
    if [ -n "$_reason" ]; then
        printf '{"ts":"%s","job_id":"%s","node":"%s","status":"%s","reason":"%s","task":"%s","head":"%s","context":"%s","lr":"%s","datasets":"%s"}\n' \
            "$(date -Iseconds)" "${SLURM_JOB_ID:-local}" \
            "${SLURM_NODELIST:-local}" "$_status" "$_reason" \
            "$TASK" "$HEAD" "${CONTEXT:-?}" "${LR:-default}" "${DATASETS:-all}" \
            >> "$_STATUS_FILE"
    else
        printf '{"ts":"%s","job_id":"%s","node":"%s","status":"%s","task":"%s","head":"%s","context":"%s","lr":"%s","datasets":"%s"}\n' \
            "$(date -Iseconds)" "${SLURM_JOB_ID:-local}" \
            "${SLURM_NODELIST:-local}" "$_status" \
            "$TASK" "$HEAD" "${CONTEXT:-?}" "${LR:-default}" "${DATASETS:-all}" \
            >> "$_STATUS_FILE"
    fi
}

# ── Auto-resume trap (SIGUSR1 fires 120s before wall time) ───────────────────
_timeout_handler() {
    _write_status "TIMEOUT_REQUEUED"
    echo ""
    echo "Time limit approaching — resubmitting for auto-resume ($(date))"
    [ -n "$_PYTHON_PID" ] && kill -TERM "$_PYTHON_PID" 2>/dev/null || true
    _TIME_LIMIT=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
        | grep -oP 'TimeLimit=\K\S+' || echo "04:00:00")
    _LOG_STEM="${_TRAIN_LOG%.log}"
    NEW_JOB=$(sbatch \
        --export=ALL \
        --time="$_TIME_LIMIT" \
        --output="${_LOG_STEM}_%j.out" \
        --error="${_LOG_STEM}_%j.err" \
        "$_SCRIPT_PATH" 2>&1)
    echo "$NEW_JOB"
    exit 0
}
trap '_timeout_handler' USR1

_write_status "STARTED"

echo "========================================================================"
echo "Mantis baseline — Stage 2 Step 1 — LoRA Fine-Tuning Sweep"
echo "========================================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Task:      ${TASK:-'(from config)'}"
echo "Head:      ${HEAD:-'(from config)'}"
echo "Datasets:  ${DATASETS:-'(all)'}"
echo "Start:     $(date)"
echo "========================================================================"
echo ""

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python scripts/train_mantis_lora.py --config $CONFIG"
[ -n "$TASK"                   ] && CMD="$CMD --task $TASK"
[ -n "$HEAD"                   ] && CMD="$CMD --head $HEAD"
[ -n "$CONTEXT"                ] && CMD="$CMD --context $CONTEXT"
[ -n "$DATASETS"               ] && CMD="$CMD --datasets $DATASETS"
[ -n "$BATCH_SIZE"             ] && CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --accum-steps $ACCUM_STEPS"
[ -n "$LR"                     ] && CMD="$CMD --lr $LR"
[ -n "$RUN_TAG"                ] && CMD="$CMD --run-tag $RUN_TAG"
[ -n "$STAGE1_CHECKPOINT"      ] && CMD="$CMD --stage1-checkpoint $STAGE1_CHECKPOINT"
[ -n "$STAGE2_30S_CHECKPOINT"  ] && CMD="$CMD --stage2-30s-checkpoint $STAGE2_30S_CHECKPOINT"

echo "Running: $CMD"
echo ""

# Run Python in background so USR1 can interrupt 'wait' immediately
set +e
eval "$CMD" &
_PYTHON_PID=$!
wait $_PYTHON_PID
EXIT_CODE=$?
trap '' USR1   # training done — ignore any late-firing USR1
# Do NOT re-enable set -e here — EXIT_CODE is already captured and the
# cleanup/status path below must not be aborted by a non-zero subcommand.

echo ""
echo "========================================================================"
echo "End time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS"
    _write_status "SUCCESS"
else
    _RESULTS_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['logging']['results_dir'])" 2>/dev/null || echo "")
    _EXP_ID="${TASK}_${HEAD}"
    [ -n "$RUN_TAG" ] && _EXP_ID="${_EXP_ID}_${RUN_TAG}"
    _REASON_FILE="${_RESULTS_DIR}/${_EXP_ID}/_failure_reason_${SLURM_JOB_ID:-local}.txt"
    _REASON=$(cat "$_REASON_FILE" 2>/dev/null | tr '"' "'" || echo "unknown")
    echo "Status: FAILED (exit code: $EXIT_CODE) — ${_REASON}"
    _write_status "FAILED" "$_REASON"
fi
echo "========================================================================"

deactivate
exit $EXIT_CODE
