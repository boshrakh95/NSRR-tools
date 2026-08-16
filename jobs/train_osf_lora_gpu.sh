#!/bin/bash
#SBATCH --job-name=osf_lora_sweep
#SBATCH --account=def-forouzan_gpu
#SBATCH --time=24:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --exclude=fc11006,fc11013,fc11010
#SBATCH --signal=B:USR1@120            # send SIGUSR1 to bash 120s before wall time
#SBATCH --output=/home/boshra95/NSRR-tools/logs_osf_lora/%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools/logs_osf_lora/%x_%j.err

# OSF baseline — Stage 2 Step 1 — LoRA Fine-Tuning Sweep
#
# Forked from jobs/train_osf_context_sweep_gpu.sh — see
# docs/TSFM_OSF_IMPLEMENTATION_PLAN.md Appendix §6 / checklist 2.3. Same
# structure, same auto-resume mechanism, same status/log conventions as
# Stage 1's job script; only the venv (still osf_env), target script,
# config, log dir, and job name differ. No W&B here — train_osf_lora.py
# doesn't wire it in (same known gap as Stage 1: wandb isn't installed in
# osf_env, see docs/TSFM_OSF_IMPLEMENTATION_PLAN.md Known Issues).
#
# Trains one (task, head_type) combination across all context lengths,
# warm-starting each context's sequence_head from the MATCHING Stage 1
# checkpoint (auto-detected by train_osf_lora.py if --stage1-checkpoint
# isn't passed — see that script's docstring). Already-finished context
# lengths are skipped automatically (safe to resubmit).
#
# Auto-resume on timeout: same mechanism as train_osf_context_sweep_gpu.sh
# (--signal=B:USR1@120 + bash trap + resume.pt, saved every epoch).
#
# GPU size: 3g.40gb (upgraded 2026-08-15 from 1g.10gb) — MIG partitions
# compute (SMs) proportionally to memory, not just VRAM, so this should be
# a real ~3x throughput gain, not just more headroom. Real 1g.10gb data
# point: apnea_binary/lstm/30s ran at ~60 min/epoch (measured from
# resume.pt's accumulated_time_min, not a guess) — full backbone
# forward+backward per raw epoch in the window, every training step,
# fundamentally different from Stage 1's cheap precomputed-embedding
# lookup. 3g.40gb timing not yet independently confirmed.
#
# ⚠️ COMPUTE SCALES ~LINEARLY WITH CONTEXT LENGTH, NOT SUB-LINEARLY —
# CombinedOSFLoRAModel.forward() runs every single raw epoch in a window
# through the full backbone individually (chunked, but not amortized), so
# 240m (480 epochs/window) costs ~480x what 30s (1 epoch/window) costs per
# training step, not a fraction of it. Confirm feasibility per context
# length before submitting long-context jobs — see
# docs/TSFM_OSF_IMPLEMENTATION_PLAN.md checklist 2.6 for the real numbers
# once measured, and the warm-start-from-30s plan for how long contexts
# are intended to be made tractable.
#
# ⚠️ WALL-TIME NOT YET FULLY CALIBRATED for 3g.40gb (checklist 2.6, not
# done yet as of this writing) — the 24h default below is a placeholder,
# not measured at this GPU size. Run a real pilot before trusting it for
# long contexts.
#
# Usage examples:
#   sbatch --export=ALL,TASK=apnea_binary,HEAD=lstm jobs/train_osf_lora_gpu.sh
#
#   # Single context, explicit Stage 1 checkpoint (skip auto-detection):
#   sbatch --export=ALL,TASK=apnea_binary,HEAD=lstm,CONTEXT=30s,\
#STAGE1_CHECKPOINT=/scratch/boshra95/psg_full/unified/results/phase0_osf/apnea_binary_lstm/context_30s/best_model.pt \
#       jobs/train_osf_lora_gpu.sh
#
#   # Filter to specific datasets:
#   sbatch --export=ALL,TASK=apnea_binary,HEAD=lstm,DATASETS="apples shhs" jobs/train_osf_lora_gpu.sh
#
# Or single default run (uses task/head from phase0_osf_lora_config.yaml):
#   sbatch jobs/train_osf_lora_gpu.sh

set -e

# Store absolute path early — needed for resubmission from within the job
_SCRIPT_PATH="$(realpath "$0")"
_PYTHON_PID=""

cd /home/boshra95/NSRR-tools
LOGS_DIR=${LOGS_DIR:-logs_osf_lora}
mkdir -p "$LOGS_DIR"
mkdir -p "$LOGS_DIR/status"

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.10.13 2>/dev/null || true

source /home/boshra95/osf_env/bin/activate

# Unbuffered stdout — without this, Python block-buffers when stdout isn't a
# TTY (i.e. always, here), so epoch progress can sit invisible in an internal
# buffer for a long time before appearing in the log file (only flushed on
# process exit/kill). Found live 2026-08-15: a real job had made genuine
# progress (verified via resume.pt) with nothing showing in the log.
export PYTHONUNBUFFERED=1

# Fail fast if CUDA is not available — avoids silent CPU fallback
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available on node $SLURM_NODELIST'" || {
    echo "ERROR: CUDA not available. Cancel and resubmit with --exclude=$SLURM_NODELIST"
    exit 1
}

# ── Job parameters ────────────────────────────────────────────────────────────
CONFIG=${CONFIG:-"configs/phase0_osf_lora_config.yaml"}
TASK=${TASK:-""}
HEAD=${HEAD:-""}
CONTEXT=${CONTEXT:-""}
DATASETS=${DATASETS:-""}
BATCH_SIZE=${BATCH_SIZE:-""}   # empty = train_osf_lora.py falls back to 32 (same as Stage 1/SleepFM)
ACCUM_STEPS=${ACCUM_STEPS:-1}  # effective_batch = BATCH_SIZE × ACCUM_STEPS; raise if BATCH_SIZE must drop to fit
LR=${LR:-""}
RUN_TAG=${RUN_TAG:-""}
STAGE1_CHECKPOINT=${STAGE1_CHECKPOINT:-""}   # empty = auto-detect matching Stage 1 checkpoint

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
        | grep -oP 'TimeLimit=\K\S+' || echo "24:00:00")
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
echo "OSF baseline — Stage 2 Step 1 — LoRA Fine-Tuning Sweep"
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
CMD="python scripts/train_osf_lora.py --config $CONFIG"
[ -n "$TASK"              ] && CMD="$CMD --task $TASK"
[ -n "$HEAD"              ] && CMD="$CMD --head $HEAD"
[ -n "$CONTEXT"           ] && CMD="$CMD --context $CONTEXT"
[ -n "$DATASETS"          ] && CMD="$CMD --datasets $DATASETS"
[ -n "$BATCH_SIZE"        ] && CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --accum-steps $ACCUM_STEPS"
[ -n "$LR"                ] && CMD="$CMD --lr $LR"
[ -n "$RUN_TAG"           ] && CMD="$CMD --run-tag $RUN_TAG"
[ -n "$STAGE1_CHECKPOINT" ] && CMD="$CMD --stage1-checkpoint $STAGE1_CHECKPOINT"

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
