#!/bin/bash
#SBATCH --job-name=ctx_sweep
#SBATCH --account=def-forouzan_gpu
#SBATCH --time=24:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --exclude=fc11006,fc11013,fc11010
#SBATCH --signal=B:USR1@120            # send SIGUSR1 to bash 120s before wall time
#SBATCH --output=/home/boshra95/NSRR-tools/logs_v3/sweep_%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools/logs_v3/sweep_%x_%j.err

# Phase 0 Step 4 — Context-Length Sweep Training
#
# Trains one (task, head_type) combination across all context lengths.
# Each context length takes ~10-30 min; 7 lengths × up to 30 epochs = ~2-4h.
# Already-finished context lengths are skipped automatically (safe to resubmit).
#
# Auto-resume on timeout:
#   --signal=B:USR1@120 sends SIGUSR1 to bash 120s before the wall-time limit.
#   Python runs in the background; 'wait' returns immediately on USR1, letting
#   the trap fire, kill Python cleanly, and resubmit this script via sbatch "$0".
#   The new job finds resume.pt (saved after every epoch) and continues training.
#   This works on Alliance Canada — no --requeue needed for timeout-based resume.
#
# Usage examples:
#   sbatch --export=ALL,TASK=apnea_binary,HEAD=lstm        jobs/train_context_sweep_gpu.sh
#   sbatch --export=ALL,TASK=apnea_binary,HEAD=mean_pool   jobs/train_context_sweep_gpu.sh
#   sbatch --export=ALL,TASK=apnea_binary,HEAD=transformer jobs/train_context_sweep_gpu.sh
#   sbatch --export=ALL,TASK=sleep_staging,HEAD=lstm       jobs/train_context_sweep_gpu.sh
#
#   # Filter to specific datasets (space-separated, quoted):
#   sbatch --export=ALL,TASK=sleep_staging,HEAD=lstm,DATASETS="shhs1 shhs2" jobs/train_context_sweep_gpu.sh
#
#   # Disable W&B for a run:
#   sbatch --export=ALL,...,NO_WANDB=1 jobs/train_context_sweep_gpu.sh
#
# W&B setup: store your API key in ~/.wandb_key (chmod 600).
#   The script loads it automatically — no interactive prompts.
#
# Or single default run (uses task/head from phase0_config.yaml):
#   sbatch jobs/train_context_sweep_gpu.sh

set -e

# Store absolute path early — needed for resubmission from within the job
_SCRIPT_PATH="$(realpath "$0")"
_PYTHON_PID=""

cd /home/boshra95/NSRR-tools
mkdir -p logs_v3
mkdir -p logs_v3/status

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11 2>/dev/null || true
# module load cuda 2>/dev/null || true

source /home/boshra95/sleepfm_env/bin/activate

export PYTHONPATH="/home/boshra95/sleepfm-clinical:/home/boshra95/sleepfm-clinical/sleepfm:$PYTHONPATH"

# Fail fast if CUDA is not available — avoids silent CPU fallback
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available on node $SLURM_NODELIST'" || {
    echo "ERROR: CUDA not available. Cancel and resubmit with --exclude=$SLURM_NODELIST"
    exit 1
}

# ── W&B setup (non-interactive) ───────────────────────────────────────────────
# Store your key once: echo "your_key_here" > ~/.wandb_key && chmod 600 ~/.wandb_key
[ -f ~/.wandb_key ] && export WANDB_API_KEY=$(cat ~/.wandb_key)
export WANDB_DIR=/tmp/wandb_${SLURM_JOB_ID}   # node-local tmp, auto-cleaned
mkdir -p "$WANDB_DIR"

# ── Job parameters ────────────────────────────────────────────────────────────
CONFIG=${CONFIG:-"configs/phase0_v3_config.yaml"}
TASK=${TASK:-""}            # empty = use config default
TASK_TYPE=${TASK_TYPE:-""}  # empty = use config default
HEAD=${HEAD:-""}            # empty = use config default
CONTEXT=${CONTEXT:-""}      # single context length, e.g. "30s" or "10m"
DATASETS=${DATASETS:-""}    # space-separated dataset names, e.g. "shhs mros"
BATCH_SIZE=${BATCH_SIZE:-""}  # training batch size (default: 32); reduce for long contexts
LR=${LR:-""}                  # learning rate override, e.g. LR=1e-4
RUN_TAG=${RUN_TAG:-""}        # suffix for experiment folder, e.g. RUN_TAG=lr1e4
WANDB_PROJECT=${WANDB_PROJECT:-"nsrr-phase0-v3"}
NO_WANDB=${NO_WANDB:-""}

# ── Job run tracking ──────────────────────────────────────────────────────────
_EXP_TAG="${TASK}_${HEAD}"
[ -n "$RUN_TAG" ] && _EXP_TAG="${_EXP_TAG}_${RUN_TAG}"
_STATUS_FILE="logs_v3/status/train_${_EXP_TAG}_${CONTEXT:-nocontext}_lr${LR:-default}.jsonl"

# Persistent training log — all resubmissions append here so the full epoch
# history is in one place regardless of how many jobs the run takes.
_TRAIN_LOG="logs_v3/train_${_EXP_TAG}_${CONTEXT:-nocontext}_lr${LR:-default}.log"
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
# Python runs in background so 'wait' returns immediately on signal.
# All env vars (TASK, HEAD, etc.) are already exported and will be forwarded
# via --export=ALL to the new job, which finds resume.pt and continues.
_timeout_handler() {
    _write_status "TIMEOUT_REQUEUED"
    echo ""
    echo "Time limit approaching — resubmitting for auto-resume ($(date))"
    [ -n "$_PYTHON_PID" ] && kill -TERM "$_PYTHON_PID" 2>/dev/null || true
    # Forward the same wall time so the resubmitted job doesn't fall back to
    # the script's 24h default
    _TIME_LIMIT=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
        | grep -oP 'TimeLimit=\K\S+' || echo "24:00:00")
    NEW_JOB=$(sbatch --export=ALL --time="$_TIME_LIMIT" "$_SCRIPT_PATH" 2>&1)
    echo "$NEW_JOB"
    exit 0
}
trap '_timeout_handler' USR1

_write_status "STARTED"

echo "========================================================================"
echo "Phase 0 Step 4 — Context-Length Sweep"
echo "========================================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Task:      ${TASK:-'(from config)'}  type=${TASK_TYPE:-'(from config)'}"
echo "Head:      ${HEAD:-'(from config)'}"
echo "Datasets:  ${DATASETS:-'(all)'}"
echo "W&B:       $([ -n "$NO_WANDB" ] && echo disabled || echo "project=$WANDB_PROJECT")"
echo "Start:     $(date)"
echo "========================================================================"
echo ""

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python scripts/train_context_sweep.py --config $CONFIG"
[ -n "$TASK"           ] && CMD="$CMD --task $TASK"
[ -n "$TASK_TYPE"      ] && CMD="$CMD --task-type $TASK_TYPE"
[ -n "$HEAD"           ] && CMD="$CMD --head $HEAD"
[ -n "$CONTEXT"        ] && CMD="$CMD --context $CONTEXT"
[ -n "$DATASETS"       ] && CMD="$CMD --datasets $DATASETS"
[ -n "$BATCH_SIZE"     ] && CMD="$CMD --batch-size $BATCH_SIZE"
[ -n "$LR"             ] && CMD="$CMD --lr $LR"
[ -n "$RUN_TAG"        ] && CMD="$CMD --run-tag $RUN_TAG"
[ -n "$WANDB_PROJECT"  ] && CMD="$CMD --wandb-project $WANDB_PROJECT"
[ -n "$NO_WANDB"       ] && CMD="$CMD --no-wandb"

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
    # Read failure reason written by Python (exp_dir/_failure_reason_<jobid>.txt)
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
