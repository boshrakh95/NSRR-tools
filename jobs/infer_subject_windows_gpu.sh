#!/bin/bash
#SBATCH --job-name=infer_windows
#SBATCH --account=def-forouzan_gpu
#SBATCH --time=05:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --exclude=fc11006
#SBATCH --signal=B:USR1@120            # send SIGUSR1 to bash 120s before wall time
#SBATCH --output=/home/boshra95/NSRR-tools/logs_v3/infer_%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools/logs_v3/infer_%x_%j.err

# Phase 0 — Subject-level inference (all windows)
#
# Loads a trained checkpoint and runs inference on ALL non-overlapping windows
# per subject (no K=5 cap).  Saves a parquet of per-window probabilities for
# downstream majority-voting / mean-prob aggregation.
#
# Already-done contexts are skipped automatically (safe to resubmit).
# Auto-resume on timeout: same mechanism as train_context_sweep_gpu.sh —
#   --signal=B:USR1@120 fires 120s before wall time, Python is killed cleanly,
#   and this script is resubmitted via sbatch "$0" with --export=ALL.
#
# Usage examples:
#   # Single context:
#   sbatch --export=ALL,TASK=apnea_binary,TASK_TYPE=seq2label,HEAD=lstm,CONTEXTS="10m" \
#       jobs/infer_subject_windows_gpu.sh
#
#   # Multiple contexts in one job (already-done are skipped automatically):
#   sbatch --export=ALL,TASK=apnea_binary,TASK_TYPE=seq2label,HEAD=lstm,CONTEXTS="30s 10m 40m 80m" \
#       jobs/infer_subject_windows_gpu.sh
#
#   # With dataset filter:
#   sbatch --export=ALL,TASK=cvd_binary,TASK_TYPE=seq2label,HEAD=lstm,CONTEXTS="30s 10m 40m",DATASETS="shhs mros apples" \
#       jobs/infer_subject_windows_gpu.sh
#
#   # Run on val split instead of test:
#   sbatch --export=ALL,...,SPLIT=val jobs/infer_subject_windows_gpu.sh
#
#   # Reproduce training eval exactly (K=5 windows, no --all-windows):
#   sbatch --export=ALL,...,NO_ALL_WINDOWS=1 jobs/infer_subject_windows_gpu.sh
#
# Time guide (all-windows, test split, H100):
#   30s context: ~9.5M items → ~20 min
#   10m context: ~475k items → ~3  min
#   40m context: ~120k items → <1  min
#   Multi-context job: sum of individual times + set --time accordingly

set -e

# Store absolute path early — needed for resubmission from within the job
_SCRIPT_PATH="$(realpath "$0")"
_PYTHON_PID=""

cd /home/boshra95/NSRR-tools
mkdir -p logs_v3
mkdir -p logs_v3/status

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11 2>/dev/null || true

source /home/boshra95/sleepfm_env/bin/activate

export PYTHONPATH="/home/boshra95/sleepfm-clinical:/home/boshra95/sleepfm-clinical/sleepfm:$PYTHONPATH"

# Fail fast if CUDA is not available
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available on node $SLURM_NODELIST'" || {
    echo "ERROR: CUDA not available. Cancel and resubmit with --exclude=$SLURM_NODELIST"
    exit 1
}

# ── Job parameters ────────────────────────────────────────────────────────────
CONFIG=${CONFIG:-"configs/phase0_v3_config.yaml"}
TASK=${TASK:-""}
TASK_TYPE=${TASK_TYPE:-"seq2label"}
HEAD=${HEAD:-"lstm"}
CONTEXTS=${CONTEXTS:-""}
SPLIT=${SPLIT:-"test"}
DATASETS=${DATASETS:-""}
NO_ALL_WINDOWS=${NO_ALL_WINDOWS:-""}   # set to 1 to use K=5 (training eval mode)
BATCH_SIZE=${BATCH_SIZE:-512}
RUN_TAG=${RUN_TAG:-""}                 # must match RUN_TAG used during training

# ── Job run tracking ──────────────────────────────────────────────────────────
_EXP_TAG="${TASK}_${HEAD}"
[ -n "$RUN_TAG" ] && _EXP_TAG="${_EXP_TAG}_${RUN_TAG}"
_STATUS_FILE="logs_v3/status/infer_${_EXP_TAG}_${SPLIT}.jsonl"

# Persistent inference log — all resubmissions append here.
_INFER_LOG="logs_v3/infer_${_EXP_TAG}_${SPLIT}.log"
exec > >(tee -a "$_INFER_LOG") 2>&1

_write_status() {
    local _status="$1"
    local _reason="${2:-}"
    if [ -n "$_reason" ]; then
        printf '{"ts":"%s","job_id":"%s","node":"%s","status":"%s","reason":"%s","task":"%s","head":"%s","contexts":"%s","split":"%s","datasets":"%s"}\n' \
            "$(date -Iseconds)" "${SLURM_JOB_ID:-local}" \
            "${SLURM_NODELIST:-local}" "$_status" "$_reason" \
            "$TASK" "$HEAD" "${CONTEXTS:-all}" "$SPLIT" "${DATASETS:-all}" \
            >> "$_STATUS_FILE"
    else
        printf '{"ts":"%s","job_id":"%s","node":"%s","status":"%s","task":"%s","head":"%s","contexts":"%s","split":"%s","datasets":"%s"}\n' \
            "$(date -Iseconds)" "${SLURM_JOB_ID:-local}" \
            "${SLURM_NODELIST:-local}" "$_status" \
            "$TASK" "$HEAD" "${CONTEXTS:-all}" "$SPLIT" "${DATASETS:-all}" \
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
        | grep -oP 'TimeLimit=\K\S+' || echo "05:00:00")
    NEW_JOB=$(sbatch --export=ALL --time="$_TIME_LIMIT" "$_SCRIPT_PATH" 2>&1)
    echo "$NEW_JOB"
    exit 0
}
trap '_timeout_handler' USR1

_write_status "STARTED"

echo "========================================================================"
echo "Phase 0 — Subject-level inference (all windows)"
echo "========================================================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Task:        ${TASK}  type=${TASK_TYPE}"
echo "Head:        ${HEAD}"
echo "Contexts:    ${CONTEXTS:-'(auto-discover)'}"
echo "Split:       ${SPLIT}"
echo "Datasets:    ${DATASETS:-'(all)'}"
echo "All windows: $([ -n "$NO_ALL_WINDOWS" ] && echo 'no (K=5)' || echo 'yes')"
echo "Start:       $(date)"
echo "========================================================================"
echo ""

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python scripts/infer_subject_windows.py --config $CONFIG"
[ -n "$TASK"           ] && CMD="$CMD --task $TASK"
[ -n "$TASK_TYPE"      ] && CMD="$CMD --task-type $TASK_TYPE"
[ -n "$HEAD"           ] && CMD="$CMD --head $HEAD"
[ -n "$CONTEXTS"       ] && CMD="$CMD --context $CONTEXTS"
[ -n "$SPLIT"          ] && CMD="$CMD --split $SPLIT"
[ -n "$DATASETS"       ] && CMD="$CMD --datasets $DATASETS"
[ -n "$NO_ALL_WINDOWS" ] && CMD="$CMD --no-all-windows"
[ -n "$RUN_TAG"        ] && CMD="$CMD --run-tag $RUN_TAG"
CMD="$CMD --batch-size $BATCH_SIZE"

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
    # Read failure reason written by Python (inference/{exp_id}/_failure_reason_<jobid>.txt)
    _RESULTS_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['logging']['results_dir'])" 2>/dev/null || echo "")
    _EXP_ID="${TASK}_${HEAD}"
    [ -n "$RUN_TAG" ] && _EXP_ID="${_EXP_ID}_${RUN_TAG}"
    _REASON_FILE="${_RESULTS_DIR}/inference/${_EXP_ID}/_failure_reason_${SLURM_JOB_ID:-local}.txt"
    _REASON=$(cat "$_REASON_FILE" 2>/dev/null | tr '"' "'" || echo "unknown")
    echo "Status: FAILED (exit code: $EXIT_CODE) — ${_REASON}"
    _write_status "FAILED" "$_REASON"
fi
echo "========================================================================"

deactivate
exit $EXIT_CODE
