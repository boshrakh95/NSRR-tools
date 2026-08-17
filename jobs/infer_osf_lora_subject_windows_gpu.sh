#!/bin/bash
#SBATCH --job-name=osf_lora_infer_windows
#SBATCH --account=def-forouzan_gpu
#SBATCH --time=05:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --exclude=fc11006,fc11013,fc11010
#SBATCH --signal=B:USR1@120            # send SIGUSR1 to bash 120s before wall time
#SBATCH --output=/home/boshra95/NSRR-tools/logs_osf_lora/%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools/logs_osf_lora/%x_%j.err

# OSF baseline — Stage 2 Step 2 — LoRA Subject-level inference (all windows)
#
# Forked from jobs/infer_osf_subject_windows_gpu.sh — see
# docs/TSFM_OSF_IMPLEMENTATION_PLAN.md checklist 2.4. Same structure, same
# auto-resume mechanism, same status/log conventions as Stage 1's/Stage 2
# training's job scripts; only the venv (still osf_env), target script,
# config, log dir, and job name differ. No TASK_TYPE/ZERO_MODALITIES here —
# Stage 2 is seq2label-only (see osf_raw_epoch_dataset.py scope note).
#
# Loads a train_osf_lora.py checkpoint (peft state dict: LoRA deltas +
# sequence_head) and runs the LoRA-adapted backbone live on raw signal for
# ALL non-overlapping windows per subject (no K=5 cap). Saves a parquet of
# per-window probabilities for downstream majority-voting / mean-prob
# aggregation — same schema as Stage 1's inference output.
#
# Already-done contexts (output parquet already exists) are skipped
# automatically (safe to resubmit). WITHIN a context, progress is also
# incrementally resumable as of 2026-08-17 (see infer_osf_lora_subject_
# windows.py's RESUMABILITY docstring note) — a timeout no longer restarts
# an "all windows" pass (millions of raw epochs, can take many hours) from
# item 0; it picks up from the last periodic checkpoint (every 5 min).
# Auto-resume on timeout: same mechanism as train_osf_lora_gpu.sh —
#   --signal=B:USR1@120 fires 120s before wall time, Python is killed cleanly,
#   and this script is resubmitted via sbatch "$0" with --export=ALL.
#
# Usage examples:
#   # Single context:
#   sbatch --export=ALL,TASK=apnea_binary,HEAD=lstm,CONTEXTS="10m" \
#       jobs/infer_osf_lora_subject_windows_gpu.sh
#
#   # Multiple contexts in one job (already-done are skipped automatically):
#   sbatch --export=ALL,TASK=apnea_binary,HEAD=lstm,CONTEXTS="30s 10m 40m 80m 120m 240m" \
#       jobs/infer_osf_lora_subject_windows_gpu.sh
#
#   # With dataset filter:
#   sbatch --export=ALL,TASK=apnea_binary,HEAD=lstm,CONTEXTS="30s 10m",DATASETS="apples shhs" \
#       jobs/infer_osf_lora_subject_windows_gpu.sh
#
#   # Run on val split instead of test:
#   sbatch --export=ALL,...,SPLIT=val jobs/infer_osf_lora_subject_windows_gpu.sh
#
#   # Reproduce training eval exactly (K=5 windows, no --all-windows):
#   sbatch --export=ALL,...,NO_ALL_WINDOWS=1 jobs/infer_osf_lora_subject_windows_gpu.sh
#
# ⚠️ WALL-TIME NOT YET CALIBRATED — 5h default copied from Stage 1's
# inference job as a placeholder. Stage 2 inference is more expensive per
# window than Stage 1's (live LoRA-adapted backbone forward pass per raw
# epoch, not a cached-embedding lookup) — revisit after checklist 2.6.

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

# Unbuffered stdout — see jobs/train_osf_lora_gpu.sh for why this matters.
export PYTHONUNBUFFERED=1

# Fail fast if CUDA is not available
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available on node $SLURM_NODELIST'" || {
    echo "ERROR: CUDA not available. Cancel and resubmit with --exclude=$SLURM_NODELIST"
    exit 1
}

# ── Job parameters ────────────────────────────────────────────────────────────
CONFIG=${CONFIG:-"configs/phase0_osf_lora_config.yaml"}
TASK=${TASK:-""}
HEAD=${HEAD:-"lstm"}
CONTEXTS=${CONTEXTS:-""}
SPLIT=${SPLIT:-"test"}
DATASETS=${DATASETS:-""}
NO_ALL_WINDOWS=${NO_ALL_WINDOWS:-""}   # set to 1 to use K=5 (training eval mode)
BATCH_SIZE=${BATCH_SIZE:-32}           # default in infer_osf_lora_subject_windows.py: 32 (raised
                                        # 2026-08-17 from 4 -- see that script's BATCH SIZE note)
RUN_TAG=${RUN_TAG:-""}                 # must match RUN_TAG used during training

# ── Job run tracking ──────────────────────────────────────────────────────────
_EXP_TAG="${TASK}_${HEAD}"
[ -n "$RUN_TAG" ] && _EXP_TAG="${_EXP_TAG}_${RUN_TAG}"
_STATUS_FILE="$LOGS_DIR/status/infer_${_EXP_TAG}_${SPLIT}.jsonl"

# Persistent inference log — all resubmissions append here.
_INFER_LOG="$LOGS_DIR/infer_${_EXP_TAG}_${SPLIT}.log"
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
    # Pass --output/--error explicitly so resubmitted jobs get the same
    # descriptive filename as the original submission (not the generic %x_%j fallback).
    _TIME_LIMIT=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
        | grep -oP 'TimeLimit=\K\S+' || echo "05:00:00")
    _LOG_STEM="${_INFER_LOG%.log}"
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
echo "OSF baseline — Stage 2 (LoRA) — Subject-level inference (all windows)"
echo "========================================================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Task:        ${TASK}  (seq2label)"
echo "Head:        ${HEAD}"
echo "Contexts:    ${CONTEXTS:-'(auto-discover)'}"
echo "Split:       ${SPLIT}"
echo "Datasets:    ${DATASETS:-'(all)'}"
echo "All windows: $([ -n "$NO_ALL_WINDOWS" ] && echo 'no (K=5)' || echo 'yes')"
echo "Start:       $(date)"
echo "========================================================================"
echo ""

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python scripts/infer_osf_lora_subject_windows.py --config $CONFIG"
[ -n "$TASK"           ] && CMD="$CMD --task $TASK"
[ -n "$HEAD"           ] && CMD="$CMD --head $HEAD"
[ -n "$CONTEXTS"       ] && CMD="$CMD --context $CONTEXTS"
[ -n "$SPLIT"          ] && CMD="$CMD --split $SPLIT"
[ -n "$DATASETS"       ] && CMD="$CMD --datasets $DATASETS"
[ -n "$NO_ALL_WINDOWS"   ] && CMD="$CMD --no-all-windows"
[ -n "$RUN_TAG"          ] && CMD="$CMD --run-tag $RUN_TAG"
CMD="$CMD --batch-size $BATCH_SIZE"

echo "Running: $CMD"
echo ""

# Run Python in background so USR1 can interrupt 'wait' immediately
set +e
eval "$CMD" &
_PYTHON_PID=$!
wait $_PYTHON_PID
EXIT_CODE=$?
trap '' USR1   # inference done — ignore any late-firing USR1
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
