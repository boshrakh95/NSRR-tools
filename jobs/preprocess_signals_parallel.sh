#!/bin/bash
#SBATCH --job-name=preprocess_signals
#SBATCH --account=def-forouzan
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=30000M
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs_v3_expand_channel/preprocess_%x_%j.out
#SBATCH --error=logs_v3_expand_channel/preprocess_%x_%j.err

# Preprocess NSRR EDF signals to HDF5 format — one dataset per job.
#
# Auto-resume on timeout (same mechanism as train_context_sweep_gpu.sh):
#   --signal=B:USR1@120 fires USR1 to bash 120 s before the wall-time limit.
#   Python runs in the background; 'wait' returns on USR1, the trap fires,
#   Python is sent SIGTERM (finishes the current subject cleanly), and this
#   script resubmits itself via sbatch with --export=ALL so all env vars
#   (DATASET, CONFIG_PATH, etc.) are forwarded to the new job.
#   The new job skips already-written HDF5 files and continues from there.
#   Works on Alliance Canada — no --requeue needed.
#
# Usage — preferred (env vars, safe for auto-resubmission):
#   DATASET=stages sbatch jobs/preprocess_signals_parallel.sh
#   DATASET=shhs CONFIG_PATH=configs/preprocessing_params_full.yaml \
#       sbatch jobs/preprocess_signals_parallel.sh
#
# Usage — legacy positional-arg style (also supported):
#   sbatch jobs/preprocess_signals_parallel.sh stages
#   sbatch jobs/preprocess_signals_parallel.sh stages --config configs/preprocessing_params_full.yaml
#   sbatch jobs/preprocess_signals_parallel.sh shhs --no-skip-existing
#   sbatch jobs/preprocess_signals_parallel.sh mros --mros-visit 2

set -e

_SCRIPT_PATH="$(realpath "$0")"
_PYTHON_PID=""

# ── Parameters — env var takes precedence, then positional arg, then default ──
DATASET=${DATASET:-${1:-stages}}
CONFIG_PATH=${CONFIG_PATH:-""}
SKIP_EXISTING=${SKIP_EXISTING:-"--skip-existing"}
REPROCESS_ANNOTATIONS=${REPROCESS_ANNOTATIONS:-""}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}
MROS_VISIT=${MROS_VISIT:-""}

# Parse any remaining positional options (if called in legacy style)
_pos1="${1:-}"
[ "$_pos1" = "$DATASET" ] && shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)                CONFIG_PATH="$2"; shift 2 ;;
        --no-skip-existing)      SKIP_EXISTING=""; shift ;;
        --skip-existing)         SKIP_EXISTING="--skip-existing"; shift ;;
        --reprocess-annotations) REPROCESS_ANNOTATIONS="--reprocess-annotations"; shift ;;
        --log-level)             LOG_LEVEL="$2"; shift 2 ;;
        --mros-visit)            MROS_VISIT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Export all params so --export=ALL forwards them on auto-resubmission
export DATASET CONFIG_PATH SKIP_EXISTING REPROCESS_ANNOTATIONS LOG_LEVEL MROS_VISIT

# ── Setup ─────────────────────────────────────────────────────────────────────
cd /home/boshra95/NSRR-tools
mkdir -p logs_v3_expand_channel

source .venv/bin/activate

# ── Auto-resume trap (fires 120 s before wall time) ───────────────────────────
_timeout_handler() {
    echo ""
    echo "[USR1] Time limit approaching — stopping Python and resubmitting ($(date))"
    [ -n "$_PYTHON_PID" ] && kill -TERM "$_PYTHON_PID" 2>/dev/null || true
    # Wait for Python to finish the current subject cleanly (SIGTERM handler in Python)
    wait "$_PYTHON_PID" 2>/dev/null || true
    _TIME_LIMIT=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
        | grep -oP 'TimeLimit=\K\S+' || echo "26:00:00")
    NEW_JOB=$(sbatch \
        --export=ALL \
        --time="$_TIME_LIMIT" \
        "$_SCRIPT_PATH" 2>&1)
    echo "$NEW_JOB"
    exit 0
}
trap '_timeout_handler' USR1

echo "========================================================================"
echo "NSRR Signal Preprocessing"
echo "========================================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Dataset:       $DATASET"
echo "Config:        ${CONFIG_PATH:-configs/preprocessing_params.yaml (default)}"
echo "Skip existing: $([ -n "$SKIP_EXISTING" ] && echo 'Yes' || echo 'No')"
[ -n "$MROS_VISIT" ] && echo "MrOS visit:    $MROS_VISIT"
echo "Start time:    $(date)"
echo "========================================================================"
echo ""

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python scripts/preprocess_signals.py --dataset $DATASET"
[ -n "$SKIP_EXISTING" ]         && CMD="$CMD $SKIP_EXISTING"
[ -n "$REPROCESS_ANNOTATIONS" ] && CMD="$CMD $REPROCESS_ANNOTATIONS"
[ -n "$CONFIG_PATH" ]           && CMD="$CMD --config $CONFIG_PATH"
[ -n "$MROS_VISIT" ]            && CMD="$CMD --mros-visit $MROS_VISIT"
CMD="$CMD --log-level $LOG_LEVEL"

echo "Running: $CMD"
echo ""

# Run Python in background so USR1 can interrupt 'wait' immediately
set +e
eval "$CMD" &
_PYTHON_PID=$!
wait $_PYTHON_PID
EXIT_CODE=$?
trap '' USR1   # disarm after Python finishes normally

echo ""
echo "========================================================================"
echo "End time:      $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Status:        SUCCESS"
else
    echo "Status:        FAILED (exit code: $EXIT_CODE)"
fi
echo "========================================================================"

deactivate
exit $EXIT_CODE
