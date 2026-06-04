#!/bin/bash
#SBATCH --job-name=preprocess_batch
#SBATCH --account=def-forouzan
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000M
#SBATCH --signal=B:USR1@120
#SBATCH --output=logs/preprocess_batch_%x_%j.out
#SBATCH --error=logs/preprocess_batch_%x_%j.err

# Batch preprocessing: process a subject index slice of one dataset.
# Use for large datasets (e.g. SHHS=8444) that exceed a single 26-hour job.
#
# Auto-resume on timeout (same mechanism as train_context_sweep_gpu.sh):
#   --signal=B:USR1@120 fires USR1 to bash 120 s before wall time.
#   Python finishes the current subject (SIGTERM handler), then exits.
#   This script resubmits itself with --export=ALL so DATASET, START_INDEX,
#   END_INDEX, CONFIG_PATH, etc. are all forwarded to the new job.
#   The new job skips already-written HDF5 files and continues.
#   Works on Alliance Canada — no --requeue needed.
#
# Usage — preferred (env vars, safe for auto-resubmission):
#   DATASET=shhs START_INDEX=0 END_INDEX=1500 \
#       CONFIG_PATH=configs/preprocessing_params_full.yaml \
#       sbatch jobs/preprocess_signals_array.sh
#
# Usage — positional-arg style (also supported):
#   sbatch jobs/preprocess_signals_array.sh <dataset> <start> <end> [options]
#   sbatch jobs/preprocess_signals_array.sh shhs 0 1500 --config configs/preprocessing_params_full.yaml
#   sbatch jobs/preprocess_signals_array.sh shhs 1500 3000 --config configs/preprocessing_params_full.yaml
#
# SHHS example — split 8444 subjects across 6 parallel jobs of ~1400 each:
#   CFG=--config configs/preprocessing_params_full.yaml
#   sbatch jobs/preprocess_signals_array.sh shhs    0  1500 $CFG
#   sbatch jobs/preprocess_signals_array.sh shhs 1500  3000 $CFG
#   sbatch jobs/preprocess_signals_array.sh shhs 3000  4500 $CFG
#   sbatch jobs/preprocess_signals_array.sh shhs 4500  6000 $CFG
#   sbatch jobs/preprocess_signals_array.sh shhs 6000  7500 $CFG
#   sbatch jobs/preprocess_signals_array.sh shhs 7500  9000 $CFG

set -e

_SCRIPT_PATH="$(realpath "$0")"
_PYTHON_PID=""

# ── Parameters — env var takes precedence, then positional arg, then default ──
DATASET=${DATASET:-${1:?"Usage: sbatch $0 <dataset> <start_index> <end_index> [options]"}}
START_INDEX=${START_INDEX:-${2:?"Usage: sbatch $0 <dataset> <start_index> <end_index> [options]"}}
END_INDEX=${END_INDEX:-${3:?"Usage: sbatch $0 <dataset> <start_index> <end_index> [options]"}}
CONFIG_PATH=${CONFIG_PATH:-""}
SKIP_EXISTING=${SKIP_EXISTING:-"--skip-existing"}
REPROCESS_ANNOTATIONS=${REPROCESS_ANNOTATIONS:-""}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}
MROS_VISIT=${MROS_VISIT:-""}

# Parse remaining positional options (if called in legacy style)
# Shift past dataset/start/end only when they came from positional args
_nshift=0
[ "${1:-}" = "$DATASET" ]     && _nshift=$((_nshift+1))
[ "${2:-}" = "$START_INDEX" ] && _nshift=$((_nshift+1))
[ "${3:-}" = "$END_INDEX" ]   && _nshift=$((_nshift+1))
for _ in $(seq 1 $_nshift 2>/dev/null); do shift 2>/dev/null || true; done
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
export DATASET START_INDEX END_INDEX CONFIG_PATH SKIP_EXISTING REPROCESS_ANNOTATIONS LOG_LEVEL MROS_VISIT

# ── Setup ─────────────────────────────────────────────────────────────────────
cd /home/boshra95/NSRR-tools
mkdir -p logs

source .venv/bin/activate

# ── Auto-resume trap (fires 120 s before wall time) ───────────────────────────
_timeout_handler() {
    echo ""
    echo "[USR1] Time limit approaching — stopping Python and resubmitting ($(date))"
    [ -n "$_PYTHON_PID" ] && kill -TERM "$_PYTHON_PID" 2>/dev/null || true
    # Wait for Python to finish the current subject cleanly (SIGTERM handler in Python)
    wait "$_PYTHON_PID" 2>/dev/null || true
    _TIME_LIMIT=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
        | grep -oP 'TimeLimit=\K\S+' || echo "08:00:00")
    NEW_JOB=$(sbatch \
        --export=ALL \
        --time="$_TIME_LIMIT" \
        "$_SCRIPT_PATH" 2>&1)
    echo "$NEW_JOB"
    exit 0
}
trap '_timeout_handler' USR1

echo "========================================================================"
echo "NSRR Signal Preprocessing — Batch Job"
echo "========================================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Dataset:       $DATASET"
echo "Index range:   [$START_INDEX, $END_INDEX)   (~$((END_INDEX - START_INDEX)) subjects)"
echo "Config:        ${CONFIG_PATH:-configs/preprocessing_params.yaml (default)}"
echo "Skip existing: $([ -n "$SKIP_EXISTING" ] && echo 'Yes' || echo 'No')"
[ -n "$MROS_VISIT" ] && echo "MrOS visit:    $MROS_VISIT"
echo "Start time:    $(date)"
echo "========================================================================"
echo ""

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python scripts/preprocess_signals.py --dataset $DATASET"
CMD="$CMD --start-index $START_INDEX --end-index $END_INDEX"
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
