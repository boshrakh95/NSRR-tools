#!/bin/bash
#SBATCH --job-name=physioomni_raw_signal_cache
#SBATCH --account=def-forouzan
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000M
#SBATCH --signal=B:USR1@120
#SBATCH --output=/home/boshra95/NSRR-tools-omni/logs_physioomni/precompute_cache_%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools-omni/logs_physioomni/precompute_cache_%x_%j.err

# PhysioOmni baseline — Stage 2 (LoRA) — Raw signal cache precompute (plan §15.3)
#
# CPU-ONLY (--account=def-forouzan, no GPU requested) — pure I/O + channel-
# resample work, no model, never touches a GPU allocation. Forked from
# jobs/precompute_osf_raw_signal_cache.sh's exact CPU-only pattern. See
# scripts/precompute_physioomni_raw_signal_cache.py's module docstring for
# the full motivation: precomputing this once avoids every Stage 2 training
# job redoing live HDF5-read + FFT-resample work on every __getitem__ (the
# root cause of a real 2+-hour GPU stall OSF's own Stage 2 hit).
#
# Auto-resume on timeout: --signal=B:USR1@120 fires 120s before wall time,
# Python is stopped cleanly, this script resubmits itself via sbatch
# --export=ALL. Already-cached subjects are skipped automatically on the
# next run.
#
# Usage — sharded run across apples+shhs+mros (13,481 subjects — NOT
# stages, see the python script's module docstring for why it's excluded
# by default):
#   sbatch --export=ALL,START=0,END=4500        jobs/precompute_physioomni_raw_signal_cache.sh
#   sbatch --export=ALL,START=4500,END=9000     jobs/precompute_physioomni_raw_signal_cache.sh
#   sbatch --export=ALL,START=9000,END=13481    jobs/precompute_physioomni_raw_signal_cache.sh
#
# Usage — single dataset:
#   sbatch --export=ALL,DATASETS=apples jobs/precompute_physioomni_raw_signal_cache.sh
#
# Usage — small test:
#   sbatch --export=ALL,END=50 jobs/precompute_physioomni_raw_signal_cache.sh
#
# START/END default to the full subject list. Already-cached subjects are
# skipped automatically (safe to re-submit) unless NO_SKIP=1.
#
# Wall-time NOT calibrated — PhysioOmni's FFT-based scipy.signal.resample
# (needed for 128->200/500Hz, unlike OSF's exact 128->64Hz 2:1 decimation)
# is plausibly slower per-sample than OSF's own precompute (1104 APPLES
# subjects in 5.1 min at --num-workers 8). Time the first real shard
# before assuming a total budget for the rest (plan §15.3).

set -e

_SCRIPT_PATH="$(realpath "$0")"
_PYTHON_PID=""

cd /home/boshra95/NSRR-tools-omni
LOGS_DIR=${LOGS_DIR:-logs_physioomni}
mkdir -p "$LOGS_DIR"

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.10.13 2>/dev/null || true
source /home/boshra95/physioomni_env/bin/activate

export PYTHONUNBUFFERED=1

# ── Job parameters ────────────────────────────────────────────────────────────
CONFIG=${CONFIG:-"configs/phase0_physioomni_lora_config.yaml"}
START=${START:-0}
END_IDX=${END:-""}
DATASETS=${DATASETS:-""}
NO_SKIP=${NO_SKIP:-""}
NUM_WORKERS=${NUM_WORKERS:-16}

echo "========================================================================"
echo "PhysioOmni baseline — Stage 2 raw signal cache precompute"
echo "========================================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Config:        $CONFIG"
echo "Subject range: [$START : ${END_IDX:-end}]"
echo "Datasets:      ${DATASETS:-'(default: apples shhs mros)'}"
echo "Workers:       $NUM_WORKERS"
echo "Start time:    $(date)"
echo "========================================================================"
echo ""

# ── Auto-resume trap (fires 120s before wall time) ────────────────────────────
_timeout_handler() {
    echo ""
    echo "[USR1] Time limit approaching — stopping and resubmitting ($(date))"
    [ -n "$_PYTHON_PID" ] && kill -TERM "$_PYTHON_PID" 2>/dev/null || true
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

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python scripts/precompute_physioomni_raw_signal_cache.py --config $CONFIG --start-idx $START --num-workers $NUM_WORKERS"
[ -n "$END_IDX"    ] && CMD="$CMD --end-idx $END_IDX"
[ -n "$DATASETS"   ] && CMD="$CMD --datasets $DATASETS"
[ -n "$NO_SKIP"    ] && CMD="$CMD --no-skip"

echo "Running: $CMD"
echo ""

set +e
eval "$CMD" &
_PYTHON_PID=$!
wait $_PYTHON_PID
EXIT_CODE=$?
trap '' USR1

echo ""
echo "========================================================================"
echo "End time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS"
else
    echo "Status: FAILED (exit code: $EXIT_CODE)"
fi
echo "========================================================================"

deactivate
