#!/bin/bash
#SBATCH --job-name=osf_raw_signal_cache
#SBATCH --account=def-forouzan
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000M
#SBATCH --signal=B:USR1@120
#SBATCH --output=/home/boshra95/NSRR-tools/logs_osf_lora/precompute_cache_%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools/logs_osf_lora/precompute_cache_%x_%j.err

# OSF baseline — Stage 2 (LoRA) — Raw signal cache precompute (checklist 2.5b)
#
# CPU-ONLY (--account=def-forouzan, no GPU requested) — this is pure I/O +
# channel-resample work, no model, so it never touches the GPU allocation.
# See scripts/precompute_osf_raw_signal_cache.py's module docstring for the
# full motivation: this eliminates the redundant raw-HDF5 reads that were
# making every Stage 2 training job spend hours before printing "Epoch 1".
#
# Auto-resume on timeout: same mechanism as preprocess_signals_parallel.sh —
#   --signal=B:USR1@120 fires 120s before wall time, Python is stopped
#   cleanly, this script resubmits itself via sbatch --export=ALL. Already-
#   cached subjects are skipped automatically on the next run.
#
# Usage — sharded full run (~15,000 subjects across apples/shhs/mros/stages):
#   sbatch --export=ALL,START=0,END=4000        jobs/precompute_osf_raw_signal_cache.sh
#   sbatch --export=ALL,START=4000,END=8000     jobs/precompute_osf_raw_signal_cache.sh
#   sbatch --export=ALL,START=8000,END=12000    jobs/precompute_osf_raw_signal_cache.sh
#   sbatch --export=ALL,START=12000,END=15100   jobs/precompute_osf_raw_signal_cache.sh
#
# Usage — single dataset:
#   sbatch --export=ALL,DATASETS=apples jobs/precompute_osf_raw_signal_cache.sh
#
# Usage — small test:
#   sbatch --export=ALL,END=50 jobs/precompute_osf_raw_signal_cache.sh
#
# START/END default to the full subject list. Already-cached .npy files are
# skipped automatically (safe to re-submit) unless NO_SKIP=1.

set -e

_SCRIPT_PATH="$(realpath "$0")"
_PYTHON_PID=""

cd /home/boshra95/NSRR-tools
LOGS_DIR=${LOGS_DIR:-logs_osf_lora}
mkdir -p "$LOGS_DIR"

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.10.13 2>/dev/null || true
source /home/boshra95/osf_env/bin/activate

# Unbuffered stdout — see jobs/train_osf_lora_gpu.sh for why this matters.
export PYTHONUNBUFFERED=1

# ── Job parameters ────────────────────────────────────────────────────────────
CONFIG=${CONFIG:-"configs/phase0_osf_lora_config.yaml"}
START=${START:-0}
END_IDX=${END:-""}
DATASETS=${DATASETS:-""}
NO_SKIP=${NO_SKIP:-""}
NUM_WORKERS=${NUM_WORKERS:-16}

echo "========================================================================"
echo "OSF baseline — Stage 2 raw signal cache precompute"
echo "========================================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Config:        $CONFIG"
echo "Subject range: [$START : ${END_IDX:-end}]"
echo "Datasets:      ${DATASETS:-'(all in config)'}"
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
CMD="python scripts/precompute_osf_raw_signal_cache.py --config $CONFIG --start-idx $START --num-workers $NUM_WORKERS"
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
exit $EXIT_CODE
