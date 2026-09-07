#!/bin/bash
#SBATCH --job-name=physioomni_extract_cpu
#SBATCH --account=def-forouzan
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000M
#SBATCH --signal=B:USR1@120
#SBATCH --output=/home/boshra95/NSRR-tools-omni/logs_physioomni/extract_cpu_%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools-omni/logs_physioomni/extract_cpu_%x_%j.err

# CPU-only pilot/debug extraction job for PhysioOmni embeddings — for small
# subject counts where the GPU queue isn't worth waiting on (see
# jobs/extract_physioomni_embeddings_gpu.sh for the real bulk-extraction job,
# Phase 1.9). Forked from jobs/precompute_osf_raw_signal_cache.sh's CPU-only
# directive pattern (def-forouzan account, no _gpu suffix, 16 CPUs, 32GB).

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
CONFIG=${CONFIG:-"configs/phase0_physioomni_config.yaml"}
DATASETS=${DATASETS:-""}
LIMIT=${LIMIT:-""}
START=${START:-0}
END_IDX=${END:-""}
NO_SKIP=${NO_SKIP:-""}

echo "========================================================================"
echo "PhysioOmni — CPU pilot/debug embedding extraction"
echo "========================================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Config:        $CONFIG"
echo "Datasets:      ${DATASETS:-'(all in config)'}"
echo "Limit:         ${LIMIT:-'(none)'}"
echo "Subject range: [$START : ${END_IDX:-end}]"
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
        | grep -oP 'TimeLimit=\K\S+' || echo "02:00:00")
    NEW_JOB=$(sbatch \
        --export=ALL \
        --time="$_TIME_LIMIT" \
        "$_SCRIPT_PATH" 2>&1)
    echo "$NEW_JOB"
    exit 0
}
trap '_timeout_handler' USR1

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python scripts/extract_physioomni_embeddings.py --config $CONFIG --cpu --start-idx $START"
[ -n "$DATASETS" ] && CMD="$CMD --datasets $DATASETS"
[ -n "$LIMIT"    ] && CMD="$CMD --limit $LIMIT"
[ -n "$END_IDX"  ] && CMD="$CMD --end-idx $END_IDX"
[ -n "$NO_SKIP"  ] && CMD="$CMD --no-skip"

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
