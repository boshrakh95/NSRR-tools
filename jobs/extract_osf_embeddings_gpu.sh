#!/bin/bash
#SBATCH --job-name=osf_emb
#SBATCH --account=def-forouzan_gpu
#SBATCH --time=04:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16000M
#SBATCH --exclude=fc11006
#SBATCH --signal=B:USR1@120
#SBATCH --output=/home/boshra95/NSRR-tools/logs_osf/embeddings_%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools/logs_osf/embeddings_%x_%j.err

# OSF baseline — Stage 1 Step 1 — Embedding extraction (GPU)
#
# Forked from jobs/extract_embeddings_gpu.sh — see
# docs/TSFM_OSF_IMPLEMENTATION_PLAN.md checklist item 1.8. Same
# --start-idx/--end-idx sharding pattern and SIGUSR1 auto-resume mechanism
# as the SleepFM extraction job; only the venv, target script, output paths,
# and job name differ. Fir cluster only (no rorqual variant exists yet).
#
# Total subjects: ~15,000 across apples(1104) + shhs(8444) + mros(3933) +
# stages(1513) — same population as SleepFM's full-channel extraction,
# since OSF reuses the same HDF5s (psg_full/) as source signals.
# Per-subject cost is NOT YET CALIBRATED for OSF on GPU (only CPU-debugged
# with --limit 2/--limit 10 so far, ~200-400s/subject on CPU) — start with
# a small --end-idx test job before committing to the full sharded run.
#
# RECOMMENDED: shard into parallel GPU jobs once per-subject GPU cost is
# known (same subject-order convention as SleepFM's script, since both
# read datasets in phase0_osf_config.yaml's dataset order):
#   sbatch --export=ALL,START=0,END=2500       jobs/extract_osf_embeddings_gpu.sh
#   sbatch --export=ALL,START=2500,END=5000    jobs/extract_osf_embeddings_gpu.sh
#   sbatch --export=ALL,START=5000,END=7500    jobs/extract_osf_embeddings_gpu.sh
#   sbatch --export=ALL,START=7500,END=9600    jobs/extract_osf_embeddings_gpu.sh
#   sbatch --export=ALL,START=9600,END=12500   jobs/extract_osf_embeddings_gpu.sh
#   sbatch --export=ALL,START=12500,END=15100  jobs/extract_osf_embeddings_gpu.sh
#
# Or single job (for testing / small subject counts):
#   sbatch --export=ALL,END=50 jobs/extract_osf_embeddings_gpu.sh
#
# START / END default to full dataset if not set.
# Already-extracted .npy files are skipped automatically (safe to re-submit)
# unless NO_SKIP=1 is set.

set -e

# Store absolute path early — needed for resubmission
_SCRIPT_PATH="$(realpath "$0")"
_PYTHON_PID=""

cd /home/boshra95/NSRR-tools
LOGS_DIR=${LOGS_DIR:-logs_osf}
mkdir -p "$LOGS_DIR"

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.10.13 2>/dev/null || true

source /home/boshra95/osf_env/bin/activate

# Fail fast if CUDA is not available
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available on node $SLURM_NODELIST'" || {
    echo "ERROR: CUDA not available. Cancel and resubmit with --exclude=$SLURM_NODELIST"
    exit 1
}

# ── Job parameters ────────────────────────────────────────────────────────────
CONFIG=${CONFIG:-"configs/phase0_osf_config.yaml"}
START=${START:-0}
END_IDX=${END:-""}      # empty = process to end of list
DATASETS=${DATASETS:-""}
NO_SKIP=${NO_SKIP:-""}  # set to 1 to re-extract even if .npy exists

echo "========================================================================"
echo "OSF Embedding Extraction — Stage 1 Step 1"
echo "========================================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Config:     $CONFIG"
echo "Subject range: [$START : ${END_IDX:-end}]"
echo "Datasets:   ${DATASETS:-'(all in config)'}"
echo "Start time: $(date)"
echo "========================================================================"
echo ""

# ── Auto-resume trap (fires 120 s before wall time) ───────────────────────────
_timeout_handler() {
    echo ""
    echo "[USR1] Time limit approaching — stopping Python and resubmitting ($(date))"
    [ -n "$_PYTHON_PID" ] && kill -TERM "$_PYTHON_PID" 2>/dev/null || true
    # Wait for Python to finish the current subject cleanly (SIGTERM handler in Python)
    wait "$_PYTHON_PID" 2>/dev/null || true
    _TIME_LIMIT=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
        | grep -oP 'TimeLimit=\K\S+' || echo "04:00:00")
    NEW_JOB=$(sbatch \
        --export=ALL \
        --time="$_TIME_LIMIT" \
        "$_SCRIPT_PATH" 2>&1)
    echo "$NEW_JOB"
    exit 0
}
trap '_timeout_handler' USR1

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python scripts/extract_osf_embeddings.py --config $CONFIG --start-idx $START"
if [ -n "$END_IDX" ]; then
    CMD="$CMD --end-idx $END_IDX"
fi
if [ -n "$DATASETS" ]; then
    CMD="$CMD --datasets $DATASETS"
fi
if [ -n "$NO_SKIP" ]; then
    CMD="$CMD --no-skip"
fi

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
echo "End time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS"
else
    echo "Status: FAILED (exit code: $EXIT_CODE)"
fi
echo "========================================================================"

deactivate
exit $EXIT_CODE
