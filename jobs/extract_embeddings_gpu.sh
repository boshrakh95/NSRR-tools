#!/bin/bash
#SBATCH --job-name=sleepfm_emb
#SBATCH --account=def-forouzan_gpu
#SBATCH --time=04:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --signal=B:USR1@120
#SBATCH --output=/home/boshra95/NSRR-tools/logs_v3_full/embeddings_%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools/logs_v3_full/embeddings_%x_%j.err

# Extract SleepFM embeddings — Phase 0 Step 1
#
# Total subjects: ~15,000 across apples(1104) + shhs(8444) + mros(3933) + stages(1513)
# Estimated GPU time (H100 MIG): ~1-3s/subject after warmup → ~2-3h per 2500-subject job
#
# Subject order matches phase0_v3_full_config.yaml datasets list:
#   apples(0-1103), shhs(1104-9547), mros(9548-13480), stages(13481-14993)
#
# RECOMMENDED: 6 parallel GPU jobs (add CONFIG=... for the full-channel run):
#   sbatch --export=ALL,START=0,END=2500,CONFIG=configs/phase0_v3_full_config.yaml      jobs/extract_embeddings_gpu.sh
#   sbatch --export=ALL,START=2500,END=5000,CONFIG=configs/phase0_v3_full_config.yaml   jobs/extract_embeddings_gpu.sh
#   sbatch --export=ALL,START=5000,END=7500,CONFIG=configs/phase0_v3_full_config.yaml   jobs/extract_embeddings_gpu.sh
#   sbatch --export=ALL,START=7500,END=9600,CONFIG=configs/phase0_v3_full_config.yaml   jobs/extract_embeddings_gpu.sh
#   sbatch --export=ALL,START=9600,END=12500,CONFIG=configs/phase0_v3_full_config.yaml  jobs/extract_embeddings_gpu.sh
#   sbatch --export=ALL,START=12500,END=15100,CONFIG=configs/phase0_v3_full_config.yaml jobs/extract_embeddings_gpu.sh
#
# Or single job (for testing / if GPU queue wait is long):
#   sbatch jobs/extract_embeddings_gpu.sh
#
# START / END default to full dataset if not set.
# Already-extracted .npy files are skipped automatically (safe to re-submit).

set -e

# Store absolute path early — needed for resubmission
_SCRIPT_PATH="$(realpath "$0")"
_PYTHON_PID=""

cd /home/boshra95/NSRR-tools
mkdir -p logs_v3_full

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11 2>/dev/null || true

source /home/boshra95/sleepfm_env/bin/activate

export PYTHONPATH="/home/boshra95/sleepfm-clinical:/home/boshra95/sleepfm-clinical/sleepfm:$PYTHONPATH"

# ── Job parameters ────────────────────────────────────────────────────────────
# CONFIG can be overridden via env variable, e.g.:
#   CONFIG=configs/phase0_v3_full_config.yaml sbatch --export=ALL,CONFIG=...,START=0,END=2500 jobs/extract_embeddings_gpu.sh
CONFIG=${CONFIG:-"configs/phase0_v3_config.yaml"}
START=${START:-0}
END_IDX=${END:-""}      # empty = process to end of list

echo "========================================================================"
echo "SleepFM Embedding Extraction — Phase 0 Step 1"
echo "========================================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Config:     $CONFIG"
echo "Subject range: [$START : ${END_IDX:-end}]"
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
CMD="python scripts/extract_sleepfm_embeddings.py --config $CONFIG --start-idx $START"
if [ -n "$END_IDX" ]; then
    CMD="$CMD --end-idx $END_IDX"
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
