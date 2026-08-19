#!/bin/bash
#SBATCH --job-name=physioomni_emb
#SBATCH --account=def-forouzan_gpu
#SBATCH --time=04:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16000M
#SBATCH --exclude=fc11006,fc11013,fc11010
#SBATCH --signal=B:USR1@120
#SBATCH --output=/home/boshra95/NSRR-tools-omni/logs_physioomni/embeddings_%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools-omni/logs_physioomni/embeddings_%x_%j.err

# PhysioOmni baseline — Phase 1 Step 9 — Embedding extraction (GPU)
#
# Forked from jobs/extract_osf_embeddings_gpu.sh — see
# docs/TSFM_PHYSIOOMNI_IMPLEMENTATION_PLAN.md checklist item 1.9. Same
# --start-idx/--end-idx sharding pattern and SIGUSR1 auto-resume mechanism
# as the OSF/SleepFM extraction jobs; only the venv, target script, output
# paths, and job name differ. Fir cluster only (no rorqual variant exists
# yet).
#
# Total subjects: ~14,994 across apples(1104) + shhs(8444) + mros(3933) +
# stages(1513) — same population as OSF's/SleepFM's full population, since
# PhysioOmni reads the same fast-channel psg/ HDF5s SleepFM uses (not
# psg_full/, unlike OSF — PhysioOmni needs only EEG/EOG/ECG/EMG, no RESP).
# Per-subject GPU cost IS calibrated (checklist 1.9, 2026-08-18, real H100
# MIG 1g.10gb slice): ~4.1s/subject, vs. ~50-450s/subject on CPU — roughly
# 15-100x faster. Measured on apples+shhs; mros/stages not yet tested but
# expected similar (same channel-loading path).
#
# chunk_batch_size WAS A/B-tested on real GPU (checklist 1.9, 2026-08-18):
# 16 vs 64, matched 20-subject shhs batches — 4.05s/subject vs 4.15s/subject,
# no meaningful difference. Unlike OSF (where this knob was its real
# Stage-2 bottleneck, 16->64 gave a measured 3.28x speedup — see CLAUDE.md's
# OSF section), PhysioOmni extraction here is not chunk_batch_size-bound.
# Kept at 16 (the config default). Real measured throughput: ~4.1s/subject
# on a single H100 MIG 1g.10gb slice — ~15,000 subjects total would take
# ~17 hours serial on one GPU, so shard into parallel jobs (see below)
# rather than running the full population in one job.
#
# RECOMMENDED: shard into parallel GPU jobs once per-subject GPU cost is
# known (same subject-order convention as SleepFM's/OSF's scripts, since
# all three read datasets in their own config's dataset order):
#   sbatch --export=ALL,START=0,END=2500       jobs/extract_physioomni_embeddings_gpu.sh
#   sbatch --export=ALL,START=2500,END=5000    jobs/extract_physioomni_embeddings_gpu.sh
#   sbatch --export=ALL,START=5000,END=7500    jobs/extract_physioomni_embeddings_gpu.sh
#   sbatch --export=ALL,START=7500,END=9600    jobs/extract_physioomni_embeddings_gpu.sh
#   sbatch --export=ALL,START=9600,END=12500   jobs/extract_physioomni_embeddings_gpu.sh
#   sbatch --export=ALL,START=12500,END=15000  jobs/extract_physioomni_embeddings_gpu.sh
#
# Or single job (for testing / small subject counts):
#   sbatch --export=ALL,END=50 jobs/extract_physioomni_embeddings_gpu.sh
#
# START / END default to full dataset if not set.
# Already-extracted .npy files are skipped automatically (safe to re-submit)
# unless NO_SKIP=1 is set.

set -e

# Store absolute path early — needed for resubmission
_SCRIPT_PATH="$(realpath "$0")"
_PYTHON_PID=""

cd /home/boshra95/NSRR-tools-omni
LOGS_DIR=${LOGS_DIR:-logs_physioomni}
mkdir -p "$LOGS_DIR"

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.10.13 2>/dev/null || true

source /home/boshra95/physioomni_env/bin/activate

# Fail fast if CUDA is not available
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available on node $SLURM_NODELIST'" || {
    echo "ERROR: CUDA not available. Cancel and resubmit with --exclude=$SLURM_NODELIST"
    exit 1
}

# ── Job parameters ────────────────────────────────────────────────────────────
CONFIG=${CONFIG:-"configs/phase0_physioomni_config.yaml"}
START=${START:-0}
END_IDX=${END:-""}      # empty = process to end of list
DATASETS=${DATASETS:-""}
NO_SKIP=${NO_SKIP:-""}  # set to 1 to re-extract even if .npy exists

echo "========================================================================"
echo "PhysioOmni Embedding Extraction — Phase 1 Step 9"
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
CMD="python scripts/extract_physioomni_embeddings.py --config $CONFIG --start-idx $START"
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
