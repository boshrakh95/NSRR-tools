#!/bin/bash
#SBATCH --job-name=mantis_emb
#SBATCH --account=def-egranger_gpu
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --time=08:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16000M
#SBATCH --signal=B:USR1@120
#SBATCH --output=/home/boshra95/NSRR-tools-mantis/logs_mantis/embeddings_%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools-mantis/logs_mantis/embeddings_%x_%j.err

# Mantis baseline — Stage 1 Step 11 — Embedding extraction (GPU) (rorqual-specific)
#
# Identical to extract_mantis_embeddings_gpu.sh except:
#   - --partition=gpubase_bygpu_b3 (required on rorqual; not valid on Fir)
#   - --exclude removed (Fir fc* node names don't exist on rorqual)
#
# See extract_mantis_embeddings_gpu.sh for the full rationale (real
# measured GPU throughput ~8.0s/subject, atomic writes, sharding
# convention, --gpu-fraction reasoning) — not repeated here.
#
# RECOMMENDED: shard into parallel GPU jobs, same population/order as the
# Fir script (subject order = concatenated apples+shhs+mros+stages list
# per configs/phase0_mantis_config.yaml's embedding.datasets order):
#   sbatch --export=ALL,START=0,END=2500       jobs/extract_mantis_embeddings_gpu_rorqual.sh
#   sbatch --export=ALL,START=2500,END=5000    jobs/extract_mantis_embeddings_gpu_rorqual.sh
#   sbatch --export=ALL,START=5000,END=7500    jobs/extract_mantis_embeddings_gpu_rorqual.sh
#   sbatch --export=ALL,START=7500,END=9600    jobs/extract_mantis_embeddings_gpu_rorqual.sh
#   sbatch --export=ALL,START=9600,END=12500   jobs/extract_mantis_embeddings_gpu_rorqual.sh
#   sbatch --export=ALL,START=12500,END=15000  jobs/extract_mantis_embeddings_gpu_rorqual.sh
#
# ⚠️ RUNNING ON BOTH CLUSTERS IN PARALLEL: --start-idx/--end-idx shard the
# SAME concatenated subject list on both Fir and Rorqual (both read the
# same configs/phase0_mantis_config.yaml, same HDF5 directory listing
# order) — pick DISJOINT [START:END) ranges across the two clusters, the
# same way you would across multiple shards on one cluster, so no subject
# is ever double-submitted. Output filenames are per-subject and atomic
# either way, so an accidental overlap would just be wasted compute, not
# a correctness problem — but disjoint ranges avoid the waste.
#
# Or single job (for testing / small subject counts):
#   sbatch --export=ALL,END=50 jobs/extract_mantis_embeddings_gpu_rorqual.sh
#
# Or restrict to one dataset:
#   sbatch --export=ALL,DATASETS="stages" jobs/extract_mantis_embeddings_gpu_rorqual.sh

set -e

# Store absolute path early — needed for resubmission
_SCRIPT_PATH="$(realpath "$0")"
_PYTHON_PID=""

cd /home/boshra95/NSRR-tools-mantis
LOGS_DIR=${LOGS_DIR:-logs_mantis}
mkdir -p "$LOGS_DIR"

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.10.13 2>/dev/null || true

source /home/boshra95/mantis_env/bin/activate

# Fail fast if CUDA is not available
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available on node $SLURM_NODELIST'" || {
    echo "ERROR: CUDA not available. Cancel and resubmit — check node health with sinfo."
    exit 1
}

# ── Job parameters ────────────────────────────────────────────────────────────
CONFIG=${CONFIG:-"configs/phase0_mantis_config.yaml"}
START=${START:-0}
END_IDX=${END:-""}      # empty = process to end of list
DATASETS=${DATASETS:-""}
NO_SKIP=${NO_SKIP:-""}  # set to 1 to re-extract even if .npy exists
GPU_FRACTION=${GPU_FRACTION:-"0.142857"}  # 1/7 of an H100 — matches the 1g.10gb slice requested above

echo "========================================================================"
echo "Mantis Embedding Extraction — Stage 1 Step 11 (rorqual)"
echo "========================================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Config:     $CONFIG"
echo "Subject range: [$START : ${END_IDX:-end}]"
echo "Datasets:   ${DATASETS:-'(all in config)'}"
echo "GPU fraction: $GPU_FRACTION"
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
CMD="python scripts/extract_mantis_embeddings.py --config $CONFIG --start-idx $START --gpu-fraction $GPU_FRACTION"
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
