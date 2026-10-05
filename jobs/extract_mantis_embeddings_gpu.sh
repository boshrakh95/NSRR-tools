#!/bin/bash
#SBATCH --job-name=mantis_emb
#SBATCH --account=def-egranger_gpu
#SBATCH --time=08:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16000M
#SBATCH --exclude=fc11006,fc11013,fc11010
#SBATCH --signal=B:USR1@120
#SBATCH --output=/home/boshra95/NSRR-tools-mantis/logs_mantis/embeddings_%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools-mantis/logs_mantis/embeddings_%x_%j.err

# Mantis baseline — Stage 1 Step 11 — Embedding extraction (GPU)
#
# Forked from jobs/extract_physioomni_embeddings_gpu.sh (itself forked from
# jobs/extract_osf_embeddings_gpu.sh) — same --start-idx/--end-idx sharding
# pattern and SIGUSR1 auto-resume mechanism as the OSF/PhysioOmni/SleepFM
# extraction jobs; only the venv, target script, output paths, job name,
# and the Mantis-specific --gpu-fraction flag differ.
#
# WHY THIS FILE DIDN'T EXIST UNTIL NOW (checklist 1.4's explicit note,
# docs/TSFM_MANTIS_IMPLEMENTATION_PLAN.md): the real jobs/*.sh production
# scripts are written only once the step-by-step implementation (dataset +
# train + infer + registry, checklist 1.7-1.10) is finished and validated
# via CPU debug/launch.json — never written piecemeal mid-implementation.
# That point is now — checklist 1.7-1.10 are all done and smoke-tested.
# This is a real, submittable production script, not a debug tool.
#
# Total subjects: ~14,994 across apples(1104) + shhs(8444) + mros(3933) +
# stages(1513) — same population as OSF's/SleepFM's full population (Mantis
# reads the same fast-channel psg/ HDF5s SleepFM/PhysioOmni use, and unlike
# PhysioOmni, Mantis DOES need stages for apnea_binary — it has a RESP
# pathway, plan §2.2/§5.8).
#
# REAL measured GPU throughput (Pilot 3 + Pilot 1/2's 100-subject run,
# 2026-09-06/07, H100 MIG 1g.10gb slice, Option D windowing — the decided
# production config): **~8.0s/subject**. At that rate the full ~14,994-subject
# population is ~33.3h serial on one GPU slice — shard into parallel jobs
# (examples below) rather than running it in one job. This is a real
# measured number (not a placeholder copy from another model — Mantis's
# per-token compute is genuinely different, ~19x OSF's/PhysioOmni's per
# plan §4.8, since it batches all 6 channels through ONE encoder per
# subject rather than looking up a single embedding).
#
# gpu-fraction: the extraction script's achieved-TFLOP/s percentage-of-peak
# instrumentation (plan §4.1) needs to know what slice of the card it
# actually has, since nvidia-smi cannot be trusted to report a MIG slice's
# true size on this cluster (plan §4's "nvidia-smi MIG-memory-misreport"
# lesson) — 1g.10gb = 1/7 of an H100, the script's own default, forwarded
# explicitly below for clarity rather than relying on the default silently
# matching whatever --gpus= is requested above.
#
# RECOMMENDED: shard into parallel GPU jobs (subject order = concatenated
# apples+shhs+mros+stages list per configs/phase0_mantis_config.yaml's
# `embedding.datasets` order, sliced globally):
#   sbatch --export=ALL,START=0,END=2500       jobs/extract_mantis_embeddings_gpu.sh
#   sbatch --export=ALL,START=2500,END=5000    jobs/extract_mantis_embeddings_gpu.sh
#   sbatch --export=ALL,START=5000,END=7500    jobs/extract_mantis_embeddings_gpu.sh
#   sbatch --export=ALL,START=7500,END=9600    jobs/extract_mantis_embeddings_gpu.sh
#   sbatch --export=ALL,START=9600,END=12500   jobs/extract_mantis_embeddings_gpu.sh
#   sbatch --export=ALL,START=12500,END=15000  jobs/extract_mantis_embeddings_gpu.sh
#
# Or single job (for testing / small subject counts):
#   sbatch --export=ALL,END=50 jobs/extract_mantis_embeddings_gpu.sh
#
# Or restrict to one dataset (e.g. finish stages last, since only
# apnea_binary needs it):
#   sbatch --export=ALL,DATASETS="stages" jobs/extract_mantis_embeddings_gpu.sh
#
# START / END default to full dataset if not set.
# Already-extracted .npy files are skipped automatically (safe to
# re-submit/shard however you like) unless NO_SKIP=1 is set — output
# filenames are per-subject ({output_dir}/{dataset}/{subject_id}.npy), so
# concurrent shards can never overwrite each other's files as long as their
# START:END ranges don't overlap. Writes are also atomic (temp file +
# os.replace(), verified 2026-09-07) — a killed/timed-out job can never
# leave a truncated .npy that a later resume would mistake for "done".

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
    echo "ERROR: CUDA not available. Cancel and resubmit with --exclude=$SLURM_NODELIST"
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
echo "Mantis Embedding Extraction — Stage 1 Step 11"
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
        | grep -oP 'TimeLimit=\K\S+' || echo "07:00:00")
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
