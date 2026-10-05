#!/bin/bash
#SBATCH --job-name=mantis_raw_signal_cache
#SBATCH --account=def-egranger
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000M
#SBATCH --signal=B:USR1@120
#SBATCH --output=/home/boshra95/NSRR-tools-mantis/logs_mantis_lora/precompute_cache_%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools-mantis/logs_mantis_lora/precompute_cache_%x_%j.err

# Mantis baseline — Stage 2 (LoRA) — Raw signal cache precompute (checklist 2.2)
#
# CPU-ONLY (--account=def-forouzan, no GPU requested) — pure I/O + reshape
# work, no model, so it never touches a GPU allocation. See
# scripts/precompute_mantis_raw_signal_cache.py's module docstring for the
# full motivation. Unlike OSF's/PhysioOmni's own precompute jobs, Mantis
# needs NO resampling (already 128 Hz) — real small-scale measurement
# (2026-09-07, 3 real APPLES subjects): ~5.3 subjects/s single-process,
# substantially faster than either sibling's FFT/decimation-based
# precompute. Time the first real sharded job before trusting a full
# 07:00:00 wall-time budget across the ~14,994-subject population — this
# default is a starting point, not yet calibrated at scale.
#
# --time=04:00:00 is a starting guess for a ~2500-subject shard at the
# measured single-process rate with 16 workers; auto-resume means an
# underestimate just costs one extra resubmission, not lost work.
#
# BEFORE THE FULL RUN: check `diskusage_report` — the full cache is
# ~720GB across ~14,994 subjects (plan §14.3; confirmed 2026-09-07 that
# /scratch has 8442/19000 GiB used, comfortable headroom, but re-check
# immediately before building since usage changes). Build cohort by
# cohort so a quota surprise is recoverable, not discovered mid-shard.
#
# Auto-resume on timeout: same mechanism as extract_mantis_embeddings_gpu.sh —
#   --signal=B:USR1@120 fires 120s before wall time, Python is stopped
#   cleanly, this script resubmits itself via sbatch --export=ALL. Already-
#   cached subjects are skipped automatically on the next run (cache_exists()
#   parses meta.json, so a subject killed mid-write is correctly retried,
#   not mistaken for done).
#
# Usage — sharded full run (~14,994 subjects across apples/shhs/mros/stages,
# same population as Stage 1's extraction — Mantis needs stages for
# apnea_binary):
#   sbatch --export=ALL,START=0,END=2500       jobs/precompute_mantis_raw_signal_cache.sh
#   sbatch --export=ALL,START=2500,END=5000    jobs/precompute_mantis_raw_signal_cache.sh
#   sbatch --export=ALL,START=5000,END=7500    jobs/precompute_mantis_raw_signal_cache.sh
#   sbatch --export=ALL,START=7500,END=9600    jobs/precompute_mantis_raw_signal_cache.sh
#   sbatch --export=ALL,START=9600,END=12500   jobs/precompute_mantis_raw_signal_cache.sh
#   sbatch --export=ALL,START=12500,END=15000  jobs/precompute_mantis_raw_signal_cache.sh
#
# Usage — single dataset:
#   sbatch --export=ALL,DATASETS=apples jobs/precompute_mantis_raw_signal_cache.sh
#
# Usage — small test:
#   sbatch --export=ALL,END=50 jobs/precompute_mantis_raw_signal_cache.sh
#
# START/END default to the full subject list. Already-cached subjects are
# skipped automatically (safe to re-submit) unless NO_SKIP=1.

set -e

_SCRIPT_PATH="$(realpath "$0")"
_PYTHON_PID=""

cd /home/boshra95/NSRR-tools-mantis
LOGS_DIR=${LOGS_DIR:-logs_mantis_lora}
mkdir -p "$LOGS_DIR"

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.10.13 2>/dev/null || true
source /home/boshra95/mantis_env/bin/activate

# Unbuffered stdout — same reasoning as OSF's/PhysioOmni's own precompute
# jobs: without this, tee/log output can lag well behind actual progress.
export PYTHONUNBUFFERED=1

# ── Job parameters ────────────────────────────────────────────────────────────
CONFIG=${CONFIG:-"configs/phase0_mantis_lora_config.yaml"}
START=${START:-0}
END_IDX=${END:-""}
DATASETS=${DATASETS:-""}
NO_SKIP=${NO_SKIP:-""}
NUM_WORKERS=${NUM_WORKERS:-16}

echo "========================================================================"
echo "Mantis baseline — Stage 2 raw signal cache precompute"
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
CMD="python scripts/precompute_mantis_raw_signal_cache.py --config $CONFIG --start-idx $START --num-workers $NUM_WORKERS"
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
