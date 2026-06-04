#!/bin/bash
#SBATCH --job-name=preprocess_batch
#SBATCH --account=def-forouzan
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=30000M
#SBATCH --output=logs/preprocess_batch_%x_%j.out
#SBATCH --error=logs/preprocess_batch_%x_%j.err

# Batch preprocessing: process a subject index slice of one dataset.
# Use this when a dataset is too large for a single 26-hour job (e.g. SHHS=8444 subjects).
# Safe to re-submit — subjects whose HDF5 already exists are skipped by default.
#
# Usage:
#   sbatch jobs/preprocess_signals_array.sh <dataset> <start> <end> [options]
#
#   dataset:  stages | shhs | apples | mros  (required)
#   start:    0-based inclusive start index   (required)
#   end:      exclusive end index             (required)
#
# Options (all optional):
#   --config PATH          Path to preprocessing config yaml (default: preprocessing_params.yaml)
#   --no-skip-existing     Re-process subjects even if HDF5 already exists
#   --reprocess-annotations  Re-extract annotations only (keeps existing HDF5)
#   --log-level LEVEL      DEBUG | INFO | WARNING | ERROR (default: INFO)
#   --mros-visit 1|2       MrOS only: which visit to process
#
# SHHS example — split 8444 subjects into 6 jobs of ~1400 each:
#   sbatch jobs/preprocess_signals_array.sh shhs    0  1500 --config configs/preprocessing_params_full.yaml
#   sbatch jobs/preprocess_signals_array.sh shhs 1500  3000 --config configs/preprocessing_params_full.yaml
#   sbatch jobs/preprocess_signals_array.sh shhs 3000  4500 --config configs/preprocessing_params_full.yaml
#   sbatch jobs/preprocess_signals_array.sh shhs 4500  6000 --config configs/preprocessing_params_full.yaml
#   sbatch jobs/preprocess_signals_array.sh shhs 6000  7500 --config configs/preprocessing_params_full.yaml
#   sbatch jobs/preprocess_signals_array.sh shhs 7500  9000 --config configs/preprocessing_params_full.yaml
#
# Smaller datasets fit in one job — use preprocess_signals_parallel.sh for those.

set -e

# ── Required positional arguments ─────────────────────────────────────────────
DATASET=${1:?"Usage: sbatch $0 <dataset> <start_index> <end_index> [options]"}
START_INDEX=${2:?"Usage: sbatch $0 <dataset> <start_index> <end_index> [options]"}
END_INDEX=${3:?"Usage: sbatch $0 <dataset> <start_index> <end_index> [options]"}

# ── Optional arguments ────────────────────────────────────────────────────────
CONFIG_PATH=""
SKIP_EXISTING="--skip-existing"
REPROCESS_ANNOTATIONS=""
LOG_LEVEL="INFO"
MROS_VISIT=""

shift 3
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"; shift 2 ;;
        --no-skip-existing)
            SKIP_EXISTING=""; shift ;;
        --skip-existing)
            SKIP_EXISTING="--skip-existing"; shift ;;
        --reprocess-annotations)
            REPROCESS_ANNOTATIONS="--reprocess-annotations"; shift ;;
        --log-level)
            LOG_LEVEL="$2"; shift 2 ;;
        --mros-visit)
            MROS_VISIT="--mros-visit $2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Setup ─────────────────────────────────────────────────────────────────────
cd /home/boshra95/NSRR-tools
mkdir -p logs

source .venv/bin/activate

echo "========================================================================"
echo "NSRR Signal Preprocessing — Batch Job"
echo "========================================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "Dataset:       $DATASET"
echo "Index range:   [$START_INDEX, $END_INDEX)   (~$((END_INDEX - START_INDEX)) subjects)"
echo "Config:        ${CONFIG_PATH:-configs/preprocessing_params.yaml (default)}"
echo "Skip existing: $([ -n "$SKIP_EXISTING" ] && echo 'Yes' || echo 'No')"
echo "Log level:     $LOG_LEVEL"
echo "Start time:    $(date)"
echo "========================================================================"
echo ""

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python scripts/preprocess_signals.py"
CMD="$CMD --dataset $DATASET"
CMD="$CMD --start-index $START_INDEX"
CMD="$CMD --end-index $END_INDEX"
[ -n "$SKIP_EXISTING" ]           && CMD="$CMD $SKIP_EXISTING"
[ -n "$REPROCESS_ANNOTATIONS" ]   && CMD="$CMD $REPROCESS_ANNOTATIONS"
[ -n "$CONFIG_PATH" ]             && CMD="$CMD --config $CONFIG_PATH"
[ -n "$MROS_VISIT" ]              && CMD="$CMD $MROS_VISIT"
CMD="$CMD --log-level $LOG_LEVEL"

echo "Running: $CMD"
echo ""

eval $CMD
EXIT_CODE=$?

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
