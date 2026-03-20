#!/bin/bash
#SBATCH --job-name=ctx_sweep
#SBATCH --account=def-forouzan_gpu
#SBATCH --time=08:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --output=/home/boshra95/NSRR-tools/logs/sweep_%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools/logs/sweep_%x_%j.err

# Phase 0 Step 4 — Context-Length Sweep Training
#
# Trains one (task, head_type) combination across all context lengths.
# Each context length takes ~10-30 min; 7 lengths × up to 30 epochs = ~2-4h.
# Already-finished context lengths are skipped automatically (safe to resubmit).
#
# Usage examples:
#   sbatch --export=ALL,TASK=apnea_binary,HEAD=lstm        jobs/train_context_sweep_gpu.sh
#   sbatch --export=ALL,TASK=apnea_binary,HEAD=mean_pool   jobs/train_context_sweep_gpu.sh
#   sbatch --export=ALL,TASK=apnea_binary,HEAD=transformer jobs/train_context_sweep_gpu.sh
#   sbatch --export=ALL,TASK=sleep_staging,HEAD=lstm       jobs/train_context_sweep_gpu.sh
#
# Or single default run (uses task/head from phase0_config.yaml):
#   sbatch jobs/train_context_sweep_gpu.sh

set -e

cd /home/boshra95/NSRR-tools
mkdir -p logs

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11 2>/dev/null || true

source /home/boshra95/sleepfm_env/bin/activate

export PYTHONPATH="/home/boshra95/sleepfm-clinical:/home/boshra95/sleepfm-clinical/sleepfm:$PYTHONPATH"

# ── Job parameters ────────────────────────────────────────────────────────────
CONFIG="configs/phase0_config.yaml"
TASK=${TASK:-""}           # empty = use config default
TASK_TYPE=${TASK_TYPE:-""} # empty = use config default
HEAD=${HEAD:-""}           # empty = use config default

echo "========================================================================"
echo "Phase 0 Step 4 — Context-Length Sweep"
echo "========================================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Task:      ${TASK:-'(from config)'}  type=${TASK_TYPE:-'(from config)'}"
echo "Head:      ${HEAD:-'(from config)'}"
echo "Start:     $(date)"
echo "========================================================================"
echo ""

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python scripts/train_context_sweep.py --config $CONFIG"
[ -n "$TASK"      ] && CMD="$CMD --task $TASK"
[ -n "$TASK_TYPE" ] && CMD="$CMD --task-type $TASK_TYPE"
[ -n "$HEAD"      ] && CMD="$CMD --head $HEAD"

echo "Running: $CMD"
echo ""
eval $CMD

EXIT_CODE=$?

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
