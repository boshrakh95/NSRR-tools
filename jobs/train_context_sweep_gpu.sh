#!/bin/bash
#SBATCH --job-name=ctx_sweep
#SBATCH --account=def-forouzan_gpu
#SBATCH --time=24:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --exclude=fc11006,fc11013,fc11010
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
#   # Filter to specific datasets (space-separated, quoted):
#   sbatch --export=ALL,TASK=sleep_staging,HEAD=lstm,DATASETS="shhs1 shhs2" jobs/train_context_sweep_gpu.sh
#
#   # Disable W&B for a run:
#   sbatch --export=ALL,...,NO_WANDB=1 jobs/train_context_sweep_gpu.sh
#
# W&B setup: store your API key in ~/.wandb_key (chmod 600).
#   The script loads it automatically — no interactive prompts.
#
# Or single default run (uses task/head from phase0_config.yaml):
#   sbatch jobs/train_context_sweep_gpu.sh

set -e

cd /home/boshra95/NSRR-tools
mkdir -p logs

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11 2>/dev/null || true
# module load cuda 2>/dev/null || true

source /home/boshra95/sleepfm_env/bin/activate

export PYTHONPATH="/home/boshra95/sleepfm-clinical:/home/boshra95/sleepfm-clinical/sleepfm:$PYTHONPATH"

# Fail fast if CUDA is not available — avoids silent CPU fallback
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available on node $SLURM_NODELIST'" || {
    echo "ERROR: CUDA not available. Cancel and resubmit with --exclude=$SLURM_NODELIST"
    exit 1
}

# ── W&B setup (non-interactive) ───────────────────────────────────────────────
# Store your key once: echo "your_key_here" > ~/.wandb_key && chmod 600 ~/.wandb_key
[ -f ~/.wandb_key ] && export WANDB_API_KEY=$(cat ~/.wandb_key)
export WANDB_DIR=/tmp/wandb_${SLURM_JOB_ID}   # node-local tmp, auto-cleaned
mkdir -p "$WANDB_DIR"

# ── Job parameters ────────────────────────────────────────────────────────────
CONFIG="configs/phase0_config.yaml"
TASK=${TASK:-""}            # empty = use config default
TASK_TYPE=${TASK_TYPE:-""}  # empty = use config default
HEAD=${HEAD:-""}            # empty = use config default
CONTEXT=${CONTEXT:-""}      # single context length, e.g. "30s" or "10m"
DATASETS=${DATASETS:-""}    # space-separated dataset names, e.g. "shhs mros"
BATCH_SIZE=${BATCH_SIZE:-""}  # training batch size (default: 32); reduce for long contexts
LR=${LR:-""}                  # learning rate override, e.g. LR=1e-4
RUN_TAG=${RUN_TAG:-""}        # suffix for experiment folder, e.g. RUN_TAG=lr1e4
WANDB_PROJECT=${WANDB_PROJECT:-"nsrr-phase0"}
NO_WANDB=${NO_WANDB:-""}

echo "========================================================================"
echo "Phase 0 Step 4 — Context-Length Sweep"
echo "========================================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Task:      ${TASK:-'(from config)'}  type=${TASK_TYPE:-'(from config)'}"
echo "Head:      ${HEAD:-'(from config)'}"
echo "Datasets:  ${DATASETS:-'(all)'}"
echo "W&B:       ${NO_WANDB:+disabled}${NO_WANDB:-project=$WANDB_PROJECT}"
echo "Start:     $(date)"
echo "========================================================================"
echo ""

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python scripts/train_context_sweep.py --config $CONFIG"
[ -n "$TASK"           ] && CMD="$CMD --task $TASK"
[ -n "$TASK_TYPE"      ] && CMD="$CMD --task-type $TASK_TYPE"
[ -n "$HEAD"           ] && CMD="$CMD --head $HEAD"
[ -n "$CONTEXT"        ] && CMD="$CMD --context $CONTEXT"
[ -n "$DATASETS"       ] && CMD="$CMD --datasets $DATASETS"
[ -n "$BATCH_SIZE"     ] && CMD="$CMD --batch-size $BATCH_SIZE"
[ -n "$LR"             ] && CMD="$CMD --lr $LR"
[ -n "$RUN_TAG"        ] && CMD="$CMD --run-tag $RUN_TAG"
[ -n "$WANDB_PROJECT"  ] && CMD="$CMD --wandb-project $WANDB_PROJECT"
[ -n "$NO_WANDB"       ] && CMD="$CMD --no-wandb"

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
