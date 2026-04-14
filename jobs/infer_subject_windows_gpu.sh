#!/bin/bash
#SBATCH --job-name=infer_windows
#SBATCH --account=def-forouzan_gpu
#SBATCH --time=00:10:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --exclude=fc11006
#SBATCH --output=/home/boshra95/NSRR-tools/logs/infer_%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools/logs/infer_%x_%j.err

# Phase 0 — Subject-level inference (all windows)
#
# Loads a trained checkpoint and runs inference on ALL non-overlapping windows
# per subject (no K=5 cap).  Saves a parquet of per-window probabilities for
# downstream majority-voting / mean-prob aggregation.
#
# Usage examples:
#   # Single context:
#   sbatch --export=ALL,TASK=apnea_binary,TASK_TYPE=seq2label,HEAD=lstm,CONTEXTS="10m" \
#       jobs/infer_subject_windows_gpu.sh
#
#   # Multiple contexts in one job (already-done are skipped automatically):
#   sbatch --export=ALL,TASK=apnea_binary,TASK_TYPE=seq2label,HEAD=lstm,CONTEXTS="30s 10m 40m 80m" \
#       jobs/infer_subject_windows_gpu.sh
#
#   # With dataset filter:
#   sbatch --export=ALL,TASK=cvd_binary,TASK_TYPE=seq2label,HEAD=lstm,CONTEXTS="30s 10m 40m",DATASETS="shhs mros apples" \
#       jobs/infer_subject_windows_gpu.sh
#
#   # Run on val split instead of test:
#   sbatch --export=ALL,...,SPLIT=val jobs/infer_subject_windows_gpu.sh
#
#   # Reproduce training eval exactly (K=5 windows, no --all-windows):
#   sbatch --export=ALL,...,NO_ALL_WINDOWS=1 jobs/infer_subject_windows_gpu.sh
#
# Time guide (all-windows, test split, H100):
#   30s context: ~9.5M items → ~20 min
#   10m context: ~475k items → ~3  min
#   40m context: ~120k items → <1  min
#   Multi-context job: sum of individual times + set --time accordingly

set -e

cd /home/boshra95/NSRR-tools
mkdir -p logs

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11 2>/dev/null || true

source /home/boshra95/sleepfm_env/bin/activate

export PYTHONPATH="/home/boshra95/sleepfm-clinical:/home/boshra95/sleepfm-clinical/sleepfm:$PYTHONPATH"

# Fail fast if CUDA is not available
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available on node $SLURM_NODELIST'" || {
    echo "ERROR: CUDA not available. Cancel and resubmit with --exclude=$SLURM_NODELIST"
    exit 1
}

# ── Job parameters ────────────────────────────────────────────────────────────
CONFIG="configs/phase0_config.yaml"
TASK=${TASK:-""}
TASK_TYPE=${TASK_TYPE:-"seq2label"}
HEAD=${HEAD:-"lstm"}
CONTEXTS=${CONTEXTS:-""}
SPLIT=${SPLIT:-"test"}
DATASETS=${DATASETS:-""}
NO_ALL_WINDOWS=${NO_ALL_WINDOWS:-""}   # set to 1 to use K=5 (training eval mode)
BATCH_SIZE=${BATCH_SIZE:-512}
RUN_TAG=${RUN_TAG:-""}                 # must match RUN_TAG used during training

echo "========================================================================"
echo "Phase 0 — Subject-level inference (all windows)"
echo "========================================================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Task:        ${TASK}  type=${TASK_TYPE}"
echo "Head:        ${HEAD}"
echo "Contexts:    ${CONTEXTS}"
echo "Split:       ${SPLIT}"
echo "Datasets:    ${DATASETS:-'(all)'}"
echo "All windows: ${NO_ALL_WINDOWS:-yes}$([ -n "$NO_ALL_WINDOWS" ] && echo 'no (K=5)')"
echo "Start:       $(date)"
echo "========================================================================"
echo ""

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python scripts/infer_subject_windows.py --config $CONFIG"
[ -n "$TASK"           ] && CMD="$CMD --task $TASK"
[ -n "$TASK_TYPE"      ] && CMD="$CMD --task-type $TASK_TYPE"
[ -n "$HEAD"           ] && CMD="$CMD --head $HEAD"
[ -n "$CONTEXTS"       ] && CMD="$CMD --context $CONTEXTS"
[ -n "$SPLIT"          ] && CMD="$CMD --split $SPLIT"
[ -n "$DATASETS"       ] && CMD="$CMD --datasets $DATASETS"
[ -n "$NO_ALL_WINDOWS" ] && CMD="$CMD --no-all-windows"
[ -n "$RUN_TAG"        ] && CMD="$CMD --run-tag $RUN_TAG"
CMD="$CMD --batch-size $BATCH_SIZE"

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
