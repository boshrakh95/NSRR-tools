#!/bin/bash
#SBATCH --job-name=find_bs
#SBATCH --account=def-forouzan_gpu
#SBATCH --partition=gpubase_bygpu_b3
#SBATCH --time=00:20:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000M
#SBATCH --output=/home/boshra95/NSRR-tools/logs_v3/%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools/logs_v3/%x_%j.err

# Phase 0 — Batch Size Probe (rorqual-specific)
#
# Probes the largest batch size that fits in GPU memory for each context length.
# Writes {exp_dir}/batch_sizes.json, which gen_commands.py train (memory_bounded
# mode) reads automatically to set per-context BATCH_SIZE.
#
# Generate this sbatch command with:
#   python scripts/gen_commands.py probe-batch <exp_id>
#
# Direct usage:
#   HEAD=transformer EXP_DIR=/scratch/.../sleep_efficiency_binary_transformer_membnd \
#   CONFIG=configs/phase0_v3_config.yaml \
#   sbatch jobs/find_batch_size_gpu_rorqual.sh
#
# Optional overrides (export before sbatch or pass via --export=ALL,...):
#   HEAD                  head architecture: lstm | transformer | mean_pool
#   EXP_DIR               output directory for batch_sizes.json
#   CONFIG                path to phase0_config.yaml
#   CONTEXTS              space-separated context lengths (default: all standard ones)
#   NUM_CLASSES           number of output classes (default: 2)
#   STARTING_BATCH_SIZE   largest batch to try first (default: 256)

set -e
cd /home/boshra95/NSRR-tools
mkdir -p logs_v3

module load python/3.11 2>/dev/null || true
source /home/boshra95/sleepfm_env/bin/activate
export PYTHONPATH="/home/boshra95/sleepfm-clinical:/home/boshra95/sleepfm-clinical/sleepfm:$PYTHONPATH"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available on $SLURM_NODELIST'" || {
    echo "ERROR: CUDA not available"
    exit 1
}

# ── Job parameters ────────────────────────────────────────────────────────────
HEAD=${HEAD:-"lstm"}
EXP_DIR=${EXP_DIR:-""}
CONFIG=${CONFIG:-"configs/phase0_v3_config.yaml"}
CONTEXTS=${CONTEXTS:-""}
NUM_CLASSES=${NUM_CLASSES:-2}
STARTING_BATCH_SIZE=${STARTING_BATCH_SIZE:-256}

echo "========================================================================"
echo "Phase 0 — Batch Size Probe"
echo "========================================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Head:      $HEAD"
echo "Config:    $CONFIG"
echo "Exp dir:   $EXP_DIR"
echo "Start:     $(date)"
echo "========================================================================"
echo ""

CMD="python scripts/find_batch_size.py --config $CONFIG --head $HEAD"
[ -n "$EXP_DIR"              ] && CMD="$CMD --exp-dir $EXP_DIR"
[ -n "$CONTEXTS"             ] && CMD="$CMD --contexts $CONTEXTS"
[ -n "$NUM_CLASSES"          ] && CMD="$CMD --num-classes $NUM_CLASSES"
[ -n "$STARTING_BATCH_SIZE"  ] && CMD="$CMD --starting-batch-size $STARTING_BATCH_SIZE"

echo "Running: $CMD"
echo ""
eval "$CMD"

echo ""
echo "========================================================================"
echo "End time: $(date)"
echo "Status: SUCCESS"
echo "========================================================================"

deactivate
