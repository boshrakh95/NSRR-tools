#!/bin/bash
#SBATCH --job-name=test_emb
#SBATCH --account=def-forouzan_gpu
#SBATCH --time=00:30:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4 
#SBATCH --mem=16000M
#SBATCH --output=/home/boshra95/NSRR-tools/logs/test_embeddings_%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools/logs/test_embeddings_%x_%j.err

# Test SleepFM embedding extraction on 2 subjects per dataset.
# Runs all four datasets sequentially so a crash on any one is clearly visible.
# Use --no-skip so it always re-runs even if .npy already exists.
#
# Usage:
#   sbatch jobs/test_embeddings_gpu.sh

set -e   # stop on first error so failures are obvious

cd /home/boshra95/NSRR-tools
mkdir -p logs

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11 2>/dev/null || true

source /home/boshra95/sleepfm_env/bin/activate

export PYTHONPATH="/home/boshra95/sleepfm-clinical:/home/boshra95/sleepfm-clinical/sleepfm:$PYTHONPATH"

CONFIG="configs/phase0_config.yaml"
PY="python scripts/extract_sleepfm_embeddings.py --config $CONFIG --limit 2 --no-skip"

echo "========================================================================"
echo "SleepFM Embedding Extraction — GPU TEST (2 subjects × 4 datasets)"
echo "========================================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time: $(date)"
echo "========================================================================"

# ── APPLES ────────────────────────────────────────────────────────────────────
echo ""
echo ">>> [1/4] APPLES"
$PY --datasets apples
echo "    APPLES OK"

# ── MrOS ─────────────────────────────────────────────────────────────────────
echo ""
echo ">>> [2/4] MrOS"
$PY --datasets mros
echo "    MrOS OK"

# ── SHHS ─────────────────────────────────────────────────────────────────────
echo ""
echo ">>> [3/4] SHHS"
$PY --datasets shhs
echo "    SHHS OK"

# ── STAGES ───────────────────────────────────────────────────────────────────
echo ""
echo ">>> [4/4] STAGES"
$PY --datasets stages
echo "    STAGES OK"

echo ""
echo "========================================================================"
echo "ALL DATASETS PASSED"
echo "End time: $(date)"
echo ""
echo "Output files written to:"
for ds in apples mros shhs stages; do
    ls /scratch/boshra95/psg/unified/embeddings/sleepfm_5sec/$ds/*.npy 2>/dev/null \
        | head -2 \
        | while read f; do
            shape=$(python3 -c "import numpy as np; a=np.load('$f'); print(a.shape)" 2>/dev/null)
            echo "  $f  shape=$shape"
          done
done
echo "========================================================================"

deactivate
