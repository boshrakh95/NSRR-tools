#!/bin/bash
#SBATCH --job-name=sleepfm_emb
#SBATCH --account=def-forouzan_gpu
#SBATCH --time=05:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --output=/home/boshra95/NSRR-tools/logs/embeddings_%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools/logs/embeddings_%x_%j.err

# Extract SleepFM embeddings — Phase 0 Step 1
#
# Total subjects: ~15,000 across apples(1104) + mros(3933) + shhs(8444) + stages(1513)
# Estimated GPU time (H100 MIG): ~1-3s/subject after warmup → ~2-3h per 2500-subject job
#
# RECOMMENDED: 6 parallel GPU jobs, each covering ~2500 subjects (4h limit)
# Subject order: apples(0-1103), mros(1104-5036), shhs(5037-13480), stages(13481-14993)
#
#   sbatch --export=ALL,START=0,END=2500      jobs/extract_embeddings_gpu.sh
#   sbatch --export=ALL,START=2500,END=5000   jobs/extract_embeddings_gpu.sh
#   sbatch --export=ALL,START=5000,END=7500   jobs/extract_embeddings_gpu.sh
#   sbatch --export=ALL,START=7500,END=10000  jobs/extract_embeddings_gpu.sh
#   sbatch --export=ALL,START=10000,END=12500 jobs/extract_embeddings_gpu.sh
#   sbatch --export=ALL,START=12500,END=15000 jobs/extract_embeddings_gpu.sh
#
# Or single job (for testing / if GPU queue wait is long):
#   sbatch jobs/extract_embeddings_gpu.sh
#
# START / END default to full dataset if not set.
# Already-extracted .npy files are skipped automatically (safe to re-submit).

set -e

cd /home/boshra95/NSRR-tools
mkdir -p logs

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11 2>/dev/null || true

source /home/boshra95/sleepfm_env/bin/activate

export PYTHONPATH="/home/boshra95/sleepfm-clinical:/home/boshra95/sleepfm-clinical/sleepfm:$PYTHONPATH"

# ── Job parameters ────────────────────────────────────────────────────────────
CONFIG="configs/phase0_config.yaml"
START=${START:-0}
END_IDX=${END:-""}      # empty = process to end of list

echo "========================================================================"
echo "SleepFM Embedding Extraction — Phase 0 Step 1"
echo "========================================================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURM_NODELIST"
echo "GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Subject range: [$START : ${END_IDX:-end}]"
echo "Start time: $(date)"
echo "========================================================================"
echo ""

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python scripts/extract_sleepfm_embeddings.py --config $CONFIG --start-idx $START"
if [ -n "$END_IDX" ]; then
    CMD="$CMD --end-idx $END_IDX"
fi

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
