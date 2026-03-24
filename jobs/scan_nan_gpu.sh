#!/bin/bash
#SBATCH --job-name=scan_nan
#SBATCH --account=def-forouzan_gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000M
#SBATCH --output=/home/boshra95/NSRR-tools/logs/scan_nan_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools/logs/scan_nan_%j.err

set -e

cd /home/boshra95/NSRR-tools
mkdir -p logs

module load python/3.11 2>/dev/null || true
source /home/boshra95/sleepfm_env/bin/activate
export PYTHONPATH="/home/boshra95/sleepfm-clinical:/home/boshra95/sleepfm-clinical/sleepfm:$PYTHONPATH"

echo "========================================================================"
echo "NaN Embedding Scanner"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo "========================================================================"
echo ""

python scripts/scan_nan_embeddings.py --workers 8

echo ""
echo "========================================================================"
echo "End time: $(date)"
echo "========================================================================"

deactivate
