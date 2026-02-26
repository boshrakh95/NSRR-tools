#!/bin/bash
#SBATCH --job-name=meta_mros
#SBATCH --time=60:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/extract_metadata_mros_%j.log
#SBATCH --error=logs/extract_metadata_mros_%j.err

# MrOS metadata extraction (slowest - ~58 hours)

cd /home/boshra95/NSRR-tools
source .venv/bin/activate

echo "Starting MrOS metadata extraction..."
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python scripts/extract_metadata.py --datasets mros --force --no-cache

echo "End time: $(date)"
echo "MrOS metadata extraction complete!"
