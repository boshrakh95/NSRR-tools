#!/bin/bash
#SBATCH --job-name=meta_apples
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=8000M
#SBATCH --account=def-forouzan
#SBATCH --output=logs/extract_metadata_apples_%j.log
#SBATCH --error=logs/extract_metadata_apples_%j.err

# APPLES metadata extraction (~2 min)

cd /home/boshra95/NSRR-tools
source .venv/bin/activate

echo "Starting APPLES metadata extraction..."
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python scripts/extract_metadata.py --datasets apples --force --no-cache

echo "End time: $(date)"
echo "APPLES metadata extraction complete!"
