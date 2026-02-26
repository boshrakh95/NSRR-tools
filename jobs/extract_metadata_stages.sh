#!/bin/bash
#SBATCH --job-name=meta_stages
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --account=def-forouzan
#SBATCH --mem=8000M
#SBATCH --output=logs/extract_metadata_stages_%j.log
#SBATCH --error=logs/extract_metadata_stages_%j.err

# STAGES metadata extraction (~10 min)

cd /home/boshra95/NSRR-tools
source .venv/bin/activate

echo "Starting STAGES metadata extraction..."
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python scripts/extract_metadata.py --datasets stages --force --no-cache

echo "End time: $(date)"
echo "STAGES metadata extraction complete!"
