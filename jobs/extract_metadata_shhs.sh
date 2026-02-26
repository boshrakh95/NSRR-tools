#!/bin/bash
#SBATCH --job-name=meta_shhs
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000M
#SBATCH --account=def-forouzan
#SBATCH --output=logs/extract_metadata_shhs_%j.log
#SBATCH --error=logs/extract_metadata_shhs_%j.err

# SHHS metadata extraction (~45 min)

cd /home/boshra95/NSRR-tools
source .venv/bin/activate

echo "Starting SHHS metadata extraction..."
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

python scripts/extract_metadata.py --datasets shhs --force --no-cache 

echo "End time: $(date)"
echo "SHHS metadata extraction complete!"
