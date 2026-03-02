#!/bin/bash
#SBATCH --job-name=meta_all
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --account=def-forouzan
#SBATCH --output=logs/extract_metadata_all_%j.log
#SBATCH --error=logs/extract_metadata_all_%j.err

# Extract metadata for ALL datasets (STAGES, SHHS, APPLES, MROS)
# Creates one unified metadata file with all subjects

cd /home/boshra95/NSRR-tools
source .venv/bin/activate

echo "================================================================================"
echo "Starting unified metadata extraction for ALL datasets"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Datasets: STAGES, SHHS, APPLES, MROS"
echo ""

python scripts/extract_metadata.py --datasets stages shhs apples mros --force

echo ""
echo "================================================================================"
echo "End time: $(date)"
echo "Unified metadata extraction complete!"
echo "================================================================================"
echo ""
echo "Output location: /scratch/boshra95/psg/unified/metadata/unified_metadata.parquet"
echo ""
