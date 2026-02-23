#!/bin/bash
#SBATCH --account=def-forouzan
#SBATCH --job-name=extract_channels
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000M
#SBATCH --time=06:00:00
#SBATCH --mail-user=boshra.khajehpiri.1@ens.etsmtl.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/boshra95/NSRR-tools/output/channel_analysis/logs/extract_channels_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools/output/channel_analysis/logs/extract_channels_%j.err

# Print job info
echo "========================================"
echo "SLURM Job: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo "========================================"
echo "Extracting channels from datasets: stages shhs apples mros"
echo "Output directory: /home/boshra95/NSRR-tools/output/channel_analysis"
echo "========================================"
echo ""

# Activate environment
source /home/boshra95/NSRR-tools/.venv/bin/activate

# Change to NSRR-tools directory
cd /home/boshra95/NSRR-tools

# Run channel extraction for all datasets
echo "Starting channel extraction..."
uv run python scripts/extract_nsrr_channels.py \
    --datasets stages shhs apples mros \
    --output-dir output/channel_analysis

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Channel extraction completed successfully!"
    echo "End time: $(date)"
    echo "========================================"
    echo ""
    echo "Results saved in:"
    echo "  - Per-dataset CSVs: output/channel_analysis/<dataset>_channels.csv"
    echo "  - All unique channels: output/channel_analysis/all_unique_channels.txt"
    echo "  - Frequency analysis: output/channel_analysis/channel_frequency.json"
else
    echo ""
    echo "========================================"
    echo "ERROR: Channel extraction failed!"
    echo "End time: $(date)"
    echo "========================================"
    exit 1
fi
