#!/bin/bash
# ==============================================================================
# Extract Channel Names from All NSRR Datasets
# ==============================================================================
# This script runs channel extraction on the compute cluster for all datasets
# 
# Usage:
#   bash run_extract_channels_cluster.sh
#
# Output will be saved to: /home/boshra95/NSRR-tools/output/channel_analysis/
# ==============================================================================

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# SLURM settings
ACCOUNT="def-forouzan"
JOB_NAME="extract_channels"
CPUS=8
MEM="16000M"
TIME="06:00:00"
MAIL_USER="boshra.khajehpiri.1@ens.etsmtl.ca"

# Paths
NSRR_TOOLS_DIR="/home/boshra95/NSRR-tools"
OUTPUT_DIR="${NSRR_TOOLS_DIR}/output/channel_analysis"
LOG_DIR="${OUTPUT_DIR}/logs"
ENV_PATH="/home/boshra95/NSRR-tools/.venv"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Datasets to process
DATASETS="stages shhs apples mros"

# ==============================================================================
# CREATE AND SUBMIT JOB
# ==============================================================================

JOB_SCRIPT="${LOG_DIR}/extract_channels_job.sh"

echo "Creating SLURM job script: ${JOB_SCRIPT}"

cat > "$JOB_SCRIPT" <<'EOF'
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
EOF

# Submit the job
echo ""
echo "Submitting job to SLURM..."
sbatch "$JOB_SCRIPT"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Job submitted successfully!"
    echo "========================================"
    echo "Monitor job status with: squeue -u $USER"
    echo "View output logs in: ${LOG_DIR}/"
    echo "Results will be saved to: ${OUTPUT_DIR}/"
else
    echo ""
    echo "ERROR: Job submission failed!"
    exit 1
fi
