#!/bin/bash
# Submit all metadata extraction jobs in parallel

cd /home/boshra95/NSRR-tools

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Submitting metadata extraction jobs..."
echo ""

# Submit all jobs
job1=$(sbatch jobs/extract_metadata_mros.sh | awk '{print $4}')
echo "MrOS:   Job ID $job1 (est. 58 hours)"

job2=$(sbatch jobs/extract_metadata_shhs.sh | awk '{print $4}')
echo "SHHS:   Job ID $job2 (est. 45 min)"

job3=$(sbatch jobs/extract_metadata_apples.sh | awk '{print $4}')
echo "APPLES: Job ID $job3 (est. 2 min)"

job4=$(sbatch jobs/extract_metadata_stages.sh | awk '{print $4}')
echo "STAGES: Job ID $job4 (est. 10 min)"

echo ""
echo "All jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "Check logs in: logs/"
