#!/bin/bash
#SBATCH --job-name=debug_nan
#SBATCH --account=def-forouzan_gpu
#SBATCH --time=01:00:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --output=/home/boshra95/NSRR-tools/logs/debug_nan_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools/logs/debug_nan_%j.err

cd /home/boshra95/NSRR-tools
source /home/boshra95/sleepfm_env/bin/activate
export PYTHONPATH="/home/boshra95/sleepfm-clinical:/home/boshra95/sleepfm-clinical/sleepfm:$PYTHONPATH"
python scripts/debug_nan.py
