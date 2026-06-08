#!/bin/bash
#SBATCH --job-name=eval_ckpt
#SBATCH --account=def-forouzan_gpu
#SBATCH --time=00:30:00
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --exclude=fc11006
#SBATCH --output=/home/boshra95/NSRR-tools/logs_v3_full/eval_ckpt_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools/logs_v3_full/eval_ckpt_%j.err

set -e
cd /home/boshra95/NSRR-tools

module load python/3.11 2>/dev/null || true
source /home/boshra95/sleepfm_env/bin/activate
export PYTHONPATH="/home/boshra95/sleepfm-clinical:/home/boshra95/sleepfm-clinical/sleepfm:$PYTHONPATH"

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'" || exit 1

CONFIG=${CONFIG:-"configs/phase0_v3_full_config.yaml"}
TASK=${TASK:-"sex_binary"}
HEAD=${HEAD:-"lstm"}
CONTEXTS=${CONTEXTS:-"30s 10m 40m"}
DATASETS=${DATASETS:-"apples shhs"}

echo "========================================"
echo "Eval checkpoint  task=$TASK  head=$HEAD"
echo "Contexts: $CONTEXTS"
echo "Datasets: $DATASETS"
echo "Start: $(date)"
echo "========================================"
echo ""

# shellcheck disable=SC2086
python scripts/eval_checkpoint.py \
    --config "$CONFIG" \
    --task "$TASK" \
    --head "$HEAD" \
    --contexts $CONTEXTS \
    --datasets $DATASETS \
    --batch-size 512

echo ""
echo "End: $(date)"
deactivate
