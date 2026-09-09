#!/bin/bash
#SBATCH --job-name=mantis_ctx_sweep
#SBATCH --account=def-egranger_gpu
#SBATCH --time=05:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --signal=B:USR1@120            # send SIGUSR1 to bash 120s before wall time
#SBATCH --output=/home/boshra95/NSRR-tools-mantis/logs_mantis/%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools-mantis/logs_mantis/%x_%j.err

# Mantis baseline — Stage 1 Step 8 — Context-Length Sweep Training (nibi-specific)
#
# Differs from train_mantis_context_sweep_gpu.sh (Fir):
#   - --gpus=h100:1 (whole GPU), NOT the MIG 1g.10gb slice Fir uses — no MIG
#     slice naming was found documented for Nibi (unverified — see
#     jobs/test_gpu_setup_nibi.sh's header for why), so this asks for a
#     whole card rather than guessing a slice string that could be rejected.
#     Costs more of the allocation per job than Fir's MIG slice; revisit if
#     Nibi turns out to support MIG once confirmed.
#   - No --partition (no partition requirement found for Nibi, unlike
#     Rorqual's required gpubase_bygpu_b3).
#   - No --exclude (no documented bad-node list for Nibi, unlike Fir's
#     fc11006/fc11013/fc11010).
#   - W&B stays ONLINE (not forced offline like Rorqual) — Nibi compute
#     nodes have internet access, unlike Rorqual/Narval.
#
# ⚠️ RUN jobs/test_gpu_setup_nibi.sh FIRST if you haven't already — it
# verifies the GPU request/account/partition assumptions above actually
# work on this cluster before you commit a real training job to them.
#
# Trains one (task, head_type) combination across all context lengths.
# Already-finished context lengths are skipped automatically (safe to resubmit).
#
# Auto-resume on timeout:
#   --signal=B:USR1@120 sends SIGUSR1 to bash 120s before the wall-time limit.
#   Python runs in the background; 'wait' returns immediately on USR1, letting
#   the trap fire, kill Python cleanly, and resubmit this script via sbatch "$0".
#   The new job finds resume.pt (saved after every epoch) and continues training.
#   This works on Alliance Canada — no --requeue needed for timeout-based resume
#   (node-failure requeue is a separate mechanism, applied at the initial sbatch
#   invocation, not inside this script — same convention as OSF's/PhysioOmni's
#   job scripts).
#
# Usage examples:
#   sbatch --export=ALL,TASK=sex_binary,HEAD=lstm        jobs/train_mantis_context_sweep_gpu_nibi.sh
#   sbatch --export=ALL,TASK=sex_binary,HEAD=mean_pool    jobs/train_mantis_context_sweep_gpu_nibi.sh
#   sbatch --export=ALL,TASK=sex_binary,HEAD=transformer  jobs/train_mantis_context_sweep_gpu_nibi.sh
#
#   # Filter to specific datasets (space-separated, quoted):
#   sbatch --export=ALL,TASK=sex_binary,HEAD=lstm,DATASETS="apples shhs" jobs/train_mantis_context_sweep_gpu_nibi.sh
#
#   # Point at the Pilot 1/2 100-subject population instead of the production
#   # embedding_dir (useful before checklist 1.11's full extraction has run):
#   sbatch --export=ALL,TASK=sex_binary,HEAD=lstm,EMBEDDING_DIR=/scratch/boshra95/psg/unified/embeddings/mantis_pilot12/D_Llast_combined jobs/train_mantis_context_sweep_gpu_nibi.sh
#
#   # Disable W&B for a run:
#   sbatch --export=ALL,...,NO_WANDB=1 jobs/train_mantis_context_sweep_gpu_nibi.sh
#
# W&B setup: store your API key in ~/.wandb_key (chmod 600).
#   The script loads it automatically — no interactive prompts. Runs online
#   here (Nibi compute nodes have internet) — no offline/sync step needed.
#   Mantis runs default to the nsrr-phase0-mantis W&B project (kept separate
#   from SleepFM's nsrr-phase0 and OSF's/PhysioOmni's own projects), see
#   train_mantis_context_sweep.py. Check whether wandb is installed in
#   mantis_env before relying on it (same known CC Go-toolchain issue that
#   affected physioomni_env/osf_env) — use NO_WANDB=1 if not.
#
# Or single default run (uses task/head from phase0_mantis_config.yaml):
#   sbatch jobs/train_mantis_context_sweep_gpu_nibi.sh

set -e

# Store absolute path early — needed for resubmission from within the job
_SCRIPT_PATH="$(realpath "$0")"
_PYTHON_PID=""

cd /home/boshra95/NSRR-tools-mantis
LOGS_DIR=${LOGS_DIR:-logs_mantis}
mkdir -p "$LOGS_DIR"
mkdir -p "$LOGS_DIR/status"

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.10.13 2>/dev/null || true

source /home/boshra95/mantis_env/bin/activate

# Fail fast if CUDA is not available — avoids silent CPU fallback
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available on node $SLURM_NODELIST'" || {
    echo "ERROR: CUDA not available. Cancel and resubmit — check node health with sinfo."
    exit 1
}

# ── W&B setup (non-interactive) ───────────────────────────────────────────────
# Store your key once: echo "your_key_here" > ~/.wandb_key && chmod 600 ~/.wandb_key
[ -f ~/.wandb_key ] && export WANDB_API_KEY=$(cat ~/.wandb_key)
export WANDB_DIR=${SLURM_TMPDIR:-/tmp}/wandb_${SLURM_JOB_ID}    # $SLURM_TMPDIR = per-job local scratch (larger/more reliable than /tmp)
mkdir -p "$WANDB_DIR"

# ── Job parameters ────────────────────────────────────────────────────────────
CONFIG=${CONFIG:-"configs/phase0_mantis_config.yaml"}
TASK=${TASK:-""}            # empty = use config default
TASK_TYPE=${TASK_TYPE:-""}  # empty = use config default
HEAD=${HEAD:-""}            # empty = use config default
CONTEXT=${CONTEXT:-""}      # single context length, e.g. "30s" or "10m"
DATASETS=${DATASETS:-""}    # space-separated dataset names, e.g. "apples shhs"
EMBEDDING_DIR=${EMBEDDING_DIR:-""}  # override dataset.embedding_dir (e.g. Pilot 1/2 population)
BATCH_SIZE=${BATCH_SIZE:-""}   # micro-batch size fed to the GPU (default: 32)
ACCUM_STEPS=${ACCUM_STEPS:-1}  # gradient accumulation steps; effective_batch = BATCH_SIZE × ACCUM_STEPS
LR=${LR:-""}                   # learning rate override, e.g. LR=1e-4
RUN_TAG=${RUN_TAG:-""}        # suffix for experiment folder, e.g. RUN_TAG=lr1e4
WANDB_PROJECT=${WANDB_PROJECT:-"nsrr-phase0-mantis"}
NO_WANDB=${NO_WANDB:-""}

# ── Job run tracking ──────────────────────────────────────────────────────────
_EXP_TAG="${TASK}_${HEAD}"
[ -n "$RUN_TAG" ] && _EXP_TAG="${_EXP_TAG}_${RUN_TAG}"
_STATUS_FILE="$LOGS_DIR/status/train_${_EXP_TAG}_${CONTEXT:-nocontext}_lr${LR:-default}.jsonl"

# Persistent training log — all resubmissions append here so the full epoch
# history is in one place regardless of how many jobs the run takes.
_TRAIN_LOG="$LOGS_DIR/train_${_EXP_TAG}_${CONTEXT:-nocontext}_lr${LR:-default}.log"
exec > >(tee -a "$_TRAIN_LOG") 2>&1

_write_status() {
    local _status="$1"
    local _reason="${2:-}"
    if [ -n "$_reason" ]; then
        printf '{"ts":"%s","job_id":"%s","node":"%s","status":"%s","reason":"%s","task":"%s","head":"%s","context":"%s","lr":"%s","datasets":"%s"}\n' \
            "$(date -Iseconds)" "${SLURM_JOB_ID:-local}" \
            "${SLURM_NODELIST:-local}" "$_status" "$_reason" \
            "$TASK" "$HEAD" "${CONTEXT:-?}" "${LR:-default}" "${DATASETS:-all}" \
            >> "$_STATUS_FILE"
    else
        printf '{"ts":"%s","job_id":"%s","node":"%s","status":"%s","task":"%s","head":"%s","context":"%s","lr":"%s","datasets":"%s"}\n' \
            "$(date -Iseconds)" "${SLURM_JOB_ID:-local}" \
            "${SLURM_NODELIST:-local}" "$_status" \
            "$TASK" "$HEAD" "${CONTEXT:-?}" "${LR:-default}" "${DATASETS:-all}" \
            >> "$_STATUS_FILE"
    fi
}

# ── Auto-resume trap (SIGUSR1 fires 120s before wall time) ───────────────────
# Python runs in background so 'wait' returns immediately on signal.
# All env vars (TASK, HEAD, etc.) are already exported and will be forwarded
# via --export=ALL to the new job, which finds resume.pt and continues.
_timeout_handler() {
    _write_status "TIMEOUT_REQUEUED"
    echo ""
    echo "Time limit approaching — resubmitting for auto-resume ($(date))"
    [ -n "$_PYTHON_PID" ] && kill -TERM "$_PYTHON_PID" 2>/dev/null || true
    # Forward the same wall time so the resubmitted job doesn't fall back to
    # the script's 24h default.
    # Pass --output/--error explicitly so resubmitted jobs get the same
    # descriptive filename as the original submission (not the generic %x_%j fallback).
    _TIME_LIMIT=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
        | grep -oP 'TimeLimit=\K\S+' || echo "24:00:00")
    _LOG_STEM="${_TRAIN_LOG%.log}"
    NEW_JOB=$(sbatch \
        --export=ALL \
        --time="$_TIME_LIMIT" \
        --output="${_LOG_STEM}_%j.out" \
        --error="${_LOG_STEM}_%j.err" \
        "$_SCRIPT_PATH" 2>&1)
    echo "$NEW_JOB"
    exit 0
}
trap '_timeout_handler' USR1

_write_status "STARTED"

echo "========================================================================"
echo "Mantis baseline — Stage 1 Step 8 — Context-Length Sweep (nibi)"
echo "========================================================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Task:      ${TASK:-'(from config)'}  type=${TASK_TYPE:-'(from config)'}"
echo "Head:      ${HEAD:-'(from config)'}"
echo "Datasets:  ${DATASETS:-'(all)'}"
echo "W&B:       $([ -n "$NO_WANDB" ] && echo disabled || echo "project=$WANDB_PROJECT")"
echo "Start:     $(date)"
echo "========================================================================"
echo ""

# ── Build command ─────────────────────────────────────────────────────────────
CMD="python scripts/train_mantis_context_sweep.py --config $CONFIG"
[ -n "$TASK"           ] && CMD="$CMD --task $TASK"
[ -n "$TASK_TYPE"      ] && CMD="$CMD --task-type $TASK_TYPE"
[ -n "$HEAD"           ] && CMD="$CMD --head $HEAD"
[ -n "$CONTEXT"        ] && CMD="$CMD --context $CONTEXT"
[ -n "$DATASETS"       ] && CMD="$CMD --datasets $DATASETS"
[ -n "$EMBEDDING_DIR"  ] && CMD="$CMD --embedding-dir $EMBEDDING_DIR"
[ -n "$BATCH_SIZE"     ] && CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --accum-steps $ACCUM_STEPS"
[ -n "$LR"             ] && CMD="$CMD --lr $LR"
[ -n "$RUN_TAG"        ] && CMD="$CMD --run-tag $RUN_TAG"
[ -n "$WANDB_PROJECT"  ] && CMD="$CMD --wandb-project $WANDB_PROJECT"
[ -n "$NO_WANDB"       ] && CMD="$CMD --no-wandb"

echo "Running: $CMD"
echo ""

# Run Python in background so USR1 can interrupt 'wait' immediately
set +e
eval "$CMD" &
_PYTHON_PID=$!
wait $_PYTHON_PID
EXIT_CODE=$?
trap '' USR1   # training done — ignore any late-firing USR1
# Do NOT re-enable set -e here — EXIT_CODE is already captured and the
# cleanup/status path below must not be aborted by a non-zero subcommand.

echo ""
echo "========================================================================"
echo "End time: $(date)"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS"
    _write_status "SUCCESS"
else
    # Read failure reason written by Python (exp_dir/_failure_reason_<jobid>.txt)
    _RESULTS_DIR=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['logging']['results_dir'])" 2>/dev/null || echo "")
    _EXP_ID="${TASK}_${HEAD}"
    [ -n "$RUN_TAG" ] && _EXP_ID="${_EXP_ID}_${RUN_TAG}"
    _REASON_FILE="${_RESULTS_DIR}/${_EXP_ID}/_failure_reason_${SLURM_JOB_ID:-local}.txt"
    _REASON=$(cat "$_REASON_FILE" 2>/dev/null | tr '"' "'" || echo "unknown")
    echo "Status: FAILED (exit code: $EXIT_CODE) — ${_REASON}"
    _write_status "FAILED" "$_REASON"
fi
echo "========================================================================"

deactivate
exit $EXIT_CODE
