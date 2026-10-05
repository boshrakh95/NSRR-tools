#!/bin/bash
#SBATCH --job-name=test_gpu_nibi
#SBATCH --account=def-egranger_gpu
#SBATCH --time=00:10:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --output=/home/boshra95/NSRR-tools-mantis/logs_mantis/%x_%j.out
#SBATCH --error=/home/boshra95/NSRR-tools-mantis/logs_mantis/%x_%j.err

# Nibi GPU-allocation smoke test — run this FIRST, before submitting any real
# Mantis job here, to confirm the SLURM directives below are actually valid
# on Nibi.
#
# Why this exists: docs.alliancecan.ca/wiki/Nibi is behind an Anubis
# JS-challenge bot wall that blocked automated fetching (would also block a
# plain `curl` from a login node — it requires a real browser). The GPU
# resource string, account format, and "no --partition needed" assumption
# below are inferred from converging secondary sources (mint.westdri.ca's
# GPU-clusters training page, the castorini/onboarding cc-guide, and the
# alliance-ml-docs skill reference), NOT read directly off Nibi's own wiki
# page. Specifically NOT verified from a primary source:
#   - --gpus=h100:1 (whole GPU) is used because no MIG slice naming (like
#     Fir's nvidia_h100_80gb_hbm3_1g.10gb) was found documented for Nibi
#     anywhere — guessing a MIG string risks a hard sbatch rejection, so
#     this deliberately asks for a whole card instead.
#   - No --partition line — every real Nibi example found omits one
#     (unlike Rorqual, which requires --partition=gpubase_bygpu_b3).
#   - No --exclude — no documented bad-node list for Nibi, unlike Fir's
#     fc11006/fc11013/fc11010.
# If this job fails at the scheduler level (rejected/held, not a runtime
# error), that's the signal one of these guesses is wrong — check the
# rejection reason with `sacct -j <jobid> --format=State,Reason` and adjust
# before touching the real train/infer _nibi.sh scripts.
#
# What this job actually checks at runtime (once it starts):
#   - nvidia-smi sees a GPU at all
#   - torch.cuda.is_available() is True, if mantis_env already exists
#     (skips gracefully if the env isn't built yet — this script has no
#     hard dependency on it, so it's safe to run before env setup too)
#   - basic tensor op on the GPU actually executes, not just reports available
#
# Usage:
#   sbatch jobs/test_gpu_setup_nibi.sh
#   # then: sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode
#   #       cat logs_mantis/test_gpu_nibi_<jobid>.out

set -e

cd /home/boshra95/NSRR-tools-mantis
mkdir -p logs_mantis

echo "========================================================================"
echo "Nibi GPU smoke test"
echo "========================================================================"
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "Account:     $SLURM_JOB_ACCOUNT"
echo "Partition:   ${SLURM_JOB_PARTITION:-(none requested)}"
echo "Start:       $(date)"
echo "========================================================================"
echo ""

echo "── nvidia-smi ──────────────────────────────────────────────────────────"
nvidia-smi || { echo "ERROR: nvidia-smi failed — no GPU visible on this node."; exit 1; }
echo ""

echo "── SLURM-reported GPU allocation ──────────────────────────────────────"
echo "SLURM_GPUS: ${SLURM_GPUS:-unset}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo ""

if [ -f /home/boshra95/mantis_env/bin/activate ]; then
    echo "── mantis_env found — running torch GPU check ────────────────────────"
    module load python/3.10.13 2>/dev/null || true
    source /home/boshra95/mantis_env/bin/activate
    python -c "
import torch
print('torch version:', torch.__version__)
assert torch.cuda.is_available(), 'CUDA not available'
print('CUDA available: True')
print('Device name:', torch.cuda.get_device_name(0))
print('Device memory (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
x = torch.randn(4096, 4096, device='cuda')
y = x @ x
torch.cuda.synchronize()
print('Matmul on GPU: OK, result mean =', y.mean().item())
"
    deactivate
    echo ""
    echo "RESULT: GPU allocation + mantis_env both verified working on Nibi."
else
    echo "── mantis_env not found yet — skipping torch check ───────────────────"
    echo "(That's fine if you're running this before building the env — the"
    echo " nvidia-smi output above already confirms the SLURM GPU request"
    echo " and node allocation work. Re-run this script after mantis_env"
    echo " exists to also confirm torch sees the GPU.)"
    echo ""
    echo "RESULT: SLURM GPU allocation verified working on Nibi (env check skipped)."
fi

echo ""
echo "========================================================================"
echo "End time: $(date)"
echo "========================================================================"
