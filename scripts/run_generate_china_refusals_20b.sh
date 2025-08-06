#!/bin/bash
#
# SLURM batch script to run `generate.py` (20B model) on the China refusals
# dataset.  A single task uses all 8 GPUs on the node.
#
# All job-specific logs are written to `slurm_logs/<jobid>/` inside the
# SLURM submission directory.
# -----------------------------------------------------------------------------
#SBATCH --job-name=generate_20b_china_refusals
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1      # single task uses all GPUs
#SBATCH --gres=gpu:8             # 8 GPUs per node
#SBATCH --cpus-per-task=16       # adjust as appropriate
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Environment activation
# -----------------------------------------------------------------------------
ENV_FILE="${SLURM_SUBMIT_DIR:-$(pwd)}/env.sh"
source "$ENV_FILE"

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
LOG_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}/slurm_logs"
mkdir -p "$LOG_ROOT"
JOB_DIR="$LOG_ROOT/${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_DIR"

# -----------------------------------------------------------------------------
# Run generation (single task uses all 8 GPUs)
# -----------------------------------------------------------------------------
cd "${SLURM_SUBMIT_DIR}/gpt"
CMD="python generate.py --model 20b --data_path inputs/china_refusals.json -e 20b-china-refusals-reasoning=med -t 8 -r medium"

echo "[$(date)] $CMD" | tee "$JOB_DIR/generate.log"

# Run and capture both stdout and stderr into the same log file
eval $CMD &>> "$JOB_DIR/generate.log"

# Move SLURM job-level stdout/err into JOB_DIR to avoid clutter
for f in "${LOG_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}."{out,err}; do
  [ -f "$f" ] && mv "$f" "$JOB_DIR/"
done

printf "\nGeneration completed. Logs in $JOB_DIR\n"