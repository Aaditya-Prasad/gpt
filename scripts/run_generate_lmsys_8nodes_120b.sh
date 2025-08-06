#!/bin/bash
#
# SLURM batch script to run `generate.py` on 8 nodes (64 GPUs total).
# Each node processes a distinct shard of the LMSYS dataset with no inter-node
# communication.
#
# Node-local log files are written to `slurm_logs/generate_<node_id>.log`.
# Standard SLURM output/error streams are also redirected to `slurm_logs`.
# -----------------------------------------------------------------------------
#SBATCH --job-name=generate_120b_lmsys
#SBATCH --nodes=1               # single node
#SBATCH --ntasks-per-node=8     # one task per GPU
#SBATCH --gres=gpu:8            # 8 GPUs per node
#SBATCH --cpus-per-task=4       # adjust as appropriate for your cluster
#SBATCH --time=24:00:00         # adjust wall-time as needed
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# Resolve path to env.sh in submission directory and load environment variables
ENV_FILE="${SLURM_SUBMIT_DIR:-$(pwd)}/env.sh"
source "$ENV_FILE"
export ENV_FILE

# Ensure the logging directory exists in submit dir
mkdir -p "${SLURM_SUBMIT_DIR:-$(pwd)}/slurm_logs"

# Create a per-job log directory
JOB_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/slurm_logs/${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_DIR"
export JOB_DIR

# -----------------------------------------------------------------------------
# Launch one independent task on each node.
#
# `SLURM_PROCID` serves as the 0-indexed identifier for the task/node.
# This ID is used to select the correct data shard and to set a unique
# experiment name for the run on that node.
# -----------------------------------------------------------------------------

srun --ntasks=8 --ntasks-per-node=8 \
     bash -c '
        source "$ENV_FILE"
        cd "$SLURM_SUBMIT_DIR/gpt"
        GPU_ID=${SLURM_LOCALID:-${SLURM_PROCID}}
        export CUDA_VISIBLE_DEVICES=${GPU_ID}
        CMD="python generate.py --model 120b --data_path inputs/lmsys-filtered_${GPU_ID}.json -e 120b-lmsys-reasoning=med_${GPU_ID} -t 1 -r medium"
        echo "[$(date)] Running on GPU ${GPU_ID}: $CMD" >&2
        eval $CMD &> ${JOB_DIR}/generate_${GPU_ID}.log
     '

# Wait for all background tasks to finish before exiting.
wait

printf "\nAll per-gpu generate.py jobs have completed.\n"

# Move SLURM output files into the per-job directory to avoid clutter
OUT_FILE="${SLURM_SUBMIT_DIR:-$(pwd)}/slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
ERR_FILE="${SLURM_SUBMIT_DIR:-$(pwd)}/slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
if [ -f "$OUT_FILE" ]; then mv "$OUT_FILE" "$JOB_DIR/"; fi
if [ -f "$ERR_FILE" ]; then mv "$ERR_FILE" "$JOB_DIR/"; fi
