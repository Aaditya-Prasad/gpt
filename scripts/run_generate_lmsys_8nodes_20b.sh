#!/bin/bash
#
# SLURM batch script to run `generate.py` on 8 nodes (64 GPUs total).
# Each node processes a distinct shard of the LMSYS dataset with no inter-node
# communication.
#
# Node-local log files are written to `slurm_logs/generate_<node_id>.log`.
# Standard SLURM output/error streams are also redirected to `slurm_logs`.
# -----------------------------------------------------------------------------
#SBATCH --job-name=generate_20b_lmsys
#SBATCH --nodes=8               # total number of nodes
#SBATCH --ntasks-per-node=1     # one task per node
#SBATCH --gres=gpu:8            # 8 GPUs per node
#SBATCH --cpus-per-task=16      # adjust as appropriate for your cluster
#SBATCH --time=24:00:00         # adjust wall-time as needed
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# Ensure the logging directory exists
mkdir -p slurm_logs

# -----------------------------------------------------------------------------
# Launch one independent task on each node.
#
# `SLURM_PROCID` serves as the 0-indexed identifier for the task/node.
# This ID is used to select the correct data shard and to set a unique
# experiment name for the run on that node.
# -----------------------------------------------------------------------------

srun --ntasks=8 --ntasks-per-node=1 \
     bash -c '
        NODE_ID=${SLURM_PROCID}
        CMD="python generate.py --model 20b --data_path inputs/lmsys-filtered_${NODE_ID}.json -e 20b-lmsys-default_${NODE_ID} -t 8"
        echo "[$(date)] Running on node $NODE_ID: $CMD" >&2
        eval $CMD &> slurm_logs/generate_${NODE_ID}.log
     '

# Wait for all background tasks to finish before exiting.
wait

printf "\nAll per-node generate.py jobs have completed.\n"