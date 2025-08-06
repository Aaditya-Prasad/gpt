#!/bin/bash
#
# SLURM batch script that reserves 1 node (8 GPUs) and launches four
# independent generation jobs, each using 2 GPUs (-t 2):
#   0) 20B model on safety_bench.json
#   1) 20B model on global_opinions.json
#   2) 120B model on safety_bench.json
#   3) 120B model on global_opinions.json
#
# Each job/task writes its own log to slurm_logs/<jobid>/generate_<task>.log
# and the SLURM stdout/err are moved into the same directory.
# -----------------------------------------------------------------------------
#SBATCH --job-name=generate_opinions_safety
#SBATCH --nodes=1
#SBATCH --ntasks=4               # 4 independent tasks
#SBATCH --gpus-per-task=2        # 2 GPUs per task → 8 GPUs total
#SBATCH --cpus-per-task=8        # adjust as appropriate
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# -----------------------------------------------------------------------------
# Environment activation
# -----------------------------------------------------------------------------
ENV_FILE="${SLURM_SUBMIT_DIR:-$(pwd)}/env.sh"
source "$ENV_FILE"
export ENV_FILE

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
LOG_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}/slurm_logs"
mkdir -p "$LOG_ROOT"
JOB_DIR="$LOG_ROOT/${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_DIR"
export JOB_DIR

# -----------------------------------------------------------------------------
# Launch tasks (each with 2 GPUs)
# -----------------------------------------------------------------------------

srun --ntasks=4 --gpus-per-task=2 bash -c '
  set -euo pipefail
  source "$ENV_FILE"

  # Move to project directory where generate.py lives
  cd "$SLURM_SUBMIT_DIR/gpt"

  TASK_ID=${SLURM_PROCID}
  GPU_PAIR_START=$(( SLURM_LOCALID * 2 ))
  GPU_PAIR_END=$(( GPU_PAIR_START + 1 ))
  export CUDA_VISIBLE_DEVICES="${GPU_PAIR_START},${GPU_PAIR_END}"

  case $TASK_ID in
    0)
      MODEL="20b"
      DATA="safety_bench.json"
      EXP="20b-safety"
      THREADS=2
      ;;
    1)
      MODEL="20b"
      DATA="global_opinions.json"
      EXP="20b-global-opinions"
      THREADS=2
      ;;
    2)
      MODEL="120b"
      DATA="safety_bench.json"
      EXP="120b-safety"
      THREADS=2
      ;;
    3)
      MODEL="120b"
      DATA="global_opinions.json"
      EXP="120b-global-opinions"
      THREADS=2
      ;;
    *)
      echo "Unknown task id $TASK_ID" >&2
      exit 1
      ;;
  esac

  CMD="python generate.py --model ${MODEL} --data_path inputs/${DATA} -e ${EXP} -t ${THREADS}"
  echo "[$(date)] Task $TASK_ID using GPUs $CUDA_VISIBLE_DEVICES: $CMD" | tee "$JOB_DIR/generate_${TASK_ID}.log"

  # Run command, append both stdout & stderr
  eval $CMD &>> "$JOB_DIR/generate_${TASK_ID}.log"
'

# -----------------------------------------------------------------------------
# Move SLURM stdout/err into per-job directory to avoid clutter
# -----------------------------------------------------------------------------
for f in "${LOG_ROOT}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}."{out,err}; do
  [ -f "$f" ] && mv "$f" "$JOB_DIR/"
done

printf "\nAll four generation tasks completed. Logs in $JOB_DIR\n"