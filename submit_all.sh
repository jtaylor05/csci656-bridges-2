#!/bin/bash
# submit_all.sh
# Submit one SLURM job per group, with inter-job dependencies so groups
# run concurrently by default. Pass --sequential to chain them instead.
# Usage: bash submit_all.sh [--sequential] [--dry-run] [--limit N]

set -euo pipefail

MEGATRON_HOME="$HOME/Megatron-LM"
INPUT_CSV="$HOME/project/configs_v2.csv"
OUTDIR="$HOME/project/results"
CONDA_ENV="megatron"
WORKERS=1          # torchrun launches per node (keep 1 for 20-iter runs)
TIMEOUT=300        # seconds per config
EXTRA_ARGS=""
SEQUENTIAL=false

for arg in "$@"; do
  case $arg in
    --sequential) SEQUENTIAL=true ;;
    --dry-run)    EXTRA_ARGS="$EXTRA_ARGS --dry-run" ;;
    --limit=*)    EXTRA_ARGS="$EXTRA_ARGS --limit=${arg#*=}" ;;
  esac
done

mkdir -p "$OUTDIR" logs

# Group definitions: id  wall_hours
declare -A WALL_HOURS=(
  [g1_dense_small]=6
  [g2_dense_large]=8
  [g3_moe_mixtral]=8
  [g4_moe_deepseek]=6
  [g5_moe_qwen]=8
)

PREV_JOB=""

for GROUP_ID in g1_dense_small g2_dense_large g3_moe_mixtral g4_moe_deepseek g5_moe_qwen; do
  WH=${WALL_HOURS[$GROUP_ID]}

  DEPEND_ARG=""
  if $SEQUENTIAL && [ -n "$PREV_JOB" ]; then
    DEPEND_ARG="--dependency=afterok:${PREV_JOB}"
  fi

  JOB_ID=$(sbatch $DEPEND_ARG \
    --job-name="mpt_${GROUP_ID}" \
    --partition=GPU \
    --nodes=1 \
    --ntasks-per-node=8 \
    --gres=gpu:8 \
    --time="${WH}:00:00" \
    --mem=128G \
    --output="logs/${GROUP_ID}_%j.out" \
    --error="logs/${GROUP_ID}_%j.err" \
    --export=ALL \
    --wrap="
      module purge
      module load cuda/11.7.1 gcc/10.2.0 anaconda3/2022.10
      conda activate ${CONDA_ENV}

      export CUDA_DEVICE_MAX_CONNECTIONS=1
      export NCCL_TIMEOUT=120
      export WANDB_DISABLED=true
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

      python $HOME/project/run_group.py \
        --group ${GROUP_ID} \
        --input ${INPUT_CSV} \
        --outdir ${OUTDIR} \
        --megatron ${MEGATRON_HOME} \
        --workers ${WORKERS} \
        --timeout ${TIMEOUT} \
        --resume \
        ${EXTRA_ARGS}
    " \
    | awk '{print $NF}')

  echo "Submitted ${GROUP_ID}  job_id=${JOB_ID}  wall=${WH}h"
  PREV_JOB=$JOB_ID
done

echo ""
echo "All groups submitted. Monitor with: squeue -u \$USER"
echo "Results will appear in: $OUTDIR"
