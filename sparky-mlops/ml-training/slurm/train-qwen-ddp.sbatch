#!/bin/bash
#SBATCH --job-name=qwen-ddp
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/srv/sparky-mlops/runs/%x-%j-%N.out
#SBATCH --time=00:59:00

set -euo pipefail

cd /srv/sparky-mlops/jobs

RUN_ID="${RUN_ID:-slurm-${SLURM_JOB_ID}}"
CONFIG_PATH="${CONFIG_PATH:-ml-training/configs/qwen-ddp-dev.yaml}"

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29500}"

export RUN_ID
export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE="${SLURM_NTASKS}"

export HF_HOME=/srv/sparky-mlops/hf-cache
export TRANSFORMERS_CACHE=/srv/sparky-mlops/hf-cache

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=enP7s7
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1

export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}"

echo "job id: ${SLURM_JOB_ID}"
echo "nodes: ${SLURM_JOB_NODELIST}"
echo "master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "world size: ${WORLD_SIZE}"

export TRITON_CACHE_DIR=/srv/sparky-mlops/cache/triton
export XDG_CACHE_HOME=/srv/sparky-mlops/cache/xdg
export TORCH_HOME=/srv/sparky-mlops/cache/torch
export HF_HOME=/srv/sparky-mlops/hf-cache
export TRANSFORMERS_CACHE=/srv/sparky-mlops/hf-cache

mkdir -p "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${HF_HOME}"

srun bash -lc '
  export RANK="${SLURM_PROCID}"
  export LOCAL_RANK="${SLURM_LOCALID}"
  export TRITON_CACHE_DIR="/srv/sparky-mlops/cache/triton/rank-${RANK}"
  mkdir -p "${TRITON_CACHE_DIR}"

  echo "rank=${RANK} local_rank=${LOCAL_RANK} host=$(hostname)"

  /srv/sparky-mlops/venv/bin/python ml-training/train_qwen_ddp.py \
    --config "'"${CONFIG_PATH}"'" \
    --run-id "'"${RUN_ID}"'"
'