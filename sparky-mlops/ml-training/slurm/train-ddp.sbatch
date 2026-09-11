#!/bin/bash
#SBATCH --job-name=ddp-train
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/srv/sparky-mlops/runs/%x-%j-%N.out
#SBATCH --time=00:15:00

set -euo pipefail

cd /srv/sparky-mlops/jobs

RUN_ID="${RUN_ID:-slurm-${SLURM_JOB_ID}}"
CONFIG_PATH="${CONFIG_PATH:-ml-training/configs/ddp-dev.yaml}"

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29500}"

export RUN_ID
export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE="${SLURM_NTASKS}"

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=enP7s7
export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1

export OMPI_MCA_btl=tcp,self
export OMPI_MCA_btl_tcp_if_include=enP7s7
export OMPI_MCA_oob_tcp_if_include=enP7s7

echo "job id: ${SLURM_JOB_ID}"
echo "nodes: ${SLURM_JOB_NODELIST}"
echo "master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "world size: ${WORLD_SIZE}"

srun bash -lc '
  export RANK="${SLURM_PROCID}"
  export LOCAL_RANK="${SLURM_LOCALID}"
  echo "rank=${RANK} local_rank=${LOCAL_RANK} host=$(hostname)"
  /srv/sparky-mlops/venv/bin/python ml-training/train_ddp.py \
    --config "'"${CONFIG_PATH}"'" \
    --run-id "'"${RUN_ID}"'"
'