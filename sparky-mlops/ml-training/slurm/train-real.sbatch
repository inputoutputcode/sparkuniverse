#!/bin/bash
#SBATCH --job-name=real-train
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/srv/sparky-mlops/runs/%x-%j-%N.out
#SBATCH --time=00:10:00

set -euo pipefail

cd /srv/sparky-mlops/jobs

RUN_ID="${RUN_ID:-slurm-${SLURM_JOB_ID}}"
CONFIG_PATH="${CONFIG_PATH:-ml-training/configs/real-dev.yaml}"

/srv/sparky-mlops/venv/bin/python ml-training/train_real.py \
  --config "${CONFIG_PATH}" \
  --run-id "${RUN_ID}"