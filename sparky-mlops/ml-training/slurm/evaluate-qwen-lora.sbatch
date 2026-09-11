#!/bin/bash
#SBATCH --job-name=qwen-eval
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/srv/sparky-mlops/runs/%x-%j-%N.out
#SBATCH --time=00:30:00

set -euo pipefail

cd /srv/sparky-mlops/jobs

RUN_ID="${RUN_ID:?RUN_ID is required}"
CONFIG_PATH="${CONFIG_PATH:-ml-training/configs/qwen-eval-dev.yaml}"

export HF_HOME=/srv/sparky-mlops/hf-cache
export TRANSFORMERS_CACHE=/srv/sparky-mlops/hf-cache
export XDG_CACHE_HOME=/srv/sparky-mlops/cache/xdg
export TORCH_HOME=/srv/sparky-mlops/cache/torch
export TRITON_CACHE_DIR=/srv/sparky-mlops/cache/triton/eval

mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}"

echo "job id: ${SLURM_JOB_ID}"
echo "job name: ${SLURM_JOB_NAME}"
echo "node: $(hostname)"
echo "run id: ${RUN_ID}"
echo "config: ${CONFIG_PATH}"
echo "python: $(/srv/sparky-mlops/venv/bin/python --version)"

/srv/sparky-mlops/venv/bin/python ml-training/evaluate_qwen_lora.py \
  --config "${CONFIG_PATH}" \
  --run-id "${RUN_ID}"