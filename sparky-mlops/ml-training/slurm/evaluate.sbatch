#!/bin/bash
#SBATCH --job-name=fake-eval
#SBATCH --partition=debug
#SBATCH --output=/srv/sparky-mlops/runs/%x-%j-%N.out
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --gres=gpu:1

set -euo pipefail

cd /srv/sparky-mlops/jobs

RUN_ID="${RUN_ID:?RUN_ID must be set}"
CONFIG_PATH="${CONFIG_PATH:-ml-training/configs/eval-dev.yaml}"
METADATA_PATH="${METADATA_PATH:-/srv/sparky-mlops/runs/${RUN_ID}/metadata.json}"

echo "job id: ${SLURM_JOB_ID}"
echo "run id: ${RUN_ID}"
echo "node: $(hostname)"
echo "user: $(whoami)"
echo "config: ${CONFIG_PATH}"
echo "metadata: ${METADATA_PATH}"
echo "python: $(python3 --version)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python3 ml-training/evaluate.py \
  --config "${CONFIG_PATH}" \
  --metadata "${METADATA_PATH}"