#!/bin/bash
#SBATCH --job-name=fake-train
#SBATCH --partition=debug
#SBATCH --output=/srv/sparky-mlops/runs/%x-%j-%N.out
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

cd /srv/sparky-mlops/jobs

RUN_ID="${RUN_ID:-slurm-${SLURM_JOB_ID}}"
CONFIG_PATH="${CONFIG_PATH:-ml-training/configs/dev.yaml}"

echo "job id: ${SLURM_JOB_ID}"
echo "run id: ${RUN_ID}"
echo "node: $(hostname)"
echo "user: $(whoami)"
echo "config: ${CONFIG_PATH}"
echo "python: $(python3 --version)"

python3 ml-training/train.py \
  --config "${CONFIG_PATH}" \
  --run-id "${RUN_ID}"
