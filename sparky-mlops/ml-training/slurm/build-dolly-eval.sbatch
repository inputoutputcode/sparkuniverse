#!/bin/bash
#SBATCH --job-name=build-eval
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/srv/sparky-mlops/runs/%x-%j-%N.out
#SBATCH --time=00:10:00

set -euo pipefail

cd /srv/sparky-mlops/jobs

CONFIG_PATH="${CONFIG_PATH:-ml-training/configs/build-dolly-eval-dev.yaml}"

export HF_HOME=/srv/sparky-mlops/hf-cache
export TRANSFORMERS_CACHE=/srv/sparky-mlops/hf-cache
export XDG_CACHE_HOME=/srv/sparky-mlops/cache/xdg

mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" ml-training/evals

/srv/sparky-mlops/venv/bin/python ml-training/build_dolly_eval_set.py \
  --config "${CONFIG_PATH}"