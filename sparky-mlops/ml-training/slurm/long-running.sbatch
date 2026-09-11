#!/bin/bash
#SBATCH --job-name=long-running
#SBATCH --partition=debug
#SBATCH --output=/srv/sparky-mlops/runs/%x-%j-%N.out
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail

echo "job id: ${SLURM_JOB_ID}"
echo "node: $(hostname)"
echo "user: $(whoami)"
echo "started: $(date)"

for i in {1..600}; do
  echo "tick ${i} $(date)"
  sleep 1
done

echo "completed: $(date)"