#!/bin/bash
#SBATCH --job-name=hello-slurm
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=/srv/sparky-mlops/runs/%x-%j-%N.out
#SBATCH --time=00:05:00

set -euxo pipefail

mkdir -p /srv/sparky-mlops/runs

echo "job id: ${SLURM_JOB_ID}"
echo "job name: ${SLURM_JOB_NAME}"
echo "node: $(hostname)"
echo "user: $(whoami)"
echo "workdir: $(pwd)"
echo "date: $(date -Is)"

echo "Slurm environment:"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST}"
echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

echo "Hello from Slurm"