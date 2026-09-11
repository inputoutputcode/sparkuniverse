#!/bin/bash
#SBATCH --job-name=nccl-collectives
#SBATCH --partition=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/srv/sparky-mlops/runs/%x-%j-%N.out
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G

set -euo pipefail

NCCL_TESTS_DIR="${NCCL_TESTS_DIR:-/srv/sparky-mlops/jobs/nccl-tests}"
cd "${NCCL_TESTS_DIR}"

echo "job id: ${SLURM_JOB_ID}"
echo "nodes: ${SLURM_JOB_NODELIST}"
echo "submit host: ${SLURM_SUBMIT_HOST}"
echo "batch host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_IFNAME=enP7s7

export OMPI_MCA_btl=tcp,self
export OMPI_MCA_btl_tcp_if_include=enP7s7
export OMPI_MCA_oob_tcp_if_include=enP7s7

export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1

echo "Interfaces:"
srun ip -brief addr

echo "Node placement:"
srun hostname

echo "GPU visibility:"
srun nvidia-smi -L

echo "all_reduce_perf"
mpirun -np 2 ./build/all_reduce_perf -b 8 -e 16G -f 2 -g 1

echo "all_gather_perf"
mpirun -np 2 ./build/all_gather_perf -b 8 -e 16G -f 2 -g 1

echo "reduce_scatter_perf"
mpirun -np 2 ./build/reduce_scatter_perf -b 8 -e 16G -f 2 -g 1

echo "alltoall_perf"
mpirun -np 2 ./build/alltoall_perf -b 8 -e 16G -f 2 -g 1