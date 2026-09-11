# Sparky MLOps Lab Notes

## Project Goal

This lab builds a small but realistic MLOps workflow on two DGX Spark systems. The goal is not to create a production ML platform immediately, but to learn the operational pieces by building them manually:

- GitLab CI/CD as the control plane
- GitLab Runner on `agenthost`
- SSH-based job submission to the DGX Spark cluster
- Slurm for batch scheduling
- SlurmDBD/accounting for job history
- Ansible for infrastructure checks and setup
- NFS shared storage for multi-node jobs
- NCCL benchmarks for GPU/network validation
- PyTorch DDP for distributed training validation
- LoRA training with Qwen as the first real model workload

## Cluster Topology

The lab uses two DGX Spark nodes:

- `spark-38fc`
  - Slurm controller
  - Slurm compute node
  - NFS server
  - Shared storage export: `/srv/sparky-mlops`

- `spark-1ac4`
  - Slurm compute node
  - NFS client
  - Mounts `/srv/sparky-mlops`

The GitLab Runner runs on `agenthost.local` using the shell executor and submits jobs through SSH to `spark-38fc`.

## Shared Storage

A central shared path was introduced because Slurm jobs may land on either node. Writing logs to local home directories caused GitLab jobs to fail when the output landed on the other node.

Final shared layout:

`/srv/sparky-mlops/`

Expected subdirectories:

- `/srv/sparky-mlops/jobs`
- `/srv/sparky-mlops/runs`
- `/srv/sparky-mlops/venv`
- `/srv/sparky-mlops/hf-cache`
- `/srv/sparky-mlops/cache`

The important lesson: once Slurm schedules across multiple nodes, all job inputs and outputs should live on shared storage or another cluster-visible filesystem.

## NFS Notes

`spark-38fc` exports:

`/srv/sparky-mlops 192.168.1.229(rw,sync,no_subtree_check)`

`spark-1ac4` mounts:

`192.168.1.82:/srv/sparky-mlops /srv/sparky-mlops nfs defaults,_netdev,vers=4 0 0`

A more robust lab entry after reboot issues:

`192.168.1.82:/srv/sparky-mlops /srv/sparky-mlops nfs defaults,_netdev,nofail,x-systemd.automount,x-systemd.requires=network-online.target,x-systemd.after=network-online.target,vers=4 0 0`

## Slurm Setup

Slurm was configured with:

- `slurmctld` on `spark-38fc`
- `slurmd` on both DGX nodes
- `munge` authentication
- one debug partition
- both nodes registered as compute nodes
- one GPU GRES per node

GPU detection required manual GRES configuration because automatic NVML detection was not available with the installed Slurm package.

Relevant symptom:

`Gres=(null)`

Working state:

`Gres=gpu:1`

## Slurm Accounting

Initially `sacct` returned:

`Slurm accounting storage is disabled`

After adding SlurmDBD/accounting support, `sacct` worked and showed job records such as:

| JobID | JobName | User | Account | State | ExitCode | Elapsed |
|---|---|---|---|---|---|---|
| 77 | hello-slurm | chris | lab | COMPLETED | 0:0 | 00:00:01 |

This was important because `scontrol show job` only helps while jobs are still retained by the controller, while `sacct` gives persistent accounting history.

## GitLab CI Integration

GitLab does not run the training itself. It orchestrates:

`GitLab CI -> GitLab Runner on agenthost -> SSH -> sbatch on spark-38fc -> Slurm schedules work`

A dedicated sync stage copies project files to shared storage:

`/srv/sparky-mlops/jobs`

This avoids stale Slurm scripts on the cluster.

Important GitLab lesson: jobs in the same stage run in parallel unless ordered with stages or `needs`.

## Common GitLab Runner Issues

The SaaS GitLab runner could not reach the local DGX systems:

`ssh: connect to host 192.168.1.82 port 22: Connection timed out`

The local shell runner on `agenthost` was required.

Another repeated message appeared during Git checkout:

`failed to store: -61`

This came from the macOS credential helper/keychain path during CI Git operations. It was noisy but did not block the jobs.

## Slurm Output Troubleshooting

A common failure was looking for output files on the wrong node or wrong path.

Bad assumption:

`~/slurm-${JOB_ID}.out`

Better pattern:

`#SBATCH --output=/srv/sparky-mlops/runs/%x-%j-%N.out`

Then search shared storage:

`find /srv/sparky-mlops/runs -maxdepth 1 -type f -name "*${JOB_ID}*.out" -print`

The `%N` token is useful because it records which node wrote the output.

## NCCL Benchmarks

The lab used NVIDIA `nccl-tests` to validate GPU communication across both DGX Spark nodes.

Collectives tested:

- `all_reduce_perf`
- `all_gather_perf`
- `reduce_scatter_perf`
- `alltoall_perf`

Useful parameters:

- `-b`: beginning message size
- `-e`: ending message size
- `-f`: size multiplier between test points
- `-g`: GPUs per process

Example command:

`mpirun -np 2 ./build/all_reduce_perf -b 8 -e 1G -f 2 -g 1`

## NCCL Networking Fixes

Open MPI initially selected the wrong interface and tried to communicate over a Docker bridge network:

`connect() to 172.18.0.1:1024 failed`

The fix was to constrain MPI and NCCL to the correct network interfaces:

`export NCCL_SOCKET_IFNAME=enP7s7`

`export OMPI_MCA_btl=tcp,self`

`export OMPI_MCA_btl_tcp_if_include=enP7s7`

`export OMPI_MCA_oob_tcp_if_include=enP7s7`

For dual rail ConnectX usage:

`export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`

The NCCL logs confirmed use of both rails with `NET/IB/0` and `NET/IB/1`.

## NCCL Results

A successful four-collective NCCL run completed in roughly 30-40 seconds for a 1G sweep.

Observed behavior:

- single rail bandwidth was roughly around 12-14 GB/s
- dual rail bandwidth improved visible collectives to roughly 22-24 GB/s
- `#wrong=0` confirmed correctness
- `algbw` and `busbw` were often similar because with two ranks the communication factor is simple

For learning, running all four collectives is useful. For quick CI validation, one small `all_reduce_perf` smoke test is enough. For deeper cluster validation, test larger sizes and all collectives manually or in scheduled validation jobs.

## Real Training Workload

NCCL validates communication primitives, but it does not fully validate a training stack.

A real training workload adds coverage for:

- Python environment consistency
- CUDA/PyTorch compatibility
- Hugging Face cache behavior
- tokenizer/model loading
- optimizer behavior
- GPU memory pressure
- CPU/GPU shared memory pressure
- DDP process group setup
- checkpoint/artifact writing
- real application-level failure modes

## LoRA Training

The first real model workload used Qwen with LoRA.

Conceptually:

`Fine-tuning = broad category`

`Full fine-tuning = update most or all base model weights`

`LoRA fine-tuning = freeze base model and train small adapter weights`

The lab used LoRA because it is practical on DGX Spark and still exercises the distributed training path.

Model:

`Qwen/Qwen2.5-0.5B-Instruct`

Dataset:

`databricks/databricks-dolly-15k`

The dataset is useful for a public demo if attribution and license terms are respected. Avoid vendoring the full dataset into the repo unless the license obligations are handled.

## Python Environment

A shared virtual environment was placed at:

`/srv/sparky-mlops/venv`

This worked in the lab because both DGX nodes have the same architecture, Python ABI, filesystem path, and compatible NVIDIA stack.

PyTorch validation:

`torch 2.14.0+cu130`

`torch.cuda.is_available() == True`

`CUDA device: NVIDIA GB10`

The NVIDIA PyPI mirror caused DNS delays:

`pypi.ngc.nvidia.com`

Temporary workaround:

`/srv/sparky-mlops/venv/bin/python -m pip --isolated install --index-url https://pypi.org/simple datasets transformers peft accelerate safetensors`

## Qwen DDP Training Result

The successful Qwen DDP LoRA job ran through GitLab, Slurm, two DGX Spark nodes, and one GPU per node.

Result summary:

| Field | Value |
|---|---|
| RUN_ID | `ci-2823845558-qwen-ddp` |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Dataset | `databricks/databricks-dolly-15k` |
| World size | `2` |
| Samples total | `12000` |
| Samples per rank | `6000` |
| Epochs | `1` |
| Batch size per rank | `1` |
| Global batch size | `2` |
| Max length | `512` |
| Learning rate | `0.0002` |
| Elapsed training time | `485.47 seconds` |
| Approximate CI duration | `9 minutes` |

Final loss:

| Rank | Host | Final Loss | Average Loss |
|---|---|---|---|
| 0 | `spark-1ac4` | `1.5867` | `1.8271` |
| 1 | `spark-38fc` | `1.3790` | `1.8309` |

Aggregate:

`avg_final_loss: 1.4828`

`avg_loss: 1.8290`

Artifact:

`/srv/sparky-mlops/runs/ci-2823845558-qwen-ddp/qwen-lora-adapter`

This adapter is not a full standalone model. It must be loaded with the base Qwen model or merged into the base model for serving.

## Slurm Resource Notes

The Qwen DDP job requested:

`#SBATCH --nodes=2`

`#SBATCH --ntasks-per-node=1`

`#SBATCH --gres=gpu:1`

`#SBATCH --cpus-per-task=4`

`#SBATCH --mem=64G`

The DGX Spark has unified CPU/GPU memory, but Slurm memory accounting still operates at the node allocation level. Requesting `64G` was a reasonable lab setting because the node was configured with about `99G` memory in Slurm to leave a safety buffer.

`MaxRSS` from `sacct` is useful for observing peak resident CPU memory, but it does not fully explain GPU or unified memory behavior.

## Time Limit Issue

A Qwen job stayed pending because its requested time exceeded the partition maximum:

`Reason: PartitionTimeLimit`

The partition had:

`MaxTime=01:00:00`

The job was adjusted to:

`#SBATCH --time=00:45:00`

## Current End State

The lab has successfully demonstrated:

- GitLab Runner on a local Mac host
- SSH-based CI access into the DGX Spark cluster
- Slurm two-node scheduling
- GPU GRES configuration
- SlurmDBD accounting
- shared NFS storage across nodes
- Ansible infrastructure checks/setup
- NCCL collective benchmarks
- PyTorch DDP smoke validation
- real Qwen LoRA DDP training
- persistent artifacts under `/srv/sparky-mlops/runs`

## Next Practical Step

The next major step is model evaluation.

The evaluation job should:

- load `Qwen/Qwen2.5-0.5B-Instruct`
- load the LoRA adapter from the run directory
- run a fixed prompt set
- write `qwen-evaluation.json`
- optionally write generated samples as Markdown or JSONL
- print a concise GitLab log summary

After evaluation works, the next phase can be serving:

`FastAPI /predict -> base Qwen + LoRA adapter -> Docker image -> Helm chart -> Kubernetes`

That would bring Kubernetes and Helm back into the lab for the model serving side, while Slurm remains responsible for training and validation workloads.