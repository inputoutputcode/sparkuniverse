# Sparky MLOps Lab

Sparky MLOps is a hands-on learning project for building an LLM fine-tuning automation pipeline on a local DGX Spark cluster.

The project focuses on learning by doing:

- GitLab CI/CD
- GitLab Runner
- SSH automation
- Ansible
- Slurm
- Python
- Bash
- Later: Docker, FastAPI, Kubernetes, and Helm

The first phase is intentionally Slurm-first. Kubernetes and Helm are optional later additions for online model serving.

## Project Rule

No agent may create, edit, delete, format, or refactor project files.

Agents may only:

- explain concepts
- review pasted snippets
- suggest commands
- help debug pasted errors
- propose exercises
- provide file content for manual use

All file changes must be made manually by the project owner.

## Architecture

Current control flow:

    GitLab.com
      |
      | pipeline trigger
      v
    GitLab Runner on agenthost
      |
      | shell executor
      v
    agenthost.local
      |
      | SSH / Ansible / SCP
      v
    DGX Spark node
      |
      | sbatch
      v
    Slurm
      |
      | executes Python job
      v
    Training / evaluation outputs

Planned full flow:

    Git push
      -> GitLab pipeline
      -> local GitLab Runner
      -> Ansible checks DGX environment
      -> GitLab copies repo-managed training files to DGX
      -> GitLab submits Slurm training job
      -> Slurm runs Python training
      -> training writes metadata/artifacts
      -> GitLab submits Slurm evaluation job
      -> evaluation writes metrics
      -> GitLab gates promotion based on metrics
      -> optional: Docker builds inference API
      -> optional: Helm deploys API to Kubernetes

## Host Roles

### GitLab.com

GitLab hosts the repository and controls CI/CD pipelines.

It is responsible for:

- source control
- pipeline execution flow
- job logs
- pass/fail status
- manual approvals
- future model promotion and deployment gates

GitLab does not directly access the DGX cluster. It delegates local work to GitLab Runner.

### agenthost

`agenthost` runs the local GitLab Runner.

It is responsible for:

- executing CI jobs with the shell executor
- reaching the DGX Spark node over SSH
- running Ansible against DGX
- copying repo-managed files to DGX
- submitting Slurm jobs
- polling Slurm job status
- printing logs and metadata in CI output

### DGX Spark

The DGX Spark node runs Slurm and executes workloads.

It is responsible for:

- Slurm controller and worker services
- GPU visibility through NVIDIA tools
- training jobs
- evaluation jobs
- batch inference jobs
- run artifacts and logs

Important remote paths:

    /home/chris/sparky-mlops-jobs
    /home/chris/sparky-mlops-runs

## Repository Layout

Planned structure:

    .
    ├── .gitlab-ci.yml
    ├── AGENTS.md
    ├── README.md
    ├── docs/
    │   ├── architecture.md
    │   ├── mlops-pipeline-design.md
    │   ├── project-endstate-and-timeline.md
    │   └── runbook.md
    ├── infra/
    │   └── ansible/
    │       ├── inventory/
    │       │   └── hosts.ini
    │       └── playbooks/
    │           └── check-dgx.yml
    ├── ml-training/
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── infer.py
    │   ├── configs/
    │   │   ├── local.yaml
    │   │   ├── dev.yaml
    │   │   ├── eval-local.yaml
    │   │   └── eval-dev.yaml
    │   ├── data/
    │   │   └── toy-instructions.jsonl
    │   ├── slurm/
    │   │   ├── hello.sbatch
    │   │   ├── train.sbatch
    │   │   ├── evaluate.sbatch
    │   │   └── infer.sbatch
    │   └── scripts/
    │       ├── submit-train.sh
    │       ├── wait-for-slurm-job.sh
    │       └── print-run-metadata.sh
    ├── inference-api/
    │   ├── app/
    │   ├── Dockerfile
    │   └── tests/
    └── helm/
        └── inference-api/
            ├── Chart.yaml
            ├── values.yaml
            ├── values-dev.yaml
            ├── values-staging.yaml
            ├── values-prod.yaml
            └── templates/

## Current Status

Completed learning milestones:

- Local GitLab Runner installed on `agenthost`
- GitLab jobs forced onto local shell runner
- CI job executes on `agenthost.local`
- CI can SSH to DGX Spark
- Ansible can reach DGX Spark
- DGX prerequisite checks run from GitLab
- Slurm installed on DGX Spark
- Slurm hello job works
- GitLab can submit Slurm jobs
- GitLab can wait for Slurm jobs
- Repo-managed Slurm scripts can be copied to DGX
- Fake Python training runs under Slurm
- Fake training writes metadata

Current phase:

    GitLab -> Slurm -> fake training -> metadata.json

Next phase:

    GitLab -> Slurm -> fake evaluation -> metrics.json -> pass/fail gate

## GitLab Runner

The project uses a local GitLab Runner instead of GitLab shared runners.

Reason:

- GitLab shared runners cannot reach the private DGX Spark network.
- The local runner has access to SSH keys and local network routes.
- The shell executor makes Slurm and Ansible integration straightforward.

Expected job header:

    Running with gitlab-runner ...
    on sparky-mlops-runner ...
    Preparing the "shell" executor
    Running on agenthost.local

## Ansible

Ansible is used to inspect and eventually configure the DGX environment.

Inventory example:

    [dgx]
    spark-38fc ansible_host=192.168.1.82 ansible_user=chris ansible_python_interpreter=/usr/bin/python3.12

Useful commands:

    ansible -i infra/ansible/inventory/hosts.ini dgx -m ping
    ansible -i infra/ansible/inventory/hosts.ini dgx -m command -a "python3 --version"
    ansible -i infra/ansible/inventory/hosts.ini dgx -m command -a "nvidia-smi"
    ansible -i infra/ansible/inventory/hosts.ini dgx -m shell -a "which sbatch || true"

Run prerequisite playbook:

    ansible-playbook -i infra/ansible/inventory/hosts.ini infra/ansible/playbooks/check-dgx.yml

## Slurm

Slurm is the batch scheduler for training, evaluation, and batch inference.

The current learning setup starts with a single-node Slurm installation so the basic concepts are easy to debug.

Initial setup:

    spark-38fc
      munge
      slurmctld
      slurmd

The planned endstate is a two-node Slurm cluster across both DGX Spark machines.

Planned Slurm cluster:

    spark-38fc
      role: Slurm controller and compute node
      services:
        - munge
        - slurmctld
        - slurmd

    spark-1ac4
      role: Slurm compute node
      services:
        - munge
        - slurmd

In this endstate, `spark-38fc` runs the Slurm controller daemon and can also execute jobs. `spark-1ac4` joins the cluster as an additional compute node.

Both nodes must have:

- matching Slurm configuration
- matching Munge authentication key
- synchronized clocks
- consistent user accounts
- network connectivity on Slurm ports
- NVIDIA driver/tooling available
- compatible Python/runtime environment

Important Slurm commands:

    sinfo
    squeue
    sbatch
    srun
    scontrol show nodes
    scontrol show job <job-id>

Slurm is used for finite jobs, not long-running services.

Examples:

- training job
- evaluation job
- batch inference job
- GPU smoke test

## Training

The first trainer is fake by design.

Purpose:

- validate automation
- exercise Slurm
- exercise GitLab polling
- produce metadata
- avoid ML dependency complexity too early

Expected training flow:

    GitLab job
      -> copy ml-training/ to DGX
      -> submit train.sbatch
      -> wait for Slurm job
      -> print Slurm output
      -> print metadata.json

Expected output:

    /home/chris/sparky-mlops-runs/<run-id>/metadata.json

## Evaluation

The next planned milestone is fake evaluation.

Planned files:

    ml-training/evaluate.py
    ml-training/configs/eval-dev.yaml
    ml-training/slurm/evaluate.sbatch

Evaluation should:

- read training metadata
- calculate a fake score
- compare score to a threshold
- write `metrics.json`
- exit `0` on pass
- exit nonzero on fail

Target flow:

    train -> metadata.json -> evaluate -> metrics.json -> GitLab pass/fail

## Kubernetes And Helm

Kubernetes is not required for the current Slurm-first lab.

Slurm can run:

- tests
- training
- evaluation
- batch inference

Kubernetes and Helm become useful later for online serving:

    GitLab
      -> build Docker image
      -> Helm deploys FastAPI inference API
      -> Kubernetes runs /predict service

Future API:

    POST /predict

Example request:

    {
      "prompt": "Explain Slurm in one paragraph."
    }

Example response:

    {
      "model_version": "ci-123-456",
      "output": "Slurm is a workload manager..."
    }

## Planned Pipeline Stages

Expected stages:

    stages:
      - smoke
      - validate
      - infra
      - train
      - evaluate
      - package
      - deploy

Current useful jobs:

    runner-smoke
    dgx-ssh-smoke
    ansible-smoke
    dgx-prereq-check
    slurm-hello
    slurm-fake-train

Planned jobs:

    slurm-fake-evaluate
    python-unit-tests
    bash-lint
    ansible-syntax-check
    slurm-real-train
    slurm-real-evaluate
    slurm-batch-infer
    build-inference-image
    helm-lint
    deploy-inference-dev

## Learning Roadmap

### Phase 1: Automation Foundation

Status: mostly complete

Scope:

- GitLab Runner
- local shell executor
- SSH to DGX
- Ansible checks
- Slurm install
- Slurm hello job
- GitLab-submitted Slurm job
- fake training

### Phase 2: Evaluation Gate

Status: next

Scope:

- fake evaluator
- metrics output
- Slurm evaluation job
- GitLab pass/fail gate

### Phase 3: Batch Inference

Status: planned

Scope:

- prompt input file
- batch inference script
- Slurm inference job
- prediction output file

### Phase 4: Real Fine-Tuning

Status: planned

Scope:

- choose small model
- choose small dataset
- add Python ML dependencies
- request GPU through Slurm
- run LoRA or QLoRA fine-tuning
- write model artifacts

### Phase 5: Online Serving

Status: optional later

Scope:

- FastAPI `/predict`
- Docker image
- Helm chart
- Kubernetes deployment
- smoke test endpoint

## Success Criteria

Phase 1 is successful when GitLab can:

- run on the local runner
- reach DGX over SSH
- run Ansible checks
- submit Slurm jobs
- wait for Slurm completion
- print Slurm logs
- read training metadata

Full project success means GitLab can:

- train a model through Slurm
- evaluate the candidate model
- gate promotion based on metrics
- run batch inference
- optionally deploy an online inference API through Helm and Kubernetes

## Key Lessons

- GitLab shared runners cannot reach private lab infrastructure.
- Use a local shell runner for LAN/DGX access.
- Runner tags prevent jobs from landing on the wrong runner.
- The runner service user controls SSH keys, PATH, and permissions.
- SSH in CI must be non-interactive.
- Ansible provides repeatable environment checks.
- Slurm executes jobs; GitLab orchestrates them.
- CI must wait for Slurm jobs before reading output files.
- Repo-managed Slurm scripts are better than server-local scripts.
- Kubernetes is optional until online serving is needed.


This project can optionally use the `databricks/databricks-dolly-15k` dataset for instruction-tuning demos. The dataset is licensed under CC-BY-SA-3.0 by Databricks. The dataset is not vendored in this repository; training code downloads it from Hugging Face at runtime.