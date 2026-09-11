# Sparky MLOps Pipeline Design

## Purpose

This project is a hands-on MLOps learning lab for building a simple LLM fine-tuning pipeline on a DGX Spark cluster.

The goal is to learn by doing:

- GitLab CI/CD
- GitLab Runner
- Ansible
- Slurm
- Python
- Bash
- Later: Kubernetes, Helm, Docker, FastAPI, and model serving

The first implementation is intentionally simple. It uses fake training before real model fine-tuning so the automation flow can be understood independently from machine learning complexity.

## Current Architecture

```text
Developer
  |
  | git push
  v
GitLab.com
  |
  | pipeline job
  v
GitLab Runner on agenthost
  |
  | shell executor
  v
agenthost.local
  |
  | SSH / Ansible
  v
DGX Spark node: spark-38fc
  |
  | sbatch
  v
Slurm
  |
  | executes job
  v
Python training script
  |
  | writes logs and metadata
  v
/home/chris/sparky-mlops-runs