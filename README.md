# Playground for DGX Spark cluster

Experiments, automation, and benchmarks for running AI workloads across local NVIDIA DGX Spark systems.

## Projects

[sparky-dynamo](./sparky-dynamo)

Automated setup for a two-node DGX Spark cluster running NVIDIA Dynamo on K3s over ConnectX-7 networking. Includes GPU Operator configuration, RDMA-related host setup, Dynamo platform deployment, and cluster teardown scripts.

 
[sparky-lmcache](./sparky-lmcache)

Benchmarking distributed LLM inference across two DGX Sparks using NVIDIA Dynamo, LMCache, NIXL, and RoCE. Focuses on measuring the impact of shared KV cache reuse on agentic workloads, with reproducible runs, observability, traces, and AIPerf results.

 
[sparky-mlops](./sparky-mlops)

Hands-on MLOps lab for building a reproducible LLM training and evaluation pipeline on DGX Spark. Uses GitLab CI/CD, Ansible, Slurm, Python, and Bash to automate cluster validation, training, evaluation, and model promotion workflows.
