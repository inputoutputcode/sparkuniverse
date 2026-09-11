# Sparky MLOps Lab: Endstate And Implementation Plan

## Project Summary

Sparky MLOps is a hands-on learning project for building an end-to-end LLM fine-tuning automation pipeline on a local DGX Spark cluster.

The project is intentionally designed as a learning lab rather than a production MLOps platform. Each technology has a clear role:

- GitLab controls automation.
- GitLab Runner provides local execution access from GitLab to the private lab network.
- Ansible prepares and validates the DGX environment.
- Slurm schedules fine-tuning, evaluation, and batch inference jobs.
- Python implements training, evaluation, and inference logic.
- Bash provides small operational wrappers and CI glue.
- Kubernetes later runs the online inference service.
- Helm later packages and deploys the inference service.
- Docker later packages the inference API runtime.

The first project phase is Slurm-first. Kubernetes and Helm are intentionally delayed until the batch workflow is understood.

## Project Rule

No agent may create, edit, delete, format, or refactor project files.

Agents may only:

- explain concepts
- review snippets pasted by the user
- suggest commands
- help debug pasted errors
- propose exercises
- provide file content for the user to manually create

All project file changes must be made manually by the user.

## Endstate Vision

The endstate is a working local MLOps pipeline where a Git commit can trigger a controlled workflow:

```text
Developer pushes code
  |
  v
GitLab pipeline starts
  |
  v
Local GitLab Runner executes on agenthost
  |
  v
GitLab job runs validation
  |
  v
Ansible checks or prepares DGX Spark
  |
  v
GitLab submits Slurm training job
  |
  v
Slurm runs Python fine-tuning on DGX GPU
  |
  v
Training writes versioned artifacts and metadata
  |
  v
GitLab submits Slurm evaluation job
  |
  v
Evaluation writes metrics and pass/fail result
  |
  v
GitLab promotes accepted model version
  |
  v
Later: GitLab builds inference image
  |
  v
Later: Helm deploys FastAPI inference service to Kubernetes
  |
  v
/predict API serves selected model version