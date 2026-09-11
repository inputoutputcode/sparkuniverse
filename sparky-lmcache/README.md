# Nemotron 3 with LMCache on two DGX Sparks

Serving NVIDIA Nemotron 3 (Nano 30B-A3B and Super, both NVFP4) under Dynamo's
aggregated deployment across two DGX Spark nodes, with LMCache multiprocess
KV-cache sharing over NIXL/RoCE. The question the benchmarks answer: **how many
more agentic-coding requests can two machines serve with P2P KV reuse than with
routing alone.**

| node | LAN | RoCE | role | ssh alias |
|---|---|---|---|---|
| spark-a | 10.0.0.11 | 10.1.0.11 | k3s server | `spark-a` |
| spark-b | 10.0.0.12 | 10.1.0.12 | k3s agent | `spark-b` |
| mac mini | — | — | control plane, AIPerf client, Grafana | — |

Commands use IP addresses. Neither `spark-a` nor `spark-a` resolves in DNS —
those names exist only in `~/.ssh/config`, so they work for `ssh` and `rsync`
but not for `curl`, where an unresolved host returns empty output that is
indistinguishable from a missing metric.

## Layout

| path | what |
|---|---|
| `observability/` | Grafana/Prometheus/Loki/Tempo on the Mini, exporters on the Sparks |
| `bench/` | measurement and serving scripts (pull from the Sparks first) |
| `traces/` | AgentX trace tooling; the trace data itself is gitignored |
| `deploy/` | k3s and Dynamo manifests |
| `runs/` | per-run results, one directory per `run_id` |

## Getting started

```bash
# 1. observability control plane on the Mini
cd observability/mini && docker compose up -d

# 2. exporters on both Sparks
cd ../spark
rsync -avP spark-exporters.sh promtail.yaml runinfo.sh 10.0.0.11:~/
rsync -avP spark-exporters.sh promtail.yaml runinfo.sh 10.0.0.12:~/
ssh 10.0.0.11 'MINI=<mac-lan-ip> ~/spark-exporters.sh'
ssh 10.0.0.12 'MINI=<mac-lan-ip> ~/spark-exporters.sh'

# 3. bring the measurement scripts into the repo
cd ../../bench && ./pull-from-sparks.sh
```

Full setup, including AIPerf on the Mini and the trace transfer, is in
[`observability/README.md`](observability/README.md).

## Run naming

Every run writes its configuration into Prometheus through
`observability/spark/runinfo.sh`, and results land in `runs/<run_id>/`:

```
runs/r2026-08-13-01/
  config.json      model, arm, chunk size, L1 size, util, cache mode, clock cap
  aiperf/          raw AIPerf artifacts
  summary.md       what it showed
```

Results without a `run_id` are not usable. Two conclusions in this project were
invalidated by a configuration detail nobody recorded.

## Conventions

- Container names on the Sparks are `obs-*` for observability and short names
  for workload. Cleanup guards must exclude `^(dcgm|obs-)`.
- Nothing is claimed here that was not measured. Where a number is derived, the
  derivation and its cross-check are both stated in `docs/findings.md`.
