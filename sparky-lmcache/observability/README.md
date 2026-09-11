# Mac Mini control plane — setup

Mac Mini runs the observability stack and the AIPerf client. The two Sparks run
only exporters and the workload. Everything is pulled from the Mini except logs
and traces, which are pushed.

| node | LAN | RoCE | role | ssh alias |
|---|---|---|---|---|
| spark-a | **10.0.0.11** | 10.1.0.11 | k3s server | `spark-a` |
| spark-b | **10.0.0.12** | 10.1.0.12 | k3s agent | `spark-b` |
| mac mini | — | — | control plane, AIPerf client | — |

**Commands in this document use IP addresses, deliberately.** Neither
`spark-a` nor `spark-a` resolves in DNS — the aliases exist only in
`~/.ssh/config`, so they work for `ssh` and `rsync` but not for `curl`, and an
unresolved host returns empty output that looks exactly like a missing metric.

The `spark-a` / `spark-b` form is still canonical in prose and in the
`node:` labels in `prometheus.yml`, because it matches `hostname` output and is
what Grafana displays.

RoCE addresses are the Spark-to-Spark fabric. The Mini talks to both over the
1 GbE LAN and never touches 192.168.177.x.

**Every section of this document runs on the Mac Mini.** Nothing here is meant
to be executed on a Spark directly — the Sparks are targets, reached over ssh.
Keep one repo clone, on the Mini, as the source of truth; clones on the Sparks
will drift and you will not notice which one you edited.

---

## 0. Prerequisites

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install --cask orbstack          # or docker desktop; orbstack is lighter on apple silicon
brew install python@3.12 rsync jq node_exporter
```

Verify the container runtime is on arm64 and can reach the Sparks:

```bash
docker version --format '{{.Server.Arch}}'     # expect arm64
for h in 10.0.0.11 10.0.0.12; do ssh user@$h 'hostname; uptime'; done
```

If ssh prompts for a password, push a key first — every later step assumes
non-interactive ssh:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/spark -N ''
for h in 10.0.0.11 10.0.0.12; do ssh-copy-id -i ~/.ssh/spark user@$h; done
cat >> ~/.ssh/config <<'EOF'
Host spark-a spark-a 10.0.0.11
  HostName 10.0.0.11
  User chris
  IdentityFile ~/.ssh/spark
  ControlPath none

Host spark-b spark-b 10.0.0.12
  HostName 10.0.0.12
  User chris
  IdentityFile ~/.ssh/spark
  ControlPath none
EOF
```

Three details that each cost time to discover:

- **The IP must be in the `Host` pattern list.** Commands in this repo use IPs;
  without them ssh falls through to the local username and prompts for a
  password.
- **`ControlPath none`** overrides colima's `~/.colima/ssh_config`, which is
  Included at the top of `~/.ssh/config` and sets a *fixed* socket path with no
  `%h` — a latent way for every host to multiplex onto one connection.
- **Use the wired addresses.** Both Sparks are on the LAN twice: spark-a at
  .82 wired / .2 WiFi, spark-b at .229 wired / .17 WiFi. Pointing an alias at
  the WiFi address puts trace transfers and metric scrapes on a shared,
  variable-latency link.

Verify before anything scrapes them — the two IPs must disagree:

```bash
for h in spark-a spark-b 10.0.0.11 10.0.0.12; do
  printf '%-16s -> ' $h; ssh -o ConnectTimeout=5 $h hostname
done
```

---

## 1. Pull the traces off 10.0.0.11

Traces live in `~/traces` on spark-a. Check sizes first — `cc-weka/` is the
raw source dump and is excluded below; the `cc-weka-s*.jsonl` subsets are what
the benchmarks consume.

```bash
ssh 10.0.0.11 'du -sh ~/traces ~/traces/cc-weka ~/aiperf/artifacts; ls -la ~/traces'
```

```bash
cd <clone>/nemotron-lmcache
mkdir -p traces/data runs/_prior-artifacts
rsync -avP --exclude 'cc-weka/' 10.0.0.11:'~/traces/'  traces/data/
rsync -avP 10.0.0.11:'~/aiperf/artifacts/'             runs/_prior-artifacts/
rsync -avP 10.0.0.11:'~/weka-*.py'                     traces/scripts/
rsync -avP 10.0.0.11:'~/trace-analyze.py'              traces/scripts/
```

One glob per rsync call. A multi-line quoted file list is passed to the remote
shell as a single filename and fails with "No such file or directory".

Both destinations are gitignored — the traces are 47 MB of subsets and the
artifacts are raw run output. `cc-weka/` is the 1.75 GB raw source, needed only
to build new subsets.

Checksum both ends. A truncated trace produces a benchmark that runs fine and
means nothing — but note the guard on an empty remote list. Without it, `diff`
compares nothing against nothing and reports success, which is how this step
passed the first time it was run while copying zero files:

```bash
ssh 10.0.0.11 'cd ~/traces && md5sum *.jsonl' | sort > /tmp/remote.md5
(cd traces/data && md5sum *.jsonl) | sort > /tmp/local.md5
if [ ! -s /tmp/remote.md5 ]; then echo "FAIL: no remote checksums"
elif diff /tmp/remote.md5 /tmp/local.md5; then echo "OK: $(wc -l < /tmp/local.md5) files match"
else echo "FAIL: mismatch"; fi
```

---

## 2. AIPerf on the Mini

Do not guess the package — mirror whatever the Spark has installed:

```bash
ssh 10.0.0.11 '~/aiperf/venv/bin/pip freeze' > /tmp/aiperf-freeze.txt
grep -iE 'aiperf|genai|perf' /tmp/aiperf-freeze.txt
```

Then build the same environment locally:

```bash
python3.12 -m venv ~/aiperf/venv
~/aiperf/venv/bin/pip install --upgrade pip
~/aiperf/venv/bin/pip install <package-name-from-the-freeze>
~/aiperf/venv/bin/aiperf --version
```

If any dependency is Linux-only, fall back to running AIPerf in a container on
the Mini rather than fighting the wheels — the client is not on the measurement
path, so a container costs nothing.

**Tokenizer.** `--tokenizer` must be the HF repo id, not a local path. That
requires network access and, for gated NVIDIA repos, a token:

```bash
~/aiperf/venv/bin/pip install huggingface_hub
export HF_TOKEN=...        # put in ~/.zshrc
~/aiperf/venv/bin/hf auth whoami
```

Verify the tokenizer resolves before a long run — this is the failure that
wastes an hour of benchmark time and produces plausible-looking wrong ISLs.

---

## 3. Observability stack

Everything below runs from the repo clone. There is no separate
`~/spark-bench` directory — that path appeared in an earlier draft and is gone.

```bash
cd <clone>/nemotron-lmcache/observability/mini
docker compose up -d
docker compose ps
```

Colima needs sizing first; the default 2 CPU / 2 GB will not run four services:

```bash
colima stop && colima start --cpu 4 --memory 8 --disk 100
unset DOCKER_HOST          # colima's context is overridden if this is set
docker version --format '{{.Server.Arch}}'   # expect arm64
```

Endpoints:

| service | url | why not the usual port |
|---|---|---|
| Grafana | http://localhost:3002 (anonymous admin) | `staycurio-web` holds 3000 |
| Prometheus | http://localhost:9091 | `staycurio-prometheus` holds 9090 |
| Loki | http://localhost:3102 | `staycurio-loki` holds 3100 |
| Tempo | OTLP on 4317/4318; query API internal only | — |

This Mac also runs the `staycurio` stack — nineteen containers including its own
Prometheus, Grafana, Loki and promtail. The defaults above sidestep it so both
can run at once. Override in `.env` if needed (see `.env.example`).

**Container-internal ports are unchanged.** Grafana reaches `prometheus:9090`,
`loki:3100` and `tempo:3200` over the compose network regardless of what is
published on the host. Only the published side moves.

Tempo's 3200 is not published — something else on this host binds it, and
Grafana reaches Tempo over the compose network anyway. To query it directly:
`docker compose exec tempo wget -qO- localhost:3200/api/echo`.

If an image tag 404s, switch that service to `:latest`, then pin whatever
version you actually got. Reproducibility matters more than novelty here.

### "address already in use" on colima

Colima forwards container ports from the VM to the host through an ssh
multiplexer. When a container fails to start, **the forward leaks** — so the
next `docker compose up` reports the port as taken by our own aborted attempt.
`lsof` shows it held by an `ssh` process pointing at
`~/.colima/_lima/colima/ssh.sock`; that is colima itself, not a rival service.

    colima restart      # clears leaked forwards

Do not kill that ssh PID directly — it is the docker socket, and killing it
takes the daemon with it.

Host ports are overridable via `.env` (see `.env.example`) for genuine
collisions, but a leaked forward is not one.

Also run node_exporter on the Mini itself. If AIPerf saturates the Mac's CPU or
its link, the results describe the client, and you need the data to rule that
out:

```bash
brew services start node_exporter
curl -s localhost:9100/metrics | head -3
```

---

## 4. Exporters on both Sparks

> **Run from the Mac Mini.** Everything here pushes *to* the Sparks. Running it
> on a Spark makes it ssh to itself, which appears to work and silently leaves
> the other node without exporters.

```bash
rsync -avP spark-exporters.sh promtail.yaml 10.0.0.11:~/
rsync -avP spark-exporters.sh promtail.yaml 10.0.0.12:~/
ssh 10.0.0.11 'MINI=<mac-lan-ip> ~/spark-exporters.sh'
ssh 10.0.0.12 'MINI=<mac-lan-ip> ~/spark-exporters.sh'
```

Every container is named with an `obs-` prefix for one reason:

> **The benchmark scripts delete containers.** Several run
> `docker ps -a --format '{{.Names}}' | grep -v '^dcgm' | xargs -r docker rm -f`,
> which spares dcgm-exporter and nothing else. That will destroy the `obs-*`
> containers on the next run — dcgm-exporter was already lost once this way.

`protect-exporters.sh` finds the affected scripts rather than rewriting every
`.sh` in the home directory, and shows the change before making it:

```bash
rsync -avP protect-exporters.sh 10.0.0.11:~/ && ssh 10.0.0.11 '~/protect-exporters.sh'
rsync -avP protect-exporters.sh 10.0.0.12:~/ && ssh 10.0.0.12 '~/protect-exporters.sh'
# review the output, then
ssh 10.0.0.11 '~/protect-exporters.sh --apply'
ssh 10.0.0.12 '~/protect-exporters.sh --apply'
```

It also lists any script that deletes containers with *no* guard at all — those
are worse and need fixing by hand. Known to carry the narrow guard:
`l2-bytes-per-token.sh`, `l2-bpt2.sh`, `l2-bpt3.sh`, `segprefix.sh`,
`dtype-nll.sh`. The older ones (`ablate.sh`, `soak.sh`, `floor-test.sh`,
`mtp-backend.sh`, `marlin-util.sh`) predate it and need checking.

---

## 5. Verify each signal moves

Do not trust a panel that has never been shown a known stimulus.

Use IP addresses here. The `spark-a` alias lives in `~/.ssh/config` and means
nothing to `curl`, which does its own DNS — an unresolved hostname returns
empty output that looks exactly like a missing metric. Display names in Grafana
come from the `node:` labels in `prometheus.yml`, so no `/etc/hosts` entry is
needed anywhere.

```bash
S=10.0.0.11        # spark-a;  10.0.0.12 for spark-b

# GPU: clocks and power should track a load
curl -s $S:9400/metrics | grep -E 'DCGM_FI_DEV_(SM_CLOCK|GPU_TEMP|POWER_USAGE)'

# unified memory: this is the number that predicts OOM, not the GPU one
curl -s $S:9100/metrics | grep -E '^node_memory_MemAvailable_bytes'

# ConnectX-7: RoCE bypasses netdev, so these are the real counters
curl -s $S:9100/metrics | grep -E '^node_infiniband_port_data_(received|transmitted)_bytes_total'

# containers
curl -s $S:8080/metrics | grep -c '^container_cpu_usage_seconds_total'

# vLLM, only while serving - empty between runs is expected
curl -s $S:8011/metrics | grep -E 'vllm:(prefix_cache|gpu_cache_usage|num_requests)'
```

**Calibrate the InfiniBand counters before trusting them.** Some drivers report
`port_data_*` in 4-byte words and node_exporter's conversion has varied across
versions. There is a known reference — 109.23 Gb/s measured on this fabric — so
record the counters, run a cross-node NIXL transfer, record again, and check the
derived rate. A result 4x off means the panel needs a divisor.

The infiniband counters deserve a positive control of their own: record them,
run a cross-node NIXL transfer, record again. If they do not move, the P2P
traffic panel is decorative.

### dcgm counter set

The stock deployment on these machines exports four fields: SM clock, GPU temp,
power and util.

**`DCGM_FI_DEV_SM_CLOCK` is the metric that catches the 513 MHz USB-PD clamp** —
the clamped value is unmistakable in the clock series, which is how
`clockwatch.sh` found it. Alert on that. `CLOCK_THROTTLE_REASONS`, added below,
only supplies attribution afterwards: USB-PD clamp vs thermal vs power cap vs
sync boost. Useful when a run degrades and you need the cause; not needed to see
that it happened.

The fields genuinely missing from the stock set are memory clock, framebuffer,
energy, XID errors and the profiling counters.

`dcgm-counters.csv` is already bind-mounted at `/etc/dcgm-exporter/custom.csv`,
so this is an edit plus a restart:

```bash
rsync -avP dcgm-counters.csv 10.0.0.11:~/dcgm-counters.csv
rsync -avP dcgm-counters.csv 10.0.0.12:~/dcgm-counters.csv
for h in 10.0.0.11 10.0.0.12; do
  ssh $h 'docker restart $(docker ps -qf name=dcgm) >/dev/null; sleep 10
          echo "--- $(hostname)"
          curl -s localhost:9400/metrics | grep -oE "^DCGM_[A-Z0-9_]+" | sort -u'
done
```

Fields unsupported on GB10 simply never appear, so the output *is* the answer to
which ones work. Two are worth particular attention if they survive:
`DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` speaks directly to the unresolved question of
why prefill runs at 36% of the machine's own bf16 peak, and
`DCGM_FI_PROF_DRAM_ACTIVE` is the measured counterpart to the bandwidth term in
the decode floor model. Both would replace inference with observation.

Framebuffer fields need validating against `node_memory_MemAvailable_bytes`
before use — on unified memory they may mirror system memory or mean nothing.

---

## 6. Run registry

Every series needs to say which configuration produced it. node_exporter's
textfile collector is the least invasive way — the benchmark script writes a
file, Prometheus picks up the labels:

```bash
# on a Spark, at the start of every run
~/runinfo.sh start r2026-08-13-01 super p2p chunk=4224 l1=28 util=0.72 mode=align
# at the end
~/runinfo.sh stop  r2026-08-13-01
```

Join in Grafana with `* on(instance) group_left(run_id,model,arm) bench_run_info`.

Without this you get graphs you cannot attribute. Two results in this project
have already been invalidated by a configuration detail nobody recorded.

---

## Open items before this is trustworthy

- **LMCache metrics** — vLLM's `/metrics` exposes nothing LMCache-related
  (verified). The image has `lmcache/v1/mp_observability/subscribers/metrics/`
  building OTel counters. Check whether it exports OTLP; if not, a log-derived
  exporter over `Stored` / `Retrieved` / `retained keys` / `triggering eviction`
  is the fallback, and those lines parse cleanly.
- **Dynamo metrics** — endpoints and the router's decision series still need
  enumerating; scrape NATS and etcd too, since router state flows through them
  and a stalled etcd looks like a router bug.
- **Time sync** — chrony on both Sparks against the Mini, with offset graphed.
  Cross-node P2P latency attribution is meaningless if clocks drift, and NTP
  failure is silent.
- **Exporter overhead** — the exporters consume UMA that competes with the
  model. Measure one identical run with and without them rather than assuming
  it is negligible.
