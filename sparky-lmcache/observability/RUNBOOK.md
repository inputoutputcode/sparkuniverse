# Bring-up runbook

Ordered fixes for everything currently broken or unverified. Run from the Mac
Mini, in the `/Users/chris/code/sparky` clone. `./verify.sh` after each phase.

Phases are ordered by what blocks what: phase 1 tells you which later items are
already broken, and phases 2–3 must happen before any benchmark or the run will
both miss a node and destroy the exporters.

```bash
cd /Users/chris/code/sparky/nemotron-lmcache/observability
chmod +x verify.sh spark/*.sh
./verify.sh          # baseline - expect many failures
```

---

## 1. Restart Prometheus with the corrected config

The running stack was started from a clone whose `prometheus.yml` still had
`10.0.0.17` — spark-b's WiFi address. Nothing it has collected for node 2
is trustworthy.

```bash
cd mini
docker compose down
docker compose up -d
sleep 15
open "http://localhost:9091/targets"
```

`/targets` is the page that would have caught this on day one. Every job should
list `10.0.0.11` and `10.0.0.12`; `vllm` and `dynamo` down between runs
is expected, nothing else should be.

---

## 2. Exporters on both nodes

spark-b has none at all. spark-a needs the cadvisor and textfile fixes.

```bash
cd ../spark
for ip in 10.0.0.11 10.0.0.12; do
  rsync -avP spark-exporters.sh promtail.yaml runinfo.sh \
             protect-exporters.sh dcgm-counters.csv $ip:~/
done

for ip in 10.0.0.11 10.0.0.12; do
  echo "=== $ip"
  ssh $ip 'docker rm -f obs-node obs-cadvisor obs-promtail 2>/dev/null
           MINI=10.0.0.94 ~/spark-exporters.sh'
done
```

`MINI` is the Mini's LAN address — promtail pushes logs there. Confirm it:
`ipconfig getifaddr en0`.

The script's own output ends with the signal checks, including a positive
control that writes a probe metric and reads it back out of `/metrics`. A
dead textfile collector means the run registry fails silently.

---

## 3. Protect the exporters from the benchmark scripts

Several benchmark scripts run
`docker ps -a | grep -v '^dcgm' | xargs -r docker rm -f`, which deletes every
`obs-*` container. Dry run first — it prints the files and lines it would touch,
and separately lists any script that deletes containers with *no* guard at all.

```bash
for ip in 10.0.0.11 10.0.0.12; do
  echo "=== $ip"; ssh $ip '~/protect-exporters.sh'
done
# review, then
for ip in 10.0.0.11 10.0.0.12; do ssh $ip '~/protect-exporters.sh --apply'; done
```

---

## 4. Expand the dcgm counter set

`~/dcgm-counters.csv` is already bind-mounted at
`/etc/dcgm-exporter/custom.csv`, so this is a file copy plus a restart. Note
`docker start` as well as `restart`: the container is often left stopped rather
than removed, and a `docker ps` filter without `-a` reports it as absent.

```bash
for ip in 10.0.0.11 10.0.0.12; do
  ssh $ip 'docker start dcgm-exporter 2>/dev/null
           docker restart dcgm-exporter >/dev/null; sleep 10
           echo "--- $(hostname)"
           curl -s localhost:9400/metrics | grep -oE "^DCGM_[A-Z0-9_]+" | sort -u'
done
```

Unsupported fields are silently dropped, so **the output is the answer** to
which ones GB10 implements. `DCGM_FI_DEV_MEM_CLOCK` is already known
unsupported. The two worth watching for are `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`
and `DCGM_FI_PROF_DRAM_ACTIVE` — if they survive, the prefill-efficiency
question and the decode floor model become measurable instead of fitted.

---

## 5. Calibrate the InfiniBand counters

Presence is not correctness. Some drivers report `port_data_*` in 4-byte words,
and a panel that is 4× wrong still looks plausible. There is a known reference:
**109.23 Gb/s = 13.65 GB/s** measured on this fabric.

```bash
ib() { ssh $1 'cat /sys/class/infiniband/*/ports/*/counters/port_rcv_data' | paste -sd+ - | bc; }

B0=$(ib 10.0.0.12)
# run a cross-node NIXL transfer here (nixl-probe.py or an LMCache P2P fetch)
B1=$(ib 10.0.0.12)
echo "delta words: $((B1-B0))   bytes if x4: $(( (B1-B0)*4 ))"
```

Divide by the transfer's wall time and compare against 13.65 GB/s. A result 4×
high or low tells you the divisor the panel needs.

While here, settle **which rail** carries P2P. Two 200 Gb/s fabrics exist —
`enp1s0f1np1` on 192.168.177.x and `enP2p1s0f1np1` on 192.168.178.x. If NIXL
only ever uses one, half the fabric is idle during the test meant to
demonstrate P2P's value.

```bash
for ip in 10.0.0.11 10.0.0.12; do
  ssh $ip 'for d in /sys/class/infiniband/*/ports/1; do
             echo "$(basename $(dirname $(dirname $d))) rcv=$(cat $d/counters/port_rcv_data)"
           done'
done
```

Run that before and after a transfer; only the rail in use moves.

---

## 6. Verify the log path end to end

promtail starting is not evidence a log line arrived.

```bash
curl -s "http://localhost:3102/loki/api/v1/label/container/values" | jq .
curl -sG "http://localhost:3102/loki/api/v1/query_range" \
  --data-urlencode 'query={container="obs-node"}' \
  --data-urlencode "start=$(($(date +%s)-3600))000000000" | jq '.data.result | length'
```

Zero streams means promtail cannot reach the Mini on 3102 — check `LOKI_PORT`
in `mini/.env` matches `LOKI_PORT` in `spark-exporters.sh`.

---

## 7. Exercise the run registry

The single most valuable piece, and it has never been run once.

```bash
ssh 10.0.0.11 '~/runinfo.sh start smoke-01 nano single chunk=2128 l1=12 util=0.40 mode=align'
sleep 10
curl -s 10.0.0.11:9100/metrics | grep '^bench_run_info'
curl -sG "http://localhost:9091/api/v1/query" --data-urlencode 'query=bench_run_info' | jq '.data.result'
ssh 10.0.0.11 '~/runinfo.sh stop smoke-01'
```

The label set must reach Prometheus, not just node_exporter — that join is what
makes every other series attributable.

---

## 8. Time sync

Cross-node latency attribution is meaningless if the clocks drift, and NTP
failure is silent.

```bash
for ip in 10.0.0.11 10.0.0.12; do
  ssh $ip 'timedatectl show -p NTPSynchronized --value; chronyc tracking 2>/dev/null | head -3'
done
```

---

## 9. Client-side coverage on the Mini

Blocked on the Homebrew ownership question. Without it, you cannot rule out the
client as the bottleneck — and the Mini also runs the `staycurio` stack, so its
load is neither zero nor constant.

```bash
brew install node_exporter && brew services start node_exporter
curl -s localhost:9100/metrics | head -3
```

Then quiesce staycurio for benchmark runs:

```bash
docker ps --format '{{.Names}}' | grep -v '^obs-' > /tmp/paused-containers.txt
xargs -r docker stop < /tmp/paused-containers.txt      # restore: docker start
```

---

## 10. vLLM metrics during an actual run

Only meaningful while serving, so it has never been checked.

```bash
ssh 10.0.0.11 'BT=4255 ~/serve-frozen.sh' &
sleep 240
curl -s 10.0.0.11:8011/metrics | grep -E 'vllm:(prefix_cache_(queries|hits)_total|gpu_cache_usage_perc|num_requests_running)'
```

Then the three-way consistency check that catches a silently no-opping
connector: vLLM `prefix_cache_hits`, LMCache `retained keys` in the logs, and
AIPerf's TTFT distribution are three views of the same event.

---

## 11. LMCache metrics — add the flag to every server invocation

**Resolved: no exporter needs writing.** The MP server has a built-in
Prometheus endpoint, disabled by default. Add `--http-port 9500` to every
`lmcache server` invocation — the benchmark scripts and the Dynamo deployment
both start one — and `prometheus.yml` already scrapes it.

```bash
lmcache server --l1-size-gb <N> --chunk-size <2128|4224> \
               --eviction-policy LRU --http-port 9500
```

Default is 9090, which collides with conventional Prometheus ports; 9500 keeps
it clear. Confirmed working:

    lmcache_mp_l1_usage_ratio            occupancy vs the 0.80 eviction watermark
    lmcache_mp_l1_memory_usage_bytes
    lmcache_mp_l1_eviction_loop_ticks_total
    lmcache_mp_active_p2p_lookup_jobs    P2P fetches in flight
    lmcache_mp_active_prefetch_jobs
    lmcache_mp_l2_adapters
    lmcache_mp_event_bus_{queue_depth,dropped_events_total,drain_lag_seconds}

More appear once traffic flows — lookup and throughput counters are created on
first event.

`--http-port` applies **only** when `--otlp-endpoint` is unset; they are
mutually exclusive. Tracing (`--enable-tracing --otlp-endpoint
http://<mini>:4317`) would push spans to Tempo but disable the Prometheus
endpoint, so getting both needs an OTel collector in front — deferred.

---

## Still to be built

- **Dynamo metrics.** Endpoints unenumerated; the NATS and etcd ports in
  `prometheus.yml` are guesses.
- **OTLP producers.** Tempo is running and receiving nothing. vLLM has
  `--otlp-traces-endpoint`; Dynamo and LMCache need investigating.
- **AIPerf ingestion.** Artifacts exist; nothing loads them into the timeline.
- **Exporter overhead.** One identical run with and without the exporters,
  to quote a number rather than assume it is negligible.
