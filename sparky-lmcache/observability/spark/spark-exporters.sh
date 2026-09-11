#!/usr/bin/env bash
# Exporters for one DGX Spark. Run on both nodes.
#   MINI=<mac-lan-ip> ./spark-exporters.sh
#
# Every container is named obs-* on purpose. The benchmark scripts wipe
# containers with `docker ps -a | grep -v '^dcgm' | xargs docker rm -f`, which
# would destroy these. Widen that guard before starting them:
#   sed -i "s/grep -v '\^dcgm'/grep -vE '^(dcgm|obs-)'/" ~/*.sh
set -euo pipefail
MINI=${MINI:?set MINI to the Mac Mini LAN address}
# Must match LOKI_PORT in observability/mini/.env (default 3102, not 3100 -
# the Mini's staycurio stack holds 3100).
LOKI_PORT=${LOKI_PORT:-3102}
# Deliberately under $HOME, not /var/lib. Over ssh there is no tty, so sudo
# fails, the script continues, and docker then creates the bind-mount path as
# root - leaving runinfo.sh unable to write and the run registry silently empty.
TEXTFILE=${TEXTFILE_DIR:-$HOME/node_exporter_textfile}
mkdir -p "$TEXTFILE"

docker rm -f obs-node obs-cadvisor obs-promtail >/dev/null 2>&1 || true

# ---- node_exporter -------------------------------------------------------
# --collector.infiniband is the important one: RoCE bypasses the kernel netdev
# path, so ConnectX-7 traffic does not appear in node_network_*_bytes_total.
# --collector.textfile carries the run registry written by runinfo.sh.
docker run -d --name obs-node --restart=unless-stopped \
  --net=host --pid=host \
  -v /:/host:ro,rslave \
  -v "$TEXTFILE":"$TEXTFILE":ro \
  quay.io/prometheus/node-exporter:v1.8.2 \
    --path.rootfs=/host \
    --collector.infiniband \
    --collector.textfile \
    --collector.textfile.directory="$TEXTFILE" \
    --collector.processes

# ---- cadvisor ------------------------------------------------------------
# v0.49.1 does NOT work here. Its bundled Docker API client is 1.41 and the
# Spark's daemon requires >= 1.44, so the docker factory fails to register and
# cadvisor silently falls back to the raw factory - producing a single id="/"
# series and no per-container metrics. The failure is only visible in
# `docker logs obs-cadvisor`. v0.52+ carries a newer client.
#
# /sys/fs/cgroup is mounted explicitly for cgroup v2. --docker_only is dropped
# so that a docker enumeration failure yields an error rather than silence.
docker run -d --name obs-cadvisor --restart=unless-stopped \
  -p 8080:8080 --privileged --device=/dev/kmsg \
  -v /:/rootfs:ro -v /var/run:/var/run:ro -v /sys:/sys:ro \
  -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
  -v /var/lib/docker/:/var/lib/docker:ro -v /dev/disk/:/dev/disk:ro \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  gcr.io/cadvisor/cadvisor:${CADVISOR_TAG:-v0.52.1} \
    --housekeeping_interval=5s \
    --store_container_labels=false \
    --docker=unix:///var/run/docker.sock

# ---- promtail ------------------------------------------------------------
# /var/log/pods and /var/log/containers are the k3s side. Without them the
# k3s-pods scrape job in promtail.yaml matches no files and fails silently,
# which is how this went unnoticed once the workload left bare docker.
# /var/log/pods holds symlink targets under /var/lib/rancher, so that path has
# to be visible too or every log line resolves to a dangling link.
docker run -d --name obs-promtail --restart=unless-stopped --net=host \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  -v /var/lib/docker/containers:/var/lib/docker/containers:ro \
  -v /var/log/pods:/var/log/pods:ro \
  -v /var/log/containers:/var/log/containers:ro \
  -v /var/lib/rancher:/var/lib/rancher:ro \
  -v "$HOME/promtail.yaml":/etc/promtail/config.yml:ro \
  -e LOKI_URL="http://${MINI}:${LOKI_PORT}/loki/api/v1/push" \
  grafana/promtail:3.3.2 \
    -config.file=/etc/promtail/config.yml -config.expand-env=true

# ---- dcgm-exporter -------------------------------------------------------
# Deployed separately, with ~/dcgm-counters.csv bind-mounted at
# /etc/dcgm-exporter/custom.csv. Note the -a: it is frequently left stopped
# rather than removed, and a filter without -a reports it as absent.
# The container is named 'dcgm-exporter' on one node and 'dcgm' on the other,
# so discover the name rather than assuming it.
DCGM=$(docker ps -a --format '{{.Names}}' | grep -m1 '^dcgm' || true)
if [ -z "$DCGM" ]; then
  echo "!! no dcgm-exporter container at all - GPU clocks, power and"
  echo "   temperature will be missing. Deploy it before benchmarking."
elif docker ps --format '{{.Names}}' | grep -qx "$DCGM"; then
  echo "dcgm container '$DCGM' already running"
else
  echo "dcgm container '$DCGM' is stopped - starting it"
  docker start "$DCGM" >/dev/null 2>&1 || echo "  !! failed to start"
  sleep 8
  if ! curl -sf -m3 localhost:9400/metrics >/dev/null; then
    echo "  !! started but 9400 is not answering. Most likely the counter csv"
    echo "     was rejected - this exporter exits on an unknown field id"
    echo "     rather than skipping it. Check: docker logs $DCGM --tail 30"
  fi
fi

sleep 5
echo "--- endpoints ---"
for p in 9100 8080 9400; do
  printf '  :%s  ' "$p"
  curl -sf "localhost:$p/metrics" >/dev/null && echo OK || echo DOWN
done

echo "--- signals that must be present ---"
curl -s localhost:9100/metrics | grep -c '^node_infiniband_' \
  | xargs -I{} echo "  infiniband series: {}   (0 means RoCE traffic is invisible)"
curl -s localhost:9100/metrics | grep -c '^node_memory_MemAvailable_bytes' \
  | xargs -I{} echo "  MemAvailable:      {}   (this predicts OOM on UMA, not the GPU metric)"
curl -s localhost:9400/metrics 2>/dev/null | grep -cE '^DCGM_FI_DEV_SM_CLOCK' \
  | xargs -I{} echo "  SM clock:          {}   (this is what catches the 513 MHz clamp)"
curl -s localhost:8080/metrics 2>/dev/null \
  | grep -c '^container_cpu_usage_seconds_total{.*name=' \
  | xargs -I{} echo "  per-container CPU: {}   (0 means cadvisor sees only the root cgroup)"

# Positive control for the textfile collector. Without this the run registry
# fails silently and every series ends up unattributable.
echo 'obs_textfile_probe 1' > "$TEXTFILE/_probe.prom"
sleep 6
if curl -s localhost:9100/metrics | grep -q '^obs_textfile_probe 1'; then
  echo "  textfile collector: OK  ($TEXTFILE)"
else
  echo "  textfile collector: FAILED - runinfo.sh will not reach prometheus"
  ls -la "$TEXTFILE"
fi
rm -f "$TEXTFILE/_probe.prom"
