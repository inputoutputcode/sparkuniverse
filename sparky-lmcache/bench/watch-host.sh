#!/usr/bin/env bash
# Sample the benchmark client's health once a minute, to a CSV.
#
#   screen -dmS watch bash -c 'exec /path/to/bench/watch-host.sh'
#   tail -f runs/host-watch.csv
#
# Every failure in this project so far has come from the Mac rather than the
# cluster: macOS Local Network gating, Colima losing LAN egress twice, an ssh
# forward dying, the login session being torn down after midnight, bash 3.2,
# missing PyYAML. Each was diagnosed afterwards from whatever state happened to
# survive, which is why several took an hour.
#
# This does not fix any of them. It records when they start, so the next one
# takes a minute instead.
#
# Deliberately does not depend on Prometheus for anything except asking
# Prometheus whether it is working, because Prometheus is one of the things
# that fails.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$REPO/runs/host-watch.csv}
INTERVAL=${INTERVAL:-60}
ENDPOINT=${ENDPOINT:-http://localhost:8000}
PROM=${PROM:-http://localhost:9091}
GRAF=${GRAF:-http://localhost:3002}
NODE=${NODE:-10.0.0.11}

FIELDS="ts,sweep,aiperf,tunnel8000,endpoint,grafana,prom_up_ok,prom_up_total,node_direct,load1,mem_free_mb,disk_free_gb,ip"
[ -s "$OUT" ] || echo "$FIELDS" > "$OUT"
echo "watching every ${INTERVAL}s -> $OUT"

http() { curl -s -o /dev/null -m 4 -w '%{http_code}' "$1" 2>/dev/null || echo 000; }
n()    { echo "${1:-0}" | tr -d ' '; }

prev=""
while true; do
  ts=$(date -u +%FT%TZ)

  sweep=$(pgrep -fc 'run-sweep\.sh' 2>/dev/null || echo 0)
  aiperf=$(pgrep -fc 'aiperf profile' 2>/dev/null || echo 0)
  tunnel=$(pgrep -fc '8000:127\.0\.0\.1:30800' 2>/dev/null || echo 0)

  endpoint=$(http "$ENDPOINT/v1/models")
  grafana=$(http "$GRAF/api/health")

  # Prometheus' own view of its targets. If ok drops to 0 while total stays
  # put, the container has lost LAN egress, which is the Colima failure and is
  # invisible from the host because the host itself can still reach the nodes.
  up_json=$(curl -s -m 5 "$PROM/api/v1/query?query=up" 2>/dev/null)
  up_total=$(grep -o '"value"' <<<"$up_json" | wc -l | tr -d ' ')
  up_ok=$(grep -o '"1"\]' <<<"$up_json" | wc -l | tr -d ' ')

  # The same target from the host, to separate "the Sparks are down" from
  # "only the container cannot reach them".
  node_direct=$(http "http://$NODE:9100/metrics")

  load1=$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}')
  pagesize=$(sysctl -n hw.pagesize 2>/dev/null || echo 16384)
  free_pages=$(vm_stat 2>/dev/null | awk '/Pages free/ {gsub(/\./,"",$3); print $3}')
  mem_free_mb=$(( ${free_pages:-0} * pagesize / 1048576 ))
  disk_free_gb=$(df -g / 2>/dev/null | awk 'NR==2 {print $4}')
  ip=$(ipconfig getifaddr en0 2>/dev/null || echo none)

  line="$ts,$(n $sweep),$(n $aiperf),$(n $tunnel),$endpoint,$grafana,$(n $up_ok),$(n $up_total),$node_direct,${load1:-NA},$mem_free_mb,${disk_free_gb:-NA},${ip:-none}"
  echo "$line" >> "$OUT"

  # Print only when something changes. A quiet watchdog that prints every
  # minute is a watchdog nobody reads, and the transition is the whole point.
  state="$(n $sweep)|$(n $aiperf)|$(n $tunnel)|$endpoint|$grafana|$(n $up_ok)|$node_direct|${ip:-none}"
  if [ "$state" != "$prev" ]; then
    [ -n "$prev" ] && echo "$ts CHANGE  sweep=$sweep aiperf=$aiperf tunnel=$tunnel endpoint=$endpoint grafana=$grafana prom_up=$up_ok/$up_total node=$node_direct ip=$ip"
    prev="$state"
  fi

  sleep "$INTERVAL"
done
