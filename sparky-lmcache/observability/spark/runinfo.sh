#!/usr/bin/env bash
# Run registry. Publishes the configuration of the current benchmark run as
# Prometheus labels via node_exporter's textfile collector, so every series can
# be attributed to the settings that produced it.
#
#   ./runinfo.sh start r2026-08-13-01 super p2p chunk=4224 l1=28 util=0.72 mode=align
#   ./runinfo.sh stop  r2026-08-13-01
#
# Join in Grafana with:
#   <any series> * on(instance) group_left(run_id,model,arm) bench_run_info
#
# Two results in this project have already been invalidated by a configuration
# detail nobody recorded. This is the fix.
set -euo pipefail
DIR=${TEXTFILE_DIR:-$HOME/node_exporter_textfile}
[ -d "$DIR" ] || { echo "no textfile dir at $DIR - run spark-exporters.sh first"; exit 1; }
[ -w "$DIR" ] || { echo "$DIR is not writable by $(whoami)"; exit 1; }
F="$DIR/bench_run.prom"
SEQF="$DIR/.bench_seq"
ACTION=${1:?start|stop}; RUN=${2:?run id}

if [ "$ACTION" = stop ]; then
  # Keep the labels and drop the value to 0, plus a stop timestamp. A bare
  # `bench_run_info{run_id} 0` loses which node, model and arm the run had, and
  # the dashboard then cannot tell "finished cleanly" from "never started".
  SEQ=$(cat "$SEQF" 2>/dev/null || echo 0)
  INFO_LINE=$(awk -v run="$RUN" '
    $1 ~ /^bench_run_info\{/ && index($0, "run_id=\"" run "\"") {
      sub(/ [^ ]+$/, " 0")
      print
      found=1
      exit
    }
    END { if (!found) print "" }
  ' "$F" 2>/dev/null || true)
  if [ -n "$INFO_LINE" ]; then
    INFO_LABELS=${INFO_LINE#bench_run_info}
    INFO_LABELS=${INFO_LABELS% 0}
  else
    INFO_LINE="bench_run_info{run_id=\"$RUN\",seq=\"$SEQ\",host=\"$(hostname)\"} 0"
    INFO_LABELS="{run_id=\"$RUN\",seq=\"$SEQ\",host=\"$(hostname)\"}"
  fi
  { echo '# HELP bench_run_info Active benchmark run and its configuration.'
    echo '# TYPE bench_run_info gauge'
    echo "$INFO_LINE"
    echo '# HELP bench_run_stop_seconds Unix time the run was closed.'
    echo '# TYPE bench_run_stop_seconds gauge'
    echo "bench_run_stop_seconds$INFO_LABELS $(date +%s)"
  } > "$F.tmp" && mv "$F.tmp" "$F"
  echo "run $RUN closed"
  exit 0
fi

# Monotonic per node. `bench_run_info == 1` cannot distinguish a live run from
# one whose stop never landed, which happened to weka-c-evict95 on spark-b
# when its wrapper was suspended before cleanup. A sequence number makes a stale
# entry obvious at a glance: the two nodes disagree, or it trails the newest
# run. Paired with bench_run_start_seconds the dashboard can show age directly.
SEQ=$(( $(cat "$SEQF" 2>/dev/null || echo 0) + 1 ))
echo "$SEQ" > "$SEQF"

MODEL=${3:?model}; ARM=${4:?arm: router-only|p2p|single}
shift 4
LABELS="run_id=\"$RUN\",seq=\"$SEQ\",model=\"$MODEL\",arm=\"$ARM\",host=\"$(hostname)\""
for kv in "$@"; do LABELS="$LABELS,${kv%%=*}=\"${kv#*=}\""; done

# Capture the settings that actually drift between runs rather than trusting
# the caller to pass them.
#
# clocks.max.sm is the hardware maximum (3003), not the applied cap. The
# applications clock is what `nvidia-smi -lgc` sets; fall back to max only if
# no cap is applied.
# Every one of these ends in `|| VAR=`, and that is not defensive habit.
# `set -euo pipefail` is on, so an assignment from a failing pipeline aborts the
# whole script, silently, with exit 1 and no message. That is exactly what
# happened once the workload moved to k3s: see IMG below.
CLK=$(nvidia-smi --query-gpu=clocks.applications.graphics --format=csv,noheader,nounits 2>/dev/null | head -1) || CLK=""
case "$CLK" in ''|*N/A*) CLK=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | head -1) || CLK="";; esac
CLKMAX=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits 2>/dev/null | head -1) || CLKMAX=""

# The workload image. k3s runs containerd, so `docker ps` sees only the obs
# containers, the filter below removes all of them, grep exits 1 with no
# output, pipefail propagates it and set -e kills the script before the
# `IMG=none` fallback can run. The run registry then goes silently missing and
# every series in the run is unattributable.
#
# Ask containerd first, since that is where the workload actually is. Fall back
# to docker for the bare-docker bench/* scripts, which do still use it.
IMG=$(sudo -n k3s ctr -n k8s.io containers ls 2>/dev/null \
       | awk '/dynamo-vllm-lmcache/ {print $2; exit}') || IMG=""
if [ -z "$IMG" ]; then
  IMG=$(docker ps --format '{{.Image}}' 2>/dev/null \
         | grep -vE 'cadvisor|promtail|node-exporter|dcgm' \
         | head -1) || IMG=""
fi
[ -n "$IMG" ] || IMG=none

LABELS="$LABELS,clock_mhz=\"${CLK:-NA}\",clock_max_mhz=\"${CLKMAX:-NA}\",image=\"$IMG\""

{ echo '# HELP bench_run_info Active benchmark run and its configuration.'
  echo '# TYPE bench_run_info gauge'
  echo "bench_run_info{$LABELS} 1"
  echo '# HELP bench_run_start_seconds Unix time the run started.'
  echo '# TYPE bench_run_start_seconds gauge'
  echo "bench_run_start_seconds{run_id=\"$RUN\"} $(date +%s)"
} > "$F.tmp" && mv "$F.tmp" "$F"

echo "run $RUN started: $LABELS"
