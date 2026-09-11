#!/usr/bin/env bash
# Read-only check of every observability signal. Run from the Mac Mini.
#
#   ./verify.sh
#
# Changes nothing. Every line is PASS, FAIL or WARN with the reason. A signal
# that is merely *present* is not enough - several checks below assert the
# value is sane, because a metric that exists and reads zero forever looks
# identical to a working one on a dashboard.
#
# macOS ships bash 3.2: no mapfile, no associative arrays.
set -u

N1_IP=${N1_IP:-10.0.0.11};  N1=spark-a
N2_IP=${N2_IP:-10.0.0.12}; N2=spark-b
PROM=${PROM:-localhost:9091}
LOKI=${LOKI:-localhost:3102}
GRAF=${GRAF:-localhost:3002}

pass=0; fail=0; warn=0
ok()   { printf '  \033[32mPASS\033[0m  %-34s %s\n' "$1" "${2:-}"; pass=$((pass+1)); }
no()   { printf '  \033[31mFAIL\033[0m  %-34s %s\n' "$1" "${2:-}"; fail=$((fail+1)); }
meh()  { printf '  \033[33mWARN\033[0m  %-34s %s\n' "$1" "${2:-}"; warn=$((warn+1)); }
hdr()  { printf '\n=== %s\n' "$1"; }

get() { curl -sf -m 5 "$1" 2>/dev/null; }

# ---------------------------------------------------------------- mini stack
hdr "Mac Mini control plane"

if get "http://$PROM/-/healthy" >/dev/null; then ok "prometheus up" "$PROM"
else no "prometheus up" "$PROM unreachable"; fi

if get "http://$LOKI/ready" >/dev/null; then ok "loki up" "$LOKI"
else no "loki up" "$LOKI unreachable"; fi

if get "http://$GRAF/api/health" >/dev/null; then ok "grafana up" "$GRAF"
else no "grafana up" "$GRAF unreachable"; fi

# The stale-config trap: prometheus was started from a clone whose config had
# the wrong address for node 2.
T=$(get "http://$PROM/api/v1/targets")
if [ -n "$T" ]; then
  if echo "$T" | grep -q "$N2_IP"; then ok "prometheus knows $N2" "$N2_IP"
  else no "prometheus knows $N2" "config predates the address fix - restart with the repo config"; fi
  UPC=$(echo "$T" | tr ',' '\n' | grep -c '"health":"up"')
  DNC=$(echo "$T" | tr ',' '\n' | grep -c '"health":"down"')
  [ "$UPC" -gt 0 ] && ok "targets up" "$UPC" || no "targets up" "none"
  [ "$DNC" -gt 0 ] && meh "targets down" "$DNC (vllm/dynamo down between runs is expected)"
else
  no "prometheus targets api" "no response"
fi

# Grafana datasources are provisioned from YAML and have never been opened.
DS=$(get "http://$GRAF/api/datasources")
if [ -n "$DS" ]; then
  for d in prometheus loki tempo; do
    echo "$DS" | grep -qi "\"type\":\"$d\"" && ok "datasource $d" || no "datasource $d" "not provisioned"
  done
else
  meh "grafana datasources" "api needs auth or grafana still starting"
fi

# Tempo receives nothing until an OTLP exporter is configured somewhere.
if echo "$T" | grep -q 'tempo'; then :; fi
# The label-values API returns {"status":"success","data":[...]} - the key is
# "data", not "values".
TR=$(get "http://$LOKI/loki/api/v1/label/container/values")
NC=$(echo "$TR" | sed -n 's/.*"data":\[\(.*\)\].*/\1/p' | tr ',' '\n' | grep -c '"')
if [ "${NC:-0}" -gt 0 ]; then
  ok "loki receiving logs" "$NC container streams"
else
  no "loki receiving logs" "no streams - check promtail_sent_entries_total on the nodes"
fi

# ------------------------------------------------------------------- sparks
check_node() {
  NAME=$1; IP=$2
  hdr "$NAME ($IP)"

  if ! ssh -o ConnectTimeout=5 -o BatchMode=yes "$IP" true 2>/dev/null; then
    no "ssh" "unreachable or key not accepted"; return
  fi
  H=$(ssh -o ConnectTimeout=5 "$IP" hostname 2>/dev/null)
  [ "$H" = "$NAME" ] && ok "ssh identity" "$H" \
                     || no "ssh identity" "expected $NAME, got '$H' - wrong host!"

  M=$(get "http://$IP:9100/metrics")
  if [ -z "$M" ]; then no "node_exporter" "9100 unreachable"; else
    ok "node_exporter" "9100"

    MA=$(echo "$M" | awk '/^node_memory_MemAvailable_bytes/{printf "%.0f", $2/1073741824}')
    [ -n "$MA" ] && ok "MemAvailable" "${MA} GiB (UMA - this predicts OOM)" \
                 || no "MemAvailable" "absent"

    # Distinguish "metric missing" from "counter is zero". The first means the
    # collector is not running; the second means no RoCE traffic since boot,
    # which is normal between runs and says nothing about the collector.
    RAILS=$(echo "$M" | grep -c '^node_infiniband_port_data_received_bytes_total')
    if [ "$RAILS" -eq 0 ]; then
      no "infiniband collector" "metric absent - RoCE traffic invisible"
    else
      ok "infiniband collector" "$RAILS ports"
      RX=$(echo "$M" | awk '/^node_infiniband_port_data_received_bytes_total/{s+=$2} END{printf "%.0f", s}')
      if [ "${RX:-0}" -eq 0 ] 2>/dev/null; then
        meh "infiniband traffic" "0 bytes since boot - counters uncalibrated (runbook phase 5)"
      else
        ok "infiniband traffic" "$(echo "$RX" | awk '{printf "%.1f GiB rx", $1/1073741824}')"
      fi
    fi

    if echo "$M" | grep -q '^node_textfile_scrape_error'; then
      E=$(echo "$M" | awk '/^node_textfile_scrape_error/{print $2}')
      [ "$E" = "0" ] && ok "textfile collector" "run registry can publish" \
                     || no "textfile collector" "scrape_error=$E"
    else
      no "textfile collector" "not enabled - runinfo.sh cannot reach prometheus"
    fi

    if echo "$M" | grep -q '^bench_run_info'; then
      ok "run registry" "$(echo "$M" | grep -m1 '^bench_run_info' | cut -c1-70)"
    else
      meh "run registry" "no bench_run_info - expected between runs"
    fi
  fi

  D=$(get "http://$IP:9400/metrics")
  if [ -z "$D" ]; then no "dcgm-exporter" "9400 unreachable (stopped? check docker ps -a)"; else
    NF=$(echo "$D" | grep -oE '^DCGM_[A-Z0-9_]+' | sort -u | wc -l | tr -d ' ')
    echo "$D" | grep -q '^DCGM_FI_DEV_SM_CLOCK' \
      && ok "dcgm SM_CLOCK" "$NF fields (SM_CLOCK catches the 513 MHz clamp)" \
      || no "dcgm SM_CLOCK" "absent"
    echo "$D" | grep -q '^DCGM_FI_PROF_' \
      && ok "dcgm profiling fields" "PIPE_TENSOR_ACTIVE / DRAM_ACTIVE available" \
      || meh "dcgm profiling fields" "absent - prefill efficiency stays inferred"
    [ "$NF" -le 4 ] && meh "dcgm counter set" "$NF fields - expanded csv not applied"
  fi

  C=$(get "http://$IP:8080/metrics")
  if [ -z "$C" ]; then no "cadvisor" "8080 unreachable"; else
    PC=$(echo "$C" | grep -c '^container_cpu_usage_seconds_total{.*name=')
    [ "$PC" -gt 0 ] && ok "cadvisor per-container" "$PC series" \
                    || no "cadvisor per-container" "root cgroup only - docker api version floor"
  fi

  V=$(get "http://$IP:8011/metrics")
  if [ -z "$V" ]; then meh "vllm /metrics" "not serving (expected between runs)"; else
    echo "$V" | grep -q 'vllm:prefix_cache_queries' \
      && ok "vllm cache metrics" || meh "vllm cache metrics" "serving but no prefix cache counters"
  fi

  # chrony or systemd-timesyncd, whichever this host runs
  TS=$(ssh -o ConnectTimeout=5 "$IP" '
    if command -v chronyc >/dev/null 2>&1 && chronyc tracking >/dev/null 2>&1; then
      chronyc tracking | awk "/System time/{print \$4, \$5, \$6}"
    else
      timedatectl show -p NTPSynchronized -p TimeUSec --value 2>/dev/null | paste -sd" " -
    fi' 2>/dev/null)
  case "$TS" in
    yes*|*seconds*) ok "time sync" "$TS" ;;
    no*)            no "time sync" "NTP not synchronised - cross-node latency unattributable" ;;
    *)              meh "time sync" "no NTP daemon reporting" ;;
  esac

  G=$(ssh -o ConnectTimeout=5 "$IP" 'grep -l "grep -v .\^dcgm" ~/*.sh 2>/dev/null | wc -l' 2>/dev/null | tr -d ' ')
  if [ "${G:-0}" -gt 0 ]; then
    no "cleanup guard" "$G script(s) will delete obs-* - run protect-exporters.sh --apply"
  else
    ok "cleanup guard" "no script deletes obs-*"
  fi
}

check_node "$N1" "$N1_IP"
check_node "$N2" "$N2_IP"

hdr "summary"
printf '  %d pass, %d fail, %d warn\n\n' "$pass" "$fail" "$warn"
[ "$fail" -eq 0 ]
