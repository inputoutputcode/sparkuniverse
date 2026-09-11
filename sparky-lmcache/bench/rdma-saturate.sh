#!/usr/bin/env bash
# Sustained RDMA bandwidth test for the two Spark ConnectX-7 rails.
#
# This is deliberately below LMCache/NIXL. It answers "what can the fabric do
# for several seconds?" before asking why an application-level KV path is lower.
set -Eeuo pipefail

A=${A:-10.0.0.11}
B=${B:-10.0.0.12}
DURATION=${DURATION:-20}
SIZE=${SIZE:-8388608}
QPS=${QPS:-8}
TX_DEPTH=${TX_DEPTH:-512}
MODE=${MODE:-dual} # single-a, single-b, dual
BASE_PORT=${BASE_PORT:-18515}
SSH_OPTS=${SSH_OPTS:-"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8"}

RAILS_A_NAMES=(rocep1s0f1 roceP2p1s0f1)
RAILS_B_IPS=(10.1.0.12 10.2.0.12)
RAILS_LABELS=(rail-a rail-b)

sshx() {
  local host=$1
  shift
  ssh $SSH_OPTS "$host" "$@"
}

rails_for_mode() {
  case "$MODE" in
    single-a) echo 0 ;;
    single-b) echo 1 ;;
    dual) echo 0 1 ;;
    *) echo "MODE must be single-a, single-b, or dual" >&2; exit 2 ;;
  esac
}

counter_file() {
  local host=$1 out=$2
  : > "$out"
  for dev in "${RAILS_A_NAMES[@]}"; do
    sshx "$host" "cat /sys/class/infiniband/$dev/ports/1/counters/port_xmit_data 2>/dev/null || echo 0" \
      | awk -v d="$dev" '{print d, $1}' >> "$out"
  done
}

cleanup() {
  for idx in $(rails_for_mode); do
    port=$((BASE_PORT + idx))
    sshx "$B" "pkill -f 'ib_write_bw.*-p $port' >/dev/null 2>&1 || true" || true
  done
}
trap cleanup EXIT

echo "=== config"
echo "  mode:      $MODE"
echo "  duration:  ${DURATION}s"
echo "  size:      $SIZE bytes"
echo "  qps:       $QPS"
echo "  tx-depth:  $TX_DEPTH"

echo
echo "=== tools"
for h in "$A" "$B"; do
  echo "--- $h"
  sshx "$h" 'hostname; command -v ib_write_bw; ibdev2netdev | grep -E "rocep1s0f1|roceP2p1s0f1"'
done

echo
echo "=== start servers on $B"
cleanup
for idx in $(rails_for_mode); do
  dev=${RAILS_A_NAMES[$idx]}
  port=$((BASE_PORT + idx))
  label=${RAILS_LABELS[$idx]}
  echo "  $label server: dev=$dev port=$port"
  sshx "$B" "nohup ib_write_bw -d $dev -i 1 -R -D $DURATION -s $SIZE -q $QPS -t $TX_DEPTH -p $port --report_gbits --report-counters=counters/port_rcv_data > /tmp/ib-write-bw-$label.server.log 2>&1 &"
done
sleep 3

before=$(mktemp /tmp/rdma-before.XXXXXX)
after=$(mktemp /tmp/rdma-after.XXXXXX)
counter_file "$A" "$before"

echo
echo "=== clients from $A"
pids=()
outs=()
for idx in $(rails_for_mode); do
  dev=${RAILS_A_NAMES[$idx]}
  peer=${RAILS_B_IPS[$idx]}
  port=$((BASE_PORT + idx))
  label=${RAILS_LABELS[$idx]}
  out="/tmp/ib-write-bw-$label.client.log"
  outs+=("$out")
  echo "  $label client: dev=$dev peer=$peer port=$port"
  sshx "$A" "ib_write_bw $peer -d $dev -i 1 -R -D $DURATION -s $SIZE -q $QPS -t $TX_DEPTH -p $port --report_gbits --report-counters=counters/port_xmit_data > $out 2>&1" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

counter_file "$A" "$after"

echo
echo "=== client reports"
for out in "${outs[@]}"; do
  echo "--- $A:$out"
  sshx "$A" "cat $out" | tail -35
done

echo
echo "=== A transmit counter deltas"
awk '
  NR==FNR {b[$1]=$2; next}
  {
    delta=$2 - b[$1]
    total += delta
    gbps = delta * 4 * 8 / 1000000000 / dur
    printf "  %-14s %14d words  %9.2f GB  %7.2f Gb/s\n", $1, delta, delta * 4 / 1000000000, gbps
  }
  END {
    gbps = total * 4 * 8 / 1000000000 / dur
    printf "  %-14s %14d words  %9.2f GB  %7.2f Gb/s\n", "total", total, total * 4 / 1000000000, gbps
  }
' dur="$DURATION" "$before" "$after"

exit "$status"
