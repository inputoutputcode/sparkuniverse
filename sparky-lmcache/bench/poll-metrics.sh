#!/usr/bin/env bash
# Run-scoped counter deltas, collected over ssh instead of Prometheus.
#
# Prometheus scrapes from a Docker container on the Mac, and on 2026-08-17 that
# container lost LAN egress while the Mac itself kept it. Every target read
# `connection refused` from inside the container and 200 from the host. Rather
# than block a three hour arm on that, poll the same endpoints directly.
#
# This also fixes a mistake Prometheus made easy. vllm:prefix_cache_hits_total
# and friends are cumulative over the worker's lifetime, and comparing a
# lifetime total against one run's prompt tokens produced a 49.7M against 44.2M
# nonsense. Here the first sample is the baseline and everything reported is a
# delta, so a run can only be compared against itself.
#
#   ./poll-metrics.sh weka-c-p2p 60 &
#   ... run the arm ...
#   kill %1   (or it stops at MAX_HOURS)
#
# ## Three things this got wrong before 2026-08-25
#
# **Labels were discarded.** The old awk did `gsub(/\{.*\}/, "", $1)`, so
# `lmcache_mp_l2_adapters{state="active"}` and `{state="draining"}` became the
# same key with two different values, and the same for every histogram bucket
# and for vllm:num_requests_waiting_by_reason. Those series are unrecoverable in
# every poll.csv written before this date -- backfill-poll.py detects and drops
# them rather than guessing. The full series identifier is now kept, quoted, so
# a CSV field containing commas survives.
#
# **The file was recreated headerless.** The header was written once with `>` at
# startup. `rm -rf runs/<id>` during a debugging session, while this was still
# running, deleted the file and the next `>>` recreated it with no header. That
# is exactly how weka-a-router/poll.csv ended up unreadable. The header is now
# rewritten whenever the file is missing.
#
# **It outlived its run by four days.** Nothing stops this when the arm ends, so
# weka-a-router/poll.csv covers 105 hours beginning two days after that arm
# finished. The filename now carries a start epoch so a relaunch cannot append
# to an older run's file, and MAX_HOURS bounds it.
set -uo pipefail

RUN_ID=${1:?run id, matching the RUN_ID passed to run-scenario.sh}
INTERVAL=${2:-60}
NODES=${NODES:-"10.0.0.11 10.0.0.12"}
MAX_HOURS=${MAX_HOURS:-16}
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$REPO/runs/$RUN_ID
mkdir -p "$OUT"

START=$(date -u +%s)
# Distinct per launch. Two pollers on one run id, or a relaunch after a crash,
# now write separate files instead of interleaving into one that no longer
# corresponds to any single window.
CSV=$OUT/poll-$START.csv
LATEST=$OUT/poll.csv
HEADER="ts,node,series,value"

# Named explicitly rather than scraped wholesale. A wildcard here would produce
# a file whose columns change between runs, which is useless for comparison.
# vllm:prefix_cache_* is the device cache and vllm:external_prefix_cache_* is
# the LMCache connector, reported separately by the same endpoint. That split is
# what arms A, B and C exist to measure, so it is worth more than anything else
# collected here.
#
# vllm:kv_cache_usage_perc is the arena. Zero preemptions proves it never
# overflowed, not how close it came, and 23 concurrent sequences at p90 length
# would consume 8.4 GB of 10 GB while leaving the run looking healthy.
WANT='vllm:num_requests_running|vllm:num_requests_waiting|vllm:num_preemptions_total|vllm:kv_cache_usage_perc|vllm:prefix_cache_queries_total|vllm:prefix_cache_hits_total|vllm:external_prefix_cache_queries_total|vllm:external_prefix_cache_hits_total|vllm:prompt_tokens_cached_total|lmcache_mp_l1_usage_ratio|lmcache_mp_p2p|lmcache_mp_l2|lmcache_mp_retrieve|lmcache_mp_store|lmcache_mp_lookup'

sample() {
  local ip=$1
  # -n is required, not tidiness. ssh reads stdin by default, so this script
  # backgrounded with & gets SIGTTIN on the first sample and suspends:
  #   [1] + suspended (tty input)  ./poll-metrics.sh
  # BatchMode stops it blocking on a host-key or password prompt for the same
  # reason.
  ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$ip" "
    curl -s --max-time 5 localhost:9090/metrics 2>/dev/null
    curl -s --max-time 5 localhost:9500/metrics 2>/dev/null
    echo rdma_port_xmit_data \$(cat /sys/class/infiniband/rocep1s0f1/ports/1/counters/port_xmit_data 2>/dev/null || echo 0)
    echo mem_available_kb \$(awk '/MemAvailable/{print \$2}' /proc/meminfo)
  " 2>/dev/null | awk -v ip="$ip" -v want="$WANT" '
    /^#/ { next }
    NF < 2 { next }
    {
      # The value is the last field. The series is everything before it, which
      # keeps the label set intact even when a label value contains a space --
      # splitting on $1 would truncate `foo{a="b c"} 1` to `foo{a="b`.
      val = $NF
      series = $0
      sub(/[ \t]+[^ \t]+[ \t]*$/, "", series)
      name = series
      sub(/\{.*/, "", name)
      if (name !~ want && name != "rdma_port_xmit_data" && name != "mem_available_kb") next
      # RFC4180: a field containing commas or quotes is wrapped in quotes with
      # internal quotes doubled. Label sets contain both.
      gsub(/"/, "\"\"", series)
      print ip "\t\"" series "\"\t" val
    }'
}

echo "polling $RUN_ID every ${INTERVAL}s for up to ${MAX_HOURS}h -> $CSV"
echo "$HEADER" > "$CSV"
# A stable name for tooling, pointing at this launch. Replaced, never appended.
ln -sf "$(basename "$CSV")" "$LATEST" 2>/dev/null || cp "$CSV" "$LATEST"

finish() {
  local n=0
  [ -f "$CSV" ] && n=$(($(wc -l < "$CSV") - 1))
  echo
  echo "poll stopped, $n samples in $CSV"
  exit 0
}
trap finish INT TERM

DEADLINE=$((START + MAX_HOURS * 3600))
while true; do
  now=$(date -u +%s)
  if [ "$now" -ge "$DEADLINE" ]; then
    echo "reached MAX_HOURS=${MAX_HOURS}, stopping rather than outliving the run"
    finish
  fi
  # The run directory has been deleted underneath this loop before, during a
  # debugging cycle, and the next append silently recreated a headerless file.
  if [ ! -f "$CSV" ]; then
    mkdir -p "$OUT"
    echo "$HEADER" > "$CSV"
    echo "  ! $CSV vanished and was recreated with a header"
  fi
  for ip in $NODES; do
    sample "$ip" | while IFS=$'\t' read -r node series value; do
      echo "$now,$node,$series,$value" >> "$CSV"
    done
  done
  sleep "$INTERVAL"
done
