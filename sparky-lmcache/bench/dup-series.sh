#!/usr/bin/env bash
# Which series does an exporter emit twice, and do the copies agree?
#
#   ./dup-series.sh                       both Sparks, vLLM 9090 and LMCache 9500
#   ./dup-series.sh 10.0.0.11 9500     one endpoint
#
# Prometheus logs this against the lmcache job on every scrape, on both nodes:
#
#   Error on ingesting samples with different value but same timestamp
#     component="scrape manager" scrape_pool=lmcache num_dropped=2
#
# It means two exposition lines share a label set and disagree on the value.
# Prometheus keeps one and discards the other, arbitrarily, so the affected
# series has been unreliable in the dashboards for as long as this has been
# happening. Identical duplicates are accepted silently, so the warning proves
# the copies differ at least sometimes.
#
# This names the series. Two passes, because the distinction matters:
#
#   DIFFER    two lines, same labels, different values -- this is what
#             Prometheus drops and what the warning counts
#   SAME      two lines, same labels, same value -- harmless, but still a
#             duplicate registration and the same underlying bug
#
# `num_dropped=2` should match the DIFFER count. If it does not, the offender is
# on the other endpoint or appears only intermittently, and this wants running a
# few times.
set -uo pipefail

if [ $# -eq 2 ]; then
  TARGETS="$1:$2"
else
  TARGETS=""
  for ip in ${NODES:-10.0.0.11 10.0.0.12}; do
    TARGETS="$TARGETS $ip:9090 $ip:9500"
  done
fi

for t in $TARGETS; do
  ip=${t%:*}; port=${t#*:}
  echo "=== $ip:$port"
  body=$(ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$ip" \
           "curl -s --max-time 5 localhost:$port/metrics" 2>/dev/null)
  if [ -z "$body" ]; then
    echo "  no response"
    continue
  fi

  # Series identity is everything before the value, so a label value containing
  # a space does not truncate the key.
  echo "$body" | awk '
    /^#/ { next }
    NF < 2 { next }
    {
      val = $NF
      key = $0
      sub(/[ \t]+[^ \t]+[ \t]*$/, "", key)
      n[key]++
      if (n[key] > 1 && seen[key] != val) diff[key] = seen[key] " vs " val
      seen[key] = val
    }
    END {
      nd = ns = 0
      for (k in n) if (n[k] > 1) {
        if (k in diff) { printf "  DIFFER  %s   %s\n", k, diff[k]; nd++ }
        else           { printf "  SAME    %s   x%d\n", k, n[k]; ns++ }
      }
      printf "  --- %d differing, %d identical duplicates, %d series total\n",
             nd, ns, length(n)
    }' | sort
done

echo
echo "A DIFFER line is a series Prometheus is silently dropping samples of."
echo "There is no LMCache flag for it: the exporter is registering a collector"
echo "more than once, which is why the count is exactly two. Either report it"
echo "upstream and live with the log noise, or drop the metric in the lmcache"
echo "job with metric_relabel_configs, which trades the noise for the metric."
