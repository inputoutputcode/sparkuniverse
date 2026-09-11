#!/usr/bin/env bash
# Measure Super LMCache store-path pressure:
#   A. GPU/device KV -> LMCache L1
#   B. LMCache L1 -> fs_native disk L2
#
# This is not an AIPerf replacement. It sends synthetic cold prompts so the
# LMCache store path is easy to see in Prometheus counters before committing to
# a full trace replay.
set -uo pipefail

ENDPOINT=${ENDPOINT:-http://localhost:8000}
HEAD_NODE=${HEAD_NODE:-10.0.0.11}
NODES=${NODES:-"10.0.0.11 10.0.0.12"}
NS=${NS:-dynamo-system}
PROMPT_WORDS=${PROMPT_WORDS:-3000}
REQUESTS=${REQUESTS:-10}
CONCURRENCY=${CONCURRENCY:-5}
GEN=${GEN:-1}
CURL_MAX=${CURL_MAX:-1800}
SETTLE=${SETTLE:-30}
CLEAR_L1=${CLEAR_L1:-1}
OUT=${OUT:-/tmp/super-lmcache-store-path-bench}
KC='export KUBECONFIG=/etc/rancher/k3s/k3s.yaml'
mkdir -p "$OUT"

sshq() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$@"; }

build_prompt() {
  local words=$1 nonce=$2 path=$3
  python3 - "$words" "$GEN" "$nonce" > "$path" <<'PY'
import json, sys
n, gen, nonce = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
words = [f"storepath{i:06d}" for i in range(n)]
print(json.dumps({
    "model": "nemotron",
    "prompt": f"Super LMCache store path bench {nonce}. Catalogue: " + " ".join(words) + "\nSummarise.",
    "max_tokens": gen,
    "temperature": 0.0,
}))
PY
}

send_one() {
  local i=$1 payload="$OUT/prompt-$i.json" tag="req-$i" t0 t1 code
  t0=$(python3 -c 'import time; print(time.time())')
  code=$(curl -s -m "$CURL_MAX" -X POST "$ENDPOINT/v1/completions" \
    -H 'Content-Type: application/json' -d @"$payload" \
    -o "$OUT/resp-$tag.json" -w '%{http_code}')
  t1=$(python3 -c 'import time; print(time.time())')
  python3 - "$OUT/resp-$tag.json" "$tag" "$code" "$t0" "$t1" "$OUT/timings.tsv" <<'PY'
import json, sys
path, tag, code, t0, t1, timings = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
wall = t1 - t0
prompt = completion = "?"
try:
    d = json.load(open(path))
    u = d.get("usage") or {}
    prompt = u.get("prompt_tokens", "?")
    completion = u.get("completion_tokens", "?")
except Exception:
    pass
with open(timings, "a") as f:
    f.write(f"{tag}\t{code}\t{wall:.6f}\t{prompt}\t{completion}\n")
print(f"  {tag}: http {code}, wall {wall:.2f}s, prompt {prompt}, completion {completion}")
PY
  [ "$code" = 200 ]
}

run_burst() {
  local active=0 rc=0
  for i in $(seq 1 "$REQUESTS"); do
    send_one "$i" &
    active=$((active + 1))
    if [ "$active" -ge "$CONCURRENCY" ]; then
      wait -n 2>/dev/null || {
        # macOS bash 3.2 has no wait -n. Fall back to batch waits.
        wait || rc=1
        active=0
        continue
      }
      active=$((active - 1))
    fi
  done
  wait || rc=1
  return "$rc"
}

snap() {
  local tag=$1
  for ip in $NODES; do
    sshq "$ip" "curl -s -m 5 http://127.0.0.1:9500/metrics" 2>/dev/null \
      | grep '^lmcache_' > "$OUT/$tag-$ip.metrics"
    sshq "$ip" "python3 - <<'PY'
import os
root = '/mnt/lmcache-l2'
count = 0
size = 0
for base, _, files in os.walk(root):
    for f in files:
        if 'NVIDIA-Nemotron-3-Super' not in f:
            continue
        count += 1
        try:
            size += os.path.getsize(os.path.join(base, f))
        except OSError:
            pass
print(f'{count} {size}')
PY" > "$OUT/$tag-$ip.superdisk"
  done
}

clear_l1() {
  echo "=== clear LMCache L1"
  pods=$(sshq "$HEAD_NODE" "$KC; kubectl -n $NS get pods -o name | grep -i vllmdecodeworker" 2>/dev/null)
  [ -n "$pods" ] || { echo "  !! no decodeworker pods found"; return 1; }
  while read -r pod; do
    [ -n "$pod" ] || continue
    printf "  %s: " "$pod"
    sshq "$HEAD_NODE" "$KC; kubectl -n $NS exec ${pod#pod/} -- lmcache kvcache clear --url http://127.0.0.1:9500" >/dev/null \
      && echo ok || echo FAILED
  done <<< "$pods"
}

summarize() {
  python3 - "$OUT" before after "$OUT/timings.tsv" "$SETTLE" $NODES <<'PY'
import math, statistics, sys
from pathlib import Path

out = Path(sys.argv[1])
before_tag, after_tag, timings_path, settle = sys.argv[2], sys.argv[3], Path(sys.argv[4]), float(sys.argv[5])
nodes = sys.argv[6:]

def read_metrics(tag, ip):
    vals = {}
    p = out / f"{tag}-{ip}.metrics"
    if not p.exists():
        return vals
    for line in p.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        vals[parts[0]] = float(parts[1])
    return vals

def sum_prefix(vals, prefix):
    return sum(v for k, v in vals.items() if k.startswith(prefix))

def disk(tag, ip):
    try:
        return [int(x) for x in (out / f"{tag}-{ip}.superdisk").read_text().split()]
    except Exception:
        return [0, 0]

rows = []
if timings_path.exists():
    for line in timings_path.read_text().splitlines():
        tag, code, wall, prompt, completion = line.split("\t")
        rows.append((tag, code, float(wall), prompt, completion))

walls = [r[2] for r in rows if r[1] == "200"]
prompt_tokens = []
for r in rows:
    try:
        prompt_tokens.append(int(r[3]))
    except ValueError:
        pass

if rows:
    print("=== request timing")
    print(f"  completed 200s: {len(walls)} / {len(rows)}")
    if walls:
        print(f"  wall p50/max:   {statistics.median(walls):.2f}s / {max(walls):.2f}s")
    if prompt_tokens:
        print(f"  prompt tokens:  total {sum(prompt_tokens):,}, avg {sum(prompt_tokens)/len(prompt_tokens):,.0f}")

print("=== LMCache store path")
for ip in nodes:
    b = read_metrics(before_tag, ip)
    a = read_metrics(after_tag, ip)
    d0 = disk(before_tag, ip)
    d1 = disk(after_tag, ip)

    l0_sum = sum_prefix(a, "lmcache_mp_l0_l1_store_throughput_GB_per_second_sum") - sum_prefix(b, "lmcache_mp_l0_l1_store_throughput_GB_per_second_sum")
    l0_count = sum_prefix(a, "lmcache_mp_l0_l1_store_throughput_GB_per_second_count") - sum_prefix(b, "lmcache_mp_l0_l1_store_throughput_GB_per_second_count")
    l2_sum = sum_prefix(a, "lmcache_mp_l2_store_throughput_GB_per_second_sum") - sum_prefix(b, "lmcache_mp_l2_store_throughput_GB_per_second_sum")
    l2_count = sum_prefix(a, "lmcache_mp_l2_store_throughput_GB_per_second_count") - sum_prefix(b, "lmcache_mp_l2_store_throughput_GB_per_second_count")

    l1_chunks = sum_prefix(a, "lmcache_mp_l1_write_chunks_total") - sum_prefix(b, "lmcache_mp_l1_write_chunks_total")
    l2_chunks = sum_prefix(a, "lmcache_mp_l2_store_completed_objects_chunks_total") - sum_prefix(b, "lmcache_mp_l2_store_completed_objects_chunks_total")
    evicted = sum_prefix(a, "lmcache_mp_l1_evicted_chunks_total") - sum_prefix(b, "lmcache_mp_l1_evicted_chunks_total")
    l1_ratio = sum_prefix(a, "lmcache_mp_l1_usage_ratio")
    l1_bytes = sum_prefix(a, "lmcache_mp_l1_memory_usage_bytes")
    l2_usage = sum_prefix(a, "lmcache_mp_l2_usage_bytes") - sum_prefix(b, "lmcache_mp_l2_usage_bytes")
    disk_bytes = d1[1] - d0[1]
    disk_files = d1[0] - d0[0]

    l0_avg = l0_sum / l0_count if l0_count > 0 else 0.0
    l2_avg = l2_sum / l2_count if l2_count > 0 else 0.0
    disk_gbs = (disk_bytes / 1e9 / settle) if settle > 0 else 0.0

    print(f"--- {ip}")
    print(f"  GPU/device -> L1 avg: {l0_avg:.2f} GB/s over {l0_count:.0f} chunks")
    print(f"  L1 -> disk L2 avg:    {l2_avg:.2f} GB/s over {l2_count:.0f} chunks")
    print(f"  L1 write chunks:      {l1_chunks:.0f}")
    print(f"  L2 stored chunks:     {l2_chunks:.0f}")
    print(f"  L1 evicted chunks:    {evicted:.0f}")
    print(f"  L1 usage now:         {l1_bytes/1e9:.2f} GB, ratio-sum {l1_ratio:.3f}")
    print(f"  L2 usage delta:       {l2_usage/1e9:.2f} GB")
    print(f"  disk growth:          {disk_files:+.0f} files, {disk_bytes/1e9:.2f} GB")
    print(f"  disk growth / settle: {disk_gbs:.2f} GB/s over {settle:.0f}s")
    if l0_count and l2_count:
        if l2_avg < l0_avg * 0.5:
            print("  likely throttle:      L1 -> disk L2 is much slower than GPU -> L1")
        elif evicted > 0 and l2_chunks < l1_chunks:
            print("  likely throttle:      L1 eviction pressure before L2 catches up")
        else:
            print("  likely throttle:      not obvious from LMCache store histograms")
PY
}

rm -f "$OUT"/prompt-*.json "$OUT"/resp-*.json "$OUT"/*.metrics "$OUT"/*.superdisk "$OUT"/timings.tsv

echo "=== endpoint"
curl -sf -m 10 "$ENDPOINT/v1/models" -o "$OUT/models.json" || {
  echo "!! endpoint not serving at $ENDPOINT"
  exit 1
}
python3 - "$OUT/models.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("  models:", ", ".join(m.get("id", "?") for m in d.get("data", [])))
PY

echo "=== worker pods"
sshq "$HEAD_NODE" "$KC; kubectl -n $NS get pods -o wide | grep -E 'super-agg-p2p|lmcache-coordinator' || true"

echo "=== test shape"
echo "  prompt words: $PROMPT_WORDS"
echo "  requests:     $REQUESTS"
echo "  concurrency:  $CONCURRENCY"
echo "  settle:       $SETTLE seconds"
echo "  clear L1:     $CLEAR_L1"

if [ "$CLEAR_L1" = 1 ]; then
  clear_l1
fi

echo "=== build prompts"
for i in $(seq 1 "$REQUESTS"); do
  build_prompt "$PROMPT_WORDS" "req-$i-$(date +%s)-$RANDOM" "$OUT/prompt-$i.json"
done
echo "  first payload bytes: $(wc -c < "$OUT/prompt-1.json")"

snap before
start=$(python3 -c 'import time; print(time.time())')
echo "=== send burst"
run_burst || {
  echo "!! one or more requests failed; raw responses in $OUT"
}
end=$(python3 -c 'import time; print(time.time())')
python3 - "$start" "$end" <<'PY'
import sys
print(f"  burst wall: {float(sys.argv[2]) - float(sys.argv[1]):.2f}s")
PY

echo "=== settle for async L2 stores"
sleep "$SETTLE"
snap after
summarize

echo
echo "Interpretation:"
echo "  If GPU/device -> L1 is high but L1 -> disk L2 is much lower, disk/offload is the throttle."
echo "  If both are high but L1 evictions climb and L2 stored chunks lag L1 writes, L1 is too small for the burst."
echo "  If both rates are similar and disk growth matches L2 usage, throttling is probably elsewhere in vLLM prefill/decode."
echo "  Raw files are in $OUT"
