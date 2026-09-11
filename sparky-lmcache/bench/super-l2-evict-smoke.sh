#!/usr/bin/env bash
# Validate Super + LMCache local disk L2 on the hot miss path.
#
# Shape:
#   1. Send one reusable anchor prompt. This should store Super KV to fs_native L2.
#   2. Clear LMCache L1 on all workers.
#   3. Send unique filler prompts to displace vLLM/device prefix-cache blocks.
#   4. Clear LMCache L1 again.
#   5. Replay the anchor prompt. Any external reuse now has to come from L2.
#
# For a fast canary, deploy with a small vLLM KV arena first:
#   KV_CACHE_MEMORY_BYTES=1000000000 MAX_NUM_SEQS=1 L1_SIZE_GB=2 ./bench/deploy-super-l2-local.sh
#   ENDPOINT=http://localhost:8000 PROMPT_WORDS=3000 FILLER_COUNT=12 ./bench/super-l2-evict-smoke.sh
#
# Against the current 16 GB arena, expect a much longer run:
#   ENDPOINT=http://localhost:8000 PROMPT_WORDS=6000 FILLER_COUNT=94 ./bench/super-l2-evict-smoke.sh
set -uo pipefail

ENDPOINT=${ENDPOINT:-http://localhost:8000}
HEAD_NODE=${HEAD_NODE:-10.0.0.11}
NODES=${NODES:-"10.0.0.11 10.0.0.12"}
NS=${NS:-dynamo-system}
PROMPT_WORDS=${PROMPT_WORDS:-3000}
FILLER_WORDS=${FILLER_WORDS:-$PROMPT_WORDS}
FILLER_COUNT=${FILLER_COUNT:-12}
GEN=${GEN:-1}
CURL_MAX=${CURL_MAX:-1800}
OUT=${OUT:-/tmp/super-l2-evict-smoke}
KC='export KUBECONFIG=/etc/rancher/k3s/k3s.yaml'
mkdir -p "$OUT"

sshq() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$@"; }

build_prompt() {
  local words=$1 nonce=$2
  python3 - "$words" "$GEN" "$nonce" <<'PY'
import json, sys
n, gen, nonce = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
words = [f"superl2evict{i:06d}" for i in range(n)]
print(json.dumps({
    "model": "nemotron",
    "prompt": f"Super disk L2 eviction smoke {nonce}. Catalogue: " + " ".join(words) + "\nSummarise.",
    "max_tokens": gen,
    "temperature": 0.0,
}))
PY
}

send_payload() {
  local tag=$1 payload=$2 t0 t1 code
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
    print(f"  {tag}: http {code}, wall {wall:.2f}s, prompt {prompt}, completion {completion}")
except Exception as e:
    print(f"  {tag}: http {code}, wall {wall:.2f}s, no JSON usage: {e}")
with open(timings, "a") as f:
    f.write(f"{tag}\t{code}\t{wall:.6f}\t{prompt}\t{completion}\n")
PY
  [ "$code" = 200 ]
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

delta_metrics() {
  local a=$1 b=$2
  for ip in $NODES; do
    echo "--- $ip metrics"
    join -j1 -o 0,1.2,2.2 \
      <(awk '{print $1, $2}' "$OUT/$a-$ip.metrics" | sort) \
      <(awk '{print $1, $2}' "$OUT/$b-$ip.metrics" | sort) 2>/dev/null \
      | awk '
        /lmcache_mp_(lookup|l0_l1|l1_|l2_|store|retrieve|load|save|prefetch)/ {
          d=$3-$2
          if (d != 0) printf "  %-84s %+g\n", $1, d
        }' | head -160
    echo "--- $ip Super disk"
    before=$(cat "$OUT/$a-$ip.superdisk" 2>/dev/null || echo "0 0")
    after=$(cat "$OUT/$b-$ip.superdisk" 2>/dev/null || echo "0 0")
    python3 - "$before" "$after" <<'PY'
import sys
b = [int(x) for x in sys.argv[1].split()]
a = [int(x) for x in sys.argv[2].split()]
print(f"  files {b[0]} -> {a[0]} ({a[0]-b[0]:+d})")
print(f"  bytes {b[1]} -> {a[1]} ({a[1]-b[1]:+d})")
PY
  done
}

sum_metric_delta() {
  local a=$1 b=$2 pattern=$3
  python3 - "$OUT" "$a" "$b" "$pattern" $NODES <<'PY'
import re, sys
out, a, b, pattern, *nodes = sys.argv[1:]
pat = re.compile(pattern)
def read(tag, ip):
    vals = {}
    try:
        for line in open(f"{out}/{tag}-{ip}.metrics"):
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2 or not pat.search(parts[0]):
                continue
            vals[parts[0]] = float(parts[1])
    except FileNotFoundError:
        pass
    return vals
total = 0.0
for ip in nodes:
    before = read(a, ip)
    after = read(b, ip)
    for key, value in after.items():
        total += value - before.get(key, 0.0)
print(total)
PY
}

disk_delta_report() {
  local a=$1 b=$2
  python3 - "$OUT" "$a" "$b" $NODES <<'PY'
import sys
out, a, b, *nodes = sys.argv[1:]
rows = []
for ip in nodes:
    def read(tag):
        try:
            return [int(x) for x in open(f"{out}/{tag}-{ip}.superdisk").read().split()]
        except Exception:
            return [0, 0]
    before = read(a)
    after = read(b)
    rows.append((ip, after[0] - before[0], after[1] - before[1]))
for ip, files, size in rows:
    if files or size:
        print(f"{ip}:{files}:{size}")
PY
}

clear_l1() {
  echo "  clearing LMCache L1 in worker pods"
  pods=$(sshq "$HEAD_NODE" "$KC; kubectl -n $NS get pods -o name | grep -i vllmdecodeworker" 2>/dev/null)
  [ -n "$pods" ] || { echo "  !! no decodeworker pods found"; return 1; }
  while read -r pod; do
    [ -n "$pod" ] || continue
    printf "    %s: " "$pod"
    sshq "$HEAD_NODE" "$KC; kubectl -n $NS exec ${pod#pod/} -- lmcache kvcache clear --url http://127.0.0.1:9500" >/dev/null \
      && echo ok || echo FAILED
  done <<< "$pods"
}

show_context() {
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
  echo "  filler words: $FILLER_WORDS"
  echo "  filler count: $FILLER_COUNT"
  echo "  generation tokens: $GEN"
  echo "  routing note: if replay stores new disk bytes on a different node than anchor-cold,"
  echo "                rerun with FILLER_COUNT parity flipped or use a one-worker deployment"
}

show_recent_errors() {
  echo
  echo "=== recent LMCache/vLLM lines"
  sshq "$HEAD_NODE" "$KC; kubectl -n $NS logs -l nvidia.com/dynamo-component=VllmDecodeWorker --tail=1000 2>/dev/null \
    | grep -iE 'No GPU context|Traceback|ValueError|RuntimeError|OOM|out of memory|preempt|Running:|Waiting:|GPU KV cache usage|Prefix cache hit rate|External prefix cache hit rate' \
    | tail -100" || true
}

rm -f "$OUT"/resp-*.json "$OUT"/*.metrics "$OUT"/*.superdisk "$OUT"/timings.tsv
show_context

anchor_nonce="anchor-$(date +%s)-$RANDOM"
build_prompt "$PROMPT_WORDS" "$anchor_nonce" > "$OUT/anchor.json"
echo "  anchor bytes: $(wc -c < "$OUT/anchor.json")"

echo
echo "=== stage 1: cold anchor store"
snap s0
send_payload anchor-cold "$OUT/anchor.json" || {
  echo "!! cold anchor failed; response in $OUT/resp-anchor-cold.json"
  show_recent_errors
  exit 1
}
sleep 10
snap s1
delta_metrics s0 s1

echo
echo "=== stage 2: clear L1, then send fillers"
clear_l1
sleep 5
snap s2
for i in $(seq 1 "$FILLER_COUNT"); do
  payload="$OUT/filler-$i.json"
  build_prompt "$FILLER_WORDS" "filler-$i-$(date +%s)-$RANDOM" > "$payload"
  send_payload "filler-$i" "$payload" || {
    echo "!! filler $i failed; response in $OUT/resp-filler-$i.json"
    show_recent_errors
    exit 1
  }
done
sleep 10
snap s3
echo "--- filler deltas"
delta_metrics s2 s3

echo
echo "=== stage 3: clear L1, replay anchor"
clear_l1
sleep 5
snap s4
send_payload anchor-replay "$OUT/anchor.json" || {
  echo "!! anchor replay failed; response in $OUT/resp-anchor-replay.json"
  show_recent_errors
  exit 1
}
sleep 10
snap s5
delta_metrics s4 s5
show_recent_errors

echo
echo "=== summary"
python3 - "$OUT/timings.tsv" <<'PY'
import sys
rows = {}
for line in open(sys.argv[1]):
    tag, code, wall, prompt, completion = line.rstrip("\n").split("\t")
    rows[tag] = (code, float(wall), prompt, completion)
for tag in ("anchor-cold", "anchor-replay"):
    if tag in rows:
        code, wall, prompt, completion = rows[tag]
        print(f"  {tag}: http {code}, wall {wall:.2f}s, prompt {prompt}, completion {completion}")
if "anchor-cold" in rows and "anchor-replay" in rows:
    cold = rows["anchor-cold"][1]
    replay = rows["anchor-replay"][1]
    print(f"  replay delta: {replay-cold:+.2f}s ({(replay/cold if cold else 0):.2f}x cold)")
PY
lookup_hits=$(sum_metric_delta s4 s5 'lmcache_mp_lookup_hit_tokens_total')
lookup_reqs=$(sum_metric_delta s4 s5 'lmcache_mp_lookup_requested_tokens_total')
l2_read=$(sum_metric_delta s4 s5 'lmcache_mp_.*(retrieve|load).*')
l2_usage=$(sum_metric_delta s4 s5 'lmcache_mp_l2_usage_bytes')
anchor_disk=$(disk_delta_report s0 s1)
replay_disk=$(disk_delta_report s4 s5)
echo "  replay lookup requested token delta: $lookup_reqs"
echo "  replay lookup hit token delta:       $lookup_hits"
echo "  replay retrieve/load delta:          $l2_read"
echo "  replay l2 usage byte delta:          $l2_usage"
echo "  anchor disk growth nodes:            ${anchor_disk:-none}"
echo "  replay disk growth nodes:            ${replay_disk:-none}"
echo
echo "Interpretation:"
echo "  GO: after the final L1 clear, anchor-replay has lookup hit tokens or real L2 retrieve/load deltas,"
echo "      no fresh replay disk growth, and replay is materially faster than anchor-cold."
echo "  STORE-ONLY: replay has lookup hit tokens = 0 and grows disk/l2_usage again."
echo "  ROUTING-MISS: replay disk growth is on a different node than anchor-cold; flip FILLER_COUNT parity"
echo "      or use a one-worker deployment."
echo "  Raw files are in $OUT"
