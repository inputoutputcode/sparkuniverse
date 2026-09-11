#!/usr/bin/env bash
# Quick validation for Super + LMCache local disk L2.
#
# This is intentionally smaller than an AIPerf run. It answers:
#   1. Does a request reach the current Super deployment?
#   2. Does LMCache issue lookup/store work?
#   3. Does fs_native L2 usage or Super-specific disk content grow?
#   4. Do the logs show the GPU-context failure that invalidates the arm?
set -uo pipefail

ENDPOINT=${ENDPOINT:-http://localhost:8000}
HEAD_NODE=${HEAD_NODE:-10.0.0.11}
NODES=${NODES:-"10.0.0.11 10.0.0.12"}
NS=${NS:-dynamo-system}
PROMPT_WORDS=${PROMPT_WORDS:-6000}
GEN=${GEN:-1}
OUT=${OUT:-/tmp/super-l2-quick-smoke}
KC='export KUBECONFIG=/etc/rancher/k3s/k3s.yaml'
mkdir -p "$OUT"

sshq() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$@"; }

build_prompt() {
  python3 - "$PROMPT_WORDS" "$GEN" "$1" <<'PY'
import json, sys
n, gen, nonce = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
words = [f"superl2item{i:06d}" for i in range(n)]
print(json.dumps({
    "model": "nemotron",
    "prompt": f"Super local L2 smoke {nonce}. Catalogue: " + " ".join(words) + "\nSummarise.",
    "max_tokens": gen,
    "temperature": 0.0,
}))
PY
}

send() {
  local tag=$1 t0 t1 code
  t0=$(python3 -c 'import time; print(time.time())')
  code=$(curl -s -m 1800 -X POST "$ENDPOINT/v1/completions" \
    -H 'Content-Type: application/json' -d @"$OUT/prompt.json" \
    -o "$OUT/resp-$tag.json" -w '%{http_code}')
  t1=$(python3 -c 'import time; print(time.time())')
  python3 - "$OUT/resp-$tag.json" "$tag" "$code" "$t0" "$t1" <<'PY'
import json, sys
path, tag, code, t0, t1 = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
try:
    d=json.load(open(path))
    u=d.get("usage") or {}
    print(f"  {tag}: http {code}, wall {t1-t0:.2f}s, prompt {u.get('prompt_tokens','?')}, completion {u.get('completion_tokens','?')}")
except Exception as e:
    print(f"  {tag}: http {code}, wall {t1-t0:.2f}s, no JSON usage: {e}")
PY
  [ "$code" = 200 ]
}

snap() {
  local tag=$1
  for ip in $NODES; do
    sshq "$ip" "curl -s -m 5 http://127.0.0.1:9500/metrics" 2>/dev/null \
      | grep -E '^lmcache_mp_(lookup|l0_l1|l1_|l2_|store|retrieve)' \
      > "$OUT/$tag-$ip.metrics"
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
        {
          d=$3-$2
          if (d != 0) printf "  %-76s %+g\n", $1, d
        }' | head -120
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

show_recent_errors() {
  echo
  echo "=== recent LMCache/vLLM errors"
  sshq "$HEAD_NODE" "$KC; kubectl -n $NS logs -l nvidia.com/dynamo-component=VllmDecodeWorker --tail=800 2>/dev/null \
    | grep -iE 'No GPU context|Error in blocking handler|Traceback|ValueError|RuntimeError|OOM|out of memory|preempt|Running:|Waiting:|GPU KV cache usage' \
    | tail -80" || true
}

echo "=== endpoint"
curl -sf -m 10 "$ENDPOINT/v1/models" -o "$OUT/models.json" || {
  echo "!! endpoint not serving at $ENDPOINT"
  exit 1
}
python3 - "$OUT/models.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print("  models:", ", ".join(m.get("id", "?") for m in d.get("data", [])))
PY

echo "=== worker pods"
sshq "$HEAD_NODE" "$KC; kubectl -n $NS get pods -o wide | grep -E 'super-agg-p2p|lmcache-coordinator' || true"

build_prompt "target-$(date +%s)-$RANDOM" > "$OUT/prompt.json"
echo "  prompt bytes: $(wc -c < "$OUT/prompt.json")"

echo
echo "=== cold request"
snap t0
send cold || { echo "!! cold request failed; response in $OUT/resp-cold.json"; show_recent_errors; exit 1; }
sleep 10
snap t1
delta_metrics t0 t1

echo
echo "=== repeat request"
send repeat || { echo "!! repeat request failed; response in $OUT/resp-repeat.json"; show_recent_errors; exit 1; }
sleep 10
snap t2
delta_metrics t1 t2
show_recent_errors

echo
echo "=== interpretation"
echo "  GO: lmcache lookup/store counters move, fs_native l2_usage or Super disk bytes grow, and repeat is faster."
echo "  NO-GO: lookup/store stay zero, l2_usage stays zero, or logs show No GPU context errors."
echo "  Raw files are in $OUT"
