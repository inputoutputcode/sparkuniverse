#!/usr/bin/env bash
# Run request-level hot-path probes against nano-hotpath-smoke.
#
# Stage A: no L1 clear between requests. The second request should hit peer L1
# over P2P because round_robin sends it to the other worker.
#
# Stage B: clear L1 after the first request. The second request can only avoid
# recompute if shared/remote L2 is configured and reachable from the other node.
set -uo pipefail

ENDPOINT=${ENDPOINT:-http://10.0.0.11:30805}
HEAD_NODE=${HEAD_NODE:-10.0.0.11}
NODES=${NODES:-"10.0.0.11 10.0.0.12"}
NS=${NS:-dynamo-system}
# This is a floor, not the final tokenizer count. The synthetic words split
# into many model tokens; 8000 produced about 78k prompt tokens in l2-smoke,
# which is long enough to span many chunks but still below max_model_len.
PROMPT_TOKENS=${PROMPT_TOKENS:-8000}
GEN=${GEN:-8}
OUT=${OUT:-/tmp/hotpath-smoke}
KC='export KUBECONFIG=/etc/rancher/k3s/k3s.yaml'
mkdir -p "$OUT"

sshq() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$@"; }

build_prompt() {
  python3 - "$PROMPT_TOKENS" "$GEN" "$1" <<'PY'
import json, sys
n, gen, nonce = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
words = [f"kvitem{i:06d}" for i in range(int(n * 1.4))]
print(json.dumps({
    "model": "nemotron",
    "prompt": f"Hotpath smoke {nonce}. Catalogue: " + " ".join(words) + "\nSummarise.",
    "max_tokens": gen,
    "temperature": 0.0,
}))
PY
}

send() {
  local tag=$1
  local t0 t1 code
  t0=$(python3 -c 'import time; print(time.time())')
  code=$(curl -s -m 900 -X POST "$ENDPOINT/v1/completions" \
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

snap_metrics() {
  local tag=$1
  for ip in $NODES; do
    sshq "$ip" "curl -s -m 5 http://127.0.0.1:9500/metrics" 2>/dev/null \
      | grep '^lmcache_' > "$OUT/$tag-$ip.metrics"
  done
}

delta_metrics() {
  local a=$1 b=$2
  for ip in $NODES; do
    echo "  --- $ip"
    join -j1 -o 0,1.2,2.2 \
      <(awk '{print $1, $2}' "$OUT/$a-$ip.metrics" | sort) \
      <(awk '{print $1, $2}' "$OUT/$b-$ip.metrics" | sort) 2>/dev/null \
      | awk '
        /lmcache_mp_(lookup|l0_l1|l1_|l2_|active_p2p)/ {
          d=$3-$2
          if (d != 0) printf "    %-64s %+d\n", $1, d
        }'
  done
}

rdma_words() {
  local ip=$1 total=0 v
  for d in rocep1s0f1 roceP2p1s0f1; do
    v=$(sshq "$ip" "cat /sys/class/infiniband/$d/ports/1/counters/port_xmit_data 2>/dev/null" || echo 0)
    total=$((total + ${v:-0}))
  done
  echo "$total"
}

snap_rdma() {
  local tag=$1
  for ip in $NODES; do
    rdma_words "$ip" > "$OUT/$tag-$ip.rdma"
  done
}

delta_rdma() {
  local a=$1 b=$2
  for ip in $NODES; do
    before=$(cat "$OUT/$a-$ip.rdma" 2>/dev/null || echo 0)
    after=$(cat "$OUT/$b-$ip.rdma" 2>/dev/null || echo 0)
    words=$((after - before))
    mb=$((words * 4 / 1048576))
    printf "  %s: %+d words = %+d MB\n" "$ip" "$words" "$mb"
  done
}

clear_l1() {
  echo "  clearing L1 through lmcache CLI inside worker pods"
  pods=$(sshq "$HEAD_NODE" "$KC; kubectl -n $NS get pods -o name | grep -i decodeworker" 2>/dev/null)
  [ -n "$pods" ] || { echo "  !! no decodeworker pods found"; return 1; }
  while read -r pod; do
    [ -n "$pod" ] || continue
    printf "    %s: " "$pod"
    sshq "$HEAD_NODE" "$KC; kubectl -n $NS exec ${pod#pod/} -- lmcache kvcache clear --url http://127.0.0.1:9500" >/dev/null \
      && echo ok || echo FAILED
  done <<< "$pods"
}

echo "=== wait for model registration"
for i in $(seq 60); do
  if curl -sf -m 10 "$ENDPOINT/v1/models" -o "$OUT/models.json" 2>/dev/null; then
    if python3 - "$OUT/models.json" <<'PY' >/dev/null 2>&1; then
import json, sys
d=json.load(open(sys.argv[1]))
ids=[m.get("id") for m in d.get("data", [])]
raise SystemExit(0 if "nemotron" in ids else 1)
PY
      echo "  nemotron visible after $((i*5))s"
      break
    fi
  fi
  [ "$i" = 60 ] && {
    echo "!! endpoint serving, but model 'nemotron' is not visible at $ENDPOINT/v1/models"
    cat "$OUT/models.json" 2>/dev/null || true
    exit 1
  }
  sleep 5
done

echo "=== stage A: hot peer L1 over P2P"
build_prompt "p2p-$(date +%s)-$RANDOM" > "$OUT/prompt.json"
snap_metrics a0; snap_rdma a0
send A1-cold || { echo "!! A1 failed; response in $OUT/resp-A1-cold.json"; exit 1; }
sleep 10
snap_metrics a1; snap_rdma a1
send A2-peer || { echo "!! A2 failed; response in $OUT/resp-A2-peer.json"; exit 1; }
sleep 10
snap_metrics a2; snap_rdma a2
echo "--- stage A metrics, cold store"
delta_metrics a0 a1
echo "--- stage A metrics, peer candidate"
delta_metrics a1 a2
echo "--- stage A RDMA during peer candidate"
delta_rdma a1 a2

echo
echo "=== stage B: L1 cleared, shared/remote L2 candidate"
build_prompt "l2-$(date +%s)-$RANDOM" > "$OUT/prompt.json"
snap_metrics b0; snap_rdma b0
send B1-cold || { echo "!! B1 failed; response in $OUT/resp-B1-cold.json"; exit 1; }
sleep 20
clear_l1
sleep 5
snap_metrics b1; snap_rdma b1
send B2-after-clear || { echo "!! B2 failed; response in $OUT/resp-B2-after-clear.json"; exit 1; }
sleep 20
snap_metrics b2; snap_rdma b2
echo "--- stage B metrics, after L1 clear"
delta_metrics b1 b2
echo "--- stage B RDMA during after-clear request"
delta_rdma b1 b2

echo
echo "Interpretation:"
echo "  Stage A works when the second request has l2_prefetch hits on the answering node and RDMA counters move."
echo "  Stage B works for shared/remote L2 when the after-clear request has l2_prefetch hits without a full l0_l1 store pattern."
echo "  If Stage B recomputes, shared/remote L2 is not actually in the hot miss path."
echo "  Raw files are in $OUT"
