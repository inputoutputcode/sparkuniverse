#!/usr/bin/env bash
# KVBM Go/No-Go request test for Super.
#
# The test stores a long prompt, sends enough other long prompts to put pressure
# on the intentionally small device KV arena, then repeats the first prompt.
# A fast repeat is the first evidence that KVBM can recover useful KV from a
# lower tier without requiring a large LMCache-style L1.
set -uo pipefail

ENDPOINT=${ENDPOINT:-http://10.0.0.11:30806}
HEAD_NODE=${HEAD_NODE:-10.0.0.11}
NODES=${NODES:-"10.0.0.11 10.0.0.12"}
NS=${NS:-dynamo-system}
PROMPT_TOKENS=${PROMPT_TOKENS:-8000}
WARM_PROMPTS=${WARM_PROMPTS:-10}
GEN=${GEN:-1}
OUT=${OUT:-/tmp/kvbm-go-nogo}
KC='export KUBECONFIG=/etc/rancher/k3s/k3s.yaml'
mkdir -p "$OUT"

sshq() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$@"; }

build_prompt() {
  local nonce=$1 path=$2
  python3 - "$PROMPT_TOKENS" "$GEN" "$nonce" > "$path" <<'PY'
import json, sys
n, gen, nonce = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
words = [f"superkv{i:06d}" for i in range(int(n * 1.4))]
print(json.dumps({
    "model": "nemotron",
    "prompt": f"KVBM smoke {nonce}. Catalogue: " + " ".join(words) + "\nSummarise.",
    "max_tokens": gen,
    "temperature": 0.0,
}))
PY
}

send_file() {
  local tag=$1 path=$2
  local t0 t1 code
  t0=$(python3 -c 'import time; print(time.time())')
  code=$(curl -s -m 1800 -X POST "$ENDPOINT/v1/completions" \
    -H 'Content-Type: application/json' -d @"$path" \
    -o "$OUT/resp-$tag.json" -w '%{http_code}')
  t1=$(python3 -c 'import time; print(time.time())')
  python3 - "$OUT/resp-$tag.json" "$tag" "$code" "$t0" "$t1" <<'PY'
import json, sys
path, tag, code, t0, t1 = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
wall = t1 - t0
try:
    d=json.load(open(path))
    u=d.get("usage") or {}
    print(f"{tag},{code},{wall:.3f},{u.get('prompt_tokens','?')},{u.get('completion_tokens','?')}")
except Exception as e:
    print(f"{tag},{code},{wall:.3f},?,? # {e}")
PY
  [ "$code" = 200 ]
}

snap() {
  local tag=$1
  for ip in $NODES; do
    sshq "$ip" "curl -s -m 5 http://127.0.0.1:9600/metrics 2>/dev/null; curl -s -m 5 http://127.0.0.1:9090/metrics 2>/dev/null" \
      | grep -iE 'kvbm|prefix|cache|offload|onboard' > "$OUT/$tag-$ip.metrics"
    sshq "$ip" "du -sb /mnt/kvbm-disk 2>/dev/null || true" > "$OUT/$tag-$ip.du"
  done
}

delta_metrics() {
  local a=$1 b=$2
  for ip in $NODES; do
    echo "--- $ip"
    join -j1 -o 0,1.2,2.2 \
      <(awk '{print $1, $2}' "$OUT/$a-$ip.metrics" | sort) \
      <(awk '{print $1, $2}' "$OUT/$b-$ip.metrics" | sort) 2>/dev/null \
      | awk '{d=$3-$2; if (d != 0) printf "  %-72s %+g\n", $1, d}' | head -120
    echo "  disk: $(cat "$OUT/$a-$ip.du" 2>/dev/null | awk '{print $1}') -> $(cat "$OUT/$b-$ip.du" 2>/dev/null | awk '{print $1}') bytes"
  done
}

echo "=== wait for model registration"
for i in $(seq 120); do
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
  [ "$i" = 120 ] && { echo "!! model not visible"; cat "$OUT/models.json" 2>/dev/null || true; exit 1; }
  sleep 5
done

echo "tag,http,wall_s,prompt_tokens,completion_tokens" | tee "$OUT/timings.csv"

build_prompt "target-$(date +%s)-$RANDOM" "$OUT/target.json"
snap t0
send_file cold-target "$OUT/target.json" | tee -a "$OUT/timings.csv" || { echo "!! cold target failed"; exit 1; }
snap t1

echo "=== warm $WARM_PROMPTS unique prompts to pressure the device arena"
for i in $(seq 1 "$WARM_PROMPTS"); do
  p="$OUT/warm-$i.json"
  build_prompt "warm-$i-$(date +%s)-$RANDOM" "$p"
  send_file "warm-$i" "$p" | tee -a "$OUT/timings.csv" || { echo "!! warm $i failed"; exit 1; }
done
snap t2

echo "=== repeat original target"
send_file repeat-target "$OUT/target.json" | tee -a "$OUT/timings.csv" || { echo "!! repeat target failed"; exit 1; }
snap t3

echo
echo "=== metric/disk deltas: cold target"
delta_metrics t0 t1
echo
echo "=== metric/disk deltas: warm pressure"
delta_metrics t1 t2
echo
echo "=== metric/disk deltas: repeat target"
delta_metrics t2 t3

echo
python3 - "$OUT/timings.csv" <<'PY'
import csv, sys
rows=list(csv.DictReader(open(sys.argv[1])))
by={r["tag"]: r for r in rows}
def wall(tag):
    return float(by[tag]["wall_s"])
cold=wall("cold-target")
repeat=wall("repeat-target")
ratio=repeat/cold if cold else 999
print(f"cold-target {cold:.2f}s")
print(f"repeat-target {repeat:.2f}s")
print(f"repeat/cold {ratio:.3f}")
if ratio <= 0.50:
    print("KVBM_GO: repeat was at least 2x faster after pressure")
elif ratio <= 0.80:
    print("KVBM_MAYBE: repeat improved, inspect metrics/logs before long run")
else:
    print("KVBM_NO_GO: repeat looks like recompute or routing missed locality")
PY

echo
echo "raw artifacts in $OUT"
