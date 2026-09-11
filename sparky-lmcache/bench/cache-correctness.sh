#!/usr/bin/env bash
# Is KV reuse on a Mamba hybrid correct, or only fast?
#
# The positive control we already have proves LMCache RETURNS a cached context.
# It does not prove the model then generates what it would have generated
# without the cache. On these models roughly 79% of what must be restored is
# SSM state rather than attention KV, so a cache that silently dropped the
# state would still look fast and would still report a hit.
#
# Three arms, same prompt, greedy:
#   A  fresh engine, empty cache          stores the context
#   B  engine restarted, cache warm       must be served by LMCache
#   C  engine AND cache both cleared      determinism control
#
# A vs C is the run-to-run noise floor and it is not zero. bf16 SSM is not
# bitwise deterministic on this stack: two runs of an identical NLL config
# differed by 0.00034 nats. So "A differs from B" proves nothing on its own.
# The question is whether A-vs-B diverges materially more than A-vs-C.
#
# Usage:
#   ./cache-correctness.sh                      Nano, 31920-token prompt
#   MODEL=/models/... CHUNK=4224 BT=8447 T=29568 UTIL=0.72 ./cache-correctness.sh
set -uo pipefail
exec 9>/tmp/spark-bench.lock; flock -n 9 || { echo "lock held"; exit 3; }

IMG=${IMG:-dynamo-vllm-lmcache:1.3.0-lmc052-arm64}
M=${MODEL:-/models/nemotron-3-nano-30b-nvfp4}
CHUNK=${CHUNK:-2128}; BT=${BT:-4255}; T=${T:-31920}
UTIL=${UTIL:-0.40}; L1_GB=${L1_GB:-12}; GEN=${GEN:-64}
OUT=${OUT:-$HOME/cache-correctness}
mkdir -p "$OUT"

# KEEP=1 leaves the containers up so their logs survive a failure. Without it a
# crash during startup destroys the only copy of the root cause.
cleanup() { [ "${KEEP:-0}" = 1 ] || docker rm -f l2 lmc >/dev/null 2>&1; }
trap cleanup EXIT

# The APIServer re-raises as "Engine core initialization failed. See root cause
# above", and the real error is in the EngineCore stream well above it. tail on
# the combined log lands past it every time.
dump_failure() {
  echo "  --- EngineCore ---"
  docker logs l2 2>&1 | grep -E '^\(EngineCore' | grep -viE '^.*(INFO|Loading safetensors)' | tail -30
  echo "  --- last engine lines ---"
  docker logs l2 2>&1 | grep -E '^\(EngineCore' | tail -8
  echo "  --- memory now ---"
  free -g | awk '/^Mem:/{print "  total "$2"G used "$3"G available "$7"G"}'
  docker ps -a --format '  {{.Names}}\t{{.Status}}' | grep -E 'l2|lmc'
}

# Wait for the previous engine's memory to come back. The threshold has to
# scale with UTIL: a fixed 70 GB is fine for Nano at 0.40 (48.7 GB) and lets
# Super at 0.72 (87.6 GB) start into a wall, where EngineCore dies during init
# and surfaces as "KeyboardInterrupt: terminated" from _interrupt_init.
settle() {
  local tot need avail used
  tot=$(free -b | awk '/^Mem:/{print $2}')
  need=$(awk -v t="$tot" -v u="$UTIL" 'BEGIN{printf "%.0f", t*u + 6e9}')
  for _ in $(seq 90); do
    avail=$(free -b | awk '/^Mem:/{print $7}')
    [ "$avail" -gt "$need" ] && break
    sleep 5
  done
  # MemAvailable says nothing about what the driver still holds. Loading 74.8 GB
  # of weights failed with NVRM NV_ERR_NO_MEMORY while MemAvailable read 97 GB,
  # because the previous engine's device allocation had not been released.
  #
  # memory.used is [N/A] on GB10 - unified memory reports no framebuffer usage,
  # the same reason DCGM drops every FB_* field. So wait for the driver to show
  # no compute apps instead, which is what NVIDIA's own Ultra preflight checks.
  # Ignore small contexts: the LMCache server holds ~200 MiB permanently by
  # design, so a bare "any compute app" test never clears.
  for _ in $(seq 60); do
    apps=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
           | awk -v m="${GPU_IDLE_MB:-1024}" '$1+0 > m' | grep -c '[0-9]')
    [ "${apps:-0}" -eq 0 ] && { sleep 5; return; }
    sleep 5
  done
  echo "  !! driver still shows ${apps} compute app(s) before start"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/     /'
  echo "     system available $(numfmt --to=iec "${avail:-0}"), needed $(numfmt --to=iec "$need")"
}

start_lmc() {
  docker rm -f lmc >/dev/null 2>&1
  docker run -d --name lmc --network host --ipc=host --gpus all --ulimit memlock=-1 \
    --entrypoint lmcache "$IMG" \
    server --l1-size-gb "$L1_GB" --chunk-size "$CHUNK" \
           --eviction-policy LRU --http-port 9500 >/dev/null
  for _ in $(seq 30); do ss -ltn 2>/dev/null | grep -q ":5555 " && return; sleep 2; done
  echo "!! lmcache server never listened on 5555"; docker logs lmc 2>&1 | tail -20; exit 1
}

start_vllm() {
  docker rm -f l2 >/dev/null 2>&1
  settle
  docker run -d --name l2 --gpus all --network host --ipc=host --ulimit memlock=-1 \
    -v ~/models:/models -e VLLM_OMNI_USE_QUACK_FP8=0 --entrypoint vllm "$IMG" serve "$M" \
      --served-model-name nemotron --trust-remote-code --load-format fastsafetensors \
      --moe-backend cutlass --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
      --mamba-cache-mode align --enable-chunked-prefill --enable-prefix-caching \
      --max-model-len 131072 --max-num-seqs 1 --max-num-batched-tokens "$BT" \
      --gpu-memory-utilization "$UTIL" \
      --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}' \
      --port 8011 >/dev/null
  for _ in $(seq 300); do
    curl -sf localhost:8011/v1/models >/dev/null 2>&1 && { sleep 10; return; }
    docker ps -q -f name=l2 | grep -q . || { echo "!! vllm exited during startup"; dump_failure; exit 1; }
    sleep 5
  done
  echo "!! vllm never came up"; dump_failure; exit 1
}

# One request, greedy, with per-token logprobs. Logprobs matter: they diverge
# before the argmax does, so they detect a corrupted state that token equality
# would miss.
gen() {
  T=$T GEN=$GEN ARM=$1 OUT=$OUT python3 - <<'PY'
import os, json, random, time, requests
T=int(os.environ['T']); GEN=int(os.environ['GEN'])
arm=os.environ['ARM']; out=os.environ['OUT']
prompt=[random.Random(1234).randrange(1000, 90000) for _ in range(T)]
t0=time.perf_counter()
r=requests.post('http://localhost:8011/v1/completions', timeout=1800, json={
    'model':'nemotron','prompt':prompt,'max_tokens':GEN,
    'temperature':0,'seed':42,'logprobs':0})
r.raise_for_status(); d=r.json(); c=d['choices'][0]
rec={'arm':arm,'wall_s':round(time.perf_counter()-t0,2),
     'text':c.get('text',''),
     'tokens':(c.get('logprobs') or {}).get('tokens'),
     'logprobs':(c.get('logprobs') or {}).get('token_logprobs'),
     'usage':d.get('usage')}
json.dump(rec, open(f'{out}/{arm}.json','w'))
print(f"  arm {arm}: {rec['wall_s']}s  {len(rec['tokens'] or [])} tokens")
PY
}

lmc_since() { docker logs lmc 2>&1 | tail -n "+$1"; }

echo "=== model $M   prompt $T tokens   generate $GEN"

echo "=== arm A: fresh engine, empty cache"
start_lmc
start_vllm
gen A
sleep 8
N=$(docker logs lmc 2>&1 | wc -l)

echo "=== arm B: engine restarted, cache retained"
start_vllm
gen B
sleep 5
echo "--- lmcache activity during B"
lmc_since $((N+1)) | grep -iE "retriev|retained keys" | tail -5 | sed 's/^/    /'
SERVED=$(lmc_since $((N+1)) | grep -c "Retrieved")
[ "$SERVED" -eq 0 ] && echo "    !! nothing retrieved - arm B was NOT served from cache, result is meaningless"

echo "=== arm C: engine and cache both cleared"
start_lmc
start_vllm
gen C

echo
OUT=$OUT python3 - <<'PY'
import os, json
out=os.environ['OUT']
a,b,c=(json.load(open(f'{out}/{x}.json')) for x in 'ABC')

def cmp(x,y):
    tx,ty=x['tokens'] or [],y['tokens'] or []
    n=min(len(tx),len(ty))
    first=next((i for i in range(n) if tx[i]!=ty[i]), None)
    same=sum(1 for i in range(n) if tx[i]==ty[i])
    lx,ly=x['logprobs'] or [],y['logprobs'] or []
    m=min(len(lx),len(ly))
    pairs=[(p,q) for p,q in zip(lx[:m],ly[:m]) if p is not None and q is not None]
    mad=sum(abs(p-q) for p,q in pairs)/len(pairs) if pairs else float('nan')
    return first, same, n, mad

fAC,sAC,nAC,mAC = cmp(a,c)
fAB,sAB,nAB,mAB = cmp(a,b)

print("  pair                first divergence   tokens identical   mean |dlogprob|")
print(f"  A vs C (noise floor) {str(fAC):>14}   {sAC:>6}/{nAC:<9}   {mAC:.6f}")
print(f"  A vs B (cache)       {str(fAB):>14}   {sAB:>6}/{nAB:<9}   {mAB:.6f}")
print()
if fAC is None and fAB is None:
    print("  VERDICT: identical across all three arms. Reuse is exact.")
elif fAC is None and fAB is not None:
    print(f"  VERDICT: engine is deterministic (A==C) but the cached run diverges")
    print(f"           at token {fAB}. KV reuse changes the output. Not sound.")
elif fAB is not None and fAC is not None and fAB < fAC:
    print(f"  VERDICT: cached run diverges earlier ({fAB}) than the noise floor")
    print(f"           ({fAC}). Suggestive of state loss, not just numerics.")
else:
    print("  VERDICT: cache divergence is within the run-to-run noise floor.")
    print("           No evidence of state loss at this prompt length.")
json.dump({'A_vs_C':{'first':fAC,'same':sAC,'n':nAC,'mad':mAC},
           'A_vs_B':{'first':fAB,'same':sAB,'n':nAB,'mad':mAB}},
          open(f'{out}/summary.json','w'), indent=1)
print(f"\n  artifacts in {out}/")
PY
