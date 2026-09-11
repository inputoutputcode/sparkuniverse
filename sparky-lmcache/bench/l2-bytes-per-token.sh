#!/usr/bin/env bash
# Bytes per token in LMCache's MP store.
# Cap the store at a known size, push distinct contexts until the first is
# evicted: bytes_per_token = size / tokens_that_fit.
#
# --http-port 9500 is not cosmetic. LMCache's HTTP server defaults to 8080,
# which is cadvisor, and the server exits with
#   [Errno 98] error while attempting to bind on address ('0.0.0.0', 8080)
# after having already logged a successful ZMQ startup, so the failure reads
# like something else went wrong.
#
# Reading the result: this prints L1 / tokens, but eviction fires at the 0.80
# watermark, so the honest figure is 0.8 x that. At 4 GB expect it to report
# ~17,900 B/token where the true value is ~14,300, consistent with the 14,848
# measured three other ways in docs/findings.md.
#
# --kv-cache-memory-bytes is what makes this a test at all. Run at
# gpu-memory-utilization 0.70 alone, the device arena is ~70 GB, about 20M
# tokens, and the probe context never leaves the *GPU* prefix cache. The
# 2026-08-16 run reported "cached" for all 78 iterations out of 2.52M tokens,
# which at 14,848 B/token would be 37 GB in a 4 GB store. It was measuring the
# device cache the whole time and could not have failed.
#
# 500 MB is ~143k device tokens, above the 4 x 31,920 needed for the running
# sequences and small enough that the probe falls through to L1. The reported
# capacity is then device + L1, so subtract the device tokens before dividing.
#
# The direct method is better and already settled this: read
# lmcache_mp_l1_memory_usage_bytes either side of one request. On 2026-08-15
# that gave 505,544,704 bytes for a 32,000-token prompt = 34,048 x 14,848,
# exact once chunk padding is accounted for. Prefer it. This script survives
# because an eviction bracket tests a different thing -- that the arena
# behaves like its configured size under pressure.
set -u
exec 9>/tmp/spark-bench.lock; flock -n 9 || { echo "lock held"; exit 3; }
IMG=dynamo-vllm-lmcache:1.3.0-lmc052-arm64
M=${MODEL:-/models/nemotron-3-nano-30b-nvfp4}
BT=${BT:-4255}
L1_GB=${L1_GB:-4}
T=${T:-31920}
NAME=nemotron; PORT=8011

docker ps -a --format '{{.Names}}' | grep -vE '^(dcgm|obs-)' | xargs -r docker rm -f >/dev/null 2>&1
[ -x ~/evict.sh ] && ~/evict.sh ~/models >/dev/null; sleep 20

echo "=== starting LMCache MP server, ${L1_GB} GB"
docker run -d --name lmc --network host --ipc=host --gpus all \
  --ulimit memlock=-1 --entrypoint lmcache "$IMG" \
  server --l1-size-gb "$L1_GB" --chunk-size 2128 --eviction-policy LRU \
         --http-port 9500 >/dev/null
for _ in $(seq 30); do
  ss -ltn 2>/dev/null | grep -q ":5555 " && { echo "  listening on 5555"; break; }
  docker ps -q -f name=lmc | grep -q . || { echo "  !! server exited"; docker logs lmc 2>&1 | tail -20; exit 1; }
  sleep 2
done

echo "=== starting vLLM"
docker run -d --name l2 --gpus all --network host --ipc=host \
  --ulimit memlock=-1 -v ~/models:/models -e VLLM_OMNI_USE_QUACK_FP8=0 \
  --entrypoint vllm "$IMG" serve "$M" \
    --served-model-name $NAME --trust-remote-code --load-format fastsafetensors \
    --moe-backend cutlass --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
    --mamba-cache-mode align --enable-chunked-prefill --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --max-model-len 131072 --max-num-seqs 4 --max-num-batched-tokens $BT \
    --gpu-memory-utilization 0.70 \
    --kv-cache-memory-bytes ${DEV_BYTES:-500000000} \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}' \
    --port $PORT >/dev/null

for _ in $(seq 300); do
  curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1 && break
  docker ps -q -f name=l2 | grep -q . || { echo "!! vllm did not start"
    docker logs l2 2>&1 | sed -n '1,/APIServer pid=1) Traceback/p' | tail -40; exit 1; }
  sleep 5
done
docker logs l2 2>&1 | grep -iE "block size|GPU KV cache size|LMCache" | sed 's/^/  /' | head -10

L1_GB=$L1_GB PORT=$PORT NAME=$NAME T=$T python3 - <<'PY'
import os, time, random, requests
URL=f"http://localhost:{os.environ['PORT']}"; M=os.environ['NAME']
L1=float(os.environ['L1_GB'])*1e9; T=int(os.environ['T'])
ctx=lambda s:[random.Random(s).randrange(1000,90000) for _ in range(T)]
def ttft(p):
    t0=time.perf_counter()
    r=requests.post(f"{URL}/v1/completions",timeout=900,stream=True,
        json={"model":M,"prompt":p,"max_tokens":1,"temperature":0,"stream":True})
    r.raise_for_status()
    for line in r.iter_lines():
        if line and line.startswith(b"data: ") and line[6:].strip()!=b"[DONE]":
            return time.perf_counter()-t0
first=ctx(1); cold=ttft(first); warm=ttft(first)
print(f"\n  {T:,}-token context   cold {cold:.1f}s   warm {warm:.1f}s")
th=cold*0.5; print(f"  evicted above {th:.1f}s\n")
for k in range(2,80):
    ttft(ctx(k)); t=ttft(first); tok=k*T
    print(f"  after {k:>2} contexts ({tok:>9,} tok):  probe {t:>5.1f}s  "
          f"{'EVICTED' if t>th else 'cached'}")
    if t>th:
        print(f"\n  {L1/1e9:.0f} GB held < {tok:,} tokens  ->  ~{L1/tok:,.0f} bytes/token")
        print(f"  (vLLM device accounting 25,357 | attention-only 4,224)")
        break
PY
docker rm -f l2 lmc >/dev/null 2>&1
