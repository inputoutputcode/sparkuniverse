#!/usr/bin/env bash
# Does vLLM #45238 fire on Nemotron Nano?
#
# The claim: in `--mamba-cache-mode align` -- the only mode these hybrids
# support -- vLLM retains one Mamba state checkpoint per request, at the last
# block boundary before the prompt ends. HybridKVCacheCoordinator requires
# every cache group to hit, so if that single checkpoint lands in the
# request-unique part of the prompt, the Mamba group misses and vetoes the
# attention groups' matches. Prefix caching drops to exactly 0%, silently.
#
#     trigger:  floor((prompt_len - 1) / block_size) * block_size > shared_prefix
#
# The issue was filed against Qwen3.5-4B at block_size 528. Nano's attention
# block is forced up to the mamba page size, 2128, so checkpoints are 4x
# sparser here and the issue says larger blocks are strictly worse.
#
# Design: hold prompt_len constant and move only the shared-prefix boundary,
# so the two arms differ in exactly the quantity the trigger depends on.
#
#   P = 6456 tokens, checkpoint = floor(6455/2128)*2128 = 6384
#     arm A, shared 6400: 6384 > 6400 is false -> checkpoint inside the
#                         shared prefix -> reuse should work
#     arm B, shared 6000: 6384 > 6000 is true  -> checkpoint in unique tokens
#                         -> reuse should collapse
#
# Run on a Spark. No LMCache: this isolates vLLM's own prefix cache, which is
# what the issue is about. Add the connector afterwards to see whether LMCache
# masks or inherits the behaviour.
set -uo pipefail
exec 9>/tmp/spark-bench.lock; flock -n 9 || { echo "lock held"; exit 3; }

IMG=${IMG:-dynamo-vllm-lmcache:1.3.0-lmc052-arm64}
M=${MODEL:-/models/nemotron-3-nano-30b-nvfp4}
BLOCK=${BLOCK:-2128}
PORT=8011; NAME=nemotron

docker rm -f vllm-align >/dev/null 2>&1

# `docker rm -f` returns before the process releases GPU memory. Without this
# the next engine dies in init_device() with a traceback that says nothing
# about memory, while nvidia-smi shows 30 GB still held by a pid whose
# container is gone. Same barrier as cache-correctness.sh.
#
# memory.used reads [N/A] on GB10 because the memory is unified, so ask about
# compute apps and threshold instead.
settle() {
  echo -n "  waiting for the GPU to drain"
  for _ in $(seq 60); do
    apps=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
           | awk -v m="${GPU_IDLE_MB:-1024}" '$1+0 > m' | grep -c '[0-9]')
    [ "${apps:-0}" -eq 0 ] && { echo " ok"; sleep 5; return; }
    echo -n "."
    sleep 5
  done
  echo
  echo "  !! GPU still held after 300 s:"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  exit 1
}
settle

echo "=== vLLM, no LMCache, block ${BLOCK}"
docker run -d --name vllm-align --gpus all --network host --ipc=host \
  --ulimit memlock=-1 -v ~/models:/models -e VLLM_OMNI_USE_QUACK_FP8=0 \
  --entrypoint vllm "$IMG" serve "$M" \
    --served-model-name $NAME --trust-remote-code --load-format fastsafetensors \
    --moe-backend cutlass --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
    --mamba-cache-mode align --enable-chunked-prefill --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --max-model-len 131072 --max-num-seqs 4 --max-num-batched-tokens 4255 \
    --gpu-memory-utilization 0.55 \
    --port $PORT >/dev/null

for _ in $(seq 300); do
  curl -sf "localhost:$PORT/v1/models" >/dev/null 2>&1 && break
  docker ps -q -f name=vllm-align | grep -q . || {
    # Not `tail -40`. The root cause sits above the API server's own
    # traceback, and tailing prints "See root cause above" and nothing else.
    echo "!! vllm exited"
    docker logs vllm-align 2>&1 \
      | grep -iE "EngineCore failed|Error|assert|out of memory|no memory|Free memory" \
      | grep -viE "^\(EngineCore pid=[0-9]+\) ERROR .*\[core\.py:[0-9]+\] +(File|return|self|\^|super|engine_core|async|with|next)" \
      | head -20
    echo "  -- GPU:"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv
    exit 1; }
  sleep 5
done
docker logs vllm-align 2>&1 | grep -iE "attention block size|GPU KV cache size" | sed 's/^/  /'

BLOCK=$BLOCK PORT=$PORT NAME=$NAME python3 - <<'PY'
import os, time, random, json, requests
URL=f"http://localhost:{os.environ['PORT']}"; M=os.environ['NAME']
B=int(os.environ['BLOCK'])
P=3*B + 72                      # 6456 for B=2128
CKPT=((P-1)//B)*B               # 6384
N=8                             # requests per arm, one shared prefix

def counters():
    m=requests.get(f"{URL}/metrics", timeout=10).text
    g=lambda k: sum(float(l.split()[-1]) for l in m.splitlines()
                    if l.startswith(k) and not l.startswith("#"))
    return g("vllm:prefix_cache_queries_total"), g("vllm:prefix_cache_hits_total")

def arm(label, shared_len, seed):
    rng=random.Random(seed)
    shared=[rng.randrange(1000,90000) for _ in range(shared_len)]
    q0,h0=counters(); ttfts=[]
    for i in range(N):
        uniq=[random.Random(9000+i).randrange(1000,90000) for _ in range(P-shared_len)]
        t0=time.perf_counter()
        r=requests.post(f"{URL}/v1/completions", timeout=900,
            json={"model":M,"prompt":shared+uniq,"max_tokens":1,"temperature":0})
        r.raise_for_status(); ttfts.append(time.perf_counter()-t0)
    time.sleep(3); q1,h1=counters()
    dq,dh=q1-q0,h1-h0
    # request 1 populates; 2..N are the ones that can hit
    print(f"  {label:<34} shared={shared_len:<5} queries={dq:>9,.0f} "
          f"hits={dh:>9,.0f}  {100*dh/dq if dq else 0:>5.1f}%   "
          f"ttft first {ttfts[0]:.2f}s  rest {sum(ttfts[1:])/(N-1):.2f}s")
    return dh/dq if dq else 0

print(f"\n  block {B}, prompt {P}, checkpoint at {CKPT}\n")
a=arm("A  checkpoint in shared prefix", CKPT+16, 11)   # 6400 > 6384
b=arm("B  checkpoint in unique tokens", CKPT-384, 22)  # 6000 < 6384

print()
if a > 0.3 and b < 0.05:
    print("  #45238 REPRODUCES on Nemotron Nano.")
    print("  Same prompt length, same request count; only the shared-prefix")
    print("  boundary moved, and reuse went from working to zero.")
elif a > 0.3 and b > 0.3:
    print("  Does not reproduce: both arms reuse. Either Nemotron's align")
    print("  implementation differs, or this build carries a fix.")
else:
    print("  Inconclusive: arm A did not reuse either. Check that the arena")
    print("  holds both arms' prefixes and that prefix caching is enabled.")
PY

docker rm -f vllm-align >/dev/null 2>&1
