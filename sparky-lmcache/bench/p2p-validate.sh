#!/usr/bin/env bash
# Does a KV cache stored on one Spark serve a request on the other, and is the
# answer correct? Two nodes, bare docker, no Kubernetes, no Dynamo.
#
# Run from the Mac Mini.
#
# Topology:
#   coordinator      spark A, http, peers register and heartbeat here
#   lmcache server   both nodes, p2p advertise on the RoCE fabric
#   vllm             both nodes, LMCacheMPConnector
#
# The test:
#   1. store a context on A
#   2. request the same context on B, which has never seen it
#   3. B can only answer from A, so a hit proves the transfer path
#   4. compare B's output against A's, greedy, to prove the transfer is correct
#
# Success is L2 > 0 in B's retained-keys line. Every single-node run so far
# reads "0 L2" because no peer adapter existed. L1 hits on B would mean the
# context leaked locally and the test proved nothing.
#
# Three transport arms, same payload, same instrument. The port counters
# distinguish them; wall time alone cannot, which is how a TCP result ended up
# in findings.md labelled RoCE.
#
#   rdma        ./p2p-validate.sh
#               UCX picks rc/dc on the pinned rails. Expect the counter to move
#               by roughly the payload.
#
#   tcp-fabric  NO_RDMA=1 ./p2p-validate.sh
#               Same cable, same addresses, TCP sockets. Isolates transport
#               from wire. This is what the 2026-08-14 run measured by
#               accident.
#
#   tcp-lan     NO_RDMA=1 UCX_DEVS=enP7s7 \
#                 A_ROCE=10.0.0.11 B_ROCE=10.0.0.12 ./p2p-validate.sh
#               1 GbE. The question this answers is whether the fabric is
#               needed at all for a KV chunk of this size, or only for Super's
#               3.11 GB at 128k.
#
# NO_RDMA withholds /dev/infiniband so UCX has no device to open and falls
# back to TCP. Do not use UCX_TLS=tcp for this: NIXL cannot build a backend
# under it and the server dies at startup with
#   nixl_cu13._bindings.nixlBackendError: NIXL_ERR_BACKEND
# Withholding the device is also the more faithful reproduction, since it is
# exactly the condition the 2026-08-14 run was in.
#
# Run all three at one payload before believing any ordering between them, and
# repeat with T=127680 for Super-sized transfers, where the balance should
# shift toward bandwidth and away from protocol overhead.
set -uo pipefail

IMG=${IMG:-dynamo-vllm-lmcache:1.3.0-lmc052-arm64}
M=${MODEL:-/models/nemotron-3-nano-30b-nvfp4}
CHUNK=${CHUNK:-2128}; BT=${BT:-4255}; T=${T:-31920}
UTIL=${UTIL:-0.40}; L1_GB=${L1_GB:-12}; GEN=${GEN:-64}

A_LAN=${A_LAN:-10.0.0.11};   A_ROCE=${A_ROCE:-10.1.0.11}
B_LAN=${B_LAN:-10.0.0.12};  B_ROCE=${B_ROCE:-10.1.0.12}
COORD_PORT=${COORD_PORT:-9300}
# Not 9400: that is dcgm-exporter. Not 9500 either, that is the lmcache
# server's own Prometheus endpoint.
P2P_PORT=${P2P_PORT:-9450}
# rocep1s0f1 -> enp1s0f1np1 -> 192.168.177.x, the address LMCache advertises.
UCX_DEVS=${UCX_DEVS:-rocep1s0f1:1}
UCX_MAX_RNDV_RAILS=${UCX_MAX_RNDV_RAILS:-1}
UCX_CM_USE_ALL_DEVICES=${UCX_CM_USE_ALL_DEVICES:-n}
# The devices whose counters prove the transport. Both rails, so that traffic
# escaping onto the second one is visible rather than silently uncounted.
IB_DEVS=${IB_DEVS:-"rocep1s0f1 roceP2p1s0f1"}
OUT=${OUT:-$HOME/p2p-validate}
mkdir -p "$OUT"

# Sum of port_xmit_data across the cabled rails, in 4-byte words. Read on the
# sending node before and after the transfer. This is the only instrument that
# distinguishes RDMA from TCP: throughput cannot, because a 474 MB transfer
# over TCP on this cable lands in the same range.
ib_xmit() {
  local host=$1 total=0 v
  for d in $IB_DEVS; do
    v=$(sshx "$host" "cat /sys/class/infiniband/$d/ports/1/counters/port_xmit_data 2>/dev/null" || echo 0)
    total=$((total + ${v:-0}))
  done
  echo "$total"
}

# The request helpers run on this machine, not in a container. macOS system
# python has no requests, the aiperf venv does.
# Local python only parses JSON, so the stdlib is enough. All HTTP happens on
# the nodes.
PY=${PY:-python3}

# Optional. If setting --p2p-advertise-url alone does not produce L2 hits, the
# peer backend may need declaring explicitly.
L2_ADAPTER=${L2_ADAPTER:-}

SSH_OPTS=${SSH_OPTS:-"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"}
sshx() { ssh -o ConnectTimeout=5 $SSH_OPTS "$1" "${@:2}"; }

ib_xmit_file() {
  local host=$1 out=$2
  : > "$out"
  for d in $IB_DEVS; do
    sshx "$host" "cat /sys/class/infiniband/$d/ports/1/counters/port_xmit_data 2>/dev/null || echo 0" \
      | awk -v d="$d" '{print d, $1}' >> "$out"
  done
}

ib_xmit_delta_report() {
  local before=$1 after=$2 total=0
  awk '
    NR==FNR {b[$1]=$2; next}
    {
      delta=$2 - b[$1]
      total += delta
      printf "    %-14s %12d words = %8.1f MB\n", $1, delta, delta * 4 / 1048576
    }
    END {
      printf "    %-14s %12d words = %8.1f MB\n", "total", total, total * 4 / 1048576
    }
  ' "$before" "$after"
}

cleanup() {
  [ "${KEEP:-0}" = 1 ] && return
  for n in "$A_LAN" "$B_LAN"; do sshx "$n" 'docker rm -f lmc vllm coord >/dev/null 2>&1' ; done
}
trap cleanup EXIT

start_lmc() {
  local lan=$1 roce=$2 id=$3
  # --device /dev/infiniband and UCX_NET_DEVICES are what make this RDMA.
  # Without the device the container has nothing to open and UCX silently
  # falls back to TCP over the same cable; without the pin UCX picks by its
  # own heuristics and takes the LAN. The 2026-08-14 run had neither, and its
  # 2.22 GB/s went into findings.md labelled as RoCE. Both are TCP tells that
  # no throughput number can reveal, only the port counters can.
  # NO_RDMA withholds the device only. UCX_NET_DEVICES still applies, and must:
  # without it UCX enumerates every interface and picks the CX-7 netdev even
  # when the peer was advertised on the LAN address, so the "1 GbE" arm
  # measured TCP over the fabric and reported 2.56 GB/s -- 20 Gb/s on a link
  # that cannot carry it. Name the interface, or the arm is not the arm.
  local dev="--device /dev/infiniband --cap-add IPC_LOCK"
  [ "${NO_RDMA:-0}" = 1 ] && dev=""
  sshx "$lan" "docker rm -f lmc >/dev/null 2>&1
    docker run -d --name lmc --network host --ipc=host --gpus all --ulimit memlock=-1 \
      $dev -e UCX_NET_DEVICES=$UCX_DEVS \
      -e UCX_MAX_RNDV_RAILS=$UCX_MAX_RNDV_RAILS \
      -e UCX_CM_USE_ALL_DEVICES=$UCX_CM_USE_ALL_DEVICES \
      --entrypoint lmcache $IMG \
      server --l1-size-gb $L1_GB --chunk-size $CHUNK --eviction-policy LRU \
             --host 0.0.0.0 --http-port 9500 --instance-id $id \
             --p2p-advertise-url ${roce}:${P2P_PORT} \
             --p2p-transfer-engine nixl \
             --coordinator-url http://${A_ROCE}:${COORD_PORT} \
             --coordinator-advertise-ip ${roce} \
             $L2_ADAPTER >/dev/null"
  for _ in $(seq 30); do
    sshx "$lan" "ss -ltn | grep -q ':5555 '" && return 0
    sshx "$lan" 'docker ps -q -f name=lmc | grep -q .' || {
      echo "  !! lmcache server exited on $lan"; sshx "$lan" 'docker logs lmc 2>&1 | tail -25'; exit 1; }
    sleep 2
  done
  echo "  !! lmcache server never listened on $lan"; exit 1
}

start_vllm() {
  local lan=$1
  sshx "$lan" "docker rm -f vllm >/dev/null 2>&1
    docker run -d --name vllm --gpus all --network host --ipc=host --ulimit memlock=-1 \
      -v ~/models:/models --entrypoint vllm $IMG serve $M \
        --served-model-name nemotron --trust-remote-code --load-format fastsafetensors \
        --moe-backend cutlass --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
        --mamba-cache-mode align --enable-chunked-prefill --enable-prefix-caching \
        --enable-prompt-tokens-details \
        --max-model-len 131072 --max-num-seqs 4 --max-num-batched-tokens $BT \
        --gpu-memory-utilization $UTIL \
        --kv-transfer-config '{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\"}' \
        --port 8011 >/dev/null"
  for _ in $(seq 300); do
    curl -sf "http://$lan:8011/v1/models" >/dev/null 2>&1 && { sleep 8; return 0; }
    sshx "$lan" 'docker ps -q -f name=vllm | grep -q .' || {
      echo "  !! vllm exited on $lan"; sshx "$lan" 'docker logs vllm 2>&1 | grep -E "^\(EngineCore" | tail -25'; exit 1; }
    sleep 5
  done
  echo "  !! vllm never came up on $lan"; exit 1
}

# The request runs ON the node against localhost, not from this machine.
# macOS gates local-network access per binary, so a venv python gets errno 65
# where curl succeeds. Driving it remotely also keeps LAN latency out of the
# timings, which is what we are measuring.
gen() {
  local lan=$1 tag=$2
  sshx "$lan" "T=$T GEN=$GEN TAG=$tag python3 -" <<'PY' > "$OUT/$tag.json"
import os, json, random, time, requests, sys
T=int(os.environ['T']); GEN=int(os.environ['GEN']); tag=os.environ['TAG']
prompt=[random.Random(4242).randrange(1000, 90000) for _ in range(T)]
t0=time.perf_counter()
r=requests.post('http://localhost:8011/v1/completions', timeout=1800, json={
    'model':'nemotron','prompt':prompt,'max_tokens':GEN,
    'temperature':0,'seed':42,'logprobs':0})
r.raise_for_status(); d=r.json(); c=d['choices'][0]
json.dump({'tag':tag,'wall_s':round(time.perf_counter()-t0,2),
           'tokens':(c.get('logprobs') or {}).get('tokens'),
           'usage':d.get('usage')}, sys.stdout)
PY
  [ -s "$OUT/$tag.json" ] || { echo "  !! $tag produced no output"; return 1; }
  "$PY" - "$OUT/$tag.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
u=r.get('usage') or {}; d=(u.get('prompt_tokens_details') or {})
print(f"  {r['tag']}: {r['wall_s']}s  cached_tokens={d.get('cached_tokens','n/a')}")
PY
}

for spec in "$A_LAN $A_ROCE" "$B_LAN $B_ROCE"; do
  set -- $spec
  if sshx "$1" "ss -ltn | grep -q ':$P2P_PORT '"; then
    echo "!! $P2P_PORT is already bound on $1"
    sshx "$1" "ss -ltnp | grep ':$P2P_PORT '" | sed 's/^/   /'
    echo "   pick another with P2P_PORT=... (9400 is dcgm-exporter, 9500 is lmcache metrics)"
    exit 1
  fi
done

echo "=== coordinator on $A_ROCE:$COORD_PORT"
echo "=== ucx"
echo "  UCX_NET_DEVICES=$UCX_DEVS"
echo "  UCX_MAX_RNDV_RAILS=$UCX_MAX_RNDV_RAILS"
echo "  UCX_CM_USE_ALL_DEVICES=$UCX_CM_USE_ALL_DEVICES"
sshx "$A_LAN" "docker rm -f coord >/dev/null 2>&1
  docker run -d --name coord --network host --entrypoint lmcache $IMG \
    coordinator --host 0.0.0.0 --port $COORD_PORT --chunk-size $CHUNK >/dev/null"
sleep 6
sshx "$A_LAN" 'docker ps -q -f name=coord | grep -q .' || {
  echo "  !! coordinator exited, check its flags"; sshx "$A_LAN" 'docker logs coord 2>&1 | tail -25'; exit 1; }

echo "=== lmcache servers"
start_lmc "$A_LAN" "$A_ROCE" spark-a
start_lmc "$B_LAN" "$B_ROCE" spark-b
sleep 5
for n in "$A_LAN" "$B_LAN"; do
  sshx "$n" 'docker logs lmc 2>&1 | grep -iE "coordinator|p2p|register" | tail -4' | sed "s/^/  $n| /"
done

echo "=== engines"
start_vllm "$A_LAN"
start_vllm "$B_LAN"

echo "=== store on A"
gen "$A_LAN" A
sleep 8
NB=$(sshx "$B_LAN" 'docker logs lmc 2>&1 | wc -l')

# A ships the blocks to B, so A's transmit counter is the one that must move.
IB_BEFORE=$(ib_xmit "$A_LAN")
ib_xmit_file "$A_LAN" "$OUT/ib-before.txt"

echo "=== request the same context on B"
gen "$B_LAN" B
sleep 5

IB_AFTER=$(ib_xmit "$A_LAN")
ib_xmit_file "$A_LAN" "$OUT/ib-after.txt"
# port_xmit_data counts 4-byte words
IB_MB=$(( (IB_AFTER - IB_BEFORE) * 4 / 1048576 ))
echo
echo "--- transport"
echo "    A port_xmit_data delta: $((IB_AFTER - IB_BEFORE)) words = ${IB_MB} MB"
ib_xmit_delta_report "$OUT/ib-before.txt" "$OUT/ib-after.txt"
if [ "$IB_MB" -gt 100 ]; then
  echo "    RDMA: the transfer crossed the RoCE fabric"
elif [ "$IB_MB" -gt 0 ]; then
  echo "    PARTIAL: some RDMA, less than the payload. Check UCX_NET_DEVICES=$UCX_DEVS"
else
  echo "    TCP: no RDMA bytes. The rate below is a TCP rate, do not label it RoCE."
fi

echo
echo "--- B lmcache activity"
sshx "$B_LAN" "docker logs lmc 2>&1 | tail -n +$((NB+1)) | grep -iE 'retained keys|retriev|p2p' | tail -6" | sed 's/^/    /'

L2=$(sshx "$B_LAN" "docker logs lmc 2>&1 | tail -n +$((NB+1)) | grep -oE '[0-9]+ L2' | tail -1 | cut -d' ' -f1")
echo
if [ "${L2:-0}" -gt 0 ] 2>/dev/null; then
  echo "  P2P CONFIRMED: B served $L2 chunks from the peer"
else
  echo "  NO P2P: B reported no L2 hits."
  echo "  Check coordinator registration above, and try L2_ADAPTER='--l2-adapter {\"type\":\"p2p\"}'"
fi

OUT=$OUT "$PY" - <<'PY'
import os, json
out=os.environ['OUT']
try:
    a=json.load(open(f'{out}/A.json')); b=json.load(open(f'{out}/B.json'))
except FileNotFoundError:
    raise SystemExit
ta,tb=a['tokens'] or [], b['tokens'] or []
n=min(len(ta),len(tb))
first=next((i for i in range(n) if ta[i]!=tb[i]), None)
print(f"  A wall {a['wall_s']}s   B wall {b['wall_s']}s")
if first is None and n:
    print(f"  output identical across nodes, {n}/{n} tokens")
else:
    print(f"  outputs diverge at token {first} of {n}")
    print("  note: cross-node numerics are not guaranteed identical, compare")
    print("  against the single-node noise floor from cache-correctness.sh")
PY

echo
echo "  artifacts in $OUT"
