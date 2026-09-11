#!/usr/bin/env bash
# Deploy Super with LMCache local fs_native L2 for quick smoke validation.
#
# Defaults are intentionally canary-sized:
#   - P2P off
#   - 20 GB vLLM device KV arena
#   - 400 GB local fs_native L2
#   - 4 GiB LMCache L1 staging tier
#   - max-num-seqs 4 to avoid wedging the engine while validating L2
#   - two workers unless WORKER_REPLICAS is overridden
set -Eeuo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
  shift
fi

REPO=$(cd "$(dirname "$0")/.." && pwd)
SRC=${SRC:-$REPO/deploy/super-agg-p2p.yaml}
HEAD_NODE=${HEAD_NODE:-10.0.0.11}
KC='export KUBECONFIG=/etc/rancher/k3s/k3s.yaml'
TMP=${TMP:-/tmp/arm-super-l2-local.yaml}
PY=${PY:-python3}

P2P_ENABLED=${P2P_ENABLED:-0}
L1_SIZE_GB=${L1_SIZE_GB:-4}
L2_ADAPTER_MODE=${L2_ADAPTER_MODE:-fs_native}
L2_MAX_GB=${L2_MAX_GB:-400}
L2_NUM_WORKERS=${L2_NUM_WORKERS:-32}
L2_PREFETCH_POLICY=${L2_PREFETCH_POLICY:-default}
L2_PREFETCH_MAX_IN_FLIGHT=${L2_PREFETCH_MAX_IN_FLIGHT:-8}
L2_USE_ODIRECT=${L2_USE_ODIRECT:-true}
L2_STORE_POLICY=${L2_STORE_POLICY:-}
EVICTION_POLICY=${EVICTION_POLICY:-LRU}
UCX_NET_DEVICES=${UCX_NET_DEVICES:-rocep1s0f1:1,roceP2p1s0f1:1}
UCX_MAX_RNDV_RAILS=${UCX_MAX_RNDV_RAILS:-2}
UCX_CM_USE_ALL_DEVICES=${UCX_CM_USE_ALL_DEVICES:-y}
KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-20000000000}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
WORKER_REPLICAS=${WORKER_REPLICAS:-2}

args=(
  "P2P_ENABLED=$P2P_ENABLED"
  "L1_SIZE_GB=$L1_SIZE_GB"
  "L2_ADAPTER_MODE=$L2_ADAPTER_MODE"
  "L2_MAX_GB=$L2_MAX_GB"
  "L2_NUM_WORKERS=$L2_NUM_WORKERS"
  "L2_PREFETCH_POLICY=$L2_PREFETCH_POLICY"
  "L2_PREFETCH_MAX_IN_FLIGHT=$L2_PREFETCH_MAX_IN_FLIGHT"
  "L2_USE_ODIRECT=$L2_USE_ODIRECT"
  "L2_STORE_POLICY=$L2_STORE_POLICY"
  "EVICTION_POLICY=$EVICTION_POLICY"
  "UCX_NET_DEVICES=$UCX_NET_DEVICES"
  "UCX_MAX_RNDV_RAILS=$UCX_MAX_RNDV_RAILS"
  "UCX_CM_USE_ALL_DEVICES=$UCX_CM_USE_ALL_DEVICES"
  "__ARG__--kv-cache-memory-bytes=$KV_CACHE_MEMORY_BYTES"
  "__ARG__--max-num-seqs=$MAX_NUM_SEQS"
  "__REPLICAS__VllmDecodeWorker=$WORKER_REPLICAS"
)

"$PY" - "$SRC" "$TMP" "${args[@]}" <<'PY'
import re
import sys
src, dst, *pairs = sys.argv[1:]
env_over = {}
arg_over = {}
replica_over = {}
for p in pairs:
    k, v = p.split("=", 1)
    if k.startswith("__ARG__"):
        arg_over[k[len("__ARG__"):]] = v
    elif k.startswith("__REPLICAS__"):
        replica_over[k[len("__REPLICAS__"):]] = v
    else:
        env_over[k] = v

lines = open(src).read().splitlines()
seen_env = set()
i = 0
while i < len(lines):
    m = re.match(r"^(\s*)-\s+name:\s+([A-Z0-9_]+)\s*$", lines[i])
    if not m or m.group(2) not in env_over:
        i += 1
        continue
    key = m.group(2)
    indent = m.group(1)
    end = i + 1
    while end < len(lines):
        if re.match(rf"^{re.escape(indent)}-\s+name:\s+", lines[end]):
            break
        end += 1
    j = i + 1
    while j < end:
        if re.match(r"^\s*value:\s*", lines[j]):
            escaped = env_over[key].replace("\\", "\\\\").replace('"', '\\"')
            lines[j] = re.sub(r"^(\s*)value:.*$", rf'\1value: "{escaped}"', lines[j])
            seen_env.add(key)
            break
        j += 1
    i += 1

seen_arg = set()
for flag, value in arg_over.items():
    for i, line in enumerate(lines[:-1]):
        if re.match(rf"^\s*-\s+{re.escape(flag)}\s*$", line):
            indent = re.match(r"^(\s*)-", line).group(1)
            lines[i + 1] = f'{indent}- "{value}"'
            seen_arg.add(flag)
            break

seen_replica = set()
for component, value in replica_over.items():
    in_component = False
    component_indent = None
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)-\s+name:\s+(.+?)\s*$", line)
        if m and m.group(2) == component:
            in_component = True
            component_indent = m.group(1)
            continue
        if in_component:
            if re.match(rf"^{re.escape(component_indent)}-\s+name:\s+", line):
                break
            m_rep = re.match(r"^(\s*)replicas:\s+\d+\s*$", line)
            if m_rep:
                lines[i] = f"{m_rep.group(1)}replicas: {value}"
                seen_replica.add(component)
                break

missing = (set(env_over) - seen_env) | (set(arg_over) - seen_arg) | (set(replica_over) - seen_replica)
if missing:
    print(f"!! requested override not found in manifest: {sorted(missing)}")
    raise SystemExit(2)

open(dst, "w").write("\n".join(lines) + "\n")
print(f"wrote {dst}")
for k in sorted(env_over):
    print(f"  {k}={env_over[k]}")
for k in sorted(arg_over):
    print(f"  {k}={arg_over[k]}")
for k in sorted(replica_over):
    print(f"  {k}.replicas={replica_over[k]}")
PY

if [ "$DRY_RUN" = 1 ]; then
  echo "dry-run: not copying or applying"
  exit 0
fi

echo "copying to $HEAD_NODE:~/arm.yaml"
scp -q "$TMP" "$HEAD_NODE:~/arm.yaml"

if [ "${SKIP_TEARDOWN:-0}" != 1 ]; then
  echo "tearing down existing DynamoGraphDeployments"
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$HEAD_NODE" "$KC
    kubectl -n dynamo-system delete dgd --all --ignore-not-found
    kubectl -n dynamo-system delete deploy lmcache-coordinator --ignore-not-found
    kubectl -n dynamo-system wait --for=delete pod \
      -l nvidia.com/dynamo-component=VllmDecodeWorker --timeout=300s >/dev/null 2>&1 || true"
fi

echo "applying ~/arm.yaml"
ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$HEAD_NODE" "$KC; kubectl apply -f ~/arm.yaml"

expected_ready=$((WORKER_REPLICAS + 2))
echo "waiting for frontend + $WORKER_REPLICAS Super worker(s) + coordinator"
for i in $(seq 150); do
  ready=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$HEAD_NODE" "$KC
    kubectl -n dynamo-system get pods --no-headers 2>/dev/null \
      | grep -E 'super-agg-p2p|lmcache-coordinator' \
      | awk '\$3==\"Running\" {split(\$2,a,\"/\"); if (a[1]==a[2] && a[2]>0) n++} END {print n+0}'" \
    2>/dev/null || echo 0)
  [ "${ready:-0}" -ge "$expected_ready" ] && { echo "ready after $((i*10))s"; exit 0; }
  [ $((i % 6)) -eq 0 ] && echo "  waiting, $ready/$expected_ready ready at $((i*10))s"
  sleep 10
done

echo "!! pods not ready"
ssh -n "$HEAD_NODE" "$KC; kubectl -n dynamo-system get pods -o wide | grep -E 'super-agg-p2p|lmcache-coordinator' || true"
ssh -n "$HEAD_NODE" "$KC; kubectl -n dynamo-system logs -l nvidia.com/dynamo-component=VllmDecodeWorker --tail=240 2>/dev/null | grep -iE 'lmcache|error|traceback|oom|killed|No GPU context' || true"
exit 1
