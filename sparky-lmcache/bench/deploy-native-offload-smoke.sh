#!/usr/bin/env bash
# Render/apply the Super native OffloadingConnector smoke manifest.
set -Eeuo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then DRY_RUN=1; shift; fi

REPO=$(cd "$(dirname "$0")/.." && pwd)
SRC=${SRC:-$REPO/deploy/super-native-offload-smoke.yaml}
HEAD_NODE=${HEAD_NODE:-10.0.0.11}
KC='export KUBECONFIG=/etc/rancher/k3s/k3s.yaml'
TMP=${TMP:-/tmp/arm-super-native-offload-smoke.yaml}
PY=${PY:-python3}

keys=(PYTHONHASHSEED UCX_NET_DEVICES UCX_MAX_RNDV_RAILS UCX_CM_USE_ALL_DEVICES VLLM_SSM_CONV_STATE_LAYOUT HF_HUB_OFFLINE)
args=()
for k in "${keys[@]}"; do
  [ "${!k+x}" = x ] && args+=("$k=${!k}")
done
[ "${KV_CACHE_MEMORY_BYTES+x}" = x ] && args+=("__ARG__--kv-cache-memory-bytes=$KV_CACHE_MEMORY_BYTES")
[ "${MAX_NUM_SEQS+x}" = x ] && args+=("__ARG__--max-num-seqs=$MAX_NUM_SEQS")
[ "${CPU_BYTES_TO_USE+x}" = x ] && args+=("__JSON__cpu_bytes_to_use=$CPU_BYTES_TO_USE")
[ "${OFFLOAD_BLOCK_SIZE+x}" = x ] && args+=("__JSON__block_size=$OFFLOAD_BLOCK_SIZE")

if [ "${#args[@]}" -gt 0 ]; then
  render_argv=("$SRC" "$TMP" "${args[@]}")
else
  render_argv=("$SRC" "$TMP")
fi

"$PY" - "${render_argv[@]}" <<'PY'
import json, re, sys
src, dst, *pairs = sys.argv[1:]
env_over, arg_over, json_over = {}, {}, {}
for p in pairs:
    k, v = p.split("=", 1)
    if k.startswith("__ARG__"):
        arg_over[k[len("__ARG__"):]] = v
    elif k.startswith("__JSON__"):
        json_over[k[len("__JSON__"):]] = int(v)
    else:
        env_over[k] = v
lines = open(src).read().splitlines()

def quote(v):
    return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'

seen_env = set()
i = 0
while i < len(lines):
    m = re.match(r"^(\s*)-\s+name:\s+([A-Z0-9_]+)\s*$", lines[i])
    if not m or m.group(2) not in env_over:
        i += 1
        continue
    key, indent = m.group(2), m.group(1)
    end = i + 1
    while end < len(lines) and not re.match(rf"^{re.escape(indent)}-\s+name:\s+", lines[end]):
        end += 1
    for j in range(i + 1, end):
        if re.match(r"^\s*value:\s*", lines[j]):
            lines[j] = re.sub(r"^(\s*)value:.*$", rf"\1value: {quote(env_over[key])}", lines[j])
            seen_env.add(key)
            break
    i += 1

seen_arg = set()
for flag, value in arg_over.items():
    for i, line in enumerate(lines[:-1]):
        if re.match(rf"^\s*-\s+{re.escape(flag)}\s*$", line):
            indent = re.match(r"^(\s*)-", line).group(1)
            lines[i + 1] = f'{indent}- "{value}"'
            seen_arg.add(flag)
            break

if json_over:
    for i, line in enumerate(lines[:-1]):
        if re.match(r"^\s*-\s+--kv-transfer-config\s*$", line):
            cfg = json.loads(lines[i + 1].split("- ", 1)[1].strip().strip("'"))
            cfg["kv_connector_extra_config"].update(json_over)
            indent = re.match(r"^(\s*)-", lines[i + 1]).group(1)
            lines[i + 1] = f"{indent}- '{json.dumps(cfg, separators=(',', ':'))}'"
            break

missing = (set(env_over) - seen_env) | (set(arg_over) - seen_arg)
if missing:
    print(f"!! requested override not found: {sorted(missing)}")
    raise SystemExit(2)
open(dst, "w").write("\n".join(lines) + "\n")
print(f"wrote {dst}")
for k in sorted(env_over): print(f"  {k}={env_over[k]}")
for k in sorted(arg_over): print(f"  {k}={arg_over[k]}")
for k in sorted(json_over): print(f"  {k}={json_over[k]}")
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

echo "waiting for frontend + two Super workers"
for i in $(seq 150); do
  ready=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$HEAD_NODE" "$KC
    kubectl -n dynamo-system get pods --no-headers 2>/dev/null \
      | grep -E 'super-native-offload-smoke' \
      | awk '\$3==\"Running\" {split(\$2,a,\"/\"); if (a[1]==a[2] && a[2]>0) n++} END {print n+0}'" \
    2>/dev/null || echo 0)
  [ "${ready:-0}" -ge 3 ] && { echo "ready after $((i*10))s"; exit 0; }
  [ $((i % 6)) -eq 0 ] && echo "  waiting, $ready/3 ready at $((i*10))s"
  sleep 10
done

echo "!! pods not ready"
ssh -n "$HEAD_NODE" "$KC; kubectl -n dynamo-system get pods -o wide | grep -E 'super-native-offload-smoke' || true"
exit 1
