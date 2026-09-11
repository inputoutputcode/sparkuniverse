#!/usr/bin/env bash
# Render and apply the Super KVBM smoke manifest as ~/arm.yaml.
#
# Examples:
#   ./bench/deploy-kvbm-smoke.sh
#   DYN_KVBM_CPU_CACHE_GB=__DELETE__ DYN_KVBM_DISK_CACHE_GB=200 ./bench/deploy-kvbm-smoke.sh
#   KV_CACHE_MEMORY_BYTES=8000000000 DYN_KVBM_CPU_CACHE_GB=8 ./bench/deploy-kvbm-smoke.sh
set -Eeuo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
  shift
fi

REPO=$(cd "$(dirname "$0")/.." && pwd)
SRC=${SRC:-$REPO/deploy/super-kvbm-smoke.yaml}
HEAD_NODE=${HEAD_NODE:-10.0.0.11}
KC='export KUBECONFIG=/etc/rancher/k3s/k3s.yaml'
TMP=${TMP:-/tmp/arm-super-kvbm-smoke.yaml}
PY=${PY:-python3}

keys=(
  DYN_KVBM_CPU_CACHE_GB DYN_KVBM_DISK_CACHE_GB DYN_KVBM_DISK_CACHE_DIR
  DYN_KVBM_METRICS DYN_KVBM_METRICS_PORT DYN_KVBM_CACHE_STATS_LOG_INTERVAL_SECS
  DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER DYN_KVBM_DISK_DISABLE_O_DIRECT
  DYN_KVBM_NIXL_BACKEND_UCX DYN_KVBM_NIXL_BACKEND_POSIX UCX_NET_DEVICES
  UCX_MAX_RNDV_RAILS UCX_CM_USE_ALL_DEVICES HF_HUB_OFFLINE
)

args=()
for k in "${keys[@]}"; do
  if [ "${!k+x}" = x ]; then
    args+=("$k=${!k}")
  fi
done
if [ "${KV_CACHE_MEMORY_BYTES+x}" = x ]; then
  args+=("__ARG__--kv-cache-memory-bytes=$KV_CACHE_MEMORY_BYTES")
fi
if [ "${MAX_NUM_SEQS+x}" = x ]; then
  args+=("__ARG__--max-num-seqs=$MAX_NUM_SEQS")
fi

if [ "${#args[@]}" -gt 0 ]; then
  render_argv=("$SRC" "$TMP" "${args[@]}")
else
  render_argv=("$SRC" "$TMP")
fi

"$PY" - "${render_argv[@]}" <<'PY'
import re
import sys
src, dst, *pairs = sys.argv[1:]
env_over = {}
arg_over = {}
for p in pairs:
    k, v = p.split("=", 1)
    if k.startswith("__ARG__"):
        arg_over[k[len("__ARG__"):]] = v
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
    if env_over[key] in ("__DELETE__", "__UNSET__"):
        del lines[i:end]
        seen_env.add(key)
        continue
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

missing = (set(env_over) - seen_env) | (set(arg_over) - seen_arg)
if missing:
    print(f"!! requested override not found in manifest: {sorted(missing)}")
    raise SystemExit(2)

open(dst, "w").write("\n".join(lines) + "\n")
print(f"wrote {dst}")
for k in sorted(env_over):
    print(f"  {k}={env_over[k]}")
for k in sorted(arg_over):
    print(f"  {k}={arg_over[k]}")
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
      | grep -E 'super-kvbm-smoke' \
      | awk '\$3==\"Running\" {split(\$2,a,\"/\"); if (a[1]==a[2] && a[2]>0) n++} END {print n+0}'" \
    2>/dev/null || echo 0)
  [ "${ready:-0}" -ge 3 ] && { echo "ready after $((i*10))s"; exit 0; }
  [ $((i % 6)) -eq 0 ] && echo "  waiting, $ready/3 ready at $((i*10))s"
  sleep 10
done

echo "!! pods not ready"
ssh -n "$HEAD_NODE" "$KC; kubectl -n dynamo-system get pods -o wide | grep -E 'super-kvbm-smoke' || true"
ssh -n "$HEAD_NODE" "$KC; kubectl -n dynamo-system logs -l nvidia.com/dynamo-component=VllmDecodeWorker --tail=200 2>/dev/null | grep -iE 'kvbm|dynamo|error|traceback|oom|killed' || true"
exit 1
