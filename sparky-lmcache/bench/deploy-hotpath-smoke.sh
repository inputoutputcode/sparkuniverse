#!/usr/bin/env bash
# Render and apply the hot-path smoke manifest as ~/arm.yaml on the k3s head.
#
# Examples:
#   ./deploy-hotpath-smoke.sh
#   L2_ADAPTER_MODE=fs_native_shared ./deploy-hotpath-smoke.sh
#   L2_ADAPTER_MODE=nixl_posix_dynamic_shared COORD_EVENT_REPORTING=1 ./deploy-hotpath-smoke.sh
#   L2_ADAPTER_MODE=resp RESP_HOST=10.0.0.11 RESP_PORT=6379 ./deploy-hotpath-smoke.sh
set -Eeuo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
  shift
fi

REPO=$(cd "$(dirname "$0")/.." && pwd)
SRC=${SRC:-$REPO/deploy/nano-hotpath-smoke.yaml}
HEAD_NODE=${HEAD_NODE:-10.0.0.11}
KC='export KUBECONFIG=/etc/rancher/k3s/k3s.yaml'
TMP=${TMP:-/tmp/arm-hotpath-smoke.yaml}

PY=${PY:-python3}
command -v "$PY" >/dev/null 2>&1 || { echo "!! no Python found at $PY"; exit 1; }

keys=(
  L1_SIZE_GB P2P_ENABLED P2P_PORT L2_PREFETCH_POLICY EVICTION_WATERMARK
  EVICTION_RATIO L2_ADAPTER_MODE L2_PATH L2_NUM_WORKERS L2_MAX_GB RESP_HOST
  RESP_PORT COORD_EVENT_REPORTING UCX_NET_DEVICES UCX_MAX_RNDV_RAILS
  UCX_CM_USE_ALL_DEVICES COORDINATOR_URL
)

args=()
for k in "${keys[@]}"; do
  if [ "${!k+x}" = x ]; then
    args+=("$k=${!k}")
  fi
done

if [ "${#args[@]}" -gt 0 ]; then
  render_argv=("$SRC" "$TMP" "${args[@]}")
else
  render_argv=("$SRC" "$TMP")
fi

"$PY" - "${render_argv[@]}" <<'PY'
import re
import sys
src, dst, *pairs = sys.argv[1:]
over = dict(p.split("=", 1) for p in pairs)
lines = open(src).read().splitlines()
seen = set()
i = 0
while i < len(lines):
    m = re.match(r"^(\s*)-\s+name:\s+([A-Z0-9_]+)\s*$", lines[i])
    if not m or m.group(2) not in over:
        i += 1
        continue
    key = m.group(2)
    indent = m.group(1)
    j = i + 1
    while j < len(lines):
        if re.match(rf"^{re.escape(indent)}-\s+name:\s+", lines[j]):
            break
        if re.match(r"^\s*value:\s*", lines[j]):
            escaped = over[key].replace("\\", "\\\\").replace('"', '\\"')
            lines[j] = re.sub(r"^(\s*)value:.*$", rf'\1value: "{escaped}"', lines[j])
            seen.add(key)
            break
        j += 1
    i += 1
missing = set(over) - seen
if missing:
    print(f"!! env not present in manifest: {sorted(missing)}")
    raise SystemExit(2)
open(dst, "w").write("\n".join(lines) + "\n")
print(f"wrote {dst}")
if over:
    print("overrides:")
    for k in sorted(over):
        print(f"  {k}={over[k]}")
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
      -l nvidia.com/dynamo-component=VllmDecodeWorker --timeout=240s >/dev/null 2>&1 || true"
fi

echo "applying ~/arm.yaml"
ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$HEAD_NODE" "$KC; kubectl apply -f ~/arm.yaml"

echo "waiting for frontend + two workers"
for i in $(seq 90); do
  ready=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$HEAD_NODE" "$KC
    kubectl -n dynamo-system get pods --no-headers 2>/dev/null \
      | grep -E 'nano-hotpath-smoke|lmcache-coordinator' \
      | awk '\$3==\"Running\" {split(\$2,a,\"/\"); if (a[1]==a[2] && a[2]>0) n++} END {print n+0}'" \
    2>/dev/null || echo 0)
  [ "${ready:-0}" -ge 4 ] && { echo "ready after $((i*10))s"; exit 0; }
  [ $((i % 6)) -eq 0 ] && echo "  waiting, $ready/4 ready at $((i*10))s"
  sleep 10
done

echo "!! pods not ready"
ssh -n "$HEAD_NODE" "$KC; kubectl -n dynamo-system get pods -o wide | grep -E 'nano-hotpath-smoke|lmcache-coordinator' || true"
exit 1
