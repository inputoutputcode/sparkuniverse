#!/usr/bin/env bash
# Check whether the current Super KVBM smoke deployment started the connector.
set -uo pipefail

HEAD_NODE=${HEAD_NODE:-10.0.0.11}
NODES=${NODES:-"10.0.0.11 10.0.0.12"}
NS=${NS:-dynamo-system}
IMG=${IMG:-dynamo-vllm-lmcache:1.3.0-lmc052-arm64}
KC='export KUBECONFIG=/etc/rancher/k3s/k3s.yaml'

sshq() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$@"; }

echo "=== image module check"
for ip in $NODES; do
  echo "--- $ip"
  sshq "$ip" "docker run --rm --entrypoint python3 $IMG -c '
import importlib
for m in [\"kvbm\", \"kvbm.vllm_integration.connector\"]:
    try:
        importlib.import_module(m)
        print(m, \"OK\")
    except Exception as e:
        print(m, \"FAIL\", repr(e))
'" 2>&1 | sed 's/^/  /'
done

echo
echo "=== pods"
sshq "$HEAD_NODE" "$KC; kubectl -n $NS get pods -o wide | grep -E 'super-kvbm-smoke|VllmDecodeWorker|frontend' || true"

echo
echo "=== pod state"
sshq "$HEAD_NODE" "$KC; kubectl -n $NS get pods -o name | grep -i 'super-kvbm-smoke.*vllmdecodeworker' || true" | while read -r pod; do
  [ -n "$pod" ] || continue
  echo "--- $pod"
  sshq "$HEAD_NODE" "$KC; kubectl -n $NS describe ${pod#pod/} 2>/dev/null" \
    | grep -E 'State:|Last State:|Reason:|Exit Code:|Started:|Finished:|Ready:|Restart Count:|Readiness|Startup|Liveness|Warning|Failed|OOM|Back-off' \
    | sed 's/^/  /'
done

echo
echo "=== worker env"
pods=$(sshq "$HEAD_NODE" "$KC; kubectl -n $NS get pods -o name | grep -i 'super-kvbm-smoke.*vllmdecodeworker' || true")
if [ -z "$pods" ]; then
  echo "  no super-kvbm-smoke worker pods found"
fi
echo "$pods" | while read -r pod; do
  [ -n "$pod" ] || continue
  echo "--- $pod"
  sshq "$HEAD_NODE" "$KC; kubectl -n $NS exec ${pod#pod/} -- sh -lc '
    printenv | grep -E \"^(DYN_KVBM|KVBM|UCX|NIXL|HF_HOME|HF_HUB_OFFLINE)\" | sort
    echo disk_dir=\${DYN_KVBM_DISK_CACHE_DIR:-}
    df -hT \${DYN_KVBM_DISK_CACHE_DIR:-/kvbm-disk} 2>/dev/null || true
    du -sh \${DYN_KVBM_DISK_CACHE_DIR:-/kvbm-disk} 2>/dev/null || true
  '" 2>/dev/null | sed 's/^/  /'
done

echo
echo "=== worker logs"
if [ -n "$pods" ]; then
  echo "$pods" | while read -r pod; do
    [ -n "$pod" ] || continue
    echo "--- $pod"
    sshq "$HEAD_NODE" "$KC; kubectl -n $NS logs ${pod#pod/} --tail=1200 2>/dev/null" \
      | grep -iE 'kvbm|DynamoConnector|GPU KV cache size|Available KV cache|cache stats|offload|onboard|disk|traceback|exception|error|failed|killed|oom|assert|ValueError|RuntimeError' \
      | tail -80
    echo "--- $pod previous"
    sshq "$HEAD_NODE" "$KC; kubectl -n $NS logs ${pod#pod/} --previous --tail=1200 2>/dev/null" \
      | grep -iE 'kvbm|DynamoConnector|GPU KV cache size|Available KV cache|cache stats|offload|onboard|disk|traceback|exception|error|failed|killed|oom|assert|ValueError|RuntimeError' \
      | tail -120
  done
fi

echo
echo "=== recent events"
sshq "$HEAD_NODE" "$KC; kubectl -n $NS get events --sort-by=.lastTimestamp 2>/dev/null | grep -E 'super-kvbm-smoke|Failed|BackOff|Unhealthy|Killing|OOM' | tail -40 || true"

echo
echo "=== metrics endpoints"
for ip in $NODES; do
  echo "--- $ip"
  for port in 9600 9090; do
    echo "  port $port"
    sshq "$ip" "curl -fsS -m 5 http://127.0.0.1:$port/metrics 2>/dev/null | grep -iE 'kvbm|cache|offload|onboard|prefix' | head -60 || true" \
      | sed 's/^/    /'
  done
done
