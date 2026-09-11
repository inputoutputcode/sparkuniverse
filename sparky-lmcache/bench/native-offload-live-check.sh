#!/usr/bin/env bash
# Diagnose the Super native OffloadingConnector smoke deployment.
set -uo pipefail

HEAD_NODE=${HEAD_NODE:-10.0.0.11}
NODES=${NODES:-"10.0.0.11 10.0.0.12"}
NS=${NS:-dynamo-system}
IMG=${IMG:-dynamo-vllm-lmcache:1.3.0-lmc052-arm64}
KC='export KUBECONFIG=/etc/rancher/k3s/k3s.yaml'
sshq() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$@"; }

echo "=== connector module check"
for ip in $NODES; do
  echo "--- $ip"
  sshq "$ip" "docker run --rm --entrypoint python3 $IMG -c '
import importlib
for m in [\"vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector\"]:
    try:
        importlib.import_module(m); print(m, \"OK\")
    except Exception as e:
        print(m, \"FAIL\", repr(e))
'" 2>&1 | sed 's/^/  /'
done

echo
echo "=== pods"
sshq "$HEAD_NODE" "$KC; kubectl -n $NS get pods -o wide | grep -E 'super-native-offload-smoke' || true"

pods=$(sshq "$HEAD_NODE" "$KC; kubectl -n $NS get pods -o name | grep -i 'super-native-offload-smoke.*vllmdecodeworker' || true")

echo
echo "=== worker env"
echo "$pods" | while read -r pod; do
  [ -n "$pod" ] || continue
  echo "--- $pod"
  sshq "$HEAD_NODE" "$KC; kubectl -n $NS exec ${pod#pod/} -- sh -lc '
    printenv | grep -E \"^(PYTHONHASHSEED|VLLM|UCX|NIXL|HF_HOME|HF_HUB_OFFLINE)\" | sort
  '" 2>/dev/null | sed 's/^/  /'
done

echo
echo "=== worker logs"
echo "$pods" | while read -r pod; do
  [ -n "$pod" ] || continue
  echo "--- $pod"
  sshq "$HEAD_NODE" "$KC; kubectl -n $NS logs ${pod#pod/} --tail=1400 2>/dev/null" \
    | grep -iE 'OffloadingConnector|offload|self_describing|kv-events|Hybrid KV|HMA|GPU KV cache size|CPU|traceback|exception|error|failed|killed|oom|VLLM_SSM_CONV_STATE_LAYOUT' \
    | tail -120
  echo "--- $pod previous"
  sshq "$HEAD_NODE" "$KC; kubectl -n $NS logs ${pod#pod/} --previous --tail=1400 2>/dev/null" \
    | grep -iE 'OffloadingConnector|offload|self_describing|kv-events|Hybrid KV|HMA|GPU KV cache size|CPU|traceback|exception|error|failed|killed|oom|VLLM_SSM_CONV_STATE_LAYOUT' \
    | tail -160
done

echo
echo "=== metrics"
for ip in $NODES; do
  echo "--- $ip"
  sshq "$ip" "curl -fsS -m 5 http://127.0.0.1:9090/metrics 2>/dev/null | grep -iE 'offload|prefix|cache|kv_cache_events|prompt_tokens|computed_tokens' | head -100 || true" \
    | sed 's/^/  /'
done
