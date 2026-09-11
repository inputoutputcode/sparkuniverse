#!/usr/bin/env bash
# Inspect the currently deployed Kubernetes arm for NIXL/P2P/RDMA evidence.
set -uo pipefail

HEAD_NODE=${HEAD_NODE:-10.0.0.11}
NODES=${NODES:-"10.0.0.11 10.0.0.12"}
NS=${NS:-dynamo-system}
KC='export KUBECONFIG=/etc/rancher/k3s/k3s.yaml'

sshq() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$@"; }

echo "=== worker pods"
sshq "$HEAD_NODE" "$KC; kubectl -n $NS get pods -o wide | grep -E 'decodeworker|nano-hotpath-smoke|nano-agg-p2p' || true"

echo
echo "=== pod env/resources that matter"
sshq "$HEAD_NODE" "$KC; kubectl -n $NS get pods -o name | grep -i decodeworker" | while read -r pod; do
  [ -n "$pod" ] || continue
  echo "--- $pod"
  sshq "$HEAD_NODE" "$KC; kubectl -n $NS exec ${pod#pod/} -- sh -lc '
    echo NODE_IP=\${NODE_IP:-}
    echo UCX_NET_DEVICES=\${UCX_NET_DEVICES:-}
    echo UCX_MAX_RNDV_RAILS=\${UCX_MAX_RNDV_RAILS:-}
    echo UCX_CM_USE_ALL_DEVICES=\${UCX_CM_USE_ALL_DEVICES:-}
    echo P2P_PORT=\${P2P_PORT:-}
    echo L2_ADAPTER_MODE=\${L2_ADAPTER_MODE:-unset}
    ls -l /dev/infiniband 2>/dev/null || echo no /dev/infiniband
    printenv | grep -E \"^(NIXL|UCX|VLLM_NIXL|LMCACHE)\" | sort || true
  '" 2>/dev/null | sed 's/^/  /'
done

echo
echo "=== lmcache logs"
sshq "$HEAD_NODE" "$KC; kubectl -n $NS logs -l nvidia.com/dynamo-component=VllmDecodeWorker --tail=800 2>/dev/null" \
  | grep -iE 'hotpath smoke|P2P on|p2p-transfer-engine|transfer engine|nixl|ucx|Registered with coordinator|Added L2 adapter|rdma|uverbs|coordinator' \
  | tail -80

echo
echo "=== lmcache status endpoints"
for ip in $NODES; do
  echo "--- $ip"
  sshq "$ip" "curl -fsS -m 5 http://127.0.0.1:9500/status 2>/dev/null | python3 -m json.tool 2>/dev/null || curl -fsS -m 5 http://127.0.0.1:9500/status 2>/dev/null || true" \
    | grep -E 'p2p|instance|adapter|coordinator|peer|state|l2' \
    | sed 's/^/  /'
done

echo
echo "=== rdma counters"
for ip in $NODES; do
  echo "--- $ip"
  sshq "$ip" 'for d in rocep1s0f1 roceP2p1s0f1; do
    x=/sys/class/infiniband/$d/ports/1/counters/port_xmit_data
    r=/sys/class/infiniband/$d/ports/1/counters/port_rcv_data
    [ -r "$x" ] && printf "%s port_xmit_data=%s port_rcv_data=%s\n" "$d" "$(cat "$x")" "$(cat "$r")"
  done' | sed 's/^/  /'
done

echo
echo "Interpretation:"
echo "  NIXL intended: logs show P2P on and transfer engine nixl."
echo "  RDMA possible: pod has /dev/infiniband and UCX_NET_DEVICES names visible HCAs."
echo "  Dual rail intended: UCX_NET_DEVICES includes rocep1s0f1:1,roceP2p1s0f1:1 and UCX_MAX_RNDV_RAILS=2."
echo "  Dual rail proven: run hotpath-smoke.sh and see both port_xmit_data/port_rcv_data counters move during the peer-hit stage."
