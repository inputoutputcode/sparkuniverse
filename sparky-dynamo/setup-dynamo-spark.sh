#!/usr/bin/env bash
# =============================================================================
# NVIDIA Dynamo on 2x DGX Spark (K3s + ConnectX-7) — one-shot node setup
#
# Usage:
#   Server node:  sudo ./setup-dynamo-spark.sh server <CX7_IFACE> <CX7_IP>
#   Agent node:   sudo ./setup-dynamo-spark.sh agent  <CX7_IFACE> <CX7_IP> <SERVER_CX7_IP> <K3S_TOKEN>
#
# Example:
#   spark-a: sudo ./setup-dynamo-spark.sh server enp1s0f1np1 192.168.177.11
#   spark-b: sudo ./setup-dynamo-spark.sh agent  enp1s0f1np1 192.168.177.12 192.168.177.11 <token>
#
# The token is printed at the end of the server run
# (from /var/lib/rancher/k3s/server/node-token).
# =============================================================================
set -euo pipefail

ROLE="${1:?role: server|agent}"
CX7_IF="${2:?ConnectX-7 interface, e.g. enp1s0f1np1}"
CX7_IP="${3:?this node's CX-7 IP, e.g. 192.168.177.11}"
SERVER_IP="${4:-}"
K3S_TOKEN="${5:-}"
DYNAMO_VERSION="${DYNAMO_VERSION:-1.2.1}"   # match chart == repo tag == image tag

log() { echo -e "\n==> $*"; }

# -----------------------------------------------------------------------------
# Phase 0: host preparation (identical on both nodes)
# -----------------------------------------------------------------------------

log "0.1 Sanity: GPU driver and CX-7 link"
nvidia-smi -L
ip -br link show "$CX7_IF" | grep -q UP || { echo "ERROR: $CX7_IF has no link — check cable/port"; exit 1; }
ip -br addr show "$CX7_IF" | grep -q "$CX7_IP" || { echo "ERROR: $CX7_IP not on $CX7_IF — set a STATIC address in netplan first"; exit 1; }

log "0.2 inotify limits (K8s exhausts the defaults)"
tee /etc/sysctl.d/99-k8s-inotify.conf >/dev/null <<'EOF'
fs.inotify.max_user_instances = 8192
fs.inotify.max_user_watches = 1048576
EOF
sysctl --system >/dev/null

log "0.3 CDI specs: refresh service for the regular spec, management spec by hand"
# Toolkit 1.18+ regenerates /var/run/cdi/nvidia.yaml via nvidia-cdi-refresh
# (the .path unit watches for driver changes). It does NOT emit the management
# spec, which the GPU Operator validator needs when toolkit.enabled=false, so
# generate that one into persistent /etc/cdi.
systemctl enable --now nvidia-cdi-refresh.service nvidia-cdi-refresh.path 2>/dev/null || true
mkdir -p /etc/cdi
nvidia-ctk cdi generate --mode=management \
  --vendor=management.nvidia.com --class=gpu \
  --output=/etc/cdi/management.nvidia.com-gpu.yaml
nvidia-ctk cdi list | grep -q 'management.nvidia.com/gpu=all' \
  || { echo "ERROR: management CDI device missing"; exit 1; }

# NOTE: /etc/cdi persists across reboots. Re-run this generate step after a
# driver or CUDA update, since the spec embeds driver-version paths.

log "0.4 MSS clamp (pods inherit CX-7 jumbo MTU; egress path is 1500)"
iptables -t mangle -C FORWARD -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --clamp-mss-to-pmtu 2>/dev/null \
  || iptables -t mangle -A FORWARD -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --clamp-mss-to-pmtu
# iptables-persistent restores too early: K3s rebuilds netfilter state after it
# and drops the rule. Re-add after K3s starts instead.
tee /etc/systemd/system/mss-clamp.service >/dev/null <<'EOF'
[Unit]
Description=Clamp TCP MSS to PMTU for pod egress
After=k3s.service k3s-agent.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c 'iptables -t mangle -C FORWARD -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --clamp-mss-to-pmtu 2>/dev/null || iptables -t mangle -A FORWARD -p tcp --tcp-flags SYN,RST SYN -j TCPMSS --clamp-mss-to-pmtu'

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable mss-clamp.service >/dev/null

log "0.45 memlock=infinity for the k3s service (UCX/RDMA memory registration)"
# CAP_IPC_LOCK is ineffective for non-root containers, so raise the limit on
# the service that spawns them. Applied to both unit names; only one exists.
for UNIT in k3s k3s-agent; do
  mkdir -p "/etc/systemd/system/${UNIT}.service.d"
  printf '[Service]\nLimitMEMLOCK=infinity\n' > "/etc/systemd/system/${UNIT}.service.d/memlock.conf"
done
systemctl daemon-reload

log "0.5 Cache dirs: HF models (pre-seed here) + vLLM compile cache"
mkdir -p /var/lib/hf-cache /var/lib/vllm-cache
chmod -R a+rwX /var/lib/hf-cache /var/lib/vllm-cache

# -----------------------------------------------------------------------------
# Phase 1: K3s
# -----------------------------------------------------------------------------

mkdir -p /etc/rancher/k3s

if [[ "$ROLE" == "server" ]]; then
  log "1.1 K3s server config (SQLite datastore — deliberately NOT 2-node etcd)"
  tee /etc/rancher/k3s/config.yaml >/dev/null <<EOF
disable:
  - traefik      # conflicts with Dynamo ingress assumptions
  - servicelb    # fights any LoadBalancer implementation over IPs
node-ip: ${CX7_IP}
advertise-address: ${CX7_IP}
flannel-iface: ${CX7_IF}   # pod traffic on the 200G link, not the egress NIC
tls-san:
  - ${CX7_IP}
write-kubeconfig-mode: "0644"
EOF

  log "1.2 Install K3s server"
  curl -sfL https://get.k3s.io | sh -s - server
  export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
  until kubectl get node >/dev/null 2>&1; do sleep 2; done

  # ---------------------------------------------------------------------------
  # Phase 2: GPU Operator (server only — cluster-wide component)
  # ---------------------------------------------------------------------------
  log "2. GPU Operator — DGX OS already ships driver + toolkit: manage NEITHER"
  command -v helm >/dev/null || curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
  helm repo add nvidia https://helm.ngc.nvidia.com/nvidia >/dev/null 2>&1 || true
  helm repo update >/dev/null
  helm upgrade --install gpu-operator nvidia/gpu-operator \
    -n gpu-operator --create-namespace \
    --set driver.enabled=false \
    --set toolkit.enabled=false

  # ---------------------------------------------------------------------------
  # Phase 3: Dynamo platform
  # ---------------------------------------------------------------------------
  log "3. Dynamo platform ${DYNAMO_VERSION} (etcd+NATS explicitly on; Grove+KAI bundled)"
  helm fetch "https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${DYNAMO_VERSION}.tgz"
  helm upgrade --install dynamo-platform "dynamo-platform-${DYNAMO_VERSION}.tgz" \
    -n dynamo-system --create-namespace \
    --set global.etcd.install=true \
    --set global.nats.install=true \
    --set "global.grove.install=true" \
    --set "global.kai-scheduler.install=true"

  log "DONE (server). Agent join token:"
  cat /var/lib/rancher/k3s/server/node-token
  echo "Run on the second Spark:"
  echo "  sudo ./setup-dynamo-spark.sh agent <CX7_IFACE> <ITS_CX7_IP> ${CX7_IP} <token-above>"

elif [[ "$ROLE" == "agent" ]]; then
  [[ -n "$SERVER_IP" && -n "$K3S_TOKEN" ]] || { echo "agent needs SERVER_CX7_IP and K3S_TOKEN"; exit 1; }

  log "1.1 K3s agent config"
  tee /etc/rancher/k3s/config.yaml >/dev/null <<EOF
node-ip: ${CX7_IP}
flannel-iface: ${CX7_IF}
EOF

  log "1.2 Reachability check, then install K3s agent"
  curl -sk "https://${SERVER_IP}:6443/ping" | grep -q pong || { echo "ERROR: server not reachable on CX-7"; exit 1; }
  curl -sfL https://get.k3s.io | K3S_URL="https://${SERVER_IP}:6443" K3S_TOKEN="${K3S_TOKEN}" sh -s - agent

  log "DONE (agent). Verify on the server: kubectl get nodes -o wide  (INTERNAL-IP = CX-7 IPs)"
else
  echo "Unknown role: $ROLE"; exit 1
fi
