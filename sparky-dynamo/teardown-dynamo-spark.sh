#!/usr/bin/env bash
# =============================================================================
# Teardown for the "Dynamo on 2x DGX Spark" setup.
# Run on the AGENT node first, then on the SERVER node.
#
#   sudo ./teardown-dynamo-spark.sh            # keep model and compile caches
#   sudo ./teardown-dynamo-spark.sh --caches   # also delete /var/lib/*-cache
#
# Note: the vllm-runtime image lives in K3s's own image store and goes away
# with the uninstall, so the rerun pulls it again (~13 GB per node).
#
# NOT touched: ConnectX-7 netplan config, NVIDIA driver, container toolkit.
# Those are prerequisites (NVIDIA "Connect Two Sparks" playbook + DGX OS).
# =============================================================================
set -uo pipefail

WIPE_CACHES="${1:-}"
log() { echo -e "\n==> $*"; }

log "1. Uninstall K3s (removes cluster, embedded containerd, images, CNI)"
if [ -x /usr/local/bin/k3s-agent-uninstall.sh ]; then
  /usr/local/bin/k3s-agent-uninstall.sh
elif [ -x /usr/local/bin/k3s-uninstall.sh ]; then
  /usr/local/bin/k3s-uninstall.sh
else
  echo "no k3s uninstall script found, skipping"
fi
rm -rf /etc/rancher/k3s /var/lib/rancher/k3s

log "2. Remove systemd units and drop-ins added by the setup"
systemctl disable --now mss-clamp.service 2>/dev/null
systemctl disable --now nvidia-cdi-management.service 2>/dev/null
rm -f /etc/systemd/system/mss-clamp.service \
      /etc/systemd/system/nvidia-cdi-management.service
rm -rf /etc/systemd/system/k3s.service.d /etc/systemd/system/k3s-agent.service.d
systemctl daemon-reload

log "3. Remove host tuning"
rm -f /etc/sysctl.d/99-k8s-inotify.conf
sysctl --system >/dev/null
iptables -t mangle -D FORWARD -p tcp --tcp-flags SYN,RST SYN \
  -j TCPMSS --clamp-mss-to-pmtu 2>/dev/null

log "4. Remove the CDI management spec (regenerated in Part 1.2)"
rm -f /etc/cdi/management.nvidia.com-gpu.yaml

if [ "$WIPE_CACHES" = "--caches" ]; then
  log "5. Remove model and compile caches (forces a fresh download)"
  rm -rf /var/lib/hf-cache /var/lib/vllm-cache
else
  log "5. Keeping /var/lib/hf-cache and /var/lib/vllm-cache (pass --caches to wipe)"
fi

log "Done. Verify the CX-7 link is still up before re-running the setup:"
ip -br addr | grep 192.168.177 || echo "WARNING: no 192.168.177.x address found"
