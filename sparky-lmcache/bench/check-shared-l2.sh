#!/usr/bin/env bash
# Verify that the host path used for shared LMCache L2 is actually shared.
#
# Same path on two nodes is not enough: a Kubernetes hostPath is still local
# unless NFS/WEKA/Lustre/etc. is mounted there on both hosts.
set -uo pipefail

A=${A:-10.0.0.11}
B=${B:-10.0.0.12}
PATH_ON_HOST=${PATH_ON_HOST:-/mnt/shared-lmcache-l2}
NONCE=${NONCE:-$(date +%s)-$RANDOM}

sshq() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$@"; }

echo "=== mount identity"
for h in "$A" "$B"; do
  echo "--- $h"
  sshq "$h" "mkdir -p '$PATH_ON_HOST'
    findmnt -T '$PATH_ON_HOST' -o TARGET,SOURCE,FSTYPE,OPTIONS -n || true
    stat -f -c 'fs_type=%T fs_id=%i' '$PATH_ON_HOST' 2>/dev/null || true
    df -hT '$PATH_ON_HOST' | tail -1" | sed 's/^/  /'
done

echo
echo "=== cross-node visibility"
sshq "$A" "mkdir -p '$PATH_ON_HOST'; printf '%s\n' '$NONCE from $A' > '$PATH_ON_HOST/.lmcache-shared-test-$NONCE'"
if sshq "$B" "test -f '$PATH_ON_HOST/.lmcache-shared-test-$NONCE'"; then
  echo "  A -> B visible: yes"
else
  echo "  A -> B visible: NO"
fi

sshq "$B" "mkdir -p '$PATH_ON_HOST'; printf '%s\n' '$NONCE from $B' > '$PATH_ON_HOST/.lmcache-shared-test-$NONCE-b'"
if sshq "$A" "test -f '$PATH_ON_HOST/.lmcache-shared-test-$NONCE-b'"; then
  echo "  B -> A visible: yes"
else
  echo "  B -> A visible: NO"
fi

echo
if sshq "$A" "test -f '$PATH_ON_HOST/.lmcache-shared-test-$NONCE-b'" \
  && sshq "$B" "test -f '$PATH_ON_HOST/.lmcache-shared-test-$NONCE'"; then
  echo "SHARED_L2_OK: $PATH_ON_HOST is visible from both nodes"
  sshq "$A" "rm -f '$PATH_ON_HOST/.lmcache-shared-test-$NONCE' '$PATH_ON_HOST/.lmcache-shared-test-$NONCE-b'" >/dev/null 2>&1
  sshq "$B" "rm -f '$PATH_ON_HOST/.lmcache-shared-test-$NONCE' '$PATH_ON_HOST/.lmcache-shared-test-$NONCE-b'" >/dev/null 2>&1
  exit 0
fi

echo "SHARED_L2_NOT_SHARED: mount a real shared filesystem at $PATH_ON_HOST on both Sparks, or use RESP/remote L2."
exit 1
