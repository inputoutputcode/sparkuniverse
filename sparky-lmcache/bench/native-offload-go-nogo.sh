#!/usr/bin/env bash
# Reuse the KVBM request-pressure test against the native offload endpoint.
set -Eeuo pipefail
ENDPOINT=${ENDPOINT:-http://10.0.0.11:30807}
OUT=${OUT:-/tmp/native-offload-go-nogo}
exec "$(dirname "$0")/kvbm-go-nogo.sh"
