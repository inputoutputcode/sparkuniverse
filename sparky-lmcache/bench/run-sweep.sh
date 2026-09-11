#!/usr/bin/env bash
# Run several arms back to back, unattended.
#
#   ./run-sweep.sh sweeps/nano-abc.conf
#
# Each line of the conf is one arm:
#
#   run_id | manifest | ENV=VAL ENV=VAL ...
#
# Blank lines and # comments are ignored. For each arm the script patches the
# worker env in a copy of the manifest, tears the old deployment down, brings
# the new one up, **verifies it is the arm it claims to be**, then runs the
# benchmark.
#
# The verification is the point. Three hours of a wrongly configured deployment
# produces a complete, plausible, wrong result, and this project has already
# lost runs to a silently stripped flag, an L1 that never started, and a loader
# that replayed 85 requests instead of 3,485. An arm that fails its checks is
# skipped, not run.
#
# Failures do not stop the sweep. A bad arm at 2am should not cost the three
# that would have run after it.
set -uo pipefail

CONF=${1:?path to a sweep conf, e.g. sweeps/nano-abc.conf}
REPO=$(cd "$(dirname "$0")/.." && pwd)
HEAD_NODE=${HEAD_NODE:-10.0.0.11}
NODES=${NODES:-"10.0.0.11 10.0.0.12"}
ENDPOINT=${ENDPOINT:-http://localhost:8000}
TRACE=${TRACE:-$REPO/traces/data/cc-weka-s85-128k/}
CONCURRENCY=${CONCURRENCY:-10}
KC='export KUBECONFIG=/etc/rancher/k3s/k3s.yaml'
SWEEP_LOG=$REPO/runs/sweep-$(date +%Y%m%d-%H%M).log

mkdir -p "$REPO/runs"
exec > >(tee -a "$SWEEP_LOG") 2>&1
echo "sweep started $(date -u +%FT%TZ), log $SWEEP_LOG"

# One sweep at a time. Two concurrent sweeps would fight over the same single
# GPU per node and both would fail in ways that look like engine bugs.
#
# A lock directory, not pgrep. `pgrep -f run-sweep.sh` also matches the
# `caffeinate -i ./run-sweep.sh ...` wrapper, because caffeinate carries the
# script name in its own command line, so the guard refused to start under the
# one invocation this is meant to be run with. mkdir is atomic and does not
# care what wraps the process.
LOCK=${SWEEP_LOCK:-/tmp/run-sweep.lock}
if ! mkdir "$LOCK" 2>/dev/null; then
  # kill -9 cannot run the trap, so a killed sweep leaves the lock behind with
  # its pid file still in it, and `rmdir` then refuses because the directory is
  # not empty. Rather than require the operator to know that, check whether the
  # holder is alive and reclaim the lock if it is not.
  holder=$(cat "$LOCK/pid" 2>/dev/null || echo "")
  if [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null; then
    echo "!! another sweep is running, pid $holder, holding $LOCK"
    exit 1
  fi
  echo "  clearing stale lock from pid ${holder:-unknown}, no such process"
  rm -rf "$LOCK"
  mkdir "$LOCK" || { echo "!! cannot create $LOCK"; exit 1; }
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK" 2>/dev/null' EXIT INT TERM

# "aiperf profile" rather than "aiperf", so a venv path in some unrelated
# command line does not look like a running benchmark.
if pgrep -f "aiperf profile" >/dev/null 2>&1; then
  echo "!! aiperf is already running. Finish or kill it first."
  exit 1
fi

sshq() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$@"; }

# The env patch needs PyYAML, and macOS system python does not ship it:
#   ModuleNotFoundError: No module named 'yaml'
# The aiperf venv has it because AIPerf reads YAML config. Find an interpreter
# that actually works rather than assuming python3 does, and fail here with a
# usable message rather than three identical tracebacks at 4am.
PY=""
for cand in "$HOME/aiperf/venv/bin/python" python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import yaml" >/dev/null 2>&1; then
    PY=$cand
    break
  fi
done
if [ -z "$PY" ]; then
  echo "!! no python with PyYAML found. Tried \$HOME/aiperf/venv/bin/python, python3, python."
  echo "   Install it:  python3 -m pip install --user pyyaml"
  exit 1
fi
echo "using $PY for manifest patching ($("$PY" -c 'import yaml,sys;print("PyYAML",yaml.__version__)'))"

# ---------------------------------------------------------------- teardown
teardown() {
  sshq "$HEAD_NODE" "$KC
    k3s kubectl -n dynamo-system delete dgd --all --ignore-not-found >/dev/null
    k3s kubectl -n dynamo-system delete deploy lmcache-coordinator --ignore-not-found >/dev/null
    k3s kubectl -n dynamo-system wait --for=delete pod \
      -l nvidia.com/dynamo-component=VllmDecodeWorker --timeout=240s >/dev/null 2>&1"
  # Deleting the DGD returns immediately while engines take tens of seconds to
  # release GPU memory. The next apply then fails inside init_device() with a
  # traceback that says nothing about memory. Wait for the GPU, not the API.
  for _ in $(seq 30); do
    local busy=0
    for ip in $NODES; do
      n=$(sshq "$ip" 'nvidia-smi --query-compute-apps=used_memory --format=csv,noheader | wc -l' 2>/dev/null || echo 1)
      [ "${n:-1}" -gt 0 ] && busy=1
    done
    [ "$busy" = 0 ] && return 0
    sleep 10
  done
  echo "  !! GPU still held after 300s"
  return 1
}

# ------------------------------------------------------------------ deploy
# Count READY, not STATUS.
#
# A worker reports STATUS=Running at READY=0/1 for the ~130 s it spends loading
# 19 GB of weights. Counting ' Running ' returned 3 after 23 s, so the sweep
# slept 30, ran verify_arm against engines that had not started, failed, and
# tore the deployment down at 66 s. Every arm failed the same way, and from the
# cluster side it looked like pods crash-looping.
wait_ready() {
  local want=$1 ready=0 i
  for i in $(seq 90); do
    ready=$(sshq "$HEAD_NODE" "$KC
      k3s kubectl -n dynamo-system get pods --no-headers 2>/dev/null \
        | grep -E 'decodeworker|frontend' \
        | awk '\$3==\"Running\" {split(\$2,a,\"/\"); if (a[1]==a[2] && a[2]>0) n++} END {print n+0}'" \
      2>/dev/null || echo 0)
    [ "${ready:-0}" -ge "$want" ] && { echo "    ready after $((i*10))s"; return 0; }
    [ $((i % 6)) -eq 0 ] && echo "    waiting, $ready/$want ready at $((i*10))s"
    sleep 10
  done
  return 1
}

# --------------------------------------------------------------- verify arm
# Every check answers "is this the arm the conf asked for", not "is it healthy".
# A healthy deployment of the wrong arm is the expensive failure.
verify_arm() {
  local name=$1 want_p2p=$2
  local log
  log=$(sshq "$HEAD_NODE" "$KC
    k3s kubectl -n dynamo-system logs -l nvidia.com/dynamo-component=VllmDecodeWorker \
      --tail=2000 2>/dev/null")

  local nodes_used
  nodes_used=$(sshq "$HEAD_NODE" "$KC
    k3s kubectl -n dynamo-system get pods -o wide --no-headers 2>/dev/null \
      | grep decodeworker | awk '{print \$7}' | sort -u | wc -l")
  if [ "${nodes_used:-0}" -lt 2 ]; then
    echo "  !! workers are not on two different nodes ($nodes_used). No peer exists."
    return 1
  fi

  # Here-strings, not pipes. `printf '%s' "$log" | grep -q PATTERN` fails even
  # when the pattern matches: grep -q exits on the first hit, printf dies with
  # EPIPE, and `set -o pipefail` takes printf's non-zero status. That reported
  # "entrypoint did not report P2P on" against a deployment that had said
  # exactly that, and killed a healthy arm.
  local adapters
  adapters=$(grep -c "Added L2 adapter" <<<"$log")
  case "$want_p2p" in
    1)
      grep -q "P2P on" <<<"$log" || { echo "  !! entrypoint did not report P2P on"; return 1; }
      [ "$adapters" -ge 2 ] || { echo "  !! only $adapters L2 adapters attached, want 2"; return 1; }
      ;;
    0)
      # LMCache is running with P2P disabled, so the entrypoint does print.
      grep -q "P2P OFF" <<<"$log" || { echo "  !! entrypoint did not report P2P OFF"; return 1; }
      [ "$adapters" -eq 0 ] || { echo "  !! $adapters L2 adapters attached but P2P should be off"; return 1; }
      ;;
    none)
      # The router-only arm runs no LMCache at all, so there is no entrypoint to
      # report anything. Checking for "P2P OFF" here failed a correctly deployed
      # arm on 2026-08-18: that string comes from the P2P entrypoint, and this
      # manifest uses router-only-worker-entrypoint. The right check is that
      # nothing LMCache-shaped exists.
      [ "$adapters" -eq 0 ] || {
        echo "  !! $adapters L2 adapters attached, but this arm has no LMCache"; return 1; }
      if grep -qiE "Registered with coordinator|LMCacheMPConnector" <<<"$log"; then
        echo "  !! LMCache is running in a router-only arm, wrong manifest applied"
        return 1
      fi
      echo "  no LMCache present, as expected for router-only"
      ;;
  esac

  # The arena is the one number that silently differs when
  # --gpu-memory-utilization gates it below what was asked for.
  # grep -m1 rather than `| head -1`, same EPIPE reason.
  local kv
  kv=$(grep -m1 -oiE "GPU KV cache size: *[0-9,]+" <<<"$log")
  echo "  ${kv:-GPU KV cache size not found in log}"

  curl -sf --max-time 10 "$ENDPOINT/v1/models" >/dev/null \
    || { echo "  !! endpoint not serving at $ENDPOINT"; return 1; }
  return 0
}

# ------------------------------------------------------------------- sweep
declare -a RESULTS=()
ARM_N=0

while IFS='|' read -r name manifest envs; do
  name=$(echo "${name:-}" | xargs); [ -z "$name" ] && continue
  case "$name" in \#*) continue ;; esac
  manifest=$(echo "${manifest:-}" | xargs)
  envs=$(echo "${envs:-}" | xargs)
  ARM_N=$((ARM_N + 1))

  echo
  echo "======================================================================"
  echo "arm $ARM_N: $name   manifest=$manifest   env: ${envs:-<none>}"
  echo "start $(date -u +%FT%TZ)"

  # Skip only a *complete* run. A summary written from a killed arm is still a
  # summary, and skipping on its existence alone let a 128-request fragment
  # stand in for a 3,485-request arm.
  if [ -s "$REPO/runs/$name/summary.json" ]; then
    verdict=$("$PY" - "$REPO/runs/$name/summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
got, want = d.get("requests") or 0, d.get("expected_requests")
if want is None:
    print("ok" if got else "partial 0 requests")
elif got >= 0.99 * want:
    print("ok")
else:
    print(f"partial {got:,.0f} of {want:,}")
PY
)
    if [ "$verdict" = ok ]; then
      echo "  already complete, skipping"
      RESULTS+=("$name SKIPPED already-done")
      continue
    fi
    echo "  existing summary is $verdict, rerunning"
    rm -rf "$REPO/runs/$name"
  fi

  SRC=$REPO/deploy/$manifest
  [ -f "$SRC" ] || { echo "  !! no such manifest: $SRC"; RESULTS+=("$name FAIL no-manifest"); continue; }
  TMP=/tmp/arm-$name.yaml

  # Patch the worker env in place rather than keeping one manifest per arm.
  # Divergence between near-identical manifests is how an A/B ends up differing
  # in something nobody noticed.
  "$PY" - "$SRC" "$TMP" $envs <<'PY'
import sys, yaml
src, dst, *pairs = sys.argv[1:]
docs = [d for d in yaml.safe_load_all(open(src)) if d]
over = {}
arg_over = {}
for p in pairs:
    k, v = p.split("=", 1)
    if k.startswith("__ARG__"):
        arg_over[k[len("__ARG__"):]] = v
    else:
        over[k] = v
applied = []
arg_applied = []
for d in docs:
    if d.get("kind") != "DynamoGraphDeployment":
        continue
    for c in d["spec"]["components"]:
        for ctr in c.get("podTemplate", {}).get("spec", {}).get("containers", []):
            for e in ctr.get("env", []) or []:
                if e["name"] in over:
                    e["value"] = over[e["name"]]
                    applied.append(f"{e['name']}={e['value']}")
            args = ctr.get("args", []) or []
            for flag, value in arg_over.items():
                if flag in args:
                    args[args.index(flag) + 1] = value
                    arg_applied.append(f"{flag}={value}")
missing = set(over) - {a.split("=")[0] for a in applied}
missing_args = set(arg_over) - {a.split("=")[0] for a in arg_applied}
if missing or missing_args:
    # A typo here would otherwise apply nothing and run the previous arm's
    # configuration under the new arm's name, which is unrecoverable after
    # the fact.
    if missing:
        print(f"!! env not present in manifest, nothing to override: {sorted(missing)}")
    if missing_args:
        print(f"!! arg not present in manifest, nothing to override: {sorted(missing_args)}")
    raise SystemExit(2)
with open(dst, "w") as f:
    yaml.safe_dump_all(docs, f, sort_keys=False)
print("  patched:", ", ".join(applied) or "<none>")
if arg_applied:
    print("  patched args:", ", ".join(arg_applied))

# Anything not overridden inherits the manifest, so print what will actually
# run rather than only what changed.
WATCH = ("L1_SIZE_GB", "P2P_ENABLED", "L2_PREFETCH_POLICY",
         "L2_ADAPTER_MODE", "L2_PATH", "L2_MAX_GB", "L2_NUM_WORKERS",
         "L2_STORE_POLICY",
         "EVICTION_WATERMARK", "EVICTION_RATIO")
for d in docs:
    if d.get("kind") != "DynamoGraphDeployment":
        continue
    for c in d["spec"]["components"]:
        if c["name"] != "VllmDecodeWorker":
            continue
        ctr = c["podTemplate"]["spec"]["containers"][0]
        env = {e["name"]: e.get("value") for e in ctr.get("env", []) or []}
        eff = {k: env[k] for k in WATCH if k in env}
        print("  effective:", eff or "<no LMCache env, router-only arm>")
        args = ctr.get("args", [])
        for flag in ("--kv-cache-memory-bytes", "--gpu-memory-utilization", "--max-num-seqs"):
            if flag in args:
                print(f"  {flag} {args[args.index(flag)+1]}")
PY
  [ $? -eq 0 ] || { RESULTS+=("$name FAIL env-patch"); continue; }

  # Three states, not two. `0` means LMCache is up with P2P switched off, which
  # still prints from the entrypoint. `none` means no LMCache in the manifest at
  # all, which prints nothing and must be verified by absence instead.
  WANT_P2P=1
  echo "$envs" | grep -q "P2P_ENABLED=0" && WANT_P2P=0
  case "$manifest" in *router-only*) WANT_P2P=none ;; esac

  echo "  tearing down"
  teardown || { RESULTS+=("$name FAIL teardown"); continue; }

  echo "  deploying"
  scp -q "$TMP" "$HEAD_NODE:~/arm.yaml" || { RESULTS+=("$name FAIL scp"); continue; }
  sshq "$HEAD_NODE" "$KC; k3s kubectl apply -f ~/arm.yaml" || { RESULTS+=("$name FAIL apply"); continue; }

  if ! wait_ready 3; then
    echo "  !! pods not Running after 600s"
    sshq "$HEAD_NODE" "$KC; k3s kubectl -n dynamo-system get pods --no-headers | grep -E 'decodeworker|frontend'"
    RESULTS+=("$name FAIL not-ready"); continue
  fi
  sleep 30

  echo "  verifying (want P2P=$WANT_P2P)"
  if ! verify_arm "$name" "$WANT_P2P"; then
    RESULTS+=("$name FAIL verify"); continue
  fi

  mkdir -p "$REPO/runs/$name"
  cp "$TMP" "$REPO/runs/$name/manifest.yaml"

  echo "  running benchmark"
  "$REPO/bench/poll-metrics.sh" "$name" 60 > "/tmp/poll-$name.log" 2>&1 &
  POLL=$!
  RUN_ID="$name" ARM=p2p MODEL=nano ENDPOINT="$ENDPOINT" \
    TRACE="$TRACE" CONCURRENCY="$CONCURRENCY" \
    "$REPO/bench/run-scenario.sh"
  rc=$?
  kill "$POLL" 2>/dev/null

  if [ $rc -eq 0 ] && [ -s "$REPO/runs/$name/summary.json" ]; then
    read_pct=$(python3 -c "
import json;d=json.load(open('$REPO/runs/$name/summary.json'))
print(f\"{d.get('cache_read_pct') or 0:.2f}% read, {d.get('req_per_hour') or 0:.0f} req/h\")" 2>/dev/null)
    RESULTS+=("$name OK ${read_pct:-no-summary}")
  else
    RESULTS+=("$name FAIL run rc=$rc")
  fi
  echo "  done $(date -u +%FT%TZ)"
done < "$CONF"

echo
echo "======================================================================"
echo "sweep finished $(date -u +%FT%TZ)"
printf '  %s\n' "${RESULTS[@]}"
echo
echo "log: $SWEEP_LOG"
