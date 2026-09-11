#!/usr/bin/env bash
# Where does per-user throughput collapse?
#
# On GB10 decode is the bottleneck, so the question that decides how many
# engineers a pair of Sparks can serve is not "what is the peak throughput"
# but "past which concurrency does adding a session make everyone slower
# without producing more total work".
#
# Two curves, from the same runs:
#
#   aggregate tokens/s   rises with concurrency, then flattens. The flat part
#                        is the machine's real output capacity.
#   per-user tokens/s    falls monotonically. Where it crosses what a person
#                        will tolerate is the session limit.
#
# The knee is where aggregate stops rising. Past it, concurrency buys queueing
# and nothing else.
#
# Synthetic prompts, not the trace: this measures the engine, and the trace's
# variable prompt lengths and cache hits would confound the sweep. ISL is set
# near the trace's median so the answer transfers.
#
# Runs from the Mac Mini against the Dynamo frontend, so the numbers are for
# the pair. Point ENDPOINT at one worker to get a single node.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
ENDPOINT=${ENDPOINT:?e.g. http://localhost:8000}
MODEL=${MODEL:-nano}
LEVELS=${LEVELS:-"1 2 4 6 8 10 12 16"}
ISL=${ISL:-32000}          # near the trace median, and one full 15-chunk context
OSL=${OSL:-512}            # long enough for steady-state decode, short enough to finish
REQS=${REQS:-}             # default: 3 x concurrency, so every level does equal work per slot
AIPERF=${AIPERF:-$HOME/aiperf/venv/bin/aiperf}
SEED=${SEED:-42}

case "$MODEL" in
  nano)  SERVED=nemotron; TOKENIZER=${TOKENIZER:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4} ;;
  super) SERVED=nemotron; TOKENIZER=${TOKENIZER:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4} ;;
  *) echo "unknown MODEL: $MODEL"; exit 2 ;;
esac

RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M)-$MODEL-knee}
OUT=$REPO/runs/$RUN_ID
mkdir -p "$OUT"

[ -x "$AIPERF" ] || { echo "!! aiperf not found at $AIPERF"; exit 1; }
curl -sf "$ENDPOINT/v1/models" >/dev/null || { echo "!! endpoint not serving: $ENDPOINT"; exit 1; }

echo "=== $RUN_ID"
echo "    ISL $ISL  OSL $OSL  levels: $LEVELS"
echo "    $ENDPOINT"
echo

printf "%6s  %12s  %12s  %10s  %10s  %8s\n" \
  conc "agg tok/s" "user tok/s" "TTFT p50" "TTFT p90" "reqs"

for C in $LEVELS; do
  N=${REQS:-$((C * 3))}
  D="$OUT/c$C"
  mkdir -p "$D"

  # --no-server-metrics: scraping the endpoint's own /metrics makes AIPerf
  # configuration finish before the record processor registers, after which it
  # aborts at 300 s. See run-scenario.sh.
  "$AIPERF" profile \
    -m "$SERVED" \
    --tokenizer "$TOKENIZER" --tokenizer-trust-remote-code \
    --url "$ENDPOINT" --endpoint-type completions --streaming \
    --synthetic-input-tokens-mean "$ISL" --synthetic-input-tokens-stddev 0 \
    --osl "$OSL" \
    --extra-inputs ignore_eos:true \
    --concurrency "$C" --request-count "$N" \
    --random-seed "$SEED" \
    --no-server-metrics \
    --artifact-dir "$D" \
    >"$D/aiperf.log" 2>&1

  python3 - "$D" "$C" <<'PY'
import json, sys, glob
d, c = sys.argv[1], sys.argv[2]
f = glob.glob(f"{d}/**/profile_export_aiperf.json", recursive=True)
if not f:
    print(f"{c:>6}  {'no export':>12}"); raise SystemExit
j = json.load(open(f[0]))
g = lambda k, s="avg": (j.get(k) or {}).get(s)
agg  = g("output_token_throughput")
user = g("output_token_throughput_per_user")
t50  = g("time_to_first_token", "p50")
t90  = g("time_to_first_token", "p90")
n    = g("request_count")
fmt = lambda v, d=1: f"{v:,.{d}f}" if isinstance(v, (int, float)) else "n/a"
print(f"{c:>6}  {fmt(agg):>12}  {fmt(user):>12}  "
      f"{fmt((t50 or 0)/1000, 2):>10}  {fmt((t90 or 0)/1000, 2):>10}  {fmt(n, 0):>8}")
PY
done

echo
echo "  Read the aggregate column: the knee is where it stops rising."
echo "  Read the per-user column against what a person will tolerate."
echo "  artifacts: $OUT"
