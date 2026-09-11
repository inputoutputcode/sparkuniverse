#!/usr/bin/env bash
# One benchmark arm, end to end: register the run, replay a trace with AIPerf,
# collect artifacts, extract the numbers that matter.
#
# Runs from the Mac Mini. The endpoint can be a single vLLM server or a Dynamo
# frontend, so the same script covers all three arms.
#
#   ARM=single   ENDPOINT=http://10.0.0.11:8011  ./run-scenario.sh
#   ARM=router   ENDPOINT=http://10.0.0.11:8000  ./run-scenario.sh
#   ARM=p2p      ENDPOINT=http://10.0.0.11:8000  ./run-scenario.sh
#
# Every run gets a run_id that reaches Prometheus through node_exporter's
# textfile collector, so the dashboards can attribute every series to this
# configuration. A run without one is not analysable later.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
ARM=${ARM:?single|router|p2p}
MODEL=${MODEL:-nano}
ENDPOINT=${ENDPOINT:?e.g. http://10.0.0.11:8000}
# No default. Every arm in docs/exp-*.md passes agent-1000.jsonl explicitly,
# and the old default pointed at cc-weka-s10-128k.jsonl, which no run has ever
# used and which needs a different block size. A default nobody uses is a trap
# waiting for the one time someone omits the variable.
TRACE=${TRACE:?path to the trace, e.g. $REPO/traces/data/agent-1000.jsonl}
CONCURRENCY=${CONCURRENCY:-5}
SEED=${SEED:-42}
AIPERF=${AIPERF:-$HOME/aiperf/venv/bin/aiperf}
NODES=${NODES:-"10.0.0.11 10.0.0.12"}

# AIPerf needs the HF repo id, not a local path. A wrong tokenizer produces
# plausible but incorrect sequence lengths and the run looks fine.
case "$MODEL" in
  nano)  SERVED=nemotron; TOKENIZER=${TOKENIZER:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4};  CHUNK=2128 ;;
  super) SERVED=nemotron; TOKENIZER=${TOKENIZER:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4}; CHUNK=4224 ;;
  *) echo "unknown MODEL: $MODEL"; exit 2 ;;
esac

RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M)-$MODEL-$ARM}
OUT=$REPO/runs/$RUN_ID
mkdir -p "$OUT/aiperf"

# The block size AIPerf uses to synthesise prompts must equal the block size
# the trace was captured with, or every hash id expands to the wrong number of
# tokens and the replayed prompts bear no relation to the recorded ones. Nothing
# downstream complains: the run completes and every number is wrong.
#
# The two families need different loaders, and picking the wrong one either
# fails loudly or, worse, synthesises every prompt at the wrong length and
# completes normally.
#
#   cc-weka-*   session-grouped SemiAnalysis Claude Code capture. AIPerf's
#               weka_trace loader reads this shape directly and preserves
#               session boundaries, turn ordering, subagent branch and
#               spawn-join, inter-turn delays, and the per-turn counts that
#               populate theoretical_prefix_cache_hit. Let the loader supply
#               its own block size from plugin metadata.
#   agent-*     flat mooncake trace at 512. Nothing above survives flattening.
#
# convert-mooncake.py exists because mooncake_trace rejects the session form.
# That was never a property of AIPerf, only of that one dataset type.
case "$(basename "$TRACE")" in
  cc-weka*)          DATASET_TYPE=weka_trace;     TRACE_BLOCK="" ;;
  agent*|64k_*|8k_*) DATASET_TYPE=mooncake_trace; TRACE_BLOCK=512 ;;
  *)                 DATASET_TYPE="";             TRACE_BLOCK="" ;;
esac
DATASET_TYPE=${DATASET_TYPE:?cannot infer loader for $(basename "$TRACE"), set DATASET_TYPE}

# Shape and loader have to agree. A session-grouped file under mooncake_trace
# dies mid-run with "Exactly one of 'input_length', 'text_input', 'messages',
# or 'payload' must be provided", after the run has already registered itself.
#
# weka_trace wants a directory of *.json, one WekaTrace each. Not JSONL, which
# it rejects with "unexpected content after document", and not a JSON array,
# which fails validation. traces/split-weka.py produces the directory.
GROUPED=0
if [ -d "$TRACE" ]; then
  GROUPED=1
  N_JSON=$(find "$TRACE" -maxdepth 1 -name '*.json' | wc -l | tr -d ' ')
  [ "$N_JSON" -gt 0 ] || { echo "!! $TRACE contains no *.json files"; exit 2; }
elif head -1 "$TRACE" | grep -q '"requests"'; then
  GROUPED=1
fi

if [ "$DATASET_TYPE" = weka_trace ] && [ ! -d "$TRACE" ]; then
  echo "!! weka_trace wants a directory of one-session .json files, not $TRACE"
  echo "   python3 $REPO/traces/split-weka.py $TRACE ${TRACE%.jsonl}/"
  exit 2
fi
if [ "$DATASET_TYPE" = mooncake_trace ] && [ "$GROUPED" = 1 ]; then
  echo "!! $TRACE is session-grouped but the loader is mooncake_trace."
  echo "   Prefer --custom-dataset-type weka_trace, which reads it as captured."
  exit 2
fi

# These are expanded below as ${ARR[@]+"${ARR[@]}"}, not "${ARR[@]}".
# This script runs from the Mac, macOS ships bash 3.2, and expanding an empty
# array as "${ARR[@]}" under `set -u` is an unbound variable error there. Bash
# fixed it in 4.4, so it passes every check on Linux and fails only on the
# machine that actually runs it:
#   ./run-scenario.sh: line 305: BLOCK_ARG[@]: unbound variable
# after the run has already registered itself in Prometheus.
#
# Only mooncake_trace needs a block size. For weka_trace the loader's own
# metadata is right and overriding it is how you get prompts at the wrong
# length.
BLOCK_SIZE=${BLOCK_SIZE:-$TRACE_BLOCK}
BLOCK_ARG=()
if [ "$DATASET_TYPE" = mooncake_trace ]; then
  [ -n "$BLOCK_SIZE" ] || { echo "!! set BLOCK_SIZE for $(basename "$TRACE")"; exit 2; }
  if [ -n "$TRACE_BLOCK" ] && [ "$BLOCK_SIZE" != "$TRACE_BLOCK" ]; then
    echo "!! BLOCK_SIZE=$BLOCK_SIZE but $(basename "$TRACE") was captured at $TRACE_BLOCK."
    echo "   Every prompt would be synthesised at the wrong length."
    exit 2
  fi
  BLOCK_ARG=(--prompt-input-tokens-block-size "$BLOCK_SIZE")
fi

# Weka loaders cap peak prompt+output per root trace. Default matches
# --max-model-len, so a session that would overflow the engine is excluded
# rather than truncated. No effect on mooncake_trace.
MAX_CONTEXT=${MAX_CONTEXT:-131072}
CTX_ARG=()
if [ "$DATASET_TYPE" = weka_trace ]; then
  # --ignore-trace-delays is not optional here.
  #
  # The Weka loader defaults to open-loop replay: each session starts at its
  # absolute recorded timestamp and turns are separated by recorded think time.
  # These captures carry 600 s think times and span up to 18 hours, so AIPerf
  # reaches PROFILING and then sends nothing, which looks exactly like a broken
  # deployment. Observed 2026-08-18: endpoint reachable, configuration clean in
  # 7.10 s, vllm:num_requests_running at 0 on both nodes indefinitely.
  #
  # A fixed --concurrency makes the run closed-loop by construction, so recorded
  # arrival times conflict with it rather than adding realism. It also matches
  # the agent-1000 arms, where mooncake_trace under concurrency ignored
  # timestamps, so the corpora stay comparable.
  #
  # The cost, and it is real: stripping inter-turn delays runs sessions
  # back-to-back, so the cache never ages between turns the way it would in
  # production. Eviction pressure here is lower than a real deployment's. If
  # that turns out to matter, --trace-idle-gap-cap-seconds keeps the gaps while
  # bounding them, at the price of a much longer run.
  CTX_ARG=(--max-context-length "$MAX_CONTEXT" --ignore-trace-delays)
fi

case "$DATASET_TYPE" in
  weka_trace)     ENDPOINT_TYPE=${ENDPOINT_TYPE:-chat} ;;
  mooncake_trace) ENDPOINT_TYPE=${ENDPOINT_TYPE:-completions} ;;
esac

# Without this AIPerf stops on a default duration budget, not on the dataset.
# The first agent-500 run ended at 299.67 s having completed 10 requests of
# 500, because average latency was 136 s. The console table then reports
# perfectly reasonable numbers over a 2% sample, which is the dangerous kind
# of wrong: nothing about the output says it is a fragment.
#
# `wc -l` is the request count only for a flat trace. A session-grouped file has
# one line per *session*, so cc-weka-s85-128k.jsonl would have run 85 requests
# instead of 3,485 and reported a complete result over 2% of the workload.
# Count the nested requests for grouped files.
count_requests() {
  if [ "$GROUPED" = 1 ]; then
    # Two shapes carry sessions. weka-subset.py writes one JSON object per
    # line, and AIPerf's weka_trace loader calls json.load() and wants a single
    # document, failing with
    #   invalid JSON: unexpected content after document: line 2 column 1
    # Handle both here so the count does not depend on which one is in use.
    python3 - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if p.is_dir():
    sessions = [json.loads(f.read_text()) for f in sorted(p.glob("*.json"))]
else:
    raw = p.read_text().strip()
    try:
        doc = json.loads(raw)
        sessions = doc if isinstance(doc, list) else doc.get("traces") or [doc]
    except json.JSONDecodeError:
        sessions = [json.loads(l) for l in raw.splitlines() if l.strip()]
n = 0
for s in sessions:
    stack = list(s.get("requests") or [])
    while stack:
        r = stack.pop()
        stack.extend(r.get("requests") or [])
        n += 1
print(n)
PY
  else
    wc -l < "$1" | tr -d ' '
  fi
}
TRACE_REQUESTS=$(count_requests "$TRACE")
REQUESTS=${REQUESTS:-$TRACE_REQUESTS}

# A ceiling, not a target. --request-count is what should end the run; this
# only stops a misconfigured one from occupying the cluster indefinitely.
# Set it well above the projection: agent-1000 is ~2.19M output tokens, which
# at the measured 195 tok/s aggregate is 3.1 h, so 8 h leaves room for the
# reuse rate coming in lower than the knee sweep's synthetic prompts gave.
#
MAXDUR=${MAXDUR:-28800}

# Finite, not `inf`.
#
# The default 30 s grace truncates requests that legitimately run for minutes
# and skews every percentile, so `inf` looked like the safe choice. It is not:
# on 2026-08-18 weka-c-rep1 sent all 3,485 requests, 3,472 completed, and then
# waited six hours for the last 13. No summary, and arms 2 and 3 never started.
#
# The longest legitimate request here is 59,903 output tokens, about 50 minutes
# at the measured 20 tok/s per user. 5400 covers that with margin and still
# bounds a hang. A run that hits this ceiling has stuck requests and should be
# investigated rather than trusted.
GRACE=${GRACE:-5400}

# AIPerf's subprocesses survive an aborted run and ignore SIGTERM. By the
# afternoon of 2026-08-15 five generations were still resident, the oldest from
# five hours earlier. A stale dataset_manager holding its ZMQ endpoints is the
# best available explanation for the "configured in 0.09 s" runs that then
# abort at 300 s with "Dataset configuration not received": the new services
# reach an old manager, and the new record processor never gets the broadcast.
#
# Unproven -- arm B succeeded with orphans present -- but they are garbage
# regardless, and they compete for CPU on the machine generating the load.
if pgrep -f aiperf >/dev/null 2>&1; then
  echo "  clearing $(pgrep -f aiperf | wc -l | tr -d ' ') orphaned aiperf processes"
  pkill -9 -f aiperf
  sleep 2
fi

# The actual cause of "Dataset configuration not received after 300.0s".
#
# AIPerf content-addresses tokenized datasets under ~/.cache/aiperf/dataset_mmap,
# keyed on input bytes + dataset type + tokenizer identity. The first run on a
# trace tokenizes it in ~7 s. Every later run is a cache hit and configuration
# completes in about a millisecond -- before the record processor has registered,
# so the dataset-configuration broadcast reaches nobody and the run dies 300 s
# later. Measured 2026-08-16: p2p and l1-48 succeeded cold; both router-only
# attempts failed warm, with CONFIGURED to PROFILING in 1 ms and the record
# processor registering 922 ms after that.
#
# This supersedes the --no-server-metrics explanation, which was wrong: both
# failures had it set and a clean process table.
#
# Cost of clearing: one re-tokenization, ~7 s. Do not reach for --cache-bust
# instead. It injects per-conversation markers into the prompts, which changes
# the token stream and destroys exactly the prefix sharing being measured.
CACHE_DIR=${AIPERF_CACHE_DIR:-$HOME/.cache/aiperf/dataset_mmap}
if [ -d "$CACHE_DIR" ]; then
  echo "  clearing AIPerf dataset cache ($(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)) to avoid the startup race"
  rm -rf "${CACHE_DIR:?}"/*
fi

[ -x "$AIPERF" ] || { echo "!! aiperf not found at $AIPERF"; exit 1; }
[ -s "$TRACE" ] || [ -d "$TRACE" ] || { echo "!! trace not found: $TRACE"; exit 1; }

curl -sf "$ENDPOINT/v1/models" >/dev/null || { echo "!! endpoint not serving: $ENDPOINT"; exit 1; }

# curl is not the client that matters. macOS gates local-network access per
# binary, so curl succeeds against a Spark while the aiperf venv python gets
#   OSError(65, 'No route to host')
# and AIPerf then sits in PROFILING sending nothing until you notice the GPUs
# are idle at 10 W. Check with the interpreter that will actually make the
# requests, and fail in a second rather than after a run's worth of silence.
PY=$(dirname "$AIPERF")/python
[ -x "$PY" ] || PY=python3
"$PY" - "$ENDPOINT" <<'PY' || exit 1
import sys, json, urllib.request
url = sys.argv[1].rstrip("/") + "/v1/models"
try:
    ids = [m["id"] for m in json.load(urllib.request.urlopen(url, timeout=5))["data"]]
except OSError as e:
    print(f"!! the aiperf interpreter cannot reach {url}: {e}")
    print("   curl works and this does not -> macOS Local Network gating.")
    print("   See docs/dead-ends.md. Grant it, or tunnel to localhost.")
    sys.exit(1)
if not ids:
    print(f"!! {url} served an empty model list"); sys.exit(1)
print(f"    endpoint ok, models: {', '.join(ids)}")
PY

echo "=== run $RUN_ID"
echo "    model $MODEL  arm $ARM  concurrency $CONCURRENCY"
echo "    trace $(basename "$TRACE")  $TRACE_REQUESTS requests, loader $DATASET_TYPE"
echo "    sending $REQUESTS"

# Configuration snapshot. Written before the run so a crash still leaves a
# record of what was attempted.
#
# BLOCK_SIZE is deliberately empty for weka_trace, where the loader supplies its
# own from plugin metadata. Interpolated bare that produced `"block_size": ,`
# and every weka run since has written invalid JSON. Nothing complained: the
# file is only read afterwards, by summarize-run.py, which catches the parse
# error and silently drops expected_requests -- so a recovered summary loses the
# one field that says whether the run finished. That is the exact check added
# after a 128-request fragment stood in for a 3,485-request arm.
BLOCK_JSON=${BLOCK_SIZE:-null}

cat > "$OUT/config.json" <<EOF
{
  "run_id": "$RUN_ID",
  "arm": "$ARM",
  "model": "$MODEL",
  "endpoint": "$ENDPOINT",
  "trace": "$(basename "$TRACE")",
  "trace_requests": $TRACE_REQUESTS,
  "requests_sent": $REQUESTS,
  "dataset_type": "$DATASET_TYPE",
  "concurrency": $CONCURRENCY,
  "seed": $SEED,
  "chunk_size": $CHUNK,
  "block_size": $BLOCK_JSON,
  "tokenizer": "$TOKENIZER",
  "started": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

for ip in $NODES; do
  ssh -o ConnectTimeout=5 "$ip" "~/runinfo.sh start $RUN_ID $MODEL $ARM chunk=$CHUNK conc=$CONCURRENCY" 2>/dev/null \
    || echo "  !! run registry not reachable on $ip, series will be unattributable"
done

# Ctrl-C leaves AIPerf's worker and record-processor subprocesses alive, and
# leaves bench_run_info advertising a run that never finished -- after which
# every later series is attributed to it. Clean both up on any exit path.
cleanup() {
  pkill -f aiperf 2>/dev/null
  for ip in $NODES; do
    ssh -o ConnectTimeout=5 "$ip" "~/runinfo.sh stop $RUN_ID" 2>/dev/null
  done
}
trap 'echo; echo "interrupted, cleaning up"; cleanup; exit 130' INT TERM

# --no-server-metrics is not an optimisation. AIPerf scrapes the inference
# endpoint's own /metrics by default, and when that endpoint answers,
# configuration completes in ~0.08 s instead of ~6.9 s. The record processor
# then registers *after* the dataset-configuration broadcast, never receives
# it, and aborts the run 300 s later:
#   Dataset configuration not received after 300.0s; aborting run
# Reproduced twice at c5 and c10. It is a startup race in AIPerf, triggered by
# an unrelated endpoint being fast. Prometheus already collects the same
# series with run-id attribution, so nothing is lost.
#
# --endpoint-type follows the loader, and getting it wrong fails every request.
#
#   mooncake_trace  -> completions. The trace is bare token blocks with no
#                      messages to send, and the default `chat` would wrap them.
#   weka_trace      -> chat. The loader reconstructs multi-turn conversations,
#                      and AIPerf's own client refuses to send them otherwise:
#                        ValueError('Completions endpoint only supports one turn.')
#                      All 40 requests failed this way on 2026-08-17 before the
#                      server saw any of them.
#
# The chat path also exercises --dyn-reasoning-parser and --dyn-tool-call-parser,
# which completions bypasses. That is closer to how an agent harness actually
# calls the model, and it means output token counts include whatever the
# template adds per turn.

# Watchdog for the race above. --no-server-metrics reduced its frequency but did
# not eliminate it: it recurred on the router-only arm at 14:07:49 on 2026-08-16,
# 335 s into a 2.2 h run. When the record processor dies the workers keep sending
# and the GPUs stay busy, so every dashboard shows a healthy run producing
# nothing. Two hours of machine time, no records.
#
# Kill the run the moment the failure appears rather than letting it burn the
# slot. The flag file is what distinguishes this from a clean finish afterwards,
# because the exit code does not.
FATAL="Dataset configuration not received"
FLAG=$OUT/.aborted-startup-race
rm -f "$FLAG"
(
  for _ in $(seq 1 60); do
    sleep 10
    [ -f "$OUT/aiperf.log" ] || continue
    if grep -q "$FATAL" "$OUT/aiperf.log" 2>/dev/null; then
      touch "$FLAG"
      echo; echo "!! record processor died, killing the run rather than burning the slot"
      pkill -9 -f aiperf
      return 0 2>/dev/null || exit 0
    fi
  done
) &
WATCHDOG=$!

START=$(date +%s)
"$AIPERF" profile \
  -m "$SERVED" \
  --tokenizer "$TOKENIZER" \
  --tokenizer-trust-remote-code \
  --input-file "$TRACE" \
  --custom-dataset-type "$DATASET_TYPE" \
  ${BLOCK_ARG[@]+"${BLOCK_ARG[@]}"} ${CTX_ARG[@]+"${CTX_ARG[@]}"} \
  --url "$ENDPOINT" \
  --streaming --use-server-token-count \
  --endpoint-type "$ENDPOINT_TYPE" \
  --no-server-metrics \
  --extra-inputs ignore_eos:true \
  --concurrency "$CONCURRENCY" --random-seed "$SEED" \
  --request-count "$REQUESTS" \
  --benchmark-duration "$MAXDUR" --benchmark-grace-period "$GRACE" \
  --artifact-dir "$OUT/aiperf" \
  2>&1 | tee "$OUT/aiperf.log"
RC=${PIPESTATUS[0]}
END=$(date +%s)
kill "$WATCHDOG" 2>/dev/null

# A run that hit the race must not fall through to the summary extractor. It
# writes plausible-looking numbers over whatever fraction completed, and nothing
# in that output says it is a fragment.
if [ -f "$FLAG" ] || grep -q "$FATAL" "$OUT/aiperf.log" 2>/dev/null; then
  echo
  echo "!! ABORTED: AIPerf's record processor never received the dataset config."
  echo "   No usable results. Before retrying:"
  echo "     pkill -9 -f aiperf && sleep 3 && pgrep -f aiperf | wc -l   # must be 0"
  echo "   Then relaunch with a fresh RUN_ID so this directory is not overwritten."
  for ip in $NODES; do
    ssh -o ConnectTimeout=5 "$ip" "~/runinfo.sh stop $RUN_ID" 2>/dev/null
  done
  exit 1
fi

for ip in $NODES; do
  ssh -o ConnectTimeout=5 "$ip" "~/runinfo.sh stop $RUN_ID" 2>/dev/null
done

echo
echo "=== results"
echo "  wall time: $((END-START)) s"

# Prefill tokens avoided. This is the cleanest cache metric: server-reported,
# per request, independent of scheduling and decode speed. Needs
# --enable-prompt-tokens-details on the engine.
# Read AIPerf's own aggregate export, not the per-record jsonl. The jsonl does
# not carry a `usage` object, so the previous version of this block reported
# zeros for a run whose console table showed 76.92% cache read -- the most
# dangerous failure mode available, since zeros look like a result.
python3 - "$OUT" "$REQUESTS" <<'PY'
import json, sys, glob
out = sys.argv[1]
expected = int(sys.argv[2])
f = glob.glob(f"{out}/aiperf/**/profile_export_aiperf.json", recursive=True)
if not f:
    print("  no profile_export_aiperf.json found, check the artifact layout")
    raise SystemExit
j = json.load(open(f[0]))

def m(key, stat="avg"):
    v = j.get(key)
    return v.get(stat) if isinstance(v, dict) else v

# Metric names confirmed against schema 1.4 / aiperf 0.12.0. The totals are
# separate keys from the per-request averages -- take the total_ ones, or the
# reuse figure comes out as a mean over requests and looks plausible while
# being the wrong quantity.
s = {
    "requests":        m("request_count"),
    "duration_s":      m("benchmark_duration"),
    "req_per_hour":    (m("request_throughput") or 0) * 3600,
    "output_tok_s":    m("output_token_throughput"),
    "output_tok_s_user": m("output_token_throughput_per_user"),
    "total_output_tokens": m("total_usage_completion_tokens") or m("total_output_tokens"),
    "total_prompt_tokens": m("total_usage_prompt_tokens") or m("total_isl"),
    "cached_prompt_tokens": m("total_usage_prompt_cache_read_tokens"),
    "cache_read_pct":  m("overall_usage_prompt_cache_read_pct"),
    "ttft_p50_ms":     m("time_to_first_token", "p50"),
    "ttft_p90_ms":     m("time_to_first_token", "p90"),
    "itl_p50_ms":      m("inter_token_latency", "p50"),
    "isl_p50":         m("input_sequence_length", "p50"),
    "osl_p50":         m("output_sequence_length", "p50"),
}

# AIPerf's own infinite-cache ceiling, from TheoreticalPrefixCacheAccumulator.
# Worth more than the trie in traces/session-structure.py: same instrument, same
# run, same denominator as the achieved figure, and phase-scoped so warmup
# cannot leak in. Comparing 84.47% measured against a ceiling computed by a
# different tool over a different denominator was never quite valid.
#
# The exact export key is not documented, so discover it rather than guess. A
# hardcoded name that silently misses would reintroduce the zeros-look-like-a-
# result failure this block already had once.
fmt = lambda v, d=1: f"{v:,.{d}f}" if isinstance(v, (int, float)) else "n/a"

# MetricResult carries count = total blocks and sum = hit blocks alongside the
# percentage. Take all three: the two integers are directly comparable to
# traces/session-structure.py's 87,207 total and 73,843 seen, so this either
# validates that trie or refutes it, which a percentage alone cannot do.
theo = {k: v for k, v in j.items() if "theoretical" in k.lower()}
if theo:
    for k, v in sorted(theo.items()):
        if isinstance(v, dict):
            s[k] = v.get("avg")
            s[k + "_total_blocks"] = v.get("count")
            s[k + "_hit_blocks"] = v.get("sum")
            print(f"  {k:<30}{fmt(s[k], 2)}%   "
                  f"{fmt(s[k+'_hit_blocks'], 0)} of {fmt(s[k+'_total_blocks'], 0)} blocks")
        else:
            s[k] = v
            print(f"  {k:<30}{v}")
else:
    print("  !! no theoretical_prefix_cache_* key in the export.")
    print("     The accumulator only enables when the loader stamps per-turn")
    print("     block counts, and that is the WEKA loader. A --custom-dataset-type")
    print("     mooncake_trace run may never populate it. If so, the ceiling has")
    print("     to keep coming from traces/session-structure.py.")
print(f"  requests            {fmt(s['requests'], 0)}")
print(f"  requests/hour       {fmt(s['req_per_hour'])}")
print(f"  output tok/s        {fmt(s['output_tok_s'])}   per user {fmt(s['output_tok_s_user'])}")
print(f"  prompt tokens       {fmt(s['total_prompt_tokens'], 0)}")
print(f"  cache read          {fmt(s['cached_prompt_tokens'], 0)}  ({fmt(s['cache_read_pct'], 2)}%)")
print(f"  TTFT p50/p90 (s)    {fmt((s['ttft_p50_ms'] or 0)/1000, 2)} / {fmt((s['ttft_p90_ms'] or 0)/1000, 2)}")
print(f"  ITL p50 (ms)        {fmt(s['itl_p50_ms'], 2)}")
if s["cache_read_pct"] is None:
    print("  !! overall_usage_prompt_cache_read_pct absent. Present only when")
    print("     the server reports cached tokens, i.e. when something hit.")
    print(f"     usage keys: {[k for k in j if 'usage' in k.lower()]}")
# A killed run still leaves an AIPerf export, and a summary built from it looks
# entirely normal. weka-b-l1only was written from 128 of 3,485 requests, 270 s
# of a 3 h arm, and run-sweep.sh then skipped that arm as "already done".
# Record what was expected so nothing downstream has to guess.
s["expected_requests"] = expected
got = s.get("requests") or 0
s["complete"] = bool(got >= 0.99 * expected)
if not s["complete"]:
    print(f"  !! INCOMPLETE: {got:,.0f} of {expected:,} requests. "
          f"Not a result. Delete runs/{out.rsplit('/', 1)[-1]} and rerun.")
json.dump(s, open(f"{out}/summary.json", "w"), indent=1)
PY

echo "  artifacts: $OUT"
exit $RC
