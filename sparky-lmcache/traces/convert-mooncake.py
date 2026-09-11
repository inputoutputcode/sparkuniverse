#!/usr/bin/env python3
"""Flatten session-grouped agent traces into AIPerf's mooncake_trace format.

The captured traces are one JSON object per *session*:

    {"id": "...", "block_size": 64, "hash_id_scope": "local",
     "models": ["claude-fable-5", ...],
     "requests": [{"t": 0.0, "in": 640, "out": 15, "hash_ids": [0..9],
                   "api_time": 1.534, "ttft": 1.255, "type": "s"}, ...]}

AIPerf's `mooncake_trace` dataset type wants one object per *request*, and
rejects the session form outright:

    Value error, Exactly one of 'input_length', 'text_input', 'messages',
    or 'payload' must be provided

**That rejection is specific to `mooncake_trace`, not to AIPerf.** AIPerf ships
`WekaTraceLoader`, a file-based loader that reads exactly this session-grouped
shape, and `SemiAnalysisCCTracesWekaLoader`, which pulls the same corpus from
HuggingFace and delegates to it. Both preserve what this script destroys:
session boundaries, subagent branch and spawn-join structure, inter-turn delays,
the per-request model mapping, and the per-turn counts that populate AIPerf's
own `theoretical_prefix_cache_hit` metric.

So flattening buys compatibility with one dataset type at the cost of every
structural property the workload has. Prefer the Weka loader where the source is
session-grouped. This script remains correct for feeding an already-flat trace
through `mooncake_trace`, which is what the Nano arms used.

Two things this does beyond flattening, both of which change the answer:

**Hash ids are renumbered per session.** `hash_id_scope: "local"` means block
10 of one session and block 10 of another are unrelated content. Replayed
as-is, AIPerf would synthesise the same text for both, every session would
appear to share a prefix, and the measured reuse would be an artefact of the
conversion rather than of the cache. Each session is offset past the previous
one's highest id.

**Timestamps stay as captured.** Every session starts near t=0, so they run
concurrently, which is the point: concurrent agent sessions are what produce
cross-node routing pressure. Sorting is by absolute time across sessions.

`api_time`, `ttft` and `think_time` are measurements from the original capture
and are deliberately dropped. They describe what some other system did.
"""
import argparse
import json
import sys

ap = argparse.ArgumentParser()
ap.add_argument("input")
ap.add_argument("output")
ap.add_argument("--model", help="keep only requests whose source model matches")
ap.add_argument("--max-input", type=int, default=131072,
                help="drop requests longer than the engine's --max-model-len")
# The captured sessions include real idle gaps -- several think_time values are
# 600 s, and the widest trace spans 18 hours. Replayed at 1.0 the benchmark
# would mostly measure an idle cluster. Scaling preserves the *order* and the
# relative spacing that produces overlapping sessions, which is what creates
# routing pressure, while making the run finish.
ap.add_argument("--time-scale", type=float, default=1.0,
                help="multiply every timestamp, e.g. 0.02 turns 18 h into 22 min")
a = ap.parse_args()

offset = 0
rows = []
sessions = kept = dropped_len = dropped_model = dropped_empty = 0
block_sizes = set()

with open(a.input) as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        s = json.loads(line)
        sessions += 1
        block_sizes.add(s.get("block_size"))
        top = 0
        for r in s.get("requests", []):
            if a.model and r.get("model") != a.model:
                dropped_model += 1
                continue
            ids = r.get("hash_ids") or []
            if ids:
                top = max(top, max(ids))
            n_in = int(r.get("in", 0))
            n_out = int(r.get("out", 0))
            if n_in > a.max_input:
                dropped_len += 1
                continue
            # AIPerf rejects these outright:
            #   max_tokens: Input should be greater than or equal to 1
            # A request that generated nothing, or had no prompt, is a failed
            # or cancelled call in the capture rather than work to replay.
            # Clamping to 1 would invent load that never happened.
            if n_in < 1 or n_out < 1:
                dropped_empty += 1
                continue
            rows.append({
                "timestamp": int(round(float(r.get("t", 0.0)) * 1000 * a.time_scale)),
                "input_length": n_in,
                "output_length": n_out,
                "hash_ids": [h + offset for h in ids],
            })
            kept += 1
        # Past this session's highest id, so no two sessions collide.
        offset += top + 1

# Not every trace in traces/data is session-grouped. Some are already one
# record per request, in which case this script has nothing to do and writing
# an empty file is worse than failing -- the run then aborts inside AIPerf
# several minutes later with an unrelated-looking error.
if not rows:
    print(f"!! no requests extracted from {a.input}", file=sys.stderr)
    print("   No line carried a 'requests' array. If the file is already one", file=sys.stderr)
    print("   record per request, feed it to AIPerf directly and read its", file=sys.stderr)
    print("   block size from the capture rather than converting. Check with:", file=sys.stderr)
    print(f"     head -1 {a.input} | python3 -m json.tool | head -20", file=sys.stderr)
    raise SystemExit(1)

rows.sort(key=lambda r: r["timestamp"])

with open(a.output, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

if len(block_sizes) != 1:
    print(f"!! mixed block_size across sessions: {block_sizes}", file=sys.stderr)
bs = block_sizes.pop() if len(block_sizes) == 1 else None

lens = [r["input_length"] for r in rows]
print(f"sessions      {sessions}")
print(f"requests      {kept}")
if dropped_model:
    print(f"  dropped, model filter   {dropped_model}")
if dropped_len:
    print(f"  dropped, over --max-input {dropped_len}")
if dropped_empty:
    print(f"  dropped, zero in or out {dropped_empty}")
print(f"block_size    {bs}   <- pass this as --prompt-input-tokens-block-size")
print(f"distinct blocks {offset}")
if lens:
    lens.sort()
    print(f"input_length  min {lens[0]}  median {lens[len(lens)//2]}  max {lens[-1]}")
    span = rows[-1]["timestamp"] / 1000
    print(f"span          {span:.0f} s  (time-scale {a.time_scale})")
    if span > 3600:
        want = 1200 / span * a.time_scale
        print(f"!! that is {span/3600:.1f} h of mostly idle replay.")
        print(f"   --time-scale {want:.4f} would compress it to about 20 min.")
