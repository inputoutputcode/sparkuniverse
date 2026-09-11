#!/usr/bin/env bash
# How far along is a running arm, and when will it finish.
#
#   ./eta.sh weka-a-router
#
# AIPerf draws its progress counter with a rich terminal UI. Under `screen -dm`
# stdout is not a TTY, so that display is suppressed entirely and aiperf.log
# stops after "AIPerf System is PROFILING". Nothing is wrong when that happens,
# but it leaves no way to answer "how long left" from the log, and the question
# got asked four times in one evening.
#
# The durable counter is the export itself: one line per completed request in
# aiperf/profile_export.jsonl. config.json is written at registration, a few
# seconds before profiling starts, so its mtime is the start time to within
# rounding.
#
# The projection is linear and the tail of this corpus is not. The longest
# request is 59,903 output tokens, roughly 50 minutes on its own, and
# weka-c-rep1 sent all 3,485 then spent six hours on the last 13. Treat the
# estimate as a floor once the run is past about 95 percent.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN=${1:?run id, e.g. weka-a-router}
D=$REPO/runs/$RUN

[ -d "$D" ] || { echo "no such run: $D"; exit 1; }
EXPORT=$D/aiperf/profile_export.jsonl
CFG=$D/config.json

[ -f "$CFG" ]    || { echo "no config.json yet, the run has not registered"; exit 1; }
[ -f "$EXPORT" ] || { echo "no profile_export.jsonl yet, no request has completed"; exit 1; }

python3 - "$EXPORT" "$CFG" <<'PY'
import json, os, sys, time

export, cfg = sys.argv[1], sys.argv[2]

n = sum(1 for _ in open(export))
want = json.load(open(cfg)).get("requests_sent") or 0
start = os.path.getmtime(cfg)
elapsed = time.time() - start

if not n or not want:
    raise SystemExit(f"{n} records, {want} expected, nothing to project")

rate = n / elapsed
left = (want - n) / rate if n < want else 0

print(f"{n:,} of {want:,}   {n / want * 100:.1f}%   {rate:.3f} req/s")
print(f"elapsed {elapsed / 3600:.2f} h   remaining {left / 60:.0f} min   "
      f"finish ~{time.strftime('%H:%M', time.localtime(time.time() + left))}")

# Past this point the linear projection is reliably optimistic, so say so
# rather than letting the number be read as a real estimate.
if n / want > 0.95:
    print("  in the tail: the remaining requests are the long ones, expect longer")
PY
