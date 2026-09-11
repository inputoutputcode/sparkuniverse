#!/usr/bin/env bash
# Start and end timestamps for every run, in UTC and PT.
#
#   ./run-times.sh              markdown table, all runs
#   ./run-times.sh weka-c-rep2  one run
#
# **Anchor on aiperf.log, not config.json.** config.json's `started` is written
# by run-scenario.sh when it registers the run, 10 to 20 s before AIPerf begins
# profiling. `duration_s` in summary.json is AIPerf's own benchmark_duration, so
# it is measured from the PROFILING line and pairing it with config.json's
# timestamp puts the end time out by that gap.
#
# aiperf.log logs in **local time**, which is PT on the benchmark host. There is
# no date on those lines, only a clock, so the date comes from config.json where
# it exists and from the file mtime where it does not.
#
# August 2026 is PDT, UTC-7. This does not handle a run that straddles a DST
# boundary; none do, and the next transition is 2026-11-01.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
ONLY=${1:-}

python3 - "$REPO" "$ONLY" <<'PY'
import json, os, re, sys
from datetime import datetime, timedelta, timezone

repo, only = sys.argv[1], sys.argv[2]
runs = os.path.join(repo, "runs")
PT = timezone(timedelta(hours=-7))
UTC = timezone.utc

def read_config(d):
    p = os.path.join(d, "config.json")
    if not os.path.exists(p):
        return {}
    raw = open(p).read()
    # `"block_size": ,` -- BLOCK_SIZE is empty for weka_trace and was
    # interpolated bare. Repaired in run-scenario.sh, but every run before
    # 2026-08-19 has the broken form on disk.
    raw = re.sub(r'("block_size"\s*:)\s*(?=[,}\n])', r'\1 null', raw)
    try:
        return json.loads(raw)
    except ValueError:
        return {}

rows = []
for name in sorted(os.listdir(runs)):
    d = os.path.join(runs, name)
    if not os.path.isdir(d) or (only and name != only):
        continue

    cfg = read_config(d)
    summ = os.path.join(d, "summary.json")
    dur = req = None
    if os.path.exists(summ):
        try:
            s = json.load(open(summ))
            dur, req = s.get("duration_s"), s.get("requests")
        except ValueError:
            pass

    log = os.path.join(d, "aiperf.log")
    clock = source = None
    if os.path.exists(log):
        for line in open(log, errors="replace"):
            m = re.search(r"(\d{2}:\d{2}:\d{2}\.\d+)\s+INFO\s+AIPerf System is PROFILING", line)
            if m:
                clock, source = m.group(1), "aiperf.log"
                break

    # Date: config.json's UTC start converted to PT, else the log's mtime.
    day = None
    if cfg.get("started"):
        day = (datetime.strptime(cfg["started"], "%Y-%m-%dT%H:%M:%SZ")
               .replace(tzinfo=UTC).astimezone(PT).date())
    elif os.path.exists(log):
        day = datetime.fromtimestamp(os.path.getmtime(log), PT).date()

    if clock and day:
        t0 = datetime.combine(day, datetime.strptime(clock, "%H:%M:%S.%f").time(), PT)
    elif cfg.get("started"):
        t0 = datetime.strptime(cfg["started"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        source = "config.json (approx, PROFILING line missing)"
    else:
        rows.append((name, cfg.get("model"), None, None, dur, req, "NO TIMESTAMP"))
        continue

    t1 = t0 + timedelta(seconds=dur) if dur else None
    rows.append((name, cfg.get("model"), t0, t1, dur, req, source))

f = lambda t, tz: t.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S") if t else "-"

print("| Run | Model | Start UTC | End UTC | Start PT | End PT | Dur | Req | Source |")
print("|---|---|---|---|---|---|---:|---:|---|")
for name, model, t0, t1, dur, req, src in rows:
    print(f"| `{name}` | {model or '-'} | {f(t0,UTC)} | {f(t1,UTC)} | "
          f"{f(t0,PT)} | {f(t1,PT)} | {f'{dur/3600:.2f} h' if dur else '-'} | "
          f"{f'{req:,.0f}' if req else '-'} | {src} |")

missing = [r[0] for r in rows if r[6] == "NO TIMESTAMP"]
if missing:
    print()
    print("No timestamp recoverable for: " + ", ".join(missing))
    print("Neither config.json nor an aiperf.log with a PROFILING line survives.")
PY
