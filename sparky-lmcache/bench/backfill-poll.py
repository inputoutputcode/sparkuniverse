#!/usr/bin/env python3
"""Rebuild Prometheus history for a run from its poll.csv, without touching what is already there.

    python3 bench/backfill-poll.py weka-c-rep2                 # dry run, writes nothing
    python3 bench/backfill-poll.py weka-c-rep2 --apply         # copies blocks into Prometheus
    python3 bench/backfill-poll.py --all --apply

Prometheus is pull-based. A window in which it was not running, or could not
reach the Sparks, is permanently empty -- restarting Colima resumes collection
from that moment and never goes back. `weka-c-rep2` ran 01:37 to 04:35 PT on
2026-08-18, straight through a Colima outage, so its dashboards are blank even
though the run itself completed normally at 91.86% cache read.

poll-metrics.sh was sampling the same endpoints over ssh every 60 s, entirely
independently of Prometheus. That file is the surviving record, and this turns
it into TSDB blocks.

## Three safety properties, in order of how badly each would hurt

**Nothing existing is modified or deleted.** Blocks are built in a staging
directory and copied in. No existing block is opened, moved or removed, and
--apply refuses if a block of the same ULID is already present.

**Recovered samples cannot collide with live samples.** Every series carries
`recovered="true"` and `run_id="..."`. A different label set is a different
series, so even if the window does overlap real data, the two sit side by side
rather than one winning. It also means anyone reading the dashboard can tell
which is which, which matters more than convenience.

**Windows that already have data are skipped by default.** Before emitting
anything, each metric is queried against the live Prometheus over the run's
window. Anything already covered is left alone, so a partial outage does not
produce a doubled line on the graph. --include-covered overrides that.

## What cannot be recovered, and why

poll-metrics.sh recorded `ts,node,metric,value` and dropped every label. For
most series that is harmless, since `node` was the only label that varied. For
two kinds it is fatal:

  *_bucket          a histogram bucket without `le` is not a bucket
  l2_adapters       two rows per scrape, state="active" and state="draining",
                    now indistinguishable

Any (ts, node, metric) key appearing more than once with differing values is
dropped for this reason and reported. Picking one of them would be inventing
data, which is worse than a gap.
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="*", help="run ids under runs/")
ap.add_argument("--all", action="store_true", help="every run that has a poll.csv")
ap.add_argument("--prom-url", default="http://localhost:9091")
ap.add_argument("--container", default="prometheus",
                help="docker container running Prometheus, for --apply")
ap.add_argument("--data-dir", default="/prometheus",
                help="Prometheus TSDB directory inside that container")
ap.add_argument("--out", default="/tmp/backfill", help="staging directory")
ap.add_argument("--apply", action="store_true", help="copy blocks in; default is a dry run")
ap.add_argument("--include-covered", action="store_true",
                help="emit metrics even where Prometheus already has data")
ap.add_argument("--no-clip", action="store_true",
                help="use the whole poll.csv rather than clipping to the run window")
ap.add_argument("--promtool", default="promtool")
a = ap.parse_args()

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"


def prom_covered(metric, start, end):
    """True if Prometheus already holds any sample for this metric in the window.

    Errors are treated as 'not covered' but reported, because refusing to
    recover data because a health check failed is the wrong default -- the
    recovered="true" label already makes collision impossible.
    """
    q = urllib.parse.urlencode({
        "query": metric, "start": start, "end": end, "step": "300",
    })
    url = f"{a.prom_url}/api/v1/query_range?{q}"
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            d = json.load(r)
    except Exception as e:                                   # noqa: BLE001
        print(f"    ! could not query Prometheus for {metric}: {e}")
        return False
    if d.get("status") != "success":
        return False
    return any(s.get("values") for s in d.get("data", {}).get("result", []))


def run_window(run):
    """(start, end) epoch seconds for the run itself, or None.

    poll-metrics.sh is started per run but nothing stops it when the run ends,
    so a poll.csv routinely outlives its arm. weka-a-router's covers 105.78 h
    beginning two days *after* that arm finished, because the poller was left
    running and kept appending under the old id. Backfilling it whole would
    stamp run_id="weka-a-router" on three days of unrelated samples.

    config.json's `started` is written 10 to 20 s before AIPerf begins, and
    duration_s is measured from AIPerf's own start, so pad both ends by a
    minute rather than trying to be exact.
    """
    d = RUNS / run
    cfg, summ = d / "config.json", d / "summary.json"
    if not cfg.exists() or not summ.exists():
        return None
    try:
        raw = re.sub(r'("block_size"\s*:)\s*(?=[,}\n])', r'\1 null', open(cfg).read())
        started = json.loads(raw).get("started")
        dur = json.load(open(summ)).get("duration_s")
    except (ValueError, OSError):
        return None
    if not started or not dur:
        return None
    from datetime import datetime, timezone
    t0 = int(datetime.strptime(started, "%Y-%m-%dT%H:%M:%SZ")
             .replace(tzinfo=timezone.utc).timestamp())
    return t0 - 60, t0 + int(dur) + 60


def load(run, window=None):
    """Return (samples, dropped, start, end). samples[(metric, node)] = [(ts, value)]."""
    p = RUNS / run / "poll.csv"
    seen = defaultdict(dict)          # (ts, node, metric) -> value
    collide = set()

    # Some poll.csv files have the `ts,node,metric,value` header and some do
    # not, depending on whether the poller created the file or appended to an
    # existing one. DictReader would silently take a data row as the header and
    # then drop every line, which is what made weka-a-router report "no usable
    # samples" against 282,992 rows of perfectly good data. Sniff instead.
    with open(p, newline="") as fh:
        first = fh.readline()
        fh.seek(0)
        if not first.startswith("ts,"):
            rows = csv.DictReader(fh, fieldnames=["ts", "node", "metric", "value"])
        else:
            rows = csv.DictReader(fh)
        for row in rows:
            # `series` is the post-2026-08-25 column and carries the full label
            # set; `metric` is the older one where labels were stripped. Files
            # of both shapes exist and both have to be readable.
            try:
                ts, node = int(row["ts"]), row["node"]
                metric = row.get("series") or row["metric"]
                val = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if window and not (window[0] <= ts <= window[1]):
                continue
            key = (ts, node, metric)
            if key in seen[metric] and seen[metric][key] != val:
                collide.add(metric)
            seen[metric][key] = val

    samples = defaultdict(list)
    lo = hi = None
    for metric, d in seen.items():
        if metric in collide or metric.endswith("_bucket"):
            continue
        for (ts, node, _), val in d.items():
            samples[(metric, node)].append((ts, val))
            lo = ts if lo is None else min(lo, ts)
            hi = ts if hi is None else max(hi, ts)
    for k in samples:
        samples[k].sort()
    return samples, sorted(collide), lo, hi


def openmetrics(run, samples, path):
    """Every family declared as a gauge.

    OpenMetrics requires a counter family `foo` to expose samples named
    `foo_total`, and these names already end in `_total`. Declaring them
    counters would make promtool reject the file. Type metadata is advisory --
    rate() and increase() work regardless -- so gauge is the honest, working
    choice. Grafana will label them gauges; that is the cost.
    """
    # A series may arrive as `name` or as `name{a="1",b="2"}`. Split the family
    # name off so the TYPE line is emitted once per family, and splice the
    # recovery labels into any existing set rather than appending a second
    # brace group, which would not parse.
    families = defaultdict(list)
    for (series, node), pts in samples.items():
        name = series.split("{", 1)[0]
        inner = series[len(name):].strip()
        inner = inner[1:-1] if inner.startswith("{") and inner.endswith("}") else ""
        families[name].append((series, inner, node, pts))

    with open(path, "w") as f:
        for name in sorted(families):
            f.write(f"# TYPE {name} gauge\n")
            f.write(f"# HELP {name} recovered from {run} poll.csv by backfill-poll.py\n")
            for _series, inner, node, pts in sorted(families[name], key=lambda r: (r[2], r[0])):
                labels = f'node="{node}",run_id="{run}",recovered="true"'
                if inner:
                    labels = f"{inner},{labels}"
                for ts, val in pts:
                    f.write(f"{name}{{{labels}}} {val} {ts}\n")
        f.write("# EOF\n")


targets = a.runs
if a.all:
    targets = sorted(d.name for d in RUNS.glob("*/") if (d / "poll.csv").exists())
if not targets:
    raise SystemExit("nothing to do: name a run, or use --all")

staging = Path(a.out)
staging.mkdir(parents=True, exist_ok=True)
built = []

for run in targets:
    print(f"\n=== {run}")
    if not (RUNS / run / "poll.csv").exists():
        print("  no poll.csv, skipping")
        continue

    win = None if a.no_clip else run_window(run)
    if win:
        print(f"  run window {win[0]} to {win[1]}  ({(win[1] - win[0]) / 3600:.2f} h)")
    elif not a.no_clip:
        print("  ! no run window derivable from config.json + summary.json;"
              " using the whole file")

    samples, collide, lo, hi = load(run, win)
    if not samples:
        if win:
            _, _, flo, fhi = load(run, None)
            if flo:
                print(f"  no samples inside the run window. poll.csv covers "
                      f"{flo} to {fhi}, which does not overlap it.")
                print("  The poller was left running under this run id after the "
                      "arm finished. --no-clip would backfill it anyway, mislabelled.")
                continue
        print("  no usable samples")
        continue
    print(f"  window {lo} to {hi}  ({(hi - lo) / 3600:.2f} h)")
    print(f"  {len(samples)} series, {sum(len(v) for v in samples.values()):,} samples")
    if collide:
        print(f"  dropped, labels not recoverable: {', '.join(collide)}")

    # One coverage query per metric, not per series, and cached -- a metric with
    # two nodes would otherwise be asked twice and could answer differently if
    # Prometheus were mid-scrape.
    covered = {}
    keep, skipped = {}, []
    for (metric, node), pts in samples.items():
        if not a.include_covered:
            if metric not in covered:
                covered[metric] = prom_covered(metric, lo, hi)
            if covered[metric]:
                skipped.append(metric)
                continue
        keep[(metric, node)] = pts
    if skipped:
        print(f"  already in Prometheus for this window, skipping: "
              f"{', '.join(sorted(set(skipped)))}")
    if not keep:
        print("  nothing left to backfill")
        continue

    om = staging / f"{run}.openmetrics"
    openmetrics(run, keep, om)
    out = staging / run
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cmd = [a.promtool, "tsdb", "create-blocks-from", "openmetrics", str(om), str(out)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        # The OpenMetrics file is already written and is the hard part, so say
        # where it is rather than losing it to a missing binary.
        print(f"  !! promtool not found on PATH. The OpenMetrics file is ready at:")
        print(f"     {om}")
        print(f"     Install it (it ships with Prometheus) or run the conversion in the")
        print(f"     Prometheus container:")
        print(f"       docker cp {om} {a.container}:/tmp/{om.name}")
        print(f"       docker exec {a.container} promtool tsdb create-blocks-from \\")
        print(f"         openmetrics /tmp/{om.name} /tmp/blocks-{run}")
        continue
    if r.returncode != 0:
        print(f"  !! promtool failed: {r.stderr.strip().splitlines()[-1] if r.stderr else r.returncode}")
        print(f"     {' '.join(cmd)}")
        continue

    blocks = [p for p in out.iterdir() if p.is_dir()]
    for b in blocks:
        try:
            meta = json.load(open(b / "meta.json"))
            n = meta.get("stats", {}).get("numSamples", "?")
            print(f"  block {b.name}  {n} samples")
        except Exception:                                    # noqa: BLE001
            print(f"  block {b.name}")
    built.append((run, blocks))

if not built:
    print("\nno blocks built")
    raise SystemExit(0)

print("\n" + "=" * 60)
if not a.apply:
    print("DRY RUN, nothing was copied. To install:")
    print(f"  python3 {Path(__file__).name} {' '.join(targets)} --apply")
    print("\nOr by hand, which is the same thing:")
    for run, blocks in built:
        for b in blocks:
            print(f"  docker cp {b} {a.container}:{a.data_dir}/{b.name}")
    print(f"  docker restart {a.container}")
    raise SystemExit(0)

for run, blocks in built:
    for b in blocks:
        dest = f"{a.data_dir}/{b.name}"
        # Refuse rather than overwrite. A ULID collision should be impossible,
        # so if it happens something is wrong and stopping is correct.
        check = subprocess.run(
            ["docker", "exec", a.container, "test", "-e", dest],
            capture_output=True)
        if check.returncode == 0:
            print(f"  !! {dest} already exists in {a.container}, refusing to overwrite")
            continue
        r = subprocess.run(["docker", "cp", str(b), f"{a.container}:{dest}"],
                           capture_output=True, text=True)
        print(f"  {'copied' if r.returncode == 0 else '!! failed'} {b.name}"
              f"{'' if r.returncode == 0 else ': ' + r.stderr.strip()}")

print(f"\nNow restart Prometheus so it opens the new blocks:")
print(f"  docker restart {a.container}")
print("\nThen in Grafana, recovered series carry recovered=\"true\" and a run_id.")
print("Existing panels will pick them up; to see only recovered data, add")
print('  {recovered="true"}')
