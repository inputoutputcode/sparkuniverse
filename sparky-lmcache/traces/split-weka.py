#!/usr/bin/env python3
"""Split a session-grouped JSONL capture into the directory AIPerf's Weka loader wants.

`weka-subset.py` writes one JSON object per line. AIPerf's `weka_trace` loader
wants something else entirely, and the mismatch is not obvious from either side:

    dataset/loader/weka_trace.py:1103   path.suffix != ".json"        -> rejected
    dataset/loader/weka_trace.py:1106   orjson.loads(path.read_bytes())
    dataset/loader/weka_trace.py:1112   WekaTrace.model_validate(blob)
    dataset/loader/weka_trace.py:1158   sorted(self._path.glob("*.json"))

One `WekaTrace` per file, and a directory of them for a corpus. Feeding it the
JSONL fails after the run has already registered itself:

    ValueError: invalid JSON: unexpected content after document:
    line 2 column 1 (char 243460)

which is json.load() reading the first session and finding a second one behind
it. A JSON array fails the same way for a different reason, since the validator
expects an object rather than a list.

    python3 traces/split-weka.py traces/data/cc-weka-s85-128k.jsonl \\
                                traces/data/cc-weka-s85-128k/
"""
import argparse
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("source", help="session-grouped .jsonl from weka-subset.py")
ap.add_argument("dest", help="directory to write one .json per session into")
a = ap.parse_args()

dest = Path(a.dest)
dest.mkdir(parents=True, exist_ok=True)

# Stale files from an earlier split would be globbed in alongside the new ones
# and replayed as extra sessions, which is the kind of contamination that shows
# up as an unexplained change in the reuse ceiling.
stale = sorted(dest.glob("*.json"))
if stale:
    print(f"removing {len(stale)} existing .json in {dest}")
    for p in stale:
        p.unlink()

n = req = 0
for line in Path(a.source).read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    trace = json.loads(line)
    sid = str(trace.get("id") or f"trace{n:04d}")
    # Session ids in this corpus are hex, but do not trust that for a filename.
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", sid)[:64]
    (dest / f"{safe}.json").write_text(json.dumps(trace))
    stack = list(trace.get("requests") or [])
    while stack:
        r = stack.pop()
        stack.extend(r.get("requests") or [])
        req += 1
    n += 1

written = len(sorted(dest.glob("*.json")))
print(f"{n} sessions, {req:,} requests -> {dest}")
if written != n:
    print(f"!! wrote {n} but {written} files are present. Duplicate session ids "
          f"collide on filename and one would silently overwrite the other.")
    raise SystemExit(1)
