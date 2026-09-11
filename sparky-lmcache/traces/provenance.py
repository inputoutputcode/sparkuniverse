#!/usr/bin/env python3
"""Where did a flattened trace come from, and can its sessions share blocks?

label-check.py established that agent-1000.jsonl is not reproducible from any
session-grouped capture in traces/data. So the claim that rests on the
conversion is unsupported, and it is load-bearing:

    "The corpus scopes block ids per session, so cross-session sharing is
     inexpressible in the trace."

That is only true if convert-mooncake.py produced the file. It offsets each
session past the previous one's highest id, which leaves a signature that
survives into the flat file:

  **session-offset ids** -> each session's block ids form a disjoint contiguous
  interval. Two sessions can never name the same block, so sharing is
  impossible by construction and the 84.7% ceiling is intra-session by
  definition.

  **globally scoped ids** -> the intervals overlap, because ids refer to
  content rather than position. Sessions *could* share and evidently did not,
  which is a fact about the workload rather than about the conversion.

Sessions are approximated by first block, which is exact under offsetting and
close enough under global scoping to answer the question.

    python3 traces/provenance.py traces/data/*.jsonl
"""
import argparse
import json
import sys
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--pairs", action="store_true",
                help="also test which files are prefixes of which")
a = ap.parse_args()


def load(path):
    """Per-request mooncake rows, or None if the file is session-grouped."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "requests" in d:
                return None
            rows.append(d)
    return rows


loaded = {}
print(f"{'file':<48}{'reqs':>7}{'roots':>7}{'blocks':>9}  id scope")
for path in a.files:
    try:
        rows = load(path)
    except json.JSONDecodeError as e:
        print(f"{path:<48}  unreadable: {e}")
        continue
    if rows is None:
        print(f"{path:<48}{'':>7}{'':>7}{'':>9}  session-grouped source, skipped")
        continue
    if not rows:
        print(f"{path:<48}{0:>7}")
        continue
    loaded[path] = rows

    by_root = defaultdict(set)
    allids = set()
    for r in rows:
        ids = r.get("hash_ids") or []
        if not ids:
            continue
        by_root[ids[0]].update(ids)
        allids.update(ids)

    # Disjoint intervals is the offsetting signature. Overlap is not: it means
    # two sessions name the same block, which only global scoping allows.
    spans = sorted((min(v), max(v), k) for k, v in by_root.items())
    overlap = sum(1 for (l1, h1, _), (l2, h2, _) in zip(spans, spans[1:]) if l2 <= h1)
    shared = sum(1 for i, (_, _, k1) in enumerate(spans)
                 for _, _, k2 in spans[i + 1:]
                 if by_root[k1] & by_root[k2])

    # Second, independent signal. convert-mooncake.py assigns ids by
    # `offset += top + 1`, so its output starts at 0 and is fully dense. A
    # corpus that names blocks by content is not obliged to be either.
    lo, hi = min(allids), max(allids)
    dense = len(allids) / (hi - lo + 1)
    tag = f"[from {lo}, {dense:.0%} dense]"

    if len(spans) < 2:
        # Not undecidable: under offsetting two sessions cannot share a first
        # block, so a single root means either one session or global ids with a
        # shared system prompt, which is the interesting case.
        scope = ("one root: either a single session, or GLOBAL ids where "
                 "sessions share an opening block")
    elif shared:
        scope = f"GLOBAL, {shared} session pairs share block ids"
    elif overlap:
        scope = f"global-ish, {overlap} spans overlap but no ids shared"
    else:
        scope = "SESSION-OFFSET, disjoint intervals, sharing impossible"

    print(f"{path:<48}{len(rows):>7,}{len(by_root):>7}{len(allids):>9,}  {tag} {scope}")

if not loaded:
    sys.exit(0)

# Detail for the multi-session files, since the verdict above is a summary.
for path, rows in loaded.items():
    by_root = defaultdict(set)
    for r in rows:
        ids = r.get("hash_ids") or []
        if ids:
            by_root[ids[0]].update(ids)
    if len(by_root) < 2 or len(by_root) > 12:
        continue
    print(f"\n{path}")
    for k, v in sorted(by_root.items(), key=lambda kv: min(kv[1])):
        n = sum(1 for r in rows if (r.get("hash_ids") or [None])[0] == k)
        print(f"  root {k:>8}  ids {min(v):>8,} to {max(v):<8,}  "
              f"{len(v):>7,} distinct  {n:>5,} requests")

if a.pairs:
    print("\nprefix relationships (which file is the head of which)")
    key = lambda rows: [(r["input_length"], r["output_length"],
                         tuple(r.get("hash_ids") or [])) for r in rows]
    keys = {p: key(r) for p, r in loaded.items()}
    found = False
    for p1, k1 in keys.items():
        for p2, k2 in keys.items():
            if p1 != p2 and len(k1) < len(k2) and k2[:len(k1)] == k1:
                print(f"  {p1} is the first {len(k1):,} of {p2}")
                found = True
    if not found:
        print("  none. No file here is the head of another.")
