#!/usr/bin/env python3
"""Schema and distribution of the weka corpus, without assuming the nesting.
   ~/weka-inspect.py [traces.jsonl] [context_limit]
"""
import json, os, sys
from collections import Counter, defaultdict

F = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/traces/cc-weka/traces.jsonl")
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 131072

def walk(o, path="$"):
    """anything with both 'in' and 'out' is a request record"""
    if isinstance(o, dict):
        if "in" in o and "out" in o:
            yield path, o
        for k, v in o.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        for v in o:
            yield from walk(v, f"{path}[]")

paths = Counter(); per_session = []; ISL = []; OSL = []; keys = Counter()
n = 0
for line in open(F):
    if not line.strip(): continue
    n += 1
    obj = json.loads(line)
    if n == 1:
        print("=== top-level keys of record 1")
        for k, v in obj.items():
            t = type(v).__name__
            extra = f"[{len(v)}]" if isinstance(v, (list, dict)) else f" = {repr(v)[:70]}"
            print(f"    {k}: {t}{extra}")
    c = 0
    for p, r in walk(obj):
        paths[p] += 1; c += 1
        keys.update(r.keys())
        try:
            ISL.append(int(r["in"])); OSL.append(int(r["out"]))
        except (TypeError, ValueError):
            pass
    per_session.append(c)

print(f"\n=== {n} lines, {sum(per_session):,} request records")
print("\n  where requests live:")
for p, c in paths.most_common(8):
    print(f"    {c:>8,}  {p}")
print(f"\n  request fields: {', '.join(sorted(keys))}")

ps = sorted(per_session)
q = lambda a, p: a[min(int(len(a)*p), len(a)-1)]
print(f"\n  requests per session: p50 {q(ps,.5)}  p90 {q(ps,.9)}  max {ps[-1]}")

ISL.sort(); OSL.sort()
print(f"\n  ISL  p50 {q(ISL,.5):>9,}  p90 {q(ISL,.9):>9,}  max {ISL[-1]:>9,}")
print(f"  OSL  p50 {q(OSL,.5):>9,}  p90 {q(OSL,.9):>9,}  max {OSL[-1]:>9,}")
fit = sum(1 for x in ISL if x <= LIMIT)
print(f"\n  fit in {LIMIT:,}: {fit:,} of {len(ISL):,} = {fit/len(ISL)*100:.1f}%")
for lim in (32768, 65536, 131072, 262144):
    f2 = sum(1 for x in ISL if x <= lim)
    print(f"    <= {lim:>7,}: {f2/len(ISL)*100:>5.1f}%")
