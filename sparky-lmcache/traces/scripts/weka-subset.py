#!/usr/bin/env python3
"""Whole sessions, no truncation. A session qualifies if every well-formed
request in it fits the window. Malformed requests are skipped, not the session.
   ~/weka-subset.py [SESSIONS] [LIMIT] [SEED]
"""
import json, os, sys, random

S     = int(sys.argv[1]) if len(sys.argv) > 1 else 10
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 262144
SEED  = int(sys.argv[3]) if len(sys.argv) > 3 else 42
SRC = os.path.expanduser("~/traces/cc-weka/traces.jsonl")
DST = os.path.expanduser(f"~/traces/cc-weka-s{S}-{LIMIT//1024}k.jsonl")

def reqs(o):
    for m in o.get("requests", []):
        yield m
        for s in m.get("requests", []):
            yield s
def val(r, k):
    v = r.get(k)
    return int(v) if str(v).lstrip("-").isdigit() else None

pool = []
for line in open(SRC):
    if not line.strip(): continue
    o = json.loads(line)
    ins = [v for r in reqs(o) if (v := val(r, "in")) is not None]
    if not ins or max(ins) > LIMIT: continue
    pool.append(o)

subs_in_pool = sum(1 for o in pool if any(m.get("requests") for m in o["requests"]))
print(f"{len(pool)} sessions fit entirely in {LIMIT:,}  ({subs_in_pool} with subagents)")
if not pool: sys.exit("none - raise the limit")
random.Random(SEED).shuffle(pool); pool = pool[:S]

with open(DST, "w") as f:
    for o in pool: f.write(json.dumps(o) + "\n")

main = sum(len(o["requests"]) for o in pool)
sub  = sum(len(m.get("requests", [])) for o in pool for m in o["requests"])
sess_sub = sum(1 for o in pool if any(m.get("requests") for m in o["requests"]))
ISL = [v for o in pool for r in reqs(o) if (v := val(r, "in"))  is not None]
OSL = [v for o in pool for r in reqs(o) if (v := val(r, "out")) is not None]
blocks = {(o.get("id", i), h) for i, o in enumerate(pool) for r in reqs(o) for h in r.get("hash_ids", [])}
q = lambda a, p: sorted(a)[min(int(len(a)*p), len(a)-1)]
ws = len(blocks)*64

print(f"\nwrote {DST}")
print(f"  {len(pool)} sessions ({sess_sub} with subagents), {main} main turns, {sub} subagent requests")
print(f"  ISL p50 {q(ISL,.5):>8,}  p90 {q(ISL,.9):>8,}  max {max(ISL):>8,}  total {sum(ISL):>12,}")
print(f"  OSL p50 {q(OSL,.5):>8,}  p90 {q(OSL,.9):>8,}  total {sum(OSL):>12,}")
print(f"  working set {ws:,} tokens ({ws/1_060_864:.1f}x device KV)   reuse {(1-ws/sum(ISL))*100:.1f}%")
print(f"  concurrency ceiling {len(pool)}\n")
print(f"  {'model':<7}{'conc':>6}{'perfect cache':>16}{'no cache':>12}")
for name, pf, dec in (("Super", 1795, {2:29.1, 5:60.5}), ("Nano", 4785, {2:105.1, 5:184.1})):
    for c, d in dec.items():
        best = (ws/(pf*2) + sum(OSL)/(d*2))/60
        worst = (sum(ISL)/(pf*2) + sum(OSL)/(d*2))/60
        print(f"  {name:<7}{'c'+str(c):>6}{best:>14.0f}m{worst:>11.0f}m")
