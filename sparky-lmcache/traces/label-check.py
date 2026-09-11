#!/usr/bin/env python3
"""Test session-detail.py's compaction and subagent labels against the capture.

session-detail.py classifies a shrinking prompt as compaction or subagent from
its new-token fraction alone. That is a heuristic and nothing in this project
should rest on one when a label is available.

The session-grouped capture carries labels the flattening step throws away:

    {"t": 0.0, "in": 640, "out": 15, "hash_ids": [...],
     "api_time": 1.534, "ttft": 1.255, "type": "s", "model": "..."}

`type` is the harness's own classification. `model` is a second signal, because
subagents in these harnesses often run on a smaller model than the main thread.
If either lines up with the heuristic, the labels in findings.md are confirmed.
If neither does, they are guesses and have to be described as guesses.

**Which source produced the benchmarked file is not recorded anywhere**, and
traces/data holds five session-grouped captures. Naming one by hand would
cross-tab a different request set than the one measured and nothing in the
output would say so. So this takes the flattened file and every candidate, and
identifies the source by reproducing it: it replays convert-mooncake.py's
filtering and ordering on each candidate and checks whether the flattened file
is a prefix of the result, comparing hash_ids and lengths.

    python3 traces/label-check.py traces/data/agent-1000.jsonl traces/data/cc-weka-*.jsonl

A candidate that reproduces the file is the source. If none does, the flattened
file was built some other way and this script says so rather than guessing.
"""
import argparse
import json
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("flat", help="the flattened trace that was benchmarked")
ap.add_argument("candidates", nargs="+", help="session-grouped captures to test")
ap.add_argument("--max-input", type=int, default=131072)
ap.add_argument("--drop", type=float, default=0.30)
a = ap.parse_args()


def load_flat(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def convert(path, max_input):
    """convert-mooncake.py's transform, kept in step with it deliberately.

    Offsets are assigned in file order and rows are sorted by timestamp
    afterwards, so both have to happen here in that order or the hash_ids come
    out different and a real source looks like a non-match.
    """
    rows, offset = [], 0
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            top = 0
            for r in s.get("requests", []):
                ids = r.get("hash_ids") or []
                if ids:
                    top = max(top, max(ids))
                n_in, n_out = int(r.get("in", 0)), int(r.get("out", 0))
                if n_in > max_input or n_in < 1 or n_out < 1:
                    continue
                rows.append({
                    "t": float(r.get("t", 0.0)),
                    "in": n_in, "out": n_out,
                    "ids": [h + offset for h in ids],
                    "sid": s.get("id"),
                    "type": r.get("type"), "model": r.get("model"),
                })
            offset += top + 1
    rows.sort(key=lambda r: r["t"])
    return rows


flat = load_flat(a.flat)
N = len(flat)
print(f"{a.flat}: {N:,} requests\n")

# Timestamps are excluded from the comparison on purpose. The converter scales
# them by --time-scale and the value used is not recorded, so a mismatch there
# would say nothing. hash_ids and both lengths are scale-invariant, and hash_ids
# alone make an accidental match implausible.
def key(r, src):
    return (r["in"], r["out"], tuple(r["ids"])) if src else \
           (r["input_length"], r["output_length"], tuple(r["hash_ids"]))


want = [key(r, False) for r in flat]
match = None
print("candidate                                      requests  reproduces?")
for c in a.candidates:
    try:
        rows = convert(c, a.max_input)
    except (json.JSONDecodeError, KeyError) as e:
        print(f"  {c:<44} {'-':>9}  unreadable: {e}")
        continue
    got = [key(r, True) for r in rows]
    if len(got) >= N and got[:N] == want:
        verdict = "YES, exact prefix"
        match = match or (c, rows)
    elif len(got) < N:
        verdict = f"no, only {len(got):,} after filtering"
    else:
        agree = sum(1 for x, y in zip(got, want) if x == y)
        verdict = f"no, {agree:,}/{N:,} rows align"
    print(f"  {c:<44} {len(got):>9,}  {verdict}")

if not match:
    print("\n!! no candidate reproduces the flattened file.")
    print("   It was not produced by convert-mooncake.py from any of these, or")
    print("   it was truncated or re-sorted afterwards. Comparing labels against")
    print("   a different request set would be worse than not comparing them.")
    raise SystemExit(1)

src, rows = match
rows = rows[:N]
print(f"\nsource identified: {src}")
print(f"sessions covered by these {N:,} requests: "
      f"{len(set(r['sid'] for r in rows))}\n")

for field in ("type", "model"):
    c = Counter(r[field] for r in rows)
    if set(c) == {None}:
        print(f"!! no `{field}` field in this capture, it cannot confirm anything")
    else:
        print(f"`{field}` values: " + ", ".join(f"{k}={v}" for k, v in c.most_common()))
print()

# Warm fraction per session. hash_id_scope is "local", so the session is the
# correct trie scope, and after offsetting it matches the flat file's global one.
tries = defaultdict(dict)
for r in rows:
    node, warm = tries[r["sid"]], 0
    for h in r["ids"]:
        if h in node:
            node, warm = node[h], warm + 1
        else:
            break
    node = tries[r["sid"]]
    for h in r["ids"]:
        node = node.setdefault(h, {})
    r["blocks"], r["new"] = len(r["ids"]), len(r["ids"]) - warm

# Same three-way split as session-detail.py.
hi = defaultdict(int)
for r in rows:
    p, fn = hi[r["sid"]], r["new"] / max(1, r["blocks"])
    r["class"] = ("compaction?" if fn > 0.50 else "subagent?" if fn < 0.20
                  else "between") if p and r["in"] < p * (1 - a.drop) else "grew or flat"
    hi[r["sid"]] = max(p, r["in"])

CLASSES = ["compaction?", "subagent?", "between", "grew or flat"]
for field in ("type", "model"):
    vals = sorted({str(r[field]) for r in rows})
    if vals == ["None"]:
        continue
    w = max(12, max(len(v) for v in vals) + 2)
    print(f"heuristic class vs `{field}`")
    print(" " * 16 + "".join(f"{v:>{w}}" for v in vals) + f"{'total':>10}")
    for cl in CLASSES:
        sub = [r for r in rows if r["class"] == cl]
        cnt = Counter(str(r[field]) for r in sub)
        print(f"  {cl:<14}" + "".join(f"{cnt.get(v, 0):>{w},}" for v in vals)
              + f"{len(sub):>10,}")
    print()

    # A label that tracks the heuristic makes it redundant, which is the good
    # outcome. A label spread evenly across classes means one of the two is
    # measuring something other than what it claims.
    shrink = [r for r in rows if r["class"] in ("compaction?", "subagent?")]
    if shrink:
        best = defaultdict(int)
        for (cl, v), n in Counter((r["class"], str(r[field])) for r in shrink).items():
            best[cl] = max(best[cl], n)
        tot = sum(best.values())
        print(f"  {tot:,} of {len(shrink):,} shrinking requests ({100*tot/len(shrink):.0f}%) "
              f"sit in their class's most common `{field}` value\n")
