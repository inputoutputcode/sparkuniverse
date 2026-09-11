#!/usr/bin/env python3
"""Do two traces contain the same sessions, when their block ids cannot be compared?

The obvious check, take a session id and look for it elsewhere, does not work
here. agent-1000.jsonl is flat mooncake and has no session field, and the
cc-weka captures scope block ids per session while the agent-* family scopes
them globally. Nothing identifying survives both encodings.

Except the turn shape. Every request carries an input and an output length, and
those are properties of the conversation rather than of the file format. A
session of 35 turns with five-digit prompt lengths is a fingerprint: the chance
that two unrelated sessions share many exact (in, out) pairs is small.

So this groups each file into sessions, then scores every cross-file session
pair by how much of the smaller one's turn multiset the larger contains.

  ~100%  the same session, present in both corpora
  ~0%    unrelated corpora

Sessions come from the `id` field when the file is session-grouped, and from
the first block id when it is flat, which is the best proxy available once the
session field is gone.

    python3 traces/fingerprint.py traces/data/agent-1000.jsonl traces/data/cc-weka-*.jsonl
"""
import argparse
import json
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--min-turns", type=int, default=5,
                help="ignore sessions shorter than this, they match by chance")
ap.add_argument("--top", type=int, default=6)
a = ap.parse_args()


def load(path):
    """session key -> Counter of (input_length, output_length)."""
    out = defaultdict(Counter)
    flat = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "requests" in d:
                sid = d.get("id")
                stack = list(d["requests"])
                while stack:
                    r = stack.pop()
                    stack.extend(r.get("requests") or [])
                    try:
                        i, o = int(r["in"]), int(r["out"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    out[sid][(i, o)] += 1
            else:
                ids = d.get("hash_ids") or []
                flat.append((ids[0] if ids else None,
                             d.get("input_length", 0), d.get("output_length", 0)))
    for sid, i, o in flat:
        out[sid][(i, o)] += 1
    return {k: v for k, v in out.items() if sum(v.values()) >= a.min_turns}


loaded = {}
for p in a.files:
    try:
        loaded[p] = load(p)
    except (json.JSONDecodeError, OSError) as e:
        print(f"{p}: unreadable, {e}")
for p, s in loaded.items():
    print(f"{p:<48}{len(s):>4} sessions, "
          f"{sum(sum(v.values()) for v in s.values()):>6,} turns")

names = list(loaded)
if len(names) < 2:
    raise SystemExit("\nneed at least two readable files to compare")

print()
for i, p1 in enumerate(names):
    for p2 in names[i + 1:]:
        pairs = []
        for k1, c1 in loaded[p1].items():
            for k2, c2 in loaded[p2].items():
                # multiset intersection, normalized by the smaller session, so a
                # short session fully contained in a long one still scores 100%
                shared = sum((c1 & c2).values())
                if shared:
                    pairs.append((shared / min(sum(c1.values()), sum(c2.values())),
                                  shared, k1, k2))
        pairs.sort(reverse=True)
        print(f"{p1}\n  vs {p2}")
        if not pairs:
            print("    no session shares a single (input, output) pair. "
                  "Unrelated corpora.\n")
            continue
        for frac, shared, k1, k2 in pairs[:a.top]:
            print(f"    {frac:>6.1%}  {shared:>5} turns shared   "
                  f"{str(k1)[:24]:<26} <-> {str(k2)[:24]}")
        best = pairs[0][0]
        verdict = ("SAME SESSIONS, the corpora overlap" if best > 0.8 else
                   "partial overlap, worth reading the pairs above" if best > 0.2 else
                   "incidental matches only, the corpora are unrelated")
        print(f"    -> {verdict}\n")
