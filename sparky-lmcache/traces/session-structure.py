#!/usr/bin/env python3
"""How much of a trace is sessions growing, and how much is starting over?

Cache reuse depends on continuation. A turn that extends an earlier prompt
finds its prefix warm; a turn that starts a new session finds nothing, fills
L1 with content nobody asks for again, and evicts something that would have
been reused. A trace dominated by restarts makes any cache look bad, and the
result would say more about the capture than the system.

Method: build a trie over each request's hash_ids and record how many leading
blocks were already present. That is the request's warm prefix -- the blocks a
perfect cache of unlimited size would serve. Requests matching zero blocks are
cold starts.

This is an upper bound on reuse. It assumes infinite capacity, so the gap
between it and the measured cache-read rate is what capacity cost you.

    python3 traces/session-structure.py traces/data/agent-1000.jsonl
"""
import argparse
import json
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--block-size", type=int, default=512,
                help="tokens per hash_id; 512 for the agent traces, 64 for cc-weka")
ap.add_argument("--chains", type=int, default=5,
                help="how many of the longest chains to print")
a = ap.parse_args()

root = {}
rows = []

with open(a.trace) as fh:
    for i, line in enumerate(fh):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        ids = d.get("hash_ids") or []
        n_in = d.get("input_length", 0)

        # walk as far as the trie already goes
        node, matched = root, 0
        for h in ids:
            if h in node:
                node = node[h]
                matched += 1
            else:
                break
        # then insert the rest
        node = root
        for h in ids:
            node = node.setdefault(h, {})

        rows.append({
            "i": i,
            "blocks": len(ids),
            "warm": matched,
            "in": n_in,
            "out": d.get("output_length", 0),
            "root": ids[0] if ids else None,
        })

n = len(rows)
cold = [r for r in rows if r["warm"] == 0]
partial = [r for r in rows if 0 < r["warm"] < r["blocks"]]
full = [r for r in rows if r["warm"] and r["warm"] == r["blocks"]]

tot_blocks = sum(r["blocks"] for r in rows)
tot_warm = sum(r["warm"] for r in rows)

print(f"trace            {a.trace}")
print(f"requests         {n:,}")
print(f"distinct roots   {len(set(r['root'] for r in rows)):,}   "
      f"(first block of each request; a proxy for session count)")
print()
print(f"cold starts      {len(cold):>6,}  {100*len(cold)/n:>5.1f}%   no leading block seen before")
print(f"continuations    {len(partial):>6,}  {100*len(partial)/n:>5.1f}%   extend an earlier prompt")
print(f"exact repeats    {len(full):>6,}  {100*len(full)/n:>5.1f}%   every block seen before")
print()
print(f"blocks total     {tot_blocks:>9,}   ({tot_blocks*a.block_size:,} tokens)")
print(f"blocks warm      {tot_warm:>9,}   {100*tot_warm/tot_blocks:>5.1f}%  <- reuse ceiling at infinite capacity")
print()

# How much of each request was already warm. A healthy agentic trace is
# heavily weighted to the right: most turns extend a long existing prefix.
buckets = Counter()
for r in rows:
    if not r["blocks"]:
        continue
    f = r["warm"] / r["blocks"]
    buckets["0%" if f == 0 else
            "1-25%" if f <= .25 else
            "26-50%" if f <= .50 else
            "51-75%" if f <= .75 else
            "76-99%" if f < 1 else "100%"] += 1
print("warm fraction per request")
for k in ("0%", "1-25%", "26-50%", "51-75%", "76-99%", "100%"):
    c = buckets.get(k, 0)
    print(f"  {k:>7}  {c:>6,}  {100*c/n:>5.1f}%  {'#' * int(60*c/n)}")
print()

# Do sessions actually grow? Group by first block and look at how the prompt
# length develops. Restarts show up as a chain whose length does not increase.
chains = {}
for r in rows:
    chains.setdefault(r["root"], []).append(r)
longest = sorted(chains.values(), key=len, reverse=True)[:a.chains]
print(f"chains by shared first block: {len(chains):,} chains, "
      f"longest {len(longest[0]) if longest else 0} requests")
print()
for c in longest:
    lens = [r["in"] for r in c]
    grow = sum(1 for x, y in zip(lens, lens[1:]) if y > x)
    # Low monotonicity is not a restart. Agentic sessions interleave subagent
    # calls that carry a shorter prompt off the same system prefix, so length
    # oscillates while the prefix keeps growing. Read the cold-start count
    # above for restarts; this line only shows the shape.
    print(f"  {len(c):>4} turns  in: {lens[0]:>7,} -> {lens[-1]:>7,}  "
          f"length rises on {100*grow/max(1, len(lens)-1):>5.1f}% of turns   "
          f"{'monotonic growth' if grow > 0.8*(len(lens)-1) else 'interleaved subagents'}")
