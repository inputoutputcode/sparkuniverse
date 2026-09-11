#!/usr/bin/env python3
"""Where do the new tokens in an agentic trace come from?

session-structure.py says 15.3% of agent-1000 is content never seen before.
That is a lot for a workload described as "a prompt that grows a little each
turn", so this script asks where it comes from and whether the sessions are
really interleaved.

Four questions:

  1. How many sessions, and do their turns interleave or run in blocks?
     Grouping by first block id is a proxy. Two conversations sharing a system
     prompt share a first block, so a "root" may be a repo rather than a
     session.

  2. How many compaction events? Compaction summarizes a full context back to
     a smaller baseline, which produces a sharp drop in prompt length and a
     prefix nothing has seen. It should show up as a large negative step with
     a low warm fraction.

  3. How many new tokens per turn, and how does that compare to the previous
     turn's output? The model's own answer is appended to the next prompt, so
     output length is a floor on new content. Anything above it is tool
     results, file reads and user text.

  4. Does new content correlate with the previous output, or arrive in bursts?

    python3 traces/session-detail.py traces/data/agent-1000.jsonl
"""
import argparse
import json
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--block-size", type=int, default=512)
ap.add_argument("--drop", type=float, default=0.30,
                help="prompt shrinking by more than this fraction counts as compaction")
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
        node, warm = root, 0
        for h in ids:
            if h in node:
                node = node[h]
                warm += 1
            else:
                break
        node = root
        for h in ids:
            node = node.setdefault(h, {})
        rows.append({"i": i, "root": ids[0] if ids else None,
                     "blocks": len(ids), "warm": warm,
                     "new": len(ids) - warm,
                     "in": d.get("input_length", 0),
                     "out": d.get("output_length", 0)})

n = len(rows)
B = a.block_size

# 1. sessions, and whether they interleave
by_root = defaultdict(list)
for r in rows:
    by_root[r["root"]].append(r["i"])

print(f"requests {n:,}, distinct first blocks {len(by_root)}\n")
print("first block   turns   index range        contiguous?")
for rt, idx in sorted(by_root.items(), key=lambda kv: -len(kv[1])):
    span = idx[-1] - idx[0] + 1
    # contiguous means the turns occupy a solid run of the file
    frac = len(idx) / span
    print(f"  {str(rt)[:10]:>10} {len(idx):>7}   {idx[0]:>5} to {idx[-1]:<6}  "
          f"{frac:>5.0%} dense  {'contiguous block' if frac > 0.9 else 'interleaved with others'}")

# how often the root changes from one request to the next. Near 0 means the
# file is sorted by session. Near 1 means turns are shuffled together.
flips = sum(1 for x, y in zip(rows, rows[1:]) if x["root"] != y["root"])
print(f"\nroot changes between consecutive requests: {flips:,} of {n-1:,} "
      f"({100*flips/(n-1):.1f}%)")

# 2. compaction, separated from subagents
#
# Both shrink the prompt, so the shrink alone does not identify either. A
# subagent carries a subset of context the cluster has already seen, so it
# arrives warm. A compaction writes a fresh summary, so the new prompt is
# mostly text nothing has seen. The new fraction separates them.
shrank = []
prev_in = defaultdict(int)
for r in rows:
    p = prev_in[r["root"]]
    if p and r["in"] < p * (1 - a.drop):
        frac_new = r["new"] / max(1, r["blocks"])
        shrank.append((r["i"], p, r["in"], r["new"] * B, frac_new))
    prev_in[r["root"]] = max(prev_in[r["root"]], r["in"])

comp = [s for s in shrank if s[4] > 0.50]
sub = [s for s in shrank if s[4] < 0.20]
mid = len(shrank) - len(comp) - len(sub)

print(f"\nprompt shrank by more than {a.drop:.0%} against the session high water mark: "
      f"{len(shrank)} times")
print(f"  mostly new (>50%), compaction-shaped      {len(comp):>4}")
print(f"  mostly warm (<20%), subagent-shaped       {len(sub):>4}")
print(f"  in between                                {mid:>4}")

print("\ncompaction-shaped requests")
for i, p, now, newtok, f in comp[:12]:
    print(f"  request {i:>4}  {p:>7,} -> {now:>7,} tokens, {newtok:>7,} new ({f:>4.0%})")
if len(comp) > 12:
    print(f"  ... and {len(comp)-12} more")
if comp:
    print(f"  new tokens carried by these: {sum(c[3] for c in comp):,}")
    # The post-compaction prompt size is the harness's baseline, and it is what
    # the next growth cycle starts from. Published harness figures describe the
    # trigger point rather than this, so it has to come from the trace.
    after = sorted(c[2] for c in comp)
    q = lambda p: after[min(len(after) - 1, int(p * len(after)))]
    print(f"  prompt after compaction: min {after[0]:,}  p25 {q(.25):,}  "
          f"p50 {q(.50):,}  p75 {q(.75):,}  max {after[-1]:,}")

# 3 and 4. where the new tokens come from
# Compared in aggregate, not per turn. The sessions interleave, so rows[i-1] is
# usually a different session and a per-turn pairing would be meaningless.
new_tok = [r["new"] * B for r in rows]
tot_new = sum(new_tok)
tot_out = sum(r["out"] for r in rows)

print(f"\nnew tokens total      {tot_new:>12,}")
print(f"  mean per request    {tot_new/n:>12,.0f}")
print(f"previous-turn output  {tot_out:>12,}   "
      f"{100*tot_out/tot_new if tot_new else 0:.1f}% of new content")
print(f"  unexplained by output {tot_new - tot_out:>10,}   "
      f"tool results, file reads, user text")

# Counts alone are misleading. A bucket holding 7% of requests can hold most
# of the new content, and it is the content that has to be prefilled.
buckets, mass = Counter(), Counter()
for v in new_tok:
    k = ("0" if v == 0 else
         "1-512" if v <= 512 else
         "513-2k" if v <= 2048 else
         "2k-8k" if v <= 8192 else
         "8k-32k" if v <= 32768 else "over 32k")
    buckets[k] += 1
    mass[k] += v
print("\nnew tokens per request      requests            new tokens in bucket")
for k in ("0", "1-512", "513-2k", "2k-8k", "8k-32k", "over 32k"):
    c, m = buckets.get(k, 0), mass.get(k, 0)
    print(f"  {k:>8}  {c:>5,}  {100*c/n:>5.1f}%   "
          f"{m:>10,}  {100*m/max(1,tot_new):>5.1f}%  {'#' * int(40*m/max(1,tot_new))}")

top = sorted(rows, key=lambda r: -r["new"])[:5]
print("\nlargest single injections of new content")
for r in top:
    print(f"  request {r['i']:>4}  {r['new']*B:>8,} new tokens of {r['in']:>7,} prompt "
          f"({100*r['new']/max(1,r['blocks']):>4.0f}% of the request)")
