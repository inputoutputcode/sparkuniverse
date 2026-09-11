#!/usr/bin/env python3
"""How evenly can a trace possibly load two nodes?

Under prefix affinity a router places whole sessions, not requests: splitting a
session across nodes is what P2P exists to make affordable, and in the
router-only arm it costs a full prefill. So the best balance any affinity-based
router can reach is the best two-way partition of the session sizes.

`agent-1000` has three sessions of 41.7 / 41.6 / 16.7%, so its only reachable
splits are 83/17, 58/42 and 100/0. The observed 70/30 was not a routing failure,
it was the trace. That is worth checking before choosing the next one.

Two load measures, because they answer different questions:

  **decode tokens** -- output volume. The Sparks saturate on decode bandwidth
  (nothing queues, batch slots sit idle), so this is the one that determines
  whether both GPUs stay busy.

  **prompt tokens** -- input volume, which sets prefill cost and cache pressure.

Reports the optimal partition, not a greedy one, so the number is a true ceiling
on what any placement policy could achieve.

    python3 traces/balance.py traces/data/cc-weka-*.jsonl
    python3 traces/balance.py traces/data/agent-1000.jsonl --block-size 512
"""
import argparse
import json
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--nodes", type=int, default=2)
ap.add_argument("--block-size", type=int, default=512,
                help="only used to report working set for flat traces")
a = ap.parse_args()


def sessions(path):
    """(name, prompt_tokens, decode_tokens, requests) per session.

    Handles both shapes: session-grouped captures keyed by their own id, and
    flat mooncake traces grouped by first block, which is the best proxy
    available once the session field is gone.
    """
    out = defaultdict(lambda: [0, 0, 0])
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
                    stack.extend(r.get("requests", []) or [])
                    try:
                        i, o = int(r["in"]), int(r["out"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if i < 1 or o < 1:
                        continue
                    out[sid][0] += i
                    out[sid][1] += o
                    out[sid][2] += 1
            else:
                ids = d.get("hash_ids") or []
                sid = ids[0] if ids else None
                out[sid][0] += d.get("input_length", 0)
                out[sid][1] += d.get("output_length", 0)
                out[sid][2] += 1
    return out


def best_split(weights, nodes):
    """Minimum achievable max-node share, over all assignments of whole sessions.

    Exact for two nodes via subset sums. Greedy for more, which is an upper
    bound rather than a ceiling, so it is labelled as such by the caller.
    """
    total = sum(weights)
    if total <= 0:
        return None
    if nodes == 2:
        reach = {0}
        for w in weights:
            reach |= {r + w for r in reach}
        half = total / 2
        best = min(reach, key=lambda s: abs(s - half))
        return max(best, total - best) / total
    loads = [0] * nodes
    for w in sorted(weights, reverse=True):
        loads[loads.index(min(loads))] += w
    return max(loads) / total


print(f"{'file':<44}{'sess':>5}{'reqs':>7}{'decode split':>15}{'prompt split':>15}")
for path in a.files:
    try:
        s = sessions(path)
    except (json.JSONDecodeError, OSError) as e:
        print(f"{path:<44}  unreadable: {e}")
        continue
    if not s:
        print(f"{path:<44}  no sessions found")
        continue
    dec = [v[1] for v in s.values()]
    pro = [v[0] for v in s.values()]
    reqs = sum(v[2] for v in s.values())
    bd, bp = best_split(dec, a.nodes), best_split(pro, a.nodes)
    exact = "" if a.nodes == 2 else "  (greedy, upper bound)"
    f = lambda x: f"{100*x:.0f}/{100*(1-x):.0f}" if x is not None else "n/a"
    print(f"{path:<44}{len(s):>5}{reqs:>7}{f(bd):>15}{f(bp):>15}{exact}")

# Detail for the candidates worth choosing between.
for path in a.files:
    try:
        s = sessions(path)
    except (json.JSONDecodeError, OSError):
        continue
    if not (1 < len(s) <= 40):
        continue
    tot_d = sum(v[1] for v in s.values()) or 1
    print(f"\n{path}  ({len(s)} sessions, concurrency ceiling {len(s)})")
    for sid, (p, d, n) in sorted(s.items(), key=lambda kv: -kv[1][1])[:12]:
        print(f"  {str(sid)[:22]:<24}{n:>5} reqs  prompt {p:>10,}  decode {d:>9,}"
              f"  {100*d/tot_d:>5.1f}%")
    if len(s) > 12:
        print(f"  ... and {len(s)-12} more")
