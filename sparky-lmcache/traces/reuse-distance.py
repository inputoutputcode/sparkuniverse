#!/usr/bin/env python3
"""How much cache does this corpus actually need, and what can sharing add?

    python3 traces/reuse-distance.py traces/data/cc-weka-s85-128k/ --chunk 4224
    python3 traces/reuse-distance.py traces/data/cc-weka-s85-128k/ --chunk 2128

Motivated by a wrong claim of mine. I had been quoting cache capacity as a
percentage of the corpus working set, 12.59M tokens, and calling Super
"oversubscribed" because a 3.60M-token arena holds 28.6% of it. Christian
pointed out that this is the wrong denominator: tokens from a session that has
finished are never asked for again, LRU evicts them, and their existence costs
nothing.

The project's own data agrees with him. Nano arm A held 16.04M tokens and read
88.96%. Arm B held 5.45M, 43% as much, and read 89.09%. Identical inside the
0.27-point floor. Everything above roughly 5M tokens on this corpus was dead
weight.

So the question is not "does the working set fit" but "is a chunk still
resident when its session comes back", which is a reuse distance question and
is answerable from the trace alone, without occupying the cluster for nine
hours per arm.

## What it simulates

Three cache topologies at each capacity, which is the point of the exercise:

    single   one cache, every request           -- a hypothetical single node
    split    two caches, sessions pinned to one -- arms A and B, KV-aware routing
    shared   one cache of twice the capacity    -- arm C's ceiling, if P2P were free

`split` against `shared` at a given per-node capacity is an upper bound on what
P2P can buy, because it assumes a peer fetch is as good as a local hit. Arm C
measured 92.00% against arm B's 89.09% on Nano, so the real figure lands below
this line, and the gap between them is the cost of the transport.

## Assumptions, and where they are weak

**Chunk identity.** If the corpus carries `hash_ids`, those are used directly
and the model is exact. Otherwise chunks are keyed by `(lineage, index)` and a
turn of L tokens is assumed to touch chunks 0..ceil(L/chunk)-1 of its lineage,
which is what prefix caching does when a session grows by appending. Run with
`--verbose` to see which was used.

**Compaction breaks the prefix chain.** When a turn's prompt is shorter than
its predecessor's, the harness has rewritten the context and the old chunks are
mostly dead. Modelled by starting a new lineage, so the reuse is not credited.
Without hash_ids this is a heuristic, and it is the largest source of error --
`session-detail.py` found these events split cleanly into compaction and
subagent shapes, and this collapses both into "new lineage".

**Arrival order.** LRU depends on interleaving, and the runs use
`--ignore-trace-delays`, so recorded timestamps do not govern. A concurrency-C
scheduler is simulated instead: C sessions active, one turn taken from each in
round robin, a new session admitted whenever one finishes. That is what AIPerf
does and it is why `--concurrency` is an argument here.

**No cross-session sharing.** A shared system prompt would be one identical
leading chunk across every session, and this model gives each lineage its own.
That understates reuse by at most one chunk per request. With hash_ids it is
handled correctly, because identical content hashes identically.

The output is a curve, and curves have knees. What matters is where this one
sits relative to the arenas actually deployed, not the absolute hit rates,
which carry all of the above error.
"""
import argparse
import json
import math
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("corpus", help="directory of one-session .json, or a .jsonl")
ap.add_argument("--chunk", type=int, default=4224,
                help="tokens per cache chunk: 2128 Nano, 4224 Super")
ap.add_argument("--hash-block", type=int, default=64,
                help="tokens per hash_id in the capture: 64 for cc-weka, 512 "
                     "for the agent traces. Only used when the corpus carries "
                     "hash_ids, but wrong here and every capacity is wrong.")
ap.add_argument("--concurrency", type=int, default=10,
                help="sessions in flight, matching the benchmark")
ap.add_argument("--capacities", default="",
                help="comma-separated token capacities; default sweeps a decade")
ap.add_argument("--verbose", action="store_true")
a = ap.parse_args()


# ----------------------------------------------------------------- loading
# Field names differ between the two corpus families and neither is documented.
# Guessing wrong here produces a plausible curve from nothing, so fail loudly
# with the keys that were actually present rather than defaulting to zero.
# `in` and `out` are what the cc-weka capture uses. The agent traces use
# `input_length`. Nothing documents either, and a missing key here would have
# silently produced a curve out of zeros, which is why the loader raises with
# the keys it did find rather than defaulting.
LEN_KEYS = ("in", "input_length", "isl", "num_input_tokens", "prompt_tokens",
            "input_tokens", "num_prompt_tokens")
HASH_KEYS = ("hash_ids", "block_hashes", "block_ids")


def field(d, names):
    for k in names:
        if k in d and d[k] is not None:
            return d[k]
    return None


def flatten(req, out):
    """Depth first, so a subagent's turns follow the parent turn that spawned it."""
    out.append(req)
    for child in req.get("requests") or []:
        flatten(child, out)


def load(path):
    p = Path(path)
    sessions = []
    if p.is_dir():
        files = sorted(p.glob("*.json"))
        if not files:
            raise SystemExit(f"no *.json in {p}")
        for f in files:
            sessions.append(json.loads(f.read_text()))
    else:
        for line in p.read_text().splitlines():
            if line.strip():
                sessions.append(json.loads(line))

    out = []
    for i, s in enumerate(sessions):
        reqs = []
        if isinstance(s.get("requests"), list):
            for r in s["requests"]:
                flatten(r, reqs)
        else:
            # A flat corpus: one object per request, no session nesting. Group
            # by whatever id it carries so the scheduler still has sessions.
            reqs = [s]
        sid = str(s.get("id") or s.get("session_id") or i)
        out.append((sid, reqs))
    return out


sessions = load(a.corpus)
n_req = sum(len(r) for _, r in sessions)
if not n_req:
    raise SystemExit("no requests found")

probe = next(r for _, rs in sessions for r in rs)
use_hash = field(probe, HASH_KEYS) is not None
if not use_hash and field(probe, LEN_KEYS) is None:
    raise SystemExit(
        "cannot find an input length or hash id field. Keys present on the "
        f"first request: {sorted(probe.keys())}")

if a.verbose:
    print(f"# {len(sessions)} sessions, {n_req:,} requests", file=sys.stderr)
    print(f"# chunk identity from {'hash_ids' if use_hash else 'lengths'}",
          file=sys.stderr)


# ------------------------------------------------------- chunk sequences
# One list of chunk keys per request, in prefix order.
def chunks_for(sid, reqs):
    seq = []
    lineage, prev_len = 0, -1
    for r in reqs:
        if use_hash:
            ids = field(r, HASH_KEYS) or []
            # hash ids are content addressed, so no lineage bookkeeping is
            # needed -- but they are *capture* blocks, 64 tokens here, and the
            # cache stores whole chunks of --chunk tokens. Keying on the raw ids
            # made one cache entry equal 64 tokens while capacity was counted in
            # 4224-token units, so a 16.04M-token arena simulated as 243k. Every
            # number in the first version of this table was wrong by that ratio.
            #
            # Group ids by the chunk their tokens fall in, and key each group by
            # its last id. The ids are prefix hashes, so the last one identifies
            # the whole span up to it, which is exactly the identity a prefix
            # cache uses. Grouping by token offset rather than by count also
            # survives --chunk not being a multiple of --hash-block: 4224/64 is
            # 66 exactly, but 2128/64 is 33.25.
            groups = {}
            for i, h in enumerate(ids):
                groups.setdefault(i * a.hash_block // a.chunk, []).append(h)
            seq.append([("h", g[-1]) for _, g in sorted(groups.items())])
            continue
        L = int(field(r, LEN_KEYS) or 0)
        if L <= 0:
            seq.append([])
            continue
        # Shorter than its predecessor means the context was rewritten. Credit
        # none of the old chunks; start a fresh lineage.
        if L < prev_len:
            lineage += 1
        prev_len = L
        n = math.ceil(L / a.chunk)
        seq.append([(sid, lineage, i) for i in range(n)])
    return seq


per_session = [(sid, chunks_for(sid, reqs)) for sid, reqs in sessions]


# ------------------------------------------------------------- scheduling
# C sessions in flight, one turn each per round, a new session admitted as soon
# as one drains. Approximates AIPerf under --ignore-trace-delays, where the
# recorded gaps are discarded and arrival is governed by slot availability.
def arrival_order():
    pending = list(range(len(per_session)))
    cursor = [0] * len(per_session)
    active = []
    order = []
    while pending or active:
        while pending and len(active) < a.concurrency:
            active.append(pending.pop(0))
        drained = []
        for s in active:
            i = cursor[s]
            if i >= len(per_session[s][1]):
                drained.append(s)
                continue
            order.append((s, i))
            cursor[s] += 1
        for s in drained:
            active.remove(s)
    return order


order = arrival_order()


# -------------------------------------------------------------------- LRU
class LRU:
    __slots__ = ("cap", "d")

    def __init__(self, cap):
        self.cap = cap
        self.d = OrderedDict()

    def touch(self, key):
        """True if already resident. Inserts either way, evicting oldest."""
        if key in self.d:
            self.d.move_to_end(key)
            return True
        self.d[key] = None
        if len(self.d) > self.cap:
            self.d.popitem(last=False)
        return False


def simulate(cap_chunks, topology):
    if topology == "single":
        caches, pick = [LRU(cap_chunks)], lambda s: 0
    elif topology == "split":
        # Sticky routing: a session always lands on the same node, which is what
        # the KV router does when it has an overlap to score.
        caches = [LRU(cap_chunks), LRU(cap_chunks)]
        pick = lambda s: s % 2
    elif topology == "shared":
        caches, pick = [LRU(2 * cap_chunks)], lambda s: 0
    else:
        raise ValueError(topology)

    hit = tot = 0
    for s, i in order:
        c = caches[pick(s)]
        for key in per_session[s][1][i]:
            tot += 1
            # A prefix cache stops at the first miss: everything after it is
            # recomputed even if it happens to be resident. Modelling every
            # chunk as an independent lookup would overstate the hit rate.
            if c.touch(key):
                hit += 1
            else:
                for rest in per_session[s][1][i][per_session[s][1][i].index(key) + 1:]:
                    tot += 1
                    c.touch(rest)
                break
    return hit, tot


# ------------------------------------------------------------------ sweep
if a.capacities:
    caps = [int(x) for x in a.capacities.split(",")]
else:
    caps = [int(1e6 * x) for x in
            (0.5, 1, 1.5, 2, 2.5, 3, 3.6, 4.5, 5.45, 7, 9, 12, 16.04, 24, 40)]

DEPLOYED = {
    2_883_584: "Super arm A as deployed, 16 GB",
    3_604_000: "Super arm A at 20 GB",
    5_450_000: "Nano arm B, device + L1",
    16_039_467: "Nano arm A, 56 GB",
}

print(f"corpus      {a.corpus}")
print(f"chunk       {a.chunk} tokens")
print(f"concurrency {a.concurrency}")
print(f"requests    {n_req:,}   sessions {len(sessions):,}")
print()
print(f"{'capacity/node':>14} {'chunks':>8} {'split':>8} {'shared':>8} {'P2P ceiling':>12}   note")

inf_hit, inf_tot = simulate(10 ** 9, "single")
for cap in caps:
    cc = max(1, cap // a.chunk)
    sh, st = simulate(cc, "split")
    hh, ht = simulate(cc, "shared")
    split = 100 * sh / st if st else 0
    shared = 100 * hh / ht if ht else 0
    note = DEPLOYED.get(cap, "")
    for k, v in DEPLOYED.items():
        if abs(cap - k) / k < 0.02:
            note = v
    print(f"{cap:>14,} {cc:>8,} {split:>7.2f}% {shared:>7.2f}% {shared-split:>+11.2f}   {note}")

print()
print(f"infinite capacity ceiling: {100*inf_hit/inf_tot:.2f}%")
print()
print("split is arms A and B. shared is the best P2P could do if a peer fetch")
print("were free. The measured arm C sits between them.")
