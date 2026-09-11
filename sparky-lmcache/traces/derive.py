#!/usr/bin/env python3
"""Is one flat trace a filtered subset of another, and what was filtered out?

`provenance.py --pairs` showed agent-1000.jsonl is *not* a prefix of
64k_400_90kv_agent_new_noschedule.jsonl, while both share roots 0/506/814 and
the same ~2M block id universe. So it is derived but not a head, and the
selection rule is unrecorded.

This matches the target's rows against the source greedily in order. A complete
match means the target is an order-preserving subset, and the rows the match
skipped are exactly what the selection dropped. Characterizing those is what
identifies the rule: a length cap shows up as every skipped row exceeding a
bound the kept rows never reach.

Matches on (input_length, output_length, hash_ids). Timestamps are excluded
because a converter may rescale them, and hash_ids alone make an accidental
match implausible.

    python3 traces/derive.py traces/data/agent-1000.jsonl \\
        traces/data/64k_400_90kv_agent_new_noschedule.jsonl
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("target", help="the derived file, e.g. agent-1000.jsonl")
ap.add_argument("source", help="the file it is suspected to come from")
a = ap.parse_args()


def rows(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                out.append((d.get("input_length", 0), d.get("output_length", 0),
                            tuple(d.get("hash_ids") or []), d))
    return out


tgt, src = rows(a.target), rows(a.source)
print(f"target {a.target}: {len(tgt):,} rows")
print(f"source {a.source}: {len(src):,} rows\n")

key = lambda r: (r[0], r[1], r[2])
i = 0
skipped = []
matched = 0
for s in src:
    if i < len(tgt) and key(s) == key(tgt[i]):
        i += 1
        matched += 1
        if i == len(tgt):
            consumed = src.index(s) + 1 if False else None
            break
    else:
        if i < len(tgt):
            skipped.append(s)

if matched != len(tgt):
    print(f"!! only {matched:,} of {len(tgt):,} target rows matched in order.")
    print("   Not an order-preserving subset. It was re-sorted, re-sampled, or")
    print("   came from a different source. Do not read the skip analysis below")
    print("   as a selection rule.")
else:
    used = matched + len(skipped)
    print(f"target IS an order-preserving subset of source.")
    print(f"  consumed the first {used:,} source rows ({100*used/len(src):.1f}%)")
    print(f"  kept {matched:,}, dropped {len(skipped):,} "
          f"({100*len(skipped)/max(1,used):.1f}% of the span)")

if not skipped:
    print("\nnothing was dropped: the target is a plain prefix.")
    raise SystemExit

kept_in = [r[0] for r in tgt]
kept_out = [r[1] for r in tgt]
drop_in = [r[0] for r in skipped]
drop_out = [r[1] for r in skipped]
q = lambda v, p: sorted(v)[min(int(p * len(v)), len(v) - 1)] if v else 0

print("\n                 kept                        dropped")
print(f"  input   min {min(kept_in):>8,}  p50 {q(kept_in,.5):>8,}  max {max(kept_in):>8,}"
      f"   min {min(drop_in):>8,}  p50 {q(drop_in,.5):>8,}  max {max(drop_in):>8,}")
print(f"  output  min {min(kept_out):>8,}  p50 {q(kept_out,.5):>8,}  max {max(kept_out):>8,}"
      f"   min {min(drop_out):>8,}  p50 {q(drop_out,.5):>8,}  max {max(drop_out):>8,}")

# A clean threshold is the signature of a length filter. State it only when the
# two ranges genuinely do not overlap, otherwise say the rule is something else.
print()
if min(drop_in) > max(kept_in):
    print(f"  every dropped row has input > {max(kept_in):,} and every kept row is at"
          f" or below it.\n  Selection rule: an input-length cap between "
          f"{max(kept_in):,} and {min(drop_in):,}.")
elif min(drop_out) > max(kept_out):
    print(f"  clean split on output length at {max(kept_out):,}.")
elif all(r[1] < 1 or r[0] < 1 for r in skipped):
    print("  every dropped row has a zero input or output length. Selection rule:"
          " drop empty requests.")
else:
    over = sum(1 for v in drop_in if v > max(kept_in))
    empty = sum(1 for r in skipped if r[0] < 1 or r[1] < 1)
    print(f"  no clean threshold. Of {len(skipped):,} dropped rows, {over:,} exceed the"
          f" kept maximum input and {empty:,} are empty.")
    print("  The remainder were dropped for some other reason, so the rule is not"
          " a simple filter.")
    for r in skipped[:5]:
        print(f"    dropped: in={r[0]:>8,} out={r[1]:>6,} blocks={len(r[2]):>5}")
