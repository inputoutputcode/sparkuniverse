#!/usr/bin/env python3
"""Add expected_requests and complete to summaries written before that check existed.

    python3 bench/backfill-complete.py --dry-run
    python3 bench/backfill-complete.py

The completeness check was added on 2026-08-18, after a 128-request fragment
stood in for a 3,485-request arm and would have been read as a 14-point P2P
result. Runs from before that date have no `complete` field, so a listing shows
`complete=None` for six finished arms and it reads like a failure. It is not:
absent means "predates the check".

Two things stop summarize-run.py from filling it in by itself.

It rebuilds the whole summary from the export and stamps `recovered: True`,
which would be a provenance lie for runs that were never recovered. This script
only adds the two missing fields and never touches a measured value.

And it reads requests_sent from config.json, which for every weka_trace run is
malformed: BLOCK_SIZE is deliberately empty for that loader and was interpolated
bare, producing `"block_size": ,`. So the parse fails, the field is dropped
silently, and the run still looks incomplete. Repair that here too, since a
broken config.json also blocks any future recovery of these runs.
"""
import argparse
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true", help="report, change nothing")
ap.add_argument("--runs", default=None, help="only this run id")
a = ap.parse_args()

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"

# `"block_size": ,` and `"block_size": }` are the two shapes an empty
# interpolation leaves behind. Anchored on the key so nothing else can match.
BARE = re.compile(r'("block_size"\s*:)\s*(?=[,}\n])')


def repair_config(path):
    """Return requests_sent, fixing invalid JSON in place if that is what blocks it."""
    if not path.exists():
        return None, "no config.json"
    raw = path.read_text()
    try:
        return json.loads(raw).get("requests_sent"), None
    except ValueError:
        pass

    fixed = BARE.sub(r"\1 null", raw)
    try:
        want = json.loads(fixed).get("requests_sent")
    except ValueError as e:
        # Do not guess at a file broken some other way. Leaving it alone and
        # saying so is better than writing something that parses but is wrong.
        return None, f"config.json invalid, not the block_size bug: {e}"
    if not a.dry_run:
        path.write_text(fixed)
    return want, "repaired config.json"


targets = sorted(d for d in RUNS.glob("*/") if (d / "summary.json").exists())
if a.runs:
    targets = [d for d in targets if d.name == a.runs]
if not targets:
    raise SystemExit("no summaries found")

changed = skipped = 0
for d in targets:
    sp = d / "summary.json"
    s = json.loads(sp.read_text())

    if "complete" in s:
        print(f"  {d.name:24} already has complete={s['complete']}")
        skipped += 1
        continue

    want, note = repair_config(d / "config.json")
    got = s.get("requests") or 0
    if not want:
        print(f"  {d.name:24} SKIP, {note or 'no requests_sent'}")
        skipped += 1
        continue

    s["expected_requests"] = want
    # Same 0.99 threshold run-scenario.sh uses, so a backfilled summary and a
    # natively written one agree. 3,484 of 3,485 passes; 128 of 3,485 does not.
    s["complete"] = bool(got >= 0.99 * want)
    if not a.dry_run:
        sp.write_text(json.dumps(s, indent=1))
    print(f"  {d.name:24} {got:>6.0f}/{want:<6} complete={s['complete']}"
          f"{'  (' + note + ')' if note else ''}")
    changed += 1

print(f"\n{changed} updated, {skipped} skipped{'  [dry run, nothing written]' if a.dry_run else ''}")
