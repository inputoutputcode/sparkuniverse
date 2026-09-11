#!/usr/bin/env python3
"""Tabulate completed runs, and report the spread when they are repeats.

Two uses. Given repeats of one configuration it prints the noise floor, which
is the number every later comparison has to beat. Given different arms it
prints them side by side and, if a floor has been measured, says whether the
differences clear it.

    python3 bench/compare-runs.py weka-c-rep1 weka-c-rep2 weka-c-rep3
    python3 bench/compare-runs.py weka-c-p2p weka-b-l1only --floor 0.4

Reads runs/<id>/summary.json, which run-scenario.sh writes.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--floor", type=float, default=None,
                help="cache-read spread in points from a noise-floor sweep")
ap.add_argument("--repeats", action="store_true",
                help="treat the runs as repeats of one config and report the spread")
a = ap.parse_args()

REPO = Path(__file__).resolve().parent.parent
FIELDS = [
    ("cache_read_pct", "cache read %", 2),
    ("req_per_hour", "requests/hour", 1),
    ("output_tok_s", "output tok/s", 1),
    ("ttft_p50_ms", "TTFT p50 ms", 0),
    ("ttft_p90_ms", "TTFT p90 ms", 0),
    ("total_prompt_tokens", "prompt tokens", 0),
    ("cached_prompt_tokens", "cache read tokens", 0),
    ("duration_s", "duration s", 0),
]

data = {}
for r in a.runs:
    p = REPO / "runs" / r / "summary.json"
    if not p.exists():
        print(f"!! no summary for {r} at {p}")
        continue
    data[r] = json.load(open(p))
if len(data) < 2:
    raise SystemExit("need at least two runs with a summary.json")

names = list(data)
w = max(18, max(len(n) for n in names) + 2)
print(f"{'':<20}" + "".join(f"{n:>{w}}" for n in names))
for key, label, dp in FIELDS:
    vals = [data[n].get(key) for n in names]
    cells = "".join(
        f"{v:>{w},.{dp}f}" if isinstance(v, (int, float)) else f"{'n/a':>{w}}"
        for v in vals)
    print(f"{label:<20}{cells}")

nums = [data[n].get("cache_read_pct") for n in names]
if all(isinstance(v, (int, float)) for v in nums):
    lo, hi = min(nums), max(nums)
    spread = hi - lo
    mean = sum(nums) / len(nums)
    print()
    if a.repeats or len(set(a.runs)) == len(a.runs) and a.repeats:
        pass
    print(f"cache read: min {lo:.2f}  max {hi:.2f}  mean {mean:.2f}  "
          f"spread {spread:.2f} points")

    if a.repeats:
        # Nothing else in this project may claim a difference smaller than this.
        print()
        print(f"NOISE FLOOR = {spread:.2f} points of cache read.")
        print(f"Any arm-to-arm difference at or below {spread:.2f} points is not")
        print("a result. Quote it as 'no difference this measurement can resolve'.")
        for key, label, dp in FIELDS[1:5]:
            v = [data[n].get(key) for n in names]
            if all(isinstance(x, (int, float)) for x in v) and min(v):
                print(f"  {label:<16} spread {max(v)-min(v):,.{dp}f} "
                      f"({100*(max(v)-min(v))/min(v):.1f}%)")
    elif a.floor is not None:
        print()
        verdict = "CLEARS the floor" if spread > a.floor else "INSIDE the floor, not a result"
        print(f"floor {a.floor:.2f} points, observed spread {spread:.2f} -> {verdict}")
