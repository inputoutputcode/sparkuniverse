#!/usr/bin/env python3
"""Write runs/<id>/summary.json from an AIPerf export that already exists.

run-scenario.sh extracts the summary itself, at the end. If the wrapper dies
after AIPerf finishes but before that step, three hours of completed work has no
summary and compare-runs.py cannot see it, even though every artifact is on
disk. weka-c-rep1 landed in exactly that state: profile_export_aiperf.json
written at 00:37, wrapper killed, no summary.

    python3 bench/summarize-run.py weka-c-rep1
    python3 bench/summarize-run.py --all

Same fields and same metric names as run-scenario.sh, so a recovered summary is
indistinguishable from one written normally.
"""
import argparse
import glob
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="*", help="run ids under runs/")
ap.add_argument("--all", action="store_true",
                help="every run with an export but no summary")
ap.add_argument("--force", action="store_true", help="overwrite an existing summary")
a = ap.parse_args()

REPO = Path(__file__).resolve().parent.parent


# No `-> dict | None` annotation. That is 3.10 syntax and it is evaluated at
# def time, so on the Mac's older python3 the script dies before running:
#   TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'
def summarize(run_dir):
    hits = glob.glob(str(run_dir / "aiperf/**/profile_export_aiperf.json"), recursive=True)
    if not hits:
        print(f"  {run_dir.name}: no profile_export_aiperf.json")
        return None
    j = json.load(open(hits[0]))

    def m(key, stat="avg"):
        v = j.get(key)
        return v.get(stat) if isinstance(v, dict) else v

    s = {
        "requests":            m("request_count"),
        "duration_s":          m("benchmark_duration"),
        "req_per_hour":        (m("request_throughput") or 0) * 3600,
        "output_tok_s":        m("output_token_throughput"),
        "output_tok_s_user":   m("output_token_throughput_per_user"),
        "total_output_tokens": m("total_usage_completion_tokens") or m("total_output_tokens"),
        "total_prompt_tokens": m("total_usage_prompt_tokens") or m("total_isl"),
        "cached_prompt_tokens": m("total_usage_prompt_cache_read_tokens"),
        "cache_read_pct":      m("overall_usage_prompt_cache_read_pct"),
        "ttft_p50_ms":         m("time_to_first_token", "p50"),
        "ttft_p90_ms":         m("time_to_first_token", "p90"),
        "itl_p50_ms":          m("inter_token_latency", "p50"),
        "isl_p50":             m("input_sequence_length", "p50"),
        "osl_p50":             m("output_sequence_length", "p50"),
        "recovered":           True,
    }
    # config.json records what the run was asked to send. Without it a recovered
    # summary cannot say whether the run finished, and a partial one looks
    # identical to a complete one.
    cfg = run_dir / "config.json"
    if cfg.exists():
        try:
            want = json.load(open(cfg)).get("requests_sent")
        except (ValueError, OSError):
            want = None
        if want:
            s["expected_requests"] = want
            s["complete"] = bool((s.get("requests") or 0) >= 0.99 * want)

    for k, v in j.items():
        if "theoretical" in k.lower():
            s[k] = v.get("avg") if isinstance(v, dict) else v
            if isinstance(v, dict):
                s[k + "_total_blocks"] = v.get("count")
                s[k + "_hit_blocks"] = v.get("sum")
    return s


targets = []
if a.all:
    for d in sorted((REPO / "runs").glob("*/")):
        if (d / "summary.json").exists() and not a.force:
            continue
        if glob.glob(str(d / "aiperf/**/profile_export_aiperf.json"), recursive=True):
            targets.append(d.name)
else:
    targets = a.runs
if not targets:
    raise SystemExit("nothing to do")

for name in targets:
    d = REPO / "runs" / name
    out = d / "summary.json"
    if out.exists() and not a.force:
        print(f"  {name}: summary.json exists, use --force")
        continue
    s = summarize(d)
    if s is None:
        continue
    json.dump(s, open(out, "w"), indent=1)
    f = lambda v, dp=1: f"{v:,.{dp}f}" if isinstance(v, (int, float)) else "n/a"
    print(f"  {name}: {f(s['cache_read_pct'], 2)}% read, "
          f"{f(s['req_per_hour'])} req/h, {f(s['duration_s'], 0)}s -> {out}")
