#!/usr/bin/env python3

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Fake evaluator for MLOps lab")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML evaluation config",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to training metadata.json",
    )
    return parser.parse_args()


def load_yaml(path):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path):
    json_path = Path(path)
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = parse_args()

    config = load_yaml(args.config)
    metadata = load_json(args.metadata)

    run_id = metadata["run_id"]
    output_dir = Path(metadata["artifact_dir"])

    minimum_score = float(config.get("minimum_score", 0.75))
    candidate_score = float(config.get("candidate_score", 0.82))

    passed = candidate_score >= minimum_score

    metrics = {
        "run_id": run_id,
        "model_name": metadata.get("model_name"),
        "dataset": metadata.get("dataset"),
        "candidate_score": candidate_score,
        "minimum_score": minimum_score,
        "passed": passed,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "metadata_path": str(Path(args.metadata).resolve()),
    }

    metrics_path = output_dir / "metrics.json"
    write_json(metrics_path, metrics)

    print(f"Evaluated run: {run_id}")
    print(f"Candidate score: {candidate_score}")
    print(f"Minimum score: {minimum_score}")
    print(f"Passed: {passed}")
    print(f"Wrote metrics: {metrics_path}")

    if not passed:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())