#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", default="/srv/sparky-mlops/runs")
    parser.add_argument("--promotion-name", default="qwen-current")
    parser.add_argument("--decision", default="accepted")
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    run_dir = runs_root / args.run_id
    promoted_dir = runs_root / "promoted"
    promotion_file = promoted_dir / f"{args.promotion_name}.json"

    adapter_path = run_dir / "qwen-lora-adapter"
    summary_path = run_dir / "qwen-eval-summary.json"
    review_path = run_dir / "qwen-eval-review.md"
    machine_results_path = run_dir / "qwen-machine-eval-results.jsonl"
    human_results_path = run_dir / "qwen-human-review-results.jsonl"

    if not adapter_path.is_dir():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    if not summary_path.is_file():
        raise FileNotFoundError(f"Evaluation summary does not exist: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if not summary.get("quality_gates_passed"):
        raise SystemExit("Refusing promotion because quality_gates_passed is false")

    promoted_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "run_id": args.run_id,
        "base_model": summary["base_model"],
        "adapter_path": str(adapter_path),
        "summary_path": str(summary_path),
        "review_path": str(review_path),
        "machine_results_path": str(machine_results_path),
        "human_results_path": str(human_results_path),
        "adapter_scored_accuracy": summary["adapter_scored_accuracy"],
        "base_scored_accuracy": summary["base_scored_accuracy"],
        "accuracy_regression_vs_base": summary["accuracy_regression_vs_base"],
        "quality_gates_passed": summary["quality_gates_passed"],
        "decision": args.decision,
    }

    promotion_file.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()