#!/usr/bin/env python3

import argparse
import json
import time
import uuid
import yaml
from datetime import datetime, timezone
from pathlib import Path

import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Fake LLM fine-tuning trainer")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML experiment config",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run ID supplied by CI or caller",
    )
    return parser.parse_args()


def load_config(path):
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main():
    args = parse_args()
    config = load_config(args.config)

    run_id = args.run_id or f"run-{uuid.uuid4().hex[:8]}"
    output_root = Path(config["output_dir"]).expanduser()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_name = config.get("model_name", "unknown-model")
    dataset = config.get("dataset", "unknown-dataset")
    epochs = int(config.get("epochs", 1))
    learning_rate = float(config.get("learning_rate", 0.0002))

    print(f"Starting fake fine-tuning run: {run_id}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Output directory: {run_dir}")

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}: training...")
        time.sleep(2)
        print(f"Epoch {epoch}/{epochs}: done")

    metadata = {
        "run_id": run_id,
        "model_name": model_name,
        "dataset": dataset,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "status": "completed",
        "started_from_config": str(Path(args.config).resolve()),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(run_dir),
    }

    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Wrote metadata: {metadata_path}")
    print("Fake fine-tuning completed successfully")


if __name__ == "__main__":
    main()
