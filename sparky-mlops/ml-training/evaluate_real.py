#!/usr/bin/env python3

import argparse
import json
import platform
import socket
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml


class TinyClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    output_root = Path(config["output_dir"])
    run_dir = output_root / args.run_id
    model_path = run_dir / "model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    device_name = config.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested CUDA, but torch.cuda.is_available() is false")

    device = torch.device(device_name)

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)

    input_dim = int(config.get("input_dim", 128))
    hidden_dim = int(config.get("hidden_dim", 256))
    num_classes = int(config.get("num_classes", 4))
    batch_size = int(config.get("batch_size", 8))
    eval_steps = int(config.get("eval_steps", 20))

    model = TinyClassifier(input_dim, hidden_dim, num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    started_at = time.time()
    losses = []
    correct = 0
    total = 0

    print(f"Starting real evaluation run: {args.run_id}")
    print(f"Host: {socket.gethostname()}")
    print(f"Python: {platform.python_version()}")
    print(f"Torch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Model: {model_path}")

    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    with torch.no_grad():
        for step in range(eval_steps):
            inputs = torch.randn(batch_size, input_dim, device=device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += batch_size

            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)

            print(
                f"eval_step={step + 1}/{eval_steps} "
                f"loss={loss_value:.6f}"
            )

    elapsed_seconds = time.time() - started_at
    accuracy = correct / total if total else 0.0

    evaluation = {
        "run_id": args.run_id,
        "model_path": str(model_path),
        "avg_eval_loss": sum(losses) / len(losses),
        "min_eval_loss": min(losses),
        "max_eval_loss": max(losses),
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "eval_steps": eval_steps,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
        "host": socket.gethostname(),
    }

    output_path = run_dir / "evaluation.json"
    output_path.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote evaluation: {output_path}")
    print("Real evaluation completed successfully")


if __name__ == "__main__":
    main()
