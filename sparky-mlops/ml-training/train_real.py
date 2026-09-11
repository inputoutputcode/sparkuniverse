#!/usr/bin/env python3

import argparse
import json
import platform
import socket
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
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
    run_dir.mkdir(parents=True, exist_ok=True)

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
    epochs = int(config.get("epochs", 2))
    steps_per_epoch = int(config.get("steps_per_epoch", 50))
    learning_rate = float(config.get("learning_rate", 0.001))

    model = TinyClassifier(input_dim, hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    started_at = time.time()
    losses = []

    print(f"Starting real training run: {args.run_id}")
    print(f"Host: {socket.gethostname()}")
    print(f"Python: {platform.python_version()}")
    print(f"Torch: {torch.__version__}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            inputs = torch.randn(batch_size, input_dim, device=device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            epoch_loss += loss_value
            losses.append(loss_value)

            print(
                f"epoch={epoch + 1}/{epochs} "
                f"step={step + 1}/{steps_per_epoch} "
                f"loss={loss_value:.6f}"
            )

        avg_epoch_loss = epoch_loss / steps_per_epoch
        print(f"epoch={epoch + 1} avg_loss={avg_epoch_loss:.6f}")

    elapsed_seconds = time.time() - started_at

    model_path = run_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    metrics = {
        "run_id": args.run_id,
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "max_loss": max(losses),
        "avg_loss": sum(losses) / len(losses),
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
    }

    metadata = {
        "run_id": args.run_id,
        "host": socket.gethostname(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config_path": str(config_path),
        "output_dir": str(run_dir),
        "model_path": str(model_path),
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote model: {model_path}")
    print(f"Wrote metrics: {run_dir / 'metrics.json'}")
    print(f"Wrote metadata: {run_dir / 'metadata.json'}")
    print("Real training completed successfully")


if __name__ == "__main__":
    main()