#!/usr/bin/env python3

import argparse
import json
import os
import platform
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel


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


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    rank = int(require_env("RANK"))
    world_size = int(require_env("WORLD_SIZE"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl")

    config_path = Path(args.config)
    config = load_config(config_path)

    output_root = Path(config["output_dir"])
    run_dir = output_root / args.run_id

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed + rank)

    input_dim = int(config.get("input_dim", 128))
    hidden_dim = int(config.get("hidden_dim", 256))
    num_classes = int(config.get("num_classes", 4))
    batch_size = int(config.get("batch_size", 8))
    epochs = int(config.get("epochs", 2))
    steps_per_epoch = int(config.get("steps_per_epoch", 50))
    learning_rate = float(config.get("learning_rate", 0.001))

    model = TinyClassifier(input_dim, hidden_dim, num_classes).to(device)
    ddp_model = DistributedDataParallel(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(ddp_model.parameters(), lr=learning_rate)

    hostname = socket.gethostname()
    started_at = time.time()
    local_losses = []

    print(
        f"Starting DDP training run={args.run_id} "
        f"rank={rank}/{world_size} local_rank={local_rank} host={hostname}"
    )
    print(f"Torch: {torch.__version__}")
    print(f"CUDA device: {torch.cuda.get_device_name(local_rank)}")

    for epoch in range(epochs):
        ddp_model.train()
        epoch_loss = 0.0

        for step in range(steps_per_epoch):
            inputs = torch.randn(batch_size, input_dim, device=device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)

            optimizer.zero_grad(set_to_none=True)
            outputs = ddp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            local_losses.append(loss_value)
            epoch_loss += loss_value

            print(
                f"rank={rank} epoch={epoch + 1}/{epochs} "
                f"step={step + 1}/{steps_per_epoch} loss={loss_value:.6f}"
            )

        avg_epoch_loss = epoch_loss / steps_per_epoch
        print(f"rank={rank} epoch={epoch + 1} avg_loss={avg_epoch_loss:.6f}")

    elapsed_seconds = time.time() - started_at

    local_summary = {
        "rank": rank,
        "host": hostname,
        "final_loss": local_losses[-1],
        "min_loss": min(local_losses),
        "max_loss": max(local_losses),
        "avg_loss": sum(local_losses) / len(local_losses),
        "num_losses": len(local_losses),
    }

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_summary)

    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)

        model_path = run_dir / "ddp-model.pt"
        torch.save(ddp_model.module.state_dict(), model_path)

        metrics = {
            "run_id": args.run_id,
            "world_size": world_size,
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "batch_size_per_rank": batch_size,
            "global_batch_size": batch_size * world_size,
            "learning_rate": learning_rate,
            "elapsed_seconds": elapsed_seconds,
            "rank_summaries": gathered,
            "avg_final_loss": sum(item["final_loss"] for item in gathered) / world_size,
            "avg_loss": sum(item["avg_loss"] for item in gathered) / world_size,
        }

        metadata = {
            "run_id": args.run_id,
            "training_type": "ddp",
            "master_addr": os.environ.get("MASTER_ADDR"),
            "master_port": os.environ.get("MASTER_PORT"),
            "world_size": world_size,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(local_rank),
            "config_path": str(config_path),
            "output_dir": str(run_dir),
            "model_path": str(model_path),
            "hosts": sorted({item["host"] for item in gathered}),
        }

        (run_dir / "ddp-metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        (run_dir / "ddp-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )

        print(f"Wrote model: {model_path}")
        print(f"Wrote metrics: {run_dir / 'ddp-metrics.json'}")
        print(f"Wrote metadata: {run_dir / 'ddp-metadata.json'}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()