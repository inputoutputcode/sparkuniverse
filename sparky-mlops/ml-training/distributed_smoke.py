#!/usr/bin/env python3

import json
import os
import platform
import socket
from pathlib import Path

import torch
import torch.distributed as dist


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    rank = int(require_env("RANK"))
    world_size = int(require_env("WORLD_SIZE"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    run_id = require_env("RUN_ID")
    output_root = Path(os.environ.get("OUTPUT_ROOT", "/srv/sparky-mlops/runs"))
    run_dir = output_root / run_id

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    hostname = socket.gethostname()
    device_name = torch.cuda.get_device_name(local_rank)

    tensor = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    payload = {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "hostname": hostname,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": device_name,
        "all_reduce_result": float(tensor.item()),
    }

    print(json.dumps(payload, sort_keys=True))

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, payload)

    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = run_dir / "distributed-smoke.json"
        output_path.write_text(json.dumps(gathered, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()