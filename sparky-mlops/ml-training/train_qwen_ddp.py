#!/usr/bin/env python3

import argparse
import json
import os
import random
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_rank_zero() -> bool:
    return get_rank() == 0


def log(message: str) -> None:
    if is_rank_zero():
        print(message, flush=True)


def setup_distributed() -> None:
    if get_world_size() <= 1:
        return

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")

    torch.cuda.set_device(get_local_rank())


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    seed = seed + get_rank()
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def build_chat_text(instruction: str, response: str, context: str = "") -> str:
    instruction = instruction.strip()
    response = response.strip()
    context = context.strip()

    if context:
        user_content = f"{instruction}\n\nContext:\n{context}"
    else:
        user_content = instruction

    return (
        "<|im_start|>user\n"
        f"{user_content}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{response}\n"
        "<|im_end|>"
    )


def tokenize_example(tokenizer, text: str, max_length: int) -> dict:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].squeeze(0)
    attention_mask = encoded["attention_mask"].squeeze(0)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def load_dolly_rows(config: dict) -> list[dict]:
    dataset = load_dataset(
        config.get("dataset_name", "databricks/databricks-dolly-15k"),
        split=config.get("dataset_split", "train"),
    )

    rows = []
    for row in dataset:
        instruction = str(row.get("instruction", "")).strip()
        response = str(row.get("response", "")).strip()
        context = str(row.get("context", "")).strip()

        if instruction and response:
            rows.append(
                {
                    "source": "dolly",
                    "instruction": instruction,
                    "context": context,
                    "response": response,
                }
            )

    random.Random(int(config.get("seed", 42))).shuffle(rows)
    return rows[: int(config.get("dataset_max_samples", 12000))]


def load_supplemental_rows(config: dict) -> list[dict]:
    path_value = config.get("supplemental_jsonl_path")

    if not path_value:
        return []

    path = Path(path_value)
    repeat = int(config.get("supplemental_repeat", 1))

    if not path.exists():
        raise FileNotFoundError(f"supplemental_jsonl_path does not exist: {path}")

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            instruction = str(row.get("instruction", "")).strip()
            response = str(row.get("response", "")).strip()
            context = str(row.get("context", "")).strip()

            if not instruction or not response:
                raise ValueError(f"supplemental row needs instruction and response: {row}")

            rows.append(
                {
                    "source": "format-supplement",
                    "instruction": instruction,
                    "context": context,
                    "response": response,
                }
            )

    return rows * repeat


def load_rank_dataset(config: dict, tokenizer) -> tuple[list[dict], dict]:
    max_length = int(config.get("max_length", 512))
    seed = int(config.get("seed", 42))
    rank = get_rank()
    world_size = get_world_size()

    dolly_rows = load_dolly_rows(config)
    supplemental_rows = load_supplemental_rows(config)

    all_rows = dolly_rows + supplemental_rows
    random.Random(seed).shuffle(all_rows)

    rank_rows = all_rows[rank::world_size]

    tokenized = []
    for row in rank_rows:
        text = build_chat_text(
            instruction=row["instruction"],
            context=row.get("context", ""),
            response=row["response"],
        )
        tokenized.append(tokenize_example(tokenizer, text, max_length))

    stats = {
        "dolly_rows": len(dolly_rows),
        "supplemental_rows_after_repeat": len(supplemental_rows),
        "total_rows": len(all_rows),
        "rank_rows": len(rank_rows),
    }

    return tokenized, stats


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.unsqueeze(0).to(device) for key, value in batch.items()}


def unwrap_model(model):
    if isinstance(model, DDP):
        return model.module
    return model


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    setup_distributed()

    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()

    set_seed(int(config.get("seed", 42)))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = torch.device(f"cuda:{local_rank}")

    run_id = args.run_id
    output_root = Path(config.get("output_dir", "/srv/sparky-mlops/runs"))
    run_dir = output_root / run_id
    adapter_subdir = config.get("adapter_subdir", "qwen-lora-adapter")
    adapter_dir = run_dir / adapter_subdir

    if is_rank_zero():
        run_dir.mkdir(parents=True, exist_ok=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    model_name = config.get("base_model", "Qwen/Qwen2.5-0.5B-Instruct")

    log(f"run id: {run_id}")
    log(f"host: {socket.gethostname()}")
    log(f"world size: {world_size}")
    log(f"base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("Loading dataset...")
    rank_dataset, dataset_stats = load_rank_dataset(config, tokenizer)

    if not rank_dataset:
        raise RuntimeError(f"Rank {rank} received no training samples")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=int(config.get("lora_r", 8)),
        lora_alpha=int(config.get("lora_alpha", 16)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )

    model = get_peft_model(model, lora_config)
    model.to(device)

    if is_rank_zero():
        model.print_trainable_parameters()

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    epochs = int(config.get("epochs", 1))
    batch_size_per_rank = int(config.get("batch_size_per_rank", 1))

    if batch_size_per_rank != 1:
        raise ValueError("This lab trainer currently expects batch_size_per_rank: 1")

    configured_steps = config.get("steps_per_epoch", "all")
    if configured_steps == "all":
        steps_per_epoch = len(rank_dataset)
    else:
        steps_per_epoch = min(int(configured_steps), len(rank_dataset))

    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
    learning_rate = float(config.get("learning_rate", 2e-4))
    weight_decay = float(config.get("weight_decay", 0.0))
    warmup_ratio = float(config.get("warmup_ratio", 0.0))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    log_every = int(config.get("log_every", 50))

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_micro_steps = steps_per_epoch * epochs
    total_optimizer_steps = max(
        1,
        (total_micro_steps + gradient_accumulation_steps - 1)
        // gradient_accumulation_steps,
    )
    warmup_steps = int(total_optimizer_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if dist.is_available() and dist.is_initialized():
        local_count = torch.tensor([len(rank_dataset)], device=device)
        gathered_counts = [torch.zeros_like(local_count) for _ in range(world_size)]
        dist.all_gather(gathered_counts, local_count)

        gathered_stats = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_stats, dataset_stats)

        if is_rank_zero():
            for index, value in enumerate(gathered_counts):
                print(f"Samples on rank {index}: {int(value.item())}", flush=True)
            print(f"dataset_stats_by_rank: {json.dumps(gathered_stats, sort_keys=True)}", flush=True)
    else:
        log(f"Samples on rank 0: {len(rank_dataset)}")
        log(f"dataset_stats: {json.dumps(dataset_stats, sort_keys=True)}")

    log(f"epochs: {epochs}")
    log(f"steps_per_epoch: {steps_per_epoch}")
    log(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    log(f"total_optimizer_steps: {total_optimizer_steps}")
    log(f"warmup_steps: {warmup_steps}")
    log(f"learning_rate: {learning_rate}")
    log(f"lora_r: {int(config.get('lora_r', 8))}")
    log(f"lora_alpha: {int(config.get('lora_alpha', 16))}")
    log(f"lora_dropout: {float(config.get('lora_dropout', 0.05))}")
    log(f"lora_target_modules: {config.get('lora_target_modules')}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    started = time.perf_counter()

    loss_sum = 0.0
    loss_min = None
    loss_max = None
    final_loss = None
    micro_step_count = 0
    optimizer_step_count = 0

    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            sample = rank_dataset[step % len(rank_dataset)]
            batch = move_batch_to_device(sample, device)

            outputs = model(**batch)
            raw_loss = outputs.loss
            loss = raw_loss / gradient_accumulation_steps
            loss.backward()

            raw_loss_value = float(raw_loss.detach().cpu())
            final_loss = raw_loss_value
            loss_sum += raw_loss_value
            micro_step_count += 1

            loss_min = raw_loss_value if loss_min is None else min(loss_min, raw_loss_value)
            loss_max = raw_loss_value if loss_max is None else max(loss_max, raw_loss_value)

            should_step = (
                micro_step_count % gradient_accumulation_steps == 0
                or step == steps_per_epoch - 1
            )

            if should_step:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step_count += 1

            if is_rank_zero() and (
                step == 0
                or (step + 1) % log_every == 0
                or step == steps_per_epoch - 1
            ):
                print(
                    f"epoch={epoch + 1}/{epochs} "
                    f"step={step + 1}/{steps_per_epoch} "
                    f"optimizer_step={optimizer_step_count} "
                    f"loss={raw_loss_value:.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.8f}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started

    local_metrics = {
        "rank": rank,
        "host": socket.gethostname(),
        "sample_count": len(rank_dataset),
        "micro_steps": micro_step_count,
        "optimizer_steps": optimizer_step_count,
        "final_loss": final_loss,
        "avg_loss": loss_sum / micro_step_count if micro_step_count else None,
        "min_loss": loss_min,
        "max_loss": loss_max,
        "elapsed_seconds": elapsed,
        "dataset_stats": dataset_stats,
    }

    if dist.is_available() and dist.is_initialized():
        gathered_metrics = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_metrics, local_metrics)
    else:
        gathered_metrics = [local_metrics]

    if is_rank_zero():
        unwrapped = unwrap_model(model)

        if bool(config.get("save_adapter", True)):
            adapter_dir.mkdir(parents=True, exist_ok=True)
            unwrapped.save_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))

        avg_final_loss = sum(item["final_loss"] for item in gathered_metrics) / len(
            gathered_metrics
        )
        avg_loss = sum(item["avg_loss"] for item in gathered_metrics) / len(
            gathered_metrics
        )

        metadata = {
            "run_id": run_id,
            "training_type": "qwen-lora-ddp",
            "base_model": model_name,
            "dataset_name": config.get("dataset_name", "databricks/databricks-dolly-15k"),
            "dataset_split": config.get("dataset_split", "train"),
            "dataset_max_samples": int(config.get("dataset_max_samples", 12000)),
            "supplemental_jsonl_path": config.get("supplemental_jsonl_path"),
            "supplemental_repeat": int(config.get("supplemental_repeat", 1)),
            "world_size": world_size,
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "batch_size_per_rank": batch_size_per_rank,
            "global_batch_size": batch_size_per_rank
            * world_size
            * gradient_accumulation_steps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_length": int(config.get("max_length", 512)),
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "max_grad_norm": max_grad_norm,
            "lora_r": int(config.get("lora_r", 8)),
            "lora_alpha": int(config.get("lora_alpha", 16)),
            "lora_dropout": float(config.get("lora_dropout", 0.05)),
            "lora_target_modules": config.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
            "python": os.popen("python3 --version").read().strip(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0),
            "adapter_path": str(adapter_dir),
        }

        metrics = {
            "run_id": run_id,
            "training_type": "qwen-lora-ddp",
            "world_size": world_size,
            "elapsed_seconds": elapsed,
            "avg_final_loss": avg_final_loss,
            "avg_loss": avg_loss,
            "rank_metrics": gathered_metrics,
        }

        save_json(run_dir / "qwen-ddp-metadata.json", metadata)
        save_json(run_dir / "qwen-ddp-metrics.json", metrics)

        print(f"Wrote metadata: {run_dir / 'qwen-ddp-metadata.json'}", flush=True)
        print(f"Wrote metrics: {run_dir / 'qwen-ddp-metrics.json'}", flush=True)
        print(f"Saved adapter: {adapter_dir}", flush=True)

    cleanup_distributed()


if __name__ == "__main__":
    main()