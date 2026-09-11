#!/usr/bin/env python3

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found")

    return json.loads(text[start : end + 1])


def score_result(task: dict, output: str) -> dict:
    task_type = task["task_type"]

    if not output.strip():
        return {
            "scored": True,
            "passed": False,
            "metric": "non_empty_output",
            "score": 0.0,
        }

    if task_type == "classification":
        expected = normalize_text(task["expected_label"])
        actual = normalize_text(output)

        first_line = actual.split("\n")[0]
        first_sentence = re.split(r"[.!?]", first_line)[0].strip()

        expected_forms = {expected}

        if expected.endswith("y"):
            expected_forms.add(expected[:-1] + "ies")
        elif expected.endswith(("s", "x", "z", "ch", "sh")):
            expected_forms.add(expected + "es")
        else:
            expected_forms.add(expected + "s")

        passed = False

        for expected_form in expected_forms:
            pattern = rf"\b{re.escape(expected_form)}\b"

            if re.search(pattern, first_sentence):
                passed = True
                break

        return {
            "scored": True,
            "passed": passed,
            "metric": "expected_label_standalone_word_or_plural",
            "score": 1.0 if passed else 0.0,
            "expected_label": expected,
            "accepted_forms": sorted(expected_forms),
            "first_sentence": first_sentence,
        }

    if task_type == "closed_qa":
        expected = normalize_text(task["expected_answer"])
        actual = normalize_text(output)
        passed = expected in actual

        return {
            "scored": True,
            "passed": passed,
            "metric": "expected_answer_substring",
            "score": 1.0 if passed else 0.0,
        }

    if task_type == "json_extraction":
        expected_json = task["expected_json"]

        try:
            parsed = extract_json_object(output)
        except Exception as exc:
            return {
                "scored": True,
                "passed": False,
                "metric": "json_parse_and_field_match",
                "score": 0.0,
                "json_parse_success": False,
                "error": str(exc),
            }

        matched = 0
        total = len(expected_json)

        for key, expected_value in expected_json.items():
            actual_value = str(parsed.get(key, ""))
            if normalize_text(str(expected_value)) in normalize_text(actual_value):
                matched += 1

        score = matched / total if total else 0.0

        return {
            "scored": True,
            "passed": score == 1.0,
            "metric": "json_parse_and_field_match",
            "score": score,
            "json_parse_success": True,
            "matched_fields": matched,
            "total_fields": total,
        }

    return {
        "scored": False,
        "passed": None,
        "metric": "human_review_required",
    }


def load_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def generate(model, tokenizer, prompt: str, config: dict) -> dict:
    messages = [{"role": "user", "content": prompt}]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(rendered_prompt, return_tensors="pt").to(model.device)
    started = time.perf_counter()

    generation_kwargs = {
        "max_new_tokens": int(config.get("max_new_tokens", 128)),
        "do_sample": bool(config.get("do_sample", False)),
        "pad_token_id": tokenizer.eos_token_id,
    }

    if generation_kwargs["do_sample"]:
        generation_kwargs["temperature"] = float(config.get("temperature", 0.7))
        generation_kwargs["top_p"] = float(config.get("top_p", 0.9))

    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)

    elapsed = time.perf_counter() - started
    new_tokens = generated[0][inputs["input_ids"].shape[-1] :]
    output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return {
        "output": output,
        "latency_seconds": elapsed,
        "generated_tokens": int(new_tokens.shape[-1]),
        "empty": not bool(output.strip()),
    }


def load_base_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    return tokenizer, model


def summarize(results: list[dict], run_id: str, config: dict, adapter_path: Path) -> dict:
    categories = defaultdict(
        lambda: {
            "total": 0,
            "scored": 0,
            "adapter_passed": 0,
            "base_passed": 0,
            "adapter_score_sum": 0.0,
            "base_score_sum": 0.0,
            "adapter_empty": 0,
            "base_empty": 0,
        }
    )

    for row in results:
        bucket = categories[row["task_type"]]
        bucket["total"] += 1

        if row["adapter"]["empty"]:
            bucket["adapter_empty"] += 1

        if row["base"]["empty"]:
            bucket["base_empty"] += 1

        if row["adapter_score"]["scored"]:
            bucket["scored"] += 1
            bucket["adapter_score_sum"] += float(row["adapter_score"].get("score", 0.0))
            bucket["base_score_sum"] += float(row["base_score"].get("score", 0.0))

            if row["adapter_score"]["passed"]:
                bucket["adapter_passed"] += 1

            if row["base_score"]["passed"]:
                bucket["base_passed"] += 1

    category_summary = {}
    total_scored = 0
    total_adapter_passed = 0
    total_base_passed = 0
    total_adapter_empty = 0
    total_base_empty = 0

    for category, bucket in sorted(categories.items()):
        scored = bucket["scored"]

        category_summary[category] = {
            **bucket,
            "adapter_accuracy": bucket["adapter_passed"] / scored if scored else None,
            "base_accuracy": bucket["base_passed"] / scored if scored else None,
            "adapter_avg_score": bucket["adapter_score_sum"] / scored if scored else None,
            "base_avg_score": bucket["base_score_sum"] / scored if scored else None,
        }

        total_scored += scored
        total_adapter_passed += bucket["adapter_passed"]
        total_base_passed += bucket["base_passed"]
        total_adapter_empty += bucket["adapter_empty"]
        total_base_empty += bucket["base_empty"]

    base_avg_latency_seconds = (
        sum(row["base"]["latency_seconds"] for row in results) / len(results)
    )

    adapter_avg_latency_seconds = (
        sum(row["adapter"]["latency_seconds"] for row in results) / len(results)
    )

    base_accuracy = total_base_passed / total_scored if total_scored else None
    adapter_accuracy = total_adapter_passed / total_scored if total_scored else None

    accuracy_regression = None
    if base_accuracy is not None and adapter_accuracy is not None:
        accuracy_regression = base_accuracy - adapter_accuracy

    gates = config.get("quality_gates", {})

    gate_results = {
        "min_machine_accuracy": {
            "threshold": gates.get("min_machine_accuracy"),
            "actual": adapter_accuracy,
            "passed": adapter_accuracy is not None
            and adapter_accuracy >= float(gates.get("min_machine_accuracy", 0.0)),
        },
        "max_empty_outputs": {
            "threshold": gates.get("max_empty_outputs"),
            "actual": total_adapter_empty,
            "passed": total_adapter_empty <= int(gates.get("max_empty_outputs", 999999)),
        },
        "max_adapter_avg_latency_seconds": {
            "threshold": gates.get("max_adapter_avg_latency_seconds"),
            "actual": adapter_avg_latency_seconds,
            "passed": adapter_avg_latency_seconds
            <= float(gates.get("max_adapter_avg_latency_seconds", 999999.0)),
        },
        "max_accuracy_regression_vs_base": {
            "threshold": gates.get("max_accuracy_regression_vs_base"),
            "actual": accuracy_regression,
            "passed": accuracy_regression is not None
            and accuracy_regression
            <= float(gates.get("max_accuracy_regression_vs_base", 1.0)),
        },
    }

    return {
        "run_id": run_id,
        "base_model": config["base_model"],
        "adapter_path": str(adapter_path),
        "machine_eval_path": config["machine_eval_path"],
        "human_eval_path": config["human_eval_path"],
        "total_tasks": len(results),
        "scored_tasks": total_scored,
        "base_passed_scored_tasks": total_base_passed,
        "adapter_passed_scored_tasks": total_adapter_passed,
        "base_scored_accuracy": base_accuracy,
        "adapter_scored_accuracy": adapter_accuracy,
        "accuracy_regression_vs_base": accuracy_regression,
        "base_empty_outputs": total_base_empty,
        "adapter_empty_outputs": total_adapter_empty,
        "base_avg_latency_seconds": base_avg_latency_seconds,
        "adapter_avg_latency_seconds": adapter_avg_latency_seconds,
        "category_summary": category_summary,
        "quality_gates": gate_results,
        "quality_gates_passed": all(item["passed"] for item in gate_results.values()),
        "requires_human_review": True,
    }


def markdown_cell(value: str, limit: int = 300) -> str:
    value = value.replace("\n", " ").replace("|", "\\|").strip()

    if len(value) > limit:
        value = value[: limit - 3] + "..."

    return value


def expected_display(row: dict) -> str:
    if "expected_label" in row:
        return str(row["expected_label"])

    if "expected_answer" in row:
        return str(row["expected_answer"])

    if "expected_json" in row:
        return json.dumps(row["expected_json"], sort_keys=True)

    if row.get("expected_behavior"):
        return str(row["expected_behavior"])

    return ""


def write_review_markdown(path: Path, summary: dict, results: list[dict]) -> None:
    lines = [
        "# Qwen LoRA Evaluation Review",
        "",
        f"Run ID: `{summary['run_id']}`",
        f"Base model: `{summary['base_model']}`",
        f"Adapter: `{summary['adapter_path']}`",
        "",
        "## Automated Summary",
        "",
        f"- Total tasks: `{summary['total_tasks']}`",
        f"- Scored tasks: `{summary['scored_tasks']}`",
        f"- Base scored accuracy: `{summary['base_scored_accuracy']}`",
        f"- Adapter scored accuracy: `{summary['adapter_scored_accuracy']}`",
        f"- Accuracy regression vs base: `{summary['accuracy_regression_vs_base']}`",
        f"- Quality gates passed: `{summary['quality_gates_passed']}`",
        f"- Base average latency seconds: `{summary['base_avg_latency_seconds']}`",
        f"- Adapter average latency seconds: `{summary['adapter_avg_latency_seconds']}`",
        "",
        "## Quality Gates",
        "",
        "| Gate | Threshold | Actual | Passed |",
        "|---|---:|---:|---|",
    ]

    for gate_name, gate in summary["quality_gates"].items():
        lines.append(
            f"| `{gate_name}` | `{gate['threshold']}` | `{gate['actual']}` | `{gate['passed']}` |"
        )

    lines.extend(
        [
            "",
            "## Category Summary",
            "",
            "| Category | Total | Scored | Base Accuracy | Adapter Accuracy | Base Empty | Adapter Empty |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for category, row in summary["category_summary"].items():
        lines.append(
            f"| `{category}` | {row['total']} | {row['scored']} | "
            f"{row['base_accuracy']} | {row['adapter_accuracy']} | "
            f"{row['base_empty']} | {row['adapter_empty']} |"
        )

    lines.extend(
        [
            "",
            "## Review Table",
            "",
            "| ID | Type | Prompt | Expected | Base Output | Adapter Output | Auto Result | Human Preference | Notes |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
    )

    for row in results:
        auto_result = "human"

        if row["adapter_score"]["scored"]:
            auto_result = "pass" if row["adapter_score"]["passed"] else "fail"

        lines.append(
            "| "
            f"`{row['id']}` | "
            f"`{row['task_type']}` | "
            f"{markdown_cell(row['prompt'], 180)} | "
            f"{markdown_cell(expected_display(row), 220)} | "
            f"{markdown_cell(row['base']['output'])} | "
            f"{markdown_cell(row['adapter']['output'])} | "
            f"{auto_result} |  |  |"
        )

    lines.extend(
        [
            "",
            "## Human Review Decision",
            "",
            "- Accepted: `yes/no`",
            "- Better than base: `yes/no/mixed`",
            "- Main improvement:",
            "- Main regression:",
            "- Next parameter change:",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    run_id = args.run_id
    output_dir = Path("/srv/sparky-mlops/runs") / run_id
    adapter_path = output_dir / config.get("adapter_subdir", "qwen-lora-adapter")

    output_dir.mkdir(parents=True, exist_ok=True)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    machine_tasks = load_jsonl(Path(config["machine_eval_path"]))
    human_tasks = load_jsonl(Path(config["human_eval_path"]))
    all_tasks = machine_tasks + human_tasks

    print(f"run id: {run_id}")
    print(f"base model: {config['base_model']}")
    print(f"adapter path: {adapter_path}")
    print(f"machine eval path: {config['machine_eval_path']}")
    print(f"human eval path: {config['human_eval_path']}")
    print(f"machine task count: {len(machine_tasks)}")
    print(f"human review task count: {len(human_tasks)}")
    print(f"total task count: {len(all_tasks)}")
    print(f"cuda available: {torch.cuda.is_available()}")

    tokenizer, base_model = load_base_model(config["base_model"])

    print("Running base model evaluation...")
    base_outputs = {}

    for index, task in enumerate(all_tasks, start=1):
        print(f"base eval {index}/{len(all_tasks)}: {task['id']}")
        base_outputs[task["id"]] = generate(base_model, tokenizer, task["prompt"], config)

    print("Loading LoRA adapter...")
    adapter_model = PeftModel.from_pretrained(base_model, str(adapter_path))
    adapter_model.eval()

    print("Running adapter evaluation...")
    results = []

    for index, task in enumerate(all_tasks, start=1):
        print(f"adapter eval {index}/{len(all_tasks)}: {task['id']}")

        adapter_result = generate(adapter_model, tokenizer, task["prompt"], config)
        base_result = base_outputs[task["id"]]

        result = {
            "id": task["id"],
            "task_type": task["task_type"],
            "prompt": task["prompt"],
            "expected_behavior": task.get("expected_behavior"),
            "base": base_result,
            "base_score": score_result(task, base_result["output"]),
            "adapter": adapter_result,
            "adapter_score": score_result(task, adapter_result["output"]),
        }

        if "expected_label" in task:
            result["expected_label"] = task["expected_label"]

        if "expected_answer" in task:
            result["expected_answer"] = task["expected_answer"]

        if "expected_json" in task:
            result["expected_json"] = task["expected_json"]

        results.append(result)

    summary = summarize(results, run_id, config, adapter_path)

    files = config["output_files"]

    machine_results = [row for row in results if row["adapter_score"]["scored"]]
    human_results = [row for row in results if not row["adapter_score"]["scored"]]

    machine_results_path = output_dir / files["machine_results_jsonl"]
    human_results_path = output_dir / files["human_results_jsonl"]
    summary_path = output_dir / files["summary_json"]
    review_path = output_dir / files["review_markdown"]

    with machine_results_path.open("w", encoding="utf-8") as handle:
        for row in machine_results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    with human_results_path.open("w", encoding="utf-8") as handle:
        for row in human_results:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_review_markdown(review_path, summary, results)

    print("Evaluation completed")
    print(f"machine results: {machine_results_path}")
    print(f"human results: {human_results_path}")
    print(f"summary: {summary_path}")
    print(f"review: {review_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not summary["quality_gates_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()