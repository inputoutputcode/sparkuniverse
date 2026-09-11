import argparse
import json
import random
from pathlib import Path

import yaml
from datasets import load_dataset


CATEGORY_MAP = {
    "classification": "classification",
    "closed_qa": "closed_qa",
    "open_qa": "open_qa",
    "summarization": "summarization",
    "information_extraction": "extraction",
    "brainstorming": "brainstorming",
    "creative_writing": "creative_writing",
}


def build_prompt(row):
    instruction = row.get("instruction", "").strip()
    context = row.get("context", "").strip()

    if context:
        return f"{instruction}\n\nContext:\n{context}"

    return instruction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_name = config.get("dataset_name", "databricks/databricks-dolly-15k")
    split = config.get("split", "train")
    seed = int(config.get("seed", 42))
    total_examples = int(config.get("total_examples", 100))
    max_per_category = int(config.get("max_per_category", 20))

    dataset = load_dataset(dataset_name, split=split)

    rows = []
    for row in dataset:
        category = CATEGORY_MAP.get(row.get("category"))
        if not category:
            continue

        prompt = build_prompt(row)
        response = row.get("response", "").strip()

        if not prompt or not response:
            continue

        rows.append(
            {
                "id": f"dolly-{len(rows):05d}",
                "task_type": category,
                "prompt": prompt,
                "reference": response,
                "source_category": row.get("category"),
            }
        )

    rng = random.Random(seed)
    rng.shuffle(rows)

    counts = {}
    selected = []

    for row in rows:
        task_type = row["task_type"]
        if counts.get(task_type, 0) >= max_per_category:
            continue

        selected.append(row)
        counts[task_type] = counts.get(task_type, 0) + 1

        if len(selected) >= total_examples:
            break

    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"Wrote {len(selected)} eval examples to {output_path}")
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()