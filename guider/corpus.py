from __future__ import annotations

import random
from typing import Iterator

from datasets import load_dataset


def _load_source(spec: dict, seed: int):
    kwargs = {"split": spec.get("split", "train"), "streaming": True}
    if spec.get("config"):
        kwargs["name"] = spec["config"]
    dataset = load_dataset(spec["name"], **kwargs)
    skip_examples = int(spec.get("skip_examples", 0) or 0)
    if skip_examples:
        dataset = dataset.skip(skip_examples)
    shuffle_buffer = int(spec.get("shuffle_buffer", 0) or 0)
    if shuffle_buffer > 1:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
    return dataset


def render_messages(messages) -> str:
    parts = []
    for message in messages or []:
        role = str(message.get("role", "user")).strip().lower()
        content = str(message.get("content", "")).strip()
        if content:
            parts.append(f"<|{role}|>\n{content}")
    return "\n\n".join(parts)


def row_to_text(row: dict, spec: dict) -> str:
    field = spec.get("message_field")
    if field:
        return render_messages(row.get(field, []))
    value = row.get(spec.get("text_field", "text"), "")
    return value if isinstance(value, str) else str(value or "")


def iter_source(spec: dict, seed: int = 42) -> Iterator[str]:
    dataset = _load_source(spec, seed)
    limit = spec.get("max_examples")
    count = 0
    for row in dataset:
        text = row_to_text(row, spec).strip()
        if not text:
            continue
        yield text
        count += 1
        if limit is not None and count >= int(limit):
            break


def iter_mixture(specs: list[dict], seed: int = 42) -> Iterator[str]:
    if not specs:
        raise ValueError("At least one dataset source is required.")
    rng = random.Random(seed)
    active = [
        {"weight": max(0.0, float(spec.get("weight", 1.0))),
         "iter": iter_source(spec, seed + i)}
        for i, spec in enumerate(specs)
    ]
    active = [item for item in active if item["weight"] > 0]
    while active:
        selected = rng.choices(active, weights=[x["weight"] for x in active], k=1)[0]
        try:
            yield next(selected["iter"])
        except StopIteration:
            active.remove(selected)


def collect_tokenizer_sample(specs: list[dict], max_chars: int, seed: int = 42) -> list[str]:
    texts = []
    total = 0
    for text in iter_mixture(specs, seed=seed):
        remaining = max_chars - total
        if remaining <= 0:
            break
        piece = text[:remaining]
        if piece:
            texts.append(piece)
            total += len(piece)
    if not texts:
        raise RuntimeError("Configured Hugging Face datasets returned no text.")
    print(f"Tokenizer sample: {total:,} chars from {len(texts):,} examples")
    return texts
