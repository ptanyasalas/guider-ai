from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from guider.config import load_config
from guider.corpus import collect_tokenizer_sample, iter_mixture
from guider.tokenizer import ByteBPETokenizer


def _validate_source(spec: dict, label: str) -> None:
    if not isinstance(spec, dict) or not spec.get("name"):
        raise ValueError(f"{label}: each source needs a dataset 'name'.")
    if spec.get("message_field") is None and not spec.get("text_field", "text"):
        raise ValueError(f"{label}: specify text_field or message_field.")
    weight = float(spec.get("weight", 1.0))
    if weight < 0:
        raise ValueError(f"{label}: source weight cannot be negative.")


def validate_config(cfg: dict) -> None:
    dc, tc = cfg["data"], cfg["tokenizer"]
    for key in ("train_sources", "val_sources"):
        sources = dc.get(key, [])
        if not sources:
            raise ValueError(f"data.{key} must contain at least one dataset source.")
        for i, spec in enumerate(sources):
            _validate_source(spec, f"data.{key}[{i}]")
    if not dc.get("tokenizer_sources", dc["train_sources"]) and not tc.get("copy_from"):
        raise ValueError("Tokenizer training needs at least one source.")
    for key in ("max_train_tokens", "max_val_tokens"):
        value = dc.get(key)
        if value is not None and int(value) <= 0:
            raise ValueError(f"data.{key} must be positive or null.")
    if int(dc.get("preprocess_batch_size", 0)) < 1:
        raise ValueError("data.preprocess_batch_size must be >= 1.")
    if not tc.get("copy_from") and int(tc.get("sample_chars", 0)) < 1:
        raise ValueError("tokenizer.sample_chars must be >= 1.")


def write_token_file(specs, tokenizer, output_path, max_tokens, batch_size, seed):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temporary file so an interrupted run never leaves a valid-looking partial .bin.
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    count = examples = 0
    try:
        with temp_path.open("wb") as handle:
            batch = []

            def flush(texts):
                nonlocal count
                encoded = tokenizer.encode_batch(texts, add_eos=True)
                ids = np.fromiter(itertools.chain.from_iterable(encoded), dtype=np.uint16)
                if max_tokens is not None:
                    ids = ids[:max(0, int(max_tokens) - count)]
                if ids.size:
                    ids.tofile(handle)
                    count += int(ids.size)
                return max_tokens is None or count < int(max_tokens)

            for text in tqdm(iter_mixture(specs, seed), desc=f"Encoding {output_path.name}"):
                batch.append(text)
                examples += 1
                if len(batch) >= batch_size:
                    keep_going = flush(batch)
                    batch = []
                    if not keep_going:
                        break
            if batch and (max_tokens is None or count < int(max_tokens)):
                flush(batch)
            handle.flush()
            os.fsync(handle.fileno())
        if count == 0:
            raise RuntimeError(f"No tokens were written to {output_path}. Check dataset names, splits and text fields.")
        os.replace(temp_path, output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return count, examples


def _probe_source(specs, label, seed):
    # Force one row from every configured source before spending time training the tokenizer.
    for index, spec in enumerate(specs):
        iterator = iter_mixture([spec], seed + index)
        try:
            next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"No usable text from {label}[{index}] dataset={spec.get('name')!r}, "
                f"split={spec.get('split', 'train')!r}. Check dataset config/split/fields."
            ) from exc


def main():
    parser = argparse.ArgumentParser(description="Stream and tokenize a configured Hugging Face corpus mixture.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    validate_config(cfg)
    dc, tc = cfg["data"], cfg["tokenizer"]
    seed = int(cfg["training"]["seed"])
    local = Path(dc["local_dir"])
    local.mkdir(parents=True, exist_ok=True)
    tokenizer_path = local / dc["tokenizer_file"]

    print("Checking configured Hugging Face sources before preprocessing...")
    _probe_source(dc["train_sources"], "data.train_sources", seed)
    _probe_source(dc["val_sources"], "data.val_sources", seed + 100)
    tokenizer_sources = dc.get("tokenizer_sources", dc["train_sources"])
    if not tc.get("copy_from"):
        _probe_source(tokenizer_sources, "data.tokenizer_sources", seed + 200)

    copy_from = tc.get("copy_from")
    if copy_from:
        source = Path(copy_from)
        if not source.exists():
            raise FileNotFoundError(f"tokenizer.copy_from not found: {source}")
        shutil.copy2(source, tokenizer_path)
        tokenizer = ByteBPETokenizer.from_file(tokenizer_path, tc["eos_token"])
        print(f"Reusing tokenizer from {source}; vocab={tokenizer.vocab_size:,}")
    else:
        sample = collect_tokenizer_sample(
            tokenizer_sources,
            int(tc["sample_chars"]),
            seed=seed,
        )
        tokenizer = ByteBPETokenizer.train(
            sample,
            vocab_size=int(tc["vocab_size"]),
            min_pair_frequency=int(tc["min_pair_frequency"]),
            eos_token=tc["eos_token"],
        )
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer saved: {tokenizer_path}; vocab={tokenizer.vocab_size:,}")

    if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("Tokenizer vocabulary is too large for uint16 token IDs.")

    train_count, train_examples = write_token_file(
        dc["train_sources"], tokenizer, local / dc["train_file"],
        dc["max_train_tokens"], int(dc["preprocess_batch_size"]), seed + 11,
    )
    val_count, val_examples = write_token_file(
        dc["val_sources"], tokenizer, local / dc["val_file"],
        dc["max_val_tokens"], int(dc["preprocess_batch_size"]), seed + 29,
    )
    metadata = {
        "version": cfg.get("run_name", "Guider 0.4.1"),
        "vocab_size": tokenizer.vocab_size,
        "eos_token_id": tokenizer.eos_token_id,
        "train_tokens": train_count,
        "val_tokens": val_count,
        "train_examples": train_examples,
        "val_examples": val_examples,
        "dtype": "uint16",
        "train_sources": dc["train_sources"],
        "val_sources": dc["val_sources"],
    }
    metadata_path = local / "metadata.json"
    metadata_tmp = metadata_path.with_suffix(".json.tmp")
    metadata_tmp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(metadata_tmp, metadata_path)

    persistent = Path(dc["persistent_dir"]) if dc.get("persistent_dir") else None
    if persistent and Path("/content/drive/MyDrive").exists():
        persistent.mkdir(parents=True, exist_ok=True)
        for filename in [dc["tokenizer_file"], dc["train_file"], dc["val_file"], "metadata.json"]:
            shutil.copy2(local / filename, persistent / filename)
        print(f"Copied processed files to {persistent}")
    else:
        print("Persistent storage not configured; processed files remain in local runtime storage.")
    print(f"Preprocessing complete: train={train_count:,}; val={val_count:,} tokens")


if __name__ == "__main__":
    main()
