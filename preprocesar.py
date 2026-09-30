from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from guider.config import load_config
from guider.corpus import collect_tokenizer_sample, iter_mixture
from guider.tokenizer import ByteBPETokenizer


def write_token_file(specs, tokenizer, output_path, max_tokens, batch_size, seed):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = examples = 0
    with output_path.open("wb") as handle:
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
    return count, examples


def main():
    parser = argparse.ArgumentParser(description="Stream and tokenize a configured Hugging Face corpus mixture.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    dc, tc = cfg["data"], cfg["tokenizer"]
    local = Path(dc["local_dir"])
    local.mkdir(parents=True, exist_ok=True)
    tokenizer_path = local / dc["tokenizer_file"]

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
            dc.get("tokenizer_sources", dc["train_sources"]),
            int(tc["sample_chars"]),
            seed=int(cfg["training"]["seed"]),
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
        dc["max_train_tokens"], int(dc["preprocess_batch_size"]),
        int(cfg["training"]["seed"]) + 11,
    )
    val_count, val_examples = write_token_file(
        dc["val_sources"], tokenizer, local / dc["val_file"],
        dc["max_val_tokens"], int(dc["preprocess_batch_size"]),
        int(cfg["training"]["seed"]) + 29,
    )
    metadata = {
        "version": cfg.get("run_name", "Guider"),
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
    (local / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    persistent = Path(dc["persistent_dir"]) if dc.get("persistent_dir") else None
    if persistent and Path("/content/drive/MyDrive").exists():
        persistent.mkdir(parents=True, exist_ok=True)
        for filename in [dc["tokenizer_file"], dc["train_file"], dc["val_file"], "metadata.json"]:
            shutil.copy2(local / filename, persistent / filename)
        print(f"Copied processed files to {persistent}")
    else:
        print("Drive is optional; processed files remain in local runtime storage.")
    print(f"Preprocessing complete: train={train_count:,}; val={val_count:,} tokens")


if __name__ == "__main__":
    main()
