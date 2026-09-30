from __future__ import annotations

import itertools
import json
import shutil
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from guider.config import load_config
from guider.tokenizer import ByteBPETokenizer

DATASET_NAME = "roneneldan/TinyStories"


def collect_tokenizer_sample(dataset, max_chars):
    texts, total = [], 0
    for row in dataset:
        text = row.get("text", "")
        if not text:
            continue
        piece = text[:max_chars - total]
        texts.append(piece)
        total += len(piece)
        if total >= max_chars:
            break
    if not texts:
        raise RuntimeError("TinyStories stream returned no text.")
    print(f"Tokenizer training sample: {total:,} characters")
    return texts


def write_token_file(dataset, tokenizer, output_path, max_tokens, batch_size):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    token_count = 0
    with output_path.open("wb") as handle:
        batch = []

        def flush(text_batch):
            nonlocal token_count
            encoded = tokenizer.encode_batch(text_batch, add_eos=True)
            flat = itertools.chain.from_iterable(encoded)
            ids = np.fromiter(flat, dtype=np.uint16)
            if max_tokens is not None:
                ids = ids[:max(0, max_tokens - token_count)]
            if ids.size:
                ids.tofile(handle)
                token_count += int(ids.size)
            return max_tokens is None or token_count < max_tokens

        for row in tqdm(dataset, desc=f"Encoding {output_path.name}"):
            text = row.get("text", "")
            if not text:
                continue
            batch.append(text)
            if len(batch) >= batch_size:
                keep_going = flush(batch)
                batch = []
                if not keep_going:
                    break
        if batch and (max_tokens is None or token_count < max_tokens):
            flush(batch)
    return token_count


def persist_files(local_dir, persistent_dir, filenames):
    if not persistent_dir or not Path("/content/drive/MyDrive").exists():
        print("Google Drive is not mounted; processed files remain on local disk.")
        return
    persistent_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        print(f"Copying {filename} to Google Drive...")
        shutil.copy2(local_dir / filename, persistent_dir / filename)


def main():
    cfg = load_config()
    dc, tc = cfg["data"], cfg["tokenizer"]
    local_dir = Path(dc["local_dir"])
    persistent_dir = Path(dc["persistent_dir"]) if dc.get("persistent_dir") else None
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming Hugging Face dataset: {DATASET_NAME}")
    sample = load_dataset(DATASET_NAME, split="train", streaming=True)
    sample_texts = collect_tokenizer_sample(sample, tc["sample_chars"])
    tokenizer = ByteBPETokenizer.train(
        sample_texts,
        vocab_size=tc["vocab_size"],
        min_pair_frequency=tc["min_pair_frequency"],
        eos_token=tc["eos_token"],
    )
    if tokenizer.vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError("Vocabulary is too large for uint16 token IDs.")

    tokenizer.save(local_dir / dc["tokenizer_file"])
    train_stream = load_dataset(DATASET_NAME, split="train", streaming=True)
    val_stream = load_dataset(DATASET_NAME, split="validation", streaming=True)

    train_count = write_token_file(
        train_stream, tokenizer, local_dir / dc["train_file"],
        dc["max_train_tokens"], dc["preprocess_batch_size"],
    )
    val_count = write_token_file(
        val_stream, tokenizer, local_dir / dc["val_file"],
        dc["max_val_tokens"], dc["preprocess_batch_size"],
    )

    metadata = {
        "dataset": DATASET_NAME, "vocab_size": tokenizer.vocab_size,
        "eos_token_id": tokenizer.eos_token_id, "train_tokens": train_count,
        "val_tokens": val_count, "dtype": "uint16",
    }
    (local_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    persist_files(
        local_dir, persistent_dir,
        [dc["tokenizer_file"], dc["train_file"], dc["val_file"], "metadata.json"],
    )
    print(f"Preprocessing complete: train={train_count:,}, val={val_count:,} tokens")


if __name__ == "__main__":
    main()
