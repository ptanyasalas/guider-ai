from pathlib import Path

import torch


def load_text(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if len(text) < 100:
        raise ValueError("Training data is too small. Put a real text corpus in data/train.txt.")
    return text


def get_batch(data, batch_size: int, block_size: int, device: str):
    if len(data) <= block_size:
        raise ValueError(
            "Dataset split is too small for block_size. "
            "Use a smaller block_size or a larger training corpus."
        )

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
