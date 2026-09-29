from pathlib import Path
import torch


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        unknown = set(text) - set(self.stoi)
        if unknown:
            raise ValueError(f"Unknown characters: {sorted(unknown)!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path: str) -> None:
        Path(path).write_text("".join(self.chars), encoding="utf-8")

    @classmethod
    def load(cls, path: str):
        chars = Path(path).read_text(encoding="utf-8")
        tokenizer = cls("")
        tokenizer.chars = list(chars)
        tokenizer.stoi = {ch: i for i, ch in enumerate(tokenizer.chars)}
        tokenizer.itos = {i: ch for i, ch in enumerate(tokenizer.chars)}
        return tokenizer


def load_text(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if len(text) < 100:
        raise ValueError("Training data is too small. Put a real text corpus in data/train.txt.")
    return text


def make_splits(text: str, train_split: float):
    data = torch.tensor(CharTokenizer(text).encode(text), dtype=torch.long)
    n = int(len(data) * train_split)
    return data[:n], data[n:]


def get_batch(data, batch_size: int, block_size: int, device: str):
    if len(data) <= block_size:
        raise ValueError("Training data must contain more tokens than block_size.")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
