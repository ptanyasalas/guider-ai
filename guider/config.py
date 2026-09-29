from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    vocab_size: int | None
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    max_steps: int
    eval_interval: int
    eval_steps: int
    weight_decay: float
    grad_clip: float
    train_split: float
    seed: int


def load_config(path: str = "config.yaml"):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return raw
