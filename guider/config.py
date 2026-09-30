from pathlib import Path

import yaml


_REQUIRED_SECTIONS = (
    "model", "tokenizer", "training", "early_stopping",
    "data", "checkpoint", "generation",
)


def load_config(path: str = "config.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must contain a top-level mapping.")
    missing = [key for key in _REQUIRED_SECTIONS if key not in raw]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")
    return raw
