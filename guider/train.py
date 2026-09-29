from pathlib import Path
import random

import torch

from .config import load_config
from .data import get_batch, load_text
from .model import GuiderLM
from .tokenizer import ByteBPETokenizer


def estimate_loss(model, train_data, val_data, cfg, device):
    model.eval()
    out = {}

    with torch.no_grad():
        for name, data in (("train", train_data), ("val", val_data)):
            losses = torch.zeros(cfg["training"]["eval_steps"])

            for k in range(len(losses)):
                x, y = get_batch(data, cfg["training"]["batch_size"], cfg["model"]["block_size"], device)
                _, loss = model(x, y)
                losses[k] = loss.item()

            out[name] = losses.mean().item()

    model.train()
    return out


def main():
    cfg = load_config()
    seed = cfg["training"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = load_text(cfg["data"]["path"])

    tokenizer_cfg = cfg["tokenizer"]
    tokenizer = ByteBPETokenizer.train(
        text,
        vocab_size=tokenizer_cfg["vocab_size"],
        min_pair_frequency=tokenizer_cfg["min_pair_frequency"],
    )

    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = int(len(encoded) * cfg["training"]["train_split"])
    train_data, val_data = encoded[:split], encoded[split:]

    block_size = cfg["model"]["block_size"]
    if len(train_data) <= block_size or len(val_data) <= block_size:
        raise ValueError(
            f"Dataset is too small for block_size={block_size}: "
            f"train_tokens={len(train_data)}, val_tokens={len(val_data)}. "
            "Use a smaller block_size or a larger corpus."
        )

    model_cfg = cfg["model"]
    model = GuiderLM(
        vocab_size=tokenizer.vocab_size,
        block_size=model_cfg["block_size"],
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        dropout=model_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    params = sum(p.numel() for p in model.parameters())
    print(
        f"Guider v0.2 | device={device} | vocab={tokenizer.vocab_size} | "
        f"merges={tokenizer.merge_count} | tokens={len(encoded)} | parameters={params:,}"
    )

    best_val = float("inf")
    checkpoint_dir = Path(cfg["checkpoint"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / cfg["checkpoint"]["filename"]

    for step in range(cfg["training"]["max_steps"] + 1):
        if step % cfg["training"]["eval_interval"] == 0:
            losses = estimate_loss(model, train_data, val_data, cfg, device)
            print(f"step {step:>6} | train {losses['train']:.4f} | val {losses['val']:.4f}")

            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": cfg,
                        "tokenizer": tokenizer.to_dict(),
                    },
                    checkpoint_path,
                )

        x, y = get_batch(train_data, cfg["training"]["batch_size"], model_cfg["block_size"], device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        optimizer.step()

    print(f"Training complete. Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
